# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<
KernelTemplate<MODEL_TYPE, NUM_HEADS>>` (`csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`,
`config.h`, `components/dequant.h`, `components/helpers.h`).

Grid `(NUM_HEADS / 64, s_q, num_sm_parts)`, 384 threads, launched on a cluster of
`NUM_HEADS / 64` CTAs, `__launch_bounds__(384, 1, CLUSTER)`, one `SharedMemoryPlan` of dynamic
shared memory. Three warpgroups:

- WG0 (`setmaxnreg.inc 192`): per top-k block of 64 tokens, P = Q K^T (36 / 32 ss WGMMA k-steps),
  the `is_kv_valid` mask and the online softmax, S = bf16(P) into registers and `sS`, and
  O[:, 0:256] += S V (rs WGMMA); per request the next Q TMA, the row sums, O / OAccum and LSE.
- WG1 (`setmaxnreg.dec 160`): O[:, 256:512] *= sScale, O += sS V (ss WGMMA).
- WG2 (`setmaxnreg.dec 152`): the producer. It gathers the block's tokens (FP8 NoPE + scales + bf16
  RoPE) with cache-hinted 128-bit loads, dequantizes the NoPE part to bf16 and stores the K block
  into `sK[buf]` in the WGMMA no-swizzle K-major layout: element (token t, dim d) at bf16 index
  `(d // 8) * 512 + t * 8 + d % 8`. With a 2-CTA cluster (h_q = 128) each CTA dequantizes half of
  the 64 tokens and also writes them into the peer CTA's buffer with `st.async`, completing the
  peer's `bar_k_remote_ready` transaction count.

V is K read transposed (the MN-major WGMMA B operand over the same bytes). Two K buffers
alternate, `buf = (block - start_block) % 2`, with one phase bit per buffer in `bar_phase_k`.

Shared memory is one linear pool at the `SharedMemoryPlan` byte offsets; `sQ`, `sS` and the
bf16 output staging tiles are 64 x 64 128B-swizzled tiles (row r at r * 128 bytes, 16-byte
chunk c at (c ^ (r % 8)) * 16). Float math is written as the PTX the source's
`--use_fast_math` build emits (`.ftz`, `ex2.approx`, `div.approx`, `lg2.approx`).
"""

import tirx_kernels.tirx_lite as txl

BLOCK_M = 64
TOPK_BLOCK = 64
HEAD_DIM_V = 512
HEAD_DIM_ROPE = 64
NUM_THREADS = 384
MAX_INIT_VAL = -1e30
#: `NamedBarriers` through CUTLASS's user overload (`+ FirstUserBarrier = 8`).
BAR_SS_READY = 8
BAR_SS_FREE = 9
BAR_OBUF_SL = 10
BAR_EPI = 11
#: `cute::TMA::CacheHintSm90::EVICT_FIRST`.
EVICT_FIRST = 0x12F0000000000000
#: `ClusterBarrier::wait`'s suspend-time hint.
TRY_WAIT_TICKS = 10000000
#: GMMA descriptors (cute `make_gmma_desc`): K-major SW128 (sQ, sS); K-major interleave (sK as
#: the QK^T B operand: LBO 1024 B, SBO 128 B); MN-major interleave (V: LBO 128 B, SBO 1024 B).
DESC_SW128 = 0x4000004000010000
DESC_K_INTER = 0x0000000800400000
DESC_V_INTER = 0x0000004000080000
#: `get_peer_addr` (helpers.h:229): the peer CTA's generic address of a shared byte.
PEER_ADDR_MASK = 16777216
L2E = 1.4426950408889634  # CUDART_L2E_F
LOG2E = 1.4426950408889634  # (float)M_LOG2E
LN2 = 0.6931471805599453

_TMA_LOAD = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_STORE_5D = "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group"
_BULK_S2G = "cp.async.bulk.global.shared::cta.bulk_group"
_LDG_256 = "ld.global.nc.L1::evict_last.L2::256B.v4.s32"
_LDG_128 = "ld.global.nc.L1::evict_last.L2::128B.v4.s32"
_ST_ASYNC = "st_async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64"
_ST_ASYNC_V4 = "st_async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32"
_WG_QK = "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16"
_WG_PV = "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16"


class Plan:
    """`KernelTemplate<MODEL_TYPE, NUM_HEADS>` constants and `SharedMemoryPlan` byte offsets."""

    def __init__(self, model_type: str, num_heads: int):
        assert model_type in ("V32", "MODEL1") and num_heads in (64, 128)
        self.v32 = model_type == "V32"
        self.cluster = num_heads // 64
        self.d = 576 if self.v32 else 512  # HEAD_DIM_K
        self.d_nope = self.d - HEAD_DIM_ROPE
        self.num_scales = 4 if self.v32 else 8
        self.tokens_per_thread = 2 if self.cluster == 1 else 1
        self.q_bytes = BLOCK_M * self.d * 2
        self.k_bytes = TOPK_BLOCK * self.d * 2
        oaccum_bytes = (63 * 520 + 512) * 4
        union = max(2 * self.k_bytes, BLOCK_M * HEAD_DIM_V * 2, oaccum_bytes)
        self.off_q = 0
        self.off_u = self.q_bytes
        self.off_s = (self.off_u + union + 1023) // 1024 * 1024
        self.off_valid = self.off_s + 8192
        self.off_m = self.off_valid + 128
        self.off_l = self.off_m + 256
        self.off_scale = self.off_l + 256
        self.off_oscale = self.off_scale + 256
        self.off_bar_q = self.off_oscale + 256
        self.off_bar_local = self.off_bar_q + 8
        self.off_bar_remote = self.off_bar_local + 16
        self.off_bar_avail = self.off_bar_remote + 16
        self.smem_bytes = (self.off_bar_avail + 16 + 1023) // 1024 * 1024


def _u32(x):
    return txl.cast(x, "uint32")


def _u64(x):
    return txl.cast(x, "uint64")


def build_main_kernel(model_type: str, num_heads: int, *, backend_config=None):
    """The TIRx `PrimFunc` for `KernelTemplate<model_type, num_heads>`."""
    pl = Plan(model_type, num_heads)
    V32, CLUSTER, D = pl.v32, pl.cluster, pl.d
    N_TILES = D // 64

    @txl.kernel()
    def flash_fwd_splitkv_mla_fp8_sparse(
        kv: txl.Tensor((txl.int64(),), "uint8"),
        indices: txl.Tensor((txl.int64(),), "int32"),
        topk_length: txl.Tensor((txl.int64(),), "int32"),
        attn_sink: txl.Tensor((txl.int64(),), "float32"),
        lse: txl.Tensor((txl.int64(),), "float32"),
        extra_kv: txl.Tensor((txl.int64(),), "uint8"),
        extra_indices: txl.Tensor((txl.int64(),), "int32"),
        extra_topk_length: txl.Tensor((txl.int64(),), "int32"),
        lse_accum: txl.Tensor((txl.int64(),), "float32"),
        o_accum: txl.Tensor((txl.int64(),), "float32"),
        tile_scheduler_metadata: txl.Tensor((txl.int64(),), "int32"),
        num_splits: txl.Tensor((txl.int64(),), "int32"),
        b: "int32",
        s_q: "int32",
        h_q: "int32",
        topk: "int32",
        page_block_size: "int32",
        sm_scale_div_log2: "float32",
        extra_page_block_size: "int32",
        extra_topk: "int32",
        stride_kv_block: "int32",
        stride_kv_row: "int32",
        stride_indices_b: "int32",
        stride_indices_s_q: "int32",
        stride_lse_b: "int32",
        stride_lse_s_q: "int32",
        stride_extra_kv_block: "int32",
        stride_extra_kv_row: "int32",
        stride_extra_indices_b: "int32",
        stride_extra_indices_s_q: "int32",
        stride_lse_accum_split: "int32",
        stride_lse_accum_s_q: "int32",
        stride_o_accum_split: "int32",
        stride_o_accum_s_q: "int32",
        stride_o_accum_h_q: "int32",
        num_sm_parts: "int32",
        tma_q: txl.Ptr(txl.TensorMap),
        tma_o: txl.Ptr(txl.TensorMap),
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(
                grid=(CLUSTER, s_q, num_sm_parts), block=NUM_THREADS, cluster=[CLUSTER, 1, 1]
            ),
            # __launch_bounds__(384, 1, CLUSTER_SIZE)
            kernel_attrs=txl.cuda.KernelAttributes(
                min_blocks_per_sm=1, max_blocks_per_cluster=CLUSTER
            ),
            backend_config=backend_config,
        )

        # ---- the SharedMemoryPlan as one linear pool ----
        smem = txl.smem_pool()
        pool = smem.alloc((pl.smem_bytes // 2,), "uint16", align=1024)
        smem.commit(pl.smem_bytes)
        pool_f32 = pool.view("float32")
        pool_u64 = pool.view("uint64")
        pool_u8 = pool.view("uint8")
        pool_base = txl.local_scalar(
            "uint32", init=_u32(txl.cuda.cvta_generic_to_shared(pool.ptr_to([0])))
        )

        def b16(byte):  # 16-bit element pointer at a byte offset
            return pool.ptr_to([byte // 2])

        def f32p(byte):
            return pool_f32.ptr_to([byte // 4])

        def u64p(byte):
            return pool_u64.ptr_to([byte // 8])

        def sw128(tile_byte, row, chunk):
            return tile_byte + row * 128 + ((chunk ^ (row % 8)) * 16)

        def bar_q():
            return u64p(pl.off_bar_q)

        def bar_at(off, buf):
            """`plan.<bar>[buf]` for a runtime buf (8-byte stride)."""
            return pool_u64.ptr_to([off // 8 + buf])

        # ---- per-thread indices (.cuh:90-96) ----
        head_block_idx = txl.cuda.block_idx("x") if CLUSTER == 2 else 0
        s_q_idx = txl.cuda.block_idx("y")
        partition_idx = txl.cuda.block_idx("z")
        idx_in_cluster = head_block_idx % 2 if CLUSTER == 2 else 0
        tid = txl.thread_id()
        warpgroup_idx = txl.local_scalar("int32")
        txl.ptx.shfl_sync.idx.b32(
            warpgroup_idx, tid // 128, txl.uint32(0), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
        )
        warp_idx = txl.local_scalar("int32")
        txl.ptx.shfl_sync.idx.b32(
            warp_idx, tid // 32, txl.uint32(0), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
        )
        t = txl.local_scalar("int32", init=tid % 128)  # idx_in_warpgroup
        lane = txl.local_scalar("int32", init=tid % 32)
        # get_AorC_row_idx (helpers.h:141): this thread's two accumulator rows.
        row0 = txl.local_scalar("int32", init=(t // 32) * 16 + (t % 32) // 4)
        rows = (row0, row0 + 8)

        # ---- PTX helpers: the fast-math forms the source compiles to ----
        def fmax(d, a, bb):
            txl.assign(d, txl.fmax(a, bb))

        def fmul(d, a, bb):
            txl.ptx["mul.ftz.f32"](d, a, bb)

        def fsub(d, a, bb):
            txl.ptx["sub.ftz.f32"](d, a, bb)

        def fadd(d, a, bb):
            txl.ptx["add.ftz.f32"](d, a, bb)

        def ffma(d, a, bb, c):
            txl.ptx["fma.rn.ftz.f32"](d, a, bb, c)

        def ex2(d, a):
            txl.ptx["ex2.approx.ftz.f32"](d, a)

        def lds(byte):
            v = txl.local_scalar("float32")
            txl.ptx["ld.shared.f32"](v, f32p(byte))
            return v

        def sts(byte, v):
            txl.ptx["st.shared.f32"](f32p(byte), v)

        def shfl_xor(v, off):
            peer = txl.local_scalar("float32")
            txl.ptx.shfl_sync.bfly.b32(
                peer, v, txl.uint32(off), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
            )
            return peer

        def wait_barrier(bar, phase):
            """`ClusterBarrier::wait`: try_wait.parity with the suspend hint, retried."""
            txl.cuda.mbarrier_wait_parity(bar, _u32(phase), txl.uint32(TRY_WAIT_TICKS))

        def expect_tx(bar, nbytes):
            txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](bar, txl.uint32(nbytes))

        def arrive(bar):
            txl.ptx["mbarrier.arrive.shared::cta.b64"](bar)

        def arrive_remote(bar, cta_id, pred):
            """`ClusterBarrier::arrive(cta_id, pred)`: mapa + a cluster-scope arrive."""
            with txl.if_(pred), txl.then_():
                local = txl.local_scalar("uint32", init=_u32(txl.cuda.cvta_generic_to_shared(bar)))
                remote = txl.local_scalar("uint32")
                txl.ptx["mapa.shared::cluster.u32"](remote, local, txl.uint32(cta_id))
                txl.ptx["mbarrier.arrive.shared::cluster.b64"](remote)

        def named_sync(bar_id, nthreads):
            txl.ptx["bar.sync"](txl.uint32(bar_id), txl.uint32(nthreads))

        def named_arrive(bar_id, nthreads):
            txl.ptx["bar.arrive"](txl.uint32(bar_id), txl.uint32(nthreads))

        def fence_async_shared():
            txl.ptx.fence.proxy.async_.shared__cta()

        def elect_one():
            return txl.cuda.elect_sync()

        def is_null(tensor):
            return txl.isnullptr(tensor.data)

        def sync_cluster():
            """`sync_all_threads_in_cluster` (config.h)."""
            if CLUSTER == 1:
                txl.cuda.cta_sync()
            else:
                txl.ptx.barrier.cluster.arrive.relaxed()
                txl.ptx.barrier.cluster.wait.acquire()

        def fence_regs(regs, n):
            for i in range(n):
                txl.cuda.wgmma.noop_barrier(regs[i])

        def desc(const, byte_expr):
            """A 64-bit GMMA descriptor: start field (shared address >> 4, 14 bits) | constant."""
            start = (byte_expr >> txl.uint32(4)) & txl.uint32(0x3FFF)
            lo = start | txl.uint32(const & 0xFFFFFFFF)
            return (txl.uint64(const >> 32) << txl.uint64(32)) | _u64(lo)

        # ---- prologue (.cuh:109-155) ----
        with txl.if_(warp_idx == 0), txl.then_():
            with txl.if_(elect_one()), txl.then_():
                txl.ptx.prefetch.tensormap(txl.address_of(tma_q))
                txl.ptx.prefetch.tensormap(txl.address_of(tma_o))
            with txl.if_(elect_one()), txl.then_():
                txl.ptx["mbarrier.init.shared::cta.b64"](bar_q(), txl.uint32(1))
                for i in range(2):
                    txl.ptx["mbarrier.init.shared::cta.b64"](
                        u64p(pl.off_bar_local + 8 * i), txl.uint32(128)
                    )
                    if CLUSTER == 2:
                        txl.ptx["mbarrier.init.shared::cta.b64"](
                            u64p(pl.off_bar_remote + 8 * i), txl.uint32(1)
                        )
                        txl.ptx["mbarrier.init.shared::cta.b64"](
                            u64p(pl.off_bar_avail + 8 * i), txl.uint32(4)
                        )
                    else:
                        txl.ptx["mbarrier.init.shared::cta.b64"](
                            u64p(pl.off_bar_avail + 8 * i), txl.uint32(256)
                        )
                txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.barrier.cluster.arrive.relaxed()

        bar_phase_k = txl.local_scalar("int32", init=0)

        # DecodingSchedMeta of this partition (.cuh:142), read by every thread.
        meta = txl.alloc_local((8,), "uint32")
        meta_u32 = tile_scheduler_metadata.view("uint32")
        txl.ptx.ld.global_.v4.u32(
            meta[0], meta[1], meta[2], meta[3], meta_u32.ptr_to([partition_idx * 8])
        )
        txl.ptx.ld.global_.v4.u32(
            meta[4], meta[5], meta[6], meta[7], meta_u32.ptr_to([partition_idx * 8 + 4])
        )
        begin_req = txl.local_scalar("int32", init=txl.cast(meta[0], "int32"))
        end_req = txl.local_scalar("int32", init=txl.cast(meta[1], "int32"))
        begin_block = txl.local_scalar("int32", init=txl.cast(meta[2], "int32"))
        meta_end_block = txl.local_scalar("int32", init=txl.cast(meta[3], "int32"))
        begin_split = txl.local_scalar("int32", init=txl.cast(meta[4], "int32"))
        first_splitted = txl.local_scalar("int32", init=txl.cast(meta[5], "int32"))
        last_splitted = txl.local_scalar("int32", init=txl.cast(meta[6], "int32"))
        with txl.if_(begin_req >= b), txl.then_():
            txl.return_(txl.int32(0))

        def launch_q_copy(batch):
            """launch_tma_copy of Q (helpers.h:188): D/64 boxes of 64 x 64, then arm bar_q."""
            for i in range(N_TILES):
                txl.ptx[_TMA_LOAD](
                    b16(pl.off_q + 8192 * i),
                    txl.address_of(tma_q),
                    txl.int32(64 * i),
                    head_block_idx * BLOCK_M,
                    s_q_idx,
                    batch,
                    bar_q(),
                    txl.uint64(EVICT_FIRST),
                )
            expect_tx(bar_q(), pl.q_bytes)

        with txl.if_(warp_idx == 0), txl.then_():
            with txl.if_(elect_one()), txl.then_():
                launch_q_copy(begin_req)
        txl.ptx.barrier.cluster.wait.acquire()

        def ldg_s32(tensor, index):
            v = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.s32(v, tensor.ptr_to([index]))
            return v

        def req_info(batch):
            """get_cur_req_info (.cuh:164-184)."""
            a = {}
            if V32:
                total_padded = topk
            else:
                tl = txl.local_scalar("int32", init=topk)
                with txl.if_(txl.Not(is_null(topk_length))), txl.then_():
                    txl.ptx.ld.global_.nc.s32(tl, topk_length.ptr_to([batch]))
                orig_padded = txl.local_scalar(
                    "int32", init=txl.max((tl + TOPK_BLOCK - 1) // TOPK_BLOCK * TOPK_BLOCK, TOPK_BLOCK)
                )
                xtl = txl.local_scalar("int32", init=extra_topk)
                with txl.if_(txl.Not(is_null(extra_topk_length))), txl.then_():
                    txl.ptx.ld.global_.nc.s32(xtl, extra_topk_length.ptr_to([batch]))
                total_padded = orig_padded + (xtl + TOPK_BLOCK - 1) // TOPK_BLOCK * TOPK_BLOCK
                a["topk_length"] = tl
                a["extra_topk_length"] = xtl
                a["num_orig"] = txl.local_scalar("int32", init=orig_padded // TOPK_BLOCK)
            a["start"] = txl.local_scalar(
                "int32", init=txl.Select(batch == begin_req, begin_block, 0)
            )
            a["end"] = txl.local_scalar(
                "int32",
                init=txl.Select(batch == end_req, meta_end_block, total_padded // TOPK_BLOCK),
            )
            a["no_split"] = txl.local_scalar(
                "int32",
                init=txl.Select(
                    batch == begin_req,
                    txl.cast(first_splitted == 0, "int32"),
                    txl.Select(batch == end_req, txl.cast(last_splitted == 0, "int32"), 1),
                ),
            )
            return a

        num_valid = txl.local_scalar("int32", init=txl.min(h_q - head_block_idx * BLOCK_M, BLOCK_M))

        def split_index(batch):
            n_split_idx = txl.Select(batch == begin_req, begin_split, 0)
            return txl.local_scalar("int32", init=ldg_s32(num_splits, batch) + n_split_idx)

        # ---------------------------------------------------------------------------
        # store_o (config.h:157-240), both attention warpgroups
        # ---------------------------------------------------------------------------
        def store_o(rO, o_scales, no_split, batch, split_idx, wg):
            with txl.if_(no_split != 0):
                with txl.then_():
                    words = txl.alloc_local((4,), "uint32")
                    tmp0 = txl.local_scalar("float32")
                    tmp1 = txl.local_scalar("float32")
                    for idx in range(16):
                        for p in range(4):
                            s = o_scales[p % 2]
                            fmul(tmp0, rO[idx * 8 + 2 * p], s)
                            fmul(tmp1, rO[idx * 8 + 2 * p + 1], s)
                            txl.ptx["cvt.rn.bf16x2.f32"](words[p], tmp1, tmp0)
                        byte = sw128(
                            pl.off_u + 8192 * (wg * 4 + idx // 4),
                            (t // 32) * 16 + t % 16,
                            2 * (idx % 4) + (t % 32) // 16,
                        )
                        txl.ptx["stmatrix.sync.aligned.x4.m8n8.shared.b16"](
                            b16(byte), words[0], words[1], words[2], words[3]
                        )
                    fence_async_shared()
                    named_sync(BAR_EPI, 256)
                    with txl.if_(tid == 0), txl.then_():
                        txl.ptx[_TMA_STORE_5D](
                            txl.address_of(tma_o),
                            txl.int32(0),
                            head_block_idx * BLOCK_M,
                            txl.int32(0),
                            s_q_idx,
                            batch,
                            b16(pl.off_u),
                        )
                        txl.ptx.cp.async_.bulk.commit_group()
                with txl.else_():
                    t0 = txl.local_scalar("float32")
                    t1 = txl.local_scalar("float32")
                    for idx in range(0, 128, 2):
                        hi = idx % 4 >= 2
                        row = rows[1 if hi else 0]
                        col = wg * 256 + (t % 4) * 2 + (idx // 4) * 8
                        s = o_scales[1 if hi else 0]
                        fmul(t0, rO[idx], s)
                        fmul(t1, rO[idx + 1], s)
                        byte = pl.off_u + 4 * (row * 520 + col)
                        if wg == 0 and idx == 0:  # the source's build splits WG0's first pair
                            txl.ptx["st.shared.f32"](f32p(byte), t0)
                            txl.ptx["st.shared.f32"](f32p(byte + 4), t1)
                        else:
                            txl.ptx["st.shared.v2.f32"](f32p(byte), t0, t1)
                    fence_async_shared()
                    named_sync(BAR_EPI, 256)
                    with txl.if_(elect_one()), txl.then_():
                        for lr in range(BLOCK_M // 8):
                            row = lr * 8 + tid // 32
                            with txl.if_(row < num_valid), txl.then_():
                                txl.ptx[_BULK_S2G](
                                    o_accum.ptr_to(
                                        [
                                            split_idx * stride_o_accum_split
                                            + s_q_idx * stride_o_accum_s_q
                                            + (head_block_idx * BLOCK_M + row) * stride_o_accum_h_q
                                        ]
                                    ),
                                    f32p(pl.off_u + 4 * row * 520),
                                    txl.uint32(HEAD_DIM_V * 4),
                                )
                        txl.ptx.cp.async_.bulk.commit_group()

        def release_k(buf):
            """`plan.bar_k_avail[buf].arrive(...)` (.cuh:264-269 / 407-412)."""
            if CLUSTER == 2:
                arrive_remote(bar_at(pl.off_bar_avail, buf), 0, t == 32)
                arrive_remote(bar_at(pl.off_bar_avail, buf), 1, t == 64)
            else:
                arrive(bar_at(pl.off_bar_avail, buf))

        def k_byte(buf):
            """Shared address of sK[buf] (runtime buf)."""
            return pool_base + txl.uint32(pl.off_u) + _u32(buf) * txl.uint32(pl.k_bytes)

        # ======================================================================
        # WG0 (.cuh:186-365)
        # ======================================================================
        def wg0():
            txl.ptx["setmaxnreg.inc.sync.aligned.u32"](192)
            rO = txl.alloc_local((128,), "float32")
            rP = txl.alloc_local((32,), "float32")
            rS = txl.alloc_local((32,), "uint16")
            rS_u32 = rS.view("uint32")
            rL = txl.alloc_local((2,), "float32")
            rM = txl.alloc_local((2,), "float32")
            sink = txl.alloc_local((2,), "float32")
            for i in range(2):
                txl.assign(sink[i], txl.float32(-float("inf")))
            with txl.if_(txl.Not(is_null(attn_sink))), txl.then_():
                for i in range(2):
                    v = txl.local_scalar("float32")
                    txl.ptx.ld.global_.nc.f32(
                        v, attn_sink.ptr_to([head_block_idx * BLOCK_M + rows[i]])
                    )
                    fmul(sink[i], v, txl.float32(L2E))
            batch = txl.local_scalar("int32", init=begin_req)
            with txl.while_(batch <= end_req):
                a = req_info(batch)
                txl.assign(rL[0], txl.float32(0.0))
                txl.assign(rL[1], txl.float32(0.0))
                txl.assign(rM[0], txl.float32(MAX_INIT_VAL))
                txl.assign(rM[1], txl.float32(MAX_INIT_VAL))
                for i in range(128):
                    txl.assign(rO[i], txl.float32(0.0))
                wait_barrier(bar_q(), (begin_req - batch) & 1)

                block = txl.local_scalar("int32", init=a["start"])
                with txl.while_(block < a["end"]):
                    buf = txl.local_scalar("int32", init=(block - a["start"]) % 2)
                    ph = (bar_phase_k >> buf) & 1
                    wait_barrier(bar_at(pl.off_bar_local, buf), ph)
                    if CLUSTER == 2:
                        wait_barrier(bar_at(pl.off_bar_remote, buf), ph)
                    # P = Q K^T (gemm<true, -1>, helpers.h:149)
                    kb = txl.local_scalar("uint32", init=k_byte(buf))
                    fence_regs(rP, 32)
                    txl.ptx.wgmma.fence.sync.aligned()
                    for i in range(N_TILES):
                        for k in range(4):
                            txl.ptx[_WG_QK](
                                *[rP[j] for j in range(32)],
                                desc(DESC_SW128, pool_base + txl.uint32(pl.off_q + 8192 * i + 32 * k)),
                                desc(DESC_K_INTER, kb + txl.uint32(2048 * (4 * i + k))),
                                (i, k) != (0, 0),
                                1,
                                1,
                                0,
                                0,
                            )
                    txl.ptx.wgmma.commit_group.sync.aligned()
                    txl.ptx.wgmma.wait_group.sync.aligned(0)
                    fence_regs(rP, 32)
                    txl.assign(bar_phase_k, bar_phase_k ^ (1 << buf))
                    with txl.if_(block != a["start"]), txl.then_():
                        named_sync(BAR_SS_FREE, 256)

                    # scale_softmax (.cuh:32-84)
                    scale_old = txl.alloc_local((2,), "float32")
                    valid_base = pl.off_valid + 64 * buf + (t % 4) * 2
                    for r in range(2):
                        cur_max = txl.local_scalar("float32", init=txl.float32(-float("inf")))
                        for g in range(8):
                            fl = txl.alloc_local((2,), "uint8")
                            txl.ptx["ld.shared.v2.u8"](fl[0], fl[1], pool_u8.ptr_to([valid_base + 8 * g]))
                            for e in range(2):
                                j = 4 * g + 2 * r + e
                                txl.assign(
                                    rP[j],
                                    txl.Select(
                                        txl.cast(fl[e], "uint16") == txl.uint16(0),
                                        txl.float32(-float("inf")),
                                        rP[j],
                                    ),
                                )
                                fmax(cur_max, cur_max, rP[j])
                        for off in (1, 2):
                            fmax(cur_max, cur_max, shfl_xor(cur_max, off))
                        fmul(cur_max, sm_scale_div_log2, cur_max)
                        old = txl.local_scalar("float32", init=rM[r])
                        fmax(rM[r], cur_max, old)
                        fsub(scale_old[r], old, rM[r])
                        ex2(scale_old[r], scale_old[r])
                        for g in range(32):
                            for e in range(2):
                                fmul(rO[4 * g + 2 * r + e], scale_old[r], rO[4 * g + 2 * r + e])
                        cur_sum = txl.local_scalar("float32", init=txl.float32(0.0))
                        tmp = txl.local_scalar("float32")
                        for g in range(8):
                            for e in range(2):
                                j = 4 * g + 2 * r + e
                                fmul(tmp, sm_scale_div_log2, rP[j])
                                fsub(tmp, tmp, rM[r])
                                ex2(rP[j], tmp)
                                txl.ptx["cvt.rn.bf16.f32"](rS[j], rP[j])
                                fadd(cur_sum, cur_sum, rP[j])
                        ffma(rL[r], scale_old[r], rL[r], cur_sum)
                    with txl.if_(t % 4 == 0), txl.then_():
                        txl.ptx["st.shared.v2.f32"](
                            f32p(pl.off_scale + 8 * (t // 4)), scale_old[0], scale_old[1]
                        )

                    # save_rPb_to_sP (config.h:136)
                    for kk in range(4):
                        byte = sw128(pl.off_s, (t // 32) * 16 + lane % 16, 2 * kk + lane // 16)
                        txl.ptx["stmatrix.sync.aligned.x4.m8n8.shared.b16"](
                            b16(byte), *[rS_u32[4 * kk + j] for j in range(4)]
                        )
                    fence_async_shared()

                    # O += S @ V[:, 0:256] (gemm<false, -1>, rs)
                    fence_regs(rS_u32, 16)
                    fence_regs(rO, 128)
                    txl.ptx.wgmma.fence.sync.aligned()
                    for k in range(4):
                        txl.ptx[_WG_PV](
                            *[rO[j] for j in range(128)],
                            *[rS_u32[4 * k + j] for j in range(4)],
                            desc(DESC_V_INTER, kb + txl.uint32(256 * k)),
                            True,
                            1,
                            1,
                            1,
                        )
                    txl.ptx.wgmma.commit_group.sync.aligned()
                    named_arrive(BAR_SS_READY, 256)
                    txl.ptx.wgmma.wait_group.sync.aligned(0)
                    fence_regs(rO, 128)
                    fence_regs(rS_u32, 16)
                    release_k(buf)
                    txl.assign(block, block + 1)

                # next Q / PDL (.cuh:272-285)
                with txl.if_(tid // 32 == 0), txl.then_():
                    with txl.if_(elect_one()), txl.then_():
                        with txl.if_(batch != end_req):
                            with txl.then_():
                                launch_q_copy(batch + 1)
                            with txl.else_():
                                txl.ptx.griddepcontrol.launch_dependents()

                # L / M across the quad, o_scales (.cuh:288-314)
                for r in range(2):
                    fadd(rL[r], rL[r], shfl_xor(rL[r], 1))
                    fadd(rL[r], rL[r], shfl_xor(rL[r], 2))
                with txl.if_(t % 4 == 0), txl.then_():
                    for r in range(2):
                        sts(pl.off_l + 4 * rows[r], rL[r])
                        sts(pl.off_m + 4 * rows[r], rM[r])
                o_scales = txl.alloc_local((2,), "float32")
                for r in range(2):
                    txl.assign(o_scales[r], txl.float32(0.0))
                    with txl.if_(rL[r] != txl.float32(0.0)), txl.then_():
                        with txl.if_(a["no_split"] != 0):
                            with txl.then_():
                                e = txl.local_scalar("float32")
                                fsub(e, sink[r], rM[r])
                                ex2(e, e)
                                fadd(e, rL[r], e)
                                txl.ptx["div.approx.ftz.f32"](o_scales[r], txl.float32(1.0), e)
                            with txl.else_():
                                txl.ptx["div.approx.ftz.f32"](o_scales[r], txl.float32(1.0), rL[r])
                    with txl.if_(t % 4 == 0), txl.then_():
                        sts(pl.off_oscale + 4 * rows[r], o_scales[r])
                named_sync(BAR_OBUF_SL, 256)

                split_idx = txl.local_scalar("int32", init=0)
                with txl.if_(a["no_split"] == 0), txl.then_():
                    txl.assign(split_idx, split_index(batch))
                store_o(rO, o_scales, a["no_split"], batch, split_idx, 0)
                with txl.if_(tid < num_valid), txl.then_():
                    cur_l = lds(pl.off_l + 4 * tid)
                    with txl.if_(a["no_split"] != 0):
                        with txl.then_():
                            out = txl.local_scalar("float32", init=txl.float32(float("inf")))
                            with txl.if_(cur_l != txl.float32(0.0)), txl.then_():
                                lg = txl.local_scalar("float32")
                                txl.ptx["lg2.approx.ftz.f32"](lg, cur_l)
                                md = txl.local_scalar("float32")
                                txl.ptx["div.approx.ftz.f32"](
                                    md, lds(pl.off_m + 4 * tid), txl.float32(LOG2E)
                                )
                                ffma(out, lg, txl.float32(LN2), md)
                            txl.ptx.st.global_.f32(
                                lse.ptr_to(
                                    [
                                        batch * stride_lse_b
                                        + s_q_idx * stride_lse_s_q
                                        + head_block_idx * BLOCK_M
                                        + tid
                                    ]
                                ),
                                out,
                            )
                        with txl.else_():
                            out = txl.local_scalar("float32", init=txl.float32(-float("inf")))
                            with txl.if_(cur_l != txl.float32(0.0)), txl.then_():
                                txl.ptx["lg2.approx.ftz.f32"](out, cur_l)
                                fadd(out, out, lds(pl.off_m + 4 * tid))
                            txl.ptx.st.global_.f32(
                                lse_accum.ptr_to(
                                    [
                                        split_idx * stride_lse_accum_split
                                        + s_q_idx * stride_lse_accum_s_q
                                        + head_block_idx * BLOCK_M
                                        + tid
                                    ]
                                ),
                                out,
                            )
                txl.ptx.cp.async_.bulk.wait_group.read(0)
                sync_cluster()
                txl.assign(batch, batch + 1)

        # ======================================================================
        # WG1 (.cuh:366-453)
        # ======================================================================
        def wg1():
            txl.ptx["setmaxnreg.dec.sync.aligned.u32"](160)
            rO = txl.alloc_local((128,), "float32")
            batch = txl.local_scalar("int32", init=begin_req)
            with txl.while_(batch <= end_req):
                a = req_info(batch)
                for i in range(128):
                    txl.assign(rO[i], txl.float32(0.0))
                block = txl.local_scalar("int32", init=a["start"])
                with txl.while_(block < a["end"]):
                    buf = txl.local_scalar("int32", init=(block - a["start"]) % 2)
                    kb = txl.local_scalar("uint32", init=k_byte(buf) + txl.uint32(32768))
                    named_sync(BAR_SS_READY, 256)
                    sc = txl.alloc_local((2,), "float32")
                    txl.ptx["ld.shared.v2.f32"](sc[0], sc[1], f32p(pl.off_scale + 8 * (t // 4)))
                    for r in range(2):
                        for g in range(32):
                            for e in range(2):
                                fmul(rO[4 * g + 2 * r + e], sc[r], rO[4 * g + 2 * r + e])
                    fence_regs(rO, 128)
                    txl.ptx.wgmma.fence.sync.aligned()
                    for k in range(4):
                        txl.ptx[_WG_PV](
                            *[rO[j] for j in range(128)],
                            desc(DESC_SW128, pool_base + txl.uint32(pl.off_s + 32 * k)),
                            desc(DESC_V_INTER, kb + txl.uint32(256 * k)),
                            True,
                            1,
                            1,
                            0,
                            1,
                        )
                    txl.ptx.wgmma.commit_group.sync.aligned()
                    txl.ptx.wgmma.wait_group.sync.aligned(0)
                    fence_regs(rO, 128)
                    release_k(buf)
                    with txl.if_(block != a["end"] - 1), txl.then_():
                        named_arrive(BAR_SS_FREE, 256)
                    txl.assign(block, block + 1)
                named_sync(BAR_OBUF_SL, 256)
                o_scales = txl.alloc_local((2,), "float32")
                for r in range(2):
                    txl.assign(o_scales[r], lds(pl.off_oscale + 4 * rows[r]))
                split_idx = txl.local_scalar("int32", init=0)
                with txl.if_(a["no_split"] == 0), txl.then_():
                    txl.assign(split_idx, split_index(batch))
                store_o(rO, o_scales, a["no_split"], batch, split_idx, 1)
                txl.ptx.cp.async_.bulk.wait_group.read(0)
                sync_cluster()
                txl.assign(batch, batch + 1)

        # ======================================================================
        # WG2: producer (.cuh:454-670)
        # ======================================================================
        def producer():
            txl.ptx["setmaxnreg.dec.sync.aligned.u32"](152)
            pw = txl.local_scalar("int32")
            txl.ptx.shfl_sync.idx.b32(
                pw, t // 32, txl.uint32(0), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
            )
            my_token_base = txl.local_scalar("int32", init=pw * 8 + lane % 8)
            TPT = pl.tokens_per_thread
            batch = txl.local_scalar("int32", init=begin_req)
            with txl.while_(batch <= end_req):
                a = req_info(batch)
                g_idx = txl.local_scalar(
                    "int32", init=batch * stride_indices_b + s_q_idx * stride_indices_s_q
                )
                g_xidx = txl.local_scalar(
                    "int32", init=batch * stride_extra_indices_b + s_q_idx * stride_extra_indices_s_q
                )
                nxt = txl.alloc_local((TPT,), "int32")
                for rnd in range(TPT):
                    pos = a["start"] * TOPK_BLOCK + idx_in_cluster * 32 + rnd * 32 + my_token_base
                    if V32:
                        txl.ptx.ld.global_.nc.s32(nxt[rnd], indices.ptr_to([g_idx + pos]))
                    else:
                        with txl.if_(a["start"] < a["num_orig"]), txl.then_():
                            txl.ptx.ld.global_.nc.s32(nxt[rnd], indices.ptr_to([g_idx + pos]))

                def process_block(block, extra, first_extra):
                    buf = txl.local_scalar("int32", init=(block - a["start"]) % 2)
                    if extra:
                        idx_t, idx_base = extra_indices, g_xidx
                        rel_block = txl.local_scalar("int32", init=block - a["num_orig"])
                        page, blk_stride = extra_page_block_size, stride_extra_kv_block
                        row_stride, kptr = stride_extra_kv_row, extra_kv
                        tlen = a["extra_topk_length"]
                    else:
                        idx_t, idx_base = indices, g_idx
                        rel_block = block
                        page, blk_stride = page_block_size, stride_kv_block
                        row_stride, kptr = stride_kv_row, kv
                        tlen = a["topk_length"] if not V32 else None
                    k_off = txl.local_scalar("int32", init=pl.off_u + buf * pl.k_bytes)
                    def peer_of(ptr):
                        """`cast_smem_ptr_to_uint(get_peer_addr(ptr))` (helpers.h:208, 232): the
                        generic address with bit 24 flipped, converted back to a shared address."""
                        g = txl.local_scalar("uint64")
                        txl.ptx["cvta.shared.u64"](g, _u64(txl.cuda.cvta_generic_to_shared(ptr)))
                        delta = txl.cast(g ^ txl.uint64(PEER_ADDR_MASK), "int64") - txl.cast(g, "int64")
                        peer = txl.ptr_byte_offset(ptr, delta, ty=ptr.ty)
                        return txl.local_scalar("uint32", init=_u32(txl.cuda.cvta_generic_to_shared(peer)))

                    def store16(byte, w):
                        """The 16-byte shared store of one bf16x8 (st.shared.v4.b32: STS.128)."""
                        txl.ptx["st.shared.v4.u32"](
                            pool.view("uint32").ptr_to([byte // 4]), w[0], w[1], w[2], w[3]
                        )

                    def store16_peer(addr32, w):
                        """st_async_128b (helpers.h:206) into the peer CTA, completing its
                        bar_k_remote_ready (st.async ... v4.b32: STAS.128)."""
                        txl.ptx[_ST_ASYNC_V4](addr32, w[0], w[1], w[2], w[3], remote_bar)

                    remote_bar = txl.local_scalar("uint32")
                    if CLUSTER == 2:
                        txl.assign(remote_bar, peer_of(bar_at(pl.off_bar_remote, buf)))

                    for rnd in range(TPT):
                        tok = txl.local_scalar("int32", init=my_token_base + rnd * 32)
                        slot = idx_in_cluster * 32 + tok
                        # token index (.cuh:515-529)
                        token = txl.local_scalar("int32")
                        if not extra:
                            txl.assign(token, nxt[rnd])
                            nxt_end = a["end"] if V32 else a["num_orig"]
                            with txl.if_(block + 1 != nxt_end), txl.then_():
                                txl.ptx.ld.global_.nc.s32(
                                    nxt[rnd], indices.ptr_to([g_idx + (block + 1) * TOPK_BLOCK + slot])
                                )
                        else:
                            if first_extra:
                                txl.ptx.ld.global_.nc.s32(
                                    token,
                                    extra_indices.ptr_to(
                                        [g_xidx + (block - a["num_orig"]) * TOPK_BLOCK + slot]
                                    ),
                                )
                            else:
                                txl.assign(token, nxt[rnd])
                            with txl.if_(block + 1 != a["end"]), txl.then_():
                                txl.ptx.ld.global_.nc.s32(
                                    nxt[rnd],
                                    extra_indices.ptr_to(
                                        [g_xidx + (block + 1 - a["num_orig"]) * TOPK_BLOCK + slot]
                                    ),
                                )
                        if not V32:
                            txl.assign(
                                token,
                                txl.Select(rel_block * TOPK_BLOCK + slot >= tlen, -1, token),
                            )
                        block_index = txl.local_scalar("int32", init=0)
                        with txl.if_(token != -1), txl.then_():
                            txl.assign(
                                block_index,
                                txl.cast(_u32(token) // _u32(page), "int32"),
                            )
                        rel_idx = txl.local_scalar(
                            "int32", init=txl.cast(_u32(token) % _u32(page), "int32")
                        )
                        scales = txl.alloc_local((pl.num_scales,), "uint16")
                        if V32:
                            g_k = txl.local_scalar(
                                "int64",
                                init=txl.cast(block_index, "int64") * txl.cast(blk_stride, "int64")
                                + txl.cast(rel_idx, "int64") * txl.cast(row_stride, "int64"),
                            )
                            sf = txl.alloc_local((4,), "int32")
                            txl.ptx[_LDG_128](sf[0], sf[1], sf[2], sf[3], kptr.ptr_to([g_k + 512]))
                        else:
                            blk = txl.local_scalar(
                                "int64",
                                init=txl.cast(block_index, "int64") * txl.cast(blk_stride, "int64"),
                            )
                            g_k = txl.local_scalar(
                                "int64", init=txl.cast(rel_idx, "int64") * txl.int64(576) + blk
                            )
                            e8 = txl.local_scalar("int64")
                            txl.ptx.ld.global_.nc.s64(
                                e8,
                                kptr.ptr_to(
                                    [blk + txl.cast(page * 576 + rel_idx * 8, "int64")]
                                ),
                            )
                        # NoPE gathers before the bar_k_avail wait (they read only global memory), as
                        # the source's build schedules them (ptxas hoists its LDGs above the try_wait).
                        n_nope = pl.d_nope // 64
                        xs = txl.alloc_local((4 * n_nope,), "int32")
                        for di in range(n_nope):
                            txl.ptx[_LDG_256](
                                xs[4 * di], xs[4 * di + 1], xs[4 * di + 2], xs[4 * di + 3],
                                kptr.ptr_to([g_k + txl.cast(16 * (lane // 8) + 64 * di, "int64")]),
                            )
                        # The scale conversion consumes the scale load: issued after the NoPE gathers so
                        # the gathers do not wait on its latency (the source's SASS order).
                        if V32:
                            for i in range(4):
                                sfv = txl.local_scalar("float32")
                                txl.ptx["mov.b32"](sfv, sf[i])
                                txl.ptx["cvt.rn.bf16.f32"](scales[i], sfv)
                        else:
                            for i in range(7):
                                # __internal_e8m0_to_bf16 (cuda_fp8.hpp:858-868): res = x << 7,
                                # then x == 0 -> 0x0040 and x == 0xFF -> 0x7FFF, as the two
                                # selp.b16 the source's build emits.
                                x = txl.local_scalar(
                                    "uint16", init=txl.cast(txl.shift_right(e8, txl.int64(8 * i)), "uint16")
                                )
                                e = txl.local_scalar("uint16", init=x & txl.uint16(0xFF))
                                r = txl.local_scalar(
                                    "uint16", init=(x << txl.uint16(7)) & txl.uint16(0x7F80)
                                )
                                is0 = txl.local_scalar("bool", init=e == txl.uint16(0))
                                is255 = txl.local_scalar("bool", init=e == txl.uint16(255))
                                txl.ptx.selp.b16(r, txl.uint16(0x0040), r, txl.ptx.pred(is0))
                                txl.ptx.selp.b16(scales[i], txl.uint16(0x7FFF), r, txl.ptx.pred(is255))
                        if rnd == 0:
                            wait_barrier(
                                bar_at(pl.off_bar_avail, buf), ((bar_phase_k >> buf) & 1) ^ 1
                            )
                        if CLUSTER == 2 and rnd == 0:
                            with txl.if_(t == 0), txl.then_():
                                expect_tx(bar_at(pl.off_bar_remote, buf), 32 * D * 2)
                        with txl.if_(token == -1), txl.then_():
                            for i in range(pl.num_scales if V32 else 7):
                                txl.ptx["cvt.rn.bf16.f32"](scales[i], txl.float32(0.0))
                        invalid = token == -1
                        # NoPE (.cuh:576-598)
                        nope_base = k_off + 2 * 8 * slot + 2 * 16 * (lane // 8) * TOPK_BLOCK
                        if CLUSTER == 2:
                            # sK_nope_peer_base = get_peer_addr(sK_nope_base) (.cuh:513), once per round
                            nope_peer = peer_of(pool_u64.ptr_to([nope_base // 8]))
                        for di in range(n_nope):
                            x = [xs[4 * di + i] for i in range(4)]
                            sidx = di // 2 if V32 else di
                            s2 = txl.local_scalar(
                                "uint32",
                                init=(_u32(scales[sidx]) << txl.uint32(16)) | _u32(scales[sidx]),
                            )
                            for half in range(2):
                                out = txl.alloc_local((4,), "uint32")
                                for q in range(2):
                                    w = txl.local_scalar(
                                        "uint32", init=txl.Select(invalid, txl.uint32(0), _u32(x[2 * half + q]))
                                    )
                                    f = txl.alloc_local((4,), "float32")
                                    for p in range(2):
                                        h2 = txl.local_scalar("uint32")
                                        txl.ptx.cvt.rn.f16x2.e4m3x2(
                                            h2,
                                            txl.cast(txl.shift_right(w, txl.uint32(16 * p)), "uint16"),
                                        )
                                        txl.ptx.cvt.f32.f16(f[2 * p], txl.cast(h2, "uint16"))
                                        txl.ptx.cvt.f32.f16(
                                            f[2 * p + 1], txl.cast(txl.shift_right(h2, txl.uint32(16)), "uint16")
                                        )
                                    for p in range(2):
                                        bf = txl.local_scalar("uint32")
                                        txl.ptx["cvt.rn.bf16x2.f32"](bf, f[2 * p + 1], f[2 * p])
                                        txl.ptx["mul.bf16x2"](out[2 * q + p], bf, s2)
                                addr = nope_base + 2 * (64 * di + 8 * half) * TOPK_BLOCK
                                store16(addr, out)
                                if CLUSTER == 2:
                                    store16_peer(
                                        nope_peer + txl.uint32(2 * (64 * di + 8 * half) * TOPK_BLOCK),
                                        out,
                                    )
                        # RoPE (.cuh:600-623)
                        rope_base = k_off + 2 * 8 * slot + 2 * (pl.d_nope + 8 * (lane // 8)) * TOPK_BLOCK
                        if CLUSTER == 2:
                            rope_peer = peer_of(pool_u64.ptr_to([rope_base // 8]))  # .cuh:607
                        rope_off = pl.d_nope + (16 if V32 else 0)
                        for di in range(2):
                            y = txl.alloc_local((4,), "int32")
                            txl.ptx[_LDG_128](
                                y[0], y[1], y[2], y[3],
                                kptr.ptr_to(
                                    [g_k + txl.cast(rope_off + 2 * (8 * (lane // 8) + 32 * di), "int64")]
                                ),
                            )
                            if not V32:
                                for i in range(4):
                                    txl.assign(y[i], txl.Select(invalid, 0, y[i]))
                            addr = rope_base + 2 * 32 * di * TOPK_BLOCK
                            yw = [_u32(y[i]) for i in range(4)]
                            store16(addr, yw)
                            if CLUSTER == 2:
                                store16_peer(rope_peer + txl.uint32(2 * 32 * di * TOPK_BLOCK), yw)
                    fence_async_shared()
                    with txl.if_(t < 32), txl.then_():
                        ix = txl.alloc_local((2,), "int32")
                        txl.ptx.ld.global_.nc.v2.s32(
                            ix[0], ix[1], idx_t.ptr_to([idx_base + rel_block * TOPK_BLOCK + lane * 2])
                        )
                        v = txl.alloc_local((2,), "uint8")
                        for e in range(2):
                            ok = ix[e] != -1
                            if not V32:
                                ok = txl.And(ok, rel_block * TOPK_BLOCK + lane * 2 + e < tlen)
                            txl.assign(v[e], txl.cast(ok, "uint8"))
                        txl.ptx["st.shared.v2.u8"](
                            pool_u8.ptr_to([pl.off_valid + 64 * buf + lane * 2]), v[0], v[1]
                        )
                    arrive(bar_at(pl.off_bar_local, buf))
                    txl.assign(bar_phase_k, bar_phase_k ^ (1 << buf))

                if V32:
                    block = txl.local_scalar("int32", init=a["start"])
                    with txl.while_(block < a["end"]):
                        process_block(block, False, False)
                        txl.assign(block, block + 1)
                else:
                    block = txl.local_scalar("int32", init=a["start"])
                    with txl.while_(block < txl.min(a["num_orig"], a["end"])):
                        process_block(block, False, False)
                        txl.assign(block, block + 1)
                    with txl.if_(a["num_orig"] < a["end"]), txl.then_():
                        process_block(txl.max(a["start"], a["num_orig"]), True, True)
                    block2 = txl.local_scalar("int32", init=txl.max(a["start"], a["num_orig"]) + 1)
                    with txl.while_(block2 < a["end"]):
                        process_block(block2, True, False)
                        txl.assign(block2, block2 + 1)
                sync_cluster()
                txl.assign(batch, batch + 1)

        with txl.if_(warpgroup_idx == 0):
            with txl.then_():
                wg0()
            with txl.else_():
                with txl.if_(warpgroup_idx == 1):
                    with txl.then_():
                        wg1()
                    with txl.else_():
                        producer()

    return flash_fwd_splitkv_mla_fp8_sparse.func

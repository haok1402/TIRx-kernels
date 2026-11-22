# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90::flash_fwd_splitkv_mla_kernel<Traits<InputT>, TmaParams>`
(`csrc/sm90/decode/dense/splitkv_mla.cuh`, `traits.h`, `config.h`).

Grid `(ceil(q_seq_per_hk / 64), h_k, num_sm_parts)`, 256 threads: two warpgroups and no producer
warpgroup (thread 0 of a warpgroup issues the TMA copies), `__launch_bounds__(256, 1, 1)`, one
`SharedMemoryPlan` of dynamic shared memory, PDL.

A CTA owns one 64-row block of folded queries (`q_seq_per_hk` rows of one KV head) and walks the
requests / KV blocks its partition's `DecodingSchedMeta` lists. KV blocks are processed in pairs:
warpgroup 0 owns the even block's softmax and output columns 0..255, warpgroup 1 the odd block's
softmax and output columns 256..511. They exchange the row max (`sM`), the rescale factors
(`sScale0`, `sScale1`) and the probability tiles (`sP0`, `sP1`) through shared memory under the
named barriers below. `sK0` / `sK1` double-buffer the paged KV blocks; V is K read transposed.

Shared memory is one linear pool at the `SharedMemoryPlan` byte offsets. Every 64 x 64 16-bit
tile is in the TMA 128B-swizzled form: row r at r * 128 bytes, its 16-byte chunk c at
(c ^ (r % 8)) * 16. The WGMMA descriptors, the ldmatrix / stmatrix lane addresses and the OOB
fill below spell that mapping explicitly. Float math is written as the PTX the source's
`--use_fast_math` build emits (`.ftz`, `ex2.approx`, `div.approx`, `lg2.approx`).
"""

import tirx_kernels.tirx_lite as txl

BLOCK_SIZE_M = 64
PAGE_BLOCK_SIZE = 64
HEAD_DIM_K = 576
HEAD_DIM_V = 512
NUM_THREADS = 256

#: `MAX_INIT_VAL_SM` / `MAX_INIT_VAL` (splitkv_mla.cuh:17-18).
MAX_INIT_VAL_SM = -1e30
MAX_INIT_VAL = -1e33
#: `traits.h::NamedBarriers` through CUTLASS's user overload (`+ FirstUserBarrier = 8`).
BAR_SSCALE0_READY = 8
BAR_SSCALE1_READY = 9
BAR_SP0_READY = 10
BAR_RO1_SP0_SV0R_ISSUED = 11
BAR_SM_INITIALIZED = 12

#: `SharedMemoryPlan` byte offsets (traits.h:70-80).
OFF_Q = 0
OFF_K0 = 73728
OFF_K1 = 147456
OFF_P0 = 221184
OFF_M = 229376
OFF_L = 229632
OFF_SCALE0 = 230144
OFF_SCALE1 = 230400
OFF_BAR_K0 = 230656
OFF_BAR_K1 = 230728
OFF_BAR_Q = 230800
SMEM_BYTES = 230816
TILE_BYTES = 64 * 64 * 2  # one 64 x 64 16-bit tile
OFF_P1 = OFF_Q + 8 * TILE_BYTES  # sP1 aliases Q tile 8
OFF_O = OFF_K0  # the epilogue's output staging aliases sK0 / sK1

#: `cute::TMA::CacheHintSm90::EVICT_FIRST`.
EVICT_FIRST = 0x12F0000000000000
#: `ClusterBarrier::wait`'s suspend-time hint (cutlass/arch/barrier.h:414).
TRY_WAIT_TICKS = 0x989680
#: GMMA descriptor high words (cute `make_gmma_desc`): K-major SW128 (LBO field 1, SBO 1024 B),
#: and the transposed V operand (MN-major SW128: LBO 8192 B to the next 64 columns, SBO 1024 B).
DESC_K_HI = 0x4000004000010000
DESC_MN_HI = 0x4000004002000000

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

_TMA_LOAD = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_STORE = "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
_BULK_S2G = "cp.async.bulk.global.shared::cta.bulk_group"


def _u32(x):
    return txl.cast(x, "uint32")


def build_main_kernel(dtype: str, *, backend_config=None):
    """The TIRx `PrimFunc` for `flash_fwd_splitkv_mla_kernel<Traits<dtype>>`."""
    ab = {"bfloat16": "bf16", "float16": "f16"}[dtype]
    wg_qk = f"wgmma.mma_async.sync.aligned.m64n64k16.f32.{ab}.{ab}"
    wg_pv = f"wgmma.mma_async.sync.aligned.m64n256k16.f32.{ab}.{ab}"
    cvt_in = f"cvt.rn.{ab}.f32"

    @txl.kernel()
    def flash_fwd_splitkv_mla(
        seqlens_k: txl.Tensor((txl.int64(),), "int32"),
        block_table: txl.Tensor((txl.int64(),), "int32"),
        tile_scheduler_metadata: txl.Tensor((txl.int64(),), "int32"),
        num_splits: txl.Tensor((txl.int64(),), "int32"),
        softmax_lse: txl.Tensor((txl.int64(),), "float32"),
        softmax_lseaccum: txl.Tensor((txl.int64(),), "float32"),
        oaccum: txl.Tensor((txl.int64(),), "float32"),
        b: "int32",
        s_q: "int32",
        q_seq_per_hk: "int32",
        q_head_per_hk: "int32",
        h_k: "int32",
        is_causal: "int32",
        scale_softmax_log2: "float32",
        block_table_batch_stride: "int32",
        num_sm_parts: "int32",
        tma_q: txl.Ptr(txl.TensorMap),
        tma_k: txl.Ptr(txl.TensorMap),
        tma_o: txl.Ptr(txl.TensorMap),
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(
                grid=((q_seq_per_hk + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, h_k, num_sm_parts),
                block=NUM_THREADS,
                programmatic_stream_serialization=True,
            ),
            # __launch_bounds__(256, 1, 1): max threads, min blocks per SM, max cluster rank.
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1, max_blocks_per_cluster=1),
            backend_config=backend_config,
        )
        del num_sm_parts

        # ---- the SharedMemoryPlan as one linear pool (byte offsets above) ----
        smem = txl.smem_pool()
        pool = smem.alloc((SMEM_BYTES // 2,), "uint16", align=1024)
        smem.commit(SMEM_BYTES)
        pool_f32 = pool.view("float32")
        pool_u64 = pool.view("uint64")
        pool_u32 = pool.view("uint32")
        pool_base = txl.local_scalar(
            "uint32", init=_u32(txl.cuda.cvta_generic_to_shared(pool.ptr_to([0])))
        )

        def b16(byte):  # 16-bit element pointer at a byte offset
            return pool.ptr_to([byte // 2])

        def f32(byte):
            return pool_f32.ptr_to([byte // 4])

        def u64(byte):
            return pool_u64.ptr_to([byte // 8])

        def bar_q():
            return u64(OFF_BAR_Q)

        def bar_k(which, i):
            return u64((OFF_BAR_K0 if which == 0 else OFF_BAR_K1) + 8 * i)

        def sw128(tile_byte, row, chunk):
            """Byte offset of 16-byte chunk `chunk` of row `row` of a 128B-swizzled tile."""
            return tile_byte + row * 128 + ((chunk ^ (row % 8)) * 16)

        # ---- per-thread indices (splitkv_mla.cuh:972-976) ----
        m_block_idx = txl.cuda.block_idx("x")
        k_head_idx = txl.cuda.block_idx("y")
        partition_idx = txl.cuda.block_idx("z")
        tid = txl.thread_id()
        warpgroup_idx = txl.local_scalar("int32", init=tid // 128)
        idx_in_wg = txl.local_scalar("int32", init=tid % 128)
        warp_in_wg = txl.local_scalar("int32", init=idx_in_wg // 32)
        lane = txl.local_scalar("int32", init=tid % 32)
        # get_AorC_row_idx (.cuh:20): this thread's two accumulator rows.
        row0 = txl.local_scalar("int32", init=warp_in_wg * 16 + lane // 4)
        row1 = txl.local_scalar("int32", init=row0 + 8)
        rows = (row0, row1)
        col_in_group = txl.local_scalar("int32", init=(idx_in_wg % 4) * 2)

        # ---- small PTX helpers (the fast-math forms the source compiles to) ----
        def fmax(d, a, bb):
            txl.ptx["max.ftz.f32"](d, a, bb)

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
            txl.ptx["ld.shared.f32"](v, f32(byte))
            return v

        def sts(byte, v):
            txl.ptx["st.shared.f32"](f32(byte), v)

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

        def named_sync(bar_id, nthreads):
            txl.ptx["bar.sync"](txl.uint32(bar_id), txl.uint32(nthreads))

        def named_arrive(bar_id, nthreads):
            txl.ptx["bar.arrive"](txl.uint32(bar_id), txl.uint32(nthreads))

        def fence_async_shared():
            txl.ptx.fence.proxy.async_.shared__cta()

        # GMMA descriptor start-address field (address >> 4, 14 bits) of the pool base. Every
        # operand offset is a compile-time constant and the pool lies below 256 KiB, so a
        # descriptor is this base plus a constant in the low word, packed with a constant high
        # word -- as cute advances its descriptors (`desc.lo += offset >> 4`).
        # The low word also carries the LBO field, so each layout gets its own base low word.
        start_field = txl.local_scalar(
            "uint32", init=(pool_base >> txl.uint32(4)) & txl.uint32(0x3FFF)
        )
        desc_lo_base = {}
        for layout in (DESC_K_HI, DESC_MN_HI):
            desc_lo_base[layout] = txl.local_scalar(
                "uint32", init=start_field | txl.uint32(layout & 0xFFFFFFFF)
            )

        def desc(layout, byte):
            """A 64-bit GMMA descriptor for the tile at pool byte offset `byte` (a constant):
            {base low word + (byte >> 4), constant high word}."""
            assert isinstance(byte, int) and byte % 16 == 0
            lo = desc_lo_base[layout] + txl.uint32(byte >> 4)
            return (txl.uint64(layout >> 32) << txl.uint64(32)) | txl.cast(lo, "uint64")

        def fence_regs(regs, n):
            for i in range(n):
                txl.cuda.wgmma.noop_barrier(regs[i])

        # ---- register state ----
        rO = txl.alloc_local((128,), "float32")  # m64n256 C fragment of this warpgroup
        rP = txl.alloc_local((32,), "float32")  # m64n64 C fragment (rP0 / rP1)
        rPb = txl.alloc_local((32,), "uint16")  # InputT copy of P, = 16 rs A registers
        rPb_u32 = rPb.view("uint32")
        rQ8 = txl.alloc_local((16,), "uint32")  # rs A fragment of Q tile 8
        rL = txl.alloc_local((2,), "float32")
        right = txl.alloc_local((2,), "int32")  # rRightBorderForQSeq
        phase_q = txl.local_scalar("int32", init=0)
        phase_k0 = txl.local_scalar("int32", init=0)
        phase_k1 = txl.local_scalar("int32", init=0)

        # ---- prologue (.cuh:994-1016) ----
        with txl.if_(tid == 0), txl.then_():
            txl.ptx.prefetch.tensormap(txl.address_of(tma_q))
            txl.ptx.prefetch.tensormap(txl.address_of(tma_k))
            txl.ptx.prefetch.tensormap(txl.address_of(tma_o))
            txl.ptx["mbarrier.init.shared::cta.b64"](bar_q(), txl.uint32(1))
            for i in range(9):
                txl.ptx["mbarrier.init.shared::cta.b64"](bar_k(0, i), txl.uint32(1))
                txl.ptx["mbarrier.init.shared::cta.b64"](bar_k(1, i), txl.uint32(1))
            fence_async_shared()
        txl.cuda.cta_sync()

        # DecodingSchedMeta of this partition (.cuh:1020), read by every thread.
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
            """launch_q_copy (.cuh:691-710): 9 boxes of 64 x 64, then arm barrier_Q."""
            with txl.if_(tid == 0), txl.then_():
                for i in range(9):
                    txl.ptx[_TMA_LOAD](
                        b16(OFF_Q + TILE_BYTES * i),
                        txl.address_of(tma_q),
                        txl.int32(64 * i),
                        m_block_idx * BLOCK_SIZE_M,
                        k_head_idx,
                        batch,
                        bar_q(),
                        txl.uint64(EVICT_FIRST),
                    )
                expect_tx(bar_q(), 64 * HEAD_DIM_K * 2)

        def launch_kv_tiles(which, block_index, tiles, issuer):
            """launch_kv_tiles_copy_tma: one 64 x 64 box per tile, no expect-tx here."""
            base = OFF_K0 if which == 0 else OFF_K1
            with txl.if_(issuer), txl.then_():
                for i in tiles:
                    txl.ptx[_TMA_LOAD](
                        b16(base + TILE_BYTES * i),
                        txl.address_of(tma_k),
                        txl.int32(64 * i),
                        txl.int32(0),
                        k_head_idx,
                        block_index,
                        bar_k(which, i),
                        txl.uint64(EVICT_FIRST),
                    )

        def ldg_s32(tensor, index):
            v = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.s32(v, tensor.ptr_to([index]))
            return v


        # ---------------------------------------------------------------------------
        # WGMMA chains
        # ---------------------------------------------------------------------------
        def qk_tile(which, i, phase, accumulate_first):
            """QKT_GEMM_ONE_TILE (.cuh:122-178): arm, wait, 4 k-steps of m64n64k16, commit."""
            kbase = OFF_K0 if which == 0 else OFF_K1
            with txl.if_(idx_in_wg == 0), txl.then_():
                expect_tx(bar_k(which, i), TILE_BYTES)
            wait_barrier(bar_k(which, i), phase)
            if i == 8:
                fence_regs(rQ8, 16)
            fence_regs(rP, 32)
            txl.ptx.wgmma.fence.sync.aligned()
            for k in range(4):
                desc_b = desc(DESC_K_HI, kbase + TILE_BYTES * i + 32 * k)
                if i != 8:
                    desc_a = desc(DESC_K_HI, OFF_Q + TILE_BYTES * i + 32 * k)
                    txl.ptx[wg_qk](
                        *[rP[j] for j in range(32)],
                        desc_a,
                        desc_b,
                        accumulate_first or k > 0,
                        1,
                        1,
                        0,
                        0,
                    )
                else:
                    txl.ptx[wg_qk](
                        *[rP[j] for j in range(32)],
                        *[rQ8[4 * k + j] for j in range(4)],
                        desc_b,
                        True,
                        1,
                        1,
                        0,
                    )
            txl.ptx.wgmma.commit_group.sync.aligned()
            fence_regs(rP, 32)
            if i == 8:
                fence_regs(rQ8, 16)

        def qk_pipeline(which, phase, stage):
            """warpgroup_cooperative_qkt_gemm<PHASE_IDX> (.cuh:196-254)."""
            if stage == 0:
                for i in (0, 1, 2, 3):
                    qk_tile(which, i, phase, i != 0)
            elif stage == 1:
                for i in (4, 5, 6, 7, 8, 0, 1, 2, 3):
                    qk_tile(which, i, phase, i != 4)
                txl.assign(phase, phase ^ 1)
            else:
                for i in (4, 5, 6, 7, 8):
                    qk_tile(which, i, phase, True)
                txl.assign(phase, phase ^ 1)

        def v_desc(which, half, k):
            base = (OFF_K0 if which == 0 else OFF_K1) + 4 * TILE_BYTES * half
            return desc(DESC_MN_HI, base + 2048 * k)

        def pv_local(which, half):
            """warpgroup_cooperative_pv_gemm_localP (.cuh:284): rO += P (registers) @ V half."""
            fence_regs(rPb_u32, 16)
            fence_regs(rO, 128)
            txl.ptx.wgmma.fence.sync.aligned()
            for k in range(4):
                txl.ptx[wg_pv](
                    *[rO[j] for j in range(128)],
                    *[rPb_u32[4 * k + j] for j in range(4)],
                    v_desc(which, half, k),
                    True,
                    1,
                    1,
                    1,
                )
            txl.ptx.wgmma.commit_group.sync.aligned()
            fence_regs(rO, 128)
            fence_regs(rPb_u32, 16)

        def pv_remote(p_off, which, half):
            """warpgroup_cooperative_pv_gemm_remoteP (.cuh:308): rO += sP @ V half."""
            fence_regs(rO, 128)
            txl.ptx.wgmma.fence.sync.aligned()
            for k in range(4):
                txl.ptx[wg_pv](
                    *[rO[j] for j in range(128)],
                    desc(DESC_K_HI, p_off + 32 * k),
                    v_desc(which, half, k),
                    True,
                    1,
                    1,
                    0,
                    1,
                )
            txl.ptx.wgmma.commit_group.sync.aligned()
            fence_regs(rO, 128)

        # ---------------------------------------------------------------------------
        # Register <-> shared tiles
        # ---------------------------------------------------------------------------
        def save_rPb_to_sP(tile_off):
            """save_rPb_to_sP (.cuh:484): stmatrix x4 per 16-column k-block."""
            for kb in range(4):
                byte = sw128(tile_off, warp_in_wg * 16 + lane % 16, 2 * kb + lane // 16)
                txl.ptx["stmatrix.sync.aligned.x4.m8n8.shared.b16"](
                    b16(byte), *[rPb_u32[4 * kb + j] for j in range(4)]
                )

        def fill_oob_V(which, half, valid):
            """fill_oob_V (.cuh:579-597): zero V tokens >= valid (they may hold NaN)."""
            v = txl.local_scalar("int32", init=txl.max(valid, 0))
            j = idx_in_wg % 64
            token = txl.local_scalar("int32", init=v + idx_in_wg // 64)
            base = (OFF_K0 if which == 0 else OFF_K1) + 4 * TILE_BYTES * half
            # cute writes through a generic pointer: st.u64 at the generic address.
            gbase = txl.local_scalar("uint64")
            txl.ptx["cvta.shared.u64"](gbase, txl.cast(pool_base, "uint64"))
            def store_zero():
                byte = sw128(base + TILE_BYTES * (j // 16), token, (j % 16) // 2) + 8 * (j % 2)
                txl.ptx["st.u64"](gbase + txl.cast(byte, "uint64"), txl.uint64(0))
                txl.assign(token, token + 2)

            # The loop as the source's build emits it: skipped when token > 63, the trip count's
            # remainder mod 4 peeled first, then a 4-way unrolled body.
            with txl.if_(token <= PAGE_BLOCK_SIZE - 1), txl.then_():
                trip = txl.local_scalar(
                    "int32",
                    init=(txl.if_then_else(token == PAGE_BLOCK_SIZE - 1, 64, 63) - token) // 2 + 1,
                )
                rem = txl.local_scalar("int32", init=trip % 4)
                for p in range(3):
                    with txl.if_(rem > p), txl.then_():
                        store_zero()
                with txl.while_(token < PAGE_BLOCK_SIZE):
                    for _ in range(4):
                        store_zero()

        # ---------------------------------------------------------------------------
        # Softmax pieces (wg0_bunch_0, wg1_bunch_0, wg0_scale_rP0, wg0_rescale_rO0)
        # ---------------------------------------------------------------------------
        def mask_row(r, start_token):
            for g in range(8):
                i = 4 * g + 2 * r
                tok = start_token + g * 8 + col_in_group
                txl.assign(
                    rP[i], txl.if_then_else(tok < right[r], rP[i], txl.float32(MAX_INIT_VAL))
                )
                txl.assign(
                    rP[i + 1],
                    txl.if_then_else(tok + 1 < right[r], rP[i + 1], txl.float32(MAX_INIT_VAL)),
                )

        def row_max(r, all_masked=False):
            cur = txl.local_scalar("float32", init=txl.float32(MAX_INIT_VAL))
            pair = txl.local_scalar("float32")
            # With every value MAX_INIT_VAL (IS_BLK0_LAST in warpgroup 1) the source's 16-way
            # max folds to that constant; only the shuffles remain.
            for g in range(0 if all_masked else 8):
                i = 4 * g + 2 * r
                fmax(pair, rP[i], rP[i + 1])
                fmax(cur, cur, pair)
            for off in (1, 2):
                fmax(cur, cur, shfl_xor(cur, off))
            fmul(cur, cur, scale_softmax_log2)
            return cur

        def exp_row(r, new_max):
            cur_sum = txl.local_scalar("float32", init=txl.float32(0.0))
            pair = txl.local_scalar("float32")
            for g in range(8):
                i = 4 * g + 2 * r
                for e in (i, i + 1):
                    fmul(rP[e], rP[e], scale_softmax_log2)
                    fsub(rP[e], rP[e], new_max)
                    ex2(rP[e], rP[e])
                txl.ptx[cvt_in](rPb[i], rP[i])
                txl.ptx[cvt_in](rPb[i + 1], rP[i + 1])
                fadd(pair, rP[i], rP[i + 1])
                fadd(cur_sum, cur_sum, pair)
            return cur_sum

        def scale_rO_row(r, s):
            for g in range(32):
                i = 4 * g + 2 * r
                fmul(rO[i], rO[i], s)
                fmul(rO[i + 1], rO[i + 1], s)

        def wg0_bunch_0(do_oob_filling, start_token):
            for r in range(2):
                row = rows[r]
                if do_oob_filling:
                    mask_row(r, start_token)
                cur_max = row_max(r)
                old = lds(OFF_M + 4 * row)
                new_max = txl.local_scalar("float32")
                fmax(new_max, old, cur_max)
                scale_old = txl.local_scalar("float32")
                fsub(scale_old, old, new_max)
                ex2(scale_old, scale_old)
                txl.cuda.warp_sync()
                with txl.if_(idx_in_wg % 4 == 0), txl.then_():
                    sts(OFF_SCALE0 + 4 * row, scale_old)
                    sts(OFF_M + 4 * row, new_max)
                scale_rO_row(r, scale_old)
                cur_sum = exp_row(r, new_max)
                ffma(rL[r], rL[r], scale_old, cur_sum)

        def wg1_bunch_0(b0l, b1l, b2l, start_token):
            for r in range(2):
                row = rows[r]
                if b1l or b2l:
                    mask_row(r, start_token)
                elif b0l:
                    for g in range(8):
                        i = 4 * g + 2 * r
                        txl.assign(rP[i], txl.float32(MAX_INIT_VAL))
                        txl.assign(rP[i + 1], txl.float32(MAX_INIT_VAL))
                cur_max = row_max(r, all_masked=b0l and not (b1l or b2l))
                old = lds(OFF_M + 4 * row)
                new_max = txl.local_scalar("float32")
                fmax(new_max, old, cur_max)
                scale_old = txl.local_scalar("float32")
                fsub(scale_old, old, new_max)
                ex2(scale_old, scale_old)
                txl.cuda.warp_sync()
                with txl.if_(idx_in_wg % 4 == 0), txl.then_():
                    sts(OFF_M + 4 * row, new_max)
                    sts(OFF_SCALE1 + 4 * row, scale_old)
                if not b0l:
                    cur_sum = exp_row(r, new_max)
                else:
                    cur_sum = txl.local_scalar("float32", init=txl.float32(0.0))
                s01 = txl.local_scalar("float32")
                fmul(s01, scale_old, lds(OFF_SCALE0 + 4 * row))
                scale_rO_row(r, s01)
                ffma(rL[r], rL[r], s01, cur_sum)

        def wg0_scale_rP0():
            tmp = txl.local_scalar("float32")
            for r in range(2):
                s1 = lds(OFF_SCALE1 + 4 * rows[r])
                for g in range(8):
                    i = 4 * g + 2 * r
                    for e in (i, i + 1):
                        fmul(tmp, rP[e], s1)
                        txl.ptx[cvt_in](rPb[e], tmp)

        def wg0_rescale_rO0():
            for r in range(2):
                s1 = lds(OFF_SCALE1 + 4 * rows[r])
                scale_rO_row(r, s1)
                fmul(rL[r], rL[r], s1)

        def next_block_index(ctx, block, ahead):
            """GET_BLOCK_INDEX (.cuh:769)."""
            v = txl.local_scalar("int32", init=0)
            with txl.if_(block + ahead < ctx["end_block"]), txl.then_():
                txl.ptx.ld.global_.nc.s32(
                    v, block_table.ptr_to([ctx["bt_base"] + block + ahead])
                )
            return v

        # ---------------------------------------------------------------------------
        # wg0_subroutine / wg1_subroutine (.cuh:742-945)
        # ---------------------------------------------------------------------------
        def wg0_block_pair(ctx, block, b0l, b1l):
            seqlen, end = ctx["seqlen_k"], ctx["end_block"]
            start_token = block * PAGE_BLOCK_SIZE
            nxt0 = next_block_index(ctx, block, 2)
            nxt1 = next_block_index(ctx, block, 3)
            wg0_bunch_0(b0l or b1l, start_token)
            named_arrive(BAR_SSCALE0_READY, NUM_THREADS)
            if b0l:
                fill_oob_V(0, 0, seqlen - start_token)
                fence_async_shared()
            pv_local(0, 0)
            txl.ptx.wgmma.wait_group.sync.aligned(0)
            named_sync(BAR_SSCALE1_READY, NUM_THREADS)
            if not b0l and not b1l:
                launch_kv_tiles(0, nxt0, range(0, 4), idx_in_wg == 0)
            wg0_scale_rP0()
            save_rPb_to_sP(OFF_P0)
            fence_async_shared()
            named_arrive(BAR_SP0_READY, NUM_THREADS)
            if not b0l:
                if b1l:
                    fill_oob_V(1, 0, seqlen - start_token - PAGE_BLOCK_SIZE)
                    fence_async_shared()
                named_sync(BAR_RO1_SP0_SV0R_ISSUED, NUM_THREADS)
                wg0_rescale_rO0()
                pv_remote(OFF_P1, 1, 0)
            if not b0l and not b1l:
                qk_pipeline(0, phase_k0, 0)
                with txl.if_(block + 3 < end), txl.then_():
                    txl.ptx.wgmma.wait_group.sync.aligned(4)
                    launch_kv_tiles(1, nxt1, range(0, 4), idx_in_wg == 0)
                qk_pipeline(0, phase_k0, 2)
            txl.ptx.wgmma.wait_group.sync.aligned(0)

        def wg1_block_pair(ctx, block, b0l, b1l, b2l):
            seqlen = ctx["seqlen_k"]
            start_token = block * PAGE_BLOCK_SIZE
            nxt0 = next_block_index(ctx, block, 2)
            nxt1 = next_block_index(ctx, block, 3)
            named_sync(BAR_SSCALE0_READY, NUM_THREADS)
            wg1_bunch_0(b0l, b1l, b2l, start_token + PAGE_BLOCK_SIZE)
            named_arrive(BAR_SSCALE1_READY, NUM_THREADS)
            if not b0l:
                save_rPb_to_sP(OFF_P1)
                if b1l:
                    fill_oob_V(1, 1, seqlen - start_token - PAGE_BLOCK_SIZE)
                    fence_async_shared()
                pv_local(1, 1)
                if not b1l:
                    fence_async_shared()
            named_sync(BAR_SP0_READY, NUM_THREADS)
            if b0l:
                fill_oob_V(0, 1, seqlen - start_token)
                fence_async_shared()
            pv_remote(OFF_P0, 0, 1)
            if not b0l:
                named_arrive(BAR_RO1_SP0_SV0R_ISSUED, NUM_THREADS)
            if not b0l and not b1l and not b2l:
                txl.ptx.wgmma.wait_group.sync.aligned(1)
                launch_kv_tiles(1, nxt1, range(4, 9), idx_in_wg == 0)
            if not b0l and not b1l:
                txl.ptx.wgmma.wait_group.sync.aligned(0)
                launch_kv_tiles(0, nxt0, range(4, 9), idx_in_wg == 0)
            if not b0l and not b1l and not b2l:
                qk_pipeline(1, phase_k1, 1)
            txl.ptx.wgmma.wait_group.sync.aligned(0)

        def wg0_request(ctx, start_block):
            """.cuh:1105-1147."""
            end = ctx["end_block"]
            # rP0 (.cuh:1107) is declared uninitialized; the source's build materializes it as
            # zeros before the guarded chain, so no path defines it inside the WGMMA stage.
            for j in range(32):
                txl.assign(rP[j], txl.float32(0.0))
            for i in range(9):
                with txl.if_(idx_in_wg == 0), txl.then_():
                    expect_tx(bar_k(0, i), TILE_BYTES)
                wait_barrier(bar_k(0, i), phase_k0)
            txl.assign(phase_k0, phase_k0 ^ 1)
            # warpgroup_cooperative_qkt_gemm_no_pipeline: 36 back-to-back ss k-steps.
            with txl.if_(start_block - 16777216 < end), txl.then_():
                fence_regs(rP, 32)
                txl.ptx.wgmma.fence.sync.aligned()
                for i in range(9):
                    for k in range(4):
                        txl.ptx[wg_qk](
                            *[rP[j] for j in range(32)],
                            desc(DESC_K_HI, OFF_Q + TILE_BYTES * i + 32 * k),
                            desc(DESC_K_HI, OFF_K0 + TILE_BYTES * i + 32 * k),
                            (i, k) != (0, 0),
                            1,
                            1,
                            0,
                            0,
                        )
                txl.ptx.wgmma.commit_group.sync.aligned()
                fence_regs(rP, 32)
            named_sync(BAR_SM_INITIALIZED, 128)
            txl.ptx.wgmma.wait_group.sync.aligned(0)
            block = txl.local_scalar("int32", init=start_block)
            with txl.while_(block < end - 2):
                wg0_block_pair(ctx, block, False, False)
                txl.assign(block, block + 2)
            with txl.if_(block + 1 < end):
                with txl.then_():
                    wg0_block_pair(ctx, block, False, True)
                with txl.else_():
                    with txl.if_(block < end), txl.then_():
                        wg0_block_pair(ctx, block, True, False)

        def wg1_request(ctx, start_block):
            """.cuh:1149-1182."""
            end = ctx["end_block"]
            for j in range(32):  # rP1 (.cuh:1151), as for rP0
                txl.assign(rP[j], txl.float32(0.0))
            with txl.if_(start_block + 1 < end), txl.then_():
                qk_pipeline(1, phase_k1, 1)
                txl.ptx.wgmma.wait_group.sync.aligned(0)
            block = txl.local_scalar("int32", init=start_block)
            with txl.while_(block < end - 3):
                wg1_block_pair(ctx, block, False, False, False)
                txl.assign(block, block + 2)
            with txl.if_(block + 2 < end):
                with txl.then_():
                    wg1_block_pair(ctx, block, False, False, True)
                    txl.assign(block, block + 2)
                    wg1_block_pair(ctx, block, True, False, False)
                with txl.else_():
                    with txl.if_(block + 1 < end):
                        with txl.then_():
                            wg1_block_pair(ctx, block, False, True, False)
                        with txl.else_():
                            with txl.if_(block < end), txl.then_():
                                wg1_block_pair(ctx, block, True, False, False)

        # ---------------------------------------------------------------------------
        # Epilogue (.cuh:1185-1265) and store_o (.cuh:608-685)
        # ---------------------------------------------------------------------------
        def epilogue(batch, n_split_idx, is_no_split):
            for r in range(2):
                fadd(rL[r], rL[r], shfl_xor(rL[r], 1))
                fadd(rL[r], rL[r], shfl_xor(rL[r], 2))
            with txl.if_(idx_in_wg % 4 == 0), txl.then_():
                sts(OFF_L + 4 * (row0 + warpgroup_idx * 64), rL[0])
                sts(OFF_L + 4 * (row0 + 8 + warpgroup_idx * 64), rL[1])
            txl.cuda.cta_sync()
            with txl.if_(warpgroup_idx == 0):
                with txl.then_():
                    fadd(rL[0], rL[0], lds(OFF_L + 4 * (row0 + 64)))
                    fadd(rL[1], rL[1], lds(OFF_L + 4 * (row0 + 8 + 64)))
                with txl.else_():
                    with txl.if_(idx_in_wg % 4 == 0), txl.then_():
                        t0 = txl.local_scalar("float32")
                        fadd(t0, lds(OFF_L + 4 * row0), rL[0])
                        sts(OFF_L + 4 * row0, t0)
                        t1 = txl.local_scalar("float32")
                        fadd(t1, lds(OFF_L + 4 * (row0 + 8)), rL[1])
                        sts(OFF_L + 4 * (row0 + 8), t1)
                    txl.cuda.warp_sync()
                    txl.assign(rL[0], lds(OFF_L + 4 * row0))
                    txl.assign(rL[1], lds(OFF_L + 4 * (row0 + 8)))
            # rL = (rL == 0 || rL != rL) ? 1 : rL
            for r in range(2):
                txl.assign(
                    rL[r],
                    txl.if_then_else(
                        txl.Or(rL[r] == txl.float32(0.0), rL[r] != rL[r]), txl.float32(1.0), rL[r]
                    ),
                )
            with txl.if_(batch + 1 <= end_req):
                with txl.then_():
                    launch_q_copy(batch + 1)
                with txl.else_():
                    txl.ptx.griddepcontrol.launch_dependents()
            num_valid = txl.local_scalar(
                "int32", init=txl.min(q_seq_per_hk - m_block_idx * BLOCK_SIZE_M, BLOCK_SIZE_M)
            )
            lse_row = txl.local_scalar(
                "int32",
                init=(batch * h_k + k_head_idx) * q_seq_per_hk + m_block_idx * BLOCK_SIZE_M,
            )
            with txl.if_(is_no_split != 0):
                with txl.then_():
                    # O = rO / rL in InputT, STSM into 8 swizzled 64 x 64 tiles, TMA store.
                    rOb = txl.alloc_local((128,), "uint16")
                    rOb_u32 = rOb.view("uint32")
                    tmp = txl.local_scalar("float32")
                    for i in range(128):
                        txl.ptx["div.approx.ftz.f32"](tmp, rO[i], rL[1 if i % 4 >= 2 else 0])
                        txl.ptx[cvt_in](rOb[i], tmp)
                    for jj in range(16):
                        tile = OFF_O + TILE_BYTES * (warpgroup_idx * 4 + jj // 4)
                        byte = sw128(tile, warp_in_wg * 16 + lane % 16, 2 * (jj % 4) + lane // 16)
                        txl.ptx["stmatrix.sync.aligned.x4.m8n8.shared.b16"](
                            b16(byte), *[rOb_u32[4 * jj + j] for j in range(4)]
                        )
                    fence_async_shared()
                    txl.cuda.cta_sync()
                    with txl.if_(tid == 0), txl.then_():
                        for jj in range(8):
                            txl.ptx[_TMA_STORE](
                                txl.address_of(tma_o),
                                txl.int32(64 * jj),
                                m_block_idx * BLOCK_SIZE_M,
                                k_head_idx,
                                batch,
                                b16(OFF_O + TILE_BYTES * jj),
                            )
                        txl.ptx.cp.async_.bulk.commit_group()
                    with txl.if_(tid < num_valid), txl.then_():
                        cur_l = lds(OFF_L + 4 * tid)
                        lse = txl.local_scalar("float32", init=txl.float32(float("inf")))
                        with txl.if_(
                            txl.Not(txl.Or(cur_l == txl.float32(0.0), cur_l != cur_l))
                        ), txl.then_():
                            lg = txl.local_scalar("float32")
                            txl.ptx["lg2.approx.ftz.f32"](lg, cur_l)
                            mdiv = txl.local_scalar("float32")
                            txl.ptx["div.approx.ftz.f32"](
                                mdiv, lds(OFF_M + 4 * tid), txl.float32(LOG2E)
                            )
                            ffma(lse, lg, txl.float32(LN2), mdiv)
                        txl.ptx.st.global_.f32(softmax_lse.ptr_to([lse_row + tid]), lse)
                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                with txl.else_():
                    split_raw = txl.local_scalar("uint32")
                    txl.ptx.ld.global_.u32(split_raw, num_splits.view("uint32").ptr_to([batch]))
                    split_idx = txl.local_scalar(
                        "int32", init=txl.cast(split_raw, "int32") + n_split_idx
                    )
                    # fp32 rows of 520 floats (the padding avoids bank conflicts).
                    t0 = txl.local_scalar("float32")
                    t1 = txl.local_scalar("float32")
                    for idx in range(0, 128, 2):
                        rl = rL[1 if idx % 4 >= 2 else 0]
                        row = row0 + (8 if idx % 4 >= 2 else 0)
                        col = warpgroup_idx * 256 + col_in_group + (idx // 4) * 8
                        txl.ptx["div.approx.ftz.f32"](t0, rO[idx], rl)
                        txl.ptx["div.approx.ftz.f32"](t1, rO[idx + 1], rl)
                        byte = OFF_O + 4 * (row * 520 + col)
                        if idx == 0:  # the source's build splits the first pair into scalars
                            txl.ptx["st.shared.f32"](f32(byte), t0)
                            txl.ptx["st.shared.f32"](f32(byte + 4), t1)
                        else:
                            txl.ptx["st.shared.v2.f32"](f32(byte), t0, t1)
                    fence_async_shared()
                    txl.cuda.cta_sync()
                    acc_row = txl.local_scalar(
                        "int32",
                        init=(split_idx * h_k + k_head_idx) * q_seq_per_hk
                        + m_block_idx * BLOCK_SIZE_M,
                    )
                    with txl.if_(tid < num_valid), txl.then_():
                        txl.ptx[_BULK_S2G](
                            oaccum.ptr_to([(acc_row + tid) * HEAD_DIM_V]),
                            f32(OFF_O + 4 * tid * 520),
                            txl.uint32(HEAD_DIM_V * 4),
                        )
                        txl.ptx.cp.async_.bulk.commit_group()
                    with txl.if_(tid < num_valid), txl.then_():
                        cur_l = lds(OFF_L + 4 * tid)
                        lse = txl.local_scalar("float32", init=txl.float32(-float("inf")))
                        with txl.if_(
                            txl.Not(txl.Or(cur_l == txl.float32(0.0), cur_l != cur_l))
                        ), txl.then_():
                            txl.ptx["lg2.approx.ftz.f32"](lse, cur_l)
                            fadd(lse, lse, lds(OFF_M + 4 * tid))
                        txl.ptx.st.global_.f32(softmax_lseaccum.ptr_to([acc_row + tid]), lse)
                    txl.ptx.cp.async_.bulk.wait_group.read(0)

        launch_q_copy(begin_req)

        # ---- per request (.cuh:1026-1266, #pragma unroll 1) ----
        batch = txl.local_scalar("int32", init=begin_req)
        with txl.while_(batch <= end_req):
            n_split_idx = txl.local_scalar(
                "int32", init=txl.if_then_else(batch == begin_req, begin_split, 0)
            )
            seqlen_k = ldg_s32(seqlens_k, batch)
            start_block = txl.local_scalar(
                "int32", init=txl.if_then_else(batch == begin_req, begin_block, 0)
            )
            end_block = txl.local_scalar(
                "int32",
                init=txl.if_then_else(
                    batch == end_req, meta_end_block, txl.truncdiv(seqlen_k + 63, 64)
                ),
            )
            is_no_split = txl.local_scalar(
                "int32",
                init=txl.if_then_else(
                    batch == begin_req,
                    txl.cast(first_splitted == 0, "int32"),
                    txl.if_then_else(batch == end_req, txl.cast(last_splitted == 0, "int32"), 1),
                ),
            )

            def get_mask_len(local_seq_q_idx):
                """get_mask_len (.cuh:948-957)."""
                g = m_block_idx * BLOCK_SIZE_M + local_seq_q_idx
                v = txl.local_scalar("int32", init=0)
                with txl.if_(g < q_seq_per_hk), txl.then_():
                    txl.assign(v, s_q - txl.truncdiv(g, q_head_per_hk) - 1)
                return v

            with txl.if_(is_causal != 0):
                with txl.then_():
                    common_mask_len = get_mask_len(BLOCK_SIZE_M - 1)
                    last_block_in_seq = txl.local_scalar(
                        "int32", init=txl.truncdiv(seqlen_k - common_mask_len + 63, 64)
                    )
                    txl.assign(
                        end_block,
                        txl.if_then_else(
                            batch == end_req,
                            txl.min(meta_end_block, last_block_in_seq),
                            last_block_in_seq,
                        ),
                    )
                    for r in range(2):
                        txl.assign(
                            right[r],
                            txl.min(seqlen_k - get_mask_len(rows[r]), end_block * PAGE_BLOCK_SIZE),
                        )
                with txl.else_():
                    txl.assign(right[0], seqlen_k)
                    txl.assign(right[1], seqlen_k)

            bt_base = txl.local_scalar("int32", init=batch * block_table_batch_stride)

            # Copy K0 and K1 (.cuh:1082-1086). Each launch `__ldg`s its block id again.
            launch_kv_tiles(0, ldg_s32(block_table, bt_base + start_block), range(9), tid == 0)
            with txl.if_(start_block + 1 < end_block), txl.then_():
                nb_a = ldg_s32(block_table, bt_base + start_block + 1)
                launch_kv_tiles(1, nb_a, range(4, 9), tid == 0)
                nb_b = ldg_s32(block_table, bt_base + start_block + 1)
                launch_kv_tiles(1, nb_b, range(0, 4), tid == 0)

            for i in range(128):
                txl.assign(rO[i], txl.float32(0.0))
            txl.assign(rL[0], txl.float32(0.0))
            txl.assign(rL[1], txl.float32(0.0))
            with txl.if_(tid < BLOCK_SIZE_M), txl.then_():
                sts(OFF_M + 4 * tid, txl.float32(MAX_INIT_VAL_SM))

            wait_barrier(bar_q(), phase_q)
            txl.assign(phase_q, phase_q ^ 1)

            # rQ8: ldmatrix of Q tile 8 into the rs A fragment (retrieve_rP_from_sP, .cuh:506).
            for kb in range(4):
                lrow = warp_in_wg * 16 + lane % 16
                byte = sw128(OFF_Q + 8 * TILE_BYTES, lrow, 2 * kb + lane // 16)
                txl.ptx["ldmatrix.sync.aligned.x4.m8n8.shared.b16"](
                    rQ8[4 * kb], rQ8[4 * kb + 1], rQ8[4 * kb + 2], rQ8[4 * kb + 3], b16(byte)
                )

            ctx = dict(seqlen_k=seqlen_k, end_block=end_block, bt_base=bt_base)
            with txl.if_(warpgroup_idx == 0):
                with txl.then_():
                    wg0_request(ctx, start_block)
                with txl.else_():
                    wg1_request(ctx, start_block)

            epilogue(batch, n_split_idx, is_no_split)
            with txl.if_(batch != end_req), txl.then_():
                txl.cuda.cta_sync()
            txl.assign(batch, batch + 1)

    return flash_fwd_splitkv_mla.func

# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ 0251105a), Copyright (c) 2024, Jay Shah,
# Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of the three SM90 backward kernels of flash-attention ``hopper/``:

- ``flash::FlashAttnBwdPreprocess`` (``flash_bwd_preprocess_kernel.h``);
- ``flash::FlashAttnBwdSm90`` (``flash_bwd_kernel_sm90.h``, ``mainloop_bwd_sm90_tma_gmma_ws.hpp``,
  ``epilogue_bwd.hpp``, ``tile_scheduler.hpp``, ``mask.h``, ``block.h``);
- ``flash::FlashAttnBwdPostprocessConvertdQ`` (``flash_bwd_postprocess_kernel.h``).

Specializations: head dim 128, bf16, MHA, fixed length, non-deterministic dQ accumulation;
causal (kBlockM 64, dQ = dS K, ``SingleTileBwdLPTScheduler``) and non-causal (kBlockM 80,
dQ^T = K^T dS^T, ``SingleTileScheduler``); kBlockN 128, two MMA warpgroups, 2-stage Q / dO / dS.

Shared memory of the main kernel is one linear pool at the byte offsets of the source's
``SharedStorage`` (``OFFSETS``). Every 16-bit tile is in the 128B-swizzled K-major form:
row r of a 64-column half at r * 128 bytes, its 16-byte chunk c at (c ^ (r % 8)) * 16
(``_sw128``). The WGMMA C fragment of an m64nN tile gives thread t (warp w = t // 32 of its
warpgroup, lane l) rows 16w + l // 4 (+ 8) and columns 8c + 2 (l % 4) + {0, 1}, registers
4c + {0, 1} (+ {2, 3} for row + 8); the rs A fragment of k-step k is bf16x2 words 4k..4k + 3 of
the same thread's converted S^T / dS^T fragment (``convert_layout_acc_Aregs``: no data movement).
Float math is the PTX the source's ``--use_fast_math`` build emits (``.ftz``, ``ex2.approx``,
contractable ``mul``/``sub``).
"""

import tirx_kernels.tirx_lite as txl

HEAD_DIM = 128
BLOCK_N = 128
NUM_THREADS = 384
NUM_MMA_THREADS = 256
LOG2E_F32 = 1.4426950408889634  # float(M_LOG2E)

#: CUTLASS ``FirstUserBarrier`` (8) + ``BwdNamedBarriers``; ``EpilogueBarrier`` is reserved id 1.
BAR_EPILOGUE = 1
BAR_PDS = 9
BAR_DQ_EMPTY = 10  # + warpgroup 0 / 1
BAR_DQ_FULL = 13  # + warpgroup 0 / 1

#: ``ClusterBarrier::wait``'s suspend-time hint.
TRY_WAIT_TICKS = 10000000
#: ``cute::TMA::CacheHintSm90``.
EVICT_NORMAL = 0x1000000000000000
EVICT_LAST = 0x14F0000000000000
#: GMMA descriptor high word (SWIZZLE_128B, SBO 1024 B) and the low-word LBO fields of the
#: export (mma_traits_sm90_gmma.hpp): K-major 1, MN-major sQ / sdO = the half stride kBlockM*128,
#: MN-major sK half (one 64-column atom) 0.
DESC_HI = 0x40000040

#: ``SharedStorage`` byte offsets (layout probe of the export).
OFFSETS = {
    64: dict(k=0, v=32768, dqacc=65536, q=98304, do=131072, lse=163840, dpsum=164352,
             ds=165888, bars=198656, total=199680),
    80: dict(k=0, v=32768, dqacc=65536, q=106496, do=147456, lse=188416, dpsum=189312,
             ds=190464, bars=231424, total=232448),
}

_TMA_LOAD = "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
_BULK_G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
_TMA_STORE = "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
_BULK_REDUCE = "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32"
_WG_SS = "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16"
_STSM = "stmatrix.sync.aligned.x4.m8n8.shared.b16"
_STSM_T = "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16"


def _u32(x):
    return txl.cast(x, "uint32")


def _i32(x):
    return txl.cast(x, "int32")


def _i64(x):
    return txl.cast(x, "int64")


def _sw128(row, chunk):
    """Byte offset of 16-byte chunk ``chunk`` (0..7) of row ``row`` in a 128B-swizzled half."""
    return row * 128 + ((chunk ^ (row % 8)) * 16)


def _block_m(is_causal):
    return 64 if is_causal else 80


# =====================================================================================
# Kernel 1: FlashAttnBwdPreprocess (flash_bwd_preprocess_kernel.h:143-245)
# =====================================================================================


def build_preprocess(head_dim, block_m, *, backend_config=None):
    assert head_dim == HEAD_DIM
    BM = block_m
    ROWS = BM // 16  # GmemTiledCopy (16 x 16) x 8: rows per thread

    @txl.kernel()
    def flash_bwd_preprocess(
        o: txl.Tensor((txl.int64(),), "bfloat16"),
        do: txl.Tensor((txl.int64(),), "bfloat16"),
        lse: txl.Tensor((txl.int64(),), "float32"),
        dpsum: txl.Tensor((txl.int64(),), "float32"),
        lse_log2: txl.Tensor((txl.int64(),), "float32"),
        dq_accum: txl.Tensor((txl.int64(),), "float32"),
        dq_semaphore: txl.Tensor((txl.int64(),), "int32"),
        s_q: "int32",
        d: "int32",
        h: "int32",
        b: "int32",
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=((s_q + BM - 1) // BM, h, b), block=256),
            # __launch_bounds__(256, 2)
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=2),
            backend_config=backend_config,
        )
        # ---- kernel body starts here (FlashAttnBwdPreprocess::operator()) ----
        m_block = txl.cuda.block_idx("x")
        bidh = txl.cuda.block_idx("y")
        bidb = txl.cuda.block_idx("z")
        t = txl.thread_id()
        rows_left = txl.local_scalar("int32", init=s_q - m_block * BM)  # seqlen_o - m_block*kBlockM
        s_q_rounded = (s_q + BM - 1) // BM * BM

        # lse = thread_idx < seqlen_o - m_block*kBlockM && thread_idx < kBlockM ? gLSE(t) : INFINITY
        lse_v = txl.local_scalar("float32", init=txl.float32(float("inf")))
        with txl.if_(txl.And(t < rows_left, t < BM)), txl.then_():
            txl.ptx.ld.global_.f32(
                lse_v, lse.ptr_to([_i64(bidb * h + bidh) * _i64(s_q) + _i64(m_block * BM + t)])
            )

        row0 = t // 16
        col8 = (t % 16) * 8
        pcol = col8 < d  # tOpO(0): the one K iteration (128 = 16 x 8)
        # O / dO rows: (b, s, h, d) contiguous; row stride h*d, head stride d, batch stride s*h*d.
        row_base = _i64(bidb) * _i64(s_q) + _i64(m_block * BM)

        def gaddr(t_, r):
            idx = ((row_base + _i64(r)) * _i64(h) + _i64(bidh)) * _i64(d) + _i64(col8)
            return t_.view("uint32").ptr_to([idx // 2])

        rO = txl.alloc_local((4 * ROWS,), "uint32")
        rdO = txl.alloc_local((4 * ROWS,), "uint32")

        def load_rows(dst, src):
            # flash::copy<Is_even_MN=false, Is_even_K=false, Clear_OOB_MN, Clear_OOB_K>: zero, then
            # the 128-bit load under (row < max_MN) && pred(k).
            for i in range(ROWS):
                for j in range(4):
                    txl.assign(dst[4 * i + j], txl.uint32(0))
                with txl.if_(txl.And(row0 + 16 * i < rows_left, pcol)), txl.then_():
                    txl.ptx.ld.global_.v4.u32(
                        dst[4 * i], dst[4 * i + 1], dst[4 * i + 2], dst[4 * i + 3], gaddr(src, row0 + 16 * i)
                    )

        load_rows(rO, o)
        load_rows(rdO, do)

        def bf16_lo(w):
            return txl.reinterpret(w << txl.uint32(16), ty="float32")

        def bf16_hi(w):
            return txl.reinterpret(w & txl.uint32(0xFFFF0000), ty="float32")

        for i in range(ROWS):
            acc = txl.local_scalar("float32")
            for e in range(8):
                w = 4 * i + e // 2
                ov = bf16_lo(rO[w]) if e % 2 == 0 else bf16_hi(rO[w])
                gv = bf16_lo(rdO[w]) if e % 2 == 0 else bf16_hi(rdO[w])
                if e == 0:
                    txl.ptx["mul.ftz.f32"](acc, gv, ov)
                else:
                    txl.ptx["fma.rn.ftz.f32"](acc, gv, ov, acc)
            # Allreduce<16> with SumOp: butterfly 8, 4, 2, 1.
            for off in (8, 4, 2, 1):
                peer = txl.local_scalar("float32")
                txl.ptx.shfl_sync.bfly.b32(
                    peer, acc, txl.uint32(off), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
                )
                txl.ptx["add.ftz.f32"](acc, acc, peer)
            r = row0 + 16 * i
            with txl.if_(col8 == 0), txl.then_():
                val = txl.if_then_else(r < rows_left, acc, txl.float32(0.0))
                txl.ptx.st.global_.f32(
                    dpsum.ptr_to([_i64(bidb * h + bidh) * _i64(s_q_rounded) + _i64(m_block * BM + r)]),
                    val,
                )

        with txl.if_(txl.And(t < s_q_rounded - m_block * BM, t < BM)), txl.then_():
            scaled = txl.local_scalar("float32")
            txl.ptx["mul.ftz.f32"](scaled, lse_v, txl.float32(LOG2E_F32))
            txl.ptx.st.global_.f32(
                lse_log2.ptr_to([_i64(bidb * h + bidh) * _i64(s_q_rounded) + _i64(m_block * BM + t)]),
                txl.if_then_else(lse_v == txl.float32(-float("inf")), txl.float32(0.0), scaled),
            )

        # Clear_dQaccum: GmemTiledCopyAccum 256 threads x 4 floats.
        acc_base = _i64(bidb * h + bidh) * _i64(s_q_rounded * HEAD_DIM) + _i64(m_block * BM * HEAD_DIM)
        for i in range(BM * HEAD_DIM // 1024):
            txl.ptx.st.global_.v4.u32(
                dq_accum.view("uint32").ptr_to([acc_base + _i64(t * 4 + 1024 * i)]),
                txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0),
            )

        # params.dq_semaphore != nullptr && thread_idx == 0 (mha_bwd always passes one).
        sem_ptr = txl.reinterpret(dq_semaphore.data, ty="uint64")
        with txl.if_(txl.And(sem_ptr != txl.uint64(0), t == 0)), txl.then_():
            txl.ptx.st.global_.u32(
                dq_semaphore.view("uint32").ptr_to([bidh + bidb * h + m_block * h * b]), txl.uint32(0)
            )

    return flash_bwd_preprocess.func


# =====================================================================================
# Kernel 3: FlashAttnBwdPostprocessConvertdQ (flash_bwd_postprocess_kernel.h:156-252)
# =====================================================================================


def build_postprocess(head_dim, is_causal, *, dkv=False, backend_config=None):
    """The dQ convert (TileShape (kBlockM, d), TiledMmadQ), or with ``dkv`` the dK / dV convert of the
    GQA build (TileShape (kBlockN, d) = (128, 128), TiledMmadKV, dKV_swapAB false: run_flash_bwd's
    PostprocessKerneldKV, softmax_scale 1)."""
    assert head_dim == HEAD_DIM
    BM = BLOCK_N if dkv else _block_m(is_causal)
    # dQ (m64n64) / dQ^T (m64n80) / dK, dV (m64n128) C fragment registers per thread
    NREG = 64 if dkv else (32 if is_causal else 40)
    swz333 = dkv or is_causal  # Swizzle<3,3,3> 8 x 64 atoms (MmaShapeN 64 or 128), else <1,3,3> 8 x 16
    SMEM_ACC = BM * HEAD_DIM * 4
    SMEM_DQ = BM * HEAD_DIM * 2
    OFF_BAR = SMEM_ACC + SMEM_DQ
    SMEM_BYTES = OFF_BAR + 128  # sizeof(SharedStorage): 49280 / 61568 / 98432 (dK, dV)

    @txl.kernel()
    def flash_bwd_convert_dq(
        dq_accum: txl.Tensor((txl.int64(),), "float32"),
        dq: txl.Tensor((txl.int64(),), "bfloat16"),
        s_q: "int32",
        d: "int32",
        h: "int32",
        b: "int32",
        softmax_scale: "float32",
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=((s_q + BM - 1) // BM, h, b), block=256),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=2),
            backend_config=backend_config,
        )
        # ---- kernel body starts here (FlashAttnBwdPostprocessConvertdQ::operator()) ----
        smem = txl.smem_pool()
        pool = smem.alloc((SMEM_BYTES // 2,), "uint16", align=128)
        smem.commit(SMEM_BYTES)
        pool_f32 = pool.view("float32")
        pool_u64 = pool.view("uint64")
        m_block = txl.cuda.block_idx("x")
        bidh = txl.cuda.block_idx("y")
        bidb = txl.cuda.block_idx("z")
        t = txl.thread_id()
        s_q_rounded = (s_q + BM - 1) // BM * BM
        bar = pool_u64.ptr_to([OFF_BAR // 8])

        acc_base = _i64(bidb * h + bidh) * _i64(s_q_rounded * HEAD_DIM) + _i64(m_block * BM * HEAD_DIM)
        with txl.if_(t == 0), txl.then_():
            txl.ptx["mbarrier.init.shared::cta.b64"](bar, txl.uint32(1))
            txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](bar, txl.uint32(SMEM_ACC))
            txl.ptx[_BULK_G2S](
                pool_f32.ptr_to([0]), dq_accum.ptr_to([acc_base]), txl.uint32(SMEM_ACC), bar
            )
        txl.cuda.cta_sync()
        txl.cuda.mbarrier_wait_parity(bar, txl.uint32(0), txl.uint32(TRY_WAIT_TICKS))

        w = t // 128
        tt = t % 128
        racc = txl.alloc_local((NREG,), "float32")
        for i in range(NREG // 4):
            txl.ptx.ld.shared.v4.f32(
                racc[4 * i], racc[4 * i + 1], racc[4 * i + 2], racc[4 * i + 3],
                pool_f32.ptr_to([w * (BM * 64) + (i * 128 + tt) * 4]),
            )
        for i in range(NREG):
            txl.ptx["mul.ftz.f32"](racc[i], racc[i], softmax_scale)
        r16 = txl.alloc_local((NREG // 2,), "uint32")
        for wd in range(NREG // 2):
            txl.ptx["cvt.rn.bf16x2.f32"](r16[wd], racc[2 * wd + 1], racc[2 * wd])

        warp = tt // 32
        lane = t % 32
        k_ = lane % 8  # stmatrix: lanes 8j..8j+7 give the 8 row addresses of matrix j
        jm = lane // 8
        i_j = jm % 2  # matrix j of instruction p covers (c = 2p + j // 2, row half i = j % 2)

        def sdq_byte(m, col):
            """Byte of element (row m, column col) of sdQ (col a multiple of 8)."""
            if swz333:  # Swizzle<3,3,3> o ((8,8),(64,2)):((64,512),(1,4096)) (dQ; dK, dV: (8,16) rows)
                return SMEM_ACC + (col // 64) * (BM * 128) + m * 128 + ((((col % 64) // 8) ^ (m % 8)) * 16)
            # Swizzle<1,3,3> o ((8,10),(16,8)):((16,128),(1,1280))
            return (
                SMEM_ACC + (col // 16) * 2560 + (m // 8) * 256 + (m % 8) * 32
                + ((((col % 16) // 8) ^ ((m % 8) // 4)) % 2) * 16
            )

        for p in range(NREG // 8):
            c_j = 2 * p + jm // 2
            if dkv:
                # dK / dV fragment (KV rows 64w + 16*warp + 8i + .., d cols 8c + ..): STSM_N.
                m = w * 64 + warp * 16 + i_j * 8 + k_
                col = c_j * 8
                txl.ptx[_STSM](pool.ptr_to([sdq_byte(m, col) // 2]), *[r16[4 * p + j] for j in range(4)])
            elif is_causal:
                # dQ fragment (q rows 16*warp + 8i + .., d cols 64w + 8c + ..): STSM_N.
                m = warp * 16 + i_j * 8 + k_
                col = w * 64 + c_j * 8
                txl.ptx[_STSM](pool.ptr_to([sdq_byte(m, col) // 2]), *[r16[4 * p + j] for j in range(4)])
            else:
                # dQ^T fragment (d rows 64w + 16*warp + 8i + .., q cols 8c + ..): STSM_T into (q, d).
                m = c_j * 8 + k_
                col = w * 64 + warp * 16 + i_j * 8
                txl.ptx[_STSM_T](pool.ptr_to([sdq_byte(m, col) // 2]), *[r16[4 * p + j] for j in range(4)])
        txl.cuda.cta_sync()

        col8 = (t % 16) * 8
        pcol = col8 < d
        rows = BM // 16
        v = txl.alloc_local((4 * rows,), "uint32")
        pool_u32 = pool.view("uint32")
        for i in range(rows):  # flash::copy smem -> regs (EvenM, Is_even_K): no predicate
            m = t // 16 + 16 * i
            txl.ptx.ld.shared.v4.u32(
                v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3], pool_u32.ptr_to([sdq_byte(m, col8) // 4])
            )
        limit = txl.min(s_q - m_block * BM, BM)
        for i in range(rows):
            m = t // 16 + 16 * i
            with txl.if_(txl.And(m < limit, pcol)), txl.then_():
                idx = ((_i64(bidb) * _i64(s_q) + _i64(m_block * BM + m)) * _i64(h) + _i64(bidh)) * _i64(d) + _i64(col8)
                txl.ptx.st.global_.v4.u32(
                    dq.view("uint32").ptr_to([idx // 2]), v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]
                )

    return flash_bwd_convert_dq.func


def build_postprocess_dkv(head_dim, *, backend_config=None):
    return build_postprocess(head_dim, True, dkv=True, backend_config=backend_config)


# =====================================================================================
# Kernel 2: FlashAttnBwdSm90 (flash_bwd_kernel_sm90.h:155-278)
# =====================================================================================


def build_main_kernel(head_dim, is_causal, *, gqa=False, backend_config=None):
    assert head_dim == HEAD_DIM
    BM = _block_m(is_causal)
    NC = BM // 8  # 8-column groups of the S^T / dP^T fragment
    KM = BM // 16  # k-steps of the dV / dK chains
    NQ = 32 if is_causal else 40  # dQ / dQ^T fragment registers
    OFF = OFFSETS[BM]
    SMEM_BYTES = OFF["total"]
    Q_TILE = BM * 256  # bytes of one Q / dO / dS stage
    HALF = BM * 128  # 64-column half stride of a Q / dO / dS stage
    TX_Q = BM * 256 + BM * 4  # TmaTransactionBytesQ + TmaTransactionBytesLSE
    LSE_STRIDE = (BM + 63) // 64 * 64  # SmemLayoutLSE stage stride (floats)
    wg_s = _WG_SS.format(n=BM)
    wg_q = _WG_SS.format(n=64 if is_causal else 80)
    wg_kv = _WG_SS.format(n=128)
    # pipeline barriers: barrier_KV, q full[2] empty[2], do full[2] empty[2]
    B_KV = OFF["bars"]

    def b_full_q(s):
        return OFF["bars"] + 16 + 8 * s

    def b_empty_q(s):
        return OFF["bars"] + 32 + 8 * s

    def b_full_do(s):
        return OFF["bars"] + 48 + 8 * s

    def b_empty_do(s):
        return OFF["bars"] + 64 + 8 * s

    def _body(lse_log2, dpsum, dq_accum, dk, dv, dk_accum, dv_accum, s_q, s_k, d, h, h_k, b, window_size_right, softmax_scale, scale_log2, lpt_total, blk_div, blk_mul, blk_shr, head_div, head_mul, head_shr, l2min_div, l2min_mul, l2min_shr, l2maj_div, l2maj_mul, l2maj_shr, l2res_div, l2res_mul, l2res_shr, num_hb_quotient, qhkh_div, qhkh_mul, qhkh_shr, tma_q, tma_do, tma_k, tma_v, tma_dk, tma_dv):
        """FlashAttnBwdSm90::operator(); the GQA build passes dk / dv / tma_dk / tma_dv = None and the MHA
        build dk_accum / dv_accum / h_k = None."""
        n_blocks = (s_k + BLOCK_N - 1) // BLOCK_N
        grid = (lpt_total,) if is_causal else (n_blocks, h, b)
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=grid, block=NUM_THREADS),
            # __launch_bounds__(384, 1): setmaxnreg needs the entry register count pinned.
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=backend_config,
        )
        # dk / dv: store_zero's plain 128-bit stores (the TMA epilogue goes through tma_dk / tma_dv).
        # ---- kernel body starts here (FlashAttnBwdSm90::operator()) ----
        smem = txl.smem_pool()
        pool = smem.alloc((SMEM_BYTES // 2,), "uint16", align=1024)
        smem.commit(SMEM_BYTES)
        pool_f32 = pool.view("float32")
        pool_u64 = pool.view("uint64")
        pool_u32 = pool.view("uint32")
        pool_base = txl.local_scalar("uint32", init=_u32(txl.cuda.cvta_generic_to_shared(pool.ptr_to([0]))))

        def b16(byte):
            return pool.ptr_to([byte // 2])

        def f32p(byte):
            return pool_f32.ptr_to([byte // 4])

        def bar(byte):
            return pool_u64.ptr_to([byte // 8])

        def bar_s(byte_fn, s):
            """Barrier of runtime stage s (0 / 1)."""
            return pool_u64.ptr_to([(byte_fn(0) // 8) + s])

        thread_idx = txl.thread_id()
        lane_pred = txl.local_scalar("uint32")
        lane_id_e = txl.local_scalar("uint32")
        txl.ptx.elect_sync(lane_id_e, lane_pred, txl.uint32(0xFFFFFFFF))  # K:166
        warp = txl.local_scalar("uint32")
        txl.ptx.shfl_sync.idx.b32(
            warp, _u32(thread_idx) // txl.uint32(32), txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF)
        )
        with txl.if_(txl.And(warp == txl.uint32(0), lane_pred == txl.uint32(1))), txl.then_():
            # The GQA epilogue prefetches nothing (CollectiveEpilogueBwdGQA::prefetch_tma_descriptors).
            for m in (tma_q, tma_do, tma_k, tma_v) + (() if gqa else (tma_dk, tma_dv)):
                txl.ptx.prefetch.tensormap(txl.address_of(m))
        wg = txl.local_scalar("uint32")
        txl.ptx.shfl_sync.idx.b32(
            wg, _u32(thread_idx) // txl.uint32(128), txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF)
        )
        with txl.if_(txl.And(warp == txl.uint32(0), lane_pred == txl.uint32(1))), txl.then_():
            txl.ptx["mbarrier.init.shared::cta.b64"](bar(B_KV), txl.uint32(1))
        # PipelineTmaAsync constructors (pipeline_q, then pipeline_do): warp 0's elected lane.
        for full_fn, empty_fn in ((b_full_q, b_empty_q), (b_full_do, b_empty_do)):
            e1 = txl.local_scalar("uint32")
            e1_lane = txl.local_scalar("uint32")
            txl.ptx.elect_sync(e1_lane, e1, txl.uint32(0xFFFFFFFF))
            with txl.if_(txl.And(warp == txl.uint32(0), e1 == txl.uint32(1))), txl.then_():
                for s in range(2):
                    txl.ptx["mbarrier.init.shared::cta.b64"](bar(full_fn(s)), txl.uint32(1))
                    txl.ptx["mbarrier.init.shared::cta.b64"](bar(empty_fn(s)), txl.uint32(NUM_MMA_THREADS))
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        # ---------------- scheduler (tile_scheduler.hpp) and m-block range (block.h:83) -------------
        def fdiv(x, div, mul, shr):
            """cutlass::FastDivmod::divide (x >= 0): ``divisor != 1 ? umulhi(x, m) >> s : x``, a branch
            around the multiply-high as nvcc emits it."""
            q = txl.local_scalar("int32")
            with txl.if_(div != 1):
                with txl.then_():
                    hi = txl.local_scalar("uint32")
                    txl.ptx["mul.hi.u32"](hi, _u32(x), mul)
                    txl.assign(q, _i32(hi >> shr))
                with txl.else_():
                    txl.assign(q, x)
            return q

        def tile():
            if is_causal:
                tile_idx = txl.cuda.block_idx("x")
                bidhb = txl.local_scalar("int32", init=fdiv(tile_idx, l2maj_div, l2maj_mul, l2maj_shr))
                l2_mod = txl.local_scalar("int32", init=tile_idx - bidhb * l2maj_div)
                nb = txl.local_scalar("int32")
                resid = txl.local_scalar("int32")
                with txl.if_(bidhb < num_hb_quotient):
                    with txl.then_():
                        txl.assign(nb, fdiv(l2_mod, l2min_div, l2min_mul, l2min_shr))
                        txl.assign(resid, l2_mod - nb * l2min_div)
                    with txl.else_():
                        txl.assign(nb, fdiv(l2_mod, l2res_div, l2res_mul, l2res_shr))
                        txl.assign(resid, l2_mod - nb * l2res_div)
                x = txl.local_scalar("int32", init=bidhb * l2min_div + resid)
                bb = txl.local_scalar("int32", init=fdiv(x, head_div, head_mul, head_shr))
                hh = txl.local_scalar("int32", init=x - bb * head_div)
                return nb, hh, bb
            return txl.cuda.block_idx("x"), txl.cuda.block_idx("y"), txl.cuda.block_idx("z")

        m_max_c = (s_q + BM - 1) // BM

        def m_min_of(n_block):
            if not is_causal:
                return txl.int32(0)
            return txl.max(0, txl.truncdiv(n_block * BLOCK_N + s_q - s_k - window_size_right, BM))

        s_q_rounded = (s_q + BM - 1) // BM * BM

        def stat_base(bidb, bidh):  # (b, h, s_q_rounded) float index
            return _i64(bidb * h + bidh) * _i64(s_q_rounded)

        def wait(bar_ptr, phase):
            txl.cuda.mbarrier_wait_parity(bar_ptr, _u32(phase), txl.uint32(TRY_WAIT_TICKS))

        def advance(stage, phase):
            with txl.if_(stage == 1):
                with txl.then_():
                    txl.assign(stage, 0)
                    txl.assign(phase, phase ^ 1)
                with txl.else_():
                    txl.assign(stage, 1)

        with txl.if_(wg == txl.uint32(0)):
            # =============================== producer warpgroup ===============================
            with txl.then_():
                txl.ptx["setmaxnreg.dec.sync.aligned.u32"](txl.uint32(24))
                wiw = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.idx.b32(
                    wiw, (_u32(thread_idx) // txl.uint32(32)) % txl.uint32(4), txl.uint32(0),
                    txl.uint32(31), txl.uint32(0xFFFFFFFF),
                )
                with txl.if_(wiw == txl.uint32(0)):
                    with txl.then_():
                        _producer_load(locals_=dict(
                            tile=tile, m_min_of=m_min_of, m_max_c=m_max_c, wait=wait, advance=advance,
                            bar=bar, bar_s=bar_s, b_full_q=b_full_q, b_empty_q=b_empty_q,
                            b_full_do=b_full_do, b_empty_do=b_empty_do, B_KV=B_KV, b16=b16, f32p=f32p,
                            OFF=OFF, Q_TILE=Q_TILE, HALF=HALF, TX_Q=TX_Q, LSE_STRIDE=LSE_STRIDE, BM=BM,
                            tma_q=tma_q, tma_do=tma_do, tma_k=tma_k, tma_v=tma_v, lse_log2=lse_log2,
                            dpsum=dpsum, stat_base=stat_base, thread_idx=thread_idx, is_causal=is_causal,
                            h=h, s_q_rounded=s_q_rounded, fdiv=fdiv, qhkh=(qhkh_div, qhkh_mul, qhkh_shr),
                        ))
                    with txl.else_():
                        with txl.if_(wiw == txl.uint32(1)), txl.then_():
                            _dq_store(locals_=dict(
                                tile=tile, m_min_of=m_min_of, m_max_c=m_max_c, OFF=OFF, BM=BM, b16=b16,
                                f32p=f32p, dq_accum=dq_accum, s_q_rounded=s_q_rounded, h=h,
                                is_causal=is_causal,
                            ))
            # ================================ MMA warpgroups ==================================
            with txl.else_():
                txl.ptx["setmaxnreg.inc.sync.aligned.u32"](txl.uint32(240))
                _mma(locals_=dict(
                    tile=tile, m_min_of=m_min_of, m_max_c=m_max_c, wait=wait, advance=advance, bar=bar,
                    bar_s=bar_s, b_full_q=b_full_q, b_empty_q=b_empty_q, b_full_do=b_full_do,
                    b_empty_do=b_empty_do, B_KV=B_KV, b16=b16, f32p=f32p, pool_base=pool_base, OFF=OFF,
                    Q_TILE=Q_TILE, HALF=HALF, LSE_STRIDE=LSE_STRIDE, BM=BM, NC=NC, KM=KM, NQ=NQ,
                    wg_s=wg_s, wg_q=wg_q, wg_kv=wg_kv, thread_idx=thread_idx, wg=wg, s_q=s_q, s_k=s_k,
                    d=d, h=h, softmax_scale=softmax_scale, scale_log2=scale_log2, tma_dk=tma_dk,
                    tma_dv=tma_dv, is_causal=is_causal, dk_out=dk, dv_out=dv, b=b, gqa=gqa, fdiv=fdiv,
                    qhkh=(qhkh_div, qhkh_mul, qhkh_shr), dk_accum=dk_accum, dv_accum=dv_accum, h_k=h_k,
                ))


    if not gqa:

        @txl.kernel()
        def flash_bwd_sm90(
            lse_log2: txl.Tensor((txl.int64(),), "float32"),
            dpsum: txl.Tensor((txl.int64(),), "float32"),
            dq_accum: txl.Tensor((txl.int64(),), "float32"),
            dk: txl.Tensor((txl.int64(),), "bfloat16"),
            dv: txl.Tensor((txl.int64(),), "bfloat16"),
            s_q: "int32",
            s_k: "int32",
            d: "int32",
            h: "int32",
            b: "int32",
            window_size_right: "int32",
            softmax_scale: "float32",
            scale_log2: "float32",
            lpt_total: "int32",
            blk_div: "int32",
            blk_mul: "uint32",
            blk_shr: "uint32",
            head_div: "int32",
            head_mul: "uint32",
            head_shr: "uint32",
            l2min_div: "int32",
            l2min_mul: "uint32",
            l2min_shr: "uint32",
            l2maj_div: "int32",
            l2maj_mul: "uint32",
            l2maj_shr: "uint32",
            l2res_div: "int32",
            l2res_mul: "uint32",
            l2res_shr: "uint32",
            num_hb_quotient: "int32",
            qhkh_div: "int32",
            qhkh_mul: "uint32",
            qhkh_shr: "uint32",
            tma_q: txl.Ptr(txl.TensorMap),
            tma_do: txl.Ptr(txl.TensorMap),
            tma_k: txl.Ptr(txl.TensorMap),
            tma_v: txl.Ptr(txl.TensorMap),
            tma_dk: txl.Ptr(txl.TensorMap),
            tma_dv: txl.Ptr(txl.TensorMap),
        ):
            _body(lse_log2, dpsum, dq_accum, dk, dv, None, None, s_q, s_k, d, h, None, b, window_size_right,
                  softmax_scale, scale_log2, lpt_total, blk_div, blk_mul, blk_shr, head_div, head_mul, head_shr, l2min_div, l2min_mul, l2min_shr, l2maj_div, l2maj_mul, l2maj_shr, l2res_div, l2res_mul, l2res_shr, num_hb_quotient,
                  qhkh_div, qhkh_mul, qhkh_shr, tma_q, tma_do, tma_k, tma_v, tma_dk, tma_dv)

    else:

        @txl.kernel()
        def flash_bwd_sm90(
            lse_log2: txl.Tensor((txl.int64(),), "float32"),
            dpsum: txl.Tensor((txl.int64(),), "float32"),
            dq_accum: txl.Tensor((txl.int64(),), "float32"),
            dk_accum: txl.Tensor((txl.int64(),), "float32"),
            dv_accum: txl.Tensor((txl.int64(),), "float32"),
            s_q: "int32",
            s_k: "int32",
            d: "int32",
            h: "int32",
            h_k: "int32",
            b: "int32",
            window_size_right: "int32",
            softmax_scale: "float32",
            scale_log2: "float32",
            lpt_total: "int32",
            blk_div: "int32",
            blk_mul: "uint32",
            blk_shr: "uint32",
            head_div: "int32",
            head_mul: "uint32",
            head_shr: "uint32",
            l2min_div: "int32",
            l2min_mul: "uint32",
            l2min_shr: "uint32",
            l2maj_div: "int32",
            l2maj_mul: "uint32",
            l2maj_shr: "uint32",
            l2res_div: "int32",
            l2res_mul: "uint32",
            l2res_shr: "uint32",
            num_hb_quotient: "int32",
            qhkh_div: "int32",
            qhkh_mul: "uint32",
            qhkh_shr: "uint32",
            tma_q: txl.Ptr(txl.TensorMap),
            tma_do: txl.Ptr(txl.TensorMap),
            tma_k: txl.Ptr(txl.TensorMap),
            tma_v: txl.Ptr(txl.TensorMap),
        ):
            _body(lse_log2, dpsum, dq_accum, None, None, dk_accum, dv_accum, s_q, s_k, d, h, h_k, b,
                  window_size_right, softmax_scale, scale_log2, lpt_total, blk_div, blk_mul, blk_shr, head_div, head_mul, head_shr, l2min_div, l2min_mul, l2min_shr, l2maj_div, l2maj_mul, l2maj_shr, l2res_div, l2res_mul, l2res_shr, num_hb_quotient,
                  qhkh_div, qhkh_mul, qhkh_shr, tma_q, tma_do, tma_k, tma_v, None, None)

    return flash_bwd_sm90.func


def _tma_load(dst, tmap, c0, c1, c2, c3, bar_ptr, hint):
    txl.ptx[_TMA_LOAD](dst, txl.address_of(tmap), _i32(c0), _i32(c1), _i32(c2), _i32(c3), bar_ptr,
                       txl.uint64(hint))


def _producer_load(locals_):
    """Producer warp 0: CollectiveMainloopBwdSm90::load + load_tail (mainloop :420-591)."""
    L = locals_
    BM, OFF = L["BM"], L["OFF"]
    bar, bar_s, wait, advance = L["bar"], L["bar_s"], L["wait"], L["advance"]
    stage = txl.local_scalar("int32", init=txl.int32(0))
    phase = txl.local_scalar("int32", init=txl.int32(1))  # make_producer_start_state
    n_block, bidh, bidb = L["tile"]()
    m_min = txl.local_scalar("int32", init=L["m_min_of"](n_block))
    m_max = L["m_max_c"]
    leader = L["thread_idx"] % 128 == 0  # PipelineParams::is_leader
    def load_q(m, stat_row):
        q_dst = OFF["q"] + stage * L["Q_TILE"]
        for half in range(2):
            _tma_load(L["b16"](q_dst + half * L["HALF"]), L["tma_q"], 64 * half, m * BM, bidh, bidb,
                      bar_s(L["b_full_q"], stage), EVICT_LAST)
        txl.ptx[_BULK_G2S](
            L["f32p"](OFF["lse"] + stage * (L["LSE_STRIDE"] * 4)),
            L["lse_log2"].ptr_to([stat_row + _i64(m * BM)]),
            txl.uint32(BM * 4),
            bar_s(L["b_full_q"], stage),
        )

    def load_do(m, stat_row):
        do_dst = OFF["do"] + stage * L["Q_TILE"]
        for half in range(2):
            _tma_load(L["b16"](do_dst + half * L["HALF"]), L["tma_do"], 64 * half, m * BM, bidh, bidb,
                      bar_s(L["b_full_do"], stage), EVICT_LAST)
        txl.ptx[_BULK_G2S](
            L["f32p"](OFF["dpsum"] + stage * (L["LSE_STRIDE"] * 4)),
            L["dpsum"].ptr_to([stat_row + _i64(m * BM)]),
            txl.uint32(BM * 4),
            bar_s(L["b_full_do"], stage),
        )

    def acquire(empty_fn, full_fn):
        wait(bar_s(empty_fn, stage), phase)
        with txl.if_(leader), txl.then_():
            txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](bar_s(full_fn, stage), txl.uint32(L["TX_Q"]))

    def body():
        # qhead_per_khead_divmod.divide(bidh) (mainloop :456, after load()'s causal early return): the KV
        # head of the K / V loads (divisor 1 for MHA, as the source runs it).
        bidh_kv = txl.local_scalar("int32", init=L["fdiv"](bidh, *L["qhkh"]))
        # gLSE / gdPsum row base of this (bidb, bidh), int64 as upstream's strides (local_tile of mLSE /
        # mdPsum). The 24-register producer is sensitive to how it is held: causal binds it once per
        # tile (re-deriving it spilled 32 B), non-causal re-derives it per use (binding it spilled 4 B
        # and made ptxas rematerialize threadIdx in the MMA loop). Both forms are spill-free. The
        # sensitivity is a codegen issue: sm90-probe ISSUE-022 (A).
        if L["is_causal"]:
            stat_row = txl.local_scalar("int64", init=L["stat_base"](bidb, bidh))
        else:
            stat_row = L["stat_base"](bidb, bidh)
        lane = txl.local_scalar("uint32")
        lane_e = txl.local_scalar("uint32")
        txl.ptx.elect_sync(lane_e, lane, txl.uint32(0xFFFFFFFF))
        with txl.if_(lane == txl.uint32(1)), txl.then_():
            acquire(L["b_empty_q"], L["b_full_q"])
            load_q(m_min, stat_row)
        with txl.if_(lane == txl.uint32(1)), txl.then_():
            txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](bar(L["B_KV"]), txl.uint32(65536))
            for half in range(2):
                _tma_load(L["b16"](OFF["k"] + half * 16384), L["tma_k"], 64 * half, n_block * BLOCK_N,
                          bidh_kv, bidb, bar(L["B_KV"]), EVICT_NORMAL)
            for half in range(2):
                _tma_load(L["b16"](OFF["v"] + half * 16384), L["tma_v"], 64 * half, n_block * BLOCK_N,
                          bidh_kv, bidb, bar(L["B_KV"]), EVICT_NORMAL)
            with txl.serial(m_min, m_max - 1, unroll=2) as m:
                acquire(L["b_empty_do"], L["b_full_do"])
                load_do(m, stat_row)
                advance(stage, phase)
                acquire(L["b_empty_q"], L["b_full_q"])
                load_q(m + 1, stat_row)
        with txl.if_(lane == txl.uint32(1)), txl.then_():
            acquire(L["b_empty_do"], L["b_full_do"])
            load_do(m_max - 1, stat_row)
            advance(stage, phase)

    if L["is_causal"]:
        with txl.if_(m_max > m_min), txl.then_():
            body()
    else:
        body()
    # load_tail: producer_tail of pipeline_q, then of pipeline_do, from the same state.
    lt = txl.local_scalar("uint32")
    lt_e = txl.local_scalar("uint32")
    txl.ptx.elect_sync(lt_e, lt, txl.uint32(0xFFFFFFFF))
    with txl.if_(lt == txl.uint32(1)), txl.then_():
        stage_do = txl.local_scalar("int32", init=stage)
        phase_do = txl.local_scalar("int32", init=phase)
        for _ in range(2):
            wait(bar_s(L["b_empty_q"], stage), phase)
            advance(stage, phase)
        for _ in range(2):
            wait(bar_s(L["b_empty_do"], stage_do), phase_do)
            advance(stage_do, phase_do)


def _dq_store(locals_):
    """Producer warp 1: CollectiveMainloopBwdSm90::store_dq (mainloop :593-667)."""
    L = locals_
    BM, OFF = L["BM"], L["OFF"]
    n_block, bidh, bidb = L["tile"]()
    m_min = txl.local_scalar("int32", init=L["m_min_of"](n_block))
    m_max = L["m_max_c"]
    half_bytes = BM * 64 * 4

    def body():
        lane = txl.local_scalar("uint32")
        lane_e = txl.local_scalar("uint32")
        txl.ptx.elect_sync(lane_e, lane, txl.uint32(0xFFFFFFFF))
        base = _i64(bidb * L["h"] + bidh) * _i64(L["s_q_rounded"] * HEAD_DIM)
        with txl.serial(m_min, m_max, unroll=2) as m:
            for w in range(2):
                txl.ptx["bar.sync"](txl.uint32(BAR_DQ_FULL + w), txl.uint32(160))
                with txl.if_(lane == txl.uint32(1)), txl.then_():
                    txl.ptx[_BULK_REDUCE](
                        L["dq_accum"].ptr_to([base + _i64(m * (BM * HEAD_DIM) + w * (BM * 64))]),
                        L["f32p"](OFF["dqacc"] + w * half_bytes),
                        txl.uint32(half_bytes),
                        txl.uint64(EVICT_LAST),
                    )
                    txl.ptx.cp.async_.bulk.commit_group()
            for w in range(2):
                with txl.if_(lane == txl.uint32(1)), txl.then_():
                    txl.ptx.cp.async_.bulk.wait_group.read(1 - w)
                txl.ptx["bar.arrive"](txl.uint32(BAR_DQ_EMPTY + w), txl.uint32(160))

    if L["is_causal"]:
        with txl.if_(m_max > m_min), txl.then_():
            body()
    else:
        body()


def _mma(locals_):
    """MMA warpgroups: mma_init, CollectiveMainloopBwdSm90::mma (bwd_step), the dK scale, and
    CollectiveEpilogueBwd::store / store_zero."""
    L = locals_
    BM, OFF, NC, KM, NQ = L["BM"], L["OFF"], L["NC"], L["KM"], L["NQ"]
    is_causal = L["is_causal"]
    bar, bar_s, wait, advance, b16 = L["bar"], L["bar_s"], L["wait"], L["advance"], L["b16"]
    thread_idx = L["thread_idx"]
    tid = txl.local_scalar("int32", init=thread_idx - 128)  # thread_idx - NumCopyThreads
    w = txl.local_scalar("uint32", init=L["wg"] - txl.uint32(1))  # MMA warpgroup 0 / 1
    t = tid % 128
    warp4 = t // 32
    lane = t % 32
    r0 = warp4 * 16 + lane // 4  # first C-fragment row of this thread
    q0 = 2 * (lane % 4)  # first C-fragment column offset

    # mma_init: warp 0 of each MMA warpgroup arrives on dQEmpty.
    wiw = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(
        wiw, (_u32(thread_idx) // txl.uint32(32)) % txl.uint32(4), txl.uint32(0), txl.uint32(31),
        txl.uint32(0xFFFFFFFF),
    )
    with txl.if_(wiw == txl.uint32(0)), txl.then_():
        txl.ptx["bar.arrive"](BAR_DQ_EMPTY - 1 + L["wg"], txl.uint32(160))

    n_block, bidh, bidb = L["tile"]()
    m_min = txl.local_scalar("int32", init=L["m_min_of"](n_block))
    m_max = L["m_max_c"]

    def mainloop_and_epilogue():
        stage = txl.local_scalar("int32", init=txl.int32(0))
        phase = txl.local_scalar("int32", init=txl.int32(0))
        rdK = txl.alloc_local((64,), "float32")
        rdV = txl.alloc_local((64,), "float32")
        for i in range(64):
            txl.assign(rdK[i], txl.float32(0.0))
            txl.assign(rdV[i], txl.float32(0.0))
        # barrier_KV: try_wait (work_idx % 2 == 0), then the blocking wait if not ready.
        _consumer_wait(bar(L["B_KV"]), txl.int32(0))
        with txl.serial(m_min, m_max, unroll=False) as m:
            _bwd_step(L, m, stage, phase, rdK, rdV, n_block, w, tid, r0, q0, warp4, lane)
        for i in range(64):
            txl.ptx["mul.ftz.f32"](rdK[i], rdK[i], L["softmax_scale"])
        if L["gqa"]:
            _epilogue_store_gqa(L, rdK, rdV, n_block, bidh, bidb, w, tid)
        else:
            _epilogue_store(L, rdK, rdV, n_block, bidh, bidb, w, tid, r0, warp4, lane)

    if is_causal:
        if L["gqa"]:  # CollectiveEpilogueBwdGQA::store_zero is empty (the accumulators are zero-filled)
            with txl.if_(m_max > m_min), txl.then_():
                mainloop_and_epilogue()
        else:
            with txl.if_(m_max <= m_min):
                with txl.then_():
                    _store_zero(L, n_block, bidh, bidb, tid)
                with txl.else_():
                    mainloop_and_epilogue()
    else:
        mainloop_and_epilogue()


def _consumer_wait(bar_ptr, phase):
    """consumer_try_wait (no suspend hint) + consumer_wait (retry loop with the hint)."""
    ready = txl.local_scalar("uint32")
    txl.ptx["mbarrier.try_wait.parity.shared::cta.b64"](ready, bar_ptr, _u32(phase))
    with txl.if_(ready == txl.uint32(0)), txl.then_():
        txl.cuda.mbarrier_wait_parity(bar_ptr, _u32(phase), txl.uint32(TRY_WAIT_TICKS))


def _desc(pool_base, byte, lbo_field):
    """64-bit GMMA descriptor of the operand at pool byte offset ``byte`` (an expression), rebuilt per
    k-step. Advancing a base descriptor as cute does (``desc.lo += offset >> 4``) is shorter in SASS but
    measured up to 1.4 % slower on this stack (sm90-probe ISSUE-022 (B)), so the rebuild stays."""
    lo = ((pool_base + _u32(byte)) >> txl.uint32(4)) & txl.uint32(0x3FFF) | txl.uint32(lbo_field << 16)
    return (txl.uint64(DESC_HI) << txl.uint64(32)) | txl.cast(lo, "uint64")


def _kmajor_step(base, k, half):
    return base + (k // 4) * half + (k % 4) * 32


def _fence(regs, n):
    for i in range(n):
        txl.cuda.wgmma.noop_barrier(regs[i])


def _bwd_step(L, m, stage, phase, rdK, rdV, n_block, w, tid, r0, q0, warp4, lane):
    """bwd_step (mainloop :834-1001) for one m_block."""
    BM, OFF, NC, KM, NQ = L["BM"], L["OFF"], L["NC"], L["KM"], L["NQ"]
    is_causal = L["is_causal"]
    bar_s, advance, b16, f32p = L["bar_s"], L["advance"], L["b16"], L["f32p"]
    pb = L["pool_base"]
    HALF, Q_TILE = L["HALF"], L["Q_TILE"]
    NS = BM // 2  # S^T / dP^T fragment registers
    q_off = OFF["q"] + stage * Q_TILE
    do_off = OFF["do"] + stage * Q_TILE
    ds_off = OFF["ds"] + stage * Q_TILE
    lse_off = OFF["lse"] + stage * (L["LSE_STRIDE"] * 4)
    dps_off = OFF["dpsum"] + stage * (L["LSE_STRIDE"] * 4)
    w_i = _i32(w)

    # ---- S^T = K Q^T (ss) ----
    rS = txl.alloc_local((NS,), "float32")
    _consumer_wait(bar_s(L["b_full_q"], stage), phase)
    _fence(rS, NS)
    txl.ptx.wgmma.fence.sync.aligned()
    k_base = OFF["k"] + w_i * (64 * 128)
    v_base = OFF["v"] + w_i * (64 * 128)
    for k in range(8):
        txl.ptx[L["wg_s"]](
            *[rS[j] for j in range(NS)],
            _desc(pb, _kmajor_step(k_base, k, 16384), 1),
            _desc(pb, _kmajor_step(q_off, k, HALF), 1),
            k > 0, 1, 1, 0, 0,
        )
    txl.ptx.wgmma.commit_group.sync.aligned()
    _fence(rS, NS)
    rLSE = txl.alloc_local((2 * NC,), "float32")
    for c in range(NC):
        txl.ptx.ld.shared.v2.f32(rLSE[2 * c], rLSE[2 * c + 1], f32p(lse_off + (8 * c + q0) * 4))
    # ---- dP^T = V dO^T (ss) ----
    rdP = txl.alloc_local((NS,), "float32")
    _consumer_wait(bar_s(L["b_full_do"], stage), phase)
    _fence(rdP, NS)
    txl.ptx.wgmma.fence.sync.aligned()
    for k in range(8):
        txl.ptx[L["wg_s"]](
            *[rdP[j] for j in range(NS)],
            _desc(pb, _kmajor_step(v_base, k, 16384), 1),
            _desc(pb, _kmajor_step(do_off, k, HALF), 1),
            k > 0, 1, 1, 0, 0,
        )
    txl.ptx.wgmma.commit_group.sync.aligned()
    _fence(rdP, NS)
    txl.ptx.wgmma.wait_group.sync.aligned(1)
    _fence(rS, NS)

    # ---- mask (mask.h, SwapAB): seqlen_k always, causal when causal ----
    thread_col_offset = w_i * 64 + r0  # first KV row (Col coordinate) of this thread
    seqlenk_col_limit = txl.local_scalar("int32", init=L["s_k"] - n_block * BLOCK_N - thread_col_offset)
    if is_causal:
        causal_row_offset = txl.local_scalar(
            "int32", init=seqlenk_col_limit - L["s_q"] + m * BM + q0
        )
        for i in range(2):
            col0 = 8 * i
            lim = txl.local_scalar(
                "int32",
                init=txl.if_then_else(col0 >= seqlenk_col_limit, BM, col0 - causal_row_offset),
            )
            for c in range(NC):
                for e in range(2):
                    reg = 4 * c + 2 * i + e
                    txl.assign(
                        rS[reg],
                        txl.if_then_else(8 * c + e < lim, txl.float32(-float("inf")), rS[reg]),
                    )
    else:
        for i in range(2):
            for c in range(NC):
                for e in range(2):
                    reg = 4 * c + 2 * i + e
                    txl.assign(
                        rS[reg],
                        txl.if_then_else(8 * i >= seqlenk_col_limit, txl.float32(-float("inf")), rS[reg]),
                    )
    # ---- P^T = exp2(S^T * scale_log2 - LSE) ----
    for c in range(NC):
        for e in range(2):
            for i in range(2):
                reg = 4 * c + 2 * i + e
                tmp = txl.local_scalar("float32")
                txl.ptx["mul.ftz.f32"](tmp, rS[reg], L["scale_log2"])
                txl.ptx["sub.ftz.f32"](tmp, tmp, rLSE[2 * c + e])
                txl.ptx["ex2.approx.ftz.f32"](rS[reg], tmp)
    rdPsum = txl.alloc_local((2 * NC,), "float32")
    for c in range(NC):
        txl.ptx.ld.shared.v2.f32(rdPsum[2 * c], rdPsum[2 * c + 1], f32p(dps_off + (8 * c + q0) * 4))
    txl.ptx.wgmma.wait_group.sync.aligned(0)
    _fence(rdP, NS)
    # ---- dS^T = P^T (dP^T - dPsum) ----
    for c in range(NC):
        for e in range(2):
            for i in range(2):
                reg = 4 * c + 2 * i + e
                tmp = txl.local_scalar("float32")
                txl.ptx["sub.ftz.f32"](tmp, rdP[reg], rdPsum[2 * c + e])
                txl.ptx["mul.ftz.f32"](rdP[reg], rS[reg], tmp)
    rP = txl.alloc_local((NS // 2,), "uint32")
    rdS = txl.alloc_local((NS // 2,), "uint32")
    for wd in range(NS // 2):
        txl.ptx["cvt.rn.bf16x2.f32"](rP[wd], rS[2 * wd + 1], rS[2 * wd])
    for wd in range(NS // 2):
        txl.ptx["cvt.rn.bf16x2.f32"](rdS[wd], rdP[2 * wd + 1], rdP[2 * wd])
    # ---- dS^T -> sdS (stmatrix.trans): matrix j of instruction p is (c = 2p + j // 2, i = j % 2) ----
    kk = lane % 8
    jm = lane // 8
    for p in range(NC // 2):
        c_j = 2 * p + jm // 2
        i_j = jm % 2
        q_row = c_j * 8 + kk  # sdS row (q)
        kv_chunk = warp4 * 2 + i_j  # 16-byte chunk (8 KV columns) within the warpgroup's 64-column half
        byte = ds_off + w_i * HALF + _sw128(q_row, kv_chunk)
        txl.ptx[_STSM_T](b16(byte), *[rdS[4 * p + j] for j in range(4)])
    # ---- dV += P^T dO (rs) ----
    _fence(rP, NS // 2)
    _fence(rdV, 64)
    txl.ptx.wgmma.fence.sync.aligned()
    for k in range(KM):
        txl.ptx[L["wg_kv"]](
            *[rdV[j] for j in range(64)],
            *[rP[4 * k + j] for j in range(4)],
            _desc(pb, do_off + k * 2048, HALF >> 4),
            True, 1, 1, 1,
        )
    txl.ptx.wgmma.commit_group.sync.aligned()
    _fence(rdV, 64)
    _fence(rP, NS // 2)
    txl.ptx.fence.proxy.async_.shared__cta()
    txl.ptx["bar.sync"](txl.uint32(BAR_PDS), txl.uint32(NUM_MMA_THREADS))
    # ---- dQ = dS K (causal) / dQ^T = K^T dS^T (non-causal), ss ----
    rdQ = txl.alloc_local((NQ,), "float32")
    _fence(rdQ, NQ)
    txl.ptx.wgmma.fence.sync.aligned()
    k_half = OFF["k"] + w_i * 16384
    for k in range(8):
        if is_causal:
            txl.ptx[L["wg_q"]](
                *[rdQ[j] for j in range(NQ)],
                _desc(pb, _kmajor_step(ds_off, k, HALF), 1),
                _desc(pb, k_half + k * 2048, 0),
                k > 0, 1, 1, 0, 1,
            )
        else:
            txl.ptx[L["wg_q"]](
                *[rdQ[j] for j in range(NQ)],
                _desc(pb, k_half + k * 2048, 0),
                _desc(pb, _kmajor_step(ds_off, k, HALF), 1),
                k > 0, 1, 1, 1, 0,
            )
    txl.ptx.wgmma.commit_group.sync.aligned()
    _fence(rdQ, NQ)
    txl.ptx.wgmma.wait_group.sync.aligned(1)
    _fence(rdV, 64)
    _release(bar_s(L["b_empty_do"], stage))
    # ---- dK += dS^T Q (rs) ----
    _fence(rdS, NS // 2)
    _fence(rdK, 64)
    txl.ptx.wgmma.fence.sync.aligned()
    for k in range(KM):
        txl.ptx[L["wg_kv"]](
            *[rdK[j] for j in range(64)],
            *[rdS[4 * k + j] for j in range(4)],
            _desc(pb, q_off + k * 2048, HALF >> 4),
            True, 1, 1, 1,
        )
    txl.ptx.wgmma.commit_group.sync.aligned()
    _fence(rdK, 64)
    _fence(rdS, NS // 2)
    txl.ptx.wgmma.wait_group.sync.aligned(1)
    _fence(rdQ, NQ)
    # ---- dQ -> sdQacc[w] (R2S 128 threads x float4), then hand over to warp 1 ----
    txl.ptx["bar.sync"](txl.cast(BAR_DQ_EMPTY + w_i, "uint32"), txl.uint32(160))
    t = tid % 128
    for i in range(NQ // 4):
        txl.ptx.st.shared.v4.f32(
            f32p(OFF["dqacc"] + w_i * (BM * 256) + (i * 128 + t) * 16),
            rdQ[4 * i], rdQ[4 * i + 1], rdQ[4 * i + 2], rdQ[4 * i + 3],
        )
    txl.ptx.fence.proxy.async_.shared__cta()
    txl.ptx["bar.arrive"](txl.cast(BAR_DQ_FULL + w_i, "uint32"), txl.uint32(160))
    txl.ptx.wgmma.wait_group.sync.aligned(0)
    _fence(rdK, 64)
    _release(bar_s(L["b_empty_q"], stage))
    advance(stage, phase)


def _release(bar_ptr):
    """ClusterBarrier::arrive(cta_id = 0): mapa + mbarrier.arrive.shared::cluster, every thread."""
    remote = txl.local_scalar("uint32")
    txl.ptx["mapa.shared::cluster.u32"](remote, _u32(txl.cuda.cvta_generic_to_shared(bar_ptr)), txl.uint32(0))
    txl.ptx["mbarrier.arrive.shared::cluster.b64"](remote)


def _epilogue_store(L, rdK, rdV, n_block, bidh, bidb, w, tid, r0, warp4, lane):
    """CollectiveEpilogueBwd::store, Use_TMA (epilogue_bwd.hpp:163-235)."""
    OFF, b16 = L["OFF"], L["b16"]
    w_i = _i32(w)
    rV16 = txl.alloc_local((32,), "uint32")
    rK16 = txl.alloc_local((32,), "uint32")
    for wd in range(32):
        txl.ptx["cvt.rn.bf16x2.f32"](rV16[wd], rdV[2 * wd + 1], rdV[2 * wd])
    for wd in range(32):
        txl.ptx["cvt.rn.bf16x2.f32"](rK16[wd], rdK[2 * wd + 1], rdK[2 * wd])
    txl.ptx["bar.sync"](txl.uint32(BAR_EPILOGUE), txl.uint32(NUM_MMA_THREADS))
    kk = lane % 8
    jm = lane // 8
    for base_off, regs in ((OFF["v"], rV16), (OFF["k"], rK16)):  # dV first, then dK
        for p in range(8):
            c_j = 2 * p + jm // 2
            i_j = jm % 2
            row = w_i * 64 + warp4 * 16 + i_j * 8 + kk  # KV row
            byte = base_off + (c_j // 8) * 16384 + _sw128(row, c_j % 8)
            txl.ptx[_STSM](b16(byte), *[regs[4 * p + j] for j in range(4)])
    txl.ptx.fence.proxy.async_.shared__cta()
    txl.ptx["bar.arrive"](txl.uint32(BAR_EPILOGUE), txl.uint32(NUM_MMA_THREADS + 32))
    ws = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(
        ws, _u32(tid) // txl.uint32(32), txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF)
    )
    with txl.if_(ws == txl.uint32(NUM_MMA_THREADS // 32 - 1)), txl.then_():
        txl.ptx["bar.sync"](txl.uint32(BAR_EPILOGUE), txl.uint32(NUM_MMA_THREADS + 32))
        el = txl.local_scalar("uint32")
        el_lane = txl.local_scalar("uint32")
        txl.ptx.elect_sync(el_lane, el, txl.uint32(0xFFFFFFFF))
        with txl.if_(el == txl.uint32(1)), txl.then_():
            for tmap, base_off in ((L["tma_dv"], OFF["v"]), (L["tma_dk"], OFF["k"])):
                for half in range(2):
                    txl.ptx[_TMA_STORE](
                        txl.address_of(tmap), txl.int32(64 * half), _i32(n_block * BLOCK_N),
                        _i32(bidh), _i32(bidb), b16(base_off + half * 16384),
                    )
            txl.ptx.cp.async_.bulk.commit_group()
    txl.ptx.cp.async_.bulk.wait_group.read(0)


def _epilogue_store_gqa(L, rdK, rdV, n_block, bidh, bidb, w, tid):
    """CollectiveEpilogueBwdGQA::store, Use_TMA, not Deterministic (epilogue_bwd.hpp:414-518): the fp32
    dV, then dK, fragment of each MMA warpgroup into sdKV (R2STiledCopydKVaccum: 128 threads x float4 per
    warpgroup half), and one 64 KiB bulk reduce-add of the whole tile into the (b, h_kv) accumulator."""
    OFF, f32p = L["OFF"], L["f32p"]
    w_i = _i32(w)
    t = tid % 128
    bidh_kv = txl.local_scalar("int32", init=L["fdiv"](bidh, *L["qhkh"]))
    s_k_rounded = (L["s_k"] + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    tile = (_i64(bidb * L["h_k"] + bidh_kv) * _i64(s_k_rounded) + _i64(n_block * BLOCK_N)) * _i64(HEAD_DIM)
    txl.ptx["bar.sync"](txl.uint32(BAR_EPILOGUE), txl.uint32(NUM_MMA_THREADS))  # K and V reads done
    for regs, accum in ((rdV, L["dv_accum"]), (rdK, L["dk_accum"])):
        if regs is rdK:
            txl.ptx["bar.sync"](txl.uint32(BAR_EPILOGUE), txl.uint32(NUM_MMA_THREADS))  # sdKV free again
        for i in range(16):
            txl.ptx.st.shared.v4.f32(
                f32p(OFF["k"] + w_i * 32768 + (i * 128 + t) * 16),
                regs[4 * i], regs[4 * i + 1], regs[4 * i + 2], regs[4 * i + 3],
            )
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx["bar.sync"](txl.uint32(BAR_EPILOGUE), txl.uint32(NUM_MMA_THREADS))
        with txl.if_(tid == 0), txl.then_():
            txl.ptx[_BULK_REDUCE](
                accum.ptr_to([tile]), f32p(OFF["k"]), txl.uint32(BLOCK_N * HEAD_DIM * 4), txl.uint64(EVICT_LAST)
            )
            txl.ptx.cp.async_.bulk.commit_group()
            txl.ptx.cp.async_.bulk.wait_group.read(0)


def _store_zero(L, n_block, bidh, bidb, tid):
    """CollectiveEpilogueBwd::store_zero (epilogue_bwd.hpp:286-323): GmemTiledCopydKV (16 rows x
    16 threads x 8 elements), 128-bit zero stores under the row and column predicates, dK then dV."""
    col8 = (tid % 16) * 8
    pcol = col8 < L["d"]
    rows_left = L["s_k"] - n_block * BLOCK_N
    for out in (L["dk_out"], L["dv_out"]):
        for k in range(BLOCK_N // 16):
            r = tid // 16 + 16 * k
            with txl.if_(txl.And(r < rows_left, pcol)), txl.then_():
                idx = ((_i64(bidb) * _i64(L["s_k"]) + _i64(n_block * BLOCK_N + r)) * _i64(L["h"]) + _i64(bidh)) * _i64(L["d"]) + _i64(col8)
                txl.ptx.st.global_.v4.u32(
                    out.view("uint32").ptr_to([idx // 2]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0)
                )

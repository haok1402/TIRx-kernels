# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Hopper (sm_90a) fp16/bf16 GEMM: C[M, N] = A[M, K] @ B[N, K]^T.

Warp-specialized and persistent, written on tvm-tirx's Hopper helpers:

* CTA tile 128 x 256 x 64. One producer warpgroup issues TMA loads into a
  ``STAGES``-deep ring (``TMABar`` full / ``MBarrier`` empty). Two consumer
  warpgroups each own 64 rows and run ``wgmma.m64n256k16`` chains
  (``lang.wgmma``) with ``wait_group 1`` overlap. ``setmaxnreg`` gives 40 / 232
  registers.
* A 2-CTA cluster along M shares the N tile. Each CTA loads half of B and
  multicasts it, and consumers release a stage in both CTAs
  (``MBarrier.arrive_multicast``).
* Persistent grid of ``max_active_clusters`` clusters (``runtime.occupancy``).
  Each walks 2 x 1 tile pairs in a grouped raster for L2 reuse.
* Any M; N and K multiples of 8 (TMA's 16-byte global-stride rule). Tails cost
  nothing extra: TMA zero-fills out-of-bounds loads (a partial last k-block adds
  zeros) and clips the epilogue's TMA stores. A tile pair whose second tile lies
  past M still runs, so the cluster's B multicast stays symmetric.

Requires tvm-tirx with ``backend/cuda/lang/wgmma.py``, ``MBarrier.arrive_multicast``
and ``runtime/occupancy.py``.
"""

from __future__ import annotations

import torch

import tvm_tirx
from tvm_tirx.script import tirx as T

BM, BN, BK = 128, 256, 64
CLUSTER = 2
GROUP_M = 8  # tile-pair rows per L2 group
DEFAULT_CLUSTERS = 66  # H100/H200 at this tile: 132 SMs / cluster 2 (see max_active_clusters)

_DT = {"fp16": "float16", "bf16": "bfloat16"}
_CVT = {"fp16": "cvt.rn.f16x2.f32", "bf16": "cvt.rn.bf16x2.f32"}


def make_kernel(
    dtype: str,
    M: int,
    N: int,
    K: int,
    *,
    num_clusters: int = DEFAULT_CLUSTERS,
    epilogue: str = "tma",
    stages: int | None = None,
    backend_config=None,
):
    from tvm_tirx.backend.cuda.lang import NamedBarriers, SMEMPool
    from tvm_tirx.backend.cuda.lang import wgmma as W
    from tvm_tirx.backend.cuda.lang.pipeline import MBarrier, TMABar

    dt = _DT[dtype]
    if epilogue not in ("tma", "direct"):
        raise ValueError(f"epilogue must be 'tma' or 'direct', got {epilogue!r}")
    tma_epi = epilogue == "tma"
    # The TMA epilogue stages each consumer's 64 x 256 C tile in shared memory (32 KB),
    # so the ring drops to 3 stages to fit 227 KB.
    STAGES = stages or (3 if tma_epi else 4)
    if min(M, N, K) < 1 or N % 8 or K % 8:
        raise ValueError(
            f"need M, N, K >= 1 and N, K multiples of 8 (TMA strides); got {M}x{N}x{K}"
        )
    if not tma_epi and (M % (BM * CLUSTER) or N % BN):
        raise ValueError(f'epilogue="direct" needs full tiles (M % {BM * CLUSTER}, N % {BN} == 0)')
    # tile pairs along M, tiles along N, k-blocks; partial ones are zero-filled / clipped by TMA
    MP, NT, NK = -(-M // (BM * CLUSTER)), -(-N // BN), -(-K // BK)
    n_pairs = MP * NT
    rounds = (n_pairs + num_clusters - 1) // num_clusters
    half_b = BN // CLUSTER
    stage_tx = (BM * BK + BN * BK) * 2
    a_stage_bytes, b_stage_bytes = BM * BK * 2, BN * BK * 2
    op = W.WGMMA(BN, dt)
    acc_n = op.num_acc
    a_tile = W.SmemTile(None, rows=64)  # one consumer's 64 rows of A, K-major
    b_tile = W.SmemTile(None, rows=BN)
    k_steps = BK * 2 // op.k_bytes
    tma = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    tma_mc = tma + ".multicast::cluster"
    tma_store = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
    stm = "stmatrix.sync.aligned.m8n8.x4.shared.b16"
    c_blocks = BN // 64  # 128-byte (64-element) column blocks of a C tile
    cvt = _CVT[dtype]
    entry_kw = {} if backend_config is None else {"backend_config": backend_config}

    # fmt: off
    @T.function
    def main(A: T.Tensor((M, K), dt, align=16), B: T.Tensor((N, K), dt, align=16),
             C: T.Tensor((M, N), dt, align=16)):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dt, 2, A.data, K, M, K * 2, BK, BM, 1, 1, 0, 3, 2, 0)
        B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dt, 2, B.data, K, N, K * 2, BK, half_b, 1, 1, 0, 3, 2, 0)
        C_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", C_map, dt, 2, C.data, N, M, N * 2, 64, 64, 1, 1, 0, 3, 2, 0)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=num_clusters * CLUSTER, block=384, cluster=CLUSTER),
                       kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1), **entry_kw)
        tx = T.cuda.thread_idx("x")
        # Cluster registers read once into locals. block_idx // CLUSTER trips a tvm-tirx LowerIntrin
        # int32-literal bug at N = 11008 (sm90-probe ISSUE-001), and every use of T.cuda.cluster_id /
        # cluster_cta_id is a fresh `asm volatile` read (ISSUE-002).
        cl = T.alloc_local([2], "int32")
        cl[0] = T.cuda.cluster_id("x")
        cl[1] = T.cuda.cluster_cta_id("x")
        cid = cl[0]
        rank = cl[1]
        A_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, BM, BK), dt, scope="shared", align=1024)])
        B_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, BN, BK), dt, scope="shared", align=1024)])
        if T.constexpr(tma_epi):
            # per consumer: c_blocks 128B-swizzled [64 rows][64 cols] boxes, as the TMA store reads them
            C_s = T.alloc_tensor(ty_args=[T.Tensor((2, c_blocks, 64, 64), dt, scope="shared", align=1024)])
        bars = NamedBarriers(reserved=(1, 2))  # ids 1, 2: per-consumer epilogue barriers (1 + wg)
        pool = SMEMPool()
        full = TMABar(pool, STAGES)
        empty = MBarrier(pool, STAGES)
        pool.commit()
        full.init(1)
        # One arrive per consumer warp per CTA of the cluster: each warp releases a stage only after its
        # own wait on it, so no lagging warp's full-barrier wait can be overtaken by the refill
        # (synccheck finding, sm90-probe KERNEL-001; DeepGEMM does the same).
        empty.init(8 * CLUSTER)
        it = T.alloc_local([1], "int32")
        it[0] = 0
        if tx == 0:
            T.ptx.prefetch.tensormap(T.address_of(A_map))
            T.ptx.prefetch.tensormap(T.address_of(B_map))
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cluster_sync()

        if tx < 128:
            # ---------------- producer warpgroup ----------------
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](40)
            if tx == 0:
                for j in T.serial(rounds):
                    tp = cid + j * num_clusters
                    if tp < n_pairs:
                        g = tp // (GROUP_M * NT)
                        first = g * GROUP_M
                        gsize = T.min(MP - first, GROUP_M)
                        m_blk = (first + (tp % (GROUP_M * NT)) % gsize) * CLUSTER + rank
                        n_blk = (tp % (GROUP_M * NT)) // gsize
                        for kt in T.serial(NK):
                            stage = it[0] % STAGES
                            empty.wait(stage, ((it[0] // STAGES) & 1) ^ 1)
                            full.arrive(stage, stage_tx)
                            T.ptx[tma](A_s.ptr_to([stage, 0, 0]), T.address_of(A_map), kt * BK, m_blk * BM,
                                       full.ptr_to([stage]))
                            T.ptx[tma_mc](B_s.ptr_to([stage, rank * half_b, 0]), T.address_of(B_map), kt * BK,
                                          n_blk * BN + rank * half_b, full.ptr_to([stage]), T.uint16(3))
                            it[0] = it[0] + 1
        else:
            # ---------------- consumer warpgroups ----------------
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](232)
            wg = tx // 128 - 1
            ct = tx % 128
            acc = T.alloc_tensor(ty_args=[T.Tensor((acc_n,), "float32", scope="local")])
            packed = T.alloc_local([1], "uint32")
            da0 = W.smem_descriptor(A_s.ptr_to([0, wg * 64, 0]))
            db0 = W.smem_descriptor(B_s.ptr_to([0, 0, 0]))
            for j in T.serial(rounds):
                tp = cid + j * num_clusters
                if tp < n_pairs:
                    g = tp // (GROUP_M * NT)
                    first = g * GROUP_M
                    gsize = T.min(MP - first, GROUP_M)
                    m_blk = (first + (tp % (GROUP_M * NT)) % gsize) * CLUSTER + rank
                    n_blk = (tp % (GROUP_M * NT)) // gsize
                    for kt in T.serial(NK):
                        stage = it[0] % STAGES
                        full.wait(stage, (it[0] // STAGES) & 1)
                        a0 = W.advance_descriptor(da0, stage * a_stage_bytes)
                        b0 = W.advance_descriptor(db0, stage * b_stage_bytes)
                        W.chain(op, acc, [a_tile.k_step(op, s, a0) for s in range(k_steps)],
                                [b_tile.k_step(op, s, b0) for s in range(k_steps)], accumulate=kt > 0)
                        W.wait(1, (acc, acc_n))
                        if kt > 0:
                            if ct % 32 == 0:
                                empty.arrive_multicast((it[0] + STAGES - 1) % STAGES, ranks=range(CLUSTER))
                        it[0] = it[0] + 1
                    W.wait(0, (acc, acc_n))
                    if ct % 32 == 0:
                        empty.arrive_multicast((it[0] + STAGES - 1) % STAGES, ranks=range(CLUSTER))
                    if T.constexpr(tma_epi):
                        # The previous tile's TMA store must have finished reading C_s.
                        if ct == 0:
                            T.ptx.cp.async_.bulk.wait_group.read(0)
                        T.ptx["bar.sync"](T.uint32(1) + T.Cast("uint32", wg), T.uint32(128))
                        w = ct // 32
                        lane = ct % 32
                        mat = lane // 8
                        srow = w * 16 + (mat % 2) * 8 + lane % 8  # row in this consumer's 64 rows
                        for p in T.unroll(acc_n // 8):  # two 8-column chunks per stmatrix.x4
                            words = T.alloc_local([4], "uint32")
                            for h in T.unroll(2):
                                r = (2 * p + h) * 4
                                T.ptx[cvt](words[2 * h], acc[r + 1], acc[r])
                                T.ptx[cvt](words[2 * h + 1], acc[r + 3], acc[r + 2])
                            col = (2 * p + mat // 2) * 8
                            chunk = ((col % 64) // 8) ^ (srow % 8)  # 128B swizzle of 16-byte chunks
                            T.ptx[stm](C_s.ptr_to([wg, col // 64, srow, chunk * 8]),
                                       words[0], words[1], words[2], words[3])
                        T.ptx.fence.proxy.async_.shared__cta()
                        T.ptx["bar.sync"](T.uint32(1) + T.Cast("uint32", wg), T.uint32(128))
                        if ct == 0:
                            for b in T.unroll(c_blocks):
                                T.ptx[tma_store](T.address_of(C_map), T.Cast("int32", n_blk * BN + b * 64),
                                                 T.Cast("int32", m_blk * BM + wg * 64), C_s.ptr_to([wg, b, 0, 0]))
                            T.ptx.cp.async_.bulk.commit_group()
                    # direct epilogue: registers -> global, two halves per 8-column group
                    for r4 in T.unroll(0 if tma_epi else acc_n // 4):
                        r = r4 * 4
                        row, col = W.acc_coord(ct, r)
                        grow = m_blk * BM + wg * 64 + row
                        gcol = n_blk * BN + col
                        T.ptx[cvt](packed[0], acc[r + 1], acc[r])
                        T.ptx.st.global_.b32(C.ptr_to([grow, gcol]), packed[0])
                        T.ptx[cvt](packed[0], acc[r + 3], acc[r + 2])
                        T.ptx.st.global_.b32(C.ptr_to([grow + 8, gcol]), packed[0])
        if T.constexpr(tma_epi):
            if tx >= 128:
                if tx % 128 == 0:
                    T.ptx.cp.async_.bulk.wait_group(0)
        T.cuda.cluster_sync()
    # fmt: on
    return main


KERNEL_META = {"name": "fp16_bf16_gemm_sm90", "category": "gemm", "runtime_cuda_archs": ["sm_90a"]}
# Shapes with >= 2 waves of 128x256 tiles at 132 SMs are the plan's B1 gate set;
# the others are tracked.
GATED_SHAPES = [
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    (2048, 8192, 1024),
    (2048, 11008, 4096),
    (4096, 4096, 11008),
    (4096, 11008, 4096),
]
TRACKED_SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (1024, 8192, 2048),
    (2048, 4096, 11008),
    (4096, 2048, 8192),
    (8192, 1024, 2048),
]
# Tails (M, N, K not multiples of the tile) and small M: correctness configs, not benchmarked for the gate.
TAIL_SHAPES = [
    (1, 4096, 4096),
    (100, 2112, 7168),
    (777, 576, 1000),
    (4096, 2112, 7168),
    (300, 520, 136),
    (129, 264, 8),
]
CONFIGS = [
    {"dtype": d, "M": M, "N": N, "K": K, "label": f"{d}_{M}x{N}x{K}"}
    for d in ("fp16", "bf16")
    for M, N, K in GATED_SHAPES + TRACKED_SHAPES + TAIL_SHAPES
]


def prepare_data(dtype, M, N, K):
    dev = torch.device("cuda", torch.cuda.current_device())
    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    A = torch.randn(M, K).to(tdt).to(dev)
    B = torch.randn(N, K).to(tdt).to(dev)
    C = torch.zeros((M, N), dtype=tdt, device=dev)
    return A, B, C


def compile_persistent(dtype, M, N, K, *, backend_config=None):
    """Two-pass build: compile, ask the driver how many clusters fit, rebuild if needed."""
    from tirx_kernels.runner import compile_kernel
    from tvm_tirx.backend.cuda.runtime import occupancy

    ex = compile_kernel(
        make_kernel(dtype, M, N, K, backend_config=backend_config), backend_config=backend_config
    )
    clusters = occupancy.max_active_clusters(ex, block=384, cluster=CLUSTER)
    if clusters != DEFAULT_CLUSTERS:
        ex = compile_kernel(
            make_kernel(dtype, M, N, K, num_clusters=clusters, backend_config=backend_config),
            backend_config=backend_config,
        )
    return ex, clusters


def get_kernel(dtype, M, N, K, *, backend_config=None, **kwargs):
    return make_kernel(dtype, M, N, K, backend_config=backend_config)


def run_test(dtype, M, N, K, *, backend_config=None, **kwargs):
    A, B, C = prepare_data(dtype, M, N, K)
    ex, _ = compile_persistent(dtype, M, N, K, backend_config=backend_config)
    C.fill_(float("nan"))  # every element, tails included, must be written
    ex(A, B, C)
    # fp32 reference, one output ulp of tolerance: cuBLAS itself picks a different (split-K)
    # algorithm at small M and differs from this kernel by 1 bf16 ulp there.
    ref = A.float() @ B.float().T
    ulp = {"fp16": 2.0**-10, "bf16": 2.0**-7}[dtype]
    torch.testing.assert_close(C.float().cpu(), ref.cpu(), rtol=ulp, atol=0.01)


def run_bench(
    dtype, M, N, K, warmup=None, repeat=None, timer=None, *, backend_config=None, **kwargs
):
    from tirx_kernels.runner import bench

    A, B, C = prepare_data(dtype, M, N, K)
    ex, _ = compile_persistent(dtype, M, N, K, backend_config=backend_config)
    C_ref = torch.zeros_like(C)
    return bench(
        {"tir": lambda: ex(A, B, C)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"torch-cublas": lambda: lambda: torch.matmul(A, B.T, out=C_ref)},
        **kwargs,
    )


_ = tvm_tirx  # imported for compile_kernel's target resolution

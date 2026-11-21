# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Hopper (sm_90a) FP8 block-scaled GEMM, DeepGEMM sm90 "1D2D" style.

``D[M, N] = (A * sfa) @ (B * sfb)^T`` with
* ``A`` e4m3 ``[M, K]`` and per-token scales ``sfa`` ``[M, K/128]``, passed MN-major as
  ``sfa_mn_major(sfa)`` ``[K/128, align(M, 4)]`` so that TMA loads a tile's scales in one box,
* ``B`` e4m3 ``[N, K]`` and per-block scales ``sfb`` ``[N/128, K/128]``,
* ``D`` bf16 ``[M, N]``.

Hopper has no block-scaled MMA. Each 128-wide K block is a 4-step fp8 ``wgmma``
chain into a temporary accumulator ``tmp``, then a promotion
``acc += tmp * sfa[row] * sfb[n_block]`` in registers.

Tile and register design (B2-design; H200, setmaxnreg 40 / 232; ``pick_config`` chooses per shape):

* ``bm = 256, bn = 128`` (default): each of the two consumer warpgroups owns two m64
  "waves" of 64 rows. That is ``acc`` 2 x 64 plus ``tmp`` 64, so 192 f32 accumulator
  registers per thread. A 2-CTA cluster along N shares the A tile through multicast.
  The promotion stays serial after ``wait_group 0``, as in DeepGEMM. Double-buffering
  ``tmp`` to overlap it is not possible: ptxas serializes every wgmma (C7514) as soon as
  a retired buffer is read while another chain is in flight.
* ``bm = 128, bn = 192`` with a cluster along M sharing B, for small-N shapes. That is
  96 + 96 registers.
* ``bn = 256`` with ``bm = 128`` spills (128 + 128 accumulators), so it is rejected.

What made the serial promotion as cheap as DeepGEMM's (SASS diff against its cubin, sm90-probe logs/b2):
``#pragma unroll 8`` on the k-loop, so nvcc interleaves the next k-block's barrier wait and scale loads with
the FFMA promotion, and a warp-uniform warpgroup index (``shfl.sync``), so wgmma descriptors stay in uniform
registers. Per chain, both kernels then issue 4 QGMMA + 64 FFMA with little else.

Warp-specialized and persistent like ``fp16_bf16_gemm_sm90``: one producer warpgroup
(TMA for A, B and the tile's A scales per stage), stmatrix plus TMA-store epilogue,
grid sized by ``max_active_clusters``. Any M; N % 8 == 0, K % 16 == 0 (TMA strides).
Tails are zero-filled and clipped by TMA.
"""

from __future__ import annotations

import torch

BK = 128
GROUP_M = 8
# setmaxnreg: 384 threads x 1 block -> 168 entry registers; 40 x 128 + 232 x 256 = 168 x 384 (DeepGEMM's split;
# a 24-register producer spills once multicast addressing is added).
PRODUCER_REGS, CONSUMER_REGS = 40, 232
SMEM_BYTES = 227 * 1024


def sfa_stride(M):
    """Row stride of the MN-major A-scale tensor ``[K/128, M]``: M rounded up to 4 (16-byte TMA stride)."""
    return -(-M // 4) * 4


def sfa_mn_major(sfa):
    """``sfa [M, K/128]`` -> ``[K/128, sfa_stride(M)]`` (zero-padded), the layout the kernel loads."""
    M, kb = sfa.shape
    out = torch.zeros(kb, sfa_stride(M), dtype=torch.float32, device=sfa.device)
    out[:, :M] = sfa.t()
    return out


def num_stages(bm, bn):
    per_stage = bm * BK + bn * BK + bm * 4  # A, B (fp8) and the A scales
    c_bytes = bm * bn * 2  # bf16 staging tile of the epilogue
    return min(
        6, (SMEM_BYTES - c_bytes - 3072) // per_stage
    )  # 3 KB: barriers + per-tile sfb (2 x K/128 floats)


def pick_config(M, N, K):
    """(bm, bn, cluster) for a shape, from the H200 sweep over the gated shapes (sm90-probe logs/b2/v4_gated.log).

    * narrow N: 128 x 192 tiles give more tiles per wave than 256 x 128 (N = 2112 -> 11 exact column tiles);
    * short K: few k-blocks per tile, so the per-tile cost dominates and sharing B across an M pair pays;
    * otherwise DeepGEMM's main tile, 256 x 128 with A multicast across an N pair.
    """
    if N <= 2304:
        return 128, 192, None
    if K <= 1536:
        return 128, 192, "m"
    return 256, 128, "n"


def make_kernel(
    M: int,
    N: int,
    K: int,
    *,
    bm: int = 256,
    bn: int = 128,
    cluster: str | None = "n",
    num_clusters: int | None = None,
    regs: tuple[int, int] = (PRODUCER_REGS, CONSUMER_REGS),
    backend_config=None,
):
    from tvm_tirx.backend.cuda.lang import NamedBarriers, SMEMPool
    from tvm_tirx.backend.cuda.lang import wgmma as W
    from tvm_tirx.backend.cuda.lang.pipeline import MBarrier, TMABar
    from tvm_tirx.script import tirx as T

    if (
        bm not in (128, 256)
        or bn % 64
        or not 64 <= bn <= 256
        or (bm == 256 and bn > 128)
        or (bm == 128 and bn > 192)
    ):
        raise ValueError(f"unsupported tile {bm}x{bn} (register budget: see the module docstring)")
    if cluster not in (None, "m", "n"):
        raise ValueError(f"cluster must be None, 'm' or 'n', got {cluster!r}")
    if min(M, N, K) < 1 or N % 8 or K % 16:
        raise ValueError(f"need N % 8 == 0 and K % 16 == 0 (TMA strides); got {M}x{N}x{K}")
    CM, CN = (2, 1) if cluster == "m" else (1, 2) if cluster == "n" else (1, 1)
    CS = CM * CN
    if num_clusters is None:
        num_clusters = 132 // CS
    STAGES = num_stages(bm, bn)
    waves = bm // 128  # m64 waves per consumer warpgroup
    MS = sfa_stride(M)
    MT, NT = -(-M // bm), -(-N // bn)
    MC, NC = (
        -(-MT // CM),
        -(-NT // CN),
    )  # cluster tiles; a partner tile past M / N still runs (symmetric multicast)
    KB = -(-K // BK)  # k-blocks; the last may be partial (TMA zero-fills)
    NB = -(-N // 128)  # 128-column sfb blocks
    n_units = MC * NC
    rounds = -(-n_units // num_clusters)
    MA, MB = CN, CM  # multicast factor of A / B
    MCS = MA * MB
    a_rows = bm // MA  # A rows this CTA loads (and multicasts when MA == 2)
    b_rows = bn // MB
    stage_tx = bm * BK + bn * BK + bm * 4
    op = W.WGMMA(bn, "float8_e4m3fn")
    acc_n = op.num_acc
    a_tile = W.SmemTile(None, rows=64)
    b_tile = W.SmemTile(None, rows=bn)
    k_steps = BK // op.k_bytes  # 4
    a_stage_bytes, b_stage_bytes = bm * BK, bn * BK
    tma = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    tma_mc = tma + ".multicast::cluster"
    tma_store = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
    stm = "stmatrix.sync.aligned.m8n8.x4.shared.b16"
    c_blocks = bn // 64
    entry_kw = {} if backend_config is None else {"backend_config": backend_config}
    # Tile starts n_blk * bn sit at a fixed offset in the 128-column sfb grid when bn % 128 == 0 or 128 % bn == 0,
    # so the sfb block of each 8-column group is a compile-time fact; for bn = 192 it is not.
    static_split = bn % 128 == 0 or 128 % bn == 0

    # fmt: off
    @T.function
    def main(A: T.Tensor((M, K), "float8_e4m3fn", align=16), SFA: T.Tensor((KB, MS), "float32", align=16),
             B: T.Tensor((N, K), "float8_e4m3fn", align=16), SFB: T.Tensor((NB, KB), "float32", align=16),
             D: T.Tensor((M, N), "bfloat16", align=16)):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float8_e4m3fn", 2, A.data, K, M, K, BK, a_rows, 1, 1, 0, 3, 2, 0)
        B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float8_e4m3fn", 2, B.data, K, N, K, BK, b_rows, 1, 1, 0, 3, 2, 0)
        S_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", S_map, "float32", 2, SFA.data, MS, KB, MS * 4, bm, 1, 1, 1, 0, 0, 2, 0)
        D_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", D_map, "bfloat16", 2, D.data, N, M, N * 2, 64, 64, 1, 1, 0, 3, 2, 0)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=num_clusters * CS, block=384, cluster=CS),
                       kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1), **entry_kw)
        tx = T.cuda.thread_idx("x")
        if T.constexpr(CS > 1):
            # Read the cluster registers once into locals: each use of T.cuda.cluster_id / cluster_cta_id is
            # a fresh `asm volatile` read (sm90-probe ISSUE-002, ~10% here), and block_idx // 2 trips the
            # LowerIntrin int32-literal bug at some N (ISSUE-001).
            cl = T.alloc_local([2], "int32")
            cl[0] = T.cuda.cluster_id("x")
            cl[1] = T.cuda.cluster_cta_id("x")
            cid = cl[0]
            rank = cl[1]
        else:
            cid = T.cuda.block_idx("x")
            rank = 0
        A_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, bm, BK), "float8_e4m3fn", scope="shared", align=1024)])
        B_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, bn, BK), "float8_e4m3fn", scope="shared", align=1024)])
        S_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, bm), "float32", scope="shared", align=128)])
        # this tile's two candidate sfb rows (blocks nb0, nb0 + 1) for every k-block, filled once per tile
        SFB_s = T.alloc_tensor(ty_args=[T.Tensor((2, KB), "float32", scope="shared", align=16)])
        # epilogue staging: [wave][consumer][64-column block] 128B-swizzled 64 x 64 boxes, as the TMA store reads them
        C_s = T.alloc_tensor(ty_args=[T.Tensor((waves, 2, c_blocks, 64, 64), "bfloat16", scope="shared", align=1024)])
        bars = NamedBarriers(reserved=(1, 2, 3))  # 1 + wg: per-consumer epilogue barriers; 3: both consumers
        pool = SMEMPool()
        full = TMABar(pool, STAGES)
        empty = MBarrier(pool, STAGES)
        pool.commit()
        full.init(1)
        empty.init(8 * MCS)  # one arrive per consumer warp per CTA of the cluster (DeepGEMM)
        # ring position as (stage, phase), advanced incrementally: no per-k-block % / // (signed int division
        # by a constant still costs a multiply-high plus sign fix-ups)
        sp = T.alloc_local([2], "int32")
        sp[0] = 0
        sp[1] = 0
        if tx == 0:
            T.ptx.prefetch.tensormap(T.address_of(A_map))
            T.ptx.prefetch.tensormap(T.address_of(B_map))
        T.ptx.fence.mbarrier_init.release.cluster()
        if T.constexpr(CS > 1):
            T.cuda.cluster_sync()
        else:
            T.cuda.cta_sync()

        if tx < 128:
            # ---------------- producer warpgroup ----------------
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](regs[0])
            if tx == 0:
                for j in T.serial(rounds):
                    t = cid + j * num_clusters
                    if t < n_units:
                        g = t // (GROUP_M * NC)
                        first = g * GROUP_M
                        gsize = T.min(MC - first, GROUP_M)
                        m_blk = (first + (t % (GROUP_M * NC)) % gsize) * CM + (rank if CM == 2 else 0)
                        n_blk = (t % (GROUP_M * NC)) // gsize * CN + (rank if CN == 2 else 0)
                        for kb in T.serial(KB):
                            stage = sp[0]
                            empty.wait(stage, sp[1] ^ 1)
                            full.arrive(stage, stage_tx)
                            if T.constexpr(MA == 2):
                                T.ptx[tma_mc](A_s.ptr_to([stage, rank * a_rows, 0]), T.address_of(A_map), kb * BK,
                                              m_blk * bm + rank * a_rows, full.ptr_to([stage]), T.uint16(3))
                            else:
                                T.ptx[tma](A_s.ptr_to([stage, 0, 0]), T.address_of(A_map), kb * BK, m_blk * bm,
                                           full.ptr_to([stage]))
                            if T.constexpr(MB == 2):
                                T.ptx[tma_mc](B_s.ptr_to([stage, rank * b_rows, 0]), T.address_of(B_map), kb * BK,
                                              n_blk * bn + rank * b_rows, full.ptr_to([stage]), T.uint16(3))
                            else:
                                T.ptx[tma](B_s.ptr_to([stage, 0, 0]), T.address_of(B_map), kb * BK, n_blk * bn,
                                           full.ptr_to([stage]))
                            T.ptx[tma](S_s.ptr_to([stage, 0]), T.address_of(S_map), m_blk * bm, kb, full.ptr_to([stage]))
                            sp[1] = sp[1] ^ T.Cast("int32", sp[0] == STAGES - 1)
                            sp[0] = T.Select(sp[0] == STAGES - 1, 0, sp[0] + 1)
                if T.constexpr(MCS > 1):
                    # Drain: wait until every stage was released by both CTAs. After that no remote arrive or
                    # multicast write targets this CTA, so it may exit; no kernel-end cluster barrier (which would
                    # park the idle warps at a cluster barrier for the whole kernel).
                    for _s in T.serial(STAGES):
                        empty.wait(sp[0], sp[1] ^ 1)
                        sp[1] = sp[1] ^ T.Cast("int32", sp[0] == STAGES - 1)
                        sp[0] = T.Select(sp[0] == STAGES - 1, 0, sp[0] + 1)
        else:
            # ---------------- consumer warpgroups ----------------
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](regs[1])
            # Broadcast the warpgroup index from lane 0 (DeepGEMM's __shfl_sync trick): nvcc then knows it is
            # warp-uniform and keeps the descriptor arithmetic in uniform registers (no R2UR per wgmma; ~2-3%).
            wgv = T.alloc_local([1], "uint32")
            T.ptx.shfl_sync.idx.b32(wgv[0], T.Cast("uint32", tx // 128), T.uint32(0), T.uint32(31),
                                    T.uint32(0xFFFFFFFF))
            wg = T.Cast("int32", wgv[0]) - 1
            ct = tx % 128
            acc = T.alloc_tensor(ty_args=[T.Tensor((waves * acc_n,), "float32", scope="local")])
            tmp = T.alloc_tensor(ty_args=[T.Tensor((acc_n,), "float32", scope="local")])
            sc = T.alloc_local([8], "float32")  # sa0, sa1, the four scale products, sb0, sb1
            da0 = W.smem_descriptor(A_s.ptr_to([0, wg * 64, 0]))
            db0 = W.smem_descriptor(B_s.ptr_to([0, 0, 0]))
            r0 = (ct // 32) * 16 + (ct % 32) // 4  # this thread's rows: r0 and r0 + 8 of its 64
            for j in T.serial(rounds):
                t = cid + j * num_clusters
                if t < n_units:
                    g = t // (GROUP_M * NC)
                    first = g * GROUP_M
                    gsize = T.min(MC - first, GROUP_M)
                    m_blk = (first + (t % (GROUP_M * NC)) % gsize) * CM + (rank if CM == 2 else 0)
                    n_blk = (t % (GROUP_M * NC)) // gsize * CN + (rank if CN == 2 else 0)
                    for i in T.unroll(waves * acc_n):
                        acc[i] = T.float32(0.0)
                    nb0 = (n_blk * bn) // 128
                    split = 128 - (n_blk * bn) % 128  # tile column where sfb block nb0 + 1 starts
                    # both consumer warpgroups are past the previous tile's k-loop before SFB_s is refilled
                    T.ptx["bar.sync"](T.uint32(3), T.uint32(256))
                    for i in T.serial(-(-2 * KB // 256)):
                        idx = (tx - 128) + i * 256
                        if idx < 2 * KB:
                            SFB_s[idx // KB, idx % KB] = SFB[T.min(nb0 + idx // KB, NB - 1), idx % KB]
                    T.ptx["bar.sync"](T.uint32(3), T.uint32(256))
                    # `#pragma unroll 8` (DeepGEMM's) lets nvcc pipeline the next k-block's barrier wait and
                    # scale loads into this one's FFMA promotion: ~6-7% (logs/b2/v4_fixes.log).
                    for kb in T.serial(KB, unroll=8):
                        stage = sp[0]
                        # B scales (shared, this tile's) before the wait, so their latency hides
                        sc[6] = SFB_s[0, kb]
                        sc[7] = SFB_s[1, kb]
                        full.wait(stage, sp[1])
                        b0 = W.advance_descriptor(db0, stage * b_stage_bytes)
                        for w in T.unroll(waves):
                            a0 = W.advance_descriptor(da0, stage * a_stage_bytes + w * 128 * BK)
                            # DeepGEMM order: read the A scales before the chain, form the products while it runs
                            sc[0] = S_s[stage, w * 128 + wg * 64 + r0]
                            sc[1] = S_s[stage, w * 128 + wg * 64 + r0 + 8]
                            W.chain(op, tmp, [a_tile.k_step(op, s, a0) for s in range(k_steps)],
                                    [b_tile.k_step(op, s, b0) for s in range(k_steps)], accumulate=False)
                            sc[2] = sc[0] * sc[6]
                            sc[3] = sc[0] * sc[7]
                            sc[4] = sc[1] * sc[6]
                            sc[5] = sc[1] * sc[7]
                            W.wait(0, (tmp, acc_n))
                            if w == waves - 1:
                                if ct % 32 == 0:  # lane 0 of every consumer warp releases the stage
                                    if T.constexpr(MCS > 1):
                                        empty.arrive_multicast(stage, ranks=range(CS))
                                    else:
                                        empty.arrive(stage)
                            # promotion: acc += tmp * sfa[row] * sfb[128-column block of the column]
                            for c in T.unroll(bn // 8):
                                if T.constexpr(static_split):
                                    first = c * 8 < 128
                                else:
                                    first = c * 8 < split
                                lo = T.Select(first, sc[2], sc[3])  # rows r0
                                hi = T.Select(first, sc[4], sc[5])  # rows r0 + 8
                                acc[w * acc_n + c * 4] = acc[w * acc_n + c * 4] + tmp[c * 4] * lo
                                acc[w * acc_n + c * 4 + 1] = acc[w * acc_n + c * 4 + 1] + tmp[c * 4 + 1] * lo
                                acc[w * acc_n + c * 4 + 2] = acc[w * acc_n + c * 4 + 2] + tmp[c * 4 + 2] * hi
                                acc[w * acc_n + c * 4 + 3] = acc[w * acc_n + c * 4 + 3] + tmp[c * 4 + 3] * hi
                        sp[1] = sp[1] ^ T.Cast("int32", sp[0] == STAGES - 1)
                        sp[0] = T.Select(sp[0] == STAGES - 1, 0, sp[0] + 1)
                    # epilogue: stmatrix into the swizzled staging tile, then TMA store (clipped at M / N)
                    if ct == 0:
                        T.ptx.cp.async_.bulk.wait_group.read(0)
                    T.ptx["bar.sync"](T.uint32(1) + T.Cast("uint32", wg), T.uint32(128))
                    lane = ct % 32
                    mat = lane // 8
                    srow = (ct // 32) * 16 + (mat % 2) * 8 + lane % 8
                    for w in T.unroll(waves):
                        for p in T.unroll(acc_n // 8):  # two 8-column chunks per stmatrix.x4
                            words = T.alloc_local([4], "uint32")
                            for h in T.unroll(2):
                                r = w * acc_n + (2 * p + h) * 4
                                T.ptx["cvt.rn.bf16x2.f32"](words[2 * h], acc[r + 1], acc[r])
                                T.ptx["cvt.rn.bf16x2.f32"](words[2 * h + 1], acc[r + 3], acc[r + 2])
                            col = (2 * p + mat // 2) * 8
                            chunk = ((col % 64) // 8) ^ (srow % 8)  # 128B swizzle of 16-byte chunks
                            T.ptx[stm](C_s.ptr_to([w, wg, col // 64, srow, chunk * 8]),
                                       words[0], words[1], words[2], words[3])
                    T.ptx.fence.proxy.async_.shared__cta()
                    T.ptx["bar.sync"](T.uint32(1) + T.Cast("uint32", wg), T.uint32(128))
                    if ct == 0:
                        for w in T.unroll(waves):
                            for b in T.unroll(c_blocks):
                                T.ptx[tma_store](T.address_of(D_map), T.Cast("int32", n_blk * bn + b * 64),
                                                 T.Cast("int32", m_blk * bm + w * 128 + wg * 64),
                                                 C_s.ptr_to([w, wg, b, 0, 0]))
                        T.ptx.cp.async_.bulk.commit_group()
            if ct == 0:
                T.ptx.cp.async_.bulk.wait_group(0)
    # fmt: on
    return main


KERNEL_META = {
    "name": "fp8_blockscale_gemm_sm90",
    "category": "gemm",
    "runtime_cuda_archs": ["sm_90a"],
}
# DeepGEMM tests/generators.py enumerate_normal, bf16-output (n, k) list. Gated: m = 4096 (its m >= 2048 rows;
# the m = 4096 "backward" rows are the same K-major shapes on sm90 fp8). Tracked: m = 128 (decode-like).
_NK = [
    (2112, 7168),
    (576, 7168),
    (24576, 1536),
    (32768, 512),
    (7168, 16384),
    (4096, 7168),
    (7168, 2048),
]
GATED_SHAPES = [(4096, n, k) for n, k in _NK]
TRACKED_SHAPES = [(128, n, k) for n, k in _NK]
# Tails and small M: correctness configs (partial tiles in M, N and K; every pick_config branch).
TAIL_SHAPES = [(1, 4096, 7168), (100, 2112, 7168), (777, 576, 1008), (300, 520, 144), (1000, 24584, 1520),
               (2049, 7176, 2064)]  # fmt: skip
CONFIGS = [
    {"M": M, "N": N, "K": K, "label": f"{M}x{N}x{K}"}
    for M, N, K in GATED_SHAPES + TRACKED_SHAPES + TAIL_SHAPES
]


def prepare_data(M, N, K):
    """FP8 operands with DeepGEMM's quantization recipe (fp32 scales, no UE8M0)."""
    dev = torch.device("cuda", torch.cuda.current_device())
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    kb = -(-K // 128)
    ap = torch.zeros(M, kb * 128, device=dev, dtype=torch.bfloat16)
    ap[:, :K] = a
    sfa = ap.view(M, kb, 128).abs().float().amax(-1).clamp(1e-4) / 448.0
    a8 = (
        (ap.view(M, kb, 128) / sfa.unsqueeze(-1))
        .to(torch.float8_e4m3fn)
        .view(M, kb * 128)[:, :K]
        .contiguous()
    )
    nb = -(-N // 128)
    bp = torch.zeros(nb * 128, kb * 128, device=dev, dtype=torch.bfloat16)
    bp[:N, :K] = b
    sfb = bp.view(nb, 128, kb, 128).abs().float().amax(dim=(1, 3)).clamp(1e-4) / 448.0
    b8 = (
        (bp.view(nb, 128, kb, 128) / sfb[:, None, :, None])
        .to(torch.float8_e4m3fn)
        .view(nb * 128, kb * 128)
    )
    b8 = b8[:N, :K].contiguous()
    d = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    return a8, sfa, b8, sfb, d


def reference(a8, sfa, b8, sfb, out_dtype=torch.bfloat16, absolute=False):
    """Dequantize and multiply in float64: an fp32 torch matmul may run in TF32, whose input rounding
    (dequantized operands are not TF32-exact) is larger than the kernel's own error. ``absolute=True``
    gives ``|A| @ |B|^T``, the scale of the accumulation error."""
    M, K = a8.shape
    N = b8.shape[0]
    kb = sfa.shape[1]
    f64 = torch.float64
    ap = torch.zeros(M, kb * 128, device=a8.device, dtype=f64)
    ap[:, :K] = a8.to(f64).abs() if absolute else a8.to(f64)
    bp = torch.zeros(-(-N // 128) * 128, kb * 128, device=a8.device, dtype=f64)
    bp[:N, :K] = b8.to(f64).abs() if absolute else b8.to(f64)
    ap = (ap.view(M, kb, 128) * sfa.to(f64).unsqueeze(-1)).view(M, -1)
    bp = (bp.view(-1, 128, kb, 128) * sfb.to(f64)[:, None, :, None]).view(bp.shape)
    return (ap @ bp[:N].t()).to(out_dtype)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def compile_persistent(M, N, K, *, config=None, backend_config=None, **kw):
    """Two-pass build: compile, ask the driver how many clusters fit, rebuild if that differs."""
    from tirx_kernels.runner import compile_kernel
    from tvm_tirx.backend.cuda.runtime import occupancy

    bm, bn, cluster = config or pick_config(M, N, K)
    cs = 1 if cluster is None else 2

    def build(nc=None):
        k = make_kernel(
            M,
            N,
            K,
            bm=bm,
            bn=bn,
            cluster=cluster,
            num_clusters=nc,
            backend_config=backend_config,
            **kw,
        )
        return compile_kernel(k, backend_config=backend_config)

    ex = build()
    clusters = occupancy.max_active_clusters(ex, block=384, cluster=cs)
    if clusters != 132 // cs:
        ex = build(clusters)
    return ex


def get_kernel(M, N, K, *, backend_config=None, **kwargs):
    bm, bn, cluster = pick_config(M, N, K)
    return make_kernel(M, N, K, bm=bm, bn=bn, cluster=cluster, backend_config=backend_config)


def run_test(M, N, K, *, backend_config=None, **kwargs):
    a8, sfa, b8, sfb, d = prepare_data(M, N, K)
    ex = compile_persistent(M, N, K, backend_config=backend_config)
    d.fill_(float("nan"))  # every element, tails included, must be written
    ex(a8, sfa_mn_major(sfa), b8, sfb, d)
    check_output(d, a8, sfa, b8, sfb)


def check_output(d, a8, sfa, b8, sfb):
    """Aggregate, element-wise and (with DeepGEMM installed) bitwise checks of a kernel output."""
    ref = reference(a8, sfa, b8, sfb, out_dtype=torch.float64)
    # DeepGEMM's aggregate criterion ...
    diff = calc_diff(d, ref)
    assert diff < 1e-3, f"calc_diff {diff}"
    # ... and an element-wise one (calc_diff alone would pass a few wrong rows): one bf16 ulp of the exact
    # result, plus the fp8 tensor-core accumulation error inside a 128-wide k-block, which scales with
    # |A| @ |B|^T (~2^-15 of it measured; 2^-12 allowed). A wrong element is off by O(|ref|) ~ sqrt(K) x
    # that floor's scale / 2^-12, far outside it.
    bound = 2.0**-8 * ref.abs() + 2.0**-12 * reference(
        a8, sfa, b8, sfb, out_dtype=torch.float64, absolute=True
    )
    bad = (d.double() - ref).abs() > bound
    if bad.any():
        r, c = (int(x) for x in bad.nonzero()[0])
        raise AssertionError(
            f"{int(bad.sum())} of {d.numel()} outputs out of tolerance (rows {int(bad.any(1).sum())}); "
            f"first at ({r}, {c}): got {float(d[r, c])}, expected {float(ref[r, c])}"
        )
    try:
        import deep_gemm
    except ImportError:
        return
    # Same recipe and the same per-k-block fp32 promotion as DeepGEMM's sm90 1D2D kernel: bit-identical output.
    d_ref = torch.empty_like(d)
    deep_gemm.fp8_gemm_nt((a8, deep_gemm.get_mn_major_tma_aligned_tensor(sfa)), (b8, sfb), d_ref)
    mismatches = int((d.view(torch.int16) != d_ref.view(torch.int16)).sum())
    assert mismatches == 0, f"{mismatches} of {d.numel()} outputs differ bitwise from DeepGEMM"


def run_bench(M, N, K, warmup=None, repeat=None, timer=None, *, backend_config=None, **kwargs):
    from tirx_kernels.runner import bench

    a8, sfa, b8, sfb, d = prepare_data(M, N, K)
    ex = compile_persistent(M, N, K, backend_config=backend_config)
    sa_t = sfa_mn_major(sfa)
    references = {}
    try:
        import deep_gemm

        d_ref = torch.empty_like(d)
        sa_dg = deep_gemm.get_mn_major_tma_aligned_tensor(sfa)
        references["deepgemm"] = lambda: (
            lambda: deep_gemm.fp8_gemm_nt((a8, sa_dg), (b8, sfb), d_ref)
        )
    except ImportError:
        pass
    return bench(
        {"tir": lambda: ex(a8, sa_t, b8, sfb, d)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references=references,
        **kwargs,
    )

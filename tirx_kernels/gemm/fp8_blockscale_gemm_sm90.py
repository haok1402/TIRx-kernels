# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Hopper (sm_90a) FP8 block-scaled GEMM, DeepGEMM sm90 "1D2D" style.

``D[M, N] = (A * sfa) @ (B * sfb)^T`` with
* ``A`` e4m3 ``[M, K]`` and per-token scales ``sfa`` ``[M, K/128]`` (passed MN-major as
  ``sfa_t = sfa.t()`` ``[K/128, M]`` so that TMA can load a CTA's 128 scales in one box),
* ``B`` e4m3 ``[N, K]`` and per-block scales ``sfb`` ``[N/128, K/128]``,
* ``D`` bf16 ``[M, N]``.

Hopper has no block-scaled MMA, so each 128-wide K block is a 4-step fp8
``wgmma`` chain into a temporary accumulator, followed by a promotion
``acc += tmp * sfa[row] * sfb[n_block]`` in registers. Two accumulators per
consumer thread (``2 * BN / 2`` f32 registers) are the register budget that sets
the N tile: see ``REGISTER_BUDGET`` and the B2-design notes.

Warp-specialized and persistent like ``fp16_bf16_gemm_sm90``: one producer warpgroup
(TMA for A, B and the CTA's 128 A-scales per stage), two consumer warpgroups of
64 rows, ``setmaxnreg`` 24 / 240, stmatrix + TMA-store epilogue.
"""

from __future__ import annotations

import torch

BM, BK = 128, 128
GROUP_M = 8
DEFAULT_CTAS = 132
# setmaxnreg: 384 threads x 1 block -> 168 entry registers; 24 x 128 + 240 x 256 = 168 x 384.
PRODUCER_REGS, CONSUMER_REGS = 24, 240


def _stages(bn):
    # per stage: A 16 KB + B bn*128 B + 512 B of A-scales; C staging 2 x 64 x bn x 2 B; <= 227 KB
    per_stage = BM * BK + bn * BK + BM * 4
    c_bytes = 2 * 64 * bn * 2
    return min(6, (227 * 1024 - c_bytes - 1024) // per_stage)


def make_kernel(M: int, N: int, K: int, *, bn: int = 128, num_ctas: int = DEFAULT_CTAS, backend_config=None):
    from tvm_tirx.backend.cuda.lang import NamedBarriers, SMEMPool
    from tvm_tirx.backend.cuda.lang import wgmma as W
    from tvm_tirx.backend.cuda.lang.pipeline import MBarrier, TMABar
    from tvm_tirx.script import tirx as T

    if bn % 64 or not 64 <= bn <= 256:
        raise ValueError(f"bn must be a multiple of 64 in [64, 256], got {bn}")
    if M % BM:
        raise ValueError(f"M must be a multiple of {BM}, got {M}")
    STAGES = _stages(bn)
    MT, NT = -(-M // BM), -(-N // bn)
    KB = -(-K // BK)  # K blocks (the last may be partial: TMA zero-fills)
    NB = -(-N // 128)  # 128-wide scale blocks of N
    n_tiles = MT * NT
    rounds = -(-n_tiles // num_ctas)
    stage_tx = BM * BK + bn * BK + BM * 4
    op = W.WGMMA(bn, "float8_e4m3fn")
    acc_n = op.num_acc
    a_tile = W.SmemTile(None, rows=64)
    b_tile = W.SmemTile(None, rows=bn)
    k_steps = BK // op.k_bytes  # 4
    a_stage_bytes, b_stage_bytes = BM * BK, bn * BK
    tma = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    tma_store = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
    stm = "stmatrix.sync.aligned.m8n8.x4.shared.b16"
    c_blocks = bn // 64
    entry_kw = {} if backend_config is None else {"backend_config": backend_config}

    # fmt: off
    @T.function
    def main(A: T.Tensor((M, K), "float8_e4m3fn", align=16), SFA: T.Tensor((KB, M), "float32", align=16),
             B: T.Tensor((N, K), "float8_e4m3fn", align=16), SFB: T.Tensor((NB, KB), "float32", align=16),
             D: T.Tensor((M, N), "bfloat16", align=16)):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float8_e4m3fn", 2, A.data, K, M, K, BK, BM, 1, 1, 0, 3, 2, 0)
        B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float8_e4m3fn", 2, B.data, K, N, K, BK, bn, 1, 1, 0, 3, 2, 0)
        S_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", S_map, "float32", 2, SFA.data, M, KB, M * 4, BM, 1, 1, 1, 0, 0, 2, 0)
        D_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", D_map, "bfloat16", 2, D.data, N, M, N * 2, 64, 64, 1, 1, 0, 3, 2, 0)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=num_ctas, block=384),
                       kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1), **entry_kw)
        tx = T.cuda.thread_idx("x")
        bid = T.cuda.block_idx("x")
        A_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, BM, BK), "float8_e4m3fn", scope="shared", align=1024)])
        B_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, bn, BK), "float8_e4m3fn", scope="shared", align=1024)])
        S_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, BM), "float32", scope="shared", align=128)])
        C_s = T.alloc_tensor(ty_args=[T.Tensor((2, c_blocks, 64, 64), "bfloat16", scope="shared", align=1024)])
        bars = NamedBarriers(reserved=(1, 2))  # 1 + wg: per-consumer epilogue barriers
        pool = SMEMPool()
        full = TMABar(pool, STAGES)
        empty = MBarrier(pool, STAGES)
        pool.commit()
        full.init(1)
        empty.init(2)  # one arrive per consumer warpgroup
        it = T.alloc_local([1], "int32")
        it[0] = 0
        if tx == 0:
            T.ptx.prefetch.tensormap(T.address_of(A_map))
            T.ptx.prefetch.tensormap(T.address_of(B_map))
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()

        if tx < 128:
            # ---------------- producer warpgroup ----------------
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](PRODUCER_REGS)
            if tx == 0:
                for j in T.serial(rounds):
                    t = bid + j * num_ctas
                    if t < n_tiles:
                        g = t // (GROUP_M * NT)
                        first = g * GROUP_M
                        gsize = T.min(MT - first, GROUP_M)
                        m_blk = first + (t % (GROUP_M * NT)) % gsize
                        n_blk = (t % (GROUP_M * NT)) // gsize
                        for kb in T.serial(KB):
                            stage = it[0] % STAGES
                            empty.wait(stage, ((it[0] // STAGES) & 1) ^ 1)
                            full.arrive(stage, stage_tx)
                            T.ptx[tma](A_s.ptr_to([stage, 0, 0]), T.address_of(A_map), kb * BK, m_blk * BM, full.ptr_to([stage]))
                            T.ptx[tma](B_s.ptr_to([stage, 0, 0]), T.address_of(B_map), kb * BK, n_blk * bn, full.ptr_to([stage]))
                            T.ptx[tma](S_s.ptr_to([stage, 0]), T.address_of(S_map), m_blk * BM, kb, full.ptr_to([stage]))
                            it[0] = it[0] + 1
        else:
            # ---------------- consumer warpgroups ----------------
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](CONSUMER_REGS)
            wg = tx // 128 - 1
            ct = tx % 128
            acc = T.alloc_tensor(ty_args=[T.Tensor((acc_n,), "float32", scope="local")])
            tmp = T.alloc_tensor(ty_args=[T.Tensor((acc_n,), "float32", scope="local")])
            da0 = W.smem_descriptor(A_s.ptr_to([0, wg * 64, 0]))
            db0 = W.smem_descriptor(B_s.ptr_to([0, 0, 0]))
            r0 = (ct // 32) * 16 + (ct % 32) // 4  # this thread's rows: r0 and r0 + 8 of its 64
            for j in T.serial(rounds):
                t = bid + j * num_ctas
                if t < n_tiles:
                    g = t // (GROUP_M * NT)
                    first = g * GROUP_M
                    gsize = T.min(MT - first, GROUP_M)
                    m_blk = first + (t % (GROUP_M * NT)) % gsize
                    n_blk = (t % (GROUP_M * NT)) // gsize
                    for i in T.unroll(acc_n):
                        acc[i] = T.float32(0.0)
                    for kb in T.serial(KB):
                        stage = it[0] % STAGES
                        full.wait(stage, (it[0] // STAGES) & 1)
                        a0 = W.advance_descriptor(da0, stage * a_stage_bytes)
                        b0 = W.advance_descriptor(db0, stage * b_stage_bytes)
                        W.chain(op, tmp, [a_tile.k_step(op, s, a0) for s in range(k_steps)],
                                [b_tile.k_step(op, s, b0) for s in range(k_steps)], accumulate=False)
                        W.wait(0, (tmp, acc_n))
                        sa0 = S_s[stage, wg * 64 + r0]
                        sa1 = S_s[stage, wg * 64 + r0 + 8]
                        if ct == 0:
                            empty.arrive(stage)
                        # promotion: acc += tmp * sfa[row] * sfb[128-column block of the column].
                        # A bn-wide tile spans at most two sfb blocks (nb0, nb0 + 1); `split` is the
                        # tile column where the second one starts.
                        nb0 = (n_blk * bn) // 128
                        split = 128 - (n_blk * bn) % 128
                        sb0 = SFB[nb0, kb]
                        sb1 = SFB[T.min(nb0 + 1, NB - 1), kb]
                        s00 = sa0 * sb0
                        s01 = sa0 * sb1
                        s10 = sa1 * sb0
                        s11 = sa1 * sb1
                        for c in T.unroll(bn // 8):
                            first = c * 8 < split
                            lo = T.Select(first, s00, s01)  # rows r0
                            hi = T.Select(first, s10, s11)  # rows r0 + 8
                            acc[c * 4] = acc[c * 4] + tmp[c * 4] * lo
                            acc[c * 4 + 1] = acc[c * 4 + 1] + tmp[c * 4 + 1] * lo
                            acc[c * 4 + 2] = acc[c * 4 + 2] + tmp[c * 4 + 2] * hi
                            acc[c * 4 + 3] = acc[c * 4 + 3] + tmp[c * 4 + 3] * hi
                        it[0] = it[0] + 1
                    # epilogue: stmatrix into the swizzled staging tile, then TMA store
                    if ct == 0:
                        T.ptx.cp.async_.bulk.wait_group.read(0)
                    T.ptx["bar.sync"](T.uint32(1) + T.Cast("uint32", wg), T.uint32(128))
                    lane = ct % 32
                    mat = lane // 8
                    srow = (ct // 32) * 16 + (mat % 2) * 8 + lane % 8
                    for p in T.unroll(acc_n // 8):
                        words = T.alloc_local([4], "uint32")
                        for h in T.unroll(2):
                            r = (2 * p + h) * 4
                            T.ptx["cvt.rn.bf16x2.f32"](words[2 * h], acc[r + 1], acc[r])
                            T.ptx["cvt.rn.bf16x2.f32"](words[2 * h + 1], acc[r + 3], acc[r + 2])
                        col = (2 * p + mat // 2) * 8
                        chunk = ((col % 64) // 8) ^ (srow % 8)
                        T.ptx[stm](C_s.ptr_to([wg, col // 64, srow, chunk * 8]), words[0], words[1], words[2], words[3])
                    T.ptx.fence.proxy.async_.shared__cta()
                    T.ptx["bar.sync"](T.uint32(1) + T.Cast("uint32", wg), T.uint32(128))
                    if ct == 0:
                        for b in T.unroll(c_blocks):
                            T.ptx[tma_store](T.address_of(D_map), T.Cast("int32", n_blk * bn + b * 64),
                                             T.Cast("int32", m_blk * BM + wg * 64), C_s.ptr_to([wg, b, 0, 0]))
                        T.ptx.cp.async_.bulk.commit_group()
            if ct == 0:
                T.ptx.cp.async_.bulk.wait_group(0)
    # fmt: on
    _ = bars
    return main


KERNEL_META = {
    "name": "fp8_blockscale_gemm_sm90",
    "category": "gemm",
    "runtime_cuda_archs": ["sm_90a"],
}
# DeepGEMM tests/generators.py enumerate_normal, bf16-output (n, k) list. Gated: m = 4096 (its m >= 2048 rows;
# the m = 4096 "backward" rows are the same K-major shapes on sm90 fp8). Tracked: m = 128 (decode-like).
_NK = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]
GATED_SHAPES = [(4096, n, k) for n, k in _NK]
TRACKED_SHAPES = [(128, n, k) for n, k in _NK]
CONFIGS = [{"M": M, "N": N, "K": K, "label": f"{M}x{N}x{K}"} for M, N, K in GATED_SHAPES + TRACKED_SHAPES]


def prepare_data(M, N, K):
    """FP8 operands with DeepGEMM's quantization recipe (fp32 scales, no UE8M0)."""
    dev = torch.device("cuda", torch.cuda.current_device())
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    kb = -(-K // 128)
    ap = torch.zeros(M, kb * 128, device=dev, dtype=torch.bfloat16)
    ap[:, :K] = a
    sfa = ap.view(M, kb, 128).abs().float().amax(-1).clamp(1e-4) / 448.0
    a8 = (ap.view(M, kb, 128) / sfa.unsqueeze(-1)).to(torch.float8_e4m3fn).view(M, kb * 128)[:, :K].contiguous()
    nb = -(-N // 128)
    bp = torch.zeros(nb * 128, kb * 128, device=dev, dtype=torch.bfloat16)
    bp[:N, :K] = b
    sfb = bp.view(nb, 128, kb, 128).abs().float().amax(dim=(1, 3)).clamp(1e-4) / 448.0
    b8 = (bp.view(nb, 128, kb, 128) / sfb[:, None, :, None]).to(torch.float8_e4m3fn).view(nb * 128, kb * 128)
    b8 = b8[:N, :K].contiguous()
    d = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    return a8, sfa, b8, sfb, d


def reference(a8, sfa, b8, sfb):
    M, K = a8.shape
    N = b8.shape[0]
    kb = sfa.shape[1]
    ap = torch.zeros(M, kb * 128, device=a8.device)
    ap[:, :K] = a8.float()
    bp = torch.zeros(-(-N // 128) * 128, kb * 128, device=a8.device)
    bp[:N, :K] = b8.float()
    ap = (ap.view(M, kb, 128) * sfa.unsqueeze(-1)).view(M, -1)
    bp = (bp.view(-1, 128, kb, 128) * sfb[:, None, :, None]).view(bp.shape)
    return (ap @ bp[:N].t()).to(torch.bfloat16)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def compile_persistent(M, N, K, *, bn=128, backend_config=None):
    from tvm_tirx.backend.cuda.runtime import occupancy

    from tirx_kernels.runner import compile_kernel

    ex = compile_kernel(make_kernel(M, N, K, bn=bn, backend_config=backend_config), backend_config=backend_config)
    ctas = occupancy.max_active_clusters(ex, block=384, cluster=1)
    if ctas != DEFAULT_CTAS:
        ex = compile_kernel(
            make_kernel(M, N, K, bn=bn, num_ctas=ctas, backend_config=backend_config),
            backend_config=backend_config,
        )
    return ex


def get_kernel(M, N, K, *, backend_config=None, **kwargs):
    return make_kernel(M, N, K, backend_config=backend_config)


def run_test(M, N, K, *, backend_config=None, **kwargs):
    a8, sfa, b8, sfb, d = prepare_data(M, N, K)
    ex = compile_persistent(M, N, K, backend_config=backend_config)
    ex(a8, sfa.t().contiguous(), b8, sfb, d)
    diff = calc_diff(d, reference(a8, sfa, b8, sfb))
    assert diff < 1e-3, f"calc_diff {diff}"

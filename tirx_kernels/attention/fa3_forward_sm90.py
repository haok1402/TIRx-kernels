# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FA3-style attention forward for Hopper (sm_90a): bf16, head_dim 128, causal or not, varlen.

Layout: FlashInfer's ragged NHD. Q, K, V and O are ``[total_tokens, heads, 128]`` with sequence ``b``
at rows ``cu_seqlens[b] : cu_seqlens[b + 1]``. This is self-attention prefill (q_len == kv_len per
sequence); a causal row attends to keys at or before its own position.

**Grid.** Each tile is one 128-row Q block of one (sequence, head); a CTA has 3 warpgroups, 384 threads.
- **Non-causal: one tile per CTA.** Every tile has the same length.
- **Causal: persistent.** One CTA per SM runs a host-planned tile list, using FlashInfer's
  ``PrefillSM90Plan`` (longest first, least-loaded CTA; see ``work_lists``). This balances the
  1..n-block causal tiles, and it overlaps each tile's epilogue with the next tile's first K and Q loads.

**Warpgroups.**
- **WG0, producer:** ``setmaxnreg`` 24. One thread issues the Q tile, then K and V blocks of 128
  rows into 2-stage rings. K and V have separate full (TMA) and empty (consumer) barriers, so QKᵀ can
  start before V lands.
- **WG1 and WG2, consumers:** ``setmaxnreg`` 240. Each owns 64 Q rows. Both read every K/V stage and
  release it per warp (8 arrivals).

**Consumer loop** (FA3's intra-warpgroup overlap; see sm90-probe ``b3/B3-design.md``). KV blocks
run in descending order. The first block is the only one that needs masking: the causal diagonal,
or the non-causal KV tail. It is peeled, and masked with one compare per register against a per-row
limit (no ``T.Or``; see ``mask_first_block``). Then for each following block:

    S = Q Kᵀ (in flight) ; rescale O ; O += P_prev V_prev (rs) ; wait 1 ; release K ;
    softmax(S) ; wait 0 ; release V_prev ; P = bf16(S)

**Online softmax:** with ``c = log2(e) / sqrt(128)``, P = exp2(S c - m c). The running sum ``l`` is
kept per lane and reduced over the quad once, at the end. O starts at zero, so every PV chain
accumulates.

**Non-finite inputs** behave as in FA3. Masked scores become -inf, so P = 0 there, but PV still
multiplies that 0 by V. A NaN or inf in a masked V row (e.g. a causal row's future keys in the same KV
block) therefore turns the output row into NaN, exactly as FA3 does. Review checked this matches FA3
element for element.

**Persistent mode** (causal default, ``persistent=``):
- **K/V rings:** they continue across tiles; ``kv_base`` is the ring position of the tile's first block.
- **Q tile:** the producer issues the next tile's first K, then waits on ``q_empty`` before loading
  its Q. Both consumer warpgroups arrive there once their last QK^T of the tile has retired.
- **O staging:** O has its own 32 KB staging tile, not the Q tile. Each warpgroup's TMA-store leader
  waits for its previous store to finish reading before the slice is rewritten.
- **Ping-pong:** both warpgroups run the same GEMM sections per tile. Warpgroup 2's last ``bar.arrive``
  of a tile is therefore the opening arrive of the next tile. Only the last tile's arrive is left
  unconsumed, which is harmless at CTA exit.
"""

import ctypes
import heapq
import math
from typing import Any

import torch
import tvm_ffi

import tirx_kernels.tirx_lite as txl
from tirx_kernels.runner import compile_kernel, resolve_backend_config
from tvm_tirx.backend.cuda.lang import NamedBarriers
from tvm_tirx.backend.cuda.lang import wgmma as W
from tvm_tirx.script import tirx as T

HEAD_DIM = 128
BM = 128  # Q rows per CTA (2 consumer warpgroups x 64)
BN = 128  # KV rows per block
STAGES = 2
E = 64  # bf16 elements per 128-byte swizzle row
TMA3D = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
TMA3D_STORE = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
STMATRIX_X4 = "stmatrix.sync.aligned.m8n8.x4.shared.b16"
FULL = 0xFFFFFFFF
QK = W.WGMMA(BN, "bfloat16")  # S[64, BN] = Q[64, 128] K[BN, 128]^T
PV = W.WGMMA(HEAD_DIM, "bfloat16", a_in_registers=True)  # O[64, 128] += P[64, BN] V[BN, 128]
QK_STEPS = HEAD_DIM * 2 // QK.k_bytes
PV_STEPS = BN // PV.k
TILE_BYTES = BN * HEAD_DIM * 2
SCALE_LOG2 = math.log2(math.e) / math.sqrt(HEAD_DIM)

KERNEL_META = {
    "name": "fa3_forward_sm90",
    "category": "attention",
    "runtime_cuda_archs": ["sm_90a"],
}
# Correctness configs: lengths that are and are not multiples of 128, varlen batches, both masks.
CONFIGS = [
    {
        "seq_lens": lens,
        "num_heads": h,
        "causal": causal,
        "label": f"{tag}_h{h}_{'causal' if causal else 'full'}",
    }
    for tag, lens, h in [
        ("s128", (128,), 2),
        ("s1000", (1000,), 2),
        ("s4096", (4096,), 4),
        ("varlen", (1, 127, 129, 640, 1777), 3),
        ("b2_s2048", (2048, 2048), 4),
        ("zero_len", (0, 0), 2),  # nothing to compute: no launch
        ("zero_mixed", (0, 300, 0, 129, 0), 2),  # empty sequences own no tiles and no rows
    ]
    for causal in (False, True)
] + [
    # B3 gate shapes (plan section B3; FlashInfer FA3 is the bench reference)
    {
        "seq_lens": (s,) * b,
        "num_heads": h,
        "causal": causal,
        "label": f"gate_b{b}_h{h}_s{s}_{'causal' if causal else 'full'}",
    }
    for b, h, s in [(4, 16, 4096), (2, 16, 8192), (1, 16, 16384), (8, 16, 4096), (4, 32, 8192)]
    for causal in (False, True)
]


class _Regs4:
    """A 4-register window of a local uint32 tensor: one rs A operand."""

    def __init__(self, tensor, start):
        self.tensor, self.start = tensor, start

    def __getitem__(self, i):
        return self.tensor[self.start + i]

    def __iter__(self):
        return iter(self.tensor[self.start + i] for i in range(4))

    def __len__(self):
        return 4


def make_kernel(
    num_heads: int, causal: bool, *, backend_config=None, pingpong=None, persistent=None
):
    bc = resolve_backend_config(backend_config)
    H = num_heads
    if pingpong is None:
        # Measured on H200 (sm90-probe logs/b3/v3.log): ping-pong helps causal slightly and costs
        # non-causal 1-2%, where the intra-warpgroup overlap already hides softmax.
        pingpong = causal
    if persistent is None:
        # Measured on H200 (sm90-probe fa3c/logs/v5*.log, v6.log): +3-6% causal; non-causal tiles
        # are all the same length and the persistent body costs 4-10% there.
        persistent = causal
    # Inter-warpgroup ping-pong (FA3's WarpScheduler): one barrier per consumer warpgroup, 256 threads
    # (its own 128 syncing + the other's 128 arriving). A warpgroup syncs on its own barrier before
    # issuing a block's GEMMs and arrives on the other's after, so one issues GEMMs while the other
    # runs softmax.
    bars = NamedBarriers()
    sched = [bars.alloc(256), bars.alloc(256)]
    assert sched[1].barrier_id == sched[0].barrier_id + 1
    epi = [
        bars.alloc(128),
        bars.alloc(128),
    ]  # per consumer warpgroup: O staged before its TMA store
    assert epi[1].barrier_id == epi[0].barrier_id + 1

    @txl.kernel()
    def fa3_forward_sm90(
        Q_map: txl.Ptr(txl.TensorMap),
        K_map: txl.Ptr(txl.TensorMap),
        V_map: txl.Ptr(txl.TensorMap),
        O_map: txl.Ptr(txl.TensorMap),
        O: txl.Tensor((txl.int64(),), "bfloat16"),
        cu_seqlens: txl.Tensor((txl.int64(),), "int32"),
        tiles: txl.Tensor((txl.int64(),), "int32"),
        work_ptr: txl.Tensor((txl.int64(),), "int32"),
        num_ctas: "int32",
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=num_ctas, block=384),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=bc,
        )
        tid = txl.thread_id()
        # 0: producer, 1..2: consumers. Bound ONCE, here, where every warp is converged: the builtin's
        # value is a full-mask __shfl_sync re-emitted at each use, and a use inside a short-circuit
        # (`||`, `?:`) runs it divergently; the causal mask hung that way (sm90-probe ISSUE-008).
        wg = txl.local_scalar("int32")
        txl.assign(wg, T.cuda.broadcast_warpgroup_id())
        smem = txl.smem_pool()
        q_s = smem.alloc([BM * HEAD_DIM], "bfloat16", align=1024)
        k_s = smem.alloc([STAGES * BN * HEAD_DIM], "bfloat16", align=1024)
        v_s = smem.alloc([STAGES * BN * HEAD_DIM], "bfloat16", align=1024)
        # O staging: its own tile when persistent (the next tile's Q loads while this one stores),
        # else the Q tile, which is dead by the epilogue.
        o_s = smem.alloc([BM * HEAD_DIM], "bfloat16", align=1024) if persistent else q_s
        q_full = txl.TMABar(smem, 1)
        k_full = txl.TMABar(smem, STAGES)
        v_full = txl.TMABar(smem, STAGES)
        k_empty = txl.MBarrier(smem, STAGES)
        v_empty = txl.MBarrier(smem, STAGES)
        q_empty = txl.MBarrier(smem, 1) if persistent else None
        q_full.init(1)
        k_full.init(1)
        v_full.init(1)
        k_empty.init(8)  # one arrival per consumer warp
        v_empty.init(8)
        if persistent:
            q_empty.init(8)
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.cuda.cta_sync()

        def ldg_i32(ptr):
            value = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.b32(value, ptr)
            return value

        cta = txl.cta_id()
        head = txl.local_scalar("int32")
        m_blk = txl.local_scalar("int32")
        seq0 = txl.local_scalar("int32")
        seq_len = txl.local_scalar("int32")
        n_blocks = txl.local_scalar("int32")
        if persistent:
            # This CTA's tiles are work_ptr[cta] .. work_ptr[cta + 1] of the table; ``kv_base`` is the
            # K/V ring position of the current tile's first block (the rings run on across tiles).
            w_begin = ldg_i32(work_ptr.ptr_to([cta]))
            w_count = txl.local_scalar("int32")
            txl.assign(w_count, ldg_i32(work_ptr.ptr_to([cta + 1])) - w_begin)
            kv_base = txl.local_scalar("int32")
            txl.assign(kv_base, 0)
        else:
            kv_base = 0

        def load_tile(w):
            batch = ldg_i32(tiles.ptr_to([3 * w]))
            txl.assign(head, ldg_i32(tiles.ptr_to([3 * w + 1])))
            txl.assign(m_blk, ldg_i32(tiles.ptr_to([3 * w + 2])))
            txl.assign(seq0, ldg_i32(cu_seqlens.ptr_to([batch])))
            txl.assign(seq_len, ldg_i32(cu_seqlens.ptr_to([batch + 1])) - seq0)
            if causal:
                txl.assign(n_blocks, m_blk + 1)  # BM == BN: KV blocks 0..m_blk
            else:
                txl.assign(n_blocks, (seq_len + BN - 1) // BN)

        if not persistent:
            # The one tile, read here by every warp before the role split (as its loads overlap the
            # barrier setup); reading it after setmaxnreg cost non-causal about 1%.
            load_tile(cta)

        def for_each_tile(body):
            """``body(t)`` for this CTA's t-th tile: every tile of its list, or tile ``cta`` only."""
            if persistent:
                with txl.serial(w_count) as t:
                    load_tile(w_begin + t)
                    body(t)
                    txl.assign(kv_base, kv_base + n_blocks)
            else:
                body(0)

        def stage_phase(n):
            return n % STAGES, (n // STAGES) & 1

        with txl.if_(wg == 0):
            # ------------------------------------------------------------ producer
            with txl.then_():
                txl.ptx["setmaxnreg.dec.sync.aligned.u32"](24)
                with txl.if_(tid == 0), txl.then_():

                    def load_q(t):
                        if persistent:
                            # the previous tile's consumers have retired their last QK^T on q_s
                            q_empty.wait(0, (t & 1) ^ 1)
                        q_full.arrive(0, tx_count=BM * HEAD_DIM * 2)
                        for a in range(HEAD_DIM // E):
                            txl.ptx[TMA3D](
                                q_s.ptr_to([a * BM * E]),
                                txl.address_of(Q_map),
                                txl.int32(a * E),
                                head,
                                seq0 + m_blk * BM,
                                q_full.ptr_to([0]),
                            )

                    def load_kv(ring, full, empty, gmap, n):
                        st, ph = stage_phase(kv_base + n)
                        row = seq0 + (n_blocks - 1 - n) * BN
                        empty.wait(st, ph ^ 1)
                        full.arrive(st, tx_count=TILE_BYTES)
                        for a in range(HEAD_DIM // E):
                            txl.ptx[TMA3D](
                                ring.ptr_to([st * BN * HEAD_DIM + a * BN * E]),
                                txl.address_of(gmap),
                                txl.int32(a * E),
                                head,
                                row,
                                full.ptr_to([st]),
                            )

                    def produce(t):
                        if persistent:  # FA3's order: the first K can go before the Q tile is free
                            load_kv(k_s, k_full, k_empty, K_map, 0)
                            load_q(t)
                            load_kv(v_s, v_full, v_empty, V_map, 0)
                            with txl.serial(n_blocks - 1) as i:
                                load_kv(k_s, k_full, k_empty, K_map, i + 1)
                                load_kv(v_s, v_full, v_empty, V_map, i + 1)
                        else:
                            load_q(t)
                            with txl.serial(n_blocks) as n:
                                load_kv(k_s, k_full, k_empty, K_map, n)
                                load_kv(v_s, v_full, v_empty, V_map, n)

                    for_each_tile(produce)
            # ------------------------------------------------------------ consumers
            with txl.else_():
                txl.ptx["setmaxnreg.inc.sync.aligned.u32"](240)
                cw = wg - 1

                def sched_sync():
                    if pingpong:
                        txl.ptx["bar.sync"](
                            txl.cast(sched[0].barrier_id + cw, "uint32"), txl.uint32(256)
                        )

                def sched_arrive():
                    if pingpong:
                        txl.ptx["bar.arrive"](
                            txl.cast(sched[0].barrier_id + 1 - cw, "uint32"), txl.uint32(256)
                        )

                # Warpgroup 1 issues first. Both warpgroups run the same GEMM sections per tile, so
                # warpgroup 2's last arrive of one tile is the opening arrive of the next; after the
                # last tile it is left unconsumed, which is harmless at CTA exit.
                if pingpong:
                    with txl.if_(cw == 1), txl.then_():
                        txl.ptx["bar.arrive"](txl.uint32(sched[0].barrier_id), txl.uint32(256))
                tw = tid & 127
                lane0 = (tid & 31) == 0
                s_acc = txl.alloc_local([QK.num_acc], "float32")
                o_acc = txl.alloc_local([PV.num_acc], "float32")
                p_half = txl.alloc_local([QK.num_acc], "bfloat16")
                p_regs = p_half.view("uint32")
                m_run = txl.alloc_local([2], "float32")
                l_run = txl.alloc_local([2], "float32")
                alpha = txl.alloc_local([2], "float32")
                mx = txl.alloc_local([2], "float32")
                mc = txl.alloc_local([2], "float32")  # m * c
                lim = txl.alloc_local([2], "int32")
                q_tile = W.SmemTile(q_s.ptr_to([cw * 64 * E]), rows=BM)

                def k_tile(st):
                    return W.SmemTile(k_s.ptr_to([st * BN * HEAD_DIM]), rows=BN)

                def v_tile(st):
                    return W.SmemTile(v_s.ptr_to([st * BN * HEAD_DIM]), rows=BN, major="MN")

                def qk_chain(st):
                    W.chain(QK, s_acc, q_tile.k_steps(QK, QK_STEPS), k_tile(st).k_steps(QK, QK_STEPS),
                            accumulate=False)  # fmt: skip

                def pv_chain(st):
                    W.chain(PV, o_acc, [_Regs4(p_regs, s * 4) for s in range(PV_STEPS)],
                            v_tile(st).k_steps(PV, PV_STEPS), accumulate=True, transpose_b=True)  # fmt: skip

                def release(bar, st):
                    with txl.if_(lane0), txl.then_():
                        bar.arrive(st)

                def fmax(dst, a, b):
                    # max.f32 is one FMNMX; T.max lowers to a NaN-propagating FSETP/FSETP/FSEL triple
                    txl.ptx["max.f32"](dst, a, b)

                def softmax(first):
                    for h in range(2):
                        txl.assign(mx[h], T.float32(-math.inf) if first else m_run[h])
                    for r in range(QK.num_acc):
                        h = (r % 4) // 2
                        fmax(mx[h], mx[h], s_acc[r])
                    for h in range(2):
                        for d in (1, 2):
                            fmax(mx[h], mx[h], T.cuda.__shfl_xor_sync(T.uint32(FULL), mx[h], d, 32))
                        if first:
                            txl.assign(alpha[h], T.float32(0.0))
                            txl.assign(l_run[h], T.float32(0.0))
                        else:
                            txl.assign(alpha[h], T.exp2((m_run[h] - mx[h]) * T.float32(SCALE_LOG2)))
                            txl.assign(l_run[h], l_run[h] * alpha[h])
                        txl.assign(m_run[h], mx[h])
                        txl.assign(mc[h], mx[h] * T.float32(SCALE_LOG2))
                    for r in range(QK.num_acc):
                        h = (r % 4) // 2
                        # S c - m c: one FFMA feeding MUFU.EX2 (was FADD + FMUL)
                        p = T.exp2(s_acc[r] * T.float32(SCALE_LOG2) - mc[h])
                        txl.assign(l_run[h], l_run[h] + p)
                        txl.assign(p_half[r], T.Cast("bfloat16", p))

                def rescale_o():
                    for r in range(PV.num_acc):
                        txl.assign(o_acc[r], o_acc[r] * alpha[(r % 4) // 2])

                def mask_first_block():
                    # Register r sits at row 16 warp + lane / 4 + 8 h, column 8 (r / 4) + r % 2 +
                    # 2 (lane % 4) of the block (W.acc_coord). It is masked iff its column is past the
                    # row's last visible key (the diagonal, and the sequence end). Folding the lane part
                    # and the block offset into a per-row limit leaves one compare against a constant
                    # per register: ISETP + FSEL. The same mask as T.Or(k_col >= seq_len, k_col > q_row),
                    # whose short-circuit `||` NVRTC compiled into a branch per register (ISSUE-017).
                    # Kept once ISSUE-017 is fixed: there, the two-compare form is branch-free and
                    # equally fast on causal, but non-causal runs 0.5-1.1% slower than with this form
                    # (sm90-probe fa3c/logs/v4_table.log).
                    lane = tw & 31
                    j0 = n_blocks - 1
                    for h in range(2):
                        last = seq_len - 1
                        if causal:
                            q_row = m_blk * BM + cw * 64 + (tw >> 5) * 16 + (lane >> 2) + 8 * h
                            last = T.min(q_row, last)
                        txl.assign(lim[h], last - j0 * BN - 2 * (lane & 3))
                    for r in range(QK.num_acc):
                        masked = T.int32(8 * (r // 4) + r % 2) > lim[(r % 4) // 2]
                        txl.assign(s_acc[r], T.Select(masked, T.float32(-math.inf), s_acc[r]))

                def consume(t):
                    for r in range(PV.num_acc):
                        txl.assign(o_acc[r], T.float32(0.0))
                    q_full.wait(0, t & 1 if persistent else 0)

                    # first (highest) KV block: the only one masked (causal diagonal / KV tail)
                    st0, ph0 = stage_phase(kv_base)
                    k_full.wait(st0, ph0)
                    sched_sync()
                    qk_chain(st0)
                    sched_arrive()
                    W.wait(0, (s_acc, QK.num_acc))
                    release(k_empty, st0)
                    mask_first_block()
                    softmax(True)

                    with txl.serial(n_blocks - 1) as i:
                        n = kv_base + i + 1
                        st, ph = stage_phase(n)
                        pst, pph = stage_phase(n - 1)
                        k_full.wait(st, ph)
                        sched_sync()
                        qk_chain(st)
                        rescale_o()
                        v_full.wait(pst, pph)
                        pv_chain(pst)
                        sched_arrive()
                        W.wait(1, (s_acc, QK.num_acc))
                        release(k_empty, st)
                        softmax(False)
                        W.wait(0, (o_acc, PV.num_acc))
                        release(v_empty, pst)

                    if persistent:  # every QK^T of this tile has retired: the next Q may load
                        release(q_empty, 0)
                    lst, lph = stage_phase(kv_base + n_blocks - 1)
                    rescale_o()
                    v_full.wait(lst, lph)
                    pv_chain(lst)
                    W.wait(0, (o_acc, PV.num_acc))
                    release(v_empty, lst)
                    epilogue()

                def epilogue():
                    # quad-reduce l, normalize, store the rows inside the sequence
                    for h in range(2):
                        for d in (1, 2):
                            txl.assign(
                                l_run[h],
                                l_run[h] + T.cuda.__shfl_xor_sync(T.uint32(FULL), l_run[h], d, 32),
                            )
                        txl.assign(alpha[h], T.float32(1.0) / l_run[h])
                    row0 = m_blk * BM + cw * 64  # this warpgroup's first Q row in the sequence
                    with txl.if_(row0 + 64 <= seq_len):
                        # Whole 64 rows inside the sequence: stmatrix into this warpgroup's own 64-row
                        # slice of the staging tile, then one TMA store per 64-column atom. Single tile:
                        # the slice is the Q tile's, which only this warpgroup reads and whose last
                        # QK^T has retired. Persistent: the O tile, once this warpgroup's previous
                        # store has read it. A ragged tile's tail rows belong to the next sequence, so
                        # partial tiles take the masked direct path below.
                        with txl.then_():
                            if persistent:
                                with txl.if_(tw == 0), txl.then_():
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                                txl.ptx["bar.sync"](
                                    txl.cast(epi[0].barrier_id + cw, "uint32"), txl.uint32(128)
                                )
                            lane = tw & 31
                            mat = lane >> 3
                            srow = (tw >> 5) * 16 + (mat & 1) * 8 + (lane & 7)
                            words = txl.alloc_local([4], "uint32")
                            for p in range(PV.num_acc // 8):  # two 8-column chunks per stmatrix.x4
                                for hh in range(2):
                                    r = (2 * p + hh) * 4
                                    txl.ptx["cvt.rn.bf16x2.f32"](
                                        words[2 * hh], o_acc[r + 1] * alpha[0], o_acc[r] * alpha[0]
                                    )
                                    txl.ptx["cvt.rn.bf16x2.f32"](
                                        words[2 * hh + 1],
                                        o_acc[r + 3] * alpha[1],
                                        o_acc[r + 2] * alpha[1],
                                    )
                                col = (2 * p + (mat >> 1)) * 8
                                # 128B swizzle of 16-byte chunks
                                chunk = ((col & 63) >> 3) ^ (srow & 7)
                                txl.ptx[STMATRIX_X4](
                                    o_s.ptr_to([(col >> 6) * BM * E + (cw * 64 + srow) * E + chunk * 8]),
                                    words[0], words[1], words[2], words[3],
                                )  # fmt: skip
                            txl.ptx.fence.proxy.async_.shared__cta()
                            txl.ptx["bar.sync"](
                                txl.cast(epi[0].barrier_id + cw, "uint32"), txl.uint32(128)
                            )
                            with txl.if_(tw == 0), txl.then_():
                                for a in range(HEAD_DIM // E):
                                    txl.ptx[TMA3D_STORE](
                                        txl.address_of(O_map),
                                        txl.int32(a * E),
                                        head,
                                        seq0 + row0,
                                        o_s.ptr_to([a * BM * E + cw * 64 * E]),
                                    )
                                txl.ptx.cp.async_.bulk.commit_group()
                                if not persistent:
                                    # shared memory outlives the read
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                        with txl.else_():
                            packed = txl.local_scalar("uint32")
                            for r in range(0, PV.num_acc, 2):
                                row, col = W.acc_coord(tw, r)
                                q_row = row0 + row
                                h = (r % 4) // 2
                                with txl.if_(q_row < seq_len), txl.then_():
                                    txl.ptx["cvt.rn.bf16x2.f32"](
                                        packed, o_acc[r + 1] * alpha[h], o_acc[r] * alpha[h]
                                    )
                                    off = (
                                        txl.cast(seq0 + q_row, "int64") * H + head
                                    ) * HEAD_DIM + col
                                    txl.ptx["st.global.b32"](O.ptr_to([off]), packed)

                for_each_tile(consume)
                if persistent:
                    with txl.if_(tw == 0), txl.then_():
                        # shared memory outlives the last read
                        txl.ptx.cp.async_.bulk.wait_group.read(0)

    return fa3_forward_sm90


# --------------------------------------------------------------------------- host side
class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte TensorMap payload."""

    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        self.ptr = ctypes.c_void_p((ctypes.addressof(self._storage) + 63) & ~63)


def _tensor_map(t, box_rows):
    """3-D map of a ragged ``[tokens, heads, 128]`` bf16 tensor: dims (d, h, tokens), box (64, 1, rows)."""
    tokens, heads, d = t.shape
    desc = _AlignedTensorMap()
    tvm_ffi.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr, "bfloat16", 3, ctypes.c_void_p(int(t.data_ptr())),
        d, heads, tokens, d * 2, heads * d * 2, E, 1, box_rows, 1, 1, 1,
        0, 3, 2, 0,
    )  # fmt: skip
    return desc


def tile_table(seq_lens, num_heads, causal, device="cuda"):
    """(sequence, head, Q block) per CTA, in launch order, for the one-tile-per-CTA grid.

    The Q blocks of one (sequence, head) are consecutive, so the CTAs in flight share a few heads'
    K/V in L2. With heads innermost, about a hundred 2 MB K/V streams thrashed the 50 MB L2. Causal
    tiles run longest first within each head (Q block descending), which also balances each wave.
    """
    tiles = []
    for b, n in enumerate(seq_lens):
        blocks = range((n + BM - 1) // BM)
        for h in range(num_heads):
            tiles.extend((b, h, m) for m in (reversed(blocks) if causal else blocks))
    return torch.tensor(tiles, dtype=torch.int32, device=device).view(-1), len(tiles)


def work_lists(seq_lens, num_heads, causal, num_ctas):
    """Per-CTA tile lists for the persistent grid: FlashInfer's ``PrefillSM90Plan``.

    Heads outer, sequences by length descending, Q blocks descending; each tile goes to the CTA with
    the least work so far, at cost ``2 * BM + kv_len`` (kv_len is the tile's causal KV extent). Each
    CTA runs its list in order, so the CTAs' first tiles are the longest and the tiles in flight
    still cover a few heads at a time (L2 locality as in ``tile_table``).
    """
    heap = [(0, c) for c in range(num_ctas)]
    lists = [[] for _ in range(num_ctas)]
    for h in range(num_heads):
        for b in sorted(range(len(seq_lens)), key=lambda b: -seq_lens[b]):
            n = seq_lens[b]
            nq = (n + BM - 1) // BM
            for m in reversed(range(nq)):
                kv_len = n if (not causal or m + 1 == nq) else (m + 1) * BM
                cost, c = heapq.heappop(heap)
                heapq.heappush(heap, (cost + 2 * BM + kv_len, c))
                lists[c].append((b, h, m))
    return lists


def get_kernel(
    num_heads=None, causal=False, *, backend_config=None, pingpong=None, persistent=None, **kwargs
):
    return make_kernel(
        num_heads, causal, backend_config=backend_config, pingpong=pingpong, persistent=persistent
    ).func


class Launcher:
    """The compiled kernel for one (heads, causal); ``bind`` specializes it to tensors and lengths.

    Construction only compiles (no CUDA allocation), so ``prepare_bench`` can build it before the
    workload receives a GPU.
    """

    def __init__(self, num_heads, causal, *, backend_config=None, pingpong=None, persistent=None):
        self.persistent = causal if persistent is None else persistent
        self.ex = compile_kernel(
            get_kernel(
                num_heads,
                causal,
                backend_config=backend_config,
                pingpong=pingpong,
                persistent=self.persistent,
            ),
            backend_config=backend_config,
        )
        self.num_heads, self.causal = num_heads, causal

    def bind(self, q, k, v, o, seq_lens):
        """A zero-argument launch of attention over ragged ``q, k, v`` into ``o``.

        ``seq_lens``: one length per sequence, summing to ``q.shape[0]``.
        """
        seq_lens = tuple(int(n) for n in seq_lens)
        if sum(seq_lens) != q.shape[0] or q.shape[1] != self.num_heads:
            raise ValueError(f"seq_lens {seq_lens} / heads do not match q {tuple(q.shape)}")
        cu = cu_seqlens(seq_lens, q.device)
        if not any(seq_lens):
            # No rows, so no tiles: a 0-CTA grid is a launch error (and a 0-row tensor map is
            # invalid). As FlashInfer, whose plan then holds no work, nothing is computed or written.
            def run():
                pass

            return run
        if self.persistent:
            num_tiles = sum((n + BM - 1) // BM for n in seq_lens) * self.num_heads
            num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
            lists = work_lists(seq_lens, self.num_heads, self.causal, min(num_sms, num_tiles))
            ptr = [0]
            for lst in lists:
                ptr.append(ptr[-1] + len(lst))
            tiles = torch.tensor(
                [t for lst in lists for t in lst], dtype=torch.int32, device=q.device
            ).view(-1)
            work_ptr = torch.tensor(ptr, dtype=torch.int32, device=q.device)
            num_ctas = len(lists)
        else:
            tiles, num_ctas = tile_table(seq_lens, self.num_heads, self.causal, q.device)
            work_ptr = torch.zeros(1, dtype=torch.int32, device=q.device)  # unused
        maps = (_tensor_map(q, BM), _tensor_map(k, BN), _tensor_map(v, BN), _tensor_map(o, 64))
        args = (*[mp.ptr for mp in maps], o.view(-1), cu, tiles, work_ptr, num_ctas)

        def run():
            self.ex(*args)

        # the host tensor maps and metadata outlive the closure
        run.keepalive = (maps, cu, tiles, work_ptr)
        return run


def cu_seqlens(seq_lens, device="cuda"):
    return torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )


def prepare_data(seq_lens, num_heads, **kwargs: Any):
    torch.manual_seed(0)
    total = sum(seq_lens)
    q, k, v = (
        torch.randn(total, num_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    return q, k, v


def reference(q, k, v, seq_lens, causal):
    """float32 attention per sequence (exact softmax), [tokens, heads, d]."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    start = 0
    for n in seq_lens:
        if n == 0:
            continue
        qs, ks, vs = (x[start : start + n].transpose(0, 1).float() for x in (q, k, v))
        out[start : start + n] = torch.nn.functional.scaled_dot_product_attention(
            qs, ks, vs, is_causal=causal
        ).transpose(0, 1)
        start += n
    return out


def run_test(seq_lens, num_heads, causal, *, backend_config=None, **kwargs):
    q, k, v = prepare_data(seq_lens, num_heads)
    o = torch.full_like(q, float("nan"))
    run = Launcher(num_heads, causal, backend_config=backend_config).bind(q, k, v, o, seq_lens)
    run()
    torch.cuda.synchronize()
    ref = reference(q, k, v, seq_lens, causal)
    torch.testing.assert_close(o.float(), ref, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- benchmarking
def _fa3_reference(q, k, v, seq_lens, num_heads, causal):
    """FA3 (FlashInfer ``backend="fa3"``, batch ragged prefill) on the same tensors, planned untimed."""
    import flashinfer

    workspace = torch.empty(512 << 20, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend="fa3")
    cu = cu_seqlens(seq_lens, q.device)
    wrapper.plan(cu, cu, num_heads, num_heads, HEAD_DIM, causal=causal, q_data_type=torch.bfloat16)
    return lambda: wrapper.run(q, k, v)


def prepare_bench(*, backend_config=None, **kwargs: Any):
    """Compile before the workload receives a GPU."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    cfg = {key: kwargs[key] for key in ("seq_lens", "num_heads", "causal")}
    state = {
        "config": cfg,
        "launcher": Launcher(cfg["num_heads"], cfg["causal"], backend_config=backend_config),
    }
    return prepared_gpu_benchmark(run_gpu, state, backend_config=backend_config)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, backend_config=None, **kwargs):
    """Allocate and measure after GPU assignment; FA3 is the reference.

    Kernel-only GPU time (CUPTI via Proton) is the metric, which is what the harness scores and the
    B3 gate uses. Event timing would include ~15-25 us of FlashInfer's Python dispatch per FA3 call.
    """
    from tirx_kernels.runner import bench

    timer = timer or "proton"
    cfg = prepared["config"]
    seq_lens, num_heads, causal = cfg["seq_lens"], cfg["num_heads"], cfg["causal"]
    q, k, v = prepare_data(seq_lens, num_heads)
    o = torch.empty_like(q)
    run = prepared["launcher"].bind(q, k, v, o, seq_lens)
    return bench(
        {"tir": run},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"fa3": lambda: _fa3_reference(q, k, v, seq_lens, num_heads, causal)},
        **kwargs,
    )


def run_bench(
    seq_lens,
    num_heads,
    causal,
    warmup=None,
    repeat=None,
    timer=None,
    *,
    backend_config=None,
    **kwargs,
):
    """Standalone wrapper over the same explicit prepare and GPU stages."""
    return prepare_bench(
        seq_lens=seq_lens, num_heads=num_heads, causal=causal, backend_config=backend_config
    ).run_gpu(warmup=warmup, repeat=repeat, timer=timer, **kwargs)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""RMSNorm for Hopper (sm_90a): the kernels found by the M3 evolution run on H200.

The two kernels are taken from the rmsnorm_h200 run's frontier (``persistent-tma-ring``, 1.053x
FlashInfer summed over the 15 task shapes, 1.041x at batch_size >= 4113). Both have the signature
``(inp, wgt, out, ctr, batch_size, grid)`` and are correct for any ``grid >= 1``. :func:`plan`
chooses the kernel and its configuration from (hidden_size, batch_size).

* **small** (batch_size <= 1024): rows are held entirely in registers, one row per CTA per
  grid-stride step. X is read once with ``ld.global.nc.v4``, the W loads overlap the reduction, and
  there is no shared-memory staging. This regime is latency bound.
* **ring** (larger batches): persistent CTAs, k per SM. Thread 0 keeps an S-stage shared-memory ring
  of whole rows filled by ``cp.async.bulk`` (mbarrier complete_tx); W stays in registers; Y is stored
  from registers. The first ``S * grid`` rows are assigned statically. After that, rows come from a
  global ticket counter fetched one row ahead, which keeps the row stream sequential in DRAM; static
  strided schedules drift and lose about 4% at 16384 rows.

Two findings from the run fixed the large shapes:
* plain ``ld.global.nc`` beats ``L1::no_allocate`` on the loads, by 2-3% of HBM throughput;
* the row order has to stay sequential (a scrambled order costs 7-8%).

**Ticket counter** (``ctr``: one 64-bit word, as two uint32 zeroed before the first launch).
* The low half counts row tickets: each fetch is ``atom.add(word, 1)``.
* The high half counts finished CTAs: each adds ``1 << 32`` as it exits.
* The CTA whose completion finds ``grid - 1`` others subtracts the whole word, so it is 0 again for
  the next launch.

Why this is guaranteed by the PTX memory model (no fences needed):
* All of it is relaxed atomic read-modify-writes on one location. Those are totally ordered in
  coherence order, and each reads the value just before it.
* One thread's RMWs to one location keep program order (base causality order, same address, same
  proxy).
* So every CTA's ticket fetches precede its completion. The last completion follows every other
  completion, and the reset follows the last completion.
* The reset therefore sees, and removes, every ticket and completion of the launch.

The run's version stored 0 instead, which is not ordered against another CTA's late ticket atomic.
An ``acq_rel`` completion atomic on a separate word would also be correct, at about the same cost.
Both cost about 0.5 us at 4113 rows (hs 4096/5120) against the run's unordered reset, and nothing
measurable elsewhere.

**One counter, one launch at a time.** Launches sharing a ``ctr`` must be stream ordered. Running
one argument tuple concurrently on two streams, or replaying a CUDA graph that captured it
concurrently with another launch of it, interleaves their tickets and corrupts rows: review saw
7 of 600 launches wrong. Give each concurrent launch its own argument tuple (``_kernel_args``
allocates a fresh counter on every call).

**64-bit element offsets.** Every row's offset is formed once per row in 64 bits, so any
``batch_size * hidden_size`` is addressed correctly; the 32-bit form wrapped beyond 2**31 elements.
Measured on H200 at every task shape, it costs nothing (within noise).

The small kernel launches ``batch_size`` CTAs itself and ignores ``grid``. The ring is correct for any
``grid``. So any pairing of a planned kernel with :func:`grid_for`'s grid computes the right result.
"""

import tirx_kernels.tirx_lite as txl

EPS = 1e-6
FULL_MASK = 0xFFFFFFFF
BULK_G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
LD_NC = "ld.global.nc.v4.b32"
LD_PLAIN = "ld.global.v4.b32"
SMALL_MAX_BATCH = 1024
# hidden_size -> warps per row for the small kernel; W uses plain loads at 5120 (1.47 vs 1.86 us).
SMALL_CFG = {4096: (8, LD_NC), 5120: (10, LD_PLAIN), 8192: (16, LD_NC)}
# hidden_size -> (warps, CTAs per SM, ring stages) for the ring kernel
RING_CFG = {4096: (4, 3, 6), 5120: (5, 2, 9)}
RING_CFG_8192 = {"le_8192_rows": (4, 2, 6), "gt_8192_rows": (8, 1, 12)}


def supported(hidden_size: int) -> bool:
    """Hidden sizes these kernels take: 8 halves per thread, at most 8 such chunks per thread."""
    return _generic_warps(hidden_size) is not None


def _generic_warps(hidden_size):
    for warps in (16, 8, 4, 2, 1, 20, 10, 5):
        if hidden_size % (256 * warps) == 0 and hidden_size // (256 * warps) <= 8:
            return warps
    return None


def plan(hidden_size: int, batch_size=None) -> dict:
    """Kernel and configuration for one shape. ``batch_size=None`` plans for a large batch."""
    if not supported(hidden_size):
        raise ValueError(f"hidden_size={hidden_size} is not supported by the sm_90 kernels")
    small_batch = batch_size is not None and batch_size <= SMALL_MAX_BATCH
    if small_batch or (hidden_size not in RING_CFG and hidden_size != 8192):
        warps, ld_w = SMALL_CFG.get(hidden_size, (_generic_warps(hidden_size), LD_NC))
        return {"kind": "small", "warps": warps, "ld_w": ld_w, "ctas_per_sm": None}
    if hidden_size == 8192:
        big = batch_size is None or batch_size > 8192
        warps, k, stages = RING_CFG_8192["gt_8192_rows" if big else "le_8192_rows"]
    else:
        warps, k, stages = RING_CFG[hidden_size]
    return {"kind": "ring", "warps": warps, "ctas_per_sm": k, "stages": stages}


def grid_for(p: dict, batch_size: int, num_sms: int) -> int:
    if p["kind"] == "small":
        return max(1, batch_size)
    return max(1, min(batch_size, num_sms * p["ctas_per_sm"]))


def make_kernel(hidden_size: int, p: dict):
    if p["kind"] == "small":
        return _make_small(hidden_size, p["warps"], p["ld_w"])
    return _make_ring(hidden_size, p["warps"], p["stages"])


def _warp_allreduce(v, peer):
    for delta in (16, 8, 4, 2, 1):
        txl.ptx.shfl_sync.bfly.b32(
            peer,
            txl.reinterpret(v[0], ty="uint32"),
            txl.uint32(delta),
            txl.uint32(31),
            txl.uint32(FULL_MASK),
        )
        txl.assign(v[0], v[0] + txl.reinterpret(peer, ty="float32"))


def _sumsq(xf, n, ss):
    parts = 4 if n >= 16 else 1
    acc = txl.alloc_local([parts], "float32")
    for j in range(parts):
        txl.assign(acc[j], txl.float32(0.0))
    for k in range(n):
        txl.assign(acc[k % parts], acc[k % parts] + xf[k] * xf[k])
    total = acc[0]
    for j in range(1, parts):
        total = total + acc[j]
    txl.assign(ss[0], total)


def _normalize_and_store(out, gptr, row, hidden_size, threads, chunks, xf, wf, ow, rms, tid):
    for c in range(chunks):
        for q in range(4):
            k = 8 * c + 2 * q
            txl.ptx["cvt.rn.f16x2.f32"](ow[q], (xf[k + 1] * rms) * wf[k + 1], (xf[k] * rms) * wf[k])
        col = (c * threads + tid) * 8
        txl.ptx["st.global.v4.b32"](gptr(out, col, row), ow[0], ow[1], ow[2], ow[3])


def _make_small(hidden_size, warps, ld_w, wide=True):
    threads = warps * 32
    chunks = hidden_size // (8 * threads)

    @txl.kernel()
    def rmsnorm_sm90_small(
        inp: txl.Tensor((txl.int64(),), "float16"),
        wgt: txl.Tensor((txl.int64(),), "float16"),
        out: txl.Tensor((txl.int64(),), "float16"),
        ctr: txl.Tensor((2,), "uint32"),
        batch_size: "int32",
        grid: "int32",
    ):
        # One row per CTA, batch_size CTAs whatever ``grid`` says (unused here), so the kernel is
        # correct with any launch arguments. A grid-stride loop costs a register spill and ~0.5 us.
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=batch_size, block=threads))
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        smem = txl.smem_pool()
        part = smem.alloc([warps], "float32") if warps > 1 else None
        xw = txl.alloc_local([4 * chunks], "uint32")
        ww = txl.alloc_local([4 * chunks], "uint32")
        ow = txl.alloc_local([4], "uint32")
        xf = txl.alloc_local([8 * chunks], "float32")
        wf = txl.alloc_local([8 * chunks], "float32")
        ss = txl.alloc_local([1], "float32")
        tmp = txl.alloc_local([1], "float32")
        rms = txl.local_scalar("float32")
        base = txl.local_scalar("int64" if wide else "int32")
        peer = txl.local_scalar("uint32")
        row = txl.cta_id()
        ROW = row
        rowbase = txl.local_scalar("int64")
        if wide:
            txl.assign(rowbase, txl.cast(row, "int64") * hidden_size)

        def gptr(tensor, off, row=None):
            # Element offset row * hidden_size + col. ``wide``: 64-bit, from a row base computed once
            # per row (any batch). Otherwise 32-bit, valid only while batch_size * hidden_size < 2**31.
            if row is None:
                txl.assign(base, off)
            elif wide:
                txl.assign(
                    base, (rowbase if row is ROW else txl.cast(row, "int64") * hidden_size) + off
                )
            else:
                txl.assign(base, row * hidden_size + off)
            return tensor.ptr_to([base if wide else txl.cast(base, "int64")])

        if True:  # one row per CTA (kept as a block: the same per-row step as the ring kernel)
            for c in range(chunks):
                col = (c * threads + tid) * 8
                txl.ptx[LD_NC](
                    xw[4 * c], xw[4 * c + 1], xw[4 * c + 2], xw[4 * c + 3],
                    gptr(inp, col, row),
                )  # fmt: skip
            for c in range(chunks):
                col = (c * threads + tid) * 8
                txl.ptx[ld_w](
                    ww[4 * c], ww[4 * c + 1], ww[4 * c + 2], ww[4 * c + 3], gptr(wgt, col)
                )
            for p in range(4 * chunks):
                txl.idioms.cast_f16x2_to_f32x2(xf, p, xw[p])
            _sumsq(xf, 8 * chunks, ss)
            _warp_allreduce(ss, peer)
            if warps > 1:
                with txl.if_(lane == 0), txl.then_():
                    txl.ptx["st.shared.f32"](part.ptr_to([warp]), ss[0])
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(threads))
                txl.assign(ss[0], txl.float32(0.0))
                for j in range(warps):
                    txl.ptx["ld.shared.f32"](tmp[0], part.ptr_to([j]))
                    txl.assign(ss[0], ss[0] + tmp[0])
            txl.assign(rms, txl.rsqrt(ss[0] / txl.float32(hidden_size) + txl.float32(EPS)))
            for p in range(4 * chunks):
                txl.idioms.cast_f16x2_to_f32x2(wf, p, ww[p])
            _normalize_and_store(out, gptr, row, hidden_size, threads, chunks, xf, wf, ow, rms, tid)

    return rmsnorm_sm90_small


def _make_ring(hidden_size, warps, stages, wide=True):
    threads = warps * 32
    chunks = hidden_size // (8 * threads)
    row_bytes = hidden_size * 2

    @txl.kernel()
    def rmsnorm_sm90_ring(
        inp: txl.Tensor((txl.int64(),), "float16"),
        wgt: txl.Tensor((txl.int64(),), "float16"),
        out: txl.Tensor((txl.int64(),), "float16"),
        ctr: txl.Tensor((2,), "uint32"),
        batch_size: "int32",
        grid: "int32",
    ):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=grid, block=threads))
        tid = txl.thread_id()
        lane = tid & 31
        warp = tid >> 5
        cta = txl.cta_id()

        smem = txl.smem_pool()
        buf = smem.alloc([stages * hidden_size], "float16", align=128)
        part = smem.alloc([2 * warps], "float32", align=16)
        rowidx = smem.alloc([stages], "int32", align=16)
        full = txl.MBarrier(smem, stages)
        full.init(1)
        txl.ptx.fence.proxy.async_.shared__cta()

        xw = txl.alloc_local([4 * chunks], "uint32")
        ww = txl.alloc_local([4 * chunks], "uint32")
        ow = txl.alloc_local([4], "uint32")
        xf = txl.alloc_local([8 * chunks], "float32")
        wf = txl.alloc_local([8 * chunks], "float32")
        ss = txl.alloc_local([1], "float32")
        tmp = txl.alloc_local([1], "float32")
        rms = txl.local_scalar("float32")
        base = txl.local_scalar("int64" if wide else "int32")
        peer = txl.local_scalar("uint32")
        it = txl.local_scalar("int32")
        row = txl.local_scalar("int32")
        ROW = row
        nrow = txl.local_scalar("int32")
        stage = txl.local_scalar("int32")
        phase = txl.local_scalar("int32")
        tk = txl.local_scalar("uint32")
        old = txl.local_scalar("uint64")
        total = txl.local_scalar("uint64")
        rowbase = txl.local_scalar("int64")
        done = txl.local_scalar("int32")
        old_tok = txl.local_scalar("uint64")

        def gptr(tensor, off, row=None):
            # Element offset row * hidden_size + col. ``wide``: 64-bit, from a row base computed once
            # per row (any batch). Otherwise 32-bit, valid only while batch_size * hidden_size < 2**31.
            if row is None:
                txl.assign(base, off)
            elif wide:
                txl.assign(
                    base, (rowbase if row is ROW else txl.cast(row, "int64") * hidden_size) + off
                )
            else:
                txl.assign(base, row * hidden_size + off)
            return tensor.ptr_to([base if wide else txl.cast(base, "int64")])

        def issue(st, r):
            txl.ptx["st.shared.s32"](rowidx.ptr_to([st]), r)
            txl.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                full.ptr_to([st]), txl.uint32(row_bytes)
            )
            txl.ptx[BULK_G2S](
                buf.ptr_to([st * hidden_size]),
                gptr(inp, 0, r),
                txl.uint32(row_bytes),
                full.ptr_to([st]),
            )

        def sentinel(st):
            txl.ptx["st.shared.s32"](rowidx.ptr_to([st]), txl.int32(-1))
            txl.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
                old_tok, full.ptr_to([st]), txl.uint32(1)
            )

        def take_ticket():
            # low 32 bits of the counter word: tickets handed out (one row each)
            txl.ptx["atom.global.add.u64"](old, ctr.ptr_to([0]), txl.uint64(1))
            txl.assign(tk, txl.cast(old & txl.uint64(0xFFFFFFFF), "uint32"))

        with txl.if_(tid == 0), txl.then_():
            txl.assign(done, 0)
            take_ticket()  # prefetched one row ahead
            for s in range(stages):
                with txl.if_(cta + s * grid < batch_size):
                    with txl.then_():
                        issue(s, cta + s * grid)
                    with txl.else_():
                        sentinel(s)
        # barrier init and thread 0's rowidx writes are visible to every waiter after this
        txl.cuda.cta_sync()

        for c in range(chunks):
            col = (c * threads + tid) * 8
            txl.ptx[LD_NC](ww[4 * c], ww[4 * c + 1], ww[4 * c + 2], ww[4 * c + 3], gptr(wgt, col))
        for p in range(4 * chunks):
            txl.idioms.cast_f16x2_to_f32x2(wf, p, ww[p])

        txl.assign(it, 0)
        txl.assign(stage, 0)
        txl.assign(phase, 0)
        with txl.while_(True):
            full.wait(stage, phase)
            txl.ptx["ld.shared.s32"](row, rowidx.ptr_to([stage]))
            with txl.if_(row < 0), txl.then_():
                txl.break_()
            if wide:
                txl.assign(rowbase, txl.cast(row, "int64") * hidden_size)
            for c in range(chunks):
                col = (c * threads + tid) * 8
                txl.ptx["ld.shared.v4.b32"](
                    xw[4 * c], xw[4 * c + 1], xw[4 * c + 2], xw[4 * c + 3],
                    buf.ptr_to([stage * hidden_size + col]),
                )  # fmt: skip
            for p in range(4 * chunks):
                txl.idioms.cast_f16x2_to_f32x2(xf, p, xw[p])
            _sumsq(xf, 8 * chunks, ss)
            _warp_allreduce(ss, peer)
            pbase = (it & 1) * warps
            with txl.if_(lane == 0), txl.then_():
                txl.ptx["st.shared.f32"](part.ptr_to([pbase + warp]), ss[0])
            # Every thread has read buf[stage] and rowidx[stage] (generic proxy) before this barrier;
            # thread 0's proxy fence then orders those reads before the refill's async-proxy write.
            txl.ptx.bar.sync(txl.uint32(1), txl.uint32(threads))
            with txl.if_(tid == 0), txl.then_():
                with txl.if_(done == 0), txl.then_():
                    # tk < batch_size first: a 64-bit ticket must not wrap into a negative int32 row.
                    txl.assign(
                        nrow,
                        txl.Select(
                            tk < txl.cast(batch_size, "uint32"),
                            stages * grid + txl.cast(tk, "int32"),
                            batch_size,
                        ),
                    )
                    with txl.if_(nrow < batch_size):
                        with txl.then_():
                            txl.ptx.fence.proxy.async_.shared__cta()
                            issue(stage, nrow)
                            take_ticket()
                        with txl.else_():
                            sentinel(stage)
                            txl.assign(done, 1)
            txl.assign(ss[0], txl.float32(0.0))
            for j in range(warps):
                txl.ptx["ld.shared.f32"](tmp[0], part.ptr_to([pbase + j]))
                txl.assign(ss[0], ss[0] + tmp[0])
            txl.assign(rms, txl.rsqrt(ss[0] / txl.float32(hidden_size) + txl.float32(EPS)))
            _normalize_and_store(out, gptr, row, hidden_size, threads, chunks, xf, wf, ow, rms, tid)
            txl.assign(it, it + 1)
            with txl.if_(stage == stages - 1):
                with txl.then_():
                    txl.assign(stage, 0)
                    txl.assign(phase, phase ^ 1)
                with txl.else_():
                    txl.assign(stage, stage + 1)

        # Counter reset (module docstring): one word, completions in the high half. The CTA whose
        # completion finds the other grid - 1 subtracts the whole word, so it ends at exactly 0.
        with txl.if_(tid == 0), txl.then_():
            txl.ptx["atom.global.add.u64"](old, ctr.ptr_to([0]), txl.uint64(1) << txl.uint64(32))
            with txl.if_((old >> txl.uint64(32)) == txl.cast(grid - 1, "uint64")), txl.then_():
                # into a register: the atomic below overwrites `old`
                txl.assign(total, old + (txl.uint64(1) << txl.uint64(32)))
                txl.ptx["atom.global.add.u64"](old, ctr.ptr_to([0]), txl.uint64(0) - total)

    return rmsnorm_sm90_ring

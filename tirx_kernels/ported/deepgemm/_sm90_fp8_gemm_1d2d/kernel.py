# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90_fp8_gemm_1d2d_impl` (`deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`).

One persistent kernel instantiation per `KernelSpec`: a TMA producer warpgroup and one or two
WGMMA math warpgroups over a `kNumStages`-deep shared-memory ring, FP32 promotion of each
128-wide K block with per-token A scales and per-block B scales, and an STSM + TMA-store
epilogue.  The block scheduler is `deep_gemm/scheduler/gemm.cuh::Scheduler`.

Every shared-memory buffer is a one-dimensional allocation at the byte offset the source
computes (`.cuh:110-126`); the 128B swizzles of A, B and D live in the TensorMaps, the WGMMA
descriptors and the explicit epilogue bank-group arithmetic.
"""

import math

import tirx_kernels.tirx_lite as txl

from .spec import GemmType, KernelSpec

#: `cute::TMA::CacheHintSm90::EVICT_NORMAL`.
EVICT_NORMAL = 0x1000000000000000
#: `ClusterBarrier::wait`'s suspend-time hint (`cutlass/arch/barrier.h:412`).
TRY_WAIT_TICKS = 0x989680
#: `cutlass::arch::NamedBarrier` reserves ids 0..7; the source's ids 0 and 1 become 8 and 9.
NAMED_BARRIER_SFB = 8
NAMED_BARRIER_STORE = 9

_TMA_LOAD = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
_TMA_LOAD_HINT = _TMA_LOAD + ".L2::cache_hint"
_TMA_LOAD_MULTICAST = _TMA_LOAD + ".multicast::cluster.L2::cache_hint"
_TMA_STORE = "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"


def _ceil_div(a, b):
    return (a + b - 1) // b


def _num_1d_blocks_per_group(spec: KernelSpec) -> int:
    """`sched::get_num_1d_blocks_per_group`: the first candidate wins ties (strict `<`)."""
    best, best_usage = 0, None
    for candidate in (8, 16):
        if spec.is_tma_multicast_on_a:
            usage = candidate * spec.block_n + _ceil_div(spec.num_sms, candidate) * spec.block_m
        else:
            usage = candidate * spec.block_m + _ceil_div(spec.num_sms, candidate) * spec.block_n
        if best_usage is None or usage < best_usage:
            best, best_usage = candidate, usage
    return best


def _u32(x):
    return txl.cast(x, "uint32")


def _wait(barrier_ptr, phase):
    """`ClusterTransactionBarrier::wait`: try_wait.parity with the suspend hint, retried."""
    done = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.while_(done == txl.uint32(0)):
        txl.ptx["mbarrier.try_wait.parity.shared::cta.b64"](
            done, barrier_ptr, _u32(phase), txl.uint32(TRY_WAIT_TICKS)
        )


class _Scheduler:
    """`sched::Scheduler` for the m-grouped types: one independent walk per role.

    The producer lane and every math thread construct their own and walk the identical block
    sequence, as in the source; no state is shared between them.
    """

    def __init__(self, spec: KernelSpec, grouped_layout, shape_m):
        self.spec = spec
        self.gl = grouped_layout
        bm, bn = spec.block_m, spec.block_n
        self.num_n_blocks = _ceil_div(spec.shape_n, bn)
        self.kind = spec.gemm_type
        # Member initializers (`gemm.cuh:40-63`).
        self.current_iter = txl.local_scalar("int32", init=-1)
        self.current_group_idx = txl.local_scalar("uint32", init=txl.uint32(0))
        self.current_m_cumsum = txl.local_scalar("uint32", init=txl.uint32(0))
        self.last_psum_m = txl.local_scalar("uint32", init=txl.uint32(0))
        self.current_psum_m = txl.local_scalar("uint32", init=txl.uint32(0))
        self.current_m_block_cumsum = txl.local_scalar("uint32", init=txl.uint32(0))
        self.num_m_blocks = txl.local_scalar("uint32", init=txl.uint32(0))
        self.num_blocks = txl.local_scalar("uint32", init=txl.uint32(0))
        self.num_blocks_in_group = txl.local_scalar("uint32", init=txl.uint32(0))
        self.is_peer_cta_alive = txl.local_scalar("uint32", init=txl.uint32(1))
        self.m_block_idx = txl.local_scalar("uint32", init=txl.uint32(0))
        self.n_block_idx = txl.local_scalar("uint32", init=txl.uint32(0))
        self.valid = txl.local_scalar("uint32", init=txl.uint32(0))
        # Constructor (`gemm.cuh:88-115`).
        txl.assign(self.num_m_blocks, (_u32(shape_m) + txl.uint32(bm - 1)) // txl.uint32(bm))
        if self.kind == GemmType.M_GROUPED_CONTIGUOUS:
            txl.assign(self.num_blocks, self.num_m_blocks * txl.uint32(self.num_n_blocks))
        elif self.kind == GemmType.M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT:
            txl.assign(self.current_psum_m, _u32(self._layout_at(0)))
            txl.assign(
                self.num_m_blocks, (self.current_psum_m + txl.uint32(bm - 1)) // txl.uint32(bm)
            )

    def _layout_at(self, index):
        value = txl.local_scalar("int32")
        txl.ptx["ld.global.s32"](value, self.gl.ptr_to([index]))
        return value

    def _swizzle(self, block_idx):
        """`get_swizzled_block_idx`, SM90 branch (with the unaligned-multicast fix)."""
        spec = self.spec
        per = _num_1d_blocks_per_group(spec)
        if per % spec.num_tma_multicast != 0:
            raise AssertionError("Invalid group size")
        on_a = spec.is_tma_multicast_on_a
        nnb = txl.uint32(self.num_n_blocks)
        primary = nnb if on_a else self.num_m_blocks
        secondary = self.num_m_blocks if on_a else nnb
        blocks_per_group = txl.local_scalar("uint32", init=secondary * txl.uint32(per))
        first = txl.local_scalar("uint32", init=block_idx // blocks_per_group * txl.uint32(per))
        in_group = txl.local_scalar("uint32", init=block_idx % blocks_per_group)
        txl.assign(self.num_blocks_in_group, txl.min(txl.uint32(per), primary - first))
        if spec.num_tma_multicast > 1:
            with txl.if_(self.num_blocks_in_group % txl.uint32(2) != txl.uint32(0)):
                with txl.then_():
                    even = txl.local_scalar("uint32", init=self.num_blocks_in_group ^ txl.uint32(1))
                    with txl.if_(in_group < even * secondary):
                        with txl.then_():
                            txl.assign(self.num_blocks_in_group, even)
                        with txl.else_():
                            txl.assign(in_group, in_group - even * secondary)
                            txl.assign(first, first + even)
                            txl.assign(self.num_blocks_in_group, txl.uint32(1))
        if on_a:
            txl.assign(self.m_block_idx, in_group // self.num_blocks_in_group)
            txl.assign(self.n_block_idx, first + in_group % self.num_blocks_in_group)
        else:
            txl.assign(self.m_block_idx, first + in_group % self.num_blocks_in_group)
            txl.assign(self.n_block_idx, in_group // self.num_blocks_in_group)

    def next(self):
        """`get_next_block`; sets `valid` and, if valid, `m_block_idx` / `n_block_idx`."""
        spec = self.spec
        bm, nnb = spec.block_m, txl.uint32(self.num_n_blocks)
        txl.assign(self.current_iter, self.current_iter + 1)
        next_block_idx = txl.local_scalar(
            "uint32", init=_u32(self.current_iter) * txl.uint32(spec.num_sms) + _u32(txl.cta_id())
        )
        txl.assign(self.valid, txl.uint32(1))
        if self.kind == GemmType.M_GROUPED_MASKED:
            searching = txl.local_scalar("uint32", init=txl.uint32(1))
            with txl.while_(searching == txl.uint32(1)):
                with txl.if_(self.current_group_idx == txl.uint32(spec.num_groups)):
                    with txl.then_():
                        txl.assign(self.valid, txl.uint32(0))
                        txl.assign(searching, txl.uint32(0))
                    with txl.else_():
                        rows = self._layout_at(self.current_group_idx)
                        txl.assign(
                            self.num_m_blocks, (_u32(rows) + txl.uint32(bm - 1)) // txl.uint32(bm)
                        )
                        cumsum = txl.local_scalar(
                            "uint32", init=self.current_m_cumsum + self.num_m_blocks
                        )
                        with txl.if_(next_block_idx < cumsum * nnb):
                            with txl.then_():
                                txl.assign(searching, txl.uint32(0))
                            with txl.else_():
                                txl.assign(self.current_group_idx, self.current_group_idx + 1)
                                txl.assign(self.current_m_cumsum, cumsum)
            with txl.if_(self.valid == txl.uint32(1)):
                with txl.then_():
                    self._swizzle(next_block_idx - self.current_m_cumsum * nnb)
        elif self.kind == GemmType.M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT:
            searching = txl.local_scalar("uint32", init=txl.uint32(1))
            with txl.while_(searching == txl.uint32(1)):
                with txl.if_(
                    next_block_idx < (self.current_m_block_cumsum + self.num_m_blocks) * nnb
                ):
                    with txl.then_():
                        txl.assign(searching, txl.uint32(0))
                    with txl.else_():
                        txl.assign(self.current_group_idx, self.current_group_idx + 1)
                        with txl.if_(self.current_group_idx == txl.uint32(spec.num_groups)):
                            with txl.then_():
                                txl.assign(self.valid, txl.uint32(0))
                                txl.assign(searching, txl.uint32(0))
                            with txl.else_():
                                txl.assign(
                                    self.last_psum_m,
                                    (self.current_psum_m + txl.uint32(bm - 1))
                                    // txl.uint32(bm)
                                    * txl.uint32(bm),
                                )
                                txl.assign(
                                    self.current_psum_m,
                                    _u32(self._layout_at(self.current_group_idx)),
                                )
                                txl.assign(
                                    self.current_m_block_cumsum,
                                    self.current_m_block_cumsum + self.num_m_blocks,
                                )
                                txl.assign(
                                    self.num_m_blocks,
                                    (self.current_psum_m - self.last_psum_m + txl.uint32(bm - 1))
                                    // txl.uint32(bm),
                                )
            with txl.if_(self.valid == txl.uint32(1)):
                with txl.then_():
                    self._swizzle(next_block_idx - self.current_m_block_cumsum * nnb)
                    txl.assign(
                        self.m_block_idx, self.m_block_idx + self.last_psum_m // txl.uint32(bm)
                    )
        else:  # MGroupedContiguous
            with txl.if_(next_block_idx >= self.num_blocks):
                with txl.then_():
                    txl.assign(self.valid, txl.uint32(0))
                with txl.else_():
                    mc = txl.uint32(spec.num_tma_multicast)
                    alive = txl.local_scalar("uint32", init=txl.uint32(0))
                    with txl.if_(nnb % mc == txl.uint32(0)):
                        with txl.then_():
                            txl.assign(alive, txl.uint32(1))
                    with txl.if_(self.num_m_blocks % mc == txl.uint32(0)):
                        with txl.then_():
                            txl.assign(alive, txl.uint32(1))
                    with txl.if_((next_block_idx ^ txl.uint32(1)) < self.num_blocks):
                        with txl.then_():
                            txl.assign(alive, txl.uint32(1))
                    txl.assign(self.is_peer_cta_alive, alive)
                    self._swizzle(next_block_idx)

    def group_offset(self, m_block_idx):
        """The group index `get_global_idx<kWithGroupOffset=true>` multiplies `shape_dim` by."""
        if self.kind == GemmType.M_GROUPED_CONTIGUOUS:
            row = self._layout_at(m_block_idx * txl.uint32(self.spec.block_m))
            return _u32(txl.max(row, 0))
        return self.current_group_idx

    def is_tma_multicast_valid(self, m_block_idx):
        valid = txl.local_scalar("uint32", init=txl.uint32(1))
        with txl.if_(self.num_blocks_in_group == txl.uint32(1)):
            with txl.then_():
                txl.assign(valid, txl.uint32(0))
            with txl.else_():
                if (
                    self.kind == GemmType.M_GROUPED_CONTIGUOUS
                    and not self.spec.is_tma_multicast_on_a
                ):
                    bm = txl.uint32(self.spec.block_m)
                    group = self._layout_at(m_block_idx * bm)
                    peer = self._layout_at((m_block_idx ^ txl.uint32(1)) * bm)
                    with txl.if_(group != peer):
                        with txl.then_():
                            txl.assign(valid, txl.uint32(0))
        return valid

    def is_computation_valid(self, m_block_idx, m_offset):
        bm = txl.uint32(self.spec.block_m)
        if self.kind == GemmType.M_GROUPED_CONTIGUOUS:
            return self._layout_at(m_offset + m_block_idx * bm) >= 0
        if self.kind == GemmType.M_GROUPED_MASKED:
            return m_offset + m_block_idx * bm < _u32(self._layout_at(self.current_group_idx))
        return m_offset + m_block_idx * bm < self.current_psum_m


def build_kernel(spec: KernelSpec, *, backend_config=None):
    """The TIRx `PrimFunc` for one `sm90_fp8_gemm_1d2d_impl` template instantiation."""
    BM, BN, BK = spec.block_m, spec.block_n, spec.block_k
    stages, math_threads, mc = spec.num_stages, spec.num_math_threads, spec.num_tma_multicast
    num_threads = spec.num_tma_threads + math_threads
    if BK != 128 or BM not in (64, 128) or spec.is_tma_multicast_on_a:
        raise NotImplementedError(f"unported specialization {spec}")
    is_masked = spec.gemm_type == GemmType.M_GROUPED_MASKED

    # ---- template constants (`.cuh:57-92`) ----
    uniform_sfb = BK % BN == 0  # kMustUseUniformedScaleB
    num_accum = BN // 2  # WGMMA m64nBNk32 f32 accumulators per thread
    wave_block_m = BM if BM <= 64 else 128
    num_waves = BM // wave_block_m
    store_threads = wave_block_m * (128 // 64)  # kNumWGMMAStoreThreads
    swizzle_d = spec.swizzle_d_mode
    if swizzle_d == 0:
        raise NotImplementedError("the padded (non-swizzled) D store is not ported")
    tma_d_block_n = swizzle_d // 2
    shape_n, shape_k = spec.shape_n, spec.shape_k  # compiled dims ("nk")
    shape_k_scales = _ceil_div(shape_k, BK)
    shape_n_sfb = _ceil_div(shape_n, BK)
    gcd = math.gcd(BK, BN)
    should_optimize = BK // gcd <= 4 and not uniform_sfb
    k_gap, k_end = gcd // 8, (BK // 8 if should_optimize else 0)
    if not uniform_sfb and not should_optimize:
        raise NotImplementedError("non-uniform B scales without the dispatched predicate")
    wgmma = f"wgmma.mma_async.sync.aligned.m64n{BN}k32.f32.e4m3.e4m3"

    # ---- shared memory, byte offsets of `.cuh:76-126` ----
    smem_d_size = _ceil_div(BM * BN * 2, 1024) * 1024
    a_stage, b_stage = BM * BK, BN * BK
    sfa_stage = _ceil_div(BM * 4, 128) * 128
    sfb_bytes = _ceil_div(shape_k_scales * (1 if uniform_sfb else 2) * 4, 8) * 8
    off_sf = smem_d_size + stages * (a_stage + b_stage)
    off_bar = off_sf + stages * sfa_stage + sfb_bytes
    if off_bar + 16 * stages > spec.smem_size:
        raise AssertionError("shared memory carve-out exceeds the heuristic's smem_size")
    # GMMA descriptor high word: SBO = 1024 >> 4 in bits [32,46), layout code 1 in [62,64).
    desc_hi = (1024 >> 4) | (1 << 30)

    @txl.kernel()
    def sm90_fp8_gemm_1d2d(
        sfb: txl.Tensor((txl.int64(),), "float32"),
        grouped_layout: txl.Tensor((txl.int64(),), "int32"),
        shape_m: "int32",
        shape_n_arg: "int32",
        shape_k_arg: "int32",
        tensor_map_a: txl.Ptr(txl.TensorMap),
        tensor_map_b: txl.Ptr(txl.TensorMap),
        tensor_map_d: txl.Ptr(txl.TensorMap),
        tensor_map_sfa: txl.Ptr(txl.TensorMap),
    ):
        # `LaunchArgs(num_sms, num_threads, smem_size, cluster)`, `__launch_bounds__(N, 1)`.
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(
                grid=spec.num_sms, block=num_threads, cluster=(mc,) if mc > 1 else None
            ),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=backend_config,
        )
        del shape_n_arg, shape_k_arg  # SHAPE_N / SHAPE_K are compiled in (`.cuh:72-74`)

        # ---- one linear, 1024-aligned dynamic pool at the source offsets ----
        smem = txl.smem_pool()
        smem_d = smem.alloc((smem_d_size // 2,), "bfloat16", align=1024)
        smem_a = smem.alloc((stages * a_stage,), "float8_e4m3fn", align=1024)
        smem_b = smem.alloc((stages * b_stage,), "float8_e4m3fn", align=1024)
        smem_sfa = smem.alloc((stages * sfa_stage // 4,), "float32", align=128)
        smem_sfb = smem.alloc((sfb_bytes // 4,), "float32", align=8)
        bars = smem.alloc((2 * stages,), "uint64", align=8)  # full[0..S), empty[S..2S)
        if smem.bytes != off_bar + 16 * stages:
            raise AssertionError(
                f"SMEM layout ends at {smem.bytes}, expected {off_bar + 16 * stages}"
            )
        smem.commit(spec.smem_size)

        def full(s):
            return bars.ptr_to([s])

        def empty(s):
            return bars.ptr_to([s + stages])

        thread_idx = _u32(txl.thread_id())
        warp_idx = txl.local_scalar("uint32")
        txl.ptx.shfl_sync.idx.b32(
            warp_idx,
            thread_idx // txl.uint32(32),
            txl.uint32(0),
            txl.uint32(31),
            txl.uint32(0xFFFFFFFF),
        )
        lane_idx = _u32(txl.lane_id())

        # ---- prologue (`.cuh:96-151`) ----
        with txl.if_(warp_idx == txl.uint32(math_threads // 32)):
            with txl.then_():
                elected = txl.local_scalar("uint32")
                elected_lane = txl.local_scalar("uint32")
                txl.ptx.elect_sync(elected_lane, elected, txl.uint32(0xFFFFFFFF))
                with txl.if_(elected == txl.uint32(1)):
                    with txl.then_():
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_a))
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_b))
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_sfa))
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_d))
        txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))

        with txl.if_(warp_idx == txl.uint32(math_threads // 32 + 1)):
            with txl.then_():
                init_elected = txl.local_scalar("uint32")
                init_lane = txl.local_scalar("uint32")
                txl.ptx.elect_sync(init_lane, init_elected, txl.uint32(0xFFFFFFFF))
                with txl.if_(init_elected == txl.uint32(1)):
                    with txl.then_():
                        for s in range(stages):
                            txl.ptx["mbarrier.init.shared::cta.b64"](full(s), txl.uint32(1))
                            txl.ptx["mbarrier.init.shared::cta.b64"](
                                empty(s), txl.uint32(mc * math_threads // 32)
                            )
                        txl.ptx.fence.mbarrier_init.release.cluster()
        if mc > 1:
            txl.ptx.barrier.cluster.arrive.relaxed.aligned()
            txl.ptx.barrier.cluster.wait.aligned()
        else:
            txl.ptx.bar.sync(txl.uint32(0))
        txl.ptx.griddepcontrol.wait()

        stage_idx = txl.local_scalar("uint32", init=txl.uint32(0))
        phase = txl.local_scalar("uint32", init=txl.uint32(0))

        def advance_pipeline():
            txl.assign(
                stage_idx,
                txl.select(stage_idx == txl.uint32(stages - 1), txl.uint32(0), stage_idx + 1),
            )
            txl.assign(phase, phase ^ txl.cast(stage_idx == txl.uint32(0), "uint32"))

        with txl.if_(warp_idx >= txl.uint32(math_threads // 32)):
            # ================= TMA producer warpgroup (`.cuh:167-214`) =================
            with txl.then_():
                txl.ptx["setmaxnreg.dec.sync.aligned.u32"](txl.uint32(40))
                with txl.if_(warp_idx == txl.uint32(math_threads // 32 + 2)):
                    with txl.then_():
                        ld_elected = txl.local_scalar("uint32")
                        ld_lane = txl.local_scalar("uint32")
                        txl.ptx.elect_sync(ld_lane, ld_elected, txl.uint32(0xFFFFFFFF))
                        with txl.if_(ld_elected == txl.uint32(1)):
                            with txl.then_():
                                _producer(
                                    spec,
                                    sched_args=(grouped_layout, shape_m),
                                    smem=(smem_a, smem_b, smem_sfa),
                                    maps=(tensor_map_a, tensor_map_b, tensor_map_sfa),
                                    full=full,
                                    empty=empty,
                                    stage_idx=stage_idx,
                                    phase=phase,
                                    advance=advance_pipeline,
                                )

            # ================= math warpgroups (`.cuh:215-445`) =================
            with txl.else_():
                _math(
                    spec,
                    sched_args=(grouped_layout, shape_m),
                    sfb=sfb,
                    smem=(smem_d, smem_a, smem_b, smem_sfa, smem_sfb),
                    tensor_map_d=tensor_map_d,
                    full=full,
                    empty=empty,
                    stage_idx=stage_idx,
                    phase=phase,
                    advance=advance_pipeline,
                    thread_idx=thread_idx,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                    consts=dict(
                        uniform_sfb=uniform_sfb,
                        num_accum=num_accum,
                        wave_block_m=wave_block_m,
                        num_waves=num_waves,
                        store_threads=store_threads,
                        swizzle_d=swizzle_d,
                        tma_d_block_n=tma_d_block_n,
                        shape_k_scales=shape_k_scales,
                        shape_n_sfb=shape_n_sfb,
                        should_optimize=should_optimize,
                        k_gap=k_gap,
                        k_end=k_end,
                        wgmma=wgmma,
                        a_stage=a_stage,
                        b_stage=b_stage,
                        sfa_stage=sfa_stage,
                        desc_hi=desc_hi,
                        is_masked=is_masked,
                    ),
                )

    return sm90_fp8_gemm_1d2d.func


def _producer(spec, *, sched_args, smem, maps, full, empty, stage_idx, phase, advance):
    """The elected lane of TMA warp `kNumMathThreads/32 + 2` (`.cuh:173-213`)."""
    smem_a, smem_b, smem_sfa = smem
    tensor_map_a, tensor_map_b, tensor_map_sfa = maps
    grouped_layout, shape_m = sched_args
    BM, BN, BK, mc = spec.block_m, spec.block_n, spec.block_k, spec.num_tma_multicast
    shape_k_scales = _ceil_div(spec.shape_k, BK)
    a_stage, b_stage = BM * BK, BN * BK
    sfa_words = _ceil_div(BM * 4, 128) * 128 // 4
    is_masked = spec.gemm_type == GemmType.M_GROUPED_MASKED

    sched = _Scheduler(spec, grouped_layout, shape_m)
    sched.next()
    with txl.while_(sched.valid == txl.uint32(1)):
        m_block = sched.m_block_idx
        # Multicast on B only (A and SFA are loaded per CTA in every ported specialization).
        mc_b = txl.local_scalar("uint32", init=txl.uint32(1))
        if mc > 1:
            mc_valid = sched.is_tma_multicast_valid(m_block)
            with txl.if_(mc_valid == txl.uint32(1)):
                with txl.then_():
                    txl.assign(mc_b, txl.uint32(mc))
        b_group = txl.local_scalar("uint32", init=sched.group_offset(m_block))
        k_block_idx = txl.local_scalar("uint32", init=txl.uint32(0))
        with txl.while_(k_block_idx < txl.uint32(shape_k_scales)):
            _wait(empty(stage_idx), phase ^ txl.uint32(1))
            k_idx = k_block_idx * txl.uint32(BK)
            group_a = sched.current_group_idx if is_masked else txl.uint32(0)
            m_idx = group_a * _u32(shape_m) + m_block * txl.uint32(BM)
            sfa_k = group_a * txl.uint32(shape_k_scales) + k_block_idx
            n_idx = b_group * txl.uint32(spec.shape_n) + sched.n_block_idx * txl.uint32(BN)
            txl.ptx[_TMA_LOAD_HINT](
                smem_a.ptr_to([stage_idx * txl.uint32(a_stage)]),
                txl.address_of(tensor_map_a),
                txl.cast(k_idx, "int32"),
                txl.cast(m_idx, "int32"),
                full(stage_idx),
                txl.uint64(EVICT_NORMAL),
            )
            txl.ptx[_TMA_LOAD_HINT](
                smem_sfa.ptr_to([stage_idx * txl.uint32(sfa_words)]),
                txl.address_of(tensor_map_sfa),
                txl.cast(m_block * txl.uint32(BM), "int32"),
                txl.cast(sfa_k, "int32"),
                full(stage_idx),
                txl.uint64(EVICT_NORMAL),
            )
            b_dst = smem_b.ptr_to([stage_idx * txl.uint32(b_stage)])

            def load_b():
                txl.ptx[_TMA_LOAD_HINT](
                    b_dst,
                    txl.address_of(tensor_map_b),
                    txl.cast(k_idx, "int32"),
                    txl.cast(n_idx, "int32"),
                    full(stage_idx),
                    txl.uint64(EVICT_NORMAL),
                )

            if mc == 1:  # `num_tma_multicast == 1` is a compile-time fact without a cluster
                load_b()
            else:
                with txl.if_(mc_b == txl.uint32(1)):
                    with txl.then_():
                        load_b()
                    with txl.else_():
                        with txl.if_(_u32(txl.cuda.cluster_cta_id("x")) == txl.uint32(0)):
                            with txl.then_():
                                txl.ptx[_TMA_LOAD_MULTICAST](
                                    b_dst,
                                    txl.address_of(tensor_map_b),
                                    txl.cast(k_idx, "int32"),
                                    txl.cast(n_idx, "int32"),
                                    full(stage_idx),
                                    txl.uint16((1 << mc) - 1),
                                    txl.uint64(EVICT_NORMAL),
                                )
            txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](
                full(stage_idx), txl.uint32(a_stage + b_stage + BM * 4)
            )
            txl.assign(k_block_idx, k_block_idx + 1)
            advance()
        sched.next()
    if mc > 1:
        # Drain, so no peer still arrives into these barriers (`.cuh:210-213`).
        drained = txl.local_scalar("uint32", init=txl.uint32(0))
        with txl.while_(drained < txl.uint32(spec.num_stages)):
            _wait(empty(stage_idx), phase ^ txl.uint32(1))
            txl.assign(drained, drained + 1)
            advance()


def _math(
    spec,
    *,
    sched_args,
    sfb,
    smem,
    tensor_map_d,
    full,
    empty,
    stage_idx,
    phase,
    advance,
    thread_idx,
    warp_idx,
    lane_idx,
    consts,
):
    """Math warpgroups: B-scale staging, WGMMA + promotion, STSM/TMA-store epilogue."""
    smem_d, smem_a, smem_b, smem_sfa, smem_sfb = smem
    grouped_layout, shape_m = sched_args
    c = consts
    BM, BN, BK, mc = spec.block_m, spec.block_n, spec.block_k, spec.num_tma_multicast
    math_threads = spec.num_math_threads
    num_accum, num_waves, wave_block_m = c["num_accum"], c["num_waves"], c["wave_block_m"]
    shape_k_scales, uniform_sfb = c["shape_k_scales"], c["uniform_sfb"]
    swizzle_d, tma_d_block_n = c["swizzle_d"], c["tma_d_block_n"]

    txl.ptx["setmaxnreg.inc.sync.aligned.u32"](txl.uint32(248 if math_threads == 128 else 232))
    math_wg_idx = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(
        math_wg_idx,
        thread_idx // txl.uint32(128),
        txl.uint32(0),
        txl.uint32(31),
        txl.uint32(0xFFFFFFFF),
    )
    r_0 = txl.local_scalar("uint32", init=warp_idx * txl.uint32(16) + lane_idx // txl.uint32(4))
    r_1 = txl.local_scalar("uint32", init=r_0 + txl.uint32(8))
    # Low descriptor words (start address >> 4, LBO 0), broadcast from lane 0 (`.cuh:223-226`).
    a_lo = _u32(
        txl.cuda.cvta_generic_to_shared(smem_a.ptr_to([math_wg_idx * txl.uint32(64 * BK)]))
    ) >> txl.uint32(4)
    b_lo = _u32(txl.cuda.cvta_generic_to_shared(smem_b.ptr_to([0]))) >> txl.uint32(4)
    a_desc_lo = txl.local_scalar("uint32")
    b_desc_lo = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(
        a_desc_lo, a_lo, txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF)
    )
    txl.ptx.shfl_sync.idx.b32(
        b_desc_lo, b_lo, txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF)
    )
    accum = txl.alloc_local((num_accum,), "float32")
    final_accum = txl.alloc_local((num_accum * num_waves,), "float32")
    scale = txl.alloc_local((8,), "float32")  # sa0, sa1, sb0, sb1, s00, s10, s01, s11
    desc_hi64 = txl.uint64(c["desc_hi"] << 32)
    sched = _Scheduler(spec, grouped_layout, shape_m)

    def empty_barrier_arrive():
        if mc == 1:
            with txl.if_(lane_idx == txl.uint32(0)):
                with txl.then_():
                    txl.ptx["mbarrier.arrive.shared::cta.b64"](empty(stage_idx))
        else:
            target = txl.select(
                sched.is_peer_cta_alive == txl.uint32(1),
                lane_idx,
                _u32(txl.cuda.cluster_cta_id("x")),
            )
            with txl.if_(lane_idx < txl.uint32(mc)):
                with txl.then_():
                    remote = txl.local_scalar("uint32")
                    txl.ptx["mapa.shared::cluster.u32"](
                        remote, _u32(txl.cuda.cvta_generic_to_shared(empty(stage_idx))), target
                    )
                    txl.ptx["mbarrier.arrive.shared::cluster.b64"](remote)

    def k_loop(former_iters):
        """One copy of the `#pragma unroll 8` K loop (`.cuh:283-349`)."""
        with txl.serial(0, shape_k_scales, unroll=8) as k_block_idx:
            a_base = a_desc_lo + stage_idx * txl.uint32(c["a_stage"] // 16)
            b_base = b_desc_lo + stage_idx * txl.uint32(c["b_stage"] // 16)
            txl.ptx["ld.shared.f32"](scale[2], smem_sfb.ptr_to([k_block_idx]))
            if not uniform_sfb:
                txl.ptx["ld.shared.f32"](scale[3], smem_sfb.ptr_to([k_block_idx + shape_k_scales]))
            _wait(full(stage_idx), phase)
            for local_idx in range(num_waves):
                m_offset = local_idx * wave_block_m
                sfa_base = stage_idx * txl.uint32(c["sfa_stage"] // 4)
                txl.ptx["ld.shared.f32"](scale[0], smem_sfa.ptr_to([sfa_base + r_0 + m_offset]))
                txl.ptx["ld.shared.f32"](scale[1], smem_sfa.ptr_to([sfa_base + r_1 + m_offset]))
                for i in range(num_accum):
                    txl.cuda.wgmma.noop_barrier(accum[i])
                txl.ptx.wgmma.fence.sync.aligned()
                for k in range(BK // 32):
                    a_desc = desc_hi64 | txl.cast(
                        a_base + txl.uint32((m_offset * BK + k * 32) // 16), "uint64"
                    )
                    b_desc = desc_hi64 | txl.cast(b_base + txl.uint32(k * 32 // 16), "uint64")
                    txl.ptx[c["wgmma"]](
                        *[accum[j] for j in range(num_accum)], a_desc, b_desc, k != 0, 1, 1
                    )
                txl.ptx.wgmma.commit_group.sync.aligned()
                for i in range(num_accum):
                    txl.cuda.wgmma.noop_barrier(accum[i])
                txl.ptx.wgmma.wait_group.sync.aligned(0)
                if local_idx == num_waves - 1:
                    empty_barrier_arrive()
                txl.assign(scale[4], scale[0] * scale[2])
                txl.assign(scale[5], scale[1] * scale[2])
                if not uniform_sfb:
                    txl.assign(scale[6], scale[0] * scale[3])
                    txl.assign(scale[7], scale[1] * scale[3])
                base = num_accum * local_idx
                for i in range(num_accum // 4):
                    # `predicate ? scale_x_0 : scale_x_1` with `i < num_former_iters` pinned to a
                    # constant by the dispatch branch this copy sits in (`.cuh:342`).
                    former = uniform_sfb or i < former_iters
                    lo = scale[4] if former else scale[6]
                    hi = scale[5] if former else scale[7]
                    for j, s in ((0, lo), (1, lo), (2, hi), (3, hi)):
                        txl.assign(
                            final_accum[base + 4 * i + j],
                            final_accum[base + 4 * i + j] + s * accum[4 * i + j],
                        )
            advance()

    sched.next()
    with txl.while_(sched.valid == txl.uint32(1)):
        m_block, n_block = sched.m_block_idx, sched.n_block_idx
        # ---- B scales for this block (`.cuh:231-251`) ----
        num_former_iters = txl.local_scalar("uint32", init=txl.uint32(BN // 8))
        num_full_iters = txl.local_scalar("uint32", init=txl.uint32(BN // 8))
        if not uniform_sfb:
            txl.assign(
                num_former_iters,
                txl.min(txl.uint32(BN), txl.uint32(BK) - n_block * txl.uint32(BN) % txl.uint32(BK))
                // txl.uint32(8),
            )
            txl.assign(
                num_full_iters,
                txl.min(txl.uint32(spec.shape_n) - n_block * txl.uint32(BN), txl.uint32(BN))
                // txl.uint32(8),
            )
        num_sfb = txl.local_scalar("uint32", init=txl.uint32(shape_k_scales))
        with txl.if_(num_former_iters < num_full_iters):
            with txl.then_():
                txl.assign(num_sfb, txl.uint32(2 * shape_k_scales))
        with txl.if_(thread_idx >= txl.uint32(32)):
            with txl.then_():
                sfb_base = txl.local_scalar(
                    "uint32",
                    init=sched.group_offset(m_block) * txl.uint32(c["shape_n_sfb"] * shape_k_scales)
                    + n_block * txl.uint32(BN) // txl.uint32(BK) * txl.uint32(shape_k_scales),
                )
                i = txl.local_scalar("uint32", init=thread_idx - txl.uint32(32))
                with txl.while_(i < num_sfb):
                    value = txl.local_scalar("float32")
                    with txl.if_(i < txl.uint32(shape_k_scales)):
                        with txl.then_():
                            txl.ptx["ld.global.f32"](value, sfb.ptr_to([sfb_base + i]))
                        with txl.else_():
                            # K-major SFB: row 1 is `(i - shape_k_scales) * 1 + shape_k_scales`.
                            txl.ptx["ld.global.f32"](
                                value,
                                sfb.ptr_to(
                                    [
                                        sfb_base
                                        + (i - txl.uint32(shape_k_scales))
                                        + txl.uint32(shape_k_scales)
                                    ]
                                ),
                            )
                    txl.ptx["st.shared.f32"](smem_sfb.ptr_to([i]), value)
                    txl.assign(i, i + txl.uint32(math_threads - 32))
        txl.ptx["bar.sync"](txl.uint32(NAMED_BARRIER_SFB), txl.uint32(math_threads))

        for i in range(num_accum * num_waves):
            txl.assign(final_accum[i], txl.float32(0))

        with txl.if_(sched.is_computation_valid(m_block, math_wg_idx * txl.uint32(64))):
            with txl.then_():
                if c["should_optimize"]:
                    # `dispatch_num_former_iters<0, kGap, kEnd>`: exactly one copy runs.
                    for former in range(0, c["k_end"] + 1, c["k_gap"]):
                        with txl.if_(num_former_iters == txl.uint32(former)):
                            with txl.then_():
                                k_loop(former)
                else:
                    k_loop(0)
            with txl.else_():
                k_skip = txl.local_scalar("uint32", init=txl.uint32(0))
                with txl.while_(k_skip < txl.uint32(shape_k_scales)):
                    _wait(full(stage_idx), phase)
                    empty_barrier_arrive()
                    txl.assign(k_skip, k_skip + 1)
                    advance()

        # ---- epilogue: STSM into swizzled D, TMA store (`.cuh:359-443`) ----
        with txl.if_(thread_idx < txl.uint32(BN // tma_d_block_n)):
            with txl.then_():
                txl.ptx.cp.async_.bulk.wait_group.read(0)
        txl.ptx["bar.sync"](txl.uint32(NAMED_BARRIER_STORE), txl.uint32(c["store_threads"]))
        for local_idx in range(num_waves):
            m_offset = local_idx * wave_block_m
            base = num_accum * local_idx
            for i in range(num_accum // 4):
                atom, in_atom = i // (tma_d_block_n // 8), i % (tma_d_block_n // 8)
                if swizzle_d // 16 == 8:
                    row = txl.uint32(in_atom // 8) + lane_idx
                    col = txl.uint32(in_atom)
                else:
                    bank_group = txl.uint32(in_atom) + lane_idx * txl.uint32(swizzle_d // 16)
                    row = bank_group // txl.uint32(8)
                    col = bank_group % txl.uint32(8)
                col = col ^ (row % txl.uint32(swizzle_d // 16))
                byte = (
                    warp_idx * txl.uint32(16 * swizzle_d)
                    + txl.uint32(m_offset * swizzle_d + atom * BM * swizzle_d)
                    + row * txl.uint32(128)
                    + col * txl.uint32(16)
                )
                lo = txl.local_scalar("uint32")
                hi = txl.local_scalar("uint32")
                txl.ptx["cvt.rn.bf16x2.f32"](
                    lo, final_accum[base + 4 * i + 1], final_accum[base + 4 * i]
                )
                txl.ptx["cvt.rn.bf16x2.f32"](
                    hi, final_accum[base + 4 * i + 3], final_accum[base + 4 * i + 2]
                )
                txl.ptx["stmatrix.sync.aligned.x2.m8n8.shared.b16"](
                    smem_d.ptr_to([byte // txl.uint32(2)]), lo, hi
                )
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.ptx["bar.sync"](txl.uint32(NAMED_BARRIER_STORE), txl.uint32(c["store_threads"]))
        with txl.if_(thread_idx < txl.uint32(BN // tma_d_block_n)):
            with txl.then_():
                in_block_n = thread_idx * txl.uint32(tma_d_block_n)
                group_d = sched.current_group_idx if c["is_masked"] else txl.uint32(0)
                txl.ptx[_TMA_STORE](
                    txl.address_of(tensor_map_d),
                    txl.cast(n_block * txl.uint32(BN) + in_block_n, "int32"),
                    txl.cast(group_d * _u32(shape_m) + m_block * txl.uint32(BM), "int32"),
                    smem_d.ptr_to([in_block_n * txl.uint32(BM)]),
                )
                txl.ptx.cp.async_.bulk.commit_group()
        txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))
        sched.next()

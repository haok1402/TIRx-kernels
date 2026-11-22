# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90_fp8_gemm_1d1d_impl` (`deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`).

One persistent kernel instantiation per `KernelSpec`: a TMA producer warpgroup and one or two
WGMMA math warpgroups over a `kNumStages`-deep shared-memory ring.  Each 128-wide K block is
promoted into FP32 with a per-row A scale and a per-column B scale (both TMA-loaded per stage),
and every block is written back with one TMA reduce-add per math warpgroup, accumulating into
the FP32 output.  For `KGroupedContiguous` the producer rewrites the A/B TensorMaps in place
(address, inner dim and stride) whenever the scheduler enters a new group and publishes them
through a per-CTA GMEM slot.  The block scheduler is `deep_gemm/scheduler/gemm.cuh::Scheduler`.

Every shared-memory buffer is a one-dimensional allocation at the byte offset the source
computes (`.cuh:68-125`); the 128B swizzle of A and B lives in the TensorMaps and the WGMMA
descriptors.
"""

import tirx_kernels.tirx_lite as txl

from .spec import GemmType, KernelSpec

#: `cute::TMA::CacheHintSm90::EVICT_NORMAL`.
EVICT_NORMAL = 0x1000000000000000
#: `ClusterBarrier::wait`'s suspend-time hint (`cutlass/arch/barrier.h:412`).
TRY_WAIT_TICKS = 0x989680
#: `cutlass::arch::NamedBarrier` reserves ids 0..7; the source's id `math_wg_idx` becomes 8 + it.
NAMED_BARRIER_BASE = 8
#: (BLOCK_M, BLOCK_N) of the multicast-on-A classes with a reviewed export (k-grouped only).
REVIEWED_MULTICAST_ON_A = frozenset({(128, 104), (128, 120)})
#: `CUtensorMap` bytes.
TENSOR_MAP_BYTES = 128

_TMA_LOAD = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
_TMA_LOAD_HINT = _TMA_LOAD + ".L2::cache_hint"
_TMA_LOAD_MULTICAST = _TMA_LOAD + ".multicast::cluster.L2::cache_hint"
_TMA_REDUCE_ADD = "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group"


def _ceil_div(a, b):
    return (a + b - 1) // b


def _align(a, b):
    return _ceil_div(a, b) * b


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
    """`ClusterTransactionBarrier::wait` (`cutlass/arch/barrier.h:408-421`): one asm retry loop around
    `mbarrier.try_wait.parity.shared::cta.b64` with the 0x989680 suspend hint."""
    txl.cuda.mbarrier_wait_parity(barrier_ptr, _u32(phase), TRY_WAIT_TICKS)


def _elect():
    """`cute::elect_one_sync()`: `elect.sync` over the full warp; true on the elected lane."""
    lane = txl.local_scalar("uint32")
    pred = txl.local_scalar("uint32")
    txl.ptx.elect_sync(lane, pred, txl.uint32(0xFFFFFFFF))
    return pred


def _with_source_fp_mode(backend_config):
    """DeepGEMM's JIT builds without fast math: `fp_mode="precise"` (no FTZ, IEEE div/sqrt, FMA
    contraction on), on the kernel so it composes with a caller's other CUDA settings."""
    config = {key: dict(value) for key, value in (backend_config or {}).items()}
    config.setdefault("cuda", {})["fp_mode"] = "precise"
    return config


class _Scheduler:
    """`sched::Scheduler` for `Normal` and `KGroupedContiguous` (`kKAlignment = kSFKSpan = 128`).

    The producer lane and every math thread construct their own and walk the identical block
    sequence (including the k-group walk), as in the source; no state is shared between them.
    """

    def __init__(self, spec: KernelSpec, grouped_layout, shape_m, shape_n, shape_k):
        self.spec = spec
        self.gl = grouped_layout
        self.is_k_grouped = spec.gemm_type == GemmType.K_GROUPED_CONTIGUOUS
        bm, bn = spec.block_m, spec.block_n
        # Member initializers (`gemm.cuh:40-63`).
        self.current_iter = txl.local_scalar("int32", init=-1)
        self.num_blocks_in_group = txl.local_scalar("uint32", init=txl.uint32(0))
        self.is_peer_cta_alive = txl.local_scalar("uint32", init=txl.uint32(1))
        if self.is_k_grouped:
            self.current_group_idx = txl.local_scalar("uint32", init=txl.uint32(0))
            self.current_shape_k = txl.local_scalar("uint32", init=_u32(shape_k))
            self.current_num_valid_groups = txl.local_scalar("uint32", init=txl.uint32(0))
            self.current_k_cumsum = txl.local_scalar("uint32", init=txl.uint32(0))
            self.current_sf_k_cumsum = txl.local_scalar("uint32", init=txl.uint32(0))
            self.next_group_idx = txl.local_scalar("uint32", init=txl.uint32(0))
            self.next_shape_k = txl.local_scalar("uint32", init=txl.uint32(0))
        else:
            # The k-group members are never written for `Normal`: their initializers are constants
            # (`current_sf_k_cumsum = 0`, `current_shape_k = shape_k`), which nvcc folds into every use.
            self.current_group_idx = txl.uint32(0)
            self.current_shape_k = _u32(shape_k)
            self.current_sf_k_cumsum = txl.uint32(0)
        self.m_block_idx = txl.local_scalar("uint32", init=txl.uint32(0))
        self.n_block_idx = txl.local_scalar("uint32", init=txl.uint32(0))
        self.valid = txl.local_scalar("uint32", init=txl.uint32(0))
        # Constructor (`gemm.cuh:88-115`).
        self.num_m_blocks = (_u32(shape_m) + txl.uint32(bm - 1)) // txl.uint32(bm)
        self.num_n_blocks = (_u32(shape_n) + txl.uint32(bn - 1)) // txl.uint32(bn)
        self.num_blocks = txl.local_scalar("uint32", init=self.num_m_blocks * self.num_n_blocks)
        # The block-group size when it is a compile-time constant (see `_swizzle`), else None.
        per = _num_1d_blocks_per_group(spec)
        on_a = spec.is_tma_multicast_on_a
        primary_const = spec.shape_n if on_a else spec.shape_m
        primary_blocks = _ceil_div(primary_const, bn if on_a else bm)
        # Compile-time facts about the block groups (see `_swizzle`), with the primary extent compiled in:
        # - `primary % per == 0`: every group has exactly `per` blocks;
        # - `primary <= per`: one group (`first` is always 0) of `primary` blocks, split by the
        #   unaligned-multicast fix-up when `primary` is odd under multicast.
        self.primary_blocks = primary_blocks if primary_const else None
        self.single_group = bool(primary_const) and primary_blocks <= per
        self.odd_single_group = (
            self.single_group and spec.num_tma_multicast > 1 and primary_blocks % 2 == 1
        )
        if primary_const and primary_blocks % per == 0:
            self.const_group_size = per
        elif self.single_group and not self.odd_single_group:
            self.const_group_size = primary_blocks
        else:
            self.const_group_size = None
        if self.is_k_grouped:
            self._get_next_k_group(self.current_group_idx, self.current_shape_k)
            txl.assign(self.next_group_idx, self.current_group_idx + txl.uint32(1))
            self._get_next_k_group(self.next_group_idx, self.next_shape_k)

    def _get_next_k_group(self, group_idx, shape_k):
        """`get_next_k_group` (`gemm.cuh:65-72`): the next group with `K > 0`, or `kNumGroups`."""
        searching = txl.local_scalar("uint32", init=txl.uint32(1))
        with txl.while_(searching == txl.uint32(1)):
            with txl.if_(group_idx < txl.uint32(self.spec.num_groups)):
                with txl.then_():
                    # The source's `int` load emits `ld.global.u32`, compared unsigned (`> 0`).
                    txl.ptx["ld.global.u32"](shape_k, self.gl.ptr_to([group_idx]))
                    with txl.if_(shape_k > txl.uint32(0)):
                        with txl.then_():
                            txl.assign(searching, txl.uint32(0))
                        with txl.else_():
                            txl.assign(group_idx, group_idx + txl.uint32(1))
                with txl.else_():
                    txl.assign(searching, txl.uint32(0))

    def _swizzle(self, block_idx):
        """`get_swizzled_block_idx`, SM90 branch (with the unaligned-multicast fix)."""
        spec = self.spec
        per = _num_1d_blocks_per_group(spec)
        if per % spec.num_tma_multicast != 0:
            raise AssertionError("Invalid group size")
        on_a = spec.is_tma_multicast_on_a
        primary = self.num_n_blocks if on_a else self.num_m_blocks
        secondary = self.num_m_blocks if on_a else self.num_n_blocks
        # The compiled primary extent lets nvcc fold the group arithmetic below (constant group size,
        # `first == 0` when there is only one group); NVRTC keeps it, and with it
        # `is_tma_multicast_valid` and the plain multicast-side load path, so the folds are written out.
        if self.single_group:
            # `block_idx < num_blocks = primary * secondary <= per * secondary`: group 0.
            first = txl.local_scalar("uint32", init=txl.uint32(0))
            in_group = txl.local_scalar("uint32", init=block_idx)
        else:
            secondary_const = spec.shape_m if on_a else spec.shape_n
            if secondary_const:
                # `secondary * per` is a compile-time constant: divide by the literal (nvcc's form).
                blocks_per_group = txl.uint32(
                    _ceil_div(secondary_const, spec.block_m if on_a else spec.block_n) * per
                )
            else:
                blocks_per_group = txl.local_scalar("uint32", init=secondary * txl.uint32(per))
            first = txl.local_scalar("uint32", init=block_idx // blocks_per_group * txl.uint32(per))
            in_group = txl.local_scalar("uint32", init=block_idx % blocks_per_group)
        if self.const_group_size is not None:
            txl.assign(self.num_blocks_in_group, txl.uint32(self.const_group_size))
        elif self.odd_single_group:
            # "Fix unaligned TMA multicast" with the (odd) group size a constant.
            even = self.primary_blocks ^ 1
            with txl.if_(in_group < txl.uint32(even) * secondary):
                with txl.then_():
                    txl.assign(self.num_blocks_in_group, txl.uint32(even))
                with txl.else_():
                    txl.assign(in_group, in_group - txl.uint32(even) * secondary)
                    txl.assign(first, txl.uint32(even))
                    txl.assign(self.num_blocks_in_group, txl.uint32(1))
        else:
            txl.assign(self.num_blocks_in_group, txl.min(txl.uint32(per), primary - first))
            if spec.num_tma_multicast > 1:
                with txl.if_(self.num_blocks_in_group % txl.uint32(2) != txl.uint32(0)):
                    with txl.then_():
                        even = txl.local_scalar(
                            "uint32", init=self.num_blocks_in_group ^ txl.uint32(1)
                        )
                        with txl.if_(in_group < even * secondary):
                            with txl.then_():
                                txl.assign(self.num_blocks_in_group, even)
                            with txl.else_():
                                txl.assign(in_group, in_group - even * secondary)
                                txl.assign(first, first + even)
                                txl.assign(self.num_blocks_in_group, txl.uint32(1))
        if self.const_group_size is not None:
            group_size = txl.uint32(self.const_group_size)
        else:
            group_size = self.num_blocks_in_group
        if on_a:
            txl.assign(self.m_block_idx, in_group // group_size)
            txl.assign(self.n_block_idx, first + in_group % group_size)
        else:
            txl.assign(self.m_block_idx, first + in_group % group_size)
            txl.assign(self.n_block_idx, in_group // group_size)

    def next(self):
        """`get_next_block`; sets `valid` and, if valid, `m_block_idx` / `n_block_idx`."""
        spec = self.spec
        txl.assign(self.current_iter, self.current_iter + 1)
        next_block_idx = txl.local_scalar(
            "uint32", init=_u32(self.current_iter) * txl.uint32(spec.num_sms) + _u32(txl.cta_id())
        )
        txl.assign(self.valid, txl.uint32(1))
        if self.is_k_grouped:
            searching = txl.local_scalar("uint32", init=txl.uint32(1))
            with txl.while_(searching == txl.uint32(1)):
                with txl.if_(self.current_group_idx == txl.uint32(spec.num_groups)):
                    with txl.then_():
                        # End of the task
                        txl.assign(self.valid, txl.uint32(0))
                        txl.assign(searching, txl.uint32(0))
                    with txl.else_():
                        with txl.if_(
                            next_block_idx
                            < (self.current_num_valid_groups + txl.uint32(1)) * self.num_blocks
                        ):
                            with txl.then_():
                                txl.assign(searching, txl.uint32(0))
                            with txl.else_():
                                # Move to the next group
                                txl.assign(
                                    self.current_sf_k_cumsum,
                                    self.current_sf_k_cumsum
                                    + (self.current_shape_k + txl.uint32(127)) // txl.uint32(128),
                                )
                                txl.assign(
                                    self.current_num_valid_groups,
                                    self.current_num_valid_groups + txl.uint32(1),
                                )
                                txl.assign(
                                    self.current_k_cumsum,
                                    self.current_k_cumsum + self.current_shape_k,
                                )
                                txl.assign(self.current_group_idx, self.next_group_idx)
                                txl.assign(self.next_group_idx, self.next_group_idx + txl.uint32(1))
                                txl.assign(self.current_shape_k, self.next_shape_k)
                                self._get_next_k_group(self.next_group_idx, self.next_shape_k)
            with txl.if_(self.valid == txl.uint32(1)):
                with txl.then_():
                    self._swizzle(next_block_idx - self.current_num_valid_groups * self.num_blocks)
        else:  # Normal
            with txl.if_(next_block_idx >= self.num_blocks):
                with txl.then_():
                    txl.assign(self.valid, txl.uint32(0))
                with txl.else_():
                    mc = txl.uint32(spec.num_tma_multicast)
                    alive = txl.local_scalar("uint32", init=txl.uint32(0))
                    with txl.if_(self.num_n_blocks % mc == txl.uint32(0)):
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

    def is_tma_multicast_valid(self):
        """`is_tma_multicast_valid` for `Normal` / k-grouped: only the group-size test.  A Python bool
        when the group size is a compile-time constant."""
        if self.const_group_size is not None:
            return self.const_group_size != 1
        return self.num_blocks_in_group != txl.uint32(1)


def build_kernel(spec: KernelSpec, *, backend_config=None):
    """The TIRx `PrimFunc` for one `sm90_fp8_gemm_1d1d_impl` template instantiation."""
    BM, BN, BK = spec.block_m, spec.block_n, spec.block_k
    stages, math_threads, mc = spec.num_stages, spec.num_math_threads, spec.num_tma_multicast
    num_threads = spec.num_tma_threads + math_threads
    is_k_grouped = spec.gemm_type == GemmType.K_GROUPED_CONTIGUOUS
    if spec.gemm_type not in (GemmType.NORMAL, GemmType.K_GROUPED_CONTIGUOUS):
        raise NotImplementedError(f"gemm type {spec.gemm_type!r} is not a 1D1D type")
    if BK != 128 or BM not in (64, 128) or spec.num_tma_threads != 128:
        raise NotImplementedError(f"unported specialization {spec}")
    if mc > 1 and spec.is_tma_multicast_on_a and (
        not is_k_grouped or (BM, BN) not in REVIEWED_MULTICAST_ON_A
    ):
        # Multicast on A is exported and reviewed for the KGroupedContiguous classes the realistic
        # MoE shapes select (`sm90_fp8_gemm_1d1d_multicast_on_a.md`); refuse the rest (other BLOCK_N,
        # BLOCK_M 64, Normal) rather than build an unverified variant.
        raise NotImplementedError(
            f"TMA multicast on A is ported for KGroupedContiguous {sorted(REVIEWED_MULTICAST_ON_A)} only"
        )

    # ---- template constants (`.cuh:56-75, 150-155`) ----
    num_accum = BN // 2  # WGMMA m64nBNk32 f32 accumulators per thread
    pipeline_unrolls = 0 if is_k_grouped else stages
    tma_registers, math_registers = (40, 232) if pipeline_unrolls == 0 else (24, 240)
    wgmma = f"wgmma.mma_async.sync.aligned.m64n{BN}k32.f32.e4m3.e4m3"

    # ---- shared memory, byte offsets of `.cuh:68-125` ----
    tm_size = 2 * TENSOR_MAP_BYTES if is_k_grouped else 0
    d_size = BM * BN * 4
    a_stage, b_stage = BM * BK, BN * BK
    sfa_stage = BM * 4
    sfb_bytes_per_stage = BN * 4
    sfb_stage = _align(sfb_bytes_per_stage, 128)
    if d_size % 1024 != 0 or sfa_stage % 128 != 0:
        raise AssertionError("source static asserts: D and SFA stage alignment")
    off_d = tm_size
    off_a = off_d + d_size
    off_b = off_a + stages * a_stage
    off_sfa = off_b + stages * b_stage
    off_sfb = off_sfa + stages * sfa_stage
    off_bar = off_sfb + stages * sfb_stage
    if off_bar + 16 * stages > spec.smem_size:
        raise AssertionError("shared memory carve-out exceeds the heuristic's smem_size")
    expect_tx = a_stage + b_stage + sfa_stage + sfb_bytes_per_stage
    # GMMA descriptor high word: SBO = 1024 >> 4 in bits [32,46), layout code 1 in [62,64).
    desc_hi64 = ((1024 >> 4) | (1 << 30)) << 32

    @txl.kernel()
    def sm90_fp8_gemm_1d1d(
        gmem_a: txl.Tensor((txl.int64(),), "float8_e4m3fn"),
        gmem_b: txl.Tensor((txl.int64(),), "float8_e4m3fn"),
        grouped_layout: txl.Tensor((txl.int64(),), "int32"),
        tensor_map_buffer: txl.Tensor((txl.int64(),), "uint8"),
        shape_m_arg: "int32",
        shape_n_arg: "int32",
        shape_k_arg: "int32",
        tensor_map_a_base: txl.Ptr(txl.TensorMap),
        tensor_map_b_base: txl.Ptr(txl.TensorMap),
        tensor_map_sfa: txl.Ptr(txl.TensorMap),
        tensor_map_sfb: txl.Ptr(txl.TensorMap),
        tensor_map_cd: txl.Ptr(txl.TensorMap),
    ):
        # `LaunchArgs(num_sms, num_threads, smem_size, cluster)`, `__launch_bounds__(N, 1)`.
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(
                grid=spec.num_sms, block=num_threads, cluster=(mc,) if mc > 1 else None
            ),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=_with_source_fp_mode(backend_config),
        )
        # "Overwrite shape constants if the compiler gives" (`.cuh:63-66`).
        shape_m = txl.uint32(spec.shape_m) if spec.shape_m else _u32(shape_m_arg)
        shape_n = txl.uint32(spec.shape_n) if spec.shape_n else _u32(shape_n_arg)
        shape_k = txl.uint32(spec.shape_k) if spec.shape_k else _u32(shape_k_arg)

        # ---- one linear, 1024-aligned dynamic pool at the source offsets ----
        smem = txl.smem_pool()
        smem_tmap = (
            smem.alloc((tm_size // 4,), "uint32", align=1024) if is_k_grouped else None
        )
        smem_d = smem.alloc((d_size // 4,), "float32", align=256 if is_k_grouped else 1024)
        smem_a = smem.alloc((stages * a_stage,), "float8_e4m3fn", align=256)
        smem_b = smem.alloc((stages * b_stage,), "float8_e4m3fn", align=256)
        smem_sfa = smem.alloc((stages * sfa_stage // 4,), "float32", align=128)
        smem_sfb = smem.alloc((stages * sfb_stage // 4,), "float32", align=128)
        bars = smem.alloc((2 * stages,), "uint64", align=8)  # full[0..S), empty[S..2S)
        for buf, want in (
            (smem_d, off_d),
            (smem_a, off_a),
            (smem_b, off_b),
            (smem_sfa, off_sfa),
            (smem_sfb, off_sfb),
            (bars, off_bar),
        ):
            if smem.offset_of(buf) != want:
                raise AssertionError(f"SMEM buffer at {smem.offset_of(buf)}, expected {want}")
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
        lane_idx = txl.local_scalar("uint32", init=thread_idx % txl.uint32(32))

        # ---- prologue (`.cuh:81-148`) ----
        with txl.if_(warp_idx == txl.uint32(math_threads // 32)):
            with txl.then_():
                with txl.if_(_elect() == txl.uint32(1)):
                    with txl.then_():
                        for tmap in (
                            tensor_map_a_base,
                            tensor_map_b_base,
                            tensor_map_sfa,
                            tensor_map_sfb,
                            tensor_map_cd,
                        ):
                            txl.ptx.prefetch.tensormap(txl.address_of(tmap))
        txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))

        with txl.if_(warp_idx == txl.uint32(math_threads // 32 + 1)):
            with txl.then_():
                with txl.if_(_elect() == txl.uint32(1)):
                    with txl.then_():
                        if is_k_grouped:
                            # `*smem_tensor_map_a = tensor_map_a_base` (`.cuh:130-131`): 16-byte
                            # vector traffic, parameter -> registers -> shared.
                            for which, tmap in enumerate((tensor_map_a_base, tensor_map_b_base)):
                                _copy_param_tensor_map_to_smem(tmap, smem_tmap, which * 32)
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

        # `get_pipeline(iter_idx)`: (iter % kNumStages, (iter / kNumStages) & 1), one state per role.
        iter_idx = (
            txl.local_scalar("uint32", init=txl.uint32(0)),  # stage = iter_idx % kNumStages
            txl.local_scalar("uint32", init=txl.uint32(0)),  # phase = (iter_idx / kNumStages) & 1
        )
        roles = dict(
            spec=spec,
            grouped_layout=grouped_layout,
            shapes=(shape_m, shape_n, shape_k),
            full=full,
            empty=empty,
            iter_idx=iter_idx,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
        )

        with txl.if_(warp_idx >= txl.uint32(math_threads // 32)):
            # ================= TMA producer warpgroup (`.cuh:170-245`) =================
            with txl.then_():
                txl.ptx["setmaxnreg.dec.sync.aligned.u32"](txl.uint32(tma_registers))
                with txl.if_(warp_idx == txl.uint32(math_threads // 32)):
                    with txl.then_():
                        with txl.if_(_elect() == txl.uint32(1)):
                            with txl.then_():
                                _producer(
                                    **roles,
                                    gmem=(gmem_a, gmem_b, tensor_map_buffer),
                                    maps=(
                                        tensor_map_a_base,
                                        tensor_map_b_base,
                                        tensor_map_sfa,
                                        tensor_map_sfb,
                                    ),
                                    smem=(smem_tmap, smem_a, smem_b, smem_sfa, smem_sfb),
                                    consts=dict(
                                        a_stage=a_stage,
                                        b_stage=b_stage,
                                        sfa_stage=sfa_stage,
                                        sfb_stage=sfb_stage,
                                        expect_tx=expect_tx,
                                        unrolls=pipeline_unrolls,
                                    ),
                                )
            # ================= math warpgroups (`.cuh:246-348`) =================
            with txl.else_():
                txl.ptx["setmaxnreg.inc.sync.aligned.u32"](txl.uint32(math_registers))
                _math(
                    **roles,
                    thread_idx=thread_idx,
                    tensor_map_cd=tensor_map_cd,
                    smem=(smem_d, smem_a, smem_b, smem_sfa, smem_sfb),
                    consts=dict(
                        num_accum=num_accum,
                        wgmma=wgmma,
                        a_stage=a_stage,
                        b_stage=b_stage,
                        sfa_stage=sfa_stage,
                        sfb_stage=sfb_stage,
                        desc_hi64=desc_hi64,
                        unrolls=pipeline_unrolls,
                    ),
                )

    return sm90_fp8_gemm_1d1d.func


def _byte_address(ptr, byte_offset):
    """A generic pointer `byte_offset` bytes past `ptr`."""
    base = txl.reinterpret(ptr, ty="uint64")
    return txl.reinterpret(base + txl.uint64(byte_offset), ty="handle")


def _copy_param_tensor_map_to_smem(tmap, smem_tmap, word_offset):
    """Copy one grid-constant 128-byte TensorMap into SMEM: 8 x 16-byte loads into registers,
    then 8 x 16-byte shared stores (the source's struct copy, `.cuh:130-131`).

    The loads are the source's `ld.param.v4.u32` (constant-bank `LDC` in SASS), through the parameter
    address `cvta.to.param` gives for the grid-constant TensorMap."""
    words = txl.alloc_local((32,), "uint32")
    base = txl.local_scalar("uint64")
    txl.ptx["cvta.to.param.u64"](base, txl.reinterpret(txl.address_of(tmap), ty="handle"))
    for j in range(8):
        txl.ptx["ld.param.v4.u32"](
            words[4 * j], words[4 * j + 1], words[4 * j + 2], words[4 * j + 3],
            base + txl.uint64(16 * j),
        )  # fmt: skip
    for j in reversed(range(8)):
        txl.ptx["st.shared.v4.u32"](
            smem_tmap.ptr_to([word_offset + 4 * j]),
            words[4 * j], words[4 * j + 1], words[4 * j + 2], words[4 * j + 3],
        )  # fmt: skip


def _copy_smem_tensor_map_to_gmem(smem_tmap, word_offset, dst):
    """`*(gmem_tensor_map_a) = *(smem_tensor_map_a)` (`.cuh:208-209`): 8 x 16-byte shared loads and
    8 x 16-byte global stores.

    Each 16-byte chunk is stored right after it is loaded, the order the source's SASS ends up in
    (ptxas interleaves nvcc's load-all-then-store-all PTX): this copy runs in the 40-register producer,
    and holding all 32 words live (the PTX order, which NVRTC's `asm volatile` copies keep) spills."""
    words = txl.alloc_local((4,), "uint32")
    for j in range(8):
        txl.ptx["ld.shared.v4.u32"](
            words[0], words[1], words[2], words[3], smem_tmap.ptr_to([word_offset + 4 * j])
        )
        txl.ptx["st.global.v4.u32"](_byte_address(dst, 16 * j), words[0], words[1], words[2], words[3])


def _pipeline(pipe, stages):
    """`get_pipeline(iter_idx ++)` (`.cuh:165-167`): this ring slot's stage and phase, then advance.

    `pipe` is the role's `(stage, phase)` pair, i.e. `(iter_idx % kNumStages, (iter_idx / kNumStages) & 1)`
    kept incrementally (stage wraps, phase flips on the wrap) instead of re-derived from the counter.
    The two are equal for every iteration; the counter form's per-copy divisions, once a K loop is
    unrolled, are all hoisted to the loop top and spill the 24/40-register producer (nvcc on the same
    generated source spills identically, so this is the expression shape, not the compiler)."""
    stage, phase_bit = pipe
    stage_idx = txl.local_scalar("uint32", init=stage)
    phase = txl.local_scalar("uint32", init=phase_bit)
    # Bound before `stage` is rewritten: an unbound expression would re-read the updated stage.
    wrap = txl.local_scalar("uint32", init=txl.cast(stage == txl.uint32(stages - 1), "uint32"))
    txl.assign(stage, txl.select(wrap == txl.uint32(1), txl.uint32(0), stage + txl.uint32(1)))
    txl.assign(phase_bit, phase_bit ^ wrap)
    return stage_idx, phase


def _producer(
    *, spec, grouped_layout, shapes, full, empty, iter_idx, warp_idx, lane_idx, gmem, maps, smem,
    consts,
):  # fmt: skip
    """The elected lane of TMA warp `kNumMathThreads/32` (`.cuh:175-244`)."""
    del warp_idx
    gmem_a, gmem_b, tensor_map_buffer = gmem
    tensor_map_a_base, tensor_map_b_base, tensor_map_sfa, tensor_map_sfb = maps
    smem_tmap, smem_a, smem_b, smem_sfa, smem_sfb = smem
    shape_m, shape_n, shape_k = shapes
    BM, BN, BK, mc = spec.block_m, spec.block_n, spec.block_k, spec.num_tma_multicast
    stages = spec.num_stages
    on_a = spec.is_tma_multicast_on_a
    is_k_grouped = spec.gemm_type == GemmType.K_GROUPED_CONTIGUOUS
    c = consts

    # Per-CTA GMEM tensor-map slots (`.cuh:99-100`).
    cta = _u32(txl.cta_id())
    gmem_tmap_a = tensor_map_buffer.ptr_to([txl.cast(cta * txl.uint32(2), "int64") * 128])
    gmem_tmap_b = tensor_map_buffer.ptr_to([txl.cast(cta * txl.uint32(2) + 1, "int64") * 128])
    tmap_a = gmem_tmap_a if is_k_grouped else txl.address_of(tensor_map_a_base)
    tmap_b = gmem_tmap_b if is_k_grouped else txl.address_of(tensor_map_b_base)

    sched = _Scheduler(spec, grouped_layout, shape_m, shape_n, shape_k)
    last_group_idx = txl.local_scalar("uint32", init=txl.uint32(spec.num_groups))
    # `while (scheduler.get_next_block(...))`: one copy of the scheduler, at the loop top.
    running = txl.local_scalar("uint32", init=txl.uint32(1))
    with txl.while_(running == txl.uint32(1)):
        sched.next()
        with txl.if_(sched.valid == txl.uint32(1)):
            with txl.then_():
                m_block, n_block = sched.m_block_idx, sched.n_block_idx
                # `num_tma_multicast_a/b` (`.cuh:179-181`): the multicast side gets `kNumTMAMulticast`.
                valid = sched.is_tma_multicast_valid() if mc > 1 else False
                if isinstance(valid, bool):
                    # Compile-time: one TMA path per operand.
                    mc_a = mc if (on_a and valid) else 1
                    mc_b = mc if (not on_a and valid) else 1
                else:
                    mc_a = txl.local_scalar("uint32", init=txl.uint32(1))
                    mc_b = txl.local_scalar("uint32", init=txl.uint32(1))
                    with txl.if_(valid):
                        with txl.then_():
                            txl.assign(mc_a if on_a else mc_b, txl.uint32(mc))
                num_k_blocks = txl.local_scalar(
                    "uint32", init=(sched.current_shape_k + txl.uint32(BK - 1)) // txl.uint32(BK)
                )
                m_idx = txl.local_scalar("uint32", init=m_block * txl.uint32(BM))
                n_idx = txl.local_scalar("uint32", init=n_block * txl.uint32(BN))

                if is_k_grouped:
                    with txl.if_(last_group_idx != sched.current_group_idx):
                        with txl.then_():
                            txl.assign(last_group_idx, sched.current_group_idx)
                            # "Directly update current tensor map" (`.cuh:194-199`).
                            k_offset = txl.cast(sched.current_k_cumsum, "int64")
                            for which, base, dim in ((0, gmem_a, shape_m), (1, gmem_b, shape_n)):
                                txl.ptx.tensormap_replace.tile.global_address.shared__cta.b1024.b64(
                                    smem_tmap.ptr_to([which * 32]),
                                    txl.reinterpret(
                                        base.ptr_to([k_offset * txl.cast(dim, "int64")]), ty="uint64"
                                    ),
                                )
                            for which in (0, 1):
                                txl.ptx.tensormap_replace.tile.global_dim.shared__cta.b1024.b32(
                                    smem_tmap.ptr_to([which * 32]), txl.uint32(0), sched.current_shape_k
                                )
                                txl.ptx.tensormap_replace.tile.global_stride.shared__cta.b1024.b64(
                                    smem_tmap.ptr_to([which * 32]),
                                    txl.uint32(0),
                                    txl.cast(sched.current_shape_k, "uint64"),
                                )
                            # "Make sure the tensor maps are not used by in-flight TMA loads before
                            # updating GMEM" (`.cuh:202-206`).
                            txl.ptx.cp.async_.bulk.commit_group()
                            txl.ptx.cp.async_.bulk.wait_group.read(0)
                            txl.ptx.bar.warp.sync(txl.uint32(1) << lane_idx)
                            _copy_smem_tensor_map_to_gmem(smem_tmap, 0, gmem_tmap_a)
                            _copy_smem_tensor_map_to_gmem(smem_tmap, 32, gmem_tmap_b)
                            txl.ptx.fence.proxy.tensormap__generic.release.gpu()
                            # "Immediately acquire current tensor map" (`.cuh:213-214`).
                            txl.ptx.fence.proxy.tensormap__generic.acquire.gpu(gmem_tmap_a)
                            txl.ptx.fence.proxy.tensormap__generic.acquire.gpu(gmem_tmap_b)

                unroll = c["unrolls"] if c["unrolls"] > 0 else None
                if unroll is None and mc > 1 and not isinstance(valid, bool):
                    # `#pragma unroll 0` leaves the factor to the compiler.  With both multicast-side load
                    # paths live (multicast validity is a runtime value: an odd block group, or a group size
                    # not fixed by the compiled shape) nvcc unrolls this loop x2 (+1 remainder copy; the
                    # 128x120, 4160x7232 and 1024x2920 exports), where NVRTC picks x4 and spills the
                    # 40-register producer: pin nvcc's x2.
                    unroll = 2
                with txl.serial(0, num_k_blocks, unroll=unroll, dtype="uint32") as k_block_idx:
                    # Wait consumer release (`.cuh:220-221`).
                    stage_idx, phase = _pipeline(iter_idx, stages)
                    _wait(empty(stage_idx), phase ^ txl.uint32(1))
                    k_idx = k_block_idx * txl.uint32(BK)
                    sf_k_idx = sched.current_sf_k_cumsum + k_block_idx

                    def tma_copy(dst, tmap, inner, outer, num_multicast, may_multicast):
                        """`tma::copy` (`tma_copy.cuh:20-55`): plain when `num_tma_multicast == 1`, else
                        the cluster-rank-0 multicast into both CTAs."""

                        def plain():
                            txl.ptx[_TMA_LOAD_HINT](
                                dst,
                                tmap,
                                txl.cast(inner, "int32"),
                                txl.cast(outer, "int32"),
                                full(stage_idx),
                                txl.uint64(EVICT_NORMAL),
                            )

                        if not may_multicast or (isinstance(num_multicast, int) and num_multicast == 1):
                            plain()  # `num_tma_multicast == 1` is a compile-time fact here
                            return
                        if isinstance(num_multicast, int):  # compile-time multicast
                            with txl.if_(_u32(txl.cuda.cluster_cta_id("x")) == txl.uint32(0)):
                                with txl.then_():
                                    txl.ptx[_TMA_LOAD_MULTICAST](
                                        dst,
                                        tmap,
                                        txl.cast(inner, "int32"),
                                        txl.cast(outer, "int32"),
                                        full(stage_idx),
                                        txl.uint16((1 << mc) - 1),
                                        txl.uint64(EVICT_NORMAL),
                                    )
                            return
                        with txl.if_(num_multicast == txl.uint32(1)):
                            with txl.then_():
                                plain()
                            with txl.else_():
                                with txl.if_(_u32(txl.cuda.cluster_cta_id("x")) == txl.uint32(0)):
                                    with txl.then_():
                                        txl.ptx[_TMA_LOAD_MULTICAST](
                                            dst,
                                            tmap,
                                            txl.cast(inner, "int32"),
                                            txl.cast(outer, "int32"),
                                            full(stage_idx),
                                            txl.uint16((1 << mc) - 1),
                                            txl.uint64(EVICT_NORMAL),
                                        )

                    mc_on_a, mc_on_b = mc > 1 and on_a, mc > 1 and not on_a
                    tma_copy(
                        smem_sfa.ptr_to([stage_idx * txl.uint32(c["sfa_stage"] // 4)]),
                        txl.address_of(tensor_map_sfa), m_idx, sf_k_idx, mc_a, mc_on_a,
                    )  # fmt: skip
                    tma_copy(
                        smem_sfb.ptr_to([stage_idx * txl.uint32(c["sfb_stage"] // 4)]),
                        txl.address_of(tensor_map_sfb), n_idx, sf_k_idx, mc_b, mc_on_b,
                    )  # fmt: skip
                    tma_copy(
                        smem_a.ptr_to([stage_idx * txl.uint32(c["a_stage"])]),
                        tmap_a, k_idx, m_idx, mc_a, mc_on_a,
                    )  # fmt: skip
                    tma_copy(
                        smem_b.ptr_to([stage_idx * txl.uint32(c["b_stage"])]),
                        tmap_b, k_idx, n_idx, mc_b, mc_on_b,
                    )  # fmt: skip
                    txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](
                        full(stage_idx), txl.uint32(c["expect_tx"])
                    )
            with txl.else_():
                txl.assign(running, txl.uint32(0))

    if mc > 1:
        # "To safely deconstruct distributed shared barriers, we need another round of empty
        # waits" (`.cuh:237-244`).
        for _ in range(stages):
            stage_idx, phase = _pipeline(iter_idx, stages)
            _wait(empty(stage_idx), phase ^ txl.uint32(1))


def _math(
    *, spec, grouped_layout, shapes, full, empty, iter_idx, warp_idx, lane_idx, thread_idx,
    tensor_map_cd, smem, consts,
):  # fmt: skip
    """Math warpgroups: WGMMA + two-sided FP32 promotion, then the TMA reduce-add epilogue."""
    smem_d, smem_a, smem_b, smem_sfa, smem_sfb = smem
    shape_m, shape_n, shape_k = shapes
    c = consts
    BN, BM, mc = spec.block_n, spec.block_m, spec.num_tma_multicast
    stages = spec.num_stages
    num_accum = c["num_accum"]
    is_k_grouped = spec.gemm_type == GemmType.K_GROUPED_CONTIGUOUS

    # "use `__shfl_sync` to encourage NVCC to use unified registers" (`.cuh:251`).
    math_wg_idx = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(
        math_wg_idx,
        thread_idx // txl.uint32(128),
        txl.uint32(0),
        txl.uint32(31),
        txl.uint32(0xFFFFFFFF),
    )
    row_idx = lane_idx // txl.uint32(4)
    col_idx = txl.local_scalar("uint32", init=lane_idx % txl.uint32(4))
    r_0 = txl.local_scalar("uint32", init=warp_idx * txl.uint32(16) + row_idx)
    r_1 = txl.local_scalar("uint32", init=r_0 + txl.uint32(8))

    accum = txl.alloc_local((num_accum,), "float32")
    final_accum = txl.alloc_local((num_accum,), "float32")
    scales_b = txl.alloc_local((num_accum // 2,), "float32")  # float2 x kNumAccum/4
    scale_a = txl.alloc_local((2,), "float32")
    sched = _Scheduler(spec, grouped_layout, shape_m, shape_n, shape_k)

    def empty_barrier_arrive(stage_idx):
        """`empty_barrier_arrive` (`.cuh:266-274`)."""
        if mc == 1:
            with txl.if_(lane_idx == txl.uint32(0)):
                with txl.then_():
                    txl.ptx["mbarrier.arrive.shared::cta.b64"](empty(stage_idx))
        else:
            target = txl.local_scalar(
                "uint32",
                init=txl.select(
                    sched.is_peer_cta_alive == txl.uint32(1),
                    lane_idx,
                    _u32(txl.cuda.cluster_cta_id("x")),
                ),
            )
            with txl.if_(lane_idx < txl.uint32(mc)):
                with txl.then_():
                    remote = txl.local_scalar("uint32")
                    txl.ptx["mapa.shared::cluster.u32"](
                        remote, _u32(txl.cuda.cvta_generic_to_shared(empty(stage_idx))), target
                    )
                    txl.ptx["mbarrier.arrive.shared::cluster.b64"](remote)

    def smem_desc(ptr):
        """`mma::sm90::make_smem_desc(ptr, 1)`: start address >> 4 in bits [0,14), SBO 1024,
        128B-swizzle layout code."""
        lo = (_u32(txl.cuda.cvta_generic_to_shared(ptr)) >> txl.uint32(4)) & txl.uint32(0x3FFF)
        return txl.uint64(c["desc_hi64"]) | txl.cast(lo, "uint64")

    # `while (scheduler.get_next_block(...))`: one copy of the scheduler, at the loop top.
    running = txl.local_scalar("uint32", init=txl.uint32(1))
    with txl.while_(running == txl.uint32(1)):
        sched.next()
        with txl.if_(sched.valid == txl.uint32(1)):
            with txl.then_():
                m_block, n_block = sched.m_block_idx, sched.n_block_idx
                current_shape_k = sched.current_shape_k if is_k_grouped else shape_k
                current_group_idx = sched.current_group_idx if is_k_grouped else txl.uint32(0)
                num_k_blocks = txl.local_scalar(
                    "uint32", init=(current_shape_k + txl.uint32(127)) // txl.uint32(128)
                )
                for i in range(num_accum):
                    txl.assign(final_accum[i], txl.float32(0))

                unroll = c["unrolls"] if c["unrolls"] > 0 else None
                with txl.serial(0, num_k_blocks, unroll=unroll, dtype="uint32"):
                    # Wait TMA arrivals (`.cuh:277-279`).
                    stage_idx, phase = _pipeline(iter_idx, stages)
                    _wait(full(stage_idx), phase)

                    # Read A / B scales: "all shared memory read must be prior to `warpgroup_arrive`"
                    # (`.cuh:281-289`).
                    sfa_base = stage_idx * txl.uint32(c["sfa_stage"] // 4)
                    txl.ptx["ld.shared.f32"](scale_a[0], smem_sfa.ptr_to([sfa_base + r_0]))
                    txl.ptx["ld.shared.f32"](scale_a[1], smem_sfa.ptr_to([sfa_base + r_1]))
                    sfb_base = stage_idx * txl.uint32(c["sfb_stage"] // 4) + col_idx * txl.uint32(2)
                    for i in range(num_accum // 4):
                        txl.ptx["ld.shared.v2.f32"](
                            scales_b[2 * i], scales_b[2 * i + 1],
                            smem_sfb.ptr_to([sfb_base + txl.uint32(8 * i)]),
                        )  # fmt: skip

                    # Commit WGMMA instructions (`.cuh:291-306`).
                    for i in range(num_accum):
                        txl.cuda.wgmma.noop_barrier(accum[i])
                    txl.ptx.wgmma.fence.sync.aligned()
                    a_stage_base = stage_idx * txl.uint32(c["a_stage"]) + math_wg_idx * txl.uint32(64 * 128)
                    b_stage_base = stage_idx * txl.uint32(c["b_stage"])
                    for k in range(128 // 32):
                        desc_a = smem_desc(smem_a.ptr_to([a_stage_base + txl.uint32(k * 32)]))
                        desc_b = smem_desc(smem_b.ptr_to([b_stage_base + txl.uint32(k * 32)]))
                        txl.ptx[c["wgmma"]](*[accum[j] for j in range(num_accum)], desc_a, desc_b, k != 0, 1, 1)
                    txl.ptx.wgmma.commit_group.sync.aligned()
                    for i in range(num_accum):
                        txl.cuda.wgmma.noop_barrier(accum[i])
                    txl.ptx.wgmma.wait_group.sync.aligned(0)

                    # Notify barrier arrival (`.cuh:309`).
                    empty_barrier_arrive(stage_idx)

                    # Promote with scales (`.cuh:312-320`): `final += scale_a * scale_b * accum`, which the
                    # source compiles (no fast math) to one non-FTZ `mul.f32` and one `fma.rn.f32`.
                    for i in range(num_accum // 4):
                        for j, sa, sb in (
                            (0, scale_a[0], scales_b[2 * i]),
                            (1, scale_a[0], scales_b[2 * i + 1]),
                            (2, scale_a[1], scales_b[2 * i]),
                            (3, scale_a[1], scales_b[2 * i + 1]),
                        ):
                            idx = 4 * i + j
                            txl.assign(final_accum[idx], final_accum[idx] + sa * sb * accum[idx])

                # ---- epilogue (`.cuh:322-346`) ----
                # Flush previous stores.
                with txl.if_(warp_idx % txl.uint32(4) == txl.uint32(0)):
                    with txl.then_():
                        with txl.if_(_elect() == txl.uint32(1)):
                            with txl.then_():
                                txl.ptx.cp.async_.bulk.wait_group.read(0)
                named_barrier = math_wg_idx + txl.uint32(NAMED_BARRIER_BASE)
                txl.ptx["bar.sync"](named_barrier, txl.uint32(128))

                # Store to D shared memory: `smem_d_0 + i * 4` on a float2 pointer is 8 columns per i.
                d_0 = r_0 * txl.uint32(BN) + col_idx * txl.uint32(2)
                d_1 = r_1 * txl.uint32(BN) + col_idx * txl.uint32(2)
                for i in range(num_accum // 4):
                    txl.ptx["st.shared.v2.f32"](
                        smem_d.ptr_to([d_0 + txl.uint32(8 * i)]), final_accum[4 * i], final_accum[4 * i + 1]
                    )
                    txl.ptx["st.shared.v2.f32"](
                        smem_d.ptr_to([d_1 + txl.uint32(8 * i)]),
                        final_accum[4 * i + 2],
                        final_accum[4 * i + 3],
                    )
                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx["bar.sync"](named_barrier, txl.uint32(128))

                # Use TMA store to write back to global memory (`.cuh:339-345`); the elected lane's own
                # r_0 / col_idx address its warpgroup's 64 rows.
                with txl.if_(warp_idx % txl.uint32(4) == txl.uint32(0)):
                    with txl.then_():
                        with txl.if_(_elect() == txl.uint32(1)):
                            with txl.then_():
                                txl.ptx[_TMA_REDUCE_ADD](
                                    txl.address_of(tensor_map_cd),
                                    txl.cast(n_block * txl.uint32(BN), "int32"),
                                    txl.cast(
                                        current_group_idx * shape_m + m_block * txl.uint32(BM) + r_0,
                                        "int32",
                                    ),
                                    smem_d.ptr_to([d_0]),
                                )
                                txl.ptx.cp.async_.bulk.commit_group()
                txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))
            with txl.else_():
                txl.assign(running, txl.uint32(0))


# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Host-side configuration of DeepGEMM's SM90 FP8 1D1D GEMM.

A line-for-line port of the descriptor and heuristic code the two SM90 1D1D entry points run
before JIT-compiling `sm90_fp8_gemm_1d1d_impl`:

- `GemmDesc`: `csrc/jit_kernels/heuristics/config.hpp:12`, as filled by
  `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:92` (normal) and `:178` (k-grouped).
- `get_best_config`: `heuristics/common.hpp:13` over `SM90ArchSpec` (`heuristics/sm90.hpp:13`)
  including its cost model, so the TIRx kernel is specialized with the exact block sizes,
  cluster shape and stage count DeepGEMM picks.
- `KernelSpec`: the template arguments `SM90FP8Gemm1D1DRuntime::generate_impl` emits
  (`impls/sm90_fp8_gemm_1d1d.hpp:33`).

Only what the 1D1D kernel accepts is modelled: FP8 x FP8 -> FP32 with accumulation, K-major
A and B, per-128 FP32 scales on both operands (`recipe = (1, 1, 128)`).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

#: `SM90ArchSpec::smem_capacity`.
SMEM_CAPACITY = 232448
#: `kNumMaxStages` in `SM90ArchSpec::get_pipeline_config`.
MAX_STAGES = 16
BLOCK_K = 128
WGMMA_M = 64
#: SM count of the SM90 parts the port targets (H100 / H200 SXM), used when a descriptor is
#: built before CUDA is initialized.
SM90_NUM_SMS = 132
#: Size of one `CUtensorMap`.
TENSOR_MAP_BYTES = 128


class GemmType(enum.IntEnum):
    """`deep_gemm/common/types.cuh::GemmType` (values as in the source enum)."""

    NORMAL = 0
    M_GROUPED_CONTIGUOUS = 1
    M_GROUPED_MASKED = 2
    K_GROUPED_CONTIGUOUS = 3
    BATCHED = 4
    M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT = 5


@dataclass(frozen=True)
class GemmDesc:
    gemm_type: GemmType
    m: int
    n: int
    k: int
    num_groups: int
    num_sms: int
    expected_m: int = 0
    expected_n: int = 0
    expected_k: int = 0
    expected_num_groups: int = 0
    #: The entry points' defaults: `fp8_fp4_gemm_nt` compiles "nk", the k-grouped API "mn".
    compiled_dims: str = "nk"

    def get_expected_m(self) -> int:
        return self.expected_m if self.expected_m > 0 else self.m

    def get_expected_n(self) -> int:
        return self.expected_n if self.expected_n > 0 else self.n

    def get_expected_k(self) -> int:
        return self.expected_k if self.expected_k > 0 else self.k

    def get_expected_num_groups(self) -> int:
        return self.expected_num_groups if self.expected_num_groups > 0 else self.num_groups


@dataclass(frozen=True)
class Layout:
    block_m: int
    block_n: int
    block_k: int
    cluster_m: int
    cluster_n: int

    @property
    def cluster_size(self) -> int:
        return self.cluster_m * self.cluster_n


@dataclass(frozen=True)
class StorageConfig:
    load_block_m: int
    load_block_n: int
    store_block_m: int
    store_block_n: int
    swizzle_a_mode: int
    swizzle_b_mode: int
    swizzle_cd_mode: int


@dataclass(frozen=True)
class PipelineConfig:
    smem_size: int
    num_stages: int


@dataclass(frozen=True)
class LaunchConfig:
    num_sms: int
    num_sms_per_cluster: int
    num_threads: int
    num_tma_threads: int
    num_math_threads: int


@dataclass(frozen=True)
class GemmConfig:
    layout: Layout
    storage_config: StorageConfig
    pipeline_config: PipelineConfig
    launch_config: LaunchConfig


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def get_swizzle_mode(block_size: int, elem_size: int) -> int:
    """`heuristics/utils.hpp::get_swizzle_mode`."""
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise AssertionError("unreachable")


def _check_validity(desc: GemmDesc) -> None:
    """`GemmDesc::check_validity` (`config.hpp:39`): the dtype checks hold by construction."""
    if desc.num_sms % 2 != 0:
        raise ValueError("DeepGEMM requires an even SM count")


def get_storage_config(desc: GemmDesc, layout: Layout) -> StorageConfig:
    """`SM90ArchSpec::get_storage_config` (K-major FP8 A/B, FP32 C/D, 1D1D kernel)."""
    del desc
    load_block_m, load_block_n = layout.block_m, layout.block_n
    # "1D1D kernel will do single warp-group stores"
    store_block_m, store_block_n = WGMMA_M, layout.block_n
    return StorageConfig(
        load_block_m,
        load_block_n,
        store_block_m,
        store_block_n,
        get_swizzle_mode(layout.block_k, 1),
        get_swizzle_mode(layout.block_k, 1),
        0,  # "We only enable swizzling for non-FP32 outputs"
    )


def get_pipeline_config(desc: GemmDesc, layout: Layout, storage: StorageConfig) -> PipelineConfig:
    """`SM90ArchSpec::get_pipeline_config` for the 1D1D kernel."""
    smem_cd = align_up(layout.block_m * layout.block_n * 4, 1024)
    smem_barriers = MAX_STAGES * 8 * 2
    smem_a_per_stage = storage.load_block_m * layout.block_k
    smem_b_per_stage = storage.load_block_n * layout.block_k
    smem_sfa_per_stage = align_up(layout.block_m * 4, 128)
    smem_sfb_per_stage = align_up(layout.block_n * 4, 128)
    smem_extra_sfb = 0  # 1D2D only
    smem_tensormap = (
        4 * TENSOR_MAP_BYTES if desc.gemm_type == GemmType.K_GROUPED_CONTIGUOUS else 0
    )
    smem_extra = smem_cd + smem_barriers + smem_extra_sfb + smem_tensormap
    smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage
    num_stages = min((SMEM_CAPACITY - smem_extra) // smem_per_stage, MAX_STAGES)
    return PipelineConfig(smem_extra + num_stages * smem_per_stage, num_stages)


def get_launch_config(desc: GemmDesc, layout: Layout) -> LaunchConfig:
    """`SM90ArchSpec::get_launch_config`."""
    num_tma_threads = 128
    num_math_threads = 128 if layout.block_m <= 64 else 256
    return LaunchConfig(
        desc.num_sms,
        layout.cluster_size,
        num_tma_threads + num_math_threads,
        num_tma_threads,
        num_math_threads,
    )


def get_layout_candidates(desc: GemmDesc) -> list[Layout]:
    """`SM90ArchSpec::get_layout_candidates` for the FP8 1D1D (FP32 output) paths."""
    if desc.gemm_type not in (GemmType.NORMAL, GemmType.K_GROUPED_CONTIGUOUS):
        raise NotImplementedError(f"gemm_type {desc.gemm_type!r} is not a 1D1D type")
    block_m_candidates = [64, 128]
    # "smaller block M can avoid TMA L2 OOB bound"
    if desc.m <= 16:
        block_m_candidates.append(16)
    if desc.m <= 32:
        block_m_candidates.append(32)
    # FP32 output: no BLOCK_M 256.

    step = math.lcm(16, 1)  # `get_block_n_multiple_of()` defaults to 1
    # "Avoid bank conflicts for 1D1D kernel FP32 output": 16, then 24, 40, ... (odd multiples of 8).
    block_n_candidates = [16]
    start, end = 24, 160  # 1D1D: `end = 160` ("Register spills")
    block_n_candidates += list(range(start, end + 1, step))
    block_k = BLOCK_K  # `128 / get_element_size(MXFP8FP4)`
    # "The number of k-groups is large (a heuristic)"
    disable_multicast = desc.gemm_type == GemmType.K_GROUPED_CONTIGUOUS and desc.num_groups > 4

    candidates = []
    max_cluster = 1 if disable_multicast else 2
    for cluster_m in range(1, max_cluster + 1):
        for cluster_n in range(1, max_cluster + 1):
            if cluster_m * cluster_n > 2:
                continue
            if desc.num_sms % (cluster_m * cluster_n) != 0:
                continue
            for block_m in block_m_candidates:
                for block_n in block_n_candidates:
                    # (The 1D2D unroll and the masked/psum multicast rules do not apply.)
                    if block_m > 128 and block_n > 128:
                        continue
                    layout = Layout(block_m, block_n, block_k, cluster_m, cluster_n)
                    storage = get_storage_config(desc, layout)
                    if storage.swizzle_a_mode % 64 != 0 or storage.swizzle_b_mode % 64 != 0:
                        continue
                    num_stages = get_pipeline_config(desc, layout, storage).num_stages
                    if num_stages < 3 or (block_m * block_n < 128 * 192 and num_stages < 4):
                        continue
                    candidates.append(layout)
    if not candidates:
        raise AssertionError("no SM90 layout candidate")
    return candidates


_INT64_MAX = np.iinfo(np.int64).max


def get_layout_cycles(desc: GemmDesc, layout: Layout) -> int:
    """`SM90ArchSpec::get_layout_info(...).num_cycles`, with the source's integer/float types."""
    num_blocks = (
        ceil_div(desc.get_expected_m(), layout.block_m)
        * ceil_div(desc.get_expected_n(), layout.block_n)
        * desc.get_expected_num_groups()
    )
    num_waves = ceil_div(num_blocks, desc.num_sms)
    # `const int l2_bandwidth_per_cycle = std::min(64. * num_sms, 8e6 / 1.3e3)`: a truncated double.
    l2_bandwidth_per_cycle = int(min(64.0 * desc.num_sms, 8e6 / 1.3e3))
    l1_bandwidth_per_cycle = 128 * desc.num_sms
    elem_size_ab, elem_size_cd = 1, 4
    expected_k = desc.get_expected_k()
    num_bytes_l2_ab = (
        expected_k
        * (layout.block_m // layout.cluster_n + layout.block_n // layout.cluster_m)
        * elem_size_ab
    )
    num_bytes_l1_ab = expected_k * (layout.block_m + layout.block_n) * elem_size_ab
    num_bytes_l1_tc = (
        expected_k * (max(WGMMA_M, layout.block_m) + layout.block_n) * elem_size_ab
        + layout.block_m * layout.block_n * elem_size_cd
    )
    # `with_accumulation` is always true on these paths.
    num_bytes_l1_l2_cd = layout.block_m * layout.block_n * elem_size_cd * 2
    num_l2_cycles = (num_bytes_l2_ab + num_bytes_l1_l2_cd) * num_blocks // l2_bandwidth_per_cycle
    num_l1_cycles = (
        (num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd)
        * num_blocks
        // l1_bandwidth_per_cycle
    )
    # `float wave_efficiency`, then `int64 / float`: both in single precision, as in C++.
    wave_efficiency = np.float32(num_blocks) / np.float32(num_waves * desc.num_sms)
    num_cycles = int(np.float32(max(num_l1_cycles, num_l2_cycles)) / wave_efficiency)
    if layout.cluster_size > 1 and num_waves <= 1:
        num_cycles = int(_INT64_MAX)
    return num_cycles


def get_best_config(desc: GemmDesc) -> GemmConfig:
    """`heuristics/common.hpp::get_best_config<SM90ArchSpec>`: the first strict minimum."""
    _check_validity(desc)
    candidates = get_layout_candidates(desc)
    best, best_cycles = candidates[0], get_layout_cycles(desc, candidates[0])
    for layout in candidates[1:]:
        cycles = get_layout_cycles(desc, layout)
        if cycles < best_cycles:
            best, best_cycles = layout, cycles
    storage = get_storage_config(desc, best)
    return GemmConfig(
        best, storage, get_pipeline_config(desc, best, storage), get_launch_config(desc, best)
    )


@dataclass(frozen=True)
class KernelSpec:
    """The `sm90_fp8_gemm_1d1d_impl` template arguments (`generate_impl`)."""

    gemm_type: GemmType
    shape_m: int  # 0: runtime
    shape_n: int
    shape_k: int
    num_groups: int
    block_m: int
    block_n: int
    block_k: int
    swizzle_a_mode: int
    swizzle_b_mode: int
    num_stages: int
    num_tma_threads: int
    num_math_threads: int
    num_tma_multicast: int
    is_tma_multicast_on_a: bool
    num_sms: int
    smem_size: int


def make_spec(desc: GemmDesc, config: GemmConfig) -> KernelSpec:
    def compiled(dim: int, name: str) -> int:
        return dim if name in desc.compiled_dims else 0

    layout, storage = config.layout, config.storage_config
    # "Requires no TMA splits" (`sm90_fp8_gemm_1d1d.hpp:113,186`)
    if storage.swizzle_a_mode != layout.block_k or storage.swizzle_b_mode != layout.block_k:
        raise AssertionError("SM90 1D1D requires no TMA splits")
    # `DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * (BLOCK_M <= 64 ? 1 : 2))` (`.cuh:258`): the source
    # cannot instantiate BLOCK_M 16 / 32 either.
    if layout.block_m not in (64, 128):
        raise NotImplementedError(f"BLOCK_M = {layout.block_m} does not compile upstream either")
    return KernelSpec(
        gemm_type=desc.gemm_type,
        shape_m=compiled(desc.m, "m"),
        shape_n=compiled(desc.n, "n"),
        shape_k=compiled(desc.k, "k"),
        num_groups=desc.num_groups,
        block_m=layout.block_m,
        block_n=layout.block_n,
        block_k=layout.block_k,
        swizzle_a_mode=storage.swizzle_a_mode,
        swizzle_b_mode=storage.swizzle_b_mode,
        num_stages=config.pipeline_config.num_stages,
        num_tma_threads=config.launch_config.num_tma_threads,
        num_math_threads=config.launch_config.num_math_threads,
        num_tma_multicast=layout.cluster_size,
        is_tma_multicast_on_a=layout.cluster_n > 1,
        num_sms=config.launch_config.num_sms,
        smem_size=config.pipeline_config.smem_size,
    )


def class_key(spec: KernelSpec) -> tuple:
    """The specialization class: what changes the emitted instruction stream."""
    return (
        "k_grouped" if spec.gemm_type == GemmType.K_GROUPED_CONTIGUOUS else "normal",
        spec.block_m,
        spec.block_n,
        spec.num_tma_multicast,
        spec.is_tma_multicast_on_a,
    )

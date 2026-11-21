# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Host-side configuration of DeepGEMM's SM90 FP8 1D2D GEMM.

A line-for-line port of the descriptor and heuristic code the m-grouped APIs run before
JIT-compiling `sm90_fp8_gemm_1d2d_impl`:

- `GemmDesc`: `csrc/jit_kernels/heuristics/config.hpp:12`, as filled by
  `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:157` (contiguous) and `:226` (masked).
- `get_best_config`: `heuristics/common.hpp:13` over `SM90ArchSpec`
  (`heuristics/sm90.hpp:13`), including its cost model, so the TIRx kernel is
  specialized with the exact block sizes, cluster shape and stage count DeepGEMM picks.
- `KernelSpec`: the template arguments `SM90FP8Gemm1D2DRuntime::generate_impl` emits
  (`impls/sm90_fp8_gemm_1d2d.hpp:35`).

Only the FP8 x FP8 -> BF16, K-major A/B, no-accumulation paths the m-grouped APIs reach are
modelled.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

#: `SM90ArchSpec::smem_capacity`.
SMEM_CAPACITY = 232448
#: `HeuristicsRuntime::kLegacyMKAlignmentForContiguousLayout`; SM90 always uses it
#: (`get_theoretical_mk_alignment_for_contiguous_layout` returns it below SM100).
MK_ALIGNMENT = 128
#: `kNumMaxStages` in `SM90ArchSpec::get_pipeline_config`.
MAX_STAGES = 16
BLOCK_K = 128
WGMMA_M = 64


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
    #: `compiled_dims="nk"`, the m-grouped APIs' default: N and K become template constants.
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
    if desc.num_sms % 2 != 0:
        raise ValueError("DeepGEMM requires an even SM count")


def get_storage_config(desc: GemmDesc, layout: Layout) -> StorageConfig:
    """`SM90ArchSpec::get_storage_config` (K-major FP8 A/B, BF16 D)."""
    load_block_m, load_block_n = layout.block_m, layout.block_n
    store_block_m, store_block_n = layout.block_m, layout.block_n  # 1D2D: not a 1D1D kernel
    return StorageConfig(
        load_block_m,
        load_block_n,
        store_block_m,
        store_block_n,
        get_swizzle_mode(layout.block_k, 1),
        get_swizzle_mode(layout.block_k, 1),
        get_swizzle_mode(store_block_n, 2),
    )


def get_pipeline_config(desc: GemmDesc, layout: Layout, storage: StorageConfig) -> PipelineConfig:
    """`SM90ArchSpec::get_pipeline_config` for the 1D2D kernel."""
    smem_cd = align_up(layout.block_m * layout.block_n * 2, 1024)
    smem_barriers = MAX_STAGES * 8 * 2
    smem_a_per_stage = storage.load_block_m * layout.block_k
    smem_b_per_stage = storage.load_block_n * layout.block_k
    smem_sfa_per_stage = align_up(layout.block_m * 4, 128)
    smem_sfb_per_stage = 0  # only the 1D1D kernel stages SFB
    use_uniform_sfb = 1 if layout.block_k % layout.block_n == 0 else 2
    smem_extra_sfb = align_up(ceil_div(desc.k, layout.block_k) * 4 * use_uniform_sfb, 8)
    smem_tensormap = 0  # only for k-grouped
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


def get_layout_candidates(desc: GemmDesc, *, mk_alignment: int = MK_ALIGNMENT) -> list[Layout]:
    """`SM90ArchSpec::get_layout_candidates` for the FP8 1D2D m-grouped paths."""
    if desc.gemm_type in (
        GemmType.M_GROUPED_CONTIGUOUS,
        GemmType.M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT,
    ):
        block_m_candidates = [mk_alignment]
    elif desc.gemm_type == GemmType.M_GROUPED_MASKED:
        block_m_candidates = [64, 128]
    else:
        raise NotImplementedError(f"gemm_type {desc.gemm_type!r} is not ported")

    step = math.lcm(16, 1)  # `get_block_n_multiple_of()` defaults to 1
    block_n_candidates = list(range(step, 192 + 1, step))  # 1D2D: `end = 192`
    block_k = BLOCK_K  # `128 / get_element_size(MXFP8FP4)`
    disable_multicast = False  # only k-grouped (> 4 groups) and batched disable it

    candidates = []
    for cluster_m in range(1, (1 if disable_multicast else 2) + 1):
        for cluster_n in range(1, (1 if disable_multicast else 2) + 1):
            if cluster_m * cluster_n > 2:
                continue
            if desc.num_sms % (cluster_m * cluster_n) != 0:
                continue
            for block_m in block_m_candidates:
                for block_n in block_n_candidates:
                    if block_n > block_k and (
                        block_n % (block_n - block_k) != 0 and block_k % (block_n - block_k) != 0
                    ):
                        continue
                    if (
                        desc.gemm_type
                        in (
                            GemmType.M_GROUPED_MASKED,
                            GemmType.M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT,
                        )
                        and ceil_div(desc.n, block_n) % (cluster_m * cluster_n) != 0
                    ):
                        continue
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
    elem_size_ab, elem_size_cd = 1, 2
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
    num_bytes_l1_l2_cd = layout.block_m * layout.block_n * elem_size_cd  # no accumulation
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


def get_best_config(desc: GemmDesc, *, mk_alignment: int = MK_ALIGNMENT) -> GemmConfig:
    """`heuristics/common.hpp::get_best_config<SM90ArchSpec>`: the first strict minimum."""
    _check_validity(desc)
    candidates = get_layout_candidates(desc, mk_alignment=mk_alignment)
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
    """The `sm90_fp8_gemm_1d2d_impl` template arguments (`generate_impl`)."""

    gemm_type: GemmType
    shape_m: int  # 0: runtime (M is never compiled for "nk")
    shape_n: int
    shape_k: int
    num_groups: int
    block_m: int
    block_n: int
    block_k: int
    swizzle_a_mode: int
    swizzle_b_mode: int
    swizzle_d_mode: int
    num_stages: int
    num_tma_threads: int
    num_math_threads: int
    num_tma_multicast: int
    is_tma_multicast_on_a: bool
    num_sms: int
    smem_size: int
    #: `kMajorSFB`: SFB stays K-major FP32 on SM90 (`layout.hpp:45`).
    major_sfb_k: bool = True


def make_spec(desc: GemmDesc, config: GemmConfig) -> KernelSpec:
    def compiled(dim: int, name: str) -> int:
        return dim if name in desc.compiled_dims else 0

    layout, storage = config.layout, config.storage_config
    if storage.swizzle_a_mode != layout.block_k or storage.swizzle_b_mode != layout.block_k:
        raise AssertionError("SM90 1D2D requires no TMA splits")  # `sm90_fp8_gemm_1d2d.hpp`
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
        swizzle_d_mode=storage.swizzle_cd_mode,
        num_stages=config.pipeline_config.num_stages,
        num_tma_threads=config.launch_config.num_tma_threads,
        num_math_threads=config.launch_config.num_math_threads,
        num_tma_multicast=layout.cluster_size,
        is_tma_multicast_on_a=layout.cluster_n > 1,
        num_sms=config.launch_config.num_sms,
        smem_size=config.pipeline_config.smem_size,
    )

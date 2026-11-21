# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TMA descriptors, compilation and launch for the SM90 FP8 1D2D GEMM.

The descriptors are the ones `sm90_m_grouped_fp8_gemm_{contiguous,masked}_1d2d` encode
(`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:177,246`) with `runtime_utils.hpp`'s
`make_tma_{a,b,cd,sf}_desc`; the 2-D encoder is shared with the SM100 port.
"""

from __future__ import annotations

from tirx_kernels.runner import cache_backend_config

from .._sm100_fp8_fp4_gemm_1d1d.spec import _encode_2d, tma_aligned_size
from .spec import GemmType, KernelSpec, ceil_div


def _grouped_tma(spec: KernelSpec) -> int:
    """`num_groups` the masked path folds into A/D/SFA's outer extent (1 for contiguous)."""
    return spec.num_groups if spec.gemm_type == GemmType.M_GROUPED_MASKED else 1


def make_tma_a_desc(a, spec: KernelSpec, *, shape_m: int, shape_k: int):
    """`make_tma_a_desc(K, a, m, k, load_block_m, block_k, a.stride(-2), groups, swizzle_a)`."""
    return _encode_2d(
        a,
        gmem_inner=shape_k,
        gmem_outer=shape_m * _grouped_tma(spec),
        smem_inner=spec.block_k,
        smem_outer=spec.block_m,
        gmem_outer_stride=int(a.stride(-2)),
        swizzle_mode=spec.swizzle_a_mode,
    )


def make_tma_b_desc(b, spec: KernelSpec, *, shape_n: int, shape_k: int):
    """`make_tma_b_desc(K, b, n, k, load_block_n, block_k, b.stride(-2), num_groups, swizzle_b)`."""
    return _encode_2d(
        b,
        gmem_inner=shape_k,
        gmem_outer=shape_n * spec.num_groups,
        smem_inner=spec.block_k,
        smem_outer=spec.block_n,
        gmem_outer_stride=int(b.stride(-2)),
        swizzle_mode=spec.swizzle_b_mode,
    )


def make_tma_d_desc(d, spec: KernelSpec, *, shape_m: int, shape_n: int):
    """`make_tma_cd_desc(d, m, n, store_block_m, store_block_n, d.stride(-2), groups, swizzle_cd)`."""
    return _encode_2d(
        d,
        gmem_inner=shape_n,
        gmem_outer=shape_m * _grouped_tma(spec),
        smem_inner=spec.block_n,
        smem_outer=spec.block_m,
        gmem_outer_stride=int(d.stride(-2)),
        swizzle_mode=spec.swizzle_d_mode,
    )


def make_tma_sfa_desc(sfa, spec: KernelSpec, *, shape_m: int, shape_k: int):
    """`make_tma_sf_desc(MN, sfa, m, k, block_m, block_k, groups, 0)` for FP32 scales."""
    shape_mn = tma_aligned_size(shape_m, int(sfa.element_size()))
    return _encode_2d(
        sfa,
        gmem_inner=shape_mn,
        gmem_outer=ceil_div(shape_k, spec.block_k) * _grouped_tma(spec),
        smem_inner=spec.block_m,
        smem_outer=1,
        gmem_outer_stride=shape_mn,
        swizzle_mode=0,
    )


@cache_backend_config
def _compile_spec_cached(spec: KernelSpec, *, backend_config=None):
    """Compile one specialization; `KernelSpec` is frozen, so it keys the cache."""
    import tvm_tirx
    from tirx_kernels.runner import cuda_target

    from .kernel import build_kernel

    func = build_kernel(spec, backend_config=backend_config)
    return tvm_tirx.tirx.compile(
        tvm_tirx.IRModule({"main": func}),
        target=cuda_target(backend_config=backend_config),
        pipeline="tirx",
    )


def compile_spec(spec: KernelSpec, *, backend_config=None):
    return _compile_spec_cached(spec, backend_config=backend_config)


def build_launch(
    spec: KernelSpec,
    *,
    executable=None,
    a,
    b,
    sfa,
    sfb,
    d,
    grouped_layout,
    shape_m: int,
    backend_config=None,
):
    """A no-argument closure issuing exactly one launch; descriptors are encoded here, outside it."""
    if executable is None:
        executable = compile_spec(spec, backend_config=backend_config)
    shape_n, shape_k = spec.shape_n, spec.shape_k
    maps = (
        make_tma_a_desc(a, spec, shape_m=shape_m, shape_k=shape_k),
        make_tma_b_desc(b, spec, shape_n=shape_n, shape_k=shape_k),
        make_tma_d_desc(d, spec, shape_m=shape_m, shape_n=shape_n),
        make_tma_sfa_desc(sfa, spec, shape_m=shape_m, shape_k=shape_k),
    )
    sfb_flat, layout_flat = sfb.reshape(-1), grouped_layout.reshape(-1)

    def launch():
        executable.mod(
            sfb_flat,
            layout_flat,
            shape_m,
            shape_n,
            shape_k,
            maps[0].ptr,
            maps[1].ptr,
            maps[2].ptr,
            maps[3].ptr,
        )

    return launch

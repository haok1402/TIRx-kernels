# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TMA descriptors, compilation and launch for the SM90 FP8 1D1D GEMM.

The descriptors are the ones `sm90_fp8_gemm_1d1d` and `sm90_k_grouped_fp8_gemm_1d1d` encode
(`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:116,189`) with `runtime_utils.hpp`'s
`make_tma_{a,b,cd,sf}_desc`; the 2-D encoder is shared with the SM100 port.
"""

from __future__ import annotations

from tirx_kernels.runner import cache_backend_config

from .._sm100_fp8_fp4_gemm_1d1d.spec import _encode_2d, tma_aligned_size
from .spec import BLOCK_K, TENSOR_MAP_BYTES, WGMMA_M, GemmType, KernelSpec, ceil_div


def make_tma_ab_desc(t, spec: KernelSpec, *, shape_mn: int, shape_k: int, block_mn: int):
    """`make_tma_{a,b}_desc(K, t, mn, k, block_mn, block_k, k, 1, swizzle)` (K-major, one group).

    For the k-grouped type this is the *base* descriptor, built on the first non-empty group;
    the kernel rewrites its address, inner dim and stride per group.
    """
    return _encode_2d(
        t,
        gmem_inner=shape_k,
        gmem_outer=shape_mn,
        smem_inner=spec.block_k,
        smem_outer=block_mn,
        gmem_outer_stride=shape_k,
        swizzle_mode=spec.swizzle_a_mode,
    )


def make_tma_sf_desc(sf, *, shape_mn: int, num_sf_k: int, block_mn: int):
    """`make_tma_sf_desc(MN, sf, mn, num_sf_k * 128, block_mn, block_k, 1, 0)` for FP32 scales."""
    aligned_mn = tma_aligned_size(shape_mn, int(sf.element_size()))
    return _encode_2d(
        sf,
        gmem_inner=aligned_mn,
        gmem_outer=num_sf_k,
        smem_inner=block_mn,
        smem_outer=1,
        gmem_outer_stride=aligned_mn,
        swizzle_mode=0,
    )


def make_tma_cd_desc(d, spec: KernelSpec, *, shape_m: int, shape_n: int, num_groups: int):
    """`make_tma_cd_desc(d, m, n, store_block_m = 64, store_block_n = BLOCK_N, d.stride(-2),
    num_groups, 0)`: one warpgroup's 64 rows per store, no swizzle (FP32 output)."""
    return _encode_2d(
        d,
        gmem_inner=shape_n,
        gmem_outer=shape_m * num_groups,
        smem_inner=spec.block_n,
        smem_outer=WGMMA_M,
        gmem_outer_stride=int(d.stride(-2)),
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


def build_launch(spec: KernelSpec, data, d, *, executable=None, backend_config=None):
    """A no-argument closure issuing exactly one launch into `d`, or `None` when the source entry
    returns early without launching (`gemm.hpp:19`: an empty problem, or K = 0)."""
    import torch

    M, N = data["M"], data["N"]
    if M == 0 or N == 0 or data["K"] == 0:
        return None
    if executable is None:
        executable = compile_spec(spec, backend_config=backend_config)
    is_k_grouped = spec.gemm_type == GemmType.K_GROUPED_CONTIGUOUS
    if is_k_grouped:
        ks = data["ks"]
        first_k = next(k for k in ks if k != 0)
        num_sf_k = sum(ceil_div(k, BLOCK_K) for k in ks)
        a_base, b_base = data["a"], data["b"]
        k_desc, groups = first_k, data["num_groups"]
        grouped_layout = data["grouped_layout"]
        # "`4` means the double buffering for both A and B operands (2 * 2)" (`gemm.hpp:387`).
        tensor_map_buffer = torch.empty(
            (spec.num_sms * 4 * TENSOR_MAP_BYTES,), dtype=torch.uint8, device=a_base.device
        )
    else:
        num_sf_k = ceil_div(data["K"], BLOCK_K)
        a_base, b_base = data["a"], data["b"]
        k_desc, groups = data["K"], 1
        # The normal entry passes null pointers (`sm90_fp8_gemm_1d1d.hpp:142`); the kernel never
        # dereferences them there.
        grouped_layout = torch.empty((1,), dtype=torch.int32, device=a_base.device)
        tensor_map_buffer = torch.empty((TENSOR_MAP_BYTES,), dtype=torch.uint8, device=a_base.device)
    maps = (
        make_tma_ab_desc(a_base, spec, shape_mn=M, shape_k=k_desc, block_mn=spec.block_m),
        make_tma_ab_desc(b_base, spec, shape_mn=N, shape_k=k_desc, block_mn=spec.block_n),
        make_tma_sf_desc(data["sfa"], shape_mn=M, num_sf_k=num_sf_k, block_mn=spec.block_m),
        make_tma_sf_desc(data["sfb"], shape_mn=N, num_sf_k=num_sf_k, block_mn=spec.block_n),
        make_tma_cd_desc(d, spec, shape_m=M, shape_n=N, num_groups=groups),
    )
    a_flat, b_flat = a_base.reshape(-1), b_base.reshape(-1)
    layout_flat, buffer_flat = grouped_layout.reshape(-1), tensor_map_buffer.reshape(-1)
    keep = (maps, a_flat, b_flat, layout_flat, buffer_flat)

    def launch():
        _ = keep  # descriptors and flat views stay alive with the closure
        executable.mod(
            a_flat,
            b_flat,
            layout_flat,
            buffer_flat,
            M,
            N,
            data["K"],
            maps[0].ptr,
            maps[1].ptr,
            maps[2].ptr,
            maps[3].ptr,
            maps[4].ptr,
        )

    return launch

# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Operands and the DeepGEMM reference for the SM90 FP8 1D2D m-grouped GEMMs.

Data mirrors `tests/generators.py::generate_m_grouped_{contiguous,masked}` on SM90, where
`get_ue8m0_usage` is False: A is cast per token `(1, 128)` and B per block `(128, 128)`, both
with FP32 scales.  The scales are then put into the layout the kernel reads by DeepGEMM's own
`transform_sf_into_required_layout` (`csrc/apis/layout.hpp:41,45`: SFA becomes MN-major and
TMA-aligned, SFB stays K-major), so the TIRx kernel and DeepGEMM read byte-identical operands.

The arch-independent comparison and benchmark helpers are shared with the SM100 1D1D port.
"""

from __future__ import annotations

from .._sm100_fp8_fp4_gemm_1d1d.data import (
    _warm_up,
    assert_within_threshold,
    bench_against_deepgemm,
    calc_diff,
    masked_slice_diff,
    psum_slice_diff,
    require_deep_gemm,
)
from .._sm100_fp8_fp4_gemm_1d1d.spec import make_actual_ms
from .spec import BLOCK_K, MK_ALIGNMENT, align_up

__all__ = [
    "assert_within_threshold",
    "bench_against_deepgemm",
    "calc_diff",
    "deepgemm_launch_m_grouped_contiguous",
    "deepgemm_launch_m_grouped_masked",
    "masked_slice_diff",
    "prepare_m_grouped_contiguous",
    "prepare_m_grouped_contiguous_psum_from_masked",
    "prepare_m_grouped_masked",
    "psum_slice_diff",
    "require_sm90",
]

#: `QuantConfig()` defaults on SM90: `gran_k_a = gran_k_b = 128`.
GRAN_K = BLOCK_K
RECIPE_A = (1, GRAN_K)
RECIPE_B = (GRAN_K, GRAN_K)


def require_sm90() -> None:
    """Skip rather than fail when the host cannot run the kernel."""
    from unittest import SkipTest

    import torch

    if not torch.cuda.is_available():
        raise SkipTest("no CUDA device")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SkipTest(
            f"needs SM90, got sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )


def _quantize_a(x):
    from deep_gemm.utils.math import per_token_cast_to_fp8

    return per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=GRAN_K)


def _quantize_grouped_b(b_bf16):
    """`grouped_cast_fp8_fp4_with_major(..., use_block_cast_for_fp8=True)`, K-major."""
    import torch
    from deep_gemm.utils.math import per_block_cast_to_fp8

    groups = [
        per_block_cast_to_fp8(b_bf16[i], use_ue8m0=False, gran_k=GRAN_K)
        for i in range(b_bf16.size(0))
    ]
    return torch.stack([g[0] for g in groups]), torch.stack([g[1] for g in groups])


def _transform_sf(sf, mn: int, k: int, recipe, num_groups: int | None = None):
    deep_gemm = require_deep_gemm()
    if num_groups is None:
        return deep_gemm.transform_sf_into_required_layout(sf, mn, k, recipe)
    return deep_gemm.transform_sf_into_required_layout(sf, mn, k, recipe, num_groups)


def prepare_m_grouped_contiguous(
    *,
    num_groups: int,
    expected_m_per_group: int,
    N: int,
    K: int,
    use_psum_layout: bool = False,
    seed: int = 0,
) -> dict:
    """Mirror `generators.generate_m_grouped_contiguous` (K-major A and B, FP8 x FP8)."""
    import torch

    require_sm90()
    torch.manual_seed(seed)
    actual_ms = make_actual_ms(num_groups, expected_m_per_group, seed)
    aligned_ms = [align_up(m, MK_ALIGNMENT) for m in actual_ms]
    M = sum(aligned_ms)

    a_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((num_groups, N, K), device="cuda", dtype=torch.bfloat16)
    grouped_layout = torch.empty(
        num_groups if use_psum_layout else M, device="cuda", dtype=torch.int32
    )
    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end, aligned_end = start + actual_m, start + aligned_m
        if use_psum_layout:
            grouped_layout[i] = actual_end
        else:
            grouped_layout[start:actual_end] = i
            grouped_layout[actual_end:aligned_end] = -1
        # The source zeroes BF16 padding so the quantized SFA padding is regular.
        a_bf16[actual_end:aligned_end] = 0
        start = aligned_end

    a_q, a_sf = _quantize_a(a_bf16)
    b_q, b_sf = _quantize_grouped_b(b_bf16)
    return {
        "M": M,
        "N": N,
        "K": K,
        "num_groups": num_groups,
        "a": a_q,
        "b": b_q,
        "sfa": _transform_sf(a_sf, M, K, RECIPE_A),
        "sfb": _transform_sf(b_sf, N, K, RECIPE_B, num_groups),
        "grouped_layout": grouped_layout,
        "d": torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
        "alignment": MK_ALIGNMENT,
        "use_psum_layout": use_psum_layout,
        "ensure_zero_padding": False,
    }


def prepare_m_grouped_contiguous_psum_from_masked(
    *, num_groups: int, expected_m_per_group: int, N: int, K: int, max_m: int = 4096, seed: int = 0
) -> dict:
    """The PSUM half of `test_m_grouped_gemm_masked`: masked operands packed by
    `generators.layout_masked_to_psum` into one `[num_groups * max_m, K]` PSUM-layout A.

    Group `j` occupies rows `[align(psum_m[j-1], 128), psum_m[j])`.  The source leaves the gap
    rows uninitialized (to test SM100's scale-packing kernel); they are zeroed here so a config is
    deterministic -- only valid rows are compared, and both kernels read the same bytes.
    """
    import torch

    require_sm90()
    torch.manual_seed(seed)
    a_bf16 = torch.randn((num_groups, max_m, K), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((num_groups, N, K), device="cuda", dtype=torch.bfloat16)
    counts = make_actual_ms(num_groups, expected_m_per_group, seed)
    if max(counts) > max_m:
        raise ValueError(f"masked_m {max(counts)} exceeds max_m {max_m}")
    psum = []
    for j, rows in enumerate(counts):
        psum.append((0 if j == 0 else align_up(psum[-1], MK_ALIGNMENT)) + rows)
    a_q, a_sf = (
        t.view(num_groups, max_m, -1) for t in _quantize_a(a_bf16.view(num_groups * max_m, K))
    )
    a_q_psum = torch.zeros((num_groups * max_m, K), device="cuda", dtype=a_q.dtype)
    a_sf_psum = torch.zeros((num_groups * max_m, a_sf.size(-1)), device="cuda", dtype=a_sf.dtype)
    last = 0
    for j, end in enumerate(psum):
        a_q_psum[last:end] = a_q[j, : end - last]
        a_sf_psum[last:end] = a_sf[j, : end - last]
        last = align_up(end, MK_ALIGNMENT)
    b_q, b_sf = _quantize_grouped_b(b_bf16)
    M = num_groups * max_m
    return {
        "M": M,
        "N": N,
        "K": K,
        "num_groups": num_groups,
        "a": a_q_psum,
        "b": b_q,
        "sfa": _transform_sf(a_sf_psum, M, K, RECIPE_A),
        "sfb": _transform_sf(b_sf, N, K, RECIPE_B, num_groups),
        "grouped_layout": torch.tensor(psum, device="cuda", dtype=torch.int32),
        "d": torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
        "alignment": MK_ALIGNMENT,
        "use_psum_layout": True,
        "ensure_zero_padding": False,
        # The source passes `expected_m_for_psum_layout = int(expected_m_per_group * 1.2)`.
        "expected_m_for_psum_layout": int(expected_m_per_group * 1.2),
    }


def prepare_m_grouped_masked(
    *, num_groups: int, expected_m_per_group: int, N: int, K: int, max_m: int = 4096, seed: int = 0
) -> dict:
    """Mirror `generators.generate_m_grouped_masked` (no psum layout)."""
    import torch

    require_sm90()
    torch.manual_seed(seed)
    a_bf16 = torch.randn((num_groups, max_m, K), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn((num_groups, N, K), device="cuda", dtype=torch.bfloat16)
    counts = make_actual_ms(num_groups, expected_m_per_group, seed)
    if max(counts) > max_m:
        raise ValueError(f"masked_m {max(counts)} exceeds max_m {max_m}")
    masked_m = torch.tensor(counts, device="cuda", dtype=torch.int32)

    a_q, a_sf = (
        t.view(num_groups, max_m, -1) for t in _quantize_a(a_bf16.view(num_groups * max_m, K))
    )
    # The source zeroes SFA rows beyond `masked_m` so the layout pass reads regular zeros.
    for j, rows in enumerate(counts):
        a_sf[j, rows:] = 0
    b_q, b_sf = _quantize_grouped_b(b_bf16)
    return {
        "M": max_m,
        "N": N,
        "K": K,
        "num_groups": num_groups,
        # The source test's `expected_m` argument: `int(expected_m_per_group * 1.2)`.
        "expected_m": int(expected_m_per_group * 1.2),
        "a": a_q,
        "b": b_q,
        "sfa": _transform_sf(a_sf, max_m, K, RECIPE_A, num_groups),
        "sfb": _transform_sf(b_sf, N, K, RECIPE_B, num_groups),
        "masked_m": masked_m,
        "grouped_layout": masked_m,
        "d": torch.empty((num_groups, max_m, N), device="cuda", dtype=torch.bfloat16),
        "alignment": MK_ALIGNMENT,
    }


def deepgemm_launch_m_grouped_contiguous(data, *, out=None):
    """`deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous` on the same operands."""
    import torch

    deep_gemm = require_deep_gemm()
    deep_gemm.set_mk_alignment_for_contiguous_layout(data["alignment"])
    out = torch.empty_like(data["d"]) if out is None else out
    a, sfa, b, sfb = data["a"], data["sfa"], data["b"], data["sfb"]
    layout, use_psum = data["grouped_layout"], data["use_psum_layout"]
    expected = data.get("expected_m_for_psum_layout")
    extra = {} if expected is None else {"expected_m_for_psum_layout": expected}

    def launch():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a, sfa), (b, sfb), out, layout, use_psum_layout=use_psum, **extra
        )

    _warm_up(launch)
    return launch, out


def deepgemm_launch_m_grouped_masked(data, *, out=None):
    """`deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked` on the same operands."""
    import torch

    deep_gemm = require_deep_gemm()
    deep_gemm.set_mk_alignment_for_contiguous_layout(data["alignment"])
    out = torch.empty_like(data["d"]) if out is None else out
    a, sfa, b, sfb = data["a"], data["sfa"], data["b"], data["sfb"]
    masked_m, expected_m = data["masked_m"], data["expected_m"]

    def launch():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked((a, sfa), (b, sfb), out, masked_m, expected_m)

    _warm_up(launch)
    return launch, out

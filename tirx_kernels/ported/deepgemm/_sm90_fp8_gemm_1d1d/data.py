# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Operands and the DeepGEMM reference for the SM90 FP8 1D1D GEMMs.

Data mirrors `tests/generators.py` on SM90, where `get_ue8m0_usage` is False and the 1D1D
recipe is `(1, 1, 128)`: both operands are cast per channel (k-grouped) or per token (normal)
with one FP32 scale per 128 K elements.  The scales are put into the layout the kernel reads by
DeepGEMM's own `get_mn_major_tma_aligned_tensor` -- what `transform_sf_into_required_layout`
and `transform_k_grouped_sf_into_required_layout` call on SM90 (`csrc/apis/layout.hpp:52,109`)
-- so the TIRx kernel and DeepGEMM read byte-identical operands.

Both kernels accumulate into D in place (`D += A @ B^T`, D starting as C).  D is a view into a
larger buffer whose guard bands hold a NaN sentinel, so the bitwise comparison of the whole
buffer also checks that neither kernel writes outside D.
"""

from __future__ import annotations

from .._sm100_fp8_fp4_gemm_1d1d.data import (
    bench_against_deepgemm,
    calc_diff,
    require_deep_gemm,
)
from .._sm100_fp8_fp4_gemm_1d1d.spec import make_actual_ks
from .spec import BLOCK_K, align_up

__all__ = [
    "assert_buffer_identical",
    "bench_against_deepgemm",
    "calc_diff",
    "deepgemm_launch_k_grouped",
    "deepgemm_launch_normal",
    "guarded_like",
    "make_ks",
    "prepare_k_grouped",
    "prepare_normal",
    "require_sm90",
]

#: `recipe = (1, 1, 128)`: the only one the SM90 1D1D kernel takes (`gemm.hpp:358`).
GRAN_K = BLOCK_K
RECIPE = (1, 1, GRAN_K)
#: A quiet-NaN FP32 bit pattern for the guard bands around D.
_SENTINEL_BITS = 0x7FC00A5A
#: Guard elements on each side of D (a multiple of 32 keeps D 128-byte aligned).
_GUARD = 4096


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


def guarded_like(c):
    """`(buffer, view)`: a sentinel-filled FP32 buffer and a contiguous view of `c`'s shape in it,
    holding a copy of `c`."""
    import torch

    numel = c.numel()
    buffer = torch.empty(numel + 2 * _GUARD, device=c.device, dtype=torch.float32)
    buffer.view(torch.int32).fill_(_SENTINEL_BITS)
    view = buffer[_GUARD : _GUARD + numel].view(c.shape)
    view.copy_(c)
    return buffer, view


def assert_buffer_identical(actual, expected, *, kernel: str, detail: str) -> None:
    """Every word of the guarded buffers -- D and both guard bands -- must match bitwise."""
    import torch

    differ = actual.view(torch.int32) != expected.view(torch.int32)
    if bool(differ.any()):
        idx = differ.nonzero()
        first = int(idx[0])
        where = "guard" if first < _GUARD or first >= actual.numel() - _GUARD else "D"
        raise AssertionError(
            f"{kernel} {detail}: {int(differ.sum())} of {differ.numel()} words differ bitwise "
            f"from DeepGEMM (first at flat {first - _GUARD} of D, in {where})"
        )


def make_ks(*, num_groups: int, expected_k_per_group: int, seed: int, group_ks=None):
    """Per-group real and 128-aligned K lengths (`generators.enumerate_k_grouped_contiguous`)."""
    real_ks = (
        list(group_ks)
        if group_ks is not None
        else make_actual_ks(num_groups, expected_k_per_group, seed)
    )
    return real_ks, [align_up(k, GRAN_K) for k in real_ks]


def _k_grouped_cast(x, ks: list[int]):
    """`generators.k_grouped_per_channel_cast_to_fp8` (`use_ue8m0 = False`, gran_k 128)."""
    import itertools

    import torch
    from deep_gemm.utils.math import align, per_channel_cast_to_fp8

    group_ends = list(itertools.accumulate(ks))
    n = x.size(1)
    x_fp8 = torch.zeros(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    sf_groups = []
    for k, end in zip(ks, group_ends):
        if k == 0:
            continue
        start = end - k
        x_group = torch.zeros((align(k, GRAN_K), n), dtype=x.dtype, device=x.device)
        x_group[:k] = x[start:end]
        x_group_fp8, x_group_sf = per_channel_cast_to_fp8(x_group, use_ue8m0=False, gran_k=GRAN_K)
        x_fp8[start:end] = x_group_fp8[:k]
        sf_groups.append(x_group_sf)
    sf = (
        torch.cat(sf_groups) if sf_groups else torch.empty((0, n), dtype=torch.float, device=x.device)
    )
    return x_fp8, sf


def prepare_k_grouped(
    *,
    num_groups: int,
    M: int,
    N: int,
    expected_k_per_group: int,
    seed: int = 0,
    group_ks=None,
) -> dict:
    """`generators.generate_k_grouped_contiguous` on SM90 (K-major A/B, FP32 D = C + A^T B).

    The SM90 entry requires every group's K to be a multiple of 128 (`sm90_fp8_gemm_1d1d.hpp:162`);
    the test feeds it the aligned lengths, with the real data zero-padded up to them.
    """
    import torch

    require_sm90()
    deep_gemm = require_deep_gemm()
    torch.manual_seed(seed)
    real_ks, aligned_ks = make_ks(
        num_groups=num_groups,
        expected_k_per_group=expected_k_per_group,
        seed=seed,
        group_ks=group_ks,
    )
    ks = aligned_ks
    total_k = sum(ks)
    layout = torch.tensor(ks, device="cuda", dtype=torch.int32)

    a = torch.randn((total_k, M), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((total_k, N), device="cuda", dtype=torch.bfloat16)
    # Zero the padding between each group's real and aligned K.
    start = 0
    for real, aligned in zip(real_ks, aligned_ks):
        a[start + real : start + aligned] = 0
        b[start + real : start + aligned] = 0
        start += aligned
    c = torch.randn((num_groups, M, N), device="cuda", dtype=torch.float) * 32

    a_fp8, sfa = _k_grouped_cast(a, ks)
    b_fp8, sfb = _k_grouped_cast(b, ks)
    # "Transpose for K Major A/B": group g is stored `[M, K_g]` K-major at offset `k_cumsum * M`.
    a_flat = torch.empty((total_k * M,), dtype=a_fp8.dtype, device="cuda")
    b_flat = torch.empty((total_k * N,), dtype=b_fp8.dtype, device="cuda")
    prefix = 0
    for k in ks:
        a_flat[prefix * M : (prefix + k) * M] = a_fp8[prefix : prefix + k].T.flatten()
        b_flat[prefix * N : (prefix + k) * N] = b_fp8[prefix : prefix + k].T.flatten()
        prefix += k
    sfa_t, sfb_t = sfa.T, sfb.T  # `(new_a, sfa.T), (new_b, sfb.T)`
    return {
        "kind": "k_grouped",
        "num_groups": num_groups,
        "M": M,
        "N": N,
        "K": total_k,
        "ks": ks,
        "real_ks": real_ks,
        "a": a_flat,
        "b": b_flat,
        "sfa_raw": sfa_t,
        "sfb_raw": sfb_t,
        # `transform_k_grouped_sf_into_required_layout` on SM90 (FP32).
        "sfa": deep_gemm.get_mn_major_tma_aligned_tensor(sfa_t),
        "sfb": deep_gemm.get_mn_major_tma_aligned_tensor(sfb_t),
        "grouped_layout": layout,
        "c": c,
    }


def prepare_normal(*, M: int, N: int, K: int, seed: int = 0) -> dict:
    """`generators.generate_normal(..., accumulate=True, out_dtype=float, Kernel1D1D)` on SM90:
    the wgrad case, both operands cast per token."""
    import torch
    from deep_gemm.utils.math import per_token_cast_to_fp8

    require_sm90()
    deep_gemm = require_deep_gemm()
    torch.manual_seed(seed)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    c = torch.randn((M, N), device="cuda", dtype=torch.float) * 32
    a_fp8, sfa = per_token_cast_to_fp8(a, use_ue8m0=False, gran_k=GRAN_K)
    b_fp8, sfb = per_token_cast_to_fp8(b, use_ue8m0=False, gran_k=GRAN_K)
    return {
        "kind": "normal",
        "M": M,
        "N": N,
        "K": K,
        "a": a_fp8,
        "b": b_fp8,
        "sfa_raw": sfa,
        "sfb_raw": sfb,
        "sfa": deep_gemm.get_mn_major_tma_aligned_tensor(sfa),
        "sfb": deep_gemm.get_mn_major_tma_aligned_tensor(sfb),
        "c": c,
    }


def deepgemm_launch_k_grouped(data, *, out=None):
    """`deep_gemm.k_grouped_fp8_gemm_nt_contiguous` (the SM90 k-grouped entry), D = C in place."""
    deep_gemm = require_deep_gemm()
    buffer, d = guarded_like(data["c"]) if out is None else out
    a = (data["a"], data["sfa_raw"])
    b = (data["b"], data["sfb_raw"])
    ks, layout = data["ks"], data["grouped_layout"]

    def launch():
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, layout, d, recipe=RECIPE)

    _warm_up_copy(launch, d, data["c"])
    return launch, (buffer, d)


def deepgemm_launch_normal(data, *, out=None, raw_scales=False):
    """`deep_gemm.fp8_fp4_gemm_nt(..., c=d, recipe=(1, 1, 128))`: the SM90 1D1D dispatch
    (`gemm.hpp:111`, `gran_n == 1`).

    By default the scales are passed already in the layout the kernel reads (`data["sfa"/"sfb"]`),
    which `transform_sf_pair_into_required_layout` returns unchanged, so each call launches only the
    GEMM kernel, as the TIRx launch does.  `raw_scales=True` passes the per-token K-major scales, for
    which the entry runs a `transpose_fp32` layout kernel per call (`gemm.hpp:106`): the same result,
    but not a like-for-like timing."""
    deep_gemm = require_deep_gemm()
    buffer, d = guarded_like(data["c"]) if out is None else out
    which = "_raw" if raw_scales else ""
    a = (data["a"], data["sfa" + which])
    b = (data["b"], data["sfb" + which])

    def launch():
        deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=d, recipe=RECIPE)

    _warm_up_copy(launch, d, data["c"])
    return launch, (buffer, d)


def _warm_up_copy(launch, d, c) -> None:
    """Run once (JIT build and descriptor caches), then restore D = C."""
    import torch

    launch()
    torch.cuda.synchronize()
    d.copy_(c)

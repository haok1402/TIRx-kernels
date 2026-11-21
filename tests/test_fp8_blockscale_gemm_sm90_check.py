# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""fp8_blockscale_gemm_sm90.check_output: accepts correct outputs, rejects what calc_diff accepts.

No kernel is compiled: the correctly rounded (float64 -> bf16) reference stands in for a correct
output, and the cases corrupt it. Needs a CUDA device for the fp64 reference only (any arch).
"""

from __future__ import annotations

import pytest
import torch

from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

SHAPES = [(300, 520, 144), (1000, 2112, 1536)]  # small K (the tightest bound) and a projection


@pytest.fixture(scope="module", params=SHAPES, ids=lambda s: "x".join(map(str, s)))
def case(request):
    torch.manual_seed(0)
    a8, sfa, b8, sfb, _ = g.prepare_data(*request.param)
    good = g.reference(a8, sfa, b8, sfb)  # correctly rounded bf16
    return a8, sfa, b8, sfb, good


def _next_ulp(x, steps=1):
    bits = x.view(torch.int16).clone()
    return (bits + steps).view(torch.bfloat16)


def test_correct_output_and_one_ulp_change_pass(case):
    a8, sfa, b8, sfb, good = case
    g.check_output(good, a8, sfa, b8, sfb, bitwise_deepgemm=False)
    off = good.clone()
    off[3, 5] = _next_ulp(off[3, 5])  # a correct kernel may round differently by 1 ulp
    g.check_output(off, a8, sfa, b8, sfb, bitwise_deepgemm=False)


def _copy_last_row(d, *_):
    d[-1] = d[-2]


def _drop_last_k_block_of_last_row(d, a8, sfa, b8, sfb):
    a = a8.clone()
    kb = sfa.shape[1]
    a.view(torch.uint8)[-1, (kb - 1) * 128 :] = 0  # e4m3 +0.0
    d[-1] = g.reference(a, sfa, b8, sfb)[-1]


def _scale_one_element(d, *_):
    d[7, 9] = d[7, 9] * 1.03 + 0.5


def _zero_last_column(d, *_):
    d[:, -1] = 0


def _one_nan(d, *_):
    d[2, 2] = float("nan")


def _one_inf(d, *_):
    d[4, 1] = float("inf")


@pytest.mark.parametrize(
    "corrupt",
    [
        _copy_last_row,
        _drop_last_k_block_of_last_row,
        _scale_one_element,
        _zero_last_column,
        _one_nan,
        _one_inf,
    ],
    ids=lambda f: f.__name__.strip("_"),
)
def test_corruptions_are_rejected_element_wise(case, corrupt):
    a8, sfa, b8, sfb, good = case
    bad = good.clone()
    corrupt(bad, a8, sfa, b8, sfb)
    # Rejected by the element-wise bound, NaN and Inf included. (On the larger shape every finite
    # case also passes DeepGEMM's calc_diff < 1e-3, which is why the element-wise check exists.)
    finite = bool(torch.isfinite(bad).all())
    if finite and bad.numel() > 10**6:
        assert g.calc_diff(bad, g.reference(a8, sfa, b8, sfb, out_dtype=torch.float64)) < 1e-3
    with pytest.raises(AssertionError, match="out of tolerance"):
        g.check_output(bad, a8, sfa, b8, sfb, bitwise_deepgemm=False)


def test_bitwise_deepgemm_is_opt_in(case, monkeypatch):
    a8, sfa, b8, sfb, good = case
    monkeypatch.delenv(g.BITWISE_DEEPGEMM_ENV, raising=False)
    g.check_output(good, a8, sfa, b8, sfb)  # default: no DeepGEMM needed, no bitwise check
    pytest.importorskip("deep_gemm")
    # The correctly rounded result is not DeepGEMM's bit pattern everywhere (accumulation order).
    with pytest.raises(AssertionError, match="differ bitwise from DeepGEMM"):
        g.check_output(good, a8, sfa, b8, sfb, bitwise_deepgemm=True)

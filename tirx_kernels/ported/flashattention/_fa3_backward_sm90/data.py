# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ 0251105a), Copyright (c) 2024, Jay Shah,
# Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Inputs, the flash_attn_3 reference, correctness checks and the benchmark for the SM90 backward.

Inputs follow ``hopper/test_flash_attn.py::test_flash_attn_output``: q, k, v ~ N(0, 1) in bf16 (BSHD),
O and LSE from flash_attn_3's own forward, dO ~ N(0, 1). The backward reference is
``torch.ops.flash_attn_3.bwd`` (``mha_bwd``), which returns ``softmax_d`` (dPsum), ``softmax_lse_log2``
and ``dq_accum`` besides writing dq, dk, dv; every one of those is compared.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

#: A quiet-NaN bit pattern every TIRx output and workspace starts from.
NAN_BF16 = 0x7FC0
NAN_F32 = 0x7FC00000


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def block_m(head_dim: int, is_causal: bool) -> int:
    """``kBlockM_sm90`` of ``mha_bwd`` (no softcap, no local window)."""
    if head_dim <= 64:
        return 128
    if head_dim <= 128:
        return 64 if is_causal else 80
    return 64


def block_n(head_dim: int) -> int:
    """``kBlockN_sm90`` of ``mha_bwd``."""
    return 128 if head_dim <= 128 else (96 if head_dim <= 192 else 80)


def require_sm90() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise unittest.SkipTest("flash_attention3_backward_sm90 needs an sm_90 GPU")


def import_fa3():
    import flash_attn_interface  # noqa: F401  (registers torch.ops.flash_attn_3)
    import torch

    return torch.ops.flash_attn_3


def prepare(
    *, batch_size, seqlen_q, seqlen_k, num_heads, head_dim, is_causal, num_heads_k=None, seed=0, **_
) -> dict[str, Any]:
    """Inputs (q, k, v, dO), the forward's O and LSE (flash_attn_3), and the workspace shapes."""
    import torch

    require_sm90()
    fa3 = import_fa3()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    b, h, d = batch_size, num_heads, head_dim
    h_k = h if num_heads_k is None else num_heads_k
    q = torch.randn(b, seqlen_q, h, d, device=dev, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(b, seqlen_k, h_k, d, device=dev, dtype=torch.bfloat16, generator=gen)
    v = torch.randn(b, seqlen_k, h_k, d, device=dev, dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(b, seqlen_q, h, d, device=dev, dtype=torch.bfloat16, generator=gen)
    scale = 1.0 / math.sqrt(d)
    out, lse, *_ = fa3.fwd(
        q, k, v, softmax_scale=scale, is_causal=is_causal, window_size_left=-1,
        window_size_right=0 if is_causal else -1, num_splits=1,
    )
    bm, bn = block_m(d, is_causal), block_n(d)
    return {
        "q": q, "k": k, "v": v, "out": out, "lse": lse, "dout": dout,
        "softmax_scale": scale, "is_causal": is_causal,
        "b": b, "s_q": seqlen_q, "s_k": seqlen_k, "h": h, "h_k": h_k, "gqa": h_k != h, "d": d,
        "block_m": bm, "block_n": bn,
        "s_q_rounded": _cdiv(seqlen_q, bm) * bm, "s_k_rounded": _cdiv(seqlen_k, bn) * bn,
    }


def run_fa3(p, dq=None, dk=None, dv=None):
    """``torch.ops.flash_attn_3.bwd``: (dq, dk, dv, softmax_d, softmax_lse_log2, dq_accum, dk_accum,
    dv_accum); the last two are None unless GQA."""
    import torch

    fa3 = import_fa3()
    dq = torch.empty_like(p["q"]) if dq is None else dq
    dk = torch.empty_like(p["k"]) if dk is None else dk
    dv = torch.empty_like(p["v"]) if dv is None else dv
    softmax_d, lse_log2, dq_accum, dk_accum, dv_accum = fa3.bwd(
        p["dout"], p["q"], p["k"], p["v"], p["out"], p["lse"], dq, dk, dv,
        softmax_scale=p["softmax_scale"], is_causal=p["is_causal"], window_size_left=-1,
        window_size_right=0 if p["is_causal"] else -1, softcap=0.0, deterministic=False, sm_margin=0,
    )
    return dq, dk, dv, softmax_d, lse_log2, dq_accum, dk_accum, dv_accum


def alloc_outputs(p) -> dict[str, Any]:
    """Our outputs and workspaces, every word a NaN sentinel."""
    import torch

    def nan_like(shape, dtype):
        if dtype == torch.bfloat16:
            return torch.full(shape, NAN_BF16, dtype=torch.int16, device="cuda").view(torch.bfloat16)
        return torch.full(shape, NAN_F32, dtype=torch.int32, device="cuda").view(torch.float32)

    b, h, d = p["b"], p["h"], p["d"]
    gqa_ws = {}
    if p["gqa"]:  # zero-filled by the launch (mha_bwd: torch fill_), NaN until then
        gqa_ws = {
            "dk_accum": nan_like((b, p["h_k"], p["s_k_rounded"] * d), torch.float32),
            "dv_accum": nan_like((b, p["h_k"], p["s_k_rounded"] * d), torch.float32),
        }
    return {
        **gqa_ws,
        "dq": nan_like(p["q"].shape, torch.bfloat16),
        "dk": nan_like(p["k"].shape, torch.bfloat16),
        "dv": nan_like(p["v"].shape, torch.bfloat16),
        "softmax_d": nan_like((b, h, p["s_q_rounded"]), torch.float32),
        "lse_log2": nan_like((b, h, p["s_q_rounded"]), torch.float32),
        "dq_accum": nan_like((b, h, p["s_q_rounded"] * d), torch.float32),
        "dq_semaphore": torch.full(
            (_cdiv(p["s_q"], p["block_m"]), b, h), -1, dtype=torch.int32, device="cuda"
        ),
    }


def _bitwise_mismatch(actual, expected) -> int:
    import torch

    if actual.dtype in (torch.bfloat16, torch.float16):
        a, e = actual.view(torch.int16), expected.view(torch.int16)
    else:
        a, e = actual.view(torch.int32), expected.view(torch.int32)
    return int((a != e).sum().item())


def max_kv_contributors(p) -> int:
    """An upper bound on the KV blocks whose dQ partials reach one Q row block: all of them (the
    causal diagonal can only lower it, which keeps this conservative). The cross-CTA bulk reduce-add
    order matters only from three on (0 + a + b == 0 + b + a in fp32)."""
    return _cdiv(p["s_k"], p["block_n"])


def check_case(config, *, backend_config=None) -> dict[str, Any]:
    """Bitwise against flash_attn_3, whole tensors from a NaN sentinel: dK, dV, dPsum, LSE*log2e; the dQ
    convert alone (upstream's dq_accum through our postprocess vs upstream's dq); dq and dq_accum under
    K-isolation (every shape; see ``_check_k_isolation``); dq itself when at most two KV blocks feed
    a row. Tolerance (upstream's test bound, against PyTorch references): dQ always."""
    import torch

    from .launch import compile_kernels, launch_backward, launch_postprocess, launch_postprocess_dkv

    p = prepare(**config)
    ex = compile_kernels(p["d"], p["is_causal"], gqa=p["gqa"], backend_config=backend_config)
    ours = alloc_outputs(p)
    launch_backward(ex, p, ours)
    ref_dq, ref_dk, ref_dv, ref_d, ref_lse2, ref_acc, ref_dkacc, ref_dvacc = run_fa3(p)
    torch.cuda.synchronize()

    report = {}
    checks = {
        "dk": (ours["dk"], ref_dk),
        "dv": (ours["dv"], ref_dv),
        # Rows >= s_q of these workspaces are written by preprocess (0 / +inf), so compare all.
        "softmax_d": (ours["softmax_d"], ref_d),
        "lse_log2": (ours["lse_log2"], ref_lse2),
    }
    for name, (a, e) in checks.items():
        report[name] = _bitwise_mismatch(a, e)
    report["dq_bitwise"] = _bitwise_mismatch(ours["dq"], ref_dq)
    report["kv_contributors"] = max_kv_contributors(p)
    group = p["h"] // p["h_k"]
    report["q_heads_per_kv_head"] = group

    # The dQ convert alone: our postprocess on upstream's accumulator must give upstream's dq.
    dq_conv = alloc_outputs(p)["dq"]
    launch_postprocess(ex, p, ref_acc, dq_conv)
    if p["gqa"]:  # and the dK / dV converts on upstream's fp32 accumulators
        dk_conv, dv_conv = alloc_outputs(p)["dk"], alloc_outputs(p)["dv"]
        launch_postprocess_dkv(ex, p, ref_dkacc, dk_conv, ref_dvacc, dv_conv)
    torch.cuda.synchronize()
    report["dq_convert"] = _bitwise_mismatch(dq_conv, ref_dq)
    must = ["softmax_d", "lse_log2", "dq_convert"]
    if p["gqa"]:
        report["dk_convert"] = _bitwise_mismatch(dk_conv, ref_dk)
        report["dv_convert"] = _bitwise_mismatch(dv_conv, ref_dv)
        must += ["dk_convert", "dv_convert"]
    # dK / dV: per-CTA results (MHA), or sums of `group` fp32 bulk reduce-adds (GQA), which are order
    # independent for at most two addends.
    if group <= 2:
        must += ["dk", "dv"]
    report.update(_check_k_isolation(ex, p))
    must += ["dq_iso", "dq_accum_iso"]
    if group > 2:
        report.update(_check_head_isolation(ex, p))
        must += ["dk_hiso", "dv_hiso", "dk_accum_hiso", "dv_accum_hiso"]
    bad = [k for k in must if report[k]]
    if report["kv_contributors"] <= 2 and report["dq_bitwise"]:
        bad.append("dq_bitwise")
    if bad:
        raise AssertionError(f"bitwise mismatches vs flash_attn_3 ({bad}): {report}")
    for name in ("dq", "dk", "dv"):
        assert not torch.isnan(ours[name]).any(), f"{name} has unwritten (sentinel) elements: {report}"
    if p["b"] * p["h"] * p["s_q"] * p["s_k"] <= TOLERANCE_MAX_SCORES:
        report.update(_check_tolerance(p, ours, (ref_dq, ref_dk, ref_dv)))
    return report


def isolated_kv_blocks(p) -> list[int]:
    """The KV blocks K-isolation keeps: the first and the last (one block if there is only one)."""
    n_blocks = _cdiv(p["s_k"], p["block_n"])
    return sorted({0, n_blocks - 1})


def _check_k_isolation(ex, p) -> dict:
    """dQ bitwise on every shape (sm90-port-review's K-isolation check). The backward takes O and LSE
    as inputs, so with K zeroed outside at most two KV blocks (O, LSE, Q, V, dO unchanged) every other
    block's dQ partial is dS * 0 = +-0 exactly, and the at-most-two remaining partials commute in fp32:
    the cross-CTA reduce order cannot matter, and dq_accum and dq must be bitwise identical to
    flash_attn_3's on the same inputs."""
    import torch

    from .launch import launch_backward

    keep = torch.zeros(p["s_k"], dtype=torch.bool, device="cuda")
    for j in isolated_kv_blocks(p):
        keep[j * p["block_n"] : (j + 1) * p["block_n"]] = True
    p_iso = {key: val for key, val in p.items() if key != "_launch_args"}
    p_iso["k"] = (p["k"] * keep[None, :, None, None].to(p["k"].dtype)).contiguous()
    ours = alloc_outputs(p_iso)
    launch_backward(ex, p_iso, ours)
    ref = run_fa3(p_iso)
    torch.cuda.synchronize()
    return {
        "dq_iso": _bitwise_mismatch(ours["dq"], ref[0]),
        "dq_accum_iso": _bitwise_mismatch(ours["dq_accum"], ref[5]),
        "iso_blocks": isolated_kv_blocks(p),
    }


def _check_head_isolation(ex, p) -> dict:
    """dK / dV bitwise for GQA groups of more than two query heads: dO zeroed for every query head but
    the first and last of each group (O, LSE, Q, K, V unchanged) makes the other heads' dK / dV
    partials exactly 0 (dP = V dO^T = 0 and dPsum = rowsum(dO * O) = 0, so dS = 0 and P^T dO = 0); the
    two remaining fp32 bulk reduce-adds commute, so dk_accum / dv_accum and dK / dV must be bitwise."""
    import torch

    from .launch import launch_backward

    group = p["h"] // p["h_k"]
    keep = torch.zeros(p["h"], dtype=torch.bool, device="cuda")
    for g0 in range(0, p["h"], group):
        keep[g0] = keep[g0 + group - 1] = True
    p_iso = {key: val for key, val in p.items() if key != "_launch_args"}
    p_iso["dout"] = (p["dout"] * keep[None, None, :, None].to(p["dout"].dtype)).contiguous()
    ours = alloc_outputs(p_iso)
    launch_backward(ex, p_iso, ours)
    ref = run_fa3(p_iso)
    torch.cuda.synchronize()
    return {
        "dk_hiso": _bitwise_mismatch(ours["dk"], ref[1]),
        "dv_hiso": _bitwise_mismatch(ours["dv"], ref[2]),
        "dk_accum_hiso": _bitwise_mismatch(ours["dk_accum"], ref[6]),
        "dv_accum_hiso": _bitwise_mismatch(ours["dv_accum"], ref[7]),
    }


def attention_ref(q, k, v, causal=False, upcast=True, reorder_ops=False):
    """``hopper/test_util.py::attention_ref`` for the cases here (MHA, no padding, no softcap, no
    dropout, window (-1, 0) when causal): same operations in the same order."""
    import torch

    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    g = q.shape[2] // k.shape[2]  # repeat(k, "b s h d -> b s (h g) d")
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    d = q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    local_mask = None
    if causal:  # construct_local_mask with window_size (-1, 0)
        row_idx = torch.arange(seqlen_q, device=q.device, dtype=torch.long)[:, None]
        col_idx = torch.arange(seqlen_k, device=q.device, dtype=torch.long)
        local_mask = col_idx > row_idx + seqlen_k - seqlen_q + 0
        scores.masked_fill_(local_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    if local_mask is not None:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    return output.to(dtype=dtype_og)


#: Above this many score elements the fp32 PyTorch reference does not fit next to the inputs on
#: one H200; such (benchmark) shapes are checked bitwise only.
TOLERANCE_MAX_SCORES = 1 << 31


def _check_tolerance(p, ours, refs) -> dict:
    """``test_flash_attn_output``'s dQ / dK / dV assertions, verbatim: with bf16 leaf tensors,
    |dx - dx_ref| <= 2 |dx_pt - dx_ref| + 2 |dx_ref + 0.3 - 0.3 - dx_ref| (rtol 2, no softcap), where
    dx_ref is the grad of the upcast reference and dx_pt that of the bf16 reorder_ops reference."""
    import torch

    q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_() for t in (p["q"], p["k"], p["v"]))
    g = p["dout"]
    out_ref = attention_ref(q_ref, k_ref, v_ref, causal=p["is_causal"])
    out_pt = attention_ref(q_ref, k_ref, v_ref, causal=p["is_causal"], upcast=False, reorder_ops=True)
    grads_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
    grads_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
    out = {}
    for name, ref, x_ref, x_pt in zip(("dq", "dk", "dv"), refs, grads_ref, grads_pt):
        atol = 2 * (x_ref + 0.3 - 0.3 - x_ref).abs().max().item()
        err = (ours[name] - x_ref).abs().max().item()
        ref_err = (ref - x_ref).abs().max().item()
        bound = 2 * (x_pt - x_ref).abs().max().item() + atol
        if not err <= bound:
            raise AssertionError(f"{name} error {err} > upstream bound {bound} (flash_attn_3: {ref_err})")
        out.update({f"{name}_err": err, f"{name}_bound": bound, f"fa3_{name}_err": ref_err})
    return out


def bench_against_fa3(
    config, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, backend_config=None
):
    """Kernel-only: our preprocess + main + postprocess against flash_attn_3's three kernels."""
    from tirx_kernels.runner import bench

    from .launch import compile_kernels, launch_backward

    p = prepare(**config)
    ex = compile_kernels(p["d"], p["is_causal"], gqa=p["gqa"], backend_config=backend_config)
    ours = alloc_outputs(p)

    def tirx():
        launch_backward(ex, p, ours)

    def build_reference():
        import torch

        bufs = [torch.empty_like(p[n]) for n in ("q", "k", "v")]
        return lambda: run_fa3(p, *bufs)

    return bench(
        {"tirx": tirx},
        references={"flash_attn_3": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )

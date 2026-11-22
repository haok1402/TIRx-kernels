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
    *, batch_size, seqlen_q, seqlen_k, num_heads, head_dim, is_causal, seed=0, **_
) -> dict[str, Any]:
    """Inputs (q, k, v, dO), the forward's O and LSE (flash_attn_3), and the workspace shapes."""
    import torch

    require_sm90()
    fa3 = import_fa3()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    b, h, d = batch_size, num_heads, head_dim
    q = torch.randn(b, seqlen_q, h, d, device=dev, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(b, seqlen_k, h, d, device=dev, dtype=torch.bfloat16, generator=gen)
    v = torch.randn(b, seqlen_k, h, d, device=dev, dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(b, seqlen_q, h, d, device=dev, dtype=torch.bfloat16, generator=gen)
    scale = 1.0 / math.sqrt(d)
    out, lse, *_ = fa3.fwd(
        q, k, v, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, scale, is_causal, -1, 0 if is_causal else -1, 0,
        0.0, True, None, 1, None, 0,
    )
    bm, bn = block_m(d, is_causal), block_n(d)
    return {
        "q": q, "k": k, "v": v, "out": out, "lse": lse, "dout": dout,
        "softmax_scale": scale, "is_causal": is_causal,
        "b": b, "s_q": seqlen_q, "s_k": seqlen_k, "h": h, "d": d,
        "block_m": bm, "block_n": bn,
        "s_q_rounded": _cdiv(seqlen_q, bm) * bm, "s_k_rounded": _cdiv(seqlen_k, bn) * bn,
    }


def run_fa3(p, dq=None, dk=None, dv=None):
    """``torch.ops.flash_attn_3.bwd``: (dq, dk, dv, softmax_d, softmax_lse_log2, dq_accum)."""
    import torch

    fa3 = import_fa3()
    dq = torch.empty_like(p["q"]) if dq is None else dq
    dk = torch.empty_like(p["k"]) if dk is None else dk
    dv = torch.empty_like(p["v"]) if dv is None else dv
    softmax_d, lse_log2, dq_accum, _, _ = fa3.bwd(
        p["dout"], p["q"], p["k"], p["v"], p["out"], p["lse"], dq, dk, dv,
        None, None, None, None, None, None, p["softmax_scale"], p["is_causal"],
        -1, 0 if p["is_causal"] else -1, 0.0, False, 0,
    )
    return dq, dk, dv, softmax_d, lse_log2, dq_accum


def alloc_outputs(p) -> dict[str, Any]:
    """Our outputs and workspaces, every word a NaN sentinel."""
    import torch

    def nan_like(shape, dtype):
        if dtype == torch.bfloat16:
            return torch.full(shape, NAN_BF16, dtype=torch.int16, device="cuda").view(torch.bfloat16)
        return torch.full(shape, NAN_F32, dtype=torch.int32, device="cuda").view(torch.float32)

    b, h, d = p["b"], p["h"], p["d"]
    return {
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
    """The most KV blocks whose dQ partials reach one Q row block (bulk reduce-add order matters
    only from three on: 0 + a + b == 0 + b + a in fp32)."""
    n_blocks = _cdiv(p["s_k"], p["block_n"])
    if not p["is_causal"]:
        return n_blocks
    # Causal (bottom-right aligned): the last Q row block sees every KV block up to its diagonal.
    return min(n_blocks, _cdiv(p["s_k"] - p["s_q"] + p["s_q"], p["block_n"]))


def check_case(config, *, backend_config=None) -> dict[str, Any]:
    """Bitwise: dK, dV, dPsum, LSE*log2e over the whole tensors (sentinel-initialised), dQ when at
    most two KV blocks feed a row, and the dQ convert alone (upstream's dq_accum through our
    postprocess vs upstream's dq). Tolerance (upstream's test bound, against an fp32 PyTorch
    reference): dQ always."""
    import torch

    from .launch import compile_kernels, launch_backward, launch_postprocess

    p = prepare(**config)
    ex = compile_kernels(p["d"], p["is_causal"], backend_config=backend_config)
    ours = alloc_outputs(p)
    launch_backward(ex, p, ours)
    ref_dq, ref_dk, ref_dv, ref_d, ref_lse2, ref_acc = run_fa3(p)
    torch.cuda.synchronize()

    report = {}
    s_q = p["s_q"]
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

    # The dQ convert alone: our postprocess on upstream's accumulator must give upstream's dq.
    dq_conv = alloc_outputs(p)["dq"]
    launch_postprocess(ex, p, ref_acc, dq_conv)
    torch.cuda.synchronize()
    report["dq_convert"] = _bitwise_mismatch(dq_conv, ref_dq)

    bad = [k for k in ("dk", "dv", "softmax_d", "lse_log2", "dq_convert") if report[k]]
    if report["kv_contributors"] <= 2 and report["dq_bitwise"]:
        bad.append("dq_bitwise")
    if bad:
        raise AssertionError(f"bitwise mismatches vs flash_attn_3: {report}")
    assert not torch.isnan(ours["dq"]).any(), "dq has unwritten (sentinel) elements"
    _check_dq_tolerance(p, ours["dq"], ref_dq)
    del s_q
    return report


def _check_dq_tolerance(p, dq, ref_dq) -> None:
    """``test_flash_attn_output``'s dQ bound: |dq - dq_ref| <= 2 |dq_pt - dq_ref| + dq_atol, with
    dq_ref from an fp32 PyTorch attention and dq_pt from the same in bf16 (reorder_ops)."""
    import torch

    q, k, v, dout = (t.float().requires_grad_() for t in (p["q"], p["k"], p["v"], p["dout"]))
    del dout
    g = p["dout"].float()

    def attn(qq, kk, vv, upcast):
        dt = torch.float32 if upcast else torch.bfloat16
        qq, kk, vv = qq.to(dt), kk.to(dt), vv.to(dt)
        s = torch.einsum("bthd,bshd->bhts", qq * p["softmax_scale"], kk)
        if p["is_causal"]:
            sq, sk = s.shape[-2], s.shape[-1]
            row = torch.arange(sq, device=s.device)[:, None]
            col = torch.arange(sk, device=s.device)[None, :]
            s = s.masked_fill(col > row + sk - sq, float("-inf"))
        a = torch.softmax(s.float(), dim=-1).to(dt)
        return torch.einsum("bhts,bshd->bthd", a, vv)

    out_ref = attn(q, k, v, True)
    (dq_ref,) = torch.autograd.grad(out_ref, (q,), g)
    out_pt = attn(q, k, v, False)
    (dq_pt,) = torch.autograd.grad(out_pt, (q,), g.to(torch.bfloat16))
    dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item()
    err = (dq.float() - dq_ref).abs().max().item()
    ref_err = (ref_dq.float() - dq_ref).abs().max().item()
    bound = 2 * (dq_pt.float() - dq_ref).abs().max().item() + dq_atol
    if not err <= bound:
        raise AssertionError(f"dq error {err} > upstream bound {bound} (flash_attn_3: {ref_err})")


def bench_against_fa3(
    config, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, backend_config=None
):
    """Kernel-only: our preprocess + main + postprocess against flash_attn_3's three kernels."""
    from tirx_kernels.runner import bench

    from .launch import compile_kernels, launch_backward

    p = prepare(**config)
    ex = compile_kernels(p["d"], p["is_causal"], backend_config=backend_config)
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

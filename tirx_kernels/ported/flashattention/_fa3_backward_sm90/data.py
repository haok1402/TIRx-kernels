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
    *, batch_size, seqlen_q, seqlen_k, num_heads, head_dim, is_causal, num_heads_k=None, seed=0,
    varlen=False, **extra,
) -> dict[str, Any]:
    """Inputs (q, k, v, dO), the forward's O and LSE (flash_attn_3), and the workspace shapes."""
    import torch

    if varlen:
        return prepare_varlen(
            batch_size=batch_size, seqlen_q=seqlen_q, seqlen_k=seqlen_k, num_heads=num_heads,
            head_dim=head_dim, is_causal=is_causal, num_heads_k=num_heads_k, seed=seed, **extra,
        )

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


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random", zero_lengths=False,
                                 generator=None):
    """``hopper/test_util.py::generate_random_padding_mask`` (with an explicit generator)."""
    import torch

    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(0 if zero_lengths else 1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device,
            generator=generator,
        )
    else:
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device,
                                generator=generator)
    if zero_lengths:  # zero lengths every 5 batches and the last batch
        for i in range(batch_size):
            if i % 5 == 0:
                lengths[i] = 0
        lengths[-1] = 0
    return torch.arange(max_seqlen, device=device)[None, :].expand(batch_size, -1) < lengths


def _unpad_input(x, attention_mask, unused_mask=None):
    """``hopper/padding.py::unpad_input``: (x_unpad, indices, cu_seqlens, max_seqlen, seqused)."""
    import torch
    import torch.nn.functional as F

    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens = all_masks.sum(dim=-1, dtype=torch.int32)
    used = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    cu = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return x.reshape(-1, *x.shape[2:])[indices], indices, cu, int(seqlens.max().item()), used


def _gen_unused_masks(padding_mask, add_unused, max_seq_len, bs, device, generator):
    """``test_flash_attn_varlen_output``'s ``_gen_unused_masks``."""
    import torch

    if add_unused:
        another_mask = generate_random_padding_mask(max_seq_len, bs, device, generator=generator)
        attn_mask = torch.logical_and(padding_mask, another_mask)
        unused_mask = torch.logical_xor(torch.logical_or(padding_mask, another_mask), attn_mask)
    else:
        attn_mask, unused_mask = padding_mask, None
    return attn_mask, unused_mask


def prepare_varlen(
    *, batch_size, seqlen_q, seqlen_k, num_heads, head_dim, is_causal, num_heads_k=None, seed=0,
    add_unused_qkv=False, seqused=True, zero_lengths_k=True, **_
) -> dict[str, Any]:
    """``test_flash_attn_varlen_output``'s inputs: padded q, k, v ~ N(0, 1) in bf16, random padding
    masks (query lengths >= 1, key lengths possibly 0 and 0 for batches 0, 5, ... and the last),
    optional unused tails (``add_unused_qkv``: seqused < the cu_seqlens lengths), unpadded as
    ``generate_qkv`` does; O and LSE from flash_attn_3's varlen forward, dO ~ N(0, 1) unpadded.
    ``generate_qkv`` always returns seqused; ``seqused=False`` drops it (cu_seqlens lengths only)."""
    import torch

    require_sm90()
    fa3 = import_fa3()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    b, h, d = batch_size, num_heads, head_dim
    h_k = h if num_heads_k is None else num_heads_k
    q_pad = torch.randn(b, seqlen_q, h, d, device=dev, dtype=torch.bfloat16, generator=gen)
    k_pad = torch.randn(b, seqlen_k, h_k, d, device=dev, dtype=torch.bfloat16, generator=gen)
    v_pad = torch.randn(b, seqlen_k, h_k, d, device=dev, dtype=torch.bfloat16, generator=gen)
    qmask = generate_random_padding_mask(seqlen_q, b, dev, "random", zero_lengths=False, generator=gen)
    kmask = generate_random_padding_mask(seqlen_k, b, dev, "random", zero_lengths=zero_lengths_k, generator=gen)
    qmask, q_unused = _gen_unused_masks(qmask, add_unused_qkv, seqlen_q, b, dev, gen)
    kmask, k_unused = _gen_unused_masks(kmask, add_unused_qkv, seqlen_k, b, dev, gen)
    q, idx_q, cu_q, max_q, used_q = _unpad_input(q_pad, qmask, q_unused)
    k, idx_k, cu_k, max_k, used_k = _unpad_input(k_pad, kmask, k_unused)
    v, *_unused = _unpad_input(v_pad, kmask, k_unused)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    seqused_q, seqused_k = (used_q, used_k) if seqused else (None, None)
    scale = 1.0 / math.sqrt(d)
    out, lse, *_ = fa3.fwd(
        q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, seqused_q=seqused_q, seqused_k=seqused_k,
        max_seqlen_q=max_q, max_seqlen_k=max_k, softmax_scale=scale, is_causal=is_causal,
        window_size_left=-1, window_size_right=0 if is_causal else -1, num_splits=1,
    )
    dout = torch.randn(out.shape, device=dev, dtype=torch.bfloat16, generator=gen)
    bm, bn = block_m(d, is_causal), block_n(d)
    total_q, total_k = q.shape[0], k.shape[0]
    return {
        "q": q, "k": k, "v": v, "out": out, "lse": lse, "dout": dout,
        "softmax_scale": scale, "is_causal": is_causal,
        # s_q / s_k are shape_Q[0] / shape_K[0] (the totals); the rounded sizes are mha_bwd's padded totals.
        "b": b, "s_q": total_q, "s_k": total_k, "h": h, "h_k": h_k, "gqa": h_k != h, "d": d,
        "block_m": bm, "block_n": bn,
        "s_q_rounded": _cdiv(total_q + b * bm, bm) * bm, "s_k_rounded": _cdiv(total_k + b * bn, bn) * bn,
        "varlen": True, "cu_seqlens_q": cu_q, "cu_seqlens_k": cu_k,
        "seqused_q": seqused_q, "seqused_k": seqused_k, "max_seqlen_q": max_q, "max_seqlen_k": max_k,
        # the padded views, for the tolerance check (upstream compares on the padded re-layout)
        "pad": {"q": q_pad, "k": k_pad, "v": v_pad, "qmask": qmask, "kmask": kmask, "q_unused": q_unused,
                "k_unused": k_unused, "idx_q": idx_q, "idx_k": idx_k},
    }


def seqlens(p, side):
    """Per-batch (offset, used length) of one side of a varlen case (SeqlenInfo's offset / seqlen)."""
    cu = p[f"cu_seqlens_{side}"].tolist()
    used = p[f"seqused_{side}"]
    used = used.tolist() if used is not None else [cu[i + 1] - cu[i] for i in range(p["b"])]
    return [(cu[i], used[i]) for i in range(p["b"])]


def written_rows_mask(p):
    """(h, s_q_rounded) bool: the dPsum / LSE_log2 rows the preprocess writes in a varlen case (every
    row of each of the ceil(seqlen / kBlockM) m-blocks at the padded offset); the rest of mha_bwd's
    new_empty workspaces is never written by either side."""
    import torch

    bm = p["block_m"]
    m = torch.zeros(p["s_q_rounded"], dtype=torch.bool)
    for bidb, (off, n) in enumerate(seqlens(p, "q")):
        offp = (off + bidb * bm) // bm * bm
        m[offp: offp + _cdiv(n, bm) * bm] = True
    return m.cuda()[None, :].expand(p["h"], -1)


def run_fa3(p, dq=None, dk=None, dv=None):
    """``torch.ops.flash_attn_3.bwd``: (dq, dk, dv, softmax_d, softmax_lse_log2, dq_accum, dk_accum,
    dv_accum); the last two are None unless GQA."""
    import torch

    fa3 = import_fa3()
    dq = torch.empty_like(p["q"]) if dq is None else dq
    dk = torch.empty_like(p["k"]) if dk is None else dk
    dv = torch.empty_like(p["v"]) if dv is None else dv
    vl = {}
    if p.get("varlen"):
        vl = dict(cu_seqlens_q=p["cu_seqlens_q"], cu_seqlens_k=p["cu_seqlens_k"], seqused_q=p["seqused_q"],
                  seqused_k=p["seqused_k"], max_seqlen_q=p["max_seqlen_q"], max_seqlen_k=p["max_seqlen_k"])
    softmax_d, lse_log2, dq_accum, dk_accum, dv_accum = fa3.bwd(
        p["dout"], p["q"], p["k"], p["v"], p["out"], p["lse"], dq, dk, dv, **vl,
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
    if p.get("varlen"):  # mha_bwd's varlen workspaces: (h, total_q_padded_rounded), batch folded in
        ws = {
            "dq": nan_like(p["q"].shape, torch.bfloat16),
            "dk": nan_like(p["k"].shape, torch.bfloat16),
            "dv": nan_like(p["v"].shape, torch.bfloat16),
            "softmax_d": nan_like((h, p["s_q_rounded"]), torch.float32),
            "lse_log2": nan_like((h, p["s_q_rounded"]), torch.float32),
            "dq_accum": nan_like((h, p["s_q_rounded"] * d), torch.float32),
            "dq_semaphore": torch.full(
                (_cdiv(p["max_seqlen_q"], p["block_m"]), b, h), -1, dtype=torch.int32, device="cuda"
            ),
        }
        if p["gqa"]:
            ws["dk_accum"] = nan_like((p["h_k"], p["s_k_rounded"], d), torch.float32)
            ws["dv_accum"] = nan_like((p["h_k"], p["s_k_rounded"], d), torch.float32)
        return ws
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
    if p.get("varlen"):
        return max([_cdiv(n, p["block_n"]) for _, n in seqlens(p, "k")] + [0])
    return _cdiv(p["s_k"], p["block_n"])


def check_case(config, *, backend_config=None) -> dict[str, Any]:
    """Bitwise against flash_attn_3, whole tensors from a NaN sentinel: dK, dV, dPsum, LSE*log2e; the dQ
    convert alone (upstream's dq_accum through our postprocess vs upstream's dq); dq and dq_accum under
    K-isolation (every shape; see ``_check_k_isolation``); dq itself when at most two KV blocks feed
    a row. Tolerance (upstream's test bound, against PyTorch references): dQ always."""
    import torch

    from .launch import compile_kernels, launch_backward, launch_postprocess, launch_postprocess_dkv

    p = prepare(**config)
    ex = compile_kernels(p["d"], p["is_causal"], gqa=p["gqa"], varlen=bool(p.get("varlen")),
                         backend_config=backend_config)
    ours = alloc_outputs(p)
    launch_backward(ex, p, ours)
    varlen = bool(p.get("varlen"))
    # varlen: rows outside every sequence's used length are written by neither side, so both start from
    # the NaN sentinel and compare bitwise as NaN patterns.
    ref_bufs = (alloc_outputs(p)[n] for n in ("dq", "dk", "dv")) if varlen else ()
    ref_dq, ref_dk, ref_dv, ref_d, ref_lse2, ref_acc, ref_dkacc, ref_dvacc = run_fa3(p, *ref_bufs)
    torch.cuda.synchronize()

    report = {}
    if p["q"].numel() == 0 or p["k"].numel() == 0:
        # mha_bwd's empty-side branch: no kernel runs, dq or dk / dv and softmax_d are zero_()ed, the
        # other workspaces stay unwritten (new_empty) on both sides.
        for name, a, e in (("dq", ours["dq"], ref_dq), ("dk", ours["dk"], ref_dk), ("dv", ours["dv"], ref_dv),
                           ("softmax_d", ours["softmax_d"], ref_d)):
            report[name] = _bitwise_mismatch(a, e)
        report["empty_side"] = True
        bad = [k for k in ("dq", "dk", "dv", "softmax_d") if report[k]]
        if bad:
            raise AssertionError(f"bitwise mismatches vs flash_attn_3 ({bad}): {report}")
        return report
    sel = written_rows_mask(p) if varlen else None  # mha_bwd's new_empty workspaces: written rows only
    checks = {
        "dk": (ours["dk"], ref_dk),
        "dv": (ours["dv"], ref_dv),
        # Rows >= s_q of these workspaces are written by preprocess (0 / +inf), so compare all.
        "softmax_d": (ours["softmax_d"], ref_d) if sel is None else (ours["softmax_d"][sel], ref_d[sel]),
        "lse_log2": (ours["lse_log2"], ref_lse2) if sel is None else (ours["lse_log2"][sel], ref_lse2[sel]),
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
        x = ours[name] if not varlen else ours[name][used_rows(p, "q" if name == "dq" else "k")]
        assert not torch.isnan(x).any(), f"{name} has unwritten (sentinel) elements: {report}"
    if varlen:
        pd = p["pad"]
        if p["b"] * p["h"] * pd["q"].shape[1] * pd["k"].shape[1] <= TOLERANCE_MAX_SCORES:
            report.update(_check_tolerance_varlen(p, ours, (ref_dq, ref_dk, ref_dv)))
    elif p["b"] * p["h"] * p["s_q"] * p["s_k"] <= TOLERANCE_MAX_SCORES:
        report.update(_check_tolerance(p, ours, (ref_dq, ref_dk, ref_dv)))
    return report


def used_rows(p, side):
    """(total,) bool: the rows of a varlen side inside some sequence's used length."""
    import torch

    m = torch.zeros(p[f"s_{side}"], dtype=torch.bool)
    for off, n in seqlens(p, side):
        m[off: off + n] = True
    return m.cuda()


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

    bn = p["block_n"]
    p_iso = {key: val for key, val in p.items() if key != "_launch_args"}
    if p.get("varlen"):  # per sequence: its own first and last KV block, from its cu_seqlens_k offset
        keep = torch.zeros(p["s_k"], dtype=torch.bool)
        for off, n in seqlens(p, "k"):
            nb = _cdiv(n, bn)
            for j in sorted({0, nb - 1}) if nb else ():
                keep[off + j * bn: off + min((j + 1) * bn, n)] = True
        keep = keep.cuda()
        p_iso["k"] = (p["k"] * keep[:, None, None].to(p["k"].dtype)).contiguous()
    else:
        keep = torch.zeros(p["s_k"], dtype=torch.bool, device="cuda")
        for j in isolated_kv_blocks(p):
            keep[j * bn : (j + 1) * bn] = True
        p_iso["k"] = (p["k"] * keep[None, :, None, None].to(p["k"].dtype)).contiguous()
    ours = alloc_outputs(p_iso)
    launch_backward(ex, p_iso, ours)
    ref = run_fa3(p_iso, alloc_outputs(p_iso)["dq"]) if p.get("varlen") else run_fa3(p_iso)
    torch.cuda.synchronize()
    if p.get("varlen"):
        sel = written_rows_mask(p)[:, :, None].expand(-1, -1, p["d"]).reshape(p["h"], -1)
        acc = _bitwise_mismatch(ours["dq_accum"][sel], ref[5][sel])
    else:
        acc = _bitwise_mismatch(ours["dq_accum"], ref[5])
    return {
        "dq_iso": _bitwise_mismatch(ours["dq"], ref[0]),
        "dq_accum_iso": acc,
        "iso_blocks": "first+last per sequence" if p.get("varlen") else isolated_kv_blocks(p),
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
    shape = [1] * (p["dout"].dim() - 2) + [p["h"], 1]  # (b, s, h, d) or varlen (total, h, d)
    p_iso["dout"] = (p["dout"] * keep.view(shape).to(p["dout"].dtype)).contiguous()
    ours = alloc_outputs(p_iso)
    launch_backward(ex, p_iso, ours)
    ref_bufs = [alloc_outputs(p_iso)[n] for n in ("dq", "dk", "dv")] if p.get("varlen") else []
    ref = run_fa3(p_iso, *ref_bufs)
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


def attention_ref_padded(q, k, v, query_padding_mask, key_padding_mask, causal=False, upcast=True,
                         reorder_ops=False):
    """``hopper/test_util.py::attention_ref`` with query / key padding masks (no softcap, no dropout,
    window (-1, 0) when causal): same operations in the same order."""
    import torch

    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    g = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    scores.masked_fill_((~key_padding_mask)[:, None, None, :], float("-inf"))
    local_mask = None
    if causal:  # construct_local_mask, window (-1, 0), per-batch lengths from the masks
        row_idx = torch.arange(seqlen_q, device=q.device, dtype=torch.long)[:, None]
        col_idx = torch.arange(seqlen_k, device=q.device, dtype=torch.long)
        sk = key_padding_mask.sum(-1)[:, None, None, None]
        sq = query_padding_mask.sum(-1)[:, None, None, None]
        local_mask = col_idx > row_idx + sk - sq + 0
        scores.masked_fill_(local_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    attention = attention.masked_fill((~key_padding_mask)[:, None, None, :], 0.0)
    if local_mask is not None:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    output.masked_fill_((~query_padding_mask)[:, :, None, None], 0.0)
    return output.to(dtype=dtype_og)


def _check_tolerance_varlen(p, ours, refs) -> dict:
    """``test_flash_attn_varlen_output``'s dQ / dK / dV assertions: grads re-padded (``dq_pad_fn`` /
    ``dk_pad_fn``), unused rows zeroed, against the padded references' grads; bound as fixed length."""
    import torch

    pd = p["pad"]
    b, sq, sk = p["b"], pd["q"].shape[1], pd["k"].shape[1]

    def pad(x, idx, s, unused):
        out = torch.zeros((b * s, *x.shape[1:]), device=x.device, dtype=x.dtype)
        out[idx] = x
        out = out.view(b, s, *x.shape[1:])
        if unused is not None:
            out.masked_fill_(unused[:, :, None, None], 0.0)
        return out

    q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_() for t in (pd["q"], pd["k"], pd["v"]))
    g = pad(p["dout"], pd["idx_q"], sq, None)
    out_ref = attention_ref_padded(q_ref, k_ref, v_ref, pd["qmask"], pd["kmask"], causal=p["is_causal"])
    out_pt = attention_ref_padded(q_ref, k_ref, v_ref, pd["qmask"], pd["kmask"], causal=p["is_causal"],
                                  upcast=False, reorder_ops=True)
    grads_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
    grads_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
    out = {}
    for name, ref, x_ref, x_pt in zip(("dq", "dk", "dv"), refs, grads_ref, grads_pt):
        idx, s, unused = (pd["idx_q"], sq, pd["q_unused"]) if name == "dq" else (pd["idx_k"], sk, pd["k_unused"])
        mine, theirs = pad(ours[name], idx, s, unused), pad(ref, idx, s, unused)
        atol = 2 * (x_ref + 0.3 - 0.3 - x_ref).abs().max().item()
        err = (mine - x_ref).abs().max().item()
        ref_err = (theirs - x_ref).abs().max().item()
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
    ex = compile_kernels(p["d"], p["is_causal"], gqa=p["gqa"], varlen=bool(p.get("varlen")),
                         backend_config=backend_config)
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

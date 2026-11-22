# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Operands, the FP8 KV-cache quantization, host-side parameters, the FlashMLA reference and the
correctness checks of the sparse FP8 decode port.

The operand generator follows `tests/lib.py::generate_testcase_for_decode` and the quantizer
`tests/quant.py::quantize_k_cache` of FlashMLA @ 9241ae3e; the PyTorch oracle is
`tests/ref.py::ref_sparse_attn_decode`. Host parameters follow `sparse_attn_decode_interface`
(`csrc/api/sparse_decode.h`).
"""

from __future__ import annotations

import random
from typing import Any

HEAD_DIM_V = 512
#: (d_qk, d_nope, d_rope, tile_size, num_tiles) of tests/quant.py::FP8KVCacheLayout.
LAYOUT = {576: (576, 512, 64, 128, 4), 512: (512, 448, 64, 64, 7)}
#: Bytes per KV token (sparse_decode.h:288-298).
BYTES_PER_TOKEN = {576: 512 + 64 * 2 + (512 // 128) * 4, 512: 448 + 64 * 2 + (448 // 64) + 1}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def require_sm90() -> None:
    from unittest import SkipTest

    import torch

    if not torch.cuda.is_available():
        raise SkipTest("CUDA is not available")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SkipTest("FlashMLA's sparse FP8 decode reference is an sm_90a kernel")


def import_flash_mla():
    import flash_mla

    return flash_mla


# ---------------------------------------------------------------------------------------------
# tests/kernelkit/generate.py
# ---------------------------------------------------------------------------------------------
def _non_contiguous_shape(shape):
    return [d + 128 if i == len(shape) - 1 else d + 1 for i, d in enumerate(shape)]


def _gen_non_contiguous_randn(shape, device):
    import torch

    # The upstream test runs with torch.set_default_dtype(torch.bfloat16).
    base = torch.randn(_non_contiguous_shape(shape), dtype=torch.bfloat16, device=device)
    return base[tuple(slice(0, d) for d in shape)]


def _non_contiguousify(t):
    import torch

    new = torch.empty(_non_contiguous_shape(t.shape), dtype=t.dtype, device=t.device)
    new = new[tuple(slice(0, d) for d in t.shape)]
    new[:] = t
    return new


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _randperm_batch(batch_size, perm_range, perm_size, padding, device):
    """`tests/lib.py::_randperm_batch` with a single padding value."""
    import torch

    torch.use_deterministic_algorithms(True)
    try:
        perm_range_max = max(int(torch.max(perm_range).item()), perm_size)
        rand = torch.rand(batch_size, perm_range_max, dtype=torch.float32, device=device)
        ar = torch.arange(0, perm_range_max, device=device).broadcast_to(batch_size, perm_range_max)
        rand[ar >= perm_range.view(batch_size, 1)] = float("-inf")
        res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
        res[res >= perm_range.view(batch_size, 1)] = padding
    finally:
        torch.use_deterministic_algorithms(False)
    return res


# ---------------------------------------------------------------------------------------------
# tests/quant.py
# ---------------------------------------------------------------------------------------------
def _scale_inv_to_ue8m0(scales_inv):
    import torch

    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(torch.float32)


def quantize_k_cache(k, d_qk):
    """`quantize_k_cache`: (num_blocks, block_size, 1, d_qk) bf16 -> the FP8 sparse layout."""
    import torch

    d, d_nope, d_rope, tile, ntiles = LAYOUT[d_qk]
    nb, bs, h_k, _ = k.shape
    assert h_k == 1
    k = k.squeeze(2)
    if d_qk == 576:
        bpt = d_nope + ntiles * 4 + 2 * d_rope
        res = torch.empty((nb, bs + 1, bpt), dtype=torch.float8_e4m3fn, device=k.device)[:, :bs, :]
        nope = res[..., :d_nope]
        scale = res[..., d_nope : d_nope + ntiles * 4].view(torch.float32)
        rope = res[..., d_nope + ntiles * 4 :].view(k.dtype)
        rope[:] = k[..., d_nope:]
        for ti in range(ntiles):
            seg = k[..., ti * tile : (ti + 1) * tile]
            s = _scale_inv_to_ue8m0(torch.abs(seg).max(dim=-1).values.float() / 448.0)
            scale[:, :, ti] = s
            nope[..., ti * tile : (ti + 1) * tile] = (seg.float() / s.unsqueeze(-1)).to(
                torch.float8_e4m3fn
            )
        return res.view(nb, bs, 1, -1)
    bpt = d_nope + 2 * d_rope + ntiles + 1
    padded = (bs * bpt + 576 - 1) // 576 * 576
    res = torch.empty((nb, padded), dtype=torch.float8_e4m3fn, device=k.device)[:, : bs * bpt]
    nope_rope = res[:, : bs * (d_nope + 2 * d_rope)].view(nb, bs, d_nope + 2 * d_rope)
    nope = nope_rope[:, :, :d_nope]
    rope = nope_rope[:, :, d_nope:].view(k.dtype)
    scale = res[:, bs * (d_nope + 2 * d_rope) :].view(nb, bs, 8)[:, :, :7].view(torch.float8_e8m0fnu)
    rope[:] = k[..., d_nope:]
    for ti in range(ntiles):
        seg = k[..., ti * tile : (ti + 1) * tile]
        s = _scale_inv_to_ue8m0(torch.abs(seg).max(dim=-1).values.float() / 448.0)
        scale[:, :, ti] = s.to(torch.float8_e8m0fnu)
        nope[:, :, ti * tile : (ti + 1) * tile] = (seg.float() / s.view(nb, bs, 1).float()).to(
            torch.float8_e4m3fn
        )
    return res.view(nb, bs, 1, -1)


def dequantize_k_cache(kq, d_qk):
    """`dequantize_k_cache`: the FP8 sparse layout -> (num_blocks, block_size, 1, d_qk) bf16."""
    import torch

    d, d_nope, d_rope, tile, ntiles = LAYOUT[d_qk]
    nb, bs, h_k, _ = kq.shape
    out = torch.empty((nb, bs, d), dtype=torch.bfloat16, device=kq.device)
    if d_qk == 576:
        kq = kq.view(nb, bs, -1)
        nope = kq[..., :d_nope]
        scale = kq[..., d_nope : d_nope + ntiles * 4].view(torch.float32)
        out[..., d_nope:] = kq[..., d_nope + ntiles * 4 :].view(torch.bfloat16)
        for ti in range(ntiles):
            out[..., ti * tile : (ti + 1) * tile] = (
                nope[..., ti * tile : (ti + 1) * tile].to(torch.float32) * scale[..., ti].unsqueeze(-1)
            )
    else:
        kq = kq.view(nb, -1)
        nope_rope = kq[:, : bs * (d_nope + 2 * d_rope)].view(nb, bs, d_nope + 2 * d_rope)
        nope = nope_rope[:, :, :d_nope]
        out[..., d_nope:] = nope_rope[:, :, d_nope:].view(torch.bfloat16)
        scale = kq[:, bs * (d_nope + 2 * d_rope) :].view(nb, bs, 8)[:, :, :7].view(torch.float8_e8m0fnu)
        for ti in range(ntiles):
            out[..., ti * tile : (ti + 1) * tile] = nope[..., ti * tile : (ti + 1) * tile].to(
                torch.bfloat16
            ) * scale[:, :, ti].to(torch.bfloat16).unsqueeze(-1)
    return out.view(nb, bs, 1, d)


def abs_indices_to_kvcache(abs_indices, block_table, block_size):
    """`abs_indices2indices_in_kvcache`."""
    import torch

    b, s_q, topk = abs_indices.shape
    _, max_blocks = block_table.shape
    abs_indices = abs_indices.clone()
    invalid = abs_indices == -1
    abs_indices[invalid] = 0
    ar = torch.arange(0, b, device=abs_indices.device).view(b, 1, 1) * max_blocks
    real = block_table.reshape(-1).index_select(0, (abs_indices // block_size + ar).view(-1))
    out = real.view(b, s_q, topk) * block_size + abs_indices % block_size
    out[invalid] = -1
    return out


# ---------------------------------------------------------------------------------------------
# tests/lib.py::generate_testcase_for_decode
# ---------------------------------------------------------------------------------------------
def _gen_kv_scope(c, s_k, block_size, topk, have_topk_length, device):
    import torch

    b, s_q, d_qk = c["b"], c["s_q"], c["d_qk"]
    seqlens = torch.full((b,), s_k, dtype=torch.int32)
    if c["is_varlen"]:
        for i in range(b):
            seqlens[i] = max(random.normalvariate(s_k, s_k / 2), s_q)
    if c["have_zero_seqlen_k"]:
        seqlens[torch.randn(b, dtype=torch.float32) > 0] = 0
    align = 4 * block_size
    max_pad = max(_cdiv(int(seqlens.max().item()), align), 1) * align
    seqlens = seqlens.to(device)
    block_table = torch.arange(b * max_pad // block_size, dtype=torch.int32, device=device)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel(), device=device)].view(b, -1)
    blocked_k = _gen_non_contiguous_randn((block_table.numel(), block_size, 1, d_qk), device) / 10
    blocked_k.clamp_(min=-1.0, max=1.0)
    abs_indices = torch.empty((b, s_q, topk), dtype=torch.int32, device=device)
    if c["is_all_indices_invalid"]:
        abs_indices.fill_(-1)
    else:
        abs_indices[:] = _randperm_batch(
            b * s_q, seqlens.repeat_interleave(s_q), topk, -1, device
        ).view(b, s_q, topk)
    indices = abs_indices_to_kvcache(abs_indices, block_table, block_size)
    topk_length = (
        torch.randint(0, topk + 1, (b,), dtype=torch.int32, device=device) if have_topk_length else None
    )
    if have_topk_length and c.get("topk_length_value") is not None:
        topk_length.fill_(c["topk_length_value"])  # edge shapes: e.g. every length 0
    if have_topk_length:
        masked = indices.clone()
        ar = torch.arange(0, topk, device=device).view(1, 1, topk).broadcast_to(b, s_q, topk)
        masked[ar >= topk_length.view(b, 1, 1)] = -1
    else:
        masked = indices
    # Never-referenced KV rows are NaN, so a kernel that reads one is caught.
    flat = blocked_k.view(-1, 1, d_qk)
    unused = torch.ones(flat.size(0), dtype=torch.bool, device=device)
    unused[masked] = False  # -1 clears the last row, as upstream's index does
    flat[unused, :, :] = float("nan")
    blocked_k = flat.view(-1, block_size, 1, d_qk)
    kq = quantize_k_cache(blocked_k, d_qk)
    fill = c.get("index_fill_beyond_topk_length")
    if have_topk_length and fill is not None:
        # Edge shapes: garbage (e.g. INT_MAX) indices past topk_length, which the kernel must not
        # dereference (splitkv_mla.cuh:531-536); the reference masks them by topk_length.
        ar = torch.arange(0, topk, device=device).view(1, 1, topk).broadcast_to(b, s_q, topk)
        indices = indices.clone()
        indices[ar >= topk_length.view(b, 1, 1)] = fill
    return {
        "seqlens": seqlens,
        "block_table": _non_contiguousify(block_table),
        "k_dequant": dequantize_k_cache(kq, d_qk),
        "kcache": kq,
        "indices": _non_contiguousify(indices),
        "topk_length": topk_length,
    }


def generate(config: dict, device="cuda") -> dict:
    import torch

    c = dict(config)
    _set_seed(c.get("seed", 0))
    b, h_q, s_q, d_qk = c["b"], c["h_q"], c["s_q"], c["d_qk"]
    q = torch.randn((b, s_q, h_q, d_qk), dtype=torch.bfloat16, device=device)
    q.clamp_(min=-1.0, max=1.0)
    attn_sink = None
    if c["enable_attn_sink"]:
        attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
        inf_mask = torch.randn((h_q,), dtype=torch.float32, device=device)
        attn_sink[inf_mask > 0.5] = float("inf")
        attn_sink[inf_mask < -0.5] = float("-inf")
        if c.get("attn_sink_value") is not None:
            attn_sink.fill_(c["attn_sink_value"])
    scope0 = _gen_kv_scope(c, c["s_k"], c["block_size"], c["topk"], c["have_topk_length"], device)
    scope1 = None
    if c.get("extra_topk") is not None:
        scope1 = _gen_kv_scope(
            c, c["extra_s_k"], c["extra_block_size"], c["extra_topk"], c["have_extra_topk_length"], device
        )
    return {
        "config": c,
        "q": _non_contiguousify(q),
        "attn_sink": attn_sink,
        "sm_scale": d_qk**-0.55,
        "kv": scope0,
        "extra": scope1,
    }


# ---------------------------------------------------------------------------------------------
# tests/ref.py::ref_sparse_attn_decode
# ---------------------------------------------------------------------------------------------
def torch_reference(g):
    import torch

    c = g["config"]
    b, s_q, h_q, d_qk = c["b"], c["s_q"], c["h_q"], c["d_qk"]

    def scope(sc):
        idx = sc["indices"]
        topk = idx.size(-1)
        invalid = idx == -1
        if sc["topk_length"] is not None:
            ar = torch.arange(0, topk, device=idx.device).view(1, 1, topk).broadcast_to(b, s_q, topk)
            invalid |= ar >= sc["topk_length"].view(b, 1, 1)
        # Upstream clamps at 0 only; the edge shapes also put INT_MAX past topk_length, so gather
        # through the mask (the masked rows are -inf-weighted either way).
        safe = torch.where(invalid, torch.zeros_like(idx), idx)
        gathered = (
            sc["k_dequant"].view(-1, d_qk).index_select(0, safe.reshape(-1)).view(b, s_q, topk, d_qk)
        )
        return gathered, invalid

    kv, invalid = scope(g["kv"])
    if g["extra"] is not None:
        kv1, invalid1 = scope(g["extra"])
        kv = torch.cat([kv, kv1], dim=2)
        invalid = torch.cat([invalid, invalid1], dim=2)
    kv = kv.view(b * s_q, -1, d_qk).float()
    kv[kv != kv] = 0.0
    q = g["q"].float().view(b * s_q, h_q, d_qk)
    w = q @ kv.transpose(-1, -2)
    w *= g["sm_scale"]
    w[invalid.view(b * s_q, 1, -1).broadcast_to(b * s_q, h_q, invalid.size(-1))] = float("-inf")
    lse = w.logsumexp(dim=-1)
    w = torch.exp(w - lse.unsqueeze(-1))
    out = (w @ kv[..., :HEAD_DIM_V]).view(b, s_q, h_q, HEAD_DIM_V)
    lse = lse.view(b, s_q, h_q)
    if g["attn_sink"] is not None:
        out *= (1.0 / (1.0 + torch.exp(g["attn_sink"].view(1, 1, h_q) - lse))).unsqueeze(-1)
    lonely = lse == float("-inf")
    out[lonely.unsqueeze(-1).broadcast_to(b, s_q, h_q, HEAD_DIM_V)] = 0.0
    lse[lonely] = float("+inf")
    return out.to(torch.bfloat16), lse.transpose(1, 2)


# ---------------------------------------------------------------------------------------------
# sparse_attn_decode_interface host side
# ---------------------------------------------------------------------------------------------
def num_sm_parts_for(num_sms: int, s_q: int, h_q: int) -> int:
    """`Decode_Sm90_Impl::get_meta` (sparse_decode.h:59-66)."""
    return max(num_sms // s_q // (h_q // 64), 1)


def prepare(**config) -> dict[str, Any]:
    """Operands plus the output / workspace tensors `sparse_attn_decode_interface` allocates."""
    import torch

    from tirx_kernels.runner import hardware_num_sms

    require_sm90()
    g = generate(config)
    c = g["config"]
    b, s_q, h_q = c["b"], c["s_q"], c["h_q"]
    parts = num_sm_parts_for(hardware_num_sms(), s_q, h_q)
    dev = g["q"].device
    g["num_sm_parts"] = parts
    g["out"] = torch.empty((b, s_q, h_q, HEAD_DIM_V), dtype=torch.bfloat16, device=dev)
    g["lse"] = torch.empty((b, s_q, h_q), dtype=torch.float32, device=dev)
    g["tile_scheduler_metadata"] = torch.empty((parts, 8), dtype=torch.int32, device=dev)
    g["num_splits"] = torch.empty((b + 1,), dtype=torch.int32, device=dev)
    g["lse_accum"] = torch.empty((b + parts, s_q, h_q), dtype=torch.float32, device=dev)
    g["out_accum"] = torch.empty((b + parts, s_q, h_q, HEAD_DIM_V), dtype=torch.float32, device=dev)
    return g


def fill_sentinel(t) -> None:
    import torch

    if t.dtype in (torch.int32,):
        t.fill_(-0x55555556)
    else:
        t.view(torch.int16 if t.element_size() == 2 else torch.int32).fill_(-1)  # all-ones NaN


def run_flashmla(g, sched_meta=None):
    """`flash_mla_with_kvcache` exactly as `tests/lib.py::run_flash_mla_decode` calls it."""
    flash_mla = import_flash_mla()
    if sched_meta is None:
        sched_meta, _ = flash_mla.get_mla_metadata()
    x = g["extra"]
    out, lse = flash_mla.flash_mla_with_kvcache(
        g["q"],
        g["kv"]["kcache"],
        None,
        None,
        HEAD_DIM_V,
        sched_meta,
        None,
        g["sm_scale"],
        False,
        True,
        g["kv"]["indices"],
        g["attn_sink"],
        x["kcache"] if x is not None else None,
        x["indices"] if x is not None else None,
        g["kv"]["topk_length"],
        x["topk_length"] if x is not None else None,
    )
    return out, lse, sched_meta


def bitwise_diff(actual, expected):
    """(number of differing elements, first differing flat index) over the raw bits."""
    import torch

    bits = torch.int16 if actual.element_size() == 2 else torch.int32
    neq = actual.contiguous().view(bits) != expected.contiguous().view(bits)
    n = int(neq.sum())
    return n, (int(neq.reshape(-1).nonzero()[0]) if n else None)


def check_is_allclose(name, ans, ref, abs_tol=1e-5, rel_tol=1e-2, cos_diff_tol=1e-7) -> None:
    """`tests/kernelkit/compare.py::check_is_allclose`, raising instead of returning False."""
    import torch

    assert ans.shape == ref.shape, f"`{name}` Shape mismatch: {ans.shape} vs {ref.shape}"
    assert ans.dtype == ref.dtype, f"`{name}` Dtype mismatch: {ans.dtype} vs {ref.dtype}"
    ans, ref = ans.clone().to(torch.float), ref.clone().to(torch.float)
    errors = []

    def deal_with_anomalies(val):
        ref_mask = (ref == val) if (val == val) else (ref != ref)
        ans_mask = (ans == val) if (val == val) else (ans != ans)
        ref[ref_mask] = 0.0
        ans[ans_mask] = 0.0
        if not torch.equal(ref_mask, ans_mask):
            errors.append(
                f"`{name}` anomaly `{val}` mismatch: {int(ans_mask.sum())} in ans but "
                f"{int(ref_mask.sum())} in ref"
            )

    for val in (float("inf"), float("-inf"), float("nan")):
        deal_with_anomalies(val)
    a64, r64 = ans.double(), ref.double()
    if float((r64 * r64).sum()) < 1e-12:
        cos_diff = 0.0
    else:
        cos_diff = 1 - 2 * float((a64 * r64).sum()) / float((a64 * a64 + r64 * r64).sum())
    raw_abs = torch.abs(ans - ref)
    raw_rel = raw_abs / (torch.abs(ref) + 1e-6)
    rel_err = raw_rel.masked_fill(raw_abs < abs_tol, 0)
    abs_err = raw_abs.masked_fill(raw_rel < rel_tol, 0)
    ok = (abs_err < abs_tol) | (rel_err < rel_tol)
    if errors:
        raise AssertionError("; ".join(errors))
    if not bool(ok.all()):
        raise AssertionError(
            f"`{name}` mismatch: {int(ok.sum())} of {ok.numel()} passed, max abs err "
            f"{float(abs_err.max())}, max rel err {float(rel_err.max())}, cos diff {cos_diff}"
        )
    if abs(cos_diff) > cos_diff_tol:
        raise AssertionError(f"`{name}` cosine diff too large: {cos_diff} vs {cos_diff_tol}")


def check_case(config, *, backend_config=None, kernel="flashmla_sm90_sparse_fp8_decode", reference=True):
    """Run one case; raise AssertionError on any mismatch. Returns a small result dict."""
    import torch

    from .launch import compile_kernels, launch_decode, launch_sched_meta, model_type_of

    g = prepare(**config)
    c = g["config"]
    ex = compile_kernels(model_type_of(c["d_qk"]), c["h_q"], backend_config=backend_config)
    detail = f"{c} parts={g['num_sm_parts']}"

    # Upstream first: its sched_meta is built by its own scheduler kernel on the first call.
    ref_out, ref_lse, sched_meta = run_flashmla(g)
    torch.cuda.synchronize()

    # Scheduler metadata: the 7 fields of every part, and num_splits, exact.
    fill_sentinel(g["tile_scheduler_metadata"])
    fill_sentinel(g["num_splits"])
    launch_sched_meta(ex, g)
    torch.cuda.synchronize()
    ref_meta, ref_splits = sched_meta.tile_scheduler_metadata, sched_meta.num_splits
    if not torch.equal(g["num_splits"], ref_splits):
        raise AssertionError(f"{kernel} {detail}: num_splits differs from FlashMLA's")
    if not torch.equal(g["tile_scheduler_metadata"][:, :7], ref_meta[:, :7]):
        bad = (g["tile_scheduler_metadata"][:, :7] != ref_meta[:, :7]).any(1).nonzero()
        raise AssertionError(f"{kernel} {detail}: scheduler metadata differs at parts {bad[:4]}")

    # Main + combine, NaN sentinel on every output and workspace.
    for name in ("out", "lse", "out_accum", "lse_accum"):
        fill_sentinel(g[name])
    launch_decode(ex, g)
    torch.cuda.synchronize()
    out, lse = g["out"], g["lse"].transpose(1, 2)
    n_out, first_out = bitwise_diff(out, ref_out)
    n_lse, first_lse = bitwise_diff(lse, ref_lse)
    if n_out or n_lse:
        raise AssertionError(
            f"{kernel} {detail}: not bitwise identical to FlashMLA: out {n_out}/{out.numel()} "
            f"(first {first_out}), lse {n_lse}/{lse.numel()} (first {first_lse})"
        )
    if reference:
        # FlashMLA's own acceptance test against the upstream PyTorch reference.
        out_ref, lse_ref = torch_reference(g)
        check_is_allclose("out", out, out_ref, abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
        check_is_allclose("lse", lse, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536)
    return {"bitwise": True, "num_sm_parts": g["num_sm_parts"]}


def bench_against_flashmla(
    executables, config, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0
):
    """Kernel-only: our main + combine against FlashMLA's (scheduler metadata precomputed by both,
    as `flash_mla_with_kvcache` does after its first call)."""
    import torch

    from tirx_kernels.runner import bench

    from .launch import launch_decode, launch_sched_meta

    g = prepare(**config)
    launch_sched_meta(executables, g)
    torch.cuda.synchronize()

    def tirx():
        launch_decode(executables, g)

    def build_reference():
        _, _, sched_meta = run_flashmla(g)
        torch.cuda.synchronize()
        return lambda: run_flashmla(g, sched_meta)

    return bench(
        {"tirx": tirx},
        references={"flashmla": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )

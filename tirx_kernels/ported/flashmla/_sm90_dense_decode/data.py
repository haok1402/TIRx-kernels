# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Operands, the FlashMLA reference, and the correctness / benchmark drivers.

`generate` replicates `tests/test_flash_mla_dense_decoding.py::generate_test_data` (same seeds,
same draws, NaN-filled KV past each request's length); `torch_reference` replicates its
`reference_torch`; `host_params` replicates the host half of `dense_attn_decode_interface`
(`csrc/api/dense_decode.h`): the GQA fold of Q, `num_sm_parts`, and every output / workspace
allocation and stride.
"""

from __future__ import annotations

import math
import random
from typing import Any

#: `config.h`.
BLOCK_SIZE_M = 64
PAGE_BLOCK_SIZE = 64
HEAD_DIM_K = 576
HEAD_DIM_V = 512
#: `sizeof(DecodingSchedMeta) / sizeof(int)` (`params.h`).
SCHED_META_INTS = 8

#: A quiet-NaN bit pattern every TIRx output starts from (bf16/fp16 halves and fp32 words).
_SENTINEL_16 = 0x7FC1
_SENTINEL_32 = 0x7FC00001


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def require_sm90() -> None:
    from unittest import SkipTest

    import torch

    if not torch.cuda.is_available():
        raise SkipTest("no CUDA device")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SkipTest("FlashMLA dense decode is SM90-only")


def import_flash_mla():
    import flash_mla

    return flash_mla


def generate(
    *,
    b,
    s_q,
    s_k,
    is_varlen,
    is_causal,
    h_q=128,
    h_kv=1,
    dtype="bfloat16",
    have_zero_seqlen_k=False,
    seed=0,
    block_size=PAGE_BLOCK_SIZE,
    d=HEAD_DIM_K,
):
    """`generate_test_data`, with the upstream test's default dtype/device made explicit."""
    import torch

    del is_causal
    dt, dev = torch_dtype(dtype), torch.device("cuda")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    assert h_q % h_kv == 0

    cache_seqlens_cpu = torch.full((b,), s_k, dtype=torch.int32, device="cpu")
    if is_varlen:
        for i in range(b):
            cache_seqlens_cpu[i] = max(random.normalvariate(s_k, s_k / 2), s_q)
    if have_zero_seqlen_k:
        zeros_mask = torch.randn(b, dtype=torch.float32, device="cpu") > 0
        cache_seqlens_cpu[zeros_mask] = 0

    max_seqlen = int(cache_seqlens_cpu.max().item())
    max_seqlen_pad = _cdiv(max_seqlen, 256) * 256
    cache_seqlens = cache_seqlens_cpu.to(dev)

    q = torch.randn(b, s_q, h_q, d, dtype=dt, device=dev) / 10
    q.clamp_(min=-1.0, max=1.0)

    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32, device=dev
    ).view(b, max_seqlen_pad // block_size)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel(), device=dev)].view(b, -1)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=dt, device=dev) / 10
    blocked_k.clamp_(min=-1.0, max=1.0)

    for i in range(b):
        cur_len = int(cache_seqlens_cpu[i].item())
        cur_num_blocks = _cdiv(cur_len, block_size)
        blocked_k[block_table[i][cur_num_blocks:]] = float("nan")
        if cur_len % block_size != 0:
            blocked_k[block_table[i][cur_num_blocks - 1]][cur_len % block_size :] = float("nan")
        block_table[i][cur_num_blocks:] = 2147480000
    return cache_seqlens, q, block_table, blocked_k


def torch_reference(cache_seqlens, block_table, q, blocked_k, dv, is_causal):
    """`reference_torch` of the upstream test (fp32 math, NaN KV zeroed, lonely-q handling)."""
    import torch

    def sdpa(query, kv, causal):
        h_q, h_kv = query.size(0), kv.size(0)
        s_q, s_k = query.shape[-2], kv.shape[-2]
        query, kv = query.float(), kv.float()
        if h_kv != 1:
            kv = kv.repeat_interleave(h_q // h_kv, dim=0)
        kv[kv != kv] = 0.0
        attn_weight = query @ kv.transpose(-2, -1)
        if causal and query.size(1) > 1:
            mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device)
            mask = mask.tril(diagonal=s_k - s_q)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float, device=query.device)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias.to(q.dtype)
        attn_weight /= math.sqrt(query.size(-1))
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ kv[..., :dv]
        lonely = lse == float("-inf")
        output[lonely.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely] = float("+inf")
        return output, lse

    b, s_q, h_q, d = q.size()
    block_size, h_kv = blocked_k.size(1), blocked_k.size(2)
    seqlens = cache_seqlens.cpu()
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=q.device)
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32, device=q.device)
    for i in range(b):
        cur_len = int(seqlens[i].item())
        idx = block_table[i][0 : _cdiv(cur_len, block_size)]
        cur_kv = blocked_k[idx].view(-1, h_kv, d)[:cur_len, ...]
        o, lse = sdpa(q[i].transpose(0, 1), cur_kv.transpose(0, 1), is_causal)
        out_ref[i] = o.transpose(0, 1)
        lse_ref[i] = lse
    return out_ref.to(q.dtype), lse_ref


def num_sm_parts_for(num_sms: int, s_q: int, h_q: int, h_kv: int) -> int:
    """`dense_attn_decode_interface`'s `num_sm_parts`."""
    return max(num_sms // h_kv // _cdiv(s_q * h_q // h_kv, BLOCK_SIZE_M), 1)


def host_params(q, blocked_k, cache_seqlens, block_table, *, is_causal, num_sms):
    """Everything `dense_attn_decode_interface` derives on the host, before its launches."""
    import torch

    b, s_q, h_q, d = q.shape
    h_kv = blocked_k.size(2)
    if s_q == 1:
        is_causal = False
    hq_per_hk = h_q // h_kv
    q_seq_per_hk = s_q * hq_per_hk
    q_folded = (
        q.view(b, s_q, h_kv, hq_per_hk, d).transpose(2, 3).reshape(b, q_seq_per_hk, h_kv, d)
    )
    num_sm_parts = num_sm_parts_for(num_sms, s_q, h_q, h_kv)
    f32 = dict(dtype=torch.float32, device=q.device)
    total_num_splits = b + num_sm_parts
    return {
        "b": b,
        "s_q": s_q,
        "h_q": h_q,
        "h_kv": h_kv,
        "q_head_per_hk": hq_per_hk,
        "q_seq_per_hk": q_seq_per_hk,
        "is_causal": bool(is_causal),
        "num_blocks": blocked_k.size(0),
        "num_sm_parts": num_sm_parts,
        "softmax_scale": d ** (-0.5),
        "q_folded": q_folded,
        "kcache": blocked_k,
        "seqlens_k": cache_seqlens,
        "block_table": block_table,
        "tile_scheduler_metadata": torch.empty(
            (num_sm_parts, SCHED_META_INTS), dtype=torch.int32, device=q.device
        ),
        "num_splits": torch.empty((b + 1,), dtype=torch.int32, device=q.device),
        "out": torch.empty((b, h_kv, q_seq_per_hk, HEAD_DIM_V), dtype=q.dtype, device=q.device),
        "lse": torch.empty((b, h_kv, q_seq_per_hk), **f32),
        "lse_accum": torch.empty((total_num_splits, h_kv, q_seq_per_hk), **f32),
        "out_accum": torch.empty((total_num_splits, h_kv, q_seq_per_hk, HEAD_DIM_V), **f32),
    }


def unfold_outputs(p):
    """The interface's final `view/transpose/reshape` of `out` and `lse` (b, s_q, h_q, dv)."""
    b, s_q, h_q, h_kv, per = p["b"], p["s_q"], p["h_q"], p["h_kv"], p["q_head_per_hk"]
    out = (
        p["out"]
        .view(b, h_kv, s_q, per, HEAD_DIM_V)
        .transpose(1, 2)
        .reshape(b, s_q, h_q, HEAD_DIM_V)
    )
    lse = p["lse"].view(b, h_kv, s_q, per).transpose(2, 3).reshape(b, h_q, s_q)
    return out, lse


def prepare(**config) -> dict[str, Any]:
    import torch

    require_sm90()
    cache_seqlens, q, block_table, blocked_k = generate(**config)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    p = host_params(
        q, blocked_k, cache_seqlens, block_table, is_causal=config["is_causal"], num_sms=num_sms
    )
    p.update(q=q, config=dict(config))
    return p


def fill_sentinel(t) -> None:
    import torch

    if t.element_size() == 2:
        t.view(torch.int16).fill_(_SENTINEL_16)
    else:
        t.view(torch.int32).fill_(_SENTINEL_32)


def run_flashmla(p, sched_meta=None):
    """The public upstream call the test makes; returns (out, lse, sched_meta)."""
    flash_mla = import_flash_mla()
    if sched_meta is None:
        sched_meta, _ = flash_mla.get_mla_metadata()
    out, lse = flash_mla.flash_mla_with_kvcache(
        p["q"],
        p["kcache"],
        p["block_table"],
        p["seqlens_k"],
        HEAD_DIM_V,
        sched_meta,
        None,
        causal=p["config"]["is_causal"],
    )
    return out, lse, sched_meta


def _bitwise_diff(actual, expected):
    import torch

    view = torch.int16 if actual.element_size() == 2 else torch.int32
    differ = actual.contiguous().view(view) != expected.contiguous().view(view)
    return int(differ.sum()), (tuple(int(i) for i in differ.nonzero()[0]) if differ.any() else None)


def check_case(config, *, backend_config=None, kernel="flashmla_sm90_dense_decode"):
    """Run one case; raise AssertionError on any mismatch. Returns a small result dict."""
    import torch

    from .launch import compile_kernels, launch_decode, launch_sched_meta

    p = prepare(**config)
    executables = compile_kernels(config.get("dtype", "bfloat16"), backend_config=backend_config)
    detail = (
        f"b={p['b']} s_q={p['s_q']} h_q={p['h_q']} h_kv={p['h_kv']} "
        f"causal={p['is_causal']} parts={p['num_sm_parts']}"
    )

    # Upstream first: its sched_meta is built by its own scheduler kernel on the first call.
    ref_out, ref_lse, sched_meta = run_flashmla(p)
    torch.cuda.synchronize()

    # Scheduler metadata: every field of every part, and num_splits, exact.
    fill_sentinel(p["tile_scheduler_metadata"])
    fill_sentinel(p["num_splits"])
    launch_sched_meta(executables, p)
    torch.cuda.synchronize()
    ref_meta, ref_splits = sched_meta.tile_scheduler_metadata, sched_meta.num_splits
    if not torch.equal(p["num_splits"], ref_splits):
        raise AssertionError(f"{kernel} {detail}: num_splits differs from FlashMLA's")
    # The 8th int is padding the source never writes; compare the 7 fields.
    if not torch.equal(p["tile_scheduler_metadata"][:, :7], ref_meta[:, :7]):
        bad = (p["tile_scheduler_metadata"][:, :7] != ref_meta[:, :7]).any(1).nonzero()
        raise AssertionError(f"{kernel} {detail}: scheduler metadata differs at parts {bad[:4]}")

    # Main + combine, NaN sentinel on every output and workspace.
    for name in ("out", "lse", "out_accum", "lse_accum"):
        fill_sentinel(p[name])
    launch_decode(executables, p)
    torch.cuda.synchronize()
    out, lse = unfold_outputs(p)

    n_out, first_out = _bitwise_diff(out, ref_out)
    n_lse, first_lse = _bitwise_diff(lse, ref_lse)
    if n_out or n_lse:
        raise AssertionError(
            f"{kernel} {detail}: not bitwise identical to FlashMLA: out {n_out}/{out.numel()} "
            f"(first {first_out}), lse {n_lse}/{lse.numel()} (first {first_lse})"
        )

    # FlashMLA's own acceptance test against the PyTorch reference.
    out_ref, lse_ref = torch_reference(
        p["seqlens_k"], p["block_table"], p["q"], p["kcache"], HEAD_DIM_V, p["config"]["is_causal"]
    )
    check_is_allclose("out", out, out_ref, abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
    check_is_allclose("lse", lse, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536)
    return {"bitwise": True, "num_sm_parts": p["num_sm_parts"]}


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
    raw_abs_err = torch.abs(ans - ref)
    raw_rel_err = raw_abs_err / (torch.abs(ref) + 1e-6)
    rel_err = raw_rel_err.masked_fill(raw_abs_err < abs_tol, 0)
    abs_err = raw_abs_err.masked_fill(raw_rel_err < rel_tol, 0)
    pass_mask = (abs_err < abs_tol) | (rel_err < rel_tol)
    if errors:
        raise AssertionError("; ".join(errors))
    if not bool(pass_mask.all()):
        raise AssertionError(
            f"`{name}` mismatch: {int(pass_mask.sum())} of {pass_mask.numel()} passed, max abs err "
            f"{float(abs_err.max())}, max rel err {float(rel_err.max())}, cos diff {cos_diff}"
        )
    if abs(cos_diff) > cos_diff_tol:
        raise AssertionError(f"`{name}` cosine diff too large: {cos_diff} vs {cos_diff_tol}")


def bench_against_flashmla(
    executables, config, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0
):
    """Kernel-only: our main + combine against FlashMLA's (scheduler metadata precomputed by both,
    as `flash_mla_with_kvcache` does after its first call)."""
    import torch

    from tirx_kernels.runner import bench

    from .launch import launch_decode, launch_sched_meta

    p = prepare(**config)
    launch_sched_meta(executables, p)
    torch.cuda.synchronize()

    def tirx():
        launch_decode(executables, p)

    def build_reference():
        _, _, sched_meta = run_flashmla(p)
        torch.cuda.synchronize()
        return lambda: run_flashmla(p, sched_meta)

    return bench(
        {"tirx": tirx},
        references={"flashmla": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )

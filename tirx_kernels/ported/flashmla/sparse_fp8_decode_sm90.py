# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM90 sparse FP8 MLA decode (top-k gathered, FP8-quantized KV cache, split-KV) -- FlashMLA's
Hopper sparse decode path.

Reference: `flash_mla.flash_mla_with_kvcache(..., is_fp8_kvcache=True, indices=...)` on SM90, which
runs `sparse_attn_decode_interface` (`csrc/api/sparse_decode.h`) and launches three kernels:

- `smxx::decode::get_mla_metadata_kernel` (`csrc/smxx/decode/get_decoding_sched_meta/
  get_decoding_sched_meta.cu`), sparse path (`topk != -1`): the split-KV tile scheduler metadata;
- `sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE,
  NUM_HEADS>>` (`csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`, `config.h`, `components/`): the
  three-warpgroup split-KV kernel (two attention warpgroups, one FP8 gather + dequant producer);
- `smxx::decode::flash_fwd_mla_combine_kernel` (`csrc/smxx/decode/combine/combine.cu`), with the
  attention sink.

Upstream instantiates four specializations (`instantiations/*.cu`):

| model_type | d_qk | KV token bytes | scales | h_q | cluster |
|---|---|---|---|---|---|
| V32 (DeepSeek-V3.2) | 576 | 656 (512 fp8 + 4 fp32 + 64 bf16) | fp32, one per 128 | 64 / 128 | 1 / 2 |
| MODEL1 | 512 | 584 (448 fp8 + 64 bf16 + 7 e8m0 + pad) | e8m0, one per 64 | 64 / 128 | 1 / 2 |
"""

from __future__ import annotations

from typing import Any

KERNEL_META = {
    "name": "flashmla_sm90_sparse_fp8_decode",
    "category": "ported.flashmla",
    "runtime_cuda_archs": ["sm_90a"],
    "reference_requirements": (
        {
            "package": "flash-mla",
            "git": {
                "url": "https://github.com/deepseek-ai/FlashMLA.git",
                "commit": "9241ae3ef9bac614dd25e45e507e089f888280e0",
            },
            "import": "flash_mla",
        },
    ),
}


def _case(
    b,
    h_q,
    s_q,
    s_k,
    is_varlen,
    topk,
    *,
    d_qk=576,
    block_size=64,
    have_topk_length=False,
    enable_attn_sink=True,
    extra_s_k=None,
    extra_topk=None,
    extra_block_size=None,
    have_extra_topk_length=False,
    is_all_indices_invalid=False,
    have_zero_seqlen_k=False,
    seed=0,
    prefix="",
) -> dict:
    """One `RawTestParamForDecode` of `tests/test_flash_mla_sparse_decoding.py` as a config dict."""
    if extra_topk is not None:
        # generate_testcase_for_decode's defaults for the extra scope.
        extra_s_k = extra_topk * 2 if extra_s_k is None else extra_s_k
        extra_block_size = block_size if extra_block_size is None else extra_block_size
    model = "v32" if d_qk == 576 else "model1"
    label = (
        f"{prefix}{model}_b{b}_hq{h_q}_sq{s_q}_sk{s_k}_topk{topk}_bs{block_size}"
        f"{'_varlen' if is_varlen else ''}{'_tl' if have_topk_length else ''}"
        f"{'' if enable_attn_sink else '_nosink'}"
        + (
            f"_x{extra_s_k}_{extra_topk}_{extra_block_size}{'_xtl' if have_extra_topk_length else ''}"
            if extra_topk is not None
            else ""
        )
        + f"{'_allinvalid' if is_all_indices_invalid else ''}{'_zeroseq' if have_zero_seqlen_k else ''}"
        + (f"_s{seed}" if seed else "")
    )
    return {
        "label": label,
        "b": b,
        "h_q": h_q,
        "s_q": s_q,
        "s_k": s_k,
        "is_varlen": is_varlen,
        "topk": topk,
        "d_qk": d_qk,
        "block_size": block_size,
        "have_topk_length": have_topk_length,
        "enable_attn_sink": enable_attn_sink,
        "extra_s_k": extra_s_k,
        "extra_topk": extra_topk,
        "extra_block_size": extra_block_size,
        "have_extra_topk_length": have_extra_topk_length,
        "is_all_indices_invalid": is_all_indices_invalid,
        "have_zero_seqlen_k": have_zero_seqlen_k,
        "seed": seed,
    }


_SK_TOPK_BS = [
    (512, 64, 2),
    (512, 64, 64),
    (512, 64, 69),
    (1024, 576, 2),
    (1024, 576, 61),
    (2046, 2048, 2),
    (2046, 2048, 64),
    (2046, 2048, 576),
]
_CORNER_SK_TOPK_BS = [(512, 64, 61), (650, 576, 53)]


def _feature_axes():
    """`gen_testcase`'s outer loops: (d_qk, have_extra_k, have_extra_topk_len, have_topk_len, h_q)."""
    for d_qk in (576, 512):
        for have_extra_k in (False, True) if d_qk == 512 else (False,):
            for have_extra_topk_len in (False, True) if have_extra_k else (False,):
                for have_topk_len in (False, True) if d_qk == 512 else (False,):
                    for h_q in (64, 128):
                        yield d_qk, have_extra_k, have_extra_topk_len, have_topk_len, h_q


def _varlens(b, have_topk_len, have_extra_topk_len):
    return (True, False) if (b == 74 and not have_topk_len and not have_extra_topk_len) else (True,)


def upstream_correctness_cases() -> list[dict]:
    """`gen_testcase`'s `correctness_cases`, in its order."""
    out = []
    for d_qk, have_extra_k, have_xtl, have_tl, h_q in _feature_axes():
        for s_k, topk, bs in _SK_TOPK_BS:
            for extra in _SK_TOPK_BS if have_extra_k else [(None, None, None)]:
                for b in (4, 74, 321):
                    for s_q in (1, 3):
                        for varlen in _varlens(b, have_tl, have_xtl):
                            out.append(
                                _case(
                                    b, h_q, s_q, s_k, varlen, topk, d_qk=d_qk, block_size=bs,
                                    have_topk_length=have_tl, enable_attn_sink=True,
                                    extra_s_k=extra[0], extra_topk=extra[1],
                                    extra_block_size=extra[2], have_extra_topk_length=have_xtl,
                                )
                            )
    return out


def upstream_corner_cases() -> list[dict]:
    """`gen_testcase`'s `corner_cases`: all-invalid indices, zero-length sequences, no sink."""
    out = []
    for d_qk, have_extra_k, have_xtl, have_tl, h_q in _feature_axes():
        for s_k, topk, bs in _CORNER_SK_TOPK_BS:
            for extra in _CORNER_SK_TOPK_BS if have_extra_k else [(None, None, None)]:
                for b in (4, 74, 321):
                    for varlen in _varlens(b, have_tl, have_xtl):
                        for all_invalid in (True, False):
                            for zero_seq in (True, False):
                                for sink in (True, False):
                                    if not (all_invalid or zero_seq or sink):
                                        continue
                                    out.append(
                                        _case(
                                            b, h_q, 3, s_k, varlen, topk, d_qk=d_qk,
                                            block_size=bs, have_topk_length=have_tl,
                                            enable_attn_sink=sink, extra_s_k=extra[0],
                                            extra_topk=extra[1], extra_block_size=extra[2],
                                            have_extra_topk_length=have_xtl,
                                            is_all_indices_invalid=all_invalid,
                                            have_zero_seqlen_k=zero_seq,
                                        )
                                    )
    return out


def upstream_performance_cases() -> list[dict]:
    """`gen_testcase`'s `performance_cases`: the production shapes and the peak-perf shapes."""
    v32 = dict(h_q=128, s_q=2, s_k=32768, is_varlen=True, topk=2048, d_qk=576)
    m1 = dict(s_q=2, s_k=16384, is_varlen=True, topk=128, d_qk=512, extra_s_k=16384, block_size=256)
    bases = [
        (dict(v32), (2, 64, 74, 128), "prod_v32_"),
        (dict(m1, h_q=64, extra_topk=512, extra_block_size=64), (2, 64, 74, 128, 148, 256), "prod_m1c1_"),
        (dict(m1, h_q=128, extra_topk=1024, extra_block_size=64), (2, 64, 74, 128, 148, 256), "prod_m1c2_"),
        (dict(m1, h_q=64, extra_topk=1024, extra_block_size=2, have_extra_topk_length=True),
         (2, 64, 74, 128, 148, 256), "prod_m1c3_"),
        (dict(m1, h_q=128, extra_topk=1024, extra_block_size=2, have_extra_topk_length=True),
         (2, 64, 74, 128, 148, 256), "prod_m1c4_"),
    ]
    out = []
    for base, bszs, prefix in bases:
        for b in bszs:
            kw = dict(base)
            h_q, s_q, s_k, varlen, topk = (kw.pop(k) for k in ("h_q", "s_q", "s_k", "is_varlen", "topk"))
            # The production cases use RawTestParam's default enable_attn_sink=True.
            out.append(_case(b, h_q, s_q, s_k, varlen, topk, prefix=prefix, **kw))
    for h_q in (64, 128):
        for d_qk in (512, 576):
            out.append(_case(148, h_q, 2, 32768, True, 16384, d_qk=d_qk, prefix="peak_"))
    return out


CONFIGS: list[dict] = []  # filled at the correctness stage

BENCH_CONFIGS: list[dict] = upstream_performance_cases()


def get_kernel(*, backend_config=None, **config):
    """The three PrimFuncs of one specialization: scheduler, main, combine."""
    from ._sm90_sparse_fp8_decode.launch import build_kernels

    config.pop("label", None)
    return list(build_kernels(config, backend_config=backend_config))


def prepare_data(**config) -> dict[str, Any]:
    from ._sm90_sparse_fp8_decode.data import prepare

    config.pop("label", None)
    return prepare(**config)


def run_test(*, backend_config=None, **config):
    """Scheduler metadata exact, outputs bitwise vs FlashMLA (whole tensors, NaN sentinel),
    and FlashMLA's own tolerances against the upstream test's PyTorch reference."""
    from ._sm90_sparse_fp8_decode.data import check_case

    config.pop("label", None)
    return check_case(config, backend_config=backend_config, kernel=KERNEL_META["name"])


def run_bench(
    *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, backend_config=None, **config
):
    raise NotImplementedError


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
    "upstream_correctness_cases",
    "upstream_corner_cases",
    "upstream_performance_cases",
]

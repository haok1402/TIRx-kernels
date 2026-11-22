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
    **extra_opts,
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
        + "".join(f"_{k}{v}" for k, v in sorted(extra_opts.items()))
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
        **extra_opts,
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


def _official_subset() -> list[dict]:
    """A cover of the upstream grid for the official runner: per feature group (model, h_q,
    topk_length, extra KV, extra_topk_length) every (s_k, topk, page) triple once (paired with
    the same extra triple), at b = 74 / s_q = 3 varlen, plus b = 4 and b = 321 at s_q = 1; and the
    corner cases with all indices invalid and with zero-length sequences. The full grids run
    through `upstream_correctness_cases` / `upstream_corner_cases` (scripted sweep)."""
    picked = []
    for d_qk, have_x, have_xtl, have_tl, h_q in _feature_axes():
        for n, (s_k, topk, bs) in enumerate(_SK_TOPK_BS):
            ex = _SK_TOPK_BS[n] if have_x else (None, None, None)
            for b, s_q in ((74, 3), (4, 1), (321, 1)) if n == 0 else ((74, 3),):
                picked.append(
                    _case(b, h_q, s_q, s_k, True, topk, d_qk=d_qk, block_size=bs,
                          have_topk_length=have_tl, enable_attn_sink=True, extra_s_k=ex[0],
                          extra_topk=ex[1], extra_block_size=ex[2], have_extra_topk_length=have_xtl)
                )
        s_k, topk, bs = _CORNER_SK_TOPK_BS[1]
        ex = _CORNER_SK_TOPK_BS[1] if have_x else (None, None, None)
        for all_invalid, zero_seq, sink in ((True, False, True), (False, True, False)):
            picked.append(
                _case(74, h_q, 3, s_k, True, topk, d_qk=d_qk, block_size=bs, have_topk_length=have_tl,
                      enable_attn_sink=sink, extra_s_k=ex[0], extra_topk=ex[1],
                      extra_block_size=ex[2], have_extra_topk_length=have_xtl,
                      is_all_indices_invalid=all_invalid, have_zero_seqlen_k=zero_seq)
            )
    return picked


INT_MAX = 2147483647

#: Edge shapes beyond the upstream grid.
EDGE_CASES = [
    # one request split over every partition (b = 1, topk 16384)
    _case(1, 128, 1, 65536, False, 16384, d_qk=576, prefix="edge_", seed=21),
    _case(1, 64, 1, 65536, False, 16384, d_qk=512, prefix="edge_", seed=22),
    # s_k < topk: -1 padding inside the index rows; page size 1
    _case(3, 64, 7, 100, True, 128, d_qk=576, block_size=1, prefix="edge_", seed=23),
    # num_sm_parts == 1 (s_q * h_q / 64 >= 132): one CTA walks every request (Q reload parity)
    _case(24, 128, 66, 256, True, 64, d_qk=576, prefix="edge_", seed=24),
    _case(20, 64, 140, 300, True, 128, d_qk=512, have_topk_length=True, prefix="edge_", seed=25),
    # MODEL1: INT_MAX indices past topk_length / extra_topk_length (no IMA, masked)
    _case(37, 64, 2, 2000, True, 512, d_qk=512, have_topk_length=True, extra_s_k=1500,
          extra_topk=256, extra_block_size=2, have_extra_topk_length=True, prefix="edge_", seed=26,
          index_fill_beyond_topk_length=INT_MAX),
    _case(37, 128, 2, 2000, True, 512, d_qk=512, have_topk_length=True, extra_s_k=1500,
          extra_topk=256, extra_block_size=2048, have_extra_topk_length=True, prefix="edge_",
          seed=27, index_fill_beyond_topk_length=INT_MAX),
    # MODEL1: every topk_length and extra_topk_length 0 (only padded, fully masked blocks)
    _case(9, 128, 3, 700, True, 128, d_qk=512, have_topk_length=True, extra_s_k=700,
          extra_topk=128, have_extra_topk_length=True, prefix="edge_", seed=28, topk_length_value=0),
    # attention sink +inf / -inf everywhere
    _case(16, 64, 2, 1024, True, 256, d_qk=576, prefix="edge_", seed=29, attn_sink_value=float("inf")),
    _case(16, 128, 2, 1024, True, 256, d_qk=512, prefix="edge_", seed=30, attn_sink_value=float("-inf")),
    # many short requests, mostly unsplit (b = 132, topk 64)
    _case(132, 64, 1, 64, False, 64, d_qk=576, prefix="edge_", seed=31),
    _case(132, 128, 1, 64, False, 64, d_qk=512, prefix="edge_", seed=32),
]

CONFIGS: list[dict] = [*_official_subset(), *EDGE_CASES]

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


def prepare_bench(*, backend_config=None, **config):
    """Compile the three kernels without touching CUDA."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    from ._sm90_sparse_fp8_decode.launch import compile_kernels, model_type_of

    config.pop("label", None)
    state = {
        "config": dict(config),
        "executables": compile_kernels(
            model_type_of(config["d_qk"]), config["h_q"], backend_config=backend_config
        ),
    }
    return prepared_gpu_benchmark(run_gpu, state, backend_config=backend_config)


def run_gpu(
    prepared,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
    backend_config=None,
    **config,
):
    from ._sm90_sparse_fp8_decode.data import bench_against_flashmla

    del backend_config
    config = {**prepared["config"], **config}
    config.pop("label", None)
    return bench_against_flashmla(
        prepared["executables"],
        config,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, backend_config=None, **config
):
    return prepare_bench(**config, backend_config=backend_config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "EDGE_CASES",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
    "upstream_correctness_cases",
    "upstream_corner_cases",
    "upstream_performance_cases",
]

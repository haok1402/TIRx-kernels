# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM90 dense MLA decode (paged KV cache, split-KV) -- FlashMLA's Hopper decode path.

Reference: `flash_mla.flash_mla_with_kvcache` with a dense bf16/fp16 KV cache on SM90, which runs
`dense_attn_decode_interface` (`csrc/api/dense_decode.h`) and launches three kernels:

- `smxx::decode::get_mla_metadata_kernel` (`csrc/smxx/decode/get_decoding_sched_meta/
  get_decoding_sched_meta.cu`): the split-KV tile scheduler metadata, once per sched_meta;
- `sm90::flash_fwd_splitkv_mla_kernel` (`csrc/sm90/decode/dense/splitkv_mla.cuh`, traits in
  `traits.h` / `config.h`): the two-warpgroup split-KV attention kernel;
- `smxx::decode::flash_fwd_mla_combine_kernel` (`csrc/smxx/decode/combine/combine.cu`): the
  split-KV LSE-weighted combine, launched with PDL after the main kernel.

The only specialization upstream has is d_qk = 576, d_v = 512, page_block_size = 64, BLOCK_SIZE_M
= 64, 256 threads; the dtype (bf16 / fp16) is the one template parameter.
"""

from __future__ import annotations

from typing import Any

KERNEL_META = {
    "name": "flashmla_sm90_dense_decode",
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

#: `tests/test_flash_mla_dense_decoding.py::main` axes.
_H_PAIRS = [
    (h_q, h_kv)
    for h_q in (1, 3, 9, 63, 64, 126, 128)
    for h_kv in (1, 2, 3, 8)
    if h_q % h_kv == 0
]


def _case(
    b,
    s_q,
    s_k,
    is_varlen,
    is_causal,
    h_q=128,
    h_kv=1,
    *,
    dtype="bfloat16",
    have_zero_seqlen_k=False,
    seed=0,
    prefix="",
) -> dict:
    tag = "bf16" if dtype == "bfloat16" else "fp16"
    label = (
        f"{prefix}{tag}_b{b}_sq{s_q}_sk{s_k}_hq{h_q}_hkv{h_kv}"
        f"{'_varlen' if is_varlen else ''}{'_causal' if is_causal else ''}"
        f"{'_zeroseq' if have_zero_seqlen_k else ''}"
    )
    return {
        "label": label,
        "b": b,
        "s_q": s_q,
        "s_k": s_k,
        "is_varlen": is_varlen,
        "is_causal": is_causal,
        "h_q": h_q,
        "h_kv": h_kv,
        "dtype": dtype,
        "have_zero_seqlen_k": have_zero_seqlen_k,
        "seed": seed,
    }


def upstream_correctness_cases(dtype="bfloat16") -> list[dict]:
    """The full `correctness_cases` grid of the upstream test (2304 cases per dtype)."""
    return [
        _case(b, s_q, s_k, varlen, causal, h_q, h_kv, dtype=dtype)
        for b in (1, 2, 6, 64)
        for s_q in (1, 2, 4)
        for s_k in (20, 140, 4096)
        for h_q, h_kv in _H_PAIRS
        for varlen in (False, True)
        for causal in (False, True)
    ]


def upstream_corner_cases(dtype="bfloat16") -> list[dict]:
    """`corner_cases`: some requests have seqlen_k == 0."""
    return [
        _case(128, 2, 4096, True, causal, h_q, h_kv, dtype=dtype, have_zero_seqlen_k=True)
        for h_q, h_kv in _H_PAIRS
        for causal in (False, True)
    ]


def upstream_performance_cases(dtype="bfloat16") -> list[dict]:
    """`performance_cases`: b = 128, h_q = 128, h_kv = 1, varlen."""
    return [
        _case(128, s_q, s_k, True, causal, dtype=dtype)
        for causal in (False, True)
        for s_q in (1, 2)
        for s_k in (4096, 8192, 16384, 32768)
    ]


def _official_subset(dtype) -> list[dict]:
    """A cover of the upstream grid for the official runner: every (h_q, h_kv) pair, and every
    value of b, s_q, s_k, varlen and causal, each at least once with each other axis's extremes.
    The full grid runs through `upstream_correctness_cases` (scripted sweep)."""
    def head_key(c):
        return (c["h_q"], c["h_kv"], c["is_causal"])

    def shape_key(c):
        return (c["b"], c["s_q"], c["s_k"], c["is_varlen"], c["is_causal"])

    grid = upstream_correctness_cases(dtype)
    picked, heads, shapes = [], set(), set()
    # Shapes first, with the 128/1 and 64/8 heads (the DeepSeek and multi-KV-head extremes).
    for case in grid:
        key = (case["h_q"], case["h_kv"], *shape_key(case))
        if (case["h_q"], case["h_kv"]) in ((128, 1), (64, 8)) and key not in shapes:
            picked.append(case)
            shapes.add(key)
    # Then every head pair, on the longest varlen sequence of the largest batch.
    for case in grid:
        if case["b"] == 64 and case["s_q"] == 2 and case["s_k"] == 4096 and case["is_varlen"]:
            if head_key(case) not in heads:
                picked.append(case)
                heads.add(head_key(case))
    return list({c["label"]: c for c in picked}.values())


#: Edge shapes beyond the upstream grid: multiple m-blocks with a partial last block (s_q * h_q
#: / h_kv = 4 * 63 = 252 rows), one-token and one-page sequences, a sequence that ends exactly on
#: a page boundary, an odd page count (the IS_BLK0_LAST tail), b = 1 with one huge split count,
#: and a long sequence whose block ids exceed 2^20 (64-bit KV offsets through the TMA coordinate).
_EDGE = [
    _case(3, 4, 1, False, False, 63, 1, seed=11, prefix="edge_"),
    _case(5, 1, 64, False, False, 128, 1, seed=12, prefix="edge_"),
    _case(5, 2, 128, False, True, 128, 1, seed=13, prefix="edge_"),
    _case(7, 1, 192, False, False, 16, 1, seed=14, prefix="edge_"),
    _case(1, 1, 131072, False, False, 128, 1, seed=15, prefix="edge_"),
    _case(2, 4, 65, True, True, 63, 3, seed=16, prefix="edge_"),
    _case(132, 1, 1000, True, False, 16, 1, seed=17, prefix="edge_"),
]

CONFIGS = [
    *_official_subset("bfloat16"),
    *upstream_corner_cases("bfloat16")[:4],
    *_official_subset("float16")[:16],
    *_EDGE,
]

#: The upstream performance sweep (bf16, DeepSeek-V3 128-head shape) plus the TP-sharded head
#: counts DeepSeek-V3 decode actually runs (16 / 32 / 64 heads per rank).
BENCH_CONFIGS = [
    *upstream_performance_cases("bfloat16"),
    *[
        _case(b, s_q, s_k, True, False, h_q, 1, prefix="tp_")
        for h_q in (16, 32, 64)
        for b, s_q, s_k in ((64, 1, 4096), (128, 2, 8192))
    ],
]


def get_kernel(*, backend_config=None, **config):
    """The three PrimFuncs of one dtype specialization: scheduler, main, combine."""
    from ._sm90_dense_decode.launch import build_kernels

    config.pop("label", None)
    return list(build_kernels(config.get("dtype", "bfloat16"), backend_config=backend_config))


def prepare_data(**config) -> dict[str, Any]:
    from ._sm90_dense_decode.data import prepare

    config.pop("label", None)
    return prepare(**config)


def prepare_bench(*, backend_config=None, **config):
    """Compile the three kernels without touching CUDA."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    from ._sm90_dense_decode.launch import compile_kernels

    config.pop("label", None)
    state = {
        "config": dict(config),
        "executables": compile_kernels(
            config.get("dtype", "bfloat16"), backend_config=backend_config
        ),
    }
    return prepared_gpu_benchmark(run_gpu, state, backend_config=backend_config)


def run_test(*, backend_config=None, **config):
    """Scheduler metadata exact, outputs bitwise vs FlashMLA (whole tensors, NaN sentinel),
    and FlashMLA's own tolerances against the upstream test's PyTorch reference."""
    from ._sm90_dense_decode.data import check_case

    config.pop("label", None)
    return check_case(config, backend_config=backend_config, kernel=KERNEL_META["name"])


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
    from ._sm90_dense_decode.data import bench_against_flashmla

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

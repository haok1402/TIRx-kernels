# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM90 k-grouped contiguous FP8 GEMM -- `KGroupedContiguous`, the MoE weight-gradient path.

Reference: `deep_gemm.k_grouped_fp8_gemm_nt_contiguous` on SM90, which dispatches to
`sm90_k_grouped_fp8_gemm_1d1d` (`csrc/apis/gemm.hpp:394`) and JIT-compiles
`sm90_fp8_gemm_1d1d_impl`.  Groups are concatenated along K; group `g` stores A as `[M, K_g]`
and B as `[N, K_g]`, both K-major, at offset `sum(K_<g) * M` (resp. `* N`), and the kernel
accumulates `D[g] += A_g @ B_g^T` into an FP32 `D[G, M, N]` through TMA reduce-add stores.
Each group's K must be a multiple of 128; one FP32 scale per 128 K elements on both operands.

Upstream source: csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:153.
"""

from __future__ import annotations

KERNEL_META = {
    "name": "deepgemm_sm90_k_grouped_fp8_gemm_contiguous",
    "category": "ported.deepgemm",
    "runtime_cuda_archs": ["sm_90a"],
    "reference_requirements": (
        {
            "package": "deep-gemm",
            "git": {
                "url": "https://github.com/deepseek-ai/DeepGEMM.git",
                "commit": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
            },
            "import": "deep_gemm",
        },
    ),
}

#: `tests/generators.py::enumerate_k_grouped_contiguous` on SM90 (`k_alignment = 128`, K-major,
#: no PSUM); the first shape has many small groups, "for stressing the SM90 in-place tensor map
#: update".
_SHAPES = [
    (8, 768, 2048, 128),
    (4, 4096, 7168, 8192),
    (4, 7168, 2048, 8192),
    (8, 4096, 7168, 4096),
    (8, 7168, 2048, 4096),
    (16, 4096, 7168, 2048),
    (16, 7168, 2048, 2048),
]


def _case(num_groups, m, n, expected_k, *, seed=0, label=None, group_ks=None) -> dict:
    case = {
        "label": label or f"g{num_groups}_m{m}_n{n}_k{expected_k}",
        "num_groups": num_groups,
        "M": m,
        "N": n,
        "expected_k_per_group": expected_k,
        "seed": seed,
    }
    if group_ks is not None:
        case["group_ks"] = tuple(group_ks)
    return case


def _with_empty_group(case: dict) -> dict:
    """`enumerate_k_grouped_contiguous_test_variants`' empty-group variant: one group's K is 0."""
    from ._sm90_fp8_gemm_1d1d.data import make_ks

    real_ks, _ = make_ks(
        num_groups=case["num_groups"],
        expected_k_per_group=case["expected_k_per_group"],
        seed=case["seed"],
    )
    real_ks[case["seed"] % len(real_ks)] = 0
    return {**case, "label": case["label"] + "_empty", "group_ks": tuple(real_ks)}


_UPSTREAM = [_case(g, m, n, k, seed=i) for i, (g, m, n, k) in enumerate(_SHAPES, start=1)]

CONFIGS = [
    *_UPSTREAM,
    *[_with_empty_group(c) for c in _UPSTREAM],
    # Edge shapes.  Leading, trailing and interior empty groups (the scheduler's group walk and
    # the first tensor-map rewrite), a single non-empty group, group K that are not multiples of
    # 128 (zero-padded up), and M / N that leave partial and odd block counts (TMA-clipped tiles,
    # the multicast pairing fix-up with <= 4 groups).
    _case(4, 4096, 7168, 2048, seed=41, label="edge_lead_trail_empty", group_ks=(0, 1000, 3000, 0)),
    _case(8, 768, 2048, 128, seed=42, label="edge_one_nonempty", group_ks=(0, 0, 0, 0, 0, 77, 0, 0)),
    _case(4, 1000, 1100, 384, seed=43, label="edge_odd_mn", group_ks=(129, 300, 1, 640)),
    _case(2, 4160, 7232, 1024, seed=44, label="edge_odd_blocks_mc", group_ks=(1024, 896)),
    _case(1, 64, 16, 128, seed=45, label="edge_single_block", group_ks=(128,)),
    _case(32, 768, 2048, 128, seed=46, label="edge_many_groups"),
]

#: Benchmarks run the upstream shapes; the edge shapes are correctness-only.
BENCH_CONFIGS = list(_UPSTREAM)


def make_desc(
    *, num_groups, M, N, expected_k_per_group, seed=0, group_ks=None, num_sms=None
):
    """The descriptor `sm90_k_grouped_fp8_gemm_1d1d` builds ("Get config using max K")."""
    from tirx_kernels.runner import hardware_num_sms

    from ._sm90_fp8_gemm_1d1d.data import make_ks
    from ._sm90_fp8_gemm_1d1d.spec import SM90_NUM_SMS, GemmDesc, GemmType

    if num_sms is None:
        num_sms = hardware_num_sms(default=SM90_NUM_SMS)
    _, ks = make_ks(
        num_groups=num_groups,
        expected_k_per_group=expected_k_per_group,
        seed=seed,
        group_ks=group_ks,
    )
    return GemmDesc(
        GemmType.K_GROUPED_CONTIGUOUS,
        m=M,
        n=N,
        k=sum(ks),
        num_groups=num_groups,
        num_sms=num_sms,
        expected_m=M,
        expected_n=N,
        expected_k=max(ks),
        expected_num_groups=num_groups,
        compiled_dims="mn",
    )


def _spec_for(config: dict):
    from ._sm90_fp8_gemm_1d1d.spec import get_best_config, make_spec

    desc = make_desc(**config)
    return make_spec(desc, get_best_config(desc))


def get_kernel(*, backend_config=None, **config):
    from ._sm90_fp8_gemm_1d1d.kernel import build_kernel

    config.pop("label", None)
    return build_kernel(_spec_for(config), backend_config=backend_config)


def prepare_data(**config):
    from ._sm90_fp8_gemm_1d1d.data import prepare_k_grouped

    config.pop("label", None)
    return prepare_k_grouped(**config)


def prepare_bench(*, backend_config=None, **config):
    """Compile the exact DeepGEMM specialization without initializing CUDA."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    from ._sm90_fp8_gemm_1d1d.launch import compile_spec

    config.pop("label", None)
    state = {
        "config": dict(config),
        "executable": compile_spec(_spec_for(config), backend_config=backend_config),
    }
    return prepared_gpu_benchmark(run_gpu, state, backend_config=backend_config)


def run_test(*, backend_config=None, **config):
    """Compile, launch and compare the whole guarded D buffer bitwise against DeepGEMM."""
    import torch

    from ._sm90_fp8_gemm_1d1d.data import (
        assert_buffer_identical,
        calc_diff,
        deepgemm_launch_k_grouped,
        guarded_like,
    )
    from ._sm90_fp8_gemm_1d1d.launch import build_launch

    config.pop("label", None)
    data = prepare_data(**config)
    buffer, d = guarded_like(data["c"])
    launch = build_launch(_spec_for(config), data, d, backend_config=backend_config)
    if launch is not None:
        launch()
    torch.cuda.synchronize()
    # The launcher warms DeepGEMM up and restores D = C; one more launch gives the result.
    ref_launch, (ref_buffer, ref_d) = deepgemm_launch_k_grouped(data)
    ref_launch()
    torch.cuda.synchronize()
    detail = f"g={data['num_groups']} M={data['M']} N={data['N']} ks={data['ks']}"
    assert_buffer_identical(buffer, ref_buffer, kernel=KERNEL_META["name"], detail=detail)
    return {"diff": calc_diff(d, ref_d), "bitwise": True, "M": data["M"], "N": data["N"]}


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
    from ._sm90_fp8_gemm_1d1d.data import (
        bench_against_deepgemm,
        deepgemm_launch_k_grouped,
        guarded_like,
    )
    from ._sm90_fp8_gemm_1d1d.launch import build_launch

    config = {**prepared["config"], **config}
    config.pop("label", None)
    data = prepare_data(**config)
    _, d = guarded_like(data["c"])
    launch = build_launch(_spec_for(config), data, d, executable=prepared["executable"])
    return bench_against_deepgemm(
        launch,
        deepgemm_launch_k_grouped,
        data,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
        M=data["M"],
        N=data["N"],
        K=data["K"],
        num_groups=data["num_groups"],
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
    "make_desc",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]

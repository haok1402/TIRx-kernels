# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM90 m-grouped masked FP8 GEMM -- `MGroupedMasked`, the MoE decode path.

Reference: `deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked` on SM90, which dispatches to
`sm90_m_grouped_fp8_gemm_masked_1d2d` (`csrc/apis/gemm.hpp:283`) and JIT-compiles
`sm90_fp8_gemm_1d2d_impl`.  A `[G, max_m, K]` and D `[G, max_m, N]` are per-group padded;
only the first `masked_m[g]` rows of group `g` are valid.

Upstream source: csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:213.
"""

from __future__ import annotations

KERNEL_META = {
    "name": "deepgemm_sm90_m_grouped_fp8_gemm_masked",
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

#: `tests/generators.py::enumerate_m_grouped_masked` (`max_m = 4096`).  Its PSUM-layout
#: half runs the contiguous PSUM kernel, which the contiguous module covers.
_GROUPS = [(32, 192), (6, 1024), (32, 20), (6, 20)]
_NK = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]


def _case(num_groups, expected_m_per_group, n, k, *, max_m=4096, seed=0) -> dict:
    return {
        "label": f"g{num_groups}_m{expected_m_per_group}_n{n}_k{k}",
        "num_groups": num_groups,
        "expected_m_per_group": expected_m_per_group,
        "N": n,
        "K": k,
        "max_m": max_m,
        "seed": seed,
    }


CONFIGS = [
    _case(g, m, n, k, seed=i)
    for i, ((g, m), (n, k)) in enumerate([(gm, nk) for gm in _GROUPS for nk in _NK], start=1)
]


def make_desc(*, num_groups, expected_m_per_group, N, K, max_m=4096, seed=0, num_sms=None):
    """The descriptor `sm90_m_grouped_fp8_gemm_masked_1d2d` builds."""
    from tirx_kernels.runner import hardware_num_sms

    from ._sm90_fp8_gemm_1d2d.spec import GemmDesc, GemmType

    del seed
    if num_sms is None:
        num_sms = hardware_num_sms()
    return GemmDesc(
        GemmType.M_GROUPED_MASKED,
        m=max_m,
        n=N,
        k=K,
        num_groups=num_groups,
        num_sms=num_sms,
        expected_m=int(expected_m_per_group * 1.2),
        expected_n=N,
        expected_k=K,
        expected_num_groups=num_groups,
    )


def _spec_for(config: dict):
    from ._sm90_fp8_gemm_1d2d.spec import get_best_config, make_spec

    desc = make_desc(**config)
    return make_spec(desc, get_best_config(desc))


def get_kernel(*, backend_config=None, **config):
    from ._sm90_fp8_gemm_1d2d.kernel import build_kernel

    config.pop("label", None)
    return build_kernel(_spec_for(config), backend_config=backend_config)


def prepare_data(**config):
    from ._sm90_fp8_gemm_1d2d.data import prepare_m_grouped_masked

    config.pop("label", None)
    return prepare_m_grouped_masked(**config)


def _tirx_launch(data, config, executable=None, backend_config=None):
    from ._sm90_fp8_gemm_1d2d.launch import build_launch

    return build_launch(
        _spec_for(config),
        executable=executable,
        a=data["a"],
        b=data["b"],
        sfa=data["sfa"],
        sfb=data["sfb"],
        d=data["d"],
        grouped_layout=data["masked_m"],
        shape_m=data["M"],
        backend_config=backend_config,
    )


def prepare_bench(*, backend_config=None, **config):
    """Compile the exact DeepGEMM specialization without initializing CUDA."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    from ._sm90_fp8_gemm_1d2d.launch import compile_spec

    config.pop("label", None)
    state = {
        "config": dict(config),
        "executable": compile_spec(_spec_for(config), backend_config=backend_config),
    }
    return prepared_gpu_benchmark(run_gpu, state, backend_config=backend_config)


def run_test(*, backend_config=None, **config):
    """Compile, launch and compare the valid rows against DeepGEMM on identical operands."""
    import torch

    from ._sm90_fp8_gemm_1d2d.data import (
        assert_within_threshold,
        deepgemm_launch_m_grouped_masked,
        masked_slice_diff,
    )

    config.pop("label", None)
    data = prepare_data(**config)
    launch = _tirx_launch(data, config, backend_config=backend_config)
    data["d"].zero_()
    launch()
    torch.cuda.synchronize()
    _, deepgemm_out = deepgemm_launch_m_grouped_masked(data)
    torch.cuda.synchronize()
    diff = masked_slice_diff(data["d"], deepgemm_out, data["masked_m"])
    return assert_within_threshold(
        diff,
        data,
        kernel=KERNEL_META["name"],
        detail=f"g={data['num_groups']} max_m={data['M']} N={data['N']} K={data['K']}",
        M=data["M"],
    )


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
    from ._sm90_fp8_gemm_1d2d.data import bench_against_deepgemm, deepgemm_launch_m_grouped_masked

    config = {**prepared["config"], **config}
    config.pop("label", None)
    data = prepare_data(**config)
    return bench_against_deepgemm(
        _tirx_launch(data, config, executable=prepared["executable"]),
        deepgemm_launch_m_grouped_masked,
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

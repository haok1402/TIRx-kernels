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


def _edge(label, num_groups, expected_m_per_group, n, k, group_ms, *, seed=0) -> dict:
    case = _case(num_groups, expected_m_per_group, n, k, seed=seed)
    case.update(label=f"edge_{label}", group_ms=tuple(group_ms))
    return case


def _coverage_case(num_groups, m, n, k, seed):
    """The export's masked call: `max_m = max(128, align(4m, 128))`, `expected_m = m`, every group m
    rows (`expected_m_per_group` is chosen so that `int(1.2 * it) == m`)."""
    expected = next(e for e in range(1, 4 * m + 2) if int(e * 1.2) == m)
    case = _case(num_groups, expected, n, k, seed=seed)
    case.update(
        label=f"cov_g{num_groups}_m{m}_n{n}_k{k}",
        max_m=max(128, (4 * m + 127) // 128 * 128),
        group_ms=(m,) * num_groups,
    )
    return case


CONFIGS = [
    *[
        _case(g, m, n, k, seed=i)
        for i, ((g, m), (n, k)) in enumerate([(gm, nk) for gm in _GROUPS for nk in _NK], start=1)
    ],
    # Coverage extension: the exported masked instantiations (decode-sized groups at EP 32-64).
    *[
        _coverage_case(*c, seed=300 + i)
        for i, c in enumerate(
            [
                # (num_groups, rows per group, N, K) -> class
                (2, 1, 1536, 2048),  # 64x32
                (2, 1, 3072, 4096),  # 64x48, runtime predicate
                (2, 1, 4096, 1536),  # 64x64
                (8, 1, 7168, 2048),  # 64x112, multicast on B
                (4, 128, 7168, 2048),  # 128x112, multicast on A
                (8, 128, 4096, 7168),  # 128x128, multicast on A
            ]
        )
    ],
    # 64x112 with multicast on B taken: groups of more than one 64-row m block.
    _edge("bn112_multicast_b", 8, 54, 7168, 2048, (128, 100, 64, 65, 200, 10, 0, 70), seed=54),
    # Edge shapes: empty groups (the scheduler skips them), a full `max_m` group, one row, and
    # row counts that are not BLOCK_M multiples (partial and fully invalid WGMMA row bands).
    _edge("empty_one_full", 4, 300, 4096, 4096, (0, 1, 4096, 777), seed=51),
    _edge("mostly_empty", 8, 20, 4096, 4096, (0, 0, 0, 5, 0, 0, 0, 0), seed=52),
    _edge("odd_m_blocks", 6, 1024, 4096, 7168, (64, 100, 129, 1000, 1200, 900), seed=53),
]


#: Benchmarks run the upstream shapes; the edge shapes are correctness-only.
BENCH_CONFIGS = [c for c in CONFIGS if not c["label"].startswith("edge_")]


def make_desc(
    *, num_groups, expected_m_per_group, N, K, max_m=4096, seed=0, group_ms=None, num_sms=None
):
    """The descriptor `sm90_m_grouped_fp8_gemm_masked_1d2d` builds."""
    from tirx_kernels.runner import hardware_num_sms

    from ._sm90_fp8_gemm_1d2d.spec import GemmDesc, GemmType

    del seed, group_ms
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
        assert_whole_output_identical,
        assert_within_threshold,
        deepgemm_launch_m_grouped_masked,
        masked_slice_diff,
        sentinel_like,
    )

    config.pop("label", None)
    data = prepare_data(**config)
    launch = _tirx_launch(data, config, backend_config=backend_config)
    data["d"].copy_(sentinel_like(data["d"]))
    launch()
    torch.cuda.synchronize()
    _, deepgemm_out = deepgemm_launch_m_grouped_masked(data, out=sentinel_like(data["d"]))
    torch.cuda.synchronize()
    diff = masked_slice_diff(data["d"], deepgemm_out, data["masked_m"])
    result = assert_within_threshold(
        diff,
        data,
        kernel=KERNEL_META["name"],
        detail=f"g={data['num_groups']} max_m={data['M']} N={data['N']} K={data['K']}",
        M=data["M"],
    )
    assert_whole_output_identical(
        data["d"], deepgemm_out, kernel=KERNEL_META["name"], detail=f"M={data['M']} N={data['N']}"
    )
    return result


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

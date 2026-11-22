# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM90 FP8 1D1D GEMM with FP32 accumulation -- `GemmType::Normal`, the dense weight-gradient path.

Reference: `deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=d, recipe=(1, 1, 128))` on SM90, which
dispatches to `sm90_fp8_gemm_1d1d` when the B scales are per token (`csrc/apis/gemm.hpp:111`,
`gran_n == 1`) and JIT-compiles the same `sm90_fp8_gemm_1d1d_impl` as the k-grouped entry:
`D[M, N] += A[M, K] @ B[N, K]^T` in FP32, one FP32 scale per 128 K elements on both operands.

Upstream source: csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:79.
"""

from __future__ import annotations

KERNEL_META = {
    "name": "deepgemm_sm90_fp8_gemm_1d1d",
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

#: `tests/generators.py::enumerate_normal` on SM90: the backward pass's FP32 + accumulate wgrad
#: (`yield override_kernel_type, ..., n, m, k, ..., True, torch.float`, Kernel1D1D, K-major),
#: i.e. `(M, N, K) = (n, 4096, k)` over `bf16_output_nk`.
_WGRAD_NK = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]


def _case(m, n, k, *, seed=0, label=None) -> dict:
    return {"label": label or f"m{m}_n{n}_k{k}", "M": m, "N": n, "K": k, "seed": seed}


_UPSTREAM = [_case(n, 4096, k, seed=i) for i, (n, k) in enumerate(_WGRAD_NK, start=1)]

CONFIGS = [
    *_UPSTREAM,
    # Edge shapes: partial M / N blocks (TMA-clipped tiles), one K block, odd block counts with a
    # multicast-capable shape, and a single tile.
    _case(1000, 1100, 384, seed=41, label="edge_odd_mn"),
    _case(4160, 7232, 128, seed=42, label="edge_one_k_block"),
    _case(6208, 2056, 2048, seed=43, label="edge_odd_blocks_mc"),
    _case(64, 16, 128, seed=44, label="edge_single_block"),
]

BENCH_CONFIGS = list(_UPSTREAM)


def make_desc(*, M, N, K, seed=0, num_sms=None):
    """The descriptor `sm90_fp8_gemm_1d1d` builds (`compiled_dims = "nk"`)."""
    from tirx_kernels.runner import hardware_num_sms

    from ._sm90_fp8_gemm_1d1d.spec import SM90_NUM_SMS, GemmDesc, GemmType

    del seed
    if num_sms is None:
        num_sms = hardware_num_sms(default=SM90_NUM_SMS)
    return GemmDesc(GemmType.NORMAL, m=M, n=N, k=K, num_groups=1, num_sms=num_sms, compiled_dims="nk")


def _spec_for(config: dict):
    from ._sm90_fp8_gemm_1d1d.spec import get_best_config, make_spec

    desc = make_desc(**config)
    return make_spec(desc, get_best_config(desc))


def get_kernel(*, backend_config=None, **config):
    from ._sm90_fp8_gemm_1d1d.kernel import build_kernel

    config.pop("label", None)
    return build_kernel(_spec_for(config), backend_config=backend_config)


def prepare_data(**config):
    from ._sm90_fp8_gemm_1d1d.data import prepare_normal

    config.pop("label", None)
    return prepare_normal(**config)


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
        deepgemm_launch_normal,
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
    ref_launch, (ref_buffer, ref_d) = deepgemm_launch_normal(data)
    ref_launch()
    torch.cuda.synchronize()
    detail = f"M={data['M']} N={data['N']} K={data['K']}"
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
    from ._sm90_fp8_gemm_1d1d.data import bench_against_deepgemm, deepgemm_launch_normal, guarded_like
    from ._sm90_fp8_gemm_1d1d.launch import build_launch

    config = {**prepared["config"], **config}
    config.pop("label", None)
    data = prepare_data(**config)
    _, d = guarded_like(data["c"])
    launch = build_launch(_spec_for(config), data, d, executable=prepared["executable"])
    return bench_against_deepgemm(
        launch,
        deepgemm_launch_normal,
        data,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
        M=data["M"],
        N=data["N"],
        K=data["K"],
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

# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM90 m-grouped contiguous FP8 GEMM -- `MGroupedContiguous[WithPsumLayout]`, the MoE path.

Reference: `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous` on SM90, which dispatches to
`sm90_m_grouped_fp8_gemm_contiguous_1d2d` (`csrc/apis/gemm.hpp:217`) and JIT-compiles
`sm90_fp8_gemm_1d2d_impl`.  FP8 E4M3 A `[M, K]` with per-token FP32 scales, FP8 B
`[G, N, K]` with per-128x128-block FP32 scales, BF16 D `[M, N]`.  `use_psum_layout` makes the
scheduler walk per-group end offsets instead of a per-row group id.

Upstream source: csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:142.
"""

from __future__ import annotations

KERNEL_META = {
    "name": "deepgemm_sm90_m_grouped_fp8_gemm_contiguous",
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

#: `tests/generators.py::enumerate_m_grouped_contiguous` on SM90: K-major A and B only,
#: both PSUM-layout settings, no `ensure_zero_padding` sweep (that is SM100-only).
_GROUPS = [(4, 8192), (8, 4096)]
_NK = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]


def _case(num_groups, expected_m_per_group, n, k, *, use_psum=False, seed=0) -> dict:
    label = f"g{num_groups}_m{expected_m_per_group}_n{n}_k{k}" + ("_psum" if use_psum else "")
    return {
        "label": label,
        "num_groups": num_groups,
        "expected_m_per_group": expected_m_per_group,
        "N": n,
        "K": k,
        "use_psum_layout": use_psum,
        "seed": seed,
    }


CONFIGS = [
    _case(g, m, n, k, use_psum=psum, seed=i)
    for i, (psum, (g, m), (n, k)) in enumerate(
        [(psum, gm, nk) for psum in (False, True) for gm in _GROUPS for nk in _NK], start=1
    )
]


def make_desc(
    *, num_groups, expected_m_per_group, N, K, use_psum_layout=False, seed=0, num_sms=None
):
    """The descriptor `sm90_m_grouped_fp8_gemm_contiguous_1d2d` builds (`expected_m = m`,
    `expected_num_groups = 1`: the source test never passes `expected_m_for_psum_layout`)."""
    from tirx_kernels.runner import hardware_num_sms

    from ._sm90_fp8_gemm_1d2d.spec import MK_ALIGNMENT, GemmDesc, GemmType, align_up
    from ._sm100_fp8_fp4_gemm_1d1d.spec import make_actual_ms

    if num_sms is None:
        num_sms = hardware_num_sms()
    m = sum(
        align_up(x, MK_ALIGNMENT) for x in make_actual_ms(num_groups, expected_m_per_group, seed)
    )
    return GemmDesc(
        GemmType.M_GROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT
        if use_psum_layout
        else GemmType.M_GROUPED_CONTIGUOUS,
        m=m,
        n=N,
        k=K,
        num_groups=num_groups,
        num_sms=num_sms,
        expected_m=m,
        expected_n=N,
        expected_k=K,
        expected_num_groups=1,
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
    from ._sm90_fp8_gemm_1d2d.data import prepare_m_grouped_contiguous

    config.pop("label", None)
    return prepare_m_grouped_contiguous(**config)


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
        grouped_layout=data["grouped_layout"],
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
    """Compile, launch and compare against DeepGEMM on byte-identical operands."""
    import torch

    from ._sm90_fp8_gemm_1d2d.data import (
        assert_within_threshold,
        calc_diff,
        deepgemm_launch_m_grouped_contiguous,
        psum_slice_diff,
    )

    config.pop("label", None)
    data = prepare_data(**config)
    launch = _tirx_launch(data, config, backend_config=backend_config)
    data["d"].zero_()
    launch()
    torch.cuda.synchronize()
    _, deepgemm_out = deepgemm_launch_m_grouped_contiguous(data)
    torch.cuda.synchronize()
    if data["use_psum_layout"]:
        diff = psum_slice_diff(data["d"], deepgemm_out, data["grouped_layout"], data["alignment"])
    else:
        # Rows marked -1 (alignment padding) are not computed by either side.
        valid = data["grouped_layout"] >= 0
        diff = calc_diff(data["d"][valid], deepgemm_out[valid])
    return assert_within_threshold(
        diff,
        data,
        kernel=KERNEL_META["name"],
        detail=f"g={data['num_groups']} M={data['M']} N={data['N']} K={data['K']} psum={data['use_psum_layout']}",
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
    from ._sm90_fp8_gemm_1d2d.data import (
        bench_against_deepgemm,
        deepgemm_launch_m_grouped_contiguous,
    )

    config = {**prepared["config"], **config}
    config.pop("label", None)
    data = prepare_data(**config)
    return bench_against_deepgemm(
        _tirx_launch(data, config, executable=prepared["executable"]),
        deepgemm_launch_m_grouped_contiguous,
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

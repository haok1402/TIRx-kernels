# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90_fp8_gemm_1d2d_impl` (`deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`).

One persistent kernel instantiation per `KernelSpec`: a TMA producer warpgroup and one or two
WGMMA math warpgroups over an `kNumStages`-deep shared-memory ring, FP32 promotion of each
128-wide K block with per-token A scales and per-block B scales, and an STSM + TMA-store
epilogue.  The block scheduler is `deep_gemm/scheduler/gemm.cuh::Scheduler`.
"""

import tirx_kernels.tirx_lite as txl

from .spec import KernelSpec


def build_kernel(spec: KernelSpec, *, backend_config=None):
    """The TIRx `PrimFunc` for one `sm90_fp8_gemm_1d2d_impl` template instantiation."""
    num_threads = spec.num_tma_threads + spec.num_math_threads

    @txl.kernel()
    def sm90_fp8_gemm_1d2d(
        sfb: txl.Tensor((txl.int64(),), "float32"),
        grouped_layout: txl.Tensor((txl.int64(),), "int32"),
        shape_m: "int32",
        shape_n: "int32",
        shape_k: "int32",
        tensor_map_a: txl.Ptr(txl.TensorMap),
        tensor_map_b: txl.Ptr(txl.TensorMap),
        tensor_map_d: txl.Ptr(txl.TensorMap),
        tensor_map_sfa: txl.Ptr(txl.TensorMap),
    ):
        # `LaunchArgs(num_sms, num_threads, smem_size, cluster)`, `__launch_bounds__(N, 1)`.
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(
                grid=spec.num_sms,
                block=num_threads,
                cluster=(spec.num_tma_multicast,) if spec.num_tma_multicast > 1 else None,
            ),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=backend_config,
        )
        # ---- sm90_fp8_gemm_1d2d_impl body starts here ----

    return sm90_fp8_gemm_1d2d.func

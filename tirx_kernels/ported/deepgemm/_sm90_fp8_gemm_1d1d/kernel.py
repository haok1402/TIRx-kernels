# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90_fp8_gemm_1d1d_impl` (`deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`)."""

import tirx_kernels.tirx_lite as txl

from .spec import KernelSpec


def build_kernel(spec: KernelSpec, *, backend_config=None):
    """The TIRx `PrimFunc` for one `sm90_fp8_gemm_1d1d_impl` template instantiation."""

    @txl.kernel()
    def sm90_fp8_gemm_1d1d(
        gmem_a: txl.Tensor((txl.int64(),), "float8_e4m3fn"),
        gmem_b: txl.Tensor((txl.int64(),), "float8_e4m3fn"),
        grouped_layout: txl.Tensor((txl.int64(),), "int32"),
        tensor_map_buffer: txl.Tensor((txl.int64(),), "uint8"),
        shape_m: "int32",
        shape_n: "int32",
        shape_k: "int32",
        tensor_map_a_base: txl.Ptr(txl.TensorMap),
        tensor_map_b_base: txl.Ptr(txl.TensorMap),
        tensor_map_sfa: txl.Ptr(txl.TensorMap),
        tensor_map_sfb: txl.Ptr(txl.TensorMap),
        tensor_map_cd: txl.Ptr(txl.TensorMap),
    ):
        # KERNEL SKETCH START: sm90_fp8_gemm_1d1d_impl body goes here.
        pass

    return sm90_fp8_gemm_1d1d.func

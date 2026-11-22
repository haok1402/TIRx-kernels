# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90_fp8_paged_mqa_logits` (`deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`)."""

import tirx_kernels.tirx_lite as txl


def build_kernel(spec, *, backend_config=None):
    """The TIRx `PrimFunc` for one `sm90_fp8_paged_mqa_logits` template instantiation."""

    @txl.kernel()
    def sm90_fp8_paged_mqa_logits(
        batch_size: "int32",
        logits_stride: "int32",
        block_table_stride: "int32",
        context_lens: txl.Tensor((txl.int64(),), "int32"),
        logits: txl.Tensor((txl.int64(),), spec.logits_dtype),
        block_table: txl.Tensor((txl.int64(),), "int32"),
        schedule_meta: txl.Tensor((txl.int64(),), "int32"),
        tensor_map_q: txl.Ptr(txl.TensorMap),
        tensor_map_kv: txl.Ptr(txl.TensorMap),
        tensor_map_kv_scales: txl.Ptr(txl.TensorMap),
        tensor_map_weights: txl.Ptr(txl.TensorMap),
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=spec.num_sms, block=128 + 512),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=backend_config,
        )
        # KERNEL BODY START: sm90_fp8_paged_mqa_logits
        pass

    return sm90_fp8_paged_mqa_logits.func

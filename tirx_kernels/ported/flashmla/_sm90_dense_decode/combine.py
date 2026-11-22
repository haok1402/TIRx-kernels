# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors


"""TIRx port of `smxx::decode::flash_fwd_mla_combine_kernel<ElementT, 512, 8, MAX_SPLITS, 256>`
(`csrc/smxx/decode/combine/combine.cu`), without an attention sink (dense decode passes none).
Launched with programmatic stream serialization after the main kernel.
"""

import tirx_kernels.tirx_lite as txl

HEAD_DIM_V = 512
BLOCK_SIZE_M = 8
NUM_THREADS = BLOCK_SIZE_M * 32
#: `MLA_NUM_SPLITS_SWITCH`.
MAX_SPLITS_BUCKETS = (32, 64, 96, 128, 160)


def max_splits_for(num_sm_parts: int) -> int:
    for bucket in MAX_SPLITS_BUCKETS:
        if num_sm_parts <= bucket:
            return bucket
    raise AssertionError(f"num_sm_parts = {num_sm_parts} > 160 is not supported by FlashMLA")


def build_combine_kernel(dtype: str, max_splits: int, *, backend_config=None):
    @txl.kernel()
    def flash_fwd_mla_combine(
        lse: txl.Tensor((txl.int64(),), "float32"),
        out: txl.Tensor((txl.int64(),), dtype),
        lse_accum: txl.Tensor((txl.int64(),), "float32"),
        o_accum: txl.Tensor((txl.int64(),), "float32"),
        num_splits: txl.Tensor((txl.int64(),), "int32"),
        stride_lse_b: "int32",
        stride_lse_s_q: "int32",
        stride_o_b: "int32",
        stride_o_s_q: "int32",
        stride_o_h_q: "int32",
        stride_lse_accum_split: "int32",
        stride_lse_accum_s_q: "int32",
        stride_o_accum_split: "int32",
        stride_o_accum_s_q: "int32",
        stride_o_accum_h_q: "int32",
        b: "int32",
        s_q: "int32",
        h_q: "int32",
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(
                grid=(b * s_q, 1, (h_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M),
                block=NUM_THREADS,
                programmatic_stream_serialization=True,
            ),
            backend_config=backend_config,
        )
        # ---- kernel body starts here (combine.cu:18) ----

    return flash_fwd_mla_combine.func

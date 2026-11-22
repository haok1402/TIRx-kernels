# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors


"""TIRx port of `sm90::flash_fwd_splitkv_mla_kernel<Traits<InputT>, TmaParams>`
(`csrc/sm90/decode/dense/splitkv_mla.cuh`, `traits.h`, `config.h`).

Grid `(ceil(q_seq_per_hk / 64), h_k, num_sm_parts)`, 256 threads (two warpgroups, no producer
warpgroup), `__launch_bounds__(256, 1, 1)`, one `SharedMemoryPlan` of dynamic shared memory, PDL.
"""

import tirx_kernels.tirx_lite as txl

BLOCK_SIZE_M = 64
PAGE_BLOCK_SIZE = 64
HEAD_DIM_K = 576
HEAD_DIM_V = 512
NUM_THREADS = 256


def build_main_kernel(dtype: str, *, backend_config=None):
    @txl.kernel()
    def flash_fwd_splitkv_mla(
        seqlens_k: txl.Tensor((txl.int64(),), "int32"),
        block_table: txl.Tensor((txl.int64(),), "int32"),
        tile_scheduler_metadata: txl.Tensor((txl.int64(),), "int32"),
        num_splits: txl.Tensor((txl.int64(),), "int32"),
        softmax_lse: txl.Tensor((txl.int64(),), "float32"),
        softmax_lseaccum: txl.Tensor((txl.int64(),), "float32"),
        oaccum: txl.Tensor((txl.int64(),), "float32"),
        b: "int32",
        s_q: "int32",
        q_seq_per_hk: "int32",
        q_head_per_hk: "int32",
        h_k: "int32",
        is_causal: "int32",
        scale_softmax_log2: "float32",
        block_table_batch_stride: "int32",
        num_sm_parts: "int32",
        tma_q: txl.Ptr(txl.TensorMap),
        tma_k: txl.Ptr(txl.TensorMap),
        tma_o: txl.Ptr(txl.TensorMap),
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(
                grid=((q_seq_per_hk + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, h_k, num_sm_parts),
                block=NUM_THREADS,
                programmatic_stream_serialization=True,
            ),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=backend_config,
        )
        # ---- kernel body starts here (splitkv_mla.cuh:961) ----

    return flash_fwd_splitkv_mla.func

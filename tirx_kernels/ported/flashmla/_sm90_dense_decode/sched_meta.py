# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors


"""TIRx port of `smxx::decode::get_mla_metadata_kernel`
(`csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`), dense (`topk == -1`)
path: one warp, dynamic shared memory of `4 * (5 * b + 1)` bytes, `__launch_bounds__(32, 1, 1)`.
"""

import tirx_kernels.tirx_lite as txl

#: `dense_attn_decode_interface`'s `GetDecodeSchedMetaParams` (`block_size_n`,
#: `fixed_overhead_num_blocks`).
BLOCK_SIZE_N = 64
FIXED_OVERHEAD_NUM_BLOCKS = 5


def build_sched_meta_kernel(*, backend_config=None):
    @txl.kernel()
    def get_mla_metadata(
        seqlens_k: txl.Tensor((txl.int64(),), "int32"),
        tile_scheduler_metadata: txl.Tensor((txl.int64(),), "int32"),
        num_splits: txl.Tensor((txl.int64(),), "int32"),
        b: "int32",
        num_sm_parts: "int32",
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=1, block=32),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=backend_config,
        )
        # ---- kernel body starts here (get_decoding_sched_meta.cu:11) ----

    return get_mla_metadata.func

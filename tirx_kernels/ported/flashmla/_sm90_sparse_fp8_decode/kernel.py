# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<
KernelTemplate<MODEL_TYPE, NUM_HEADS>>` (`csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`)."""

import tirx_kernels.tirx_lite as txl


def build_main_kernel(model_type: str, num_heads: int, *, backend_config=None):
    """The TIRx `PrimFunc` for `KernelTemplate<model_type, num_heads>`."""

    @txl.kernel()
    def flash_fwd_splitkv_mla_fp8_sparse():
        # ---- TIRx implementation starts here ----
        pass

    return flash_fwd_splitkv_mla_fp8_sparse.func

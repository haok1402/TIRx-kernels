# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ 0251105a), Copyright (c) 2024, Jay Shah,
# Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of the three SM90 backward kernels of flash-attention ``hopper/``:

- ``flash::FlashAttnBwdPreprocess`` (``flash_bwd_preprocess_kernel.h``);
- ``flash::FlashAttnBwdSm90`` (``flash_bwd_kernel_sm90.h``, ``mainloop_bwd_sm90_tma_gmma_ws.hpp``,
  ``epilogue_bwd.hpp``, ``tile_scheduler.hpp``, ``mask.h``);
- ``flash::FlashAttnBwdPostprocessConvertdQ`` (``flash_bwd_postprocess_kernel.h``).
"""

import tirx_kernels.tirx_lite as txl


def build_preprocess(head_dim, block_m, *, backend_config=None):
    @txl.kernel()
    def flash_bwd_preprocess():
        # ---- kernel body starts here (FlashAttnBwdPreprocess::operator()) ----
        pass

    return flash_bwd_preprocess


def build_main_kernel(head_dim, is_causal, *, backend_config=None):
    @txl.kernel()
    def flash_bwd_sm90():
        # ---- kernel body starts here (FlashAttnBwdSm90::operator()) ----
        pass

    return flash_bwd_sm90


def build_postprocess(head_dim, is_causal, *, backend_config=None):
    @txl.kernel()
    def flash_bwd_convert_dq():
        # ---- kernel body starts here (FlashAttnBwdPostprocessConvertdQ::operator()) ----
        pass

    return flash_bwd_convert_dq

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Shared TIRx port of DeepGEMM's `sm90_fp8_paged_mqa_logits` implementation.

Submodules: `launch` (TMA descriptors) and `kernel` (the `PrimFunc` body).  The public
kernel module is :mod:`tirx_kernels.ported.deepgemm.sm90_paged_mqa_logits_fp8`.
"""

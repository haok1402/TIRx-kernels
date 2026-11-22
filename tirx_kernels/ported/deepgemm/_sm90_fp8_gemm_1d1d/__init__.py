# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Shared TIRx port of DeepGEMM's SM90 `sm90_fp8_gemm_1d1d` implementation.

Submodules: `spec` (descriptors, heuristics, kernel specialization), `data` (operands and the
DeepGEMM reference), `kernel` (the `PrimFunc`), `launch` (TMA descriptors and launch).  The
public modules `sm90_k_grouped_fp8_gemm_contiguous` and `sm90_fp8_gemm_1d1d` under
:mod:`tirx_kernels.ported.deepgemm` pin one descriptor each.
"""

from .spec import GemmDesc, GemmType, KernelSpec, get_best_config, make_spec

__all__ = ["GemmDesc", "GemmType", "KernelSpec", "get_best_config", "make_spec"]

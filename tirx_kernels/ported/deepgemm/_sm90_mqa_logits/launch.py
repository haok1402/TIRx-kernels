# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TMA descriptors of `sm90_fp8_mqa_logits` (csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp:96-102).

Each is `make_tma_2d_desc` with the arguments the source passes; the encoder is shared with the
SM100 1D1D GEMM port.
"""

from __future__ import annotations

from .._sm100_fp8_fp4_gemm_1d1d.spec import _encode_2d, tma_aligned_size

BLOCK_KV = 256


def make_tensor_maps(data, spec):
    """(q, kv, kv_scales, weights), the kernel's TensorMap argument order."""
    config = data["config"]
    s, skv = config.seq_len, config.seq_len_kv
    h, d, bq = spec.num_heads, spec.head_dim, spec.block_q
    kv_fp8, kv_scales = data["kv_in"]
    weights = data["weights"]
    q = data["q_in"].reshape(s * h, d)
    tensor_map_q = _encode_2d(
        q, gmem_inner=d, gmem_outer=s * h, smem_inner=d, smem_outer=bq * h,
        gmem_outer_stride=d, swizzle_mode=d,
    )
    tensor_map_kv = _encode_2d(
        kv_fp8, gmem_inner=d, gmem_outer=skv, smem_inner=d, smem_outer=BLOCK_KV,
        gmem_outer_stride=d, swizzle_mode=d,
    )
    tensor_map_kv_scales = _encode_2d(
        kv_scales, gmem_inner=tma_aligned_size(skv, int(kv_scales.element_size())),
        gmem_outer=1, smem_inner=BLOCK_KV, smem_outer=1, gmem_outer_stride=0, swizzle_mode=0,
    )
    tensor_map_weights = _encode_2d(
        weights, gmem_inner=h, gmem_outer=s, smem_inner=h, smem_outer=bq,
        gmem_outer_stride=int(weights.stride(0)), swizzle_mode=0,
    )
    return tensor_map_q, tensor_map_kv, tensor_map_kv_scales, tensor_map_weights

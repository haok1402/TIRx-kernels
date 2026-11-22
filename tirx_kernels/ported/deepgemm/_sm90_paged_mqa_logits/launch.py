# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TMA descriptors of `sm90_fp8_paged_mqa_logits` (csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp:337-352).

The fused KV cache `[num_blocks, block_kv, 1, head_dim + 4]` (uint8) is split exactly as
`fp8_fp4_paged_mqa_logits` does (csrc/apis/attention.hpp:271-283): FP8 values
`[num_blocks, block_kv, head_dim]` with block stride `block_kv * (head_dim + 4)` bytes, and the
FP32 scales right after a block's values.
"""

from __future__ import annotations

import torch

from .._sm100_fp8_fp4_gemm_1d1d.spec import _encode_2d, _encode_3d

BLOCK_KV = 64


def make_tensor_maps(data, spec):
    """(q, kv, kv_scales, weights), the kernel's TensorMap argument order."""
    config = data["config"]
    b, n, h, d = config.batch_size, config.next_n, spec.num_heads, spec.head_dim
    next_n_atom = 2 if n >= 2 else 1
    fused = data["fused_kv_cache"]
    num_kv_blocks = fused.shape[0]
    block_stride = int(fused.stride(0))  # bytes
    kv = torch.as_strided(fused.view(torch.uint8), (num_kv_blocks, BLOCK_KV, d), (block_stride, d, 1))
    q = data["q"].view(torch.uint8).reshape(b * n * h, d)
    tensor_map_q = _encode_2d(
        q, gmem_inner=d, gmem_outer=b * n * h, smem_inner=d, smem_outer=next_n_atom * h,
        gmem_outer_stride=int(data["q"].stride(2)), swizzle_mode=d,
    )
    tensor_map_kv = _encode_3d(
        kv, gmem_inner=d, gmem_outer=BLOCK_KV, batch=num_kv_blocks, smem_inner=d, smem_outer=BLOCK_KV,
        stride_outer=d, stride_batch=block_stride, swizzle_mode=d,
    )
    flat = fused.view(torch.uint8).reshape(-1)
    scales = flat[BLOCK_KV * d :].view(torch.float32) if (flat.numel() - BLOCK_KV * d) % 4 == 0 else None
    if scales is None:
        raise AssertionError("fused KV cache size is not a multiple of 4 past the first block's values")
    tensor_map_kv_scales = _encode_2d(
        scales, gmem_inner=BLOCK_KV, gmem_outer=num_kv_blocks, smem_inner=BLOCK_KV, smem_outer=1,
        gmem_outer_stride=block_stride // 4, swizzle_mode=0,
    )
    weights = data["weights"]
    tensor_map_weights = _encode_2d(
        weights, gmem_inner=h, gmem_outer=b * n, smem_inner=h, smem_outer=next_n_atom,
        gmem_outer_stride=int(weights.stride(0)), swizzle_mode=0,
    )
    return tensor_map_q, tensor_map_kv, tensor_map_kv_scales, tensor_map_weights

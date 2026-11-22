# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TMA descriptors, compilation and launch of the three dense-decode kernels.

The descriptors are the ones `run_flash_splitkv_mla_kernel` builds with `cute::make_tma_copy`
(`splitkv_mla.cuh:1281-1328`): Q and O as `(q_seq_per_hk, d, h_k, b)`, the paged K cache as
`(PAGE_BLOCK_SIZE, HEAD_DIM_K, h_k, num_blocks)`, each copied in 64 x 64 boxes into 128B-swizzled
shared memory.  Encoded with `cuTensorMapEncodeTiled` outside the timed launch.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any

from tirx_kernels.runner import cache_backend_config

from .combine import max_splits_for

#: CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B (what CuTe's make_tma_copy
#: encodes for these SW128 boxes; checked against the export).
_SWIZZLE_128B = 3
_L2_PROMOTION = 2


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode(tensor, dims, strides_bytes, box, *, swizzle=_SWIZZLE_128B, l2=_L2_PROMOTION):
    import tvm_ffi

    desc = _AlignedTensorMap()
    rank = len(dims)
    dtype = {2: str(tensor.dtype).removeprefix("torch.")}[tensor.element_size()]
    tvm_ffi.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        dtype,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides_bytes,
        *box,
        *((1,) * rank),
        0,
        swizzle,
        l2,
        0,
    )
    return desc


def make_tensor_maps(p):
    """`tma_Q`, `tma_K`, `tma_O` (innermost dimension first)."""
    q, k, o = p["q_folded"], p["kcache"], p["out"]
    if k.size(0) == 0:
        # FlashMLA cannot take this either: its K TensorMap encode fails and the kernel faults.
        raise ValueError(
            "the paged KV cache has no blocks (e.g. a batch whose every sequence is empty); "
            "this is outside FlashMLA's dense-decode domain"
        )
    es = q.element_size()
    tma_q = _encode(
        q,
        (q.size(3), q.size(1), q.size(2), q.size(0)),
        (q.stride(1) * es, q.stride(2) * es, q.stride(0) * es),
        (64, 64, 1, 1),
    )
    tma_k = _encode(
        k,
        (k.size(3), k.size(1), k.size(2), k.size(0)),
        (k.stride(1) * es, k.stride(2) * es, k.stride(0) * es),
        (64, 64, 1, 1),
    )
    # out is (b, h_k, q_seq_per_hk, d_v): o_row_stride = stride(2), o_head_stride = stride(1).
    tma_o = _encode(
        o,
        (o.size(3), o.size(2), o.size(1), o.size(0)),
        (o.stride(2) * es, o.stride(1) * es, o.stride(0) * es),
        (64, 64, 1, 1),
    )
    return tma_q, tma_k, tma_o


def build_kernels(dtype: str, max_splits: int | None = None, *, backend_config=None):
    from tirx_kernels.runner import hardware_num_sms

    from .combine import build_combine_kernel
    from .kernel import build_main_kernel
    from .sched_meta import build_sched_meta_kernel

    if max_splits is None:
        max_splits = max_splits_for(hardware_num_sms())
    return (
        build_sched_meta_kernel(backend_config=backend_config),
        build_main_kernel(dtype, backend_config=backend_config),
        build_combine_kernel(dtype, max_splits, backend_config=backend_config),
    )


@cache_backend_config
def _compile_sched_meta(*, backend_config=None):
    from tirx_kernels.runner import compile_kernel

    from .sched_meta import build_sched_meta_kernel

    return compile_kernel(build_sched_meta_kernel(backend_config=backend_config), backend_config=backend_config)


@cache_backend_config
def _compile_main(dtype: str, *, backend_config=None):
    from tirx_kernels.runner import compile_kernel

    from .kernel import build_main_kernel

    return compile_kernel(build_main_kernel(dtype, backend_config=backend_config), backend_config=backend_config)


@cache_backend_config
def _compile_combine(dtype: str, max_splits: int, *, backend_config=None):
    from tirx_kernels.runner import compile_kernel

    from .combine import build_combine_kernel

    return compile_kernel(
        build_combine_kernel(dtype, max_splits, backend_config=backend_config),
        backend_config=backend_config,
    )


@dataclass
class Executables:
    dtype: str
    backend_config: Any
    sched_meta: Any
    main: Any
    combine: dict = field(default_factory=dict)

    def combine_for(self, num_sm_parts: int):
        ms = max_splits_for(num_sm_parts)
        if ms not in self.combine:
            self.combine[ms] = _compile_combine(self.dtype, ms, backend_config=self.backend_config)
        return self.combine[ms]


def compile_kernels(dtype: str, *, backend_config=None) -> Executables:
    """Compile the scheduler and main kernels, and the combine for the profile's SM count."""
    from tirx_kernels.runner import hardware_num_sms

    ex = Executables(
        dtype,
        backend_config,
        _compile_sched_meta(backend_config=backend_config),
        _compile_main(dtype, backend_config=backend_config),
    )
    ex.combine_for(hardware_num_sms())
    return ex


def _null_i32():
    """A zero-size tensor, i.e. a null data pointer: the kernels' `nullptr` operand."""
    import torch

    return torch.empty(0, dtype=torch.int32, device="cuda")


def _null_f32():
    import torch

    return torch.empty(0, dtype=torch.float32, device="cuda")


def launch_sched_meta(ex: Executables, p) -> None:
    """`run_get_decoding_sched_meta_kernel` for the dense path."""
    ex.sched_meta.mod(
        p["seqlens_k"],
        _null_i32(),  # topk_length (sparse only): null
        _null_i32(),  # extra_topk_length (sparse only): null
        p["tile_scheduler_metadata"].reshape(-1),
        p["num_splits"],
        p["b"],
        64,  # block_size_n
        5,  # fixed_overhead_num_blocks
        -1,  # topk: dense
        -1,  # extra_topk: dense_attn_decode_interface passes -1
        p["num_sm_parts"],
    )


def scale_softmax_log2(softmax_scale: float) -> float:
    """`float(softmax_scale * M_LOG2E)` with `softmax_scale` the interface's `float` argument."""
    import struct

    def f32(x):
        return struct.unpack("f", struct.pack("f", x))[0]

    return f32(f32(softmax_scale) * 1.44269504088896340736)


def _decode_args(p):
    """The main and combine kernels' arguments, built once per case outside timed launches."""
    if "_decode_args" in p:
        return p["_decode_args"]
    maps = make_tensor_maps(p)
    b, s_q, h_q, h_kv = p["b"], p["s_q"], p["h_q"], p["h_kv"]
    q_seq_per_hk, dv = p["q_seq_per_hk"], p["out"].size(3)
    main_args = (
        p["seqlens_k"],
        p["block_table"].reshape(-1),
        p["tile_scheduler_metadata"].reshape(-1),
        p["num_splits"],
        p["lse"].reshape(-1),
        p["lse_accum"].reshape(-1),
        p["out_accum"].reshape(-1),
        b,
        s_q,
        q_seq_per_hk,
        p["q_head_per_hk"],
        h_kv,
        int(p["is_causal"]),
        scale_softmax_log2(p["softmax_scale"]),
        int(p["block_table"].stride(0)),
        p["num_sm_parts"],
        maps[0].ptr,
        maps[1].ptr,
        maps[2].ptr,
    )
    # `CombineParams` as `dense_attn_decode_interface` fills it (dense_decode.h:186-206).
    combine_args = (
        p["lse"].reshape(-1),
        p["out"].reshape(-1),
        p["lse_accum"].reshape(-1),
        p["out_accum"].reshape(-1),
        p["num_splits"],
        _null_f32(),  # attn_sink: dense decode passes nullptr
        h_kv * q_seq_per_hk,
        h_q,
        h_q * s_q * dv,
        h_q * dv,
        dv,
        h_kv * q_seq_per_hk,
        h_q,
        h_q * s_q * dv,
        h_q * dv,
        dv,
        b,
        s_q,
        h_q,
    )
    p["_decode_args"] = (maps, main_args, combine_args)
    return p["_decode_args"]


def launch_decode(ex: Executables, p) -> None:
    """The main split-KV kernel followed by the combine kernel (both with PDL)."""
    _, main_args, combine_args = _decode_args(p)
    ex.main.mod(*main_args)
    ex.combine_for(p["num_sm_parts"]).mod(*combine_args)

# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TMA descriptors, compilation and launch of the sparse FP8 decode kernels.

`KernelTemplate::run` (`splitkv_mla.cuh:686-780`) builds the Q map with `cute::make_tma_copy`
over `(h_q, d_qk, s_q, b)` and `SmemLayoutQ` (64 x 64 boxes, 128B swizzle), and the 5D O map by
hand: `(64, h_q, 8, s_q, b)`, box `(64, 64, 8, 1, 1)`, 128B swizzle, L2 promotion 256B. They are
encoded with `cuTensorMapEncodeTiled` outside the timed launches.
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass, field
from typing import Any

from tirx_kernels.runner import cache_backend_config

from .._sm90_dense_decode.combine import max_splits_for

_SWIZZLE_128B = 3
_L2_128B = 2
_L2_256B = 3


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode(tensor, dims, strides_bytes, box, *, l2):
    import tvm_ffi

    desc = _AlignedTensorMap()
    rank = len(dims)
    tvm_ffi.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        "bfloat16",
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides_bytes,
        *box,
        *((1,) * rank),
        0,
        _SWIZZLE_128B,
        l2,
        0,
    )
    return desc


def make_tensor_maps(p):
    """`tma_Q` and `tensor_map_o` (innermost dimension first)."""
    q, o = p["q"], p["out"]
    b, s_q, h_q, d_qk = q.shape
    tma_q = _encode(
        q,
        (d_qk, h_q, s_q, b),
        (q.stride(2) * 2, q.stride(1) * 2, q.stride(0) * 2),
        (64, 64, 1, 1),
        l2=_L2_128B,
    )
    tma_o = _encode(
        o,
        (64, h_q, 512 // 64, s_q, b),
        (o.stride(2) * 2, 64 * 2, o.stride(1) * 2, o.stride(0) * 2),
        (64, 64, 512 // 64, 1, 1),
        l2=_L2_256B,
    )
    return tma_q, tma_o


_NULLS = {}


def null(dtype):
    """A zero-size CUDA tensor: its data pointer is null, the kernels' `nullptr` for an absent
    optional operand (the TIRx ABI takes a tensor in every pointer slot)."""
    import torch

    if dtype not in _NULLS:
        _NULLS[dtype] = torch.empty(0, dtype=dtype, device="cuda")
        assert _NULLS[dtype].data_ptr() == 0
    return _NULLS[dtype]


def opt(t, dtype):
    return null(dtype) if t is None else t


def flat(t, dtype=None):
    """A 1-D view from `t`'s first element over every element its strides reach."""
    import torch

    if dtype is not None:
        t = t.view(dtype)
    span = 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()) if n > 0)
    return torch.as_strided(t, (span,), (1,))


def model_type_of(d_qk: int) -> str:
    return {576: "V32", 512: "MODEL1"}[d_qk]


@cache_backend_config
def _compile_sched_meta(*, backend_config=None):
    from tirx_kernels.runner import compile_kernel

    from .._sm90_dense_decode.sched_meta import build_sched_meta_kernel

    return compile_kernel(build_sched_meta_kernel(backend_config=backend_config), backend_config=backend_config)


@cache_backend_config
def _compile_main(model_type: str, num_heads: int, *, backend_config=None):
    from tirx_kernels.runner import compile_kernel

    from .kernel import build_main_kernel

    return compile_kernel(
        build_main_kernel(model_type, num_heads, backend_config=backend_config),
        backend_config=backend_config,
    )


@cache_backend_config
def _compile_combine(max_splits: int, *, backend_config=None):
    from tirx_kernels.runner import compile_kernel

    from .._sm90_dense_decode.combine import build_combine_kernel

    return compile_kernel(
        build_combine_kernel("bfloat16", max_splits, backend_config=backend_config),
        backend_config=backend_config,
    )


def build_kernels(config, *, backend_config=None):
    from tirx_kernels.runner import hardware_num_sms

    from .._sm90_dense_decode.combine import build_combine_kernel
    from .._sm90_dense_decode.sched_meta import build_sched_meta_kernel
    from .data import num_sm_parts_for
    from .kernel import build_main_kernel

    parts = num_sm_parts_for(hardware_num_sms(), config.get("s_q", 1), config["h_q"])
    return (
        build_sched_meta_kernel(backend_config=backend_config),
        build_main_kernel(model_type_of(config["d_qk"]), config["h_q"], backend_config=backend_config),
        build_combine_kernel("bfloat16", max_splits_for(parts), backend_config=backend_config),
    )


@dataclass
class Executables:
    backend_config: Any
    sched_meta: Any
    main: Any
    combine: dict = field(default_factory=dict)

    def combine_for(self, num_sm_parts: int):
        ms = max_splits_for(num_sm_parts)
        if ms not in self.combine:
            self.combine[ms] = _compile_combine(ms, backend_config=self.backend_config)
        return self.combine[ms]


def compile_kernels(model_type: str, num_heads: int, *, backend_config=None) -> Executables:
    return Executables(
        backend_config,
        _compile_sched_meta(backend_config=backend_config),
        _compile_main(model_type, num_heads, backend_config=backend_config),
    )


def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def launch_sched_meta(ex: Executables, g) -> None:
    """`run_get_decoding_sched_meta_kernel` for the sparse path (sparse_decode.h:425-439)."""
    c = g["config"]
    x = g["extra"]
    import torch

    ex.sched_meta.mod(
        null(torch.int32),  # seqlens_k: dense only
        opt(g["kv"]["topk_length"], torch.int32),
        opt(x["topk_length"] if x is not None else None, torch.int32),
        g["tile_scheduler_metadata"].reshape(-1),
        g["num_splits"],
        c["b"],
        64,  # block_size_n = DecodeImplMeta::block_size_topk
        5,  # fixed_overhead_num_blocks
        c["topk"],
        c["extra_topk"] if x is not None else 0,
        g["num_sm_parts"],
    )


def _args(g):
    """The main and combine kernels' arguments, built once per case outside timed launches."""
    if "_args" in g:
        return g["_args"]
    import torch

    c = g["config"]
    maps = make_tensor_maps(g)
    q, kvq, out, lse = g["q"], g["kv"]["kcache"], g["out"], g["lse"]
    idx = g["kv"]["indices"]
    x = g["extra"]
    b, s_q, h_q, d_qk = q.shape
    # KernelTemplate::run's host checks (splitkv_mla.cuh:688-703).
    if c["topk"] % 64 != 0:
        raise ValueError(f"topk = {c['topk']} must be a multiple of 64 (TOPK_BLOCK_SIZE)")
    if h_q % 64 != 0:
        raise ValueError(f"h_q = {h_q} must be a multiple of 64 (BLOCK_M)")
    if d_qk == 576:
        if x is not None or g["kv"]["topk_length"] is not None:
            raise ValueError("V3.2 does not support an extra KV cache or a dynamic topk length")
        if kvq.stride(1) != 656:
            raise ValueError(f"V3.2 KV row stride must be 656 bytes, got {kvq.stride(1)}")
    else:
        for name, cache in (("kv", kvq), ("extra kv", x["kcache"] if x is not None else None)):
            if cache is not None and cache.stride(1) != 584:
                raise ValueError(f"MODEL1 {name} page blocks must be contiguous (row stride 584 bytes)")
    la, oa = g["lse_accum"], g["out_accum"]
    sm_scale = f32(g["sm_scale"])
    main_args = (
        flat(kvq, torch.uint8),
        flat(idx),
        opt(g["kv"]["topk_length"], torch.int32),
        opt(g["attn_sink"], torch.float32),
        lse.reshape(-1),
        flat(x["kcache"], torch.uint8) if x is not None else null(torch.uint8),
        flat(x["indices"]) if x is not None else null(torch.int32),
        opt(x["topk_length"] if x is not None else None, torch.int32),
        la.reshape(-1),
        oa.reshape(-1),
        g["tile_scheduler_metadata"].reshape(-1),
        g["num_splits"],
        b,
        s_q,
        h_q,
        c["topk"],
        kvq.size(1),
        f32(sm_scale * 1.44269504088896340736),
        x["kcache"].size(1) if x is not None else 0,
        c["extra_topk"] if x is not None else 0,
        kvq.stride(0),
        kvq.stride(1),
        idx.stride(0),
        idx.stride(1),
        lse.stride(0),
        lse.stride(1),
        x["kcache"].stride(0) if x is not None else 0,
        x["kcache"].stride(1) if x is not None else 0,
        x["indices"].stride(0) if x is not None else 0,
        x["indices"].stride(1) if x is not None else 0,
        la.stride(0),
        la.stride(1),
        oa.stride(0),
        oa.stride(1),
        oa.stride(2),
        g["num_sm_parts"],
        maps[0].ptr,
        maps[1].ptr,
    )
    # `CombineParams` as `sparse_attn_decode_interface` fills it (sparse_decode.h:470-489).
    combine_args = (
        lse.reshape(-1),
        out.reshape(-1),
        la.reshape(-1),
        oa.reshape(-1),
        g["num_splits"],
        opt(g["attn_sink"], torch.float32),
        lse.stride(0),
        lse.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        la.stride(0),
        la.stride(1),
        oa.stride(0),
        oa.stride(1),
        oa.stride(2),
        b,
        s_q,
        h_q,
    )
    g["_args"] = (maps, main_args, combine_args)
    return g["_args"]


def launch_decode(ex: Executables, g) -> None:
    """The main split-KV kernel, then the combine kernel (with PDL)."""
    _, main_args, combine_args = _args(g)
    ex.main.mod(*main_args)
    ex.combine_for(g["num_sm_parts"]).mod(*combine_args)

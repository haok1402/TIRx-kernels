# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ 0251105a), Copyright (c) 2024, Jay Shah,
# Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TMA descriptors, scheduler parameters, compilation and launch of the three backward kernels.

The descriptors are the ones ``run_flash_bwd`` (``hopper/flash_bwd_launch_template.h``) builds with
``make_tma_copy_A_sm90`` / ``make_tma_copy_B_sm90`` / ``make_tma_copy``: Q, dO, K, V, dK, dV as
``(seqlen, d, head, batch)`` with d contiguous, copied in boxes of 64 columns (128 B, SWIZZLE_128B)
by kBlockM rows (Q, dO) or 128 rows (K, V, dK, dV). The scheduler's ``FastDivmod`` constants are the
host-side ``cutlass::FastDivmod`` values. Everything here is built once per case, outside timed
launches.
"""

from __future__ import annotations

import ctypes
import math
import struct
from dataclasses import dataclass
from typing import Any

from tirx_kernels.runner import cache_backend_config

from .data import block_m, block_n

#: CU_TENSOR_MAP_SWIZZLE_128B and CU_TENSOR_MAP_L2_PROMOTION_L2_128B (cute's make_tma_copy defaults).
_SWIZZLE_128B = 3
_L2_PROMOTION = 2


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_bshd(t, box_rows):
    """A (b, s, h, d) bf16 tensor as the 4-D TensorMap (d, s, h, b), box (64, box_rows, 1, 1)."""
    import tvm_ffi

    desc = _AlignedTensorMap()
    es = t.element_size()
    dims = (t.size(3), t.size(1), t.size(2), t.size(0))
    strides = (t.stride(1) * es, t.stride(2) * es, t.stride(0) * es)
    tvm_ffi.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        "bfloat16",
        4,
        ctypes.c_void_p(int(t.data_ptr())),
        *dims,
        *strides,
        64,
        box_rows,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        _SWIZZLE_128B,
        _L2_PROMOTION,
        0,
    )
    return desc


def fast_divmod(d: int) -> tuple[int, int, int]:
    """``cutlass::FastDivmod(d)``: (divisor, multiplier, shift_right)."""
    if d == 1:
        return 1, 0, 0
    # find_log2(x) = 31 - clz(x) rounded up when x is not a power of two (cutlass/fast_math.h).
    p = 31 + _find_log2(d)
    m = ((1 << p) + d - 1) // d
    return d, m & 0xFFFFFFFF, p - 32


def _find_log2(x: int) -> int:
    a = 31 - (32 - x.bit_length())  # 31 - clz(x)
    b = 0 if (x & (x - 1)) == 0 else 1
    return a + b


def lpt_params(n_blocks: int, h: int, b: int, s_q: int, d: int = 128) -> dict[str, Any]:
    """``SingleTileBwdLPTScheduler::to_underlying_arguments`` (tile_scheduler.hpp:385-400). Its
    ``args.seqlen_k`` slot receives ``params.seqlen_q`` from ``run_flash_bwd`` (the Q / dO / dQaccum
    footprint of one head), so the L2 swizzle is sized by s_q."""
    size_one_qdo_head = s_q * (d + d) * 2
    size_one_dqaccum_head = s_q * d * 4
    size_one_head = size_one_qdo_head + size_one_dqaccum_head
    size_l2 = 40 * 1024 * 1024
    swizzle = 1 if size_l2 < size_one_head else (1 << (31 - _clz32(size_l2 // size_one_head)))
    hb_rem = (h * b) % swizzle
    return {
        "total_blocks": n_blocks * h * b,
        "block": fast_divmod(n_blocks),
        "head": fast_divmod(h),
        "l2_minor": fast_divmod(swizzle),
        "l2_major": fast_divmod(swizzle * n_blocks),
        "l2_minor_residual": fast_divmod(hb_rem if hb_rem > 0 else 1),
        "num_hb_quotient": (h * b) // swizzle,
    }


def _clz32(x: int) -> int:
    return 32 - x.bit_length()


def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def scale_log2(softmax_scale: float) -> float:
    """``float(args.softmax_scale * M_LOG2E)`` with softmax_scale a float."""
    return f32(f32(softmax_scale) * 1.44269504088896340736)


def build_kernels(head_dim: int, is_causal: bool, *, backend_config=None):
    from .kernel import build_main_kernel, build_postprocess, build_preprocess

    bm = block_m(head_dim, is_causal)
    return (
        build_preprocess(head_dim, bm, backend_config=backend_config),
        build_main_kernel(head_dim, is_causal, backend_config=backend_config),
        build_postprocess(head_dim, is_causal, backend_config=backend_config),
    )


@cache_backend_config
def _compile(head_dim: int, is_causal: bool, *, backend_config=None):
    from tirx_kernels.runner import compile_kernel

    return tuple(
        compile_kernel(k, backend_config=backend_config)
        for k in build_kernels(head_dim, is_causal, backend_config=backend_config)
    )


@dataclass
class Executables:
    head_dim: int
    is_causal: bool
    pre: Any
    main: Any
    post: Any


def compile_kernels(head_dim: int, is_causal: bool, *, backend_config=None) -> Executables:
    pre, main, post = _compile(head_dim, is_causal, backend_config=backend_config)
    return Executables(head_dim, is_causal, pre, main, post)


def _args(p, ours):
    """Kernel arguments of one case and one output set, built once outside timed launches."""
    key = id(ours)
    cache = p.setdefault("_launch_args", {})
    if key in cache:
        return cache[key]
    b, s_q, s_k, h, d = p["b"], p["s_q"], p["s_k"], p["h"], p["d"]
    bm, bn = p["block_m"], p["block_n"]
    maps = (
        _encode_bshd(p["q"], bm),
        _encode_bshd(p["dout"], bm),
        _encode_bshd(p["k"], bn),
        _encode_bshd(p["v"], bn),
        _encode_bshd(ours["dk"], bn),
        _encode_bshd(ours["dv"], bn),
    )
    pre_args = (
        p["out"].reshape(-1),
        p["dout"].reshape(-1),
        p["lse"].reshape(-1),
        ours["softmax_d"].reshape(-1),
        ours["lse_log2"].reshape(-1),
        ours["dq_accum"].reshape(-1),
        ours["dq_semaphore"].reshape(-1),
        s_q,
        d,
        h,
        b,
    )
    n_blocks = (s_k + bn - 1) // bn
    lpt = lpt_params(n_blocks, h, b, s_q, d)
    sched = (
        lpt["total_blocks"],
        *lpt["block"],
        *lpt["head"],
        *lpt["l2_minor"],
        *lpt["l2_major"],
        *lpt["l2_minor_residual"],
        lpt["num_hb_quotient"],
    )
    main_args = (
        ours["lse_log2"].reshape(-1),
        ours["softmax_d"].reshape(-1),
        ours["dq_accum"].reshape(-1),
        ours["dk"].reshape(-1),
        ours["dv"].reshape(-1),
        s_q,
        s_k,
        d,
        h,
        b,
        0 if p["is_causal"] else -1,  # window_size_right as mha_bwd sets it
        f32(p["softmax_scale"]),
        scale_log2(p["softmax_scale"]),
        *sched,
        *(m.ptr for m in maps),
    )
    post_args = (ours["dq_accum"].reshape(-1), ours["dq"].reshape(-1), s_q, d, h, b, f32(p["softmax_scale"]))
    cache[key] = (maps, pre_args, main_args, post_args)
    return cache[key]


def launch_backward(ex: Executables, p, ours) -> None:
    """preprocess, main, postprocess (no PDL: ``launch_with_pdl`` is false upstream)."""
    _, pre_args, main_args, post_args = _args(p, ours)
    ex.pre.mod(*pre_args)
    ex.main.mod(*main_args)
    ex.post.mod(*post_args)


def launch_postprocess(ex: Executables, p, dq_accum, dq) -> None:
    """The dQ convert alone, on a given accumulator."""
    ex.post.mod(dq_accum.reshape(-1), dq.reshape(-1), p["s_q"], p["d"], p["h"], p["b"], f32(p["softmax_scale"]))

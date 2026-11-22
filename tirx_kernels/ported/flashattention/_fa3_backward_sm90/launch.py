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


def build_kernels(head_dim: int, is_causal: bool, *, gqa: bool = False, varlen: bool = False,
                  backend_config=None):
    """preprocess, main, dQ convert (+ the dK / dV convert for GQA: ``run_flash_bwd``'s
    ``PostprocessKerneldKV``, launched twice). ``varlen``: the ``Varlen = true`` instantiations
    (cu_seqlens / seqused inputs)."""
    from .kernel import build_main_kernel, build_postprocess, build_postprocess_dkv, build_preprocess

    bm = block_m(head_dim, is_causal)
    kernels = (
        build_preprocess(head_dim, bm, varlen=varlen, backend_config=backend_config),
        build_main_kernel(head_dim, is_causal, gqa=gqa, varlen=varlen, backend_config=backend_config),
        build_postprocess(head_dim, is_causal, varlen=varlen, backend_config=backend_config),
    )
    if gqa:
        kernels += (build_postprocess_dkv(head_dim, varlen=varlen, backend_config=backend_config),)
    return kernels


@cache_backend_config
def _compile(head_dim: int, is_causal: bool, gqa: bool, varlen: bool = False, *, backend_config=None):
    from tirx_kernels.runner import compile_kernel

    return tuple(
        compile_kernel(k, backend_config=backend_config)
        for k in build_kernels(head_dim, is_causal, gqa=gqa, varlen=varlen, backend_config=backend_config)
    )


@dataclass
class Executables:
    head_dim: int
    is_causal: bool
    pre: Any
    main: Any
    post: Any
    gqa: bool = False
    post_dkv: Any = None
    varlen: bool = False


def compile_kernels(head_dim: int, is_causal: bool, *, gqa: bool = False, varlen: bool = False,
                    backend_config=None) -> Executables:
    built = _compile(head_dim, is_causal, gqa, varlen, backend_config=backend_config)
    return Executables(head_dim, is_causal, *built[:3], gqa=gqa, post_dkv=built[3] if gqa else None,
                       varlen=varlen)


def _null_i32():
    """A null ``int const*`` (cu_seqlens / seqused absent): a 0-element tensor, data pointer 0."""
    import torch

    return torch.empty(0, dtype=torch.int32, device="cuda")


def _varlen_args(p):
    """(cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k); seqused each possibly null. The port covers
    both cu_seqlens given (flash_attn_varlen_func's case): seqused without cu_seqlens, or one-sided
    cu_seqlens, take other upstream host paths (padded workspaces on the fixed-length kernels; batch
    strides set from cu_seqlens_q only) and are rejected."""
    if p.get("cu_seqlens_q") is None or p.get("cu_seqlens_k") is None:
        raise ValueError("varlen needs both cu_seqlens_q and cu_seqlens_k (seqused_q / seqused_k optional)")
    return tuple(p[n] if p.get(n) is not None else _null_i32()
                 for n in ("cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k"))


def _args(p, ours):
    """Kernel arguments of one case and one output set, built once outside timed launches."""
    key = id(ours)
    cache = p.setdefault("_launch_args", {})
    if key in cache:
        return cache[key]
    b, s_q, s_k, h, h_k, d = p["b"], p["s_q"], p["s_k"], p["h"], p["h_k"], p["d"]
    bm, bn = p["block_m"], p["block_n"]
    gqa = p["gqa"]
    # The GQA epilogue reduces into fp32 dK / dV accumulators instead of storing bf16 dK / dV.
    dk_acc = ours.get("dk_accum")
    dv_acc = ours.get("dv_accum")
    varlen = bool(p.get("varlen"))
    # varlen: (total, h, d) as the 4-D (d, total, h, 1) map (batch extent 1, run_flash_bwd's batch_q = 1).
    as4 = (lambda t: t.unsqueeze(0)) if varlen else (lambda t: t)
    maps = (
        _encode_bshd(as4(p["q"]), bm),
        _encode_bshd(as4(p["dout"]), bm),
        _encode_bshd(as4(p["k"]), bn),
        _encode_bshd(as4(p["v"]), bn),
    )
    if not varlen:  # the varlen MHA epilogue stores without TMA (Use_TMA = !Varlen)
        maps += (_encode_bshd(ours["dk"], bn), _encode_bshd(ours["dv"], bn))
    vptr = _varlen_args(p) if varlen else ()
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
    if varlen:  # cu_seqlens_q, seqused_q, grid rows (max_seqlen_q), shape_dPsum[0]
        pre_args += (vptr[0], vptr[2], p["max_seqlen_q"], p["s_q_rounded"])
    # Grid and LPT swizzle use the maximum lengths (params.seqlen_q / seqlen_k = max_seqlen_*).
    msq, msk = (p["max_seqlen_q"], p["max_seqlen_k"]) if varlen else (s_q, s_k)
    n_blocks = (msk + bn - 1) // bn
    lpt = lpt_params(n_blocks, h, b, msq, d)
    sched = (
        lpt["total_blocks"],
        *lpt["block"],
        *lpt["head"],
        *lpt["l2_minor"],
        *lpt["l2_major"],
        *lpt["l2_minor_residual"],
        lpt["num_hb_quotient"],
    )
    head = (s_q, s_k, d, h) + ((h_k,) if gqa else ()) + (b,)
    tail = (
        0 if p["is_causal"] else -1,  # window_size_right as mha_bwd sets it
        f32(p["softmax_scale"]),
        scale_log2(p["softmax_scale"]),
        *sched,
        *fast_divmod((h + h_k - 1) // h_k),  # qhead_per_khead_divmod (divisor 1 for MHA)
    )
    if varlen:  # cu_seqlens / seqused, the scheduler's seqlen (and grid), shape_LSE[0], dKaccum rows
        tail += (*vptr, msk, p["s_q_rounded"], p["s_k_rounded"])
    if gqa:  # dK / dV fp32 accumulators instead of the bf16 outputs; only the Q, dO, K, V TensorMaps
        main_args = (
            ours["lse_log2"].reshape(-1), ours["softmax_d"].reshape(-1), ours["dq_accum"].reshape(-1),
            dk_acc.reshape(-1), dv_acc.reshape(-1), *head, *tail, *(m.ptr for m in maps[:4]),
        )
    else:
        main_args = (
            ours["lse_log2"].reshape(-1), ours["softmax_d"].reshape(-1), ours["dq_accum"].reshape(-1),
            ours["dk"].reshape(-1), ours["dv"].reshape(-1), *head, *tail, *(m.ptr for m in maps),
        )
    post_args = (ours["dq_accum"].reshape(-1), ours["dq"].reshape(-1), *_post_tail(p, "q"), f32(p["softmax_scale"]))
    post_args += _post_varlen(p, "q")
    dkv_args = ()
    if gqa:
        # PostprocessKerneldKV: TileShape (kBlockN, d), softmax_scale 1.f, (seqlen_k, h_k) heads.
        dkv_args = (
            (ours["dk_accum"].reshape(-1), ours["dk"].reshape(-1), *_post_tail(p, "k"), 1.0, *_post_varlen(p, "k")),
            (ours["dv_accum"].reshape(-1), ours["dv"].reshape(-1), *_post_tail(p, "k"), 1.0, *_post_varlen(p, "k")),
        )
    cache[key] = (maps, pre_args, main_args, post_args, dkv_args)
    return cache[key]


def launch_backward(ex: Executables, p, ours) -> None:
    """preprocess, main, postprocess (no PDL: ``launch_with_pdl`` is false upstream)."""
    if ex.gqa:
        # mha_bwd zero-fills dk_accum / dv_accum with torch fill_(0.0) (flash_api_stable.cpp:1565, :1567),
        # an elementwise fill kernel; zero_() would be a cudaMemsetAsync, which a kernel trace omits.
        ours["dk_accum"].fill_(0.0)
        ours["dv_accum"].fill_(0.0)
    # mha_bwd (flash_api_stable.cpp:1629-1643): no kernel runs on an empty side; it zeroes the outputs
    # instead (torch zero_), e.g. a varlen batch whose sequences all have length 0.
    total_q, total_k = p["q"].shape[0] * (1 if p.get("varlen") else p["q"].shape[1]), \
        p["k"].shape[0] * (1 if p.get("varlen") else p["k"].shape[1])
    if not (total_q > 0 and total_k > 0):
        if total_k > 0:
            ours["dk"].zero_()
            ours["dv"].zero_()
            ours["softmax_d"].zero_()
        elif total_q > 0:
            ours["dq"].zero_()
            ours["softmax_d"].zero_()
        return
    _, pre_args, main_args, post_args, dkv_args = _args(p, ours)
    ex.pre.mod(*pre_args)
    ex.main.mod(*main_args)
    ex.post.mod(*post_args)
    for args in dkv_args:  # dK convert, then dV convert
        ex.post_dkv.mod(*args)


def _post_tail(p, side):
    """(seqlen, d, heads, b) of the dQ (side "q") or dK / dV (side "k") convert."""
    return (p[f"s_{side}"], p["d"], p["h"] if side == "q" else p["h_k"], p["b"])


def _post_varlen(p, side):
    """The varlen convert's extra arguments: cu_seqlens, seqused, grid rows (max_seqlen), shape[0] of
    the accumulator (the padded total)."""
    if not p.get("varlen"):
        return ()
    cu_q, cu_k, su_q, su_k = _varlen_args(p)
    cu, su = (cu_q, su_q) if side == "q" else (cu_k, su_k)
    return (cu, su, p[f"max_seqlen_{side}"], p[f"s_{side}_rounded"])


def launch_postprocess(ex: Executables, p, dq_accum, dq) -> None:
    """The dQ convert alone, on a given accumulator."""
    ex.post.mod(dq_accum.reshape(-1), dq.reshape(-1), *_post_tail(p, "q"), f32(p["softmax_scale"]),
                *_post_varlen(p, "q"))


def launch_postprocess_dkv(ex: Executables, p, dk_accum, dk, dv_accum, dv) -> None:
    """The dK and dV converts alone, on given fp32 accumulators (GQA)."""
    for acc, out in ((dk_accum, dk), (dv_accum, dv)):
        ex.post_dkv.mod(acc.reshape(-1), out.reshape(-1), *_post_tail(p, "k"), 1.0, *_post_varlen(p, "k"))

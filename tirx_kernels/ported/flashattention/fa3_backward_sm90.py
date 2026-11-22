# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ 0251105a), Copyright (c) 2024, Jay Shah,
# Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM90 FlashAttention-3 backward (bf16, MHA) -- flash-attention's Hopper ``hopper/`` path.

Reference: ``flash_attn_interface._flash_attn_backward`` (``flash_attn_3._C.bwd``) on SM90, which runs
``mha_bwd`` (``hopper/flash_api_stable.cpp``) -> ``run_mha_bwd_hdim{64,128,256}<90, bf16>``
(``hopper/flash_bwd_launch_template.h``) and launches three kernels:

- ``flash::FlashAttnBwdPreprocess`` (``hopper/flash_bwd_preprocess_kernel.h``): dPsum = rowsum(dO * O),
  LSE * log2(e), and zeroing of the fp32 dQ accumulator;
- ``flash::FlashAttnBwdSm90`` (``hopper/flash_bwd_kernel_sm90.h``) with
  ``CollectiveMainloopBwdSm90`` (``hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp``) and
  ``CollectiveEpilogueBwd`` (``hopper/epilogue_bwd.hpp``): one CTA per (KV block, head, batch),
  a producer warpgroup (TMA warp and dQ bulk-reduce warp) and two MMA warpgroups;
- ``flash::FlashAttnBwdPostprocessConvertdQ`` (``hopper/flash_bwd_postprocess_kernel.h``): fp32 dQ
  accumulator -> dQ * softmax_scale in bf16.

Specializations (``run_mha_bwd_hdim128<90, bf16, false>``): causal uses kBlockM = 64, kBlockN = 128,
dQ_swapAB = false and the ``SingleTileBwdLPTScheduler``; non-causal uses kBlockM = 80, kBlockN = 128,
dQ_swapAB = true and the ``SingleTileScheduler``. Both: Stages = Stages_dO = Stages_dS = 2,
SdP_swapAB = true, dKV_swapAB = false, 2 MMA warpgroups, AtomLayoutMSdP = 1, AtomLayoutNdKV = 2,
AtomLayoutMdQ = 1 (so dV and dK are register-sourced WGMMAs), non-deterministic dQ accumulation.
"""

from __future__ import annotations

from typing import Any

KERNEL_META = {
    "name": "flash_attention3_backward_sm90",
    "category": "ported.flashattention",
    "runtime_cuda_archs": ["sm_90a"],
    "reference_requirements": (
        {
            "package": "flash-attn-3",
            "git": {
                "url": "https://github.com/Dao-AILab/flash-attention.git",
                "commit": "0251105a2fb19d2957484b7f023cd8c115286ced",
            },
            "import": "flash_attn_interface",
        },
    ),
}

#: ``hopper/test_flash_attn.py::test_flash_attn_output`` (seqlen_q, seqlen_k) axis.
UPSTREAM_SEQLENS = [
    (1, 1),
    (64, 128),
    (128, 192),
    (256, 256),
    (239, 1),
    (799, 3),
    (113, 203),
    (113, 128),
    (128, 217),
    (113, 211),
    (108, 256),
    (256, 512),
    (384, 256),
    (640, 128),
    (512, 256),
    (1024, 1024),
    (1023, 1024),
    (1024, 1023),
    (4096, 4096),
    (4224, 4224),
]


def _case(b, s_q, s_k, h, d, causal, *, seed=0, prefix="") -> dict:
    label = f"{prefix}b{b}_sq{s_q}_sk{s_k}_h{h}_d{d}_{'causal' if causal else 'noncausal'}"
    return {
        "label": label,
        "batch_size": b,
        "seqlen_q": s_q,
        "seqlen_k": s_k,
        "num_heads": h,
        "head_dim": d,
        "is_causal": causal,
        "seed": seed,
    }


def upstream_test_cases(head_dim=128) -> list[dict]:
    """The MHA bf16 backward cases of ``test_flash_attn_output`` (batch 9, or 2 when seqlen_k > 2048;
    6 heads; d = dv; no softcap, no local window)."""
    return [
        _case(9 if s_k <= 2048 else 2, s_q, s_k, 6, head_dim, causal)
        for s_q, s_k in UPSTREAM_SEQLENS
        for causal in (False, True)
    ]


#: Edge shapes beyond the upstream grid: exactly one and two KV blocks, KV and Q tails of 1 row,
#: seqlen_q a multiple of the non-causal kBlockM (80) and of the causal one (64) only, many heads
#: and batches (the LPT scheduler's L2 swizzle and its remainder path), and seqlen_q > seqlen_k
#: causal (KV blocks with no Q rows: the store_zero epilogue).
_EDGE = [
    _case(3, 160, 128, 5, 128, False, seed=1, prefix="edge_"),
    _case(3, 192, 256, 5, 128, True, seed=2, prefix="edge_"),
    _case(2, 257, 129, 3, 128, False, seed=3, prefix="edge_"),
    _case(2, 257, 129, 3, 128, True, seed=4, prefix="edge_"),
    _case(1, 2000, 300, 2, 128, True, seed=5, prefix="edge_"),
    _case(1, 300, 2000, 2, 128, True, seed=6, prefix="edge_"),
    _case(7, 1000, 1000, 37, 128, True, seed=7, prefix="edge_"),
    _case(5, 3000, 3000, 13, 128, False, seed=8, prefix="edge_"),
]

CONFIGS = [*upstream_test_cases(128), *_EDGE]

#: ``hopper/benchmark_attn.py``-style shapes: 16k tokens per batch, hidden 2048 (16 heads of 128).
BENCH_CONFIGS = [
    _case(16384 // s, s, s, 16, 128, causal, prefix="bench_")
    for s in (512, 1024, 2048, 4096, 8192, 16384)
    for causal in (False, True)
]


def get_kernel(*, backend_config=None, **config):
    """The three PrimFuncs of one specialization: preprocess, main backward, dQ postprocess."""
    from ._fa3_backward_sm90.launch import build_kernels

    config.pop("label", None)
    return list(
        build_kernels(config["head_dim"], config["is_causal"], backend_config=backend_config)
    )


def prepare_data(**config) -> dict[str, Any]:
    from ._fa3_backward_sm90.data import prepare

    config.pop("label", None)
    return prepare(**config)


def run_test(*, backend_config=None, **config):
    """dK / dV / dPsum / LSE*log2(e) bitwise against flash_attn_3, dQ bitwise when at most two KV
    blocks contribute to a row (the fp32 bulk-reduce order cannot matter then) and otherwise within
    the upstream test's tolerance; whole outputs from a NaN sentinel."""
    from ._fa3_backward_sm90.data import check_case

    config.pop("label", None)
    return check_case(config, backend_config=backend_config)


def run_bench(
    *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, backend_config=None, **config
):
    from ._fa3_backward_sm90.data import bench_against_fa3

    config.pop("label", None)
    return bench_against_fa3(
        config,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
        backend_config=backend_config,
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
    "upstream_test_cases",
]

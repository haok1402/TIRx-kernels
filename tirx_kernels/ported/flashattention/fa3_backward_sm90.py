# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ 0251105a), Copyright (c) 2024, Jay Shah,
# Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM90 FlashAttention-3 backward (bf16, head dim 128; MHA, GQA / MQA; fixed length and varlen) --
flash-attention's Hopper ``hopper/`` path.

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

GQA / MQA (h != h_k) selects the ``GQA = true`` instantiations (CollectiveEpilogueBwdGQA: fp32 dK / dV
reduce-added across query heads, then two more convert launches); ``varlen`` configs (cu_seqlens_q and
cu_seqlens_k, seqused optional) select the ``Varlen = true`` ones (SeqlenInfo offsets, padded
workspaces, the MHA epilogue's non-TMA store).
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


#: ``hopper/test_flash_attn.py::test_flash_attn_varlen_output`` (seqlen_q, seqlen_k) axis.
UPSTREAM_VARLEN_SEQLENS = [
    (1, 1),
    (1, 3),
    (2, 1),
    (511, 1),
    (3, 513),
    (64, 128),
    (128, 128),
    (256, 256),
    (113, 203),
    (128, 217),
    (113, 211),
    (108, 256),
    (256, 512),
    (307, 256),
    (640, 128),
    (512, 256),
    (1024, 1024),
    (1023, 1024),
    (1024, 1023),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
]


def _case(b, s_q, s_k, h, d, causal, *, h_k=None, seed=0, prefix="", varlen=None) -> dict:
    """One config; ``h_k`` (KV heads) set only for GQA / MQA, which selects upstream's GQA path.
    ``varlen``: None for fixed length, else the varlen options (``add_unused_qkv``, ``seqused``,
    ``zero_lengths_k``); s_q / s_k are then the padded maximum lengths of the generated batch."""
    gqa = h_k is not None and h_k != h
    vtag = ""
    if varlen is not None:
        vtag = "varlen_" + ("unused_" if varlen.get("add_unused_qkv") else "") + (
            "noseqused_" if varlen.get("seqused") is False else "")
    label = (
        f"{prefix}{vtag}b{b}_sq{s_q}_sk{s_k}_h{h}{f'_hk{h_k}' if gqa else ''}_d{d}_"
        f"{'causal' if causal else 'noncausal'}"
    )
    config = {
        "label": label,
        "batch_size": b,
        "seqlen_q": s_q,
        "seqlen_k": s_k,
        "num_heads": h,
        "head_dim": d,
        "is_causal": causal,
        "seed": seed,
    }
    if gqa:
        config["num_heads_k"] = h_k
    if varlen is not None:
        config["varlen"] = True
        config.update(varlen)
    return config


def upstream_test_cases(head_dim=128, mha_type="mha") -> list[dict]:
    """The bf16 backward cases of ``test_flash_attn_output`` for one ``mha_type`` (batch 9, or 2 when
    seqlen_k > 2048; 6 heads, KV heads 6 / 2 / 1 for mha / gqa / mqa; d = dv; no softcap, no local)."""
    h_k = {"mha": None, "gqa": 2, "mqa": 1}[mha_type]
    return [
        _case(9 if s_k <= 2048 else 2, s_q, s_k, 6, head_dim, causal, h_k=h_k)
        for s_q, s_k in UPSTREAM_SEQLENS
        for causal in (False, True)
    ]


def upstream_varlen_test_cases(head_dim=128, mha_type="mha") -> list[dict]:
    """The bf16 backward cases of ``test_flash_attn_varlen_output`` for one ``mha_type``: batch 9 (2
    when seqlen_q > 2048), 6 heads, KV heads 6 / 2 / 1, add_unused_qkv in {False, True}, seed
    seqlen_q + seqlen_k + d + 2 * causal as upstream's (no local, no softcap, no qv)."""
    h_k = {"mha": None, "gqa": 2, "mqa": 1}[mha_type]
    return [
        _case(9 if s_q <= 2048 else 2, s_q, s_k, 6, head_dim, causal, h_k=h_k,
              seed=s_q + s_k + head_dim + 2 * int(causal), prefix=f"{mha_type}_" if h_k else "",
              varlen={"add_unused_qkv": unused})
        for s_q, s_k in UPSTREAM_VARLEN_SEQLENS
        for causal in (False, True)
        for unused in (False, True)
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

#: GQA edge shapes: two query heads per KV head (dK / dV reduce order cannot matter: bitwise), the
#: Llama-3 8B / 70B per-device head ratios (32 / 8, 64 / 8), and MQA with many query heads.
_EDGE_GQA = [
    _case(3, 300, 300, 4, 128, False, h_k=2, seed=11, prefix="edge_"),
    _case(3, 300, 300, 4, 128, True, h_k=2, seed=12, prefix="edge_"),
    _case(2, 1024, 1024, 32, 128, True, h_k=8, seed=13, prefix="edge_"),
    _case(1, 2048, 2048, 64, 128, False, h_k=8, seed=14, prefix="edge_"),
    _case(2, 500, 700, 16, 128, True, h_k=1, seed=15, prefix="edge_"),
]

#: Varlen edges: cu_seqlens without seqused (the other SeqlenInfo branch), many sequences and heads
#: (LPT swizzle remainder), long sequences (more than two KV blocks per row), GQA 32 / 8.
_EDGE_VARLEN = [
    _case(9, 300, 700, 6, 128, False, seed=21, prefix="edge_", varlen={"seqused": False}),
    _case(9, 700, 300, 6, 128, True, seed=22, prefix="edge_", varlen={"seqused": False}),
    _case(17, 1000, 1000, 13, 128, True, seed=23, prefix="edge_", varlen={"add_unused_qkv": True}),
    _case(3, 3000, 3000, 32, 128, True, h_k=8, seed=24, prefix="edge_", varlen={}),
    _case(3, 3000, 3000, 32, 128, False, h_k=8, seed=25, prefix="edge_", varlen={"seqused": False}),
]

#: hd64 edges: exactly one and two 128-row m-blocks, causal s_q > s_k (store_zero) and s_q < s_k (the
#: first masking loop ending early), many heads (LPT remainder), GQA 32 / 4, varlen without seqused.
_EDGE_HD64 = [
    _case(3, 256, 128, 5, 64, False, seed=31, prefix="edge_"),
    _case(3, 128, 256, 5, 64, True, seed=32, prefix="edge_"),
    _case(1, 2000, 300, 2, 64, True, seed=33, prefix="edge_"),
    _case(1, 300, 2000, 2, 64, True, seed=34, prefix="edge_"),
    _case(7, 1000, 1000, 37, 64, True, seed=35, prefix="edge_"),
    _case(2, 2048, 2048, 32, 64, True, h_k=4, seed=36, prefix="edge_"),
    _case(9, 700, 300, 6, 64, False, seed=37, prefix="edge_", varlen={"seqused": False}),
]

CONFIGS = [
    *upstream_test_cases(128),
    *_EDGE,
    *upstream_test_cases(128, "gqa"),
    *upstream_test_cases(128, "mqa"),
    *_EDGE_GQA,
    *upstream_varlen_test_cases(128),
    *upstream_varlen_test_cases(128, "gqa"),
    *upstream_varlen_test_cases(128, "mqa"),
    *_EDGE_VARLEN,
    # head dim 64 (kBlockM = kBlockN = 128, SeparateMaskingIterations, ShuffleLSE): the same upstream grids
    *upstream_test_cases(64),
    *upstream_test_cases(64, "gqa"),
    *upstream_test_cases(64, "mqa"),
    *upstream_varlen_test_cases(64),
    *upstream_varlen_test_cases(64, "gqa"),
    *upstream_varlen_test_cases(64, "mqa"),
    *_EDGE_HD64,
]

#: ``hopper/benchmark_attn.py``-style shapes: 16k tokens per batch, hidden 2048 (16 heads of 128).
BENCH_CONFIGS = [
    _case(16384 // s, s, s, 16, 128, causal, prefix="bench_")
    for s in (512, 1024, 2048, 4096, 8192, 16384)
    for causal in (False, True)
] + [
    # Llama-3 8B attention (32 query heads, 8 KV heads of 128), 16k tokens per batch.
    _case(16384 // s, s, s, 32, 128, causal, h_k=8, prefix="bench_")
    for s in (1024, 4096, 8192)
    for causal in (False, True)
] + [
    # Packed (varlen) batches of about 16k tokens: 16 / 4 sequences of max length 1024 / 4096 (lengths
    # in [max - 20, max], no empty sequences), MHA 16 heads and Llama-3 8B GQA 32 / 8.
    _case(16384 // s, s, s, h, 128, causal, h_k=hk, prefix="bench_",
          varlen={"zero_lengths_k": False, "seqused": False})
    for s in (1024, 4096)
    for h, hk in ((16, None), (32, 8))
    for causal in (False, True)
] + [
    # head dim 64, 16k tokens per batch: 32 heads of 64 (hidden 2048), GQA 32 / 4 at s4096, and a packed batch
    _case(16384 // s, s, s, 32, 64, causal, prefix="bench_")
    for s in (1024, 4096, 16384)
    for causal in (False, True)
] + [
    _case(4, 4096, 4096, 32, 64, causal, h_k=4, prefix="bench_") for causal in (False, True)
] + [
    _case(4, 4096, 4096, 32, 64, causal, prefix="bench_", varlen={"zero_lengths_k": False, "seqused": False})
    for causal in (False, True)
]


def get_kernel(*, backend_config=None, **config):
    """The three PrimFuncs of one specialization: preprocess, main backward, dQ postprocess."""
    from ._fa3_backward_sm90.launch import build_kernels

    config.pop("label", None)
    gqa = config.get("num_heads_k", config["num_heads"]) != config["num_heads"]
    return list(
        build_kernels(config["head_dim"], config["is_causal"], gqa=gqa, varlen=bool(config.get("varlen")),
                      backend_config=backend_config)
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
    "upstream_varlen_test_cases",
]

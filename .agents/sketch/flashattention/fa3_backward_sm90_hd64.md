<!--
This sketch is a design artifact, not source. It is written once, reviewed
once, and then frozen: the implementation follows it, and where the two
disagree the TIRx module linked below is what runs.

Documents a TIRx port of Dao-AILab/flash-attention
(https://github.com/Dao-AILab/flash-attention @ 0251105a2fb19d2957484b7f023cd8c115286ced),
hopper/flash_bwd_launch_template.h (run_mha_bwd_hdim64), hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp,
hopper/epilogue_bwd.hpp, hopper/flash_bwd_preprocess_kernel.h, hopper/flash_bwd_postprocess_kernel.h.
Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# fa3_backward_sm90, head dim 64: delta sketch

A delta on the frozen, reviewer-approved sketches [`fa3_backward_sm90.md`](fa3_backward_sm90.md) (MHA, hd128),
[`fa3_backward_sm90_gqa.md`](fa3_backward_sm90_gqa.md) and [`fa3_backward_sm90_varlen.md`](fa3_backward_sm90_varlen.md).
Every statement in them holds unless this file replaces it. Implementation: the same TIRx module, where the head
dim is a compile-time parameter of every kernel builder, as `kHeadDim` is upstream.

## Specialization

`run_mha_bwd_hdim64<90, bf16, Has_softcap = false>` (`flash_bwd_launch_template.h:312-321`) uses one tile config
for causal and non-causal:

`run_mha_bwd_dispatch<90, bf16, kBlockM = 128, kBlockN = 128, kHeadDim = 64, ..., Stages_dO = 2, Stages_dS = 2,
SdP_swapAB = true, dKV_swapAB = false, dQ_swapAB = false, NumMmaWarpGroups = 2, AtomLayoutMSdP = 1,
AtomLayoutNdKV = 2, AtomLayoutMdQ = 2, V_in_regs = false>`

It is dispatched like hd128, over Varlen x GQA (x Deterministic = false), so the port covers 8 main kernels:
causal / non-causal x MHA / GQA x fixed / varlen.

- **Flags from the mainloop** (`mainloop_bwd_sm90_tma_gmma_ws.hpp:75-76, 257-264`):
  - `Mma_dKV_is_RS = true`;
  - `Mma_dQ_is_RS = false`;
  - `dQacc_use_TMA = true`;
  - `Slice_dQKV_Mma = false`;
  - **new:** `SeparateMaskingIterations = true` (`kHeadDim <= 64`) and `ShuffleLSE = ShuffledPsum = true`
    (`SdP_swapAB && kHeadDim <= 64`).
- **Scheduler:** as hd128. Causal: `SingleTileBwdLPTScheduler`, whose swizzle is sized with headdim 64:
  `size_one_head = s_q * (64 + 64) * 2 + s_q * 64 * 4`. Non-causal: `SingleTileScheduler`.
- **mha_bwd host sizes:** `kBlockM_sm90 = 128` and `kBlockN_sm90 = 128` for d <= 64 without softcap; `d_rounded = 64`.
  All padded sizes (`seqlen_q_rounded`, `total_q_padded_rounded`, workspaces) use kBlockM 128 and 64 columns.
- **Scope: d = dv = 64 exactly.** `mha_bwd` routes every head size <= 64 to this instantiation (d_rounded 64; the
  column predicates `col8 < d` mask the rest). The port supports d = 64 only, and its host rejects other head dims,
  so the LPT swizzle's runtime d / dv (`tile_scheduler.hpp:389-391`) are 64 here.

## Line-info export

`.porting/fa3_backward_sm90/source_export/`:
- **`hd64.ptx` / `.sass` / `.ptxas.log`:** the MHA fixed-length pair, split into `hd64_main_{causal,noncausal}.ptx`,
  `hd64_pre.ptx`, `hd64_post.ptx`.
- **`hd64x6.*` (`export_hd64x.sh`, `-DEXPORT_HD64_ALL`):** the GQA, varlen and varlen-GQA pairs.
- **Layout probe:** `layout_probe.txt`, sections `hd64 causal` / `hd64 noncausal`.

ptxas, MHA fixed:
- **main causal:** 168 registers, **24 B spill stores / 48 B loads, 24 B stack**;
- **main non-causal:** 168, 0 B;
- **preprocess:** 64;
- **dQ convert:** 55.

GQA / varlen (`hd64x6.ptxas.log`), all at 168 registers:
- **GQA:** causal 24 B / 48 B, non-causal 0 B;
- **varlen MHA:** causal 16 B / 28 B, non-causal 0 B;
- **varlen GQA:** causal 20 B / 20 B, non-causal 0 B.

The preprocesses use 64 registers and the converts 55, none spilling. The spill gate is "<= upstream's" per kernel:
the non-causal mains must be spill-free.

Static PTX, MHA fixed (per instantiation):

| op | hd64 causal | hd64 non-causal | hd128 causal (for reference) |
| --- | --- | --- | --- |
| `wgmma ... m64n128k16` (S^T, dP^T) | 16 | 8 | 8 (m64n128k16: dV, dK at hd128) |
| `wgmma ... m64n64k16` | 48 | 24 | 24 |
| `ex2.approx` | 128 | 64 | 32 |
| `shfl.sync` | 138 | 74 | 10 |
| `stmatrix ... trans` (dS^T) | 16 | 8 | 4 |
| `stmatrix` (epilogue STSM_N) | 8 | 8 | 16 |

Causal instantiates `bwd_step` twice, once per masking loop (below), so its per-step counts double. Per step:
- **S^T:** 4 x m64n128k16 (k over d = 64);
- **dP^T:** 4 x m64n128k16;
- **dV:** 8 x m64n64k16 rs (k over the 128 q columns of P^T);
- **dK:** 8 x m64n64k16 rs;
- **dQ:** 8 x m64n64k16 ss (k over the 128 KV rows);
- **ex2:** 64 per thread;
- **ShuffleLSE / ShuffledPsum:** 64 `shfl.sync.idx`.

## SharedStorage (`layout_probe.txt`, hd64)

`sizeof(SharedStorage) = 200704`, tensors at 0 (199680 B), pipelines at 199680 (`barrier_KV` +0, `pipeline_q` +16,
`pipeline_do` +48, as hd128):

| member | byte offset | bytes | layout |
| --- | --- | --- | --- |
| sK | 0 | 16384 | `Sw<3,4,3> o ((8,16),(64,1)):((64,512),(1,0))`: 128 rows x 64 cols, one SW128 atom column |
| sV | 16384 | 16384 | same |
| sdQacc | 32768 | 32768 | `(4096, 2):(1, 4096)` floats: one 16 KiB half per MMA warpgroup |
| sQ | 65536 | 2 x 16384 | `Sw<3,4,3> o ((8,16),(64,1),(1,2)):((64,512),(1,0),(0,8192))`: stage stride 16384 B |
| sdO | 98304 | 2 x 16384 | same |
| sLSE | 131072 | 2 x 512 | `(128, 2):(1, 128)` floats (stage stride 128 floats) |
| sdPsum | 132096 | 2 x 512 | same |
| sP | 133120 | (unused: Mma_dKV_is_RS) | |
| sdS | 134144 | 2 x 32768 | `Sw<3,4,3> o ((8,16),(64,2),(1,2)):((64,512),(1,8192),(0,16384))`: 128 q rows x 128 KV columns as two 64-column SW128 halves 16384 B apart; stage stride 32768 B |
| epilogue sdK / sdV | 0 / 16384 | 16384 each | `SmemLayoutdKV` = sK's layout (aliases sK, sV) |
| GQA epilogue sdKVaccum | 0 | 32768 | `(4096, 2):(1, 4096)` floats |

`TmaTransactionBytesQ = TmaTransactionBytesK = TmaTransactionBytesV = 16384`, `TmaTransactionBytesLSE = 512`, so the
Q / dO stage `expect_tx` is `16384 + 512`, and barrier_KV's is `2 * 16384`.

## Main kernel

### Producer warp 0

Changes against hd128, everything else as there:
- **Copy count:** each Q / dO / K / V tile is **one** TMA box `{64, 128}` (d = 64 is one 64-column box), not two.
  The coordinates are (0, row, head, batch), and the TensorMaps are built with box `{64, 128}`.
  - `instruction_selection: cp.async.bulk.tensor.4d...L2::cache_hint`, x1 per tile.
- **LSE / dPsum:** bulk copies of `128 * 4 = 512` B per m-block (`cp.async.bulk.shared::cluster.global`), into the
  stage at `131072 + stage * 512` / `132096 + stage * 512`.

### dQ writer warp 1

As hd128 with `kBlockM * kHeadDim / 2 = 4096` floats (16384 B) per warpgroup half:
- **Source:** `sdQacc + w * 16384`.
- **Destination:** gdQaccum float `base + m_block * (128 * 64) + w * 4096`.
- `instruction_selection: cp.reduce.async.bulk ... add.f32, 16384 B`.

### MMA warpgroups: `bwd_step` (`mainloop :834-1001`)

`w` = MMA warpgroup, `t = tid % 128`, `warp = t / 32`, `lane = t % 32`.

The pipeline protocol of the hd128 `bwd_step` is unchanged and sits at the same positions:
- `consumer_wait` on pipe_q.full before S^T and on pipe_do.full before dP^T;
- the `wgmma_fence` / `wgmma_commit` pair around each of the five GEMMs;
- the stage advance at the end.

The block below restates only what changes.

```python
def bwd_step(m_block, mask_fn):                 # NS = 64 (S^T / dP^T m64n128 fragment per thread)
    rS = wgmma(m64n128k16, A=sK[64w:64w+64, :] K-major, B=sQ[stage] K-major, k-steps 4)   # S^T, ss
    # instruction_selection: wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16; extent: 4;
    #   descriptors: K-major SW128 (LBO 1), A base sK + w * 8192 B, k-step +32 B; B base sQ + stage * 16384, +32 B
    # ShuffleLSE: thread loads kStatsPerThread = 4 LSE values (:845-855)
    rLSE[i] = sLSE[stage][q_of(lane % 4, lane / 4 + 8 * i)]   for i in 0..3
    # instruction_selection: ld.shared.u32 (32-bit, the value bit-cast); extent: 4
    rdP = wgmma(m64n128k16, A=sV[64w:64w+64, :] K-major, B=sdO[stage] K-major, k-steps 4)     # dP^T, ss
    # instruction_selection: wgmma ... m64n128k16; extent: 4; K-major SW128 (LBO 1), A base sV + w * 8192 B,
    #   B base sdO + stage * 16384, k-step +32 B
    wgmma_wait(1)
    mask_fn(rS, m_block)                        # mask.h SwapAB, as hd128 with NC = 16 column groups
    for mi in range(32):                        # mi = 2c + e: q column q_of(lane % 4, mi) = 8c + 2(lane % 4) + e
        lse = shfl_sync(rLSE[mi // 8], src_lane=(mi % 8) * 4 + lane % 4)
        # instruction_selection: shfl.sync.idx.b32; extent: 32 (one per mi, shared by both KV rows)
        for each of the thread's 2 KV rows: P = ex2(S * scale_log2 - lse)
        # instruction_selection: mul.ftz.f32 + sub.ftz.f32 (FFMA) + ex2.approx.ftz.f32; extent: 64
    rdPsum[i] = sdPsum[stage][q_of(lane % 4, lane / 4 + 8 * i)]   for i in 0..3   (ShuffledPsum, :870-878)
    # instruction_selection: ld.shared.u32; extent: 4
    wgmma_wait(0)
    for mi in range(32):
        dpsum = shfl_sync(rdPsum[mi // 8], src_lane=(mi % 8) * 4 + lane % 4)
        # instruction_selection: shfl.sync.idx.b32; extent: 32
        dS = P * (dP - dpsum)                   # instruction_selection: sub.ftz.f32, mul.ftz.f32; extent: 64
    rP = cvt_bf16x2(P); rdS = cvt_bf16x2(dS)    # instruction_selection: cvt.rn.bf16x2.f32; extent: 32 + 32
    # dS^T -> sdS[stage] (q rows, KV columns): STSM_T, matrix j of instruction p = (c = 2p + j // 2, i = j % 2),
    # q row 8c + lane % 8, KV chunk warp * 2 + i of the warpgroup's 64-column half (w); p in 0..7
    # instruction_selection: stmatrix.sync.aligned.x4.trans.m8n8.shared.b16; extent: 8
    rdV += wgmma_rs(m64n64k16, A=rP[4k:4k+4], B=sdO[stage] rows 16k.. MN-major, k-steps 8)
    # instruction_selection: wgmma ... m64n64k16 (A from registers); extent: 8; B desc MN-major SW128, LBO 0,
    #   base sdO + stage * 16384 + k * 2048
    fence_proxy_async(); named_sync(PdS, 256)
    rdQ = wgmma(m64n64k16, A=sdS[stage][64w:64w+64, :] K-major, B=sK MN-major, k-steps 8)    # dQ, ss
    # instruction_selection: wgmma ... m64n64k16, flags (scale-d k > 0, 1, 1, tnspA 0, tnspB 1); extent 8;
    #   A desc K-major SW128 LBO 1, base sdS + stage * 32768 + w * 8192, k-step +32 B within a 64-column half
    #   and +16384 B to the second half (k >= 4); B desc MN-major LBO 0, base sK + k * 2048
    wgmma_wait(1); release(pipeline_do)
    rdK += wgmma_rs(m64n64k16, A=rdS[4k:4k+4], B=sQ[stage] MN-major, k-steps 8)
    # instruction_selection: wgmma ... m64n64k16 (A from registers), flags (1, 1, 1); extent: 8; B desc
    #   MN-major SW128, LBO 0, base sQ + stage * 16384 + k * 2048
    wgmma_wait(1)
    named_sync(dQEmpty + w, 160); st_shared_v4(sdQacc + w * 16384 + (i * 128 + t) * 16, rdQ[4i:4i+4]) i in 0..7
    # instruction_selection: bar.sync dQEmpty + w, 160; st.shared.v4.u32 (128-bit); extent: 8
    fence_proxy_async(); named_arrive(dQFull + w, 160)
    wgmma_wait(0); release(pipeline_q)
```

- **Fragments:**
  - **S^T / dP^T:** m64n128 C. KV rows `64w + 16 * warp + lane / 4 (+8)`, q columns `8c + 2(lane % 4) + {0, 1}`,
    c in 0..15, registers `4c + {0, 1}` (+{2, 3}).
  - **dV / dK:** m64n64 C, 32 registers. KV rows as above; d columns `8c + 2(lane % 4) + {0, 1}`, c in 0..7.
  - **dQ:** m64n64 C. q rows `64w + 16 * warp + lane / 4 (+8)`; d columns as dV / dK.
- **ShuffleLSE indexing:** `tLSEsLSE` is the C fragment's column partition of sLSE. Its entry j is q column
  `8 * (j // 2) + 2 * (lane % 4) + j % 2`. The thread loads entries `lane / 4 + 8i` (i in 0..3). The value for
  entry `mi` comes from lane `(mi % 8) * 4 + lane % 4`, which holds entry `(mi % 8) + 8 * (mi // 8) = mi` in its
  register `mi // 8`. The reads past 127 never happen (j < 32).

### Masking iterations (`mainloop :1003-1031`)

```python
if causal:                                         # SeparateMaskingIterations
    m_block_masking_max = ((n_block + 1) * 128 - 1 + seqlen_q - seqlen_k - window_size_right) // 128 + 1   # C division
    for m_block in range(m_block_min, min(m_block_max, m_block_masking_max)):
        bwd_step(m_block, mask(seqlenk=True, causal=True))
for m_block in range(m, m_block_max):              # continues from where the first loop stopped
    bwd_step(m_block, mask(seqlenk=True, causal=False))
```

- **Code shape:** each loop is its own `bwd_step` instantiation (CUTLASS_PRAGMA_NO_UNROLL).
- **Non-causal:** only the second loop, with the seqlen_k mask, as hd128 non-causal.
- **Causal division:** `m_block_masking_max` uses C integer division (truncation) on a value that can be
  negative.

### Epilogue (MHA, TMA; `epilogue_bwd.hpp:163-235`)

The structure is unchanged; only the sizes shrink:
- **Conversion:** cvt 16 + 16 words.
- **STSM_N into sdV then sdK:** matrix j of instruction p = (c = 2p + j // 2, i = j % 2), row
  `64w + 16 * warp + 8i + lane % 8`, 16-byte chunk c (c in 0..7), byte `row * 128 + ((c ^ (row % 8)) * 16)`.
  - `instruction_selection: stmatrix ... x4.m8n8; extent: 4 + 4`.
- **TMA store:** one box `{64, 128}` per tensor, dV then dK (2 instead of 4).
- **Varlen (non-TMA) store:** `GmemTiledCopydKV` with `kGmemThreadsPerRow = gcd(64 / 8, 256) = 8`. That gives
  32 rows per pass and 4 passes, so `ld.shared` 128-bit x 4 + 4 and `st.global` 128-bit x 4 + 4.
- **`store_zero`:** the same partition, 8 threads per row, 4 passes.

### GQA epilogue (`epilogue_bwd.hpp:414-518`)

As the GQA sketch with 32 fragment registers:
- **R2S:** `st.shared.v4` x 8 per thread into `sdKVaccum + w * 16384 + (i * 128 + t) * 16`.
- **Reduce:** one `cp.reduce.async.bulk` of **32768 B** per tensor, dV then dK.
- **Accumulator tile:** float `((bb * h_k + bidh_kv) * s_k_rounded + offset_k_padded + n_block * 128) * 64`.

## Preprocess (`TileShape_MK = (128, 64)`)

- **`GmemTiledCopy`:** `kGmemThreadsPerRow = 8`, layout `((8, 32), 8)`. Each thread reads row `t / 8 + 32 * r`
  (r in 0..3), column `8 * (t % 8)`.
- **dPsum:** per row, `mul.ftz` + 7 `fma.rn.ftz` (8 elements), then `Allreduce<8>`: `shfl.sync.bfly` 4, 2, 1 with
  `add.ftz`.
  - Written by `t % 8 == 0` for its 4 rows.
- **LSE:** loaded for `t < s_q - m_block * 128 and t < 128`, else +inf (`preprocess_kernel.h:167`).
  - LSE_log2 is written for `t < round_up(seqlen, 128) - m_block * 128 and t < 128` (`:227`). This is the hd128
    code with kBlockM 128.
- **dQaccum clear:** `128 * 64 / 1024 = 8` iterations of 256 x float4.
- **Grid:** `(ceil(s_q / 128), h, b)`; `__launch_bounds__(256, 2)`.

## dQ convert and dK / dV convert (`TileShape (128, 64)`, 256 threads)

The two are identical in shape: `TiledMmadQ` and `TiledMmadKV` are both m64n64 per warpgroup, warpgroups along M.

- **smem:** `sAcc` 32768 B at 0, `sdQ` 16384 B at 32768, barrier at 49152; `SharedStorageSize = 49280`.
- **S2R:** register `4i + j` from `sAcc[w * 4096 + (i * 128 + t) * 4 + j]`, i in 0..7 (`ld.shared` 128-bit; extent
  8). Then `mul.ftz` by the scale (32), and `cvt.rn.bf16x2` (16).
- **STSM_N:** matrix j of instruction p = (c = 2p + j // 2, i = j % 2), row `64w + 16 * warp + 8i + lane % 8`,
  column 8c (p in 0..3; extent 4).
  - `SmemLayoutdQ = Sw<3,3,3> o ((8,16),(64,1))`: byte `m * 128 + (((c // 8) ^ (m % 8)) * 16)`.
- **gmem copy:** 8 threads per row, 32 rows per pass, 4 passes. All shared reads (128-bit, extent 4) come first,
  then the stores under `r < min(seqlen - m_block * 128, 128) and col8 < d` (extent 4).
- **Varlen:** the runtime path of the varlen sketch with kBlock = 128.

## Correctness expectations

The hd128 gates apply unchanged:
- dPsum, LSE_log2 and the converts bitwise;
- dK / dV bitwise for MHA and for GQA groups <= 2, and under q-head isolation otherwise;
- dQ / dQaccum bitwise under K-isolation;
- upstream's tolerance.

**Test grid:**
- `test_flash_attn_output` and `test_flash_attn_varlen_output` at `d = 64` (their `COMPILED_HDIMS` axis), with
  `dv_vals` restricted to `d` (dv 256 / 512 at d 64 are the hdimdiff instantiations, out of scope);
- the same mha / gqa / mqa, causal and `add_unused_qkv` axes as hd128.

<!--
This sketch is a design artifact, not source. It is written once, reviewed
once, and then frozen: the implementation follows it, and where the two
disagree the TIRx module linked below is what runs.

Documents a TIRx port of Dao-AILab/flash-attention
(https://github.com/Dao-AILab/flash-attention @ 0251105a2fb19d2957484b7f023cd8c115286ced),
hopper/seqlen.h, hopper/tile_scheduler.hpp, hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp,
hopper/epilogue_bwd.hpp, hopper/flash_bwd_preprocess_kernel.h, hopper/flash_bwd_postprocess_kernel.h,
hopper/flash_bwd_launch_template.h, hopper/flash_api_stable.cpp. Copyright (c) 2024, Jay Shah,
Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# fa3_backward_sm90, varlen: delta sketch

A delta on the reviewer-approved sketches [`fa3_backward_sm90.md`](fa3_backward_sm90.md) (MHA) and
[`fa3_backward_sm90_gqa.md`](fa3_backward_sm90_gqa.md) (GQA). Both are frozen, and every statement in them
holds unless this file replaces it. This sketch covers `Varlen = true`, the instantiation `run_mha_bwd_dispatch`
selects when `cu_seqlens_q` or `cu_seqlens_k` is non-null (`flash_bwd_launch_template.h:299`). Each of the
four existing main kernels (causal / non-causal x MHA / GQA) gets one, and every other template argument is
unchanged. The preprocess gets one (`FlashAttnBwdPreprocess<..., Clear_dQaccum = true, Varlen = true>`).
`seqused_q` / `seqused_k` are optional runtime inputs of the same instantiation. Implementation: the same
TIRx module (`tirx_kernels/ported/flashattention/_fa3_backward_sm90/`), where varlen is a compile-time flag
of every kernel builder, as `Varlen` is a template argument upstream.

**Scope of the inputs.** Upstream decides three things on different conditions:
- the `Varlen` kernels run when either `cu_seqlens` is given (`flash_bwd_launch_template.h:299`);
- the padded workspaces are used when any of `cu_seqlens_q/k` or `seqused_q/k` is given
  (`flash_api_stable.cpp:1405, 1546-1575`);
- the converts always honour `seqused` (`postprocess_kernel.h:170`).

This port covers the case `flash_attn_varlen_func` and the upstream varlen test use: **both `cu_seqlens_q` and
`cu_seqlens_k` given, `seqused_q` / `seqused_k` each optional**. Its host wrapper rejects:
- `seqused` without `cu_seqlens`, which upstream runs as the fixed-length kernels on padded workspaces with the
  converts clipped to `seqused`;
- a single-sided `cu_seqlens`, which upstream runs with `set_params_dgrad`'s dK / dV batch strides set only when
  `cu_seqlens_q` is null (`flash_api_stable.cpp:303-308`). With `cu_seqlens_q` set and `cu_seqlens_k` null,
  every batch's dK / dV would land on batch 0's rows.

The kernels still carry every runtime null test of the source (the `cu_seqlens == nullptr` branches
included), so they are the source's kernels. Only the host restricts the inputs.

## Line-info export

`.porting/fa3_backward_sm90/source_export/`:
- **Build:** `export_driver.cu -DEXPORT_VARLEN_ALL` via `export_varlen.sh`, compiled alone in
  `export_driver_varlen.cu`. It instantiates
  `run_flash_bwd<90, 128, {64, 80}, 128, bf16, causal, false, false, /*Varlen*/ true, false, /*GQA*/ {false, true}, ...>`.
  The non-causal ones have `dQ_swapAB = true`; every other argument is as in the base / GQA exports.
- **Output:** `hd128_varlen4.ptx` (9 entries), split into `hd128_varlen{,_gqa}_main_{causal,noncausal}.ptx`
  and `hd128_varlen_pre_{causal,noncausal}.ptx`. The varlen build reuses the dQ / dK-dV postprocess
  entries unchanged: the postprocess has no `Varlen` template argument and always builds
  `SeqlenInfo<true>`.
- **ptxas, MHA and GQA mains alike:** 168 registers everywhere.
  - **causal:** **8 B spill stores, 12 B spill loads, 8 B stack**;
  - **non-causal:** **4 B / 4 B, 8 B stack**.
- **ptxas, others:** preprocess 64 (m64) / 72 (m80) registers, 0 B. Postprocesses as before.

Static PTX deltas against the fixed-length mains (non-scalar ops; each cell "fixed -> varlen"):

| op | MHA causal | MHA non-causal | GQA causal | GQA non-causal |
| --- | --- | --- | --- | --- |
| `ld.global.u32` (+`.s32`) | 0 -> 37 (+2) | 0 -> 31 (+2) | 0 -> 32 | 0 -> 26 |
| `cp.async.bulk.tensor.4d.global.shared::cta` (dK / dV TMA store) | 4 -> 0 | 4 -> 0 | 0 | 0 |
| `ld.shared.v2.u64` (sdK / sdV -> registers) | 0 -> 16 | 0 -> 16 | 0 | 0 |
| `st.global.v2.u64` | 16 -> 32 (store_zero + store) | 0 -> 32 | 0 | 0 |
| `fence.proxy.async.shared::cta` | 3 -> 2 | 3 -> 2 | = | = |
| `bar.arrive` | 9 -> 8 | 9 -> 8 | = | = |
| `elect.sync`, `shfl.sync.idx` | 9 -> 8, 10 -> 9 | 9 -> 8, 10 -> 9 | = | = |
| `cp.async.bulk.commit_group` / `wait_group.read` | 7 -> 6 | 7 -> 6 | = | = |
| `prefetch.tensormap` | 6 -> 4 | 6 -> 4 | 4 | 4 |

WGMMA, TMA loads, LSE / dPsum bulk loads, the dQ path, mask, exp2 and the GQA reduce are unchanged. The
`ld.global` counts come from the cu_seqlens / seqused reads of every `SeqlenInfo` / scheduler instance
(below), which nvcc duplicates into its branches. The port matches the instances and their reads, not
nvcc's duplication count.

## Varlen semantics (`seqlen.h`)

`SeqlenInfo<Varlen, kBlock>(bidb, seqlen_static, cu_seqlens, seqused)`, and `SeqlenInfoQK<Varlen, kBlockM>`,
which is the same pair for Q (kBlockM) and K (no padded offset):

```python
offset        = 0 if cu_seqlens is null else cu_seqlens[bidb]
offset_padded = 0 if cu_seqlens is null else (cu_seqlens[bidb] + bidb * kBlock) // kBlock * kBlock
seqlen        = seqused[bidb] if seqused is not null
                else (cu_seqlens[bidb + 1] - cu_seqlens[bidb] if cu_seqlens is not null else seqlen_static)
# instruction_selection: setp.eq.s64 ptr, 0 + a branch around an unpredicated ld.global.u32 (int32
#   cu_seqlens / seqused; nvcc also folds some into ld.global.s32 to a 64-bit register); extent: scalar
```

- **Null tests:** the port tests nullness at runtime on the raw pointer, as the source does. A null input
  is passed as a 0-element tensor, whose data pointer is 0. The test is
  `reinterpret(ptr, uint64) != 0`, the form the preprocess already uses for `dq_semaphore`.
- **Batch index:** `is_varlen_q = cu_seqlens_q != null` and `is_varlen_k = cu_seqlens_k != null` choose
  whether a global tensor is indexed with batch `bidb` or batch `0`, the stride-0 batch of the varlen
  shapes.
- **Static lengths:** `seqlen_static` is `shape_Q[0]` / `shape_K[0]`, i.e. `total_q` / `total_k` when that
  side is varlen and `seqlen_q` / `seqlen_k` otherwise.

## Host (`flash_api_stable.cpp mha_bwd`, `flash_bwd_launch_template.h:40-288`)

- **Inputs, varlen side:** `q`, `dout`, `out`: `(total_q, h, d)`; `k`, `v`: `(total_k, h_k, d)`;
  `softmax_lse`: `(h, total_q)`, the varlen forward's layout; `cu_seqlens_*`: int32 `(b + 1)`; `seqused_*`: int32
  `(b)`.
- **Outputs:** `dq`: `(total_q, h, d)`; `dk`, `dv`: `(total_k, h_k, d)`.
  - Strides: row `h * d` (`h_k * d`), head `d`, element 1. The batch index is always 0, so the batch stride
    (`set_params_dgrad`'s, both sides varlen) is never applied.
  - Rows outside every sequence's used length are written by no kernel.
  - `max_seqlen_q` / `max_seqlen_k` give `seqlen_q` / `seqlen_k`, which size the grids, the semaphore and
    the LPT swizzle.
- **Padded workspaces (varlen):**
  - `total_q_padded_rounded = round_up(total_q + b * kBlockM, kBlockM)`;
  - `total_k_padded_rounded = round_up(total_k + b * 128, 128)`;
  - `softmax_d`, `softmax_lse_log2`: `(h, total_q_padded_rounded)`; `dq_accum`: `(h, total_q_padded_rounded * 128)`;
  - GQA: `dk_accum`, `dv_accum`: `(h_k, total_k_padded_rounded, 128)`, `fill_(0.0)`;
  - `dq_semaphore`: `(ceil(max_seqlen_q / kBlockM), b, h)` as before.
- **Kernel shapes / strides:**
  - **Rows:** `seqlen_q_rounded` / `seqlen_k_rounded` become the padded totals on a varlen side.
  - **Batch strides:** become 0.
  - **TensorMaps:** Q, dO, K, V are built over `(d, total, h, 1)`: the same 4-D encoding with a batch extent
    of 1. There are **no dK / dV TensorMaps** in the varlen MHA build. With `Use_TMA = false` the epilogue's
    TMA type is `std::nullptr_t` and its params hold `nullptr` (`epilogue_bwd.hpp:31, 93-102, 137-161`).
- **Grids:**
  - preprocess and dQ convert: `(ceil(max_seqlen_q / kBlockM), h, b)`;
  - main: non-causal `(ceil(max_seqlen_k / 128), h, b)`; causal (LPT) `ceil(max_seqlen_k / 128) * h * b`,
    with the swizzle sized by `max_seqlen_q`;
  - dK / dV convert: `(ceil(max_seqlen_k / 128), h_k, b)`.
- **New kernel arguments (varlen builds only):**
  - **Pointers:** `cu_seqlens_q`, `cu_seqlens_k`, `seqused_q`, `seqused_k` (each possibly null).
  - **`s_q_rounded`, `s_k_rounded`:** the `seqlen_*_rounded` shape entries above. They are host values, as
    upstream passes them. The fixed builds keep computing them in-kernel, as reviewed.
  - **`sched_seqlen`:** the scheduler's `params.seqlen` = `max_seqlen_k`, used only when both K pointers
    are null.
  - **Existing `s_q` / `s_k`:** now mean `shape_Q[0]` / `shape_K[0]`.

  The ABI of the fixed-length builds is unchanged.

Varlen MHA main kernel (the GQA one is the GQA delta's plus the same varlen inputs):

```python
@specialize(causal in {True, False}, gqa=False, varlen=True)
def flash_bwd_sm90(LSE_log2, dPsum, dQaccum, dk, dv, s_q, s_k, d, h, b, window_size_right, softmax_scale,
                   scale_log2, sched_params..., qhkh_div, qhkh_mul, qhkh_shr,
                   cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, max_seqlen_k, s_q_rounded, s_k_rounded,
                   tma_q, tma_do, tma_k, tma_v):         # dropped against the fixed MHA build: tma_dk, tma_dv
    if warp == 0 and lane_pred:
        prefetch_tensormap(tma_q, tma_do, tma_k, tma_v)
        # instruction_selection: prefetch.tensormap x4 (CollectiveEpilogueBwd::prefetch_tma_descriptors
        #   only prefetches when Use_TMA); extent: 4   -- replaces the base sketch's x6
    ...
```

## Main kernel

### Scheduler validity (`tile_scheduler.hpp:96-118`, `:440-468`)

Every role's `get_initial_work` decodes the tile as before, then reads the KV length of `bidb`:

```python
seqlen = seqused_k[bidb] if seqused_k else (cu_seqlens_k[bidb+1] - cu_seqlens_k[bidb] if cu_seqlens_k else sched_seqlen)
valid  = n_block * 128 < seqlen                 # SingleTileScheduler (non-causal)
valid  = n_block < ceil_div(seqlen, 128)         # SingleTileBwdLPTScheduler (causal), not SPT
# instruction_selection: predicated ld.global (u32 / s32); extent: scalar per role
```

An invalid tile (`bidb = -1`) skips that role's work loop entirely. That covers the producer's `load`, the
dQ warp's `store_dq`, the MMA warpgroups' `mma` and the epilogue, and no `store_zero` runs. Everything
outside the loop stays:
- the prologue (barrier init, TensorMap prefetch, `cta_sync`);
- `setmaxnreg`;
- `mma_init`'s dQEmpty arrive;
- `load_tail` (the producer tail waits on the start state, as for an empty causal tile).

### Per-tile sequence info

Each role builds `seqlen_info = SeqlenInfoQK(bidb, s_q, s_k, cu_seqlens_q, cu_seqlens_k, seqused_q,
seqused_k)` after the tile decode. That is the producer `load` (mainloop :434), `store_dq` (:602), `mma`
(:700), and the epilogue's `SeqlenInfo<Varlen, 128>(bidb, shape_dK[0], cu_seqlens_k, seqused_k)`
(epilogue :229 / :287 / :432). The values replace the fixed build's constants:

| fixed build | varlen build |
| --- | --- |
| `seqlen_q = s_q`, `seqlen_k = s_k` (mask, m-block range) | `seqlen_info.seqlen_q`, `.seqlen_k` |
| `m_block_max = ceil(s_q / BM)`; causal `m_block_min` from `s_q - s_k` | the same formulas on the per-batch lengths (`block.h:83-100`) |
| Q / dO TMA row `m * BM`, batch `bidb` | row `offset_q + m * BM`, batch `0 if is_varlen_q else bidb` |
| K / V TMA row `n_block * 128`, batch `bidb` | row `offset_k + n_block * 128`, batch `0 if is_varlen_k else bidb` |
| LSE / dPsum float base `(bidb * h + bidh) * s_q_rounded` | `(bb_q * h + bidh) * s_q_rounded + offset_q_padded`, `bb_q = 0 if is_varlen_q else bidb` |
| dQaccum float base `(bidb * h + bidh) * s_q_rounded * 128` | `((bb_q * h + bidh) * s_q_rounded + offset_q_padded) * 128` |
| GQA `dk/dv_accum` base `((bidb * h_k + bidh_kv) * s_k_rounded + n_block * 128) * 128` | `((bb_k * h_k + bidh_kv) * s_k_rounded + offset_k_padded + n_block * 128) * 128` |

- **Empty tiles:** the producer's, `store_dq`'s and `mma`'s early exits on `m_block_max <= m_block_min` now
  hold for non-causal too (`Is_causal || Is_local || Varlen`, mainloop :442, :611, :708). A KV block of a
  batch with `seqlen_q = 0` is valid in the scheduler and empty in the mainloop.
  - **MHA:** it runs `store_zero`, which the non-causal varlen build therefore gains.
  - **GQA:** it runs nothing, as before.
- **Producer's row base:** the producer keeps its causal / non-causal split of how the int64 row base is held
  (ISSUE-022 (A)). With `offset_q_padded` added it is re-measured. Upstream itself spills 8 B / 4 B here, so
  the gate is "spill <= upstream's".
- **TMA reads past a batch:** a Q / dO / K / V box that runs past the end of its batch reads the next batch's
  rows, or zero past `total`. This is upstream's behaviour too: the mask (per-batch `seqlen_k` column limit)
  and the preprocess's `LSE_log2 = +inf` / `dPsum = 0` for rows past `seqlen_q` make those rows contribute
  nothing.

### MHA epilogue, varlen: `CollectiveEpilogueBwd::store` with `Use_TMA = false` (`epilogue_bwd.hpp:163-276`)

```python
def epilogue_store_varlen(rdK, rdV, n_block, bidh, bidb):
    # cvt.rn.bf16x2.f32 x64 and STSM_N into sdV then sdK exactly as the TMA build
    # (named_sync(1, 256) first). SmemLayoutdKV is now SmemLayoutAtomdKVSTG = Swizzle<3,3,3> o (8,64):(64,1)
    # tiled to (128, 128), M first: byte(r, c) = (c // 64) * 16384 + r * 128 + (((c % 64) // 8) ^ (r % 8)) * 16
    # + (c % 8) * 2, the same bytes as the TMA build's SW128 layout, so the STSM addresses are unchanged.
    named_sync(1, 256)                           # instruction_selection: bar.sync 1, 256 (no fence, no bar.arrive 1,288)
    si = SeqlenInfo(bidb, s_k, cu_seqlens_k, seqused_k)   # kBlock 128
    bb = 0 if is_varlen_k else bidb
    # GmemTiledCopydKV: 256 threads as (16 rows, 16 threads per row), 8 bf16 (16 B) per thread, 8 row passes
    col8 = (tid % 16) * 8
    for src in (sdV, sdK):                       # flash::copy smem -> registers, EvenN: no predicate
        for k in range(8):
            r = tid // 16 + 16 * k
            reg[src][k] = ld_shared_128(byte(r, col8))
            # instruction_selection: ld.shared.v2.u64; extent: 16
    rows = min(si.seqlen - n_block * 128, 128)
    for dst, src in ((dV, sdV), (dK, sdK)):      # flash::copy registers -> gmem, dV first
        for k in range(8):
            r = tid // 16 + 16 * k
            if r < rows and col8 < d:
                st_global_128(dst[bb, si.offset + n_block * 128 + r, bidh, col8:col8 + 8], reg[src][k])
                # instruction_selection: st.global.v2.u64; extent: 16
```

`store_zero` (epilogue :280-318) is the base sketch's, with row `si.offset + n_block * 128 + r`, batch `bb`
and the row limit `si.seqlen - n_block * 128`.

### GQA epilogue, varlen (`epilogue_bwd.hpp:432-437`)

Unchanged except for the accumulator base in the table above (`offset_padded` of
`SeqlenInfo<Varlen, 128>(bidb, shape_dKaccum[0], cu_seqlens_k, seqused_k)`, batch `bb_k`). The 64 KiB
reduce-add covers the whole 128-row tile even past `seqlen_k`. The padded layout keeps it inside this batch's
padding.

## Preprocess, varlen (`flash_bwd_preprocess_kernel.h:145-245`)

```python
si = SeqlenInfo(bidb, s_q, cu_seqlens_q, seqused_q)         # kBlock = kBlockM
if is_varlen and m_block * BM >= si.seqlen: return          # instruction_selection: setp + bra; extent: scalar
bb = 0 if is_varlen else bidb
# O / dO rows si.offset + m_block * BM + r of batch bb, row predicate r < si.seqlen - m_block * BM (as before
# with seqlen_o = si.seqlen). LSE source (h, total_q): index (bb * h + bidh) * s_q + si.offset + m_block * BM + t,
# predicate t < si.seqlen - m_block * BM (else +inf).
# dPsum / LSE_log2 at float (bb * h + bidh) * s_q_rounded + si.offset_padded + m_block * BM + row; LSE_log2
# rows t < round_up(si.seqlen, BM) - m_block * BM.
# dQaccum clear: float ((bb * h + bidh) * s_q_rounded + si.offset_padded) * 128 + m_block * BM * 128, extent
# BM * 128 floats (GmemTiledCopyAccum, 256 threads x 4 floats, BM * 128 / 1024 iterations), as before.
# dq_semaphore[bidh + bidb * h + m_block * h * b] = 0 under the same null test (index uses bidb, not bb).
```

## dQ / dK-dV convert, varlen (`flash_bwd_postprocess_kernel.h:160-253`)

Upstream has one postprocess kernel that always builds `SeqlenInfo<true>` and tests `cu_seqlens` at runtime.
The base sketch kept the fixed-length path only. Here the varlen builds pass `cu_seqlens`, `seqused` and
`s_rounded`, and the kernel runs the source's runtime path:

```python
si = SeqlenInfo(bidb, s, cu_seqlens, seqused)               # kBlock = kBlockM (dQ) or 128 (dK, dV)
if is_varlen and m_block * BM >= si.seqlen: return
bb = 0 if is_varlen else bidb
# accumulator g2s from float index ((bb * heads + bidh) * s_rounded + si.offset_padded + m_block * BM) * 128
# output rows bb, si.offset + m_block * BM + r, predicate r < min(si.seqlen - m_block * BM, BM) and col8 < d
```

Everything else (bulk g2s, S2R, scale, cvt, STSM / STSM_T, swizzle, the smem read-back) is unchanged.

## Launch sequence

The launch sequence is unchanged:
1. (GQA) `dk_accum.fill_(0)`, `dv_accum.fill_(0)`;
2. preprocess, main, dQ convert;
3. (GQA) dK convert, dV convert.

## Correctness expectations

**Test grid:** upstream's `test_flash_attn_varlen_output` for bf16 d128:
- 22 `(seqlen_q, seqlen_k)` pairs x causal x `add_unused_qkv` (seqused) x mha / gqa / mqa;
- b 9 (2 when `seqlen_q > 2048`), h 6, h_k 6 / 2 / 1;
- `generate_random_padding_mask`: query lengths >= 1; key lengths may be 0, and are 0 for batches 0, 5, ... and
  the last.

**Bitwise gates, from NaN sentinels:** the same set as fixed length:
- dPsum, LSE_log2, the dQ / dK / dV converts;
- dK / dV outright for MHA, and for GQA with <= 2 q-heads per KV head;
- dQ / dQaccum under K-isolation;
- dK / dV / accumulators under q-head isolation.

**Varlen detail of the gates:**
- **K-isolation:** keeps the first and last KV block of every batch, counted from that batch's own
  `cu_seqlens_k`.
- **Positions nobody writes:** padding rows, unused (`seqused`) rows, the padded workspace tails. They keep the
  sentinel on both sides and are compared bitwise as NaN patterns.
- **Tolerance:** dQ / dK / dV are additionally checked against upstream's test tolerance on the padded
  re-layout with the padding masks, as the upstream test does.

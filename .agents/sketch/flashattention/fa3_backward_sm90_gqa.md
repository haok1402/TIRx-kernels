<!--
This sketch is a design artifact, not source. It is written once, reviewed
once, and then frozen: the implementation follows it, and where the two
disagree the TIRx module linked below is what runs.

Documents a TIRx port of Dao-AILab/flash-attention
(https://github.com/Dao-AILab/flash-attention @ 0251105a2fb19d2957484b7f023cd8c115286ced),
hopper/epilogue_bwd.hpp (CollectiveEpilogueBwdGQA), hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp,
hopper/flash_bwd_postprocess_kernel.h, hopper/flash_bwd_launch_template.h. Copyright (c) 2024, Jay Shah,
Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# fa3_backward_sm90, GQA / MQA: delta sketch

A delta on the reviewer-approved MHA sketch [`fa3_backward_sm90.md`](fa3_backward_sm90.md) (frozen; every
statement there holds unless this file replaces it). It covers the GQA specializations of the same two
`run_flash_bwd` instantiations: `GQA = true` (selected by `mha_bwd` when `h != h_k`, MQA included), all
other template arguments identical. Implementation: the same TIRx module
(`tirx_kernels/ported/flashattention/_fa3_backward_sm90/`); the GQA build is a compile-time flag of the
kernel builders.

## Line-info export

`.porting/fa3_backward_sm90/source_export/`: `export_driver.cu` with `-DEXPORT_GQA` instantiates
`run_flash_bwd<90, 128, 64, 128, bf16, true, false, false, false, false, /*GQA*/ true, 2, 2, true, false, false, 2, 1, 2, 1, false>`
and the `<..., 80, ..., false, ..., true, ..., true, ...>` non-causal sibling (`export_gqa.sh hd128_gqa`):
`hd128_gqa.ptx` (9 entries: the MHA six plus the two GQA mains and the dK/dV postprocess), split into
`hd128_gqa_main_causal.ptx`, `hd128_gqa_main_noncausal.ptx`, `hd128_gqa_post_dkv.ptx`. ptxas: both GQA
mains 168 registers, **0 B spill** (the MHA non-causal main's 4 B spill does not occur here); the dK/dV
postprocess 88 registers, 0 B.

Static PTX deltas against the MHA mains; each cell is "causal / non-causal":

| op | MHA | GQA |
| --- | --- | --- |
| `st.shared.v4.u32` | 8 / 10 (dQ R2S) | 40 / 42 (dQ + 16 dV + 16 dK R2S) |
| `cp.reduce.async.bulk...add.f32` | 6 / 6 (dQ) | 8 / 8 (+ dV, dK) |
| `cp.async.bulk.commit_group` | 7 / 7 | 8 / 8 |
| `cp.async.bulk.wait_group.read` | 7 / 7 | 8 / 8 |
| `bar.sync` | 11 / 11 | 13 / 13 |
| `bar.arrive` | 9 / 9 | 8 / 8 |
| `prefetch.tensormap` | 6 / 6 | 4 / 4 (no dK / dV TensorMaps) |
| `stmatrix...x4.m8n8` (epilogue STSM_N) | 16 / 16 | 0 / 0 |
| `cp.async.bulk.tensor.4d.global.shared::cta` (TMA store) | 4 / 4 | 0 / 0 |
| `st.global.v2.u64` (store_zero) | 16 / 0 | 0 / 0 |
| `mul.hi.u32` | 13 / 1 | 14 / 2 (+ the epilogue's qhead_per_khead divmod) |

WGMMA, TMA-load, LSE / dPsum, mask, exp2 and dQ-path counts are unchanged.

## What changes

### Host (`flash_bwd_launch_template.h:117-184`, `:241-288`; `flash_api_stable.cpp mha_bwd`)

- `dk_accum`, `dv_accum`: fp32 `(b, h_k, seqlen_k_rounded * 128)`, zero-filled by `mha_bwd`
  (`torch::stable::fill_`, two launches before the backward) -- the port zero-fills them the same way, as
  part of the launch sequence.
- Main kernel arguments: K / V TensorMaps over `(d, s_k, h_k, b)`; the epilogue gets the dK / dV accumulator
  pointers instead of dK / dV TensorMaps; `qhead_per_khead = ceil(h / h_k)` as a FastDivmod (divisor,
  multiplier, shift). The scheduler still enumerates query heads (`num_head = h`).
- After the dQ convert, two more launches of the same `FlashAttnBwdPostprocessConvertdQ` template with
  `TileShape_NK = (128, 128)`, `kNThreads = 256` (NumEpilogueThreads), `TiledMmadKV`, `dKV_swapAB = false`,
  `softmax_scale = 1.f`: dK from `dk_accum`, then dV from `dv_accum`, grid `(ceil(s_k / 128), h_k, b)`.

### Main kernel

```python
@specialize(causal in {True, False}, gqa=True)
def flash_bwd_sm90(tma_q, tma_do, tma_k, tma_v, LSE_log2, dPsum, dQaccum, dk_accum, dv_accum,
                   s_q, s_k, d, h, h_k, b, window_size_right, softmax_scale, scale_log2, sched_params...,
                   qhkh_div, qhkh_mul, qhkh_shr):
    # Dropped against the MHA build: tma_dk, tma_dv (the GQA epilogue stores no bf16 dK / dV) and the
    # bf16 dK / dV pointers (store_zero is empty). qhead_per_khead = ceil(h / h_k) as one FastDivmod
    # triple (the source's mainloop and epilogue Params each carry a copy of the same value).
    if warp == 0 and lane_pred:
        prefetch_tensormap(tma_q, tma_do, tma_k, tma_v)
        # instruction_selection: prefetch.tensormap x4 (CollectiveEpilogueBwdGQA::prefetch_tma_descriptors
        #   is empty, epilogue_bwd.hpp:410-412); extent: 4   -- replaces the base sketch's x6
    ...
```

Producer warp 0 (`mainloop ... :456`): the only change is the KV head of the K and V loads.

```python
    bidh_kv = fdiv(bidh, qhkh)                                   # qhead_per_khead_divmod.divide
    # instruction_selection: setp.eq.s32 divisor, 1 + predicated branch around mul.hi.u32 + shr.u32
    #   (FastDivmod::divide); extent: scalar. The MHA build runs the same divide with divisor 1 (the
    #   base sketch omits it; the implementation carries it in both builds, as the source does).
    copy_g2s(K[bidb, n_block*128:+128, bidh_kv, :], sK, completion=barrier_KV, cache=EVICT_NORMAL)
    copy_g2s(V[bidb, n_block*128:+128, bidh_kv, :], sV, completion=barrier_KV, cache=EVICT_NORMAL)
    # instruction_selection: cp.async.bulk.tensor.4d ... .L2::cache_hint 0x1000000000000000, box {64, 128}
    #   x2 each (TensorMaps over h_k heads); extent: 4
```

Q, dO, LSE, dPsum loads, the dQ-reduce warp, `bwd_step` and the dK scale are unchanged (they index query
heads). `mma()` returns false for an empty causal tile as before, but the GQA epilogue's `store_zero` is
empty (`epilogue_bwd.hpp:524-532`: the accumulators are already zero), so the empty-tile branch emits
nothing.

Epilogue `CollectiveEpilogueBwdGQA::store` (`epilogue_bwd.hpp:414-518`, `Use_TMA`, not Deterministic), the
256 MMA threads, `tid` = thread index within them, `w` = MMA warpgroup, `t = tid % 128`:

```python
def epilogue_store_gqa(rdK, rdV, n_block, bidh, bidb):           # rdK already * softmax_scale
    bidh_kv = fdiv(bidh, qhkh)                                   # divmod (bidh_idx_in_group unused)
    # instruction_selection: setp.eq.s32 divisor, 1 + predicated branch around mul.hi.u32 + shr.u32;
    #   extent: scalar
    sdKV = alias(pool[0 : 65536], f32)                           # TensorStorageTMA smem_dkv, 1024-aligned union
                                                                 # with the mainloop storage (aliases sK, sV)
    named_sync(1, 256)                                           # EpilogueBarrier: all WGs done reading K, V
    # instruction_selection: bar.sync 1, 256
    copy_r2s(rdV, sdKV[w * 8192 + (i * 128 + t) * 4 + j] for register 4i + j)   # R2STiledCopydKVaccum
    # instruction_selection: st.shared.v4.u32 (128-bit); extent: 16
    fence_proxy_async()                                          # instruction_selection: fence.proxy.async.shared::cta
    named_sync(1, 256)                                           # instruction_selection: bar.sync 1, 256
    if tid == 0:
        reduce_add_s2g(sdKV, dv_accum[bidb, bidh_kv, n_block * 16384 : +16384], cache=EVICT_LAST)
        # instruction_selection: cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32,
        #   65536 B; extent: 1
        bulk_commit()                                            # instruction_selection: cp.async.bulk.commit_group
        bulk_wait_read(0)                                        # instruction_selection: cp.async.bulk.wait_group.read 0
    named_sync(1, 256)                                           # instruction_selection: bar.sync 1, 256
    copy_r2s(rdK, sdKV[w * 8192 + (i * 128 + t) * 4 + j] for register 4i + j)
    # instruction_selection: st.shared.v4.u32; extent: 16
    fence_proxy_async()                                          # instruction_selection: fence.proxy.async.shared::cta
    named_sync(1, 256)                                           # instruction_selection: bar.sync 1, 256
    if tid == 0:
        reduce_add_s2g(sdKV, dk_accum[bidb, bidh_kv, n_block * 16384 : +16384], cache=EVICT_LAST)
        # instruction_selection: cp.reduce.async.bulk ... add.f32, 65536 B; extent: 1
        bulk_commit()                                            # instruction_selection: cp.async.bulk.commit_group
        bulk_wait_read(0)                                        # instruction_selection: cp.async.bulk.wait_group.read 0
```

There is no bf16 conversion, STSM, TMA store, or `bar.arrive 1, 288` in the GQA epilogue; the
`tdKrdK *= softmax_scale` of `mma()` stays. `dv_accum` / `dk_accum` index: float
`((bidb * h_k + bidh_kv) * seqlen_k_rounded + n_block * 128) * 128`, i.e. per (b, h_kv) a sequence of
128 x 128 tiles, each holding the two warpgroups' halves in the C-fragment register order above.

### dK / dV convert (`flash_bwd_postprocess_kernel.h`, `TileShape_MK = (128, 128)`, 256 threads)

The MHA sketch's kernel 3 with these values: `kBlockM = 128` (KV rows), `NREG = 64` (the m64n128 dV/dK C
fragment), `sAcc` 65536 B at 0, `sdKV` 32768 B at 65536, barrier at 98304 (`SharedStorageSize` 98432),
`softmax_scale` argument 1.0 (runtime, so the `mul.ftz.f32` x64 remains), grid `(ceil(s_k / 128), h_k, b)`,
`__launch_bounds__(256, 2)`; `seqlen = s_k`.

- `MmaShapeN` of `TiledMmadKV` = 128 -> `kBlockKSmem = 64`, `Swizzle<3,3,3>` 8 x 64 atoms:
  `byte(m, c) = (c // 64) * 16384 + m * 128 + ((((c % 64) // 8) ^ (m % 8)) * 16) + (c % 8) * 2`.
- S2R: register `4i + j` of thread `t` of warpgroup `w` from `sAcc[w * 8192 + (i * 128 + t) * 4 + j]`
  (`ld.shared` 128-bit; extent 16); `mul.ftz.f32` by the scale (extent 64); `cvt.rn.bf16x2.f32` (extent 32);
  STSM_N (dKV_swapAB = false): matrix j of instruction p is
  (c = 2p + j // 2, i = j % 2), row m = 64w + 16 * warp + 8i + lane % 8, column 8c
  (`stmatrix.sync.aligned.x4.m8n8.shared.b16`; extent 8); gmem copy 16 threads per row, 8 row
  iterations, all shared reads (`ld.shared` 128-bit, extent 8) then the stores under
  `r < min(s_k - n_block * 128, 128) and col8 < d` (`st.global` 128-bit; extent 8).
- Launch order: dK convert, then dV convert (`flash_bwd_launch_template.h:287-288`).

## Correctness expectations

dK / dV are now sums over `qhead_per_khead` CTAs' fp32 bulk reduce-adds, in arrival order: bitwise
reproducible against flash_attn_3 when at most two query heads share a KV head (`0 + a + b == 0 + b + a`),
otherwise within the upstream test's tolerance. The dK / dV convert alone is checked bitwise by feeding it
upstream's `dk_accum` / `dv_accum` (returned by `mha_bwd`). dQ, dPsum, LSE * log2 e as for MHA.

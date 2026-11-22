<!--
Copyright (c) 2025 DeepSeek
Copyright (c) 2026 The TIRx Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.

This design sketch documents a TIRx port of DeepGEMM's
deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh together with the
scheduler and TMA/WGMMA helpers it instantiates. See NOTICE and licenses/ for
upstream attribution.
-->

# SM90 FP8 1D2D GEMM (m-grouped), coverage extension: delta sketch

**Status: DRAFT, awaiting sketch review.** Instruction selections are taken from the line-info export of
all 17 new specializations (`.porting/sm90_fp8_gemm_1d2d/source_export_ext/`, driver
`sm90-probe/dg90/export_ext.py`; each case checked against a bf16 reference, calc_diff < 1e-3). The
per-class PTX totals the port must reproduce are in the appendix.

This is a non-executable delta against the approved base sketch
[`sm90_fp8_gemm_1d2d.md`](sm90_fp8_gemm_1d2d.md), which stays authoritative for everything not
restated here. The base sketch covers BLOCK_N in {96, 128, 192}, multicast on B or none. This delta
widens the scope to the specializations DeepGEMM's SM90 heuristic picks for realistic MoE shapes.
Those 17 classes are 25% of 552 DeepSeek-V3/V3.2 and Qwen3-MoE shapes; see `sm90-probe/logs/dg90_reach.json`.

| new class (kind, BM x BN, cluster m x n) | B scales | D swizzle (bytes) / `TMA_D_BLOCK_N` / stores per block | multicast |
| --- | --- | --- | --- |
| masked 64x32, contig/psum 128x32 | uniform (`128 % 32 == 0`) | 64 / 32 / 1 | none |
| masked 64x48, contig/psum 128x48 | non-uniform, **not dispatched** (`128/gcd = 8 > 4`) | 32 / 16 / 3 | none |
| masked 64x64, contig/psum 128x64 | uniform | 128 / 64 / 1 | none |
| masked 64x112 (2x1) | non-uniform, not dispatched | 32 / 16 / 7 | on B |
| contig/masked/psum 128x112 (1x2) | non-uniform, not dispatched | 32 / 16 / 7 | **on A** |
| contig/masked/psum 128x128 (1x2) | uniform | 128 / 64 / 2 | **on A** |
| contig 128x160 (2x1) | non-uniform, **dispatched**, BN > BK | 64 / 32 / 5 | on B |

## Delta 1: multicast on A (`kIsTMAMulticastOnA`, `cluster_n = 2`)

```python
# scheduler (gemm.cuh:14-26, 117-153): group along N instead of M
kNum1DBlocksPerGroup = argmin_{c in (8, 16)} c*BLOCK_N + ceil(kNumSMs/c)*BLOCK_M   # first wins ties
primary, secondary = num_n_blocks, num_m_blocks
m_block_idx = in_group // num_blocks_in_group
n_block_idx = first + in_group % num_blocks_in_group
# (the unaligned-multicast tail fix applies unchanged)
is_tma_multicast_valid(m_block_idx) = num_blocks_in_group != 1      # contiguous on-A: no group compare

# producer (.cuh:178-205): A and SFA are shared by the two CTAs (same m block, adjacent n blocks)
mc_a = kNumTMAMulticast if valid else 1;  mc_b = 1
copy_g2s(tmap_a.coord(k*128, m_idx), smem_a[s], full[s], multicast_mask=0b11 if mc_a > 1 else None)
copy_g2s(tmap_sfa.coord(m_block*BM, sfa_k), smem_sfa[s], full[s], multicast_mask=0b11 if mc_a > 1 else None)
#   with mc_a > 1 only cluster rank 0 issues them (tma_copy.cuh:46-55)
# instruction_selection (export, contig 128x112 on A): A and SFA use only
#   cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint
#   (tma_copy.cuh:50, mask %rs = 0b11, hint EVICT_NORMAL); B uses only the plain ...bytes.L2::cache_hint form
#   (tma_copy.cuh:31). The A/SFA plain branch is absent from the PTX: shape_n is a template constant, so nvcc
#   proves num_blocks_in_group != 1 (ceil(N/BN) a multiple of the group width) and folds is_tma_multicast_valid
#   to true. The port keeps the source's runtime branch and relies on the same folding; the PTX counts check it.
copy_g2s(tmap_b.coord(k*128, n_idx), smem_b[s], full[s])         # per CTA
expect_bytes(full[s], A_STAGE + B_STAGE + BM*4)                   # unchanged
```

The consumers' empty-barrier arrivals, `is_peer_cta_alive` and the drain are unchanged from the base
sketch. Heuristic legality: masked and psum require `ceil(N/BN) % 2 == 0` (`sm90.hpp:84-89`).

## Delta 2: uniform B scales at BLOCK_N 32 and 64

This is the same path as BLOCK_N 128 in the base sketch:
- `kMustUseUniformedScaleB` holds;
- one `scale_b_0` per K block;
- `num_sfb = shape_k_scales`;
- one dispatch copy;
- the promotion predicate is the constant `True`.

Only the WGMMA shape (m64n32k32 or m64n64k32; `kNumAccum` = 16 or 32) and the D atoms change.
- **instruction_selection (export):** `wgmma.mma_async.sync.aligned.m64n{32,64}k32.f32.e4m3.e4m3` (32 per
  instantiation, as for BLOCK_N 128); promotion 16 `mul.f32` + `kNumAccum*8` `fma.rn.f32`, no `selp`;
  `stmatrix.sync.aligned.x2.m8n8.shared.b16` x `kNumAccum/4` (4 or 8). No `.ftz`.

## Delta 3: non-uniform B scales without the dispatch (BLOCK_N 48, 112)

`kShouldOptimize` is false because `128/gcd(128, BN) = 8 > 4`.
- `sel = 0`, there is one copy, and the predicate in the promotion is the **runtime**
  `i < num_former_iters` (`.cuh:342`; "making it as predicates is very important for performance").
- `num_former_iters = min(BN, 128 - n_block*BN % 128) / 8` per block.
- `num_sfb = 2 * shape_k_scales` when `num_former_iters < num_full_iters`.

```python
for i in static_range(kNumAccum // 4):
    p = i < num_former_iters                       # runtime, per element group
    fma(dst[4i+0], select(p, s00, s01), accum[4i+0])   # ... x4
# instruction_selection (export, 128x112 and 128x48): nvcc factors the select out of the products:
#   per i (i = 0 folds to the first scale; 13 selp per unrolled copy at BN 112): setp on min.u32(...) -> selp.f32 sb = p ? scale_b_0 : scale_b_1,
#   then mul.f32 sa0*sb and mul.f32 sa1*sb, then 4 fma.rn.f32 (.cuh:342-346, "(predicate ? s00 : s01) * acc").
#   That is bitwise the source expression (sa*(p ? b0 : b1) == p ? sa*b0 : sa*b1). Totals for 128x112:
#   104 selp.f32, 208 mul.f32, 448 fma.rn.f32. The port emits exactly this form: txl.select on the B
#   scales, then explicit PTX mul.f32 x2 and fma.rn.f32 x4 (ISSUE-010 workaround: asm volatile muls would
#   block nvcc's own factoring, so the port writes the factored form itself).
```

## Delta 4: dispatched non-uniform BLOCK_N 160 (BN > BK)

`gcd(128, 160) = 32` gives `kShouldOptimize`, `kGap = 4` and `kEnd = 16`. A 160-wide block spans two
128-wide SFB blocks for every `n_block`.
- `num_former_iters` takes 16, 12, 8 or 4, so the 0 branch should never be taken.
- The export decides which copies the compiler keeps.
- The 1D2D unroll legality check, `160 % (160-128) == 0`, holds.
- **instruction_selection (export):** 160 `m64n160k32`, 144 `mul.f32`, 3200 `fma.rn.f32`, no `selp.f32`: every
  kept copy has a constant predicate pattern. The port reproduces the copies the base sketch's dispatch
  emits and checks these totals.

## Delta 5: 32B and 64B D swizzles (BLOCK_N 32/48/112/160)

`TMA_D_BLOCK_N = kSwizzleDMode / 2`. The epilogue uses the general bank-group form, since the
128B-swizzle shortcut does not apply:

```python
bank_group = in_atom + lane_idx * (kSwizzleDMode // 16)
row, col = bank_group // 8, bank_group % 8
col ^= row % (kSwizzleDMode // 16)
d_off = warp*(16*kSwizzleDMode) + m_off*kSwizzleDMode + atom*BLOCK_M*kSwizzleDMode + row*128 + col*16
```

- `BLOCK_N / TMA_D_BLOCK_N` threads each issue one TMA store: 1, 3, 7 or 5, per the table above.
- **instruction_selection (export):** unchanged `stmatrix.sync.aligned.x2.m8n8.shared.b16` (`kNumAccum/4`)
  and one `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` site (issued by `BLOCK_N / TMA_D_BLOCK_N`
  threads).

## Port changes this implies

- `spec.make_spec`: accept `cluster_n = 2` and these BLOCK_N values once each class is exported, reviewed
  and correct.
- `kernel.py`:
  - the producer's A/SFA multicast branch (the scheduler's on-A swizzle is already written);
  - the runtime-predicate promotion path, which today raises `NotImplementedError`;
  - nothing new in the epilogue, whose general bank-group branch exists but has not run.
- CONFIGS: one shape per new class from the reach table, plus edges (odd N-block counts with
  on-A multicast).

## Appendix: per-class PTX totals from the export (fidelity targets)

Counts are per instantiation, over the whole kernel. The TMA-load counts depend on nvcc's unrolling of
`tma_copy`'s atom loop and are compared, not designed for.

| psum 128x112_s6, on A | 32 m64n112k32 | 208 | 448 | 104 | 14 | 8 / 4 | 1 | 0 |
| psum 128x128_s5, on A | 32 m64n128k32 | 16 | 512 | 0 | 16 | 8 / 4 | 1 | 0 |
| psum 128x32_s10, none | 32 m64n32k32 | 16 | 128 | 0 | 4 | 0 / 48 | 1 | 0 |
| psum 128x48_s9, none | 32 m64n48k32 | 80 | 192 | 40 | 6 | 0 / 12 | 1 | 0 |
| psum 128x64_s8, none | 32 m64n64k32 | 16 | 256 | 0 | 8 | 0 / 36 | 1 | 0 |
| contig 128x112_s6, on A | 32 m64n112k32 | 208 | 448 | 104 | 14 | 8 / 4 | 1 | 0 |
| contig 128x128_s5, on A | 32 m64n128k32 | 16 | 512 | 0 | 16 | 8 / 4 | 1 | 0 |
| contig 128x160_s5, on B | 160 m64n160k32 | 144 | 3200 | 0 | 20 | 4 / 12 | 1 | 0 |
| contig 128x32_s10, none | 32 m64n32k32 | 16 | 128 | 0 | 4 | 0 / 12 | 1 | 0 |
| contig 128x48_s9, none | 32 m64n48k32 | 80 | 192 | 40 | 6 | 0 / 12 | 1 | 0 |
| contig 128x64_s8, none | 32 m64n64k32 | 16 | 256 | 0 | 8 | 0 / 36 | 1 | 0 |
| masked 128x112_s6, on A | 32 m64n112k32 | 208 | 448 | 104 | 14 | 8 / 4 | 1 | 0 |
| masked 128x128_s5, on A | 32 m64n128k32 | 16 | 512 | 0 | 16 | 8 / 4 | 1 | 0 |
| masked 64x112_s9, on B | 32 m64n112k32 | 208 | 448 | 104 | 14 | 4 / 12 | 1 | 0 |
| masked 64x32_s16, none | 32 m64n32k32 | 16 | 128 | 0 | 4 | 0 / 48 | 1 | 0 |
| masked 64x48_s15, none | 32 m64n48k32 | 80 | 192 | 40 | 6 | 0 / 12 | 1 | 0 |
| masked 64x64_s13, none | 32 m64n64k32 | 16 | 256 | 0 | 8 | 0 / 36 | 1 | 0 |

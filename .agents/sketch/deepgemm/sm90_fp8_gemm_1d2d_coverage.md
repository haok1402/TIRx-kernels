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

# producer (.cuh:178-205): the base sketch's producer (base :293-322) applies unchanged, with
mc_a = kNumTMAMulticast if (kIsTMAMulticastOnA and valid) else 1    # now 2 when valid
mc_b = 1                                                            # on-A: B is never multicast
# so, per K block (base :305-322, explicit branch kept):
#   A, SFA:  if mc_a == 1: copy_g2s(plain)          elif cluster_rank() == 0: copy_g2s(multicast_mask=0b11)
#   B:       copy_g2s(plain)                        (mc_b == 1)
#   expect_bytes(full[s], A_STAGE + B_STAGE + BM*4)                  # 31232 B at 128x112; unchanged formula
# instruction_selection (export, contig/masked/psum 128x112 and 128x128 on A, N blocks even):
#   A, SFA multicast branch: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
#     .multicast::cluster.L2::cache_hint, mask operand %rs = 3, hint EVICT_NORMAL (tma_copy.cuh:50); extent: one box
#     into both CTAs.
#   A, SFA plain branch: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
#     (tma_copy.cuh:31), per CTA. ABSENT from these exports: N is a template constant and ceil(N/BN) is even, so
#     nvcc proves num_blocks_in_group != 1 (first is a multiple of kNum1DBlocksPerGroup, so num_n_blocks - first is
#     even) and folds is_tma_multicast_valid to true. With ceil(N/BN) odd (legal only for MGroupedContiguous,
#     sm90.hpp:87-91) the tail column has num_blocks_in_group == 1 and this branch is live; its selection comes
#     from the odd-N edge export (contig g4 n6352 k2048, 57 N blocks, `export_ext.py` RAW_CASES).
#   B: only the plain form above, per CTA (tma_copy.cuh:31).
#   expect_bytes: mbarrier.arrive.expect_tx.shared::cta.b64, one per K block (base :323-324).
#   Producer K loop (.cuh:183, a runtime-form loop over the compile-time num_total_k_blocks): nvcc unrolls it by
#     the expect_tx count per class in the appendix (4, 12 or 16). The port keeps the base's `while_` over the same
#     constant trip count, so the same backend decision applies; the per-class count is compared, and a mismatch
#     goes to the perf ledger (it cannot change results).
```

The consumers' empty-barrier arrivals, `is_peer_cta_alive` and the drain are unchanged from the base
sketch. Heuristic legality (`sm90.hpp`):
- any cluster of 2, on A or on B: masked and psum require `ceil(N/BN) % 2 == 0` (`:87-91`); plain contiguous does not;
- `num_sms % 2 == 0` (`:78`);
- one-wave shapes never get a cluster (`:234-235`), so CONFIGS that exercise multicast need more than one wave.

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
# instruction_selection (export; contig 128x112 kernel.ptx :1695-1717, :1851; 128x48 :1227-1229, :1287):
#   nvcc factors the select out of the products, and merges i = 0 and i = 1. Per K block:
#   - i in {0, 1}: ONE pair under p01 = (num_former_iters != 0)  [setp.ne.s32 on 8*nfi = min(128 - n*BN%128, BN)]:
#       selp.f32 sb = p01 ? scale_b_0 : scale_b_1; mul.f32 s0 = sa0*sb; mul.f32 s1 = sa1*sb; 8 fma.rn.f32.
#     (n*BN % 128 is a multiple of 16 for BN in {48, 112}, so nfi is even and `0 < nfi` == `1 < nfi`.)
#   - i = 2 .. kNumAccum/4 - 2: per i, setp.gt.u32 (8*nfi > 8*i) -> selp.f32 -> 2 mul.f32 -> 4 fma.rn.f32.
#   - last i: the compare is taken before the min, (128 - n*BN%128) > BN - 1; same selp/mul/fma.
#   Totals: BN 112: 13 selp + 26 mul per K block (104 / 208 over 8 unrolled bodies); BN 48: 5 + 10 (40 / 80).
#   Bitwise: (p ? sa*b0 : sa*b1) == sa*(p ? b0 : b1) -- the select picks exact operands, mul.rn is deterministic
#   and there is no .ftz -- so writing the factored form changes no result.
#   The port writes exactly this form with explicit PTX (ISSUE-010 workaround; asm muls cannot be CSE'd, so the
#   shared i = 0/1 pair is written once, not per i): txl.select on the B scales, ptx mul.f32 x2, ptx fma.rn.f32 x4.
#   The last-i compare and the per-i setp come from the same comparisons written on the source's values.
```

## Delta 4: dispatched non-uniform BLOCK_N 160 (BN > BK)

`gcd(128, 160) = 32` gives `kShouldOptimize`, `kGap = 4` and `kEnd = 16`. A 160-wide block spans two
128-wide SFB blocks for every `n_block`.
- `num_former_iters` takes 16, 12, 8 or 4, so the 0 branch should never be taken.
- The export decides which copies the compiler keeps.
- The 1D2D unroll legality check, `160 % (160-128) == 0`, holds.
- Legality: the 1D2D unroll rule `sm90.hpp:84` (`160 % (160 - 128) == 0`) and the kernel's assert `.cuh:59-61`
  (`gcd(160, 128) = 32 == 160 - 128`).
- **Kept copies (export):** all five, `num_former_iters` in {0, 4, 8, 12, 16}, including the never-taken 0 copy (as with
  BN 192 in the base): 160 `m64n160k32` = 5 copies x 32.
- **instruction_selection (export):** 144 `mul.f32` = 8 bodies x (2 + 4 + 4 + 4 + 4): the 0 copy has only `s01`/`s11`,
  copies 4..16 have all four (16 < `kNumAccum/4` = 20). 3200 `fma.rn.f32` = 80 x 8 x 5. No `selp.f32`: every kept
  copy has a constant predicate pattern. The port's `k_loop(former)` copies with the existing `uses_former` /
  `uses_latter` emission (`kernel.py`) produce exactly this set.

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
- CONFIGS (one per class = the exported instantiation, so the PTX targets apply; `INDEX.txt` g/n/k):
  - contig/psum 128x32 `g2 n1536 k2048`; 128x48 `g2 n3072 k4096`; 128x64 `g2 n4096 k1536`;
    128x112 on A `g4 n7168 k2048`; 128x128 on A `g8 n4096 k7168`; contig 128x160 on B `g8 n7168 k2048`
    (already a partial last N block: 7168 = 44 x 160 + 128);
  - masked 64x32 / 64x48 / 64x64 with the same n/k; masked 64x112 on B `g8 n7168 k2048`; masked 128x112 / 128x128 on A
    as contig.
- Edges:
  - contig 128x112 on A with an odd N-block count, `g4 n6352 k2048` (57 N blocks; plain A/SFA branch live in the
    tail column, `is_peer_cta_alive` false there), exported;
  - runtime predicate with a partial last N block: BN 48 / 112 with `N % BN != 0` (`num_full_iters < BLOCK_N/8`);
  - a block straddling a 128 boundary (`num_sfb = 2 * shape_k_scales`), e.g. BN 112 at n_block 1;
  - multicast shapes with more than one wave (one-wave shapes never pick a cluster).

## Appendix: per-class PTX totals from the export (fidelity targets)

Counts are static PTX instruction lines per instantiation, over the whole kernel. The `tma_copy` atom loop runs
once for A, B and SFA; the TMA-load and `expect_tx` multiplicity is nvcc's unroll of the producer K loop
(`.cuh:183`): 4 for most classes, 12 for the BN 64 classes (K = 1536, 12 K blocks fully unrolled), 16 for psum 128x32
and masked 64x32 (K = 2048, fully unrolled), but 4 for contig 128x32 at the same K.

| class (kind BMxBN_stages, multicast) | wgmma | mul.f32 | fma.rn.f32 | selp.f32 | stmatrix.x2 | TMA load multicast / plain | expect_tx (producer unroll) | TMA store sites | .ftz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| psum 128x112_s6, on A | 32 m64n112k32 | 208 | 448 | 104 | 14 | 8 / 4 | 4 | 1 | 0 |
| psum 128x128_s5, on A | 32 m64n128k32 | 16 | 512 | 0 | 16 | 8 / 4 | 4 | 1 | 0 |
| psum 128x32_s10, none | 32 m64n32k32 | 16 | 128 | 0 | 4 | 0 / 48 | 16 | 1 | 0 |
| psum 128x48_s9, none | 32 m64n48k32 | 80 | 192 | 40 | 6 | 0 / 12 | 4 | 1 | 0 |
| psum 128x64_s8, none | 32 m64n64k32 | 16 | 256 | 0 | 8 | 0 / 36 | 12 | 1 | 0 |
| contig 128x112_s6, on A | 32 m64n112k32 | 208 | 448 | 104 | 14 | 8 / 4 | 4 | 1 | 0 |
| contig 128x128_s5, on A | 32 m64n128k32 | 16 | 512 | 0 | 16 | 8 / 4 | 4 | 1 | 0 |
| contig 128x160_s5, on B | 160 m64n160k32 | 144 | 3200 | 0 | 20 | 4 / 12 | 4 | 1 | 0 |
| contig 128x32_s10, none | 32 m64n32k32 | 16 | 128 | 0 | 4 | 0 / 12 | 4 | 1 | 0 |
| contig 128x48_s9, none | 32 m64n48k32 | 80 | 192 | 40 | 6 | 0 / 12 | 4 | 1 | 0 |
| contig 128x64_s8, none | 32 m64n64k32 | 16 | 256 | 0 | 8 | 0 / 36 | 12 | 1 | 0 |
| masked 128x112_s6, on A | 32 m64n112k32 | 208 | 448 | 104 | 14 | 8 / 4 | 4 | 1 | 0 |
| masked 128x128_s5, on A | 32 m64n128k32 | 16 | 512 | 0 | 16 | 8 / 4 | 4 | 1 | 0 |
| masked 64x112_s9, on B | 32 m64n112k32 | 208 | 448 | 104 | 14 | 4 / 12 | 4 | 1 | 0 |
| masked 64x32_s16, none | 32 m64n32k32 | 16 | 128 | 0 | 4 | 0 / 48 | 16 | 1 | 0 |
| masked 64x48_s15, none | 32 m64n48k32 | 80 | 192 | 40 | 6 | 0 / 12 | 4 | 1 | 0 |
| masked 64x64_s13, none | 32 m64n64k32 | 16 | 256 | 0 | 8 | 0 / 36 | 12 | 1 | 0 |

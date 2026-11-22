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

# SM90 FP8 1D2D GEMM (m-grouped), cluster-1 BLOCK_N 96/128/192: delta sketch

**Status: DRAFT, awaiting sketch review.** This is a non-executable delta against the approved base sketch
[`sm90_fp8_gemm_1d2d.md`](sm90_fp8_gemm_1d2d.md) and coverage delta
[`sm90_fp8_gemm_1d2d_coverage.md`](sm90_fp8_gemm_1d2d_coverage.md). Both stay authoritative for everything not
restated here. The instruction selections come from the line-info export of all 10 classes below
(`.porting/sm90_fp8_gemm_1d2d/source_export_ext/`, `export_ext.py --cluster1-only`). Each export was checked against a
bf16 reference with calc_diff < 1e-3.

## Why

DeepGEMM's heuristic picks BLOCK_N 96, 128 or 192 **without a cluster** for many realistic MoE shapes. A one-wave
grid never gets a cluster (`sm90.hpp:234-235`), and psum and masked shapes with an odd N-block count cannot have
one (`sm90.hpp:87-91`); otherwise the cycle model can simply prefer cluster 1. On the review's 378-shape grid (DeepSeek-V3, Qwen3-235B, Qwen3-30B; 4/8/32 groups; 8-512 rows per
group), these are 95 shapes (25%). Before this delta, `make_spec`'s class gate refused them.

| new class (kind, BM x BN, stages) | B scales | exported instantiation (g, N, K) |
| --- | --- | --- |
| contiguous, psum 128x96 s7 | non-uniform, dispatched (`kGap` 4, `kEnd` 16) | g4 N3072 K4096 |
| masked 128x96 s7; masked 64x96 s10 | same | g4 N3072 K4096 |
| contiguous, psum, masked 128x128 s5; masked 64x128 s8 | uniform | g4 N4096 K7168 |
| contiguous, masked 128x192 s4 | non-uniform, dispatched (`kGap` 8, `kEnd` 16; BLOCK_N > BLOCK_K) | g4 N3072 K4096 |

## Delta: none in the algorithm

Every operation already exists in the base sketch, which is unchanged:
- **The `kNumTMAMulticast == 1` path** (base sketch, producer `mc == 1` branch; `.cuh:178-180` with
  `is_tma_multicast_valid` never consulted): every load is the plain per-CTA `tma::copy`. The empty-barrier arrive
  is the local `lane_idx == 0` form, and there is no drain. The base classes psum 128x192 mc1 and masked 64x192 mc1
  already run this path.
- **BLOCK_N 96/128/192 promotion** (uniform at 128; the `dispatch_num_former_iters` copies at 96 and 192) and
  their D swizzles are the base sketch's, at BLOCK_M 64 and 128.

## instruction_selection (export, per class; static PTX lines per instantiation)

- **TMA loads:** only the plain `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`
  (`tma_copy.cuh:31`). There is no `.multicast::cluster` form and no `mapa`, so the arrive is local.
- **Compute:** the base classes' forms, `wgmma.mma_async.sync.aligned.m64n{96,128,192}k32.f32.e4m3.e4m3`, non-FTZ
  `mul.f32` / `fma.rn.f32`, no `selp.f32`; stmatrix x2 and one TMA-store site. The counts are below.
- **Producer K-loop unroll:** nvcc chooses it, as in the coverage delta (the `expect_tx` column). The port leaves the
  loop as the source writes it.

| class (kind BMxBN_stages, multicast) | g/n/k | wgmma | mul.f32 | fma.rn.f32 | selp.f32 | stmatrix.x2 | TMA load multicast / plain | expect_tx | TMA store sites | .ftz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| psum 128x128_s5, none | g4_n4096_k7168 | 32 m64n128k32 | 16 | 512 | 0 | 16 | 0 / 24 | 8 | 1 | 0 |
| psum 128x96_s7, none | g4_n3072_k4096 | 128 m64n96k32 | 96 | 1536 | 0 | 12 | 0 / 12 | 4 | 1 | 0 |
| contig 128x128_s5, none | g4_n4096_k7168 | 32 m64n128k32 | 16 | 512 | 0 | 16 | 0 / 12 | 4 | 1 | 0 |
| contig 128x192_s4, none | g4_n3072_k4096 | 96 m64n192k32 | 80 | 2304 | 0 | 24 | 0 / 12 | 4 | 1 | 0 |
| contig 128x96_s7, none | g4_n3072_k4096 | 128 m64n96k32 | 96 | 1536 | 0 | 12 | 0 / 12 | 4 | 1 | 0 |
| masked 128x128_s5, none | g4_n4096_k7168 | 32 m64n128k32 | 16 | 512 | 0 | 16 | 0 / 12 | 4 | 1 | 0 |
| masked 128x192_s4, none | g4_n3072_k4096 | 96 m64n192k32 | 80 | 2304 | 0 | 24 | 0 / 12 | 4 | 1 | 0 |
| masked 128x96_s7, none | g4_n3072_k4096 | 128 m64n96k32 | 96 | 1536 | 0 | 12 | 0 / 12 | 4 | 1 | 0 |
| masked 64x128_s8, none | g4_n4096_k7168 | 32 m64n128k32 | 16 | 512 | 0 | 16 | 0 / 12 | 4 | 1 | 0 |
| masked 64x96_s10, none | g4_n3072_k4096 | 128 m64n96k32 | 96 | 1536 | 0 | 12 | 0 / 12 | 4 | 1 | 0 |

## Port changes

- **`spec.PORTED_CLASSES`:** add the 10 classes. There is no kernel change.
- **CONFIGS:** one `cov_` shape per class, each the exact exported instantiation above, every group with the same row
  count. Edges come from the existing mc1 base edges: odd M-block counts and empty groups.

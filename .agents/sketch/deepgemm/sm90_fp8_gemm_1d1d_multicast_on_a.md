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
deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh together with the
scheduler and TMA helpers it instantiates. See NOTICE and licenses/ for
upstream attribution.
-->

# SM90 FP8 1D1D GEMM, TMA multicast on A: delta sketch

A non-executable delta against the base sketch
[`sm90_fp8_gemm_1d1d.md`](sm90_fp8_gemm_1d1d.md); everything not stated here is unchanged. The
implementation is [`tirx_kernels/ported/deepgemm/_sm90_fp8_gemm_1d1d/kernel.py`](../../../tirx_kernels/ported/deepgemm/_sm90_fp8_gemm_1d1d/kernel.py),
which is the source of truth.

**Why.** No upstream SM90 1D1D test shape selects multicast on A (`cluster_n = 2`, i.e.
`kIsTMAMulticastOnA = true`), but DeepGEMM's heuristic does for realistic MoE `down`-projection
wgrad shapes with a small N (`sm90-probe/dgkg90/reach.py`, run through the port's own gate):
Qwen3-30B-A3B at EP 64 (2 experts per rank, `M = 2048, N = 768`, every token count) and
Qwen3-235B-A22B at EP 32 with 512 tokens per expert (`M = 4096, N = 1536`). 7 of 248 realistic
shapes; without this delta the port refuses them.

**Source export.** Line-info PTX/SASS of the two instantiations those shapes select, built through
DeepGEMM's JIT (`DG_JIT_WITH_LINEINFO=1`, `sm90-probe/dgkg90/export.sh kg 7 8`), each call checked
with upstream's `calc_diff < 1e-3` (5.2e-9, 2.7e-9):

| class | instantiation (SHAPE_M, SHAPE_N, groups) | N blocks | `kNum1DBlocksPerGroup` |
| --- | --- | --- | --- |
| `KGroupedContiguous` 128x104 s5 mc2, on A | (2048, 768, 2) | 8 (even) | 16 |
| `KGroupedContiguous` 128x120 s5 mc2, on A | (4096, 1536, 4) | 13 (odd) | 16 |

Both are `source_export/KGroupedContiguous_128x1{04,20}_s5_mc2_onA=true/` (INDEX.txt).
`kNum1DBlocksPerGroup`: with multicast on A the usage is `c * BLOCK_N + ceil(132 / c) * BLOCK_M`
(`gemm.cuh:20`): 104 -> {8: 3008, 16: 2816}, 120 -> {8: 3136, 16: 3072}, so 16 for both.

**In scope.** `kIsTMAMulticastOnA = true` with `kNumTMAMulticast = 2` for `KGroupedContiguous`
(compiled "mn", so SHAPE_N is a constant). **Out of scope.** Multicast on A for `Normal` (no
realistic shape on the grid selects it: N is compiled there too, but none of the 48 dense shapes
picks `cluster_n = 2`); the kernel builds it through the same code, it is simply not exported.

## Changes against the base sketch

### Scheduler (`get_swizzled_block_idx`, on-A branch)

```python
# primary = num_n_blocks, secondary = num_m_blocks: groups run along N (`gemm.cuh:121-122`).
# The general swizzle, unchanged from the base sketch, runs at run time (`gemm.cuh:123-141`):
group_idx = block_idx // (secondary * 16); first = group_idx * 16; in_group = block_idx % (secondary * 16)
num_blocks_in_group = min(16, num_n_blocks - first)
# instruction_selection: SHAPE_M is compiled, so `secondary * 16` is a constant power of two (16 * 16 = 256
#   for 128x104, 32 * 16 = 512 for 128x120): group_idx / in_group are shr.u32 / and.b32, `first` an
#   and.b32 of the shifted index, then sub.s32 + min.u32 with 16; extent: scalar per block
if num_blocks_in_group % 2 != 0:            # "Fix unaligned TMA multicast" (`gemm.cuh:133-141`)
    if in_group < (num_blocks_in_group ^ 1) * secondary: num_blocks_in_group ^= 1
    else: in_group -= (num_blocks_in_group ^ 1) * secondary; first += num_blocks_in_group ^ 1; num_blocks_in_group = 1
# instruction_selection: 128x120: a runtime branch (and.b32 parity test, xor/shl, setp.lt.u32 + bra);
#   128x104: removed by nvcc (see below); extent: scalar per block
m = in_group // num_blocks_in_group; n = first + in_group % num_blocks_in_group   # on A (`gemm.cuh:146-148`)
# instruction_selection: div.u32 by the runtime num_blocks_in_group, remainder as mul.lo + sub;
#   extent: scalar per block
```

**Value-level facts (not instruction selection).** With SHAPE_N compiled and
`num_n_blocks <= kNum1DBlocksPerGroup` there is one group: `block_idx < num_blocks =
num_n_blocks * num_m_blocks <= 16 * num_m_blocks`, so `group_idx = 0`, `first = 0`,
`num_blocks_in_group = num_n_blocks`.
- **128x104 (8 N blocks):** `8 - first`, `first` a multiple of 16, is even for every `first` nvcc
  must allow, so it removes the fix-up and `is_tma_multicast_valid` (`num_blocks_in_group == 1` is
  impossible): no plain A/SFA load is emitted. The swizzle arithmetic itself stays at run time.
- **128x120 (13 N blocks):** `num_blocks_in_group` is 13, then 12 or 1 after the fix-up: the last
  N block (`n = 12`) runs with `num_tma_multicast_a = 1` and loads A/SFA plainly; both paths are
  emitted.

The TIRx port writes these value-level facts out at trace time (`first = 0`, the constant group size
or the constant-split fix-up), because NVRTC does not derive the parity fact; the selected scheduler
instructions there differ (fewer, constant operands), the producer's TMA paths are the source's.

### Role 1: TMA producer

```python
mc_a = kNumTMAMulticast if (kIsTMAMulticastOnA and valid) else 1     # the multicast side is now A/SFA
mc_b = 1                                                               # B/SFB: always plain
for (tmap, dst, inner, outer, mc) in ((tmap_sfa, sfa[s], m_idx, sf_k, mc_a),
                                     (tmap_sfb, sfb[s], n_idx, sf_k, 1),
                                     (tmap_a,   a[s],   k_idx, m_idx, mc_a),
                                     (tmap_b,   b[s],   k_idx, n_idx, 1)):
    if mc == 1:
        copy_g2s(tmap.coord(inner, outer), dst, completion=full[s], cache_hint=EVICT_NORMAL)
        # instruction_selection: cp.async.bulk.tensor.2d.shared::cluster.global
        #   .mbarrier::complete_tx::bytes.L2::cache_hint; extent: one box
    elif cluster_rank() == 0:
        copy_g2s(tmap.coord(inner, outer), dst, completion=full[s], multicast_mask=0b11,
                 cache_hint=EVICT_NORMAL)
        # instruction_selection: mov %cluster_ctarank (TIRx: %cluster_ctaid.x, equal for a (2,1,1)
        #   cluster); cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
        #   .multicast::cluster.L2::cache_hint (mask u16 3); extent: one box into both CTAs
expect_bytes(full[s], A_STAGE + B_STAGE + SFA_STAGE + BLOCK_N * 4)    # unchanged: each CTA expects all
# instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64
```

The rank-1 CTA issues only its own B/SFB loads; rank 0's multicast A/SFA boxes complete on both CTAs'
`full[s]` (the same expect-tx count on both). The drain loop and the cluster barrier pair are the
base sketch's multicast path.

### Role 2: math warpgroups

Unchanged: `empty_barrier_arrive` is the base sketch's multicast form (`mapa.shared::cluster.u32` +
`mbarrier.arrive.shared::cluster.b64` from lanes 0..1 of each warp; `is_peer_cta_alive` is never
written for k-grouped, so the target is `lane_idx`).

## Instruction-selection summary (static PTX counts, `source_export/opcode_hist.txt`)

| instruction | KG 128x104 s5 mc2 on A | KG 128x120 s5 mc2 on A |
| --- | --- | --- |
| `wgmma.mma_async...m64n{104,120}k32.f32.e4m3.e4m3` | 4 | 4 |
| `mul.f32` / `fma.rn.f32` (each) | 52 | 60 |
| `ld.shared.v2.f32` (B scales) / `ld.shared.f32` (A scales) | 13 / 2 | 15 / 2 |
| `st.shared.v2.f32` (D) | 26 | 30 |
| TMA load `...L2::cache_hint` (plain) | 14 | 12 |
| TMA load `...multicast::cluster.L2::cache_hint` | 14 | 6 |
| `mbarrier.arrive.expect_tx` | 7 | 3 |
| `mbarrier.try_wait.parity` | 13 | 9 |
| `mapa` + `mbarrier.arrive.shared::cluster.b64` | 1 + 1 | 1 + 1 |
| `ld.param.v4` / `st.shared.v4` / `ld.shared.v4` / `st.global.v4` | 16 each | 16 each |
| `tensormap.replace` address / dim / stride | 2 / 2 / 2 | 2 / 2 / 2 |
| `barrier.cluster.arrive` + `wait`, `bar.sync` | 1, 2 | 1, 2 |

- **128x104:** 7 producer copies (x4 unroll + 3 remainder, as in the base k-grouped classes); each
  copy issues 2 multicast (SFA, A) and 2 plain (SFB, B) loads: the plain A/SFA path is removed by the
  parity fact above (`num_blocks_in_group = min(16, 8 - first)` stays a runtime register, but is
  always even, so `== 1` is impossible).
- **128x120:** nvcc keeps both A/SFA paths (the runtime split above) and emits only 3 producer copies,
  an x2 unroll plus a one-copy remainder (12 plain = 3 x (2 plain A/SFA + 2 B/SFB), 6 multicast = 3 x 2).
  The unroll factor is an nvcc choice, recorded as an ISA-comparison fact; the TIRx port pins the same
  x2 for this case (NVRTC's x4 spills the 40-register producer).
- Promotion, WGMMA, epilogue and tensor-map rewrite counts are the base k-grouped ones at the new
  BLOCK_N.

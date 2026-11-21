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

# SM90 FP8 1D2D GEMM (m-grouped): coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new IR, a
builder API, or a mathematical reference. It shows the TIRx kernel as an explicit
runtime ABI and launch; explicit linear GMEM/SMEM/register storage with byte offsets;
the persistent scheduler and the role control flow; primitive directional copies and
primitive computation; explicit stage/phase changes and publication/reuse edges; and
hardware instruction selection, read from the source's line-info PTX export.

The implementation represented by this sketch is maintained in
[`tirx_kernels/ported/deepgemm/_sm90_fp8_gemm_1d2d/`](../../../tirx_kernels/ported/deepgemm/_sm90_fp8_gemm_1d2d/__init__.py),
with two entry modules (`sm90_m_grouped_fp8_gemm_contiguous.py`,
`sm90_m_grouped_fp8_gemm_masked.py`) that pin one descriptor each. Those modules are the
source of truth.

Source export: line-info PTX and SASS of all 42 instantiations the 48 upstream SM90
m-grouped test calls reach (`test_m_grouped_gemm_contiguous` and
`test_m_grouped_gemm_masked`, including the masked test's psum half), built through
DeepGEMM's own JIT with `DG_JIT_WITH_LINEINFO=1` by `sm90-probe/dg90/export_full.py`, every
call checked with upstream's `calc_diff < 1e-3`. They fall into 8 classes:

| class (`gemm_type`, BMxBN, stages, multicast) | instantiations | B scales |
| --- | --- | --- |
| `MGroupedContiguous` 128x192 s4 mc2 (on B) | 8 | non-uniform |
| `MGroupedContiguousWithPsumLayout` 128x192 s4 mc2 (on B) | 16 | non-uniform |
| `MGroupedContiguousWithPsumLayout` 128x192 s4 mc1 | 2 | non-uniform |
| `MGroupedMasked` 128x192 s4 mc2 (on B) | 8 | non-uniform |
| `MGroupedMasked` 64x192 s6 mc2 (on B) | 3 | non-uniform |
| `MGroupedMasked` 64x192 s6 mc1 | 2 | non-uniform |
| `MGroupedMasked` 64x128 s8 mc2 (on B) | 2 | uniform |
| `MGroupedMasked` 64x96 s10 mc2 (on B) | 1 | non-uniform, BN < BK |

`.porting/sm90_fp8_gemm_1d2d/source_export/INDEX.txt` maps every instantiation to its
directory; one representative per class is linked by class name. No upstream SM90 shape
selects multicast on A (`cluster_n = 2`), so that branch is out of scope (below). Counts below
are from the `MGroupedContiguous_128x192_s4_mc2` representative unless stated, as
instruction lines minus predicated duplicates.

**In scope.** `kGemmType` = `MGroupedContiguous`, `MGroupedContiguousWithPsumLayout`,
`MGroupedMasked`; FP8 e4m3 A and B, both K-major; BF16 D without accumulation; FP32
per-token SFA (MN-major, TMA-aligned) and per-128x128-block SFB (K-major,
`kMajorSFB == K`); `BLOCK_K = 128`; `BLOCK_M` in {64, 128}; `BLOCK_N` in {96, 128, 192}
(non-uniform B scales with the `dispatch_num_former_iters` fold, and uniform scales);
`kNumTMAMulticast` 1, or 2 with multicast on B (`kIsTMAMulticastOnA == false`);
`compiled_dims = "nk"`; `EpilogueIdentity`. This is exactly the set the export covers.

**Out of scope.** Multicast on A (`kIsTMAMulticastOnA`), and `BLOCK_N` values whose
non-uniform B scales are not dispatched (48, 80, 112, 144: a runtime promotion predicate)
or that no upstream shape selects -- the TIRx `make_spec` rejects such a heuristic choice
instead of building an unverified variant; the on-A branches are kept below only where the
source shares code, marked `[on-A: out of scope]`. `Normal`, `Batched` and k-grouped GEMM types (3-D TMA, batch index);
`kMajorSFB == MN`; `BLOCK_M < 64` (single-warp-group store threads); `kSwizzleDMode == 0`
(padded D) -- no in-scope configuration selects it; FP32 output; tile (`Tx`) primitives.

## Pipeline at a glance

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0 .. M/32-1 (math, M = kNumMathThreads = 128 or 256) | per scheduler block: stage SFB rows to SMEM, then per K block one WGMMA chain per warpgroup into `accum`, FP32 promotion into `final_accum`; then STSM to swizzled D and TMA store | wait `full[stage]`; arrive `empty[stage]` (to every CTA of the cluster); named barriers 8 (SFB staged) and 9 (D tile) |
| M/32 (TMA WG, warp 0) | prefetch the four TensorMaps | none |
| M/32+1 (TMA WG, warp 1) | initialize `full`/`empty` barriers | publishes barrier init via `fence.mbarrier_init` + CTA/cluster sync |
| M/32+2 (TMA WG, warp 2, one elected lane) | persistent loop: per block, per K block, TMA A + SFA + B into one stage | wait `empty[stage]`; publish `full[stage]` with expect-tx A+B+SFA bytes |
| M/32+3 (TMA WG, warp 3) | idle after `setmaxnreg.dec` | none |

Math warpgroup `w` owns rows `[w*64, w*64+64)` of the block (one `m64nBLOCK_Nk32`
WGMMA row band); with `BLOCK_M = 64` there is one math warpgroup. Both the producer
lane and every math thread construct their own scheduler and walk the identical block
sequence, so `stage_idx`/`phase` stay in lockstep without communication.

## Primitive vocabulary

```python
specialize(...)     # compile-time variant selection
launch(...)         # launch topology and attributes
tile(...)           # linear storage: dtype, element count, placement, alignment
reg_tile(...)       # role-local register array (no layout)
at(buf, byte_off)   # address of a byte offset inside a linear buffer
desc(...)           # WGMMA SMEM matrix descriptor (64-bit, fields written explicitly)

copy_g2s(tmap.coord(...), dst, completion, multicast_mask=None)   # TMA load
copy_s2g(src, tmap.coord(...))                                    # TMA store
copy_g2r(src, dst)     # plain global load of one element into a register
copy_r2s(src, dst)     # shared store (st.shared or stmatrix)
copy_s2r(src, dst)     # shared load

gemm(acc, a_desc, b_desc, scale_d, instr)   # one wgmma.mma_async
fill(dst, value)                             # register initialization
mul(dst, a, b); fma(acc, a, b)               # FP32 promotion
select(dst, pred, a, b); cast(dst, src, rounding)
elect_predicate(mask); shuffle_index(dst, src, lane)
```

Schedule operations: `prefetch`, `init`, `fence`, `wait`, `arrive`, `expect_bytes`,
`remote_arrive`, `cta_sync`, `cluster_sync`, `named_barrier`, `pdl_wait`,
`setmaxnreg`, `wgmma_fence`, `wgmma_arrive`, `wgmma_commit`, `wgmma_wait`,
`store_fence`, `store_commit`, `store_wait`. There are no compound ops: no
`scheduler`, `epilogue`, `promote`, or `gemm_1d2d`.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
specialize(KernelSpec(
    gemm_type, SHAPE_N, SHAPE_K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K=128,
    kSwizzleAMode=128, kSwizzleBMode=128, kSwizzleDMode,          # D: 128 (BN%64==0) / 64 (BN=96)
    kNumStages, kNumTMAThreads=128, kNumMathThreads,               # 128 if BLOCK_M<=64 else 256
    kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs=132))
# instruction_selection: none; extent: the template arguments of `generate_impl`
# (sm90_fp8_gemm_1d2d.hpp:35), produced by spec.make_spec from the ported heuristic

kMustUseUniformedScaleB = BLOCK_K % BLOCK_N == 0                # BN=128: True; 96/192: False
WGMMA = m64n{BLOCK_N}k32.f32.e4m3.e4m3 (SS); kNumAccum = BLOCK_N / 2
WAVE_BLOCK_M = BLOCK_M if BLOCK_M <= 64 else 128                 # one wave per K block
kNumWGMMAStoreThreads = WAVE_BLOCK_M * 2                         # 128 or 256
TMA_D_BLOCK_N = kSwizzleDMode / 2                                # bf16 columns per store atom
num_total_k_blocks = shape_k_scales = SHAPE_K / 128              # compile-time ("nk")
shape_n_sfb = ceil(SHAPE_N / 128)

def kernel(sfb: f32*, grouped_layout: i32*, shape_m: u32, shape_n: u32, shape_k: u32,
           tmap_a, tmap_b, tmap_d, tmap_sfa: __grid_constant__ CUtensorMap):
    launch(grid=(kNumSMs,1,1), block=(128 + kNumMathThreads,1,1),
           cluster=(kNumTMAMulticast,1,1) if kNumTMAMulticast > 1 else None,
           dynamic_smem=smem_size, launch_bounds=(128 + kNumMathThreads, 1))
    # instruction_selection: .maxntid N,1,1 + .minnctapersm 1 (PTX header); cluster and
    #   dynamic smem are launch attributes; no PDL attribute (DeepGEMM default)
    shape_n, shape_k = SHAPE_N, SHAPE_K       # compiled dims override the runtime args

    # ---- storage: one dynamic, 1024-aligned linear pool --------------------
    smem = tile(u8, smem_size, space=shared.dyn, align=1024)
    SMEM_D_SIZE  = align(BLOCK_M * BLOCK_N * 2, 1024)
    A_STAGE      = BLOCK_M * 128           # fp8 bytes per stage
    B_STAGE      = BLOCK_N * 128
    SFA_STAGE    = align(BLOCK_M * 4, 128)
    SF_OFF       = SMEM_D_SIZE + kNumStages * (A_STAGE + B_STAGE)
    SFB_BYTES    = align(shape_k_scales * (1 if kMustUseUniformedScaleB else 2) * 4, 8)
    off_d          = 0                                      # bf16 D tile, swizzled atoms
    off_a(s)       = SMEM_D_SIZE + s * A_STAGE              # 128B-swizzled [BLOCK_M rows x 128B]
    off_b(s)       = SMEM_D_SIZE + kNumStages * A_STAGE + s * B_STAGE
    off_sfa(s)     = SF_OFF + s * SFA_STAGE                 # f32[BLOCK_M], linear
    off_sfb        = SF_OFF + kNumStages * SFA_STAGE        # f32[shape_k_scales] (x2 rows)
    off_full(s)    = off_sfb + SFB_BYTES + 8 * s            # mbarrier (u64)
    off_empty(s)   = off_full(kNumStages) + 8 * s
    # Math WG w's A descriptor starts at off_a(s) + w*64*128, inside stage s for every
    # in-scope BLOCK_M (the `.cuh:87-89` padding assert only matters for BLOCK_M < 64).

    warp_idx = shuffle_index(thread_idx // 32, src_lane=0)
    # instruction_selection: shfl.sync.idx.b32 (warp-uniform broadcast); extent: scalar
    lane_idx = lane_id()
    # instruction_selection: mov %laneid; extent: scalar

    # ---- prologue -----------------------------------------------------------
    if warp_idx == kNumMathThreads // 32 and elect_predicate(0xffffffff):
        # instruction_selection: elect.sync; extent: one lane
        for t in (tmap_a, tmap_b, tmap_sfa, tmap_d):
            prefetch(t)
            # instruction_selection: prefetch.tensormap [param]; extent: 4 instructions
    warp_sync()
    # instruction_selection: bar.warp.sync -1; extent: one per warp

    if warp_idx == kNumMathThreads // 32 + 1 and elect_predicate(0xffffffff):
        for s in static_range(kNumStages):
            init(full[s] @ at(smem, off_full(s)), arrive_count=1)
            init(empty[s] @ at(smem, off_empty(s)),
                 arrive_count=kNumTMAMulticast * kNumMathThreads // 32)
            # instruction_selection: mbarrier.init.shared::cta.b64; extent: 2*kNumStages
        fence(mbarrier_init, scope=cluster)
        # instruction_selection: fence.mbarrier_init.release.cluster; extent: one
    if kNumTMAMulticast > 1:
        cluster_sync(arrive="relaxed")
        # instruction_selection: barrier.cluster.arrive.relaxed.aligned +
        #   barrier.cluster.wait.aligned; extent: whole cluster
    else:
        cta_sync()
        # instruction_selection: bar.sync 0; extent: whole CTA
    pdl_wait()
    # instruction_selection: griddepcontrol.wait; extent: one (no-op without the PDL attribute)

    sched = Scheduler(gemm_type, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast,
                      kIsTMAMulticastOnA, kNumSMs, shape_m, shape_n, grouped_layout)
    # see "Persistent scheduler" below; psum reads grouped_layout[0] here (ld.global.u32)
    stage_idx, phase = 0, 0
    def advance_pipeline(k):            # k += 1; stage wraps; phase flips on wrap
        stage_idx = 0 if stage_idx == kNumStages - 1 else stage_idx + 1
        phase ^= (stage_idx == 0)
```

### Persistent scheduler (`scheduler/gemm.cuh`)

```python
# Initial state (constructor + member initializers): current_iter = -1,
# current_group_idx = 0, current_m_cumsum = 0, last_psum_m = 0, current_m_block_cumsum = 0,
# is_peer_cta_alive = True (stays True for masked and psum, read by empty_barrier_arrive),
# num_blocks_in_group set by every swizzle().
# kNum1DBlocksPerGroup = argmin over {8, 16} (first candidate wins ties: strict `<`) of
#   candidate*BLOCK_N + ceil(kNumSMs/candidate)*BLOCK_M   (multicast on A: group on N)  [on-A: out of scope]
#   candidate*BLOCK_M + ceil(kNumSMs/candidate)*BLOCK_N   (otherwise: group on M)
num_m_blocks = ceil(shape_m / BLOCK_M); num_n_blocks = ceil(shape_n / BLOCK_N)
# MGroupedContiguous: num_blocks = num_m_blocks * num_n_blocks
# psum: current_psum_m = grouped_layout[0]; num_m_blocks = ceil(current_psum_m / BLOCK_M)
# masked: num_m_blocks recomputed per group from grouped_layout[g] (= masked_m[g])

def get_next_block():                    # returns (m_block_idx, n_block_idx) or False
    next_block_idx = (++current_iter) * kNumSMs + block_idx_x
    if MGroupedMasked:
        while True:                       # skip whole groups
            if current_group_idx == kNumGroups: return False
            num_m_blocks = ceil(grouped_layout[current_group_idx] / BLOCK_M)
            # instruction_selection: ld.global.u32 (scheduler/gemm.cuh:207); extent: scalar
            if next_block_idx < (current_m_cumsum + num_m_blocks) * num_n_blocks: break
            current_group_idx += 1; current_m_cumsum += num_m_blocks
        swizzle(next_block_idx - current_m_cumsum * num_n_blocks)
    elif MGroupedContiguousWithPsumLayout:
        while next_block_idx >= (current_m_block_cumsum + num_m_blocks) * num_n_blocks:
            if ++current_group_idx == kNumGroups: return False
            last_psum_m = align(current_psum_m, BLOCK_M)
            current_psum_m = grouped_layout[current_group_idx]
            # instruction_selection: ld.global.u32; extent: scalar
            current_m_block_cumsum += num_m_blocks
            num_m_blocks = ceil((current_psum_m - last_psum_m) / BLOCK_M)
        swizzle(next_block_idx - current_m_block_cumsum * num_n_blocks)
        m_block_idx += last_psum_m / BLOCK_M
    else:  # MGroupedContiguous
        if next_block_idx >= num_blocks: return False
        is_peer_cta_alive = (num_n_blocks % kNumTMAMulticast == 0 or
                             num_m_blocks % kNumTMAMulticast == 0 or
                             (next_block_idx ^ 1) < num_blocks)
        swizzle(next_block_idx)

def swizzle(block_idx):                  # get_swizzled_block_idx, SM90 branch
    primary   = num_n_blocks if kIsTMAMulticastOnA else num_m_blocks
    secondary = num_m_blocks if kIsTMAMulticastOnA else num_n_blocks
    per_group = secondary * kNum1DBlocksPerGroup
    first = (block_idx // per_group) * kNum1DBlocksPerGroup; in_group = block_idx % per_group
    num_blocks_in_group = min(kNum1DBlocksPerGroup, primary - first)
    if kNumTMAMulticast > 1 and num_blocks_in_group % 2 != 0:   # unaligned multicast tail
        if in_group < (num_blocks_in_group ^ 1) * secondary: num_blocks_in_group ^= 1
        else: in_group -= (num_blocks_in_group ^ 1) * secondary
              first += num_blocks_in_group ^ 1; num_blocks_in_group = 1
    if kIsTMAMulticastOnA: m = in_group // num_blocks_in_group; n = first + in_group % num_blocks_in_group  # [on-A: out of scope]
    else:                  m = first + in_group % num_blocks_in_group; n = in_group // num_blocks_in_group

def group_offset(m_block_idx):            # get_global_idx<kWithGroupOffset=true>
    # contiguous: max(0, grouped_layout[m_block_idx*BLOCK_M])
    # instruction_selection: ld.global.u32 + max.s32 (cute/numeric/math.hpp:51); extent: scalar
    # masked / psum: current_group_idx
def is_tma_multicast_valid(m_block_idx):
    if num_blocks_in_group == 1: return False
    if MGroupedContiguous and not kIsTMAMulticastOnA:
        return grouped_layout[m_block_idx*BLOCK_M] == grouped_layout[(m_block_idx^1)*BLOCK_M]
        # instruction_selection: 2 x ld.global.u32 (scheduler/gemm.cuh:302-303)
    return True
def is_computation_valid(m_block_idx, m_offset):
    # contiguous: grouped_layout[m_offset + m_block_idx*BLOCK_M] >= 0  (ld.global.u32 :315)
    # masked:     m_offset + m_block_idx*BLOCK_M < grouped_layout[current_group_idx]
    #             instruction_selection: ld.global.u32 (scheduler/gemm.cuh:317) + setp.lt
    # contiguous: instruction_selection: ld.global.u32 (:315) + setp.lt.s32 p, v, 0;
    #             @p bra (to the invalid-band loop)
    # psum:       m_offset + m_block_idx*BLOCK_M < current_psum_m
```

### Role 1: TMA producer (warp M/32+2, one elected lane)

```python
    if warp_idx >= kNumMathThreads // 32:
        setmaxnreg(dec, 40)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 40; extent: TMA warpgroup
        if warp_idx == kNumMathThreads // 32 + 2 and elect_predicate(0xffffffff):
            # instruction_selection: elect.sync; extent: one lane
            while get_next_block():
                valid = is_tma_multicast_valid(m_block_idx)
                mc_a = kNumTMAMulticast if (kIsTMAMulticastOnA and valid) else 1   # always 1 in scope [on-A: out of scope]
                mc_b = kNumTMAMulticast if (not kIsTMAMulticastOnA and valid) else 1
                for k_block_idx in runtime_range(num_total_k_blocks, advance=advance_pipeline):
                    wait(empty[stage_idx], parity=phase ^ 1)
                    # instruction_selection: LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64
                    #   P1, [bar], phase, 0x989680 (suspend-time hint); @P1 bra DONE;
                    #   bra LAB_WAIT (cutlass/arch/barrier.h:412-421); extent: one stage
                    m_idx = (group_offset(...) if MGroupedMasked else 0) * shape_m \
                            + m_block_idx * BLOCK_M         # masked folds groups into rows
                    for (tmap, dst, inner, outer, box, mc) in (
                            (tmap_a,   at(smem, off_a(stage_idx)),   k_block_idx*128, m_idx,                BLOCK_M, mc_a),
                            (tmap_sfa, at(smem, off_sfa(stage_idx)), m_block_idx*BLOCK_M,
                                       (current_group_idx if MGroupedMasked else 0) * shape_k_scales + k_block_idx, 1, mc_a),
                            (tmap_b,   at(smem, off_b(stage_idx)),   k_block_idx*128,
                                       group_offset(m_block_idx) * shape_n + n_block_idx * BLOCK_N, BLOCK_N, mc_b)):
                        if mc == 1:
                            copy_g2s(tmap.coord(inner, outer), dst, completion=full[stage_idx],
                                     cache_hint=EVICT_NORMAL)
                            # instruction_selection: cp.async.bulk.tensor.2d.shared::cluster.global
                            #   .mbarrier::complete_tx::bytes.L2::cache_hint; extent: one box
                            #   (A/B: one 128B swizzle atom = the whole 128-wide K block)
                        elif cluster_rank() == 0:
                            copy_g2s(tmap.coord(inner, outer), dst, completion=full[stage_idx],
                                     multicast_mask=0b11, cache_hint=EVICT_NORMAL)
                            # instruction_selection: cp.async.bulk.tensor.2d.shared::cluster.global
                            #   .mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint
                            #   (mask 3, u16 operand); extent: one box into both CTAs
                    expect_bytes(full[stage_idx], A_STAGE + B_STAGE + BLOCK_M * 4)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64;
                    #   extent: one (arrive count 1)
            if kNumTMAMulticast > 1:      # drain so no peer still arrives into our barriers
                for i in runtime_range(kNumStages, advance=advance_pipeline):
                    wait(empty[stage_idx], parity=phase ^ 1)
                    # instruction_selection: the same try_wait.parity + 0x989680 retry loop
```

### Role 2: math warpgroups (warps 0 .. M/32-1)

```python
    else:
        setmaxnreg(inc, 248 if kNumMathThreads == 128 else 232)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32; extent: math warpgroups
        math_wg_idx = shuffle_index(thread_idx // 128, src_lane=0)
        # instruction_selection: shfl.sync.idx.b32; extent: scalar
        r_0 = warp_idx * 16 + lane_idx // 4; r_1 = r_0 + 8   # accumulator rows of this thread
        # 64-bit GMMA descriptor fields, written explicitly (`make_smem_desc(ptr, 1)`):
        #   bits [0,14)  start address >> 4      bits [16,30) LBO >> 4 = 0
        #   bits [32,46) SBO >> 4 = 1024 >> 4    bits [49,52) base offset = 0
        #   bits [62,64) = 1 (the 128B-swizzle code); K-major, 8-row groups 1024 B apart
        a_desc = desc(start=off_a(0) + math_wg_idx * 64 * 128, lbo=0, sbo=1024, bits_62_63=1)
        b_desc = desc(start=off_b(0), lbo=0, sbo=1024, bits_62_63=1)
        a_desc_lo = shuffle_index(a_desc.lo32, 0); b_desc_lo = shuffle_index(b_desc.lo32, 0)
        # instruction_selection: shfl.sync.idx.b32 x2; only the low (start-address) word is
        #   ever rewritten below; the high word is a constant

        while get_next_block():
            # ---- B scales for this block -> SMEM -----------------------------------
            num_former_iters = BLOCK_N // 8; num_full_iters = num_former_iters
            if not kMustUseUniformedScaleB:
                num_former_iters = min(BLOCK_N, 128 - n_block_idx * BLOCK_N % 128) // 8
                num_full_iters   = min(shape_n - n_block_idx * BLOCK_N, BLOCK_N) // 8
            num_sfb = shape_k_scales * (1 if num_former_iters >= num_full_iters else 2)
            if thread_idx >= 32:                    # warp 0 overlaps the previous TMA store
                base = group_offset(m_block_idx) * shape_n_sfb * shape_k_scales \
                       + (n_block_idx * BLOCK_N // 128) * shape_k_scales        # K-major SFB
                for i in runtime_range(thread_idx - 32, num_sfb, step=kNumMathThreads - 32):
                    # K-major SFB (stride_n = shape_k_scales, stride_k = 1): row 1 at element
                    # (i - shape_k_scales) * 1 + shape_k_scales == i, so both rows read sfb[base + i]
                    v = reg_tile(f32, 1)
                    copy_g2r(sfb[base + i], v)
                    # instruction_selection: ld.global.f32 (no .nc); extent: scalar per i
                    copy_r2s(v, at(smem, off_sfb + 4 * i))
                    # instruction_selection: st.shared.f32; extent: scalar per i
            named_barrier(8, kNumMathThreads)
            # instruction_selection: bar.sync 8, kNumMathThreads; extent: math warpgroups

            accum = reg_tile(f32, kNumAccum)
            final_accum = reg_tile(f32, kNumAccum * (BLOCK_M // WAVE_BLOCK_M)); fill(final_accum, 0)
            do_wgmma_store = True          # BLOCK_M >= 64 for every in-scope config

            def empty_barrier_arrive():
                if kNumTMAMulticast == 1:
                    if lane_idx == 0: arrive(empty[stage_idx])
                    # instruction_selection: mbarrier.arrive.shared::cta.b64; extent: one per warp
                else:
                    target = lane_idx if sched.is_peer_cta_alive else cluster_rank()
                    if lane_idx < kNumTMAMulticast:
                        remote_arrive(empty[stage_idx], cta=target)
                        # instruction_selection: mapa.shared::cluster.u32 +
                        #   mbarrier.arrive.shared::cluster.b64; extent: lanes 0..1 of each warp

            if is_computation_valid(m_block_idx, math_wg_idx * 64):
                # `dispatch_num_former_iters<0, kGap, kEnd>(sel, body)`: an if-chain
                #   if sel == 0: body(); elif sel == kGap: body(); ... elif sel == kEnd: body()
                # with sel = num_former_iters if kShouldOptimize else 0, kGap = gcd(128,BN)/8,
                # kEnd = 128/8 if kShouldOptimize else 0, and
                # kShouldOptimize = 128/gcd(128,BN) <= 4 and not kMustUseUniformedScaleB.
                # Exactly one branch runs. Inside branch c the predicate below compares with the
                # *runtime* `num_former_iters`, which that branch's condition pins to c, so the
                # compiler folds it. Uniform B scales (BN=128): kShouldOptimize is false, there
                # is one branch (sel = 0), and the predicate is the constant `True`
                # (kMustUseUniformedScaleB), so only s00/s10 exist (16 mul.f32 in the 64x128 PTX).
                # The compiler removes only branches it can prove unreachable: BN=192 keeps
                # {0, 8, 16} (3 copies, 96 WGMMAs; the 0 branch is emitted although never taken),
                # BN=96 keeps 4 (128 WGMMAs: compares against 0/32/64, then `< 96`; its 16 branch
                # is removed), BN=128 has 1 (32 WGMMAs).
                sel = num_former_iters if kShouldOptimize else 0
                for c in static_range(0, kEnd + 1, kGap):
                  if sel == c:
                    # instruction_selection: setp.eq + bra per branch; extent: block-uniform
                    for k_block_idx in runtime_range(num_total_k_blocks, unroll=8,
                                                     advance=advance_pipeline):
                        a_lo = a_desc_lo + stage_idx * (A_STAGE // 16)
                        b_lo = b_desc_lo + stage_idx * (B_STAGE // 16)
                        scale_b_0 = copy_s2r(at(smem, off_sfb + 4 * k_block_idx))
                        if not kMustUseUniformedScaleB:
                            scale_b_1 = copy_s2r(at(smem, off_sfb + 4 * (k_block_idx + shape_k_scales)))
                        # instruction_selection: ld.shared.f32; extent: scalar (x2 non-uniform)
                        wait(full[stage_idx], parity=phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64
                        #   with the 0x989680 hint inside the retry loop; extent: one stage
                        for local_idx in static_range(BLOCK_M // WAVE_BLOCK_M):    # = 1
                            m_off = local_idx * WAVE_BLOCK_M
                            scale_a_0 = copy_s2r(at(smem, off_sfa(stage_idx) + 4 * (r_0 + m_off)))
                            scale_a_1 = copy_s2r(at(smem, off_sfa(stage_idx) + 4 * (r_1 + m_off)))
                            # instruction_selection: ld.shared.f32 x2; read before the arrive so
                            #   the next block cannot overwrite them
                            wgmma_fence_operand(accum)     # register ordering only, no instruction
                            wgmma_arrive()
                            # instruction_selection: wgmma.fence.sync.aligned; extent: warpgroup
                            for k in static_range(128 // 32):              # 4 k-steps
                                gemm(accum, desc(lo=a_lo + (m_off * 128 + k * 32) // 16),
                                     desc(lo=b_lo + k * 32 // 16), scale_d=(k != 0),
                                     instr=WGMMA)
                                # instruction_selection: wgmma.mma_async.sync.aligned
                                #   .m64n{BLOCK_N}k32.f32.e4m3.e4m3 (scale-d predicate);
                                #   extent: one instruction
                            wgmma_commit()
                            # instruction_selection: wgmma.commit_group.sync.aligned
                            wgmma_fence_operand(accum)
                            wgmma_wait(0)
                            # instruction_selection: wgmma.wait_group.sync.aligned 0
                            if local_idx == BLOCK_M // WAVE_BLOCK_M - 1:
                                empty_barrier_arrive()
                            s00 = mul(scale_a_0, scale_b_0); s10 = mul(scale_a_1, scale_b_0)
                            if not kMustUseUniformedScaleB:
                                s01 = mul(scale_a_0, scale_b_1); s11 = mul(scale_a_1, scale_b_1)
                            # instruction_selection: mul.f32 x2 (x4); extent: scalars
                            dst = final_accum[kNumAccum * local_idx :]
                            for i in static_range(kNumAccum // 4):
                                p = kMustUseUniformedScaleB or i < num_former_iters   # .cuh:342
                                fma(dst[4i+0], select(p, s00, s01), accum[4i+0])
                                fma(dst[4i+1], select(p, s00, s01), accum[4i+1])
                                fma(dst[4i+2], select(p, s10, s11), accum[4i+2])
                                fma(dst[4i+3], select(p, s10, s11), accum[4i+3])
                                # instruction_selection: fma.rn.f32 x4 (the selects fold: p is a
                                #   constant in every in-scope copy);
                                #   extent: kNumAccum FMAs per wave
            else:   # whole row band invalid: keep the pipeline in lockstep
                for k_block_idx in runtime_range(num_total_k_blocks, advance=advance_pipeline):
                    wait(full[stage_idx], parity=phase)
                    # instruction_selection: try_wait.parity + 0x989680 retry loop (`.cuh:354`)
                    empty_barrier_arrive()     # selection as above (`.cuh:355`)

            # ---- epilogue: registers -> swizzled SMEM D -> TMA store -----------------
            if thread_idx < BLOCK_N // TMA_D_BLOCK_N:
                store_wait(read=0)
                # instruction_selection: cp.async.bulk.wait_group.read 0; extent: the
                #   previous block's TMA store, by its issuing threads only
            named_barrier(9, kNumWGMMAStoreThreads)
            # instruction_selection: bar.sync 9, kNumWGMMAStoreThreads
            for local_idx in static_range(BLOCK_M // WAVE_BLOCK_M):
                m_off = local_idx * WAVE_BLOCK_M
                src = final_accum[kNumAccum * local_idx :]
                for i in static_range(kNumAccum // 4):
                    atom, in_atom = i // (TMA_D_BLOCK_N // 8), i % (TMA_D_BLOCK_N // 8)
                    bank_group = in_atom + lane_idx * (kSwizzleDMode // 16)
                    if kSwizzleDMode // 16 == 8:          # 128B swizzle shortcut
                        row, col = in_atom // 8 + lane_idx, in_atom
                    else:
                        row, col = bank_group // 8, bank_group % 8
                    col ^= row % (kSwizzleDMode // 16)
                    d_off = (off_d + warp_idx * (16 * kSwizzleDMode) + m_off * kSwizzleDMode
                             + atom * BLOCK_M * kSwizzleDMode + row * 128 + col * 16)
                    lo = cast((src[4i+0], src[4i+1]), bf16x2, rn)
                    hi = cast((src[4i+2], src[4i+3]), bf16x2, rn)
                    # instruction_selection: cvt.rn.bf16x2.f32 x2; extent: 4 values
                    copy_r2s((lo, hi), at(smem, d_off))
                    # instruction_selection: stmatrix.sync.aligned.x2.m8n8.shared.b16 (only
                    #   lanes 0..15 supply addresses); extent: kNumAccum/4 per wave
            store_fence()
            # instruction_selection: fence.proxy.async.shared::cta
            named_barrier(9, kNumWGMMAStoreThreads)
            # instruction_selection: bar.sync 9, kNumWGMMAStoreThreads
            if thread_idx < BLOCK_N // TMA_D_BLOCK_N:
                n_in = thread_idx * TMA_D_BLOCK_N
                m_idx = (group_offset(...) if MGroupedMasked else 0) * shape_m + m_block_idx * BLOCK_M
                copy_s2g(at(smem, off_d + n_in * BLOCK_M * 2),
                         tmap_d.coord(n_block_idx * BLOCK_N + n_in, m_idx))
                # instruction_selection: cp.async.bulk.tensor.2d.global.shared::cta.bulk_group;
                #   extent: one [BLOCK_M x TMA_D_BLOCK_N] atom per issuing thread
                store_commit()
                # instruction_selection: cp.async.bulk.commit_group
            warp_sync()
            # instruction_selection: bar.warp.sync -1
```

## Logical GEMM shapes and owners

| Op | Shape | Owner | Per |
| --- | --- | --- | --- |
| WGMMA | m64 n`BLOCK_N` k32, FP8 x FP8 -> FP32, A/B from SMEM | each math warpgroup, rows `math_wg_idx*64 ..+64` | 4 per K block (`BLOCK_K/32`) |
| promotion | `kNumAccum` FMAs (`final += s * accum`) | each math thread | per K block |
| TMA A / SFA / B | [`BLOCK_M` x 128] fp8 / [`BLOCK_M`] f32 / [`BLOCK_N` x 128] fp8 | producer lane (multicast: rank 0 for the multicast operand) | per K block |
| TMA D | [`BLOCK_M` x `TMA_D_BLOCK_N`] bf16 | thread `t < BLOCK_N/TMA_D_BLOCK_N` | per block |

## TensorMap fields (`launch.py`, from `runtime_utils.hpp`)

| map | gmem dims (inner, outer) | box (inner, outer) | swizzle | outer stride |
| --- | --- | --- | --- | --- |
| A | (K, M * [G if masked]) | (128, BLOCK_M) | 128B | K |
| B | (K, N * G) | (128, BLOCK_N) | 128B | K |
| D | (N, M * [G if masked]) | (TMA_D_BLOCK_N, BLOCK_M) | `kSwizzleDMode` | N |
| SFA | (align(M,4), ceil(K/128) * [G if masked]) | (BLOCK_M, 1) | none | align(M,4) |

All loads use `L2_PROMOTION_256B`, no OOB fill; data type from the tensor (e4m3 -> u8 code,
bf16, f32).

## Storage lifetimes

| region | written by | read by | released |
| --- | --- | --- | --- |
| `smem_a/b/sfa[s]` | TMA (`full[s]` completes on bytes) | WGMMA (A/B), math `ld.shared` (SFA) | `empty[s]` after `wgmma.wait_group 0` + SFA reads, per warp, to every CTA that wrote it |
| `smem_sfb` | math threads >= 32, per block | math threads, per K block | next block, after `bar.sync 8` |
| `smem_d` | `stmatrix` of all store threads | TMA store | next block's `wait_group.read 0` + `bar.sync 9` |

## Static specialization boundary

Compile time: everything in `KernelSpec` plus `SHAPE_N`/`SHAPE_K`; runtime: `shape_m`,
`grouped_layout` contents, tensor addresses. A new N or K, group count or block config
is a new instantiation, exactly as DeepGEMM's JIT keys it.

## TIRx module and benchmark contract

`KERNEL_META` names `deepgemm_sm90_m_grouped_fp8_gemm_{contiguous,masked}`,
`runtime_cuda_archs = ["sm_90a"]`, reference requirement DeepGEMM @ 559d79fb. CONFIGS are
the upstream SM90 test shapes; `run_test` compares against
`deep_gemm.m_grouped_fp8_fp4_gemm_nt_{contiguous,masked}` on byte-identical operands
(valid rows only); `run_gpu` times both with `bench_against_deepgemm`.

## Instruction-selection summary

Placement and schedule select the instructions; the counts are evidence, not
computation. For `MGroupedContiguous_128x192_s4_mc2_g4`:

| instruction | static count | cause |
| --- | --- | --- |
| `wgmma.mma_async...m64n192k32.f32.e4m3.e4m3` | 96 | 4 k-steps x unroll 8 x 3 `num_former_iters` copies |
| `wgmma.fence` / `commit_group` / `wait_group` | 24 each | one per wave per unrolled K block per copy |
| `fma.rn.f32` | 2304 | 96 accumulators x 8 x 3 |
| `ld.shared.f32` | 96 | B-scale (2) and A-scale (2) reads per unrolled K block per copy |
| `cp.async.bulk.tensor.2d...L2::cache_hint` | 12 | A, SFA, B non-multicast paths (unrolled/duplicated by the compiler) |
| `...multicast::cluster.L2::cache_hint` | 4 | multicast path (rank 0) |
| `mbarrier.try_wait.parity.shared::cta.b64` | 64 | producer empty waits, math full waits per unrolled K block, drain |
| `mapa` + `mbarrier.arrive.shared::cluster.b64` | 56 | multicast-aware empty releases |
| `stmatrix.sync.aligned.x2.m8n8.shared.b16` | 24 | `kNumAccum/4` = 24 per thread |
| `bar.sync 8/9` | 3 | SFB staged, D written, D fenced |
| `setmaxnreg.dec 40` / `.inc 232` | 1 / 1 | role register split |
| `griddepcontrol.wait`, `fence.mbarrier_init.release.cluster`, `barrier.cluster.*` | 1 each | prologue |

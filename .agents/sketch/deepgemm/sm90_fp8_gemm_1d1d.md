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
scheduler and TMA/WGMMA/tensor-map helpers it instantiates. See NOTICE and
licenses/ for upstream attribution.
-->

# SM90 FP8 1D1D GEMM (k-grouped and normal, FP32 accumulate): coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new IR, a
builder API, or a mathematical reference. It shows the TIRx kernel as an explicit
runtime ABI and launch; explicit linear GMEM/SMEM/register storage with byte offsets;
the persistent scheduler, the in-kernel tensor-map rewrite and the role control flow;
primitive directional copies and primitive computation; explicit stage/phase changes and
publication/reuse edges; and hardware instruction selection, read from the source's
line-info PTX export.

The implementation represented by this sketch is maintained in
[`tirx_kernels/ported/deepgemm/_sm90_fp8_gemm_1d1d/`](../../../tirx_kernels/ported/deepgemm/_sm90_fp8_gemm_1d1d/__init__.py),
with two entry modules (`sm90_k_grouped_fp8_gemm_contiguous.py`, `sm90_fp8_gemm_1d1d.py`)
that pin one descriptor each. Those modules are the source of truth.

Source export: line-info PTX, SASS and cubin of all 14 instantiations the upstream SM90 tests
reach -- the 7 `test_k_grouped_gemm_contiguous` shapes (`k_grouped_fp8_gemm_nt_contiguous`)
and the 7 FP32 + accumulate wgrad calls of `test_gemm` (`fp8_fp4_gemm_nt(c=d, recipe=(1,1,128))`)
-- built through DeepGEMM's own JIT with `DG_JIT_WITH_LINEINFO=1` by
`sm90-probe/dgkg90/export_driver.py`, every call checked with upstream's `calc_diff < 1e-3`
against an FP32 dequantized reference (all pass, diff <= 8e-9). The empty-group test variants
reuse the same instantiations (the group count and K are runtime data). DeepGEMM's
`DG_PRINT_CONFIGS` output equals the ported heuristic's pick for all 14. They fall into 6
classes:

| class (`gemm_type`, BMxBN, stages, multicast) | instantiations | why |
| --- | --- | --- |
| `KGroupedContiguous` 128x136 s4 mc1 | 1 | `(8, 768, 2048, 128)`: 1 wave, multicast disabled by the cost model |
| `KGroupedContiguous` 128x152 s4 mc1 | 4 | 8 and 16 groups: `num_groups > 4` disables multicast |
| `KGroupedContiguous` 128x152 s4 mc2 (on B) | 2 | 4 groups |
| `Normal` 128x136 s4 mc2 (on B) | 1 | M = 2112 |
| `Normal` 128x152 s4 mc2 (on B) | 5 | M in {24576, 32768, 7168, 4096, 7168} |
| `Normal` 64x152 s6 mc2 (on B) | 1 | M = 576 |

`.porting/sm90_fp8_gemm_1d1d/source_export/INDEX.txt` maps every instantiation to its cache
directory; one representative per class is copied there under the class name, and
`opcode_hist.txt` holds the per-class static opcode counts summarized at the end. Counts are
instruction lines (a predicated line counts once). Unless stated, line references are to the
`KGroupedContiguous_128x136_s4_mc1` representative.

**In scope.** `kGemmType` = `KGroupedContiguous` and `Normal`; FP8 e4m3 A and B, both K-major;
FP32 C/D with accumulation through TMA reduce-add; FP32 per-128-K scales on both operands
(MN-major, TMA-aligned); `BLOCK_K = 128`; `BLOCK_M` in {64, 128}; `BLOCK_N` any 1D1D candidate
(16, 24, 40, ..., 152 -- the code path is BLOCK_N-generic; the exported values are 136 and 152);
`kNumTMAMulticast` 1, or 2 on B; compiled dims "mn" (k-grouped) and "nk" (normal).

**Out of scope.** Multicast on A (`kIsTMAMulticastOnA`): no upstream shape selects it; the
branches are kept below where the source shares code, marked `[on-A]`. `BLOCK_M` 16/32
(the heuristic offers them for M <= 32 but the source's static assert rejects them). PSUM
k-grouped layouts and MN-major operands (SM100 only); BF16 output; tile (`Tx`) primitives.

## Pipeline at a glance

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0 .. M/32-1 (math, M = kNumMathThreads = 128 or 256) | per scheduler block, per K block: one WGMMA chain per warpgroup into `accum`, FP32 promotion with per-row A scales and per-column B scales into `final_accum`; then `st.shared.v2.f32` into D and one TMA reduce-add per warpgroup | wait `full[stage]`; arrive `empty[stage]` (per warp, to every CTA of the cluster); named barrier `8 + math_wg_idx` (128 threads) twice per block |
| M/32 (TMA WG, warp 0) | prefetch the five TensorMaps; then, one elected lane: the persistent producer loop (k-grouped: tensor-map rewrite per group; per K block TMA SFA + SFB + A + B into one stage) | wait `empty[stage]`; publish `full[stage]` with expect-tx A+B+SFA+SFB bytes; k-grouped: release the rewritten maps to the tensormap proxy |
| M/32+1 (TMA WG, warp 1) | k-grouped: copy the two base TensorMaps into SMEM; initialize `full`/`empty` barriers | publishes barrier init (and the SMEM maps) via `fence.mbarrier_init` + CTA/cluster sync |
| M/32+2, M/32+3 | idle after `setmaxnreg.dec` | none |

Math warpgroup `w` owns rows `[w*64, w*64+64)` of the block; with `BLOCK_M = 64` there is
one math warpgroup. The producer lane and every math thread construct their own scheduler
and walk the identical block sequence (including the k-group walk), so `stage_idx`/`phase`
stay in lockstep without communication.

## Primitive vocabulary

```python
specialize(...)     # compile-time variant selection
launch(...)         # launch topology and attributes
tile(...)           # linear storage: dtype, element count, placement, alignment
reg_tile(...)       # role-local register array (no layout)
at(buf, byte_off)   # address of a byte offset inside a linear buffer
desc(...)           # WGMMA SMEM matrix descriptor (64-bit, fields written explicitly)

copy_g2s(tmap.coord(...), dst, completion, multicast_mask=None)   # TMA load
reduce_s2g(src, tmap.coord(...), op=add)                         # TMA reduce store
copy_p2r(param, dst)   # grid-constant parameter bytes -> registers
copy_r2g(src, dst)     # registers -> global plain store
copy_g2r(src, dst)     # plain global load of one element into a register
copy_r2s(src, dst)     # shared store
copy_s2r(src, dst)     # shared load
tmap_replace(field, smem_map, value)   # in-place tensor-map field rewrite

gemm(acc, a_desc, b_desc, scale_d, instr)   # one wgmma.mma_async
fill(dst, value)                             # register initialization
mul(dst, a, b); fma(acc, a, b)               # FP32 promotion
elect_predicate(mask); shuffle_index(dst, src, lane)
```

Schedule operations: `prefetch`, `init`, `fence`, `wait`, `arrive`, `expect_bytes`,
`remote_arrive`, `cta_sync`, `cluster_sync`, `named_barrier`, `pdl_wait`,
`setmaxnreg`, `wgmma_fence`, `wgmma_arrive`, `wgmma_commit`, `wgmma_wait`,
`store_fence`, `store_commit`, `store_wait`, `tmap_commit`, `tmap_wait`,
`tmap_release`, `tmap_acquire`. There are no compound ops: no `scheduler`,
`epilogue`, `promote`, or `update_tensor_map`.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
specialize(KernelSpec(
    gemm_type,                       # KGroupedContiguous (3) or Normal (0)
    SHAPE_M, SHAPE_N, SHAPE_K,       # compiled dims: "mn" (k-grouped, SHAPE_K = 0) / "nk" (normal, SHAPE_M = 0)
    kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K=128,
    kSwizzleAMode=128, kSwizzleBMode=128,
    kNumStages, kNumTMAThreads=128, kNumMathThreads,       # 128 if BLOCK_M <= 64 else 256
    kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs=132))
# instruction_selection: none; extent: the template arguments of `generate_impl`
# (sm90_fp8_gemm_1d1d.hpp:33), produced by spec.make_spec from the ported heuristic

WGMMA = m64n{BLOCK_N}k32.f32.e4m3.e4m3 (SS); kNumAccum = BLOCK_N / 2
kNumPipelineUnrolls = 0 if KGroupedContiguous else kNumStages
kNumTMARegisters, kNumMathRegisters = (40, 232) if kNumPipelineUnrolls == 0 else (24, 240)

def kernel(gmem_a: fp8*, gmem_b: fp8*, grouped_layout: i32*, tensor_map_buffer: CUtensorMap*,
           shape_m: u32, shape_n: u32, shape_k: u32,
           tmap_a_base, tmap_b_base, tmap_sfa, tmap_sfb, tmap_cd: __grid_constant__ CUtensorMap):
    launch(grid=(kNumSMs,1,1), block=(128 + kNumMathThreads,1,1),
           cluster=(kNumTMAMulticast,1,1) if kNumTMAMulticast > 1 else None,
           dynamic_smem=smem_size, launch_bounds=(128 + kNumMathThreads, 1))
    # instruction_selection: .maxntid N,1,1 + .minnctapersm 1 (PTX header); the five maps are
    #   `.param .align 128 .b8 [128]`; cluster and dynamic smem are launch attributes
    shape_m, shape_n, shape_k = SHAPE_M or shape_m, SHAPE_N or shape_n, SHAPE_K or shape_k

    # ---- storage: one dynamic, 1024-aligned linear pool --------------------
    smem = tile(u8, smem_size, space=shared.dyn, align=1024)
    TM_SIZE    = 2 * 128 if KGroupedContiguous else 0          # SMEM tensor-map copies
    D_SIZE     = BLOCK_M * BLOCK_N * 4                        # FP32 D tile (multiple of 1024)
    A_STAGE    = BLOCK_M * 128;  B_STAGE = BLOCK_N * 128       # fp8 bytes per stage
    SFA_STAGE  = BLOCK_M * 4                                   # multiple of 128 (asserted)
    SFB_STAGE  = align(BLOCK_N * 4, 128)
    SF_OFF     = TM_SIZE + D_SIZE + kNumStages * (A_STAGE + B_STAGE)
    BAR_OFF    = SF_OFF + kNumStages * (SFA_STAGE + SFB_STAGE)
    off_tmap_a, off_tmap_b = 0, 128                           # k-grouped only
    off_d       = TM_SIZE                                     # f32[BLOCK_M][BLOCK_N], row stride BLOCK_N
    off_a(s)    = TM_SIZE + D_SIZE + s * A_STAGE              # 128B-swizzled [BLOCK_M rows x 128B]
    off_b(s)    = TM_SIZE + D_SIZE + kNumStages * A_STAGE + s * B_STAGE
    off_sfa(s)  = SF_OFF + s * SFA_STAGE                      # f32[BLOCK_M], linear
    off_sfb(s)  = SF_OFF + kNumStages * SFA_STAGE + s * SFB_STAGE   # f32[BLOCK_N], linear
    off_full(s) = BAR_OFF + 8 * s;  off_empty(s) = BAR_OFF + 8 * (kNumStages + s)
    # e.g. 128x136 s4 k-grouped: D @256, A @69888, B @135424, SFA @205056 (stride 512),
    #   SFB @207104 (stride 640), full @209664, empty @209696 (read from the export).
    gmem_tmap_a = tensor_map_buffer + 2 * block_idx_x; gmem_tmap_b = gmem_tmap_a + 1   # 128 B each

    warp_idx = shuffle_index(thread_idx // 32, src_lane=0)
    # instruction_selection: shfl.sync.idx.b32 (warp-uniform broadcast); extent: scalar
    lane_idx = thread_idx % 32

    # ---- prologue -----------------------------------------------------------
    if warp_idx == kNumMathThreads // 32 and elect_predicate(0xffffffff):
        # instruction_selection: elect.sync; extent: one lane
        for t in (tmap_a_base, tmap_b_base, tmap_sfa, tmap_sfb, tmap_cd):
            prefetch(t)
            # instruction_selection: cvta.param.u64 + prefetch.tensormap [generic]; extent: 5
    warp_sync()
    # instruction_selection: bar.warp.sync -1

    if warp_idx == kNumMathThreads // 32 + 1 and elect_predicate(0xffffffff):
        if KGroupedContiguous:
            for (src, off) in ((tmap_a_base, off_tmap_a), (tmap_b_base, off_tmap_b)):
                # `*smem_tensor_map_a = tensor_map_a_base` (.cuh:130-131): a 128-byte struct copy
                staged = reg_tile(u32x4, 8)
                copy_p2r(src[0:128], staged)
                # instruction_selection: ld.param.v4.u32 x8 ([param + 16j]); extent: 128 B per map
                copy_r2s(staged, at(smem, off)[0:128])
                # instruction_selection: st.shared.v4.u32 x8 (emitted from offset 112 down to 0);
                #   extent: 128 B per map
        for s in static_range(kNumStages):
            init(full[s] @ at(smem, off_full(s)), arrive_count=1)
            init(empty[s] @ at(smem, off_empty(s)),
                 arrive_count=kNumTMAMulticast * kNumMathThreads // 32)
            # instruction_selection: mbarrier.init.shared::cta.b64; extent: 2*kNumStages
        fence(mbarrier_init, scope=cluster)
        # instruction_selection: fence.mbarrier_init.release.cluster
    if kNumTMAMulticast > 1:
        cluster_sync(arrive="relaxed")
        # instruction_selection: barrier.cluster.arrive.relaxed.aligned +
        #   barrier.cluster.wait.aligned
    else:
        cta_sync()
        # instruction_selection: bar.sync 0
    pdl_wait()
    # instruction_selection: griddepcontrol.wait

    sched = Scheduler(gemm_type, shape_m, shape_n, shape_k, grouped_layout)   # see below
    iter_idx = 0                # one counter per role, advanced once per K block / drain step
    def advance(stage_idx, phase):   # get_pipeline(iter_idx ++) (.cuh:165-168)
        iter_idx += 1
        return iter_idx % kNumStages, (iter_idx // kNumStages) & 1
    stage_idx, phase = 0, 0     # = get_pipeline(0)
```

### Persistent scheduler (`scheduler/gemm.cuh`, `kKAlignment = kSFKSpan = 128`)

```python
# Members: current_iter = -1, current_group_idx = 0, current_num_valid_groups = 0,
#   current_k_cumsum = 0, current_sf_k_cumsum = 0, current_shape_k = shape_k,
#   is_peer_cta_alive = True (never written for k-grouped), next_group_idx, next_shape_k.
# kNum1DBlocksPerGroup = argmin over {8, 16} (strict `<`, first wins ties) of
#   candidate*BLOCK_N + ceil(kNumSMs/candidate)*BLOCK_M   (multicast on A: group on N)
#   candidate*BLOCK_M + ceil(kNumSMs/candidate)*BLOCK_N   (otherwise: group on M)
num_m_blocks = ceil(shape_m / BLOCK_M); num_n_blocks = ceil(shape_n / BLOCK_N)
num_blocks = num_m_blocks * num_n_blocks
if KGroupedContiguous:                        # constructor (`gemm.cuh:107-113`)
    get_next_k_group(current_group_idx, current_shape_k)
    next_group_idx = current_group_idx + 1
    get_next_k_group(next_group_idx, next_shape_k)

def get_next_k_group(group_idx, shape_k):     # skip empty groups
    while group_idx < kNumGroups:
        shape_k = grouped_layout[group_idx]
        # instruction_selection: ld.global.u32 (`gemm.cuh:68`); extent: scalar per probe
        #   (the constructor's first call is fully unrolled over kNumGroups by the compiler)
        if shape_k > 0: break
        group_idx += 1

def get_next_block():                         # -> (m_block_idx, n_block_idx) or False
    next_block_idx = (++current_iter) * kNumSMs + block_idx_x
    if KGroupedContiguous:
        while True:
            if current_group_idx == kNumGroups: return False
            if next_block_idx < (current_num_valid_groups + 1) * num_blocks: break
            current_sf_k_cumsum += ceil(current_shape_k / 128)
            current_num_valid_groups += 1
            current_k_cumsum += current_shape_k
            current_group_idx = next_group_idx; next_group_idx += 1
            current_shape_k = next_shape_k
            get_next_k_group(next_group_idx, next_shape_k)
        swizzle(next_block_idx - current_num_valid_groups * num_blocks)
    else:  # Normal
        if next_block_idx >= num_blocks: return False
        is_peer_cta_alive = (num_n_blocks % kNumTMAMulticast == 0 or
                             num_m_blocks % kNumTMAMulticast == 0 or
                             (next_block_idx ^ 1) < num_blocks)
        swizzle(next_block_idx)

def swizzle(block_idx):                       # get_swizzled_block_idx, SM90 branch
    primary   = num_n_blocks if kIsTMAMulticastOnA else num_m_blocks
    secondary = num_m_blocks if kIsTMAMulticastOnA else num_n_blocks
    per_group = secondary * kNum1DBlocksPerGroup
    first = (block_idx // per_group) * kNum1DBlocksPerGroup; in_group = block_idx % per_group
    num_blocks_in_group = min(kNum1DBlocksPerGroup, primary - first)
    if kNumTMAMulticast > 1 and num_blocks_in_group % 2 != 0:   # unaligned multicast tail
        if in_group < (num_blocks_in_group ^ 1) * secondary: num_blocks_in_group ^= 1
        else: in_group -= (num_blocks_in_group ^ 1) * secondary
              first += num_blocks_in_group ^ 1; num_blocks_in_group = 1
    if kIsTMAMulticastOnA: m = in_group // num_blocks_in_group; n = first + in_group % num_blocks_in_group
    else:                  m = first + in_group % num_blocks_in_group; n = in_group // num_blocks_in_group
    # instruction_selection: div.u32 / rem by the runtime num_blocks_in_group; extent: scalar

def is_tma_multicast_valid(m_block_idx):     # Normal and k-grouped: only the group-size test
    return num_blocks_in_group != 1
```

### Role 1: TMA producer (warp M/32, one elected lane)

```python
    if warp_idx >= kNumMathThreads // 32:
        setmaxnreg(dec, kNumTMARegisters)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 40 (k-grouped) / 24 (normal)
        if warp_idx == kNumMathThreads // 32 and elect_predicate(0xffffffff):
            # instruction_selection: elect.sync; extent: one lane
            last_group_idx = kNumGroups
            while get_next_block():
                valid = is_tma_multicast_valid(m_block_idx)
                mc_a = kNumTMAMulticast if (kIsTMAMulticastOnA and valid) else 1
                mc_b = kNumTMAMulticast if (not kIsTMAMulticastOnA and valid) else 1
                num_k_blocks = ceil(sched.current_shape_k / 128)
                m_idx = m_block_idx * BLOCK_M; n_idx = n_block_idx * BLOCK_N

                if KGroupedContiguous and last_group_idx != sched.current_group_idx:
                    last_group_idx = sched.current_group_idx
                    k_off = sched.current_k_cumsum
                    tmap_replace(global_address, at(smem, off_tmap_a), gmem_a + k_off * shape_m)
                    tmap_replace(global_address, at(smem, off_tmap_b), gmem_b + k_off * shape_n)
                    # instruction_selection: tensormap.replace.tile.global_address.shared::cta
                    #   .b1024.b64 [smem32], addr64; extent: 2
                    for off in (off_tmap_a, off_tmap_b):
                        tmap_replace(global_dim, at(smem, off), ord=0, value=sched.current_shape_k)
                        # instruction_selection: tensormap.replace.tile.global_dim.shared::cta.b1024.b32
                        #   [smem as u64], 0, u32; extent: 2
                        tmap_replace(global_stride, at(smem, off), ord=0, value=u64(sched.current_shape_k))
                        # instruction_selection: tensormap.replace.tile.global_stride.shared::cta
                        #   .b1024.b64 [smem as u64], 0, u64; extent: 2
                    tmap_commit()
                    # instruction_selection: cp.async.bulk.commit_group
                    tmap_wait(read=0)
                    # instruction_selection: cp.async.bulk.wait_group.read 0 (no in-flight TMA
                    #   still reads the maps)
                    warp_sync(mask=1 << lane_idx)
                    # instruction_selection: bar.warp.sync (1 << lane); orders the GMEM stores
                    #   after the wait (`.cuh:206`)
                    for (off, dst) in ((off_tmap_a, gmem_tmap_a), (off_tmap_b, gmem_tmap_b)):
                        # `*(gmem_tensor_map_a) = *(smem_tensor_map_a)` (.cuh:208-209)
                        staged = reg_tile(u32x4, 8)
                        copy_s2r(at(smem, off)[0:128], staged)
                        # instruction_selection: ld.shared.v4.u32 x8; extent: 128 B per map
                        copy_r2g(staged, dst[0:128])
                        # instruction_selection: st.global.v4.u32 x8 (offset 112 down to 0, global
                        #   address via cvta.to.global); extent: 128 B per map
                    tmap_release(scope=gpu)
                    # instruction_selection: fence.proxy.tensormap::generic.release.gpu
                    tmap_acquire(gmem_tmap_a, 128); tmap_acquire(gmem_tmap_b, 128)
                    # instruction_selection: fence.proxy.tensormap::generic.acquire.gpu [gen64], 128; x2

                tmap_a = gmem_tmap_a if KGroupedContiguous else tmap_a_base   # generic GMEM vs param
                tmap_b = gmem_tmap_b if KGroupedContiguous else tmap_b_base
                for k_block_idx in runtime_range(num_k_blocks, unroll=kNumPipelineUnrolls):
                    # nvcc unrolls this loop x4 with a 3-copy remainder for k-grouped (`unroll 0`
                    #   is not "do not unroll" there) and xkNumStages for normal (see the summary).
                    wait(empty[stage_idx], parity=phase ^ 1)
                    # instruction_selection: LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64
                    #   P1, [bar], phase, 10000000; @P1 bra DONE; bra LAB_WAIT; extent: one stage
                    sf_k = sched.current_sf_k_cumsum + k_block_idx       # Normal: 0 + k_block_idx
                    for (tmap, dst, inner, outer, mc) in (
                            (tmap_sfa, at(smem, off_sfa(stage_idx)), m_idx, sf_k, mc_a),
                            (tmap_sfb, at(smem, off_sfb(stage_idx)), n_idx, sf_k, mc_b),
                            (tmap_a,   at(smem, off_a(stage_idx)),   k_block_idx * 128, m_idx, mc_a),
                            (tmap_b,   at(smem, off_b(stage_idx)),   k_block_idx * 128, n_idx, mc_b)):
                        if mc == 1:
                            copy_g2s(tmap.coord(inner, outer), dst, completion=full[stage_idx],
                                     cache_hint=EVICT_NORMAL)
                            # instruction_selection: cp.async.bulk.tensor.2d.shared::cluster.global
                            #   .mbarrier::complete_tx::bytes.L2::cache_hint; extent: one box
                            #   (SFA/SFB: [BLOCK_MN x 1] f32; A/B: one 128B-swizzle atom)
                        elif cluster_rank() == 0:
                            copy_g2s(tmap.coord(inner, outer), dst, completion=full[stage_idx],
                                     multicast_mask=0b11, cache_hint=EVICT_NORMAL)
                            # instruction_selection: mov %cluster_ctarank; cp.async.bulk.tensor.2d
                            #   .shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
                            #   .L2::cache_hint (mask u16 3); extent: one box into both CTAs
                    expect_bytes(full[stage_idx], A_STAGE + B_STAGE + SFA_STAGE + BLOCK_N * 4)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], tx
                    stage_idx, phase = advance(stage_idx, phase)
            if kNumTMAMulticast > 1:      # drain so no peer still arrives into our barriers
                for s in static_range(kNumStages):
                    wait(empty[stage_idx], parity=phase ^ 1)
                    # instruction_selection: the same try_wait.parity retry loop; extent: kNumStages
                    stage_idx, phase = advance(stage_idx, phase)
```

### Role 2: math warpgroups (warps 0 .. M/32-1)

```python
    else:
        setmaxnreg(inc, kNumMathRegisters)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32 232 (k-grouped) / 240 (normal)
        math_wg_idx = shuffle_index(thread_idx // 128, src_lane=0)
        # instruction_selection: shfl.sync.idx.b32
        row_idx, col_idx = lane_idx // 4, lane_idx % 4
        r_0 = warp_idx * 16 + row_idx; r_1 = r_0 + 8           # accumulator rows of this thread

        while get_next_block():
            current_shape_k = sched.current_shape_k if KGroupedContiguous else shape_k
            current_group_idx = sched.current_group_idx if KGroupedContiguous else 0
            num_k_blocks = ceil(current_shape_k / 128)
            accum = reg_tile(f32, kNumAccum); final_accum = reg_tile(f32, kNumAccum)
            fill(final_accum, 0)
            # instruction_selection: one mov.f32 0f00000000 then register copies (mov.f32);
            #   extent: kNumAccum
            scales_b = reg_tile(f32x2, kNumAccum // 4)

            def empty_barrier_arrive(s):
                if kNumTMAMulticast == 1:
                    if lane_idx == 0: arrive(empty[s])
                    # instruction_selection: mbarrier.arrive.shared::cta.b64; extent: one per warp
                else:
                    target = lane_idx if sched.is_peer_cta_alive else cluster_rank()
                    if lane_idx < kNumTMAMulticast:
                        remote_arrive(empty[s], cta=target)
                        # instruction_selection: mapa.shared::cluster.u32 +
                        #   mbarrier.arrive.shared::cluster.b64; extent: lanes 0..1 of each warp

            for k_block_idx in runtime_range(num_k_blocks, unroll=kNumPipelineUnrolls):
                # k-grouped: kept rolled; normal: unrolled xkNumStages, one in-body exit test when
                #   kNumStages does not divide SHAPE_K/128 (see the summary)
                wait(full[stage_idx], parity=phase)
                # instruction_selection: try_wait.parity retry loop with 10000000; extent: one stage
                scale_a_0 = copy_s2r(at(smem, off_sfa(stage_idx) + 4 * r_0))
                scale_a_1 = copy_s2r(at(smem, off_sfa(stage_idx) + 4 * r_1))
                # instruction_selection: ld.shared.f32 x2 (64-bit shared address operand)
                for i in static_range(kNumAccum // 4):
                    scales_b[i] = copy_s2r(at(smem, off_sfb(stage_idx) + 4 * (8 * i + 2 * col_idx)))
                    # instruction_selection: ld.shared.v2.f32; extent: kNumAccum/4
                # (all SMEM reads precede the arrive: "next scheduled block polluting the results")
                wgmma_fence_operand(accum)          # register ordering only, no instruction
                wgmma_arrive()
                # instruction_selection: wgmma.fence.sync.aligned
                for k in static_range(128 // 32):
                    gemm(accum,
                         desc(start=off_a(stage_idx) + math_wg_idx * 64 * 128 + k * 32, lbo=0, sbo=1024, bits_62_63=1),
                         desc(start=off_b(stage_idx) + k * 32, lbo=0, sbo=1024, bits_62_63=1),
                         scale_d=(k != 0), instr=WGMMA)
                    # instruction_selection: wgmma.mma_async.sync.aligned.m64n{BLOCK_N}k32.f32.e4m3.e4m3
                    #   (scale-d predicate; descriptors `(addr >> 4) & 0x3FFF | 0x40000040_00000000`);
                    #   extent: one instruction, 4 per K block
                wgmma_commit()
                # instruction_selection: wgmma.commit_group.sync.aligned
                wgmma_fence_operand(accum)
                wgmma_wait(0)
                # instruction_selection: wgmma.wait_group.sync.aligned 0
                empty_barrier_arrive(stage_idx)
                for i in static_range(kNumAccum // 4):
                    for (j, sa, sb) in ((0, scale_a_0, scales_b[i].x), (1, scale_a_0, scales_b[i].y),
                                        (2, scale_a_1, scales_b[i].x), (3, scale_a_1, scales_b[i].y)):
                        s = mul(sa, sb)
                        # instruction_selection: mul.f32 (no .ftz); extent: one per accumulator
                        fma(final_accum[4i + j], s, accum[4i + j])
                        # instruction_selection: fma.rn.f32 final = s * accum + final; extent: one
                        #   per accumulator (`final += sa * sb * accum`, contracted, .cuh:316-319)
                stage_idx, phase = advance(stage_idx, phase)

            # ---- epilogue: registers -> FP32 SMEM D -> one TMA reduce-add per warpgroup ----
            if warp_idx % 4 == 0 and elect_predicate(0xffffffff):
                store_wait(read=0)
                # instruction_selection: elect.sync + cp.async.bulk.wait_group.read 0; extent: the
                #   previous block's reduce of this warpgroup
            named_barrier(8 + math_wg_idx, 128)
            # instruction_selection: bar.sync (8 + math_wg_idx), 128 (cutlass reserves ids 0..7)
            for i in static_range(kNumAccum // 4):
                # `smem_d_0 + i * 4` on a float2 pointer: 32 bytes (8 columns) per i (.cuh:330-334)
                copy_r2s((final_accum[4i+0], final_accum[4i+1]),
                         at(smem, off_d + 4 * (r_0 * BLOCK_N + 8 * i + 2 * col_idx)))
                copy_r2s((final_accum[4i+2], final_accum[4i+3]),
                         at(smem, off_d + 4 * (r_1 * BLOCK_N + 8 * i + 2 * col_idx)))
                # instruction_selection: st.shared.v2.f32 x2 (64-bit shared address; offsets step
                #   +32 B per i, r_1 rows +8*BLOCK_N*4 B); extent: kNumAccum/2
            store_fence()
            # instruction_selection: fence.proxy.async.shared::cta
            named_barrier(8 + math_wg_idx, 128)
            # instruction_selection: bar.sync (8 + math_wg_idx), 128
            if warp_idx % 4 == 0 and elect_predicate(0xffffffff):
                # the elected lane's own r_0 / col_idx address the source rows (lane 0: r_0 = 64 * wg)
                reduce_s2g(at(smem, off_d + 4 * (r_0 * BLOCK_N + col_idx * 2)),
                           tmap_cd.coord(n_block_idx * BLOCK_N,
                                         current_group_idx * shape_m + m_block_idx * BLOCK_M + r_0),
                           op=add)
                # instruction_selection: cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group
                #   [tmap_cd (cvta.param), {n, m}], [smem32]; extent: one [64 x BLOCK_N] f32 box
                store_commit()
                # instruction_selection: cp.async.bulk.commit_group
            warp_sync()
            # instruction_selection: bar.warp.sync -1
```

## Logical GEMM shapes and owners

| Op | Shape | Owner | Per |
| --- | --- | --- | --- |
| WGMMA | m64 n`BLOCK_N` k32, FP8 x FP8 -> FP32, A/B from SMEM | each math warpgroup, rows `math_wg_idx*64 ..+64` | 4 per K block |
| promotion | `kNumAccum` x (`mul.f32` + `fma.rn.f32`) | each math thread | per K block |
| TMA SFA / SFB / A / B | [`BLOCK_M`] f32 / [`BLOCK_N`] f32 / [`BLOCK_M` x 128] fp8 / [`BLOCK_N` x 128] fp8 | producer lane (multicast: rank 0 for the multicast side) | per K block |
| TMA reduce-add D | [64 x `BLOCK_N`] f32 | elected lane of warp 0 of each math warpgroup | per block |
| tensor-map rewrite | 2 maps x (address, dim0, stride0) + 2 x 128 B SMEM->GMEM | producer lane | per k-group change (k-grouped) |

## TensorMap fields (`launch.py`, from `runtime_utils.hpp`)

| map | gmem dims (inner, outer) | box (inner, outer) | swizzle | outer stride (bytes) |
| --- | --- | --- | --- | --- |
| A (base) | (K or first_k, M) | (128, BLOCK_M) | 128B | K or first_k |
| B (base) | (K or first_k, N) | (128, BLOCK_N) | 128B | K or first_k |
| A/B after rewrite (k-grouped) | (K_g, M / N) at `gmem_a + k_cumsum*M` / `gmem_b + k_cumsum*N` | unchanged | 128B | K_g |
| SFA | (align(M, 4), num_sf_k) | (BLOCK_M, 1) | none | align(M, 4) * 4 |
| SFB | (align(N, 4), num_sf_k) | (BLOCK_N, 1) | none | align(N, 4) * 4 |
| CD | (N, M * G) | (BLOCK_N, 64) | none | N * 4 |

`num_sf_k = sum ceil(K_g / 128)` (k-grouped) or `ceil(K / 128)`; all maps use
`L2_PROMOTION_256B` and `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` (out-of-bounds box elements still
load as zero, which the partial M/N tiles rely on; the TMA reduce-add clips them). k-grouped `first_k` is the first non-empty group's K; with
every group empty (`sum K = 0`) the entry returns before launching (`gemm.hpp:36`).

## Storage lifetimes

| region | written by | read by | released |
| --- | --- | --- | --- |
| SMEM tensor maps | warp M/32+1 (param copy), then producer `tensormap.replace` | producer (copy to GMEM) | in place, after `wait_group.read 0` |
| GMEM `tensor_map_buffer[2b..2b+1]` | producer `st.global.v4` + release fence | TMA loads of A/B (tensormap proxy, after acquire) | next group change, after `wait_group.read 0` |
| `smem_a/b/sfa/sfb[s]` | TMA (`full[s]` completes on bytes) | WGMMA (A/B), math `ld.shared` (SFA/SFB) | `empty[s]` after `wgmma.wait_group 0`, per warp, to every CTA that wrote it |
| `smem_d` rows of warpgroup w | `st.shared.v2.f32` of WG w | TMA reduce of WG w | next block's `wait_group.read 0` + `bar.sync 8+w` |

## Static specialization boundary

Compile time: everything in `KernelSpec` plus the compiled dims; runtime: `shape_k`
(k-grouped) or `shape_m` (normal), `grouped_layout` contents, tensor addresses. As in
DeepGEMM's JIT, a new compiled dim, group count or block config is a new instantiation.

## TIRx module and benchmark contract

`KERNEL_META` names `deepgemm_sm90_k_grouped_fp8_gemm_contiguous` and
`deepgemm_sm90_fp8_gemm_1d1d`, `runtime_cuda_archs = ["sm_90a"]`, reference requirement
DeepGEMM @ 559d79fb. CONFIGS are the upstream SM90 test shapes (k-grouped: 7 shapes plus the
empty-group variants; normal: the 7 FP32 + accumulate wgrad shapes) plus edge shapes;
`run_test` compares the whole guarded D buffer bitwise against
`deep_gemm.k_grouped_fp8_gemm_nt_contiguous` / `deep_gemm.fp8_fp4_gemm_nt(c=d,
recipe=(1,1,128))` on byte-identical operands; `run_gpu` times both with
`bench_against_deepgemm`.

## Instruction-selection summary

Placement, compile-time shapes and schedule select the instructions; the counts are evidence,
not computation. Static PTX counts per class (`source_export/opcode_hist.txt`):

| instruction | KG 128x136 mc1 | KG 128x152 mc1 | KG 128x152 mc2 | NM 128x136 mc2 | NM 128x152 mc2 | NM 64x152 s6 mc2 |
| --- | --- | --- | --- | --- | --- | --- |
| `wgmma.mma_async...m64n{BN}k32.f32.e4m3.e4m3` | 4 | 4 | 4 | 16 | 16 | 24 |
| `wgmma.fence` / `commit_group` / `wait_group 0` | 1 | 1 | 1 | 4 | 4 | 6 |
| `mul.f32` / `fma.rn.f32` (each) | 68 | 76 | 76 | 272 | 304 | 456 |
| `ld.shared.f32` (A scales) | 2 | 2 | 2 | 8 | 8 | 12 |
| `ld.shared.v2.f32` (B scales) | 17 | 19 | 19 | 68 | 76 | 114 |
| `st.shared.v2.f32` (D) | 34 | 38 | 38 | 34 | 38 | 38 |
| `ld.param.v4.u32` / `st.shared.v4.u32` (map -> SMEM) | 16 | 16 | 16 | 0 | 0 | 0 |
| `ld.shared.v4.u32` / `st.global.v4.u32` (SMEM map -> GMEM) | 16 | 16 | 16 | 0 | 0 | 0 |
| `tensormap.replace` address / dim / stride (each) | 2 | 2 | 2 | 0 | 0 | 0 |
| `fence.proxy.tensormap::generic` release / acquire | 1 / 2 | 1 / 2 | 1 / 2 | 0 | 0 | 0 |
| TMA load `...L2::cache_hint` | 28 | 28 | 14 | 16 | 16 | 24 |
| TMA load `...multicast::cluster.L2::cache_hint` | 0 | 0 | 14 | 8 | 8 | 12 |
| `cp.reduce.async.bulk.tensor.2d...add.bulk_group` | 1 | 1 | 1 | 1 | 1 | 1 |
| `mbarrier.try_wait.parity` (retry loop) | 8 | 8 | 12 | 12 | 12 | 18 |
| `mbarrier.arrive.expect_tx` | 7 | 7 | 7 | 4 | 4 | 6 |
| `mbarrier.arrive.shared::cta` / `mapa` + `arrive.shared::cluster` | 1 / 0 | 1 / 0 | 0 / 1 | 0 / 4 | 0 / 4 | 0 / 6 |
| `cp.async.bulk.commit_group` / `wait_group.read` | 2 / 2 | 2 / 2 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| `bar.sync` / `bar.warp.sync` | 3 / 3 | 3 / 3 | 2 / 3 | 2 / 2 | 2 / 2 | 2 / 2 |
| `ld.global.u32` (k-group walk) | 11 | 19 | 7 | 0 | 0 | 0 |
| `elect.sync`, `prefetch.tensormap` | 5, 5 | 5, 5 | 5, 5 | 5, 5 | 5, 5 | 5, 5 |

What the counts encode:

- **K loops.** k-grouped (`#pragma unroll 0`): the math loop stays rolled (4 WGMMAs, one
  wave); nvcc nevertheless unrolls the *producer* loop x4 with a 3-copy remainder (7 copies of
  the 4-load group: 28 = 7 x 4 TMA loads, 7 expect-tx). Normal (`#pragma unroll kNumStages`):
  both loops are unrolled xkNumStages (K is compiled), so 16/24 WGMMAs and 4/6 producer copies.
  When `SHAPE_K / 128` is not a multiple of kNumStages (64x152 s6: 56 K blocks) nvcc keeps no
  remainder copies: each unrolled loop carries one in-body exit test against the constant trip
  count (`setp.eq ..., 56` after the 2nd copy), since the loop's start index is per-block
  constant 0.
- **Multicast paths.** With `SHAPE_M` compiled (k-grouped "mn"), `num_m_blocks` is a constant
  and every block group has an even size, so `is_tma_multicast_valid` folds to true and the
  plain B/SFB path disappears (KG mc2: 14 plain A/SFA + 14 multicast B/SFB). With M at runtime
  (normal "nk"), each producer copy keeps both B/SFB paths (2 plain A/SFA + 2 plain + 2
  multicast B/SFB).
- **Promotion.** One `mul.f32 sa*sb` plus one `fma.rn.f32` per accumulator per K block:
  `final += sa * sb * accum` contracted as `fma(sa*sb, accum, final)`, no `.ftz`.
- **Constructor.** `get_next_k_group`'s first call is unrolled over `kNumGroups` (8 probes for
  8 groups, 16 for 16), hence `ld.global.u32` tracks the group count.
- **CTA sync.** `bar.sync 0` without a cluster, the cluster barrier pair with one; the epilogue's
  two named barriers are `bar.sync 8 + math_wg_idx, 128`.

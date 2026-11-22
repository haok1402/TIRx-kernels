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
deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh together with the
TMA/WGMMA/barrier helpers it instantiates. See NOTICE and licenses/ for
upstream attribution.
-->

# SM90 FP8 MQA logits (DeepSeek-V3.2 indexer): coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new IR, a
builder API, or a mathematical reference. It shows the TIRx kernel as an explicit
runtime ABI and launch; explicit linear GMEM/SMEM/register storage with byte offsets;
the persistent Q-block loop and the role control flow; primitive directional copies and
primitive computation; explicit stage/phase changes and publication/reuse edges; and
hardware instruction selection, read from the source's line-info PTX export.

The implementation represented by this sketch is maintained in
[`tirx_kernels/ported/deepgemm/_sm90_mqa_logits/`](../../../tirx_kernels/ported/deepgemm/_sm90_mqa_logits/__init__.py),
with the entry module `sm90_mqa_logits_fp8.py`. Those modules are the source of truth.

Source export: line-info PTX and SASS of all 24 instantiations the SM90 API can build
(`kNumHeads` {32, 64} x `kHeadDim` {32, 64, 128} x `kIsCompressedLogits` x logits dtype
{float, bfloat16}), built through DeepGEMM's own JIT with `DG_JIT_WITH_LINEINFO=1` by
`.porting/sm90_mqa_logits_fp8/source_export/export_source.py` (driving
`deep_gemm.fp8_fp4_mqa_logits` with upstream's data recipe, every call checked against
upstream's reference and simulated tolerances). `source_export/INDEX.txt` maps each
instantiation to its directory with its key-opcode counts. The representative cited below
is `64,128,false,float` (`kernel.sm90_fp8_mqa_logits.4f1b31d5...`); per-specialization
differences are only in the constants named below. Counts are instruction lines minus
predicated duplicates (there are none on the counted opcodes).

**In scope.** All 24 instantiations above, with the SM90 host constants
`BLOCK_Q = 128 / kNumHeads` (4 or 2), `BLOCK_KV = 256`, `kNumQStages = kNumKVStages = 3`,
`kNumTMAThreads = 128`, `kNumMathThreads = 512`, `kNumSMs` = device SM count (132 on H200).
FP8 e4m3 Q and KV, FP32 per-KV-row scales, FP32 weights, `uint32` row ranges.

**Out of scope.** The SM100 kernel and the FP4/MXFP formats (SM100-only); the
`smxx_clean_logits` kernel `fp8_fp4_mqa_logits` runs after this one when
`clean_logits=True` (a separate kernel; the port's outputs are compared on the raw kernel
output, `clean_logits=False`); the paged kernel `sm90_fp8_paged_mqa_logits` (a different
template); tile (`Tx`) primitives.

## Pipeline at a glance

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0 .. 15 (4 math warpgroups) | per Q block: per-row `[ks, ke)` loads, weights to registers; per KV block: one WGMMA chain (`m64n128k32`, A = this warpgroup's 64 KV rows, B = the whole Q block), ReLU x weight head reduction, scale, 2 butterfly shuffles, logits stores | wait `full_q[q_stage]`, `full_kv[kv_stage]`; every thread arrives `empty_kv[kv_stage]` after `wgmma.wait_group 0`, and `empty_q[q_stage]` after the Q block's last KV block |
| 16 (TMA WG warp 0) | elected lane: prefetch the four TensorMaps; elected lane: init barriers; elected lane: first Q block, then the persistent loop issuing next-Q (Q + weights) and all KV blocks (KV + scales) | wait `empty_q`/`empty_kv` (parity `phase ^ 1`); publish `full_q`/`full_kv` with expect-tx bytes |
| 17 .. 19 (TMA WG) | exit right after `setmaxnreg.dec 32` | none |

All four math warpgroups consume the same Q stage and the same KV stage: warpgroup `w`
computes logits for KV rows `[64w, 64w+64)` of the 256-row KV block, warp `wp` (0..15) owns
KV rows `[16wp, 16wp+16)`. Producer lane and every math thread compute `load_schedule`
independently from global memory and walk the same Q-block and KV-block sequence, so the
stage/phase cursors stay in lockstep without communication.

## Primitive vocabulary

```python
specialize(...)     # compile-time variant selection
launch(...)         # launch topology and attributes
tile(...)           # linear storage: dtype, element count, placement, alignment
reg_tile(...)       # role-local register array (no layout)
at(buf, byte_off)   # address of a byte offset inside a linear buffer
desc(...)           # WGMMA SMEM matrix descriptor (64-bit, fields written explicitly)

copy_g2s(tmap.coord(...), dst, completion)   # TMA load
copy_g2r(src, dst)     # plain global load into a register
copy_s2r(src, dst)     # shared load
copy_r2g(src, dst, predicate=None)   # global store

gemm(acc, a_desc, b_desc, scale_d, instr)    # one wgmma.mma_async
max(dst, a, b); mul(dst, a, b); fma(dst, a, b, c); add(dst, a, b)   # FP32, IEEE (non-FTZ)
min(dst, a, b); umax(dst, a, b)              # u32 scalars
cast(dst, src, rounding)
elect_predicate(mask); shuffle_index(dst, src, lane); shuffle_xor(dst, src, lane_mask)
```

Schedule operations: `prefetch`, `init`, `fence`, `wait`, `arrive`, `expect_bytes`,
`warp_sync`, `cta_sync`, `pdl_wait`, `setmaxnreg`, `wgmma_fence`, `wgmma_arrive`,
`wgmma_commit`, `wgmma_wait`. There are no compound ops: no `scheduler`, `reduce_heads`,
`attention`, or `logits`.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
specialize(KernelSpec(kNumHeads in {32, 64}, kHeadDim in {32, 64, 128},
                      kIsCompressedLogits, logits_dtype in {f32, bf16}, kNumSMs=132))
# instruction_selection: none; extent: the template arguments of `generate_impl`
#   (sm90_fp8_mqa_logits.hpp:42-69), with the fixed SM90 host constants below
BLOCK_Q = 128 // kNumHeads; BLOCK_KV = 256; kNumQStages = kNumKVStages = 3
kNumTMAThreads = 128; kNumMathThreads = 512
WGMMA = m64n128k32.f32.e4m3.e4m3 (SS)          # FP8MMASelector<BLOCK_Q * kNumHeads = 128>
kNumAccum = 64                                  # f32 accumulators per thread
kNumKSteps = kHeadDim // 32                     # 1, 2 or 4 WGMMAs per chain
kNumAccumPerReduce = kNumHeads // 2             # accumulators per Q row (64 / BLOCK_Q)

def kernel(seq_len: u32, seq_len_kv: u32, max_seqlen_k: u32, stride_logits: u32,
           cu_seq_len_k_start: u32*, cu_seq_len_k_end: u32*, logits: logits_dtype*,
           tmap_q, tmap_kv, tmap_kv_scales, tmap_weights: __grid_constant__ CUtensorMap):
    launch(grid=(kNumSMs,1,1), block=(640,1,1), cluster=None,
           dynamic_smem=smem_size, launch_bounds=(640, 1))
    # instruction_selection: .maxntid 640,1,1 + .minnctapersm 1 (PTX header); dynamic smem is
    #   a launch attribute; DeepGEMM adds the PDL attribute only after `deep_gemm.set_pdl(True)`
    #   (off by default: csrc/jit/device_runtime.hpp:16, handle.hpp:101-105); the kernel
    #   always executes griddepcontrol.wait, a no-op without the attribute
    num_q_blocks = ceil_div(seq_len, BLOCK_Q)         # u32

    # ---- prologue (`.cuh:50-56`) ---------------------------------------------
    if thread_idx // 32 == 16 and elect_predicate(0xffffffff):
        # instruction_selection: elect.sync; extent: one lane of warp 16
        for t in (tmap_q, tmap_kv, tmap_kv_scales, tmap_weights):
            prefetch(t)
            # instruction_selection: prefetch.tensormap [param]; extent: 4 instructions
    warp_sync()
    # instruction_selection: bar.warp.sync -1; extent: every warp

    # ---- storage: one dynamic linear pool, aligned to kSwizzleAlignment = 8*kHeadDim ----
    smem = tile(u8, smem_size, space=shared.dyn, align=8 * kHeadDim)
    Q_ST  = BLOCK_Q * kNumHeads * kHeadDim          # fp8 bytes per Q stage (= 128 * kHeadDim)
    KV_ST = BLOCK_KV * kHeadDim                     # fp8 bytes per KV stage
    W_ST  = BLOCK_Q * kNumHeads * 4                 # = 512
    SF_ST = BLOCK_KV * 4                            # = 1024
    off_q(s)        = s * Q_ST                      # [128 rows x kHeadDim B], TMA-swizzled kHeadDim B
    off_kv(s)       = 3 * Q_ST + s * KV_ST          # [256 rows x kHeadDim B], same swizzle
    off_w(s)        = 3 * (Q_ST + KV_ST) + s * W_ST # f32[BLOCK_Q][kNumHeads], linear
    off_sf(s)       = 3 * (Q_ST + KV_ST + W_ST) + s * SF_ST   # f32[256], linear
    BAR             = off_sf(3)                     # mbarriers follow the scales
    off_full_q(s)   = BAR + 8 * s
    off_empty_q(s)  = BAR + 8 * (3 + s)
    off_full_kv(s)  = BAR + 8 * (6 + s)
    off_empty_kv(s) = BAR + 8 * (9 + s)
    # e.g. H64 D128: off_kv(0) = 49152, off_w(0) = 147456, off_sf(0) = 148992,
    #   BAR = 152064 (PTX: barrier addresses smem_buffer + 152064 .. +152152)
    # Host smem_size (`.hpp:113-123`) = BAR + 12*8 + 4*2*8 (unused tail) + 4.

    # ---- barrier init (`.cuh:97-114`) -----------------------------------------
    is_tma_load_warp = 512 <= thread_idx < 544
    if is_tma_load_warp and elect_predicate(0xffffffff):
        # instruction_selection: elect.sync; extent: one lane of warp 16
        for s in static_range(3):
            init(full_q[s] @ at(smem, off_full_q(s)), arrive_count=1)
            init(empty_q[s] @ at(smem, off_empty_q(s)), arrive_count=512)
        for s in static_range(3):
            init(full_kv[s] @ at(smem, off_full_kv(s)), arrive_count=1)
            init(empty_kv[s] @ at(smem, off_empty_kv(s)), arrive_count=512)
        # instruction_selection: mbarrier.init.shared::cta.b64; extent: 12 instructions
        fence(mbarrier_init, scope=cluster)
        # instruction_selection: fence.mbarrier_init.release.cluster; extent: one
    cta_sync()
    # instruction_selection: bar.sync 0; extent: whole CTA

    # ---- schedule state (`.cuh:120-154`) ----------------------------------------
    block_q_idx = block_idx_x; q_iter_idx = 0            # u32, both roles
    num_total_kv_blocks = 0
    seq_k_start = reg_tile(u32, BLOCK_Q); seq_k_end = reg_tile(u32, BLOCK_Q)

    def load_schedule(q_iter_offset):                    # `.cuh:127-144`, fully unrolled
        start, end = 0xffffffff, 0
        for i in static_range(BLOCK_Q):
            q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1)
            copy_g2r(cu_seq_len_k_start[q_idx], seq_k_start[i])
            copy_g2r(cu_seq_len_k_end[q_idx], seq_k_end[i])
            # instruction_selection: ld.global.u32 x2 (no .nc, 64-bit address from
            #   mul.wide.u32 q_idx, 4); extent: 2 * BLOCK_Q per call
            start = min(start, min(seq_k_start[i], seq_len_kv))
            end = umax(end, min(seq_k_end[i], seq_len_kv))
            # instruction_selection: min.u32 / max.u32; extent: scalars
        start = start & ~3                               # TMA alignment of the scale load
        # instruction_selection: and.b32 start, -4
        it = q_iter_idx + q_iter_offset
        return (it % 3, (it // 3) & 1, start, (end - start + 255) >> 8)
        # instruction_selection: mul.wide.u32 by 0xAAAAAAAB + shr 33 (division by 3),
        #   (255 - start + end) >> 8 for ceil_div(end - start, 256); extent: scalars

    def kv_pipeline(kv_block_idx):                       # `.cuh:146-154`
        it = num_total_kv_blocks + kv_block_idx
        return (it % 3, (it // 3) & 1)                  # same division-by-3 selection

    pdl_wait()
    # instruction_selection: griddepcontrol.wait; extent: one
```

### Role 1: TMA producer (warp 16; warps 17-19 exit)

```python
    if thread_idx >= 512:
        setmaxnreg(dec, 32)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 32; extent: TMA warpgroup
        if not is_tma_load_warp: return
        # instruction_selection: setp.gt.u32 tid, 543 + bra to the exit

        def issue_tma_q(stage, q_block):                 # `.cuh:167-171`
            copy_g2s(tmap_q.coord(0, q_block * BLOCK_Q * kNumHeads),
                     at(smem, off_q(stage)), completion=full_q[stage], cache_hint=EVICT_NORMAL)
            copy_g2s(tmap_weights.coord(0, q_block * BLOCK_Q),
                     at(smem, off_w(stage)), completion=full_q[stage], cache_hint=EVICT_NORMAL)
            # instruction_selection: cp.async.bulk.tensor.2d.shared::cluster.global
            #   .mbarrier::complete_tx::bytes.L2::cache_hint (hint 0x1000000000000000);
            #   extent: one box each (Q: [128 rows x kHeadDim B], one swizzle atom wide;
            #   weights: [BLOCK_Q x kNumHeads] f32, unswizzled)
            expect_bytes(full_q[stage], Q_ST + W_ST)
            # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64 (immediate,
            #   16896 for H64 D128); extent: one

        if elect_predicate(0xffffffff) and block_q_idx < num_q_blocks:
            # instruction_selection: elect.sync (first of two in this role); extent: one lane
            issue_tma_q(0, block_q_idx)

        if elect_predicate(0xffffffff):
            # instruction_selection: elect.sync (second); the whole loop runs on that lane
            while block_q_idx < num_q_blocks:
                q_stage, q_phase, kv_start, num_kv_blocks = load_schedule(1)
                wait(empty_q[q_stage], parity=q_phase ^ 1)
                # instruction_selection: LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64
                #   P1, [bar], parity, 0x989680 (suspend-time hint); @P1 bra DONE;
                #   bra LAB_WAIT (cutlass/arch/barrier.h:412-424); extent: one Q stage
                if block_q_idx + kNumSMs < num_q_blocks:
                    issue_tma_q(q_stage, block_q_idx + kNumSMs)   # selection as above

                for kv_block_idx in runtime_range(num_kv_blocks, unroll=4):
                    # nvcc unrolls the `#pragma unroll` runtime loop by 4 with a remainder of
                    # up to 3 copies after it: 7 KV issue sites in the PTX (try_wait at
                    # 4 body + 3 remainder sites); extent: num_kv_blocks iterations
                    kv_stage, kv_phase = kv_pipeline(kv_block_idx)
                    wait(empty_kv[kv_stage], parity=kv_phase ^ 1)
                    # instruction_selection: the same try_wait.parity + 0x989680 retry loop
                    kv_row = kv_start + kv_block_idx * BLOCK_KV
                    copy_g2s(tmap_kv.coord(0, kv_row), at(smem, off_kv(kv_stage)),
                             completion=full_kv[kv_stage], cache_hint=EVICT_NORMAL)
                    copy_g2s(tmap_kv_scales.coord(kv_row, 0), at(smem, off_sf(kv_stage)),
                             completion=full_kv[kv_stage], cache_hint=EVICT_NORMAL)
                    # instruction_selection: cp.async.bulk.tensor.2d...L2::cache_hint x2;
                    #   extent: one [256 x kHeadDim B] box + one [256] f32 box
                    expect_bytes(full_kv[kv_stage], KV_ST + SF_ST)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64
                    #   (immediate, 33792 for H64 D128)
                num_total_kv_blocks += num_kv_blocks
                block_q_idx += kNumSMs; q_iter_idx += 1
        # no drain: the producer simply ends
```

### Role 2: math warpgroups (warps 0 .. 15)

```python
    else:
        setmaxnreg(inc, 112)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32 112; extent: math warpgroups
        warp_idx = shuffle_index(thread_idx // 32, src_lane=0)
        # instruction_selection: shfl.sync.idx.b32 (the only idx shuffle); extent: scalar
        warpgroup_idx = warp_idx // 4
        lane_idx = lane_id()
        # instruction_selection: mov.u32 %laneid; extent: scalar
        accum = reg_tile(f32, 64)
        weights = reg_tile(f32, BLOCK_Q * kNumHeads // 4)     # weights[i][j], j < kNumHeads/4
        warp_offset = warp_idx * 16
        v_0_offset = lane_idx // 4; v_1_offset = lane_idx // 4 + 8
        # 64-bit GMMA descriptor fields, written explicitly (`make_smem_desc(ptr,
        #   to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8)`):
        #   bits [0,14) start address >> 4      bits [16,30) LBO >> 4 = 0
        #   bits [32,46) SBO >> 4 = 8*kHeadDim >> 4    bits [49,52) base offset = 0
        #   bits [62,64) layout type: B128 = 1 (D128), B64 = 2 (D64), B32 = 3 (D32)
        #   K-major, 8-row groups 8*kHeadDim bytes apart; k-step k advances the start by 32 B

        while block_q_idx < num_q_blocks:
            q_stage, q_phase, kv_start, num_kv_blocks = load_schedule(0)
            # (the 2*BLOCK_Q ld.global.u32 above, issued by every math thread)
            wait(full_q[q_stage], parity=q_phase)
            # instruction_selection: try_wait.parity + 0x989680 retry loop; extent: one stage
            for i in static_range(BLOCK_Q):
                for j in static_range(kNumHeads // 4):
                    copy_s2r(at(smem, off_w(q_stage) + 4 * (i * kNumHeads + (j // 2) * 8
                                                        + (j & 1) + (lane_idx % 4) * 2)),
                             weights[i][j])
                    # instruction_selection: ld.shared.f32 (inline asm, u64 cvta address);
                    #   extent: BLOCK_Q * kNumHeads / 4 = 32 scalars

            for kv_block_idx in runtime_range(num_kv_blocks):
                # `#pragma unroll` on a runtime trip count; nvcc emits ONE copy of this body
                # (1 wgmma.fence / commit / wait in the PTX): not unrolled
                kv_stage, kv_phase = kv_pipeline(kv_block_idx)
                wait(full_kv[kv_stage], parity=kv_phase)
                # instruction_selection: try_wait.parity + 0x989680 retry loop
                scale_kv_0 = copy_s2r(at(smem, off_sf(kv_stage) + 4 * (warp_offset + v_0_offset)))
                scale_kv_1 = copy_s2r(at(smem, off_sf(kv_stage) + 4 * (warp_offset + v_1_offset)))
                # instruction_selection: ld.shared.f32 x2 (after the full_kv wait)
                wgmma_fence_operand(accum)          # register ordering only, no instruction
                wgmma_arrive()
                # instruction_selection: wgmma.fence.sync.aligned; extent: warpgroup
                for k in static_range(kNumKSteps):
                    gemm(accum,
                         desc(start=off_kv(kv_stage) + warpgroup_idx * 64 * kHeadDim + 32 * k),
                         desc(start=off_q(q_stage) + 32 * k),
                         scale_d=(k != 0), instr=WGMMA)
                    # instruction_selection: wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3
                    #   .e4m3 {64 x f32}, descA, descB, p (scale-d = k != 0), 1, 1;
                    #   extent: kNumKSteps instructions (1 / 2 / 4)
                wgmma_commit()
                # instruction_selection: wgmma.commit_group.sync.aligned
                wgmma_fence_operand(accum)
                wgmma_wait(0)
                # instruction_selection: wgmma.wait_group.sync.aligned 0
                arrive(empty_kv[kv_stage])
                # instruction_selection: mbarrier.arrive.shared::cta.b64 _, [bar]; extent:
                #   every math thread (arrive count 512)

                kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset      # u32
                for i in static_range(BLOCK_Q):
                    a = accum[i * kNumAccumPerReduce :]           # this row's accumulators
                    # Accumulator a[J] (J < kNumHeads/2) belongs to head 8*(J/4) + 2*(lane%4) +
                    # (J&1) and KV row v_0 (J%4 < 2) or v_1 (J%4 >= 2); weight index
                    # w(J) = (J/4)*2 + (J&1). nvcc contracts the source's
                    # `sum[k] = t(k); sum[k] += t(4+k); sum[k] += t(4j+k) ...` (t(J) = fmaxf(a[J],0)
                    # * weights[i][w(J)]) into exactly this chain, per k in 0..3:
                    for k in static_range(4):
                        r0 = max(a[k], 0.0);  r1 = max(a[4 + k], 0.0)
                        # instruction_selection: max.f32 (non-FTZ, zero register); extent: scalar
                        s_k = mul(r1, weights[i][2 + (k & 1)])
                        # instruction_selection: mul.f32 (non-FTZ) -- the j=1 product, rounded
                        s_k = fma(r0, weights[i][k & 1], s_k)
                        # instruction_selection: fma.rn.f32 (non-FTZ) -- the j=0 product fused
                        for j in static_range(2, kNumHeads // 8):
                            r = max(a[4 * j + k], 0.0)
                            s_k = fma(r, weights[i][2 * j + (k & 1)], s_k)
                            # instruction_selection: max.f32 + fma.rn.f32; extent: scalar
                    # (the PTX interleaves the four k-chains; each chain's order is as above)
                    t_0 = add(s_0, s_1)
                    # instruction_selection: add.f32 (non-FTZ); extent: scalar
                    v_0 = mul(scale_kv_0, t_0)
                    # instruction_selection: mul.f32 (non-FTZ); extent: scalar
                    t_1 = add(s_2, s_3)
                    # instruction_selection: add.f32 (non-FTZ); extent: scalar
                    v_1 = mul(scale_kv_1, t_1)
                    # instruction_selection: mul.f32 (non-FTZ); extent: scalar
                    for m in (1, 2):                                  # static, source order
                        o_0 = shuffle_xor(v_0, m)
                        # instruction_selection: shfl.sync.bfly.b32 (lane mask m, clamp 31,
                        #   member mask -1); extent: scalar
                        v_0 = add(v_0, o_0)
                        # instruction_selection: add.f32 (non-FTZ); extent: scalar
                        o_1 = shuffle_xor(v_1, m)
                        # instruction_selection: shfl.sync.bfly.b32 (lane mask m, clamp 31,
                        #   member mask -1); extent: scalar
                        v_1 = add(v_1, o_1)
                        # instruction_selection: add.f32 (non-FTZ); extent: scalar
                    q_offset = u64(block_q_idx * BLOCK_Q + i) * u64(stride_logits)
                    # instruction_selection: mul.wide.u32; extent: scalar (64-bit offsets)
                    for (v, vo) in ((v_0, v_0_offset), (v_1, v_1_offset)):
                        if kIsCompressedLogits:
                            pred = seq_k_start[i] <= kv_offset + vo and kv_offset + vo < seq_k_end[i]
                            # instruction_selection: setp.gt.u32 / setp.ge.u32 + or.pred +
                            #   @p bra over the conversion and the store
                            if pred:
                                vc = cast(v, logits_dtype, rn)
                                # instruction_selection: cvt.rn.bf16.f32 (bf16); none (f32);
                                #   extent: scalar
                                copy_r2g(vc, logits[q_offset + kv_offset + vo - seq_k_start[i]])
                                # instruction_selection: st.global.f32 (f32) / st.global.u16
                                #   (bf16); extent: scalar
                        else:
                            vc = cast(v, logits_dtype, rn)
                            # instruction_selection: cvt.rn.bf16.f32 (bf16); none (f32);
                            #   extent: scalar
                            copy_r2g(vc, logits[q_offset + kv_offset + vo])
                            # instruction_selection: st.global.f32 (f32) / st.global.u16 (bf16);
                            #   extent: scalar (2 stores per row per thread, 2*BLOCK_Q)
            num_total_kv_blocks += num_kv_blocks
            arrive(empty_q[q_stage])
            # instruction_selection: mbarrier.arrive.shared::cta.b64; extent: every math
            #   thread (arrive count 512)
            block_q_idx += kNumSMs; q_iter_idx += 1
```

## Logical GEMM shapes and owners

| Op | Shape | Owner | Per |
| --- | --- | --- | --- |
| WGMMA | m64 n128 k32, FP8 x FP8 -> FP32, A = KV rows (SMEM, K-major), B = Q rows `[BLOCK_Q*kNumHeads]` (SMEM, K-major) | math warpgroup `w`, KV rows `64w ..+64` | `kHeadDim/32` per KV block |
| head reduction | `kNumHeads/2` accumulators -> 2 values per Q row per thread, then 2 butterfly steps over `lane % 4` | each math thread | per KV block per Q row |
| TMA Q / weights | [128 x kHeadDim] fp8 / [BLOCK_Q x kNumHeads] f32 | producer lane | per Q block |
| TMA KV / scales | [256 x kHeadDim] fp8 / [256] f32 | producer lane | per KV block |
| logits store | 2 elements per thread per Q row, rows `block_q_idx*BLOCK_Q + i`, cols `kv_offset + {v_0, v_1}` | each math thread | per KV block |

## TensorMap fields (`launch.py`, from `sm90_fp8_mqa_logits.hpp:102-111` / `make_tma_2d_desc`)

| map | gmem dims (inner, outer) | box (inner, outer) | swizzle | outer stride (elements) |
| --- | --- | --- | --- | --- |
| Q | (kHeadDim, seq_len * kNumHeads) | (kHeadDim, BLOCK_Q * kNumHeads) | kHeadDim B | kHeadDim |
| KV | (kHeadDim, seq_len_kv) | (kHeadDim, 256) | kHeadDim B | kHeadDim |
| KV scales | (align(seq_len_kv, 4), 1) | (256, 1) | none | 0 |
| weights | (kNumHeads, seq_len) | (kNumHeads, BLOCK_Q) | none | weights.stride(0) |

All use `L2_PROMOTION_256B` and no OOB fill (out-of-range rows read as zeros; the expect-tx
count is always the full box).

## Storage lifetimes

| region | written by | read by | released |
| --- | --- | --- | --- |
| `smem_q[s]`, `smem_w[s]` | TMA (`full_q[s]` completes on bytes) | WGMMA B (all KV blocks of the Q block); weights `ld.shared` once per Q block | `empty_q[s]` by all 512 math threads after the Q block's last KV block |
| `smem_kv[s]`, `smem_sf[s]` | TMA (`full_kv[s]`) | WGMMA A (rows of each warpgroup); scale `ld.shared` before the WGMMA | `empty_kv[s]` by all 512 math threads after `wgmma.wait_group 0` |
| `weights`, `seq_k_start/end` registers | per Q block | every KV block of the Q block | next Q block |

## Static specialization boundary

Compile time: `KernelSpec` (heads, head dim, compressed, logits dtype, SM count); runtime:
`seq_len`, `seq_len_kv`, `max_seqlen_k`, `stride_logits`, the ranges and tensor addresses.
One instantiation per `KernelSpec`, exactly as DeepGEMM's JIT keys it (its key also contains
the fixed block/stage/thread constants).

## TIRx module and benchmark contract

`KERNEL_META` names `deepgemm_sm90_fp8_mqa_logits`, `runtime_cuda_archs = ["sm_90a"]`,
reference requirement DeepGEMM @ 559d79fb. `run_test` runs both kernels into identical
sentinel-filled logits storage (DeepGEMM with `clean_logits=False`, its block reused by the
caching allocator, checked by address) and requires the whole storage, written or not, to be
bitwise identical; then upstream's reference/simulated tolerances on the TIRx output.
`run_gpu` times both.

## Instruction-selection summary

Placement and schedule select the instructions; the counts are evidence, not computation.
Per-specialization counts are in `source_export/INDEX.txt`; for `64,128,false,float`:

| instruction | static count | cause |
| --- | --- | --- |
| `wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3` | 4 | `kHeadDim/32` k-steps, KV loop not unrolled (1 / 2 / 4 for D 32 / 64 / 128) |
| `wgmma.fence` / `commit_group` / `wait_group 0` | 1 each | one chain per KV block |
| `cp.async.bulk.tensor.2d...L2::cache_hint` | 18 | Q + weights at 2 sites (first block, next block) = 4; KV + scales at 7 sites (unroll 4 + remainder 3) = 14 |
| `mbarrier.arrive.expect_tx.shared::cta.b64` | 9 | 2 Q sites + 7 KV sites |
| `mbarrier.try_wait.parity.shared::cta.b64` | 10 | producer: empty_q 1 + empty_kv 7; math: full_q 1 + full_kv 1 |
| `mbarrier.arrive.shared::cta.b64` | 2 | math: empty_kv, empty_q |
| `max.f32` | 64 | ReLU of the 64 accumulators |
| `mul.f32` | 8 + 4 | first product of each k-chain (4 per row); `* scale_kv` (2 per row) -- H32: 16 + 8 |
| `fma.rn.f32` | 56 | `(kNumHeads/8 - 1) * 4` per row -- H32: 48 |
| `add.f32` | 4 + 8 | `s_0+s_1`, `s_2+s_3` (2 per row); butterfly adds (4 per row) -- H32: 8 + 16 |
| `shfl.sync.bfly.b32` / `shfl.sync.idx.b32` | 8 / 1 | 4 per row / the warp-index broadcast -- H32: 16 / 1 |
| `ld.shared.f32` | 34 | 32 weights + 2 KV scales |
| `ld.global.u32` | 8 | `2 * BLOCK_Q` range loads in each role -- H32: 16 |
| `st.global.f32` (or `cvt.rn.bf16.f32` + `st.global.u16`) | 4 | `2 * BLOCK_Q` -- H32: 8 |
| `elect.sync` | 4 | prefetch, barrier init, first-Q, producer loop |
| `setmaxnreg.dec 32` / `.inc 112` | 1 / 1 | role register split |
| `griddepcontrol.wait`, `fence.mbarrier_init.release.cluster`, `bar.sync 0`, `bar.warp.sync -1`, `prefetch.tensormap` x4, `mbarrier.init` x12 | -- | prologue |
| FP32 ops carry no `.ftz`: DeepGEMM's JIT builds without `--use_fast_math` | | the port must emit the same non-FTZ forms |

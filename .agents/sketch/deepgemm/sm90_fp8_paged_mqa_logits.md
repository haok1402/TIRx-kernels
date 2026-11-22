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
deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh together with the
scheduler and TMA/WGMMA/barrier helpers it instantiates. See NOTICE and licenses/ for
upstream attribution.
-->

# SM90 FP8 paged MQA logits (DeepSeek-V3.2 decode indexer): coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new IR, a
builder API, or a mathematical reference. It shows the TIRx kernel as an explicit
runtime ABI and launch; explicit linear GMEM/SMEM/register storage with byte offsets;
the persistent task walk and the role control flow; primitive directional copies and
primitive computation; explicit stage/phase changes and publication/reuse edges; and
hardware instruction selection, read from the source's line-info PTX export.

The implementation represented by this sketch is maintained in
[`tirx_kernels/ported/deepgemm/_sm90_paged_mqa_logits/`](../../../tirx_kernels/ported/deepgemm/_sm90_paged_mqa_logits/__init__.py),
with the entry module `sm90_paged_mqa_logits_fp8.py`. Those modules are the source of truth.

Source export: line-info PTX and SASS of all 24 instantiations the SM90 API can build
(`kNextN` {1, 2} x `kNumHeads` {32, 64} x `kHeadDim` {32, 64, 128} x logits {float,
bfloat16}), built through DeepGEMM's own JIT with `DG_JIT_WITH_LINEINFO=1` by
`.porting/sm90_paged_mqa_logits_fp8/source_export/export_paged.py` (driving
`deep_gemm.fp8_fp4_paged_mqa_logits` with DeepGEMM's own schedule metadata; every output
checked against the reference). `source_export/INDEX.txt` lists each instantiation's
directory and key-opcode counts. The representative cited below is `2,64,128,float`
(`kernel.sm90_fp8_paged_mqa_logits.d84628e9...`). Counts are instruction lines minus
predicated duplicates.

**In scope.** The 24 instantiations above with the SM90 host constants `BLOCK_KV = 64`,
`SPLIT_KV = 256` (so `kNumMathWarpGroups = 4`, `kNumBlocksPerSplit = 4`), 3 Q and 3 KV stages,
`kNumTMAThreads = 128`, `kNumMathThreads = 512`, `kIsContextLens2D = true`, `kIsVarlen = false`,
`kNumNextNAtoms = 1`. The SM count is not a source template argument: the source launches
`get_num_sms()` CTAs at run time and reads `schedule_meta[sm]` / `[sm + 1]`. The port fixes it
per compiled specialization (`KernelSpec.num_sms`, the device SM count, 132 on H200), which must
equal the `num_sms` DeepGEMM's metadata kernel was run with (`schedule_meta` has `num_sms + 1`
rows). FP8 e4m3 Q and KV, FP32 per-token KV scales
fused into the paged cache, FP32 weights.

**Out of scope.** The metadata kernel `sm90_paged_mqa_logits_metadata` (DeepGEMM's own kernel
computes `schedule_meta`, as for the SM100 port); varlen (`kIsVarlen`, SM100-only on the API);
1D context lens and `clean_logits` (rejected by the API for 2D lens); `next_n` > 2 (rejected
by the SM90 API); FP4/MXFP formats; tile (`Tx`) primitives.

## Pipeline at a glance

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0 .. 15 (math warpgroup g = 0..3) | per task (q, kv split): on a new q, release the previous Q stage and read weights; one WGMMA chain (`m64n{kNextN*H}k32`, A = this group's 64-token KV block, B = the q's Q rows); head reduction, scale, 2 butterflies, store `kNextN` rows x 2 values | wait `full_q[q_stage]` (shared), `full_kv_g[kv_stage]` (own ring); arrive `empty_kv_g` (count 128) after `wgmma.wait_group 0`; arrive `empty_q` (count 512) when the q changes |
| 16 + g (TMA warp g, all 32 lanes) | per task: Q prefetch (issued by group 0 only, waited by every group); block-table window refresh; one KV block (3D TMA) + its scales into ring g | wait `empty_q` / `empty_kv_g` (parity `phase ^ 1`); publish `full_q` (group 0) / `full_kv_g` with expect-tx |

Every role builds its own `SM90PagedMQALogitsScheduler` from `schedule_meta[sm]` and
`schedule_meta[sm + 1]` and walks the identical task sequence, so all stage/phase cursors stay
in lockstep. A task is `(q_idx, kv_idx, num_kv)`; `kv_idx` steps by 4 blocks, group g handles
block `kv_idx + g`.

## Primitive vocabulary

```python
specialize(...); launch(...); tile(...); reg_tile(...); at(buf, byte_off); desc(...)

copy_g2s(tmap.coord(...), dst, completion)    # TMA load (2D or 3D)
copy_g2r(src, dst)                            # plain global load into registers (scalar or v2)
copy_s2r(src, dst)                            # shared load
copy_r2g(src, dst)                            # global store

gemm(acc, a_desc, b_desc, scale_d, instr)     # one wgmma.mma_async
max(dst, a, b); mul(dst, a, b); fma(dst, a, b, c); add(dst, a, b)   # FP32 IEEE (non-FTZ)
min/umax/add/ceil_div on u32 scalars; cast(dst, src, rounding)
elect_predicate(mask); shuffle_index(dst, src, lane); shuffle_xor(dst, src, lane_mask)
```

Schedule operations: `prefetch`, `init`, `fence`, `wait`, `arrive`, `expect_bytes`,
`warp_sync`, `cta_sync`, `pdl_wait`, `setmaxnreg`, `wgmma_fence`, `wgmma_arrive`,
`wgmma_commit`, `wgmma_wait`. No compound ops (`scheduler`, `reduce_heads`, `logits`).

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
specialize(KernelSpec(kNextN in {1, 2}, kNumHeads in {32, 64}, kHeadDim in {32, 64, 128},
                      logits_dtype in {f32, bf16}, num_sms=132 (port-side constant, see above)))
# instruction_selection: none; extent: the template arguments of `generate_impl`
#   (sm90_fp8_mqa_logits.hpp:268-285) with BLOCK_KV=64, kIsContextLens2D=true,
#   kIsVarlen=false, 3/3 stages, SPLIT_KV=256, 128/512 threads
kNumMathWarpGroups = 4; N = kNextN * kNumHeads            # 32, 64 or 128
WGMMA = m64{N}k32.f32.e4m3.e4m3 (SS); kNumAccum = N / 2
kNumKSteps = kHeadDim / 32; kNumAccumPerReduce = kNumHeads / 2

def kernel(batch_size: u32, logits_stride: u32, block_table_stride: u32,
           context_lens: u32*, logits: logits_dtype*, block_table: u32*, indices: u32* (unused),
           schedule_meta: u32*, tmap_q, tmap_kv, tmap_kv_scales, tmap_weights: CUtensorMap):
    launch(grid=(kNumSMs,1,1), block=(640,1,1), dynamic_smem=smem_size, launch_bounds=(640, 1))
    # instruction_selection: .maxntid 640,1,1 + .minnctapersm 1; PDL attribute only with
    #   deep_gemm.set_pdl(True)

    warp_idx = shuffle_index(thread_idx // 32, src_lane=0)
    # instruction_selection: shfl.sync.idx.b32 (`.cuh:46`)
    warpgroup_idx = warp_idx // 4
    lane_idx = lane_id()
    # instruction_selection: mov.u32 %laneid

    if warp_idx == 16 and elect_predicate(0xffffffff):
        # instruction_selection: elect.sync
        for t in (tmap_q, tmap_kv, tmap_kv_scales, tmap_weights):
            prefetch(t)
            # instruction_selection: prefetch.tensormap [param]; extent: 4
    warp_sync()
    # instruction_selection: bar.warp.sync -1

    # ---- storage: one dynamic pool aligned to SW = 8 * kHeadDim (`.cuh:61-104`) ----
    smem = tile(u8, smem_size, space=shared.dyn, align=SW)
    Q_ST  = kNextN * kNumHeads * kHeadDim; W_ST = align(kNextN * kNumHeads * 4, SW)
    Q_PIPE = 3 * (Q_ST + W_ST) + align(3 * 8 * 2, SW)
    KV_ST = 64 * kHeadDim; SF_ST = align(64 * 4, SW)
    KV_PIPE = 3 * (KV_ST + SF_ST) + align(3 * 8 * 2, SW)
    off_q(s)        = s * Q_ST
    off_w(s)        = 3 * Q_ST + s * W_ST
    off_full_q(s)   = 3 * (Q_ST + W_ST) + 8 * s;  off_empty_q(s) = off_full_q(3) + 8 * s
    kv_group_idx = shuffle_index(warp_idx_from_tid - 16 if thread_idx >= 512 else warpgroup_idx, 0)
    # instruction_selection: shfl.sync.idx.b32 (`.cuh:94`); TMA warp g and math warpgroup g
    #   share group g
    G = Q_PIPE + KV_PIPE * kv_group_idx              # this group's KV region
    off_kv(s)       = G + s * KV_ST                  # [64 tokens x kHeadDim B], TMA-swizzled
    off_sf(s)       = G + 3 * KV_ST + s * SF_ST      # f32[64]
    off_full_kv(s)  = G + 3 * (KV_ST + SF_ST) + 8 * s;  off_empty_kv(s) = off_full_kv(3) + 8 * s
    # e.g. 2,64,128: off_w(0) = 49152, Q barriers at 52224, group 0 KV at 53248, its scales at
    #   77824, its barriers at 80896; KV_PIPE = 28672 (PTX smem_buffer offsets)
    # Host smem_size = Q_PIPE + 4 * KV_PIPE + 4*2*8 + 4 (`.hpp:353-362`).

    # ---- barrier init (`.cuh:108-128`) ----
    if warp_idx >= 16 and elect_predicate(0xffffffff):
        # instruction_selection: elect.sync (every TMA warp, its own elected lane)
        if kv_group_idx == 0:
            for s in static_range(3):
                init(full_q[s] @ at(smem, off_full_q(s)), arrive_count=1)
                init(empty_q[s] @ at(smem, off_empty_q(s)), arrive_count=512)
        if kv_group_idx < 4:
            for s in static_range(3):
                init(full_kv_g[s] @ at(smem, off_full_kv(s)), arrive_count=1)
                init(empty_kv_g[s] @ at(smem, off_empty_kv(s)), arrive_count=128)
        # instruction_selection: mbarrier.init.shared::cta.b64; extent: 6 (+6 for group 0)
        fence(mbarrier_init, scope=cluster)
        # instruction_selection: fence.mbarrier_init.release.cluster
    cta_sync()
    # instruction_selection: bar.sync 0

    pdl_wait()
    # instruction_selection: griddepcontrol.wait
```

### In-kernel scheduler (`scheduler/sm90_paged_mqa_logits.cuh:138-246`, kNumBlocksPerSplit = 4)

```python
    def refresh(q):                                  # `refresh_num_kv_and_advance` (sched:179-192)
        ctx_len = reg_tile(u32, 1)
        copy_g2r(context_lens[q * kNextN + kNextN - 1], ctx_len)
        # instruction_selection: ld.global.u32 (64-bit address); extent: scalar
        cur_num_kv = ceil_div(ctx_len, 64)          # scalar bookkeeping (add 63, shr 6)

    # Constructor (sched:194-210): emitted once, before the role branch; every thread (640)
    #   executes it.
    (cur_q, cur_kv) = copy_g2r(schedule_meta[2*sm : 2*sm+2]);  cur_kv *= 4
    (end_q, end_kv) = copy_g2r(schedule_meta[2*sm+2 : 2*sm+4]);  end_kv *= 4
    # instruction_selection: ld.global.v2.u32 x2 (uint2 reads); extent: one site before the role
    #   branch, every thread
    refresh(cur_q)

    def fetch_next_task():  # -> (valid, q_idx, kv_idx, num_kv)
        out = (cur_q, cur_kv, cur_num_kv)
        if cur_q == end_q and cur_kv == end_kv: return False, out
        cur_kv += 4
        if cur_kv >= cur_num_kv:
            cur_kv = 0; cur_q += 1
            if exist(cur_q): refresh(cur_q)
            # (the refresh load sits under the exist predicate: sched:236-238, 190)
        return True, out
    def exist(q): return q < end_q or (q == end_q and 0 < end_kv)

    def q_pipeline(it): return it % 3, (it // 3) & 1       # division by 3 via mul.wide magic
    def kv_pipeline(it): return it % 3, (it // 3) & 1
    q_iter_idx = 0; kv_iter_idx = 0
```

### Role 1: TMA warps (warps 16..19, all lanes; `.cuh:151-215`)

```python
    if warp_idx >= 16:
        setmaxnreg(dec, 64)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 64
        if kv_group_idx >= 4: return            # never taken: 4 TMA warps, 4 groups
        # instruction_selection: setp.gt.u32 group, 3 + bra

        def issue_tma_q(stage, q):              # `.cuh:157-163`
            if kv_group_idx == 0 and elect_predicate(0xffffffff):
                # instruction_selection: elect.sync
                copy_g2s(tmap_q.coord(0, q * kNextN * kNumHeads), at(smem, off_q(stage)),
                         completion=full_q[stage], cache_hint=EVICT_NORMAL)
                copy_g2s(tmap_weights.coord(0, q * kNextN), at(smem, off_w(stage)),
                         completion=full_q[stage], cache_hint=EVICT_NORMAL)
                # instruction_selection: cp.async.bulk.tensor.2d.shared::cluster.global
                #   .mbarrier::complete_tx::bytes.L2::cache_hint x2; extent: one box each
                expect_bytes(full_q[stage], Q_ST + kNextN * kNumHeads * 4)
                # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64
                #   (16896 for 2,64,128)

        q_idx = batch_size                       # "none"
        fetched, (next_q, next_kv, next_num_kv) = fetch_next_task()
        if fetched:
            issue_tma_q(0, next_q); q_iter_idx = 1      # first Q (`.cuh:171-172`)
        kv_block_idx_ptr = 32; kv_block_idx_storage = reg_tile(u32, 1)

        while fetched:
            prefetch_q = q_idx != next_q and exist(next_q + 1)
            q_idx, kv_idx, num_kv = next_q, next_kv, next_num_kv
            if prefetch_q:                       # every group waits, group 0 issues
                q_stage, q_phase = q_pipeline(q_iter_idx); q_iter_idx += 1
                wait(empty_q[q_stage], parity=q_phase ^ 1)
                # instruction_selection: LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64
                #   P1, [bar], parity, 0x989680; @P1 bra DONE; bra LAB_WAIT
                issue_tma_q(q_stage, q_idx + 1)
            if kv_idx == 0 or kv_block_idx_ptr == 32:    # refresh the 32-task window
                kv_block_idx_ptr = 0
                j = kv_idx + kv_group_idx + lane_idx * 4
                kv_block_idx_storage = 0
                if j < num_kv:
                    copy_g2r(block_table[u64(q_idx) * block_table_stride + j], kv_block_idx_storage)
                    # instruction_selection: ld.global.u32 (64-bit offset: mul.lo.s64);
                    #   extent: one per lane
            kv_block_idx = shuffle_index(kv_block_idx_storage, src_lane=kv_block_idx_ptr)
            # instruction_selection: shfl.sync.idx.b32 (runtime lane, full mask); extent: scalar
            kv_block_idx_ptr += 1
            kv_stage, kv_phase = kv_pipeline(kv_iter_idx); kv_iter_idx += 1
            wait(empty_kv_g[kv_stage], parity=kv_phase ^ 1)
            # instruction_selection: the same try_wait.parity + 0x989680 retry loop
            if elect_predicate(0xffffffff):
                # instruction_selection: elect.sync
                copy_g2s(tmap_kv.coord(0, 0, kv_block_idx), at(smem, off_kv(kv_stage)),
                         completion=full_kv_g[kv_stage], cache_hint=EVICT_NORMAL)
                # instruction_selection: cp.async.bulk.tensor.3d.shared::cluster.global
                #   .mbarrier::complete_tx::bytes.L2::cache_hint; extent: one [D x 64 x 1] box
                copy_g2s(tmap_kv_scales.coord(0, kv_block_idx), at(smem, off_sf(kv_stage)),
                         completion=full_kv_g[kv_stage], cache_hint=EVICT_NORMAL)
                # instruction_selection: cp.async.bulk.tensor.2d...L2::cache_hint; [64 x 1] f32
                expect_bytes(full_kv_g[kv_stage], KV_ST + 64 * 4)
                # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64
            fetched, (next_q, next_kv, next_num_kv) = fetch_next_task()
        # The loop is not unrolled (one 3D TMA site in the PTX); no drain at the end.
```

### Role 2: math warpgroups (warps 0..15; `.cuh:216-330`)

```python
    else:
        setmaxnreg(inc, 104)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32 104
        accum = reg_tile(f32, kNumAccum); weights = reg_tile(f32, kNextN * kNumHeads / 4)
        sub_warp_offset = (warp_idx % 4) * 16
        v_0_offset = lane_idx // 4; v_1_offset = v_0_offset + 8
        # 64-bit GMMA descriptor fields, written explicitly (`make_smem_desc(ptr,
        #   to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8)`, mma/sm90.cuh:197-209, 283-291):
        #   bits [0,14) start address >> 4      bits [16,30) LBO >> 4 = 0
        #   bits [32,46) SBO >> 4 = 8*kHeadDim >> 4    bits [49,52) base offset = 0
        #   bits [62,64) layout: B128 = 1 (D128), B64 = 2 (D64), B32 = 3 (D32)
        #   -> high word or.b64 0x4000004000000000 (D128), 0x8000002000000000 (D64),
        #      0xC000001000000000 (D32); k-step k adds 32 B to the start address.
        #   A = this group's whole 64-token KV block; B = the q's kNextN*H rows.
        q_idx = batch_size

        while True:
            valid, (next_q, next_kv, next_num_kv) = fetch_next_task()
            if not valid: break
            if q_idx != next_q:                  # the q changes
                if q_iter_idx > 0:
                    arrive(empty_q[(q_iter_idx - 1) % 3])
                    # instruction_selection: mbarrier.arrive.shared::cta.b64; every math thread
                q_stage, q_phase = q_pipeline(q_iter_idx); q_iter_idx += 1
                wait(full_q[q_stage], parity=q_phase)
                # instruction_selection: try_wait.parity + 0x989680 retry loop
                for i in static_range(kNextN):
                    for j in static_range(kNumHeads // 4):
                        copy_s2r(at(smem, off_w(q_stage) + 4 * (i * kNumHeads + (j // 2) * 8
                                                             + (j & 1) + (lane_idx % 4) * 2)),
                                 weights[i][j])
                        # instruction_selection: ld.shared.f32; extent: kNextN * kNumHeads / 4
            q_idx, kv_idx = next_q, next_kv
            kv_offset = u64(q_idx * kNextN) * u64(logits_stride) \
                        + ((kv_idx + kv_group_idx) * 64 + sub_warp_offset)
            kv_stage, kv_phase = kv_pipeline(kv_iter_idx); kv_iter_idx += 1
            wait(full_kv_g[kv_stage], parity=kv_phase)
            # instruction_selection: try_wait.parity + 0x989680 retry loop
            wgmma_fence_operand(accum)
            wgmma_arrive()
            # instruction_selection: wgmma.fence.sync.aligned
            for k in static_range(kNumKSteps):
                gemm(accum, desc(start=off_kv(kv_stage) + 32 * k),
                     desc(start=off_q(q_stage) + 32 * k), scale_d=(k != 0), instr=WGMMA)
                # instruction_selection: wgmma.mma_async.sync.aligned.m64{N}k32.f32.e4m3.e4m3,
                #   scale-d predicate; extent: kNumKSteps (1 / 2 / 4)
            wgmma_commit()
            # instruction_selection: wgmma.commit_group.sync.aligned
            wgmma_fence_operand(accum)
            scale_kv_0 = copy_s2r(at(smem, off_sf(kv_stage) + 4 * (sub_warp_offset + v_0_offset)))
            scale_kv_1 = copy_s2r(at(smem, off_sf(kv_stage) + 4 * (sub_warp_offset + v_1_offset)))
            # instruction_selection: ld.shared.f32 x2, between the commit and the wait
            wgmma_wait(0)
            # instruction_selection: wgmma.wait_group.sync.aligned 0
            arrive(empty_kv_g[kv_stage])
            # instruction_selection: mbarrier.arrive.shared::cta.b64; every thread of the group

            for i in static_range(kNextN):
                a = accum[i * kNumAccumPerReduce :]
                # nvcc's contraction of `sum[k] = t(k); sum[k] += t(4j+k)`, in the PTX's order:
                # head group j outer, chain k inner (the four k-chains are interleaved).
                r0 = reg_tile(f32, 4)
                for k in static_range(4):
                    r0[k] = max(a[k], 0.0)
                    # instruction_selection: max.f32 (non-FTZ, zero register)
                for k in static_range(4):
                    r1 = max(a[4 + k], 0.0)
                    # instruction_selection: max.f32
                    s_k = mul(r1, weights[i][2 + (k & 1)])
                    # instruction_selection: mul.f32 (the j=1 product, rounded)
                    s_k = fma(r0[k], weights[i][k & 1], s_k)
                    # instruction_selection: fma.rn.f32 (the j=0 product fused)
                for j in static_range(2, kNumHeads // 8):
                    for k in static_range(4):
                        r = max(a[4 * j + k], 0.0)
                        # instruction_selection: max.f32
                        s_k = fma(r, weights[i][2 * j + (k & 1)], s_k)
                        # instruction_selection: fma.rn.f32
                t_0 = add(s_0, s_1)
                # instruction_selection: add.f32
                v_0 = mul(scale_kv_0, t_0)
                # instruction_selection: mul.f32
                t_1 = add(s_2, s_3)
                # instruction_selection: add.f32
                v_1 = mul(scale_kv_1, t_1)
                # instruction_selection: mul.f32
                for m in (1, 2):
                    o_0 = shuffle_xor(v_0, m)
                    # instruction_selection: shfl.sync.bfly.b32 (lane mask m, clamp 31, mask -1)
                    v_0 = add(v_0, o_0)
                    # instruction_selection: add.f32
                    o_1 = shuffle_xor(v_1, m)
                    # instruction_selection: shfl.sync.bfly.b32
                    v_1 = add(v_1, o_1)
                    # instruction_selection: add.f32
                base = kv_offset + u64(i) * u64(logits_stride) + v_0_offset
                vc_0 = cast(v_0, logits_dtype, rn)
                # instruction_selection: cvt.rn.bf16.f32 (bf16); none (f32)
                copy_r2g(vc_0, logits[base])
                # instruction_selection: st.global.f32 / st.global.u16 [base]
                vc_1 = cast(v_1, logits_dtype, rn)
                # instruction_selection: cvt.rn.bf16.f32 (bf16); none (f32)
                copy_r2g(vc_1, logits[base + 8])
                # instruction_selection: st.global.f32 / .u16 [base + 8 elements] (immediate
                #   offset +32 / +16 B); no masking: whole 64-token blocks are written
```

## Logical GEMM shapes and owners

| Op | Shape | Owner | Per |
| --- | --- | --- | --- |
| WGMMA | m64 n`kNextN*H` k32, A = 64 tokens of block `kv_idx+g`, B = the q's `kNextN*H` rows | math warpgroup g | `kHeadDim/32` per task |
| head reduction | `kNumHeads/2` accumulators -> 2 values per next_n row per thread | each math thread | per task per next_n row |
| TMA Q / weights | [`kNextN*H` x D] fp8 / [`kNextN` x H] f32 | group 0's elected lane | per q (prefetched one q ahead) |
| TMA KV / scales | [D x 64 x 1] fp8 (3D, batch = block index) / [64] f32 | TMA warp g's elected lane | per task |

## TensorMap fields (`launch.py`, from `sm90_fp8_mqa_logits.hpp:337-352`)

| map | gmem dims | box | swizzle | strides (elements) |
| --- | --- | --- | --- | --- |
| Q | (D, batch*next_n*H) | (D, next_n_atom*H) | D B | q.stride(2) = D |
| KV | (D, 64, num_blocks) | (D, 64, 1) | D B | (D, 64*(D+4)) bytes |
| KV scales | (64, num_blocks) | (64, 1) | none | 64*(D+4)/4 |
| weights | (H, batch*next_n) | (H, next_n_atom) | none | weights.stride(0) |

`next_n_atom` = 2 if next_n >= 2 else 1 (= kNextN on SM90). L2 promotion 256B, no OOB fill.

## Storage lifetimes

| region | written by | read by | released |
| --- | --- | --- | --- |
| `smem_q[s]`, `smem_w[s]` | TMA (group 0 issues; `full_q[s]`) | every group's WGMMA B; weights `ld.shared` on q change | `empty_q[s]` by all 512 math threads when their q changes |
| `smem_kv_g[s]`, `smem_sf_g[s]` | TMA warp g (`full_kv_g[s]`) | warpgroup g's WGMMA A; scale `ld.shared` | `empty_kv_g[s]` by warpgroup g's 128 threads after `wgmma.wait_group 0` |
| `kv_block_idx_storage` (TMA lanes) | per refresh | `shfl.sync.idx` per task | next refresh (`kv_idx == 0` or 32 tasks) |

## Static specialization boundary

Compile time: `KernelSpec` (next_n, heads, head dim, logits dtype, SM count); runtime: batch,
strides, context lens, block table, schedule metadata, tensor addresses.

## TIRx module and benchmark contract

`KERNEL_META` names `deepgemm_sm90_fp8_paged_mqa_logits`, `runtime_cuda_archs = ["sm_90a"]`,
DeepGEMM @ 559d79fb. Both kernels consume the same `schedule_meta` from DeepGEMM's metadata
kernel and write into identical sentinel-filled logits storage; the whole storage must be
bitwise identical, then upstream's simulated-reference tolerance.

## Instruction-selection summary

Per-specialization counts are in `source_export/INDEX.txt`; for `2,64,128,float`:

| instruction | static count | cause |
| --- | --- | --- |
| `wgmma.mma_async...m64n128k32.f32.e4m3.e4m3` | 4 | `kHeadDim/32` (1/2/4); N = 32/64/128 by next_n x heads |
| `cp.async.bulk.tensor.2d...L2::cache_hint` | 5 | Q + weights at 2 sites (first q, prefetch) = 4; KV scales 1 |
| `cp.async.bulk.tensor.3d...L2::cache_hint` | 1 | KV block (loop not unrolled) |
| `mbarrier.arrive.expect_tx.shared::cta.b64` | 3 | Q x2 sites, KV x1 |
| `mbarrier.try_wait.parity.shared::cta.b64` | 4 | empty_q, empty_kv (TMA); full_q, full_kv (math) |
| `mbarrier.arrive.shared::cta.b64` | 2 | empty_q, empty_kv (math) |
| `max.f32` / `mul.f32` / `fma.rn.f32` / `add.f32` | 64 / 12 / 56 / 12 | per next_n row: H/2 max, 4+2 mul, 4*(H/8-1) fma, 2+4 add |
| `shfl.sync.bfly.b32` / `shfl.sync.idx.b32` | 8 / 3 | 4 per row / warp index, group index, block-table window |
| `ld.shared.f32` | 34 | weights (`kNextN*H/4` = 32) + 2 scales |
| `ld.global.v2.u32` / `ld.global.u32` | 2 / 5 | v2: scheduler constructor (one site before the role branch); u32: constructor refresh 1 + TMA-role fetch refreshes 2 + math-role fetch refresh 1 + block table 1 |
| `st.global.f32` (bf16: `cvt.rn.bf16.f32` + `st.global.u16`) | 4 | 2 per next_n row |
| `elect.sync` | 5 | prefetch, init, first-Q, prefetch-Q, KV issue |
| `setmaxnreg.dec 64` / `.inc 104` | 1 / 1 | role register split |
| FP32 ops carry no `.ftz` | | DeepGEMM builds without fast math |

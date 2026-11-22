<!--
This sketch is a design artifact, not source. It is written once, reviewed
once, and then frozen: the implementation follows it, and where the two
disagree the TIRx module linked below is what runs.

Documents a TIRx port of Dao-AILab/flash-attention
(https://github.com/Dao-AILab/flash-attention @ 0251105a2fb19d2957484b7f023cd8c115286ced),
hopper/flash_bwd_preprocess_kernel.h, hopper/flash_bwd_kernel_sm90.h,
hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp, hopper/epilogue_bwd.hpp, hopper/tile_scheduler.hpp,
hopper/flash_bwd_postprocess_kernel.h. Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang,
Vijay Thakkar, Pradeep Ramani, Tri Dao. SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# fa3_backward_sm90: coarse WASP pipeline sketch

This non-executable design sketch describes the storage, warp roles, pipelines, control flow and
PTX-level operations of the TIRx port of FlashAttention-3's SM90 backward pass
([`tirx_kernels/ported/flashattention/_fa3_backward_sm90/`](../../../tirx_kernels/ported/flashattention/_fa3_backward_sm90/__init__.py),
entry module `tirx_kernels/ported/flashattention/fa3_backward_sm90.py`). That module is the
authoritative implementation.

## Scope

Source: flash-attention @ `0251105a`, `hopper/`, built by `hopper/setup.py` (`-O3 --use_fast_math
-lineinfo -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 -DNDEBUG`,
`-gencode arch=compute_90a,code=sm_90a`, nvcc 13.2). The host path is `mha_bwd`
(`flash_api_stable.cpp:1338`) -> `run_mha_bwd_hdim128<90, bf16, false>`
(`flash_bwd_launch_template.h:345`) -> `run_mha_bwd_dispatch` (Varlen = GQA = Deterministic =
false) -> `run_flash_bwd` (`:25`), which launches three kernels.

In scope: bf16, head_dim = head_dim_v = 128, MHA (h == h_k), fixed length, no softcap, no local
window, non-deterministic dQ accumulation, causal and non-causal. These are exactly two
specializations of `run_flash_bwd`:

| | causal | non-causal |
| --- | --- | --- |
| template | `<90, 128, 64, 128, bf16, true, false, false, false, false, false, 2, 2, true, false, false, 2, 1, 2, 1, false>` | `<90, 128, 80, 128, bf16, false, ..., 2, 2, true, false, true, 2, 1, 2, 1, false>` |
| kBlockM x kBlockN | 64 x 128 | 80 x 128 |
| dQ_swapAB | false: dQ (64 x 128) = dS K, each MMA WG one 64-column half | true: dQ^T (128 x 80) = K^T dS^T, each MMA WG one 64-row half |
| scheduler | `SingleTileBwdLPTScheduler<false, 128, false>` (1-D grid, L2 swizzle) | `SingleTileScheduler<false, false, false, 128>` (3-D grid) |
| empty KV block | possible (`m_block_max <= m_block_min`): `store_zero` | impossible: no `store_zero` code |

Fixed by the template for both: Stages (Q) = Stages_dO = Stages_dS = 2, SdP_swapAB = true,
dKV_swapAB = false, NumMmaWarpGroups = 2, AtomLayoutMSdP = 1, AtomLayoutNdKV = 2,
AtomLayoutMdQ = 1, Mma_dP_is_RS = false. Derived (printed by the export's layout probe):
`Mma_dKV_is_RS = 1` (dV and dK take P^T and dS^T from registers), `Mma_dQ_is_RS = 0`,
`dQacc_use_TMA = 1`, `Slice_dQKV_Mma = 0`, `SeparateMaskingIterations = 0`, `ShuffleLSE =
ShuffledPsum = 0`, `smem_p` empty.

Out of scope (later steps, separate instantiations): varlen (`Varlen = true`: non-TMA epilogue,
scheduler validity), GQA/MQA (`CollectiveEpilogueBwdGQA` + dK/dV postprocess), deterministic
mode (dq semaphore), head_dim 64 (128 x 128, separate masking iterations, shuffled LSE) and 256
(64 x 80, dKV_swapAB, sliced dQ MMA with register atomics), fp16, softcap, local. Tile (`Tx`)
primitives everywhere.

## The line-info export this sketch is annotated from

`.porting/fa3_backward_sm90/source_export/`, produced by `export.sh` from `export_driver.cu`,
which explicitly instantiates the two `run_flash_bwd` specializations above with `setup.py`'s
nvcc flags and the build's feature flags (`-DFLASHATTENTION_DISABLE_{SPLIT,PAGEDKV,...}`):
`hd128.ptx` (43789 lines, 5109 `.loc`, 6 entries), split per kernel into
`hd128_{pre_m64,pre_m80,main_causal,main_noncausal,post_m64,post_m80}.ptx`, with emission-order
skeletons `*_skel.txt` (innermost `.loc` + inlined-at site per run), `hd128.cubin`,
`hd128.sass` and `hd128.nvdisasm`. `layout_probe.txt` prints the exact `SharedStorage` offsets,
`TmaTransactionBytes` and every cute layout / TiledMMA / copy atom of both specializations,
from the same types `run_flash_bwd` builds. ptxas: main kernel 168 registers (the entry count;
`setmaxnreg` 24 / 240), 16 barriers, causal 0 spill, non-causal 4 B spill; preprocess 64 / 70
registers; postprocess 56 / 64. Counts below are static PTX instruction lines of the causal
kernel unless marked; the `.file` numbers are resolved to file names.

Facts from the export that the source text does not settle:

1. **Named-barrier ids.** `BwdNamedBarriers` goes through `cutlass::arch::NamedBarrier` with
   `FirstUserBarrier = 8`: PdS = 9 (256 threads), dQEmpty = 10 / 11 (160), dQFull = 13 / 14
   (160). The epilogue barrier is `ReservedNamedBarriers::EpilogueBarrier` = id 1 (256 and
   288). `__syncthreads` is `bar.sync 0`.
2. **mbarriers.** `barrier_KV` at pool 198656 (+0) count 1; pipeline_q full[0..1] at +16/+24
   count 1, empty[0..1] at +32/+40 count **256** (every MMA thread arrives, unpredicated:
   `mapa.shared::cluster.u32` to rank 0 + `mbarrier.arrive.shared::cluster.b64`); pipeline_do
   full +48/+56, empty +64/+72. Inits interleave full/empty per stage, each pipeline followed by
   `fence.mbarrier_init.release.cluster`. Non-causal: the same at 231424.
3. **Waits.** Producer acquire and the consumer's second try are `mbarrier.try_wait.parity` with
   the suspend-time hint 10000000 in a retry loop (`ClusterBarrier::wait`); the consumer's first
   try is a single `try_wait.parity` without hint whose result selects whether the loop runs
   (`consumer_try_wait` + `consumer_wait`). SASS: `SYNCS.PHASECHK.TRANS64.TRYWAIT` 32,
   `NANOSLEEP.SYNCS` 15.
4. **TMA.** Every Q / dO / K / V tile is two `cp.async.bulk.tensor.4d.shared::cluster.global.
   mbarrier::complete_tx::bytes.L2::cache_hint` boxes of 64 columns (SW128); 20 static sites (the
   producer loop is unrolled by 2). Q and dO carry `EVICT_LAST` = 0x14F0000000000000; K and V
   carry the default 0x1000000000000000. LSE and dPsum rows are `cp.async.bulk.shared::cluster.
   global.mbarrier::complete_tx::bytes` of kBlockM * 4 bytes on the same full barrier.
5. **dQ accumulation** is `cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32`
   of kBlockM * 64 * 4 bytes (16384 / 20480) per MMA warpgroup half, hint EVICT_LAST, issued by
   an elected lane of producer warp 1; 6 static sites (loop unrolled by 2, plus peel).
6. **WGMMA chains** (causal / non-causal): S^T and dP^T: 8 x `m64n64k16` / `m64n80k16` ss,
   flags `(1, 1, 0, 0)`; dQ: 8 x `m64n64k16` ss flags `(1, 1, 0, 1)` (B = K MN-major) / dQ^T: 8 x
   `m64n80k16` ss flags `(1, 1, 1, 0)` (A = K MN-major); dV and dK: 4 / 5 x `m64n128k16` **rs**
   (A = 4 bf16x2 registers), flags `(1, 1, 1)` (B MN-major). Descriptor high word
   0x4000004000000000 (SWIZZLE_128B, SBO 1024 B). `wgmma.wait_group` 1 after S^T + dP^T, 0
   before dS, 1 after dQ (retires dV), 1 after dK (retires dQ), 0 at the step end.
7. **Float forms** (fast math): exp2 argument `mul.ftz.f32` + `sub.ftz.f32` (contracted by
   ptxas to one `FFMA.FTZ`, 32 / 40 in SASS) then `ex2.approx.ftz.f32`; dS = `sub.ftz.f32`
   (FADD) then `mul.ftz.f32`; bf16 packing `cvt.rn.bf16x2.f32`; dK scale `mul.ftz.f32` x64;
   masking `selp.f32 0fFF800000` (FSEL 32 / 40); preprocess dot `mul.ftz.f32` + 7 x
   `fma.rn.ftz.f32` per row, 16-lane butterfly `shfl.sync.bfly.b32` + `add.ftz.f32`.
8. **Smem stores.** dS^T to `sdS` with `stmatrix.sync.aligned.x4.trans.m8n8.shared.b16` (4 / 5 per
   thread per step); dQ fp32 to `sdQacc` with `st.shared.v4.u32` (8 / 10 per step); epilogue dK,
   dV with `stmatrix.sync.aligned.x4.m8n8.shared.b16` (16); postprocess `stmatrix` x4 (causal,
   non-transposed, Swizzle<3,3,3>) / x4.trans (non-causal, Swizzle<1,3,3>).
9. **Loads of row statistics.** LSE and dPsum per step: `ld.shared.v2.f32` x8 / x10 each (the
   16 / 20 Q columns of the thread's S^T fragment, stride-0 over kv rows).

## Pipeline at a glance

Three kernels, in stream order:

| Kernel | Grid x block | Role-local program | Publication / reuse |
| --- | --- | --- | --- |
| `flash_bwd_preprocess` | (ceil(s_q/kBlockM), h, b) x 256 | per CTA: a kBlockM x 128 tile of O and dO -> dPsum rows, LSE*log2e rows, zero one dQaccum tile, zero one dq_semaphore word | none (stream order) |
| `flash_bwd_sm90` | causal: (ceil(s_k/128) * h * b); non-causal: (ceil(s_k/128), h, b) x 384 | per CTA: one (n_block, head, batch) KV block; loops over the m_blocks that see it | see below |
| `flash_bwd_convert_dq` | (ceil(s_q/kBlockM), h, b) x 256 | per CTA: one dQaccum tile -> dQ * softmax_scale, bf16 | stream order after the main kernel |

Main kernel roles:

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0 (WG0, elected lane) | TMA Q[m] + LSE[m] into `pipe_q` stage, K and V once into `barrier_KV`, dO[m] + dPsum[m] into `pipe_do` stage; producer tail | `pipe_q.full`, `pipe_do.full`, `barrier_KV`; waits `pipe_q.empty`, `pipe_do.empty` |
| 1 (WG0) | per m_block, per MMA WG w: wait dQFull_w, bulk reduce-add `sdQacc[w]` into dQaccum, release dQEmpty_w after the read completes | named barriers 13/14 (sync), 10/11 (arrive) |
| 2, 3 (WG0) | idle after `setmaxnreg.dec 24` | none |
| 4..7 (WG1), 8..11 (WG2) | per m_block: S^T, dP^T (ss), P^T, dS^T in registers, dS^T -> `sdS` (stmatrix.trans), dV += P^T dO (rs), dQ (ss) -> `sdQacc[w]`, dK += dS^T Q (rs); at the tile end dK *= scale, epilogue | `pipe_q.empty`, `pipe_do.empty` (256 arrivals); PdS barrier 9 between the WGs; dQEmpty/dQFull with warp 1; epilogue barrier 1 |

WG1 owns KV rows 0..63 of the block, WG2 rows 64..127, for S^T, dP^T, dV and dK. For dQ (causal)
WG w owns dQ columns 64w..64w+63 of all kBlockM rows; for dQ^T (non-causal) WG w owns rows
64w..64w+63 of dQ^T, i.e. the same dQ columns. The two WGs exchange dS through `sdS`.

## Primitive vocabulary

Structural operations (no data movement):

```python
tile(...)        # declare storage, dtype, logical shape, placement (linear SMEM byte offsets)
view(...)        # index the same storage differently (explicit offset function)
alias(...)       # exact storage aliasing with disjoint lifetimes
reg_tile(...)    # declare a role-local register fragment (explicit per-thread element map)
```

Copies state their direction: `copy_g2s` (TMA tensor or bulk), `copy_s2g` (TMA store),
`reduce_add_s2g` (bulk reduce-add), `copy_g2r`, `copy_r2g`, `copy_s2r`, `copy_r2s`.
Computation: `fill`, `cast`, `add`, `sub`, `mul`, `fma`, `exp2`, `select`, `shuffle_xor`,
`gemm(dst, a, b, accumulate)`. Schedule operations: `mbar_init`, `fence_mbar_init`,
`arrive_expect_tx`, `wait(bar, phase)`, `try_wait(bar, phase)`, `arrive(bar)`,
`named_sync(id, n)`, `named_arrive(id, n)`, `fence_proxy_async()`, `wgmma_fence()`,
`wgmma_commit()`, `wgmma_wait(n)`, `bulk_commit()`, `bulk_wait_read(n)`, `setmaxnreg`,
`cta_sync()`, `prefetch_tensormap`, `elect_one()`.

## Shared storage and address maps

All SMEM of the main kernel is one linear dynamic pool, 1024-byte aligned, at the byte offsets
the layout probe printed:

| object | causal offset | non-causal offset | bytes | content map (explicit) |
| --- | ---: | ---: | ---: | --- |
| `sK` | 0 | 0 | 32768 | `sw128_kmajor(row r in 0..127, col c in 0..127) = (c // 64) * 16384 + r * 128 + (((c % 64) // 8) ^ (r % 8)) * 16 + (c % 8) * 2` |
| `sV` | 32768 | 32768 | 32768 | same map |
| `sdQacc` | 65536 | 65536 | 2 x kBlockM*64*4 | half w at `w * kBlockM * 256`; float `i*512 + t*4 + j` of half w = register `4i + j` of MMA thread t of WG w (dQ C fragment, row-major register order) |
| `sQ` | 98304 | 106496 | 2 x kBlockM*256 | stage s at `s * kBlockM * 256`; inside a stage `sw128_kmajor` with half stride `kBlockM * 128` |
| `sdO` | 131072 | 147456 | same | same |
| `sLSE` | 163840 | 188416 | 512 / 832 | float `m + s * round_up(kBlockM, 64)` |
| `sdPsum` | 164352 | 189312 | 512 / 832 | same |
| `sdS` | 165888 | 190464 | 2 x kBlockM*256 | stage s at `s * kBlockM * 256`; (m, n) K-major (n contiguous), `sw128_kmajor` with half stride `kBlockM * 128` |
| `sdK` (epilogue) | 0 | 0 | 32768 | alias of `sK`: (n, d), `sw128_kmajor` |
| `sdV` (epilogue) | 32768 | 32768 | 32768 | alias of `sV` |
| `barrier_KV` | 198656 | 231424 | 8 | mbarrier |
| `pipe_q.full[s]`, `.empty[s]` | +16 + 8s, +32 + 8s | same | | mbarriers |
| `pipe_do.full[s]`, `.empty[s]` | +48 + 8s, +64 + 8s | same | | mbarriers |
| total | 199680 | 232448 | | `cudaFuncAttributeMaxDynamicSharedMemorySize` |

`sdK`/`sdV` alias `sK`/`sV`; the epilogue's `named_sync(1, 256)` orders them after the last
read of K and V by both WGs.

GMMA descriptors (`desc(addr, kind)`): low word = start address field `(addr >> 4) & 0x3FFF` (bits
0-13) | LBO field << 16, high word 0x40000040 (SWIZZLE_128B in bits 62-63, SBO field 64 = 1024 B).
The export's descriptor high/low constants (`or.b64` immediates, mma_traits_sm90_gmma.hpp:216/327):
K-major operands (sK, sV, sQ, sdO in S^T/dP^T; sdS in dQ) `0x4000004000010000`, i.e. LBO field 1
(16 B, ignored for swizzled K-major); MN-major sQ / sdO (dV, dK B operands, two 64-column halves)
`0x4000004002000000` causal / `0x4000004002800000` non-causal, i.e. LBO = the half stride kBlockM*128
(8192 / 10240 B); MN-major sK half (dQ B / dQ^T A operand, one 64-column atom per WG)
`0x4000004000000000`, LBO 0. The start-address field is added to the low word per operand and k-step. K-major operand, k-step `k` of 16 elements:
`base + (k // 4) * half_stride + (k % 4) * 32`. MN-major operand (tnsp = 1), k-step `k` over 16
rows: `base + k * 2048`, LBO as listed above (`kBlockM*128` for sQ/sdO, 0 for the sK half).

Register fragments (per MMA thread t of a WG, lane = t % 32, warp = t // 32): the WGMMA C
fragment of an m64nN tile holds rows `16*warp + lane//4 (+8)` and columns `8*c + 2*(lane%4)
(+1)` for c in 0..N/8-1, register `4c + {0, 1}` (row) / `{2, 3}` (row + 8). The rs A fragment of
k-step k is the four bf16x2 registers made from C registers `8k .. 8k+7` of the m64N S^T/dP^T
tile (FA3's `convert_layout_acc_Aregs`: no data movement).

## Complete sketch

### Kernel 1: `flash_bwd_preprocess` (`flash_bwd_preprocess_kernel.h:143-245`)

```python
@kernel(grid=(cdiv(s_q, kBlockM), h, b), block=256)   # __launch_bounds__(256, 2)
def flash_bwd_preprocess(O, dO, LSE, dPsum, LSE_log2, dQaccum, dq_semaphore, s_q, d, h, b):
    # d (head dim, get<1>(shape_O)) is a runtime value: the column predicate below reads it.
    m_block, bidh, bidb = block_idx
    t = thread_idx; row0 = t // 16; col8 = (t % 16) * 8          # GmemTiledCopy (16 x 16) x 8
    pcol = col8 < d                                                # tOpO (one K iteration: 128 = 16 x 8)
    lse = LSE[bidb, bidh, m_block*kBlockM + t] if t < s_q - m_block*kBlockM and t < kBlockM else +inf
    # instruction_selection: ld.global.f32 (predicated by branch); extent: scalar
    for i in range(kBlockM // 16):                                 # all O rows first (flash::copy of tOgO)
        r = row0 + 16*i
        rO[i] = copy_g2r(O[bidb, m_block*kBlockM + r, bidh, col8:col8+8],
                         mask=(r < s_q - m_block*kBlockM) and pcol, fill=0)   # Clear_OOB_MN, Clear_OOB_K
        # instruction_selection: ld.global.v2.u64 (128-bit) under setp.lt + and.pred branch; zero fill by mov;
        #   extent: kBlockM // 16 (4 / 5)
    for i in range(kBlockM // 16):                                 # then all dO rows (flash::copy of tOgdO)
        r = row0 + 16*i
        rdO[i] = copy_g2r(dO[bidb, m_block*kBlockM + r, bidh, col8:col8+8],
                          mask=(r < s_q - m_block*kBlockM) and pcol, fill=0)
        # instruction_selection: ld.global.v2.u64 (128-bit), same predicate; extent: 4 / 5
    for i in range(kBlockM // 16):
        o, g = cast(rO[i], f32), cast(rdO[i], f32)                  # bf16 -> f32 by shl/and (no cvt)
        acc = mul(g[0], o[0]); acc = fma(g[j], o[j], acc) for j in 1..7
        # instruction_selection: mul.ftz.f32 + 7 x fma.rn.ftz.f32; extent: per row
        for off in (8, 4, 2, 1):
            acc = add(acc, shuffle_xor(acc, off))
        # instruction_selection: shfl.sync.bfly.b32 (mask 0xffffffff, clamp 0x1f) + add.ftz.f32; extent: 4 per row
        if col8 == 0:
            dPsum[bidb, bidh, m_block*kBlockM + r] = acc if r < s_q - m_block*kBlockM else 0
            # instruction_selection: st.global.f32 (selp for the 0); extent: one per row, lane-0 threads
    if t < round_up(s_q, kBlockM) - m_block*kBlockM and t < kBlockM:
        LSE_log2[bidb, bidh, m_block*kBlockM + t] = 0 if lse == -inf else mul(lse, LOG2E)
        # instruction_selection: mul.ftz.f32 + selp + st.global.f32; extent: scalar
    fill(dQaccum[bidb, bidh, m_block*kBlockM*128 + t*4 + 1024*i : +4], 0) for i in range(kBlockM*128 // 1024)
    # instruction_selection: st.global.v2.u64 (128-bit zero); extent: 8 / 10 per thread
    if dq_semaphore != nullptr and t == 0:                         # mha_bwd always passes one (non-null)
        # instruction_selection: setp.eq.s64 (pointer null test) + setp (t == 0), branch
        dq_semaphore[bidh + bidb*h + m_block*h*b] = 0              # instruction_selection: st.global.u32
```

### Kernel 2: `flash_bwd_sm90` (`flash_bwd_kernel_sm90.h:155-278`)

```python
@specialize(causal in {True, False})
@kernel(grid=sched_grid, block=384, min_blocks_per_sm=1, dyn_smem=199680 | 232448)
def flash_bwd_sm90(tma_q, tma_do, tma_k, tma_v, tma_dk, tma_dv, LSE_log2, dPsum, dQaccum, dK, dV,
                   s_q, s_k, d, h, b, window_size_right, softmax_scale, scale_log2, sched_params...):
    # window_size_right (0 for causal, -1 otherwise, from mha_bwd) and d are runtime kernel parameters:
    # every role's m-block range reads window_size_right (ld.param + sub.s32), store_zero reads d.
    pool = tile(SMEM, u8, dyn_bytes, align=1024)                   # offsets: table above (the PTX declares the
                                                                    # extern pool .align 16; 1024 B comes from the
                                                                    # runtime base, as upstream relies on)
    lane_pred = elect_one()                                        # K:166, one elect.sync reused below
    # instruction_selection: elect.sync; extent: scalar
    warp = shfl_idx(thread_idx // 32, 0)                           # canonical_warp_idx_sync
    # instruction_selection: shfl.sync.idx.b32; extent: scalar
    if warp == 0 and lane_pred:
        prefetch_tensormap(tma_q, tma_do, tma_k, tma_v, tma_dk, tma_dv)
        # instruction_selection: prefetch.tensormap x6; extent: 6
    wg = shfl_idx(thread_idx // 128, 0)                            # canonical_warp_group_idx
    # instruction_selection: shfl.sync.idx.b32; extent: scalar
    if warp == 0 and lane_pred:
        mbar_init(barrier_KV, 1)                                   # instruction_selection: mbarrier.init.shared::cta.b64
    for pipe in (pipe_q, pipe_do):                                 # PipelineTmaAsync ctor, each: warp 0 elected lane
        if warp == 0 and elect_one():
            for s in (0, 1):
                mbar_init(pipe.full[s], 1); mbar_init(pipe.empty[s], 256)
                # instruction_selection: mbarrier.init.shared::cta.b64 x4 (full0, empty0, full1, empty1)
        fence_mbar_init()                                          # instruction_selection: fence.mbarrier_init.release.cluster
    cta_sync()                                                     # instruction_selection: bar.sync 0

    if wg == 0:                                                    # ---------------- producer warpgroup
        setmaxnreg(dec, 24)                                        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 24
        wiw = shfl_idx(warp % 4, 0)                                # instruction_selection: shfl.sync.idx.b32; extent: scalar
        if wiw == 0:
            producer_load_warp()
        elif wiw == 1:
            dq_store_warp()
    else:                                                          # ---------------- MMA warpgroups
        setmaxnreg(inc, 240)                                       # instruction_selection: setmaxnreg.inc.sync.aligned.u32 240
        mma_warpgroups(w = wg - 1, tid = thread_idx - 128)
```

#### Tile scheduler (`tile_scheduler.hpp:37-117`, `:366-470`)

```python
def tile():                                                        # get_initial_work; get_next_work -> invalid
    if causal:                                                     # SingleTileBwdLPTScheduler, SPT = false
        # host: size_one_head = s_q*(128+128)*2 + s_q*128*4 (TileSchedulerArguments.seqlen_k slot receives
        #       params.seqlen_q in run_flash_bwd); swizzle = 1 if 40 MiB < size_one_head
        #       else 2**floor(log2(40 MiB // size_one_head)); hb_rem = (h*b) % swizzle
        bidhb, l2_mod = divmod(block_idx.x, swizzle * n_blocks)    # FastDivmod (mul.hi.u32 + shr)
        if bidhb < (h*b) // swizzle: n_block, r = divmod(l2_mod, swizzle)
        else:                        n_block, r = divmod(l2_mod, hb_rem)
        bidb, bidh = divmod(bidhb * swizzle + r, h)
        # instruction_selection: mul.hi.u32 + shr.u32 + mul.lo/sub (FastDivmod); extent: 4 divmods
    else:
        n_block, bidh, bidb = block_idx.x, block_idx.y, block_idx.z
    return n_block, bidh, bidb
def m_range(n_block):                                              # BlockMN::get_m_block_min_max (block.h:83)
    m_max = cdiv(s_q, kBlockM)
    m_min = max(0, (n_block*128 + s_q - s_k - window_size_right) // kBlockM) if causal else 0   # C int division
    # instruction_selection: shl + add/sub.s32 + shr-based signed division by the constant kBlockM + max.s32
    return m_min, m_max
```

#### Producer warp 0 (`mainloop_bwd_sm90_tma_gmma_ws.hpp:420-591`)

```python
def producer_load_warp():
    stage, phase = 0, 1                                            # make_producer_start_state
    n_block, bidh, bidb = tile()
    m_min, m_max = m_range(n_block)
    if not (causal and m_max <= m_min):                            # load() early return (causal only)
        lane = elect_one()                                         # instruction_selection: elect.sync
        leader = thread_idx % 128 == 0                             # PipelineParams::is_leader
        if lane:
            wait(pipe_q.empty[stage], phase)                       # producer_acquire
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 + hint 10000000, retry loop
            if leader:
                arrive_expect_tx(pipe_q.full[stage], kBlockM*256 + kBlockM*4)   # 16640 / 20800
                # instruction_selection: setp.ne + bra, mbarrier.arrive.expect_tx.shared::cta.b64
            copy_g2s(Q[bidb, m_min*kBlockM:+kBlockM, bidh, :], sQ[stage], completion=pipe_q.full[stage], cache=EVICT_LAST)
            # instruction_selection: cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
            #   x2 boxes {64 cols, kBlockM rows}, coords (64*half, m*kBlockM, bidh, bidb); extent: 2
            copy_g2s(LSE_log2[bidb, bidh, m_min*kBlockM:+kBlockM], sLSE[stage], completion=pipe_q.full[stage])
            # instruction_selection: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes, kBlockM*4 B
        if lane:
            arrive_expect_tx(barrier_KV, 65536)                    # instruction_selection: mbarrier.arrive.expect_tx
            copy_g2s(K[bidb, n_block*128:+128, bidh, :], sK, completion=barrier_KV, cache=EVICT_NORMAL)
            copy_g2s(V[...same...],                     sV, completion=barrier_KV, cache=EVICT_NORMAL)
            # instruction_selection: cp.async.bulk.tensor.4d ... .L2::cache_hint 0x1000000000000000;
            #   box {64, 128} x2 each; extent: 4
            for m in range(m_min, m_max - 1, unroll=2):
                wait(pipe_do.empty[stage], phase)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 + hint 10000000, retry loop
                if leader:
                    arrive_expect_tx(pipe_do.full[stage], kBlockM*256 + kBlockM*4)
                    # instruction_selection: setp.ne + bra, mbarrier.arrive.expect_tx.shared::cta.b64
                copy_g2s(dO[bidb, m*kBlockM:+kBlockM, bidh, :], sdO[stage], completion=pipe_do.full[stage], cache=EVICT_LAST)
                # instruction_selection: cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes
                #   .L2::cache_hint 0x14F0000000000000, box {64, kBlockM}; extent: 2
                copy_g2s(dPsum[bidb, bidh, m*kBlockM:+kBlockM], sdPsum[stage], completion=pipe_do.full[stage])
                # instruction_selection: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes, kBlockM*4 B
                stage, phase = advance(stage, phase)               # one shared state (Q_dO_same_stages)
                wait(pipe_q.empty[stage], phase)
                # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 + hint 10000000, retry loop
                if leader:
                    arrive_expect_tx(pipe_q.full[stage], kBlockM*256 + kBlockM*4)
                    # instruction_selection: setp.ne + bra, mbarrier.arrive.expect_tx.shared::cta.b64
                copy_g2s(Q[bidb, (m+1)*kBlockM:+kBlockM, bidh, :], sQ[stage], completion=pipe_q.full[stage], cache=EVICT_LAST)
                # instruction_selection: cp.async.bulk.tensor.4d ... .L2::cache_hint 0x14F0000000000000; extent: 2
                copy_g2s(LSE_log2[bidb, bidh, (m+1)*kBlockM:+kBlockM], sLSE[stage], completion=pipe_q.full[stage])
                # instruction_selection: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes; extent: 1
        # scheduler_prefetch(): no-op
        if lane:
            wait(pipe_do.empty[stage], phase)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 + hint 10000000, retry loop
            if leader:
                arrive_expect_tx(pipe_do.full[stage], kBlockM*256 + kBlockM*4)
                # instruction_selection: setp.ne + bra, mbarrier.arrive.expect_tx.shared::cta.b64
            copy_g2s(dO[bidb, (m_max-1)*kBlockM:+kBlockM, bidh, :], sdO[stage], completion=pipe_do.full[stage], cache=EVICT_LAST)
            # instruction_selection: cp.async.bulk.tensor.4d ... .L2::cache_hint 0x14F0000000000000; extent: 2
            copy_g2s(dPsum[bidb, bidh, (m_max-1)*kBlockM:+kBlockM], sdPsum[stage], completion=pipe_do.full[stage])
            # instruction_selection: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes; extent: 1
            stage, phase = advance(stage, phase)
    if elect_one():                                                # load_tail: producer_tail of both pipelines
        # instruction_selection: elect.sync
        stage_do, phase_do = stage, phase                          # M:557, and producer_tail takes the state by value
        for s in range(2):
            wait(pipe_q.empty[stage], phase); stage, phase = advance(stage, phase)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 + hint 10000000, retry loop; extent: 2
        for s in range(2):
            wait(pipe_do.empty[stage_do], phase_do); stage_do, phase_do = advance(stage_do, phase_do)
            # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 + hint 10000000, retry loop; extent: 2
```

#### Producer warp 1: dQ bulk reduce (`:593-667`)

```python
def dq_store_warp():
    n_block, bidh, bidb = tile()
    m_min, m_max = m_range(n_block)
    if causal and m_max <= m_min: return
    lane = elect_one()                                             # instruction_selection: elect.sync
    for m in range(m_min, m_max, unroll=2):
        for w in (0, 1):
            named_sync(13 + w, 160)                                # dQFull_w: instruction_selection: bar.sync 13|14, 160
            if lane:
                reduce_add_s2g(sdQacc[w], dQaccum[bidb, bidh, m*kBlockM*128 + w*kBlockM*64 : +kBlockM*64], cache=EVICT_LAST)
                # instruction_selection: cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32,
                #   kBlockM*256 B; extent: 1
                bulk_commit()                                      # instruction_selection: cp.async.bulk.commit_group
        for w in (0, 1):
            if lane: bulk_wait_read(1 - w)                         # instruction_selection: cp.async.bulk.wait_group.read 1|0
            named_arrive(10 + w, 160)                              # dQEmpty_w: instruction_selection: bar.arrive 10|11, 160
```

#### MMA warpgroups (`flash_bwd_kernel_sm90.h:241-276`, `mainloop ... :669-1042`, `epilogue_bwd.hpp:163-276`)

```python
def mma_warpgroups(w, tid):                                        # w in {0,1} (canonical WG 1, 2), tid in 0..255
    stage, phase = 0, 0
    if shfl_idx(warp % 4, 0) == 0:                                 # instruction_selection: shfl.sync.idx.b32; extent: scalar
        named_arrive(10 + w, 160)                                  # mma_init: dQEmpty_w; instruction_selection: bar.arrive
    n_block, bidh, bidb = tile()
    m_min, m_max = m_range(n_block)
    if causal and m_max <= m_min:
        store_zero(n_block, bidh, bidb); return
    rdK = reg_tile(f32, 64)   ; fill(rdK, 0)                       # m64n128 C fragment: KV rows 64w.., d 0..127
    rdV = reg_tile(f32, 64)   ; fill(rdV, 0)                       # instruction_selection: mov.f32 0 x128
    if not try_wait(barrier_KV, 0): wait(barrier_KV, 0)            # work_idx % 2 == 0
    # instruction_selection: try_wait.parity (no hint) + selp; then retry loop with hint 10000000
    for m in range(m_min, m_max):                                  # CUTLASS_PRAGMA_NO_UNROLL: one body
        bwd_step(m)
    mul(rdK, rdK, softmax_scale)                                   # instruction_selection: mul.ftz.f32 x64
    epilogue_store(rdK, rdV, n_block, bidh, bidb)
```

`bwd_step(m)` (`:834-1001`), with `NM = kBlockM` (64 / 80), `NC = NM // 8` column groups per
fragment (8 / 10), `KM = NM // 16` k-steps of the dV/dK chains (4 / 5):

```python
def bwd_step(m):
    rS  = reg_tile(f32, NM // 2)                                   # S^T C fragment: rows = KV 64w.., cols = Q m*NM..
    ready = try_wait(pipe_q.full[stage], phase)                    # consumer_try_wait(pipeline_q)
    # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 (no suspend hint) + selp.b32; extent: 1
    if not ready:
        wait(pipe_q.full[stage], phase)                            # consumer_wait
        # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 + hint 10000000, retry loop
    wgmma_fence()                                                  # instruction_selection: wgmma.fence.sync.aligned
    for k in range(8):
        gemm(rS, desc(sK + 64w*128, kmajor, k), desc(sQ[stage], kmajor, k), accumulate=k > 0)
        # instruction_selection: wgmma.mma_async.sync.aligned.m64n{64|80}k16.f32.bf16.bf16, ss,
        #   scale-d = (k > 0), flags (1, 1, 0, 0); extent: 8
    wgmma_commit()                                                 # instruction_selection: wgmma.commit_group.sync.aligned
    rLSE = copy_s2r(sLSE[stage][q cols of the thread's fragment])  # 2 * NC values
    # instruction_selection: ld.shared.v2.f32; extent: NC
    rdP = reg_tile(f32, NM // 2)
    ready = try_wait(pipe_do.full[stage], phase)                   # consumer_try_wait(pipeline_do)
    # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 (no suspend hint) + selp.b32; extent: 1
    if not ready:
        wait(pipe_do.full[stage], phase)                           # consumer_wait
        # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64 + hint 10000000, retry loop
    wgmma_fence()                                                  # instruction_selection: wgmma.fence.sync.aligned
    for k in range(8):
        gemm(rdP, desc(sV + 64w*128, kmajor, k), desc(sdO[stage], kmajor, k), accumulate=k > 0)
        # instruction_selection: wgmma ... m64n{64|80}k16 ss, flags (1, 1, 0, 0); extent: 8
    wgmma_commit()                                                 # instruction_selection: wgmma.commit_group.sync.aligned
    wgmma_wait(1)                                                  # instruction_selection: wgmma.wait_group.sync.aligned 1
    # mask (mask.h:62-170, SwapAB = true; Seqlenk mask always, causal mask when causal)
    for i in (0, 1):                                               # the thread's two KV rows n = 16*warp + lane//4 + 8i
        if causal:
            lim = NM if col0_i >= seqlenk_col_limit else col0_i - causal_row_offset     # mask.h:138
            # instruction_selection: setp.gt + selp.b32; extent: 2
            for c in range(NC): for e in (0, 1):
                rS[4c + 2i + e] = select(qcol0(c, e) < lim, -inf, rS[...])              # mask.h:141
                # instruction_selection: setp.gt.s32 + selp.f32 0fFF800000; extent: NM // 4 per i
        else:
            if n_row_i >= s_k - n_block*128:                       # seqlen_k mask over the KV row
                fill(rS[row i], -inf)                              # instruction_selection: selp.f32; extent: NM // 2 total
    for each element (c, e, i):
        rS[...] = exp2(sub(mul(rS[...], scale_log2), rLSE[2c + e]))
        # instruction_selection: mul.ftz.f32 + sub.ftz.f32 (ptxas: FFMA.FTZ) + ex2.approx.ftz.f32; extent: NM // 2
    rdPsum = copy_s2r(sdPsum[stage][q cols])                       # instruction_selection: ld.shared.v2.f32; extent: NC
    wgmma_wait(0)                                                  # instruction_selection: wgmma.wait_group.sync.aligned 0
    for each element: rdP[...] = mul(rS[...], sub(rdP[...], rdPsum[2c + e]))
    # instruction_selection: sub.ftz.f32 (FADD) + mul.ftz.f32; extent: NM // 2
    rP  = cast(rS,  bf16x2)                                        # instruction_selection: cvt.rn.bf16x2.f32; extent: NM // 4
    rdS = cast(rdP, bf16x2)                                        # instruction_selection: cvt.rn.bf16x2.f32; extent: NM // 4
    copy_r2s(rdS, sdS[stage] at (q m, kv n) transposed)            # SM90_U16x8_STSM_T, 8x8 tiles of the C fragment
    # instruction_selection: stmatrix.sync.aligned.x4.trans.m8n8.shared.b16; extent: NM // 16 (4 / 5)
    wgmma_fence()                                                  # instruction_selection: wgmma.fence.sync.aligned
    for k in range(KM):                                            # dV += P^T dO, A from registers
        gemm(rdV, rP[4k:4k+4], desc(sdO[stage], mnmajor, k), accumulate=True)
        # instruction_selection: wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 rs, flags (1, 1, 1); extent: KM
    wgmma_commit()                                                 # no wait (wg_wait = -1)
    # instruction_selection: wgmma.commit_group.sync.aligned
    fence_proxy_async()                                            # instruction_selection: fence.proxy.async.shared::cta
    named_sync(9, 256)                                             # PdS: instruction_selection: bar.sync 9, 256
    rdQ = reg_tile(f32, 32 if causal else 40)
    wgmma_fence()                                                  # instruction_selection: wgmma.fence.sync.aligned
    for k in range(8):
        if causal:   # dQ[:, 64w..] = dS (NM x 128) K[:, 64w..]
            gemm(rdQ, desc(sdS[stage], kmajor, k), desc(sK + 16384*w, mnmajor, k), accumulate=k > 0)
            # instruction_selection: wgmma ... m64n64k16 ss, flags (1, 1, 0, 1); extent: 8
        else:        # dQ^T[64w.., :] = K^T[64w.., :] dS^T
            gemm(rdQ, desc(sK + 16384*w, mnmajor, k), desc(sdS[stage], kmajor, k), accumulate=k > 0)
            # instruction_selection: wgmma ... m64n80k16 ss, flags (1, 1, 1, 0); extent: 8
    wgmma_commit()                                                 # instruction_selection: wgmma.commit_group.sync.aligned
    wgmma_wait(1)                                                  # retires the dV chain
    # instruction_selection: wgmma.wait_group.sync.aligned 1
    arrive(pipe_do.empty[stage])                                   # release dO: every thread
    # instruction_selection: mapa.shared::cluster.u32 (rank 0) + mbarrier.arrive.shared::cluster.b64
    wgmma_fence()                                                  # instruction_selection: wgmma.fence.sync.aligned
    for k in range(KM):                                            # dK += dS^T Q, A from registers
        gemm(rdK, rdS[4k:4k+4], desc(sQ[stage], mnmajor, k), accumulate=True)
        # instruction_selection: wgmma ... m64n128k16 rs, flags (1, 1, 1); extent: KM
    wgmma_commit()                                                 # instruction_selection: wgmma.commit_group.sync.aligned
    wgmma_wait(1)                                                  # retires the dQ chain
    # instruction_selection: wgmma.wait_group.sync.aligned 1
    named_sync(10 + w, 160)                                        # dQEmpty_w: instruction_selection: bar.sync 10|11, 160
    copy_r2s(rdQ, sdQacc[w] at float i*512 + tid_w*4 + j for register 4i + j)
    # instruction_selection: st.shared.v4.u32; extent: 8 / 10
    fence_proxy_async()                                            # instruction_selection: fence.proxy.async.shared::cta
    named_arrive(13 + w, 160)                                      # dQFull_w: instruction_selection: bar.arrive 13|14, 160
    wgmma_wait(0)                                                  # retires the dK chain
    # instruction_selection: wgmma.wait_group.sync.aligned 0
    arrive(pipe_q.empty[stage])                                    # release Q: mapa + mbarrier.arrive.shared::cluster.b64
    stage, phase = advance(stage, phase)
```

Mask constants (per thread, `mask.h:60-68`, `:126-128`): `thread_col_offset` = the thread's first
KV row in the block (`16*warp + lane//4 + 64w`), `seqlenk_col_limit = s_k - n_block*128 -
thread_col_offset`, `thread_row_offset = 2*(lane % 4)` (first Q column), `causal_row_offset =
seqlenk_col_limit - s_q + m*NM + thread_row_offset`; `col0_i = 8i` and `qcol0(c, e) = 8c + e` are
thread-0 coordinates (compile-time constants).

Epilogue (`epilogue_bwd.hpp:163-276`, `Use_TMA` since not Varlen):

```python
def epilogue_store(rdK, rdV, n_block, bidh, bidb):
    rdV16 = cast(rdV, bf16x2); rdK16 = cast(rdK, bf16x2)           # instruction_selection: cvt.rn.bf16x2.f32 x64
    named_sync(1, 256)                                             # EpilogueBarrier: instruction_selection: bar.sync 1, 256
    copy_r2s(rdV16, sdV at (kv row, d))                            # SM90_U32x4_STSM_N
    copy_r2s(rdK16, sdK at (kv row, d))
    # instruction_selection: stmatrix.sync.aligned.x4.m8n8.shared.b16; extent: 8 + 8
    fence_proxy_async()                                            # instruction_selection: fence.proxy.async.shared::cta
    named_arrive(1, 288)                                           # instruction_selection: bar.arrive 1, 288
    if shfl_idx(tid // 32, 0) == 7:                                # last MMA warp (absolute warp 11)
        # instruction_selection: shfl.sync.idx.b32; extent: scalar
        named_sync(1, 288)                                         # instruction_selection: bar.sync 1, 288
        if elect_one():
            copy_s2g(sdV, dV[bidb, n_block*128:+128, bidh, :]); copy_s2g(sdK, dK[...])
            # instruction_selection: cp.async.bulk.tensor.4d.global.shared::cta.bulk_group, box {64, 128} x2 each; extent: 4
            bulk_commit()                                          # instruction_selection: cp.async.bulk.commit_group
    bulk_wait_read(0)                                              # every thread; instruction_selection: cp.async.bulk.wait_group.read 0

def store_zero(n_block, bidh, bidb):                               # causal only (epilogue_bwd.hpp:286-323)
    col8 = (tid % 16) * 8                                          # GmemTiledCopydKV: 256 threads, (16 rows x 16) x 8 elems
    pcol = col8 < d                                                # tdKVpdK / tdKVpdV (one K iteration)
    # instruction_selection: setp.ge.s32 (col8 >= d) feeding or.pred below; extent: scalar
    for k in range(8):                                             # dK first (flash::copy of tdKVrdKV -> tdKVgdK)
        r = tid // 16 + 16*k
        if r < s_k - n_block*128 and pcol:                         # Is_even_MN = Is_even_K = false, no clears
            copy_r2g(0, dK[bidb, n_block*128 + r, bidh, col8 : col8+8])
            # instruction_selection: st.global.v2.u64 (128-bit zero) under setp.ge + or.pred branch; extent: 8
    for k in range(8):                                             # then dV, same predicates
        r = tid // 16 + 16*k
        if r < s_k - n_block*128 and pcol:
            copy_r2g(0, dV[bidb, n_block*128 + r, bidh, col8 : col8+8])
            # instruction_selection: st.global.v2.u64 (128-bit zero) under or.pred branch; extent: 8
```

### Kernel 3: `flash_bwd_convert_dq` (`flash_bwd_postprocess_kernel.h:156-252`)

```python
@kernel(grid=(cdiv(s_q, kBlockM), h, b), block=256, smem=49280 | 61568)   # __launch_bounds__(256, 2)
def flash_bwd_convert_dq(dQaccum, dQ, s_q, d, h, b, softmax_scale):
    sAcc = tile(SMEM, f32, kBlockM*128, offset=0)                  # flat, the main kernel's dQaccum order
    sdQ  = tile(SMEM, bf16, kBlockM*128, offset=kBlockM*512)       # (m, col) element map below
    # causal (Swizzle<3,3,3> o ((8,8),(64,2)):((64,512),(1,4096)), 16 B chunks XOR row % 8):
    #   byte(m, c) = (c // 64) * (kBlockM*128) + m * 128 + ((((c % 64) // 8) ^ (m % 8)) * 16) + (c % 8) * 2
    # non-causal (Swizzle<1,3,3> o ((8,10),(16,8)):((16,128),(1,1280)), 8 x 16 atoms, chunk bit XOR row bit 2):
    #   byte(m, c) = (c // 16) * 2560 + (m // 8) * 256 + (m % 8) * 32
    #                + (((c % 16) // 8) ^ ((m % 8) // 4)) * 16 + (c % 8) * 2
    bar  = tile(SMEM, u64, 1, offset=kBlockM*768 rounded)          # barrier_dQaccum
    if thread_idx == 0:
        mbar_init(bar, 1); arrive_expect_tx(bar, kBlockM*512)
        copy_g2s(dQaccum[bidb, bidh, m_block*kBlockM*128 : +kBlockM*128], sAcc, completion=bar)
        # instruction_selection: mbarrier.init + mbarrier.arrive.expect_tx + cp.async.bulk.shared::cluster.global
        #   .mbarrier::complete_tx::bytes (kBlockM*512 B); extent: 1 each
    cta_sync()                                                     # instruction_selection: bar.sync 0
    wait(bar, 0)                                                   # instruction_selection: try_wait.parity + hint loop
    w, t = thread_idx // 128, thread_idx % 128
    racc = copy_s2r(sAcc[w*kBlockM*64 + i*512 + t*4 : +4] for i)   # register 4i + j of the dQ (dQ^T) C fragment
    # instruction_selection: ld.shared.v2.u64 (128-bit); extent: 8 / 10
    racc = mul(racc, softmax_scale)                                # instruction_selection: mul.ftz.f32; extent: 32 / 40
    r16 = cast(racc, bf16x2)                                       # instruction_selection: cvt.rn.bf16x2.f32; extent: 16 / 20
    copy_r2s(r16, sdQ)    # causal: STSM_N into (m, d); non-causal: STSM_T of the dQ^T fragment into (m, d)
    # instruction_selection: stmatrix.sync.aligned.x4[.trans].m8n8.shared.b16; extent: 4 / 5
    cta_sync()                                                     # instruction_selection: bar.sync 0
    col8 = (thread_idx % 16) * 8                                   # GmemTiledCopy (16 x 16) x 8
    pcol = col8 < d                                                # tdQpdQ
    for i in range(kBlockM // 16):                                 # all shared reads first (EvenM, Is_even_K: no predicate)
        v[i] = copy_s2r(sdQ[thread_idx // 16 + 16*i, col8 : col8+8])
        # instruction_selection: ld.shared.v2.u64 (128-bit); extent: 4 / 5
    for i in range(kBlockM // 16):                                 # then the predicated global stores
        r = thread_idx // 16 + 16*i
        if r < min(s_q - m_block*kBlockM, kBlockM) and pcol:
            copy_r2g(v[i], dQ[bidb, m_block*kBlockM + r, bidh, col8 : col8+8])
            # instruction_selection: st.global.v2.u64 (128-bit) under setp + and/or.pred branch; extent: 4 / 5
```

The postprocess is built with `SeqlenInfo<true>` (always varlen-capable): it reads
`cu_seqlens` only when the pointer is non-null (the 4 `ld.global.u32` sites are dead for
fixed length); the port passes no `cu_seqlens` and keeps the fixed-length path.

## Instruction-selection summary

| op | source | causal | non-causal |
| --- | --- | --- | --- |
| S^T, dP^T | `flash::gemm` ss | 16 x HGMMA.64x64x16 | 16 x HGMMA.64x80x16 |
| dQ / dQ^T | `flash::gemm` ss | 8 x HGMMA.64x64x16 (B tnsp) | 8 x HGMMA.64x80x16 (A tnsp) |
| dV, dK | `flash::gemm` rs | 8 x HGMMA.64x128x16 (A regs, B tnsp) | 10 |
| exp2 | `exp2f` fast-math | FFMA.FTZ + MUFU.EX2 x32 | x40 |
| mask | `mask.h` | FSEL x32 (causal + seqlen_k) | FSEL x40 (seqlen_k) |
| dS^T -> smem | STSM_T | 4 x STSM.16.MT88.4 | 5 |
| dQ -> smem | R2S 128-bit | 8 x STS.128 | 10 |
| dQ -> gmem | bulk reduce | UBLKRED, 2 per m_block | 2 |
| Q, dO, K, V loads | TMA | UTMALDG.4D, 2 per tile | same |
| LSE, dPsum loads | bulk copy | UBLKCP, 1 per tile | same |
| dK, dV store | TMA store | 4 x UTMASTG.4D | same |
| registers | `setmaxnreg` | 24 / 240, entry 168 | same |

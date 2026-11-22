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

This design sketch documents a TIRx port of FlashMLA's (MIT-licensed)
csrc/sm90/decode/sparse_fp8/, csrc/smxx/decode/combine/combine.cu and
csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu. See NOTICE
and licenses/ for upstream attribution.
-->

# FlashMLA SM90 sparse FP8 decode (top-k gathered, split-KV MLA): coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new IR, a builder API,
or a mathematical reference. It shows the TIRx kernels as explicit runtime ABIs and launches;
explicit linear GMEM/SMEM/register storage with byte offsets; warpgroup-role control flow;
primitive directional copies and primitive computation; explicit phase changes and
publication/reuse edges; and hardware instruction selection, read from the source's line-info
PTX export.

The implementation represented by this sketch is maintained in
[`tirx_kernels/ported/flashmla/_sm90_sparse_fp8_decode/`](../../../tirx_kernels/ported/flashmla/_sm90_sparse_fp8_decode/__init__.py)
(`kernel.py`, `launch.py`, `data.py`), with the entry module
`tirx_kernels/ported/flashmla/sparse_fp8_decode_sm90.py`. The scheduler and combine kernels are
the dense-decode port's (`_sm90_dense_decode/sched_meta.py`, `combine.py`), extended with the
branches this path takes. Those modules are the source of truth.

Source: FlashMLA @ 9241ae3ef9bac614dd25e45e507e089f888280e0.
Source export: `.porting/flashmla_sm90_sparse_fp8_decode/source_export/`
(`v32_persistent_h64.ptx`, `v32_persistent_h128.ptx`, `model1_persistent_h64.ptx`,
`model1_persistent_h128.ptx`, `combine.ptx`, `get_decoding_sched_meta.ptx`, with cubins, SASS and
ptxas logs), compiled by `sm90-probe/fmla90s/export_source.sh` with `setup.py`'s nvcc flags (`-O3
--use_fast_math -lineinfo --source-in-ptx --register-usage-level=10`, `-arch=sm_90a`).
`ptx_hist2.py` gives static opcode counts (inline-asm `{op ...;}` included); counts below are
static PTX instruction counts. ptxas, all four main-kernel instantiations: 168 registers (the
entry allocation that the warpgroups' `setmaxnreg` redistributes), 0 spill, 12 barriers.

**In scope.** All four main-kernel instantiations upstream builds (`instantiations/*.cu`):
`KernelTemplate<MODEL_TYPE, NUM_HEADS>` with MODEL_TYPE in {V32, MODEL1} and NUM_HEADS in {64, 128}
(NUM_HEADS = h_q exactly; CLUSTER_SIZE = NUM_M_BLOCKS = NUM_HEADS / 64). The sparse (`topk != -1`)
scheduler branch; the combine kernel with its attention-sink branch (`attn_sink` may be null or
not). Split and non-split requests; invalid indices (-1); MODEL1's `topk_length`, extra KV cache,
extra indices and `extra_topk_length`.

**Out of scope.** The SM100 sparse decode kernels; any h_q other than 64 / 128 (the API
rejects them); `h_kv != 1`; PDL on the main kernel (upstream comments it out); Tile (`Tx`)
primitives.

## Pipeline at a glance

Three kernels, in stream order (the scheduler only on the first call of a sched_meta):

| Kernel | Grid x block | Role-local program | Publication / reuse |
| --- | --- | --- | --- |
| `get_mla_metadata` | 1 x 32 | lanes: per-request block counts (sparse: from topk / topk_length / extra) into SMEM; lane 0: serial greedy partition walk; lanes: copy `num_splits` out | warp shuffles, `bar.warp.sync` |
| `flash_fwd_splitkv_mla_fp8_sparse` | (h_q/64, s_q, num_sm_parts) x 384, cluster (h_q/64, 1, 1) | per CTA (per cluster): one 64-head block of one query token, over the requests / top-k blocks its partition's `DecodingSchedMeta` lists | see the main-kernel table |
| `flash_fwd_mla_combine` | (b*s_q, 1, ceil(h_q/8)) x 256 | warp w: one q head of one (request, s_q): LSE-weighted sum over that request's splits, attention sink folded into the LSE | `griddepcontrol.wait` after the main kernel's `launch_dependents` (PDL) |

Main kernel roles (three warpgroups; `warpgroup_idx = canonical_warp_group_idx()`, a
`shfl.sync.idx` broadcast):

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| WG0 (warps 0-3), `setmaxnreg.inc 192` | per KV block b: P = Q K_b^T (ss, all 64 tokens), mask by `is_kv_valid`, online softmax, S = bf16(P) into registers and `sS`, rO[:, 0:256] += S V_b[:, 0:256] (rs); per request: next-Q TMA, row-sum, O / OAccum, LSE | `bar_k_local_ready[buf]` (+ `bar_k_remote_ready[buf]` in a cluster): wait; named 8 `sScale_and_sS_ready` (arrive), 9 `sScale_and_sS_free` (sync, from the second block), 10 `oBuf_free_and_sL_ready` (sync), 11 `epilogue_r2s_ready` (sync); `bar_k_avail[buf]`: arrive |
| WG1 (warps 4-7), `setmaxnreg.dec 160` | per KV block: rO[:, 256:512] *= sScale, rO += sS V_b[:, 256:512] (ss) | named 8 (sync), 9 (arrive, not after the last block), 10, 11 (sync); `bar_k_avail[buf]`: arrive |
| WG2 (warps 8-11), `setmaxnreg.dec 152`: producer | per KV block: gather the block's 64 (h64) / this CTA's 32 (h128) top-k tokens: FP8 NoPE + scales + bf16 RoPE with cache-hinted 128-bit loads, dequantize to bf16, store into `sK[buf]` in the WGMMA no-swizzle K-major layout (h128: also `st.async` into the peer CTA's `sK[buf]`), `is_kv_valid` | `bar_k_avail[buf]`: wait; `bar_k_local_ready[buf]`: arrive (128); h128: `bar_k_remote_ready[buf]`: own expect-tx 32*D*2 (36864 V32 | 32768 MODEL1), completed by the peer's `st.async` |
| all 384 (h64) / both CTAs (h128) | per request end: `sync_all_threads_in_cluster` | `bar.sync 0` / `barrier.cluster.arrive.relaxed` + `wait.acquire` |

WG0 owns output columns 0..255 and the softmax; WG1 owns columns 256..511 and follows WG0's
rescale factors (`sScale`) and probabilities (`sS`). Two K buffers alternate, `buf_idx =
(block_idx - start_block_idx) % 2`, with one phase bit per buffer packed into `bar_phase_k`.

## Primitive vocabulary

```python
specialize(...)     # compile-time variant selection
launch(...)         # launch topology and attributes
tile(...)           # linear storage: dtype, element count, placement, alignment
reg_tile(...)       # role-local register array (no layout)
at(buf, byte_off)   # address of a byte offset inside a linear buffer
desc(...)           # WGMMA SMEM matrix descriptor (64-bit, fields written explicitly)
peer(addr)          # the peer CTA's address of the same shared byte (generic address ^ 1<<24)

copy_g2s(tmap.coord(...), dst, completion, cache_hint)   # TMA load
copy_s2g(src, tmap.coord(...))                           # TMA store
copy_s2g_bulk(src, dst, bytes)                           # non-tensor bulk async store
copy_g2r(src, dst)     # global load into registers
copy_r2g(src, dst)     # global store
copy_r2s(src, dst)     # shared store (st.shared or stmatrix)
copy_r2peer(src, dst, completion)   # st.async to the peer CTA's shared memory with complete_tx
copy_s2r(src, dst)     # shared load

gemm(acc, a, b_desc, scale_d, instr)    # one wgmma.mma_async (a: descriptor or 4 registers)
fill(dst, value); cast(dst, src, rounding)
add / sub / mul / fma / max / div / exp2 / log2 (dst, ...)   # one PTX float op each
mul_bf16x2(dst, a, b)                   # one packed bf16 multiply
select(dst, pred, a, b); shuffle_xor(dst, src, mask)
```

Schedule operations: `prefetch`, `init`, `fence`, `wait`, `arrive`, `arrive_remote`,
`expect_bytes`, `cta_sync`, `cluster_arrive`, `cluster_wait`, `named_sync`, `named_arrive`,
`pdl_trigger`, `pdl_wait`, `elect_one`, `setmaxnreg`, `wgmma_fence`, `wgmma_commit`,
`wgmma_wait`, `store_commit`, `store_wait`. There are no compound ops: no `softmax`,
`dequant`, `attention`, `rescale`, `combine`, or `schedule`.

## Shared address functions (used by every SMEM operand below)

```python
# sQ, sS, oBuf: 64 x 64 16-bit "SW128 tiles" of 8192 B: row r at r*128, its 16-byte chunk c
# (elements 8c..8c+7) at chunk position c ^ (r % 8)  (GMMA::Layout_K_SW128_Atom, applied to the
# address bits; every tile base below is a multiple of 1024 bytes from the pool base).
def sw128(tile_base, r, c):  return tile_base + r*128 + ((c ^ (r % 8)) * 16)
# sQ: 64 x D (D = HEAD_DIM_K) as D/64 tiles; column d in tile d // 64, chunk (d % 64) // 8.

# sK[buf]: 64 tokens x D, WGMMA "INTERLEAVE" (no swizzle) K-major layout
# (tile_to_shape(GMMA::Layout_INTER_Atom<bf16, K>, (64, 64), Step<1,2>), tiled along D):
# 8 x 8 core matrices of 16 B rows; element (token t, dim d) at bf16 index
#     kidx(t, d) = (d // 8) * 512 + t * 8 + (d % 8)        # bytes: 2 * kidx
# i.e. each 8-dim column group is one contiguous 64 x 16 B = 1024 B slab, tokens consecutive.
# V is K read transposed: V[d][t] = K[t][d] for d < 512 (SmemLayoutKTilesTransposed); WG0's
# V half is dims 0..255 (slabs 0..31), WG1's dims 256..511 (slabs 32..63, byte offset 32768).

# Accumulator / A-fragment ownership (m64nN, f32): thread t of a warpgroup holds rows
# row0 = 16*(t//32) + (t%32)//4 and row1 = row0 + 8 (get_AorC_row_idx, helpers.h:141); for column
# group g (8 columns): registers 4g+0, 4g+1 -> (row0, 8g + 2*(t%4) + {0,1}), 4g+2, 4g+3 -> (row1,
# same). For m64n64 rP that is 32 registers; packed in pairs to bf16 (rS), registers 4kb..4kb+3
# of k-block kb are exactly the rs A fragment of that k-block.

# stmatrix lane address of k-block kb of a 64 x 64 SW128 tile, warp w of the warpgroup:
def lane_addr(tile_base, w, lane, kb):  return sw128(tile_base, 16*w + lane % 16, 2*kb + lane // 16)
```

## Complete sketch

### Host dispatch and launches (`sparse_attn_decode_interface`, csrc/api/sparse_decode.h)

```python
def sparse_decode(q (b, s_q, h_q, d_qk) bf16, kv (num_blocks, page_block_size, 1, bytes_per_token),
                  indices (b, s_q, topk) i32, topk_length (b,)?, attn_sink (h_q,) f32?,
                  extra_kv?, extra_indices (b, s_q, extra_topk)?, extra_topk_length (b,)?,
                  sm_scale, sched_meta):
    model_type = V32 if d_qk == 576 else MODEL1                          # :318-325
    num_sm_parts = max(num_sms // s_q // (h_q // 64), 1)                 # Decode_Sm90_Impl::get_meta
    out = empty(b, s_q, h_q, 512, bf16); lse = empty(b, s_q, h_q, f32)   # :315-316
    if sched_meta.tile_scheduler_metadata is None:                        # :419-440, first call only
        tile_scheduler_metadata = empty(num_sm_parts, 8, i32); num_splits = empty(b + 1, i32)
        launch get_mla_metadata<<<1, 32, 4*(5b+1)>>>(b, s_q, block_size_n=64,
            fixed_overhead_num_blocks=5, topk, extra_topk (0 when absent), topk_length,
            extra_topk_length, seqlens_k=nullptr, ..., num_sm_parts)       # Kernel 1
    lse_accum = empty(b + num_sm_parts, s_q, h_q, f32)                     # :456
    o_accum = empty(b + num_sm_parts, s_q, h_q, 512, f32)                  # :457
    sm_scale_div_log2 = float(sm_scale * LOG_2_E)                          # :387
    host asserts (KernelTemplate::run, .cuh:688-703): h_kv == 1, topk % 64 == 0, d_v == 512,
        h_q % 64 == 0; V32: no extra KV, no topk_length, stride_kv_row == 656;
        MODEL1: stride_kv_row == 584 (and the extra cache's)
    tma_Q = make_tma_copy(SM90_TMA_LOAD, q as (h_q, d_qk, s_q, b) : (stride_q_h_q, 1, stride_q_s_q,
                          stride_q_b), SmemLayoutQ)      # 64 x 64 boxes, SW128 (.cuh:705-716)
    tensor_map_o = cuTensorMapEncodeTiled(bf16, 5D, out, size (64, h_q, 8, s_q, b),
        strides (stride_o_h_q*2, 128, stride_o_s_q*2, stride_o_b*2) B, box (64, 64, 8, 1, 1),
        SWIZZLE_128B, L2_PROMOTION_L2_256B, OOB_FILL_NONE)                  # .cuh:718-743
    launch flash_fwd_splitkv_mla_fp8_sparse<<<(h_q/64, s_q, num_sm_parts), 384,
        sizeof(SharedMemoryPlan)>>> on a cluster (h_q/64, 1, 1), no PDL attribute   # Kernel 2
    launch flash_fwd_mla_combine<bf16, 512, 8, MAX_SPLITS, 256><<<(b*s_q, 1, ceil(h_q/8)), 256>>>
        with programmatic_stream_serialization = 1, attn_sink as given     # Kernel 3
    return out, lse.transpose(1, 2)
```

### Kernel 1: `get_mla_metadata` (get_decoding_sched_meta.cu), sparse branch

Identical to the dense sketch (`flashmla/sm90_dense_decode.md`, Kernel 1) except the
per-request token count, which now takes the runtime `topk != -1` branch:

```python
specialize(block_size_n=64, fixed_overhead_num_blocks=5)            # runtime parameters upstream
    for i in runtime_range(lane, b, step=32):
        if topk == -1:  s_k = copy_g2r(seqlens_k[i])                      # dense (not this path)
        else:
            s_k = copy_g2r(topk_length[i]) if topk_length != nullptr else topk
            # instruction_selection: ld.global.nc.s32 (__ldg), predicated on the pointer;
            #   extent: scalar per i (the loop is unrolled x4 + remainder in the PTX: :33-40)
            if s_k == 0: s_k = 1                                          # selp.b32
            if extra_topk != 0:
                s_k = ceil(s_k, block_size_n)       # ku::ceil: (a+b-1) - (a+b-1) % b; add.s32 + rem.s32 + sub.s32
                s_k += copy_g2r(extra_topk_length[i]) if extra_topk_length != nullptr else extra_topk
                # instruction_selection: ld.global.nc.s32, predicated on the pointer
        seqlens_s[i] = s_k                                                # st.shared.u32
        ... (blocks, totals, the lane-0 walk and the num_splits copy as in the dense sketch)
```

### Kernel 2: `flash_fwd_splitkv_mla_fp8_sparse` (splitkv_mla.cuh, config.h, components/)

```python
specialize(MODEL_TYPE in {V32, MODEL1}, NUM_HEADS in {64, 128})
D = HEAD_DIM_K = 576 (V32) | 512 (MODEL1); HEAD_DIM_V = 512; D_ROPE = 64; D_NOPE = D - 64
QUANT_TILE = 128 (V32) | 64 (MODEL1); NUM_SCALES = 4 (V32, fp32) | 8 (MODEL1: 7 e8m0 + pad)
CLUSTER = NUM_M_BLOCKS = NUM_HEADS / 64; BLOCK_M = TOPK_BLOCK = 64; NUM_K_BUFS = 2
TOKENS_PER_THREAD = 2 if CLUSTER == 1 else 1; TOKENS_PER_ROUND = 32
WG_QK = wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16    # ss, A and B K-major
WG_PV = wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16   # B MN-major (tnspB = 1)
MAX_INIT_VAL = -1e30
BAR_SS_READY, BAR_SS_FREE, BAR_OBUF_SL, BAR_EPI = 8, 9, 10, 11  # NamedBarriers + 8 (CUTLASS user ids)

def kernel(params: SparseAttnDecodeParams (__grid_constant__: b, s_q, h_q, topk, page_block_size,
               sm_scale_div_log2, the strides, and pointers kv, indices, topk_length, attn_sink,
               lse, extra_kv, extra_indices, extra_topk_length, lse_accum, o_accum,
               tile_scheduler_metadata, num_splits, extra_page_block_size),
           tma_params: { tma_Q, tensor_map_o } (__grid_constant__)):
    launch(grid=(CLUSTER, s_q, num_sm_parts), block=384, cluster=(CLUSTER, 1, 1),
           dynamic_smem=sizeof(SharedMemoryPlan), launch_bounds=(384, 1, CLUSTER))
    # instruction_selection: .maxntid 384,1,1 + .minnctapersm 1 + .maxclusterrank CLUSTER
    head_block_idx = ctaid.x if CLUSTER == 2 else 0; s_q_idx = ctaid.y; partition_idx = ctaid.z
    idx_in_cluster = head_block_idx % 2 if CLUSTER == 2 else 0
    warpgroup_idx = shuffle_idx(tid // 128, lane 0)      # canonical_warp_group_idx
    # instruction_selection: shfl.sync.idx.b32 (and the same for canonical_warp_idx_sync)
    t = tid % 128                                         # idx_in_warpgroup
```

```python
    # ---- storage: one linear dynamic pool (`extern __shared__ char wksp_buf[]`), SharedMemoryPlan ----
    # byte offsets (V32 | MODEL1); the PTX addresses every field as wksp_buf + constant.
    smem = tile(u8, 231424 | 208896, space=shared.dyn, align=1024)
    sQ        = at(smem, 0)                  # 73728 | 65536 B: D/64 SW128 tiles (TMA Q)
    u         = at(smem, 73728 | 65536)      # union, 147456 | 133088 B (MODEL1: oAccumBuf, 63*520+512 floats):
    sK[buf]   = at(u, buf * (73728 | 65536)) #   two K buffers, kidx() mapping
    oBuf      = at(u, 0)                     #   epilogue alias: 64 x 512 bf16 as 8 SW128 tiles
    oAccumBuf = at(u, 0)                     #   epilogue alias: 64 x 512 f32, row stride 520 floats
    sS        = at(smem, 221184 | 198656)    # 8192 B (alignas 1024): one SW128 tile, S in bf16
    is_kv_valid[buf] = at(smem, 229376 | 206848) + 64*buf   # bool[2][64]
    sM = at(smem, 229504 | 206976); sL = at(smem, 229760 | 207232)        # f32[64]
    sScale = at(smem, 230016 | 207488); sOScale = at(smem, 230272 | 207744)   # f32[64]
    bar_q = at(smem, 230528 | 208000)
    bar_k_local_ready[buf] = at(smem, 230536 | 208008) + 8*buf
    bar_k_remote_ready[buf] = at(smem, 230552 | 208024) + 8*buf
    bar_k_avail[buf] = at(smem, 230568 | 208040) + 8*buf
    # (the struct ends at 230584 | 208056 and its size rounds up to its 1024 alignment)
    # The extern pool is `.align 16` in the PTX; the source relies on the dynamic pool starting
    # at the CTA shared base, so the SW128 tiles are 1024-aligned. The TIRx pool states the
    # alignment explicitly (align=1024): a deliberate, behaviour-neutral deviation.

    # ---- prologue (.cuh:109-155) ----
    if warp == 0 and elect_one():
        prefetch(tma_Q); prefetch(tensor_map_o)
        # instruction_selection: elect.sync + prefetch.tensormap [param]; extent: 2
    if warp == 0 and elect_one():
        init(bar_q, 1)
        if CLUSTER == 2: for i in 0, 1: init(bar_k_local_ready[i], 128); init(bar_k_remote_ready[i], 1); init(bar_k_avail[i], 4)
        else:            for i in 0, 1: init(bar_k_local_ready[i], 128); init(bar_k_avail[i], 256)
        # instruction_selection: mbarrier.init.shared::cta.b64; extent: 5 (h64) | 7 (h128)
        fence(mbarrier_init, release, cluster)      # fence.mbarrier_init.release.cluster
    cluster_arrive(relaxed)                         # barrier.cluster.arrive.relaxed (also at CLUSTER 1)
    bar_phase_k = 0
    meta = copy_g2r(tile_scheduler_metadata[partition_idx*8 : +8])
    # instruction_selection: ld.global.v4.u32 x2 (every thread; no .nc)
    if meta.begin_req >= b: return
    if warp == 0 and elect_one():
        launch_q_copy(meta.begin_req)
    cluster_wait(acquire)                           # barrier.cluster.wait.acquire

    def launch_q_copy(batch):                       # launch_tma_copy (helpers.h:188)
        for i in static_range(D // 64):
            copy_g2s(tma_Q.coord(64*i, head_block_idx*64, s_q_idx, batch), at(sQ, 8192*i),
                     completion=bar_q, cache_hint=EVICT_FIRST)
            # instruction_selection: cp.async.bulk.tensor.4d.shared::cluster.global
            #   .mbarrier::complete_tx::bytes.L2::cache_hint, hint 0x12F0000000000000;
            #   extent: 9 (V32) | 8 (MODEL1) boxes of 64 x 64
        expect_bytes(bar_q, 64*D*2)
        # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64 _ (after the copies)

    def wait(bar, phase):    # ClusterBarrier::wait
        # instruction_selection: LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 P1, [bar],
        #   phase, 10000000; @P1 bra DONE; bra LAB_WAIT

    def req_info(batch):    # get_cur_req_info (.cuh:164-184), evaluated by every warpgroup
        if MODEL_TYPE == V32:
            total_padded = topk
        else:
            tl = copy_g2r(topk_length[batch]) if topk_length != nullptr else topk
            xtl = copy_g2r(extra_topk_length[batch]) if extra_topk_length != nullptr else extra_topk
            # instruction_selection: ld.global.nc.s32, each predicated on its pointer
            orig_padded = max(ceil(tl, 64), 64); total_padded = orig_padded + ceil(xtl, 64)
            num_orig_blocks = orig_padded // 64
        start = meta.begin_block if batch == meta.begin_req else 0
        end = meta.end_block if batch == meta.end_req else total_padded // 64
        no_split = (not meta.first_splitted) if batch == meta.begin_req else \
                   ((not meta.last_splitted) if batch == meta.end_req else True)

    def sync_cluster():     # sync_all_threads_in_cluster (config.h)
        if CLUSTER == 1: cta_sync()                                 # bar.sync 0
        else: cluster_arrive(relaxed); cluster_wait(acquire)        # barrier.cluster.arrive.relaxed / wait.acquire
```

#### WG0 (`warpgroup_idx == 0`, .cuh:186-365)

```python
    setmaxnreg(inc, 192)                       # setmaxnreg.inc.sync.aligned.u32 192
    rO = reg_tile(f32, 128)                    # m64n256 C fragment (columns 0..255)
    rP = reg_tile(f32, 32); rS = reg_tile(u32, 16)   # m64n64 C fragment; its bf16 rs A fragment
    rAttn_sink = [-inf, -inf]
    if attn_sink != nullptr:
        for i in 0, 1:
            rAttn_sink[i] = mul(copy_g2r(attn_sink[head_block_idx*64 + row_i]), L2E)
            # instruction_selection: ld.global.nc.f32 + mul.ftz.f32 (x 0f3FB8AA3B); extent: 2
    for batch in runtime_range(meta.begin_req, meta.end_req + 1):    # #pragma unroll 1
        args = req_info(batch)
        rL = [0, 0]; rM = [MAX_INIT_VAL, MAX_INIT_VAL]; fill(rO, 0)   # register moves only
        wait(bar_q, (meta.begin_req - batch) & 1)
        for block in runtime_range(args.start, args.end):             # CUTE_NO_UNROLL
            buf = (block - args.start) % 2; ph = (bar_phase_k >> buf) & 1
            wait(bar_k_local_ready[buf], ph)
            if CLUSTER == 2: wait(bar_k_remote_ready[buf], ph)
            # -- P = Q K^T over D (gemm<true, -1>, helpers.h:149) --
            wgmma_fence_operand(rP); wgmma_fence()                    # wgmma.fence.sync.aligned
            for i in static_range(D // 64):
                for k in static_range(4):
                    gemm(rP, desc_SW128(at(sQ, 8192*i)) + 2*k,
                         desc_INTER(at(sK[buf], 2*kidx(0, 64*i + 16*k))),
                         scale_d=(i, k) != (0, 0), instr=WG_QK)
                    # instruction_selection: wgmma ... m64n64k16 (ss), scale-d predicate;
                    #   extent: 36 (V32) | 32 (MODEL1) back to back
            wgmma_commit(); wgmma_wait(0)            # commit_group; wait_group 0 (gemm<true, -1>)
            wgmma_fence_operand(rP)                  # (register-ordering hint, no PTX instruction)
            bar_phase_k ^= 1 << buf
            if block != args.start:
                named_sync(BAR_SS_FREE, 256)        # bar.sync 9, 256 (WG1 done with sS / sScale)

            # -- scale_softmax (.cuh:32-84), rows r = 0, 1 --
            for r in (0, 1):
                for j in 0..15:                     # this thread's 16 values of row r
                    col = 8*(j // 2) + 2*(t % 4) + (j % 2)
                    valid = copy_s2r(is_kv_valid[buf][col])
                    # instruction_selection: ld.shared.v2.u8 (two columns) + setp.eq.s16;
                    #   extent: 8 per row (16 flags)
                    rP[j of r] = select(valid, rP[j of r], -inf)          # selp.f32
                cur_max = max over the 16 values                           # max.ftz.f32
                cur_max = max(cur_max, shuffle_xor(cur_max, 1)); cur_max = max(cur_max, shuffle_xor(cur_max, 2))
                # instruction_selection: shfl.sync.bfly.b32 + max.ftz.f32; extent: 2 + 2
                cur_max = mul(cur_max, sm_scale_div_log2)                   # mul.ftz.f32
                old = rM[r]; rM[r] = max(cur_max, old)                      # max.ftz.f32
                scale_old[r] = exp2(sub(old, rM[r]))                        # sub.ftz + ex2.approx.ftz
                for j in 0..63: rO[j of r] = mul(rO[j of r], scale_old[r])  # mul.ftz.f32; extent: 64
                cur_sum = 0
                for j in 0..15:
                    rP[j] = exp2(sub(mul(rP[j], sm_scale_div_log2), rM[r]))  # mul.ftz + sub.ftz + ex2.approx.ftz
                    rS[...] = cast(rP[j], bf16, rn)                          # cvt.rn.bf16.f32 (scalar)
                    cur_sum = add(cur_sum, rP[j])                            # add.ftz.f32
                rL[r] = fma(rL[r], scale_old[r], cur_sum)                    # fma.rn.ftz.f32
            if t % 4 == 0:
                copy_r2s((scale_old[0], scale_old[1]), sScale[2*(t // 4) : +2])
                # instruction_selection: st.shared.v2.f32 (indexed by quad, not by row)

            # -- save_rPb_to_sP (config.h:136) --
            for kb in static_range(4):
                copy_r2s(rS[4kb : 4kb+4], lane_addr(sS, t // 32, lane, kb))
                # instruction_selection: stmatrix.sync.aligned.x4.m8n8.shared.b16; extent: 4
            fence(proxy_async_shared)               # fence.proxy.async.shared::cta

            # -- rO += S @ V[:, 0:256] (gemm<false, -1>, rs) --
            wgmma_fence_operand(rS, rO); wgmma_fence()
            for k in static_range(4):
                gemm(rO, rS[4k : 4k+4], desc_MN_INTER(at(sK[buf], 2*kidx(16*k, 0))), scale_d=1, instr=WG_PV)
                # instruction_selection: wgmma ... m64n256k16 (rs, tnspB = 1); extent: 4
            wgmma_commit()                          # wgmma.commit_group.sync.aligned
            named_arrive(BAR_SS_READY, 256)         # bar.arrive 8, 256 (after the commit)
            wgmma_wait(0)                           # wgmma.wait_group.sync.aligned 0
            wgmma_fence_operand(rO, rS)
            if CLUSTER == 2:
                arrive_remote(bar_k_avail[buf], cta=0, pred=t == 32)
                arrive_remote(bar_k_avail[buf], cta=1, pred=t == 64)
                # instruction_selection: mapa.shared::cluster.u32 + mbarrier.arrive.shared::cluster.b64 _;
                #   extent: 2 (each predicated)
            else:
                arrive(bar_k_avail[buf])            # mbarrier.arrive.shared::cta.b64 _ (all 128)

        # -- next Q / PDL (.cuh:272-285) --
        if tid // 32 == 0 and elect_one():
            if batch != meta.end_req: launch_q_copy(batch + 1)
            else: pdl_trigger()                     # griddepcontrol.launch_dependents
        # -- row sums (.cuh:288-314) --
        for r in (0, 1):
            rL[r] = add(rL[r], shuffle_xor(rL[r], 1)); rL[r] = add(rL[r], shuffle_xor(rL[r], 2))
            # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32; extent: 2 + 2
        if t % 4 == 0:
            for r in (0, 1): copy_r2s(rL[r], sL[row_r]); copy_r2s(rM[r], sM[row_r])
            # instruction_selection: st.shared.f32; extent: 4
        for r in (0, 1):
            if args.no_split: o_scale[r] = 0 if rL[r] == 0 else div(1, add(rL[r], exp2(sub(rAttn_sink[r], rM[r]))))
            else:             o_scale[r] = 0 if rL[r] == 0 else div(1, rL[r])
            # instruction_selection: setp.eq.ftz.f32 + branch; (sub.ftz + ex2.approx.ftz + add.ftz)
            #   + div.approx.ftz.f32 (__fdividef)
            if t % 4 == 0: copy_r2s(o_scale[r], sOScale[row_r])           # st.shared.f32
        named_sync(BAR_OBUF_SL, 256)                # bar.sync 10, 256
        rL[r] = 1 if rL[r] == 0 else rL[r]          # (dead after this point in the source)
        num_valid = min(h_q - head_block_idx*64, 64)
        if not args.no_split:                       # .cuh:345-346, before store_o
            split_idx = copy_g2r(num_splits[batch]) + (meta.begin_split if batch == meta.begin_req else 0)
            # instruction_selection: ld.global.nc.s32 (__ldg)
        store_o(rO, columns 0..255, o_scale, args.no_split, batch, split_idx)      # below
        if args.no_split:
            if tid < num_valid:
                L = copy_s2r(sL[tid])
                lse = +inf if L == 0 else fma(log2(L), ln2, div(copy_s2r(sM[tid]), log2e))
                # instruction_selection: ld.shared.f32 x2; setp.eq.ftz.f32; lg2.approx.ftz.f32
                #   (logf under fast math), div.approx.ftz.f32 (sM / 0f3FB8AA3B), then
                #   fma.rn.ftz.f32 (lg2 x 0f3F317218 + div); st.global.f32
                copy_r2g(lse, lse_ptr[batch*stride_lse_b + s_q_idx*stride_lse_s_q + head_block_idx*64 + tid])
        else:
            if tid < num_valid:
                L = copy_s2r(sL[tid])
                lse = -inf if L == 0 else add(log2(L), copy_s2r(sM[tid]))
                # instruction_selection: lg2.approx.ftz.f32 + add.ftz.f32; st.global.f32
                copy_r2g(lse, lse_accum[split_idx*stride_lse_accum_split + s_q_idx*stride_lse_accum_s_q + head_block_idx*64 + tid])
        store_wait(read=0)                          # cp.async.bulk.wait_group.read 0
        sync_cluster()
```

#### WG1 (`warpgroup_idx == 1`, .cuh:366-453)

```python
    setmaxnreg(dec, 160)                       # setmaxnreg.dec.sync.aligned.u32 160
    rO = reg_tile(f32, 128)                    # columns 256..511
    for batch in runtime_range(meta.begin_req, meta.end_req + 1):    # #pragma unroll 1
        args = req_info(batch); fill(rO, 0)
        for block in runtime_range(args.start, args.end):             # CUTE_NO_UNROLL
            buf = (block - args.start) % 2
            named_sync(BAR_SS_READY, 256)                             # bar.sync 8, 256
            s = copy_s2r(sScale[2*(t // 4) : +2])                     # ld.shared.v2.f32
            for r in (0, 1): for j in 0..63: rO[j of r] = mul(rO[j of r], s[r])   # mul.ftz.f32; extent: 128
            wgmma_fence_operand(rO); wgmma_fence()
            for k in static_range(4):
                gemm(rO, desc_SW128(sS) + 2*k, desc_MN_INTER(at(sK[buf], 2*kidx(16*k, 256))), scale_d=1, instr=WG_PV)
                # instruction_selection: wgmma ... m64n256k16 (ss, tnspB = 1); extent: 4
            wgmma_commit(); wgmma_wait(0)
            wgmma_fence_operand(rO)
            if CLUSTER == 2:
                arrive_remote(bar_k_avail[buf], cta=0, pred=t == 32); arrive_remote(bar_k_avail[buf], cta=1, pred=t == 64)
            else:
                arrive(bar_k_avail[buf])                              # mbarrier.arrive.shared::cta.b64 _
            if block != args.end - 1:
                named_arrive(BAR_SS_FREE, 256)                        # bar.arrive 9, 256
        named_sync(BAR_OBUF_SL, 256)                                  # bar.sync 10, 256
        o_scale = [copy_s2r(sOScale[row_0]), copy_s2r(sOScale[row_1])]   # ld.shared.f32 x2
        if not args.no_split:                       # .cuh:440-441, before store_o
            split_idx = copy_g2r(num_splits[batch]) + (meta.begin_split if batch == meta.begin_req else 0)
            # instruction_selection: ld.global.nc.s32 (__ldg)
        store_o(rO, columns 256..511, o_scale, args.no_split, batch, split_idx)
        store_wait(read=0)                                            # cp.async.bulk.wait_group.read 0
        sync_cluster()
```

#### `store_o<IS_NO_SPLIT>` (config.h:157-240), both attention warpgroups

```python
def store_o(rO, wg, o_scale, no_split, batch, split_idx):
    if no_split:
        # bf16 O through oBuf (64 x 512, 8 SW128 tiles; this warpgroup: tiles 4wg..4wg+3)
        for idx in static_range(16):                 # one 16 x 16 chunk per iteration
            a = [cast2(mul(rO[8idx + 2p], o_scale[p % 2]), mul(rO[8idx + 2p + 1], o_scale[p % 2]), bf16x2, rn)
                 for p in 0..3]
            # instruction_selection: mul.ftz.f32 x8 + cvt.rn.bf16x2.f32 x4 (__float22bfloat162_rn)
            copy_r2s(a, lane_addr(at(oBuf, 8192*(4wg + idx // 4)), t // 32, lane, idx % 4))
            # instruction_selection: stmatrix.sync.aligned.x4.m8n8.shared.b16; extent: 16
        fence(proxy_async_shared)                    # fence.proxy.async.shared::cta
        named_sync(BAR_EPI, 256)                     # bar.sync 11, 256
        if tid == 0:
            copy_s2g(oBuf, tensor_map_o.coord(0, head_block_idx*64, 0, s_q_idx, batch))
            # instruction_selection: cp.async.bulk.tensor.5d.global.shared::cta.bulk_group;
            #   extent: one 64 x 64 x 8 box (rows past h_q are clipped by the map)
            store_commit()                           # cp.async.bulk.commit_group
    else:
        for i in static_range(0, 128, 2):
            row = row0 + 8*((i % 4) >= 2); col = 256*wg + 2*(t % 4) + 8*(i // 4)
            copy_r2s((mul(rO[i], o_scale[..]), mul(rO[i+1], o_scale[..])), at(oAccumBuf, 4*(row*520 + col)))
            # instruction_selection: mul.ftz.f32 x2 + st.shared.v2.f32; extent: 64 in WG1; in WG0
            #   the i = 0 pair is two scalar st.shared.f32 (+0, +4 of oAccumBuf) and the other 63
            #   pairs st.shared.v2.f32 (v32_persistent_h64.ptx:6471, 6475)
        fence(proxy_async_shared)                    # fence.proxy.async.shared::cta
        named_sync(BAR_EPI, 256)                     # bar.sync 11, 256
        if elect_one():                              # one lane of each of the 8 attention warps
            for lr in static_range(8):
                row = 8*lr + tid // 32
                if row < num_valid:
                    copy_s2g_bulk(at(oAccumBuf, 4*row*520),
                                  o_accum + split_idx*stride_o_accum_split + s_q_idx*stride_o_accum_s_q
                                  + (head_block_idx*64 + row)*stride_o_accum_h_q, 2048)
                    # instruction_selection: cp.async.bulk.global.shared::cta.bulk_group, 2048 B;
                    #   extent: up to 8 per warp
            store_commit()                           # cp.async.bulk.commit_group
```

#### WG2: producer (`warpgroup_idx == 2`, .cuh:454-670)

```python
    setmaxnreg(dec, 152)                       # setmaxnreg.dec.sync.aligned.u32 152
    w = shuffle_idx(t // 32, lane 0); lane = t % 32           # shfl.sync.idx.b32
    my_token_base = w*8 + lane % 8              # the 8 tokens of this warp; lane // 8: a 16-dim slice
    for batch in runtime_range(meta.begin_req, meta.end_req + 1):    # CUTE_NO_UNROLL
        args = req_info(batch)
        gIdx = indices + batch*stride_indices_b + s_q_idx*stride_indices_s_q
        gXIdx = extra_indices + batch*stride_extra_indices_b + s_q_idx*stride_extra_indices_s_q
        for rnd in static_range(TOKENS_PER_THREAD):
            if MODEL_TYPE == V32 or args.start < args.num_orig_blocks:
                nxt[rnd] = copy_g2r(gIdx[64*args.start + 32*idx_in_cluster + 32*rnd + my_token_base])
                # instruction_selection: ld.global.nc.s32 (__ldg); extent: 1 per round

        def process_block(block, EXTRA, FIRST_EXTRA):    # .cuh:483-647
            buf = (block - args.start) % 2
            idx_base, page, blk_stride, row_stride, kptr = \
                (gXIdx + 64*(block - num_orig), extra_page_block_size, stride_extra_kv_block,
                 stride_extra_kv_row, extra_kv) if EXTRA else \
                (gIdx + 64*block, page_block_size, stride_kv_block, stride_kv_row, kv)
            rel_block = block - num_orig if EXTRA else block; tlen = xtl if EXTRA else tl
            peer_remote = peer(bar_k_remote_ready[buf])     # generic ^ 1<<24 (cvta + xor.b64 + cvta.to)
            for rnd in static_range(TOKENS_PER_THREAD):
                tok = my_token_base + 32*rnd
                k_nope_base = at(sK[buf], 2*kidx(32*idx_in_cluster + tok, 16*(lane // 8)))
                # -- token index, next one prefetched (.cuh:515-529) --
                if not EXTRA:
                    token = nxt[rnd]
                    if block + 1 != (args.end if MODEL_TYPE == V32 else args.num_orig_blocks):
                        nxt[rnd] = copy_g2r(gIdx[64*(block+1) + 32*idx_in_cluster + tok])   # ld.global.nc.s32
                else:
                    token = copy_g2r(gXIdx[64*(block - num_orig) + 32*idx_in_cluster + tok]) if FIRST_EXTRA else nxt[rnd]
                    # instruction_selection: ld.global.nc.s32 (first extra block only)
                    if block + 1 != args.end:
                        nxt[rnd] = copy_g2r(gXIdx[64*(block+1-num_orig) + 32*idx_in_cluster + tok])
                if MODEL_TYPE == MODEL1 and 64*rel_block + 32*idx_in_cluster + tok >= tlen:
                    token = -1                                                  # selp.b32
                block_index = 0 if token == -1 else u32(token) // u32(page)     # div.u32 (branch-guarded)
                rel_idx = u32(token) % u32(page)                                # rem.u32
                # -- scales (.cuh:541-562) --
                if MODEL_TYPE == V32:
                    gK = kptr + block_index*blk_stride + rel_idx*row_stride
                    sf = copy_g2r(gK[512 : 528] as f32x4)
                    # instruction_selection: ld.global.nc.L1::evict_last.L2::128B.v4.s32; extent: 1
                    scales = [cast(sf[i], bf16, rn) for i in 0..3]               # cvt.rn.bf16.f32 x4
                else:
                    gK = kptr + block_index*blk_stride + rel_idx*576
                    e = copy_g2r(kptr[block_index*blk_stride + page*576 + 8*rel_idx : +8] as u64)
                    # instruction_selection: ld.global.nc.s64 (__ldg of int64_t); extent: 1
                    scales = [e8m0_to_bf16(byte i of e) for i in 0..6]
                    # instruction_selection: __nv_cvt_e8m0x2_to_bf162raw in software on sm_90:
                    #   per byte: and/shl.b16 ((e << 7) & 0x7F80), setp.eq.s16 e == 0 / e == 255,
                    #   selp.b16 0x0040 / 0x7FFF; extent: 7 (the 8th byte is padding)
                if rnd == 0:
                    wait(bar_k_avail[buf], ((bar_phase_k >> buf) & 1) ^ 1)   # try_wait.parity loop
                if CLUSTER == 2 and rnd == 0 and t == 0:
                    expect_bytes(bar_k_remote_ready[buf], 32*D*2)   # 32*(HEAD_DIM_NOPE+HEAD_DIM_ROPE)*2
                    # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64 _, 36864 (V32) |
                    #   32768 (MODEL1); extent: 1 per block (one per inlined process_block copy)
                if token == -1: scales = [bf16(0)] * NUM_SCALES                 # cvt.rn.bf16.f32 of 0
                # -- NoPE: gather, dequantize, store (.cuh:576-598) --
                for di in static_range(D_NOPE // 64):          # 8 (V32) | 7 (MODEL1)
                    x = copy_g2r(gK[16*(lane // 8) + 64*di : +16] as fp8x16)
                    # instruction_selection: ld.global.nc.L1::evict_last.L2::256B.v4.s32; extent: 1
                    if token == -1: x = 0                                       # selp.b32 x4
                    s2 = (scales[di // 2 if V32 else di],) * 2                  # mov.b32 {s, s}
                    for half in 0, 1:                                           # 8 values each
                        for q4 in 0, 1:                                         # fp8x4 groups
                            for p in 0, 1:
                                h2 = cvt(x bytes 2p..2p+1 of the group, f16x2 <- e4m3x2)
                                # instruction_selection: cvt.rn.f16x2.e4m3x2 (from cuda_fp8.hpp)
                                f = (cvt(h2.lo, f32 <- f16), cvt(h2.hi, f32 <- f16))   # cvt.f32.f16 x2
                            b2 = cast2(f pair, bf16x2, rn)                       # cvt.rn.bf16x2.f32 x2
                            y = mul_bf16x2(b2, s2)                               # mul.bf16x2 x2
                        copy_r2s(y (8 bf16), at(k_nope_base, 2*64*(64*di + 8*half)))
                        # instruction_selection: st.shared.v2.u64 (16 B, bfi.b64 packing); extent: 1
                        if CLUSTER == 2:
                            copy_r2peer(y, peer(same address), completion=peer_remote)
                            # instruction_selection: st.async.weak.shared::cluster
                            #   .mbarrier::complete_tx::bytes.v2.s64; extent: 1
                # -- RoPE (.cuh:600-623): bf16, not quantized --
                k_rope_base = at(sK[buf], 2*kidx(32*idx_in_cluster + tok, D_NOPE + 8*(lane // 8)))
                for di in static_range(2):
                    y = copy_g2r(gK[D_NOPE + (16 if V32 else 0) + 2*(8*(lane // 8) + 32*di) : +16] as bf16x8)
                    # instruction_selection: ld.global.nc.L1::evict_last.L2::128B.v4.s32; extent: 1
                    if MODEL_TYPE == MODEL1 and token == -1: y = 0               # selp.b32 x4
                    copy_r2s(y, at(k_rope_base, 2*64*32*di))
                    # instruction_selection: st.shared.v2.u64 (V32 h64 / h128, MODEL1 h128) |
                    #   st.shared.v4.u32 (MODEL1 h64); extent: 1
                    if CLUSTER == 2: copy_r2peer(y, peer(...), completion=peer_remote)   # st.async ... v2.s64
            fence(proxy_async_shared)                  # fence.proxy.async.shared::cta
            if t < 32:
                ix = copy_g2r(idx_base[2*lane : +2])   # ld.global.nc.v2.s32
                v = (ix.x != -1, ix.y != -1) if V32 else \
                    (ix.x != -1 and 64*rel_block + 2*lane < tlen, ix.y != -1 and 64*rel_block + 2*lane + 1 < tlen)
                copy_r2s(v, is_kv_valid[buf][2*lane : +2])          # st.shared.v2.u8
            arrive(bar_k_local_ready[buf])             # mbarrier.arrive.shared::cta.b64 _ (all 128)
            bar_phase_k ^= 1 << buf

        if MODEL_TYPE == V32:
            for block in runtime_range(args.start, args.end): process_block(block, False, False)
        else:
            for block in runtime_range(args.start, min(args.num_orig_blocks, args.end)):
                process_block(block, False, False)
            if args.num_orig_blocks < args.end:
                process_block(max(args.start, args.num_orig_blocks), True, True)
            for block in runtime_range(max(args.start, args.num_orig_blocks) + 1, args.end):
                process_block(block, True, False)
            # (three separately inlined copies of process_block in the PTX; every loop CUTE_NO_UNROLL)
        sync_cluster()
```

### Kernel 3: `flash_fwd_mla_combine` (combine.cu), attention-sink branch

Identical to the dense sketch's Kernel 3 except the `params.attn_sink != nullptr` branch, which
this path takes when a sink is given (between the LSE store and the scale table):

```python
    if attn_sink != nullptr:                      # runtime branch (setp.eq.s64 on the pointer)
        sink = copy_g2r(attn_sink[8*h_block + warp])          # ld.global.nc.f32 (__ldg)
        if global_lse != +inf:                                # setp.neu.ftz.f32
            global_lse = add(global_lse, log2(add(1, exp2(sub(mul(sink, L2E), global_lse)))))
            # instruction_selection: mul.ftz.f32 (x 0f3FB8AA3B) + sub.ftz.f32 + ex2.approx.ftz.f32
            #   + add.ftz.f32 (1.0) + lg2.approx.ftz.f32 + add.ftz.f32 (combine.cu:108)
        else:
            global_lse = +inf if sink == -inf else mul(sink, L2E)
            # instruction_selection: setp.eq.ftz.f32 + mul.ftz.f32 + selp.f32 (:111)
```

## Logical GEMM shapes and owners

| Op | Shape (M x N x K) | A | B | Owner | Per |
| --- | --- | --- | --- | --- | --- |
| QK^T | 64 x 64 x D (D/16 k16 steps) | sQ (SW128, K-major, ss) | sK[buf] (INTER, K-major) | WG0 | KV block |
| PV local | 64 x 256 x 64 (4 x k16) | rS registers (rs) | V = sK[buf] dims 0..255 (INTER, MN-major) | WG0 | KV block |
| PV remote | 64 x 256 x 64 (4 x k16) | sS (SW128, K-major, ss) | V = sK[buf] dims 256..511 | WG1 | KV block |

## WGMMA descriptors (written explicitly)

| name | start | LBO (bytes) | SBO (bytes) | layout (bits 62-63) | k-step advance | PTX constant (`or.b64` with the start field) |
| --- | --- | --- | --- | --- | --- | --- |
| `desc_SW128` (sQ tile i, sS) | tile >> 4 (14 bits) | 16 (field 1) | 1024 | 1 = 128B swizzle | +32 B (+2), next tile +8192 B | 4611686293305360384 = 0x4000004000010000 |
| `desc_INTER` (sK, K-major B of QK^T) | sK[buf] >> 4 | 1024 (field 64: next 8-dim slab) | 128 (field 8: next 8 tokens) | 0 = interleave | +2048 B (+128) per k16 | 34363932672 = 0x0000000800400000 |
| `desc_MN_INTER` (V, MN-major B of PV) | (sK[buf] + 32768*wg) >> 4 | 128 (field 8) | 1024 (field 64) | 0 | +256 B (+16) per k16 (16 tokens) | 274878431232 = 0x0000004000080000 |

WGMMA immediates: QK^T ss `scale-a 1, scale-b 1, tnspA 0, tnspB 0` with the scale-d predicate
`(i, k) != (0, 0)`; PV `..., 1, 1, 0, 1` (rs: `1, 1, 1`) with scale-d 1 (the PV TiledMMAs keep
cute's default `ScaleOut::One`). Evidence: `v32_persistent_h64.ptx`, the `or.b64` constants
before the first QK^T (WG0) and PV (WG1) chains and the per-step `add.s32` on the low word
(+128 for K, +16 for V, +2 for sQ / sS).

## TensorMap fields

| map | gmem dims (innermost first) | strides (bytes) | box | swizzle | L2 promotion | used by |
| --- | --- | --- | --- | --- | --- | --- |
| tma_Q | (D, h_q, s_q, b) | (stride_q_h_q, stride_q_s_q, stride_q_b) x 2 | (64, 64, 1, 1) | 128B | as cute `make_tma_copy` | D/64 loads per request |
| tensor_map_o | (64, h_q, 8, s_q, b) | (stride_o_h_q*2, 128, stride_o_s_q*2, stride_o_b*2) | (64, 64, 8, 1, 1) | 128B | L2_256B | one store per non-split request |

## Storage lifetimes

| region | written by | read by | reuse edge |
| --- | --- | --- | --- |
| `sQ` | TMA (bar_q) | WG0 QK^T | next request's TMA issued by WG0 warp 0 after its last QK^T of the request retired (`wgmma.wait_group 0`) |
| `sK[buf]` | producer `st.shared` (+ peer `st.async`) | WG0 QK^T + PV (V dims 0..255), WG1 PV (dims 256..511) | `bar_k_avail[buf]` (256 local arrivals, or 2 + 2 per CTA pair) after both PVs retired |
| `is_kv_valid[buf]` | producer warp 0 | WG0 softmax | published by `bar_k_local_ready[buf]` |
| `sS`, `sScale` | WG0 (stmatrix + fence; st.shared.v2.f32) | WG1 PV (ss), WG1 rescale | `bar.arrive 8` / `bar.sync 8`; freed by WG1's `bar.arrive 9` / WG0's `bar.sync 9` |
| `sL`, `sM`, `sOScale` | WG0 (t % 4 == 0) | WG0 LSE (tid < num_valid), WG1 o_scale | `bar.sync 10` |
| `oBuf` / `oAccumBuf` (over sK) | both WGs (stmatrix / st.shared.v2.f32) | TMA store / bulk stores | written only after `bar.sync 10`, i.e. after both PVs of the last block; read after `bar.sync 11`; `wait_group.read 0` + the request-end cluster sync before the producer refills sK |

## mbarriers

| barrier | arrive count | expect-tx | arriving threads | waiting threads, phase |
| --- | --- | --- | --- | --- |
| bar_q | 1 | 64*D*2 (after the D/64 copies) | WG0 warp-0 elected lane (first: warp 0 elected lane) | WG0, `(begin_req - batch) & 1` |
| bar_k_local_ready[buf] | 128 | none | all producer threads | WG0, bit `buf` of `bar_phase_k` |
| bar_k_remote_ready[buf] (h128) | 1 | 32*D*2 = 36864 (V32) / 32768 (MODEL1) (producer t == 0, own CTA) | own producer t == 0; tx completed by the peer CTA's `st.async` | WG0, same bit |
| bar_k_avail[buf] | 256 (h64) / 4 (h128) | none | h64: WG0 + WG1 threads; h128: WG0/WG1 t == 32 -> CTA 0, t == 64 -> CTA 1 | producer, bit `buf` ^ 1 |

## TIRx module and benchmark contract

`KERNEL_META` names `flashmla_sm90_sparse_fp8_decode`, `runtime_cuda_archs = ["sm_90a"]`,
reference requirement FlashMLA @ 9241ae3e. `run_test` runs `flash_mla.flash_mla_with_kvcache`
with the upstream test's data (`generate_testcase_for_decode`: NaN in every KV row no index
selects, -1 indices, varlen, zero-length, topk_length, extra KV) and requires: scheduler metadata
(7 fields of every part) and `num_splits` identical; `out` and `lse` bitwise identical from a
NaN-sentinel start; and FlashMLA's own `check_is_allclose` tolerances against the PyTorch
reference. `run_bench` times our main + combine against FlashMLA's call (kernel time, sched_meta
precomputed).

## Static specialization boundary

Compile time: MODEL_TYPE and NUM_HEADS (main kernel), MAX_SPLITS (combine). Everything else (b,
s_q, topk, page block sizes, strides, the optional pointers, num_sm_parts) is runtime, as
upstream.

## Instruction-selection summary (static PTX, V32 h64 | V32 h128 | MODEL1 h64)

| instruction | count | cause |
| --- | --- | --- |
| `wgmma...m64n64k16.f32.bf16.bf16` (ss) | 36 / 36 / 32 | QK^T over D (one inlined copy) |
| `wgmma...m64n256k16.f32.bf16.bf16` | 8 / 8 / 8 | 4 rs (WG0) + 4 ss (WG1) |
| `cvt.rn.f16x2.e4m3x2` / `cvt.f32.f16` / `cvt.rn.bf16x2.f32` (producer) / `mul.bf16x2` | 128 / 256 / 128 / 128 (V32 h64) | 2 rounds x 8 dims x 16 fp8 |
| `ld.global.nc.L1::evict_last.L2::256B.v4.s32` | 16 / 8 / 42 | NoPE gathers (MODEL1: 3 process_block copies x 2 rounds x 7) |
| `ld.global.nc.L1::evict_last.L2::128B.v4.s32` | 6 / 3 / 12 | V32 scales + RoPE; MODEL1 RoPE |
| `st.shared.v2.u64` (producer) | 36 / 18 / 84 (MODEL1 h128: 48) | NoPE 16 B stores; plus the RoPE stores in V32 and MODEL1 h128 (V32 h64: 32 + 4; V32 h128: 16 + 2; MODEL1 h64: 84 NoPE = 3 copies x 2 rounds x 14; MODEL1 h128: 3 x (14 + 2)) |
| `st.shared.v4.u32` (producer) | 0 / 0 / 12 (MODEL1 h128: 0) | MODEL1 h64 RoPE stores: 3 process_block copies x 2 rounds x 2 |
| `st.async...v2.s64` | 0 / 18 / 0 | h128 peer stores |
| `mapa.shared::cluster.u32` + `mbarrier.arrive.shared::cluster.b64` | 0 / 4 / 0 | h128 `bar_k_avail` arrivals |
| `stmatrix.x4` | 36 | sS (4) + O no-split (16 per WG) |
| `cp.async.bulk.tensor.4d` (Q) / `.5d` (O) | 18 / 2 | Q at start + next request; O one per WG copy of store_o |
| `cp.async.bulk.global.shared::cta` | 16 | oAccum rows, 8 per WG copy |
| `bar.sync` / `bar.arrive` named | ids 8..11 | sS ready / free, oBuf+sL, epilogue |
| `ex2.approx.ftz.f32` / `max.ftz.f32` | 36 / 38 | softmax + sink |
| `div.approx.ftz.f32` | 5 | o_scale (2 paths x 2) + LSE |
| `barrier.cluster.arrive.relaxed` / `wait.acquire` | 1+ / 1+ (h64), 4 / 4 (h128) | start + per-request cluster sync |
| `griddepcontrol.launch_dependents` | 1 | last request |
| `setmaxnreg` | inc 192 x1, dec 160, dec 152 | role register split |

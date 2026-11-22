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
csrc/sm90/decode/dense/splitkv_mla.cuh, csrc/smxx/decode/combine/combine.cu and
csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu. See NOTICE
and licenses/ for upstream attribution.
-->

# FlashMLA SM90 dense decode (split-KV MLA): coarse WASP pipeline sketch

This file is a non-executable design sketch. It is not a Python module, a new IR, a builder
API, or a mathematical reference. It shows the TIRx kernels as explicit runtime ABIs and
launches; explicit linear GMEM/SMEM/register storage with byte offsets; role control flow;
primitive directional copies and primitive computation; explicit phase changes and
publication/reuse edges; and hardware instruction selection, read from the source's line-info
PTX export.

The implementation represented by this sketch is maintained in
[`tirx_kernels/ported/flashmla/_sm90_dense_decode/`](../../../tirx_kernels/ported/flashmla/_sm90_dense_decode/__init__.py)
(`sched_meta.py`, `kernel.py`, `combine.py`, `launch.py`), with the entry module
`tirx_kernels/ported/flashmla/dense_decode_sm90.py`. Those modules are the source of truth.

Source: FlashMLA @ 9241ae3ef9bac614dd25e45e507e089f888280e0.
Source export: `.porting/flashmla_sm90_dense_decode/source_export/` (`bf16.ptx`, `fp16.ptx`,
`combine.ptx`, `get_decoding_sched_meta.ptx`, with cubins and SASS), compiled by
`sm90-probe/fmla90/export_source.sh` with `setup.py`'s nvcc flags (`-O3 --use_fast_math
-lineinfo --source-in-ptx --register-usage-level=10`, `-arch=sm_90a`). `bf16_byline.txt` is the
per-source-line instruction histogram the annotations below are read from
(`ptx_byline.py`); `bf16_skel.txt` is the same in emission order. Counts are static PTX
instruction counts of the bf16 instantiation. ptxas: 240 registers, 0 spill, 13 barriers.

**In scope.** The only main-kernel specialization upstream has: `Traits<InputT>` with InputT in
{`bfloat16_t`, `half_t`} (bf16 is shown; fp16 differs only in `f16` WGMMA operand type and
`cvt.rn.f16.f32`), BLOCK_SIZE_M = 64, PAGE_BLOCK_SIZE = 64, HEAD_DIM_K = 576, HEAD_DIM_V = 512,
256 threads, causal and non-causal, split and non-split requests; the dense (`topk == -1`)
scheduler; the combine kernel `<InputT, 512, BLOCK_SIZE_M = 8, MAX_SPLITS in {32, 64, 96, 128,
160}, 256>` without an attention sink (dense decode passes `attn_sink = nullptr`).

**Out of scope.** The sparse (`topk != -1`) scheduler branch, the combine's attention-sink
branch, `prefetch_kv_tiles` (defined but unused upstream), FP8 KV caches, any page size other than
64, and `head_size_k = 512` (accepted by the API check but rejected by
`FLASH_ASSERT(d == 576)` in the launcher). Tile (`Tx`) primitives.

## Pipeline at a glance

Three kernels, in stream order (the scheduler only on the first call of a sched_meta):

| Kernel | Grid x block | Role-local program | Publication / reuse |
| --- | --- | --- | --- |
| `get_mla_metadata` | 1 x 32 | lanes: per-request block counts into SMEM; lane 0: serial greedy partition walk; lanes: copy `num_splits` out | warp shuffles, `bar.warp.sync` |
| `flash_fwd_splitkv_mla` | (ceil(q_seq_per_hk/64), h_k, num_sm_parts) x 256 | per CTA: one 64-row Q block of one KV head, over the requests/blocks its partition's `DecodingSchedMeta` lists | see the main-kernel table |
| `flash_fwd_mla_combine` | (b*s_q, 1, ceil(h_q/8)) x 256 | warp w: one q head row of one (request, s_q): LSE-weighted sum over that request's splits | `griddepcontrol.wait` after the main kernel's `launch_dependents` (PDL) |

Main kernel roles (no producer warpgroup; TMA is issued by thread 0 of a warpgroup):

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| thread 0 | prefetch TensorMaps, init 19 mbarriers; Q TMA (9 boxes); K TMA for the first two blocks of each request; TMA O store; next-request Q TMA | `barrier_Q` (expect-tx 73728), `K0[i]`/`K1[i]` (TMA without expect-tx; the consumer arrives with expect-tx 8192) |
| WG0 (warps 0-3) | per block pair (b, b+1): softmax of P0 = Q K_b^T (rows), rO0 += P0 V_b[:, 0:256] (rs), rO0 += P1 V_{b+1}[:, 0:256] (ss from sP1), P0' = Q K_{b+2}^T (ss/rs); K TMA tiles 0..3 of b+2 into sK0, of b+3 into sK1 | named barriers 8 (sScale0Ready, arrive), 9 (sScale1Ready, sync), 10 (sP0Ready, arrive), 11 (rO1sP0sV0RIssued, sync), 12 (sMInitialized, 128 threads) |
| WG1 (warps 4-7) | per block pair: softmax of P1 = Q K_{b+1}^T, rO1 += P1 V_{b+1}[:, 256:512] (rs), rO1 += P0 V_b[:, 256:512] (ss from sP0), P1' = Q K_{b+3}^T; K TMA tiles 4..8 of b+3 into sK1, of b+2 into sK0 | 8 (sync), 9 (arrive), 10 (sync), 11 (arrive) |
| all 256 | epilogue per request: row-sum reduction, O / OAccum store, LSE | `bar.sync 0`, bulk-async store groups |

WG0 owns output columns 0..255, WG1 columns 256..511 (`rO`, 64 x 256 f32 each). Both own all
64 rows; the row max `sM` is shared in SMEM and the two warpgroups alternate updating it:
WG0 for even blocks, WG1 for odd blocks, publishing the rescale factors through `sScale0` /
`sScale1`.

## Primitive vocabulary

```python
specialize(...)     # compile-time variant selection
launch(...)         # launch topology and attributes
tile(...)           # linear storage: dtype, element count, placement, alignment
reg_tile(...)       # role-local register array (no layout)
at(buf, byte_off)   # address of a byte offset inside a linear buffer
desc(...)           # WGMMA SMEM matrix descriptor (64-bit, fields written explicitly)

copy_g2s(tmap.coord(...), dst, completion, cache_hint)   # TMA load
copy_s2g(src, tmap.coord(...))                           # TMA store
copy_s2g_bulk(src, dst, bytes)                           # non-tensor bulk async store
copy_g2r(src, dst)     # global load into registers
copy_r2g(src, dst)     # global store
copy_r2s(src, dst)     # shared store (st.shared, st (generic), or stmatrix)
copy_s2r(src, dst)     # shared load (ld.shared or ldmatrix)

gemm(acc, a, b_desc, scale_d, instr)    # one wgmma.mma_async (a: descriptor or 4 registers)
fill(dst, value); cast(dst, src, rounding)
add / sub / mul / fma / max / div / exp2 / log2 (dst, ...)   # one PTX float op each
select(dst, pred, a, b); shuffle_xor(dst, src, mask)
```

Schedule operations: `prefetch`, `init`, `fence`, `wait`, `arrive`, `expect_bytes`, `cta_sync`,
`warp_sync`, `named_sync`, `named_arrive`, `pdl_trigger`, `pdl_wait`, `wgmma_fence`,
`wgmma_commit`, `wgmma_wait`, `store_commit`, `store_wait`. There are no compound ops: no
`softmax`, `attention`, `rescale`, `combine`, or `schedule`.

## Shared address functions (used by every SMEM operand below)

```python
# A "tile" is 64 rows x 64 16-bit elements = 8192 B, written by one TMA box with
# CU_TENSOR_MAP_SWIZZLE_128B: row r occupies bytes [r*128, r*128+128); its 16-byte chunk c
# (elements 8c..8c+7) sits at chunk position c ^ (r % 8)   (cute Swizzle<3,4,3> on the byte
# address: addr ^ ((addr & 0x380) >> 3); every tile base is 1024-aligned -- implicitly in the
# source, see the pool declaration below).
def sw128(tile_base, r, c):            # byte offset of chunk c of row r
    return tile_base + r * 128 + ((c ^ (r % 8)) * 16)

# 64 x 576 operands (sQ, sK0, sK1) are 9 such tiles: column d lives in tile d // 64,
# chunk (d % 64) // 8 (cute tile_to_shape(GMMA::Layout_K_SW128_Atom, (64, 576))).
def opnd(base, r, d):  return sw128(base + (d // 64) * 8192, r, (d % 64) // 8) + (d % 8) * 2

# V is K read transposed: V[d][t] = K[t][d] for d < 512 (SmemLayoutV = composition(SmemLayoutK,
# (512, 64) row-major)); the V half h (h = 0: dims 0..255, h = 1: 256..511) is tiles 4h..4h+3.

# ldmatrix/stmatrix lane address for the 16-element k-block kb of a 64 x 64 tile, warp w of
# the warpgroup (rows 16w..16w+15): row = 16w + lane % 16, chunk = 2*kb + lane // 16
def lane_addr(tile_base, w, lane, kb):  return sw128(tile_base, 16*w + lane % 16, 2*kb + lane // 16)
# (PTX: element index (tid_in_wg*32 & 3072) | (tid*64 & 960) | (tid >> 1 & 8), + kb*32 bytes,
#  then the address-bit swizzle; bf16_skel.txt, $L__BB0_29.)

# Accumulator / A-fragment ownership (m64nN, f32): thread t of a warpgroup holds rows
# row0 = 16*(t//32) + (t%32)//4 and row1 = row0 + 8; for column group g (8 columns): registers
# 4g+0, 4g+1 -> (row0, 8g + 2*(t%4) + {0,1}), 4g+2, 4g+3 -> (row1, same columns)
# (get_AorC_row_idx, .cuh:20). For m64n64 rP that is 32 registers; packed in pairs to
# 16-bit, registers 4kb..4kb+3 of the k-block kb are exactly the rs A fragment of that
# k-block (rP_retiled, .cuh:292).
```

## Complete sketch

### Host dispatch and launches (`dense_attn_decode_interface`, csrc/api/dense_decode.h)

```python
def dense_decode(q (b, s_q, h_q, 576), kcache (num_blocks, 64, h_k, 576), seqlens_k, block_table,
                 softmax_scale, is_causal, sched_meta):
    if s_q == 1: is_causal = False                                     # dense_decode.h:71
    q_head_per_hk = h_q // h_k; q_seq_per_hk = s_q * q_head_per_hk
    q = q.view(b, s_q, h_k, q_head_per_hk, 576).transpose(2, 3) \
         .reshape(b, q_seq_per_hk, h_k, 576)                            # :76-77 (GQA fold; a copy
                                                                        #   when the transpose is not trivial)
    num_sm_parts = max(num_sms // h_k // ceil(s_q*h_q // h_k / 64), 1)  # :78
    out = empty(b, h_k, q_seq_per_hk, 512, InputT); lse = empty(b, h_k, q_seq_per_hk, f32)   # :90-91
    if sched_meta.tile_scheduler_metadata is None:                      # :95-113, first call only
        tile_scheduler_metadata = empty(num_sm_parts, 8, i32); num_splits = empty(b + 1, i32)
        launch get_mla_metadata<<<1, 32, 4*(5b+1)>>>(seqlens_k, ..., b, num_sm_parts)  # Kernel 1
    scale_softmax_log2 = float(softmax_scale * M_LOG2E)                 # :139
    total_num_splits = b + num_sm_parts                                 # :164
    lse_accum = empty(total_num_splits, h_k, q_seq_per_hk, f32)         # :165
    out_accum = empty(total_num_splits, h_k, q_seq_per_hk, 512, f32)    # :166
    launch flash_fwd_splitkv_mla<<<(ceil(q_seq_per_hk/64), h_k, num_sm_parts), 256, 230816>>>
        with programmatic_stream_serialization = 1                      # Kernel 2 (.cuh:1336-1351)
    MAX_SPLITS = min(m in (32, 64, 96, 128, 160) if num_sm_parts <= m)   # combine.cu:165-182
    assert num_sm_parts <= 160                                          # host FLASH_ASSERT
    launch flash_fwd_mla_combine<InputT, 512, 8, MAX_SPLITS, 256>
        <<<(b*s_q, 1, ceil(h_q/8)), 256, dynamic_smem 0>>>             # Kernel 3 (combine.cu:192-210)
        with programmatic_stream_serialization = 1, launch_bounds (256)
        # CombineParams (dense_decode.h:187-207): lse / out and lse_accum / out_accum strides
        #   (b: h_k*q_seq_per_hk, s_q: h_q) and (b: h_q*s_q*512, s_q: h_q*512, h_q: 512)
    out = out.view(b, h_k, s_q, q_head_per_hk, 512).transpose(1, 2).reshape(b, s_q, h_q, 512)  # :219
    lse = lse.view(b, h_k, s_q, q_head_per_hk).transpose(2, 3).reshape(b, h_q, s_q)             # :221
```

### Kernel 1: `get_mla_metadata` (get_decoding_sched_meta.cu)

```python
specialize(topk=-1, block_size_n=64, fixed_overhead_num_blocks=5)   # dense_decode.h:103-105
def get_mla_metadata(seqlens_k: i32*, tile_scheduler_metadata: i32* (num_sm_parts x 8),
                     num_splits: i32* (b+1), b: i32, num_sm_parts: i32):
    launch(grid=1, block=32, dynamic_smem=4*(5*b+1), launch_bounds=(32, 1, 1))
    smem = tile(i32, 5*b+1, space=shared.dyn)
    num_blocks_s = at(smem, 0); num_splits_s = at(smem, 4*b); seqlens_s = at(smem, 4*(2b+1))
    first_block_s = at(smem, 4*(3b+1)); last_block_s = at(smem, 4*(4b+1))

    total = 0
    for i in runtime_range(lane, b, step=32):
        s_k = copy_g2r(seqlens_k[i])
        # instruction_selection: ld.global.nc.s32 (__ldg); extent: scalar per i (the b loop is
        #   unrolled x4 + remainder: 5 copies in the PTX). The runtime `topk != -1` (sparse)
        #   branch beside it is dead for dense decode and is not ported.
        seqlens_s[i] = s_k                           # instruction_selection: st.shared.u32
        # last = max(s_k-1, 0) / block_size_n: max.s32 + div.s32 (block_size_n is a runtime
        #   parameter upstream, = 64; kept runtime); the four smem writes: st.shared.u32 x4
        last = max(s_k - 1, 0) // 64                 # first block 0
        num_blocks_s[i] = last + 1; first_block_s[i] = 0; last_block_s[i] = last
        total += last + 1 + 5
    for off in (16, 8, 4, 2, 1):
        total += shuffle_xor(total, off)             # instruction_selection: shfl.sync.bfly.b32 x5 + add.s32
    warp_sync()                                      # bar.warp.sync -1

    if lane == 0:                                     # serial greedy walk
        payload = ceil(total / num_sm_parts) + 5
        req = blk = split = cum = 0; num_splits_s[0] = 0
        for part in runtime_range(num_sm_parts):
            meta = (begin_req=req, begin_block=blk + first_block_s[req], begin_split=split,
                    first_splitted=(blk != 0)); remain = payload
            while req < b:
                left = num_blocks_s[req] - blk
                if remain >= left + 5:
                    cum += split + 1; num_splits_s[req + 1] = cum
                    remain -= left + 5; req += 1; blk = 0; split = 0
                else:
                    if remain - 5 > 0: blk += remain - 5; split += 1; remain = 0
                    break
            meta.end_req = req if blk > 0 else req - 1
            meta.end_block = blk + first_block_s[req] if blk > 0 else \
                             (0 if seqlens_s[req-1] == 0 else last_block_s[req-1] + 1)
            meta.last_splitted = meta.end_block != last_block_s[meta.end_req] + 1 \
                                 and seqlens_s[meta.end_req] != 0
            if meta.begin_req == meta.end_req:
                meta.first_splitted = meta.last_splitted = meta.first_splitted or meta.last_splitted
            # payload: div.s32 (cutlass::ceil_div); the walk's num_blocks_s / first_block_s /
            #   last_block_s / seqlens_s reads: ld.shared.u32; num_splits_s writes: st.shared.u32
            copy_r2g(meta (7 ints; word 8 untouched), tile_scheduler_metadata[part*8 : part*8+8])
            # instruction_selection: st.global.v4.u32 + st.global.v2.u32 + st.global.u32
            #   (fields 0..3, 4..5, 6); extent: one struct per part
        assert req == b and blk == 0 and split == 0
        # instruction_selection: FLASH_DEVICE_ASSERT -> vprintf + trap; the TIRx port keeps the
        #   check and the trap (the message print is diagnostics only)
    warp_sync()
    for i in runtime_range(lane, b + 1, step=32):
        num_splits[i] = num_splits_s[i]              # ld.shared.u32 + st.global.u32
```

### Kernel 2: `flash_fwd_splitkv_mla` (splitkv_mla.cuh)

```python
specialize(InputT in {bf16, f16})
WG_QK  = wgmma.mma_async.sync.aligned.m64n64k16.f32.{bf16|f16}.{bf16|f16}    # K-major A and B
WG_PV  = wgmma.mma_async.sync.aligned.m64n256k16.f32.{bf16|f16}.{bf16|f16}   # B MN-major (tnspB=1)
MAX_INIT_VAL_SM = -1e30; MAX_INIT_VAL = -1e33
BAR_SCALE0, BAR_SCALE1, BAR_P0, BAR_RO1, BAR_SM = 8, 9, 10, 11, 12   # NamedBarriers + 8 (CUTLASS user ids)

def kernel(params: DenseAttnDecodeParams (flattened: b, s_q, q_seq_per_hk, q_head_per_hk, h_k,
                is_causal, scale_softmax_log2, block_table_batch_stride, num_sm_parts and the
                pointers seqlens_k, block_table, tile_scheduler_metadata, num_splits,
                softmax_lse, softmax_lseaccum, oaccum),
           tma_Q, tma_K, tma_O: __grid_constant__ CUtensorMap):
    launch(grid=(ceil(q_seq_per_hk/64), h_k, num_sm_parts), block=256,
           dynamic_smem=230816, launch_bounds=(256, 1, 1), programmatic_stream_serialization=1)
    # instruction_selection: .maxntid 256,1,1 + .minnctapersm 1 (+ .maxclusterrank 1);
    #   the PDL attribute is a launch attribute
    m_block_idx, k_head_idx, partition_idx = ctaid.x, ctaid.y, ctaid.z
    warpgroup_idx = tid // 128        # per-thread shr (not broadcast) in the source PTX
    t = tid % 128                     # idx_in_warpgroup

    # ---- storage: one linear dynamic pool (the extern `wksp_buf`), SharedMemoryPlan offsets ----
    smem = tile(u8, 230816, space=shared.dyn, align=1024)
    # The source declares `extern __shared__ char wksp_buf[]` (PTX `.extern .shared .align 16`)
    # and relies implicitly on the dynamic pool starting at the CTA shared-window base (no static
    # shared memory), so every tile offset below, a multiple of 1024, is 1024-aligned as the
    # 128B swizzle (applied to absolute address bits) requires. The TIRx pool states that
    # alignment explicitly (align=1024): a deliberate, behaviour-neutral deviation.
    sQ = at(smem, 0)          # 73728 B: 9 tiles (Q columns 0..575), opnd() mapping
    sK0 = at(smem, 73728)     # 73728 B: K block (64 tokens x 576), opnd() mapping
    sK1 = at(smem, 147456)
    sP0 = at(smem, 221184)    # 8192 B: one 64x64 tile (P of the even block, InputT)
    sM = at(smem, 229376)     # f32[64] running row max (log2 domain)
    sL = at(smem, 229632)     # f32[128] row-sum reduction workspace ([wg][row])
    sScale0 = at(smem, 230144); sScale1 = at(smem, 230400)   # f32[64] each
    barK0[i] = at(smem, 230656 + 8*i); barK1[i] = at(smem, 230728 + 8*i)   # i = 0..8
    barQ = at(smem, 230800)
    sP1 = at(sQ, 8 * 8192)    # alias: Q tile 8 (Q columns 512..575) after rQ8 is loaded
    sO = at(smem, 73728)      # alias over sK0/sK1, epilogue only

    # ---- prologue (.cuh:994-1021) ----
    if tid == 0:
        for m in (tma_Q, tma_K, tma_O): prefetch(m)
        # instruction_selection: prefetch.tensormap [param]; extent: 3
        init(barQ, 1); for i in 0..8: init(barK0[i], 1); init(barK1[i], 1)
        # instruction_selection: mbarrier.init.shared::cta.b64; extent: 19 (Q, then K0/K1 interleaved)
        fence(proxy_async_shared)
        # instruction_selection: fence.proxy.async.shared::cta (fence_view_async_shared); extent: 1
    cta_sync()
    # instruction_selection: bar.sync 0
    meta = copy_g2r(tile_scheduler_metadata[partition_idx*8 : +8])
    # instruction_selection: ld.global.v4.u32 x2 (every thread; no .nc); extent: 32 B
    if meta.begin_req >= b: return
    phase_Q = phase_K0 = phase_K1 = 0
    launch_q_copy(meta.begin_req)                      # defined below

    def launch_q_copy(batch):                          # .cuh:691-710
        if tid == 0:
            for i in static_range(9):
                copy_g2s(tma_Q.coord(64*i, m_block_idx*64, k_head_idx, batch),
                         at(sQ, 8192*i), completion=barQ, cache_hint=EVICT_FIRST)
                # instruction_selection: cp.async.bulk.tensor.4d.shared::cluster.global
                #   .mbarrier::complete_tx::bytes.L2::cache_hint, hint 0x12F0000000000000;
                #   extent: 9 boxes of 64 x 64 (box (64, 64, 1, 1))
            expect_bytes(barQ, 73728)
            # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64 _ (after the copies)

    def launch_kv_tiles(dst, bars, block_index, tiles, issuer):   # launch_kv_tiles_copy_tma
        if issuer:
            for i in tiles:                                        # static, in the given order
                copy_g2s(tma_K.coord(64*i, 0, k_head_idx, block_index), at(dst, 8192*i),
                         completion=bars[i], cache_hint=EVICT_FIRST)
                # instruction_selection: cp.async.bulk.tensor.4d...L2::cache_hint,
                #   hint 0x12F0000000000000; extent: one 64 x 64 box per tile; no expect-tx here

    def wait_tile(bar, phase):    # ClusterBarrier::wait
        # instruction_selection: LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 P1, [bar],
        #   phase, 0x989680; @P1 bra DONE; bra LAB_WAIT (barrier.h:412-421)

    # ---- per request of this partition (.cuh:1027, #pragma unroll 1) ----
    for batch in runtime_range(meta.begin_req, meta.end_req + 1):
        n_split_idx = meta.begin_split if batch == meta.begin_req else 0
        seqlen_k = copy_g2r(seqlens_k[batch])
        # instruction_selection: ld.global.nc.s32 (__ldg); extent: scalar, every thread
        start_block = meta.begin_block if batch == meta.begin_req else 0
        end_block = meta.end_block if batch == meta.end_req else ceil(seqlen_k / 64)
        is_no_split = (not meta.first_splitted) if batch == meta.begin_req else \
                      ((not meta.last_splitted) if batch == meta.end_req else True)
        if is_causal:                                    # .cuh:1036-1061
            mask_len(row) = (s_q - (m_block_idx*64 + row) // q_head_per_hk - 1)
                            if m_block_idx*64 + row < q_seq_per_hk else 0
            # instruction_selection: div.s32 + not.b32 (get_mask_len); extent: 3 per request
            end_block = min(meta.end_block, ceil((seqlen_k - mask_len(63)) / 64)) \
                        if batch == meta.end_req else ceil((seqlen_k - mask_len(63)) / 64)
            right[r] = min(seqlen_k - mask_len(row_r), end_block * 64)   for this thread's rows r = 0, 1
        else:
            right[0] = right[1] = seqlen_k
        bt = block_table + batch * block_table_batch_stride

        launch_kv_tiles(sK0, barK0, copy_g2r(bt[start_block]), range(0, 9), issuer=tid == 0)
        # instruction_selection: ld.global.nc.s32 (block id, every thread); 9 TMA boxes (thread 0)
        if start_block + 1 < end_block:
            nb_a = copy_g2r(bt[start_block + 1])
            # instruction_selection: ld.global.nc.s32 (every thread, before the thread-0 guard)
            launch_kv_tiles(sK1, barK1, nb_a, range(4, 9), issuer=tid == 0)   # 5 TMA boxes
            nb_b = copy_g2r(bt[start_block + 1])
            # instruction_selection: ld.global.nc.s32 again, same address, every thread: the two
            #   `launch_kv_tiles_copy_tma` calls each `__ldg` the block id, and the TMA `asm volatile`
            #   (memory clobber) between them keeps the compiler from merging the loads
            launch_kv_tiles(sK1, barK1, nb_b, range(0, 4), issuer=tid == 0)   # 4 TMA boxes

        rO = reg_tile(f32, 128); fill(rO, 0); rL = [0, 0]
        # instruction_selection: register initialization only (mov.f32 0f00000000), no memory op
        if tid < 64: copy_r2s(MAX_INIT_VAL_SM, sM[tid])
        # instruction_selection: st.shared.u32 0xF149F2CA; extent: threads 0..63
        wait_tile(barQ, phase_Q); phase_Q ^= 1
        rQ8 = reg_tile(u32, 16)
        for kb in static_range(4):
            copy_s2r(lane_addr(at(sQ, 8*8192), warp_in_wg, lane, kb), rQ8[4kb:4kb+4])
            # instruction_selection: ldmatrix.sync.aligned.x4.m8n8.shared.b16; extent: 4 per thread
            #   (all 256 threads; both warpgroups hold the same rs A fragment of Q tile 8)

        if warpgroup_idx == 0:
            # ======================= WG0 (.cuh:1105-1147) =======================
            rP0 = reg_tile(f32, 32)
            for i in static_range(9):
                if t == 0: expect_bytes(barK0[i], 8192)
                # instruction_selection: mbarrier.arrive.expect_tx.shared::cta.b64 _; extent: 1
                wait_tile(barK0[i], phase_K0)
            phase_K0 ^= 1
            if start_block - 16777216 < end_block:     # always true; kept (source: prevents a spill)
                wgmma_fence_operand(rP0); wgmma_fence()
                # instruction_selection: wgmma.fence.sync.aligned
                for i in static_range(9):              # Q / K0 tiles 0..8
                    for k in static_range(4):
                        gemm(rP0, desc_K(at(sQ, 8192*i)) + 2*k, desc_K(at(sK0, 8192*i)) + 2*k,
                             scale_d=(i, k) != (0, 0), instr=WG_QK)
                        # instruction_selection: wgmma ... m64n64k16 (ss), scale-d predicate p;
                        #   extent: 36 back to back (.cuh:93)
                wgmma_commit()                         # wgmma.commit_group.sync.aligned
                wgmma_fence_operand(rP0)
            named_sync(BAR_SM, 128)
            # instruction_selection: bar.sync 12, 128 (sM written by threads 0..63 visible to WG0)
            wgmma_wait(0)                              # wgmma.wait_group.sync.aligned 0
            block = start_block
            while block < end_block - 2:               # #pragma unroll 1
                wg0_block_pair(block, B0L=False, B1L=False); block += 2
            if block + 1 < end_block:   wg0_block_pair(block, B0L=False, B1L=True)
            elif block < end_block:     wg0_block_pair(block, B0L=True,  B1L=False)
        else:
            # ======================= WG1 (.cuh:1149-1182) =======================
            rP1 = reg_tile(f32, 32)
            if start_block + 1 < end_block:
                qk_pipeline(sK1, barK1, rP1, phase_K1, PHASE_1)     # below; flips phase_K1
                wgmma_wait(0)                          # wgmma.wait_group.sync.aligned 0
            block = start_block
            while block < end_block - 3:
                wg1_block_pair(block, False, False, False); block += 2
            if block + 2 < end_block:
                wg1_block_pair(block, False, False, True); block += 2
                wg1_block_pair(block, True, False, False)
            elif block + 1 < end_block: wg1_block_pair(block, False, True, False)
            elif block < end_block:     wg1_block_pair(block, True, False, False)

        # ---- epilogue (.cuh:1185-1265) ----
        for r in (0, 1):
            rL[r] += shuffle_xor(rL[r], 1); rL[r] += shuffle_xor(rL[r], 2)
            # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32; extent: 4 + 4
        my_row = row0(t)
        if t % 4 == 0:
            copy_r2s(rL[0], sL[my_row + 64*warpgroup_idx]); copy_r2s(rL[1], sL[my_row + 8 + 64*warpgroup_idx])
            # instruction_selection: st.shared.f32 x2
        cta_sync()                                     # bar.sync 0
        if warpgroup_idx == 0:
            rL[0] = add(rL[0], copy_s2r(sL[my_row + 64])); rL[1] = add(rL[1], copy_s2r(sL[my_row + 72]))
            # instruction_selection: ld.shared.f32 + add.ftz.f32; extent: 2
        else:
            if t % 4 == 0:
                sL[my_row] = add(sL[my_row], rL[0]); sL[my_row + 8] = add(sL[my_row + 8], rL[1])
                # instruction_selection: ld.shared.f32 + add.ftz.f32 + st.shared.f32; extent: 2
            warp_sync()                                # bar.warp.sync -1
            rL = [copy_s2r(sL[my_row]), copy_s2r(sL[my_row + 8])]   # ld.shared.f32 x2
        for r in (0, 1):
            rL[r] = select(rL[r] == 0 or isnan(rL[r]), 1.0, rL[r])
            # instruction_selection: setp.equ.ftz.f32 + selp.f32; extent: 2
        if batch + 1 <= meta.end_req:
            launch_q_copy(batch + 1)                   # sQ is free: every QK^T of this request retired
        else:
            pdl_trigger()
            # instruction_selection: griddepcontrol.launch_dependents; extent: every thread
        num_valid = min(q_seq_per_hk - m_block_idx*64, 64)

        if is_no_split:
            # store_o<true> (.cuh:622-658): O = rO / rL, InputT, STSM into a 64 x 512 SW128 tile
            # (8 tiles of 64x64 at sO + 8192*j; this warpgroup's columns are tiles 4wg..4wg+3)
            for i in static_range(128):
                rOb[i] = cast(div(rO[i], rL[(i % 4) >= 2]), InputT, rn)
                # instruction_selection: div.approx.ftz.f32 + cvt.rn.bf16.f32; extent: 128 each
            for j in static_range(16):                 # 16 x (4 matrices of 8x8)
                copy_r2s(rOb[8j : 8j+8] as 4 u32, lane_addr(at(sO, 8192*(4*warpgroup_idx + j//4)),
                                                             warp_in_wg, lane, j % 4))
                # instruction_selection: stmatrix.sync.aligned.x4.m8n8.shared.b16; extent: 16
            fence(proxy_async_shared)                  # fence.proxy.async.shared::cta
            cta_sync()                                 # bar.sync 0
            if tid == 0:
                for j in static_range(8):
                    copy_s2g(at(sO, 8192*j), tma_O.coord(64*j, m_block_idx*64, k_head_idx, batch))
                    # instruction_selection: cp.async.bulk.tensor.4d.global.shared::cta.bulk_group;
                    #   extent: 8 boxes of 64 x 64 (rows >= q_seq_per_hk are clipped by the map)
                store_commit()                         # cp.async.bulk.commit_group
            if tid < num_valid:
                L = copy_s2r(sL[tid])
                lse = +inf if (L == 0 or isnan(L)) else \
                      fma(log2(L), ln2, div(copy_s2r(sM[tid]), log2e))
                # instruction_selection: setp.equ.ftz.f32; lg2.approx.ftz.f32;
                #   div.approx.ftz.f32 (sM / 0f3FB8AA3B); fma.rn.ftz.f32 (x 0f3F317218)
                copy_r2g(lse, softmax_lse[(batch*h_k + k_head_idx)*q_seq_per_hk + m_block_idx*64 + tid])
                # instruction_selection: st.global.f32
            store_wait(read=0)                         # cp.async.bulk.wait_group.read 0 (every thread)
        else:
            split_idx = copy_g2r(num_splits[batch]) + n_split_idx
            # instruction_selection: ld.global.u32 (plain; "no __ldg because of PDL")
            # store_o<false> (.cuh:660-683): fp32 rows of 520 floats (bank-conflict padding)
            for i in static_range(0, 128, 2):
                row = row0(t) + 8*((i % 4) >= 2); col = 256*warpgroup_idx + 2*(t % 4) + 8*(i // 4)
                copy_r2s((div(rO[i], rL[..]), div(rO[i+1], rL[..])), at(sO, 4*(row*520 + col)))
                # instruction_selection: div.approx.ftz.f32 x2 per pair (128); the i = 0 pair is
                #   two scalar st.shared.f32, the other 63 pairs st.shared.v2.f32
            fence(proxy_async_shared)                  # fence.proxy.async.shared::cta
            cta_sync()                                 # bar.sync 0
            if tid < num_valid:
                copy_s2g_bulk(at(sO, 4*tid*520),
                              oaccum + ((split_idx*h_k + k_head_idx)*q_seq_per_hk + m_block_idx*64 + tid)*512*4, 2048)
                # instruction_selection: cp.async.bulk.global.shared::cta.bulk_group, 2048 B;
                #   extent: one row per thread
                store_commit()                         # cp.async.bulk.commit_group
                L = copy_s2r(sL[tid])
                lse = -inf if (L == 0 or isnan(L)) else add(log2(L), copy_s2r(sM[tid]))
                # instruction_selection: setp.equ.ftz.f32; lg2.approx.ftz.f32 + add.ftz.f32
                copy_r2g(lse, softmax_lseaccum[(split_idx*h_k + k_head_idx)*q_seq_per_hk + m_block_idx*64 + tid])
                # instruction_selection: st.global.f32
            store_wait(read=0)                         # cp.async.bulk.wait_group.read 0 (every thread)
        if batch != meta.end_req: cta_sync()           # bar.sync 0
```

#### QK^T pipelined (`warpgroup_cooperative_qkt_gemm<PHASE>`, .cuh:196-254)

```python
# One "tile step" for Q/K column tile i (QKT_GEMM_ONE_TILE): the consumer arms the TMA barrier
# itself, waits, then issues 4 k-steps as one WGMMA batch.
def qk_tile(sK, bars, i, rP, phase, accumulate_first):
    if t == 0: expect_bytes(bars[i], 8192)      # mbarrier.arrive.expect_tx.shared::cta.b64 _
    wait_tile(bars[i], phase)                   # try_wait.parity + 0x989680 loop
    wgmma_fence_operand(rP); wgmma_fence()      # wgmma.fence.sync.aligned
    for k in static_range(4):
        if i != 8:
            gemm(rP, desc_K(at(sQ, 8192*i)) + 2*k, desc_K(at(sK, 8192*i)) + 2*k,
                 scale_d=(accumulate_first or k > 0), instr=WG_QK)
            # instruction_selection: wgmma m64n64k16 (ss), scale-d predicate; extent: 4
        else:
            gemm(rP, rQ8[4k:4k+4], desc_K(at(sK, 8*8192)) + 2*k, scale_d=True, instr=WG_QK)
            # instruction_selection: wgmma m64n64k16 (rs, A = rQ8); extent: 4
    wgmma_commit()                              # wgmma.commit_group.sync.aligned (no wait)
    wgmma_fence_operand(rP)

def qk_pipeline(sK, bars, rP, phase, PHASE):
    if PHASE == PHASE_0:                       # WG0: first 4 tiles of the next even block
        for i in (0, 1, 2, 3):
            qk_tile(sK, bars, i, rP, phase, accumulate_first=(i != 0))
    elif PHASE == PHASE_1:                     # WG1: all 9 tiles, tiles 4..8 first
        for i in (4, 5, 6, 7, 8, 0, 1, 2, 3):
            qk_tile(sK, bars, i, rP, phase, accumulate_first=(i != 4))
        phase ^= 1
    else:                                      # PHASE_2, WG0: the remaining 5 tiles
        for i in (4, 5, 6, 7, 8):
            qk_tile(sK, bars, i, rP, phase, accumulate_first=True)
        phase ^= 1
# (accumulate_ of the ss TiledMMA starts Zero and is set One after its first k-step; the rs
#  TiledMMA starts One. In the PTX each is a `setp.ne.b32 p, r, 0` on a register.)
```

#### WG0 block pair (`wg0_subroutine<B0L, B1L>`, .cuh:742-832)

```python
def wg0_block_pair(block, B0L, B1L):
    start_token = 64 * block
    nxt0 = 0 if block + 2 >= end_block else copy_g2r(bt[block + 2])
    nxt1 = 0 if block + 3 >= end_block else copy_g2r(bt[block + 3])
    # instruction_selection: setp.ge.s32 + predicated ld.global.nc.s32; extent: 2 (every thread)
    rPb = reg_tile(u32, 16)

    # -- wg0_bunch_0<DO_OOB_FILLING = B0L or B1L> (.cuh:331-391), rows r = 0, 1 --
    for r in (0, 1):
        if B0L or B1L:
            for each owned (row_r, column j) of rP0:
                rP0[j] = select(start_token + j < right[r], rP0[j], MAX_INIT_VAL)
                # instruction_selection: setp.lt.s32 + selp.f32; extent: 32 per row pair
        cur_max = max over the 16 owned values (seed MAX_INIT_VAL)
        # instruction_selection: max.ftz.f32; extent: 16 per row
        cur_max = max(cur_max, shuffle_xor(cur_max, 1)); cur_max = max(cur_max, shuffle_xor(cur_max, 2))
        # instruction_selection: shfl.sync.bfly.b32 + max.ftz.f32; extent: 2 + 2
        cur_max = mul(cur_max, scale_log2)                         # mul.ftz.f32
        old = copy_s2r(sM[row_r])                                   # ld.shared.f32
        new_max = max(old, cur_max)                                 # max.ftz.f32
        scale_old = exp2(sub(old, new_max))                         # sub.ftz.f32 + ex2.approx.ftz.f32
        warp_sync()                                                 # bar.warp.sync -1
        if t % 4 == 0:
            copy_r2s(scale_old, sScale0[row_r]); copy_r2s(new_max, sM[row_r])   # st.shared.f32 x2
        for j in owned columns: rO[j] = mul(rO[j], scale_old)      # mul.ftz.f32; extent: 64 per row
        cur_sum = 0
        for j in owned columns, pairs (j, j+1):
            rP0[j] = exp2(sub(mul(rP0[j], scale_log2), new_max))   # mul.ftz + sub.ftz + ex2.approx.ftz
            rPb[...] = cast(rP0[j], InputT, rn)                     # cvt.rn.bf16.f32 (scalar)
            cur_sum = add(cur_sum, add(rP0[j], rP0[j+1]))           # add.ftz.f32 x2 per pair
        rL[r] = fma(rL[r], scale_old, cur_sum)                      # fma.rn.ftz.f32
    named_arrive(BAR_SCALE0, 256)                                   # bar.arrive 8, 256

    if B0L:
        fill_oob_V(sK0, half=0, valid=seqlen_k - start_token)       # below
        fence(proxy_async_shared)                                   # fence.proxy.async.shared::cta
    # rO0 += P0 @ V0[:, 0:256]  (warpgroup_cooperative_pv_gemm_localP, .cuh:284)
    wgmma_fence_operand(rPb, rO)            # register ordering only, no instruction
    wgmma_fence()                           # wgmma.fence.sync.aligned
    for k in static_range(4):
        gemm(rO, rPb[4k:4k+4], desc_MN(at(sK0, 0)) + 128*k, scale_d=True, instr=WG_PV)
        # instruction_selection: wgmma m64n256k16 (rs), tnspB 1, scale-d predicate; extent: 4
    wgmma_commit()                          # wgmma.commit_group.sync.aligned
    wgmma_fence_operand(rO, rPb)
    wgmma_wait(0)                                                   # wgmma.wait_group.sync.aligned 0

    named_sync(BAR_SCALE1, 256)                                     # bar.sync 9, 256
    if not B0L and not B1L:
        launch_kv_tiles(sK0, barK0, nxt0, range(0, 4), issuer=t == 0)   # 4 TMA boxes
    for r in (0, 1):                                                # wg0_scale_rP0 (.cuh:529)
        s1 = copy_s2r(sScale1[row_r])                               # ld.shared.f32
        for j owned: rPb[...] = cast(mul(rP0[j], s1), InputT, rn)   # mul.ftz.f32 + cvt.rn.bf16.f32
    for kb in static_range(4):                                      # save_rPb_to_sP(sP0)
        copy_r2s(rPb[4kb:4kb+4], lane_addr(sP0, warp_in_wg, lane, kb))
        # instruction_selection: stmatrix.sync.aligned.x4.m8n8.shared.b16; extent: 4
    fence(proxy_async_shared)                                       # fence.proxy.async.shared::cta
    named_arrive(BAR_P0, 256)                                       # bar.arrive 10, 256

    if not B0L:
        if B1L:
            fill_oob_V(sK1, half=0, valid=seqlen_k - start_token - 64)
            fence(proxy_async_shared)                               # fence.proxy.async.shared::cta
        named_sync(BAR_RO1, 256)                                    # bar.sync 11, 256
        for r in (0, 1):                                            # wg0_rescale_rO0 (.cuh:553)
            s1 = copy_s2r(sScale1[row_r])                           # ld.shared.f32
            rO[owned] = mul(rO[owned], s1); rL[r] = mul(rL[r], s1)  # mul.ftz.f32; 64 + 1 per row
        # rO0 += P1 @ V1[:, 0:256] (remoteP, .cuh:308): A = sP1 = Q tile 8
        wgmma_fence_operand(rO); wgmma_fence()                      # wgmma.fence.sync.aligned
        for k in static_range(4):
            gemm(rO, desc_K(sP1) + 2*k, desc_MN(at(sK1, 0)) + 128*k, scale_d=True, instr=WG_PV)
            # instruction_selection: wgmma m64n256k16 (ss), tnspA 0, tnspB 1; extent: 4
        wgmma_commit(); wgmma_fence_operand(rO)                     # wgmma.commit_group.sync.aligned

    if not B0L and not B1L:
        qk_pipeline(sK0, barK0, rP0, phase_K0, PHASE_0)             # 4 tile steps, 16 WGMMAs
    if not B0L and not B1L and block + 3 < end_block:               # __builtin_expect(true)
        wgmma_wait(4)                                               # wgmma.wait_group.sync.aligned 4
        launch_kv_tiles(sK1, barK1, nxt1, range(0, 4), issuer=t == 0)
    if not B0L and not B1L:
        qk_pipeline(sK0, barK0, rP0, phase_K0, PHASE_2)             # 5 tile steps (tile 8 rs)
    wgmma_wait(0)                                                   # wgmma.wait_group.sync.aligned 0
```

#### WG1 block pair (`wg1_subroutine<B0L, B1L, B2L>`, .cuh:854-945)

```python
def wg1_block_pair(block, B0L, B1L, B2L):
    start_token = 64 * block
    nxt0 = ...; nxt1 = ...            # as WG0 (ld.global.nc.s32, predicated)
    rP1b = reg_tile(u32, 16)
    named_sync(BAR_SCALE0, 256)                                      # bar.sync 8, 256

    # -- wg1_bunch_0 (.cuh:406-475), tokens of block+1 --
    for r in (0, 1):
        if B1L or B2L:
            rP1[j] = select(start_token + 64 + j < right[r], rP1[j], MAX_INIT_VAL)  # setp.lt.s32 + selp.f32
        elif B0L:
            rP1[all owned] = MAX_INIT_VAL                            # (no instruction: folded)
        cur_max = MAX_INIT_VAL
        for j in owned columns of row_r:
            cur_max = max(cur_max, rP1[j])
            # instruction_selection: max.ftz.f32; extent: 16 per row -- absent in the B0L
            #   instances, where rP1 is the constant MAX_INIT_VAL and the max folds away
        for off in (1, 2):
            cur_max = max(cur_max, shuffle_xor(cur_max, off))
            # instruction_selection: shfl.sync.bfly.b32 + max.ftz.f32; extent: 2 + 2 (also in the
            #   B0L instances, on the folded constant)
        cur_max = mul(cur_max, scale_log2)                           # mul.ftz.f32
        old = copy_s2r(sM[row_r]); new_max = max(old, cur_max)       # ld.shared.f32, max.ftz.f32
        scale_old = exp2(sub(old, new_max))                          # sub.ftz + ex2.approx.ftz
        warp_sync()                                                  # bar.warp.sync -1
        if t % 4 == 0:
            copy_r2s(new_max, sM[row_r]); copy_r2s(scale_old, sScale1[row_r])   # st.shared.f32 x2
        cur_sum = 0
        if not B0L:
            rP1[j] = exp2(sub(mul(rP1[j], scale_log2), new_max)); rP1b[..] = cast(rP1[j], InputT, rn)
            cur_sum = add(cur_sum, add(rP1[j], rP1[j+1]))
            # instruction_selection: mul.ftz + sub.ftz + ex2.approx.ftz, cvt.rn.bf16.f32, add.ftz x2
        s01 = mul(scale_old, copy_s2r(sScale0[row_r]))               # ld.shared.f32 + mul.ftz.f32
        rO[owned] = mul(rO[owned], s01)                              # mul.ftz.f32; 64 per row
        rL[r] = fma(rL[r], s01, cur_sum)                             # fma.rn.ftz.f32
    named_arrive(BAR_SCALE1, 256)                                    # bar.arrive 9, 256

    if not B0L:
        for kb in static_range(4):                                   # save_rPb_to_sP(sP1)
            copy_r2s(rP1b[4kb:4kb+4], lane_addr(sP1, warp_in_wg, lane, kb))   # stmatrix x4; extent 4
        if B1L:
            fill_oob_V(sK1, half=1, valid=seqlen_k - start_token - 64)
            fence(proxy_async_shared)                                # fence.proxy.async.shared::cta
        # rO1 += P1 @ V1[:, 256:512] (localP)
        wgmma_fence_operand(rP1b, rO); wgmma_fence()                 # wgmma.fence.sync.aligned
        for k in static_range(4):
            gemm(rO, rP1b[4k:4k+4], desc_MN(at(sK1, 4*8192)) + 128*k, scale_d=True, instr=WG_PV)
            # instruction_selection: wgmma m64n256k16 (rs), tnspB 1; extent: 4
        wgmma_commit(); wgmma_fence_operand(rO, rP1b)                # wgmma.commit_group.sync.aligned
        if not B1L:
            fence(proxy_async_shared)
            # instruction_selection: fence.proxy.async.shared::cta (makes the stmatrix'd sP1
            #   visible to WG0's async-proxy WGMMA read; with B1L the fill's fence already did)
    named_sync(BAR_P0, 256)                                          # bar.sync 10, 256
    if B0L:
        fill_oob_V(sK0, half=1, valid=seqlen_k - start_token)
        fence(proxy_async_shared)                                    # fence.proxy.async.shared::cta
    # rO1 += P0 @ V0[:, 256:512] (remoteP): A = sP0
    wgmma_fence_operand(rO); wgmma_fence()                           # wgmma.fence.sync.aligned
    for k in static_range(4):
        gemm(rO, desc_K(sP0) + 2*k, desc_MN(at(sK0, 4*8192)) + 128*k, scale_d=True, instr=WG_PV)
        # instruction_selection: wgmma m64n256k16 (ss), tnspB 1; extent: 4
    wgmma_commit(); wgmma_fence_operand(rO)                          # wgmma.commit_group.sync.aligned
    if not B0L: named_arrive(BAR_RO1, 256)                           # bar.arrive 11, 256

    if not B0L and not B1L and not B2L:
        wgmma_wait(1)                                                # wgmma.wait_group 1
        launch_kv_tiles(sK1, barK1, nxt1, range(4, 9), issuer=t == 0)   # thread 128; 5 boxes
    if not B0L and not B1L:
        wgmma_wait(0)                                                # wgmma.wait_group.sync.aligned 0
        launch_kv_tiles(sK0, barK0, nxt0, range(4, 9), issuer=t == 0)   # 5 boxes
    if not B0L and not B1L and not B2L:
        qk_pipeline(sK1, barK1, rP1, phase_K1, PHASE_1)              # 9 tile steps, 36 WGMMAs
    wgmma_wait(0)
    # instruction_selection: wgmma.wait_group.sync.aligned 0 (unconditional: the source keeps it
    #   outside the `if` so nvcc does not serialize the QK^T chain)
```

#### `fill_oob_V` (.cuh:579-597)

```python
def fill_oob_V(sK, half, valid):     # zero V tokens >= valid of V half `half` (NaN guard)
    valid = max(valid, 0)                                            # max.s32
    j = t % 64                                                       # 8-byte column (4 elements)
    for token in runtime_range(valid + t // 64, 64, step=2):
        copy_r2s(0, sw128(at(sK, 8192*(4*half + j // 16)), token, (j % 16) // 2) + 8*(j % 2))
        # instruction_selection: st.u64 [generic address of the swizzled shared byte], 0
        #   (cute writes through a generic pointer). Emitted shape: guard `token0 > 63` skips the
        #   loop; trip count ((((token0 == 63) ? 64 : 63) - token0) >> 1) + 1; the trip % 4 remainder (0..3 stores) is
        #   peeled first, then a 4-way unrolled body: 3 + 4 st.u64 per instance (35 in the kernel,
        #   5 instances)
```

### Kernel 3: `flash_fwd_mla_combine` (combine.cu)

```python
specialize(InputT, HEAD_DIM_V=512, BLOCK_SIZE_M=8, MAX_SPLITS in {32,64,96,128,160}, NUM_THREADS=256)
def combine(params: CombineParams (lse, out, lse_accum, o_accum, num_splits, strides, b, s_q, h_q)):
    launch(grid=(b*s_q, 1, ceil(h_q/8)), block=256, static_smem=4*8*MAX_SPLITS,
           programmatic_stream_serialization=1)
    batch = ctaid.x // s_q; s_q_idx = ctaid.x - batch*s_q; h_block = ctaid.z
    warp, lane = tid // 32, tid % 32
    if warp >= min(8, h_q - 8*h_block): return
    begin = copy_g2r(num_splits[batch]); end = copy_g2r(num_splits[batch + 1])
    # instruction_selection: ld.global.nc.s32 x2 (__ldg, before the PDL wait)
    if end - begin == 1: return
    assert end - begin <= MAX_SPLITS             # FLASH_DEVICE_ASSERT: setp + vprintf + trap
    smem_buf = tile(f32, 8*MAX_SPLITS, space=shared.static)
    pdl_wait()                                   # griddepcontrol.wait
    oacc = o_accum + begin*stride_o_accum_split + s_q_idx*stride_o_accum_s_q
                   + (8*h_block + warp)*stride_o_accum_h_q
    datas = [copy_g2r(oacc[4*lane + 128*i : +4]) for i in 0..3]
    # instruction_selection: ld.global.v4.f32 x4 (plain, "incompatible with PDL")
    for i in static_range(ceil(MAX_SPLITS / 32)):
        split = 32*i + lane
        local_lse[i] = copy_g2r(lse_accum[begin*stride_lse_accum_split + s_q_idx*stride_lse_accum_s_q
                                          + 8*h_block + split*stride_lse_accum_split + warp]) \
                       if split < n else -inf
        # instruction_selection: setp.ge.s32 + predicated ld.global.f32; extent: per i
    max_lse = -inf
    for i in static_range(ceil(MAX_SPLITS / 32)):
        max_lse = max(max_lse, local_lse[i])                         # max.ftz.f32; extent: per i
    for off in (16, 8, 4, 2, 1):
        max_lse = max(max_lse, shuffle_xor(max_lse, off))            # shfl.sync.bfly.b32 + max.ftz.f32; x5
    max_lse = select(max_lse == -inf, 0, max_lse)                    # setp.eq.ftz.f32 + selp.f32
    sum_lse = 0
    for i in static_range(ceil(MAX_SPLITS / 32)):
        sum_lse = add(sum_lse, exp2(sub(local_lse[i], max_lse)))     # sub.ftz + ex2.approx.ftz + add.ftz
    for off in (16, 8, 4, 2, 1):
        sum_lse = add(sum_lse, shuffle_xor(sum_lse, off))            # shfl.sync.bfly.b32 + add.ftz.f32; x5
    global_lse = +inf if (sum_lse == 0 or sum_lse == -inf) else add(log2(sum_lse), max_lse)
    # instruction_selection: setp.eq.ftz.f32 x2 + or.pred; lg2.approx.ftz.f32 + add.ftz.f32
    if lane == 0:
        copy_r2g(div(global_lse, log2e), lse[batch*stride_lse_b + s_q_idx*stride_lse_s_q + 8*h_block + warp])
        # instruction_selection: div.approx.ftz.f32 + st.global.f32
    # (the `params.attn_sink != nullptr` branch is dead for dense decode; not ported)
    for i: smem_buf[warp][32*i + lane] = exp2(sub(local_lse[i], global_lse))
    # instruction_selection: sub.ftz.f32 + ex2.approx.ftz.f32 + st.shared.f32; extent: per i
    warp_sync()                                  # bar.warp.sync -1
    result = 0 (4 x float4)
    for split in runtime_range(n):               # #pragma unroll 1
        s = copy_s2r(smem_buf[warp][split])                          # ld.shared.f32
        for i in 0..3:
            result[i].xyzw = fma(s, datas[i].xyzw, result[i].xyzw)   # fma.rn.ftz.f32 x4
            if split != n - 1:
                datas[i] = copy_g2r(oacc[(split+1)*stride_o_accum_split + 4*lane + 128*i : +4])
                # instruction_selection: ld.global.v4.f32 (plain); extent: 4 per split
    o = out + batch*stride_o_b + s_q_idx*stride_o_s_q + (8*h_block + warp)*stride_o_h_q
    for i in 0..3: copy_r2g(cast(result[i], InputT, rn) x4 as u64, o[4*lane + 128*i])
    # instruction_selection: cvt.rn.bf16.f32 x4 (scalar) + bfi.b64 packing + st.global.u64;
    #   extent: 4 per lane
```

## Logical GEMM shapes and owners

| Op | Shape (M x N x K) | A | B | Owner | Per |
| --- | --- | --- | --- | --- | --- |
| QK^T tile | 64 x 64 x 64 (4 x k16) | sQ tile i (ss) or rQ8 (rs, i = 8) | sK tile i | WG0 (even blocks) / WG1 (odd blocks) | 9 tiles per block |
| PV local | 64 x 256 x 64 (4 x k16) | rPb / rP1b registers (rs) | V half (MN-major) | WG0: V0 half 0; WG1: V1 half 1 | per block pair |
| PV remote | 64 x 256 x 64 (4 x k16) | sP1 (WG0) / sP0 (WG1), K-major SW128 (ss) | V half | WG0: V1 half 0; WG1: V0 half 1 | per block pair |

## WGMMA descriptors (`desc_*`, written explicitly)

| name | start | LBO (bytes) | SBO (bytes) | layout (bits 62-63) | k-step advance | PTX constant (hi word) |
| --- | --- | --- | --- | --- | --- | --- |
| `desc_K(tile)` (Q, K, sP0, sP1) | tile >> 4 (14 bits) | 16 (field 1; ignored by the hardware for swizzled K-major, but part of the emitted constant) | 1024 | 1 = 128B swizzle | +32 B (+2 in the start field) | `or.b64 ..., 4611686293305360384` = 0x4000004000010000 |
| `desc_MN(tile0)` (V half) | first of 4 column tiles | 8192 (next 64 columns) | 1024 (next 8 tokens) | 1 | +2048 B (+128) | `or.b64 ..., 4611686293338849280` = 0x4000004002000000 |

## TensorMap fields (`launch.py`, cute `make_tma_copy` of `run_flash_splitkv_mla_kernel`)

| map | gmem dims (innermost first) | strides (bytes) | box | swizzle | used by |
| --- | --- | --- | --- | --- | --- |
| tma_Q | (576, q_seq_per_hk, h_k, b) | (h_k*576, 576, q_seq_per_hk*h_k*576) x 2 | (64, 64, 1, 1) | 128B | 9 loads per request |
| tma_K | (576, 64, h_k, num_blocks) | (h_k*576, 576, 64*h_k*576) x 2 | (64, 64, 1, 1) | 128B | one load per tile |
| tma_O | (512, q_seq_per_hk, h_k, b) | (512, q_seq_per_hk*512, h_k*q_seq_per_hk*512) x 2 | (64, 64, 1, 1) | 128B | 8 stores per non-split request |

Every map: 16-bit element type (bf16 / f16 by InputT), element strides 1, interleave
`CU_TENSOR_MAP_INTERLEAVE_NONE`, swizzle `CU_TENSOR_MAP_SWIZZLE_128B`, L2 promotion
`CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, OOB fill `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` (out-of-range
elements read as zero: Q rows past q_seq_per_hk; O rows past it are not written)
(cute `make_tma_copy`, copy_traits_sm90_tma.hpp:1033-1035).

## Storage lifetimes

| region | written by | read by | reuse edge |
| --- | --- | --- | --- |
| `sQ` tiles 0..7 | TMA (barQ) | QK^T ss of both WGs | next request's Q TMA after the epilogue's `rL` reduction (`bar.sync 0` passed; every QK^T chain retired) |
| `sQ` tile 8 | TMA (barQ) | ldmatrix into rQ8 (both WGs) | becomes `sP1` |
| `sP1` (= Q tile 8) | WG1 stmatrix of P1 | WG0 PV remote (ss) | published by WG1's `fence.proxy.async` + `bar.arrive 11` (or the fill fence), consumed after `bar.sync 11` |
| `sP0` | WG0 stmatrix of rescaled P0 | WG1 PV remote | `fence.proxy.async` + `bar.arrive 10` / `bar.sync 10` |
| `sK0` tiles 0..3 / 4..8 | TMA by WG0 thread 0 / WG1 thread 128 (`nxt0`) | QK^T (WG0), V halves (WG0 tiles 0..3, WG1 tiles 4..7) | WG0 refills 0..3 after its rO0 += P0 V0 wait and `bar.sync 9`; WG1 refills 4..8 after `wait_group 0` of its remote PV on V0 |
| `sK1` tiles 0..3 / 4..8 | TMA by WG0 (`nxt1`, after `wait_group 4`) / WG1 (after `wait_group 1`) | QK^T (WG1), V halves | as above |
| `sM`, `sScale0/1` | WG0 (even) / WG1 (odd), lane % 4 == 0 | the other WG after `bar.sync 8/9` | per block pair |
| `sL` | both WGs, lane % 4 == 0 | epilogue | per request, `bar.sync 0` |
| `sO` (over sK0/sK1) | STSM (no split) / st.shared.v2 (split) | TMA store / bulk store | `cp.async.bulk.wait_group.read 0` before the next request's K TMA |

## mbarriers

| barrier | offset | arrive count | expect-tx | arriving thread | phase variable |
| --- | --- | --- | --- | --- | --- |
| barQ | 230800 | 1 | 73728 (thread 0, after its 9 copies) | thread 0 | `phase_Q` (all threads) |
| barK0[i] | 230656 + 8i | 1 | 8192 (consumer thread t == 0 before its wait) | WG0 t=0 (tiles 0..8 of block 0 / even blocks), WG0 (PHASE_0/2) | `phase_K0` (WG0 only) |
| barK1[i] | 230728 + 8i | 1 | 8192 | WG1 t=0 (thread 128) | `phase_K1` (WG1 only) |

## TIRx module and benchmark contract

`KERNEL_META` names `flashmla_sm90_dense_decode`, `runtime_cuda_archs = ["sm_90a"]`, reference
requirement FlashMLA @ 9241ae3e. `run_test` runs `flash_mla.flash_mla_with_kvcache` on the
upstream test's data (`generate_test_data`, NaN KV past each length) and requires: scheduler
metadata (7 fields of every part) and `num_splits` identical; `out` and `lse` bitwise identical
from a NaN-sentinel start; and FlashMLA's own `check_is_allclose` tolerances against
`reference_torch`. `run_gpu` times our main + combine against FlashMLA's call (kernel time,
sched_meta precomputed).

## Static specialization boundary

Compile time: InputT, and for the combine MAX_SPLITS. Everything else (b, s_q, h_q, h_k,
causal, seqlens, block table, num_sm_parts) is runtime, as upstream.

## Instruction-selection summary (bf16 main kernel, static PTX)

| instruction | count | cause |
| --- | --- | --- |
| `wgmma...m64n64k16.f32.bf16.bf16` (ss) | 132 | 36 (WG0 request prologue) + 32 (WG0 `(F,F)` pair: PHASE_0 16 + PHASE_2 tiles 4..7 16) + 32 (WG1 request prologue PHASE_1) + 32 (WG1 `(F,F,F)` pair) |
| `wgmma...m64n64k16` (rs, rQ8) | 12 | tile 8 of PHASE_1 (2 copies) and PHASE_2 (1 copy), 4 k-steps each |
| `wgmma...m64n256k16` (rs, tnspB) | 24 | PV local, 4 k-steps x (3 WG0 copies + 3 WG1 copies with `!B0L`) |
| `wgmma...m64n256k16` (ss, tnspB) | 28 | PV remote, 4 k-steps x (2 WG0 copies with `!B0L` + 5 WG1 call sites) |
| `wgmma.fence` / `commit_group` | 41 / 41 | one per tile step or PV chain |
| `wgmma.wait_group` | 17 | per the waits above, per template copy |
| `cp.async.bulk.tensor.4d...L2::cache_hint` (loads) | 59 | Q: 2 x 9 = 18; K at request start: 9 + 5 + 4 = 18; WG0: 4 + 4 = 8 (the `(F,F)` copy); WG1: 3 x 5 = 15 (two runs in `(F,F,F)`, one in `(F,F,T)`) |
| `cp.async.bulk.tensor.4d.global.shared::cta.bulk_group` | 8 | O store |
| `cp.async.bulk.global.shared::cta.bulk_group` | 1 | OAccum row store |
| `mbarrier.try_wait.parity` / `arrive.expect_tx` | 37 / 38 | tile waits / consumer arms + Q |
| `stmatrix.x4` | 40 | save P (4 x 6 copies) + O (16) |
| `ldmatrix.x4` | 4 | rQ8 |
| `bar.sync` named / `bar.arrive` | 16 + 5 x `bar.sync 0` / 14 | ids 8..12 |
| `max.ftz.f32` / `ex2.approx.ftz.f32` | 240 / 208 | softmax (fast math: `fmaxf`, `exp2f`) |
| `div.approx.ftz.f32` | 257 | O / rL (128 + 128) + LSE |
| `fence.proxy.async.shared::cta` | 13 | publication fences |
| `griddepcontrol.launch_dependents` | 1 | last request |

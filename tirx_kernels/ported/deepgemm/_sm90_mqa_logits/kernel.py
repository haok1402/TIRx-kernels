# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90_fp8_mqa_logits` (`deep_gemm/impls/sm90_fp8_mqa_logits.cuh`).

One persistent kernel instantiation per `KernelSpec`: one TMA warp (warp 16, one elected lane)
streams a Q block (Q + per-head weights) into a 3-stage Q ring and every 256-row KV block of the
block's key range (KV + per-row scales) into a 3-stage KV ring; four WGMMA math warpgroups each
compute `m64n128k32` chains (A = 64 KV rows, B = the Q block), reduce ReLU(score) x weight over
the heads, scale by the KV row scale and store the logits.

Every shared-memory buffer is a one-dimensional allocation at the byte offset the source computes
(`.cuh:58-95`); the `kHeadDim`-byte swizzle of Q and KV lives in the TensorMaps and the WGMMA
descriptors.  DeepGEMM builds without fast math, so the kernel compiles with `fp_mode="precise"`
(no FTZ); the head reduction is written in the operation order nvcc emits for the source.
"""

import tirx_kernels.tirx_lite as txl

#: `cute::TMA::CacheHintSm90::EVICT_NORMAL`.
EVICT_NORMAL = 0x1000000000000000
#: `ClusterBarrier::wait`'s suspend-time hint (`cutlass/arch/barrier.h:412`).
TRY_WAIT_TICKS = 0x989680
FULL_MASK = 0xFFFFFFFF

BLOCK_KV = 256
NUM_Q_STAGES = 3
NUM_KV_STAGES = 3
NUM_TMA_THREADS = 128
NUM_MATH_THREADS = 512

_TMA_LOAD_HINT = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
)
#: `cute::SM90::GMMA::LayoutType` per swizzle: B128 = 1, B64 = 2, B32 = 3 (`to_swizzle_cute_type`).
_GMMA_LAYOUT = {128: 1, 64: 2, 32: 3}


def _u32(x):
    return txl.cast(x, "uint32")


def _wait(barrier_ptr, phase):
    """`ClusterTransactionBarrier::wait`: cutlass's one-asm try_wait retry loop (suspend hint 0x989680).

    One asm block, not a TIRx loop around the single-shot `try_wait`, so a loop around a wait stays
    unrollable as in the source (the producer's KV loop is unrolled by 4 + a 3-copy remainder).
    """
    txl.cuda.mbarrier_wait_parity(barrier_ptr, _u32(phase), TRY_WAIT_TICKS)


#: DeepGEMM's JIT builds without fast math: IEEE FP32 (no FTZ, precise division), FMA contraction on.
FP_MODE = {"cuda": {"fp_mode": "precise"}}


def _with_fp_mode(backend_config):
    from tvm_tirx.backend.config import merge_backend_configs

    return merge_backend_configs(FP_MODE, backend_config)


def _elect():
    """`cute::elect_one_sync()`: the returned uint32 is 1 on the elected lane."""
    elected = txl.local_scalar("uint32")
    lane = txl.local_scalar("uint32")
    txl.ptx.elect_sync(lane, elected, txl.uint32(FULL_MASK))
    return elected


def build_kernel(spec, *, backend_config=None):
    """The TIRx `PrimFunc` for one `sm90_fp8_mqa_logits` template instantiation."""
    H, D, BQ = spec.num_heads, spec.head_dim, spec.block_q
    if H not in (32, 64) or D not in (32, 64, 128) or BQ * H != 128:
        raise NotImplementedError(f"unported specialization {spec}")
    num_threads = NUM_TMA_THREADS + NUM_MATH_THREADS
    tma_warp = NUM_MATH_THREADS // 32  # warp 16
    num_k_steps = D // 32
    num_accum = 64  # m64n128 f32 accumulators per thread
    accum_per_row = H // 2  # kNumAccumPerReduce
    weights_per_row = H // 4
    wgmma = "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3"
    is_compressed = spec.is_compressed_logits
    is_bf16 = spec.logits_dtype == "bfloat16"

    # ---- shared memory, byte offsets of `.cuh:58-95` ----
    q_stage = BQ * H * D
    kv_stage = BLOCK_KV * D
    w_stage = BQ * H * 4
    sf_stage = BLOCK_KV * 4
    off_bar = NUM_Q_STAGES * (q_stage + w_stage) + NUM_KV_STAGES * (kv_stage + sf_stage)
    # GMMA descriptor high word: SBO = 8*D >> 4 in bits [32,46), layout type in bits [62,64).
    desc_hi = ((8 * D) >> 4) | (_GMMA_LAYOUT[D] << 30)

    @txl.kernel()
    def sm90_fp8_mqa_logits(
        seq_len_arg: "int32",
        seq_len_kv_arg: "int32",
        max_seqlen_k: "int32",
        stride_logits_arg: "int32",
        cu_seq_len_k_start: txl.Tensor((txl.int64(),), "int32"),
        cu_seq_len_k_end: txl.Tensor((txl.int64(),), "int32"),
        logits: txl.Tensor((txl.int64(),), spec.logits_dtype),
        tensor_map_q: txl.Ptr(txl.TensorMap),
        tensor_map_kv: txl.Ptr(txl.TensorMap),
        tensor_map_kv_scales: txl.Ptr(txl.TensorMap),
        tensor_map_weights: txl.Ptr(txl.TensorMap),
    ):
        # `LaunchArgs(num_sms, 128 + 512, smem_size)`, `__launch_bounds__(640, 1)`.
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=spec.num_sms, block=num_threads),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=_with_fp_mode(backend_config),
        )
        del max_seqlen_k  # only selects kIsCompressedLogits on the host
        seq_len = txl.local_scalar("uint32", init=_u32(seq_len_arg))
        seq_len_kv = txl.local_scalar("uint32", init=_u32(seq_len_kv_arg))
        stride_logits = txl.local_scalar("uint32", init=_u32(stride_logits_arg))
        num_q_blocks = txl.local_scalar(
            "uint32", init=(seq_len + txl.uint32(BQ - 1)) // txl.uint32(BQ)
        )
        thread_idx = _u32(txl.thread_id())

        # ---- prologue: TensorMap prefetch (`.cuh:50-56`) ----
        with txl.if_(thread_idx // txl.uint32(32) == txl.uint32(tma_warp)):
            with txl.then_():
                elected = _elect()
                with txl.if_(elected == txl.uint32(1)):
                    with txl.then_():
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_q))
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_kv))
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_kv_scales))
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_weights))
        txl.ptx.bar.warp.sync(txl.uint32(FULL_MASK))

        # ---- one linear dynamic pool at the source offsets, aligned to 8 * kHeadDim ----
        smem = txl.smem_pool()
        smem_q = smem.alloc((NUM_Q_STAGES * q_stage,), "float8_e4m3fn", align=8 * D)
        smem_kv = smem.alloc((NUM_KV_STAGES * kv_stage,), "float8_e4m3fn", align=8 * D)
        smem_w = smem.alloc((NUM_Q_STAGES * w_stage // 4,), "float32", align=4)
        smem_sf = smem.alloc((NUM_KV_STAGES * sf_stage // 4,), "float32", align=4)
        bars = smem.alloc((2 * NUM_Q_STAGES + 2 * NUM_KV_STAGES,), "uint64", align=8)
        if smem.bytes != off_bar + 8 * (2 * NUM_Q_STAGES + 2 * NUM_KV_STAGES):
            raise AssertionError(f"SMEM layout ends at {smem.bytes}, expected the source's")
        smem.commit(spec.smem_size)

        def full_q(s):
            return bars.ptr_to([s])

        def empty_q(s):
            return bars.ptr_to([s + NUM_Q_STAGES])

        def full_kv(s):
            return bars.ptr_to([s + 2 * NUM_Q_STAGES])

        def empty_kv(s):
            return bars.ptr_to([s + 2 * NUM_Q_STAGES + NUM_KV_STAGES])

        # ---- barrier init (`.cuh:97-114`) ----
        is_tma_load_warp = txl.local_scalar(
            "uint32",
            init=txl.cast(
                txl.And(
                    thread_idx >= txl.uint32(NUM_MATH_THREADS),
                    thread_idx < txl.uint32(NUM_MATH_THREADS + 32),
                ),
                "uint32",
            ),
        )
        with txl.if_(is_tma_load_warp == txl.uint32(1)):
            with txl.then_():
                init_elected = _elect()
                with txl.if_(init_elected == txl.uint32(1)):
                    with txl.then_():
                        for s in range(NUM_Q_STAGES):
                            txl.ptx["mbarrier.init.shared::cta.b64"](full_q(s), txl.uint32(1))
                            txl.ptx["mbarrier.init.shared::cta.b64"](
                                empty_q(s), txl.uint32(NUM_MATH_THREADS)
                            )
                        for s in range(NUM_KV_STAGES):
                            txl.ptx["mbarrier.init.shared::cta.b64"](full_kv(s), txl.uint32(1))
                            txl.ptx["mbarrier.init.shared::cta.b64"](
                                empty_kv(s), txl.uint32(NUM_MATH_THREADS)
                            )
                        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))

        # ---- block schedule (`.cuh:120-154`) ----
        block_q_idx = txl.local_scalar("uint32", init=_u32(txl.cta_id()))
        q_iter_idx = txl.local_scalar("uint32", init=txl.uint32(0))
        num_total_kv_blocks = txl.local_scalar("uint32", init=txl.uint32(0))
        seq_k_start = txl.alloc_local((BQ,), "uint32")
        seq_k_end = txl.alloc_local((BQ,), "uint32")
        kv_start = txl.local_scalar("uint32")
        num_kv_blocks = txl.local_scalar("uint32")
        q_stage_idx = txl.local_scalar("uint32")
        q_phase = txl.local_scalar("uint32")

        def load_schedule(q_iter_offset):
            """`load_schedule` (`.cuh:127-144`): per-row ranges, the block's KV window."""
            start = txl.local_scalar("uint32", init=txl.uint32(0xFFFFFFFF))
            end = txl.local_scalar("uint32", init=txl.uint32(0))
            for i in range(BQ):
                q_idx = txl.min(block_q_idx * txl.uint32(BQ) + txl.uint32(i), seq_len - 1)
                txl.ptx["ld.global.u32"](seq_k_start[i], cu_seq_len_k_start.ptr_to([q_idx]))
                txl.ptx["ld.global.u32"](seq_k_end[i], cu_seq_len_k_end.ptr_to([q_idx]))
                txl.assign(start, txl.min(start, txl.min(seq_k_start[i], seq_len_kv)))
                txl.assign(end, txl.max(end, txl.min(seq_k_end[i], seq_len_kv)))
            # TMA alignment requirement of the KV-scale load
            txl.assign(kv_start, start // txl.uint32(4) * txl.uint32(4))
            it = q_iter_idx + txl.uint32(q_iter_offset)
            txl.assign(q_stage_idx, it % txl.uint32(NUM_Q_STAGES))
            txl.assign(q_phase, it // txl.uint32(NUM_Q_STAGES) & txl.uint32(1))
            txl.assign(
                num_kv_blocks,
                (end - kv_start + txl.uint32(BLOCK_KV - 1)) // txl.uint32(BLOCK_KV),
            )

        def kv_pipeline(kv_block_idx):
            """`get_kv_pipeline` (`.cuh:148-153`): stage and phase of a KV block."""
            it = num_total_kv_blocks + kv_block_idx
            return it % txl.uint32(NUM_KV_STAGES), it // txl.uint32(NUM_KV_STAGES) & txl.uint32(1)

        txl.ptx.griddepcontrol.wait()

        with txl.if_(thread_idx >= txl.uint32(NUM_MATH_THREADS)):
            # ================= TMA warpgroup (`.cuh:158-205`) =================
            with txl.then_():
                txl.ptx["setmaxnreg.dec.sync.aligned.u32"](txl.uint32(32))
                # Only the first warp remains; warps 17-19 fall through to the kernel end.
                with txl.if_(is_tma_load_warp == txl.uint32(1)):
                    with txl.then_():
                        _producer(
                            spec,
                            maps=(
                                tensor_map_q,
                                tensor_map_kv,
                                tensor_map_kv_scales,
                                tensor_map_weights,
                            ),
                            smem=(smem_q, smem_kv, smem_w, smem_sf),
                            bars=(full_q, empty_q, full_kv, empty_kv),
                            sched=dict(
                                block_q_idx=block_q_idx,
                                q_iter_idx=q_iter_idx,
                                num_q_blocks=num_q_blocks,
                                num_total_kv_blocks=num_total_kv_blocks,
                                kv_start=kv_start,
                                num_kv_blocks=num_kv_blocks,
                                q_stage_idx=q_stage_idx,
                                q_phase=q_phase,
                                load_schedule=load_schedule,
                                kv_pipeline=kv_pipeline,
                            ),
                        )

            # ================= math warpgroups (`.cuh:206-327`) =================
            with txl.else_():
                txl.ptx["setmaxnreg.inc.sync.aligned.u32"](txl.uint32(112))
                # NOTES: `__shfl_sync` so NVCC can keep the warp index in a uniform register.
                warp_idx = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.idx.b32(
                    warp_idx,
                    thread_idx % txl.uint32(NUM_MATH_THREADS) // txl.uint32(32),
                    txl.uint32(0),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                warpgroup_idx = warp_idx // txl.uint32(4)
                lane_idx = txl.local_scalar("uint32", init=_u32(txl.lane_id()))
                accum = txl.alloc_local((num_accum,), "float32")
                weights = txl.alloc_local((BQ * weights_per_row,), "float32")
                warp_offset = txl.local_scalar("uint32", init=warp_idx * txl.uint32(16))
                v_0_offset = txl.local_scalar("uint32", init=lane_idx // txl.uint32(4))
                v_1_offset = txl.local_scalar(
                    "uint32", init=lane_idx // txl.uint32(4) + txl.uint32(8)
                )
                q_lo = txl.local_scalar(
                    "uint32", init=_u32(txl.cuda.cvta_generic_to_shared(smem_q.ptr_to([0])))
                )
                kv_lo = txl.local_scalar(
                    "uint32",
                    init=_u32(
                        txl.cuda.cvta_generic_to_shared(
                            smem_kv.ptr_to([warpgroup_idx * txl.uint32(64 * D)])
                        )
                    ),
                )
                desc_hi64 = txl.uint64(desc_hi << 32)
                zero = txl.float32(0.0)

                with txl.while_(block_q_idx < num_q_blocks):
                    load_schedule(0)
                    _wait(full_q(q_stage_idx), q_phase)
                    w_base = q_stage_idx * txl.uint32(w_stage // 4)
                    for i in range(BQ):
                        for j in range(weights_per_row):
                            txl.ptx["ld.shared.f32"](
                                weights[i * weights_per_row + j],
                                smem_w.ptr_to(
                                    [
                                        w_base
                                        + txl.uint32(i * H + (j // 2) * 8 + (j & 1))
                                        + lane_idx % txl.uint32(4) * txl.uint32(2)
                                    ]
                                ),
                            )

                    # `q_offset` of each row is invariant over the KV blocks; nvcc keeps it out of
                    # the KV loop for the source, so it is formed once per Q block here.
                    q_offsets = txl.alloc_local((BQ,), "uint64")
                    for i in range(BQ):
                        txl.assign(
                            q_offsets[i],
                            txl.cast(block_q_idx * txl.uint32(BQ) + txl.uint32(i), "uint64")
                            * txl.cast(stride_logits, "uint64"),
                        )

                    with txl.serial(0, num_kv_blocks, unroll=True, dtype="uint32") as kv_block_idx:
                        kv_stage_idx, kv_phase = kv_pipeline(kv_block_idx)
                        kv_stage_l = txl.local_scalar("uint32", init=kv_stage_idx)
                        _wait(full_kv(kv_stage_l), kv_phase)
                        scale_kv_0 = txl.local_scalar("float32")
                        scale_kv_1 = txl.local_scalar("float32")
                        sf_base = kv_stage_l * txl.uint32(sf_stage // 4) + warp_offset
                        txl.ptx["ld.shared.f32"](scale_kv_0, smem_sf.ptr_to([sf_base + v_0_offset]))
                        txl.ptx["ld.shared.f32"](scale_kv_1, smem_sf.ptr_to([sf_base + v_1_offset]))

                        for r in range(num_accum):
                            txl.cuda.wgmma.noop_barrier(accum[r])
                        txl.ptx.wgmma.fence.sync.aligned()
                        a_base = kv_lo + kv_stage_l * txl.uint32(kv_stage)
                        b_base = q_lo + q_stage_idx * txl.uint32(q_stage)
                        for k in range(num_k_steps):
                            a_desc = desc_hi64 | txl.cast(
                                (a_base + txl.uint32(32 * k)) >> txl.uint32(4), "uint64"
                            )
                            b_desc = desc_hi64 | txl.cast(
                                (b_base + txl.uint32(32 * k)) >> txl.uint32(4), "uint64"
                            )
                            txl.ptx[wgmma](
                                *[accum[r] for r in range(num_accum)], a_desc, b_desc, k != 0, 1, 1
                            )
                        txl.ptx.wgmma.commit_group.sync.aligned()
                        for r in range(num_accum):
                            txl.cuda.wgmma.noop_barrier(accum[r])
                        txl.ptx.wgmma.wait_group.sync.aligned(0)

                        # Release KV empty
                        txl.ptx["mbarrier.arrive.shared::cta.b64"](empty_kv(kv_stage_l))

                        # Reduce over the heads and store (`.cuh:274-317`)
                        kv_offset = txl.local_scalar(
                            "uint32",
                            init=kv_start + kv_block_idx * txl.uint32(BLOCK_KV) + warp_offset,
                        )
                        for i in range(BQ):
                            _reduce_and_store(
                                accum,
                                i * accum_per_row,
                                weights,
                                i * weights_per_row,
                                H,
                                scale_kv_0,
                                scale_kv_1,
                                zero,
                                q_offset=q_offsets[i],
                                kv_offset=kv_offset,
                                v_offsets=(v_0_offset, v_1_offset),
                                ks=seq_k_start[i],
                                ke=seq_k_end[i],
                                logits=logits,
                                is_compressed=is_compressed,
                                is_bf16=is_bf16,
                            )
                    txl.assign(num_total_kv_blocks, num_total_kv_blocks + num_kv_blocks)

                    # Release Q empty
                    txl.ptx["mbarrier.arrive.shared::cta.b64"](empty_q(q_stage_idx))

                    # Jump to the next block
                    txl.assign(block_q_idx, block_q_idx + txl.uint32(spec.num_sms))
                    txl.assign(q_iter_idx, q_iter_idx + txl.uint32(1))

    return sm90_fp8_mqa_logits.func


def _producer(spec, *, maps, smem, bars, sched):
    """The TMA warp (`.cuh:158-205`): Q blocks and their KV blocks, from one elected lane."""
    tensor_map_q, tensor_map_kv, tensor_map_kv_scales, tensor_map_weights = maps
    smem_q, smem_kv, smem_w, smem_sf = smem
    full_q, empty_q, full_kv, empty_kv = bars
    H, D, BQ = spec.num_heads, spec.head_dim, spec.block_q
    q_stage, kv_stage = BQ * H * D, BLOCK_KV * D
    w_stage, sf_stage = BQ * H * 4, BLOCK_KV * 4
    block_q_idx, q_iter_idx = sched["block_q_idx"], sched["q_iter_idx"]
    num_q_blocks = sched["num_q_blocks"]
    num_total_kv_blocks = sched["num_total_kv_blocks"]

    def issue_tma_q(stage_idx, block_idx):
        txl.ptx[_TMA_LOAD_HINT](
            smem_q.ptr_to([stage_idx * txl.uint32(q_stage)]),
            txl.address_of(tensor_map_q),
            txl.int32(0),
            txl.cast(block_idx * txl.uint32(BQ * H), "int32"),
            full_q(stage_idx),
            txl.uint64(EVICT_NORMAL),
        )
        txl.ptx[_TMA_LOAD_HINT](
            smem_w.ptr_to([stage_idx * txl.uint32(w_stage // 4)]),
            txl.address_of(tensor_map_weights),
            txl.int32(0),
            txl.cast(block_idx * txl.uint32(BQ), "int32"),
            full_q(stage_idx),
            txl.uint64(EVICT_NORMAL),
        )
        txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](
            full_q(stage_idx), txl.uint32(q_stage + w_stage)
        )

    first = _elect()
    with txl.if_(txl.And(first == txl.uint32(1), block_q_idx < num_q_blocks)):
        with txl.then_():
            issue_tma_q(txl.uint32(0), block_q_idx)

    # Only the first lane persistently schedules over blocks
    looper = _elect()
    with txl.if_(looper == txl.uint32(1)):
        with txl.then_():
            with txl.while_(block_q_idx < num_q_blocks):
                sched["load_schedule"](1)
                q_stage_idx, q_phase = sched["q_stage_idx"], sched["q_phase"]
                kv_start, num_kv_blocks = sched["kv_start"], sched["num_kv_blocks"]

                # Wait Q consumer release
                _wait(empty_q(q_stage_idx), q_phase ^ txl.uint32(1))

                # Issue TMA Q
                next_block_q_idx = block_q_idx + txl.uint32(spec.num_sms)
                with txl.if_(next_block_q_idx < num_q_blocks):
                    with txl.then_():
                        issue_tma_q(q_stage_idx, next_block_q_idx)

                # Issue TMA KV
                with txl.serial(0, num_kv_blocks, unroll=True, dtype="uint32") as kv_block_idx:
                    kv_stage_idx, kv_phase = sched["kv_pipeline"](kv_block_idx)
                    kv_stage_l = txl.local_scalar("uint32", init=kv_stage_idx)
                    _wait(empty_kv(kv_stage_l), kv_phase ^ txl.uint32(1))
                    kv_row = kv_start + kv_block_idx * txl.uint32(BLOCK_KV)
                    txl.ptx[_TMA_LOAD_HINT](
                        smem_kv.ptr_to([kv_stage_l * txl.uint32(kv_stage)]),
                        txl.address_of(tensor_map_kv),
                        txl.int32(0),
                        txl.cast(kv_row, "int32"),
                        full_kv(kv_stage_l),
                        txl.uint64(EVICT_NORMAL),
                    )
                    txl.ptx[_TMA_LOAD_HINT](
                        smem_sf.ptr_to([kv_stage_l * txl.uint32(sf_stage // 4)]),
                        txl.address_of(tensor_map_kv_scales),
                        txl.cast(kv_row, "int32"),
                        txl.int32(0),
                        full_kv(kv_stage_l),
                        txl.uint64(EVICT_NORMAL),
                    )
                    txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](
                        full_kv(kv_stage_l), txl.uint32(kv_stage + sf_stage)
                    )
                txl.assign(num_total_kv_blocks, num_total_kv_blocks + num_kv_blocks)

                # Jump to the next block
                txl.assign(block_q_idx, next_block_q_idx)
                txl.assign(q_iter_idx, q_iter_idx + txl.uint32(1))


def _reduce_and_store(
    accum,
    a0,
    weights,
    w0,
    H,
    scale_kv_0,
    scale_kv_1,
    zero,
    *,
    q_offset,
    kv_offset,
    v_offsets,
    ks,
    ke,
    logits,
    is_compressed,
    is_bf16,
):
    """One Q row of `.cuh:274-317`, in the order nvcc emits for it (non-FTZ under `fp_mode="precise"`).

    Accumulator `a0 + J` holds head `8*(J//4) + 2*(lane%4) + (J&1)` for KV row `v_0` (J%4 < 2)
    or `v_1` (J%4 >= 2), weight `w0 + (J//4)*2 + (J&1)`.  nvcc contracts the source's
    `sum[k] = t(k); sum[k] += t(4 + k); ...` as `fma(max(a[k]), w, max(a[4+k]) * w')` and then one
    `fma` per further head group; that rounding order is reproduced here.
    """

    def relu(j):
        # `fmaxf(x, 0)`: IEEE maxNum, one `max.f32` (FMNMX).
        return txl.local_scalar("float32", init=txl.fmax(accum[a0 + j], zero))

    def w(j):
        return weights[w0 + (j // 4) * 2 + (j & 1)]

    # Plain FP32 arithmetic under `fp_mode="precise"`; NVRTC contracts `a * b + c` to `fma.rn.f32` as nvcc
    # does for the source: `s_k = fma(max(a[k]), w, max(a[4+k]) * w')`, then one `fma` per head group.
    sums = []
    for k in range(4):
        r0 = relu(k)
        r1 = relu(4 + k)
        s = txl.local_scalar("float32", init=r1 * w(4 + k))
        txl.assign(s, r0 * w(k) + s)
        for j in range(2, H // 8):
            r = relu(4 * j + k)
            txl.assign(s, r * w(4 * j + k) + s)
        sums.append(s)

    v_0 = txl.local_scalar("float32", init=scale_kv_0 * (sums[0] + sums[1]))
    v_1 = txl.local_scalar("float32", init=scale_kv_1 * (sums[2] + sums[3]))

    # Inter-thread reduction
    for m in (1, 2):
        for v in (v_0, v_1):
            o = txl.local_scalar("uint32")
            txl.ptx.shfl_sync.bfly.b32(
                o,
                txl.reinterpret(v, ty="uint32"),
                txl.uint32(m),
                txl.uint32(31),
                txl.uint32(FULL_MASK),
            )
            txl.assign(v, v + txl.reinterpret(o, ty="float32"))

    # Store into the global memory
    # `v_1_offset == v_0_offset + 8`: like nvcc for the source, address both values of a row from one
    # 64-bit base (`[base]` and `[base + 8 elements]`).
    index = q_offset + txl.cast(kv_offset, "uint64") + txl.cast(v_offsets[0], "uint64")
    if is_compressed:
        index = index - txl.cast(ks, "uint64")
    ptr_0 = logits.ptr_to([txl.cast(index, "int64")])
    ptr_1 = txl.ptx.addr(ptr_0, 8 * (2 if is_bf16 else 4))
    for v, vo, ptr in ((v_0, v_offsets[0], ptr_0), (v_1, v_offsets[1], ptr_1)):
        # The compressed store is predicated (`@p st.global`) rather than branched around: nvrtc
        # sinks the address chain into a guarded region, and ptxas then keeps the branch instead
        # of the predicated store the source's SASS has.
        col = kv_offset + vo
        pred = txl.And(ks <= col, col < ke) if is_compressed else None
        if is_bf16:
            bits = txl.local_scalar("uint16", init=txl.uint16(0))
            if pred is None:
                txl.ptx.cvt.rn.bf16.f32(bits, v)
                txl.ptx["st.global.u16"](ptr, bits)
            else:
                txl.ptx.cvt.rn.bf16.f32(bits, v, pred=pred, preserve_dst=True)
                txl.ptx["st.global.u16"](ptr, bits, pred=pred)
        elif pred is None:
            txl.ptx["st.global.f32"](ptr, v)
        else:
            txl.ptx["st.global.f32"](ptr, v, pred=pred)

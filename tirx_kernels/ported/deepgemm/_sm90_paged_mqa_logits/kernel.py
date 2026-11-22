# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of `sm90_fp8_paged_mqa_logits` (`deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`).

One persistent kernel instantiation per `KernelSpec`. Every role walks the same task list with
its own copy of `SM90PagedMQALogitsScheduler` (`scheduler/sm90_paged_mqa_logits.cuh`), driven by
the schedule metadata DeepGEMM's metadata kernel produced. A task is (q, kv split of 4 blocks).

- TMA warp g (warps 16..19, all lanes) streams KV block `kv_idx + g` and its scales into its own
  3-stage ring, through a 32-task block-table window held in lanes. Group 0 also prefetches the
  next q's Q rows and weights into the shared 3-stage Q ring.
- Math warpgroup g computes `m64n{next_n*H}k32` WGMMA chains against the q's Q rows, then the
  ReLU x weight head reduction (non-FTZ under `fp_mode="precise"`, in nvcc's order), the KV-scale multiply,
  two butterflies, and the stores.

Every shared-memory buffer is a byte offset into one linear pool (`.cuh:62-106`). The
`kHeadDim`-byte swizzle of Q and KV lives in the TensorMaps and the WGMMA descriptors.
"""

import tirx_kernels.tirx_lite as txl

from .._sm90_mqa_logits.kernel import EVICT_NORMAL, FULL_MASK, _elect, _u32, _wait, _with_fp_mode

BLOCK_KV = 64
NUM_Q_STAGES = 3
NUM_KV_STAGES = 3
NUM_MATH_WARPGROUPS = 4  # SPLIT_KV / 64
NUM_BLOCKS_PER_SPLIT = NUM_MATH_WARPGROUPS
NUM_MATH_THREADS = 128 * NUM_MATH_WARPGROUPS

_TMA_2D = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
_TMA_3D = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
_GMMA_LAYOUT = {128: 1, 64: 2, 32: 3}


def _align(x, a):
    return (x + a - 1) // a * a


class _Scheduler:
    """`SM90PagedMQALogitsScheduler<kNextN, true, false, 64, 4, 1>` (sched:138-246), per thread."""

    def __init__(self, next_n, schedule_meta, context_lens):
        self.next_n = next_n
        self.context_lens = context_lens
        self.cur_q = txl.local_scalar("uint32")
        self.cur_kv = txl.local_scalar("uint32")
        self.end_q = txl.local_scalar("uint32")
        self.end_kv = txl.local_scalar("uint32")
        self.cur_num_kv = txl.local_scalar("uint32")
        sm = _u32(txl.cta_id())
        # Constructor (sched:194-210): two uint2 reads of the metadata, `uint2` pointer + sm_idx and + 1,
        # i.e. one unsigned 64-bit base and an 8-byte immediate for the second, as nvcc forms it.
        meta_ptr = schedule_meta.ptr_to([txl.cast(sm, "uint64") * txl.uint64(2)])
        txl.ptx["ld.global.v2.u32"](self.cur_q, self.cur_kv, meta_ptr)
        txl.ptx["ld.global.v2.u32"](self.end_q, self.end_kv, txl.ptx.addr(meta_ptr, 8))
        txl.assign(self.cur_kv, self.cur_kv * txl.uint32(NUM_BLOCKS_PER_SPLIT))
        txl.assign(self.end_kv, self.end_kv * txl.uint32(NUM_BLOCKS_PER_SPLIT))
        self.refresh(self.cur_q)

    def refresh(self, q):
        """`refresh_num_kv_and_advance` (sched:179-192), 2D context lens, non-varlen."""
        ctx_len = txl.local_scalar("uint32")
        txl.ptx["ld.global.u32"](
            ctx_len,
            self.context_lens.ptr_to(
                [q * txl.uint32(self.next_n) + txl.uint32(self.next_n - 1)]
            ),
        )
        txl.assign(
            self.cur_num_kv, (ctx_len + txl.uint32(BLOCK_KV - 1)) // txl.uint32(BLOCK_KV)
        )

    def exist(self, q):
        """`exist_q_atom_idx` (sched:242-244)."""
        return txl.Or(
            q < self.end_q, txl.And(q == self.end_q, txl.uint32(0) < self.end_kv)
        )

    def fetch(self, valid, q_idx, kv_idx, num_kv):
        """`fetch_next_task` (sched:223-240): writes the task and whether one existed."""
        txl.assign(q_idx, self.cur_q)
        txl.assign(kv_idx, self.cur_kv)
        txl.assign(num_kv, self.cur_num_kv)
        txl.assign(valid, txl.uint32(1))
        with txl.if_(txl.And(self.cur_q == self.end_q, self.cur_kv == self.end_kv)):
            with txl.then_():
                txl.assign(valid, txl.uint32(0))
            with txl.else_():
                txl.assign(self.cur_kv, self.cur_kv + txl.uint32(NUM_BLOCKS_PER_SPLIT))
                with txl.if_(self.cur_kv >= self.cur_num_kv):
                    with txl.then_():
                        txl.assign(self.cur_kv, txl.uint32(0))
                        txl.assign(self.cur_q, self.cur_q + txl.uint32(1))
                        with txl.if_(self.exist(self.cur_q)):
                            with txl.then_():
                                self.refresh(self.cur_q)


def build_kernel(spec, *, backend_config=None):
    """The TIRx `PrimFunc` for one `sm90_fp8_paged_mqa_logits` template instantiation."""
    NN, H, D = spec.next_n, spec.num_heads, spec.head_dim
    if NN not in (1, 2) or H not in (32, 64) or D not in (32, 64, 128):
        raise NotImplementedError(f"unported specialization {spec}")
    N = NN * H  # WGMMA N
    num_accum = N // 2
    accum_per_row = H // 2
    weights_per_row = H // 4
    num_k_steps = D // 32
    wgmma = f"wgmma.mma_async.sync.aligned.m64n{N}k32.f32.e4m3.e4m3"
    is_bf16 = spec.logits_dtype == "bfloat16"
    elem = 2 if is_bf16 else 4

    # ---- shared-memory byte offsets (`.cuh:62-106`, `.hpp:353-362`) ----
    sw = 8 * D
    q_st = NN * H * D
    w_bytes = NN * H * 4
    w_st = _align(w_bytes, sw)
    q_pipe = NUM_Q_STAGES * (q_st + w_st) + _align(NUM_Q_STAGES * 8 * 2, sw)
    kv_st = BLOCK_KV * D
    sf_st = _align(BLOCK_KV * 4, sw)
    kv_pipe = NUM_KV_STAGES * (kv_st + sf_st) + _align(NUM_KV_STAGES * 8 * 2, sw)
    off_q_bar = NUM_Q_STAGES * (q_st + w_st)
    total = q_pipe + NUM_MATH_WARPGROUPS * kv_pipe
    if total + NUM_MATH_WARPGROUPS * 2 * 8 + 4 != spec.smem_size:
        raise AssertionError("pool layout disagrees with the host smem size")
    desc_hi = ((8 * D) >> 4) | (_GMMA_LAYOUT[D] << 30)

    @txl.kernel()
    def sm90_fp8_paged_mqa_logits(
        batch_size_arg: "int32",
        logits_stride_arg: "int32",
        block_table_stride_arg: "int32",
        context_lens: txl.Tensor((txl.int64(),), "int32"),
        logits: txl.Tensor((txl.int64(),), spec.logits_dtype),
        block_table: txl.Tensor((txl.int64(),), "int32"),
        schedule_meta: txl.Tensor((txl.int64(),), "int32"),
        tensor_map_q: txl.Ptr(txl.TensorMap),
        tensor_map_kv: txl.Ptr(txl.TensorMap),
        tensor_map_kv_scales: txl.Ptr(txl.TensorMap),
        tensor_map_weights: txl.Ptr(txl.TensorMap),
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=spec.num_sms, block=128 + NUM_MATH_THREADS),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=_with_fp_mode(backend_config),
        )
        batch_size = txl.local_scalar("uint32", init=_u32(batch_size_arg))
        logits_stride = txl.local_scalar("uint32", init=_u32(logits_stride_arg))
        block_table_stride = txl.local_scalar("uint32", init=_u32(block_table_stride_arg))
        thread_idx = _u32(txl.thread_id())

        # NOTES: `__shfl_sync` so NVCC keeps the warp index in a uniform register (`.cuh:45-48`).
        warp_idx = txl.local_scalar("uint32")
        txl.ptx.shfl_sync.idx.b32(
            warp_idx, thread_idx // txl.uint32(32), txl.uint32(0), txl.uint32(31), txl.uint32(FULL_MASK)
        )
        warpgroup_idx = txl.local_scalar("uint32", init=warp_idx // txl.uint32(4))
        lane_idx = txl.local_scalar("uint32", init=_u32(txl.lane_id()))

        # ---- TensorMap prefetch (`.cuh:54-60`) ----
        with txl.if_(warp_idx == txl.uint32(NUM_MATH_THREADS // 32)):
            with txl.then_():
                elected = _elect()
                with txl.if_(elected == txl.uint32(1)):
                    with txl.then_():
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_q))
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_kv))
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_kv_scales))
                        txl.ptx.prefetch.tensormap(txl.address_of(tensor_map_weights))
        txl.ptx.bar.warp.sync(txl.uint32(FULL_MASK))

        # ---- one linear pool, aligned to 8 * kHeadDim ----
        smem = txl.smem_pool()
        pool = smem.alloc((total,), "uint8", align=sw)
        smem.commit(spec.smem_size)

        def at(byte_off):
            return pool.ptr_to([byte_off])

        def off_q(s):
            return s * txl.uint32(q_st)

        def off_w(s):
            return txl.uint32(NUM_Q_STAGES * q_st) + s * txl.uint32(w_st)

        def full_q(s):
            return at(txl.uint32(off_q_bar) + s * txl.uint32(8))

        def empty_q(s):
            return at(txl.uint32(off_q_bar + 8 * NUM_Q_STAGES) + s * txl.uint32(8))

        # TMA warp g and math warpgroup g share KV group g (`.cuh:94`).
        group_src = txl.select(
            thread_idx >= txl.uint32(NUM_MATH_THREADS),
            (thread_idx - txl.uint32(NUM_MATH_THREADS)) // txl.uint32(32),
            warpgroup_idx,
        )
        kv_group_idx = txl.local_scalar("uint32")
        txl.ptx.shfl_sync.idx.b32(
            kv_group_idx, group_src, txl.uint32(0), txl.uint32(31), txl.uint32(FULL_MASK)
        )
        group_base = txl.local_scalar(
            "uint32", init=txl.uint32(q_pipe) + txl.uint32(kv_pipe) * kv_group_idx
        )

        def off_kv(s):
            return group_base + s * txl.uint32(kv_st)

        def off_sf(s):
            return group_base + txl.uint32(NUM_KV_STAGES * kv_st) + s * txl.uint32(sf_st)

        def full_kv(s):
            return at(group_base + txl.uint32(NUM_KV_STAGES * (kv_st + sf_st)) + s * txl.uint32(8))

        def empty_kv(s):
            return at(
                group_base + txl.uint32(NUM_KV_STAGES * (kv_st + sf_st) + 8 * NUM_KV_STAGES)
                + s * txl.uint32(8)
            )

        # ---- barrier init (`.cuh:108-128`) ----
        with txl.if_(warp_idx >= txl.uint32(NUM_MATH_THREADS // 32)):
            with txl.then_():
                init_elected = _elect()
                with txl.if_(init_elected == txl.uint32(1)):
                    with txl.then_():
                        with txl.if_(kv_group_idx == txl.uint32(0)):
                            with txl.then_():
                                for s in range(NUM_Q_STAGES):
                                    txl.ptx["mbarrier.init.shared::cta.b64"](
                                        full_q(txl.uint32(s)), txl.uint32(1)
                                    )
                                    txl.ptx["mbarrier.init.shared::cta.b64"](
                                        empty_q(txl.uint32(s)), txl.uint32(NUM_MATH_THREADS)
                                    )
                        with txl.if_(kv_group_idx < txl.uint32(NUM_MATH_WARPGROUPS)):
                            with txl.then_():
                                for s in range(NUM_KV_STAGES):
                                    txl.ptx["mbarrier.init.shared::cta.b64"](
                                        full_kv(txl.uint32(s)), txl.uint32(1)
                                    )
                                    txl.ptx["mbarrier.init.shared::cta.b64"](
                                        empty_kv(txl.uint32(s)), txl.uint32(128)
                                    )
                        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))

        txl.ptx.griddepcontrol.wait()

        sched = _Scheduler(NN, schedule_meta, context_lens)
        q_iter_idx = txl.local_scalar("uint32", init=txl.uint32(0))
        kv_iter_idx = txl.local_scalar("uint32", init=txl.uint32(0))

        def pipeline(it):
            return it % txl.uint32(3), it // txl.uint32(3) & txl.uint32(1)

        valid = txl.local_scalar("uint32")
        next_q = txl.local_scalar("uint32")
        next_kv = txl.local_scalar("uint32")
        next_num_kv = txl.local_scalar("uint32")
        q_idx = txl.local_scalar("uint32", init=batch_size)  # "none" (`.cuh:166,225`)
        kv_idx = txl.local_scalar("uint32")
        q_stage_idx = txl.local_scalar("uint32")
        q_phase = txl.local_scalar("uint32")

        with txl.if_(warp_idx >= txl.uint32(NUM_MATH_THREADS // 32)):
            # ================= TMA warps (`.cuh:151-215`) =================
            with txl.then_():
                txl.ptx["setmaxnreg.dec.sync.aligned.u32"](txl.uint32(64))
                with txl.if_(kv_group_idx < txl.uint32(NUM_MATH_WARPGROUPS)):
                    with txl.then_():
                        num_kv = txl.local_scalar("uint32")

                        def issue_tma_q(stage, q):
                            with txl.if_(kv_group_idx == txl.uint32(0)):
                                with txl.then_():
                                    q_elected = _elect()
                                    with txl.if_(q_elected == txl.uint32(1)):
                                        with txl.then_():
                                            txl.ptx[_TMA_2D](
                                                at(off_q(stage)),
                                                txl.address_of(tensor_map_q),
                                                txl.int32(0),
                                                txl.cast(q * txl.uint32(NN * H), "int32"),
                                                full_q(stage),
                                                txl.uint64(EVICT_NORMAL),
                                            )
                                            txl.ptx[_TMA_2D](
                                                at(off_w(stage)),
                                                txl.address_of(tensor_map_weights),
                                                txl.int32(0),
                                                txl.cast(q * txl.uint32(NN), "int32"),
                                                full_q(stage),
                                                txl.uint64(EVICT_NORMAL),
                                            )
                                            txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](
                                                full_q(stage), txl.uint32(q_st + w_bytes)
                                            )

                        # Prefetch the first Q
                        sched.fetch(valid, next_q, next_kv, next_num_kv)
                        with txl.if_(valid == txl.uint32(1)):
                            with txl.then_():
                                issue_tma_q(txl.uint32(0), next_q)
                                txl.assign(q_iter_idx, txl.uint32(1))

                        kv_block_idx_ptr = txl.local_scalar("uint32", init=txl.uint32(32))
                        kv_block_idx_storage = txl.local_scalar("uint32", init=txl.uint32(0))

                        with txl.while_(valid == txl.uint32(1)):
                            # Prefetch next Q when current Q changes
                            prefetch_q = txl.local_scalar(
                                "uint32",
                                init=txl.cast(
                                    txl.And(q_idx != next_q, sched.exist(next_q + txl.uint32(1))),
                                    "uint32",
                                ),
                            )
                            txl.assign(q_idx, next_q)
                            txl.assign(kv_idx, next_kv)
                            txl.assign(num_kv, next_num_kv)

                            # Wait Q consumer release and issue TMA Q
                            with txl.if_(prefetch_q == txl.uint32(1)):
                                with txl.then_():
                                    stage, phase = pipeline(q_iter_idx)
                                    qs = txl.local_scalar("uint32", init=stage)
                                    qp = txl.local_scalar("uint32", init=phase)
                                    txl.assign(q_iter_idx, q_iter_idx + txl.uint32(1))
                                    _wait(empty_q(qs), qp ^ txl.uint32(1))
                                    issue_tma_q(qs, q_idx + txl.uint32(1))

                            # Read KV block index (a 32-task window across the lanes)
                            with txl.if_(
                                txl.Or(kv_idx == txl.uint32(0), kv_block_idx_ptr == txl.uint32(32))
                            ):
                                with txl.then_():
                                    txl.assign(kv_block_idx_ptr, txl.uint32(0))
                                    j = txl.local_scalar(
                                        "uint32",
                                        init=kv_idx + kv_group_idx
                                        + lane_idx * txl.uint32(NUM_MATH_WARPGROUPS),
                                    )
                                    txl.assign(kv_block_idx_storage, txl.uint32(0))
                                    with txl.if_(j < num_kv):
                                        with txl.then_():
                                            txl.ptx["ld.global.u32"](
                                                kv_block_idx_storage,
                                                block_table.ptr_to(
                                                    [
                                                        txl.cast(q_idx, "int64")
                                                        * txl.cast(block_table_stride, "int64")
                                                        + txl.cast(j, "int64")
                                                    ]
                                                ),
                                            )
                            kv_block_idx = txl.local_scalar("uint32")
                            txl.ptx.shfl_sync.idx.b32(
                                kv_block_idx,
                                kv_block_idx_storage,
                                kv_block_idx_ptr,
                                txl.uint32(31),
                                txl.uint32(FULL_MASK),
                            )
                            txl.assign(kv_block_idx_ptr, kv_block_idx_ptr + txl.uint32(1))

                            # Wait KV consumer release
                            stage, phase = pipeline(kv_iter_idx)
                            kvs = txl.local_scalar("uint32", init=stage)
                            kvp = txl.local_scalar("uint32", init=phase)
                            txl.assign(kv_iter_idx, kv_iter_idx + txl.uint32(1))
                            _wait(empty_kv(kvs), kvp ^ txl.uint32(1))

                            # Issue TMA KV
                            kv_elected = _elect()
                            with txl.if_(kv_elected == txl.uint32(1)):
                                with txl.then_():
                                    txl.ptx[_TMA_3D](
                                        at(off_kv(kvs)),
                                        txl.address_of(tensor_map_kv),
                                        txl.int32(0),
                                        txl.int32(0),
                                        txl.cast(kv_block_idx, "int32"),
                                        full_kv(kvs),
                                        txl.uint64(EVICT_NORMAL),
                                    )
                                    txl.ptx[_TMA_2D](
                                        at(off_sf(kvs)),
                                        txl.address_of(tensor_map_kv_scales),
                                        txl.int32(0),
                                        txl.cast(kv_block_idx, "int32"),
                                        full_kv(kvs),
                                        txl.uint64(EVICT_NORMAL),
                                    )
                                    txl.ptx["mbarrier.arrive.expect_tx.shared::cta.b64"](
                                        full_kv(kvs), txl.uint32(kv_st + BLOCK_KV * 4)
                                    )

                            # Fetch next task
                            sched.fetch(valid, next_q, next_kv, next_num_kv)

            # ================= math warpgroups (`.cuh:216-330`) =================
            with txl.else_():
                txl.ptx["setmaxnreg.inc.sync.aligned.u32"](txl.uint32(104))
                accum = txl.alloc_local((num_accum,), "float32")
                weights = txl.alloc_local((NN * weights_per_row,), "float32")
                sub_warp_offset = txl.local_scalar(
                    "uint32", init=warp_idx % txl.uint32(4) * txl.uint32(16)
                )
                v_0_offset = txl.local_scalar("uint32", init=lane_idx // txl.uint32(4))
                v_0_offset_u64 = txl.local_scalar("uint64", init=txl.cast(v_0_offset, "uint64"))
                pool_lo = txl.local_scalar(
                    "uint32", init=_u32(txl.cuda.cvta_generic_to_shared(at(txl.uint32(0))))
                )
                desc_hi64 = txl.uint64(desc_hi << 32)
                zero = txl.float32(0.0)

                # `while (scheduler.fetch_next_task(...))`: one fetch site at the loop head.
                running = txl.local_scalar("uint32", init=txl.uint32(1))
                with txl.while_(running == txl.uint32(1)):
                    sched.fetch(valid, next_q, next_kv, next_num_kv)
                    with txl.if_(valid == txl.uint32(0)):
                        with txl.then_():
                            txl.assign(running, txl.uint32(0))
                        with txl.else_():
                            # Current Q changes
                            with txl.if_(q_idx != next_q):
                                with txl.then_():
                                    # Release Last Q empty
                                    with txl.if_(q_iter_idx > txl.uint32(0)):
                                        with txl.then_():
                                            txl.ptx["mbarrier.arrive.shared::cta.b64"](
                                                empty_q((q_iter_idx - txl.uint32(1)) % txl.uint32(3))
                                            )
                                    # Wait TMA Q arrival
                                    stage, phase = pipeline(q_iter_idx)
                                    txl.assign(q_stage_idx, stage)
                                    txl.assign(q_phase, phase)
                                    txl.assign(q_iter_idx, q_iter_idx + txl.uint32(1))
                                    _wait(full_q(q_stage_idx), q_phase)
                                    # Read weights
                                    w_byte = off_w(q_stage_idx)
                                    for i in range(NN):
                                        for j in range(weights_per_row):
                                            txl.ptx["ld.shared.f32"](
                                                weights[i * weights_per_row + j],
                                                at(
                                                    w_byte
                                                    + txl.uint32(4 * (i * H + (j // 2) * 8 + (j & 1)))
                                                    + lane_idx % txl.uint32(4) * txl.uint32(8)
                                                ),
                                            )

                            # Get current Q and KV index
                            txl.assign(q_idx, next_q)
                            txl.assign(kv_idx, next_kv)

                            # Calculate KV offset in advance
                            kv_offset = txl.local_scalar(
                                "uint64",
                                init=txl.cast(q_idx * txl.uint32(NN), "uint64")
                                * txl.cast(logits_stride, "uint64")
                                + txl.cast(
                                    (kv_idx + kv_group_idx) * txl.uint32(BLOCK_KV) + sub_warp_offset,
                                    "uint64",
                                ),
                            )

                            # Wait TMA KV arrival
                            stage, phase = pipeline(kv_iter_idx)
                            kvs = txl.local_scalar("uint32", init=stage)
                            kvp = txl.local_scalar("uint32", init=phase)
                            txl.assign(kv_iter_idx, kv_iter_idx + txl.uint32(1))
                            _wait(full_kv(kvs), kvp)

                            # Issue WGMMA
                            for r in range(num_accum):
                                txl.cuda.wgmma.noop_barrier(accum[r])
                            txl.ptx.wgmma.fence.sync.aligned()
                            a_base = pool_lo + off_kv(kvs)
                            b_base = pool_lo + off_q(q_stage_idx)
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

                            # Read per-KV scales
                            scale_kv_0 = txl.local_scalar("float32")
                            scale_kv_1 = txl.local_scalar("float32")
                            sf_byte = off_sf(kvs) + (sub_warp_offset + v_0_offset) * txl.uint32(4)
                            txl.ptx["ld.shared.f32"](scale_kv_0, at(sf_byte))
                            txl.ptx["ld.shared.f32"](scale_kv_1, at(sf_byte + txl.uint32(32)))

                            # Wait WGMMA
                            txl.ptx.wgmma.wait_group.sync.aligned(0)

                            # Release KV empty
                            txl.ptx["mbarrier.arrive.shared::cta.b64"](empty_kv(kvs))

                            # Reduce over the head dim and store
                            for i in range(NN):
                                _reduce_and_store(
                                    accum,
                                    i * accum_per_row,
                                    weights,
                                    i * weights_per_row,
                                    H,
                                    scale_kv_0,
                                    scale_kv_1,
                                    zero,
                                    uniform_index=kv_offset
                                    + txl.cast(txl.uint32(i), "uint64") * txl.cast(logits_stride, "uint64"),
                                    v_0_u64=v_0_offset_u64,
                                    logits=logits,
                                    is_bf16=is_bf16,
                                    elem=elem,
                                )

    return sm90_fp8_paged_mqa_logits.func


def _reduce_and_store(
    accum, a0, weights, w0, H, scale_kv_0, scale_kv_1, zero, *, uniform_index, v_0_u64, logits, is_bf16, elem
):
    """One next_n row of `.cuh:295-328`, in the order nvcc emits (non-FTZ under `fp_mode="precise"`).

    Accumulator `a0 + J` belongs to weight `w0 + (J//4)*2 + (J&1)`. nvcc contracts
    `sum[k] = t(k); sum[k] += t(4j + k)` to `fma(max(a[k]), w, max(a[4+k]) * w')`, followed by one
    `fma` per further head group: head group j is the outer loop, chain k the inner.
    """

    def w(j):
        return weights[w0 + (j // 4) * 2 + (j & 1)]

    def relu(j):
        # `fmaxf(x, 0)`: IEEE maxNum, one `max.f32` (FMNMX).
        return txl.local_scalar("float32", init=txl.fmax(accum[a0 + j], zero))

    # Plain FP32 arithmetic under `fp_mode="precise"` (NVRTC contracts `a * b + c` to `fma.rn.f32` as
    # nvcc does).
    r0 = [relu(k) for k in range(4)]
    sums = []
    for k in range(4):
        r1 = relu(4 + k)
        s = txl.local_scalar("float32", init=r1 * w(4 + k))
        txl.assign(s, r0[k] * w(k) + s)
        sums.append(s)
    for j in range(2, H // 8):
        for k in range(4):
            r = relu(4 * j + k)
            txl.assign(sums[k], r * w(4 * j + k) + sums[k])

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

    # Store into the global memory: `[base]` and `[base + 8 elements]` (v_1_offset = v_0_offset + 8).
    # The warp-uniform row offset first, the per-thread v0 last, as nvcc forms it for the source;
    # `arithmetic_fence` (ISSUE-016) keeps NVRTC from reassociating v0 into the uniform part.
    index = txl.local_scalar("uint64", init=txl.cuda.arithmetic_fence(uniform_index) + v_0_u64)
    ptr_0 = logits.ptr_to([txl.cast(index, "int64")])
    ptr_1 = txl.ptx.addr(ptr_0, 8 * elem)
    for v, ptr in ((v_0, ptr_0), (v_1, ptr_1)):
        if is_bf16:
            bits = txl.local_scalar("uint16", init=txl.uint16(0))
            txl.ptx.cvt.rn.bf16.f32(bits, v)
            txl.ptx["st.global.u16"](ptr, bits)
        else:
            txl.ptx["st.global.f32"](ptr, v)

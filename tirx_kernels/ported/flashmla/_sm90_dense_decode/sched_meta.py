# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors


"""TIRx port of `smxx::decode::get_mla_metadata_kernel`
(`csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`): one warp, dynamic shared
memory of `4 * (5 * b + 1)` bytes, `__launch_bounds__(32, 1, 1)`. The per-request token count
follows the runtime `topk` branch, as upstream: dense decode passes `topk = -1` and `seqlens_k`;
sparse decode passes `topk` (and `extra_topk`, 0 without an extra KV cache) with optional
`topk_length` / `extra_topk_length` (null tensors when absent) and no `seqlens_k`.
"""

import tirx_kernels.tirx_lite as txl

#: `dense_attn_decode_interface`'s `GetDecodeSchedMetaParams` (`block_size_n`,
#: `fixed_overhead_num_blocks`).
BLOCK_SIZE_N = 64
FIXED_OVERHEAD_NUM_BLOCKS = 5


def build_sched_meta_kernel(*, backend_config=None):
    @txl.kernel()
    def get_mla_metadata(
        seqlens_k: txl.Tensor((txl.int64(),), "int32"),
        topk_length: txl.Tensor((txl.int64(),), "int32"),
        extra_topk_length: txl.Tensor((txl.int64(),), "int32"),
        tile_scheduler_metadata: txl.Tensor((txl.int64(),), "int32"),
        num_splits: txl.Tensor((txl.int64(),), "int32"),
        b: "int32",
        block_size_n: "int32",
        fixed_overhead_num_blocks: "int32",
        topk: "int32",
        extra_topk: "int32",
        num_sm_parts: "int32",
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=4 * (b * 5 + 1)),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=backend_config,
        )
        # get_decoding_sched_meta.cu:22-27: five int arrays in one dynamic pool.
        smem = txl.alloc_tensor([b * 5 + 1], "int32", scope="shared.dyn", align=16)
        num_blocks_s = 0  # [b]
        num_splits_s = b  # [b + 1]
        seqlens_s = b * 2 + 1  # [b]
        first_block_s = b * 3 + 1  # [b]
        last_block_s = b * 4 + 1  # [b]
        lane = txl.thread_id()

        # :29-55: per-request block ranges.
        total = txl.local_scalar("int32", init=0)
        i = txl.local_scalar("int32", init=lane)
        with txl.while_(i < b):
            cur_s_k = txl.local_scalar("int32")
            with txl.if_(topk == -1):
                with txl.then_():
                    # Dense model: cur_s_k = the actual s_k.
                    txl.ptx.ld.global_.nc.s32(cur_s_k, seqlens_k.ptr_to([i]))
                with txl.else_():
                    # Sparse model: cur_s_k = topk (+ extra topk).
                    txl.assign(cur_s_k, topk)
                    with txl.if_(txl.Not(txl.isnullptr(topk_length.data))), txl.then_():
                        txl.ptx.ld.global_.nc.s32(cur_s_k, topk_length.ptr_to([i]))
                    with txl.if_(cur_s_k == 0), txl.then_():
                        txl.assign(cur_s_k, 1)  # the main loop is never empty
                    with txl.if_(extra_topk != 0), txl.then_():
                        # ku::ceil(cur_s_k, block_size_n) rounds up to a multiple.
                        up = txl.local_scalar("int32", init=cur_s_k + block_size_n - 1)
                        txl.assign(cur_s_k, up - up % block_size_n)
                        xs = txl.local_scalar("int32", init=extra_topk)
                        with txl.if_(txl.Not(txl.isnullptr(extra_topk_length.data))), txl.then_():
                            txl.ptx.ld.global_.nc.s32(xs, extra_topk_length.ptr_to([i]))
                        txl.assign(cur_s_k, cur_s_k + xs)
            txl.ptx.st.shared.s32(smem.ptr_to([seqlens_s + i]), cur_s_k)
            first_token = txl.local_scalar("int32", init=0)
            last_token = txl.local_scalar("int32", init=txl.max(cur_s_k - 1, 0))
            cur_first = txl.local_scalar("int32", init=first_token // block_size_n)
            cur_last = txl.local_scalar("int32", init=last_token // block_size_n)
            nblk = txl.local_scalar("int32", init=cur_last - cur_first + 1)
            txl.assign(total, total + nblk + fixed_overhead_num_blocks)
            txl.ptx.st.shared.s32(smem.ptr_to([num_blocks_s + i]), nblk)
            txl.ptx.st.shared.s32(smem.ptr_to([first_block_s + i]), cur_first)
            txl.ptx.st.shared.s32(smem.ptr_to([last_block_s + i]), cur_last)
            txl.assign(i, i + 32)
        peer = txl.local_scalar("int32")
        for off in (16, 8, 4, 2, 1):
            txl.ptx.shfl_sync.bfly.b32(
                peer, total, txl.uint32(off), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
            )
            txl.assign(total, total + peer)
        txl.cuda.warp_sync()

        def lds(base, idx):
            v = txl.local_scalar("int32")
            txl.ptx.ld.shared.s32(v, smem.ptr_to([base + idx]))
            return v

        # :61-99: lane 0 walks the requests greedily, `payload` blocks per partition.
        with txl.if_(lane == 0), txl.then_():
            payload = txl.local_scalar(
                "int32",
                init=(total + num_sm_parts - 1) // num_sm_parts + fixed_overhead_num_blocks,
            )
            now_req = txl.local_scalar("int32", init=0)
            now_block = txl.local_scalar("int32", init=0)
            now_split = txl.local_scalar("int32", init=0)
            cum_splits = txl.local_scalar("int32", init=0)
            txl.ptx.st.shared.s32(smem.ptr_to([num_splits_s]), txl.int32(0))
            meta = txl.alloc_local((8,), "int32")
            with txl.serial(num_sm_parts, unroll=False) as part:
                txl.assign(meta[0], now_req)  # begin_req_idx
                txl.assign(meta[2], now_block + lds(first_block_s, now_req))  # begin_block_idx
                txl.assign(meta[4], now_split)  # begin_split_idx
                txl.assign(meta[5], txl.cast(now_block != 0, "int32"))  # is_first_req_splitted
                remain = txl.local_scalar("int32", init=payload)
                walking = txl.local_scalar("int32", init=1)
                with txl.while_(txl.And(walking == 1, now_req < b)):
                    remain_blocks = txl.local_scalar(
                        "int32", init=lds(num_blocks_s, now_req) - now_block
                    )
                    with txl.if_(remain >= remain_blocks + fixed_overhead_num_blocks):
                        with txl.then_():
                            txl.assign(cum_splits, cum_splits + now_split + 1)
                            txl.ptx.st.shared.s32(
                                smem.ptr_to([num_splits_s + now_req + 1]), cum_splits
                            )
                            txl.assign(remain, remain - (remain_blocks + fixed_overhead_num_blocks))
                            txl.assign(now_req, now_req + 1)
                            txl.assign(now_block, 0)
                            txl.assign(now_split, 0)
                        with txl.else_():
                            with txl.if_(remain - fixed_overhead_num_blocks > 0), txl.then_():
                                txl.assign(now_block, now_block + remain - fixed_overhead_num_blocks)
                                txl.assign(now_split, now_split + 1)
                                txl.assign(remain, 0)
                            txl.assign(walking, 0)
                # end_req_idx / end_block_idx / is_last_req_splitted
                with txl.if_(now_block > 0):
                    with txl.then_():
                        txl.assign(meta[1], now_req)
                        txl.assign(meta[3], now_block + lds(first_block_s, now_req))
                    with txl.else_():
                        txl.assign(meta[1], now_req - 1)
                        with txl.if_(lds(seqlens_s, now_req - 1) == 0):
                            with txl.then_():
                                txl.assign(meta[3], 0)
                            with txl.else_():
                                txl.assign(meta[3], lds(last_block_s, now_req - 1) + 1)
                txl.assign(
                    meta[6],
                    txl.cast(
                        txl.And(
                            meta[3] != lds(last_block_s, meta[1]) + 1,
                            lds(seqlens_s, meta[1]) != 0,
                        ),
                        "int32",
                    ),
                )
                with txl.if_(meta[0] == meta[1]), txl.then_():
                    either = txl.local_scalar(
                        "int32", init=txl.cast(txl.Or(meta[5] != 0, meta[6] != 0), "int32")
                    )
                    txl.assign(meta[5], either)
                    txl.assign(meta[6], either)
                # tile_scheduler_metadata_ptr[i] = cur_meta (7 fields; the pad word is not written)
                words = meta.view("uint32")
                dst = part * 8
                txl.ptx.st.global_.v4.u32(
                    tile_scheduler_metadata.view("uint32").ptr_to([dst]),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                txl.ptx.st.global_.v2.u32(
                    tile_scheduler_metadata.view("uint32").ptr_to([dst + 4]), words[4], words[5]
                )
                txl.ptx.st.global_.u32(
                    tile_scheduler_metadata.view("uint32").ptr_to([dst + 6]), words[6]
                )
            txl.cuda.trap_when_assert_failed(
                txl.And(now_req == b, txl.And(now_block == 0, now_split == 0))
            )
        txl.cuda.warp_sync()

        # :104-106
        j = txl.local_scalar("int32", init=lane)
        with txl.while_(j <= b):
            v = lds(num_splits_s, j)
            txl.ptx.st.global_.s32(num_splits.ptr_to([j]), v)
            txl.assign(j, j + 32)

    return get_mla_metadata.func

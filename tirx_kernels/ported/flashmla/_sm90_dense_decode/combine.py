# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors


"""TIRx port of `smxx::decode::flash_fwd_mla_combine_kernel<ElementT, 512, 8, MAX_SPLITS, 256>`
(`csrc/smxx/decode/combine/combine.cu`), without an attention sink (dense decode passes none).
Launched with programmatic stream serialization after the main kernel.
"""

import tirx_kernels.tirx_lite as txl

HEAD_DIM_V = 512
BLOCK_SIZE_M = 8
NUM_THREADS = BLOCK_SIZE_M * 32
#: `MLA_NUM_SPLITS_SWITCH`.
MAX_SPLITS_BUCKETS = (32, 64, 96, 128, 160)


def max_splits_for(num_sm_parts: int) -> int:
    for bucket in MAX_SPLITS_BUCKETS:
        if num_sm_parts <= bucket:
            return bucket
    raise AssertionError(f"num_sm_parts = {num_sm_parts} > 160 is not supported by FlashMLA")


_LOG2E = 1.4426950408889634  # (float)M_LOG2E


def _fmax(dst, a, b):
    # IEEE maxNum (`fmaxf`): one max.ftz.f32 under the source's fast-math build.
    txl.assign(dst, txl.fmax(a, b))


def build_combine_kernel(dtype: str, max_splits: int, *, backend_config=None):
    cvt = "cvt.rn.bf16.f32" if dtype == "bfloat16" else "cvt.rn.f16.f32"
    elems = HEAD_DIM_V // (32 * 4)  # float4 chunks per lane
    lse_frags = (max_splits + 31) // 32

    @txl.kernel()
    def flash_fwd_mla_combine(
        lse: txl.Tensor((txl.int64(),), "float32"),
        out: txl.Tensor((txl.int64(),), dtype),
        lse_accum: txl.Tensor((txl.int64(),), "float32"),
        o_accum: txl.Tensor((txl.int64(),), "float32"),
        num_splits: txl.Tensor((txl.int64(),), "int32"),
        stride_lse_b: "int32",
        stride_lse_s_q: "int32",
        stride_o_b: "int32",
        stride_o_s_q: "int32",
        stride_o_h_q: "int32",
        stride_lse_accum_split: "int32",
        stride_lse_accum_s_q: "int32",
        stride_o_accum_split: "int32",
        stride_o_accum_s_q: "int32",
        stride_o_accum_h_q: "int32",
        b: "int32",
        s_q: "int32",
        h_q: "int32",
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(
                grid=(b * s_q, 1, (h_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M),
                block=NUM_THREADS,
                programmatic_stream_serialization=True,
            ),
            backend_config=backend_config,
        )
        del b
        # `__shared__ float smem_buf[BLOCK_SIZE_M][MAX_SPLITS]` (static), [warp][split].
        smem_buf = txl.alloc_tensor([BLOCK_SIZE_M * max_splits], "float32", scope="shared")

        # combine.cu:21-42: one warp per q head, BLOCK_SIZE_M heads per CTA.
        batch_s_q_idx = txl.cuda.block_idx("x")
        h_block_idx = txl.cuda.block_idx("z")
        thread_idx = txl.thread_id()
        warp_idx = txl.local_scalar("int32", init=thread_idx // 32)
        lane_idx = txl.local_scalar("int32", init=thread_idx % 32)
        batch_idx = txl.local_scalar("int32", init=batch_s_q_idx // s_q)
        s_q_idx = txl.local_scalar("int32", init=batch_s_q_idx - batch_idx * s_q)
        num_valid_heads = txl.local_scalar(
            "int32", init=txl.min(BLOCK_SIZE_M, h_q - BLOCK_SIZE_M * h_block_idx)
        )
        with txl.if_(warp_idx >= num_valid_heads), txl.then_():
            txl.return_(txl.int32(0))
        start_split = txl.local_scalar("int32")
        end_split = txl.local_scalar("int32")
        txl.ptx.ld.global_.nc.s32(start_split, num_splits.ptr_to([batch_idx]))
        txl.ptx.ld.global_.nc.s32(end_split, num_splits.ptr_to([batch_idx + 1]))
        my_num_splits = txl.local_scalar("int32", init=end_split - start_split)
        with txl.if_(my_num_splits == 1), txl.then_():
            txl.return_(txl.int32(0))
        txl.cuda.trap_when_assert_failed(my_num_splits <= max_splits)

        lse_accum_base = txl.local_scalar(
            "int32",
            init=start_split * stride_lse_accum_split
            + s_q_idx * stride_lse_accum_s_q
            + h_block_idx * BLOCK_SIZE_M,
        )
        lse_base = txl.local_scalar(
            "int32",
            init=batch_idx * stride_lse_b + s_q_idx * stride_lse_s_q + h_block_idx * BLOCK_SIZE_M,
        )

        # combine.cu:59-69: wait for the main kernel, then prefetch split 0's row.
        txl.ptx.griddepcontrol.wait()
        oaccum_base = txl.local_scalar(
            "int32",
            init=start_split * stride_o_accum_split
            + s_q_idx * stride_o_accum_s_q
            + (h_block_idx * BLOCK_SIZE_M + warp_idx) * stride_o_accum_h_q,
        )
        datas = txl.alloc_local((elems * 4,), "float32")
        for i in range(elems):
            txl.ptx.ld.global_.v4.f32(
                datas[4 * i],
                datas[4 * i + 1],
                datas[4 * i + 2],
                datas[4 * i + 3],
                o_accum.ptr_to([oaccum_base + lane_idx * 4 + i * 128]),
            )

        # combine.cu:72-119: warp w reduces the LSEs of q head w over the splits.
        local_lse = txl.alloc_local((lse_frags,), "float32")
        for i in range(lse_frags):
            split_idx = txl.local_scalar("int32", init=i * 32 + lane_idx)
            txl.assign(local_lse[i], txl.float32(-float("inf")))
            with txl.if_(split_idx < my_num_splits), txl.then_():
                txl.ptx.ld.global_.f32(
                    local_lse[i],
                    lse_accum.ptr_to([lse_accum_base + split_idx * stride_lse_accum_split + warp_idx]),
                )
        max_lse = txl.local_scalar("float32", init=txl.float32(-float("inf")))
        for i in range(lse_frags):
            _fmax(max_lse, max_lse, local_lse[i])
        peer = txl.local_scalar("float32")
        for off in (16, 8, 4, 2, 1):
            txl.ptx.shfl_sync.bfly.b32(
                peer, max_lse, txl.uint32(off), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
            )
            _fmax(max_lse, max_lse, peer)
        txl.assign(
            max_lse,
            txl.if_then_else(max_lse == txl.float32(-float("inf")), txl.float32(0.0), max_lse),
        )
        sum_lse = txl.local_scalar("float32", init=txl.float32(0.0))
        tmp = txl.local_scalar("float32")
        for i in range(lse_frags):
            txl.ptx["sub.ftz.f32"](tmp, local_lse[i], max_lse)
            txl.ptx["ex2.approx.ftz.f32"](tmp, tmp)
            txl.ptx["add.ftz.f32"](sum_lse, sum_lse, tmp)
        for off in (16, 8, 4, 2, 1):
            txl.ptx.shfl_sync.bfly.b32(
                peer, sum_lse, txl.uint32(off), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
            )
            txl.ptx["add.ftz.f32"](sum_lse, sum_lse, peer)
        global_lse = txl.local_scalar("float32", init=txl.float32(float("inf")))
        with txl.if_(
            txl.Not(txl.Or(sum_lse == txl.float32(0.0), sum_lse == txl.float32(-float("inf"))))
        ), txl.then_():
            txl.ptx["lg2.approx.ftz.f32"](global_lse, sum_lse)
            txl.ptx["add.ftz.f32"](global_lse, global_lse, max_lse)
        with txl.if_(lane_idx == 0), txl.then_():
            lse_out = txl.local_scalar("float32")
            txl.ptx["div.approx.ftz.f32"](lse_out, global_lse, txl.float32(_LOG2E))
            txl.ptx.st.global_.f32(lse.ptr_to([lse_base + warp_idx]), lse_out)
        for i in range(lse_frags):
            txl.ptx["sub.ftz.f32"](tmp, local_lse[i], global_lse)
            txl.ptx["ex2.approx.ftz.f32"](tmp, tmp)
            txl.ptx.st.shared.f32(smem_buf.ptr_to([warp_idx * max_splits + i * 32 + lane_idx]), tmp)
        txl.cuda.warp_sync()

        # combine.cu:123-160: serial LSE-weighted accumulation, prefetching the next split.
        result = txl.alloc_local((elems * 4,), "float32")
        for j in range(elems * 4):
            txl.assign(result[j], txl.float32(0.0))
        lse_scale = txl.local_scalar("float32")
        with txl.serial(my_num_splits, unroll=False) as split:
            txl.ptx.ld.shared.f32(lse_scale, smem_buf.ptr_to([warp_idx * max_splits + split]))
            for i in range(elems):
                for v in range(4):
                    txl.ptx["fma.rn.ftz.f32"](result[4 * i + v], lse_scale, datas[4 * i + v], result[4 * i + v])
                with txl.if_(split != my_num_splits - 1), txl.then_():
                    txl.ptx.ld.global_.v4.f32(
                        datas[4 * i],
                        datas[4 * i + 1],
                        datas[4 * i + 2],
                        datas[4 * i + 3],
                        o_accum.ptr_to(
                            [oaccum_base + (split + 1) * stride_o_accum_split + lane_idx * 4 + i * 128]
                        ),
                    )
        out_base = txl.local_scalar(
            "int32",
            init=batch_idx * stride_o_b
            + s_q_idx * stride_o_s_q
            + (h_block_idx * BLOCK_SIZE_M + warp_idx) * stride_o_h_q,
        )
        for i in range(elems):
            conv = txl.alloc_local((4,), "uint16")
            for v in range(4):
                txl.ptx[cvt](conv[v], result[4 * i + v])
            txl.ptx.st.global_.u64(
                out.view("uint64").ptr_to([(out_base + lane_idx * 4 + i * 128) // 4]),
                conv.view("uint64")[0],
            )

    return flash_fwd_mla_combine.func

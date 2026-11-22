# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashMLA's SM90 sparse FP8 decode (`csrc/sm90/decode/sparse_fp8/`).

Submodules: `data` (operands, the FP8 KV-cache quantization, host-side parameters, the FlashMLA
reference, checks), `kernel` (the split-KV `PrimFunc`), `launch` (TMA descriptors, compilation,
launch). The scheduler-metadata and combine kernels are shared with the dense decode port
(`.._sm90_dense_decode.sched_meta` / `.combine`): upstream runs the same `smxx/decode` kernels
for both.
"""

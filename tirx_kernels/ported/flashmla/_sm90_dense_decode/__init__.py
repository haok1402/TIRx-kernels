# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Shared TIRx port of FlashMLA's SM90 dense decode (`csrc/sm90/decode/dense/`).

Submodules: `data` (operands, host-side parameters, the FlashMLA reference, checks), `sched_meta`,
`kernel` and `combine` (the three `PrimFunc`s), `launch` (TMA descriptors, compilation, launch).
"""

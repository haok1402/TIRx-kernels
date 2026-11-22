# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of DeepGEMM's SM90 FP8 MQA logits kernel (the DeepSeek-V3.2 indexer).

Upstream sources:
- deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh (`sm90_fp8_mqa_logits`)
- csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp (TMA descriptors, shared memory, launch)
- csrc/apis/attention.hpp (`fp8_fp4_mqa_logits`: block sizes and logits allocation)

For `q[seq_len, H, D]` (FP8), `kv[seq_len_kv, D]` (FP8) with per-row FP32 scales, and per-head
FP32 weights, row `i` of the output is
`logits[i, j] = kv_scale[j] * sum_h relu(q[i, h] . kv[j]) * weights[i, h]` for `j` in
`[cu_seq_len_k_start[i], cu_seq_len_k_end[i])` (dense, at column `j`) or at column
`j - cu_seq_len_k_start[i]` (compressed).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.runner import cache_backend_config, cuda_target

# `fp8_fp4_mqa_logits` on SM90 (csrc/apis/attention.hpp:140-173, .hpp:91-94).
BLOCK_QH = 128
BLOCK_KV = 256
NUM_Q_STAGES = 3
NUM_KV_STAGES = 3
NUM_SPECIALIZED_THREADS = 128
NUM_MATH_THREADS = 512
SM90_SMEM_CAPACITY = 232448

#: `calc_diff` tolerances of tests/test_attention.py::test_mqa_logits (`ref_diff_tol`).
_REF_DIFF_TOL = 1e-3
_SIM_DIFF_TOL_F32 = 5e-6
_SIM_DIFF_TOL_BF16 = 3e-5

#: Quiet-NaN bit patterns both outputs start from, so unwritten elements are compared too.
_SENTINEL_BITS = {"float32": 0x7FC00001, "bfloat16": 0x7FC1}


def _align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


@dataclass(frozen=True)
class MQALogitsSM90Config:
    seq_len: int = 128
    seq_len_kv: int = 1024
    num_heads: int = 64
    head_dim: int = 128
    logits_dtype: str = "float32"
    compressed_logits: bool = False
    # How `cu_seq_len_k_{start,end}` are generated: "cp" / "nocp" are upstream's
    # `generate_ks_ke_tests`; "random" draws arbitrary (possibly empty, unaligned) ranges.
    ks_ke: str = "nocp"
    seed: int = 0

    @property
    def block_q(self) -> int:
        return BLOCK_QH // self.num_heads

    @property
    def aligned_seq_len(self) -> int:
        return _align_up(self.seq_len, self.block_q)

    def validate(self) -> None:
        if self.num_heads not in (32, 64):
            raise ValueError("SM90 MQA logits supports num_heads 32 or 64")
        if self.head_dim not in (32, 64, 128):
            raise ValueError("SM90 MQA logits supports head_dim 32, 64 or 128")
        if self.logits_dtype not in _SENTINEL_BITS:
            raise ValueError("logits_dtype must be 'float32' or 'bfloat16'")
        if self.seq_len <= 0 or self.seq_len_kv <= 0:
            raise ValueError("sequence lengths must be positive")
        if self.ks_ke not in ("cp", "nocp", "random"):
            raise ValueError("ks_ke must be 'cp', 'nocp' or 'random'")
        if self.ks_ke == "cp" and (self.seq_len_kv % self.seq_len != 0 or self.seq_len % 2 != 0):
            raise ValueError("the CP schedule needs seq_len_kv % seq_len == 0 and an even seq_len")
        if self.ks_ke == "nocp" and self.seq_len_kv < self.seq_len:
            raise ValueError("the no-CP schedule needs seq_len_kv >= seq_len")


@dataclass(frozen=True)
class KernelSpec:
    """The template arguments of one `sm90_fp8_mqa_logits` instantiation (`.hpp:45-65`)."""

    num_heads: int
    head_dim: int
    is_compressed_logits: bool
    logits_dtype: str
    num_sms: int

    @property
    def block_q(self) -> int:
        return BLOCK_QH // self.num_heads

    @property
    def smem_size(self) -> int:
        """`sm90_fp8_mqa_logits`'s host carve-out (`.hpp:104-115`)."""
        q_stage = self.block_q * self.num_heads * self.head_dim
        w_stage = self.block_q * self.num_heads * 4
        kv_stage = BLOCK_KV * self.head_dim
        sf_stage = BLOCK_KV * 4
        size = NUM_Q_STAGES * (q_stage + w_stage) + NUM_KV_STAGES * (kv_stage + sf_stage)
        size += (NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 + (NUM_MATH_THREADS // 128) * 2) * 8
        return size + 4


def _label(c: dict[str, Any]) -> str:
    dtype = "f32" if c["logits_dtype"] == "float32" else "bf16"
    mode = "compressed" if c["compressed_logits"] else "dense"
    return (
        f"s{c['seq_len']}_skv{c['seq_len_kv']}_h{c['num_heads']}_d{c['head_dim']}"
        f"_{dtype}_{mode}_{c['ks_ke']}"
    )


def _case(**kwargs: Any) -> dict[str, Any]:
    config = {
        "seq_len": kwargs["seq_len"],
        "seq_len_kv": kwargs["seq_len_kv"],
        "num_heads": kwargs.get("num_heads", 64),
        "head_dim": kwargs.get("head_dim", 128),
        "logits_dtype": kwargs.get("logits_dtype", "float32"),
        "compressed_logits": kwargs.get("compressed_logits", False),
        "ks_ke": kwargs.get("ks_ke", "nocp"),
        "seed": kwargs.get("seed", 0),
    }
    config["label"] = _label(config)
    return config


KERNEL_META = {
    "name": "deepgemm_sm90_fp8_mqa_logits",
    "category": "ported.deepgemm",
    "runtime_cuda_archs": ["sm_90a"],
    "reference_requirements": (
        {
            "package": "deep-gemm",
            "git": {
                "url": "https://github.com/deepseek-ai/DeepGEMM.git",
                "commit": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
            },
            "import": "deep_gemm",
        },
    ),
}

# tests/test_attention.py::test_mqa_logits on SM90 enumerates every
# (logits dtype) x (dense+clean | compressed) x seq_len {2048, 8192} x seq_len_kv {8192, 65536}
# x num_heads {32, 64} x head_dim {32, 64, 128} x (cp | nocp) case: 192 calls, all run by
# sm90-probe's full-grid script.  `CONFIGS` keeps every specialization (heads x head_dim x dtype
# x mode, 24 kernels) at upstream's smallest shape with both schedules, upstream's largest shapes
# for the production indexer specialization (64 heads, head_dim 128), and edge shapes.
UPSTREAM_SM90_GRID = [
    _case(
        seq_len=s,
        seq_len_kv=skv,
        num_heads=h,
        head_dim=d,
        logits_dtype=dt,
        compressed_logits=comp,
        ks_ke=ks,
        seed=i,
    )
    for i, (dt, comp, s, skv, h, d, ks) in enumerate(
        (dt, comp, s, skv, h, d, ks)
        for dt in ("bfloat16", "float32")
        for comp in (False, True)
        for s in (2048, 8192)
        for skv in (8192, 65536)
        for h in (32, 64)
        for d in (32, 64, 128)
        for ks in ("cp", "nocp")
    )
]

_SPECIALIZATION_CASES = [
    c for c in UPSTREAM_SM90_GRID if c["seq_len"] == 2048 and c["seq_len_kv"] == 8192
]
_PRODUCTION_LARGE_CASES = [
    c
    for c in UPSTREAM_SM90_GRID
    if c["num_heads"] == 64
    and c["head_dim"] == 128
    and c["seq_len"] == 8192
    and c["seq_len_kv"] == 65536
]
EDGE_CASES = [
    # seq_len not a multiple of BLOCK_Q, a single query, seq_len_kv not a multiple of 4 or 256.
    _case(seq_len=1, seq_len_kv=1, num_heads=32, head_dim=32, ks_ke="nocp", seed=900),
    _case(seq_len=3, seq_len_kv=257, num_heads=32, head_dim=64, ks_ke="random", seed=901),
    _case(seq_len=1001, seq_len_kv=4099, num_heads=64, head_dim=128, ks_ke="random", seed=902),
    _case(
        seq_len=1001,
        seq_len_kv=4099,
        num_heads=64,
        head_dim=128,
        compressed_logits=True,
        logits_dtype="bfloat16",
        ks_ke="random",
        seed=903,
    ),
    _case(
        seq_len=777,
        seq_len_kv=3001,
        num_heads=32,
        head_dim=32,
        compressed_logits=True,
        ks_ke="random",
        seed=904,
    ),
    # More query blocks than SMs times Q stages: the persistent loop wraps the Q ring.
    _case(seq_len=4099, seq_len_kv=4099, num_heads=64, head_dim=64, ks_ke="nocp", seed=905),
]
CONFIGS = _SPECIALIZATION_CASES + _PRODUCTION_LARGE_CASES + EDGE_CASES

BENCH_CONFIGS = [
    c
    for c in UPSTREAM_SM90_GRID
    if c["num_heads"] == 64 and c["head_dim"] == 128 and c["ks_ke"] == "cp"
]


def _make_config(**kwargs: Any) -> MQALogitsSM90Config:
    kwargs = {k: v for k, v in kwargs.items() if k != "label"}
    config = MQALogitsSM90Config(**kwargs)
    config.validate()
    return config


def load_deep_gemm():
    try:
        import deep_gemm
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"DeepGEMM unavailable: {exc}") from exc
    if not hasattr(deep_gemm, "fp8_fp4_mqa_logits"):
        raise SkipTest("DeepGEMM unavailable: missing fp8_fp4_mqa_logits")
    return deep_gemm


def _require_sm90() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("no CUDA device")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SkipTest("SM90 FP8 MQA logits needs compute capability 9.0")


def _generate_ks_ke(config: MQALogitsSM90Config) -> tuple[torch.Tensor, torch.Tensor]:
    """`generate_ks_ke_tests` (tests/test_attention.py:119), plus arbitrary ranges for edges."""
    s, skv = config.seq_len, config.seq_len_kv
    if config.ks_ke == "nocp":
        ks = torch.zeros(s, dtype=torch.int32, device="cuda")
        ke = torch.arange(s, dtype=torch.int32, device="cuda") + (skv - s)
        return ks, ke
    if config.ks_ke == "cp":
        chunk_size = s // 2
        cp_size = skv // s
        cp_id = cp_size // 3
        i = torch.arange(chunk_size, dtype=torch.int32, device="cuda")
        ke = torch.cat([cp_id * chunk_size + i, (cp_size * 2 - 1 - cp_id) * chunk_size + i])
        return torch.zeros(s, dtype=torch.int32, device="cuda"), ke.to(torch.int32)
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    a = torch.randint(0, skv + 1, (s,), generator=gen)
    b = torch.randint(0, skv + 1, (s,), generator=gen)
    ks, ke = torch.minimum(a, b), torch.maximum(a, b)
    ke[::7] = ks[::7]  # some empty rows
    return ks.to(torch.int32).cuda(), ke.to(torch.int32).cuda()


def _ref_mqa_logits(q, kv, weights, ks, ke) -> torch.Tensor:
    """`ref_fp8_mqa_logits` (tests/test_attention.py:84), chunked over queries."""
    seq_len_kv = kv.shape[0]
    q_f32, kv_f32 = q.float(), kv.float()
    cols = torch.arange(0, seq_len_kv, device="cuda")
    logits = torch.empty((q.shape[0], seq_len_kv), device="cuda", dtype=torch.float32)
    chunk = max(1, (256 * 1024 * 1024) // max(1, seq_len_kv * q.shape[1] * 4))
    for c0 in range(0, q.shape[0], chunk):
        c1 = min(c0 + chunk, q.shape[0])
        score = torch.einsum("mhd,nd->hmn", q_f32[c0:c1], kv_f32)
        part = (score.relu() * weights[c0:c1].transpose(0, 1).unsqueeze(-1)).sum(dim=0)
        mask = (cols[None, :] >= ks[c0:c1, None]) & (cols[None, :] < ke[c0:c1, None])
        logits[c0:c1] = part.masked_fill(~mask, float("-inf"))
    return logits


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm = load_deep_gemm()
    _require_sm90()
    config = _make_config(**kwargs)
    from deep_gemm.utils import per_custom_dims_cast_to_fp8

    torch.manual_seed(config.seed)
    s, skv, h, d = config.seq_len, config.seq_len_kv, config.num_heads, config.head_dim
    q = torch.randn(s, h, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(skv, d, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(s, h, device="cuda", dtype=torch.float32)
    ks, ke = _generate_ks_ke(config)
    q_in = q.to(torch.float8_e4m3fn)
    kv_in = per_custom_dims_cast_to_fp8(kv, (0,), False)
    max_seqlen_k = int((ke - ks).max().item()) if config.compressed_logits else 0
    if config.compressed_logits and max_seqlen_k == 0:
        max_seqlen_k = 1
    return {
        "config": config,
        "deep_gemm": deep_gemm,
        "q": q,
        "kv": kv,
        "q_in": q_in,
        "kv_in": kv_in,
        "weights": weights,
        "ks": ks,
        "ke": ke,
        "max_seqlen_k": max_seqlen_k,
        "num_sms": torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
    }


def kernel_spec(config: MQALogitsSM90Config, num_sms: int) -> KernelSpec:
    return KernelSpec(
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        is_compressed_logits=config.compressed_logits,
        logits_dtype=config.logits_dtype,
        num_sms=num_sms,
    )


def get_kernel(*, backend_config=None, num_sms: int = 132, **kwargs: Any):
    """The TIRx `PrimFunc` of the `sm90_fp8_mqa_logits` instantiation `kwargs` selects."""
    from ._sm90_mqa_logits.kernel import build_kernel

    config = _make_config(**kwargs)
    return build_kernel(kernel_spec(config, num_sms), backend_config=backend_config)


@cache_backend_config
def _compile_cached(spec: KernelSpec, *, backend_config=None):
    import tvm_tirx

    from ._sm90_mqa_logits.kernel import build_kernel

    func = build_kernel(spec, backend_config=backend_config)
    return tvm_tirx.tirx.compile(
        tvm_tirx.IRModule({"main": func}),
        target=cuda_target(backend_config=backend_config),
        pipeline="tirx",
    )


def compile_spec(spec: KernelSpec, *, backend_config=None):
    return _compile_cached(spec, backend_config=backend_config)


def logits_storage(config: MQALogitsSM90Config, max_seqlen_k: int) -> tuple[int, int]:
    """`fp8_fp4_mqa_logits`'s allocation (csrc/apis/attention.hpp:146-158): rows, row stride."""
    alignment = 1024 // _torch_dtype(config.logits_dtype).itemsize
    if max_seqlen_k == 0:
        stride = _align_up(config.seq_len_kv + BLOCK_KV, alignment)
    else:
        stride = _align_up(_align_up(max_seqlen_k, BLOCK_KV), alignment)
    return config.aligned_seq_len, stride


def sentinel_storage(config: MQALogitsSM90Config, max_seqlen_k: int) -> torch.Tensor:
    rows, stride = logits_storage(config, max_seqlen_k)
    dtype = _torch_dtype(config.logits_dtype)
    out = torch.empty((rows, stride), device="cuda", dtype=dtype)
    int_view = torch.int32 if dtype == torch.float32 else torch.int16
    bits = _SENTINEL_BITS[config.logits_dtype]
    if int_view == torch.int16 and bits >= 0x8000:
        bits -= 0x10000
    out.view(int_view).fill_(bits)
    return out


def build_launch(data: dict[str, Any], logits: torch.Tensor, *, executable=None, backend_config=None):
    """A no-argument closure issuing exactly one TIRx launch; descriptors are encoded here."""
    from ._sm90_mqa_logits.launch import make_tensor_maps

    config: MQALogitsSM90Config = data["config"]
    spec = kernel_spec(config, data["num_sms"])
    if executable is None:
        executable = compile_spec(spec, backend_config=backend_config)
    maps = make_tensor_maps(data, spec)
    stride = int(logits.stride(0))
    flat = logits.reshape(-1)
    ks, ke = data["ks"], data["ke"]

    def launch():
        executable.mod(
            config.seq_len,
            config.seq_len_kv,
            data["max_seqlen_k"],
            stride,
            ks,
            ke,
            flat,
            *(m.ptr for m in maps),
        )

    return launch


def run_deepgemm(data: dict[str, Any], *, clean_logits: bool) -> torch.Tensor:
    """`deep_gemm.fp8_fp4_mqa_logits`, returning the full logits storage it allocated."""
    config: MQALogitsSM90Config = data["config"]
    out = data["deep_gemm"].fp8_fp4_mqa_logits(
        q=(data["q_in"], None),
        kv=data["kv_in"],
        weights=data["weights"],
        cu_seq_len_k_start=data["ks"],
        cu_seq_len_k_end=data["ke"],
        clean_logits=clean_logits,
        max_seqlen_k=data["max_seqlen_k"],
        logits_dtype=_torch_dtype(config.logits_dtype),
    )
    rows, stride = logits_storage(config, data["max_seqlen_k"])
    if out.stride(0) != stride:
        raise AssertionError(f"DeepGEMM logits stride {out.stride(0)} != expected {stride}")
    return torch.as_strided(out, (rows, stride), (stride, 1))


def run_deepgemm_into_sentinel(data: dict[str, Any]) -> torch.Tensor:
    """DeepGEMM's raw kernel output (`clean_logits=False`) on sentinel-filled storage.

    `fp8_fp4_mqa_logits` allocates its own output with `torch.empty`; a sentinel tensor of the
    same size is freed just before the call so the caching allocator hands its block back, and
    the reuse is checked by address.
    """
    config: MQALogitsSM90Config = data["config"]
    torch.cuda.synchronize()
    probe = sentinel_storage(config, data["max_seqlen_k"])
    address = probe.data_ptr()
    del probe
    out = run_deepgemm(data, clean_logits=False)
    if out.data_ptr() != address:
        raise AssertionError("DeepGEMM output did not reuse the sentinel block")
    return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def assert_bitwise_identical(actual: torch.Tensor, expected: torch.Tensor, *, detail: str) -> None:
    differ = _bits(actual) != _bits(expected)
    if bool(differ.any()):
        first = tuple(int(i) for i in differ.nonzero()[0])
        raise AssertionError(
            f"{detail}: {int(differ.sum())} of {differ.numel()} logits storage elements differ "
            f"bitwise from DeepGEMM (first at {first})"
        )


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return float((1 - 2 * (x * y).sum() / denominator).item())


def check_against_reference(data: dict[str, Any], storage: torch.Tensor) -> tuple[float, float]:
    """Upstream's validation (tests/test_attention.py:224-238) on a TIRx output."""
    config: MQALogitsSM90Config = data["config"]
    s, skv = config.seq_len, config.seq_len_kv
    ks, ke = data["ks"], data["ke"]
    q_sim = data["q_in"].to(torch.bfloat16)
    kv_fp8, kv_sf = data["kv_in"]
    kv_sim = (kv_fp8.float() * kv_sf.unsqueeze(1)).to(torch.bfloat16)
    ref = _ref_mqa_logits(data["q"], data["kv"], data["weights"], ks, ke)
    sim = _ref_mqa_logits(q_sim, kv_sim, data["weights"], ks, ke)
    if config.compressed_logits:
        logits = torch.full((s, skv), float("-inf"), device="cuda")
        cols = torch.arange(skv, device="cuda")[None, :]
        src = (cols - ks[:, None]).clamp(min=0, max=storage.shape[1] - 1)
        gathered = storage[:s].float().gather(1, src.long())
        valid = (cols >= ks[:, None]) & (cols < ke[:, None])
        logits = torch.where(valid, gathered, logits)
    else:
        logits = storage[:s, :skv].float()
        cols = torch.arange(skv, device="cuda")[None, :]
        valid = (cols >= ks[:, None]) & (cols < ke[:, None])
        logits = logits.masked_fill(~valid, float("-inf"))  # `clean_logits=True`
    ref_neginf = ref == float("-inf")
    if not torch.equal(logits == float("-inf"), ref_neginf):
        raise AssertionError("-inf mask differs from the reference")
    logits = logits.masked_fill(ref_neginf, 0)
    diff = _calc_diff(logits, ref.masked_fill(ref_neginf, 0))
    sim_diff = _calc_diff(logits, sim.masked_fill(ref_neginf, 0))
    sim_tol = _SIM_DIFF_TOL_BF16 if config.logits_dtype == "bfloat16" else _SIM_DIFF_TOL_F32
    if not diff < _REF_DIFF_TOL:
        raise AssertionError(f"diff {diff:.3g} >= {_REF_DIFF_TOL}")
    if not sim_diff < sim_tol:
        raise AssertionError(f"simulated diff {sim_diff:.3g} >= {sim_tol}")
    return diff, sim_diff


def run_test(*, backend_config=None, **kwargs: Any) -> None:
    """Bitwise against DeepGEMM over the whole logits storage, then upstream's tolerances."""
    data = prepare_data(**kwargs)
    config: MQALogitsSM90Config = data["config"]
    expected = run_deepgemm_into_sentinel(data)
    actual = sentinel_storage(config, data["max_seqlen_k"])
    build_launch(data, actual, backend_config=backend_config)()
    torch.cuda.synchronize()
    assert_bitwise_identical(actual, expected, detail=_label(asdict(config)))
    check_against_reference(data, actual)


def prepare_bench(*, backend_config=None, **kwargs: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    _require_sm90()
    config = _make_config(**kwargs)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    executable = compile_spec(kernel_spec(config, num_sms), backend_config=backend_config)
    return prepared_gpu_benchmark(
        run_gpu, {"config": dict(kwargs), "executable": executable}, backend_config=backend_config
    )


def run_gpu(prepared, *, backend_config=None, **kwargs: Any) -> dict[str, Any]:
    from tirx_kernels.runner import bench

    kwargs = {**prepared["config"], **kwargs}
    protocol = {
        name: kwargs.pop(name)
        for name in ("warmup", "repeat", "timer", "rounds", "cooldown_s")
        if name in kwargs
    }
    data = prepare_data(**kwargs)
    config: MQALogitsSM90Config = data["config"]
    logits = sentinel_storage(config, data["max_seqlen_k"])
    launch = build_launch(
        data, logits, executable=prepared["executable"], backend_config=backend_config
    )
    launch()
    torch.cuda.synchronize()
    expected = run_deepgemm_into_sentinel(data)
    assert_bitwise_identical(logits, expected, detail=_label(asdict(config)))
    del expected
    torch.cuda.empty_cache()

    def _deepgemm():
        return lambda: run_deepgemm(data, clean_logits=False)

    result = bench(
        {"tirx": launch},
        warmup=protocol.get("warmup"),
        repeat=protocol.get("repeat"),
        timer=protocol.get("timer"),
        references={"deepgemm": _deepgemm},
        rounds=protocol.get("rounds", 1),
        cooldown_s=protocol.get("cooldown_s", 1.0),
    )
    result["max_diff"] = 0.0
    return result


def run_bench(*, backend_config=None, **kwargs: Any) -> dict[str, Any]:
    protocol = {
        name: kwargs.pop(name)
        for name in ("warmup", "repeat", "timer", "rounds", "cooldown_s")
        if name in kwargs
    }
    return prepare_bench(**kwargs, backend_config=backend_config).run_gpu(**protocol)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "EDGE_CASES",
    "KERNEL_META",
    "UPSTREAM_SM90_GRID",
    "MQALogitsSM90Config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]

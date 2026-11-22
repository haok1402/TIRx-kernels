# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of DeepGEMM's SM90 FP8 paged MQA logits kernel (the DeepSeek-V3.2 decode indexer).

Upstream sources:
- deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh (`sm90_fp8_paged_mqa_logits`)
- deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh (`SM90PagedMQALogitsScheduler`)
- csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp (TMA descriptors, shared memory, launch)
- csrc/apis/attention.hpp (`fp8_fp4_paged_mqa_logits`: split, block size and logits allocation)

The schedule metadata comes from DeepGEMM's own `get_paged_mqa_logits_metadata` kernel, as for the
SM100 port.  For query `(b, t)` of `q[batch, next_n, H, D]` and the paged FP8 KV cache, row
`b * next_n + t` of the output holds `kv_scale[j] * sum_h relu(q[b, t, h] . kv[j]) * weights[b*next_n+t, h]`
for every key position `j` of the batch's KV blocks (the kernel writes whole 64-token blocks).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.runner import cache_backend_config, cuda_target

from .paged_mqa_logits_fp8 import _make_block_table, _make_context_lens, _make_fused_kv_cache

# `fp8_fp4_paged_mqa_logits` on SM90 (csrc/apis/attention.hpp:340-353, .hpp:328-370).
BLOCK_KV = 64
SPLIT_KV = 256
NUM_Q_STAGES = 3
NUM_KV_STAGES = 3
NUM_SPECIALIZED_THREADS = 128
NUM_MATH_THREADS = (SPLIT_KV // 64) * 128

_REF_DIFF_TOL = 1e-3
_SIM_DIFF_TOL_F32 = 5e-6
_SIM_DIFF_TOL_BF16 = 3e-5
_SENTINEL_BITS = {"float32": 0x7FC00001, "bfloat16": 0x7FC1}
_CONTEXT_PATTERNS = ("random_2d", "sglang_fixed", "sglang_ragged", "upstream")
#: Batches whose rows the torch reference is evaluated on for large cases (the reference is a
#: per-batch Python loop); the bitwise comparison with DeepGEMM always covers the whole output.
_REF_MAX_BATCHES = 64


def _align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


@dataclass(frozen=True)
class PagedMQALogitsSM90Config:
    batch_size: int = 1
    next_n: int = 1
    max_num_pages: int = 4
    num_pages: int = 128
    num_heads: int = 64
    head_dim: int = 128
    logits_dtype: str = "float32"
    seed: int = 0
    # "random_2d" / "sglang_fixed" / "sglang_ragged" are the SM100 port's generators;
    # "upstream" is tests/test_attention.py::test_paged_mqa_logits (avg_kv = max_num_pages * 64 / 1.3).
    context_pattern: str = "random_2d"
    avg_kv: int = 0

    page_size = BLOCK_KV  # the SM100 port's generators read these names
    varlen = False

    @property
    def max_context_len(self) -> int:
        return self.max_num_pages * self.page_size

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.next_n not in (1, 2):
            raise ValueError("SM90 paged MQA logits supports next_n 1 or 2")
        if self.num_heads not in (32, 64):
            raise ValueError("SM90 paged MQA logits supports num_heads 32 or 64")
        if self.head_dim not in (32, 64, 128):
            raise ValueError("head_dim must be 32, 64 or 128")
        if self.logits_dtype not in _SENTINEL_BITS:
            raise ValueError("logits_dtype must be 'float32' or 'bfloat16'")
        if self.context_pattern not in _CONTEXT_PATTERNS:
            raise ValueError(f"context_pattern must be one of {_CONTEXT_PATTERNS}")
        if self.context_pattern == "upstream":
            if self.avg_kv <= 0:
                raise ValueError("the upstream pattern needs avg_kv")
        elif self.max_num_pages <= 0 or self.num_pages < self.max_num_pages:
            raise ValueError("num_pages must cover max_num_pages")


@dataclass(frozen=True)
class KernelSpec:
    """The template arguments of one `sm90_fp8_paged_mqa_logits` instantiation (`.hpp:268-285`)."""

    next_n: int
    num_heads: int
    head_dim: int
    logits_dtype: str
    num_sms: int

    @property
    def smem_size(self) -> int:
        """`sm90_fp8_paged_mqa_logits`'s host carve-out (`.hpp:357-367`)."""
        sw = self.head_dim * 8
        q_stage = self.next_n * self.num_heads * self.head_dim
        w_stage = _align_up(self.next_n * self.num_heads * 4, sw)
        q_pipe = NUM_Q_STAGES * (q_stage + w_stage) + _align_up(NUM_Q_STAGES * 8 * 2, sw)
        kv_stage = BLOCK_KV * self.head_dim
        sf_stage = _align_up(BLOCK_KV * 4, sw)
        kv_pipe = NUM_KV_STAGES * (kv_stage + sf_stage) + _align_up(NUM_KV_STAGES * 8 * 2, sw)
        num_wg = SPLIT_KV // 64
        return q_pipe + num_wg * kv_pipe + num_wg * 2 * 8 + 4


def _label(c: dict[str, Any]) -> str:
    dtype = "f32" if c["logits_dtype"] == "float32" else "bf16"
    if c["context_pattern"] == "upstream":
        ctx = f"avg{c['avg_kv']}"
    else:
        ctx = f"mp{c['max_num_pages']}_{c['context_pattern']}"
    return f"b{c['batch_size']}_n{c['next_n']}_h{c['num_heads']}_d{c['head_dim']}_{ctx}_{dtype}"


def _case(**kwargs: Any) -> dict[str, Any]:
    c = {
        "batch_size": kwargs["batch_size"],
        "next_n": kwargs.get("next_n", 1),
        "max_num_pages": kwargs.get("max_num_pages", 4),
        "num_pages": kwargs.get("num_pages", 128),
        "num_heads": kwargs.get("num_heads", 64),
        "head_dim": kwargs.get("head_dim", 128),
        "logits_dtype": kwargs.get("logits_dtype", "float32"),
        "seed": kwargs.get("seed", 0),
        "context_pattern": kwargs.get("context_pattern", "random_2d"),
        "avg_kv": kwargs.get("avg_kv", 0),
    }
    c["label"] = _label(c)
    return c


KERNEL_META = {
    "name": "deepgemm_sm90_fp8_paged_mqa_logits",
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

# tests/test_attention.py::test_paged_mqa_logits on SM90: fp8, f32 weights, block_kv 64, 2D context
# lens without cleaning, batch {256, 4096} x avg_kv {8192, 65536} (batch * avg_kv <= 32M tokens),
# next_n {1, 2}, heads {32, 64}, head_dim 128, logits {bf16, f32}: 24 calls.
UPSTREAM_SM90_GRID = [
    _case(
        batch_size=b,
        next_n=n,
        num_heads=h,
        logits_dtype=dt,
        context_pattern="upstream",
        avg_kv=avg,
        seed=5000 + i,
    )
    for i, (dt, b, n, h, avg) in enumerate(
        (dt, b, n, h, avg)
        for dt in ("bfloat16", "float32")
        for b in (256, 4096)
        for n in (1, 2)
        for h in (32, 64)
        for avg in (8192, 65536)
        if b * avg <= 32 * 1024 * 1024
    )
]

# The SM100 port's coverage, restricted to what the SM90 API accepts (next_n <= 2).
DSA_INDEXER_LIKE_COVERAGE = [
    _case(
        batch_size=b,
        max_num_pages=mp,
        num_pages=max(11923, mp),
        logits_dtype=dt,
        seed=2000 + i,
    )
    for i, (dt, b, mp) in enumerate(
        (dt, b, mp)
        for dt in ("float32", "bfloat16")
        for b in (1, 2, 4, 8, 16)
        for mp in (1, 8, 32, 128)
    )
]
_SGLANG_CONTEXT_PAGES = (64, 160, 512, 1280, 2048)
_SGLANG_TARGET_VERIFY_POINTS = ((1, 64), (2, 2048), (6, 160), (10, 512), (16, 1280))
SGLANG_BENCH_CONFIGS = [
    _case(
        batch_size=b,
        max_num_pages=mp,
        num_pages=max(11923, b * mp),
        num_heads=h,
        seed=3000 + i,
        context_pattern="sglang_fixed",
    )
    for i, (h, b, mp) in enumerate(
        (h, b, mp) for h in (32, 64) for b in (1, 2, 4, 8, 16) for mp in _SGLANG_CONTEXT_PAGES
    )
] + [
    _case(
        batch_size=b,
        next_n=2,
        max_num_pages=mp,
        num_pages=max(11923, b * mp),
        num_heads=h,
        seed=4000 + i,
        context_pattern="sglang_fixed",
    )
    for i, (h, (b, mp)) in enumerate(
        (h, p) for h in (32, 64) for p in _SGLANG_TARGET_VERIFY_POINTS
    )
]
# Every (next_n, heads, head_dim, logits dtype) instantiation once, and edge shapes.
SPECIALIZATION_CASES = [
    _case(
        batch_size=3,
        next_n=n,
        max_num_pages=9,
        num_pages=64,
        num_heads=h,
        head_dim=d,
        logits_dtype=dt,
        seed=6000 + i,
        context_pattern="sglang_ragged",
    )
    for i, (n, h, d, dt) in enumerate(
        (n, h, d, dt)
        for n in (1, 2)
        for h in (32, 64)
        for d in (32, 64, 128)
        for dt in ("float32", "bfloat16")
    )
]
EDGE_CASES = [
    # context shorter than one page, a single page, odd lengths, more SMs than work, ring wraps
    _case(batch_size=1, next_n=1, max_num_pages=1, num_pages=4, seed=7000),
    _case(batch_size=1, next_n=2, max_num_pages=1, num_pages=4, seed=7001),
    _case(batch_size=7, next_n=2, max_num_pages=3, num_pages=32, num_heads=32, seed=7002),
    _case(batch_size=300, next_n=1, max_num_pages=5, num_pages=2048, seed=7003),
    _case(batch_size=33, next_n=2, max_num_pages=40, num_pages=2048, seed=7004,
          context_pattern="sglang_ragged", logits_dtype="bfloat16"),
]
CONFIGS = SPECIALIZATION_CASES + EDGE_CASES + DSA_INDEXER_LIKE_COVERAGE + UPSTREAM_SM90_GRID[:6]
BENCH_CONFIGS = SGLANG_BENCH_CONFIGS


def _make_config(**kwargs: Any) -> PagedMQALogitsSM90Config:
    kwargs = {k: v for k, v in kwargs.items() if k != "label"}
    config = PagedMQALogitsSM90Config(**kwargs)
    config.validate()
    return config


def load_deep_gemm():
    try:
        import deep_gemm
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"DeepGEMM unavailable: {exc}") from exc
    for name in ("fp8_fp4_paged_mqa_logits", "get_paged_mqa_logits_metadata"):
        if not hasattr(deep_gemm, name):
            raise SkipTest(f"DeepGEMM unavailable: missing {name}")
    return deep_gemm


def _require_sm90() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("no CUDA device")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SkipTest("SM90 FP8 paged MQA logits needs compute capability 9.0")


def _upstream_inputs(config: PagedMQALogitsSM90Config):
    """`test_paged_mqa_logits`'s non-varlen, 2D-context-lens recipe for SM90."""
    b, n, avg = config.batch_size, config.next_n, config.avg_kv
    context_lens = torch.randint(int(0.7 * avg), int(1.3 * avg), (b,), device="cuda", dtype=torch.int32)
    blocks_per_q = (context_lens + BLOCK_KV - 1) // BLOCK_KV
    max_model_len = int(blocks_per_q.max().item()) * BLOCK_KV
    total_blocks = int(blocks_per_q.sum().item())
    block_table = torch.zeros((b, int(blocks_per_q.max().item())), device="cuda", dtype=torch.int32)
    pool = torch.randperm(total_blocks, device="cuda", dtype=torch.int32)
    offset = 0
    for i, nb in enumerate(blocks_per_q.tolist()):
        block_table[i, :nb] = pool[offset : offset + nb]
        offset += nb
    lens_2d = ((context_lens.unsqueeze(1) + 1) * torch.rand(b, n, device="cuda")).int()
    lens_2d[:, -1] = context_lens
    return lens_2d.contiguous(), block_table, total_blocks, max_model_len


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm = load_deep_gemm()
    _require_sm90()
    config = _make_config(**kwargs)
    torch.manual_seed(config.seed)
    if config.context_pattern == "upstream":
        context_lens, block_table, num_pages, max_context_len = _upstream_inputs(config)
        gen = PagedMQALogitsSM90Config(**{**asdict(config), "num_pages": num_pages,
                                          "max_num_pages": max_context_len // BLOCK_KV})
    else:
        gen = config
        context_lens = _make_context_lens(config)
        block_table = _make_block_table(config)
        max_context_len = config.max_context_len
    b, n, h, d = config.batch_size, config.next_n, config.num_heads, config.head_dim
    q_bf16 = torch.randn(b, n, h, d, device="cuda", dtype=torch.bfloat16).clamp_(-2.0, 2.0)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn).contiguous()
    fused_kv_cache, kv_dequant = _make_fused_kv_cache(gen, keep_dequant=True)
    weights = torch.randn(b * n, h, device="cuda", dtype=torch.float32).contiguous()
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, BLOCK_KV, num_sms)
    return {
        "config": config,
        "gen": gen,
        "deep_gemm": deep_gemm,
        "q": q_fp8,
        "fused_kv_cache": fused_kv_cache,
        "kv_dequant": kv_dequant,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "schedule_meta": schedule_meta,
        "max_context_len": max_context_len,
        "num_sms": num_sms,
    }


def kernel_spec(config: PagedMQALogitsSM90Config, num_sms: int) -> KernelSpec:
    return KernelSpec(
        next_n=config.next_n,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        logits_dtype=config.logits_dtype,
        num_sms=num_sms,
    )


def get_kernel(*, backend_config=None, num_sms: int = 132, **kwargs: Any):
    from ._sm90_paged_mqa_logits.kernel import build_kernel

    return build_kernel(kernel_spec(_make_config(**kwargs), num_sms), backend_config=backend_config)


@cache_backend_config
def _compile_cached(spec: KernelSpec, *, backend_config=None):
    import tvm_tirx

    from ._sm90_paged_mqa_logits.kernel import build_kernel

    func = build_kernel(spec, backend_config=backend_config)
    return tvm_tirx.tirx.compile(
        tvm_tirx.IRModule({"main": func}),
        target=cuda_target(backend_config=backend_config),
        pipeline="tirx",
    )


def compile_spec(spec: KernelSpec, *, backend_config=None):
    return _compile_cached(spec, backend_config=backend_config)


def logits_storage(data: dict[str, Any]) -> tuple[int, int]:
    """`fp8_fp4_paged_mqa_logits`'s allocation (attention.hpp:330-334): rows, row stride."""
    config: PagedMQALogitsSM90Config = data["config"]
    alignment = 1024 // _torch_dtype(config.logits_dtype).itemsize
    stride = _align_up(_align_up(data["max_context_len"], SPLIT_KV), alignment)
    return config.batch_size * config.next_n, stride


def sentinel_storage(data: dict[str, Any]) -> torch.Tensor:
    config: PagedMQALogitsSM90Config = data["config"]
    rows, stride = logits_storage(data)
    out = torch.empty((rows, stride), device="cuda", dtype=_torch_dtype(config.logits_dtype))
    if config.logits_dtype == "float32":
        out.view(torch.int32).fill_(_SENTINEL_BITS["float32"])
    else:
        out.view(torch.int16).fill_(_SENTINEL_BITS["bfloat16"])
    return out


def build_launch(data: dict[str, Any], logits: torch.Tensor, *, executable=None, backend_config=None):
    """A no-argument closure issuing exactly one TIRx launch; descriptors are encoded here."""
    from ._sm90_paged_mqa_logits.launch import make_tensor_maps

    config: PagedMQALogitsSM90Config = data["config"]
    spec = kernel_spec(config, data["num_sms"])
    if executable is None:
        executable = compile_spec(spec, backend_config=backend_config)
    maps = make_tensor_maps(data, spec)
    flat = logits.reshape(-1)
    block_table = data["block_table"]

    def launch():
        executable.mod(
            config.batch_size,
            int(logits.stride(0)),
            int(block_table.stride(0)),
            data["context_lens"].reshape(-1),
            flat,
            block_table.reshape(-1),
            data["schedule_meta"].reshape(-1),
            *(m.ptr for m in maps),
        )

    return launch


def run_deepgemm(data: dict[str, Any]) -> torch.Tensor:
    """`deep_gemm.fp8_fp4_paged_mqa_logits` (no cleaning), returning its full logits storage."""
    config: PagedMQALogitsSM90Config = data["config"]
    out = data["deep_gemm"].fp8_fp4_paged_mqa_logits(
        (data["q"], None),
        data["fused_kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["schedule_meta"],
        data["max_context_len"],
        False,
        _torch_dtype(config.logits_dtype),
        None,
    )
    rows, stride = logits_storage(data)
    if out.stride(0) != stride:
        raise AssertionError(f"DeepGEMM logits stride {out.stride(0)} != expected {stride}")
    return torch.as_strided(out, (rows, stride), (stride, 1))


def run_deepgemm_into_sentinel(data: dict[str, Any]) -> torch.Tensor:
    """DeepGEMM's output on sentinel-filled storage (its block reused by the caching allocator)."""
    for _ in range(3):
        torch.cuda.synchronize()
        probe = sentinel_storage(data)
        address = probe.data_ptr()
        del probe
        out = run_deepgemm(data)
        if out.data_ptr() == address:
            return out
        del out
    raise AssertionError("DeepGEMM output did not reuse the sentinel block")


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


def _ref_rows(q, kv_dequant, weights, context_lens, block_table, max_len):
    """`ref_paged_mqa_logits` (tests/test_attention.py:256, 2D context lens), one batch at a time.

    Returns `[len(q) * next_n, max_len]` f32 logits, -inf past each row's context length.
    """
    n = q.shape[1]
    out = torch.full((q.shape[0] * n, max_len), float("-inf"), device="cuda")
    for i in range(q.shape[0]):
        ctx = int(context_lens[i, -1].item())
        nb = (ctx + BLOCK_KV - 1) // BLOCK_KV
        kv = kv_dequant[block_table[i, :nb].long()].reshape(nb * BLOCK_KV, -1).float()[:ctx]
        s = torch.einsum("nhd,td->nht", q[i].float(), kv)
        logits = (s.relu() * weights[i * n : (i + 1) * n, :, None]).sum(dim=1)
        out[i * n : (i + 1) * n, :ctx] = logits
    return out


def check_against_reference(data: dict[str, Any], storage: torch.Tensor) -> tuple[float, float]:
    """Upstream's validation (`test_paged_mqa_logits`) on (a sample of) the TIRx output."""
    config: PagedMQALogitsSM90Config = data["config"]
    b, n = config.batch_size, config.next_n
    batches = torch.arange(b)
    if b > _REF_MAX_BATCHES:
        batches = torch.randperm(b, generator=torch.Generator().manual_seed(config.seed))[:_REF_MAX_BATCHES]
    max_len = data["max_context_len"]
    sel = batches.cuda()
    rows = (sel[:, None] * n + torch.arange(n, device="cuda")[None, :]).reshape(-1)
    q, w = data["q"][sel], data["weights"][rows]
    lens, table = data["context_lens"][sel], data["block_table"][sel]
    # The dequantized FP8 operands: upstream's "simulated" reference.
    sim = _ref_rows(q.to(torch.bfloat16), data["kv_dequant"], w, lens, table, max_len)
    logits = storage[rows, :max_len].float()
    positions = torch.arange(max_len, device="cuda")[None, :]
    mask = ~(positions < lens.reshape(-1, 1))
    logits, sim = logits.masked_fill(mask, 0), sim.masked_fill(mask, 0)
    sim_diff = _calc_diff(logits, sim)
    tol = _SIM_DIFF_TOL_BF16 if config.logits_dtype == "bfloat16" else _SIM_DIFF_TOL_F32
    if not sim_diff < tol:
        raise AssertionError(f"simulated diff {sim_diff:.3g} >= {tol}")
    return sim_diff, sim_diff


def run_test(*, backend_config=None, **kwargs: Any) -> None:
    """Bitwise against DeepGEMM over the whole logits storage, then the reference tolerance."""
    data = prepare_data(**kwargs)
    expected = run_deepgemm_into_sentinel(data)
    actual = sentinel_storage(data)
    build_launch(data, actual, backend_config=backend_config)()
    torch.cuda.synchronize()
    assert_bitwise_identical(actual, expected, detail=_label(asdict(data["config"])))
    del expected
    check_against_reference(data, actual)


def prepare_bench(*, backend_config=None, **kwargs: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    from tirx_kernels.runner import hardware_num_sms

    # Prepare stays CUDA-free: the SM count is the compile profile's (TIRX_PREPARE_NUM_SMS, or the
    # H100/H200 SXM count, not the runner's B200 default). run_gpu recompiles if the device differs.
    config = _make_config(**kwargs)
    num_sms = hardware_num_sms(default=132)
    executable = compile_spec(kernel_spec(config, num_sms), backend_config=backend_config)
    return prepared_gpu_benchmark(
        run_gpu,
        {"config": dict(kwargs), "executable": executable, "num_sms": num_sms},
        backend_config=backend_config,
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
    logits = sentinel_storage(data)
    # The grid and the schedule metadata use the device's SM count: use the prepared kernel only if
    # it was compiled for the same count.
    executable = prepared["executable"] if prepared["num_sms"] == data["num_sms"] else None
    launch = build_launch(data, logits, executable=executable, backend_config=backend_config)
    launch()
    torch.cuda.synchronize()
    expected = run_deepgemm_into_sentinel(data)
    assert_bitwise_identical(logits, expected, detail=_label(asdict(data["config"])))
    del expected
    torch.cuda.empty_cache()

    def _deepgemm():
        return lambda: run_deepgemm(data)

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
    "SPECIALIZATION_CASES",
    "UPSTREAM_SM90_GRID",
    "PagedMQALogitsSM90Config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]

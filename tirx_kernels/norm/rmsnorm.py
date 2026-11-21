# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Persistent RMSNorm (SM100 family and SM90) expressed as one traced tirx-lite device body."""

import math
from typing import Any

import torch

import tirx_kernels.tirx_lite as txl
from tirx_kernels.norm import _rmsnorm_sm90 as _sm90
from tirx_kernels.runner import bench, hardware_num_sms, resolve_backend_config

eps = 1e-6
F16_BYTES = 2
SM_COUNT = 152
# Hopper parts: H100 SXM / H200 have 132 SMs. The persistent grid is sized from the
# device when CUDA is initialized (runner.hardware_num_sms), else from this default.
SM90_SM_COUNT = 132
SM90_CTAS_PER_SM = 8  # persistent CTAs per SM on Hopper (see make_kernel)


def ceildiv(a, b):
    return (a + b - 1) // b


def prepare_data(batch_size, dim):
    torch.manual_seed(42)
    return (
        torch.randn(batch_size, dim, dtype=torch.float16, device="cuda"),
        torch.randn(dim, dtype=torch.float16, device="cuda"),
    )


KERNEL_META = {
    "name": "rmsnorm",
    "category": "norm",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_90a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "f2e04400e330fb2debe0bf8730d9424a1d37927f",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}
CONFIGS = (
    [
        {"hidden_size": hs, "batch_size": bs, "label": f"hs{hs}_bs{bs}"}
        for hs in [128, 4096, 5120, 8192]
        for bs in [1, 2, 4, 8, 16, 32, 64, 128, 4113]
    ]
    + [
        # the large-batch regime (the sm_90 ticket ring, incl. its >8192-row configuration)
        {"hidden_size": hs, "batch_size": 16384, "label": f"hs{hs}_bs16384"}
        for hs in [4096, 5120, 8192]
    ]
    + [
        # More than 2**31 elements: a 32-bit row * hidden_size wraps here (about 4.4 GB per tensor).
        {"hidden_size": 8192, "batch_size": 270000, "label": "hs8192_bs270000_gt2p31"},
        {"hidden_size": 128, "batch_size": (1 << 24) + 1024, "label": "hs128_bs16778240_gt2p31"},
    ]
)
LARGE_ELEMENTS = 1 << 31


FULL_MASK = 0xFFFFFFFF
LD_G_V4 = "ld.global.v4.b32"
ST_G_V4 = "st.global.v4.b32"
LD_S_V4 = "ld.shared.v4.f32"
ST_S_V4 = "st.shared.v4.f32"
LD_S_F32 = "ld.shared.f32"
ST_S_F32 = "st.shared.f32"
MUL_F32X2 = "mul.rz.ftz.f32x2"
MUL_F32 = "mul.rz.ftz.f32"
CVT_F16X2 = "cvt.rn.f16x2.f32"


def _is_sm90(backend_config):
    """Whether to build the sm_90 variant. With no CUDA device and no explicit arch (e.g. a CPU-only
    analysis host), keep building the default sm_100-family kernel, as before the sm_90 path."""
    try:
        arch = resolve_backend_config(backend_config)["cuda"]["arch"]
    except ValueError:
        return False
    return arch.startswith("sm_9")


def make_kernel(hidden_size: int, *, backend_config=None):
    """Trace the kernel for one ``hidden_size``. Batch size stays dynamic.

    On sm_90 the packed ``mul.rz.ftz.f32x2`` (sm_100+) becomes two scalar
    ``mul.rz.ftz.f32``: packed f32x2 arithmetic is defined lane by lane, so the
    result is bit-identical, and Hopper has no packed FMUL to lose. It needs no
    opaque helper call, so the kernel passes tirx-lite's IR contract with no
    ``allowed_func_calls`` grant. The persistent grid uses the Hopper SM count.
    The sm_100-family kernel is unchanged.
    """
    sm90 = _is_sm90(backend_config)
    # Persistent grid: one CTA per SM on the sm_100 family, as before. On H200 one
    # 256-thread CTA per SM keeps too few loads in flight to approach HBM bandwidth;
    # 1/2/4/8 CTAs per SM measured 0.43/0.67/0.88/0.92x flashinfer at hs4096 x 16384.
    sm_count = hardware_num_sms(default=SM90_SM_COUNT) * SM90_CTAS_PER_SM if sm90 else SM_COUNT
    # orig:L650-653 — the schedule is a pure function of hidden_size.
    vec = math.gcd(16 // F16_BYTES, hidden_size)
    block = min(256, hidden_size // vec)
    bdx = 32
    bdy = ceildiv(block, 32)
    nthreads = bdx * bdy
    n_tiles = ceildiv(hidden_size, vec * nthreads)
    if vec != 8:
        # The original's body is written around a fixed vector width: one
        # `ld.global.v4.b32` is 4 words = 8 halves, and the shared staging is
        # two `*.v4.f32`. gcd(8, hidden_size) < 8 silently mismatches those.
        raise ValueError(
            f"hidden_size={hidden_size} gives vec={vec}; this kernel's v4 "
            "load/store shape requires hidden_size % 8 == 0"
        )

    # min_blocks_per_sm is left at its default (None), which omits the second
    # __launch_bounds__ operand — the original's exact spelling, ptxas picking
    # its own occupancy target. Measured (nvcc -Xptxas -v), original vs port:
    #     hs=128  34 | 34    hs=4096  32 | 32
    #     hs=5120 32 | 32    hs=8192  32 | 32
    # i.e. identical at every configured hidden_size. This replaces an earlier
    # `min_blocks_per_sm=8` pin, which was needed only while the substrate
    # always emitted the operand: pinning 1 let ptxas spend up to 255 registers
    # per thread and it took 54 where the original took 32 (a measured ~1.2%),
    # and 8 was the smallest value that clawed the allocation back. `None` is
    # strictly better — it also matches at hs=128, where the pin gave 32 vs the
    # original's 34.
    @txl.kernel()
    def rmsnorm(
        inp: txl.Tensor((txl.int64(),), "float16"),
        wgt: txl.Tensor((txl.int64(),), "float16"),
        out: txl.Tensor((txl.int64(),), "float16"),
        # This Tensor parameter is 1-D, so the [batch, hidden] shape cannot carry the row
        # count the way the original's match_buffer does; it is an argument.
        batch_size: "int32",
    ):
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=sm_count, block=bdy * 32),
            backend_config=backend_config,
        )

        bx = txl.cta_id()
        tid = txl.thread_id()
        lane = tid & 31  # orig tx
        warp = tid >> 5  # orig ty; orig's `thread_id` is ty*bdx+tx == tid

        # orig:L667-670. swizzle=None is the identity — a plain row-major tirx
        # buffer, addressed through ptr_to and raw ld/st like the original.
        smem = txl.smem_pool()
        x_smem = smem.alloc([hidden_size], "float32")
        sum_sq_smem = smem.alloc([bdy], "float32")

        # orig:L671-682. A traced body has no annotated-declaration form, so
        # every register array is an explicit local.
        iw = txl.alloc_local([vec // 2], "uint32")  # input_words
        ww = txl.alloc_local([vec // 2], "uint32")  # weight_words
        ow = txl.alloc_local([vec // 2], "uint32")  # output_words
        xf = txl.alloc_local([vec], "float32")  # input_vec_f32
        wf = txl.alloc_local([vec], "float32")  # weight_vec_f32
        xv = txl.alloc_local([vec], "float32")  # x_vec
        pk = txl.local_scalar("uint64")  # packed_mul
        ss = txl.alloc_local([1], "float32")  # sum_sq
        rms = txl.local_scalar("float32")  # rms_norm
        row = txl.local_scalar("int32")  # idx
        goff = txl.local_scalar("int32")  # global element offset (see gidx)
        rowoff = txl.local_scalar("int64")  # row * hidden_size, 64-bit (see row_gidx)
        roff = txl.local_scalar("int64")

        def cta_sync(bar_id):
            txl.ptx.bar.sync(txl.uint32(bar_id), txl.uint32(nthreads))

        def gidx(offset):
            """A global element index, promoted to the buffer's int64 axis ONCE.

            This Tensor parameter's extent is an ``int64`` Var, so an index built out of
            int32 terms is widened term by term — three ``IMAD.WIDE`` where the
            original (whose match_buffer shape is int32) does the whole address
            in 32 bits and widens at the pointer.

            Casting the *expression* is not enough: the simplifier distributes a
            cast over a sum and the int64 terms come back. Landing the finished
            offset in an int32 local gives the cast a single Var to sit on, and
            that reproduces the original's arithmetic exactly. The int32 range
            this implies is the original's too (largest config: 4113*8192).
            """
            txl.assign(goff, offset)
            return txl.cast(goff, "int64")

        def row_gidx(col):
            """The element index ``row * hidden_size + col`` of the current row, in 64 bits.

            ``batch_size * hidden_size`` may reach 2**31 (e.g. 262144 rows at 8192), and a
            32-bit ``row * hidden_size`` then wraps to a wrong row. The row's offset is formed
            once per row in ``rowoff``, so each access adds only its in-row column (< hidden_size,
            32-bit-safe) to it.
            """
            txl.assign(roff, rowoff + txl.cast(col, "int64"))
            return roff

        def warp_sum(acc):
            """Butterfly sum of ``acc[0]`` over the 32 lanes — orig:L725-729.

            ``shfl.sync.bfly.b32`` in DPS form: the destination pins the warp
            collective to this convergent point, one instruction per round,
            where the original's annotated assignment puts it.  ``bdx`` is 32,
            so the clamp/segmask operand is the width-32 value 31.
            """
            peer = txl.local_scalar("uint32")
            for delta in (16, 8, 4, 2, 1):
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret(acc[0], ty="uint32"),
                    txl.uint32(delta),
                    txl.uint32(31),
                    txl.uint32(FULL_MASK),
                )
                txl.assign(acc[0], acc[0] + txl.reinterpret(peer, ty="float32"))

        def scale_pair(dst, i, a, b):
            """``dst[2i:2i+2] = a * b`` lane by lane, ``a`` and ``b`` given as float pairs.

            sm_100 family: one packed f32x2 multiply. The instruction takes one
            64-bit destination and two 64-bit sources, not a two-element float
            window (port notes §3.2), so the product lands in ``pk`` and is
            unpacked with float2_x/float2_y — exactly the original's register
            shape (orig:L785-799). sm_90: the same rounding and ftz per lane.
            """
            if sm90:
                for lane in range(2):
                    txl.ptx[MUL_F32](dst[2 * i + lane], a[lane], b[lane])
            else:
                txl.ptx[MUL_F32X2](pk, txl.cuda.make_float2(*a), txl.cuda.make_float2(*b))
                txl.ptx.mov.b32(dst[2 * i], txl.cuda.float2_x(pk))
                txl.ptx.mov.b32(dst[2 * i + 1], txl.cuda.float2_y(pk))

        txl.assign(row, bx)
        with txl.while_(row < batch_size):
            txl.assign(rowoff, txl.cast(row, "int64") * hidden_size)
            # ---- pass 1: read x, accumulate sum(x^2), stage x in f32 smem ---
            txl.assign(ss[0], txl.float32(0.0))
            with txl.serial(n_tiles) as ki:
                for kv in range(vec):
                    txl.ptx.mov.b32(xv[kv], txl.float32(0.0))
                st = (ki * nthreads + tid) * vec
                with txl.if_(st < hidden_size), txl.then_():
                    txl.ptx[LD_G_V4](iw[0], iw[1], iw[2], iw[3], inp.ptr_to([row_gidx(st)]))
                    for pair in range(vec // 2):
                        txl.idioms.cast_f16x2_to_f32x2(xf, pair, iw[pair])
                    for kv in range(vec):
                        txl.assign(ss[0], ss[0] + xf[kv] * xf[kv])
                        txl.ptx.mov.b32(xv[kv], xf[kv])
                    txl.ptx[ST_S_V4](x_smem.ptr_to([st]), xv[0], xv[1], xv[2], xv[3])
                    txl.ptx[ST_S_V4](x_smem.ptr_to([st + 4]), xv[4], xv[5], xv[6], xv[7])

            # ---- CTA reduction: warp butterfly, then warp 0 over the warps --
            warp_sum(ss)
            with txl.if_(lane == 0), txl.then_():
                txl.ptx[ST_S_F32](sum_sq_smem.ptr_to([warp]), ss[0])
            cta_sync(0)
            with txl.if_(warp == 0):
                with txl.then_():
                    with txl.if_(lane < bdy):
                        with txl.then_():
                            txl.ptx[LD_S_F32](ss[0], sum_sq_smem.ptr_to([lane]))
                        with txl.else_():
                            txl.assign(ss[0], txl.float32(0.0))
                    warp_sum(ss)
                    with txl.if_(lane == 0), txl.then_():
                        txl.ptx[ST_S_F32](sum_sq_smem.ptr_to([0]), ss[0])
            cta_sync(0)
            txl.ptx[LD_S_F32](ss[0], sum_sq_smem.ptr_to([0]))
            txl.assign(rms, txl.rsqrt(ss[0] / txl.float32(hidden_size) + txl.float32(eps)))

            # ---- pass 2: rescale by rms, apply the weight, write out --------
            with txl.serial(n_tiles) as ki:
                for kv in range(vec):
                    txl.ptx.mov.b32(wf[kv], txl.float32(0.0))
                    txl.ptx.mov.b32(xv[kv], txl.float32(0.0))
                st = (ki * nthreads + tid) * vec
                with txl.if_(st < hidden_size), txl.then_():
                    txl.ptx[LD_G_V4](ww[0], ww[1], ww[2], ww[3], wgt.ptr_to([gidx(st)]))
                    txl.ptx[LD_S_V4](xv[0], xv[1], xv[2], xv[3], x_smem.ptr_to([st]))
                    txl.ptx[LD_S_V4](xv[4], xv[5], xv[6], xv[7], x_smem.ptr_to([st + 4]))
                    for pair in range(vec // 2):
                        txl.idioms.cast_f16x2_to_f32x2(wf, pair, ww[pair])
                # Both multiplies run UNGUARDED, on the zeros written above
                # when this lane is past the end (orig:L784-799). Only the
                # loads and the store are predicated.
                for pair in range(vec // 2):
                    scale_pair(xf, pair, (xv[2 * pair], xv[2 * pair + 1]), (rms, rms))
                for pair in range(vec // 2):
                    scale_pair(
                        xf, pair, (xf[2 * pair], xf[2 * pair + 1]), (wf[2 * pair], wf[2 * pair + 1])
                    )
                with txl.if_(st < hidden_size), txl.then_():
                    for pair in range(vec // 2):
                        txl.ptx[CVT_F16X2](ow[pair], xf[2 * pair + 1], xf[2 * pair])
                    txl.ptx[ST_G_V4](out.ptr_to([row_gidx(st)]), ow[0], ow[1], ow[2], ow[3])
            cta_sync(1)
            txl.assign(row, row + sm_count)

    return rmsnorm


def _sm90_device(tensor) -> bool:
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] == 9


def _kernel_args(input_data, weights, output):
    """Launch arguments for the kernel ``get_kernel`` returns for these tensors' device and shape.

    On Hopper (and a hidden size the sm_90 kernels take) the kernels also take a ticket counter,
    zeroed here and returned to zero by every launch, and the grid.

    **Do not launch one returned tuple concurrently**: not on two streams, and not from a CUDA graph
    replayed concurrently with another launch of it. The launches would share the counter, and
    their rows get corrupted. Sequential launches on one stream, and sequential graph replays, are
    fine. Call this once per concurrent launch site; every call allocates its own counter.
    """
    batch_size, hidden_size = input_data.shape[0], input_data.shape[-1]
    args = (input_data.view(-1), weights, output.view(-1), batch_size)
    if not (_sm90_device(input_data) and _sm90.supported(hidden_size)):
        return args
    p = _sm90.plan(hidden_size, batch_size)
    grid = _sm90.grid_for(p, batch_size, hardware_num_sms(default=SM90_SM_COUNT))
    ctr = torch.zeros(2, dtype=torch.uint32, device=input_data.device)
    return (*args[:3], ctr, batch_size, grid)


def get_kernel(hidden_size, batch_size=None, *, backend_config=None, **kwargs):
    """The kernel for ``hidden_size`` (and, on sm_90, ``batch_size``'s plan; any batch runs on it).

    sm_90: the H200-evolved kernels of ``_rmsnorm_sm90`` (register kernel up to 1024 rows,
    persistent ticket ring above), with this module's original sm_90 port for hidden sizes they do
    not take (e.g. 128).
    """
    if _is_sm90(backend_config) and _sm90.supported(hidden_size):
        return _sm90.make_kernel(hidden_size, _sm90.plan(hidden_size, batch_size)).func
    return make_kernel(hidden_size, backend_config=backend_config).func


def prepare_bench(*, backend_config=None, **kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {
        "config": dict(kwargs),
        "executable": compile_kernel(
            get_kernel(**kwargs, backend_config=backend_config), backend_config=backend_config
        ),
    }
    return prepared_gpu_benchmark(run_gpu, state, backend_config=backend_config)


def run_test(hidden_size, batch_size, *, backend_config=None, **kwargs):
    """Compile, run, and verify rmsnorm kernel."""
    import torch

    from tirx_kernels.runner import compile_kernel

    input_data, weights = prepare_data(batch_size, hidden_size)
    kernel = get_kernel(hidden_size, batch_size, backend_config=backend_config)
    ex = compile_kernel(kernel, backend_config=backend_config)
    output_tir = torch.empty((batch_size, hidden_size), dtype=torch.float16, device="cuda")
    args = _kernel_args(input_data, weights, output_tir)
    # Launch repeatedly on one argument tuple: the sm_90 ring's ticket counter must come back to
    # zero after every launch, or a later launch skips rows (NaN-filled first, so a skipped row fails).
    for _ in range(3):
        output_tir.fill_(float("nan"))
        ex(*args)
    torch.cuda.synchronize()
    if len(args) > 4:
        counter = args[3].cpu().tolist()
        assert counter == [0, 0], f"ticket counter not reset: {counter}"
    if batch_size * hidden_size >= LARGE_ELEMENTS:
        _check_large(output_tir, input_data, weights)
        return
    # FlashInfer's rmsnorm on the same inputs is the arbiter (the same library
    # the benchmark path compares against).
    import flashinfer

    ref = torch.empty_like(output_tir)
    flashinfer.norm.rmsnorm(input_data, weights, eps, enable_pdl=False, out=ref)
    torch.cuda.synchronize()
    torch.testing.assert_close(output_tir.cpu(), ref.cpu(), rtol=0.001, atol=0.001)


def _check_large(output, input_data, weights):
    """Above 2**31 elements: every row against a float32 torch reference, in chunks.

    FlashInfer is not assumed to be 64-bit-safe at this size. Every row is checked, not a sample:
    a wrapped 32-bit offset corrupts rows far from the 2**31 boundary too (on H200, 270000 x 8192
    with 32-bit offsets first goes wrong at row 7856).
    """
    import torch

    batch = input_data.shape[0]
    chunk = 8192
    for start in range(0, batch, chunk):
        x = input_data[start : start + chunk].float()
        ref = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weights.float()).half()
        torch.testing.assert_close(
            output[start : start + chunk], ref, rtol=0.001, atol=0.001, msg=f"rows from {start}"
        )


# timer=None inherits the global default (proton). Proton matters here: rmsnorm is a
# tiny (~2µs) kernel whose event wall is ~3x inflated by launch overhead, and its
# reference is flashinfer (Python-dispatch-heavy). Proton measures the true ~2µs kernel
# time and an undistorted ratio.
def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, backend_config=None, **kwargs):
    """Allocate, validate, and measure after GPU assignment."""
    return _run_gpu(
        prepared["executable"],
        **prepared["config"],
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        **kwargs,
    )


def _run_gpu(ex, hidden_size, batch_size, warmup=None, repeat=None, timer=None, **kwargs):
    """GPU-stage implementation shared by suite and standalone execution."""

    import torch

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    input_data, weights = prepare_data(batch_size, hidden_size)
    input_cuda = input_data.cuda()
    weights_cuda = weights.cuda()
    output_cuda = torch.empty((batch_size, hidden_size), dtype=torch.float16, device="cuda")

    args = _kernel_args(input_cuda, weights_cuda, output_cuda)
    funcs = {"tir": lambda: ex(*args)}

    def _flashinfer():
        import flashinfer

        out_fi = torch.zeros_like(input_cuda)
        return lambda: flashinfer.norm.rmsnorm(
            input_cuda, weights_cuda, eps, enable_pdl=False, out=out_fi
        )

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer": _flashinfer},
        **kwargs,
    )


def run_bench(
    hidden_size, batch_size, warmup=None, repeat=None, timer=None, *, backend_config=None, **kwargs
):
    """Standalone wrapper over the same explicit prepare and GPU stages."""
    return prepare_bench(
        hidden_size=hidden_size, batch_size=batch_size, backend_config=backend_config
    ).run_gpu(warmup=warmup, repeat=repeat, timer=timer, **kwargs)

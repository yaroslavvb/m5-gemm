"""Benchmark tiled fp32 GEMM kernels against MPS on Apple Silicon.

Two kernels are tried:

* ``async_copy.metal`` -- the original tutorial kernel from
  https://percisely.xyz/gemm. Relies on inline-asm linkage to the
  undocumented ``air.simdgroup_async_copy_2d`` intrinsic. The Metal-4
  toolchain (macOS 26+) rejects ``__asm("air.*")`` strings, so this
  fails to compile. We attempt it and report the failure.
* ``sync_copy.metal`` -- same tiling structure, but with a synchronous
  cooperative load that only uses public Metal APIs. Compiles and runs
  on Metal 4.

Run ``python matmul.py --dim 4096`` to reproduce the headline number.
"""
import argparse
import ctypes
import time

import numpy as np

from metal import compiled, compiled_from_metallib, MetalCompileError
from mps_matmul import run_mps_matmul


def time_matmul(kernel, gridShape, threadShape, A, B, C, alpha, beta, samples=10):
    n, k = A.shape
    assert B.shape[0] == k, f"inner dim missmatch: {B.shape[0]} vs. {k}"
    k, m = B.shape
    assert C.shape == (n, m), f"C dim missmatch: {C.shape} vs. ({n},{m})"
    A, B, C = A.astype(np.float32), B.astype(np.float32), C.astype(np.float32)
    goal = A @ B * alpha + C * beta
    times = []
    for i in range(samples):
        res, t = kernel(
            lambda threadw, threadmax: gridShape(n, m, threadw, threadmax),
            lambda threadw, threadmax: threadShape(n, m, threadw, threadmax),
            False,
            ctypes.c_uint32(n),
            ctypes.c_uint32(k),
            ctypes.c_uint32(m),
            ctypes.c_float(alpha),
            ctypes.c_float(beta),
            A.ravel(),
            B.ravel(),
            C.ravel(),
        )
        out_buf = res[-1]
        out = np.frombuffer(
            out_buf.contents().as_buffer(out_buf.length()), dtype=np.float32
        ).reshape((n, m))
        if not np.allclose(out, goal, rtol=1e-4):
            max_abs = np.max(np.abs(out - goal))
            max_rel = max_abs / max(np.max(np.abs(goal)), 1e-30)
            raise AssertionError(
                f"incorrect on iter ({i}): max_abs={max_abs:.4g}, max_rel={max_rel:.4g}"
            )
        times.append(t)
    return sum(times) / len(times)


def tile_config(SW=2, SIMD_TILE=4, TILE_K=2):
    """Return (grid_fn, threadgroup_fn, constants) for a given tiling."""

    def grid(n, m, tw, tmax):
        assert n % 8 == 0 and m % 8 == 0, "dims must be multiples of 8"
        assert SW * SW * tw <= tmax, "too many simd tiles per threadgroup"
        assert (
            m % (8 * SIMD_TILE * SW) == 0 and n % (8 * SIMD_TILE * SW) == 0
        ), f"dims must be multiples of {8 * SIMD_TILE * SW}"
        return (tw, m // (8 * SIMD_TILE), n // (8 * SIMD_TILE))

    def tg(n, m, tw, tmax):
        return (tw, SW, SW)

    return (grid, tg, {"SW": SW, "SIMD_TILE": SIMD_TILE, "TILE_K": TILE_K})


# Map of kernel file -> (grid_fn, tg_fn, constants).
KERNELS = {
    "async_copy.metal":   tile_config(SW=2, SIMD_TILE=4, TILE_K=2),
    "sync_copy.metal":    tile_config(SW=2, SIMD_TILE=4, TILE_K=2),
    "sync_copy_db.metal": tile_config(SW=2, SIMD_TILE=4, TILE_K=2),
}


def measure(trials, dim, kernels, offline=False):
    f = lambda: (np.ones(trials) * dim).astype(np.uint)
    ns, ms, ks = f(), f(), f()
    f = lambda rs, cs: [np.random.rand(r, c).astype(np.float32) for r, c in zip(rs, cs)]
    As, Bs, Cs = f(ns, ks), f(ks, ms), f(ns, ms)
    f = lambda: np.random.rand(trials).astype(np.float32)
    alphas, betas = f(), f()
    for kernelfile, cfg in kernels.items():
        grid, thread, constants = cfg
        try:
            if offline:
                kernel = compiled_from_metallib(
                    kernelfile, constants, ("-ffast-math", "-O3")
                )("matmul")
            else:
                kernel = compiled(open(kernelfile).read(), constants)("matmul")
        except MetalCompileError as exc:
            yield (kernelfile, None, str(exc).splitlines()[0])
            continue
        results = []
        for A, B, C, alpha, beta in zip(As, Bs, Cs, alphas, betas):
            results.append(
                (C.shape, time_matmul(kernel, grid, thread, A, B, C, alpha, beta))
            )
        yield (kernelfile, results, None)
    mps = []
    for A, B, C, alpha, beta in zip(As, Bs, Cs, alphas, betas):
        mps.append(run_mps_matmul(A, B, C, alpha, beta))
    yield ("Metal Performance Shaders", mps, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument(
        "--kernels",
        nargs="*",
        default=list(KERNELS.keys()),
        help="subset of kernel files to run (default: all)",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="run sync_copy.metal across a sweep of (SW, SIMD_TILE, TILE_K) configs",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="precompile kernels via xcrun metal -O3 -ffast-math",
    )
    args = p.parse_args()
    flops = 2 * args.dim ** 3  # 1 mul + 1 add per output element * dim^3
    if args.sweep:
        sweep_kernels = ["sync_copy.metal", "sync_copy_db.metal"]
        configs = [
            (sw, st, tk)
            for sw in (2, 3, 4)
            for st in (2, 4, 8)
            for tk in (1, 2, 4, 8)
        ]
        rows = []
        for kf in sweep_kernels:
            short = kf.replace(".metal", "").replace("sync_copy", "sc")
            for sw, st, tk in configs:
                label = f"{short} SW={sw} ST={st} TK={tk}"
                kernels = {kf: tile_config(SW=sw, SIMD_TILE=st, TILE_K=tk)}
                try:
                    for kernel, results, err in measure(args.trials, args.dim, kernels, args.offline):
                        if kernel == "Metal Performance Shaders":
                            continue
                        if err is not None:
                            rows.append((label, None, err.strip()[:80]))
                        else:
                            avg = sum(t for _, t in results) / len(results)
                            rows.append((label, flops / avg / 1e12, f"{avg*1e3:.3f} ms"))
                except (AssertionError, RuntimeError) as exc:
                    rows.append((label, None, type(exc).__name__ + ": " + str(exc)[:80]))
        rows.sort(key=lambda r: -(r[1] or 0))
        print(f"\nSweep results @ {args.dim}x{args.dim}:")
        for label, tflops, note in rows:
            if tflops is None:
                print(f"  {label:34}  SKIP  ({note})")
            else:
                print(f"  {label:34}  {tflops:6.2f} TFLOPS  ({note})")
        return
    selected = {k: KERNELS[k] for k in args.kernels}
    for kernel, results, err in measure(args.trials, args.dim, selected):
        if err is not None:
            print(f"{kernel}: SKIPPED ({err})")
            continue
        avg = sum(t for _, t in results) / len(results)
        tflops = flops / avg / 1e12
        print(
            f"{kernel}:\n"
            f"\t{args.dim}x{args.dim} fp32 matmul\n"
            f"\taverage GPU time: {avg*1e3:.3f} ms\n"
            f"\tTFLOPS: {tflops:.2f}"
        )


if __name__ == "__main__":
    main()

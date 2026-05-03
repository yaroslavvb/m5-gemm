# m5-gemm — fast fp32 GEMM on Apple M5 Max

A Metal-4 port of the tutorial **["Fast Matrix Multiply on Apple GPU"](https://percisely.xyz/gemm)** by Zeke Medley. The original repository ([0xekez/metal-matmul](https://github.com/0xekez/metal-matmul)) targets the M2 (Metal 3) and relies on an undocumented `simdgroup_async_copy_2d` intrinsic that no longer compiles on macOS 26 / Metal 4. This repo:

* keeps the article's tile-tensor-core structure,
* replaces the blocked async-copy with a synchronous cooperative load (with an optional double-buffered variant), and
* benchmarks the result against Metal Performance Shaders (MPS) on Apple **M5 Max** (40-core GPU, 128 GB).

## Headline (4096 × 4096, fp32)

| Kernel | TFLOPS | ms / matmul |
|---|---:|---:|
| `sync_copy.metal` (this work) | **13.5** | 10.1 |
| `sync_copy_db.metal` (double-buffered) | 13.1 | 10.4 |
| Metal Performance Shaders | 11.7 | 11.7 |

Across sizes (best of 3 runs × 5 trials, ms = avg GPU time):

| Size | sync_copy | sync_copy_db | MPS |
|---:|---:|---:|---:|
| 1024² | 3.3 TF | **10.7 TF** | 2.9 TF |
| 2048² | 7.7 TF | **9.0 TF** | 5.2 TF |
| 4096² | **13.5 TF** | 13.1 TF | 11.7 TF |
| 8192² | 13.0 TF | 12.7 TF | **13.5 TF** |

`sync_copy_db` wins at small sizes where launch and DRAM-fetch latency dominate (the prologue + double-buffered prefetch hides them). `sync_copy` wins at 4096 where the compiler can fully unroll a smaller loop. At 8192 we're memory-bandwidth-bound and MPS catches up.

## Why we can't run the article verbatim on Metal 4

The article smuggles two private compiler intrinsics in via clang inline-asm linkage:

```c
thread _simdgroup_event_t* __metal_simdgroup_async_copy_2d(...)
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");
void __metal_wait_simdgroup_events(int, thread _simdgroup_event_t**)
  __asm("air.wait_simdgroup_events");
```

The Metal-4 frontend (macOS 26+, `metalfe-32023.883`, both runtime `MTLDevice.newLibraryWithSource` and offline `xcrun metal`) now rejects every `__asm("air.*")` string with `error: illegal string literal in 'asm'`. The diagnostic still helpfully points out that the new intrinsic is `__metal_wait_wg_events` (taking `__metal_threadgroup_event_t*`), but the matching `air.wg_async_copy_2d` companion is not exposed via any C-callable wrapper. Direct LLVM-IR linkage doesn't work either — the offline compiler crashes (SIGSEGV) on `-x ir` input that calls `air.*` symbols, regardless of target triple.

In short: **on Metal 4, the article's inline-asm trick is permanently closed.** Running `python async_copy.py` reports the failure with a clear message; running `python matmul.py` skips the original kernel and benchmarks the synchronous replacement.

## How the synchronous kernel works

Same tiling as the article:

* threadgroup of `SW × SW` simdgroups, each owning a `(8·SIMD_TILE)²` output sub-tile (default `SW=2, SIMD_TILE=4` ⇒ 64×64 per threadgroup, 32×32 per simdgroup);
* K-loop over reduction tiles of `BK = TILE_K · 8` columns of A and rows of B;
* threadgroup memory holds `A_tg [BM × BK]` and `B_tg [BK × BN]`;
* per simdgroup, an unrolled grid of `simdgroup_float8x8` accumulators consumed by `simdgroup_load` + `simdgroup_multiply_accumulate` (the 8×8 tensor-core ops).

The replacement load is a coalesced cooperative copy: each thread of the threadgroup walks `(BM·BK)/NTHREADS` `float4`s from device memory into threadgroup memory, with consecutive threads writing consecutive `float4`s so accesses are coalesced both in DRAM and in threadgroup banks. With `NTHREADS = SW²·32 = 128` and `BM·BK = 1024`, that's 2 `float4` loads per thread.

`sync_copy_db.metal` adds a second pair of buffers and prefetches tile `l+1` while computing on tile `l`. Without `simdgroup_async_copy` this isn't true compute/load overlap — the same threads issue both — but it gives the scheduler enough independent instructions to hide threadgroup-memory latency through ILP.

The `__attribute__((max_total_threads_per_threadgroup(SW*SW*32)))` hint on the kernel turned out to be the single biggest practical win: it lets the register allocator commit to the static threadgroup size and stops spilling.

## Things that did **not** help

* `xcrun metal -O3 -ffast-math` produced bit-identical performance to runtime compilation — Apple's runtime path uses the same backend.
* True double buffering with synchronous loads (vs. single buffer with the same tile size) — small win at small dims, small loss at large dims, washes out.
* `SIMD_TILE = 8`: doubling the per-simdgroup output tile spilled accumulators (64 `simdgroup_float8x8` per simd) and ran 10× slower.
* `SW = 3`: usable only for sizes divisible by 96 or 192; never beat `SW = 2` at the same total threads.
* Apple's new `<metal_cooperative_tensor>` (Metal 4) looked promising but ships only as a generic Layout interface — no built-in GEMM Layout, so it would be a from-scratch rewrite, not an optimization.

## Reproducing

```bash
# 1. Python venv with PyObjC + numpy
python3.13 -m venv .venv     # any 3.11+ works
.venv/bin/pip install -r requirements.txt

# 2. (Optional) install the Metal Toolchain so --offline works
xcodebuild -downloadComponent MetalToolchain   # ~700 MB

# 3. Bench
.venv/bin/python matmul.py --dim 4096 --trials 5
.venv/bin/python matmul.py --dim 4096 --trials 5 --offline   # uses xcrun metal -O3 -ffast-math

# Sweep configurations
.venv/bin/python matmul.py --dim 4096 --trials 2 --sweep --offline

# Original article microbenchmark — reports the Metal-4 incompatibility cleanly
.venv/bin/python async_copy.py
```

Without the Metal Toolchain, drop `--offline` — runtime compilation goes through the same backend and gets the same TFLOPS.

## File map

| File | Purpose |
|---|---|
| `async_copy.metal` | Original article kernel. Doesn't compile on Metal 4; kept for posterity. |
| `sync_copy.metal` | Replacement kernel — same tiling, synchronous coalesced loads. |
| `sync_copy_db.metal` | Same kernel with double-buffered prefetch. |
| `metal.py` | PyObjC harness: runtime + offline (`xcrun metal`) compile, dispatch, GPU timing. Generation-agnostic MTLBuffer detection (works on M2's `AGXG14GFamilyBuffer` through M5 Max's `AGXG17XFamilyBuffer`). |
| `mps_matmul.py` | Reference benchmark via `MPSMatrixMultiplication`. |
| `matmul.py` | Bench harness — single dim, sweep, or offline mode. |
| `async_copy.py` | Original `simdgroup_async_copy` microbench — exits with a clear error on Metal 4. |

## Compatibility notes from porting

* M5 Max buffers come back as `AGXG17XFamilyBuffer` (M2 was `AGXG14GFamilyBuffer`). The host harness now checks `conformsToProtocol_(MTLBuffer)` instead of using a brittle isinstance against the chip-specific class.
* All thread-position attribute parameters of a Metal kernel must have the same scalar/vector arity. Mixing `ushort3 [[thread_position_in_threadgroup]]` with `ushort [[threads_per_threadgroup]]` is now an error; promoting both to `ushort3` and computing the linear index manually fixes it.
* `numpy 2.x` and the latest `pyobjc` 12.x both work with Python 3.13 / 3.14.

## Credits

Core algorithm and original code: **Zeke Medley** ([@0xekez](https://github.com/0xekez)) — see [the article](https://percisely.xyz/gemm) and [metal-matmul](https://github.com/0xekez/metal-matmul). This repo is a port; the BSD-3-Clause license is preserved in `LICENSE`.

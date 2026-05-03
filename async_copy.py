"""Microbenchmark for ``simdgroup_async_copy``.

Measures the cost of cooperatively loading an 8 x 1024 float tile into
threadgroup memory using k SIMD groups (k = 1, 4, 7, ..., 31). Counter-
intuitively, the single-loader configuration tends to win because the
arithmetic to compute who-loads-what dominates the actual transfer.
"""
import sys

import numpy as np

from metal import compiled


def kernel(rows, cols, k):
    read = (cols + k - 1) // k
    i = 0
    tile_sizes = []
    while i < cols:
        tile_sizes += [min(read, cols - i)]
        i += read
    assert len(tile_sizes) == k
    reads = []
    offset = 0
    for i, tile_size in enumerate(tile_sizes):
        reads += [
            f"""
if (s_pos=={i}) {{
  thread _simdgroup_event_t* event = __metal_simdgroup_async_copy_2d(
    sizeof(float),
    alignof(float),
    reinterpret_cast<threadgroup void*>(A_tg+{offset}),
    ulong({cols}),
    1,
    ulong2({tile_size}, {rows}),
    reinterpret_cast<const device void*>(A+{offset}),
    ulong({cols}),
    1,
    ulong2({tile_size}, {rows}),
    long2(0),
    0
  );
  __metal_wait_simdgroup_events(1,&event);
}}
"""
        ]
        offset += tile_size
    reads = "".join(reads)
    return f"""
#include <metal_compute>
using namespace metal;

struct _simdgroup_event_t;

thread _simdgroup_event_t* __metal_simdgroup_async_copy_2d(
  ulong,               // sizeof(element)
  ulong,               // alignof(element)
  threadgroup void *,  // dst
  ulong,               // elements_per_row
  ulong,               // stride?
  ulong2,              // tile_size
  const device void *, // source
  ulong,
  ulong,
  ulong2,
  long2,               // source origin?
  int)                 // clamp mode
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

void __metal_wait_simdgroup_events(
  int, // len(events)
  thread _simdgroup_event_t**
)
  __asm("air.wait_simdgroup_events");

kernel void async_copy(
  const device float* A,
  ushort s_pos [[simdgroup_index_in_threadgroup]]
) {{
  threadgroup float A_tg[{rows}*{cols}];
  {reads}
  threadgroup_barrier(mem_flags::mem_threadgroup);
}}
"""


def measure(rows, cols, k, samples=10000):
    source = kernel(rows, cols, k)
    from metal import MetalCompileError
    try:
        kernel_fn = compiled(source, {})("async_copy")
    except MetalCompileError as exc:
        msg = str(exc)
        if "illegal string literal in 'asm'" in msg:
            sys.stderr.write(
                "\nUpstream microbenchmark requires the undocumented intrinsic\n"
                "  air.simdgroup_async_copy_2d / air.wait_simdgroup_events\n"
                "linked via inline asm. Apple's Metal-4 toolchain (macOS 26+)\n"
                "rejects __asm(\"air.*\") strings, so this benchmark is no\n"
                "longer runnable. The matmul.py benchmark uses a synchronous\n"
                "load instead.\n"
            )
            sys.exit(2)
        raise
    A = np.random.randn(rows, cols)
    times = []
    for _ in range(samples):
        _, time = kernel_fn(
            lambda t, m: (32 * k, 1, 1), lambda t, m: (32 * k, 1, 1), False, A.ravel()
        )
        times += [time]
    return sum(times) / len(times)


if __name__ == "__main__":
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    print("# k\tavg_time_seconds")
    for k in range(1, 33, 3):
        t = measure(8, 1024, k, samples=samples)
        print(f"{k}\t{t}")

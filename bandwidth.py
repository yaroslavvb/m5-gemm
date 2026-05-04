"""Empirical unified-memory bandwidth on the GPU.

A STREAM-style probe: dispatch a kernel that copies (or just reads) N
bytes between device-storage buffers, large enough to defeat caches.
Useful as a sanity check against Apple's bandwidth specs and as an
upper bound on what any GPU kernel can pull through the unified memory
fabric.

Usage:
    .venv/bin/python bandwidth.py
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metal import compiled  # noqa: E402


def main():
    src = open(os.path.join(HERE, "bandwidth.metal")).read()
    lib = compiled(src, {})
    copy = lib("stream_copy")
    read = lib("stream_read")

    # Sweep sizes that span L2-resident through DRAM-bound territory.
    sizes_gib = [0.25, 0.5, 1.0, 2.0, 4.0]
    for gib in sizes_gib:
        nbytes = int(gib * (1 << 30))
        n4 = nbytes // 16  # float4 elements
        a = np.random.rand(n4 * 4).astype(np.float32)
        b = np.zeros_like(a)
        sink = np.zeros(4, dtype=np.float32)

        # COPY: counts both the read and the write toward bandwidth.
        times = []
        for _ in range(5):
            _, t = copy(
                lambda tw, tm: (n4, 1, 1),
                lambda tw, tm: (1024, 1, 1),
                False,
                a, b, ctypes.c_uint32(n4),
            )
            times.append(t)
        avg = sum(times[1:]) / (len(times) - 1)  # drop warmup
        bw = 2 * nbytes / avg / 1e9
        print(
            f"copy {gib:5.2f} GiB:  {avg*1e3:7.2f} ms  -> {bw:7.1f} GB/s  (read+write)"
        )

        # READ: just the load; the conditional store is dead but the
        # compiler can't prove it without seeing the input data.
        times = []
        for _ in range(5):
            _, t = read(
                lambda tw, tm: (n4, 1, 1),
                lambda tw, tm: (1024, 1, 1),
                False,
                a, sink, ctypes.c_uint32(n4),
            )
            times.append(t)
        avg = sum(times[1:]) / (len(times) - 1)
        bw = nbytes / avg / 1e9
        print(
            f"read {gib:5.2f} GiB:  {avg*1e3:7.2f} ms  -> {bw:7.1f} GB/s  (read only)"
        )


if __name__ == "__main__":
    main()

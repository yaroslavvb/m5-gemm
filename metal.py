"""Thin PyObjC wrapper around a Metal compute pipeline.

Compiles a Metal source string with optional preprocessor macros (or
loads a precompiled .metallib via ``compiled_from_metallib``), and
returns a callable that dispatches the named kernel and reports the GPU
time of a single dispatch.

Original: https://github.com/0xekez/metal-matmul (Apple M2)
This version replaces the M2-specific ``AGXG14GFamilyBuffer`` isinstance
check with an MTLBuffer protocol-conformance check so it works on any
Apple Silicon generation (tested on M5 Max, where the runtime class is
``AGXG17XFamilyBuffer``).
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile

import Metal
import numpy as np
import objc
from Foundation import NSURL

_MTL_BUFFER_PROTO = objc.protocolNamed("MTLBuffer")


def _is_mtl_buffer(obj) -> bool:
    return hasattr(obj, "conformsToProtocol_") and obj.conformsToProtocol_(
        _MTL_BUFFER_PROTO
    )


class MetalCompileError(RuntimeError):
    pass


def _gpu():
    return Metal.MTLCreateSystemDefaultDevice()


def _build_metallib(source_path: str, constants: dict, extra_flags=()) -> str:
    """Invoke ``xcrun metal`` to produce a .metallib next to source_path.

    Caches the output by a hash of (source bytes, constants, flags) so
    repeated calls during a sweep don't re-shell-out. Requires the Metal
    Toolchain (``xcodebuild -downloadComponent MetalToolchain``).
    """
    if shutil.which("xcrun") is None:
        raise MetalCompileError("xcrun not found; install Xcode + Metal Toolchain")
    src = open(source_path, "rb").read()
    key = hashlib.sha256()
    key.update(src)
    key.update(repr(sorted(constants.items())).encode())
    key.update(repr(extra_flags).encode())
    digest = key.hexdigest()[:16]
    cache_dir = os.path.expanduser("~/.cache/m5-gemm")
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"{os.path.basename(source_path)}.{digest}.metallib")
    if os.path.exists(out):
        return out
    with tempfile.TemporaryDirectory() as td:
        air = os.path.join(td, "k.air")
        defs = [f"-D{k}={v}" for k, v in constants.items()]
        cmd = ["xcrun", "metal", *defs, *extra_flags, "-c", source_path, "-o", air]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise MetalCompileError(f"xcrun metal failed:\n{r.stderr}")
        cmd = ["xcrun", "metallib", air, "-o", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise MetalCompileError(f"xcrun metallib failed:\n{r.stderr}")
    return out


def _make_caller(gpu, lib):
    """Wrap a compiled ``MTLLibrary`` as a callable for a named kernel."""
    def function(name):
        pipeline = lib.newFunctionWithName_(name)
        if pipeline is None:
            raise MetalCompileError(f"function '{name}' not found in library")
        pipeline, err = gpu.newComputePipelineStateWithFunction_error_(pipeline, None)
        if err is not None:
            raise MetalCompileError(str(err))
        queue = gpu.newCommandQueue()
        calls = 0

        def call(gridShape, threadShape, debug, *args):
            nonlocal calls
            args = [
                gpu.newBufferWithBytes_length_options_(
                    arg.tobytes(), arg.nbytes, Metal.MTLResourceStorageModeShared
                )
                if isinstance(arg, np.ndarray)
                else arg
                for arg in args
            ]
            calls += 1
            if debug:
                captureManager = Metal.MTLCaptureManager.sharedCaptureManager()
                captureDesc = Metal.MTLCaptureDescriptor()
                captureDesc.setCaptureObject_(gpu)
                captureDesc.setDestination_(
                    Metal.MTLCaptureDestinationGPUTraceDocument
                )
                captureDesc.setOutputURL_(
                    NSURL.fileURLWithPath_(f"captures/{name}_{calls}.gputrace")
                )
                ok, err = captureManager.startCaptureWithDescriptor_error_(
                    captureDesc, None
                )
                assert ok, str(err)
            buffer = queue.commandBuffer()
            encoder = buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            smem = []
            for i, arg in enumerate(args):
                if _is_mtl_buffer(arg):
                    smem += [arg]
                    encoder.setBuffer_offset_atIndex_(arg, 0, i)
                else:
                    encoder.setBytes_length_atIndex_(arg, ctypes.sizeof(arg), i)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(
                    *gridShape(
                        pipeline.threadExecutionWidth(),
                        pipeline.maxTotalThreadsPerThreadgroup(),
                    )
                ),
                Metal.MTLSizeMake(
                    *threadShape(
                        pipeline.threadExecutionWidth(),
                        pipeline.maxTotalThreadsPerThreadgroup(),
                    )
                ),
            )
            encoder.endEncoding()
            buffer.commit()
            buffer.waitUntilCompleted()
            if debug:
                captureManager.stopCapture()
            t = buffer.GPUEndTime() - buffer.GPUStartTime()
            return smem, t

        return call

    return function


def compiled_from_metallib(source_path, constants, extra_flags=()):
    """Compile ``source_path`` offline with ``xcrun metal``, then load the
    resulting metallib via Metal's runtime. Returns the same factory that
    :func:`compiled` returns.
    """
    metallib = _build_metallib(source_path, constants, extra_flags)
    gpu = _gpu()
    url = NSURL.fileURLWithPath_(metallib)
    lib, err = gpu.newLibraryWithURL_error_(url, None)
    if err is not None:
        raise MetalCompileError(str(err))
    return _make_caller(gpu, lib)


# in order to debug, the METAL_CAPTURE_ENABLED=1 needs to be set in
# the environment.
def compiled(kernel, constants):
    gpu = _gpu()
    opts = Metal.MTLCompileOptions()
    opts.setPreprocessorMacros_(constants)
    lib, err = gpu.newLibraryWithSource_options_error_(kernel, opts, None)
    if err is not None:
        raise MetalCompileError(str(err))
    return _make_caller(gpu, lib)

"""Reference benchmark: same GEMM via Metal Performance Shaders."""
import Metal
import MetalPerformanceShaders as MPS
import numpy as np


def run_mps_matmul(A, B, C, alpha, beta, samples=3):
    n, k = A.shape
    assert B.shape[0] == k, f"inner dim missmatch: {B.shape[0]} vs. {k}"
    k, m = B.shape
    assert C.shape == (n, m), f"C dim missmatch: {C.shape} vs. ({n},{m})"
    A, B, C = A.astype(np.float32), B.astype(np.float32), C.astype(np.float32)
    goal = A @ B * alpha + C * beta
    gpu = Metal.MTLCreateSystemDefaultDevice()
    assert MPS.MPSSupportsMTLDevice(gpu), "metal performance shaders unsupported"
    queue = gpu.newCommandQueue()
    times = []
    for i in range(samples):
        buffer = queue.commandBuffer()
        A_buf = gpu.newBufferWithBytes_length_options_(
            A.tobytes(), A.nbytes, Metal.MTLResourceStorageModeShared
        )
        B_buf = gpu.newBufferWithBytes_length_options_(
            B.tobytes(), B.nbytes, Metal.MTLResourceStorageModeShared
        )
        C_buf = gpu.newBufferWithBytes_length_options_(
            C.tobytes(), C.nbytes, Metal.MTLResourceStorageModeShared
        )
        descA = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            n, k, k * 4, MPS.MPSDataTypeFloat32
        )
        descB = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            k, m, m * 4, MPS.MPSDataTypeFloat32
        )
        descC = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            n, m, m * 4, MPS.MPSDataTypeFloat32
        )
        matA = MPS.MPSMatrix.alloc().initWithBuffer_offset_descriptor_(A_buf, 0, descA)
        matB = MPS.MPSMatrix.alloc().initWithBuffer_offset_descriptor_(B_buf, 0, descB)
        matC = MPS.MPSMatrix.alloc().initWithBuffer_offset_descriptor_(C_buf, 0, descC)
        gemm = MPS.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
            gpu,
            False,  # transpose A?
            False,  # transpose B?
            n,
            m,
            k,
            alpha,
            beta,
        )
        gemm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
            buffer, matA, matB, matC
        )
        buffer.commit()
        buffer.waitUntilCompleted()
        time = buffer.GPUEndTime() - buffer.GPUStartTime()
        res = np.frombuffer(C_buf.contents().as_buffer(C_buf.length()), dtype=np.float32)
        res = res.reshape((n, m))
        assert np.allclose(res, goal, rtol=1e-4), f"mps incorrect iter {i}"
        times += [time]
    return (A.shape, sum(times) / len(times))

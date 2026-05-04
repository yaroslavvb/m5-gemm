#include <metal_compute>
using namespace metal;

kernel void stream_copy(
    const device float4* src [[buffer(0)]],
    device float4*       dst [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]])
{
    if (tid >= n) return;
    dst[tid] = src[tid];
}

kernel void stream_read(
    const device float4* src [[buffer(0)]],
    device float*        sink[[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float4 v = src[tid];
    if ((v.x + v.y + v.z + v.w) == -1.0e30f) sink[0] = 1.0f;  // never taken; defeats DCE
}

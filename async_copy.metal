// Tiled GEMM kernel from https://percisely.xyz/gemm
// Uses simdgroup_async_copy + 8x8 tensor cores (simdgroup_float8x8).

#include <metal_simdgroup_matrix>
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

template<
  ushort tile_cols,
  ushort tile_rows
>
inline thread _simdgroup_event_t* simdgroup_async_copy(
  const device float* src,
  const ushort2 src_pos,
  const ushort2 src_shape,
  threadgroup float* tile) {
  src = src + src_pos.x+src_pos.y*src_shape.x;
  return __metal_simdgroup_async_copy_2d(
    sizeof(float),
    alignof(float),
    reinterpret_cast<threadgroup void*>(tile),
    ulong(tile_cols),
    1,
    ulong2(tile_cols, tile_rows),
    reinterpret_cast<const device void*>(src),
    ulong(src_shape.x),
    1,
    ulong2(tile_cols, tile_rows),
    long2(0),
    0);
}

template<ushort DIM>
inline void simdgroup_multiply(
  threadgroup float* A,
  threadgroup float* B,
  ushort2 c_pos,
  thread simdgroup_float8x8 &acc
) {
  simdgroup_float8x8 A_simd;
  simdgroup_float8x8 B_simd;
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < DIM*8; i+=8) {
    simdgroup_load(A_simd, A, DIM*8, ulong2(i, c_pos.y));
    simdgroup_load(B_simd, B, SW*SIMD_TILE*8, ulong2(c_pos.x, i));
    simdgroup_multiply_accumulate(acc, A_simd, B_simd, acc);
  }
}

kernel void matmul(
  constant ushort& n,
  constant ushort& k,
  constant ushort& m,
  constant float& alpha,
  constant float& beta,
  const device float* A,
  const device float* B,
  device float* C,
  ushort3 t_pos    [[thread_position_in_grid]],
  ushort3 t_tg_pos [[thread_position_in_threadgroup]],
  ushort  s_pos    [[simdgroup_index_in_threadgroup]]
) {
  threadgroup float A_tg[SW*SIMD_TILE*8*TILE_K*8];
  threadgroup float B_tg[SW*SIMD_TILE*8*TILE_K*8];

  ushort2 c_origin = t_pos.yz*8*SIMD_TILE;
  ushort2 a_origin = ushort2(0,c_origin.y);
  ushort2 b_origin = ushort2(c_origin.x,0);

  simdgroup_float8x8 acc[SIMD_TILE][SIMD_TILE];
  for (ushort i=0; i<SIMD_TILE; i++)
      for (ushort j=0; j<SIMD_TILE; j++)
	acc[i][j] = simdgroup_float8x8(0);

  ushort k_tiles = k/(8*TILE_K);
  for (ushort l = 0; l < k_tiles; l++) {
    ushort2 a_pos = a_origin+ushort2(l*8*TILE_K,0);
    ushort2 b_pos = b_origin+ushort2(0,l*8*TILE_K);
    if (s_pos==0) {
      thread _simdgroup_event_t* events[2];
      events[0] = simdgroup_async_copy<TILE_K*8,SW*SIMD_TILE*8>(
        A,
        a_pos,
        ushort2(k,n),
        A_tg);
      events[1] = simdgroup_async_copy<SW*SIMD_TILE*8,TILE_K*8>(
        B,
        b_pos,
        ushort2(m,k),
        B_tg);
      __metal_wait_simdgroup_events(2,events);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort i=0; i<SIMD_TILE; i++)
      for (ushort j=0; j<SIMD_TILE; j++)
	simdgroup_multiply<TILE_K>(
          A_tg,B_tg,
	  t_tg_pos.yz*8*SIMD_TILE+ushort2(i*8,j*8),
	  acc[i][j]);
    threadgroup_barrier(mem_flags::mem_none);
  }

  if (c_origin.x<m&&c_origin.y<n) {
    simdgroup_float8x8 c_simd;
    for (ushort i=0; i<SIMD_TILE; i++)
      for (ushort j=0; j<SIMD_TILE; j++) {
	ulong2 pos = ulong2(c_origin+ushort2(i*8,j*8));
	simdgroup_load(c_simd,C,m,pos);
	simdgroup_multiply(c_simd,c_simd, simdgroup_float8x8(beta));
	simdgroup_multiply_accumulate(c_simd,acc[i][j],simdgroup_float8x8(alpha),c_simd);
	simdgroup_store(c_simd,C,m,pos);
      }
  }
}

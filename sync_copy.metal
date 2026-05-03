// Tiled fp32 GEMM via 8x8 simdgroup matrices ("tensor cores").
//
// Same tiling structure as https://percisely.xyz/gemm, but with the
// undocumented `air.simdgroup_async_copy_2d` intrinsic replaced by a
// straightforward cooperative load. Apple's Metal-4 compiler (macOS 26+)
// rejects __asm("air.*") linkage, so the article's technique no longer
// compiles. This kernel only uses public Metal APIs.
//
// Compile-time constants set by the host via -D:
//   SW         tile-of-tiles edge length (threadgroup is SW x SW simdgroups)
//   SIMD_TILE  simdgroups own a SIMD_TILE x SIMD_TILE grid of 8x8 matrices
//   TILE_K     reduction tile = TILE_K * 8 elements

#include <metal_simdgroup_matrix>
#include <metal_compute>
using namespace metal;

constant constexpr ushort BM = SW * SIMD_TILE * 8;  // threadgroup output rows
constant constexpr ushort BN = SW * SIMD_TILE * 8;  // threadgroup output cols
constant constexpr ushort BK = TILE_K * 8;          // reduction tile

template <ushort DIM>
inline void simdgroup_multiply_tile(
    threadgroup float *A,
    threadgroup float *B,
    ushort2 c_pos,
    thread simdgroup_float8x8 &acc)
{
  simdgroup_float8x8 A_simd, B_simd;
#pragma clang loop unroll(full)
  for (ushort i = 0; i < DIM * 8; i += 8) {
    simdgroup_load(A_simd, A, DIM * 8, ulong2(i, c_pos.y));
    simdgroup_load(B_simd, B, BN,      ulong2(c_pos.x, i));
    simdgroup_multiply_accumulate(acc, A_simd, B_simd, acc);
  }
}

// Cooperative threadgroup load of a (rows x cols) slab from `src`
// (row-major, stride `src_stride`) into `dst` (row-major, stride `cols`).
// `tid` is the linear thread index within the threadgroup; `nthreads`
// is the threadgroup size (compile-time).
//
// Layout: each thread loads `total/nthreads` floats; consecutive thread
// indices step horizontally so the access pattern is coalesced. Both
// `cols` and `rows*cols/nthreads` must divide cleanly.
template <ushort rows, ushort cols, ushort nthreads>
inline void load_tile(
    const device float *src,
    uint src_stride,
    threadgroup float *dst,
    ushort tid)
{
  constexpr ushort per_thread = (rows * cols + nthreads - 1) / nthreads;

  // Use vec4 loads when alignment permits.
  if ((cols % 4) == 0 && (per_thread % 4) == 0) {
    constexpr ushort cols4 = cols / 4;
    constexpr ushort total4 = rows * cols4;
    auto src4 = reinterpret_cast<const device float4 *>(src);
    auto dst4 = reinterpret_cast<threadgroup float4 *>(dst);
    uint stride4 = src_stride / 4;
#pragma clang loop unroll(full)
    for (ushort i = 0; i < total4; i += nthreads) {
      ushort idx = i + tid;
      if (idx >= total4) break;
      ushort r = idx / cols4;
      ushort c = idx - r * cols4;
      dst4[idx] = src4[uint(r) * stride4 + c];
    }
    return;
  }

  constexpr ushort total = rows * cols;
#pragma clang loop unroll(full)
  for (ushort i = 0; i < total; i += nthreads) {
    ushort idx = i + tid;
    if (idx >= total) break;
    ushort r = idx / cols;
    ushort c = idx - r * cols;
    dst[idx] = src[uint(r) * src_stride + c];
  }
}

kernel void __attribute__((max_total_threads_per_threadgroup(SW * SW * 32)))
matmul(
    constant uint  &n,
    constant uint  &k,
    constant uint  &m,
    constant float &alpha,
    constant float &beta,
    const device float *A,
    const device float *B,
    device float       *C,
    ushort3 t_tg_pos       [[thread_position_in_threadgroup]],
    ushort3 tg_pos         [[threadgroup_position_in_grid]],
    ushort3 tg_size        [[threads_per_threadgroup]])
{
  threadgroup float A_tg[BM * BK];
  threadgroup float B_tg[BK * BN];

  uint tg_row = uint(tg_pos.z) * BM;  // top row of this threadgroup's C tile
  uint tg_col = uint(tg_pos.y) * BN;  // left col of this threadgroup's C tile

  simdgroup_float8x8 acc[SIMD_TILE][SIMD_TILE];
  for (ushort i = 0; i < SIMD_TILE; i++)
    for (ushort j = 0; j < SIMD_TILE; j++)
      acc[i][j] = simdgroup_float8x8(0);

  constexpr ushort NTHREADS = SW * SW * 32;  // 32 lanes per simdgroup
  ushort tid_in_tg =
      (t_tg_pos.z * tg_size.y + t_tg_pos.y) * tg_size.x + t_tg_pos.x;

  uint k_tiles = k / BK;
  for (uint l = 0; l < k_tiles; l++) {
    uint k_off = l * BK;
    load_tile<BM, BK, NTHREADS>(A + tg_row * k + k_off, k, A_tg, tid_in_tg);
    load_tile<BK, BN, NTHREADS>(B + k_off * m + tg_col, m, B_tg, tid_in_tg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort2 simd_origin = ushort2(t_tg_pos.y, t_tg_pos.z) * (8 * SIMD_TILE);
    for (ushort i = 0; i < SIMD_TILE; i++)
      for (ushort j = 0; j < SIMD_TILE; j++)
        simdgroup_multiply_tile<TILE_K>(
            A_tg, B_tg,
            simd_origin + ushort2(i * 8, j * 8),
            acc[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  ushort2 simd_origin = ushort2(t_tg_pos.y, t_tg_pos.z) * (8 * SIMD_TILE);
  uint c_row0 = tg_row + simd_origin.y;
  uint c_col0 = tg_col + simd_origin.x;
  if (c_col0 < m && c_row0 < n) {
    simdgroup_float8x8 c_simd;
    for (ushort i = 0; i < SIMD_TILE; i++)
      for (ushort j = 0; j < SIMD_TILE; j++) {
        ulong2 pos = ulong2(c_col0 + i * 8, c_row0 + j * 8);
        simdgroup_load(c_simd, C, m, pos);
        simdgroup_multiply(c_simd, c_simd, simdgroup_float8x8(beta));
        simdgroup_multiply_accumulate(c_simd, acc[i][j], simdgroup_float8x8(alpha), c_simd);
        simdgroup_store(c_simd, C, m, pos);
      }
  }
}

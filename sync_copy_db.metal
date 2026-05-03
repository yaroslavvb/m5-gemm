// Tiled fp32 GEMM with double-buffered (software-pipelined) loads.
//
// Same per-simdgroup compute as sync_copy.metal, but the K-loop loads
// the *next* threadgroup tile while computing on the *current* one.
// Without an async-DMA primitive this isn't true overlap, but it gives
// the scheduler enough independent instructions to hide threadgroup
// memory latency through ILP.
//
// Compile-time constants set by the host via -D:
//   SW         tile-of-tiles edge length (threadgroup is SW x SW simdgroups)
//   SIMD_TILE  simdgroups own a SIMD_TILE x SIMD_TILE grid of 8x8 matrices
//   TILE_K     reduction tile = TILE_K * 8 elements

#include <metal_simdgroup_matrix>
#include <metal_compute>
using namespace metal;

constant constexpr ushort BM = SW * SIMD_TILE * 8;
constant constexpr ushort BN = SW * SIMD_TILE * 8;
constant constexpr ushort BK = TILE_K * 8;
constant constexpr ushort NTHREADS = SW * SW * 32;

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

template <ushort rows, ushort cols, ushort nthreads>
inline void load_tile(
    const device float *src,
    uint src_stride,
    threadgroup float *dst,
    ushort tid)
{
  if ((cols % 4) == 0 && (rows * cols) % (nthreads * 4) == 0) {
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
  threadgroup float A_tg[2][BM * BK];
  threadgroup float B_tg[2][BK * BN];

  uint tg_row = uint(tg_pos.z) * BM;
  uint tg_col = uint(tg_pos.y) * BN;

  simdgroup_float8x8 acc[SIMD_TILE][SIMD_TILE];
  for (ushort i = 0; i < SIMD_TILE; i++)
    for (ushort j = 0; j < SIMD_TILE; j++)
      acc[i][j] = simdgroup_float8x8(0);

  ushort tid = (t_tg_pos.z * tg_size.y + t_tg_pos.y) * tg_size.x + t_tg_pos.x;

  uint k_tiles = k / BK;

  // Prologue: load tile 0 into buffer 0.
  load_tile<BM, BK, NTHREADS>(A + tg_row * k, k, A_tg[0], tid);
  load_tile<BK, BN, NTHREADS>(B + tg_col,     m, B_tg[0], tid);

  ushort cur = 0;
  ushort2 simd_origin = ushort2(t_tg_pos.y, t_tg_pos.z) * (8 * SIMD_TILE);

  for (uint l = 0; l < k_tiles; l++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ushort nxt = ushort(1) - cur;
    if (l + 1 < k_tiles) {
      uint k_off = (l + 1) * BK;
      load_tile<BM, BK, NTHREADS>(A + tg_row * k + k_off, k, A_tg[nxt], tid);
      load_tile<BK, BN, NTHREADS>(B + k_off * m + tg_col, m, B_tg[nxt], tid);
    }
    threadgroup float *A_use = A_tg[cur];
    threadgroup float *B_use = B_tg[cur];
    for (ushort i = 0; i < SIMD_TILE; i++)
      for (ushort j = 0; j < SIMD_TILE; j++)
        simdgroup_multiply_tile<TILE_K>(
            A_use, B_use,
            simd_origin + ushort2(i * 8, j * 8),
            acc[i][j]);
    cur = nxt;
  }

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

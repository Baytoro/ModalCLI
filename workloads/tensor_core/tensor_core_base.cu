#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpSize = 32;

__global__ void tensor_core_wmma_kernel(
    float* out,
    int total_warps,
    int kernel_iters,
    int repeat,
    int accumulators) {
  const int warp_id_in_block = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;

  if (global_warp_id >= total_warps) {
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  half* smem_a = reinterpret_cast<half*>(smem_raw);
  half* smem_b = smem_a + warps_per_block * kWmmaM * kWmmaK;
  float* smem_c = reinterpret_cast<float*>(smem_b + warps_per_block * kWmmaK * kWmmaN);

  half* tile_a = smem_a + warp_id_in_block * kWmmaM * kWmmaK;
  half* tile_b = smem_b + warp_id_in_block * kWmmaK * kWmmaN;
  float* tile_c = smem_c + warp_id_in_block * kWmmaM * kWmmaN;

  for (int idx = lane_id; idx < kWmmaM * kWmmaK; idx += kWarpSize) {
    tile_a[idx] = __float2half(1.0f + static_cast<float>(idx % 7) * 0.001f);
  }
  for (int idx = lane_id; idx < kWmmaK * kWmmaN; idx += kWarpSize) {
    tile_b[idx] = __float2half(1.0f + static_cast<float>(idx % 5) * 0.001f);
  }
  for (int idx = lane_id; idx < kWmmaM * kWmmaN; idx += kWarpSize) {
    tile_c[idx] = 0.0f;
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c0;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c1;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c2;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c3;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c4;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c5;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c6;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c7;

  wmma::fill_fragment(c0, 0.0f);
  wmma::fill_fragment(c1, 0.0f);
  wmma::fill_fragment(c2, 0.0f);
  wmma::fill_fragment(c3, 0.0f);
  wmma::fill_fragment(c4, 0.0f);
  wmma::fill_fragment(c5, 0.0f);
  wmma::fill_fragment(c6, 0.0f);
  wmma::fill_fragment(c7, 0.0f);

  wmma::load_matrix_sync(a_frag, tile_a, kWmmaK);
  wmma::load_matrix_sync(b_frag, tile_b, kWmmaK);

  #pragma unroll 1
  for (int i = 0; i < kernel_iters; ++i) {
    #pragma unroll 1
    for (int r = 0; r < repeat; ++r) {
      if (accumulators >= 1) wmma::mma_sync(c0, a_frag, b_frag, c0);
      if (accumulators >= 2) wmma::mma_sync(c1, a_frag, b_frag, c1);
      if (accumulators >= 3) wmma::mma_sync(c2, a_frag, b_frag, c2);
      if (accumulators >= 4) wmma::mma_sync(c3, a_frag, b_frag, c3);
      if (accumulators >= 5) wmma::mma_sync(c4, a_frag, b_frag, c4);
      if (accumulators >= 6) wmma::mma_sync(c5, a_frag, b_frag, c5);
      if (accumulators >= 7) wmma::mma_sync(c6, a_frag, b_frag, c6);
      if (accumulators >= 8) wmma::mma_sync(c7, a_frag, b_frag, c7);
    }
  }

  wmma::store_matrix_sync(tile_c, c0, kWmmaN, wmma::mem_row_major);
  __syncthreads();

  if (lane_id == 0) {
    float sum = 0.0f;
    for (int i = 0; i < kWmmaM * kWmmaN; ++i) {
      sum += tile_c[i];
    }
    out[global_warp_id] = sum;
  }
}

}  // namespace

torch::Tensor tensor_core_cuda(
    torch::Tensor out,
    int64_t num_blocks,
    int64_t block_size,
    int64_t kernel_iters,
    int64_t repeat,
    int64_t accumulators) {
  const int blocks = static_cast<int>(num_blocks);
  const int threads = static_cast<int>(block_size);
  const int warps_per_block = threads / kWarpSize;
  const int total_warps = static_cast<int>(out.numel());
  const int requested_accumulators = static_cast<int>(accumulators);
  const int effective_accumulators = requested_accumulators > 8 ? 8 : requested_accumulators;

  TORCH_CHECK(total_warps == blocks * warps_per_block, "out size must be num_blocks * (block_size / 32)");
  TORCH_CHECK(effective_accumulators >= 1, "accumulators must be >= 1");
  TORCH_CHECK(threads % kWarpSize == 0, "block_size must be divisible by 32");

  const size_t shmem_bytes = static_cast<size_t>(warps_per_block) *
      (kWmmaM * kWmmaK * sizeof(half) +
       kWmmaK * kWmmaN * sizeof(half) +
       kWmmaM * kWmmaN * sizeof(float));

  tensor_core_wmma_kernel<<<blocks, threads, shmem_bytes>>>(
      out.data_ptr<float>(),
      total_warps,
      static_cast<int>(kernel_iters),
      static_cast<int>(repeat),
      effective_accumulators);

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

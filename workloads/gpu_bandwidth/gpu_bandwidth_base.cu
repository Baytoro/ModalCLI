#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void gpu_bandwidth_copy_kernel(const float* src, float* dst, int64_t n, int copy_iters) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  #pragma unroll 1
  for (int it = 0; it < copy_iters; ++it) {
    for (int64_t i = tid * 4; i + 3 < n; i += stride * 4) {
      const float4 v = *reinterpret_cast<const float4*>(src + i);
      *reinterpret_cast<float4*>(dst + i) = v;
    }
    for (int64_t i = (n / 4) * 4 + tid; i < n; i += stride) {
      dst[i] = src[i];
    }
  }
}

torch::Tensor gpu_bandwidth_cuda(torch::Tensor src, int64_t copy_iters) {
  auto dst = torch::empty_like(src);
  const int64_t n = src.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > 32768) {
    blocks = 32768;
  }

  gpu_bandwidth_copy_kernel<<<blocks, threads>>>(
      src.data_ptr<float>(),
      dst.data_ptr<float>(),
      n,
      static_cast<int>(copy_iters));
  C10_CUDA_CHECK(cudaGetLastError());
  return dst;
}

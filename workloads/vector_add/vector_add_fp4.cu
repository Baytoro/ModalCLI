#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>

__global__ void vector_add_fp4_kernel(
    const float* a,
    const float* b,
    float* out,
    int64_t n) {
  const int64_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t base = vec_idx * 4;
  if (base >= n) {
    return;
  }

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(a + base) | reinterpret_cast<uintptr_t>(b + base) |
        reinterpret_cast<uintptr_t>(out + base)) &
       0xF) == 0;

  if (aligned && base + 3 < n) {
    const float4 av = *reinterpret_cast<const float4*>(a + base);
    const float4 bv = *reinterpret_cast<const float4*>(b + base);
    float4 ov;
    ov.x = av.x + bv.x;
    ov.y = av.y + bv.y;
    ov.z = av.z + bv.z;
    ov.w = av.w + bv.w;
    *reinterpret_cast<float4*>(out + base) = ov;
    return;
  }

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int64_t i = base + j;
    if (i < n) {
      out[i] = a[i] + b[i];
    }
  }
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
  auto out = torch::empty_like(a);
  const int64_t n = a.numel();
  constexpr int threads = 256;
  const int64_t vec_n = (n + 3) / 4;
  const int blocks = (static_cast<int>(vec_n) + threads - 1) / threads;

  vector_add_fp4_kernel<<<blocks, threads>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      out.data_ptr<float>(),
      n);

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

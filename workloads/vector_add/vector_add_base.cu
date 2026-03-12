#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

__global__ void vector_add_base_kernel(
    const float* a,
    const float* b,
    float* out,
    int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
  auto out = torch::empty_like(a);
  const int64_t n = a.numel();
  constexpr int threads = 256;
  const int blocks = (static_cast<int>(n) + threads - 1) / threads;

  vector_add_base_kernel<<<blocks, threads>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      out.data_ptr<float>(),
      n);

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

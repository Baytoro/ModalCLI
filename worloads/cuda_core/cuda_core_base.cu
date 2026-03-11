#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void fp32_fma_bench_kernel(float* out, int n, int kernel_iters, int unroll) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  // Use multiple independent accumulators to reduce dependency-chain stalls.
  float a0 = 1.000001f, b0 = 1.000002f, c0 = 0.0f;
  float a1 = 1.000003f, b1 = 1.000004f, c1 = 0.0f;
  float a2 = 1.000005f, b2 = 1.000006f, c2 = 0.0f;
  float a3 = 1.000007f, b3 = 1.000008f, c3 = 0.0f;
  float a4 = 1.000009f, b4 = 1.000010f, c4 = 0.0f;
  float a5 = 1.000011f, b5 = 1.000012f, c5 = 0.0f;
  float a6 = 1.000013f, b6 = 1.000014f, c6 = 0.0f;
  float a7 = 1.000015f, b7 = 1.000016f, c7 = 0.0f;

  #pragma unroll 1
  for (int i = 0; i < kernel_iters; ++i) {
    #pragma unroll 1
    for (int u = 0; u < unroll; ++u) {
      c0 = fmaf(a0, b0, c0);
      c1 = fmaf(a1, b1, c1);
      c2 = fmaf(a2, b2, c2);
      c3 = fmaf(a3, b3, c3);
      c4 = fmaf(a4, b4, c4);
      c5 = fmaf(a5, b5, c5);
      c6 = fmaf(a6, b6, c6);
      c7 = fmaf(a7, b7, c7);

      // Small perturbation to discourage aggressive constant folding.
      a0 += 1e-7f; b0 -= 1e-7f;
      a1 += 1e-7f; b1 -= 1e-7f;
      a2 += 1e-7f; b2 -= 1e-7f;
      a3 += 1e-7f; b3 -= 1e-7f;
      a4 += 1e-7f; b4 -= 1e-7f;
      a5 += 1e-7f; b5 -= 1e-7f;
      a6 += 1e-7f; b6 -= 1e-7f;
      a7 += 1e-7f; b7 -= 1e-7f;
    }
  }

  out[tid] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
}

torch::Tensor cuda_core_cuda(torch::Tensor out, int64_t kernel_iters, int64_t unroll) {
  const int n = static_cast<int>(out.numel());
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  fp32_fma_bench_kernel<<<blocks, threads>>>(
      out.data_ptr<float>(),
      n,
      static_cast<int>(kernel_iters),
      static_cast<int>(unroll));

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

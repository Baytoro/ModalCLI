#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void reduce_sum_base_kernel(const float* x, float* out, int64_t n) {
  extern __shared__ float sdata[];

  const int tid = threadIdx.x;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  float sum = 0.0f;
  for (int64_t i = idx; i < n; i += stride) {
    sum += x[i];
  }

  sdata[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sdata[tid] += sdata[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

torch::Tensor reduce_sum_cuda(torch::Tensor x, int64_t block_size, int64_t num_blocks) {
  auto out = torch::zeros({1}, x.options());
  const int threads = static_cast<int>(block_size);
  const int blocks = static_cast<int>(num_blocks);
  const int64_t n = x.numel();
  const size_t shmem_bytes = static_cast<size_t>(threads) * sizeof(float);

  C10_CUDA_CHECK(cudaMemset(out.data_ptr<float>(), 0, sizeof(float)));
  reduce_sum_base_kernel<<<blocks, threads, shmem_bytes>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      n);
  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

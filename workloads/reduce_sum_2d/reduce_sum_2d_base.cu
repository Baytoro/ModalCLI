#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void reduce_sum_2d_base_kernel(const float* x, float* out, int rows, int cols) {
  extern __shared__ float sdata[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  float thread_sum = 0.0f;
  const int row_offset = row * cols;
  for (int c = tid; c < cols; c += blockDim.x) {
    thread_sum += x[row_offset + c];
  }

  sdata[tid] = thread_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sdata[tid] += sdata[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[row] = sdata[0];
  }
}

torch::Tensor reduce_sum_2d_cuda(torch::Tensor x, int64_t block_size) {
  const int rows = static_cast<int>(x.size(0));
  const int cols = static_cast<int>(x.size(1));
  const int threads = static_cast<int>(block_size);
  const int blocks = rows;
  const size_t shmem_bytes = static_cast<size_t>(threads) * sizeof(float);

  auto out = torch::empty({rows}, x.options());
  reduce_sum_2d_base_kernel<<<blocks, threads, shmem_bytes>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      rows,
      cols);
  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

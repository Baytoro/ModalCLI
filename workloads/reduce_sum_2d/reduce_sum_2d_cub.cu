#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <int BLOCK_THREADS>
__global__ void reduce_sum_2d_cub_kernel(const float* x, float* out, int rows, int cols) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int row_offset = row * cols;
  float thread_sum = 0.0f;
  for (int c = tid; c < cols; c += BLOCK_THREADS) {
    thread_sum += x[row_offset + c];
  }

  const float row_sum = BlockReduce(temp_storage).Sum(thread_sum);
  if (tid == 0) {
    out[row] = row_sum;
  }
}

torch::Tensor reduce_sum_2d_cuda(torch::Tensor x, int64_t block_size) {
  const int rows = static_cast<int>(x.size(0));
  const int cols = static_cast<int>(x.size(1));
  const int threads = static_cast<int>(block_size);
  const int blocks = rows;
  auto out = torch::empty({rows}, x.options());

  switch (threads) {
    case 64:
      reduce_sum_2d_cub_kernel<64><<<blocks, 64>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
      break;
    case 128:
      reduce_sum_2d_cub_kernel<128><<<blocks, 128>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
      break;
    case 256:
      reduce_sum_2d_cub_kernel<256><<<blocks, 256>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
      break;
    case 512:
      reduce_sum_2d_cub_kernel<512><<<blocks, 512>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
      break;
    case 1024:
      reduce_sum_2d_cub_kernel<1024><<<blocks, 1024>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
      break;
    default:
      TORCH_CHECK(
          false,
          "reduce_sum_2d_cub supports block_size in {64, 128, 256, 512, 1024}, got ",
          threads);
  }

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

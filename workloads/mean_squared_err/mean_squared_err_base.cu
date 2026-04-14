#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

__global__ void mean_squared_err_kernel(
    const float* predictions,
    const float* targets,
    float* sq_errors,
    int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float diff = predictions[i] - targets[i];
    sq_errors[i] = diff * diff;
  }
}

__global__ void reduce_sum_kernel(
    const float* input,
    float* output,
    int64_t n) {
  __shared__ float shared_data[256];
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int64_t j = i; j < n; j += blockDim.x * gridDim.x) {
    sum += input[j];
  }
  shared_data[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_data[tid] += shared_data[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(output, shared_data[0]);
  }
}

torch::Tensor mean_squared_err_cuda(torch::Tensor predictions, torch::Tensor targets) {
  auto sq_errors = torch::empty_like(predictions);
  const int64_t n = predictions.numel();
  constexpr int threads = 256;
  const int blocks = (static_cast<int>(n) + threads - 1) / threads;

  mean_squared_err_kernel<<<blocks, threads>>>(
      predictions.data_ptr<float>(),
      targets.data_ptr<float>(),
      sq_errors.data_ptr<float>(),
      n);

  C10_CUDA_CHECK(cudaGetLastError());

  auto total = torch::empty({1}, dtype=torch::kFloat32, device=torch::kCUDA);
  total.fill_(0.0f);
  
  const int reduce_blocks = 1024;
  reduce_sum_kernel<<<reduce_blocks, threads>>>(
      sq_errors.data_ptr<float>(),
      total.data_ptr<float>(),
      n);

  C10_CUDA_CHECK(cudaGetLastError());

  float mse = total.item<float>() / static_cast<float>(n);
  auto result = torch::tensor(mse, dtype=torch::kFloat32, device=torch::kCUDA);
  
  return result;
}

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void mean_squared_err_kernel(
    const float* predictions,
    const float* targets,
    float* sq_errors,
    int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float diff = predictions[i] - targets[i];
    sq_errors[i] = diff * diff;
  }
}

__global__ void reduce_sum_kernel(
    const float* input,
    float* output,
    int n) {
  __shared__ float shared_data[256];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int j = i; j < n; j += blockDim.x * gridDim.x) {
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

extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
  float* sq_errors = nullptr;
  cudaMalloc(&sq_errors, N * sizeof(float));
  
  constexpr int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  mean_squared_err_kernel<<<blocks, threads>>>(predictions, targets, sq_errors, N);
  C10_CUDA_CHECK(cudaGetLastError());

  float* total = nullptr;
  cudaMalloc(&total, sizeof(float));
  cudaMemset(total, 0, sizeof(float));
  
  const int reduce_blocks = 1024;
  reduce_sum_kernel<<<reduce_blocks, threads>>>(sq_errors, total, N);
  C10_CUDA_CHECK(cudaGetLastError());

  float mse_value;
  cudaMemcpy(&mse_value, total, sizeof(float), cudaMemcpyDeviceToHost);
  *mse = mse_value / static_cast<float>(N);

  cudaFree(sq_errors);
  cudaFree(total);
}

torch::Tensor mean_squared_err_cuda(torch::Tensor predictions, torch::Tensor targets) {
  const int64_t n = predictions.numel();
  auto result = torch::empty({1}, dtype=torch::kFloat32, device=torch::kCUDA);
  
  solve(predictions.data_ptr<float>(), targets.data_ptr<float>(), result.data_ptr<float>(), static_cast<int>(n));
  
  return result;
}

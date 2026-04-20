#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int THREAD_PER_BLOCK = 256;
constexpr int BLOCK_TILE = 4096;

__global__ void softmax_max_kernel(const float *input, int *max_out, int64_t n) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    float local_max = -INFINITY;
    for (int64_t i = idx; i < n; i += stride) {
        local_max = fmaxf(local_max, input[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_out, *reinterpret_cast<int *>(&sdata[0]));
    }
}

__global__ void softmax_exp_sum_kernel(const float *input, float *sum_out, int64_t n, float max_val) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    float local_sum = 0.0f;
    for (int64_t i = idx; i < n; i += stride) {
        local_sum += expf(input[i] - max_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_out, sdata[0]);
    }
}

__global__ void softmax_compute_kernel(const float *input, float *output, int64_t n, float max_val, float sum_inv) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = expf(input[i] - max_val) * sum_inv;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    const int64_t n = x.numel();
    auto output = torch::empty_like(x);

    int *d_max;
    float *d_sum;
    C10_CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));
    C10_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    C10_CUDA_CHECK(cudaMemset(d_max, 0, sizeof(int)));
    C10_CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    const int threads = THREAD_PER_BLOCK;
    const int blocks = static_cast<int>((n + BLOCK_TILE - 1) / BLOCK_TILE);
    const size_t shmem_bytes = static_cast<size_t>(threads) * sizeof(float);

    softmax_max_kernel<<<blocks, threads, shmem_bytes>>>(x.data_ptr<float>(), d_max, n);
    C10_CUDA_CHECK(cudaGetLastError());

    int max_bits;
    C10_CUDA_CHECK(cudaMemcpy(&max_bits, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    float max_val = reinterpret_cast<float &>(max_bits);

    softmax_exp_sum_kernel<<<blocks, threads, shmem_bytes>>>(x.data_ptr<float>(), d_sum, n, max_val);
    C10_CUDA_CHECK(cudaGetLastError());

    float sum_val;
    C10_CUDA_CHECK(cudaMemcpy(&sum_val, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    float sum_inv = 1.0f / sum_val;

    const int compute_blocks = static_cast<int>((n + threads - 1) / threads);
    softmax_compute_kernel<<<compute_blocks, threads>>>(x.data_ptr<float>(), output.data_ptr<float>(), n, max_val,
                                                        sum_inv);
    C10_CUDA_CHECK(cudaGetLastError());

    C10_CUDA_CHECK(cudaFree(d_max));
    C10_CUDA_CHECK(cudaFree(d_sum));

    return output;
}
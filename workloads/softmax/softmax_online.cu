#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>

constexpr int TILE = 2048;
constexpr int THREAD_PER_BLOCK = 512;
__device__ float d_max_arr[THREAD_PER_BLOCK];
__device__ float d_sum_arr[THREAD_PER_BLOCK];

__device__ void warp_softmax(float &sum_v, float &max_v) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        float next_sum = __shfl_down_sync(mask, sum_v, offset);
        float next_max = __shfl_down_sync(mask, max_v, offset);
        float new_max = fmaxf(max_v, next_max);
        sum_v = sum_v * __expf(max_v - new_max) + next_sum * __expf(next_max - new_max);
        max_v = new_max;
    }
}

__global__ void softmax_kernel_sum_max(const float *input, float *output, int N) {
    int blk_start = blockIdx.x * TILE;
    int blk_end = min(blk_start + TILE, N);
    __shared__ float s_max[THREAD_PER_BLOCK];
    __shared__ float s_sum[THREAD_PER_BLOCK];
    float max_v = -std::numeric_limits<float>::infinity();
    float sum_v = 0.0f;
    for (int i = blk_start + threadIdx.x; i < blk_end; i += blockDim.x) {
        float val = input[i];
        if (max_v > val) {
            sum_v += __expf(val - max_v);
        } else {
            sum_v = sum_v * __expf(max_v - val) + 1.0f;
            max_v = val;
        }
    }
    s_max[threadIdx.x] = max_v;
    s_sum[threadIdx.x] = sum_v;
    __syncthreads();
    for (int i = THREAD_PER_BLOCK / 2; i >= 32; i /= 2) {
        if (threadIdx.x < i) {
            float max_val1 = s_max[threadIdx.x];
            float max_val2 = s_max[threadIdx.x + i];
            float sum_val1 = s_sum[threadIdx.x];
            float sum_val2 = s_sum[threadIdx.x + i];
            float new_max = fmaxf(max_val1, max_val2);
            s_sum[threadIdx.x] = sum_val1 * __expf(max_val1 - new_max) + sum_val2 * __expf(max_val2 - new_max);
            s_max[threadIdx.x] = new_max;
        }
        __syncthreads();
    }
    float r_sum = s_sum[threadIdx.x];
    float r_max = s_max[threadIdx.x];
    warp_softmax(r_sum, r_max);
    if (threadIdx.x == 0) {
        d_max_arr[blockIdx.x] = r_max;
        d_sum_arr[blockIdx.x] = r_sum;
    }
}

__global__ void softmax_kernel(const float *input, float *output, int N) {
    int max_size = gridDim.x;
    __shared__ float s_max[THREAD_PER_BLOCK];
    __shared__ float s_sum[THREAD_PER_BLOCK];
    float sum_val1 = 0.0f;
    float max_val1 = -std::numeric_limits<float>::infinity();
    for (int i = threadIdx.x; i < max_size; i += blockDim.x) {
        float sum_val2 = d_sum_arr[i];
        float max_val2 = d_max_arr[i];
        float new_max = fmaxf(max_val1, max_val2);
        sum_val1 = sum_val1 * __expf(max_val1 - new_max) + sum_val2 * __expf(max_val2 - new_max);
        max_val1 = new_max;
    }
    s_max[threadIdx.x] = max_val1;
    s_sum[threadIdx.x] = sum_val1;
    __syncthreads();
    for (int i = THREAD_PER_BLOCK / 2; i >= 32; i /= 2) {
        if (threadIdx.x < i) {
            max_val1 = s_max[threadIdx.x];
            float max_val2 = s_max[threadIdx.x + i];
            sum_val1 = s_sum[threadIdx.x];
            float sum_val2 = s_sum[threadIdx.x + i];
            float new_max = fmaxf(max_val1, max_val2);
            s_sum[threadIdx.x] = sum_val1 * __expf(max_val1 - new_max) + sum_val2 * __expf(max_val2 - new_max);
            s_max[threadIdx.x] = new_max;
        }
        __syncthreads();
    }
    float r_sum = s_sum[threadIdx.x];
    float r_max = s_max[threadIdx.x];
    warp_softmax(r_sum, r_max);
    s_sum[threadIdx.x] = r_sum;
    s_max[threadIdx.x] = r_max;
    __syncthreads();
    float sum_val = s_sum[0];
    float max_val = s_max[0];
    int blk_start = blockIdx.x * TILE;
    int blk_end = min(blk_start + TILE, N);
    for (int i = blk_start + threadIdx.x; i < blk_end; i += blockDim.x) {
        float val = input[i];
        output[i] = __expf(val - max_val) / sum_val;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    const int64_t n = x.numel();
    auto output = torch::empty_like(x);
    const int threadsPerBlock = THREAD_PER_BLOCK;
    const int blocksPerGrid = static_cast<int>((n + TILE - 1) / TILE);
    softmax_kernel_sum_max<<<blocksPerGrid, threadsPerBlock>>>(x.data_ptr<float>(), output.data_ptr<float>(),
                                                               static_cast<int>(n));
    C10_CUDA_CHECK(cudaGetLastError());
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(x.data_ptr<float>(), output.data_ptr<float>(),
                                                       static_cast<int>(n));
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}
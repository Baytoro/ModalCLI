#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int THREAD_PER_BLOCK_F4 = 256;
constexpr int BLOCK_TILE = 4096;

__global__ void reduce_sum_float4_kernel(const float *input, float *output, int64_t n) {
    int blk_start = BLOCK_TILE * blockIdx.x;
    int blk_end = min(blk_start + BLOCK_TILE, static_cast<int>(n));
    int blk_len = blk_end - blk_start;
    int blk_align_len = blk_len / 4 * 4;

    __shared__ float smem[THREAD_PER_BLOCK_F4];

    float r_sum = 0.0f;
    int i = threadIdx.x * 4;
    for (; i < blk_align_len; i += (THREAD_PER_BLOCK_F4 * 4)) {
        float4 val = *reinterpret_cast<const float4 *>(input + blk_start + i);
        val.z += val.x;
        val.w += val.y;
        r_sum += val.z;
        r_sum += val.w;
    }

    for (; i < blk_len; i++) {
        r_sum += input[blk_start + i];
    }

    smem[threadIdx.x] = r_sum;
    __syncthreads();

    for (int offset = (THREAD_PER_BLOCK_F4 / 2); offset >= 32; offset /= 2) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        }
        __syncthreads();
    }

    r_sum = smem[threadIdx.x];
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        r_sum += __shfl_down_sync(0xffffffff, r_sum, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, r_sum);
    }
}

torch::Tensor reduce_sum_cuda(torch::Tensor x, int64_t block_size, int64_t num_blocks) {
    auto out = torch::zeros({1}, x.options());
    const int64_t n = x.numel();
    const int blocks = static_cast<int>(num_blocks);
    constexpr int threads = THREAD_PER_BLOCK_F4;

    C10_CUDA_CHECK(cudaMemset(out.data_ptr<float>(), 0, sizeof(float)));
    reduce_sum_float4_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
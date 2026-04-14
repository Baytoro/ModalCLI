#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int BLOCK_TILE = 2048;
constexpr int THREAD_PER_BLOCK = 256;
constexpr int VEC_SIZE = 4;

__global__ void mean_squared_err_kernel(const float *predictions, const float *targets, float *mse, int N) {
    int blk_start = blockIdx.x * BLOCK_TILE;
    int blk_end = min(blk_start + BLOCK_TILE, N);
    int blk_len = blk_end - blk_start;
    int align_blk_len = (blk_len / VEC_SIZE) * VEC_SIZE;

    __shared__ float smem[THREAD_PER_BLOCK];

    float r_sum = 0.0f;
    int i = threadIdx.x * VEC_SIZE;
    for (; i < align_blk_len; i += THREAD_PER_BLOCK * VEC_SIZE) {
        float4 a = *reinterpret_cast<const float4 *>(predictions + blk_start + i);
        float4 b = *reinterpret_cast<const float4 *>(targets + blk_start + i);
        r_sum += (a.x - b.x) * (a.x - b.x);
        r_sum += (a.y - b.y) * (a.y - b.y);
        r_sum += (a.z - b.z) * (a.z - b.z);
        r_sum += (a.w - b.w) * (a.w - b.w);
    }

    for (; i < blk_len; i++) {
        float a = predictions[blk_start + i];
        float b = targets[blk_start + i];
        r_sum += (a - b) * (a - b);
    }

    smem[threadIdx.x] = r_sum;
    __syncthreads();

    for (int offset = THREAD_PER_BLOCK / 2; offset >= 32; offset /= 2) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        }
        __syncthreads();
    }

    r_sum = smem[threadIdx.x];
    auto mask = __activemask();
    for (int offset = 16; offset > 0; offset /= 2) {
        r_sum += __shfl_down_sync(mask, r_sum, offset);
    }

    if (threadIdx.x == 0) {
        r_sum /= static_cast<float>(N);
        atomicAdd(mse, r_sum);
    }
}

extern "C" void solve(const float *predictions, const float *targets, float *mse, int N) {
    dim3 block(THREAD_PER_BLOCK);
    dim3 grid((N + BLOCK_TILE - 1) / BLOCK_TILE);
    mean_squared_err_kernel<<<grid, block>>>(predictions, targets, mse, N);
}

torch::Tensor mean_squared_err_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int64_t n = predictions.numel();
    auto result = torch::zeros({1}, dtype = torch::kFloat32, device = torch::kCUDA);

    solve(predictions.data_ptr<float>(), targets.data_ptr<float>(), result.data_ptr<float>(), static_cast<int>(n));

    return result;
}

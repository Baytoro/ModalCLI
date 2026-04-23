#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// 每个 block 处理的数据量: BLOCK_TILE elements
// VEC_SIZE=4 表示每次向量化成 int4 访问
// THREAD_PER_BLOCK=256 每个 block 的线程数
constexpr int BLOCK_TILE = 8192 * 32;
constexpr int VEC_SIZE = 4;
constexpr int THREAD_PER_BLOCK = 256;

// Simple histogram kernel without cluster
__global__ void histogram_kernel(const int *input, int *histogram, int N, int num_bins) {
    // 声明 shared memory 用于 block 内累加
    __shared__ int smem[1024];

    // 初始化 shared memory 为 0 (num_bins <= 1024)
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        smem[i] = 0;
    }
    __syncthreads();

    // 计算当前 block 负责的数据范围
    int blk_start = blockIdx.x * BLOCK_TILE;
    int blk_end = min(blk_start + BLOCK_TILE, N);
    int blk_len = blk_end - blk_start;

    // 对齐到 VEC_SIZE 的长度
    int align_blk_len = (blk_len / VEC_SIZE) * VEC_SIZE;

    // 主循环: 向量化加载 + atomicAdd 到 shared memory
    // 每个线程一次处理 VEC_SIZE=4 个元素
    int i = threadIdx.x * VEC_SIZE;
    for (; i < align_blk_len; i += blockDim.x * VEC_SIZE) {
        int4 r_i = *(int4 *)(input + blk_start + i);
        // 对 4 个值分别 atomicAdd 到 shared memory 对应 bin
        atomicAdd(&smem[r_i.x], 1);
        atomicAdd(&smem[r_i.y], 1);
        atomicAdd(&smem[r_i.z], 1);
        atomicAdd(&smem[r_i.w], 1);
    }
    __syncthreads();

    // 处理尾部不满 VEC_SIZE 的元素
    for (; i < blk_len; ++i) {
        int r_i = input[blk_start + i];
        atomicAdd(&smem[r_i], 1);
    }
    __syncthreads();

    // 将 block 内累加结果写入全局 histogram
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(histogram + i, smem[i]);
    }
}

// PyTorch wrapper
torch::Tensor histogram_cuda(torch::Tensor input, int64_t num_bins) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kInt32, "only int32 is supported");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const int64_t N = input.numel();
    auto histogram_out = torch::zeros({num_bins}, input.options().dtype(torch::kInt32));

    const int threads = THREAD_PER_BLOCK;
    const int blocks = static_cast<int>((N + BLOCK_TILE - 1) / BLOCK_TILE);

    histogram_kernel<<<blocks, threads>>>(input.data_ptr<int>(), histogram_out.data_ptr<int>(), static_cast<int>(N),
                                          static_cast<int>(num_bins));
    C10_CUDA_CHECK(cudaGetLastError());

    return histogram_out;
}
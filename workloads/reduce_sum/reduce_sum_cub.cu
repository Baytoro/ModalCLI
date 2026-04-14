#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

torch::Tensor reduce_sum_cuda(torch::Tensor x, int64_t block_size, int64_t num_blocks) {
    (void)block_size;
    (void)num_blocks;

    auto out = torch::zeros({1}, x.options());
    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    C10_CUDA_CHECK(cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, x.data_ptr<float>(), out.data_ptr<float>(),
                                          static_cast<int>(x.numel())));

    auto temp = torch::empty({static_cast<long long>(temp_storage_bytes)},
                             torch::TensorOptions().device(x.device()).dtype(torch::kUInt8));

    C10_CUDA_CHECK(cub::DeviceReduce::Sum(temp.data_ptr(), temp_storage_bytes, x.data_ptr<float>(),
                                          out.data_ptr<float>(), static_cast<int>(x.numel())));

    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

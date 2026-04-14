#include <torch/extension.h>

torch::Tensor reduce_sum_cuda(torch::Tensor x, int64_t block_size, int64_t num_blocks);

torch::Tensor reduce_sum(torch::Tensor x, int64_t block_size, int64_t num_blocks) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 1, "x must be a 1D tensor");
    TORCH_CHECK(block_size > 0 && block_size <= 1024, "block_size must be in (0, 1024]");
    TORCH_CHECK(num_blocks > 0, "num_blocks must be > 0");
    return reduce_sum_cuda(x, block_size, num_blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("reduce_sum", &reduce_sum, "1D reduce-sum (CUDA)"); }

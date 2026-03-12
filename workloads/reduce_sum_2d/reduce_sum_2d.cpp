#include <torch/extension.h>

torch::Tensor reduce_sum_2d_cuda(torch::Tensor x, int64_t block_size);

torch::Tensor reduce_sum_2d(torch::Tensor x, int64_t block_size) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 is supported");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
  TORCH_CHECK(block_size > 0 && block_size <= 1024, "block_size must be in (0, 1024]");
  return reduce_sum_2d_cuda(x, block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_sum_2d", &reduce_sum_2d, "2D row-wise reduce-sum (CUDA)");
}

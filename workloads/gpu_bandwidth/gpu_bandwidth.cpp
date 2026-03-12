#include <torch/extension.h>

torch::Tensor gpu_bandwidth_cuda(torch::Tensor src, int64_t copy_iters);

torch::Tensor gpu_bandwidth(torch::Tensor src, int64_t copy_iters) {
  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  TORCH_CHECK(src.scalar_type() == torch::kFloat32, "only float32 is supported");
  TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
  TORCH_CHECK(src.dim() == 1, "src must be a 1D tensor");
  TORCH_CHECK(copy_iters > 0, "copy_iters must be > 0");
  return gpu_bandwidth_cuda(src, copy_iters);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gpu_bandwidth", &gpu_bandwidth, "GPU memory bandwidth benchmark (copy)");
}

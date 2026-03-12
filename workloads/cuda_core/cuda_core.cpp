#include <torch/extension.h>

torch::Tensor cuda_core_cuda(torch::Tensor out, int64_t kernel_iters, int64_t unroll);

torch::Tensor cuda_core(torch::Tensor out, int64_t kernel_iters, int64_t unroll) {
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "only float32 is supported");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.dim() == 1, "out must be 1D");
  TORCH_CHECK(kernel_iters > 0, "kernel_iters must be > 0");
  TORCH_CHECK(unroll > 0, "unroll must be > 0");
  return cuda_core_cuda(out, kernel_iters, unroll);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_core", &cuda_core, "FP32 FMA CUDA core benchmark kernel");
}

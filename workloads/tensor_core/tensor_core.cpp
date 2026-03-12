#include <torch/extension.h>

torch::Tensor tensor_core_cuda(
    torch::Tensor out,
    int64_t num_blocks,
    int64_t block_size,
    int64_t kernel_iters,
    int64_t repeat,
    int64_t accumulators);

torch::Tensor tensor_core(
    torch::Tensor out,
    int64_t num_blocks,
    int64_t block_size,
    int64_t kernel_iters,
    int64_t repeat,
    int64_t accumulators) {
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.dim() == 1, "out must be a 1D tensor");
  TORCH_CHECK(num_blocks > 0, "num_blocks must be > 0");
  TORCH_CHECK(block_size > 0, "block_size must be > 0");
  TORCH_CHECK(block_size % 32 == 0, "block_size must be a multiple of warp size (32)");
  TORCH_CHECK(block_size <= 1024, "block_size must be <= 1024");
  TORCH_CHECK(kernel_iters > 0, "kernel_iters must be > 0");
  TORCH_CHECK(repeat > 0, "repeat must be > 0");
  TORCH_CHECK(accumulators > 0, "accumulators must be > 0");
  return tensor_core_cuda(out, num_blocks, block_size, kernel_iters, repeat, accumulators);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tensor_core", &tensor_core, "Tensor Core WMMA microbenchmark (FP16 input, FP32 accumulate)");
}

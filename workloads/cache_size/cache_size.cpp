#include <torch/extension.h>

torch::Tensor cache_size_cuda(
    torch::Tensor buf,
    torch::Tensor sizes_bytes,
    int64_t probe_iters,
    int64_t repeat,
    const std::string& report_path);

torch::Tensor cache_size(
    torch::Tensor buf,
    torch::Tensor sizes_bytes,
    int64_t probe_iters,
    int64_t repeat,
    const std::string& report_path) {
  TORCH_CHECK(buf.is_cuda(), "buf must be a CUDA tensor");
  TORCH_CHECK(buf.scalar_type() == torch::kFloat32, "buf must be float32");
  TORCH_CHECK(buf.is_contiguous(), "buf must be contiguous");
  TORCH_CHECK(buf.dim() == 1, "buf must be 1D");

  TORCH_CHECK(!sizes_bytes.is_cuda(), "sizes_bytes must be a CPU tensor");
  TORCH_CHECK(sizes_bytes.scalar_type() == torch::kInt64, "sizes_bytes must be int64");
  TORCH_CHECK(sizes_bytes.dim() == 1, "sizes_bytes must be 1D");
  TORCH_CHECK(sizes_bytes.numel() > 0, "sizes_bytes must be non-empty");

  TORCH_CHECK(probe_iters > 0, "probe_iters must be > 0");
  TORCH_CHECK(repeat > 0, "repeat must be > 0");
  TORCH_CHECK(!report_path.empty(), "report_path must not be empty");

  return cache_size_cuda(buf, sizes_bytes, probe_iters, repeat, report_path);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cache_size", &cache_size, "Estimate GPU L1/L2 cache sizes by latency sweep");
}

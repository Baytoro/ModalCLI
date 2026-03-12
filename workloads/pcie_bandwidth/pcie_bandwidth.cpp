#include <torch/extension.h>

torch::Tensor pcie_bandwidth_cuda(
    torch::Tensor host_buf,
    torch::Tensor device_buf,
    int64_t transfer_iters,
    int64_t nbytes);

torch::Tensor pcie_bandwidth(
    torch::Tensor host_buf,
    torch::Tensor device_buf,
    int64_t transfer_iters,
    int64_t nbytes) {
  TORCH_CHECK(!host_buf.is_cuda(), "host_buf must be a CPU tensor");
  TORCH_CHECK(device_buf.is_cuda(), "device_buf must be a CUDA tensor");
  TORCH_CHECK(host_buf.scalar_type() == torch::kFloat32, "host_buf must be float32");
  TORCH_CHECK(device_buf.scalar_type() == torch::kFloat32, "device_buf must be float32");
  TORCH_CHECK(host_buf.is_contiguous(), "host_buf must be contiguous");
  TORCH_CHECK(device_buf.is_contiguous(), "device_buf must be contiguous");
  TORCH_CHECK(transfer_iters > 0, "transfer_iters must be > 0");
  TORCH_CHECK(nbytes > 0, "nbytes must be > 0");
  TORCH_CHECK(
      static_cast<int64_t>(host_buf.numel()) * static_cast<int64_t>(sizeof(float)) >= nbytes,
      "host_buf is smaller than nbytes");
  TORCH_CHECK(
      static_cast<int64_t>(device_buf.numel()) * static_cast<int64_t>(sizeof(float)) >= nbytes,
      "device_buf is smaller than nbytes");
  return pcie_bandwidth_cuda(host_buf, device_buf, transfer_iters, nbytes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pcie_bandwidth", &pcie_bandwidth, "PCIe bandwidth benchmark");
}

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

torch::Tensor pcie_bandwidth_cuda(
    torch::Tensor host_buf,
    torch::Tensor device_buf,
    int64_t kernel_iters,
    int64_t nbytes) {
  auto stream = at::cuda::getDefaultCUDAStream();
  auto* dst = host_buf.data_ptr<float>();
  const auto* src = device_buf.data_ptr<float>();

  for (int i = 0; i < static_cast<int>(kernel_iters); ++i) {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst,
        src,
        static_cast<size_t>(nbytes),
        cudaMemcpyDeviceToHost,
        stream));
  }
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  C10_CUDA_CHECK(cudaGetLastError());
  return device_buf;
}

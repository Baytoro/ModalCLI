#include <torch/extension.h>

torch::Tensor softmax_cuda(torch::Tensor x);

torch::Tensor softmax(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 1, "x must be a 1D tensor");
    return softmax_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("softmax", &softmax, "Softmax (CUDA)"); }
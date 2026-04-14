#include <torch/extension.h>

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    return vector_add_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("vector_add", &vector_add, "Vector add (CUDA)"); }

#include <torch/extension.h>

torch::Tensor mean_squared_err_cuda(torch::Tensor predictions, torch::Tensor targets);

torch::Tensor mean_squared_err(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "shape mismatch");
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    return mean_squared_err_cuda(predictions, targets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mean_squared_err", &mean_squared_err, "Mean Squared Error (CUDA)"); }

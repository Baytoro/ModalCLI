#include <torch/extension.h>

torch::Tensor histogram_cuda(torch::Tensor input, int64_t num_bins);

torch::Tensor histogram(torch::Tensor input, int64_t num_bins) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kInt32, "only int32 is supported");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    return histogram_cuda(input, num_bins);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("histogram", &histogram, "Histogram (CUDA)"); }
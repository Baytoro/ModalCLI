#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace {

__global__ void pointer_chase_kernel(const int* next_idx, int* sink, int probe_iters) {
  int idx = 0;
  #pragma unroll 1
  for (int i = 0; i < probe_iters; ++i) {
    idx = next_idx[idx];
  }
  sink[0] = idx;
}

int round_down_pow2(int v) {
  int p = 1;
  while ((p << 1) > 0 && (p << 1) <= v) {
    p <<= 1;
  }
  return p;
}

std::pair<int, int> estimate_l1_l2_bytes(
    const std::vector<int64_t>& sizes,
    const std::vector<float>& lat_ns) {
  if (sizes.empty() || lat_ns.empty() || sizes.size() != lat_ns.size()) {
    return {0, 0};
  }

  int l1_idx = -1;
  int l2_idx = -1;
  constexpr float ratio_threshold = 1.20f;

  for (size_t i = 1; i < lat_ns.size(); ++i) {
    if (lat_ns[i - 1] <= 0.0f) {
      continue;
    }
    const float ratio = lat_ns[i] / lat_ns[i - 1];
    if (ratio > ratio_threshold) {
      if (l1_idx < 0) {
        l1_idx = static_cast<int>(i) - 1;
      } else {
        l2_idx = static_cast<int>(i) - 1;
        break;
      }
    }
  }

  if (l1_idx < 0) {
    l1_idx = 0;
  }
  if (l2_idx < 0) {
    l2_idx = static_cast<int>(sizes.size()) - 1;
  }
  if (l2_idx < l1_idx) {
    l2_idx = l1_idx;
  }

  return {
      static_cast<int>(sizes[static_cast<size_t>(l1_idx)]),
      static_cast<int>(sizes[static_cast<size_t>(l2_idx)]),
  };
}

}  // namespace

torch::Tensor cache_size_cuda(
    torch::Tensor buf,
    torch::Tensor sizes_bytes,
    int64_t probe_iters,
    int64_t repeat,
    const std::string& report_path) {
  constexpr int stride_elems = 32;  // 128B stride.

  auto sink = torch::empty({1}, torch::TensorOptions().device(buf.device()).dtype(torch::kInt32));
  auto next_dev = torch::empty({buf.numel()}, torch::TensorOptions().device(buf.device()).dtype(torch::kInt32));
  auto latencies = torch::empty({sizes_bytes.numel()}, torch::TensorOptions().dtype(torch::kFloat32));

  const auto* sizes_ptr = sizes_bytes.data_ptr<int64_t>();
  auto* lat_ptr = latencies.data_ptr<float>();
  std::vector<int64_t> sizes_vec;
  std::vector<float> lat_vec;
  sizes_vec.reserve(static_cast<size_t>(sizes_bytes.numel()));
  lat_vec.reserve(static_cast<size_t>(sizes_bytes.numel()));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  C10_CUDA_CHECK(cudaEventCreate(&start));
  C10_CUDA_CHECK(cudaEventCreate(&stop));

  for (int64_t i = 0; i < sizes_bytes.numel(); ++i) {
    int64_t ws_bytes = sizes_ptr[i];
    if (ws_bytes < static_cast<int64_t>(sizeof(float) * 2)) {
      ws_bytes = sizeof(float) * 2;
    }
    int elems = static_cast<int>(ws_bytes / static_cast<int64_t>(sizeof(int)));
    if (elems > static_cast<int>(buf.numel())) {
      elems = static_cast<int>(buf.numel());
    }
    elems = round_down_pow2(elems);
    if (elems < 2) {
      elems = 2;
    }
    std::vector<int> next_host(static_cast<size_t>(elems));
    for (int j = 0; j < elems; ++j) {
      next_host[static_cast<size_t>(j)] = (j + stride_elems) & (elems - 1);
    }
    C10_CUDA_CHECK(cudaMemcpy(
        next_dev.data_ptr<int>(),
        next_host.data(),
        static_cast<size_t>(elems) * sizeof(int),
        cudaMemcpyHostToDevice));

    float total_ms = 0.0f;
    for (int r = 0; r < static_cast<int>(repeat); ++r) {
      C10_CUDA_CHECK(cudaEventRecord(start));
      pointer_chase_kernel<<<1, 1>>>(
          next_dev.data_ptr<int>(),
          sink.data_ptr<int>(),
          static_cast<int>(probe_iters));
      C10_CUDA_CHECK(cudaEventRecord(stop));
      C10_CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0.0f;
      C10_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      total_ms += ms;
    }
    C10_CUDA_CHECK(cudaGetLastError());

    const float avg_ms = total_ms / static_cast<float>(repeat);
    const float lat_ns = (avg_ms * 1e6f) / static_cast<float>(probe_iters);

    lat_ptr[i] = lat_ns;
    sizes_vec.push_back(static_cast<int64_t>(elems) * static_cast<int64_t>(sizeof(int)));
    lat_vec.push_back(lat_ns);
  }

  C10_CUDA_CHECK(cudaEventDestroy(start));
  C10_CUDA_CHECK(cudaEventDestroy(stop));

  auto [estimated_l1, estimated_l2] = estimate_l1_l2_bytes(sizes_vec, lat_vec);

  std::ofstream ofs(report_path, std::ios::out | std::ios::trunc);
  if (ofs.is_open()) {
    ofs << "{";
    ofs << "\"estimated_l1_bytes\":" << estimated_l1 << ",";
    ofs << "\"estimated_l2_bytes\":" << estimated_l2 << ",";
    ofs << "\"latency_ns\":[";
    for (size_t i = 0; i < lat_vec.size(); ++i) {
      if (i > 0) {
        ofs << ",";
      }
      ofs << lat_vec[i];
    }
    ofs << "]";
    ofs << "}";
    ofs.close();
  }

  return latencies;
}

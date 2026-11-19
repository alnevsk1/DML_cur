#include "preprocess.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace dml {
namespace {
constexpr int size = detail::imageSize;
constexpr int plane = size * size;
constexpr int roundBias = 1 << (detail::filterShift - 1);
constexpr unsigned int blockSize = 256;

void checkCuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> struct CudaFree {
  void operator()(T *value) const { cudaFree(value); }
};
template <class T> using Device = std::unique_ptr<T, CudaFree<T>>;

template <class T> Device<T> deviceArray(size_t count) {
  T *memory = nullptr;
  checkCuda(cudaMalloc(&memory, count * sizeof(T)));
  return Device<T>(memory);
}

template <class T> Device<T> upload(const T *values, size_t count) {
  auto memory = deviceArray<T>(count);
  checkCuda(cudaMemcpy(memory.get(), values, count * sizeof(T),
                       cudaMemcpyHostToDevice));
  return memory;
}

struct Filter {
  int first;
  int count;
  size_t offset;
};

struct Axis {
  Device<Filter> filters;
  Device<int> coefficients;
};

Axis prepareAxis(int source) {
  const auto weights = detail::bicubicWeights(source, size);
  std::array<Filter, size> filters{};
  std::vector<int> coefficients;
  for (int i = 0; i < size; ++i) {
    const auto &row = weights[i];
    filters[i] = {row.start, static_cast<int>(row.coefficients.size()),
                  coefficients.size()};
    coefficients.insert(coefficients.end(), row.coefficients.begin(),
                        row.coefficients.end());
  }
  return {upload(filters.data(), filters.size()),
          upload(coefficients.data(), coefficients.size())};
}

__device__ unsigned char rounded(long long value) {
  const auto pixel = value >> detail::filterShift;
  return static_cast<unsigned char>(pixel < 0 ? 0 : (pixel > 255 ? 255 : pixel));
}

__global__ void horizontalPass(const uchar3 *input, int width, size_t pixels,
                               const Filter *filters, const int *coefficients,
                               uchar3 *output) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pixels)
    return;
  const auto filter = filters[i % size];
  const auto *row = input + (i / size) * width + filter.first;
  long long blue = roundBias, green = roundBias, red = roundBias;
  for (int k = 0; k < filter.count; ++k) {
    const auto pixel = row[k];
    const long long weight = coefficients[filter.offset + k];
    blue += pixel.x * weight;
    green += pixel.y * weight;
    red += pixel.z * weight;
  }
  output[i] = make_uchar3(rounded(blue), rounded(green), rounded(red));
}

__global__ void verticalNormalize(const uchar3 *input, bool sameHeight,
                                  const Filter *filters, const int *coefficients,
                                  const float *normalize, float *output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int x = i % size;
  const int y = i / size;
  uchar3 pixel;
  if (sameHeight) {
    pixel = input[i];
  } else {
    const auto filter = filters[y];
    long long blue = roundBias, green = roundBias, red = roundBias;
    for (int k = 0; k < filter.count; ++k) {
      const auto src = input[(static_cast<size_t>(filter.first) + k) * size + x];
      const long long weight = coefficients[filter.offset + k];
      blue += src.x * weight;
      green += src.y * weight;
      red += src.z * weight;
    }
    pixel = make_uchar3(rounded(blue), rounded(green), rounded(red));
  }
  // Filtering keeps BGR byte order. The TensorRT binding is RGB CHW.
  output[i] = normalize[pixel.z];
  output[plane + i] = normalize[256 + pixel.y];
  output[2 * plane + i] = normalize[512 + pixel.x];
}

struct FrameBuffers {
  explicit FrameBuffers(cv::Size dimensions)
      : dimensions(dimensions),
        input(deviceArray<uchar3>(static_cast<size_t>(dimensions.width) *
                                  dimensions.height)) {
    if (dimensions.width != size) {
      horizontal = deviceArray<uchar3>(static_cast<size_t>(size) *
                                       dimensions.height);
      x = prepareAxis(dimensions.width);
    }
    if (dimensions.height != size)
      y = prepareAxis(dimensions.height);
  }

  cv::Size dimensions;
  Device<uchar3> input, horizontal;
  Axis x, y;
};
} // namespace

struct CudaPreprocessor::Impl {
  Device<float> normalization;
  std::unique_ptr<FrameBuffers> frame;
};

CudaPreprocessor::CudaPreprocessor(const std::array<double, 3> &mean,
                                   const std::array<double, 3> &stddev)
    : impl_(std::make_unique<Impl>()) {
  std::array<float, 3 * 256> table{};
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 256; ++i) {
      const auto value = detail::normalizePixel(i, mean[c], stddev[c]);
      if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite image normalization");
      table[c * 256 + i] = value;
    }
  }
  impl_->normalization = upload(table.data(), table.size());
}
CudaPreprocessor::~CudaPreprocessor() = default;

void CudaPreprocessor::enqueue(const cv::Mat &bgr, float *deviceOutput,
                               cudaStream_t stream, float *hostTrace) {
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::runtime_error("CUDA preprocessing expects nonempty BGR uint8");
  if (!impl_->frame || impl_->frame->dimensions != bgr.size())
    impl_->frame = std::make_unique<FrameBuffers>(bgr.size());
  auto &frame = *impl_->frame;
  const auto rowBytes = static_cast<size_t>(bgr.cols) * sizeof(uchar3);
  checkCuda(cudaMemcpy2DAsync(frame.input.get(), rowBytes, bgr.data, bgr.step,
                              rowBytes, bgr.rows, cudaMemcpyHostToDevice, stream));
  const uchar3 *intermediate = frame.input.get();
  if (frame.horizontal) {
    const auto pixels = static_cast<size_t>(size) * bgr.rows;
    const auto blocks = static_cast<unsigned int>((pixels + blockSize - 1) / blockSize);
    horizontalPass<<<blocks, blockSize, 0, stream>>>(
        frame.input.get(), bgr.cols, pixels, frame.x.filters.get(),
        frame.x.coefficients.get(), frame.horizontal.get());
    checkCuda(cudaGetLastError());
    intermediate = frame.horizontal.get();
  }
  static_assert(plane % blockSize == 0);
  verticalNormalize<<<plane / blockSize, blockSize, 0, stream>>>(
      intermediate, bgr.rows == size, frame.y.filters.get(),
      frame.y.coefficients.get(), impl_->normalization.get(), deviceOutput);
  checkCuda(cudaGetLastError());
  if (hostTrace)
    checkCuda(cudaMemcpyAsync(hostTrace, deviceOutput, 3 * plane * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
}
} // namespace dml

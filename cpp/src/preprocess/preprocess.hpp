#pragma once
#include "inference/trt_engine.hpp"
#include <array>
#include <opencv2/core.hpp>
namespace dml {
namespace detail {
inline constexpr int imageSize = 1024;
inline constexpr int filterShift = 22;

struct BicubicWeights {
  int start;
  std::vector<int> coefficients;
};
std::vector<BicubicWeights> bicubicWeights(int source, int target);

inline float normalizePixel(int pixel, double mean, double stddev) {
  // Match the reference loader's float64 calculation with FP32 constants.
  return static_cast<float>((pixel / 255.0 - static_cast<float>(mean)) /
                            static_cast<float>(stddev));
}
} // namespace detail

class CudaPreprocessor {
public:
  CudaPreprocessor(const std::array<double, 3> &mean,
                   const std::array<double, 3> &stddev);
  ~CudaPreprocessor();

  // Enqueue on the vision encoder stream. When provided, hostTrace receives
  // the exact FP32 CHW input after the caller synchronizes the stream.
  void enqueue(const cv::Mat &bgr, float *deviceOutput, cudaStream_t stream,
               float *hostTrace = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

cv::Mat resizePilBicubic(const cv::Mat &rgb, cv::Size size);
Tensor preprocess(const cv::Mat &bgr, const std::array<double, 3> &mean,
                  const std::array<double, 3> &stddev);
cv::Mat restoreLogits(const Tensor &logits, cv::Size size);
} // namespace dml

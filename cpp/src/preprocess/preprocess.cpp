#include "preprocess.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
namespace dml {
namespace detail {
// PIL-compatible separable bicubic (a=-0.5), antialias support and signed
// 22-bit weights. Algorithm reference:
// https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
std::vector<BicubicWeights> bicubicWeights(int source, int target) {
  std::vector<BicubicWeights> result;
  const double scale = static_cast<double>(source) / target;
  const double filterScale = std::max(1.0, scale);
  for (int i = 0; i < target; ++i) {
    const double center = (i + 0.5) * scale;
    int first = std::max(0, static_cast<int>(center - 2 * filterScale + 0.5));
    int last =
        std::min(source, static_cast<int>(center + 2 * filterScale + 0.5));
    std::vector<double> ws;
    double total = 0;
    for (int j = first; j < last; ++j) {
      const double x = std::abs((j - center + 0.5) / filterScale);
      double w = 0;
      if (x < 1)
        w = 1 + x * x * (1.5 * x - 2.5);
      else if (x < 2)
        w = -0.5 * (((x - 5) * x + 8) * x - 4);
      ws.push_back(w);
      total += w;
    }
    BicubicWeights row{first, {}};
    for (double w : ws) {
      const double v = w / total * (1 << filterShift);
      row.coefficients.push_back(static_cast<int>(v + (v < 0 ? -0.5 : 0.5)));
    }
    result.push_back(std::move(row));
  }
  return result;
}
} // namespace detail
namespace {
uchar rounded(int64_t value) {
  return static_cast<uchar>(std::clamp<int64_t>(value >> detail::filterShift, 0, 255));
}
} // namespace
cv::Mat resizePilBicubic(const cv::Mat &rgb, cv::Size size) {
  if (rgb.empty() || rgb.type() != CV_8UC3 || size.width <= 0 ||
      size.height <= 0)
    throw std::runtime_error("resizePilBicubic expects nonempty RGB uint8");
  cv::Mat horizontal;
  if (rgb.cols == size.width)
    horizontal = rgb;
  else {
    horizontal.create(rgb.rows, size.width, CV_8UC3);
    const auto wx = detail::bicubicWeights(rgb.cols, size.width);
    for (int y = 0; y < rgb.rows; ++y) {
      auto *dst = horizontal.ptr<cv::Vec3b>(y);
      const auto *src = rgb.ptr<cv::Vec3b>(y);
      for (int x = 0; x < size.width; ++x)
        for (int c = 0; c < 3; ++c) {
          int64_t sum = 1 << 21;
          const auto &w = wx[x];
          for (size_t k = 0; k < w.coefficients.size(); ++k)
            sum +=
                static_cast<int64_t>(src[w.start + k][c]) * w.coefficients[k];
          dst[x][c] = rounded(sum);
        }
    }
  }
  if (rgb.rows == size.height)
    return horizontal.clone();
  cv::Mat output(size, CV_8UC3);
  const auto wy = detail::bicubicWeights(rgb.rows, size.height);
  for (int y = 0; y < size.height; ++y) {
    auto *dst = output.ptr<cv::Vec3b>(y);
    const auto &w = wy[y];
    for (int x = 0; x < size.width; ++x)
      for (int c = 0; c < 3; ++c) {
        int64_t sum = 1 << 21;
        for (size_t k = 0; k < w.coefficients.size(); ++k)
          sum += static_cast<int64_t>(horizontal.ptr<cv::Vec3b>(
                     w.start + static_cast<int>(k))[x][c]) *
                 w.coefficients[k];
        dst[x][c] = rounded(sum);
      }
  }
  return output;
}
Tensor preprocess(const cv::Mat &bgr, const std::array<double, 3> &mean,
                  const std::array<double, 3> &stddev) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  auto image = resizePilBicubic(rgb, {1024, 1024});
  std::vector<float> values(3 * 1024 * 1024);
  for (int y = 0; y < 1024; ++y) {
    const auto *row = image.ptr<cv::Vec3b>(y);
    for (int x = 0; x < 1024; ++x)
      for (int c = 0; c < 3; ++c)
        values[c * 1024 * 1024 + y * 1024 + x] =
            detail::normalizePixel(row[x][c], mean[c], stddev[c]);
  }
  return floats({1, 3, 1024, 1024}, std::move(values));
}
cv::Mat restoreLogits(const Tensor &logits, cv::Size size) {
  if (logits.shape != std::vector<int64_t>{1, 1, 1024, 1024})
    throw std::runtime_error("Invalid mask shape");
  cv::Mat src(1024, 1024, CV_32F, const_cast<float *>(logits.values.data())),
      result;
  cv::resize(src, result, size, 0, 0, cv::INTER_LINEAR);
  return result;
}
} // namespace dml

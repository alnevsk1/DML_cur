#include "tracker/sam2_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace {
using Clock = std::chrono::steady_clock;

struct Result {
  cv::Mat mask;
  cv::Rect attemptedCrop;
  const char *route = "full";
  const char *fallback = "";
  int calls = 0;
  double visionMs = 0;
};

cv::Rect makeCrop(cv::Rect box, cv::Size image) {
  const int side = std::max(256, 6 * std::max(box.width, box.height));
  const int width = std::min(side, image.width);
  const int height = std::min(side, image.height);
  const int x = std::clamp(box.x + (box.width - width) / 2, 0, image.width - width);
  const int y = std::clamp(box.y + (box.height - height) / 2, 0, image.height - height);
  return {x, y, width, height};
}

bool nearCropEdge(cv::Rect box, cv::Rect crop, cv::Size image) {
  const int xGuard = crop.width / 8;
  const int yGuard = crop.height / 8;
  return (crop.x > 0 && box.x < xGuard) ||
         (crop.y > 0 && box.y < yGuard) ||
         (crop.br().x < image.width && box.br().x > crop.width - xGuard) ||
         (crop.br().y < image.height && box.br().y > crop.height - yGuard);
}

cv::Point interiorPoint(const cv::Mat &mask, cv::Rect box) {
  cv::Mat padded, distance;
  cv::copyMakeBorder(mask(box), padded, 1, 1, 1, 1,
                     cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0));
  cv::distanceTransform(padded, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
  cv::Point point;
  cv::minMaxLoc(distance, nullptr, nullptr, nullptr, &point);
  return box.tl() + point - cv::Point(1, 1);
}

bool consistent(cv::Rect previous, cv::Rect current, int lastArea, int area) {
  if (area == 0 || area * 4LL < lastArea || area > lastArea * 4LL)
    return false;
  const int margin = std::max({16, previous.width, previous.height});
  const cv::Rect vicinity(previous.x - margin, previous.y - margin,
                          previous.width + 2 * margin, previous.height + 2 * margin);
  const cv::Point center = current.tl() + cv::Point(current.width / 2, current.height / 2);
  return vicinity.contains(center);
}

// Each tracker's spatial features, positions and object pointers stay in its
// own coordinate system. Full-frame memory is held while the crop is active.
class SequenceTracker {
public:
  explicit SequenceTracker(bool roi)
      : full_(engines_, constants_, "mixed"),
        cropped_(roi ? std::make_unique<dml::Sam2Tracker>(engines_, constants_, "mixed")
                     : nullptr) {}

  void reset(int total) {
    total_ = total;
    full_.reset(total);
    crop_ = {};
    previousImage_.release();
    previousMask_.release();
    previousBox_ = {};
    previousArea_ = 0;
    cooldown_ = 0;
  }

  Result process(const cv::Mat &image, const cv::Point2f *click) {
    Result result;
    if (cropped_ && !click && cooldown_ == 0 && previousArea_ > 0) {
      if (crop_.empty())
        startCrop(image.size(), result);
      if (!crop_.empty()) {
        result.attemptedCrop = crop_;
        const auto local = predict(*cropped_, image(crop_), nullptr, result);
        const int area = cv::countNonZero(local);
        const cv::Rect localBox = cv::boundingRect(local);
        const cv::Rect box = localBox + crop_.tl();
        if (area == 0)
          result.fallback = "empty";
        else if (nearCropEdge(localBox, crop_, image.size()))
          result.fallback = "edge";
        else if (!consistent(previousBox_, box, previousArea_, area))
          result.fallback = "geometry";
        else {
          result.route = "roi";
          result.mask = cv::Mat::zeros(image.size(), CV_8UC1);
          local.copyTo(result.mask(crop_));
          remember(image, result.mask, box, area);
          return result;
        }
        crop_ = {};
      }
    }

    result.mask = predict(full_, image, click, result);
    if (cropped_) {
      const int area = cv::countNonZero(result.mask);
      remember(image, result.mask, cv::boundingRect(result.mask), area);
      if (result.fallback[0] != '\0')
        cooldown_ = 2;
      else if (cooldown_ > 0)
        --cooldown_;
    }
    return result;
  }

private:
  static cv::Mat predict(dml::Sam2Tracker &tracker, const cv::Mat &image,
                         const cv::Point2f *click, Result &result) {
    dml::Sam2Timings timings;
    const auto logits = tracker.process(image, click, {}, &timings);
    ++result.calls;
    result.visionMs += timings.graphMs.at("vision_encoder");
    return dml::restoreLogits(logits, image.size()) > 0;
  }

  void remember(const cv::Mat &image, const cv::Mat &mask, cv::Rect box, int area) {
    previousImage_ = image;
    previousMask_ = mask;
    previousBox_ = box;
    previousArea_ = area;
  }

  void startCrop(cv::Size size, Result &result) {
    const auto candidate = makeCrop(previousBox_, size);
    if (candidate.size() == size)
      return;
    const auto box = previousBox_ - candidate.tl();
    if (nearCropEdge(box, candidate, size))
      return;

    result.attemptedCrop = candidate;
    const auto point = interiorPoint(previousMask_, previousBox_) - candidate.tl();
    const cv::Point2f prompt(point);
    cropped_->reset(total_);
    const auto seed = predict(*cropped_, previousImage_(candidate), &prompt, result);
    const cv::Mat reference = previousMask_(candidate);
    const int area = cv::countNonZero(seed);
    const int overlap = cv::countNonZero(seed & reference);
    const int united = area + previousArea_ - overlap;
    if (!seed.at<unsigned char>(point) || 4LL * overlap < united) {
      result.fallback = "seed";
      return;
    }
    crop_ = candidate;
  }

  const fs::path engines_ = "models/engines/sam2.1_tiny_hf";
  const fs::path constants_ = "models/onnx/sam2.1_tiny_hf/constants.json";
  dml::Sam2Tracker full_;
  std::unique_ptr<dml::Sam2Tracker> cropped_;
  int total_ = 0;
  int cooldown_ = 0;
  cv::Rect crop_, previousBox_;
  int previousArea_ = 0;
  cv::Mat previousImage_, previousMask_;
};

std::string maskName(int index) {
  std::ostringstream name;
  name << "frame_" << std::setw(4) << std::setfill('0') << index;
  return name.str();
}
} // namespace

int main(int argc, char **argv) try {
  if (argc != 4 || (std::string(argv[3]) != "full" && std::string(argv[3]) != "roi"))
    throw std::runtime_error("Usage: roi_sequence MANIFEST EMPTY_OUTPUT full|roi");
  cv::FileStorage manifest(argv[1], cv::FileStorage::READ);
  const auto frames = manifest["frames"];
  const int total = static_cast<int>(frames.size());
  constexpr int warmup = 4;
  if (!manifest.isOpened() || total <= warmup)
    throw std::runtime_error("Expected a manifest with at least five frames");
  const cv::Point2f click(static_cast<float>(manifest["click_x"]),
                          static_cast<float>(manifest["click_y"]));
  const fs::path output(argv[2]);
  if (fs::exists(output) && !fs::is_empty(output))
    throw std::runtime_error("Output directory must be empty");
  fs::create_directories(output);

  SequenceTracker tracker(std::string(argv[3]) == "roi");
  auto load = [&](int i) {
    auto image = cv::imread(static_cast<std::string>(frames[i]));
    if (image.empty())
      throw std::runtime_error("Cannot read frame " + std::to_string(i));
    return image;
  };
  const auto expected = load(0).size();
  tracker.reset(total);
  for (int i = 0; i < warmup; ++i) {
    const auto image = load(i);
    if (image.size() != expected)
      throw std::runtime_error("Frame size changed within the sequence");
    tracker.process(image, i == 0 ? &click : nullptr);
  }

  tracker.reset(total);
  std::ofstream csv(output / "timing.csv");
  csv << "index,total_ms,vision_ms,calls,route,fallback,crop_x,crop_y,crop_w,crop_h\n";
  for (int i = 0; i < total; ++i) {
    const auto image = load(i);
    if (image.size() != expected)
      throw std::runtime_error("Frame size changed within the sequence");
    const auto start = Clock::now();
    const auto result = tracker.process(image, i == 0 ? &click : nullptr);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const auto &crop = result.attemptedCrop;
    csv << i << ',' << ms << ',' << result.visionMs << ',' << result.calls << ','
        << result.route << ',' << result.fallback << ',' << crop.x << ',' << crop.y
        << ',' << crop.width << ',' << crop.height << '\n';
    const auto dir = output / maskName(i);
    fs::create_directory(dir);
    if (!cv::imwrite((dir / "mask.png").string(), result.mask))
      throw std::runtime_error("Cannot save mask");
  }
  if (!csv)
    throw std::runtime_error("Cannot save timings");
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}

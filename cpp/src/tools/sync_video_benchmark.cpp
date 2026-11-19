#include "preprocess/preprocess.hpp"
#include "tracker/sam2_tracker.hpp"

#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::array<const char *, 5> graphs{
    "vision_encoder", "pointer_tpos", "memory_attention", "mask_decoder",
    "memory_encoder"};
enum Metric : size_t {
  decode, track, preprocess, vision, pointer, attention, decoder, memory,
  postprocess, render, pipeline, metricCount
};
constexpr std::array<const char *, metricCount> metricNames{
    "decode_ms", "track_ms", "preprocess_ms", "vision_encoder_ms",
    "pointer_tpos_ms", "memory_attention_ms", "mask_decoder_ms",
    "memory_encoder_ms", "postprocess_ms", "render_ms", "pipeline_ms"};

struct Row {
  std::array<double, metricCount> ms{};
  int maskPixels = 0;
  cv::Rect bounds;
  size_t deviceUsed = 0;
};

struct Preview {
  int index;
  cv::Mat bgra;
};

double elapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void cudaCheck(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

size_t deviceBytesUsed() {
  size_t free = 0, total = 0;
  cudaCheck(cudaMemGetInfo(&free, &total));
  return total - free;
}

double mib(size_t bytes) { return bytes / (1024.0 * 1024.0); }

int integerArg(const wchar_t *arg) {
  size_t end = 0;
  const int value = std::stoi(arg, &end);
  if (end != std::wstring_view(arg).size())
    throw std::runtime_error("Expected an integer argument");
  return value;
}

cv::Mat renderPreview(const cv::Mat &frame, const cv::Mat &mask) {
  constexpr int width = 800, height = 450;
  const double scale = std::min(width / static_cast<double>(frame.cols),
                                height / static_cast<double>(frame.rows));
  const cv::Size size(std::max(1, static_cast<int>(frame.cols * scale)),
                      std::max(1, static_cast<int>(frame.rows * scale)));
  cv::Mat display, resizedMask;
  cv::resize(frame, display, size, 0, 0,
             scale < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
  cv::resize(mask, resizedMask, size, 0, 0, cv::INTER_NEAREST);
  const cv::Scalar color(194, 199, 49);
  cv::Mat tint(size, CV_8UC3, color), blended;
  cv::addWeighted(display, 0.6, tint, 0.4, 0, blended);
  blended.copyTo(display, resizedMask);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(resizedMask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  cv::drawContours(display, contours, -1, color, 1);
  cv::Mat bgra;
  cv::cvtColor(display, bgra, cv::COLOR_BGR2BGRA);
  return bgra;
}

std::optional<Row> processFrame(cv::VideoCapture &video, dml::Sam2Tracker &tracker,
                                int index, const cv::Point2f &seed, bool measure,
                                cv::Mat &preview) {
  Row row;
  cv::Mat frame;
  const auto begin = Clock::now();
  if (!video.read(frame))
    return std::nullopt;
  if (frame.type() != CV_8UC3)
    throw std::runtime_error("Expected a BGR frame at index " + std::to_string(index));
  row.ms[decode] = elapsedMs(begin);
  if (index == 0 &&
      (seed.x < 0 || seed.x >= frame.cols || seed.y < 0 || seed.y >= frame.rows))
    throw std::runtime_error("Click must be inside the decoded frame");

  dml::Sam2Timings timing;
  auto start = Clock::now();
  const auto logits = tracker.process(frame, index == 0 ? &seed : nullptr, {},
                                      measure ? &timing : nullptr);
  row.ms[track] = elapsedMs(start);
  start = Clock::now();
  cv::Mat mask = dml::restoreLogits(logits, frame.size()) > 0;
  row.ms[postprocess] = elapsedMs(start);
  start = Clock::now();
  preview = renderPreview(frame, mask);
  row.ms[render] = elapsedMs(start);
  row.ms[pipeline] = elapsedMs(begin);

  if (measure) {
    row.ms[preprocess] = timing.preprocessMs;
    for (size_t i = 0; i < graphs.size(); ++i) {
      const auto graph = timing.graphMs.find(graphs[i]);
      if (graph != timing.graphMs.end())
        row.ms[vision + i] = graph->second;
      else if (index != 0 || (vision + i != pointer && vision + i != attention))
        throw std::runtime_error("Missing timing for " + std::string(graphs[i]));
    }
    row.deviceUsed = deviceBytesUsed();
    row.maskPixels = cv::countNonZero(mask);
    row.bounds = cv::boundingRect(mask);
  }
  return row;
}

void saveCsv(const fs::path &path, const std::vector<Row> &rows) {
  std::ofstream out(path);
  out << "frame";
  for (const auto *name : metricNames)
    out << ',' << name;
  out << ",mask_pixels,bbox_x,bbox_y,bbox_w,bbox_h,device_used_mib\n";
  out << std::fixed << std::setprecision(4);
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto &row = rows[i];
    out << i;
    for (size_t m = 0; m < metricCount; ++m) {
      out << ',';
      if (i != 0 || (m != pointer && m != attention))
        out << row.ms[m];
    }
    out << ',' << row.maskPixels << ',' << row.bounds.x << ',' << row.bounds.y
        << ',' << row.bounds.width << ',' << row.bounds.height << ','
        << mib(row.deviceUsed) << '\n';
  }
  out.flush();
  if (!out)
    throw std::runtime_error("Cannot write frame timings");
}

void saveStats(cv::FileStorage &out, const std::vector<Row> &rows) {
  double totalMs = 0;
  size_t peak = 0;
  int emptyMasks = 0;
  for (const auto &row : rows) {
    totalMs += row.ms[pipeline];
    peak = std::max(peak, row.deviceUsed);
    emptyMasks += row.maskPixels == 0;
  }
  out << "pipeline_fps" << 1000 * (rows.size() - 1) / (totalMs - rows[0].ms[pipeline])
      << "pipeline_total_seconds" << totalMs / 1000
      << "seed_pipeline_ms" << rows[0].ms[pipeline]
      << "seed_mask_pixels" << rows[0].maskPixels
      << "empty_mask_frames" << emptyMasks
      << "device_peak_sampled_mib" << mib(peak)
      << "propagation_timings" << "{";
  for (size_t m = 0; m < metricCount; ++m) {
    std::vector<double> values;
    values.reserve(rows.size() - 1);
    for (size_t i = 1; i < rows.size(); ++i)
      values.push_back(rows[i].ms[m]);
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double fraction) {
      return values[static_cast<size_t>(std::ceil(fraction * values.size())) - 1];
    };
    out << metricNames[m] << "{" << "p50" << percentile(0.5)
        << "p95" << percentile(0.95) << "}";
  }
  out << "}";
}
} // namespace

int wmain(int argc, wchar_t **argv) try {
  if (argc < 6 || argc > 8)
    throw std::runtime_error(
        "Usage: sync_video_benchmark VIDEO X Y fp32|mixed OUTPUT [WARMUP [LIMIT]]");
  const fs::path path = fs::absolute(argv[1]);
  const cv::Point2f seed(static_cast<float>(integerArg(argv[2])),
                         static_cast<float>(integerArg(argv[3])));
  const std::string precision = fs::path(argv[4]).u8string();
  if (precision != "fp32" && precision != "mixed")
    throw std::runtime_error("Choose fp32 or mixed");
  const fs::path output = argv[5];
  const int warmup = argc >= 7 ? integerArg(argv[6]) : 32;
  const int limit = argc == 8 ? integerArg(argv[7]) : std::numeric_limits<int>::max();
  if (limit < 2)
    throw std::runtime_error("Frame limit needs at least two frames");
  if (warmup < 2)
    throw std::runtime_error("Warm-up needs at least two frames");
  if (fs::exists(output) && !fs::is_empty(output))
    throw std::runtime_error("Output directory must be empty");
  fs::create_directories(output);

  cv::VideoCapture video(path.u8string());
  if (!video.isOpened())
    throw std::runtime_error("Cannot open video");
  const double count = video.get(cv::CAP_PROP_FRAME_COUNT);
  const double fps = video.get(cv::CAP_PROP_FPS);
  if (!std::isfinite(count) || count < 1 ||
      count > std::numeric_limits<int>::max() || !std::isfinite(fps) || fps <= 0)
    throw std::runtime_error("Invalid video metadata");
  const int sourceFrames = static_cast<int>(count);
  const int expected = std::min(sourceFrames, limit);
  const int previewInterval = std::clamp(expected / 4, 1, 120);
  // OpenCV can estimate the count from duration. Long clips already use the
  // capped pointer normalization in Sam2Tracker, so let decoding determine EOF.
  const int trackerBudget = sourceFrames >= 16
                                ? std::numeric_limits<int>::max()
                                : sourceFrames;
  const std::string backend = video.getBackendName();

  cudaDeviceProp gpu{};
  cudaCheck(cudaGetDeviceProperties(&gpu, 0));
  const size_t baselineUsed = deviceBytesUsed();
  const fs::path root = fs::current_path();
  const auto loadStart = Clock::now();
  dml::Sam2Tracker tracker(root / "models/engines/sam2.1_tiny_hf",
                           root / "models/onnx/sam2.1_tiny_hf/constants.json",
                           precision);
  const double loadMs = elapsedMs(loadStart);
  tracker.reset(trackerBudget);
  cv::Mat preview;
  for (int i = 0; i < warmup; ++i)
    if (!processFrame(video, tracker, i, seed, false, preview))
      throw std::runtime_error("Not enough decoded frames for warm-up");
  video.release();
  if (!video.open(path.u8string()))
    throw std::runtime_error("Cannot reopen video after warm-up");
  tracker.reset(trackerBudget);

  std::vector<Row> rows;
  rows.reserve(expected);
  std::vector<Preview> previews;
  const auto wallStart = Clock::now();
  for (int i = 0; i < limit; ++i) {
    auto row = processFrame(video, tracker, i, seed, true, preview);
    if (!row)
      break;
    rows.push_back(*row);
    if (i % previewInterval == 0)
      previews.push_back({i, preview});
    if ((i + 1) % 30 == 0)
      std::cerr << precision << "  " << i + 1 << " frames processed\n";
  }
  const double wallSeconds = elapsedMs(wallStart) / 1000;
  const int frames = static_cast<int>(rows.size());
  if (frames < 2)
    throw std::runtime_error("Not enough decoded frames for statistics");
  if (previews.back().index != frames - 1)
    previews.push_back({frames - 1, preview});

  saveCsv(output / "frames.csv", rows);
  for (const auto &sample : previews) {
    std::ostringstream name;
    name << "preview_" << std::setfill('0') << std::setw(4) << sample.index
         << ".jpg";
    cv::Mat bgr;
    cv::cvtColor(sample.bgra, bgr, cv::COLOR_BGRA2BGR);
    if (!cv::imwrite((output / name.str()).u8string(), bgr))
      throw std::runtime_error("Cannot write preview");
  }
  cv::FileStorage report((output / "summary.json").u8string(),
                         cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
  if (!report.isOpened())
    throw std::runtime_error("Cannot write summary");
  report << "video" << path.u8string() << "backend" << backend
         << "source_fps" << fps << "reported_source_frames" << sourceFrames
         << "frames" << frames << "warmup_frames" << warmup
         << "click_x" << seed.x << "click_y" << seed.y << "precision" << precision
         << "gpu" << gpu.name << "opencv" << CV_VERSION
         << "tensorrt_headers" << static_cast<int>(NV_TENSORRT_VERSION)
         << "device_before_model_mib" << mib(baselineUsed)
         << "engine_load_ms" << loadMs << "wall_seconds" << wallSeconds;
  saveStats(report, rows);
  std::cout << precision << " finished " << frames << " frames in "
            << wallSeconds << " s\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ERROR: " << error.what() << '\n';
  return 1;
}

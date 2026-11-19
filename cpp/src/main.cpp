#include "preprocess/preprocess.hpp"
#include "tracker/sam2_tracker.hpp"
#include <NvInferVersion.h>
#include <chrono>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <stdexcept>
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace {
struct Mouse {
  bool pending = false;
  cv::Point2f point;
  double scale = 1;
};
void mouseCallback(int event, int x, int y, int, void *user) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    auto &m = *static_cast<Mouse *>(user);
    m.point = {static_cast<float>(x / m.scale),
               static_cast<float>(y / m.scale)};
    m.pending = true;
  }
}
void freshOutput(const fs::path &out) {
  if (fs::exists(out) && !fs::is_empty(out))
    throw std::runtime_error("Output directory must be empty: " + out.string());
  fs::create_directories(out);
}
void recordRun(const fs::path &out, int argc, char **argv) {
  cv::FileStorage record((out / "run.json").string(),
                         cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
  if (!record.isOpened())
    throw std::runtime_error("Cannot write run metadata");
  record << "argv" << "[";
  for (int i = 0; i < argc; ++i)
    record << argv[i];
  record << "]";
  record << "opencv" << CV_VERSION << "tensorrt_headers" << NV_TENSORRT_VERSION;
  record << "msvc" << _MSC_VER;
  cudaDeviceProp device{};
  if (cudaGetDeviceProperties(&device, 0) != cudaSuccess)
    throw std::runtime_error("Cannot query CUDA device");
  record << "gpu" << device.name;
  record << "timing_scope"
         << "diagnostic including trace writes; not a benchmark";
}
std::string frameName(int index) {
  std::ostringstream s;
  s << "frame_" << std::setfill('0') << std::setw(4) << index;
  return s.str();
}
} // namespace
int main(int argc, char **argv) try {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help") {
      std::cout
          << "uav_tracker --video FILE [--precision fp32|fp16|mixed] [--root "
             "DIR]\n"
          << "uav_tracker --sequence manifest.json --output EMPTY_DIR "
             "[--precision fp32|fp16|mixed]\n"
          << "uav_tracker --cases DIR --output EMPTY_DIR [--precision "
             "fp32|fp16|mixed]\n"
          << "uav_tracker --preprocess DIR --output EMPTY_DIR\n"
          << "Space: pause/play; R: reset; Esc: exit. Click on paused frame to "
             "select.\n";
      return 0;
    }
    if (i + 1 >= argc)
      throw std::runtime_error("Expected value after option");
    const std::string key = argv[i];
    if (key != "--video" && key != "--precision" && key != "--root" &&
        key != "--sequence" && key != "--output" && key != "--cases" &&
        key != "--preprocess" && key != "--part" && key != "--smoke")
      throw std::runtime_error("Unknown option: " + key);
    if (!args.emplace(key, argv[i + 1]).second)
      throw std::runtime_error("Duplicate option: " + key);
  }
  int modes = args.count("--video") + args.count("--sequence") +
              args.count("--cases") + args.count("--preprocess");
  if (modes != 1)
    throw std::runtime_error("Specify exactly one of --video, --sequence, "
                             "--cases, --preprocess (see --help)");
  const fs::path root =
      args.count("--root") ? fs::path(args.at("--root")) : fs::current_path();
  const fs::path bundle = root / "models/onnx/sam2.1_tiny_hf";
  const fs::path engines = root / "models/engines/sam2.1_tiny_hf";
  const std::string precision =
      args.count("--precision") ? args.at("--precision") : "fp32";
  if (precision != "fp32" && precision != "fp16" && precision != "mixed")
    throw std::runtime_error("Invalid precision");
  if (precision == "fp16")
    std::cerr << "FP16 experimental: recorded video parity failure; prefer "
                 "fp32 or validated mixed mode.\n";
  if (args.count("--preprocess")) {
    fs::path out = args.at("--output");
    freshOutput(out);
    recordRun(out, argc, argv);
    std::array<double, 3> mean{}, stddev{};
    cv::FileStorage constants((bundle / "constants.json").string(),
                              cv::FileStorage::READ);
    if (!constants.isOpened())
      throw std::runtime_error("Cannot read constants");
    for (int c = 0; c < 3; ++c) {
      mean[c] = static_cast<double>(constants["image_mean"][c]);
      stddev[c] = static_cast<double>(constants["image_std"][c]);
    }
    size_t count = 0;
    for (const auto &entry : fs::directory_iterator(args.at("--preprocess")))
      if (entry.path().extension() == ".png") {
        auto bgr = cv::imread(entry.path().string());
        if (bgr.empty())
          throw std::runtime_error("Cannot read preprocessing sample");
        auto rgb = cv::Mat();
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        auto resized = dml::resizePilBicubic(rgb, {1024, 1024});
        cv::cvtColor(resized, resized, cv::COLOR_RGB2BGR);
        if (!cv::imwrite((out / entry.path().filename()).string(), resized))
          throw std::runtime_error("Cannot write resized sample");
        dml::saveRaw(out / (entry.path().stem().string() + ".bin"),
                     dml::preprocess(bgr, mean, stddev));
        ++count;
      }
    if (!count)
      throw std::runtime_error("No PNG samples");
    return 0;
  }
  if (args.count("--cases")) {
    if (args.count("--part") && args.at("--part") != "mask_decoder" &&
        args.at("--part") != "vision_encoder" &&
        args.at("--part") != "memory_encoder" &&
        args.at("--part") != "memory_attention" &&
        args.at("--part") != "pointer_tpos")
      throw std::runtime_error("Unknown part");
    fs::path out = args.at("--output");
    freshOutput(out);
    recordRun(out, argc, argv);
    for (const std::string part :
         {"vision_encoder", "mask_decoder", "memory_encoder",
          "memory_attention", "pointer_tpos"}) {
      if (args.count("--part") && args.at("--part") != part)
        continue;
      dml::TrtEngine engine(
          engines /
          (part + "_" +
           (precision == "mixed" ? (part == "mask_decoder" ? "fp32" : "fp16")
                                 : precision) +
           ".engine"));
      auto feed = engine.inputLayout();
      for (auto &[name, t] : feed)
        t = dml::readRaw(fs::path(args.at("--cases")) / part / (name + ".bin"),
                         std::move(t));
      for (const auto &[name, t] : engine.run(feed))
        dml::saveRaw(out / part / (name + ".bin"), t);
      std::cout << "case OK: " << part << '\n';
    }
    return 0;
  }
  dml::Sam2Tracker tracker(engines, bundle / "constants.json", precision);
  if (args.count("--sequence")) {
    cv::FileStorage manifest(args.at("--sequence"), cv::FileStorage::READ);
    if (!manifest.isOpened())
      throw std::runtime_error("Cannot read manifest");
    const auto frames = manifest["frames"];
    int total = static_cast<int>(frames.size());
    tracker.reset(total);
    cv::Point2f click(static_cast<float>(manifest["click_x"]),
                      static_cast<float>(manifest["click_y"]));
    fs::path out = args.at("--output");
    freshOutput(out);
    recordRun(out, argc, argv);
    std::ofstream metrics(out / "timing.csv");
    metrics << "frame,total_ms\n";
    int i = 0;
    for (const auto &entry : frames) {
      auto image = cv::imread(static_cast<std::string>(entry));
      if (image.empty())
        throw std::runtime_error("Cannot read sequence image");
      auto start = Clock::now();
      auto dir = out / frameName(i);
      auto logits = tracker.process(image, i == 0 ? &click : nullptr, dir);
      auto restored = dml::restoreLogits(logits, image.size());
      auto mask = restored > 0;
      if (!cv::imwrite((dir / "mask.png").string(), mask))
        throw std::runtime_error("Cannot save mask");
      metrics << i << ','
              << std::chrono::duration<double, std::milli>(Clock::now() - start)
                     .count()
              << '\n';
      std::cout << "frame " << i++ << " complete\n";
    }
    if (!metrics.good())
      throw std::runtime_error("Cannot write timing");
    return 0;
  }
  const bool smoke = args.count("--smoke") != 0;
  cv::Point2f smokeClick;
  fs::path smokeOut;
  int smokeStep = 0;
  if (smoke) {
    cv::FileStorage config(args.at("--smoke"), cv::FileStorage::READ);
    if (!config.isOpened())
      throw std::runtime_error("Cannot open smoke manifest");
    smokeClick = {static_cast<float>(config["click_x"]),
                  static_cast<float>(config["click_y"])};
    smokeOut = args.at("--output");
    freshOutput(smokeOut);
    recordRun(smokeOut, argc, argv);
  }
  cv::VideoCapture video(args.at("--video"));
  if (!video.isOpened())
    throw std::runtime_error("Cannot open video");
  const int total = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
  if (total < 1)
    throw std::runtime_error("Video must provide a positive frame count");
  tracker.reset(total);
  cv::Mat frame, logits;
  if (!video.read(frame))
    throw std::runtime_error("Empty video");
  Mouse mouse;
  bool paused = true, advance = false;
  int index = 0;
  double latency = 0;
  const std::string window = "DML_cur | SAM 2.1-tiny";
  cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(window, mouseCallback, &mouse);
  while (true) {
    if (advance) {
      if (!video.read(frame))
        break;
      ++index;
      advance = false;
      if (tracker.active()) {
        auto start = Clock::now();
        logits = dml::restoreLogits(tracker.process(frame), frame.size());
        latency =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
      }
    }
    if (mouse.pending) {
      if (paused && !tracker.active()) {
        tracker.reset(total - index);
        auto start = Clock::now();
        logits = dml::restoreLogits(tracker.process(frame, &mouse.point),
                                    frame.size());
        latency =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
      }
      mouse.pending = false;
    }
    auto display = frame.clone();
    if (!logits.empty()) {
      cv::Mat mask = logits > 0, tint = display.clone(), blended;
      tint.setTo(cv::Scalar(40, 220, 40), mask);
      cv::addWeighted(display, 0.6, tint, 0.4, 0, blended);
      display = blended;
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_SIMPLE);
      cv::drawContours(display, contours, -1, {0, 255, 0}, 1);
    }
    mouse.scale = std::min({1.0, 1280.0 / display.cols, 760.0 / display.rows});
    cv::resize(display, display, {}, mouse.scale, mouse.scale);
    std::ostringstream status;
    status << precision << " | frame " << index << " | " << std::fixed
           << std::setprecision(1) << latency << " ms | "
           << (latency > 0 ? 1000.0 / latency : 0) << " proc FPS";
    cv::putText(display, status.str(), {12, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                {255, 255, 255}, 1, cv::LINE_AA);
    cv::putText(display,
                paused
                    ? "PAUSED | click to select after R | Space play | Esc exit"
                    : "PLAY | Space pause | R reset",
                {12, 48}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1,
                cv::LINE_AA);
    cv::imshow(window, display);
    int key = cv::waitKeyEx(20);
    if (smoke) {
      // Script application events internally; this does not inject OS
      // mouse/keyboard input.
      if (smokeStep == 0 || smokeStep == 3) {
        if (smokeStep == 3 && tracker.active())
          throw std::runtime_error("Reset failed");
        mouseCallback(cv::EVENT_LBUTTONDOWN,
                      static_cast<int>(smokeClick.x * mouse.scale),
                      static_cast<int>(smokeClick.y * mouse.scale), 0, &mouse);
        key = -1;
      } else if (smokeStep == 1) {
        if (!tracker.active() || logits.empty() ||
            cv::countNonZero(logits > 0) == 0)
          throw std::runtime_error("Click initialization failed");
        key = ' ';
      } else if (smokeStep == 2) {
        if (index != 1 || paused)
          throw std::runtime_error("Playback failed");
        key = 'r';
      } else if (smokeStep == 4) {
        if (!tracker.active() || logits.empty())
          throw std::runtime_error("Reselection failed");
        fs::create_directories(root / "reports/figures");
        if (!cv::imwrite(
                (root / "reports/figures/sam2_cpp_sync_smoke.png").string(),
                display))
          throw std::runtime_error("Cannot save rendered smoke frame");
        cv::FileStorage result((smokeOut / "report.json").string(),
                               cv::FileStorage::WRITE |
                                   cv::FileStorage::FORMAT_JSON);
        result << "passed" << 1 << "video_frames" << total << "processed_frame"
               << index;
        result
            << "scope"
            << "Scripted native HighGUI render/click/play/reset/reselection; "
               "physical input not verified";
        key = 27;
      }
      ++smokeStep;
    }
    if (key == 27 || cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1)
      break;
    if (key == 'r' || key == 'R') {
      tracker.reset(total - index);
      logits.release();
      paused = true;
      latency = 0;
    }
    if (key == ' ')
      paused = !paused;
    advance = !paused;
  }
  cv::destroyAllWindows();
  return 0;
} catch (const std::exception &error) {
  std::cerr << "ERROR: " << error.what() << '\n';
  return 1;
}

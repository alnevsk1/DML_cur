#pragma once

#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

namespace dml::ui {
enum class Precision { fp32, mixed };
enum class Action { open, select, advance, seek, reset, precision };

struct Command {
  Action action;
  std::filesystem::path path;
  cv::Point2f point;
  int index = 0;
  Precision precision = Precision::fp32;
  bool track = false;
  bool realtime = false;
};

struct Frame {
  std::filesystem::path path;
  // Image data is shared with the window for reading only.
  cv::Mat bgr;
  cv::Mat mask;
  int index = 0;
  int count = 0;
  double fps = 0;
  double latencyMs = 0;
  int skipped = 0;
  bool tracking = false;
  bool eof = false;
  std::optional<cv::Point2f> seed;
  std::string error;
};

class Session {
public:
  explicit Session(std::filesystem::path root);
  ~Session();

  void submit(Command command);
  std::optional<Frame> takeResult();
  std::optional<Frame> takeTrackedResult();

private:
  struct TrackJob {
    Frame frame;
    std::optional<cv::Point2f> point;
    Precision precision;
    std::uint64_t revision;
    std::uint64_t trackRevision;
    bool realtime;
  };

  void stop();
  void runCapture();
  void runTracker();

  const std::filesystem::path root_;
  std::mutex mutex_;
  std::condition_variable ready_;
  bool closing_ = false;
  bool trackFailed_ = false;
  std::uint64_t revision_ = 0;
  std::uint64_t trackRevision_ = 0;
  std::optional<Command> pending_;
  std::optional<Frame> result_;
  // One waiting frame: real-time playback replaces it as new frames arrive.
  std::optional<TrackJob> pendingTrack_;
  std::optional<Frame> trackedResult_;
  std::thread captureWorker_;
  std::thread trackerWorker_;
};
} // namespace dml::ui

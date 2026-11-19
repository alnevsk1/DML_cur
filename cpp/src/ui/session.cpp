#include "session.hpp"

#include "preprocess/preprocess.hpp"
#include "tracker/sam2_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <opencv2/videoio.hpp>

namespace dml::ui {
namespace {
using Clock = std::chrono::steady_clock;

class Playback {
public:
  Frame run(const Command &command) {
    frame_.error.clear();
    try {
      switch (command.action) {
      case Action::open:
        open(command.path);
        break;
      case Action::advance:
        if (!frame_.bgr.empty() && !frame_.eof)
          readAt(frame_.index + 1, true);
        break;
      case Action::seek:
      case Action::reset:
      case Action::precision:
        if (!frame_.bgr.empty() && command.index != frame_.index)
          readAt(command.index);
        break;
      case Action::select:
        if (frame_.bgr.empty() || command.index != frame_.index)
          throw std::runtime_error("Выбранный кадр больше недоступен.");
        break;
      }
    } catch (const std::exception &error) {
      if (command.action == Action::open) {
        video_.release();
        nextIndex_ = -1;
        frame_ = {};
      }
      frame_.error = error.what();
    }
    return frame_;
  }

private:
  void open(const std::filesystem::path &path) {
    frame_ = {};
    video_.release();
    nextIndex_ = -1;
    if (!std::filesystem::is_regular_file(path) || !video_.open(path.u8string()))
      throw std::runtime_error("Не удалось открыть видео. " + path.u8string());
    const double count = video_.get(cv::CAP_PROP_FRAME_COUNT);
    const double fps = video_.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(count) || count < 1 ||
        count > std::numeric_limits<int>::max() || !std::isfinite(fps) ||
        fps <= 0 || count / fps > std::numeric_limits<long long>::max() / 2000.0)
      throw std::runtime_error("Не удалось определить число кадров или частоту кадров видео.");
    frame_.path = path;
    frame_.count = static_cast<int>(count);
    frame_.fps = fps;
    nextIndex_ = 0;
    readAt(0);
  }

  void readAt(int index, bool allowEnd = false) {
    const int next = std::exchange(nextIndex_, -1);
    if (next != index && !video_.set(cv::CAP_PROP_POS_FRAMES, index))
      throw std::runtime_error("Не удалось перейти к кадру " + std::to_string(index + 1));
    cv::Mat bgr;
    if (!video_.read(bgr)) {
      if (allowEnd && next == index) {
        frame_.eof = true;
        frame_.count = frame_.index + 1;
        return;
      }
      throw std::runtime_error("Не удалось декодировать кадр " + std::to_string(index + 1));
    }
    if (bgr.type() != CV_8UC3)
      throw std::runtime_error("Ожидался видеокадр в формате BGR.");
    if (next != index &&
        std::abs(video_.get(cv::CAP_PROP_POS_FRAMES) - (index + 1.0)) > 0.5)
      throw std::runtime_error("Видеодекодер не дошёл до нужного кадра.");
    frame_.bgr = std::move(bgr);
    frame_.index = index;
    frame_.count = std::max(frame_.count, index + 1);
    frame_.eof = false;
    nextIndex_ = index + 1;
  }

  cv::VideoCapture video_;
  Frame frame_;
  int nextIndex_ = -1;
};

class Track {
public:
  explicit Track(const std::filesystem::path &root)
      : engines_(root / "models/engines/sam2.1_tiny_hf"),
        constants_(root / "models/onnx/sam2.1_tiny_hf/constants.json") {}

  void clear() {
    if (tracker_)
      tracker_->reset(1);
    lastIndex_ = -1;
    skipped_ = 0;
  }

  void run(Frame &frame, const std::optional<cv::Point2f> &point,
           Precision precision) {
    if (point) {
      if (precision != precision_) {
        tracker_.reset();
        precision_ = precision;
      }
      if (!tracker_) {
        try {
          tracker_ = std::make_unique<Sam2Tracker>(
              engines_, constants_, precision == Precision::mixed ? "mixed" : "fp32");
        } catch (const std::exception &error) {
          throw std::runtime_error(std::string("Не удалось загрузить SAM 2.1-tiny. ") +
                                   error.what());
        }
      }
      const int remaining = frame.count - frame.index;
      // OpenCV can estimate the count. Long tracks already use capped pointer
      // normalization, so allow frames beyond that estimate.
      tracker_->reset(remaining >= 16 ? std::numeric_limits<int>::max()
                                      : remaining);
    } else if (lastIndex_ < 0 || frame.index <= lastIndex_) {
      throw std::runtime_error("Для трекинга кадры должны поступать по порядку.");
    }
    const auto start = Clock::now();
    frame.mask = restoreLogits(
                     tracker_->process(frame.bgr, point ? &*point : nullptr),
                     frame.bgr.size()) > 0;
    frame.latencyMs =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (!point)
      skipped_ += frame.index - lastIndex_ - 1;
    lastIndex_ = frame.index;
    frame.skipped = skipped_;
    frame.tracking = true;
    frame.seed = point;
  }

private:
  const std::filesystem::path engines_;
  const std::filesystem::path constants_;
  std::unique_ptr<Sam2Tracker> tracker_;
  Precision precision_ = Precision::fp32;
  int lastIndex_ = -1;
  int skipped_ = 0;
};
} // namespace

Session::Session(std::filesystem::path root) : root_(std::move(root)) {
  captureWorker_ = std::thread(&Session::runCapture, this);
  try {
    trackerWorker_ = std::thread(&Session::runTracker, this);
  } catch (...) {
    stop();
    throw;
  }
}

Session::~Session() { stop(); }

void Session::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closing_ = true;
  }
  ready_.notify_all();
  if (captureWorker_.joinable())
    captureWorker_.join();
  if (trackerWorker_.joinable())
    trackerWorker_.join();
}

void Session::submit(Command command) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++revision_;
    if (command.action != Action::advance) {
      ++trackRevision_;
      pendingTrack_.reset();
      trackedResult_.reset();
      trackFailed_ = false;
    }
    result_.reset();
    pending_ = std::move(command);
  }
  ready_.notify_all();
}

std::optional<Frame> Session::takeResult() {
  std::lock_guard<std::mutex> lock(mutex_);
  return std::exchange(result_, std::nullopt);
}

std::optional<Frame> Session::takeTrackedResult() {
  std::lock_guard<std::mutex> lock(mutex_);
  return std::exchange(trackedResult_, std::nullopt);
}

void Session::runCapture() {
  Playback playback;
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    ready_.wait(lock, [this] { return closing_ || pending_.has_value(); });
    if (closing_)
      return;
    Command command = std::move(*pending_);
    pending_.reset();
    const auto revision = revision_;
    lock.unlock();
    auto frame = playback.run(command);
    lock.lock();
    if (revision != revision_ || closing_)
      continue;

    const bool select = command.action == Action::select;
    const bool track = !trackFailed_ && frame.error.empty() &&
                       (select || (command.track && !frame.eof));
    if (track) {
      pendingTrack_ = TrackJob{
          frame, select ? std::optional<cv::Point2f>(command.point) : std::nullopt,
          command.precision, revision, trackRevision_, command.realtime};
      ready_.notify_all();
    }
    if (!track || command.realtime)
      result_ = std::move(frame);
  }
}

void Session::runTracker() {
  Track tracker(root_);
  std::uint64_t revision = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    ready_.wait(lock, [this, &revision] {
      return closing_ || revision != trackRevision_ || pendingTrack_.has_value();
    });
    if (closing_)
      return;
    const bool reset = revision != trackRevision_;
    revision = trackRevision_;
    auto job = std::exchange(pendingTrack_, std::nullopt);
    lock.unlock();
    if (reset)
      tracker.clear();
    if (job) {
      try {
        tracker.run(job->frame, job->point, job->precision);
      } catch (const std::exception &error) {
        tracker.clear();
        job->frame.error = error.what();
      }
    }
    lock.lock();
    if (!job || job->trackRevision != trackRevision_ || closing_)
      continue;
    if (!job->frame.error.empty()) {
      trackFailed_ = true;
      pendingTrack_.reset();
    }
    if (job->realtime)
      trackedResult_ = std::move(job->frame);
    else if (job->revision == revision_)
      result_ = std::move(job->frame);
  }
}
} // namespace dml::ui

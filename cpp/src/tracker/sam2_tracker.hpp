#pragma once
#include "preprocess/preprocess.hpp"
#include <array>
#include <deque>
#include <map>
#include <opencv2/core.hpp>
#include <vector>
namespace dml {
struct Sam2Timings {
  // Host time spent queuing preprocessing on the vision stream.
  double preprocessMs = 0;
  // Wall time of graph calls excluding preprocessing submission. Vision also
  // includes any queued preprocessing that is still running on the GPU.
  std::map<std::string, double> graphMs;
};

class Sam2Tracker {
public:
  Sam2Tracker(const std::filesystem::path &engines,
              const std::filesystem::path &constants,
              const std::string &precision);
  void reset(int totalFrames);
  bool active() const { return index_ > 0; }
  Tensor process(const cv::Mat &bgr, const cv::Point2f *click = nullptr,
                 const std::filesystem::path &trace = {},
                 Sam2Timings *timings = nullptr);
  const std::array<double, 3> &mean() const { return mean_; }
  const std::array<double, 3> &stddev() const { return std_; }

private:
  struct Memory {
    int index;
    Tensor tokens, pos;
  };
  struct Pointer {
    int index;
    Tensor value;
  };
  std::map<std::string, std::unique_ptr<TrtEngine>> engines_;
  std::array<double, 3> mean_{}, std_{};
  std::unique_ptr<CudaPreprocessor> preprocessor_;
  std::array<std::array<float, 64>, 7> temporal_{};
  std::vector<float> memoryScratch_, positionScratch_;
  Memory seed_{};
  Pointer seedPointer_{};
  std::deque<Memory> recent_;
  std::deque<Pointer> pointers_;
  int index_ = 0;
  int totalFrames_ = 0;
};
} // namespace dml

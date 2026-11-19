#include "sam2_tracker.hpp"
#include <algorithm>
#include <chrono>
#include <opencv2/core/persistence.hpp>
#include <stdexcept>
namespace dml {
namespace detail {
void enqueueVisionSequence(const float *input, float *output, cudaStream_t stream);
}
namespace {
Tensor sequence(const Tensor &input) {
  if (input.shape != std::vector<int64_t>{1, 256, 64, 64} ||
      input.type != nvinfer1::DataType::kFLOAT || !input.deviceData ||
      input.deviceTransform)
    throw std::runtime_error("Expected a device FP32 vision feature");
  Tensor view;
  view.shape = {4096, 1, 256};
  view.deviceData = input.deviceData;
  view.deviceTransform = detail::enqueueVisionSequence;
  return view;
}
} // namespace
Sam2Tracker::Sam2Tracker(const std::filesystem::path &engines,
                         const std::filesystem::path &constants,
                         const std::string &precision) {
  if (precision != "fp32" && precision != "fp16" && precision != "mixed")
    throw std::runtime_error("Precision must be fp32, fp16 or mixed");
  cv::FileStorage fs(constants.string(),
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (!fs.isOpened())
    throw std::runtime_error("Cannot read constants");
  for (const auto &spec :
       std::map<std::string, int>{{"image_size", 1024},
                                  {"feat_size", 64},
                                  {"hidden_dim", 256},
                                  {"mem_dim", 64},
                                  {"num_maskmem", 7},
                                  {"max_object_pointers", 16}})
    if (static_cast<int>(fs[spec.first]) != spec.second)
      throw std::runtime_error("Unsupported constants: " + spec.first);
  if (fs["image_mean"].size() != 3 || fs["image_std"].size() != 3 ||
      fs["memory_temporal_positional_encoding"].size() != 7)
    throw std::runtime_error("Malformed constants");
  for (int c = 0; c < 3; ++c) {
    mean_[c] = static_cast<double>(fs["image_mean"][c]);
    std_[c] = static_cast<double>(fs["image_std"][c]);
  }
  for (int t = 0; t < 7; ++t) {
    if (fs["memory_temporal_positional_encoding"][t].size() != 64)
      throw std::runtime_error("Invalid temporal PE");
    for (int c = 0; c < 64; ++c)
      temporal_[t][c] =
          static_cast<float>(fs["memory_temporal_positional_encoding"][t][c]);
  }
  preprocessor_ = std::make_unique<CudaPreprocessor>(mean_, std_);
  for (const std::string part :
       {"vision_encoder", "mask_decoder", "memory_encoder", "memory_attention",
        "pointer_tpos"})
    engines_[part] = std::make_unique<TrtEngine>(
        engines /
        (part + "_" +
         (precision == "mixed" ? (part == "mask_decoder" ? "fp32" : "fp16")
                               : precision) +
         ".engine"));
}
void Sam2Tracker::reset(int totalFrames) {
  if (totalFrames < 1)
    throw std::runtime_error("Frame count must be known and positive");
  index_ = 0;
  totalFrames_ = totalFrames;
  recent_.clear();
  pointers_.clear();
  seed_ = {};
  seedPointer_ = {};
}
Tensor Sam2Tracker::process(const cv::Mat &bgr, const cv::Point2f *click,
                            const std::filesystem::path &trace,
                            Sam2Timings *timings) {
  if (totalFrames_ <= index_)
    throw std::runtime_error("Frame count exceeded; reset tracker");
  if ((index_ == 0) != (click != nullptr))
    throw std::runtime_error(
        "Initial frame needs one click; reset before choosing again");
  using Clock = std::chrono::steady_clock;
  const auto elapsedMs = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
  };
  if (timings)
    timings->graphMs.clear();
  const auto saveGraph = [&](const std::string &part, const Tensors &result) {
    if (!trace.empty())
      for (const auto &[name, t] : result)
        saveRaw(trace / (part + "__" + name + ".bin"), t);
  };
  const auto run = [&](const std::string &part, const Tensors &feed,
                       TrtEngine::DeviceOutputs deviceOutputs = {}) {
    const auto start = timings ? Clock::now() : Clock::time_point{};
    auto result = engines_.at(part)->run(feed, deviceOutputs);
    if (timings)
      timings->graphMs.emplace(part, elapsedMs(start));
    saveGraph(part, result);
    return result;
  };
  constexpr int imageSize = detail::imageSize;
  Tensor pixels;
  pixels.shape = {1, 3, imageSize, imageSize};
  if (!trace.empty())
    pixels.values.resize(pixels.count());
  const auto visionStart = timings ? Clock::now() : Clock::time_point{};
  auto features = engines_.at("vision_encoder")->runWithDeviceInput(
      "pixel_values", pixels.shape, [&](float *device, cudaStream_t stream) {
        const auto start = timings ? Clock::now() : Clock::time_point{};
        preprocessor_->enqueue(bgr, device, stream,
                               trace.empty() ? nullptr : pixels.values.data());
        if (timings)
          timings->preprocessMs = elapsedMs(start);
      },
      {"feats0", "feats1", "feats2", "vision_pos_embed", "feats2_no_mem"});
  if (timings)
    timings->graphMs.emplace("vision_encoder",
                             elapsedMs(visionStart) - timings->preprocessMs);
  if (!trace.empty())
    saveRaw(trace / "pixel_values.bin", pixels);
  saveGraph("vision_encoder", features);
  // Age the bank by accepted observations, including across occlusions.
  const int memoryIndex =
      pointers_.empty() ? std::min(index_, 1) : pointers_.back().index + 1;
  Tensor conditioned;
  if (index_ == 0)
    conditioned = std::move(features.at("feats2_no_mem"));
  else {
    std::vector<const Memory *> selected{&seed_};
    for (const auto &m : recent_)
      selected.push_back(&m);
    while (selected.size() < 7)
      selected.push_back(selected.back());
    auto &memory = memoryScratch_;
    auto &position = positionScratch_;
    memory.clear();
    memory.reserve(28736 * 64);
    position.resize(28736 * 64);
    auto positionOut = position.begin();
    for (const Memory *m : selected) {
      memory.insert(memory.end(), m->tokens.values.begin(),
                    m->tokens.values.end());
      int row = m->index == 0 ? 6 : memoryIndex - m->index - 1;
      if (row < 0 || row > 6)
        throw std::runtime_error("Temporal offset out of range");
      if (m->pos.values.size() != 4096 * 64)
        throw std::runtime_error("Unexpected memory position size");
      const auto &offset = temporal_[row];
      for (auto token = m->pos.values.begin(); token != m->pos.values.end();
           token += offset.size())
        positionOut = std::transform(
            offset.begin(), offset.end(), token, positionOut,
            [](float temporal, float spatial) { return spatial + temporal; });
    }
    std::vector<const Pointer *> ptrs{&seedPointer_};
    for (auto it = pointers_.rbegin(); it != pointers_.rend(); ++it)
      ptrs.push_back(&*it);
    while (ptrs.size() < 16)
      ptrs.push_back(ptrs.back());
    std::vector<float> diffs;
    for (const auto *p : ptrs)
      diffs.push_back(static_cast<float>(memoryIndex - p->index) /
                      std::max(std::min(totalFrames_, 16) - 1, 1));
    auto tpos = run("pointer_tpos",
                    {{"normalized_diffs", floats({16}, std::move(diffs))}});
    for (size_t i = 0; i < ptrs.size(); ++i) {
      const auto &v = ptrs[i]->value.values;
      memory.insert(memory.end(), v.begin(), v.end());
      const auto &pointerPos = tpos.at("pointer_pos").values;
      for (int token = 0; token < 4; ++token)
        positionOut = std::copy_n(pointerPos.begin() + i * 64, 64, positionOut);
    }
    Tensors inputs;
    inputs.emplace("current_vision_features", sequence(features.at("feats2")));
    inputs.emplace("current_vision_position_embeddings",
                   sequence(features.at("vision_pos_embed")));
    inputs.emplace("memory", floats({28736, 1, 64}, std::move(memory)));
    inputs.emplace("memory_pos", floats({28736, 1, 64}, std::move(position)));
    auto attention = run("memory_attention", inputs, {"conditioned_feats"});
    // Inference has synchronized, so the input storage can be reused.
    memory = std::move(inputs.at("memory").values);
    position = std::move(inputs.at("memory_pos").values);
    conditioned = std::move(attention.at("conditioned_feats"));
  }
  std::vector<float> points{0, 0};
  if (click) {
    if (click->x < 0 || click->x >= bgr.cols || click->y < 0 ||
        click->y >= bgr.rows)
      throw std::runtime_error("Click outside image");
    points = {click->x * 1024 / bgr.cols, click->y * 1024 / bgr.rows};
  }
  auto decoded = [&] {
    Tensors inputs;
    inputs.emplace("feats0", std::move(features.at("feats0")));
    inputs.emplace("feats1", std::move(features.at("feats1")));
    inputs.emplace("feats2_cond", std::move(conditioned));
    inputs.emplace("input_points", floats({1, 1, 1, 2}, std::move(points)));
    inputs.emplace("input_labels", ints({1, 1, 1}, {click ? 1 : -1}));
    return run("mask_decoder", inputs);
  }();
  auto score = std::move(decoded.at("object_score_logits"));
  const bool objectPresent = score.values.at(0) > 0.f;
  score.shape = {1, 1};
  Tensors encoderInputs;
  encoderInputs.emplace("feats2", std::move(features.at("feats2")));
  encoderInputs.emplace("high_res_mask", std::move(decoded.at("high_res_mask")));
  encoderInputs.emplace("object_score_logits", std::move(score));
  encoderInputs.emplace("binarize", floats({}, {index_ == 0 ? 1.f : 0.f}));
  auto encoded = run("memory_encoder", encoderInputs);
  Memory mem{memoryIndex, std::move(encoded.at("memory_tokens")),
             std::move(encoded.at("memory_pos"))};
  Pointer pointer{memoryIndex, std::move(decoded.at("object_pointer"))};
  if (index_ == 0) {
    seed_ = std::move(mem);
    seedPointer_ = std::move(pointer);
  } else if (objectPresent) {
    recent_.push_back(std::move(mem));
    if (recent_.size() > 6)
      recent_.pop_front();
    pointers_.push_back(std::move(pointer));
    if (pointers_.size() > 15)
      pointers_.pop_front();
  }
  ++index_;
  return std::move(encoderInputs.at("high_res_mask"));
}
} // namespace dml


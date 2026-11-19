#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
namespace dml {
struct Tensor {
  std::vector<int64_t> shape;
  nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
  std::vector<float> values;
  std::vector<int32_t> integers;
  // Borrowed FP32 storage. The consumer must finish before the producing
  // engine is run again or destroyed.
  void *deviceData = nullptr;
  // Optional conversion from the borrowed storage into an input binding.
  void (*deviceTransform)(const float *, float *, cudaStream_t) = nullptr;
  size_t count() const;
  const void *data() const;
  void *data();
};
using Tensors = std::map<std::string, Tensor>;
Tensor floats(std::vector<int64_t> shape, std::vector<float> values);
Tensor ints(std::vector<int64_t> shape, std::vector<int32_t> values);
void saveRaw(const std::filesystem::path &path, const Tensor &tensor);
Tensor readRaw(const std::filesystem::path &path, Tensor tensor);
class TrtEngine {
public:
  explicit TrtEngine(const std::filesystem::path &path);
  ~TrtEngine();
  TrtEngine(const TrtEngine &) = delete;
  TrtEngine &operator=(const TrtEngine &) = delete;
  Tensors run(const Tensors &inputs);
  // Fill the sole FP32 input with finite values on the execution stream.
  using DeviceInputWriter = std::function<void(float *, cudaStream_t)>;
  using DeviceOutputs = std::initializer_list<std::string_view>;
  Tensors run(const Tensors &inputs, DeviceOutputs deviceOutputs) {
    return run(inputs, nullptr, deviceOutputs);
  }
  Tensors runWithDeviceInput(const std::string &name,
                             std::vector<int64_t> shape,
                             const DeviceInputWriter &writeInput,
                             DeviceOutputs deviceOutputs = {});
  Tensors inputLayout() const;

private:
  Tensors run(const Tensors &inputs, const DeviceInputWriter *writeInput,
              DeviceOutputs deviceOutputs);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace dml

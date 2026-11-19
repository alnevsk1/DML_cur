#include "trt_engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(_M_X64)
#include <emmintrin.h>
#endif
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace dml {
namespace detail {
void enqueueFiniteCheck(const float *values, size_t count, unsigned *flags,
                        unsigned mask, cudaStream_t stream);
}
namespace {
void check(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}
void cudaCheck(cudaError_t status) {
  check(status == cudaSuccess,
        std::string("CUDA: ") + cudaGetErrorString(status));
}
bool allFinite(const std::vector<float> &values) {
  size_t i = 0;
#if defined(_M_X64)
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  const auto exponent = _mm_set1_epi32(0x7f800000);
  for (; values.size() - i >= 4; i += 4) {
    __m128i bits;
    std::memcpy(&bits, values.data() + i, sizeof(bits));
    const auto nonfinite =
        _mm_cmpeq_epi32(_mm_and_si128(bits, exponent), exponent);
    if (_mm_movemask_epi8(nonfinite) != 0)
      return false;
  }
#endif
  for (; i < values.size(); ++i)
    if (!std::isfinite(values[i]))
      return false;
  return true;
}

// Drain pending copies on failure before their host buffers leave scope.
struct StreamWork {
  cudaStream_t stream;
  ~StreamWork() {
    if (stream)
      cudaStreamSynchronize(stream);
  }
  void finish() {
    const auto status = cudaStreamSynchronize(stream);
    stream = nullptr;
    cudaCheck(status);
  }
};
struct Logger final : nvinfer1::ILogger {
  void log(Severity s, const char *msg) noexcept override {
    if (s <= Severity::kWARNING)
      std::cerr << "[TRT] " << msg << '\n';
  }
};
struct Buffer {
  void *pointer = nullptr;
  size_t bytes = 0;
  ~Buffer() {
    if (pointer)
      cudaFree(pointer);
  }
  void reserve(size_t n) {
    if (n <= bytes)
      return;
    if (pointer) {
      cudaCheck(cudaFree(pointer));
      pointer = nullptr;
      bytes = 0;
    }
    cudaCheck(cudaMalloc(&pointer, n));
    bytes = n;
  }
};
std::vector<int64_t> shapeOf(nvinfer1::Dims d) {
  check(d.nbDims >= 0, "Unresolved TensorRT dimensions");
  return {d.d, d.d + d.nbDims};
}
} // namespace
size_t Tensor::count() const {
  size_t n = 1;
  for (auto d : shape) {
    check(d > 0 && static_cast<uint64_t>(d) <=
                       std::numeric_limits<size_t>::max() / n,
          "Invalid tensor extent");
    n *= static_cast<size_t>(d);
  }
  return n;
}
const void *Tensor::data() const {
  if (deviceData)
    throw std::logic_error("Device tensor has no host data");
  return type == nvinfer1::DataType::kFLOAT
             ? static_cast<const void *>(values.data())
             : integers.data();
}
void *Tensor::data() {
  if (deviceData)
    throw std::logic_error("Device tensor has no host data");
  return type == nvinfer1::DataType::kFLOAT ? static_cast<void *>(values.data())
                                            : integers.data();
}
Tensor floats(std::vector<int64_t> shape, std::vector<float> values) {
  Tensor t;
  t.shape = std::move(shape);
  t.values = std::move(values);
  check(t.count() == t.values.size(), "Float tensor size mismatch");
  return t;
}
Tensor ints(std::vector<int64_t> shape, std::vector<int32_t> values) {
  Tensor t;
  t.type = nvinfer1::DataType::kINT32;
  t.shape = std::move(shape);
  t.integers = std::move(values);
  check(t.count() == t.integers.size(), "Int tensor size mismatch");
  return t;
}
void saveRaw(const std::filesystem::path &path, const Tensor &t) {
  if (t.deviceTransform)
    throw std::logic_error("Cannot save a pending device transform");
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  const auto bytes = t.count() * 4;
  if (t.deviceData) {
    std::vector<float> host(t.count());
    cudaCheck(cudaMemcpy(host.data(), t.deviceData, bytes, cudaMemcpyDeviceToHost));
    out.write(reinterpret_cast<const char *>(host.data()),
              static_cast<std::streamsize>(bytes));
  } else {
    out.write(static_cast<const char *>(t.data()),
              static_cast<std::streamsize>(bytes));
  }
  check(out.good(), "Cannot write " + path.string());
}
Tensor readRaw(const std::filesystem::path &path, Tensor t) {
  t.deviceData = nullptr;
  t.deviceTransform = nullptr;
  check(std::filesystem::file_size(path) == t.count() * 4,
        "Raw file size mismatch: " + path.string());
  if (t.type == nvinfer1::DataType::kFLOAT)
    t.values.resize(t.count());
  else
    t.integers.resize(t.count());
  std::ifstream in(path, std::ios::binary);
  in.read(static_cast<char *>(t.data()),
          static_cast<std::streamsize>(t.count() * 4));
  check(in.good(), "Cannot read " + path.string());
  return t;
}
struct TrtEngine::Impl {

  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::map<std::string, Buffer> buffers;
  Buffer outputFlags;
  cudaStream_t stream = nullptr;
  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }
};
TrtEngine::TrtEngine(const std::filesystem::path &path)
    : impl_(std::make_unique<Impl>()) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  check(input.good(), "Cannot open engine: " + path.string());
  auto bytes = input.tellg();
  check(bytes > 0, "Empty engine");
  std::vector<char> blob(static_cast<size_t>(bytes));
  input.seekg(0);
  input.read(blob.data(), bytes);
  check(input.good(), "Cannot read engine");
  auto &p = *impl_;
  static Logger logger;
  p.runtime.reset(nvinfer1::createInferRuntime(logger));
  check(bool(p.runtime), "createInferRuntime failed");
  p.engine.reset(p.runtime->deserializeCudaEngine(blob.data(), blob.size()));
  check(bool(p.engine), "deserializeCudaEngine failed");
  p.context.reset(p.engine->createExecutionContext());
  check(bool(p.context), "createExecutionContext failed");
  cudaCheck(cudaStreamCreate(&p.stream));
  for (int i = 0; i < p.engine->getNbIOTensors(); ++i) {
    const char *name = p.engine->getIOTensorName(i);
    auto type = p.engine->getTensorDataType(name);
    check(type == nvinfer1::DataType::kFLOAT ||
              type == nvinfer1::DataType::kINT32,
          "Unsupported tensor type");
    check(p.engine->getTensorLocation(name) ==
              nvinfer1::TensorLocation::kDEVICE,
          "Expected device tensor");
    check(p.engine->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR,
          "Expected linear tensor");
  }
}
TrtEngine::~TrtEngine() = default;
Tensors TrtEngine::inputLayout() const {
  Tensors result;
  const auto &e = *impl_->engine;
  for (int i = 0; i < e.getNbIOTensors(); ++i) {
    const char *name = e.getIOTensorName(i);
    if (e.getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
      continue;
    Tensor t;
    t.type = e.getTensorDataType(name);
    t.shape = shapeOf(e.getTensorShape(name));
    for (auto &d : t.shape)
      if (d < 0)
        d = std::string(name) == "normalized_diffs" ? 16 : 1;
    result.emplace(name, std::move(t));
  }
  return result;
}
Tensors TrtEngine::run(const Tensors &inputs) {
  return run(inputs, nullptr, {});
}
Tensors TrtEngine::runWithDeviceInput(const std::string &name,
                                      std::vector<int64_t> shape,
                                      const DeviceInputWriter &writeInput,
                                      DeviceOutputs deviceOutputs) {
  Tensor input;
  input.shape = std::move(shape);
  Tensors inputs;
  inputs.emplace(name, std::move(input));
  return run(inputs, &writeInput, deviceOutputs);
}
Tensors TrtEngine::run(const Tensors &inputs,
                       const DeviceInputWriter *writeInput,
                       DeviceOutputs deviceOutputs) {
  auto &p = *impl_;
  check(deviceOutputs.size() <= 32, "Too many device outputs");
  const auto outputMask = [deviceOutputs](std::string_view name) {
    const auto it = std::find(deviceOutputs.begin(), deviceOutputs.end(), name);
    return it == deviceOutputs.end() ? 0u : 1u << (it - deviceOutputs.begin());
  };
  auto layout = inputLayout();
  check(inputs.size() == layout.size(), "Input count mismatch");
  for (const auto &[name, expected] : layout) {
    const auto &t = inputs.at(name);
    check(t.type == expected.type && t.shape == expected.shape,
          "Input type/shape mismatch: " + name);
    if (t.deviceTransform && !t.deviceData)
      throw std::runtime_error("Missing device transform source: " + name);
    if (t.deviceData) {
      check(t.type == nvinfer1::DataType::kFLOAT && t.values.empty() &&
                t.integers.empty() && !writeInput,
            "Invalid borrowed device input: " + name);
    } else if (!writeInput) {
      check(t.count() == (t.type == nvinfer1::DataType::kFLOAT
                              ? t.values.size()
                              : t.integers.size()),
            "Input size mismatch");
      if (t.type == nvinfer1::DataType::kFLOAT && !allFinite(t.values))
        throw std::runtime_error("Nonfinite input: " + name);
    }
    nvinfer1::Dims d{};
    d.nbDims = static_cast<int>(t.shape.size());
    for (int j = 0; j < d.nbDims; ++j)
      d.d[j] = t.shape[j];
    check(p.context->setInputShape(name.c_str(), d),
          "setInputShape failed: " + name);
  }
  check(p.context->inferShapes(0, nullptr) == 0, "Unresolved shapes");
  Tensors outputs;
  unsigned invalidOutputs = 0;
  StreamWork work{p.stream};
  for (int i = 0; i < p.engine->getNbIOTensors(); ++i) {
    std::string name = p.engine->getIOTensorName(i);
    Tensor t;
    t.shape = shapeOf(p.context->getTensorShape(name.c_str()));
    t.type = p.engine->getTensorDataType(name.c_str());
    const Tensor *input =
        p.engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT
            ? &inputs.at(name)
            : nullptr;
    if (input && input->deviceData && !input->deviceTransform) {
      check(p.context->setTensorAddress(name.c_str(), input->deviceData),
            "setTensorAddress failed");
      continue;
    }
    auto &b = p.buffers[name];
    b.reserve(t.count() * 4);
    check(p.context->setTensorAddress(name.c_str(), b.pointer),
          "setTensorAddress failed");
    if (input) {
      if (writeInput)
        (*writeInput)(static_cast<float *>(b.pointer), p.stream);
      else if (input->deviceTransform)
        input->deviceTransform(static_cast<const float *>(input->deviceData),
                               static_cast<float *>(b.pointer), p.stream);
      else
        cudaCheck(cudaMemcpyAsync(b.pointer, input->data(), t.count() * 4,
                                  cudaMemcpyHostToDevice, p.stream));
    } else {
      if (outputMask(name)) {
        check(t.type == nvinfer1::DataType::kFLOAT, "Expected FP32 device output");
        t.deviceData = b.pointer;
      } else if (t.type == nvinfer1::DataType::kFLOAT)
        t.values.resize(t.count());
      else
        t.integers.resize(t.count());
      outputs.emplace(name, std::move(t));
    }
  }
  check(p.context->enqueueV3(p.stream), "enqueueV3 failed");
  if (deviceOutputs.size()) {
    p.outputFlags.reserve(sizeof(invalidOutputs));
    auto *flags = static_cast<unsigned *>(p.outputFlags.pointer);
    cudaCheck(cudaMemsetAsync(flags, 0, sizeof(invalidOutputs), p.stream));
    for (const auto name : deviceOutputs) {
      const auto &t = outputs.at(std::string(name));
      detail::enqueueFiniteCheck(static_cast<const float *>(t.deviceData),
                                 t.count(), flags, outputMask(name), p.stream);
    }
  }
  for (auto &[name, t] : outputs)
    if (!t.deviceData)
      cudaCheck(cudaMemcpyAsync(t.data(), p.buffers.at(name).pointer,
                                t.count() * 4, cudaMemcpyDeviceToHost, p.stream));
  if (deviceOutputs.size())
    cudaCheck(cudaMemcpyAsync(&invalidOutputs, p.outputFlags.pointer,
                              sizeof(invalidOutputs), cudaMemcpyDeviceToHost,
                              p.stream));
  work.finish();
  for (const auto &[name, t] : outputs)
    if (t.deviceData ? (invalidOutputs & outputMask(name)) != 0
                     : !allFinite(t.values))
      throw std::runtime_error("Nonfinite output: " + name);
  return outputs;
}
} // namespace dml

#include <cuda_runtime.h>

#include <stdexcept>

namespace dml::detail {
namespace {
constexpr unsigned blockSize = 256;

__global__ void finiteCheck(const float *values, size_t count, unsigned *flags,
                            unsigned mask) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // All ones in the FP32 exponent identifies infinities and NaNs.
  const bool invalid =
      i < count && (__float_as_uint(values[i]) & 0x7f800000u) == 0x7f800000u;
  const int any = __syncthreads_or(invalid);
  if (threadIdx.x == 0 && any)
    atomicOr(flags, mask);
}
} // namespace

void enqueueFiniteCheck(const float *values, size_t count, unsigned *flags,
                        unsigned mask, cudaStream_t stream) {
  const auto blocks = static_cast<unsigned>((count + blockSize - 1) / blockSize);
  finiteCheck<<<blocks, blockSize, 0, stream>>>(values, count, flags, mask);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace dml::detail

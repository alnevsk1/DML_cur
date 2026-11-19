#include <cuda_runtime.h>

#include <stdexcept>

namespace dml::detail {
namespace {
constexpr int tileSize = 32;
constexpr int tileRows = 8;
constexpr int channels = 256;
constexpr int positions = 4096;
static_assert(channels % tileSize == 0 && positions % tileSize == 0);

// Transpose C by HW to HW by C. The extra column avoids bank conflicts
// when threads read the tile by column.
__global__ void visionSequence(const float *input, float *output) {
  __shared__ float tile[tileSize][tileSize + 1];
  const int p = blockIdx.x * tileSize + threadIdx.x;
  const int c = blockIdx.y * tileSize + threadIdx.y;
  for (int row = 0; row < tileSize; row += tileRows)
    tile[threadIdx.y + row][threadIdx.x] = input[(c + row) * positions + p];
  __syncthreads();
  const int outC = blockIdx.y * tileSize + threadIdx.x;
  const int outP = blockIdx.x * tileSize + threadIdx.y;
  for (int row = 0; row < tileSize; row += tileRows)
    output[(outP + row) * channels + outC] = tile[threadIdx.x][threadIdx.y + row];
}
} // namespace

void enqueueVisionSequence(const float *input, float *output,
                           cudaStream_t stream) {
  const dim3 grid(positions / tileSize, channels / tileSize);
  const dim3 block(tileSize, tileRows);
  visionSequence<<<grid, block, 0, stream>>>(input, output);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace dml::detail

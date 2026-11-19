// Этап 2: проверка окружения.
// 1) OpenCV собран с CUDA и видит GPU; 2) кадр проходит GPU-предобработку
// (upload -> cuda::resize -> cuda::cvtColor BGR->RGB -> download) и совпадает с CPU-версией;
// 3) TensorRT линкуется и создаёт builder; 4) у videoio есть бэкенд FFmpeg.
//
// Запуск: env_check [image_path] [out_path]
//   без image_path используется синтетический кадр 3840x2160 (4K, как съёмка с телефона).

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferVersion.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio/registry.hpp>

namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << '\n';
    }
};

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Гладкий кадр (градиент + объекты + размытие). Белый шум не годится для сравнения GPU/CPU:
// при сильном даунскейле любое субпиксельное расхождение интерполяции даёт разницу ~255.
cv::Mat makeSyntheticFrame() {
    cv::Mat frame(2160, 3840, CV_8UC3);  // (H, W, 3) BGR
    for (int y = 0; y < frame.rows; ++y) {
        auto* row = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < frame.cols; ++x) {
            row[x] = {static_cast<uchar>(x * 255 / frame.cols), static_cast<uchar>(y * 255 / frame.rows),
                      static_cast<uchar>(128)};
        }
    }
    cv::rectangle(frame, {600, 400}, {1400, 900}, {40, 160, 40}, cv::FILLED);
    cv::circle(frame, {1920, 1080}, 40, {0, 0, 255}, cv::FILLED);  // «мелкий объект»
    cv::GaussianBlur(frame, frame, {0, 0}, 3.0);
    return frame;
}

}  // namespace

int main(int argc, char** argv) {
    constexpr int kInputSize = 1024;  // вход энкодера SAM 2 — 1024x1024
    // INTER_AREA: при даунскейле 4K -> 1024 усредняет пиксели и не теряет мелкие объекты
    // (INTER_LINEAR берёт 2x2 соседей и даёт алиасинг; к тому же GPU/CPU реализации на краях расходятся).
    constexpr int kInterp = cv::INTER_AREA;
    const std::string outPath = argc > 2 ? argv[2] : "env_check_out.png";

    // --- 1. OpenCV + CUDA ---
    std::cout << "OpenCV " << CV_VERSION << '\n';
    const int devices = cv::cuda::getCudaEnabledDeviceCount();
    if (devices <= 0) {
        std::cerr << "FAIL: OpenCV does not see a CUDA device (count=" << devices << ")\n";
        return 1;
    }
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    // --- 2. GPU-предобработка кадра ---
    cv::Mat frame = argc > 1 ? cv::imread(argv[1], cv::IMREAD_COLOR) : makeSyntheticFrame();
    if (frame.empty()) {
        std::cerr << "FAIL: cannot read image " << argv[1] << '\n';
        return 1;
    }
    std::cout << "Input: " << frame.cols << "x" << frame.rows << '\n';

    cv::cuda::Stream stream;
    cv::cuda::GpuMat gSrc, gResized, gRgb;
    cv::Mat gpuResult;

    // Pinned (page-locked) копия кадра: DMA без промежуточного буфера драйвера.
    cv::cuda::HostMem pinnedFrame(frame.rows, frame.cols, frame.type(),
                                  cv::cuda::HostMem::PAGE_LOCKED);
    frame.copyTo(pinnedFrame.createMatHeader());

    auto gpuCompute = [&] {
        cv::cuda::resize(gSrc, gResized, {kInputSize, kInputSize}, 0, 0, kInterp, stream);
        cv::cuda::cvtColor(gResized, gRgb, cv::COLOR_BGR2RGB, 0, stream);  // модель ждёт RGB
    };
    auto gpuPass = [&](bool pinned) {
        if (pinned) {
            gSrc.upload(pinnedFrame, stream);
        } else {
            gSrc.upload(frame, stream);
        }
        gpuCompute();
        gRgb.download(gpuResult, stream);
        stream.waitForCompletion();
    };
    gpuPass(false);  // прогрев: инициализация контекста и ядер

    constexpr int kIters = 50;
    auto t0 = Clock::now();
    for (int i = 0; i < kIters; ++i) gpuPass(false);
    const double gpuMs = msSince(t0) / kIters;

    t0 = Clock::now();
    for (int i = 0; i < kIters; ++i) gpuPass(true);
    const double gpuPinnedMs = msSince(t0) / kIters;

    // Разбивка по стадиям (pageable-память), каждая стадия с синхронизацией.
    double uploadMs = 0.0, computeMs = 0.0, downloadMs = 0.0;
    for (int i = 0; i < kIters; ++i) {
        t0 = Clock::now();
        gSrc.upload(frame, stream);
        stream.waitForCompletion();
        uploadMs += msSince(t0);
        t0 = Clock::now();
        gpuCompute();
        stream.waitForCompletion();
        computeMs += msSince(t0);
        t0 = Clock::now();
        gRgb.download(gpuResult, stream);
        stream.waitForCompletion();
        downloadMs += msSince(t0);
    }

    cv::Mat cpuResized, cpuResult;
    t0 = Clock::now();
    for (int i = 0; i < kIters; ++i) {
        cv::resize(frame, cpuResized, {kInputSize, kInputSize}, 0, 0, kInterp);
        cv::cvtColor(cpuResized, cpuResult, cv::COLOR_BGR2RGB);
    }
    const double cpuMs = msSince(t0) / kIters;

    cv::Mat diff;
    cv::absdiff(gpuResult, cpuResult, diff);
    std::vector<cv::Mat> diffCh;
    cv::split(diff, diffCh);
    const cv::Mat diffMax = cv::max(cv::max(diffCh[0], diffCh[1]), diffCh[2]);  // (H, W) max по каналам
    double maxDiff = 0.0;
    cv::Point maxLoc;
    cv::minMaxLoc(diffMax, nullptr, &maxDiff, nullptr, &maxLoc);
    const double meanDiff = cv::mean(diff.reshape(1))[0];
    const int bigDiffCount = cv::countNonZero(diffMax > 2);

    std::printf("Preprocess %dx%d -> %dx%d (resize + BGR->RGB), avg of %d iters:\n", frame.cols,
                frame.rows, kInputSize, kInputSize, kIters);
    std::printf("  GPU total, pageable host mem: %.3f ms/frame\n", gpuMs);
    std::printf("  GPU total, pinned host mem:   %.3f ms/frame\n", gpuPinnedMs);
    std::printf("    upload %.3f | compute %.3f | download %.3f ms (pageable, synced per stage)\n",
                uploadMs / kIters, computeMs / kIters, downloadMs / kIters);
    std::printf("  CPU:                          %.3f ms/frame\n", cpuMs);
    std::printf("  |GPU - CPU|: max %.0f at (%d, %d), mean %.4f, pixels with diff > 2: %d of %d\n",
                maxDiff, maxLoc.x, maxLoc.y, meanDiff, bigDiffCount, kInputSize * kInputSize);

    cv::Mat toSave;
    cv::cvtColor(gpuResult, toSave, cv::COLOR_RGB2BGR);  // imwrite ждёт BGR
    cv::imwrite(outPath, toSave);
    std::cout << "Saved: " << outPath << '\n';

    // --- 3. TensorRT ---
    std::printf("TensorRT headers %d.%d.%d, library %d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                NV_TENSORRT_PATCH, getInferLibVersion());
    TrtLogger logger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (builder == nullptr) {
        std::cerr << "FAIL: createInferBuilder returned null\n";
        return 1;
    }
    std::cout << "TensorRT builder OK, fast FP16: " << (builder->platformHasFastFp16() ? "yes" : "no")
              << '\n';
    delete builder;

    int cudaRt = 0;
    cudaRuntimeGetVersion(&cudaRt);
    std::printf("CUDA runtime %d.%d\n", cudaRt / 1000, (cudaRt % 1000) / 10);

    // --- 4. Видео-бэкенды ---
    std::cout << "videoio FFmpeg backend: "
              << (cv::videoio_registry::hasBackend(cv::CAP_FFMPEG) ? "yes" : "NO") << '\n';

    // CPU-путь для 8U считает в фиксированной точке, GPU — во float: единичные расхождения
    // на границах объектов допустимы, систематических быть не должно.
    const bool ok = meanDiff < 0.5 && maxDiff <= 8.0;
    std::cout << (ok ? "ENV CHECK: OK" : "ENV CHECK: GPU/CPU mismatch") << '\n';
    return ok ? 0 : 2;
}

cmake_minimum_required(VERSION 3.24)
project(uav_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Пути к SDK (версии: docs/versions.md) ---
set(OpenCV_DIR "C:/opencv/install" CACHE PATH "OpenCV install dir (with OpenCVConfig.cmake)")
set(TENSORRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../TensorRT-10.16.1.11" CACHE PATH "Unpacked TensorRT dir")

find_package(OpenCV 4.14 REQUIRED COMPONENTS
    core imgproc imgcodecs videoio highgui
    cudaarithm cudawarping cudaimgproc)
find_package(CUDAToolkit 12.9 REQUIRED)
enable_language(CUDA)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS "${TENSORRT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(TENSORRT_NVINFER nvinfer_10 PATHS "${TENSORRT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(TENSORRT_ONNXPARSER nvonnxparser_10 PATHS "${TENSORRT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(tensorrt INTERFACE)
target_include_directories(tensorrt INTERFACE "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(tensorrt INTERFACE "${TENSORRT_NVINFER}" "${TENSORRT_ONNXPARSER}" CUDA::cudart)

# --- Этап 2: проверка окружения (OpenCV CUDA + TensorRT) ---
add_executable(env_check src/tools/env_check.cpp)
target_link_libraries(env_check PRIVATE ${OpenCV_LIBS} tensorrt)

# Synchronous HF SAM 2.1-tiny application and reproducible headless validation.
add_library(sam2_tracker
    src/inference/trt_engine.cpp
    src/inference/trt_checks.cu
    src/preprocess/preprocess.cpp
    src/preprocess/preprocess_cuda.cu
    src/tracker/sam2_tracker.cpp
    src/tracker/vision_sequence.cu)
set_target_properties(sam2_tracker PROPERTIES
    CUDA_ARCHITECTURES 89 CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(sam2_tracker PUBLIC src)
target_link_libraries(sam2_tracker PUBLIC ${OpenCV_LIBS} tensorrt)
if(MSVC)
    target_compile_options(sam2_tracker PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/fp:precise>")
    target_compile_definitions(sam2_tracker PUBLIC NOMINMAX)
endif()
add_executable(uav_tracker src/main.cpp)
target_link_libraries(uav_tracker PRIVATE sam2_tracker)

add_executable(uav_tracker_gui WIN32 src/ui/main.cpp src/ui/session.cpp)
target_compile_definitions(uav_tracker_gui PRIVATE UNICODE _UNICODE)
target_link_libraries(uav_tracker_gui PRIVATE sam2_tracker comctl32 comdlg32 shell32)
if(MSVC)
    target_compile_options(uav_tracker_gui PRIVATE /W4 /permissive- /utf-8)
endif()

add_executable(sync_video_benchmark src/tools/sync_video_benchmark.cpp)
target_link_libraries(sync_video_benchmark PRIVATE sam2_tracker)
if(MSVC)
    target_compile_options(sync_video_benchmark PRIVATE /W4 /permissive-)
endif()

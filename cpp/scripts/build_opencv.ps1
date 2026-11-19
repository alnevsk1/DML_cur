# Сборка OpenCV 4.14.0 + opencv_contrib с CUDA для Windows (параметры — docs/versions.md).
# Ожидает исходники из zip-архивов тега 4.14.0 с GitHub, распакованные в $Root:
#   $Root\opencv-4.14.0, $Root\opencv_contrib-4.14.0
# Запуск:  powershell -ExecutionPolicy Bypass -File cpp\scripts\build_opencv.ps1 [-Stage configure|build|all]
param(
    [string]$Root = "C:\opencv",
    [ValidateSet("configure", "build", "all")]
    [string]$Stage = "all",
    [int]$Jobs = 4
)
$ErrorActionPreference = "Stop"

$BuildTools = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$VcVars = Join-Path $BuildTools "VC\Auxiliary\Build\vcvars64.bat"
$CMakeDir = Join-Path $BuildTools "Common7\IDE\CommonExtensions\Microsoft\CMake"

$Src = Join-Path $Root "opencv-4.14.0"
$Contrib = Join-Path $Root "opencv_contrib-4.14.0\modules"
$Build = Join-Path $Root "build"
$Install = Join-Path $Root "install"

# Окружение MSVC x64 (cl.exe, link.exe, INCLUDE/LIB) — переносим переменные из vcvars64.bat в текущий процесс.
cmd /c "`"$VcVars`" >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}
$env:Path = "$(Join-Path $CMakeDir 'CMake\bin');$(Join-Path $CMakeDir 'Ninja');$env:Path"

# CMake разбирает '\' в значениях -D как escape-последовательность ('\P' → ошибка) — передаём пути с '/'.
function ConvertTo-CMakePath([string]$Path) { $Path.Replace('\', '/') }
$CudaRoot = ConvertTo-CMakePath $env:CUDA_PATH

$Modules = @(
    "core", "imgproc", "imgcodecs", "videoio", "highgui",
    "cudev", "cudaarithm", "cudafilters", "cudaimgproc", "cudawarping"
) -join ","

$CMakeArgs = @(
    "-S", $Src, "-B", $Build, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$(ConvertTo-CMakePath $Install)",
    "-DOPENCV_EXTRA_MODULES_PATH=$(ConvertTo-CMakePath $Contrib)",
    "-DBUILD_LIST=$Modules",
    # CUDA: только архитектура RTX 4060 (Ada, SM 8.9)
    "-DWITH_CUDA=ON",
    "-DCUDA_TOOLKIT_ROOT_DIR=$CudaRoot",
    "-DCUDA_ARCH_BIN=8.9",
    "-DCUDA_ARCH_PTX=",
    "-DWITH_CUBLAS=ON",
    "-DWITH_CUFFT=ON",
    # Не нужно: инференс через TensorRT, декодирование через cv::VideoCapture
    "-DWITH_CUDNN=OFF",
    "-DOPENCV_DNN_CUDA=OFF",
    "-DWITH_NVCUVID=OFF",
    "-DWITH_NVCUVENC=OFF",
    "-DWITH_OPENGL=OFF",
    # Без тестов, примеров и биндингов
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_DOCS=OFF",
    "-DBUILD_JAVA=OFF",
    "-DBUILD_opencv_python3=OFF",
    "-DBUILD_opencv_python_bindings_generator=OFF",
    "-DBUILD_opencv_java_bindings_generator=OFF",
    "-DINSTALL_TESTS=OFF"
)

if ($Stage -in "configure", "all") {
    cmake @CMakeArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }
}
if ($Stage -in "build", "all") {
    cmake --build $Build --config Release --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }
    cmake --install $Build --config Release
    if ($LASTEXITCODE -ne 0) { throw "Install failed ($LASTEXITCODE)" }
}

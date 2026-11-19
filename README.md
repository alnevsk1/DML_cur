# DML_cur

Отслеживание одного объекта по клику на видео. SAM 2.1-tiny, C++17, OpenCV, TensorRT, Win32.

## Запуск

Нужны Windows, NVIDIA GPU (сборка настроена для SM 8.9), Visual Studio 2022 Build Tools, CUDA 12.9, OpenCV 4.14 с CUDA в `C:\opencv\install` и TensorRT 10.16.1.11 в корне проекта. `CUDA_PATH` должен указывать на CUDA Toolkit.

Скачайте пять ONNX-графов и `constants.json` из [набора SAM 2.1-tiny](https://huggingface.co/square-zero-labs/sam2.1-tiny-video-onnx/tree/3b2984dd865f6e9d2cc6aed0be6a5a5c2eb352ce) в `models/onnx/sam2.1_tiny_hf/`.

В PowerShell из корня проекта соберите движки для смешанного режима:

```powershell
$onnx = 'models/onnx/sam2.1_tiny_hf'
$engines = 'models/engines/sam2.1_tiny_hf'
New-Item -ItemType Directory -Force $engines | Out-Null
. .\cpp\scripts\env.ps1

$profiles = @{
    mask_decoder = '--optShapes=input_points:1x1x1x2,input_labels:1x1x1'
    pointer_tpos = '--optShapes=normalized_diffs:16'
}
foreach ($graph in 'vision_encoder', 'mask_decoder', 'memory_encoder', 'memory_attention', 'pointer_tpos') {
    $precision = 'fp32'
    $extra = @()
    if ($graph -ne 'mask_decoder') {
        $precision = 'fp16'
        $extra += '--fp16'
    }
    if ($profiles.ContainsKey($graph)) {
        $extra += $profiles[$graph]
    }
    & .\TensorRT-10.16.1.11\bin\trtexec.exe "--onnx=$onnx/$graph.onnx" "--saveEngine=$engines/${graph}_$precision.engine" --noTF32 --builderOptimizationLevel=0 --memPoolSize=workspace:2048M --skipInference @extra
    if ($LASTEXITCODE -ne 0) { throw "Ошибка сборки движка $graph" }
}
```

Соберите и запустите приложение:

```powershell
& .\cpp\scripts\build.ps1 -Config Release
& .\cpp\scripts\launch_gui.cmd
```

Откройте локальное видео, выберите «Смешанная (декодер FP32)», щёлкните по объекту на паузе и нажмите «Пуск». Для воспроизведения по времени источника включите «Режим реального времени» на паузе.

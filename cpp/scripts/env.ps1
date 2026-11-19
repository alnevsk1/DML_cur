# Добавляет в PATH текущей сессии DLL OpenCV, TensorRT и CUDA, чтобы запускать собранные .exe.
# Использование (dot-source, из корня проекта):  . .\cpp\scripts\env.ps1
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dirs = @(
    "C:\opencv\install\x64\vc17\bin",
    (Join-Path $ProjectRoot "TensorRT-10.16.1.11\bin"),  # в Windows-архиве DLL лежат в bin\, в lib\ только .lib
    (Join-Path $env:CUDA_PATH "bin")
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { Write-Warning "Not found: $d"; continue }
    if (($env:Path -split ";") -notcontains $d) { $env:Path = "$d;$env:Path" }
}

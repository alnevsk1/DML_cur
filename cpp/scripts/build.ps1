# Сборка C++-части (cpp/) через Ninja в окружении MSVC x64.
# Запуск из корня проекта:  powershell -ExecutionPolicy Bypass -File cpp\scripts\build.ps1 [-Config Release|Debug]
# Для запуска собранных .exe нужны DLL OpenCV и TensorRT в PATH — см. cpp\scripts\env.ps1.
param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release"
)
$ErrorActionPreference = "Stop"

$BuildTools = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$VcVars = Join-Path $BuildTools "VC\Auxiliary\Build\vcvars64.bat"
$CMakeDir = Join-Path $BuildTools "Common7\IDE\CommonExtensions\Microsoft\CMake"

$CppDir = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $CppDir "build\$($Config.ToLower())"

cmd /c "`"$VcVars`" >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}
$env:Path = "$(Join-Path $CMakeDir 'CMake\bin');$(Join-Path $CMakeDir 'Ninja');$env:Path"

cmake -S $CppDir -B $Build -G Ninja "-DCMAKE_BUILD_TYPE=$Config"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }
cmake --build $Build
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }

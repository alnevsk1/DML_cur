@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1
if not exist ".\cpp\build\release\uav_tracker_gui.exe" (
    echo Build the Release target first with cpp\scripts\build.ps1.
    popd
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; . '.\cpp\scripts\env.ps1'; $gui = Start-Process -FilePath '.\cpp\build\release\uav_tracker_gui.exe' -Wait -PassThru; exit $gui.ExitCode"
set "GUI_EXIT=%ERRORLEVEL%"
popd
if not "%GUI_EXIT%"=="0" pause
exit /b %GUI_EXIT%

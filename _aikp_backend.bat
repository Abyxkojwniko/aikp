@echo off
chcp 65001 >nul
title AIKP 引擎（请勿关闭此窗口）
cd /d "%~dp0"

rem ---- locate conda activate.bat ----
set "CONDA_ACT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not exist "%CONDA_ACT%" set "CONDA_ACT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not exist "%CONDA_ACT%" set "CONDA_ACT=C:\ProgramData\miniconda3\Scripts\activate.bat"
if not exist "%CONDA_ACT%" set "CONDA_ACT=C:\ProgramData\anaconda3\Scripts\activate.bat"

if not exist "%CONDA_ACT%" (
    echo [错误] 找不到 conda（Miniconda / Anaconda）。
    echo 请先安装 conda 并创建名为 aikp 的环境。
    pause
    exit /b 1
)

call "%CONDA_ACT%" aikp
cd /d "%~dp0backend"
python server.py

echo.
echo [引擎已停止] 按任意键关闭。
pause >nul

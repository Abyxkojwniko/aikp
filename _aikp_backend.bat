@echo off
chcp 65001 >nul
title AIKP 引擎（请勿关闭此窗口）
cd /d "%~dp0"

rem ---- pick the Python runtime: local .venv first, else portable (bundle) ----
set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=%~dp0tools\python\python.exe"
if not exist "%PYEXE%" (
    echo [错误] 找不到 Python 运行环境（.venv 或 tools\python）。
    echo 请改用「启动游戏.bat」启动，它会自动配置环境。
    pause
    exit /b 1
)

cd /d "%~dp0backend"
"%PYEXE%" server.py

echo.
echo [引擎已停止] 按任意键关闭。
pause >nul

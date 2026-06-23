@echo off
chcp 65001 >nul
title AIKP 引擎（请勿关闭此窗口）
cd /d "%~dp0"

rem ---- run inside the project-local virtualenv (.venv) ----
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [错误] 找不到虚拟环境 .venv。
    echo 请改用「启动游戏.bat」启动，它会自动配置环境。
    pause
    exit /b 1
)

cd /d "%~dp0backend"
"%VENV_PY%" server.py

echo.
echo [引擎已停止] 按任意键关闭。
pause >nul

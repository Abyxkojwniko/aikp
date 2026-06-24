@echo off
chcp 936 >nul
title AIKP 启动器
cd /d "%~dp0"

echo ============================================
echo            AIKP   AI 跑团主持
echo ============================================
echo.
echo  正在启动游戏。首次启动会自动配置环境（可能需要几分钟下载），
echo  之后每次启动都很快，请耐心等待...
echo.

echo  [0/4] 检查并自动配置运行环境（首次较慢，之后秒过）...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_aikp_setup.ps1"
if errorlevel 1 (
    echo.
    echo  [!] 环境自动配置失败，无法启动。请把上面的红色错误反馈。
    pause
    exit /b 1
)
echo      环境已就绪 √
echo.

echo  [1/4] 清理可能残留的旧进程（确保加载最新代码）...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>nul
ping -n 2 127.0.0.1 >nul

echo  [2/4] 启动游戏引擎（后端，端口 8001）...
start "AIKP 引擎（请勿关闭）" /min "%~dp0_aikp_backend.bat"

echo  [3/4] 启动游戏界面（前端，端口 8000）...
start "AIKP 界面（请勿关闭）" /min "%~dp0_aikp_frontend.bat"

echo.
echo  [4/4] 等待服务就绪（不会死等，就绪后自动打开浏览器）
echo.

REM ==== 等后端 8001 就绪 ====
set /a n=0
:WAIT_BACKEND
netstat -ano | findstr ":8001" | findstr "LISTENING" >nul 2>nul
if not errorlevel 1 goto BACKEND_OK
set /a n+=1
if %n% geq 90 (
    echo.
    echo  [!] 引擎 90 秒内没起来。请打开最小化的「AIKP 引擎」窗口看报错。
    goto WAIT_FRONTEND
)
<nul set /p "=#"
ping -n 2 127.0.0.1 >nul
goto WAIT_BACKEND
:BACKEND_OK
echo.
echo      引擎已就绪 (8001) √

REM ==== 等前端 8000 就绪 ====
:WAIT_FRONTEND
set /a m=0
:LOOP_FRONTEND
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>nul
if not errorlevel 1 goto FRONTEND_OK
set /a m+=1
if %m% geq 90 (
    echo.
    echo  [!] 界面 90 秒内没起来。请打开最小化的「AIKP 界面」窗口看报错。
    goto FINISH
)
<nul set /p "=#"
ping -n 2 127.0.0.1 >nul
goto LOOP_FRONTEND
:FRONTEND_OK
echo.
echo      界面已就绪 (8000) √
echo.
echo  正在打开浏览器...
start "" "http://127.0.0.1:8000"

:FINISH
echo.
echo ============================================
echo  如果浏览器没自动打开，手动访问：
echo      http://127.0.0.1:8000
echo.
echo  后台有两个最小化的黑窗口在跑游戏，请勿关闭。
echo  结束游戏：双击「停止游戏.bat」
echo ============================================
echo.
echo  （本启动窗口 12 秒后自动关闭，不影响游戏运行）
timeout /t 12 /nobreak >nul
exit

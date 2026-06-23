@echo off
chcp 65001 >nul
title AIKP 界面（请勿关闭此窗口）
cd /d "%~dp0"

rem ---- prefer the portable Node (tools\node) if it was auto-installed ----
if exist "%~dp0tools\node\node.exe" set "PATH=%~dp0tools\node;%PATH%"

cd /d "%~dp0Tavern\SillyTavern"
node server.js

echo.
echo [界面已停止] 按任意键关闭。
pause >nul

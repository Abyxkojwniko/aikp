@echo off
chcp 936 >nul
title 关闭AIKP
echo 正在关闭 AIKP 的所有窗口和服务...

rem 1) free the game ports (python backend / node frontend) + their process trees
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING"') do taskkill /F /T /PID %%a >nul 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /T /PID %%a >nul 2>nul

rem 2) close the AIKP cmd windows (engine / frontend) by their command line
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'cmd.exe' -and ($_.CommandLine -like '*_aikp_backend.bat*' -or $_.CommandLine -like '*_aikp_frontend.bat*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo 已全部关闭，本窗口即将关闭。
timeout /t 1 /nobreak >nul
exit

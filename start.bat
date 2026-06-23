@echo off
echo ========================================
echo          AIKP - TRPG GM Server
echo ========================================
echo.
echo [1/2] Starting backend on port 8001...
start "AIKP Backend" /min cmd /c "call C:\Users\Abyxkojw\miniconda3\Scripts\activate.bat aikp && cd /d E:\aikp\backend && python server.py"
timeout /t 3 /nobreak >nul
echo [2/2] Starting SillyTavern on port 8000...
cd /d E:\aikp\Tavern\SillyTavern
echo Press Ctrl+C to stop.
node server.js
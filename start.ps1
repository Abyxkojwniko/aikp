# AIKP One-Click Launcher
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "         AIKP - TRPG GM Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start backend
Write-Host "[1/2] Starting backend on port 8001..." -ForegroundColor Yellow
$condaActivate = "C:\Users\Abyxkojw\miniconda3\Scripts\activate.bat"
$backend = Start-Process -PassThru -WindowStyle Minimized -FilePath cmd -ArgumentList "/c", "call $condaActivate aikp && cd /d E:\aikp\backend && python server.py"
Start-Sleep -Seconds 3
Write-Host "       Backend PID: $($backend.Id)" -ForegroundColor Green

# Start frontend
Write-Host "[2/2] Starting SillyTavern on port 8000..." -ForegroundColor Yellow
Set-Location "E:\aikp\Tavern\SillyTavern"
Write-Host "       Press Ctrl+C to stop. Opening browser..." -ForegroundColor Green
node server.js

# Cleanup on exit
Write-Host "Shutting down backend..." -ForegroundColor Yellow
Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
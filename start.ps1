# AIKP One-Click Launcher (PowerShell)
# Auto-configures the environment on first run, then starts both servers.
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "         AIKP - TRPG GM Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# [0/2] Ensure Python / Node / venv / deps (idempotent)
& "$Root\_aikp_setup.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "环境配置失败。" -ForegroundColor Red; exit 1 }

# [1/2] Backend on 8001 (project-local venv)
Write-Host "[1/2] Starting backend on port 8001..." -ForegroundColor Yellow
$venvPy  = Join-Path $Root '.venv\Scripts\python.exe'
$backend = Start-Process -PassThru -WindowStyle Minimized -FilePath $venvPy `
    -ArgumentList 'server.py' -WorkingDirectory (Join-Path $Root 'backend')
Start-Sleep -Seconds 3
Write-Host "       Backend PID: $($backend.Id)" -ForegroundColor Green

# [2/2] Frontend on 8000 (use portable Node if present)
$portableNode = Join-Path $Root 'tools\node'
if (Test-Path $portableNode) { $env:PATH = "$portableNode;$env:PATH" }
Write-Host "[2/2] Starting SillyTavern on port 8000..." -ForegroundColor Yellow
Write-Host "       Press Ctrl+C to stop." -ForegroundColor Green
Set-Location (Join-Path $Root 'Tavern\SillyTavern')
node server.js

# Cleanup on exit
Write-Host "Shutting down backend..." -ForegroundColor Yellow
Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue

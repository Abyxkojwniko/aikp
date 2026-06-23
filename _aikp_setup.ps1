# =========================================================================
#  AIKP 环境自动配置脚本 (_aikp_setup.ps1)
#  Idempotent bootstrapper invoked by 启动游戏.bat before launching.
#  Ensures everything a player needs WITHOUT manual setup:
#    1. Python   — uses a system Python if present, else downloads a
#                  portable Python into  tools\python\  (no admin needed)
#    2. .venv    — creates a project-local virtualenv + installs backend deps
#    3. Node.js  — uses a system Node if present, else downloads a portable
#                  Node into  tools\node\
#    4. npm deps — runs `npm install` for the SillyTavern frontend
#
#  Re-running is cheap: each step is skipped when already satisfied.
#  To force a clean reconfigure, delete the  .venv  and  tools  folders.
#  Comments are English; player-facing messages are Chinese (per project).
# =========================================================================

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'   # speeds up Invoke-WebRequest a lot

# --- Versions to fetch when nothing usable is found on the machine ---
$PY_VERSION   = '3.11.9'
$NODE_VERSION = '20.18.1'

$Root      = $PSScriptRoot
$Tools     = Join-Path $Root 'tools'
$Venv      = Join-Path $Root '.venv'
$VenvPy    = Join-Path $Venv 'Scripts\python.exe'
$ReqFile   = Join-Path $Root 'backend\requirements.txt'
$StRoot    = Join-Path $Root 'Tavern\SillyTavern'

function Say($msg)  { Write-Host "  $msg" -ForegroundColor Gray }
function Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Fail($msg) {
    Write-Host "`n[配置失败] $msg" -ForegroundColor Red
    Write-Host "请把以上红字截图反馈。常见原因：网络不通、被防火墙拦截。" -ForegroundColor Yellow
    exit 1
}

function Get-CommandPath($name) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source } else { return $null }
}

function Download($url, $dest) {
    Say "下载 $url"
    # Prefer curl.exe (built into Win10/11) — fast & reliable; fall back to IWR.
    $curl = Get-CommandPath 'curl.exe'
    if ($curl) {
        & $curl -L --fail --silent --show-error -o $dest $url
        if ($LASTEXITCODE -ne 0) { throw "curl 下载失败 (exit $LASTEXITCODE): $url" }
    } else {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    }
    if (-not (Test-Path $dest)) { throw "下载后文件不存在: $dest" }
}

# =========================================================================
# 1. Locate (or install) a base Python able to build the venv
# =========================================================================
function Resolve-BasePython {
    # a) portable Python we installed on a previous run
    $portable = Join-Path $Tools 'python\python.exe'
    if (Test-Path $portable) { return $portable }

    # b) a system Python (py launcher or python on PATH) — resolve to a real exe
    $py = Get-CommandPath 'py'
    if ($py) {
        try {
            $exe = (& $py -3 -c "import sys;print(sys.executable)" 2>$null)
            if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe)) { return $exe.Trim() }
        } catch {}
    }
    $python = Get-CommandPath 'python'
    if ($python -and $python -notlike '*WindowsApps*') { return $python }  # skip MS Store stub

    return $null
}

function Install-PortablePython {
    Step "未检测到 Python，正在自动下载安装（约 30MB，免管理员）..."
    New-Item -ItemType Directory -Force -Path $Tools | Out-Null
    $url     = "https://www.python.org/ftp/python/$PY_VERSION/python-$PY_VERSION-amd64.exe"
    $exe     = Join-Path $Tools "python-$PY_VERSION-amd64.exe"
    $target  = Join-Path $Tools 'python'
    Download $url $exe
    Say "静默安装 Python $PY_VERSION 到 tools\python ..."
    # Per-user install to a project-local dir — no admin, no PATH pollution.
    $pyArgs = @('/quiet','InstallAllUsers=0','PrependPath=0','Include_pip=1',
                'Include_test=0','Include_launcher=0','Include_doc=0',
                'SimpleInstall=1',"TargetDir=$target")
    $p = Start-Process -FilePath $exe -ArgumentList $pyArgs -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Python 安装器退出码 $($p.ExitCode)" }
    Remove-Item $exe -ErrorAction SilentlyContinue
    $portable = Join-Path $target 'python.exe'
    if (-not (Test-Path $portable)) { throw "安装后未找到 $portable" }
    return $portable
}

# =========================================================================
# 2. Create venv + install backend deps (idempotent via a hash marker)
# =========================================================================
function Ensure-Venv {
    Step "检查后端运行环境 (.venv) ..."
    if (-not (Test-Path $VenvPy)) {
        $base = Resolve-BasePython
        if (-not $base) { $base = Install-PortablePython }
        Say "用 $base 创建虚拟环境 .venv ..."
        # $base is a real python.exe path; call operator handles spaces in paths.
        & $base -m venv $Venv
        if (-not (Test-Path $VenvPy)) { throw "创建 .venv 失败" }
        Ok "虚拟环境已创建"
    } else {
        Ok "虚拟环境已存在"
    }
}

function Ensure-PyDeps {
    Step "检查后端依赖 ..."
    if (-not (Test-Path $ReqFile)) { throw "缺少 $ReqFile" }
    $marker = Join-Path $Venv '.deps.hash'
    $want   = (Get-FileHash $ReqFile -Algorithm SHA256).Hash
    $have   = if (Test-Path $marker) { Get-Content $marker -Raw } else { '' }
    if ($have.Trim() -eq $want) { Ok "依赖已是最新"; return }

    Say "安装/更新 Python 依赖（首次约 1-3 分钟）..."
    & $VenvPy -m pip install --upgrade pip --quiet
    & $VenvPy -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) { throw "pip install 失败" }
    Set-Content -Path $marker -Value $want -Encoding ascii
    Ok "后端依赖就绪"
}

# =========================================================================
# 3. Locate (or install) Node.js — exported via tools\node for the frontend
# =========================================================================
function Ensure-Node {
    Step "检查 Node.js ..."
    $portableNode = Join-Path $Tools 'node\node.exe'
    if (Test-Path $portableNode) {
        $env:PATH = (Join-Path $Tools 'node') + ';' + $env:PATH
        Ok "使用便携版 Node (tools\node)"
        return
    }
    $node = Get-CommandPath 'node'
    if ($node) { Ok "使用系统 Node ($node)"; return }

    Step "未检测到 Node.js，正在自动下载便携版（约 30MB，免安装）..."
    New-Item -ItemType Directory -Force -Path $Tools | Out-Null
    $name = "node-v$NODE_VERSION-win-x64"
    $url  = "https://nodejs.org/dist/v$NODE_VERSION/$name.zip"
    $zip  = Join-Path $Tools "$name.zip"
    Download $url $zip
    Say "解压 Node ..."
    Expand-Archive -Path $zip -DestinationPath $Tools -Force
    Remove-Item $zip -ErrorAction SilentlyContinue
    $extracted = Join-Path $Tools $name
    $final     = Join-Path $Tools 'node'
    if (Test-Path $final) { Remove-Item $final -Recurse -Force }
    Rename-Item -Path $extracted -NewName 'node'
    if (-not (Test-Path $portableNode)) { throw "Node 解压后未找到 node.exe" }
    $env:PATH = $final + ';' + $env:PATH
    Ok "便携版 Node 就绪 (tools\node)"
}

function Ensure-NodeModules {
    Step "检查前端依赖 (node_modules) ..."
    if (-not (Test-Path $StRoot)) { throw "缺少前端目录 $StRoot" }
    $nm = Join-Path $StRoot 'node_modules'
    if (Test-Path $nm) { Ok "前端依赖已存在"; return }

    Say "安装前端依赖（首次约 1-3 分钟）..."
    $portableNpm = Join-Path $Tools 'node\npm.cmd'
    $npm = if (Test-Path $portableNpm) { $portableNpm } else { 'npm' }
    Push-Location $StRoot
    try {
        & $npm install --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw "npm install 失败" }
    } finally { Pop-Location }
    Ok "前端依赖就绪"
}

# =========================================================================
# Run all steps
# =========================================================================
try {
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "        AIKP 首次启动环境自动配置" -ForegroundColor Cyan
    Write-Host "  （已配置过则秒过，无需手动安装任何东西）" -ForegroundColor DarkGray
    Write-Host "============================================" -ForegroundColor Cyan

    Ensure-Venv
    Ensure-PyDeps
    Ensure-Node
    Ensure-NodeModules

    Write-Host "`n[环境就绪] 准备启动游戏..." -ForegroundColor Green
    exit 0
}
catch {
    Fail $_.Exception.Message
}

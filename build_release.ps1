# =========================================================================
#  build_release.ps1 — assemble a fully self-contained, relocatable Windows
#  bundle of AIKP (portable Python + deps + portable Node + node_modules +
#  offline embedding model + a double-clickable AIKP.exe), then zip it.
#
#  Why not just ship a .venv? A venv hard-codes the absolute path of its base
#  Python, so it breaks the moment a player extracts to a different folder.
#  Instead we install deps DIRECTLY into a portable Python (tools\python),
#  which is relocatable when invoked as  python.exe server.py.
#
#  Usage:   powershell -ExecutionPolicy Bypass -File build_release.ps1 -Version v0.1.0
#  Output:  dist\AIKP-Portable-<version>-win64.zip   (and dist\AIKP\ staging)
# =========================================================================
[CmdletBinding()]
param([string]$Version = 'v0.1.0')

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'

$PY_VERSION   = '3.11.9'
$NODE_VERSION = '20.18.1'

$Repo  = $PSScriptRoot
$Dist  = Join-Path $Repo 'dist'
$Stage = Join-Path $Dist 'AIKP'
$Zip   = Join-Path $Dist "AIKP-Portable-$Version-win64.zip"
$Req   = Join-Path $Repo 'backend\requirements.txt'

function Step($m){ Write-Host "`n==== $m ====" -ForegroundColor Cyan }
function Say($m){  Write-Host "  $m" -ForegroundColor Gray }
function Ok($m){   Write-Host "  [OK] $m" -ForegroundColor Green }

function Download($url,$dest){
    Say "下载 $url"
    $curl = (Get-Command curl.exe -ErrorAction SilentlyContinue).Source
    if($curl){ & $curl -L --fail --silent --show-error -o $dest $url; if($LASTEXITCODE -ne 0){ throw "下载失败: $url" } }
    else     { Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing }
    if(-not (Test-Path $dest)){ throw "下载后文件不存在: $dest" }
}

# robocopy returns a bitmask; 0-7 = success, >=8 = real failure
function Robo($src,$dst,[string[]]$xd,[string[]]$xf){
    $a = @($src,$dst,'/E','/NFL','/NDL','/NJH','/NJS','/NP','/R:1','/W:1')
    if($xd){ $a += '/XD'; $a += $xd }
    if($xf){ $a += '/XF'; $a += $xf }
    robocopy @a | Out-Null
    if($LASTEXITCODE -ge 8){ throw "robocopy 失败 ($src -> $dst), code $LASTEXITCODE" }
    $global:LASTEXITCODE = 0
}

# ---------------------------------------------------------------- 0. clean
Step "准备打包目录 $Stage"
if(Test-Path $Stage){ Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
$tmp = Join-Path $Dist '_tmp'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# ------------------------------------------- 1. portable Python + backend deps
Step "安装便携 Python $PY_VERSION 并装后端依赖"
$pyDir = Join-Path $Stage 'tools\python'
$pyExe = Join-Path $pyDir 'python.exe'
$inst  = Join-Path $tmp "python-$PY_VERSION-amd64.exe"
Download "https://www.python.org/ftp/python/$PY_VERSION/python-$PY_VERSION-amd64.exe" $inst
$p = Start-Process -FilePath $inst -Wait -PassThru -ArgumentList @(
    '/quiet','InstallAllUsers=0','PrependPath=0','Include_pip=1',
    'Include_test=0','Include_launcher=0','Include_doc=0','SimpleInstall=1',
    "TargetDir=$pyDir")
if($p.ExitCode -ne 0){ throw "Python 安装器退出码 $($p.ExitCode)" }
if(-not (Test-Path $pyExe)){ throw "未找到 $pyExe" }
Say "pip install 依赖（chromadb/onnxruntime 较大，请耐心）..."
& $pyExe -m pip install --upgrade pip --quiet
& $pyExe -m pip install -r $Req
if($LASTEXITCODE -ne 0){ throw "pip install 失败" }
$hash = (Get-FileHash $Req -Algorithm SHA256).Hash
Set-Content -Path (Join-Path $pyDir '.aikp_deps.hash') -Value $hash -Encoding ascii
Ok "便携 Python 就绪"

# ------------------------------------------------------- 2. portable Node.js
Step "下载便携 Node.js $NODE_VERSION"
$nodeName = "node-v$NODE_VERSION-win-x64"
$nodeZip  = Join-Path $tmp "$nodeName.zip"
Download "https://nodejs.org/dist/v$NODE_VERSION/$nodeName.zip" $nodeZip
Expand-Archive -Path $nodeZip -DestinationPath $tmp -Force
Move-Item (Join-Path $tmp $nodeName) (Join-Path $Stage 'tools\node')
if(-not (Test-Path (Join-Path $Stage 'tools\node\node.exe'))){ throw "Node 解压失败" }
Ok "便携 Node 就绪"

# -------------------------------------------------- 3. copy the app source
# IMPORTANT: robocopy /XD with a BARE name excludes that name ANYWHERE in the
# tree. We must exclude the app's own top-level dirs by FULL PATH, otherwise
# e.g. excluding "dist"/"cache" would also delete every node_modules\**\dist
# and break packages like "yaml". Only __pycache__/.git are excluded by name
# (we genuinely want those gone recursively).
Step "拷贝程序文件"
$beSrc = Join-Path $Repo 'backend'
Robo $beSrc (Join-Path $Stage 'backend') `
     @('__pycache__', (Join-Path $beSrc 'sessions'), (Join-Path $beSrc 'uploads'),
       (Join-Path $beSrc 'world_book'), (Join-Path $beSrc '_chroma')) `
     @('_t.py','_test_parser.py','_read_xlsx.py','rebuild_backend.py')
$stSrc = Join-Path $Repo 'Tavern\SillyTavern'
Robo $stSrc (Join-Path $Stage 'Tavern\SillyTavern') `
     @('.git', (Join-Path $stSrc 'data'), (Join-Path $stSrc 'backups'),
       (Join-Path $stSrc 'cache'), (Join-Path $stSrc 'thumbnails'), (Join-Path $stSrc 'dist')) `
     @((Join-Path $stSrc 'config.yaml'))
# root files needed to run
$rootFiles = @('启动游戏.bat','停止游戏.bat','_aikp_backend.bat','_aikp_frontend.bat',
               '_aikp_setup.ps1','start.bat','start.ps1','.env.example',
               'README.md','LICENSE','NOTICE')
foreach($f in $rootFiles){
    $src = Join-Path $Repo $f
    if(Test-Path $src){ Copy-Item $src (Join-Path $Stage $f) -Force }
}
# empty models/ for the player's own world books
New-Item -ItemType Directory -Force -Path (Join-Path $Stage 'models') | Out-Null
Set-Content -Path (Join-Path $Stage 'models\.gitkeep') -Value '' -Encoding ascii
# ASCII shim that AIKP.exe launches (avoids a Chinese filename in the exe)
$shim = "@echo off`r`nchcp 65001 >nul`r`ncd /d `"%~dp0`"`r`ncall `"%~dp0启动游戏.bat`"`r`n"
[System.IO.File]::WriteAllText((Join-Path $Stage '_aikp_launch.bat'), $shim, (New-Object System.Text.UTF8Encoding($false)))
Ok "程序文件已拷贝"

# --------------------------------------- 4. precache offline embedding model
Step "预缓存语义检索模型（离线用）"
$warm = Join-Path $tmp 'warm.py'
Set-Content -Path $warm -Encoding ascii -Value @'
import chromadb.utils.embedding_functions as ef
fn = ef.ONNXMiniLM_L6_V2()
fn(["warmup"])
print("model ready")
'@
try {
    & $pyExe $warm
    if($LASTEXITCODE -ne 0){ throw "warmup exit $LASTEXITCODE" }
    $modelSrc = Join-Path $env:USERPROFILE '.cache\chroma\onnx_models\all-MiniLM-L6-v2'
    if(Test-Path (Join-Path $modelSrc 'onnx\model.onnx')){
        $modelDst = Join-Path $Stage 'runtime_cache\chroma_onnx\all-MiniLM-L6-v2'
        New-Item -ItemType Directory -Force -Path $modelDst | Out-Null
        Copy-Item (Join-Path $modelSrc '*') $modelDst -Recurse -Force
        Ok "离线模型已打包"
    } else { Write-Host "  [警告] 未找到下载的模型，跳过（玩家首次索引时会联网下载）" -ForegroundColor Yellow }
} catch {
    Write-Host "  [警告] 模型预缓存失败：$($_.Exception.Message)（不致命，玩家首次索引时会联网下载）" -ForegroundColor Yellow
}

# ----------------------------------------------- 5. compile AIKP.exe launcher
Step "编译 AIKP.exe 启动器"
$csc = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if(Test-Path $csc){
    # Pure-ASCII launcher source (references the ASCII shim, not a Chinese name).
    $csFile = Join-Path $tmp 'AIKP.cs'
    Set-Content -Path $csFile -Encoding ascii -Value @'
using System;
using System.Diagnostics;
using System.IO;
class AikpLauncher {
    static int Main() {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string bat = Path.Combine(dir, "_aikp_launch.bat");
        if (!File.Exists(bat)) {
            Console.Error.WriteLine("Cannot find _aikp_launch.bat next to AIKP.exe.");
            return 1;
        }
        try {
            Process.Start(new ProcessStartInfo {
                FileName = bat, WorkingDirectory = dir, UseShellExecute = true
            });
            return 0;
        } catch (Exception e) {
            Console.Error.WriteLine(e.Message);
            return 1;
        }
    }
}
'@
    $exeOut = Join-Path $Stage 'AIKP.exe'
    & $csc /nologo /target:winexe "/out:$exeOut" $csFile
    if($LASTEXITCODE -ne 0 -or -not (Test-Path $exeOut)){ throw "csc 编译失败" }
    Ok "AIKP.exe 已生成"
} else {
    Write-Host "  [警告] 未找到 csc.exe，跳过 exe（玩家可直接双击 启动游戏.bat）" -ForegroundColor Yellow
}

# ----------------------------------------------------------------- 6. zip it
Step "打包为 zip"
if(Test-Path $Zip){ Remove-Item $Zip -Force }
$tar = (Get-Command tar.exe -ErrorAction SilentlyContinue).Source
if($tar){ & $tar -a -c -f $Zip -C $Dist 'AIKP'; if($LASTEXITCODE -ne 0){ throw "tar 打包失败" } }
else    { Compress-Archive -Path $Stage -DestinationPath $Zip -Force }
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue

$sizeMB = [math]::Round(((Get-Item $Zip).Length/1MB),1)
$stageMB= [math]::Round(((Get-ChildItem $Stage -Recurse -Force | Measure-Object Length -Sum).Sum/1MB),1)
Write-Host "`n========================================" -ForegroundColor Green
Write-Host " 打包完成" -ForegroundColor Green
Write-Host "   解压后大小: $stageMB MB" -ForegroundColor Green
Write-Host "   压缩包:     $Zip  ($sizeMB MB)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

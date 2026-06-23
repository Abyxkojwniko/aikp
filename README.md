# AIKP — AI 跑团主持

AI 驱动的 TRPG（CoC / D&D）游戏主持人（KP/GM）。前端为定制版 SillyTavern，后端为 FastAPI，
通过 OpenAI 兼容接口对接 LLM。把跑团模组解析成结构化世界书后，以「确定性状态机 + LLM 叙事」的
方式主持一局游戏。

> 模组内容（商业版权）不随仓库分发，请自行准备并放入 `models/`。

## 下载（最省事：整合包）

到 [Releases](../../releases) 下载 `AIKP-Portable-*-win64.zip`（约 430MB），解压后：

1. 复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key（在 https://platform.deepseek.com/ 获取）。
2. **双击 `AIKP.exe`**（或 `启动游戏.bat`）。

整合包内置便携版 Python、Node、全部依赖和离线语义模型，**无需联网配置环境、无需装任何东西**，解压即玩。

如果你更想从源码运行（仓库体积小、自动联网配置），见下面「快速开始」。

## 快速开始（Windows，源码 + 零配置）

1. 下载 / 克隆本仓库。
2. 复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key（在 https://platform.deepseek.com/ 获取）。
3. **双击 `启动游戏.bat`**。

就这样。首次启动会**自动**完成所有环境配置——无需手动安装 Python、Node 或任何依赖：

- 检测不到 Python → 自动下载便携版到 `tools\python\`（免管理员）
- 检测不到 Node.js → 自动下载便携版到 `tools\node\`
- 自动创建虚拟环境 `.venv` 并安装后端依赖
- 自动为前端执行 `npm install`
- 全部就绪后自动打开浏览器到 http://127.0.0.1:8000

首次配置需要联网，可能花几分钟下载（取决于网速）；之后每次启动都很快。
结束游戏：双击 **`停止游戏.bat`**。

> 想强制重新配置环境？删除项目下的 `.venv` 和 `tools` 文件夹，再双击 `启动游戏.bat` 即可。

## 手动安装（进阶 / 非 Windows）

如果你想自己管理环境（或在 macOS / Linux 上运行）：

```bash
git clone <your-repo-url> aikp
cd aikp

# 1) 后端依赖（任选 venv 或 conda）
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r backend/requirements.txt

# 2) 前端依赖（SillyTavern，需要 Node.js 20+）
cd Tavern/SillyTavern
npm install
cd ../..

# 3) 配置 API Key
cp .env.example .env            # 然后编辑 .env，填入你的 DEEPSEEK_API_KEY
```

手动启动（两个终端）：

```bash
# 终端 1 — 后端
.venv/Scripts/python backend/server.py        # http://localhost:8001

# 终端 2 — 前端
cd Tavern/SillyTavern && node server.js        # http://localhost:8000
```

打开浏览器访问 http://localhost:8000 即可开始。

## 环境要求

- 仅 Windows 一键启动：**无需预装任何东西**（脚本会自动下载便携版 Python 3.11 与 Node.js 20）。
- 手动安装：Python 3.10+、Node.js 20+。
- 一个 DeepSeek API Key（OpenAI 兼容）。

## 配置说明

API Key 解析顺序：请求头 `Authorization` → 环境变量 `DEEPSEEK_API_KEY`（`.env`） → SillyTavern secrets。

| 变量 | 默认 |
|---|---|
| `DEEPSEEK_API_KEY` | （必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | `deepseek-chat` |

## 目录

```
启动游戏.bat            一键启动（首次自动配置环境）
停止游戏.bat            关闭所有服务
_aikp_setup.ps1         环境自动配置脚本（被启动脚本调用）
backend/                后端（FastAPI + 游戏引擎 + 模组解析器）
Tavern/SillyTavern/     定制版前端（含 public/scripts/extensions/aikp 扩展）
models/                 放置你的世界书（不随仓库分发）
.venv/  tools/          自动生成的运行环境（已 gitignore，不随仓库分发）
```

## 许可

本仓库包含两部分，分别授权（详见根目录 `NOTICE`）：

- **`backend/` 及启动脚本（原创）** — Apache License 2.0（见 `LICENSE`）
- **`Tavern/SillyTavern/`（修改版 SillyTavern 前端）** — AGPL-3.0（见 `Tavern/SillyTavern/LICENSE`）

两部分以独立进程通过 HTTP 通信；Apache-2.0 仅覆盖原创后端，不覆盖捆绑的 SillyTavern 前端。

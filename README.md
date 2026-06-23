# AIKP — AI 跑团主持

AI 驱动的 TRPG（CoC / D&D）游戏主持人（KP/GM）。前端为定制版 SillyTavern，后端为 FastAPI，
通过 OpenAI 兼容接口对接 LLM。把跑团模组解析成结构化世界书后，以「确定性状态机 + LLM 叙事」的
方式主持一局游戏。

> 模组内容（商业版权）不随仓库分发，请自行准备并放入 `models/`。

## 环境要求

- Python 3.10+ （建议用 conda 创建名为 `aikp` 的环境）
- Node.js 18+ （SillyTavern 前端需要）
- 一个 DeepSeek API Key（OpenAI 兼容）：https://platform.deepseek.com/

## 安装

```bash
git clone <your-repo-url> aikp
cd aikp

# 1) 后端依赖
conda create -n aikp python=3.10 -y
conda activate aikp
pip install -r backend/requirements.txt

# 2) 前端依赖（SillyTavern）
cd Tavern/SillyTavern
npm install
cd ../..

# 3) 配置 API Key
cp .env.example .env          # 然后编辑 .env，填入你的 DEEPSEEK_API_KEY
```

## 运行

点击启动游戏.bat

手动启动（两个终端）：

```bash
# 终端 1 — 后端
conda activate aikp && cd backend && python server.py        # http://localhost:8001

# 终端 2 — 前端
cd Tavern/SillyTavern && node server.js                       # http://localhost:8000
```

打开浏览器访问 http://localhost:8000 即可开始。

## 配置说明

API Key 解析顺序：请求头 `Authorization` → 环境变量 `DEEPSEEK_API_KEY`（`.env`） → SillyTavern secrets。

| 变量 | 默认 |
|---|---|
| `DEEPSEEK_API_KEY` | （必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | `deepseek-chat` |

## 目录

```
backend/                后端（FastAPI + 游戏引擎 + 模组解析器）
Tavern/SillyTavern/     定制版前端（含 public/scripts/extensions/aikp 扩展）
models/                 放置你的世界书（不随仓库分发）
```

## 许可

本仓库包含两部分，分别授权（详见根目录 `NOTICE`）：

- **`backend/` 及启动脚本（原创）** — Apache License 2.0（见 `LICENSE`）
- **`Tavern/SillyTavern/`（修改版 SillyTavern 前端）** — AGPL-3.0（见 `Tavern/SillyTavern/LICENSE`）

两部分以独立进程通过 HTTP 通信；Apache-2.0 仅覆盖原创后端，不覆盖捆绑的 SillyTavern 前端。

# AIKP — AI 跑团主持

作者：[@Abyxkojw](https://github.com/Abyxkojwniko)

AI 驱动的 TRPG（CoC / D&D）游戏主持人（KP/GM）。前端为定制版 SillyTavern，后端为 FastAPI，
通过 OpenAI 兼容接口对接 LLM。把跑团模组解析成结构化世界书后，以「AI 语义理解 +
事实状态/事件验证 + LLM 叙事」的方式主持游戏。

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
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` |

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

## 运行时边界

- AI 仅从「当前可见对象 + 背包对象」的封闭候选集中解析玩家动作。
- 代码验证位置、可见性、持有关系、锁和道具等前置条件。
- 验证后的变化写入 `world_events`，当前世界事实从事件与 `entity_facts` 得到。
- 旧模组的 `states/triggers` 仅作为明确检定和原文分支的兼容规则。
- NPC 对话必须选择交谈对象；普通 object 可选择以消除歧义，也可用自然语言指定。
- 场景导航只暴露当前可达出口；玩家可选择稳定场景 ID，再用“去那里”等自然表达移动。
- 会话分别记录 `discovered_scene_ids`、`visited_scene_ids` 和 `selected_scene_id`，隐藏或未解锁出口不会进入候选。
- 解析器优先通读完整模组，先建立粗粒度故事支柱、实体注册表和分支树，再按节点回到原文取证并重建高细节场景；每个节点按原文忠实度、细节、因果、状态、分支和叙事层级评分，低于阈值会携带缺陷报告重试。书中故事、回忆、梦境等不可进入层级不会误入地图。超出 `AIKP_FULL_REBUILD_MAX_CHARS` 的文档才使用旧兼容流程，且不会静默截断结尾。
- 普通物品是带 `home_scene` 的场景实例，同名门、书柜等不会跨房间合并；运行时动态位置账本覆盖静态场景原文，防止已移动物品被叙述回原处。

## 许可

本仓库包含两部分，分别授权（详见根目录 `NOTICE`）：

- **`backend/` 及启动脚本（原创）** — Apache License 2.0（见 `LICENSE`）
- **`Tavern/SillyTavern/`（修改版 SillyTavern 前端）** — AGPL-3.0（见 `Tavern/SillyTavern/LICENSE`）

两部分以独立进程通过 HTTP 通信；Apache-2.0 仅覆盖原创后端，不覆盖捆绑的 SillyTavern 前端。

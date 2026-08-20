# AIKP — AI 跑团主持

作者：[@Abyxkojw](https://github.com/Abyxkojwniko)

AI 驱动的 TRPG 游戏主持人（KP/GM）。前端为定制版 SillyTavern，后端为 FastAPI，
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
| `DEEPSEEK_MODEL` | `deepseek-chat` |
| `AIKP_NARRATIVE_AUDIT` | `strict`（`always` 为失效开放，`off` 可关闭） |
| `AIKP_NARRATIVE_AUDITOR_MODEL` | 与 `DEEPSEEK_MODEL` 相同 |

叙事审计默认使用严格模式，每个需要 LLM 叙述的回合会增加一次独立审计调用；发现冲突时还会增加一次定向修复调用。在线审计不可用或返回无效 JSON 时，严格模式会退回仅由已提交事实生成的保守叙述。离线测试会明确记录为 `skipped_offline`。

### Grounded runtime 方法

当前运行时采用 `propose -> validate -> commit -> narrate -> audit -> repair/fallback`：模型提出动作解释与叙述，代码只提交封闭世界中通过校验的事件，叙述不能直接写 SAN、信任、位置、物品或 NPC 生命周期。实现思路借鉴了以下工作的评价问题，但不声称复现其完整方法或达到其榜单结果：

- [Orchestrated Reality](https://arxiv.org/html/2606.16014)：将生成、工具执行、验证和修复拆成可审计阶段。
- [NCP-Bench](https://arxiv.org/html/2608.08160)：把跨长程的 setup/payoff 作为显式 narrative commitment 持续追踪。
- [WSE-bench](https://arxiv.org/html/2608.15654)：用“既有事实、当前冲突、缺失的调和事件”检查世界状态矛盾。

世界事件按批次在副本上归约，通过类型、位置、所有权与生命周期不变量后一次提交，并记录前后状态哈希。动作解析、场景进入和玩家掷骰也使用同一事务边界；任何后置事件失败都会回滚整次行动。

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
- 同一动作产生的多个变化先在副本上执行并检查世界不变量，全部通过才原子提交；每批事件共享 `commit_id`，并记录提交前后的规范状态哈希，失败不会留下半个动作。
- LLM 叙事生成后会与当前场景、实体位置/生死/开锁状态、已提交事件和剧情承诺进行冲突审计；冲突答案只修复一次，仍不一致则使用仅由已提交事实生成的回退叙述。
- 旧模组的 `states/triggers` 仅作为明确检定和原文分支的兼容规则。
- NPC 对话必须选择交谈对象；普通 object 可选择以消除歧义，也可用自然语言指定。
- 场景导航只暴露当前可达出口；玩家可选择稳定场景 ID，再用“去那里”等自然表达移动。
- 会话分别记录 `discovered_scene_ids`、`visited_scene_ids` 和 `selected_scene_id`，隐藏或未解锁出口不会进入候选。
- 解析器优先通读完整模组，先建立粗粒度故事支柱、实体注册表和分支树，再按节点回到原文取证并重建高细节场景；超长规则书/模组合集先逐窗口生成带原文编号的文档地图，再综合全局树，不再因超过单次上下文上限直接退回旧分段解析。每个节点按原文忠实度、细节、因果、状态、分支和叙事层级评分，低于阈值会携带缺陷报告重试。书中故事、回忆、梦境等不可进入层级不会误入地图。
- 多冒险合集保留独立 `scenario_id` 子树；跨故事剧情边会使重建质量门失败。场景、NPC、物品和线索在运行时自动使用故事命名空间，同名房间或角色不会互相覆盖。前端仅在合集世界中显示冒险选择菜单，切换会建立隔离的新会话并保留已导入的角色卡。
- 世界书分别记录具体 `ruleset` 与 `dice_system`；解析器可识别 CoC、D&D、RuneQuest、BRP、Dragonbane、Pendragon、7th Sea 和 GUMSHOE。当前自动技能检定适配到 Pendragon 为止，7th Sea/GUMSHOE 会明确标记为尚无自动检定适配，而不会伪装成 D&D。具体角色卡与完整战斗规则仍以所用系统适配器为准。
- 普通物品是带 `home_scene` 的场景实例，同名门、书柜等不会跨房间合并；运行时动态位置账本覆盖静态场景原文，防止已移动物品被叙述回原处。

## 许可

本仓库包含两部分，分别授权（详见根目录 `NOTICE`）：

- **`backend/` 及启动脚本（原创）** — Apache License 2.0（见 `LICENSE`）
- **`Tavern/SillyTavern/`（修改版 SillyTavern 前端）** — AGPL-3.0（见 `Tavern/SillyTavern/LICENSE`）

两部分以独立进程通过 HTTP 通信；Apache-2.0 仅覆盖原创后端，不覆盖捆绑的 SillyTavern 前端。

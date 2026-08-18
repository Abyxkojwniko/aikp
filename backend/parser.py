# -*- coding: utf-8 -*-
"""AIKP Module Parser — LLM-driven text → world book parsing pipeline.

Multi-pass:
  Pass 0: Document overview (plot, mystery, key characters)
  Pass 1: Per-chunk entity extraction (scenes, NPCs, items, clues)
  Pass 2: Global relationship linking (scene graph, clue chains, NPC affiliations)
  Pass 3: Condition inference (skill checks, DCs, state triggers, event sequences)
  Merge: Combine results → world_book.json
"""

from __future__ import annotations

import json
import os
import time
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import DEEPSEEK_BASE_URL, WORLD_BOOK_DIR, BACKEND_DIR
from chunker import chunk_document, Chunk

UPLOADS_DIR = os.path.join(BACKEND_DIR, "uploads")


# ── Prompt Templates ──────────────────────────────────────────

PASS0_SYSTEM = """Analyze this TRPG module IN FULL. Extract a high-level summary.

IMPORTANT: Read the ENTIRE text carefully. The character introduction section lists NPCs by name, but they may appear later under descriptions (e.g. "the young man", "a tall figure"). You must identify ALL aliases.

Return ONLY valid JSON (no markdown, no explanation):
{
  "title": "module title",
  "mystery": "one sentence core mystery",
  "opening": "VERBATIM opening narration — find the section labeled 导入/开场/开幕 or the first scene where players start, then copy MULTIPLE PARAGRAPHS verbatim. Include: (1) the shared situation setup, (2) the first scene atmosphere, (3) initial NPC introductions. Copy at least 300 characters. Do NOT summarize — paste original Chinese text exactly.",
  "starting_scene": "the CHINESE NAME of the first main investigation/exploration location (NOT a transit point like a bus stop, market, train station, or waiting area). This is where players first interact with NPCs and search for clues — e.g. '黎明公馆门口' not '香槟集市候车点'",
  "npcs": [{"name":"正式名字","aliases":["别名1","描述性称呼"],"role":"职业/身份","brief":"一句话描述","first_scene":"首次出场的场景名"}],
  "locations": ["location names"],
  "phases": ["intro","investigation","climax","resolution"],
  "genre": "horror/investigation/fantasy",
  "rule_system": "dnd" or "coc"
}

NPC aliases: list ALL ways each character is referred to in the text. For example, if character 尾金星杉 is also called "年轻人" or "年轻男子" in scene descriptions, include those as aliases.
One-off unnamed characters (old woman who appears once, a random climber) should NOT be in this list — only recurring named characters.
rule_system detection: if the text mentions CoC/克苏鲁/d100/SAN/理智/幸运 -> "coc"; if D&D/DnD/d20/DC/AC -> "dnd"; default "dnd"."""

PASS1_SYSTEM = """Extract ALL scenes, NPCs, items, and clues from this TRPG module chunk.

SCENE DESC: Write 3-5 sentences. Include atmosphere, key visual details, and what players first notice. Copy key phrases verbatim from the original text where possible.
SCENE PURPOSE: One KP-only sentence — what this scene contributes to the overall mystery/story.
SCENE TYPE: "location" (can be revisited) or "event" (one-time occurrence).

For each CLUE players can find in a scene:
  "reveals": what piece of the mystery this exposes (KP perspective, e.g. "证明伯爵曾被谋杀")
  "points_to": scene_id this clue naturally leads players toward next (empty string if none)

SOURCE GROUNDING: Every scene, NPC, item, clue, and event must include
"source_quote": an exact 1-3 sentence quote copied from CHUNK TEXT that proves
the entry exists. Never paraphrase or invent a quote. Use an empty string when
the chunk contains no supporting sentence.

Return ONLY valid JSON:
{
  "scenes": [{"id":"short_english_slug","name":"中文名","desc":"3-5 sentence description","purpose":"one KP sentence","type":"location","source_quote":"exact quote"}],
  "npcs": [{"id":"...","name":"...","scene":"scene_id","profession":"...","public_label":"原文中玩家在未知姓名时实际看见的称呼，必须逐字出现于该场景原文且不得包含姓名或秘密身份；无法确认则留空","appearance":"...","personality":"...","dialogue":{"topic":"line"},"source_quote":"exact quote"}],
  "items": [{"id":"...","name":"...","scene":"scene_id","desc":"...","source_quote":"exact quote"}],
  "clues": [{"id":"...","name":"...","scene":"scene_id","desc":"...","check":"Skill DC","reveals":"...","points_to":"","source_quote":"exact quote"}],
  "events": [{"id":"...","name":"...","desc":"...","scene":"scene_id","trigger":"...","source_quote":"exact quote"}]
}"""

SCENE_COVERAGE_SYSTEM = """Audit source-backed module sections that were not mapped to scenes.
Select ONLY sections that are physical playable locations, or numbered solo-adventure
encounter/choice nodes that the player can actually enter during play.
Reject chapters, rules, credits, character sheets, handouts, endings, background lore,
and non-playable event descriptions.
You cannot create text or names. Return candidate_start values only from the supplied list.
Return ONLY valid JSON:
{"additions":[{"candidate_start":123,"kind":"location or event","reason":"brief classification reason"}]}
"""

PASS2_SYSTEM = """Given entities extracted from a TRPG module, do two things:

1. Build the scene graph (how scenes connect via exits)
2. Extract story beats — the ordered milestones of the investigation

A beat = a milestone players must reach to advance the story. Order them as they happen in the module.
Each beat: what players need to find (critical_clues), what the KP should know (kp_note), what unlocks next.

CRITICAL: In critical_clues and optional_clues, use ONLY the exact clue IDs from the "clue_index" field provided.
These are English slug strings like "count_body", "clue_hugo_plan". Do NOT use Chinese descriptions.
If a beat has no specific clue requirement, use "visited" as advance_when and leave critical_clues empty.

Return ONLY valid JSON:
{
  "scene_graph": {
    "scene_id": {"exits": {"移动关键词": "target_scene_id"}}
  },
  "story_beats": [
    {
      "id": "beat_slug",
      "name": "节拍名（如：初到公馆、温室调查、对峙管家）",
      "kp_note": "KP视角一两句：这一节拍的目标是什么，玩家应找到什么，之后故事往哪走",
      "scenes": ["scene_id"],
      "critical_clues": ["exact_clue_id_from_clue_index"],
      "optional_clues": ["exact_clue_id_from_clue_index"],
      "advance_when": "any_critical",
      "unlocks_scenes": ["scene_id"]
    }
  ]
}
Story beats must follow the module's actual narrative order. Use "visited" as advance_when when no specific clue is required."""

PASS3_SYSTEM = """You are a TRPG analyst. Using ONLY the extracted entities provided below, finalize the world book.
CRITICAL: DO NOT invent new entities. Only use what's in the input data.
For each entity, add game mechanics:
- If a clue needs a check, add: check="Skill DC" (e.g. "Perception 12")
- If there's a SAN check, add: san_check="1/1d3"  
- Add NPC dialogue, scene exits, item descriptions based on the text
Return valid JSON with keys: scenes, entities, plot_outline.
Entities should be a dict like {"entity_id": {"type":"npc","name":"...","scene":"...","profession":"...","appearance":"...","personality":"...","dialogue":{...}}}.
Preserve profession and appearance from input data. Output ONLY JSON, no markdown."""


PASS_MECHANICS_SYSTEM = """你是 TRPG 规则分析师。从下面的【模组原文】里提取所有「判定点」，关联到给定的实体清单，构造成实体状态机。这一步决定了游戏里玩家能不能真正掷骰子，非常重要，请尽量找全。

判定点包括：
- 技能检定：〈侦查〉〈聆听〉〈幸运〉〈攀爬〉、DEX*5、力量检定 等
- SAN/理智检定：SANcheck X/Y、减少 Nd SAN、理智检定失败 等
- 成败分支：若成功→A，若失败→B

把每个判定点关联到下面清单里的某个【实体id】（优先关联到具体的物品/线索/场景实体）。绝不新增清单外的实体。

输出 ONLY JSON，格式：
{
  "实体id": {
    "initial_state": "起始状态名(如 hidden/present/locked/sealed)",
    "states": {
      "起始状态名": {
        "triggers": ["玩家会说的动作关键词,如 侦查/搜索/查看/喝/饮用/攀爬/阅读"],
        "check": "技能名 难度" (CoC 只填技能名即可,如 "侦查";若原文有数字难度就带上,如 "侦查 50"),
        "san_check": "成功损失/失败损失" (如 "0/1" "1/1d3" "0/1d100"),
        "on_pass": {"to_state":"成功后状态名","narration":"成功时发生什么(照原文,不要编造)"},
        "on_fail": {"to_state":"失败后状态名","narration":"失败时发生什么(照原文)"}
      }
    }
  }
}
规则：
- check 和 san_check 按需出现：纯 SAN 判定只填 san_check；纯技能判定只填 check；两者都涉及就都填。
- narration 必须照原文描述，不要编造内容。
- 只提取原文明确写了的判定；原文没判定的实体不要出现在输出里。
- triggers 是玩家可能说出口的动作词，方便引擎匹配。"""


PASS_STORYLINE_SYSTEM = """你是 TRPG 剧情分析师。模组里有些 NPC 会**贯穿多个场景/节拍**（比如一起行动的同伴、反复出现的反派）。请从【模组原文】里，为每个给定的 NPC 提取他们的【故事线】——也就是这个 NPC 在剧情推进的各个阶段，分别做什么、想什么、揭示什么。

这是给 KP 用的，目的是让 KP 知道每个 NPC 的来龙去脉和走向，从而主动、连贯地扮演他们（而不是每个场景都把 NPC 当陌生人）。

要求：
- 只为清单里的 NPC 提取，绝不新增清单外的人。
- 故事线按剧情顺序排列，每一段关联到一个【节拍id】（用给定的 story_beats 的 id）或【场景id】。
- 每段写这个 NPC 在该阶段的**具体行为/动机/关键台词或转变**（KP视角，忠于原文，不要编造）。
- 如果某 NPC 只在一个场景出现且没有跨场景弧线，给一段即可。
- 包含 NPC 的**隐藏真相/最终结局**（如果原文写了），标注为 KP 机密。

输出 ONLY JSON，格式：
{
  "npc_id": {
    "arc": [
      {"beat": "节拍id或场景id", "does": "该阶段这个NPC做什么/想什么/揭示什么（忠于原文）"}
    ],
    "secret": "这个NPC的隐藏真相或最终结局（原文写了才填，KP机密；没有就留空）"
  }
}
规则：npc_id 必须用清单里给定的 id。arc 按剧情先后排序。does 忠于原文，不编造。"""


PASS_NPC_STYLE_SYSTEM = """你是 TRPG 角色分析师。根据每个 NPC 的【人设/性格描述】，判定他们的【说话风格 style】，并为他们的每个【对话话题】判定【需要的信任度 dialogue_trust】。

这是给 KP 用的：风格决定 KP 扮演该 NPC 时话的多少和语气；信任度决定玩家要多熟才能从该 NPC 嘴里问出这个话题。

【style】三项都必须从下列固定取值里选（不得自创、不得用中文）：
- verbosity（话多少）：many_words（健谈/滔滔不绝）| normal（正常）| few_words（寡言/简练，1-2句）| grunt（几乎只用单字/点头/哼）
- tone（语气）：cheerful（开朗愉快）| nervous（紧张焦虑）| gruff（冷硬粗暴/低沉）| academic（严谨理性/学术拘谨）| neutral（中性）
- initiative（主动性）：active（主动搭话/自来熟）| passive（被动，被问才答）

【dialogue_trust】：为每个给定话题给一个 0-100 的整数。粗略基准（再按性格与话题私密程度微调，越私密越高）：
- 寒暄/问候/公开信息：0
- 一般话题：10-20
- 个人/家庭：约 25
- 恐惧/弱点/创伤：约 30
- 秘密/过去/隐私：50 或更高

输出 ONLY JSON，格式：
{
  "npc_id": {
    "style": {"verbosity":"...","tone":"...","initiative":"..."},
    "dialogue_trust": {"话题名": 整数}
  }
}
规则：npc_id 用清单里给定的 id；style 三项只能用上面列出的英文取值；dialogue_trust 的话题名必须与给定话题完全一致；只判定清单里的 NPC，不新增。"""


# ── Parser Class ──────────────────────────────────────────────

class ModuleParser:
    def __init__(self, api_key: str, base_url: str = DEEPSEEK_BASE_URL, model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._last_error = None  # set when an LLM call fails

    def _llm(self, system: str, user: str, temperature: float = 0.3,
             max_tokens: int = 4096, json_mode: bool = False) -> str:
        try:
            print(f"[PARSER:LLM] Calling {self.model} with {len(user)} chars (max_tokens={max_tokens})...", flush=True)
            kwargs = {}
            if json_mode:
                # Force valid JSON from the API — avoids intermittent literal
                # newlines / unescaped quotes inside string values that break
                # _parse_json recovery.
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs,
            )
            result = resp.choices[0].message.content or ""
            print(f"[PARSER:LLM] Response: {len(result)} chars, starts: {result[:80]}", flush=True)
            return result
        except Exception as e:
            msg = str(e)[:200]
            print(f"[PARSER:LLM] ERROR: {msg}", flush=True)
            # Store error for status reporting; return empty so pipeline continues
            self._last_error = msg
            return ""

    def _parse_json(self, raw: str) -> dict:
        raw = raw.strip()
        # Catch markdown code fences first
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if m:
            raw = m.group(1)
        else:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                raw = m.group()

        # Try multiple parse strategies
        for strategy, result in self._try_parse(raw):
            if result is not None:
                return result
            print(f"[PARSER:JSON] {strategy} failed, trying next...", flush=True)

        print(f"[PARSER:JSON] All strategies failed, raw[:200]: {raw[:200]}", flush=True)
        return {}

    def _try_parse(self, raw: str):
        """Generator of (strategy_name, parsed_dict_or_None)."""
        # Strategy 1: direct parse
        try:
            yield ("direct", json.loads(raw))
            return
        except json.JSONDecodeError:
            pass

        # Strategy 2: fix trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', raw)
        if fixed != raw:
            try:
                yield ("trailing_comma", json.loads(fixed))
                return
            except json.JSONDecodeError:
                pass

        # Strategy 3: truncate at last complete structure
        # Walk backwards to find the last valid JSON token boundary
        depth = 0
        in_string = False
        escape = False
        last_good = len(raw)
        for i, ch in enumerate(raw):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch in '{[':
                    depth += 1
                elif ch in '}]':
                    depth -= 1
                    if depth <= 0:
                        last_good = i + 1
        if last_good < len(raw):
            truncated = raw[:last_good]
            if depth > 0:
                truncated += '}' * depth  # close unclosed objects
            try:
                yield ("truncated", json.loads(truncated))
                return
            except json.JSONDecodeError:
                # Try closing arrays too
                truncated2 = raw[:last_good]
                for ch in reversed(truncated2):
                    if ch == '[':
                        truncated2 += ']'
                    elif ch == '{':
                        truncated2 += '}'
                    else:
                        break
                try:
                    yield ("truncated2", json.loads(truncated2))
                    return
                except json.JSONDecodeError:
                    pass

        yield ("all_failed", None)

    # ── Pass 0: Overview ───────────────────────────────────────

    def pass0_overview(self, text: str, max_chars: int = 60000) -> dict:
        snippet = text[:max_chars]
        # If the actual game text (正文/导入) section starts beyond the initial snippet,
        # append it so the LLM can extract a proper opening.
        game_text_hint = ""
        for marker in ("正文\n导入", "正文\n开场", "\n导入\n", "\n开场\n", "\n开幕\n"):
            idx = text.find(marker, max_chars // 2)  # search in second half
            if idx > 0 and idx < len(text):
                # Append the game text section (up to 3000 chars)
                game_text_hint = f"\n\n=== 正文/导入（游戏文本）===\n{text[idx:idx+3000]}"
                print(f"[PARSER:0] Found game text section at char {idx}, appending hint", flush=True)
                break
        raw = self._llm(PASS0_SYSTEM,
                        f"Analyze this TRPG module:\n\n{snippet}{game_text_hint}",
                        max_tokens=4096)
        result = self._parse_json(raw)

        # Pass 0.5: Code-extract ■登场人物 as verification source
        code_chars = _extract_character_section(text)
        if code_chars:
            result["_code_characters"] = code_chars
            print(f"[PARSER:0.5] Code-extracted {len(code_chars)} "
                  f"character entries from text", flush=True)

        return result

    # ── Pass 1: Per-chunk entity extraction ────────────────────

    def pass1_extract_chunk(self, chunk: Chunk, overview: dict, chunk_idx: int, total_chunks: int) -> dict:
        ctx = f"This is chunk {chunk_idx + 1}/{total_chunks}: {chunk.title}\n"
        if overview:
            ctx += f"Module: {overview.get('title', 'Unknown')}\n"
            ctx += f"Mystery: {overview.get('mystery', '')}\n"
            npcs = overview.get("npcs", [])
            if npcs:
                ctx += ("=== CANONICAL NPC ROSTER (use these names) ===\n"
                        + json.dumps(npcs, ensure_ascii=False, indent=1)
                        + "\nWhen you encounter a character matching "
                        "a roster entry (by alias or description), "
                        "use that entry's canonical name.\n")
        ctx += f"\n--- CHUNK TEXT ---\n{chunk.text}"

        raw = self._llm(PASS1_SYSTEM, ctx, temperature=0.3)
        result = self._parse_json(raw)
        _bind_chunk_provenance(result, chunk)
        return result

    def pass1_extract_all(self, chunks: list[Chunk], overview: dict) -> list[dict]:
        results = []
        for i, chunk in enumerate(chunks):
            print(f"  Pass 1: chunk {i+1}/{len(chunks)} ({chunk.char_count}c)", flush=True)
            r = self.pass1_extract_chunk(chunk, overview, i, len(chunks))
            r["_chunk_index"] = i
            results.append(r)
            time.sleep(0.5)  # Rate limit avoidance
        return results

    # ── Pass 2: Global linking ─────────────────────────────────

    def pass2_link(self, pass1_results: list[dict], overview: dict) -> dict:
        all_scenes, all_npcs, all_clues = [], [], []
        for r in pass1_results:
            all_scenes.extend(r.get("scenes", []))
            all_npcs.extend(r.get("npcs", []))
            all_clues.extend(r.get("clues", []))

        # Compact clue index: scene_id → [{id, check, reveals}]
        # LLM MUST use these exact IDs in story_beats.critical_clues
        clue_index: dict[str, list] = {}
        for c in all_clues:
            sid = c.get("scene", "")
            cid = c.get("id", "")
            if sid and cid:
                clue_index.setdefault(sid, []).append({
                    "id": cid,
                    "check": c.get("check", ""),
                    "reveals": c.get("reveals", "")[:60],
                })
        # Also collect clues embedded in scene objects
        for s in all_scenes:
            sid = s.get("id", "")
            for c in s.get("clues", []):
                cid = c.get("id", "")
                if cid and sid:
                    existing = [e["id"] for e in clue_index.get(sid, [])]
                    if cid not in existing:
                        clue_index.setdefault(sid, []).append({
                            "id": cid,
                            "check": c.get("check", ""),
                            "reveals": c.get("reveals", "")[:60],
                        })

        # Compact scene list (id + name + purpose only — no full descs)
        scene_list = [
            {"id": s.get("id"), "name": s.get("name"),
             "purpose": s.get("purpose", "")[:80]}
            for s in all_scenes if s.get("id")
        ]
        # Deduplicate by id (keep first)
        seen_ids: set = set()
        scene_list_dedup = []
        for s in scene_list:
            if s["id"] not in seen_ids:
                seen_ids.add(s["id"])
                scene_list_dedup.append(s)

        summary = {
            "overview": {
                "title": overview.get("title"),
                "mystery": overview.get("mystery"),
                "starting_scene": overview.get("starting_scene"),
                "npcs": [{"name": n.get("name"), "scene": n.get("scene"),
                           "role": n.get("profession", "")}
                          for n in all_npcs[:20]],
            },
            "scenes": scene_list_dedup,
            "clue_index": clue_index,
        }

        user = (f"Link these extracted entities and extract story beats.\n"
                f"USE ONLY the clue IDs from clue_index — never invent clue names.\n\n"
                f"{json.dumps(summary, ensure_ascii=False, indent=2)[:8000]}")
        raw = self._llm(PASS2_SYSTEM, user, temperature=0.2)
        return self._parse_json(raw)

    # ── Pass 3: Condition inference ────────────────────────────

    def pass3_finalize(self, pass1_results: list[dict], pass2_result: dict, overview: dict) -> dict:
        combined = {
            "overview": overview,
            "entities": {
                "scenes": _deduplicate_entities("scenes", pass1_results),
                "npcs": _deduplicate_entities("npcs", pass1_results),
                "items": _deduplicate_entities("items", pass1_results),
                "clues": _deduplicate_entities("clues", pass1_results),
            },
            "pass2": pass2_result,
        }
        user = f"Finalize this world book. Add game mechanics (skill checks, DCs, SAN checks).\n{json.dumps(combined, ensure_ascii=False, indent=2)[:8000]}"
        raw = self._llm(PASS3_SYSTEM, user, temperature=0.2, max_tokens=8192)
        return self._parse_json(raw)

    # ── Pass 1.5: Code-based Entity Deduplication ────────────

    def pass1_5_deduplicate(self, pass1_results: list[dict],
                            overview: dict) -> list[dict]:
        """Deduplicate NPCs across chunks using Pass 0 roster + aliases.

        Steps:
          1. Name + alias matching (code)
          2. Small LLM call for remaining unmatched (Pass 1.5b)
          3. Merge groups, remap scene references
"""

        all_npcs, all_scenes = [], []
        all_items, all_clues, all_events = [], [], []
        for r in pass1_results:
            all_npcs.extend(r.get("npcs", []))
            all_scenes.extend(r.get("scenes", []))
            all_items.extend(r.get("items", []))
            all_clues.extend(r.get("clues", []))
            all_events.extend(r.get("events", []))

        roster = overview.get("npcs", [])
        roster_map: dict[str, dict] = {}
        alias_to_canon: dict[str, str] = {}
        for entry in roster:
            name = entry.get("name", "")
            if not name:
                continue
            norm = _norm_name(name)
            roster_map[norm] = entry
            alias_to_canon[norm] = norm
            for alias in entry.get("aliases", []):
                a = _norm_name(alias)
                if a:
                    alias_to_canon[a] = norm

        # Also incorporate code-extracted characters (Pass 0.5)
        for ce in overview.get("_code_characters", []):
            ce_norm = _norm_name(ce.get("name", ""))
            if ce_norm and ce_norm in roster_map:
                roster_map[ce_norm]["_code_desc"] = ce.get("desc", "")

        # --- Step 1: Match by name OR alias ---
        groups: dict[str, list[dict]] = {}
        unmatched: list[dict] = []
        for npc in all_npcs:
            npc_name = npc.get("name", "")
            npc_norm = _norm_name(npc_name)
            canon = alias_to_canon.get(npc_norm)
            if not canon:
                for a_norm, c_norm in alias_to_canon.items():
                    if _names_match(npc_name, a_norm):
                        canon = c_norm
                        break
            if canon:
                groups.setdefault(canon, []).append(npc)
            else:
                unmatched.append(npc)

        print(f"[PARSER:1.5] Name/alias-matched "
              f"{sum(len(g) for g in groups.values())} NPCs → "
              f"{len(groups)} groups, {len(unmatched)} unmatched",
              flush=True)

        # --- Step 2 (1.5b): Small LLM call for remaining ---
        if unmatched and roster_map:
            resolved = self._resolve_aliases_llm(
                unmatched, roster_map, overview)
            still_unmatched = []
            for npc in unmatched:
                npc_name = npc.get("name", "")
                target = resolved.get(npc_name)
                if target:
                    target_norm = _norm_name(target)
                    if target_norm in roster_map:
                        groups.setdefault(target_norm, []).append(npc)
                        print(f"[PARSER:1.5b] LLM-matched "
                              f"'{npc_name}' → '{target}'",
                              flush=True)
                        continue
                still_unmatched.append(npc)
        else:
            still_unmatched = list(unmatched)

        # --- Step 3: Merge each group ---
        merged_npcs: list[dict] = []
        id_mapping: dict[str, str] = {}
        for canon_norm, entities in groups.items():
            rentry = roster_map.get(canon_norm)
            cname = (rentry["name"] if rentry
                     else entities[0].get("name", ""))
            cid = _pick_best_npc_id(cname, entities)
            merged = _merge_npc_group(cid, cname, entities, rentry)
            merged_npcs.append(merged)
            for ent in entities:
                old_id = ent.get("id", "")
                if old_id and old_id != cid:
                    id_mapping[old_id] = cid

        for npc in still_unmatched:
            merged_npcs.append(npc)
            print(f"[PARSER:1.5] Kept as one-off: "
                  f"'{npc.get('name')}'", flush=True)

        # --- Step 4: Remap NPC references in scenes ---
        _remap_scene_npcs(all_scenes, id_mapping)

        # --- Step 5: Dedup items/clues/events by ID ---
        merged_items = _dedup_list_by_id(all_items)
        merged_clues = _dedup_list_by_id(all_clues)
        merged_events = _dedup_list_by_id(all_events)

        print(f"[PARSER:1.5] Result: {len(merged_npcs)} NPCs "
              f"(was {len(all_npcs)}), "
              f"{len(all_scenes)} scenes", flush=True)

        return [{
            "scenes": all_scenes,
            "npcs": merged_npcs,
            "items": merged_items,
            "clues": merged_clues,
            "events": merged_events,
            "_id_mapping": id_mapping,
        }]

    def _resolve_aliases_llm(self, unmatched: list[dict],
                             roster_map: dict,
                             overview: dict) -> dict[str, str]:
        """Pass 1.5b: Small LLM call to resolve unmatched NPCs.

        Input: ~500 chars (roster + descriptions).
        Output: {unnamed_description: canonical_name}.
        """
        roster_lines = []
        for norm, entry in roster_map.items():
            name = entry.get("name", "")
            brief = entry.get("brief", "")
            aliases = entry.get("aliases", [])
            code_desc = entry.get("_code_desc", "")
            line = f"- {name}: {brief}"
            if aliases:
                line += f" (已知别名: {', '.join(aliases)})"
            if code_desc:
                line += f"\n  原文描述: {code_desc[:100]}"
            roster_lines.append(line)

        unmatched_lines = []
        for npc in unmatched:
            name = npc.get("name", "")
            pers = npc.get("personality", "")[:60]
            scene = npc.get("scene", "")
            unmatched_lines.append(
                f"- \"{name}\" (场景:{scene}, 性格:{pers})")

        prompt = _ALIAS_RESOLVE_PROMPT.format(
            roster="\n".join(roster_lines),
            unmatched="\n".join(unmatched_lines))

        print(f"[PARSER:1.5b] LLM alias resolution: "
              f"{len(unmatched)} unmatched, "
              f"{len(prompt)} chars input", flush=True)

        raw = self._llm(
            "You match unnamed character descriptions to named "
            "characters. Return ONLY valid JSON.",
            prompt, temperature=0.1, max_tokens=512)
        if not raw:
            return {}

        try:
            data = self._parse_json(raw)
            if isinstance(data, list):
                result = {}
                for item in data:
                    uname = item.get("unnamed", "")
                    cname = item.get("canonical_name")
                    conf = item.get("confidence", "low")
                    if uname and cname and conf in ("high", "medium"):
                        result[uname] = cname
                return result
            return {}
        except Exception:
            return {}

    # ── Pass 1.7: Source Text Binding ──────────────────────────

    def pass1_7_bind_source_text(self, pass1_results: list[dict],
                                 original_text: str) -> None:
        """Bind source text segments to scenes. Modifies in place."""
        segments = _split_text_segments(original_text)
        if not segments:
            print("[PARSER:1.7] No segments found", flush=True)
            return

        skip_titles = {"(preamble)", "■真相", "■登场人物"}
        scenes = pass1_results[0].get("scenes", [])
        npcs = pass1_results[0].get("npcs", [])
        npc_idx = {n.get("id", ""): n for n in npcs if n.get("id")}

        bound = 0
        for scene in scenes:
            best_seg, best_score = None, 0
            for seg in segments:
                if seg["title"] in skip_titles:
                    continue
                score = _score_segment(
                    scene.get("name", ""), scene.get("desc", ""),
                    scene.get("npcs", []), npc_idx, seg["text"])
                if score > best_score:
                    best_score = score
                    best_seg = seg
            if best_seg and best_score >= 3:
                scene["source_text"] = best_seg["text"]
                scene["source_start"] = best_seg.get("start", 0)
                if re.fullmatch(r'\d{1,3}', str(best_seg.get("title", ""))):
                    scene["source_section_number"] = int(best_seg["title"])
                bound += 1

        print(f"[PARSER:1.7] Bound {bound}/{len(scenes)} scenes "
              f"to source text", flush=True)

    def pass1_8_recover_scenes(self, pass1_results: list[dict],
                               original_text: str) -> None:
        """Recover omitted playable locations from a closed source-section set."""
        if not pass1_results:
            return
        scene_data = pass1_results[0].setdefault("scenes", [])
        candidates = _scene_coverage_candidates(original_text, scene_data)
        if not candidates:
            print("[PARSER:1.8] Scene coverage complete", flush=True)
            return

        numeric = [item for item in candidates if item.get("kind_hint") == "event"]
        ordinary = [item for item in candidates if item.get("kind_hint") != "event"]
        accepted: list[dict] = _apply_scene_coverage_repair(
            scene_data, numeric, {
                "additions": [
                    {"candidate_start": item["start"], "kind": "event"}
                    for item in numeric
                ],
            })
        for offset in range(0, len(ordinary), 20):
            batch = ordinary[offset:offset + 20]
            compact = [{
                "candidate_start": item["start"],
                "heading": item["name"],
                "excerpt": item["text"][:700],
            } for item in batch]
            try:
                raw = self._llm(
                    SCENE_COVERAGE_SYSTEM,
                    "Existing scenes:\n"
                    + json.dumps([
                        {"id": s.get("id"), "name": s.get("name")}
                        for s in scene_data
                    ], ensure_ascii=False)
                    + "\n\nUnmapped source candidates:\n"
                    + json.dumps(compact, ensure_ascii=False, indent=1),
                    temperature=0.1,
                    max_tokens=2048,
                )
                proposal = self._parse_json(raw)
            except Exception as exc:
                print(f"[PARSER:1.8] Coverage batch failed: {exc}", flush=True)
                continue
            accepted.extend(_apply_scene_coverage_repair(
                scene_data + accepted, batch, proposal))

        if accepted:
            scene_data.extend(accepted)
        print(
            f"[PARSER:1.8] Recovered {len(accepted)}/{len(candidates)} "
            "unmapped source sections as scenes",
            flush=True,
        )

    def pass1_9_recover_marked_entities(self, pass1_results: list[dict],
                                        original_text: str) -> None:
        """Recover explicit NPC/item markers that semantic extraction omitted."""
        if not pass1_results:
            return
        recovered = _recover_source_marked_entities(
            pass1_results[0], original_text)
        print(
            f"[PARSER:1.9] Recovered {recovered['npcs']} NPCs and "
            f"{recovered['items']} items from source markers",
            flush=True,
        )

    # ── Pass 3.5: Validation ──────────────────────────────────

    def pass_game_mechanics(self, text: str, world_book: dict) -> None:
        """Extract dice/SAN check points from the ORIGINAL module text and merge
        them into entity state machines, so checks are actually rollable in play.
        Pass 3 can't do this — it only sees extracted entities, not the text."""
        entities = world_book.get("entities", {})
        if not entities:
            return
        ent_list = [
            f'{eid}（{e.get("name","")}, {e.get("type","")}, 场景:{e.get("scene","")}）'
            for eid, e in entities.items() if isinstance(e, dict)
        ]
        scenes = world_book.get("scenes", {})
        scene_list = [f'{sid}（{s.get("name","")}）'
                      for sid, s in scenes.items() if isinstance(s, dict)]

        added = 0
        windows = list(_iter_text_windows(text))
        for window_index, window in enumerate(windows, start=1):
            print(
                f"[PARSER] Game mechanics window {window_index}/{len(windows)}",
                flush=True,
            )
            user = (
                f"== 模组原文（第 {window_index}/{len(windows)} 段）==\n{window}\n\n"
                f"== 实体清单（id（名字, 类型, 场景）） ==\n" + "\n".join(ent_list) + "\n\n"
                f"== 场景清单（id（名字）） ==\n" + "\n".join(scene_list) + "\n\n"
                f"只提取当前原文段明确出现的判定点，关联到上面的实体 id，按要求输出 JSON。"
            )
            raw = self._llm(
                PASS_MECHANICS_SYSTEM,
                user,
                temperature=0.1,
                max_tokens=8192,
            )
            result = self._parse_json(raw)
            if not isinstance(result, dict):
                print(
                    f"[PARSER] Game mechanics window {window_index}: no valid result",
                    flush=True,
                )
                continue

            for tid, sm in result.items():
                if not isinstance(sm, dict):
                    continue
                target = entities.get(tid) or scenes.get(tid)
                if not isinstance(target, dict):
                    continue
                states = sm.get("states")
                if isinstance(states, dict) and states:
                    before = len(target.get("states", {}))
                    target.setdefault("states", {}).update(states)
                    if sm.get("initial_state") and not target.get("initial_state"):
                        target["initial_state"] = sm["initial_state"]
                    elif not target.get("initial_state"):
                        target["initial_state"] = next(iter(states))
                    added += len(target["states"]) - before
        print(f"[PARSER] Game mechanics: added check/SAN states to {added} entities", flush=True)

    def pass_npc_storylines(self, text: str, world_book: dict) -> None:
        """Extract per-NPC storylines (arc across beats/scenes) from the ORIGINAL
        text, so the KP knows each NPC's trajectory and can play them coherently
        across the scenes they travel through. Stored as entity['storyline'].

        Focused on multi-scene NPCs (companions, recurring characters) but covers
        all NPCs. Like pass_game_mechanics, it needs the full text — extracted
        entities alone don't carry the arc."""
        entities = world_book.get("entities", {})
        npcs = {eid: e for eid, e in entities.items()
                if isinstance(e, dict) and e.get("type") == "npc"}
        if not npcs:
            return
        npc_list = [
            f'{eid}（{e.get("name","")}，出现场景：'
            f'{"、".join(e.get("all_scenes") or [e.get("scene","")])}）'
            for eid, e in npcs.items()
        ]
        beats = world_book.get("story_beats", [])
        beat_list = [f'{b.get("id","")}（{b.get("name","")}）'
                     for b in beats if isinstance(b, dict)]

        added = 0
        windows = list(_iter_text_windows(text))
        for window_index, window in enumerate(windows, start=1):
            print(
                f"[PARSER] NPC storylines window {window_index}/{len(windows)}",
                flush=True,
            )
            user = (
                f"== 模组原文（第 {window_index}/{len(windows)} 段）==\n{window}\n\n"
                f"== NPC 清单（id（名字，出现场景）） ==\n" + "\n".join(npc_list) + "\n\n"
                f"== 剧情节拍 story_beats（id（名字），按顺序） ==\n"
                + ("\n".join(beat_list) if beat_list else "（无，请用场景id作为阶段标记）") + "\n\n"
                f"只提取当前原文段明确出现的 NPC 故事线，按要求输出 JSON。"
            )
            raw = self._llm(
                PASS_STORYLINE_SYSTEM,
                user,
                temperature=0.1,
                max_tokens=8192,
                json_mode=True,
            )
            result = self._parse_json(raw)
            if not isinstance(result, dict):
                print(
                    f"[PARSER] NPC storylines window {window_index}: no valid result",
                    flush=True,
                )
                continue

            for nid, data in result.items():
                target = entities.get(nid)
                if not isinstance(target, dict) or not isinstance(data, dict):
                    continue
                arc = data.get("arc")
                if isinstance(arc, list) and arc:
                    existing = target.setdefault("storyline", [])
                    known = {
                        (step.get("beat", ""), step.get("does", ""))
                        for step in existing if isinstance(step, dict)
                    }
                    for step in arc:
                        if not isinstance(step, dict):
                            continue
                        key = (step.get("beat", ""), step.get("does", ""))
                        if key not in known:
                            existing.append(step)
                            known.add(key)
                            added += 1
                secret = data.get("secret")
                if secret and isinstance(secret, str) and not target.get("storyline_secret"):
                    target["storyline_secret"] = secret
        print(f"[PARSER] NPC storylines: added arcs to {added} NPCs", flush=True)

    def pass_npc_style(self, world_book: dict) -> None:
        """Determine each NPC's speaking style + per-topic trust thresholds with a
        single LLM pass at PARSE time, baked into static data (entity['style'] and
        entity['dialogue_trust']). This replaces the runtime keyword heuristics in
        npc_state (infer_style/default_trust), which stay only as a fallback for
        old world books or if this pass fails. Style at parse-time is more robust
        across modules than guessing from a hand-tuned keyword table at runtime."""
        entities = world_book.get("entities", {})
        npcs = {eid: e for eid, e in entities.items()
                if isinstance(e, dict) and e.get("type") == "npc"}
        if not npcs:
            return
        blocks = []
        for eid, e in npcs.items():
            topics = list((e.get("dialogue") or {}).keys())
            blocks.append(
                f'{eid}：名字={e.get("name","")}，职业={e.get("profession","")}\n'
                f'  性格：{(e.get("personality","") or "（无）")[:400]}\n'
                f'  对话话题：{("、".join(topics) if topics else "（无）")}'
            )
        user = ("== NPC 清单 ==\n" + "\n\n".join(blocks)
                + "\n\n请为每个 NPC 判定 style 和 dialogue_trust，按要求输出 JSON。")
        raw = self._llm(PASS_NPC_STYLE_SYSTEM, user, temperature=0.0,
                        max_tokens=4096, json_mode=True)
        result = self._parse_json(raw)
        if not isinstance(result, dict):
            print("[PARSER] NPC style: no valid result", flush=True)
            return

        _VERB = {"many_words", "normal", "few_words", "grunt"}
        _TONE = {"cheerful", "nervous", "gruff", "academic", "neutral"}
        _INIT = {"active", "passive"}
        styled = 0
        for nid, data in result.items():
            target = entities.get(nid)
            if not isinstance(target, dict) or not isinstance(data, dict):
                continue
            st = data.get("style")
            if isinstance(st, dict):
                # Clamp to the known vocabulary; fall back to safe defaults so a
                # bad enum can never break downstream behavior shaping.
                target["style"] = {
                    "verbosity": st.get("verbosity") if st.get("verbosity") in _VERB else "normal",
                    "tone": st.get("tone") if st.get("tone") in _TONE else "neutral",
                    "initiative": st.get("initiative") if st.get("initiative") in _INIT else "passive",
                }
                styled += 1
            dt = data.get("dialogue_trust")
            if isinstance(dt, dict):
                valid_topics = set((target.get("dialogue") or {}).keys())
                clean = {}
                for topic, val in dt.items():
                    if topic not in valid_topics:
                        continue
                    try:
                        clean[topic] = max(0, min(100, int(val)))
                    except (TypeError, ValueError):
                        continue
                if clean:
                    target["dialogue_trust"] = clean
        print(f"[PARSER] NPC style: set style for {styled} NPCs", flush=True)

    def pass3_5_validate(self, world_book: dict) -> list[str]:
        """Validate world book completeness."""
        issues: list[str] = []
        entities = world_book.get("entities", {})
        scenes = world_book.get("scenes", {})

        npc_names: set[str] = set()
        for eid, ent in entities.items():
            if not isinstance(ent, dict) or ent.get("type") != "npc":
                continue
            name = ent.get("name", "")
            if not name:
                issues.append(f"NPC '{eid}' missing name")
            elif name in npc_names:
                issues.append(f"Duplicate NPC name: '{name}'")
            npc_names.add(name)
            if not ent.get("personality"):
                issues.append(f"NPC '{name or eid}' missing personality")
            if not ent.get("dialogue"):
                issues.append(f"NPC '{name or eid}' missing dialogue")

        for sid, scene in scenes.items():
            if not isinstance(scene, dict):
                continue
            if not scene.get("desc"):
                issues.append(f"Scene '{sid}' missing desc")
            if not scene.get("source_text"):
                issues.append(f"Scene '{sid}' missing source_text")
            for npc_id in scene.get("npcs", []):
                if npc_id not in entities:
                    issues.append(
                        f"Scene '{sid}' → unknown NPC '{npc_id}'")

        if not world_book.get("opening"):
            issues.append("Missing opening")
        ss = world_book.get("starting_scene", "")
        if not ss:
            issues.append("Missing starting_scene")
        elif ss not in scenes:
            issues.append(f"starting_scene '{ss}' not in scenes")

        # Validate story_beats
        clue_ids = {c.get("id") for s in scenes.values() if isinstance(s, dict)
                    for c in s.get("clues", []) if isinstance(c, dict)}
        for beat in world_book.get("story_beats", []):
            bid = beat.get("id", "?")
            for sid in beat.get("scenes", []):
                if sid not in scenes:
                    issues.append(f"Beat '{bid}' references unknown scene '{sid}'")
            for cid in beat.get("critical_clues", []):
                if cid not in clue_ids:
                    issues.append(f"Beat '{bid}' references unknown clue '{cid}'")

        return issues

    # ── Full pipeline ──────────────────────────────────────────

    def parse(self, text: str) -> dict:
        print(f"[PARSER] Starting pipeline: {len(text)} chars", flush=True)

        # Pass 0
        print("[PARSER] Pass 0: Overview...", flush=True)
        overview = self.pass0_overview(text)

        # Chunk
        chunks = chunk_document(text)
        print(f"[PARSER] Document chunked: {len(chunks)} chunks", flush=True)

        # Pass 1
        print("[PARSER] Pass 1: Entity extraction per chunk...", flush=True)
        pass1 = self.pass1_extract_all(chunks, overview)

        # Pass 1.5: Code-based dedup
        print("[PARSER] Pass 1.5: Deduplicating entities...", flush=True)
        pass1 = self.pass1_5_deduplicate(pass1, overview)

        # Pass 1.7: Source text binding
        print("[PARSER] Pass 1.7: Binding source text...", flush=True)
        self.pass1_7_bind_source_text(pass1, text)

        # Pass 1.8: closed-set coverage repair. The model may classify only
        # source sections selected by code; names and descriptions remain verbatim.
        print("[PARSER] Pass 1.8: Auditing scene coverage...", flush=True)
        self.pass1_8_recover_scenes(pass1, text)

        print("[PARSER] Pass 1.9: Recovering marked entities...", flush=True)
        self.pass1_9_recover_marked_entities(pass1, text)

        # Pass 2: Global linking + story beat extraction
        print("[PARSER] Pass 2: Global linking + story beats...", flush=True)
        pass2 = self.pass2_link(pass1, overview)

        # Code assembly (replaces Pass 3 LLM rebuild)
        print("[PARSER] Assembling world book (code)...", flush=True)
        world_book = _assemble_world_book(pass1[0], pass2, overview)

        # Merge overview metadata
        world_book.setdefault("name", overview.get("title", "Unknown"))
        world_book.setdefault("description", overview.get("mystery", ""))
        world_book.setdefault("version", "0.1.0")

        # Opening narration is player-facing and therefore must be traceable to
        # the uploaded document. Pass 0 occasionally returns a polished summary
        # despite being asked for a verbatim quote; never trust that as canon.
        opening = overview.get("opening", "")
        opening_verified = _source_contains_text(text, opening)
        if not opening_verified:
            if opening:
                print("[PARSER] Rejected ungrounded Pass 0 opening", flush=True)
            opening = _extract_opening(text)
            opening_verified = bool(opening and _source_contains_text(text, opening))
        if opening:
            world_book["opening"] = opening
        world_book["_opening_source_verified"] = opening_verified

        # starting_scene is already resolved by _assemble_world_book;
        # no further override needed here.

        # Rule system: LLM detection → fallback heuristic
        rule_system = overview.get("rule_system", "")
        if rule_system not in ("dnd", "coc"):
            rule_system = _detect_rule_system(text)
        world_book["rule_system"] = rule_system

        # Extract PL-facing information (player rules/warnings)
        pl_info = _extract_pl_info(text)
        if pl_info:
            world_book["pl_info"] = pl_info

        # Some modules use an action-cost clock that is distinct from chat
        # turns (for example actions worth 0/1/2 development rounds). Extract
        # the explicit table and milestones in code so runtime progression does
        # not depend on the narrator remembering arithmetic from prose.
        action_clocks = _extract_action_clocks(text)
        if action_clocks:
            world_book["action_clocks"] = action_clocks

        # Pass 3.6: Game mechanics — extract dice/SAN check points from the
        # ORIGINAL text and merge into entity state machines (makes checks rollable).
        print("[PARSER] Pass 3.6: Extracting game mechanics (checks/SAN)...", flush=True)
        try:
            self.pass_game_mechanics(text, world_book)
        except Exception as e:
            print(f"[PARSER] Game mechanics error: {e}", flush=True)

        # Pass 3.7: NPC storylines — per-NPC arc across beats/scenes, so the KP
        # plays travelling/recurring NPCs coherently instead of as strangers.
        print("[PARSER] Pass 3.7: Extracting NPC storylines...", flush=True)
        try:
            self.pass_npc_storylines(text, world_book)
        except Exception as e:
            print(f"[PARSER] NPC storyline error: {e}", flush=True)

        # Pass 3.8: NPC style + per-topic trust — bake speaking style and dialogue
        # trust thresholds into static data, replacing the runtime keyword
        # heuristics (npc_state.infer_style/default_trust, now fallback-only).
        print("[PARSER] Pass 3.8: Inferring NPC style/trust...", flush=True)
        try:
            self.pass_npc_style(world_book)
        except Exception as e:
            print(f"[PARSER] NPC style error: {e}", flush=True)

        # Pass 3.5: Validation
        print("[PARSER] Pass 3.5: Validating...", flush=True)
        issues = self.pass3_5_validate(world_book)
        if issues:
            world_book["_validation_issues"] = issues
            for issue in issues:
                print(f"[PARSER:VALIDATE] {issue}", flush=True)

        print(f"[PARSER] Complete: {world_book.get('name')} "
              f"(scenes={len(world_book.get('scenes', {}))}, "
              f"entities={len(world_book.get('entities', {}))})",
              flush=True)
        return world_book


# ── Opening Text Extraction ──────────────────────────────────

_PROVENANCE_COLLECTIONS = ("scenes", "npcs", "items", "clues", "events")


def _iter_text_windows(
    text: str,
    window_chars: int = 28000,
    overlap_chars: int = 1200,
):
    """Yield the entire module in bounded, overlapping analysis windows."""
    if not text:
        return
    if window_chars <= overlap_chars:
        raise ValueError("window_chars must be greater than overlap_chars")

    start = 0
    while start < len(text):
        hard_end = min(len(text), start + window_chars)
        end = hard_end
        if hard_end < len(text):
            boundary = text.rfind("\n\n", start + window_chars // 2, hard_end)
            if boundary > start:
                end = boundary
        yield text[start:end]
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)


def _bind_chunk_provenance(result: dict, chunk: Chunk) -> None:
    """Verify LLM quotes and attach an exact source excerpt when possible."""
    if not isinstance(result, dict):
        return

    source = chunk.text
    for collection in _PROVENANCE_COLLECTIONS:
        entries = result.get(collection, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            quote = entry.get("source_quote", "")
            if not isinstance(quote, str):
                quote = ""
            quote = quote.strip()

            if quote and quote in source:
                entry["source_quote"] = quote
                entry["source_verified"] = True
            else:
                entry.pop("source_quote", None)
                entry["source_verified"] = False
                name = str(entry.get("name", "")).strip()
                if name:
                    index = source.find(name)
                    if index >= 0:
                        start = max(0, index - 240)
                        end = min(len(source), index + len(name) + 360)
                        entry["source_quote"] = source[start:end].strip()
                        entry["source_verified"] = True

            entry["source_chunk"] = chunk.index


_OPENING_PATTERNS = [
    re.compile(r'(?:导入|开场|开幕|开始|导入部分)[：:]\s*(.*?)(?=\n\n\n|\n[■★◆#【§]|\Z)', re.DOTALL),
    re.compile(r'(?:向玩家|给玩家|对PL)[朗读念讲述说](.*?)(?=\n\n\n|\n[■★◆#【§]|\Z)', re.DOTALL),
    # Header-only form: "导入\n" on its own line, content follows
    re.compile(r'^(?:导入|开场|开幕)\s*\n(.*?)(?=\n(?:第一日|第二日|HO\d|■|\Z))', re.DOTALL | re.MULTILINE),
]

def _extract_opening(text: str) -> str:
    """Extract opening narration from module text via regex patterns."""
    for pat in _OPENING_PATTERNS:
        m = pat.search(text)
        if m:
            block = m.group(1).strip()
            if len(block) > 30:
                return block
    return ""


def _source_contains_text(source: str, candidate: str) -> bool:
    """Verify candidate prose against source while tolerating PDF line wrapping."""
    if not source or not candidate or not isinstance(candidate, str):
        return False
    if candidate in source:
        return True
    normalize = lambda value: re.sub(r'\s+', ' ', value).strip()
    normalized_candidate = normalize(candidate)
    return len(normalized_candidate) >= 30 and normalized_candidate in normalize(source)


def _extract_action_clocks(text: str) -> dict:
    """Extract explicit Chinese action-cost clocks and milestone passages."""
    milestone_pattern = re.compile(
        r'◇\s*当\s*(?P<name>[^\n]{0,20}?轮次)\s*到达\s*'
        r'[\[［【]?\s*(?P<at>\d+)\s*[\]］】]?\s*时'
    )
    milestone_matches = list(milestone_pattern.finditer(text or ""))
    if not milestone_matches:
        return {}

    clock_name = milestone_matches[0].group("name").strip()
    action_pattern = re.compile(
        r'(?m)^□\s*(?P<label>[^\n（(]{1,30})[（(]\s*'
        r'(?P<increment>\d+|/)\s*[）)]'
    )
    actions = []
    known_triggers = {
        "尝试站起": ["站起", "起身"],
        "尝试开门": ["开门", "门把"],
        "尝试冲水": ["冲水", "冲水按钮"],
        "查看排气扇": ["排气扇"],
        "查看读物架": ["读物架", "报纸", "杂志"],
        "查看设备": ["查看设备", "手机", "手表"],
        "沟通外界": ["沟通外界", "打电话", "发信息", "发送信息"],
    }
    for match in action_pattern.finditer(text):
        if match.group("increment") == "/":
            continue
        label = match.group("label").strip()
        base = re.sub(r'^(?:尝试|查看)', '', label).strip()
        triggers = [label]
        if base and base != label:
            triggers.append(base)
        triggers.extend(known_triggers.get(label, []))
        actions.append({
            "label": label,
            "triggers": list(dict.fromkeys(t for t in triggers if t)),
            "increment": int(match.group("increment")),
            "source_quote": match.group(0).strip(),
        })

    outcome_actions = []
    violence_match = re.search(
        r'暴力行为.{0,900}?大失败\s*3\s*/\s*失败\s*0\s*/\s*'
        r'成功[、,\s]*困难成功\s*/\s*1\s*/\s*'
        r'极难成功\s*2\s*/\s*大成功\s*3',
        text or "", flags=re.DOTALL)
    if violence_match:
        outcome_actions.append({
            "label": "暴力行为",
            "triggers": ["暴力", "攻击", "破坏", "砸", "踹", "撞", "撬", "射击", "开枪"],
            "outcome_increments": {
                "critical_failure": 3, "fumble": 3, "failure": 0,
                "success": 1, "hard_success": 1,
                "extreme_success": 2, "critical_success": 3,
            },
            "source_quote": violence_match.group(0)[-240:].strip(),
        })

    milestones = []
    for index, match in enumerate(milestone_matches):
        next_start = (milestone_matches[index + 1].start()
                      if index + 1 < len(milestone_matches) else len(text))
        structural = re.search(r'(?m)^■', text[match.end():next_start])
        end = match.end() + structural.start() if structural else next_start
        narration = text[match.end():end].strip()
        # Keep enough canonical prose to narrate the event, without attaching
        # an entire following chapter when PDF extraction has sparse headings.
        narration = narration[:1800].strip()
        # Parenthetical dice/SAN directions are KP-facing mechanics. They remain
        # in source context, but must not be emitted as player-facing prose.
        narration = re.sub(
            r'[（(][^）)]*(?:检定|SAN|守密人|\bKP\b)[^）)]*[）)]',
            '', narration, flags=re.IGNORECASE | re.DOTALL)
        narration = re.sub(
            r'(?m)^\s*[—－-]+\s*\d+\s*[—－-]+\s*$', '', narration).strip()
        at = int(match.group("at"))
        milestones.append({
            "at": at,
            "flag": f"action_clock_{at}",
            "narration": narration,
            "source_quote": match.group(0).strip(),
        })

    if not actions and not milestones:
        return {}
    return {
        "development_round": {
            "name": clock_name,
            "initial": 0,
            "default_increment": 0,
            "actions": actions,
            "outcome_actions": outcome_actions,
            "milestones": milestones,
        }
    }


# ── PL Information Extraction ─────────────────────────────────

_PL_INFO_PATTERNS = [
    re.compile(r'\[PL向[信情]息\].*?(?=\n\n|\n[■★◆#【]|\Z)', re.DOTALL),
    re.compile(r'(?:PL|玩家)[向用]信息[：:].*?(?=\n\n|\n[■★◆#【]|\Z)', re.DOTALL),
    re.compile(r'特殊规则[：:].*?(?=\n\n|\n[■★◆#【]|\Z)', re.DOTALL),
    re.compile(r'请向PL公开.*?(?=\n\n|\n[■★◆#【]|\Z)', re.DOTALL),
]

def _extract_pl_info(text: str) -> str:
    """Extract player-facing information blocks from module text."""
    results = []
    for pat in _PL_INFO_PATTERNS:
        for m in pat.finditer(text):
            block = m.group(0).strip()
            if block and block not in results:
                results.append(block)
    return "\n\n".join(results) if results else ""


# ── Rule System Detection ─────────────────────────────────────

_COC_KEYWORDS = re.compile(
    r'克苏鲁|COC|[Cc]all\s+of\s+[Cc]thulhu|d100|1d100|SAN值|理智值|理智检定|幸运检定|'
    r'技能值|POW|STR|DEX|CON|APP|SIZ|INT|EDU|侦查|聆听|图书馆',
    re.IGNORECASE,
)

def _detect_rule_system(text: str) -> str:
    """Fallback heuristic: scan first 4000 chars for system indicators."""
    snippet = text[:4000]
    if _COC_KEYWORDS.search(snippet):
        return "coc"
    return "dnd"


# ── Deduplication ──────────────────────────────────────────────

def _deduplicate_entities(key: str, pass1_results: list[dict]) -> list[dict]:
    seen = set()
    merged = []
    for r in pass1_results:
        for entity in r.get(key, []):
            eid = entity.get("id", "")
            if eid and eid not in seen:
                seen.add(eid)
                merged.append(entity)
    return merged


def _dedup_list_by_id(entities: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for e in entities:
        eid = e.get("id", "")
        if eid and eid not in seen:
            seen.add(eid)
            result.append(e)
    return result


def _dedup_scenes(all_scenes: list[dict]) -> dict[str, dict]:
    """Merge scenes with same ID: longest desc wins, npcs/clues are unioned."""
    by_id: dict[str, dict] = {}
    for scene in all_scenes:
        sid = scene.get("id", "")
        if not sid:
            continue
        if sid not in by_id:
            by_id[sid] = dict(scene)
            by_id[sid].setdefault("clues", [])
            by_id[sid].setdefault("npcs", [])
        else:
            ex = by_id[sid]
            if len(scene.get("desc", "")) > len(ex.get("desc", "")):
                ex["desc"] = scene["desc"]
            if len(scene.get("purpose", "")) > len(ex.get("purpose", "")):
                ex["purpose"] = scene["purpose"]
            if scene.get("source_text") and not ex.get("source_text"):
                ex["source_text"] = scene["source_text"]
            for field in (
                    "source_start", "source_section_number", "source_recovered"):
                if scene.get(field) is not None and ex.get(field) is None:
                    ex[field] = scene[field]
            existing_npcs = set(ex.get("npcs", []))
            for n in scene.get("npcs", []):
                if n not in existing_npcs:
                    ex.setdefault("npcs", []).append(n)
                    existing_npcs.add(n)
            existing_clue_ids = {c.get("id") for c in ex.get("clues", [])}
            for c in scene.get("clues", []):
                if c.get("id") and c["id"] not in existing_clue_ids:
                    ex.setdefault("clues", []).append(c)
                    existing_clue_ids.add(c["id"])
    return by_id


def _assemble_world_book(pass1_data: dict, pass2_result: dict, overview: dict) -> dict:
    """Code-based world book assembly — replaces Pass 3 LLM rebuild.

    Takes deduplicated Pass 1 data + Pass 2 scene graph/beats and produces
    the final world book dict without any LLM involvement in structure.
    """
    # Scenes: dedup by ID, merge desc/npcs/clues
    scenes = _dedup_scenes(pass1_data.get("scenes", []))

    # Apply scene exits from Pass 2 scene graph
    scene_graph = pass2_result.get("scene_graph", {})
    for sid, gdata in scene_graph.items():
        if sid in scenes and isinstance(gdata, dict):
            scenes[sid]["exits"] = gdata.get("exits", {})
    _apply_numbered_scene_edges(scenes)

    # Build flat entities dict: NPCs + items (type-tagged)
    entities: dict[str, dict] = {}
    for npc in pass1_data.get("npcs", []):
        npc = dict(npc)
        npc["type"] = "npc"
        eid = npc.get("id", "")
        if eid:
            entities[eid] = npc
    for item in pass1_data.get("items", []):
        item = dict(item)
        item.setdefault("type", "item")
        eid = item.get("id", "")
        if eid:
            entities[eid] = item

    # Embed standalone clues (those with scene= field) into their scenes
    for clue in pass1_data.get("clues", []):
        sid = clue.get("scene", "")
        cid = clue.get("id", "")
        if not sid or not cid:
            continue
        if sid not in scenes:
            continue
        existing_ids = {c.get("id") for c in scenes[sid].get("clues", [])}
        if cid not in existing_ids:
            scenes[sid].setdefault("clues", []).append({
                "id": cid,
                "desc": clue.get("desc", ""),
                "check": clue.get("check", ""),
                "reveals": clue.get("reveals", ""),
                "points_to": clue.get("points_to", ""),
            })

    # Collect all real clue IDs from scenes
    real_clue_ids: set[str] = set()
    for s in scenes.values():
        for c in s.get("clues", []):
            if isinstance(c, dict) and c.get("id"):
                real_clue_ids.add(c["id"])

    # Filter beat clue references: remove IDs not present in scenes.
    # If all critical_clues were invalid, downgrade advance_when to "visited"
    # so the beat doesn't block progression permanently.
    story_beats = pass2_result.get("story_beats", [])
    for beat in story_beats:
        if not isinstance(beat, dict):
            continue
        for field in ("critical_clues", "optional_clues"):
            raw_ids = beat.get(field, [])
            valid = [cid for cid in raw_ids if cid in real_clue_ids]
            invalid = [cid for cid in raw_ids if cid not in real_clue_ids]
            if invalid:
                print(f"[PARSER:BEATS] Beat '{beat.get('id')}' {field} removed invalid IDs: {invalid}", flush=True)
            beat[field] = valid
        # Downgrade if critical_clues went empty and beat required them
        if not beat.get("critical_clues") and beat.get("advance_when") in ("any_critical", "all_critical"):
            print(f"[PARSER:BEATS] Beat '{beat.get('id')}' downgraded to 'visited' (no valid critical_clues)", flush=True)
            beat["advance_when"] = "visited"

    # Resolve starting_scene: overview may return Chinese name or wrong ID.
    # Try: exact ID match → exact name match → partial name match → first scene with clues.
    raw_start = overview.get("starting_scene", "")
    starting_scene = ""
    if raw_start in scenes:
        starting_scene = raw_start
    else:
        # Exact name match
        starting_scene = next(
            (sid for sid, s in scenes.items()
             if isinstance(s, dict) and s.get("name", "") == raw_start),
            ""
        )
    if not starting_scene and raw_start:
        # Partial name match (e.g. "香槟集市候车点" matches scene name "香槟集市")
        starting_scene = next(
            (sid for sid, s in scenes.items()
             if isinstance(s, dict) and (
                 raw_start in s.get("name", "") or s.get("name", "") in raw_start
             )),
            ""
        )
    if not starting_scene and scenes:
        # Fallback: first scene that has clues (likely the real play area, not transit)
        starting_scene = next(
            (sid for sid, s in scenes.items()
             if isinstance(s, dict) and s.get("clues")),
            next(iter(scenes))
        )
    if raw_start and starting_scene != raw_start:
        print(f"[PARSER] starting_scene resolved: {raw_start!r} → {starting_scene!r}", flush=True)

    # If resolved scene is empty (no clues/npcs) it's probably a transit point.
    # Prefer the first scene that actually has clues or NPC references.
    if starting_scene:
        s = scenes.get(starting_scene, {})
        is_empty = not s.get("clues") and not s.get("npcs")
        if is_empty:
            richer = next(
                (sid for sid, sd in scenes.items()
                 if isinstance(sd, dict) and (sd.get("clues") or sd.get("npcs"))),
                None
            )
            if richer:
                print(f"[PARSER] starting_scene upgraded from empty scene {starting_scene!r} → {richer!r}", flush=True)
                starting_scene = richer

    return {
        "scenes": scenes,
        "entities": entities,
        "story_beats": story_beats,
        "starting_scene": starting_scene,
    }


# ── Pass 0.5: Code-extract Character Section ────────────────

_CHAR_SECTION_PATTERN = re.compile(
    r'■\s*登[场場]人物(.*?)(?=\n■|\n★|\n【|\Z)', re.DOTALL)

_CHAR_ENTRY_PATTERN = re.compile(
    r'[・·•]\s*(.+?)(?:[（(].*?[)）])?\s*\n(.*?)(?=\n[・·•]|\n■|\n★|\n【|\Z)',
    re.DOTALL)


def _extract_character_section(text: str) -> list[dict]:
    """Extract NPC entries from ■登场人物 section via regex."""
    m = _CHAR_SECTION_PATTERN.search(text)
    if not m:
        return []
    section = m.group(1)
    chars = []
    for cm in _CHAR_ENTRY_PATTERN.finditer(section):
        name = cm.group(1).strip()
        desc = cm.group(2).strip()
        if name and desc:
            chars.append({"name": name, "desc": desc})
    return chars


# ── Pass 1.5b: Small LLM Alias Resolution ───────────────────

_ALIAS_RESOLVE_PROMPT = """Given these named characters and unnamed descriptions, match them.

NAMED CHARACTERS:
{roster}

UNMATCHED DESCRIPTIONS:
{unmatched}

Return ONLY valid JSON — a list of matches:
[{{"unnamed": "描述", "canonical_name": "正式名字", "confidence": "high/medium/low"}}]

If an unnamed description does NOT match any named character (e.g. a one-off stranger), set canonical_name to null.
Only match with high/medium confidence. Do NOT force-match."""


# ── Pass 1.5 / 1.7 Helpers ──────────────────────────────────

_COMMON_CHINESE = frozenset(
    "的了不是在有我他她它们你这那个一二三四五六七八九十"
    "大小上下中和与也就都要到会很还人为"
)


def _norm_name(name: str) -> str:
    return (name.replace(" ", "").replace("　", "")
            .replace("・", "").replace("\xb7", "").strip())


def _names_match(a: str, b: str) -> bool:
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) >= 2 and na in nb:
        return True
    if len(nb) >= 2 and nb in na:
        return True
    return False


def _distinctive_overlap(text_a: str, text_b: str) -> int:
    ca = set(re.findall(r'[一-鿿]', text_a)) - _COMMON_CHINESE
    cb = set(re.findall(r'[一-鿿]', text_b)) - _COMMON_CHINESE
    return len(ca & cb)


def _npc_full_text(npc: dict) -> str:
    parts = [npc.get("name", ""), npc.get("personality", ""),
             npc.get("appearance", "")]
    d = npc.get("dialogue", {})
    if isinstance(d, dict):
        parts.extend(str(v) for v in d.values())
    return " ".join(p for p in parts if p)


def _pick_best_npc_id(name: str, entities: list[dict]) -> str:
    generic = ("npc_young", "npc_read", "npc_old", "npc_man",
                "npc_woman", "npc_tall", "npc_mysterious")
    ids = [e.get("id", "") for e in entities if e.get("id")]
    good = [i for i in ids if not any(i.lower().startswith(g) for g in generic)]
    if good:
        return max(good, key=len)
    return ids[0] if ids else f"npc_{_norm_name(name)}"


def _merge_npc_group(cid: str, cname: str, entities: list[dict],
                     roster_entry: dict | None) -> dict:
    merged = {
        "type": "npc", "id": cid, "name": cname,
        "aliases": [], "personality": "", "dialogue": {},
    }
    seen_names = {_norm_name(cname)}
    all_scenes = []
    source_quotes = []

    for ent in entities:
        norm = _norm_name(ent.get("name", ""))
        if norm and norm not in seen_names:
            merged["aliases"].append(ent["name"])
            seen_names.add(norm)
        if len(ent.get("personality", "")) > len(merged["personality"]):
            merged["personality"] = ent["personality"]
        d = ent.get("dialogue", {})
        if isinstance(d, dict):
            for k, v in d.items():
                if k not in merged["dialogue"]:
                    merged["dialogue"][k] = v
        s = ent.get("scene", "")
        if s and s not in all_scenes:
            all_scenes.append(s)
        for field in ("profession", "appearance"):
            if not merged.get(field) and ent.get(field):
                merged[field] = ent[field]
        if ent.get("states") and not merged.get("states"):
            merged["states"] = ent["states"]
        quote = ent.get("source_quote", "")
        if quote and quote not in source_quotes:
            source_quotes.append(quote)

    if all_scenes:
        merged["scene"] = all_scenes[0]
        if len(all_scenes) > 1:
            merged["all_scenes"] = all_scenes
    if source_quotes:
        merged["source_quote"] = source_quotes[0]
        if len(source_quotes) > 1:
            merged["source_quotes"] = source_quotes
        merged["source_verified"] = True
    if roster_entry and not merged.get("profession"):
        merged["profession"] = roster_entry.get("role", "")
    if not merged["aliases"]:
        del merged["aliases"]
    return merged


def _remap_scene_npcs(scenes: list[dict], npc_id_map: dict) -> list[dict]:
    for scene in scenes:
        if "npcs" in scene:
            seen = set()
            remapped = []
            for nid in scene["npcs"]:
                new_id = npc_id_map.get(nid, nid)
                if new_id not in seen:
                    seen.add(new_id)
                    remapped.append(new_id)
            scene["npcs"] = remapped
    return scenes


_SEGMENT_MARKER = re.compile(
    r'^(?:■|★|◆|【|END[：:]|'
    r'HO[123](?:\s|$|[：:])|'
    r'(?:正文|导入|结局|第[一二三四五六七八九十0-9]+(?:日|幕|章|节)|'
    r'主线|支线|分支|探索)(?:\s*$|[：:].*)|'
    r'(?:LOCATION|SCENE|CHAPTER|PART)\s+'
    r'(?:[A-Z0-9IVX]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)'
    r'\s*[:.-]?)'
    r'|^[A-Z][A-Z0-9’\'&(), -]{4,70}$'
)

_ENGLISH_SECTION_WORDS = frozenset({
    "introduction", "background", "overview", "setup", "conclusion",
    "aftermath", "rewards", "development", "beginning", "ending",
})
_ENGLISH_TITLE_CONNECTORS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "versus", "with",
})


def _is_english_title_case_heading(value: str) -> bool:
    """Conservatively recognize standalone English scenario subheadings."""
    if (not (2 <= len(value) <= 70) or not re.match(r'[A-Z]', value)
            or value.endswith("-")):
        return False
    if re.search(r'[.!?;,:]|\d|%|\b\d+[dD]\d+\b', value):
        return False
    if re.match(
            r'^(?:Author|Editor|Layout|Cover|Interior|Cartography|Proofreading|'
            r'Design|Character Sheet|Pre-generated|Copyright|Chaosium|'
            r'Hit Points?|Rune Points?|Magic Points?|Movement Rate|Equipment|'
            r'Weapons?|Skills?|Traits?|Passions?)\b',
            value, flags=re.IGNORECASE):
        return False
    if value.casefold() in _ENGLISH_SECTION_WORDS:
        return True
    words = re.findall(r"[A-Za-z][A-Za-z’'\-]*", value)
    if not (2 <= len(words) <= 9):
        return False
    if words[-1].casefold() in _ENGLISH_TITLE_CONNECTORS:
        return False
    significant = [word for word in words
                   if word.casefold() not in _ENGLISH_TITLE_CONNECTORS]
    return bool(significant) and all(word[0].isupper() for word in significant)

_CHECK_LABEL_PARTS = frozenset({
    "侦查", "聆听", "灵感", "幸运", "图书馆", "图书馆使用", "教育", "英语",
    "艺术", "乔装", "交涉技能", "力量", "敏捷", "体质", "外貌", "意志",
    "心理学", "密码学", "潜行", "导航", "攀爬", "闪避", "急救", "医学",
    "神秘学", "克苏鲁神话", "会计", "话术", "说服", "恐吓", "魅惑",
    "斗殴", "射击", "追踪", "职业模板", "职业模版",
    "str", "con", "siz", "dex", "int", "pow", "app", "edu", "san",
})


def _is_bracket_check_line(stripped: str) -> bool:
    """Distinguish `【skill】 result` from real bracketed event headings."""
    match = re.match(r'^【([^】]+)】(.*)$', stripped)
    if not match:
        return False
    label, suffix = match.groups()
    # A bracket followed by prose is an inline check/result, not a boundary.
    if suffix.strip().lstrip("：:").strip():
        return True
    parts = [part.strip().lower() for part in re.split(r'[/／、+]|\bor\b', label)]
    return bool(parts) and all(
        part in _CHECK_LABEL_PARTS
        or re.fullmatch(r'(?:困难|极难)?(?:str|con|siz|dex|int|pow|app|edu)\*?\d*', part)
        for part in parts
    )


def _split_text_segments(text: str) -> list[dict]:
    lines = text.split('\n')
    stripped_lines = {line.strip() for line in lines if line.strip()}
    line_counts = Counter(line.strip() for line in lines if line.strip())
    toc_headings = {
        match.group(1).strip().casefold()
        for line in lines
        if (match := re.match(
            r'^\s*(.{2,70}?)\s*\.{2,}\s*\d+\s*$', line.strip()))
    }
    solo_references = {
        int(value) for value in re.findall(
            r'\b(?:go|turn|proceed)\s+to\s+(\d{1,3})\b', text,
            flags=re.IGNORECASE)
    }
    solo_numbered = len(solo_references) >= 10

    def neighboring_nonempty(index: int, direction: int) -> str:
        cursor = index + direction
        while 0 <= cursor < len(lines):
            candidate = lines[cursor].strip()
            if candidate:
                return candidate
            cursor += direction
        return ""

    numeric_section_indices = set()
    if solo_numbered:
        candidates: dict[int, list[int]] = {}
        for index, line in enumerate(lines):
            stripped = line.strip()
            if re.fullmatch(r'\d{1,3}', stripped):
                value = int(stripped)
                if 1 <= value <= max(solo_references):
                    candidates.setdefault(value, []).append(index)
        for indices in candidates.values():
            def numeric_score(index: int) -> tuple[int, int]:
                before = neighboring_nonempty(index, -1)
                after = neighboring_nonempty(index, 1)
                score = 0
                if (len(after) >= 10 and re.search(r'[a-z]', after)
                        and re.search(r'[A-Z]', after)):
                    score += 6
                if (len(before) >= 4 and before.upper() == before
                        and re.search(r'[A-Z]', before)):
                    score -= 5
                if (len(after) >= 4 and after.upper() == after
                        and re.search(r'[A-Z]', after)):
                    score -= 5
                return score, index
            best = max(indices, key=numeric_score)
            if numeric_score(best)[0] > 0:
                numeric_section_indices.add(best)

    def is_heading(line: str, index: int) -> bool:
        stripped = line.strip()
        numeric_solo_heading = index in numeric_section_indices
        next_nonempty = neighboring_nonempty(index, 1)
        english_scene_heading = bool(re.match(
            r'^(?:location|scene|chapter|part)\s+'
            r'(?:[a-z0-9ivx]+|one|two|three|four|five|six|seven|eight|nine|ten)'
            r'\s*[:.-]?',
            stripped, flags=re.IGNORECASE))
        toc_heading = (
            stripped.casefold() in toc_headings
            or f"{stripped} {next_nonempty}".casefold() in toc_headings
        )
        title_case_heading = _is_english_title_case_heading(stripped)
        if not (_SEGMENT_MARKER.match(stripped) or english_scene_heading
                or numeric_solo_heading
                or toc_heading or title_case_heading):
            return False
        if numeric_solo_heading:
            return True
        previous_nonempty = neighboring_nonempty(index, -1)
        if (title_case_heading and re.fullmatch(r'\d{1,3}', previous_nonempty)
                and index >= 2):
            before_page_number = neighboring_nonempty(index - 1, -1)
            if (len(before_page_number) >= 4
                    and before_page_number.upper() == before_page_number
                    and re.search(r'[A-Z]', before_page_number)):
                return False
        if (title_case_heading and (
                previous_nonempty.casefold() in {
                    "written by", "editing", "editor", "cover art", "interior art",
                    "cartography", "layout", "book design", "character sheet",
                }
                or previous_nonempty.casefold().endswith(" by"))):
            return False
        if _is_bracket_check_line(stripped):
            return False
        # Dot-leader/page-number rows belong to the table of contents.
        if re.search(r'\.{2,}\s*\d*\s*$', stripped):
            return False
        if re.fullmatch(r'HO[123]\d+', stripped, flags=re.IGNORECASE):
            return False
        if re.fullmatch(
                r'(?:主线|支线|分支|探索|HO[123]).*\D\d{1,3}',
                stripped, flags=re.IGNORECASE):
            return False
        # Contents entries can be extracted without dot leaders, for example
        # `主线：委托16`, while the actual heading later appears without the
        # page number. Only discard the numbered form when that exact base
        # heading exists elsewhere in the document.
        numbered = re.fullmatch(r'(.+?)(\d{1,3})', stripped)
        if numbered and numbered.group(1).rstrip() in stripped_lines:
            return False
        # Long `HO2：...` lines are character speech, not an investigator
        # handout/branch boundary. Real HO section labels are short names.
        ho_suffix = re.match(r'^HO[123]\s*[：:](.+)$', stripped,
                             flags=re.IGNORECASE)
        if ho_suffix and (
                len(ho_suffix.group(1).strip()) > 8
                or re.search(r'[，。！？?!]', ho_suffix.group(1))):
            return False
        # Character sheets and monster stat blocks are often uppercase and fit
        # the generic heading regex. They are content, not scene boundaries.
        if len(re.findall(
                r'\b(?:STR|CON|SIZ|DEX|INT|POW|APP|EDU)(?=\s|\d)',
                stripped)) >= 2:
            return False
        # PDF extraction repeats the book title at the top of every page. Such
        # running headers are not scene boundaries. Explicit numbered structural
        # headings remain boundaries even if a contents page repeats them.
        explicit = re.match(
            r'^(?:■|★|◆|【|END[：:]|(?:LOCATION|SCENE|CHAPTER|PART)\s+)',
            stripped,
        ) or english_scene_heading
        return bool(explicit or toc_heading or line_counts[stripped] <= 2)

    segments = []
    title = "(preamble)"
    current = []
    start = 0
    for i, line in enumerate(lines):
        if is_heading(line, i):
            if current:
                segments.append({"title": title,
                                 "text": "\n".join(current).strip(),
                                 "start": start})
            title = line.strip()
            current = [line]
            start = i
        else:
            current.append(line)
    if current:
        segments.append({"title": title,
                         "text": "\n".join(current).strip(),
                         "start": start})
    return segments


_NON_SCENE_HEADING_RE = re.compile(
    r'(?:contents?|credits?|copyright|introduction|background|overview|setup|'
    r'conclusion|aftermath|rewards?|development|appendix|handouts?|characters?|'
    r'investigators?|keepers?|rules?|mechanics?|statistics?|monsters?|ending|'
    r'目录|版权|前言|背景|概述|概要|真相|登场人物|人物介绍|角色卡|规则|机制|'
    r'奖励|结局|后日谈|附录|手册|线索汇总|事件表)',
    flags=re.IGNORECASE,
)


def _clean_scene_heading(title: str) -> str:
    value = str(title or "").strip()
    value = re.sub(r'^[■★◆]+\s*', '', value)
    value = re.sub(r'^【\s*([^】]+)\s*】$', r'\1', value)
    value = re.sub(
        r'^(?:LOCATION|SCENE)\s+(?:[A-Z0-9IVX]+|ONE|TWO|THREE|FOUR|FIVE|'
        r'SIX|SEVEN|EIGHT|NINE|TEN)\s*[:.\-]?\s*',
        '', value, flags=re.IGNORECASE)
    value = re.sub(r'^(?:场景|地点)\s*[：:]\s*', '', value)
    return value.strip(' ：:.-')


def _scene_coverage_candidates(original_text: str,
                               existing_scenes: list[dict]) -> list[dict]:
    """Return unmapped, source-grounded structural sections for LLM classification."""
    bound_texts = {
        str(scene.get("source_text", "")).strip()
        for scene in existing_scenes
        if str(scene.get("source_text", "")).strip()
    }
    known_names = {
        _norm_name(scene.get("name", ""))
        for scene in existing_scenes
        if _norm_name(scene.get("name", ""))
    }
    candidates: list[dict] = []
    for segment in _split_text_segments(original_text):
        raw_title = str(segment.get("title", "")).strip()
        source_text = str(segment.get("text", "")).strip()
        numeric_node = bool(re.fullmatch(r'\d{1,3}', raw_title))
        name = f"Section {raw_title}" if numeric_node else _clean_scene_heading(raw_title)
        if raw_title == "(preamble)" or not (2 <= len(name) <= 80):
            continue
        if len(source_text) < 80 or source_text in bound_texts:
            continue
        if _NON_SCENE_HEADING_RE.search(name):
            continue
        normalized_title = _norm_name(name)
        if normalized_title in known_names:
            continue
        if not numeric_node and not re.search(r'[A-Za-z\u4e00-\u9fff]', name):
            continue
        candidates.append({
            "start": int(segment.get("start", 0)),
            "name": name,
            "text": source_text,
            "kind_hint": "event" if numeric_node else "location",
        })
    return candidates


def _recovered_scene_id(name: str, start: int, existing_ids: set[str]) -> str:
    ascii_name = name.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r'[^a-z0-9]+', '_', ascii_name).strip('_')[:48]
    base = slug or f"source_scene_{start}"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _apply_scene_coverage_repair(existing_scenes: list[dict],
                                 candidates: list[dict],
                                 proposal: object) -> list[dict]:
    """Accept only candidate references; all player-facing content stays verbatim."""
    additions = proposal.get("additions", []) if isinstance(proposal, dict) else []
    by_start = {item["start"]: item for item in candidates}
    existing_ids = {str(scene.get("id", "")) for scene in existing_scenes}
    existing_names = {_norm_name(scene.get("name", "")) for scene in existing_scenes}
    accepted: list[dict] = []
    used_starts: set[int] = set()

    for item in additions if isinstance(additions, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("candidate_start"))
        except (TypeError, ValueError):
            continue
        candidate = by_start.get(start)
        if not candidate or start in used_starts:
            continue
        name = candidate["name"]
        if _norm_name(name) in existing_names:
            continue
        scene_id = _recovered_scene_id(name, start, existing_ids)
        source_text = candidate["text"]
        proposed_kind = str(item.get("kind", "")).strip().lower()
        scene_kind = proposed_kind if proposed_kind in {"location", "event"} else candidate.get("kind_hint", "location")
        accepted.append({
            "id": scene_id,
            "name": name,
            "desc": source_text[:1000],
            "purpose": "",
            "type": scene_kind,
            "source_text": source_text,
            "source_quote": source_text[:300],
            "source_start": start,
            "source_section_number": (
                int(name.removeprefix("Section "))
                if re.fullmatch(r'Section \d{1,3}', name) else None
            ),
            "source_recovered": True,
            "npcs": [],
            "clues": [],
            "exits": {},
        })
        used_starts.add(start)
        existing_ids.add(scene_id)
        existing_names.add(_norm_name(name))
    return accepted


def _apply_numbered_scene_edges(scenes: dict[str, dict]) -> None:
    """Build solo-adventure node edges from explicit source cross-references."""
    number_to_ids: dict[int, list[str]] = {}
    for scene_id, scene in scenes.items():
        if not isinstance(scene, dict):
            continue
        number = scene.get("source_section_number")
        if isinstance(number, int):
            number_to_ids.setdefault(number, []).append(scene_id)

    reference_re = re.compile(
        r'\b(?:go|turn|proceed)\s+to\s+(\d{1,3})\b|'
        r'(?:转到|转至|前往|跳转至)\s*(\d{1,3})',
        flags=re.IGNORECASE,
    )
    for scene in scenes.values():
        if not isinstance(scene, dict) or not isinstance(
                scene.get("source_section_number"), int):
            continue
        exits = scene.setdefault("exits", {})
        if not isinstance(exits, dict):
            exits = {}
            scene["exits"] = exits
        for match in reference_re.finditer(str(scene.get("source_text", ""))):
            raw_number = match.group(1) or match.group(2)
            target_ids = number_to_ids.get(int(raw_number), [])
            if len(target_ids) == 1:
                exits.setdefault(f"Section {raw_number}", target_ids[0])


_SOURCE_ENTITY_MARKERS = (
    ("npcs", "npc", re.compile(
        r'^\s*👤\s*([^：:\n]{1,50}?)(?:\s*[：:]\s*(.*))?$')),
    ("items", "item", re.compile(
        r'^\s*📖\s*([^：:\n]{1,60}?)(?:\s*[：:]\s*(.*))?$')),
)


def _source_entity_id(kind: str, name: str, scene_id: str) -> str:
    digest = hashlib.sha1(
        f"{kind}\0{name}\0{scene_id}".encode("utf-8")).hexdigest()[:10]
    return f"source_{kind}_{digest}"


def _recover_source_marked_entities(pass1_data: dict,
                                    original_text: str) -> dict[str, int]:
    """Add only explicit icon-marked entities and bind them to source scenes."""
    scenes = pass1_data.setdefault("scenes", [])
    segments = _split_text_segments(original_text)
    segment_starts = sorted(
        int(segment.get("start", 0)) for segment in segments)
    scene_by_start: dict[int, list[dict]] = {}
    for scene in scenes:
        starts = scene.get("source_starts", [])
        if not isinstance(starts, list):
            starts = []
        if isinstance(scene.get("source_start"), int):
            starts = [scene["source_start"], *starts]
        for start in dict.fromkeys(starts):
            if isinstance(start, int):
                scene_by_start.setdefault(start, []).append(scene)

    existing_names = {
        collection: {
            _norm_name(entity.get("name", ""))
            for entity in pass1_data.setdefault(collection, [])
            if _norm_name(entity.get("name", ""))
        }
        for collection in ("npcs", "items")
    }
    counts = {"npcs": 0, "items": 0}

    for line_number, raw_line in enumerate(original_text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        segment_start = max(
            (start for start in segment_starts if start <= line_number),
            default=-1,
        )
        scene_options = scene_by_start.get(segment_start, [])
        if not scene_options:
            continue
        scene = scene_options[0]
        scene_id = str(scene.get("id", ""))
        if not scene_id:
            continue

        for collection, kind, pattern in _SOURCE_ENTITY_MARKERS:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group(1).strip(' ・·•')
            description = (match.group(2) or "").strip()
            normalized = _norm_name(name)
            if not normalized or normalized in existing_names[collection]:
                break
            entity = {
                "id": _source_entity_id(kind, name, scene_id),
                "type": kind,
                "name": name,
                "scene": scene_id,
                "desc": description,
                "description": description,
                "source_quote": line,
                "source_line": line_number,
                "source_recovered": True,
                "initial_state": "present" if kind == "npc" else "hidden",
            }
            if kind == "npc":
                entity.update({
                    "public_label": "",
                    "appearance": description,
                    "personality": "",
                    "dialogue": {},
                })
            else:
                entity["portable"] = True
            pass1_data[collection].append(entity)
            existing_names[collection].add(normalized)
            counts[collection] += 1
            break
    return counts


def _score_segment(scene_name: str, scene_desc: str, npc_ids: list[str],
                   npc_index: dict, seg_text: str) -> int:
    score = 0
    seg_flat = seg_text.replace(" ", "").replace("\n", "")
    name_norm = _norm_name(scene_name)
    if name_norm and len(name_norm) >= 2 and name_norm in seg_flat:
        score += 5
    for phrase in re.findall(r'[一-鿿]{3,}', scene_desc):
        if phrase in seg_text:
            score += 2
    for npc_id in npc_ids:
        npc = npc_index.get(npc_id, {})
        npc_name = _norm_name(npc.get("name", ""))
        if npc_name and len(npc_name) >= 2 and npc_name in seg_flat:
            score += 3
        for alias in npc.get("aliases", []):
            a = _norm_name(alias)
            if a and len(a) >= 2 and a in seg_flat:
                score += 2
    return score


# ── World Book IO ─────────────────────────────────────────────

def save_world_book(name: str, data: dict) -> str:
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', name)
    module_dir = os.path.join(WORLD_BOOK_DIR, safe_name)
    os.makedirs(module_dir, exist_ok=True)
    path = os.path.join(module_dir, f"{safe_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_upload_text(upload_id: str) -> str:
    for ext in (".txt", ".md", ".docx", ".pdf"):
        path = os.path.join(UPLOADS_DIR, f"{upload_id}{ext}")
        if os.path.exists(path):
            if ext == ".docx":
                return _extract_docx_text_text(path)
            if ext == ".pdf":
                return _extract_pdf_text_text(path)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(f"Upload {upload_id} not found")


def _extract_docx_text_text(path: str) -> str:
    import zipfile, xml.etree.ElementTree as ET
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    lines = []
    for p in root.iter(f"{{{ns['w']}}}p"):
        line = "".join(t.text or "" for t in p.iter(f"{{{ns['w']}}}t"))
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def _extract_pdf_text_text(path: str) -> str:
    """Extract plain text from a .pdf file using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        lines = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                lines.append(text)
        result = "\n".join(lines)
        if not result.strip():
            return "[PDF contains no extractable text. If this is a scanned document, upload through SillyTavern which uses client-side pdf.js for better extraction.]"
        return result
    except ImportError:
        return "[pypdf not installed. Run: pip install pypdf]"
    except Exception as e:
        return f"[Error extracting PDF: {e}]"


# ── Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import DEEPSEEK_API_KEY

    if len(sys.argv) < 2:
        print("Usage: python parser.py <file.txt> [api_key]")
        sys.exit(1)

    api_key = sys.argv[2] if len(sys.argv) > 2 else DEEPSEEK_API_KEY
    if not api_key:
        print("Error: No API key. Set DEEPSEEK_API_KEY in config.py or pass as arg.")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        text = f.read()

    parser = ModuleParser(api_key=api_key)
    world = parser.parse(text)

    out_path = save_world_book(world.get("name", "parsed_module"), world)
    print(f"Saved to: {out_path}")

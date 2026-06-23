# -*- coding: utf-8 -*-
"""AIKP Reference Resolver — figure out WHICH scene NPC a player is referring to.

The hard problem: players refer to NPCs by name ("尾金星杉"), by description
("那个年轻人"), by situation ("骂我的人"), or by self-coined nickname ("野猪").
Static parser-time aliases can never cover the last three — they are produced
at play time.

Design (closed-world, code-constrains-LLM — see CLAUDE.md invariants):

    player reference X
      ① O(1) lookup   name / entity-id / player-coined nickname   (lookup_known_reference)
      ② miss + intent → closed-world LLM sub-query                (llm_resolve_reference)
            returns one of the REAL scene NPCs, or null
      ③ null          → caller treats as "not found" (code denial, no narration LLM)

The LLM sub-query is a *separate* structured call, NOT a paragraph injected into
the narration prompt. Its output schema only permits {one of the real NPCs | null},
so inventing a new character is structurally impossible — unlike open narration.
"""

from __future__ import annotations

import json
import re

from npc_context import keyword_match


# ── O(1) known-reference lookup (name / id / nickname) ─────────

def _name_to_scene_eid(name: str, scene_entity_ids: list[str],
                       entity_index: dict[str, dict]) -> str | None:
    """Map an NPC display name back to its entity id within the current scene."""
    nn = (name or "").lower().replace(" ", "")
    if not nn:
        return None
    for eid in scene_entity_ids:
        ei = entity_index.get(eid, {})
        if ei.get("type") != "npc":
            continue
        en = (ei.get("name") or "").lower().replace(" ", "")
        if en == nn:
            return eid
    return None


def lookup_known_reference(player_input: str, scene_entity_ids: list[str],
                           entity_index: dict[str, dict],
                           session: dict) -> list[str]:
    """Deterministic O(1) resolution: exact name/id, then player-coined nicknames.

    Returns matched NPC entity ids (possibly several). No LLM, no fuzzy guessing.
    """
    # 1. exact name / entity-id (existing behavior)
    ids = keyword_match(player_input, scene_entity_ids, entity_index)
    if ids:
        return ids

    # 2. player-coined nicknames stored on npc_states[name].dynamic.nicknames
    inp_nospace = player_input.lower().replace(" ", "")
    npc_states = session.get("npc_states", {})
    for name, st in npc_states.items():
        nicknames = st.get("dynamic", {}).get("nicknames", []) if isinstance(st, dict) else []
        for nick in nicknames:
            n = (nick or "").lower().replace(" ", "")
            if n and n in inp_nospace:
                eid = _name_to_scene_eid(name, scene_entity_ids, entity_index)
                if eid:
                    return [eid]
    return []


# ── Deterministic dialogue/keyword match (rules over prompts) ──

# Generic words that must NOT drive a match (they appear in many references).
_GENERIC_WORDS = frozenset({
    "年轻", "男子", "男人", "女子", "女人", "那个", "这个", "刚才", "现在",
    "说话", "名字", "询问", "他的", "她的", "我们", "你们", "他们", "什么",
    "怎么", "一个", "这里", "那里", "搭话", "交谈", "对话", "聊天", "先生",
    "家伙", "人物", "角色", "旁边", "附近", "刚刚",
})


def dialogue_keyword_match(player_input: str, scene_entity_ids: list[str],
                           entity_index: dict[str, dict],
                           world: dict) -> str | None:
    """Deterministic resolution when the player refers to an NPC by something
    they SAID — e.g. 「说不要命的」「刚才啐人的」. Extract 2–4 char Chinese
    fragments from the input and look them up verbatim in each NPC's dialogue
    lines. If exactly one NPC owns the most-specific match (>=3 chars), return
    it — no LLM guessing. This is the rules-over-prompts path that fixes the
    LLM picking the wrong NPC by personality vibe.
    """
    entities = world.get("entities", {})
    # Only 3-4 char fragments — 2-char ones are function-word noise ("的人",
    # "子的") that collide across unrelated dialogue and cause false matches.
    frags: set[str] = set()
    for run in re.findall(r'[一-鿿]+', player_input):
        for n in (4, 3):
            for i in range(len(run) - n + 1):
                frags.add(run[i:i + n])
    frags = {f for f in frags if f not in _GENERIC_WORDS}
    if not frags:
        return None

    scores: dict[str, int] = {}
    for eid in scene_entity_ids:
        if entity_index.get(eid, {}).get("type") != "npc":
            continue
        dlg = entities.get(eid, {}).get("dialogue", {})
        if not isinstance(dlg, dict):
            continue
        text = " ".join(
            (v if isinstance(v, str)
             else (v.get("text", "") if isinstance(v, dict) else str(v)))
            for v in dlg.values()
        )
        if not text:
            continue
        # Weight by fragment length — longer fragments are more specific.
        hit = sum(len(f) for f in frags if f in text)
        if hit:
            scores[eid] = hit

    if not scores:
        return None
    best = max(scores.values())
    winners = [eid for eid, s in scores.items() if s == best]
    # Require a specific match (>=3 chars) AND a unique winner.
    if len(winners) == 1 and best >= 3:
        return winners[0]
    return None


# ── Recent-behavior context for the sub-query ──────────────────

def _recent_behavior(eid: str, name: str, session: dict, entity: dict) -> str:
    """One short line of what this NPC most recently said/did.

    "骂我的人" only resolves to 尾金 because 尾金 just sneered at the player —
    so the sub-query needs each NPC's latest behavior, not just static traits.
    Falls back to a representative line of module dialogue.
    """
    npc_states = session.get("npc_states", {})
    st = npc_states.get(name) or npc_states.get(eid)
    if isinstance(st, dict):
        interactions = st.get("dynamic", {}).get("interactions", [])
        if interactions:
            last = interactions[-1]
            resp = (last.get("response") or "").strip()
            if resp:
                return resp[:80]
    # Fallback: a representative module dialogue line
    dialogue = entity.get("dialogue", {})
    if isinstance(dialogue, dict) and dialogue:
        for key in ("encounter", "greeting"):
            if key in dialogue:
                v = dialogue[key]
                txt = v if isinstance(v, str) else (v.get("text", "") if isinstance(v, dict) else "")
                if txt:
                    return str(txt)[:80]
        first = next(iter(dialogue.values()))
        txt = first if isinstance(first, str) else (first.get("text", "") if isinstance(first, dict) else "")
        if txt:
            return str(txt)[:80]
    return "（暂无明确言行）"


# ── Closed-world LLM sub-query ─────────────────────────────────

_RESOLVE_SYSTEM = (
    "你是 TRPG 系统内部的「指代消解」模块。你的唯一职责是判断玩家所指的对象是"
    "当前场景在场角色中的哪一个，或者根本不在场。你绝不叙事、绝不编造新角色。"
)


def llm_resolve_reference(target: str, player_input: str,
                          scene_entity_ids: list[str],
                          entity_index: dict[str, dict],
                          world: dict, session: dict,
                          api_key: str, base_url: str,
                          model: str,
                          recent_conversation: str = "",
                          scene_text: str = "") -> dict:
    """Closed-world structured resolution.

    Returns {"npc_id": <scene eid> | None, "persistent": bool, "reason": str}.
    On any failure, returns npc_id=None (conservative: deny rather than invent).

    recent_conversation: last N turns of dialogue — lets the sub-query resolve
        cross-scene references like "那个说不要命的人" from actual session history.
    scene_text: current scene's source text or description — grounds in-scene refs.
    """
    entities = world.get("entities", {})
    npc_ids = [eid for eid in scene_entity_ids
               if entity_index.get(eid, {}).get("type") == "npc"]
    if not npc_ids:
        return {"npc_id": None, "persistent": False, "reason": "no npc in scene"}

    roster_lines = []
    name_by_label = {}
    for eid in npc_ids:
        e = entities.get(eid, {})
        name = e.get("name", eid)
        name_by_label[name] = eid
        appr = (e.get("appearance") or "")[:50]
        pers = (e.get("personality") or "")[:50]
        # Role/title (夫人/管家/队长/女仆长…) so references by role resolve, even
        # for a disguised NPC the player only knows by role. Strip parenthetical
        # secret notes ("男扮女装") — only the public-facing role is needed to match.
        role = re.sub(r'[（(][^）)]*[）)]', '', e.get("profession", "") or "").strip()[:30]
        recent = _recent_behavior(eid, name, session, e)
        # Runtime-extracted observable traits (读书/啐人/敲背包…) — lets the LLM
        # resolve behavior references like 「看书的」 even when the trait only
        # appeared in narration, and handles synonyms (看书≈读书) that exact
        # matching can't.
        _st = session.get("npc_states", {}).get(name)
        traits = _st.get("dynamic", {}).get("traits", []) if isinstance(_st, dict) else []
        # Give the LLM ALL of this NPC's lines — a descriptive reference
        # ("说我们不要命的男子") may match ANY line, not just the latest.
        dlg = e.get("dialogue", {})
        said = ""
        if isinstance(dlg, dict) and dlg:
            said = "；".join(
                (v if isinstance(v, str)
                 else (v.get("text", "") if isinstance(v, dict) else str(v)))
                for v in dlg.values()
            )[:200]
        roster_lines.append(
            f'- id="{name}" | 身份/称呼：{role or "（未知）"} | 外貌：{appr or "（未知）"} | '
            f'性格：{pers or "（未知）"} | '
            f'标志行为：{("、".join(traits)) if traits else "（无）"} | '
            f'说过的话：{said or "（无）"} | 最近：{recent}'
        )
    roster = "\n".join(roster_lines)

    # Build context prefix: scene text + conversation history give the sub-query
    # ground truth for cross-scene refs ("那个说不要命的人") and in-scene refs.
    context_parts = []
    if scene_text:
        context_parts.append(f"【当前场景描述/原文】\n{scene_text[:600]}")
    if recent_conversation:
        context_parts.append(f"【最近对话记录（含NPC发言，可据此解析历史指代）】\n{recent_conversation}")
    context_prefix = "\n\n".join(context_parts)

    # If target == player_input (no specific ref extracted), phrase differently.
    is_full_input = (target == player_input)
    if is_full_input:
        target_line = (f'玩家输入："{player_input}"\n'
                       f"请判断玩家是否在指代某个在场角色，如果是，是哪一个？")
    else:
        target_line = (f'玩家说："{player_input}"\n'
                       f'其中用「{target}」来指代某个对象。')

    user_prompt = (
        (context_prefix + "\n\n" if context_prefix else "")
        + target_line + "\n\n"
        + f"当前场景在场的角色【仅有】以下这些，没有任何其他人：\n{roster}\n\n"
        + f'问题：玩家{"是否在指代某人，指的" if is_full_input else f"用「{target}」指的"}是上面哪一个角色？\n'
        + "判定规则（重要，按顺序）：\n"
        + "1. 优先查【最近对话记录】：若记录里某个角色说过/做过和玩家描述吻合的事，"
        + "直接命中。例：记录里「尾金星杉说'你这是不要命啊'」→「说不要命的人」=尾金星杉。\n"
        + "2. 再查角色的『说过的话』栏，逐字核对台词片段——不要凭性格印象猜。\n"
        + "3. 按外貌/身份/行为指代时，查『外貌』『性格』『标志行为』栏（注意同义：「看书」≈「读书」）。\n"
        + "4. 按身份/头衔/称呼指代时（如「夫人」「管家」「女仆长」「队长」「医生」），"
        + "查『身份/称呼』栏匹配——哪怕该角色是伪装的，玩家用其表面身份称呼也算命中。\n"
        + f"5. 明确对应某角色→返回其 id；不在名单里的人或物→返回 null。绝不编造。\n\n"
        + "只输出 JSON，格式：\n"
        + f'{{"npc_id": "<上面某个id 或 null>", '
        + f'"persistent_nickname": true/false, "reason": "<10字内理由>"}}\n'
        + f'persistent_nickname：「{target}」是否值得长期记住的特指外号'
        + '（如"野猪""读书的"为 true；"那个人""他"等泛指为 false）。'
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RESOLVE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as e:
        print(f"[REF] LLM resolve failed: {e}", flush=True)
        return {"npc_id": None, "persistent": False, "reason": "llm error"}

    label = data.get("npc_id")
    if label in (None, "null", "", "None"):
        return {"npc_id": None, "persistent": False,
                "reason": str(data.get("reason", ""))[:40]}

    # Map the returned label back to a real scene entity id
    eid = name_by_label.get(label) or _name_to_scene_eid(
        str(label), scene_entity_ids, entity_index)
    if not eid:
        # LLM returned something not in roster — treat as not found
        print(f"[REF] LLM returned off-roster label {label!r} → denied", flush=True)
        return {"npc_id": None, "persistent": False, "reason": "off-roster"}

    return {
        "npc_id": eid,
        "persistent": bool(data.get("persistent_nickname", False)),
        "reason": str(data.get("reason", ""))[:40],
    }

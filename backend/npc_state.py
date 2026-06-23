# -*- coding: utf-8 -*-
"""AIKP NPC State — static+dynamic storage, merge, style inference, interaction CRUD.

Design: docs/npc-design.md
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import NpcStatic, NpcDynamic, NpcInteraction


# ── Style Inference (Chinese keywords -> behavior constraints) ──

def infer_style(personality: str) -> dict:
    """Guess verbosity / tone / initiative from personality text."""
    p = personality.lower()

    # Verbosity
    if any(kw in p for kw in ["沉默", "寡言", "话少", "简练", "少言", "不多说"]):
        verbosity = "few_words"
    elif any(kw in p for kw in ["几乎不说", "嗯", "只言片语", "只点头", "不答"]):
        verbosity = "grunt"
    elif any(kw in p for kw in ["话多", "热情", "健谈", "滔滔不绝", "喜欢说", "爱讲"]):
        verbosity = "many_words"
    else:
        verbosity = "normal"

    # Tone
    if any(kw in p for kw in ["冷", "粗鲁", "暴躁", "刻薄", "低沉", "不近人情", "严厉"]):
        tone = "gruff"
    elif any(kw in p for kw in ["紧张", "焦虑", "不安", "恐慌", "慌张", "神经质"]):
        tone = "nervous"
    elif any(kw in p for kw in ["严谨", "学术", "冷静", "理性", "拘谨"]):
        tone = "academic"
    elif any(kw in p for kw in ["开心", "开朗", "愉快", "乐观", "活泼", "轻快"]):
        tone = "cheerful"
    else:
        tone = "neutral"

    # Initiative
    if any(kw in p for kw in ["主动", "搭话", "自来熟", "积极"]):
        initiative = "active"
    else:
        initiative = "passive"

    return {"verbosity": verbosity, "tone": tone, "initiative": initiative}


# ── Trust Defaults ──

def default_trust(topic: str) -> int:
    """Assign default trust_required based on topic name pattern."""
    t = topic.lower()
    if any(kw in t for kw in ["greeting", "hello", "intro", "greet", "问候", "你好"]):
        return 0
    if any(kw in t for kw in ["secret", "past", "hidden", "private", "秘密", "过去", "隐藏", "隐私"]):
        return 50
    if any(kw in t for kw in ["fear", "weakness", "trauma", "害怕", "恐惧", "创伤", "弱点"]):
        return 30
    if any(kw in t for kw in ["personal", "family", "personal", "家人", "个人"]):
        return 25
    return 15  # general topic


# ── NPC Merge ──

def merge_npcs(world: dict) -> dict[str, dict]:
    """Merge same-name NPC entities from world book into unified records.

    Same character appears under different entity IDs in different scenes.
    Merge by name, preserving all unique dialogue topics.
    """
    entities = world.get("entities", {})
    # Legacy compat: also check npcs dict
    legacy = world.get("npcs", {})

    merged: dict[str, dict] = {}

    def _extract(eid: str, entity: dict):
        name = entity.get("name", eid)
        if not name:
            return

        # Find or create entry
        if name not in merged:
            merged[name] = {
                "static": {
                    "name": name,
                    "profession": entity.get("profession", ""),
                    "appearance": entity.get("appearance", ""),
                    "personality": entity.get("personality", ""),
                    "style": entity.get("style") or infer_style(
                        entity.get("personality", "")
                    ),
                    "dialogue": {},
                },
                "dynamic": {
                    "stage": "stranger",
                    "trust": 0,
                    "mood": "neutral",
                    "revealed": [],
                    "interactions": [],
                    "summary": "",
                    "nicknames": [],  # player-coined nicknames for this NPC
                    "traits": [],     # runtime-extracted observable traits (读书/啐人/敲背包…)
                    "disclosure": {   # what the player is ALLOWED to be told (KP knows everything)
                        "name": False,       # has this NPC's real name been revealed to the player?
                        "background": False, # has their motive/secret/appearance detail been revealed?
                    },
                },
                "_ids": [eid],  # track original entity IDs
            }

        entry = merged[name]
        entry["_ids"].append(eid)

        # Merge static: use richer data
        s = entry["static"]
        for field in ["profession", "appearance", "personality"]:
            existing = s.get(field, "")
            new_val = entity.get(field, "")
            if new_val and len(new_val) > len(existing):
                s[field] = new_val

        # Re-infer style if personality was updated
        if s["personality"]:
            s["style"] = infer_style(s["personality"])

        # Merge dialogue topics (unique by topic name)
        dialogue = entity.get("dialogue", {})
        for topic, data in (dialogue.items() if isinstance(dialogue, dict) else {}):
            text = data if isinstance(data, str) else data.get("text", str(data))
            trust_req = data.get("trust_required") if isinstance(data, dict) else None
            if trust_req is None:
                trust_req = default_trust(topic)
            if topic not in s["dialogue"]:
                s["dialogue"][topic] = {"text": text, "trust_required": trust_req}

    # Process entities dict
    for eid, entity in entities.items():
        if entity.get("type") == "npc":
            _extract(eid, entity)

    # Process legacy npcs dict
    for npc_id, npc in legacy.items():
        npc_with_type = {**npc, "type": "npc"}
        _extract(npc_id, npc_with_type)

    # Initialize disclosure from the opening narration: an NPC whose real name
    # already appears in the opening text is known to the player (e.g. a famous
    # climber introduced by name); everyone else starts hidden until revealed
    # through play (asking their name, an event, reaching a scene…).
    opening = world.get("opening", "") or ""
    for nm, rec in merged.items():
        d = rec["dynamic"].setdefault("disclosure", {"name": False, "background": False})
        d["name"] = nm in opening

    return merged


# ── Interaction CRUD ──

def add_interaction(
    npc_dynamic: dict,
    turn: int,
    player_action: str,
    response: str,
    trust_delta: int = 0,
    revealed_topic: str | None = None,
    revealed_summary: str = "",
):
    """Record a new interaction and update dynamic state."""
    interactions = npc_dynamic.setdefault("interactions", [])
    interactions.append({
        "turn": turn,
        "player": player_action[:80],
        "response": response[:120],
        "delta": trust_delta,
        "revealed": revealed_topic,
    })

    # Update trust
    npc_dynamic["trust"] = max(-100, min(100, npc_dynamic.get("trust", 0) + trust_delta))

    # Update stage based on trust threshold
    trust = npc_dynamic["trust"]
    if trust < 0:
        npc_dynamic["stage"] = "hostile"
    elif trust < 16:
        npc_dynamic["stage"] = "stranger"
    elif trust < 41:
        npc_dynamic["stage"] = "cautious"
    elif trust < 71:
        npc_dynamic["stage"] = "trusting"
    else:
        npc_dynamic["stage"] = "close"

    # Record revealed topic
    if revealed_topic:
        revealed = npc_dynamic.setdefault("revealed", [])
        if not any(r["topic"] == revealed_topic for r in revealed):
            revealed.append({
                "topic": revealed_topic,
                "turn": turn,
                "summary": revealed_summary[:200],
            })

    # Trim interactions to 5, compress old to summary
    while len(interactions) > 5:
        old = interactions.pop(0)
        summary_parts = npc_dynamic.get("summary", "")
        npc_dynamic["summary"] = (
            f"{summary_parts}T{old['turn']}: {old['player']} -> {old['response']}\n"
        )[:500]

    return npc_dynamic


def add_nickname(npc_dynamic: dict, nickname: str) -> bool:
    """Record a player-coined nickname for this NPC (deduped).

    Returns True if newly added. Nicknames are used for O(1) reference
    resolution on later turns ("和野猪说话" -> the NPC).
    """
    nickname = (nickname or "").strip()
    if not nickname:
        return False
    nicknames = npc_dynamic.setdefault("nicknames", [])
    if nickname in nicknames:
        return False
    nicknames.append(nickname)
    return True


def add_traits(npc_dynamic: dict, traits: list[str]) -> int:
    """Record runtime-observed traits for an NPC (deduped). Used so descriptive
    references like 「读书的」「啐人的」 can resolve to the right NPC even when the
    trait only appeared in narration, not in the module's static data."""
    cur = npc_dynamic.setdefault("traits", [])
    added = 0
    for t in traits:
        t = (t or "").strip()
        if t and t not in cur:
            cur.append(t)
            added += 1
    return added


def set_mood(npc_dynamic: dict, mood: str, reason: str = ""):
    """Update NPC mood."""
    npc_dynamic["mood"] = mood
    if reason:
        npc_dynamic["_mood_reason"] = reason
    return npc_dynamic

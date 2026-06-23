# -*- coding: utf-8 -*-
"""AIKP State Manager — Session persistence, turn log, state snapshot, entity memories."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from models import (
    Session, TurnLog, PlayerState, EntityMemory,
    create_session,
)
from config import SESSIONS_DIR


# ── Session Persistence ────────────────────────────────────────

def session_path(chat_id: str) -> Path:
    return Path(SESSIONS_DIR) / f"{chat_id}.json"


def load_session(chat_id: str) -> Session:
    path = session_path(chat_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _ensure_session_defaults(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[STATE] Corrupt session file for {chat_id}: {e}", flush=True)
            # Rename corrupt file for debugging, create fresh session
            try:
                corrupt_path = path.with_suffix(".corrupt")
                path.rename(corrupt_path)
                print(f"[STATE] Renamed corrupt session to {corrupt_path}", flush=True)
            except Exception:
                pass
    return create_session(chat_id, "")


def save_session(session: Session) -> None:
    session["updated_at"] = time.time()
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path = session_path(session["chat_id"])
    # atomic write: write to temp first, then rename
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _ensure_session_defaults(data: dict) -> Session:
    defaults = create_session("", "")
    for key, default in defaults.items():
        if key not in data:
            data[key] = default
    return Session(**{k: v for k, v in data.items() if k in defaults})


# ── Turn Log ───────────────────────────────────────────────────

def write_turn_log(
    session: Session,
    turn: int,
    scene: str,
    player_input: str,
    gm_response: str,
    dice_result: Optional[dict] = None,
    entity_state_changes: Optional[dict] = None,
    npc_changes: Optional[dict] = None,
    new_flags: Optional[list] = None,
    items_obtained: Optional[list] = None,
    items_used: Optional[list] = None,
    scene_transition: Optional[str] = None,
) -> TurnLog:
    entry: TurnLog = TurnLog(
        turn=turn,
        scene=scene,
        timestamp=time.time(),
        player_input=player_input,
        gm_response=gm_response,
        dice_result=dice_result,
        entity_state_changes=entity_state_changes or {},
        npc_changes=npc_changes or {},
        new_flags=new_flags or [],
        items_obtained=items_obtained or [],
        items_used=items_used or [],
        scene_transition=scene_transition,
    )
    session.setdefault("turn_log", []).append(entry)
    session["current_turn"] = turn
    return entry


def update_last_turn_response(session: Session, gm_response: str) -> None:
    """Update the GM response text in the most recent turn log entry.
    Used by the SSE streaming handler to backfill the actual response
    after the stream completes."""
    log = session.get("turn_log", [])
    if log:
        log[-1]["gm_response"] = gm_response


# ── State Snapshot ─────────────────────────────────────────────

def compute_state_snapshot(
    session: Session,
    world: dict,
    scene_index: dict[str, list[str]],
    entity_index: dict[str, dict],
) -> str:
    """Generate natural language state summary from session data.
    Pure code computation — no LLM, no embedding. Milliseconds.
    """
    ps = session.get("player_state", {})
    current_scene_id = ps.get("current_scene", "")
    scene = world.get("scenes", {}).get(current_scene_id, {})
    entity_states = session.get("entity_states", {})
    flags = session.get("flags", [])
    dispositions = session.get("npc_dispositions", {})
    turn_log = session.get("turn_log", [])
    turn = session.get("current_turn", 0)
    plot_phase = session.get("plot_phase", "intro")

    lines = []

    # Header
    lines.append(f"=== CURRENT STATE (Turn {turn}) ===")
    lines.append(f"Scene: {scene.get('name', current_scene_id)} [visited]")

    # Investigator (from imported character card, if any)
    pc_name = ps.get("name", "")
    pc_prof = ps.get("profession", "")
    if pc_name or pc_prof or ps.get("attributes"):
        ident = f"调查员: {pc_name or '未命名'}"
        if pc_prof:
            ident += f"（{pc_prof}）"
        lines.append(ident)
        attrs = ps.get("attributes", {})
        if attrs:
            order = ["力量", "敏捷", "意志", "体质", "外貌",
                     "教育", "体型", "智力", "幸运"]
            astr = " ".join(f"{a}{attrs[a]}" for a in order if a in attrs)
            if astr:
                lines.append(f"属性: {astr}")
        # Key skills (alias-collapsed, value-bearing), capped to keep prompt lean
        from card_parser import canonical_skills
        key_skills = sorted(canonical_skills(ps.get("skills", {})).items(),
                            key=lambda kv: -kv[1])
        if key_skills:
            top = " ".join(f"{k}{v}" for k, v in key_skills[:12])
            lines.append(f"主要技能: {top}")
        bg = ps.get("background", "")
        if bg:
            lines.append(f"背景: {bg[:80]}")

    lines.append(
        f"HP: {ps.get('hp', '?')}/{ps.get('max_hp', '?')} | "
        f"SAN: {ps.get('san', '?')}/{ps.get('max_san', '?')} | "
        f"MP: {ps.get('mp', '?')}/{ps.get('max_mp', '?')} | "
        f"Inventory: {', '.join(ps.get('inventory', [])) or 'empty'}"
    )
    lines.append(f"Plot phase: {plot_phase}")

    # Entities in current scene
    scene_entity_ids = scene_index.get(current_scene_id, [])
    if scene_entity_ids:
        lines.append("")
        lines.append("Entities here:")
        for eid in scene_entity_ids:
            einfo = entity_index.get(eid, {})
            etype = einfo.get("type", "?")
            ename = einfo.get("name", eid)
            estate = entity_states.get(eid, "unknown")
            entity = world.get("entities", {}).get(eid, {})
            state_def = entity.get("states", {}).get(estate, {})
            desc = state_def.get("description") or entity.get("description") or entity.get("personality", "")
            # Truncate long descriptions
            if len(desc) > 80:
                desc = desc[:77] + "..."
            icon = _entity_state_icon(etype, estate)
            lines.append(f"  {icon} {ename} [{estate.upper()}] — {desc}")

    # Flags
    if flags:
        lines.append("")
        lines.append(f"Global flags: {', '.join(flags)}")

    # NPC dispositions
    if dispositions:
        lines.append("")
        lines.append("NPC dispositions:")
        for npc_id, val in dispositions.items():
            ninfo = entity_index.get(npc_id, {})
            nname = ninfo.get("name", npc_id)
            mood = _disposition_label(val)
            lines.append(f"  {nname}: {mood} ({val:+d})")

    # Recent turns summary
    recent_logs = turn_log[-5:] if len(turn_log) >= 5 else turn_log
    if recent_logs and len(turn_log) > 0:
        lines.append("")
        lines.append("Recent events:")
        for entry in recent_logs:
            t = entry.get("turn", "?")
            inp = entry.get("player_input", "")
            if len(inp) > 60:
                inp = inp[:57] + "..."
            lines.append(f"  T{t}: {inp}")
            if entry.get("dice_result"):
                dr = entry["dice_result"]
                lines.append(
                    f"      → {dr.get('skill_name', '?')} check: "
                    f"d20={dr['d20']} + {dr.get('skill_value', 0)} = "
                    f"{dr.get('total', '?')} (DC={dr.get('difficulty', '?')}) "
                    f"[{dr.get('verdict', '?').upper()}]"
                )
            if entry.get("entity_state_changes"):
                for eid, change in entry["entity_state_changes"].items():
                    e_name = entity_index.get(eid, {}).get("name", eid)
                    lines.append(f"      → {e_name}: {change}")

    return "\n".join(lines)


def _entity_state_icon(entity_type: str, state: str) -> str:
    icons = {
        "clue": {"hidden": "[?]", "found": "[!]", "read": "[i]", "default": "[?]"},
        "item": {"default": "[-]", "in_inventory": "[+]", "used": "[x]"},
        "npc": {"default": "[@]"},
        "door": {"locked": "[L]", "unlocked": "[U]", "default": "[-]"},
        "container": {"closed": "[-]", "opened": "[+]", "default": "[-]"},
    }
    type_icons = icons.get(entity_type, {"default": "[ ]"})
    return type_icons.get(state, type_icons.get("default", "[ ]"))


def _disposition_label(value: int) -> str:
    if value >= 70:
        return "devoted"
    if value >= 40:
        return "friendly"
    if value >= 10:
        return "cooperative"
    if value >= -10:
        return "neutral"
    if value >= -40:
        return "suspicious"
    if value >= -70:
        return "hostile"
    return "vengeful"


# ── Entity Memory Extraction ───────────────────────────────────

def extract_entity_memories(
    turn_log_entry: TurnLog,
    entity_index: dict[str, dict],
) -> list[EntityMemory]:
    """After each turn, extract new entity memories from the log entry."""
    memories: list[EntityMemory] = []
    turn = turn_log_entry["turn"]
    scene = turn_log_entry["scene"]

    # Entity state changes → "state_change" or "discovery" memory
    for eid, change in turn_log_entry.get("entity_state_changes", {}).items():
        einfo = entity_index.get(eid, {})
        etype = einfo.get("type", "")
        ename = einfo.get("name", eid)
        old_state, new_state = _parse_state_change(change)

        mem_type = "discovery" if etype == "clue" and new_state in ("found", "read") else "state_change"
        importance = 1.0 if etype == "clue" and new_state in ("found", "read") else 0.5

        memories.append(EntityMemory(
            entity_id=eid,
            turn=turn,
            type=mem_type,
            summary=f"{ename}: {old_state} → {new_state}",
            importance=importance,
            scene=scene,
        ))

    # NPC disposition changes → "dialogue" memory
    for npc_id, changes in turn_log_entry.get("npc_changes", {}).items():
        ninfo = entity_index.get(npc_id, {})
        nname = ninfo.get("name", npc_id)
        disp_change = changes.get("disposition")
        if disp_change is not None:
            memories.append(EntityMemory(
                entity_id=npc_id,
                turn=turn,
                type="dialogue",
                summary=f"{nname} disposition changed by {disp_change:+d}",
                importance=0.5,
                scene=scene,
            ))

    return memories


def _parse_state_change(change_str: str) -> tuple[str, str]:
    if "→" in change_str:
        parts = change_str.split("→")
        return parts[0].strip(), parts[1].strip()
    return "?", change_str


def append_entity_memories(session: Session, memories: list[EntityMemory]) -> None:
    memories_dict = session.setdefault("entity_memories", {})
    for mem in memories:
        eid = mem["entity_id"]
        memories_dict.setdefault(eid, []).append(mem)


# ── Conversation Summary ───────────────────────────────────────

def get_recent_conversation(session: Session, raw_turns: int = 5) -> str:
    """Get last N turns of raw conversation for prompt."""
    log = session.get("turn_log", [])
    recent = log[-raw_turns:] if len(log) >= raw_turns else log
    if not recent:
        return ""

    lines = ["=== RECENT CONVERSATION ==="]
    for entry in recent:
        lines.append(f"[Turn {entry['turn']}] Player: {entry['player_input']}")
        gm = entry.get("gm_response", "")
        if len(gm) > 200:
            gm = gm[:197] + "..."
        lines.append(f"GM: {gm}")
        lines.append("")
    return "\n".join(lines)


def get_or_compress_conversation_summary(
    session: Session,
    api_key: str,
    base_url: str = "https://api.deepseek.com/v1",
    compress_every: int = 10,
) -> str:
    """Get mid-range conversation summary.
    Compresses turns [5:15] via LLM every `compress_every` turns.
    Returns existing summary if not due for recompression.
    """
    turn = session.get("current_turn", 0)
    log = session.get("turn_log", [])

    # Check cached summary
    cached = session.get("_cached_summary", "")
    cached_at = session.get("_cached_summary_turn", 0)

    if cached and (turn - cached_at) < compress_every:
        return cached

    # Determine range to summarize: turns older than 5 but younger than 15
    if len(log) <= 5:
        return ""

    start_idx = max(0, len(log) - 15)
    end_idx = max(0, len(log) - 5)
    to_compress = log[start_idx:end_idx]

    if not to_compress:
        return cached  # keep old summary if no new data to compress

    # Build compression prompt
    narrative_parts = []
    for entry in to_compress:
        narrative_parts.append(
            f"[T{entry['turn']}] {entry['player_input']}\n"
            f"GM: {entry.get('gm_response', '')[:300]}"
        )

    prompt = (
        "Summarize the following RPG session events in 3-5 concise sentences. "
        "Focus on: what the player did, what they discovered, NPC interactions, "
        "and important decisions. Keep it brief.\n\n"
        + "\n\n".join(narrative_parts)
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            stream=False,
        )
        summary = resp.choices[0].message.content or ""
        session["_cached_summary"] = summary
        session["_cached_summary_turn"] = turn
        return summary
    except Exception as e:
        print(f"[STATE] Summary compression failed: {e}", flush=True)
        return cached  # fallback to old summary


# ── Session Integration Helpers ────────────────────────────────

def initialize_session_from_world(session: Session, world: dict) -> Session:
    """Set initial scene and NPC states for a new session based on world book."""
    from npc_state import merge_npcs

    scenes = world.get("scenes", {})
    entities = world.get("entities", {})

    # Pick first scene: explicit starting_scene → plot_outline → first in dict
    first_scene_id = world.get("starting_scene", "")
    if not first_scene_id:
        plot = world.get("plot_outline", {})
        if isinstance(plot, dict):
            phases = plot.get("phases", [])
            if phases and isinstance(phases, list):
                first_scene_id = phases[0].get("starting_scene", "")
    if not first_scene_id:
        first_scene_id = next(iter(scenes.keys())) if scenes else ""

    # Validate: a parsed starting_scene that isn't a real scene key.
    # Try resolving by scene name (parser may emit Chinese name instead of ID).
    if first_scene_id not in scenes:
        matched = next(
            (sid for sid, s in scenes.items()
             if isinstance(s, dict) and s.get("name", "") == first_scene_id),
            None
        )
        first_scene_id = matched if matched else (next(iter(scenes.keys())) if scenes else "")

    session["player_state"]["current_scene"] = first_scene_id
    session["model"] = world.get("name", "")

    # Rule system from world book
    rule_system = world.get("rule_system", "dnd")
    session["rule_system"] = rule_system

    # CoC default skills (Chinese names to match world book check fields)
    if rule_system == "coc":
        session["player_state"]["skills"] = {
            "侦查": 25, "聆听": 20, "图书馆利用": 20,
            "心理学": 10, "意志": 50, "力量": 50,
            "敏捷": 50, "体质": 50, "外貌": 50,
            "体型": 50, "智力": 60, "教育": 60,
            "幸运": 50, "神秘学": 5, "医学": 5,
            "导航": 10, "攀爬": 20, "游泳": 20,
            "闪避": 25, "斗殴": 25, "射击": 25,
            "驾驶": 20, "电气维修": 10, "机械维修": 10,
            "说服": 10, "话术": 5, "恐吓": 15, "魅惑": 15,
        }
        session["player_state"]["san"] = 50
        session["player_state"]["max_san"] = 99

    # Initialize entity states to their initial_state
    for eid, entity in entities.items():
        session["entity_states"][eid] = entity.get("initial_state", "default")

    # Build NPC states from world book entities (merge same-name NPCs)
    session["npc_states"] = merge_npcs(world)

    # Initialize beat tracking from story_beats
    beats = world.get("story_beats", [])
    if beats:
        session["current_beat_id"] = beats[0].get("id", "")
        session["completed_beats"] = []
        session["unlocked_scenes"] = []

    return session

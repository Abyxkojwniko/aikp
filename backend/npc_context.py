# -*- coding: utf-8 -*-
"""AIKP NPC Context — 6-layer prompt assembly for GM injection.

Design: docs/npc-design.md

Layers:
  0: RULES  (forever cache)  — handled by config.py GM_SYSTEM_PROMPT
  1: SCENE  (per scene)      — build_scene_layer()
  2: STORY  (per scene)      — build_story_layer()
  3: NPC_HIT (per turn)      — build_npc_hit()
  4: RETRIEVAL               — delegated to npc_retrieval.py
  5: RECENT                  — get_recent_conversation() from state_manager
"""

from __future__ import annotations


# ── Keyword Matching ───────────────────────────────────────────

def keyword_match(player_input: str, scene_entity_ids: list[str],
                  entity_index: dict[str, dict]) -> list[str]:
    """Return NPC entity IDs mentioned in player input.

    Matches by: entity name (space-tolerant), or entity ID substring.
    """
    inp = player_input.lower()
    inp_nospace = inp.replace(" ", "")
    matched = []
    for eid in scene_entity_ids:
        einfo = entity_index.get(eid, {})
        if einfo.get("type") != "npc":
            continue
        name = (einfo.get("name") or "").lower()
        name_nospace = name.replace(" ", "")
        if name and (name in inp or name_nospace in inp_nospace):
            matched.append(eid)
        elif eid.lower() in inp:
            matched.append(eid)
    return matched


# ── Layer 1: Scene ─────────────────────────────────────────────

def build_scene_layer(scene_id: str, world: dict,
                      scene_index: dict[str, list[str]],
                      entity_index: dict[str, dict]) -> str:
    """Build SCENE layer: location + present NPC/item names (1 line each)."""
    scenes = world.get("scenes", {})
    scene = scenes.get(scene_id, {})

    lines = []
    name = scene.get("name", scene_id)
    desc = scene.get("desc", "") or scene.get("description", "")
    lines.append(f"=== {name} ===")
    lines.append(desc)

    # Exits
    exits = scene.get("exits", {})
    if isinstance(exits, dict) and exits:
        lines.append("Exits: " + " | ".join(exits.keys()))
    elif isinstance(exits, list) and exits:
        exit_names = []
        for eid in exits:
            ts = scenes.get(eid, {})
            exit_names.append(ts.get("name", eid))
        lines.append("Exits: " + " | ".join(exit_names))

    # Present NPCs (name only, 1 line each)
    eids = scene_index.get(scene_id, [])
    npc_ids = [eid for eid in eids
               if entity_index.get(eid, {}).get("type") == "npc"]
    if npc_ids:
        npc_lines = _npc_summary_lines(npc_ids, entity_index)
        lines.extend(npc_lines)

    # Present items/entities (name only, 1 line each)
    other_ids = [eid for eid in eids
                 if entity_index.get(eid, {}).get("type") != "npc"]
    if other_ids:
        item_lines = _entity_summary_lines(other_ids, entity_index, world)
        lines.extend(item_lines)

    return "\n".join(lines)


def _npc_summary_lines(npc_ids: list[str], entity_index: dict[str, dict]) -> list[str]:
    """One line per NPC: name + profession."""
    result = []
    for eid in npc_ids:
        einfo = entity_index.get(eid, {})
        name = einfo.get("name", eid)
        prof = einfo.get("profession", "")
        if prof:
            result.append(f"[NPC] {name} — {prof}")
        else:
            result.append(f"[NPC] {name}")
    return result


def _entity_summary_lines(entity_ids: list[str], entity_index: dict[str, dict],
                          world: dict) -> list[str]:
    """One line per entity: name + check info if any."""
    result = []
    entities = world.get("entities", {})
    for eid in entity_ids:
        einfo = entity_index.get(eid, {})
        name = einfo.get("name", eid)
        etype = einfo.get("type", "?")
        entity = entities.get(eid, {})
        check_str = ""
        # Check states for checks
        for state_def in entity.get("states", {}).values():
            if isinstance(state_def, dict):
                chk = state_def.get("check", "")
                if chk:
                    check_str = f" [{chk}]"
                    break
                chk = state_def.get("san_check", "")
                if chk:
                    check_str = f" [SAN {chk}]"
                    break
        result.append(f"[{etype}] {name}{check_str}")
    return result


# ── Layer 2: Story ─────────────────────────────────────────────

def build_story_layer(session: dict) -> str:
    """Return LLM-compressed summary of prior scenes (STORY layer).

    Cached as session["_story_summary"]. Regenerated on scene transition.
    """
    return session.get("_story_summary", "")


def compress_story(session: dict, api_key: str = "",
                   base_url: str = "https://api.deepseek.com/v1") -> str:
    """Compress prior scene turn_log into story summary via LLM.

    Called on scene transition. Merges hard facts (trust changes,
    revelations, flags, decisions) into a natural 3-5 sentence paragraph.
    """
    log = session.get("turn_log", [])
    if not log:
        return ""

    current_scene = session["player_state"].get("current_scene", "")
    # Only summarize turns from old scene
    current_scene_turns = [t for t in log if t.get("scene") == current_scene]
    old_turns = [t for t in log if t.get("scene") != current_scene]

    # Build narrative lines from turn_log
    narrative = []
    for entry in old_turns[-20:]:
        t = entry.get("turn", "?")
        inp = entry.get("player_input", "")[:100]
        resp = entry.get("gm_response", "")[:150]
        narrative.append(f"T{t}: {inp}\n  GM: {resp}")

    # Include NPC state changes from old turns
    for entry in old_turns[-20:]:
        for change in entry.get("entity_state_changes", {}).items():
            pass  # already in narrative
        for change in entry.get("npc_changes", {}).items():
            narrative.append(f"  NPC changed: {change}")

    if not narrative:
        return session.get("_story_summary", "")

    prompt = (
        "Summarize the following RPG session events in 3-5 concise sentences. "
        "Include: key discoveries, NPC relationship changes, important decisions. "
        "Keep it brief. Output only the summary text, no markdown.\n\n"
        + "\n".join(narrative)
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        summary = resp.choices[0].message.content or ""
        session["_story_summary"] = (session.get("_story_summary", "") +
                                     "\n" + summary).strip()
        return session["_story_summary"]
    except Exception as e:
        print(f"[NPC_CTX] Story compression failed: {e}", flush=True)
        return session.get("_story_summary", "")


# ── Layer 3: NPC Hit ───────────────────────────────────────────

def build_npc_hit(matched_npc_ids: list[str], session: dict,
                  entity_index: dict[str, dict]) -> str:
    """Build NPC_HIT layer for NPCs the player is currently interacting with.

    Includes: personality, style, trust, revealed list, hidden topics, recent interactions.
    Only one paragraph per NPC.
    """
    npc_states = session.get("npc_states", {})
    if not matched_npc_ids or not npc_states:
        return ""

    lines = []
    for eid in matched_npc_ids[:2]:  # max 2 NPCs to limit tokens
        einfo = entity_index.get(eid, {})
        name = einfo.get("name", eid)

        # Find merged NPC state by name
        state = _find_npc_state(name, npc_states)
        if not state:
            continue

        static = state.get("static", {})
        dynamic = state.get("dynamic", {})

        # Header: name + style + trust
        style = static.get("style", {})
        lines.append(
            f"[{name}] "
            f"{static.get('profession', '')} | "
            f"trust:{dynamic.get('trust', 0)} | "
            f"{dynamic.get('mood', 'neutral')}"
        )

        # Personality
        pers = static.get("personality", "")
        if pers:
            lines.append(f"  Personality: {pers[:120]}")

        # Revealed topics
        revealed = dynamic.get("revealed", [])
        if revealed:
            lines.append(f"  Already told player:")
            for r in revealed:
                lines.append(f"    T{r['turn']} {r['topic']}: {r['summary'][:100]}")

        # Hidden topics
        dialogue = static.get("dialogue", {})
        unrevealed = [
            (t, d) for t, d in dialogue.items()
            if not any(r["topic"] == t for r in revealed)
        ]
        if unrevealed:
            lines.append(f"  Not yet revealed:")
            for topic, data in unrevealed:
                trust_req = data.get("trust_required", 0)
                req_flag = data.get("requires_flag", "")
                flag_str = f" + {req_flag}" if req_flag else ""
                lines.append(f"    {topic} (trust>{trust_req}{flag_str})")

        # Recent interactions
        interactions = dynamic.get("interactions", [])
        if interactions:
            lines.append(f"  Recent interactions:")
            for ix in interactions[-3:]:
                lines.append(
                    f"    T{ix['turn']}: {ix['player'][:60]} "
                    f"-> {ix['response'][:80]}"
                )

        lines.append("")

    return "\n".join(lines).strip()


def _find_npc_state(name: str, npc_states: dict) -> dict | None:
    """Find NPC state by name, with fuzzy matching."""
    name_lower = name.lower()
    # Exact match
    if name_lower in {k.lower() for k in npc_states}:
        for k, v in npc_states.items():
            if k.lower() == name_lower:
                return v
    # Substring match
    for k, v in npc_states.items():
        if name_lower in k.lower() or k.lower() in name_lower:
            return v
    return None

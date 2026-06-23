# -*- coding: utf-8 -*-
"""AIKP Plot Pusher — stall detection + NPC-driven plot progression.

Design: docs/npc-design.md

Logic:
  1. Detect stall: no new discoveries in >= 5 turns, player has not said
     "wait / stay / examine more"
  2. Generate push: select NPC with unlocked-but-unrevealed topic
  3. Fallback: suggest scene transition
"""

from __future__ import annotations

STALL_THRESHOLD = 5  # turns without discovery before intervention

STALL_KEYWORDS = [
    "等等", "等一下", "先看看", "再观察", "再检查", "再搜索",
    "先不急着", "再待一会", "再转转", "等一下再说",
    "wait", "stay", "hold on", "one more look", "look around again",
]


def detect_stall(session: dict) -> bool:
    """Check if the game is stalled (no discovery for N turns).

    Returns True if stalled, False otherwise.
    """
    turn = session.get("current_turn", 0)
    turn_log = session.get("turn_log", [])

    if turn < 3:
        return False  # too early

    # Check last N turns for any new discoveries
    last_turns = turn_log[-STALL_THRESHOLD:]
    if len(last_turns) < STALL_THRESHOLD:
        return False

    for entry in last_turns:
        # Entity state change = discovery
        if entry.get("entity_state_changes"):
            return False
        # New flags = discovery
        if entry.get("new_flags"):
            return False
        # Items obtained = discovery
        if entry.get("items_obtained"):
            return False

    # Check player didn't explicitly say "wait"
    for entry in last_turns:
        inp = entry.get("player_input", "").lower()
        for kw in STALL_KEYWORDS:
            if kw in inp:
                return False  # player intentionally staying

    return True


def generate_push(session: dict, scene_index: dict[str, list[str]],
                  entity_index: dict[str, dict], world: dict) -> str:
    """Generate a push hint when the game is stalled.

    Priority:
      1. NPC with unlocked topic not yet revealed -> hint at them
      2. Undiscovered entity in scene -> hint at it
      3. Scene transition suggestion

    Returns a single line of hint text, or empty string.
    """
    if not detect_stall(session):
        return ""

    current_scene = session["player_state"].get("current_scene", "")
    scene_entities = scene_index.get(current_scene, [])
    npc_states = session.get("npc_states", {})

    # Priority 1: NPC with unlocked topic
    npc_ids = [eid for eid in scene_entities
               if entity_index.get(eid, {}).get("type") == "npc"]
    for eid in npc_ids:
        einfo = entity_index.get(eid, {})
        name = einfo.get("name", eid)

        # Find merged NPC state
        state = _find_npc(name, npc_states)
        if not state:
            continue

        static = state.get("static", {})
        dynamic = state.get("dynamic", {})
        trust = dynamic.get("trust", 0)
        revealed = {r["topic"] for r in dynamic.get("revealed", [])}
        dialogue = static.get("dialogue", {})

        for topic, data in dialogue.items():
            trust_req = data.get("trust_required", 0)
            if topic not in revealed and trust >= trust_req:
                return f"{name}似乎欲言又止，瞥了{data.get('hint_text', '你')}一眼。"

    # Priority 2: Undiscovered entity
    entity_states = session.get("entity_states", {})
    other_ids = [eid for eid in scene_entities
                 if entity_index.get(eid, {}).get("type") != "npc"]
    for eid in other_ids:
        state = entity_states.get(eid, "default")
        if state in ("default", "hidden", "present", "unknown"):
            einfo = entity_index.get(eid, {})
            name = einfo.get("name", eid)
            return f"你注意到{name}似乎有什么不同寻常之处。"

    # Priority 3: Suggest scene transition
    scenes = world.get("scenes", {})
    scene = scenes.get(current_scene, {})
    exits = scene.get("exits", {})
    if isinstance(exits, dict) and exits:
        exit_names = list(exits.keys())
        return f"天色不早了。或许该{exit_names[0]}了。"
    elif isinstance(exits, list) and exits:
        ts = scenes.get(exits[0], {})
        target_name = ts.get("name", exits[0])
        return f"天色不早了。或许该前往{target_name}了。"

    return ""


def _find_npc(name: str, npc_states: dict) -> dict | None:
    """Fuzzy-find NPC state by name."""
    name_lower = name.lower()
    for k, v in npc_states.items():
        if k.lower() == name_lower or name_lower in k.lower() or k.lower() in name_lower:
            return v
    return None

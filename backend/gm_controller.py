# -*- coding: utf-8 -*-
"""AIKP GM Controller — Beat-based story progression + anti-derailment."""

from __future__ import annotations
from models import Session


# ── Legacy phase constants (kept for fallback) ────────────────────
PLOT_PHASES = ("intro", "investigation", "climax", "resolution")
PHASE_TRANSITIONS = {
    "intro": "investigation",
    "investigation": "climax",
    "climax": "resolution",
    "resolution": None,
}


# ── Beat-aware story check ─────────────────────────────────────────

def check_story_beat(session: Session, world: dict) -> dict:
    """Evaluate current story beat; advance if conditions met. Returns KP context dict.

    Falls back to check_plot_phase() if world has no story_beats."""
    story_beats = world.get("story_beats", [])
    if not story_beats:
        return check_plot_phase(session)

    discovered = set(session.get("discovered_clues", []))
    current_beat_id = session.get("current_beat_id", "")
    completed = list(session.get("completed_beats", []))

    # Find current beat index
    idx = 0
    if current_beat_id:
        for i, b in enumerate(story_beats):
            if b.get("id") == current_beat_id:
                idx = i
                break
    else:
        current_beat_id = story_beats[0].get("id", "")
        session["current_beat_id"] = current_beat_id

    current_beat = story_beats[idx]
    result = {
        "current_beat": current_beat,
        "beat_advanced": False,
        "new_beat": None,
        "hints": [],
        "derailment_level": 0,
        "derailment_action": None,
        "phase": session.get("plot_phase", "investigation"),
    }

    # Check advance condition
    critical = current_beat.get("critical_clues", [])
    advance_when = current_beat.get("advance_when", "any_critical")
    current_scene = session["player_state"].get("current_scene", "")

    if critical:
        found_critical = [c for c in critical if c in discovered]
        if advance_when == "all_critical":
            should_advance = len(found_critical) >= len(critical)
        else:
            should_advance = len(found_critical) >= 1
    elif advance_when == "visited":
        beat_scenes = current_beat.get("scenes", [])
        should_advance = bool(beat_scenes and current_scene in beat_scenes)
    else:
        should_advance = False

    # Advance to next beat
    if should_advance and idx + 1 < len(story_beats):
        next_beat = story_beats[idx + 1]
        next_id = next_beat.get("id", "")
        if next_id and next_id not in completed:
            completed.append(current_beat_id)
            session["completed_beats"] = completed
            session["current_beat_id"] = next_id
            for sid in next_beat.get("unlocks_scenes", []):
                ul = session.setdefault("unlocked_scenes", [])
                if sid not in ul:
                    ul.append(sid)
            result["beat_advanced"] = True
            result["new_beat"] = next_beat
            result["current_beat"] = next_beat
            print(f"[CONTROLLER] Beat: {current_beat_id} → {next_id}", flush=True)

    # Hints: missing critical clues in current scene
    active_beat = result["current_beat"]
    result["hints"] = _beat_hints(active_beat, discovered, current_scene, world)

    # Anti-derailment
    turn_log = session.get("turn_log", [])
    turns_since = _count_turns_since_discovery(turn_log)
    phase = session.get("plot_phase", "investigation")
    result["derailment_level"] = _derailment_level(turns_since, phase)
    result["derailment_action"] = _derailment_action(result["derailment_level"])

    return result


def _beat_hints(beat: dict, discovered: set, current_scene: str, world: dict) -> list[str]:
    hints = []
    critical = beat.get("critical_clues", [])
    missing = [c for c in critical if c not in discovered]
    if not missing:
        return hints
    scene_clues = world.get("scenes", {}).get(current_scene, {}).get("clues", [])
    missing_here = [c for c in scene_clues
                    if isinstance(c, dict) and c.get("id") in missing]
    if missing_here:
        parts = []
        for c in missing_here[:2]:
            chk = f"（{c['check']}）" if c.get("check") else ""
            parts.append(f"{c.get('desc', c.get('id', ''))[:30]}{chk}")
        hints.append("本场景有关键线索未发现：" + "；".join(parts))
    return hints


# ── Beat context for assemble_context ─────────────────────────────

def build_beat_context(world: dict, session: dict, current_scene_id: str) -> str:
    """Build KP story-progress block. Uses story_beats if available,
    falls back to legacy plot_outline block."""
    story_beats = world.get("story_beats", [])
    if not story_beats:
        return _legacy_plot_block(world, session, current_scene_id)

    current_beat_id = session.get("current_beat_id", "")
    completed = session.get("completed_beats", [])
    discovered = set(session.get("discovered_clues", []))

    idx = 0
    for i, b in enumerate(story_beats):
        if b.get("id") == current_beat_id:
            idx = i
            break
    beat = story_beats[idx]

    lines = ["=== 剧情节拍（KP视角）==="]

    if completed:
        lines.append("已完成：" + " → ".join(completed))

    lines.append(f"当前节拍：【{beat.get('name', current_beat_id)}】")
    if beat.get("kp_note"):
        lines.append(f"  {beat['kp_note']}")

    # Clue status
    critical = beat.get("critical_clues", [])
    found = [c for c in critical if c in discovered]
    missing = [c for c in critical if c not in discovered]
    if found:
        lines.append(f"  ✓ 已找到：{', '.join(found)}")
    if missing:
        scene_clues = world.get("scenes", {}).get(current_scene_id, {}).get("clues", [])
        here = [c for c in scene_clues
                if isinstance(c, dict) and c.get("id") in missing]
        not_here = [m for m in missing
                    if not any(isinstance(c, dict) and c.get("id") == m
                               for c in scene_clues)]
        if here:
            lines.append("  ✗ 本场景未找到的关键线索：")
            for c in here:
                chk = f"（需要 {c['check']}）" if c.get("check") else ""
                lines.append(f"    · {c.get('desc', c.get('id', ''))}{chk}")
                if c.get("reveals"):
                    lines.append(f"      发现后揭示：{c['reveals']}")
        if not_here:
            lines.append(f"  ✗ 其他场景还缺：{', '.join(not_here)}")

    # Current scene purpose
    purpose = world.get("scenes", {}).get(current_scene_id, {}).get("purpose", "")
    if purpose:
        lines.append(f"当前场景目的：{purpose}")

    # Next beat preview
    if idx + 1 < len(story_beats):
        nxt = story_beats[idx + 1]
        lines.append(f"下一节拍（条件满足时）：{nxt.get('name', '')}")
        for sid in nxt.get("unlocks_scenes", [])[:2]:
            s = world.get("scenes", {}).get(sid, {})
            if s:
                lines.append(f"  → 将开放：{s.get('name', sid)}")

    lines.append("（以上是KP专用视角，用来自然引导故事，不要直接剧透给玩家）")
    return "\n".join(lines)


# ── Legacy plot_outline fallback ───────────────────────────────────

def _legacy_plot_block(world: dict, session: dict, current_scene_id: str) -> str:
    """Original plot_outline logic for modules without story_beats."""
    plot_outline = world.get("plot_outline")
    phases = _flatten_plot_phases(plot_outline)
    if not phases:
        return ""

    current_phase_id = session.get("plot_phase", "")
    idx = 0
    for i, p in enumerate(phases):
        if p.get("scene") == current_scene_id:
            idx = i
            break
        if p.get("phase_id") == current_phase_id:
            idx = i
    matched_by_scene = any(p.get("scene") == current_scene_id for p in phases)
    if matched_by_scene:
        session["plot_phase"] = phases[idx]["phase_id"]

    current = phases[idx]
    flags = session.get("flags", [])
    lines = ["=== 剧情进度（KP视角）==="]
    if idx > 0:
        lines.append("已完成阶段：" + " → ".join(p["phase_id"] for p in phases[:idx]))
    lines.append(f"当前阶段：{current['phase_id']}")
    if current.get("event"):
        lines.append(f"  关键事件：{current['event']}")
    scene_clues = world.get("scenes", {}).get(current_scene_id, {}).get("clues", [])
    pending = [c for c in scene_clues
               if isinstance(c, dict) and f"{c.get('id', '')}_discovered" not in flags]
    if pending:
        lines.append("  本场景待发现：")
        for c in pending:
            chk = f"（需要 {c['check']}）" if c.get("check") else ""
            lines.append(f"    · {c.get('desc', c.get('id', ''))}{chk}")
    if idx + 1 < len(phases):
        nxt = phases[idx + 1]
        lines.append(f"下一阶段：{nxt['phase_id']}")
        if nxt.get("event"):
            lines.append(f"  将发生：{nxt['event']}")
        if nxt.get("scene"):
            nxt_scene = world.get("scenes", {}).get(nxt["scene"], {})
            lines.append(f"  → 引导前往：{nxt_scene.get('name', nxt['scene'])}")
    lines.append("（以上是KP专用视角，不要直接剧透给玩家）")
    return "\n".join(lines)


def _flatten_plot_phases(plot_outline) -> list[dict]:
    if not plot_outline:
        return []
    if isinstance(plot_outline, list):
        result = []
        for i, p in enumerate(plot_outline):
            if isinstance(p, dict):
                result.append({"phase_id": p.get("phase_id", str(i)), **p})
            elif isinstance(p, str):
                result.append({"phase_id": str(i), "event": p})
        return result
    if isinstance(plot_outline, dict):
        result = []
        for k, v in plot_outline.items():
            if isinstance(v, str):
                result.append({"phase_id": k, "event": v})
            elif isinstance(v, dict):
                if any(key in v for key in ("scene", "event", "clues", "checks")):
                    result.append({"phase_id": k, **v})
                else:
                    for sk, sv in v.items():
                        if isinstance(sv, dict):
                            result.append({"phase_id": f"{k}.{sk}", **sv})
                        elif isinstance(sv, str):
                            result.append({"phase_id": f"{k}.{sk}", "event": sv})
        return result
    return []


# ── Legacy check_plot_phase (fallback when no story_beats) ────────

def check_plot_phase(session: Session) -> dict:
    phase = session.get("plot_phase", "intro")
    flags = session.get("flags", [])
    turn_log = session.get("turn_log", [])
    current_scene = session["player_state"].get("current_scene", "")
    clues_found = sum(1 for f in flags if f.endswith("_discovered"))
    scenes_visited = len(set(e.get("scene", "") for e in turn_log if e.get("scene")))
    result = {
        "phase": phase,
        "should_advance": False,
        "new_phase": None,
        "hints": [],
        "derailment_level": 0,
        "derailment_action": None,
        "current_beat": None,
    }
    if phase == "intro":
        if scenes_visited >= 2 or clues_found >= 1:
            result["should_advance"] = True
            result["new_phase"] = "investigation"
    elif phase == "investigation":
        if clues_found >= 2 or current_scene in ("climax_scene",):
            result["should_advance"] = True
            result["new_phase"] = "climax"
    elif phase == "climax":
        if "climax_resolved" in flags:
            result["should_advance"] = True
            result["new_phase"] = "resolution"
    if result["should_advance"] and result["new_phase"]:
        session["plot_phase"] = result["new_phase"]
        print(f"[CONTROLLER] Phase: {phase} → {result['new_phase']}", flush=True)
    turns_since = _count_turns_since_discovery(turn_log)
    result["derailment_level"] = _derailment_level(turns_since, phase)
    result["derailment_action"] = _derailment_action(result["derailment_level"])
    result["hints"].extend(_generate_hints(session, phase))
    return result


# ── inject_controller_context ──────────────────────────────────────

def inject_controller_context(controller_result: dict) -> str:
    parts = []
    beat = controller_result.get("current_beat")
    if beat:
        name = beat.get("name", "")
        note = beat.get("kp_note", "")
        if name or note:
            parts.append(f"PLOT BEAT: {name}. {note}".strip())
    else:
        phase_ctx = get_phase_context(controller_result.get("phase", "intro"))
        if phase_ctx:
            parts.append(phase_ctx)
    hints = controller_result.get("hints", [])
    if hints:
        parts.append("GM GUIDANCE: " + " | ".join(hints))
    derailment = controller_result.get("derailment_action")
    if derailment:
        parts.append("PACING NOTE: " + derailment)
    return "\n".join(parts) if parts else ""


# ── Shared helpers ─────────────────────────────────────────────────

def _count_turns_since_discovery(turn_log: list) -> int:
    count = 0
    for entry in reversed(turn_log):
        if entry.get("entity_state_changes") or entry.get("new_flags"):
            return count
        if entry.get("scene_transition"):
            return count
        count += 1
    return count


def _derailment_level(turns_since_clue: int, phase: str) -> int:
    if phase in ("climax", "resolution"):
        return 0
    if turns_since_clue <= 3:
        return 0
    if turns_since_clue <= 5:
        return 1
    if turns_since_clue <= 8:
        return 2
    if turns_since_clue <= 12:
        return 3
    return 4


def _derailment_action(level: int) -> str | None:
    if level == 0:
        return None
    if level == 1:
        return "Subtle hint: include an environmental detail that might point toward something worth investigating."
    if level == 2:
        return "NPC nudge: a relevant NPC could approach or speak, offering a clue or direction."
    if level == 3:
        return "Explicit reminder: gently reiterate the current objective or mention an undiscovered clue."
    return "Soft redirect: the environment naturally guides toward the main plot area. Never say 'you can't'."


def _generate_hints(session: Session, phase: str) -> list[str]:
    hints = []
    if phase != "investigation":
        return hints
    turn = session.get("current_turn", 0)
    if turn % 5 != 0:
        return hints
    entity_states = session.get("entity_states", {})
    clues_in_scene = [eid for eid, state in entity_states.items()
                      if state in ("hidden", "visible")]
    if clues_in_scene:
        hints.append("HINT: There are things in this scene that could be investigated.")
    return hints


def get_phase_context(phase: str) -> str:
    contexts = {
        "intro": (
            "PLOT PHASE: INTRODUCTION. Emphasize atmosphere, describe the environment vividly, "
            "introduce NPCs naturally. Softly guide toward the first clue or exit."
        ),
        "investigation": (
            "PLOT PHASE: INVESTIGATION. Describe scenes with investigative detail. "
            "NPCs should reveal information when asked the right questions. Maintain mystery and tension."
        ),
        "climax": (
            "PLOT PHASE: CLIMAX. Describe with urgency. The truth is close. Consequences are real."
        ),
        "resolution": (
            "PLOT PHASE: RESOLUTION. Describe the aftermath. Tie up loose threads."
        ),
    }
    return contexts.get(phase, "")

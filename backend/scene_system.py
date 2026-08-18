# -*- coding: utf-8 -*-
"""Authoritative scene discovery and destination selection.

The narrator may interpret a movement request, but it cannot choose a scene
outside the current scene's validated exits. Player-facing scene lists expose
only reachable destinations, never the whole module map.
"""

from __future__ import annotations

from typing import Any


def _resolve_scene_id(value: Any, world: dict) -> str:
    scenes = world.get("scenes", {})
    raw = str(value or "").strip()
    if raw in scenes and scenes.get(raw, {}).get("navigable", True):
        return raw
    return next(
        (
            sid
            for sid, scene in scenes.items()
            if isinstance(scene, dict)
            and scene.get("navigable", True)
            and str(scene.get("name", "")).strip() == raw
        ),
        "",
    )


def ensure_scene_state(session: dict, world: dict) -> None:
    """Migrate old sessions and maintain monotonic discovery/visit sets."""
    ps = session.setdefault("player_state", {})
    current = str(ps.get("current_scene", ""))
    scenes = world.get("scenes", {})

    discovered = session.setdefault("discovered_scene_ids", [])
    visited = session.setdefault("visited_scene_ids", [])
    session.setdefault("selected_scene_id", None)

    if current in scenes:
        if current not in discovered:
            discovered.append(current)
        if current not in visited:
            visited.append(current)


def scene_exit_records(current_scene: dict, world: dict) -> list[dict]:
    """Normalize legacy string exits and metadata-rich exit definitions."""
    exits = current_scene.get("exits", {})
    pairs = exits.items() if isinstance(exits, dict) else [("", item) for item in (exits or [])]
    records: list[dict] = []

    for keyword, raw in pairs:
        metadata = raw if isinstance(raw, dict) else {}
        target_value = (
            metadata.get("target")
            or metadata.get("scene_id")
            or metadata.get("to")
            if metadata
            else raw
        )
        target_id = _resolve_scene_id(target_value, world)
        if not target_id:
            continue
        records.append({
            "scene_id": target_id,
            "keyword": str(metadata.get("label") or keyword or "").strip(),
            "hidden": bool(metadata.get("hidden", False)),
            "requires_flag": str(metadata.get("requires_flag", "")).strip(),
        })
    return records


def _exit_available(record: dict, session: dict) -> bool:
    scene_id = record["scene_id"]
    unlocked = set(session.get("unlocked_scenes", []))
    discovered = set(session.get("discovered_scene_ids", []))
    required_flag = record.get("requires_flag", "")
    if required_flag and required_flag not in set(session.get("flags", [])):
        return False
    if record.get("hidden") and scene_id not in unlocked and scene_id not in discovered:
        return False
    return True


def list_available_scenes(session: dict, world: dict) -> list[dict]:
    """Return only destinations currently reachable from the active scene."""
    ensure_scene_state(session, world)
    scenes = world.get("scenes", {})
    current_id = str(session.get("player_state", {}).get("current_scene", ""))
    current = scenes.get(current_id, {})
    visited = set(session.get("visited_scene_ids", []))
    roster: list[dict] = []
    seen: set[str] = set()

    for record in scene_exit_records(current, world):
        target_id = record["scene_id"]
        if target_id in seen or not _exit_available(record, session):
            continue
        scene = scenes.get(target_id, {})
        if not isinstance(scene, dict):
            continue
        seen.add(target_id)
        roster.append({
            "id": target_id,
            "label": scene.get("name", target_id),
            "exit_label": record.get("keyword", ""),
            "visited": target_id in visited,
        })

    discovered = session.setdefault("discovered_scene_ids", [])
    for item in roster:
        if item["id"] not in discovered:
            discovered.append(item["id"])
    return roster


def reconcile_scene_target(session: dict, roster: list[dict]) -> bool:
    """Clear a selected destination that is no longer reachable."""
    selected = session.get("selected_scene_id")
    valid = {item.get("id") for item in roster}
    if selected and selected not in valid:
        session["selected_scene_id"] = None
        return True
    return False


def select_scene_target(session: dict, scene_id: str | None, world: dict) -> dict:
    """Set a destination after validating it against the current exit roster."""
    roster = list_available_scenes(session, world)
    if scene_id is None or not str(scene_id).strip():
        session["selected_scene_id"] = None
    else:
        target = str(scene_id).strip()
        if target not in {item["id"] for item in roster}:
            raise ValueError("Scene is not currently reachable")
        session["selected_scene_id"] = target
    return {
        "selected_scene_id": session.get("selected_scene_id"),
        "scenes": roster,
    }


def selected_movement_target(player_input: str, session: dict, world: dict,
                             has_move_intent: bool) -> str:
    """Resolve an explicit clicked destination for an actual movement request."""
    if not has_move_intent:
        return ""
    selected = session.get("selected_scene_id")
    roster = list_available_scenes(session, world)
    valid = {item["id"] for item in roster}
    return str(selected) if selected in valid else ""


def commit_scene_transition(session: dict, world: dict, scene_id: str) -> None:
    """Record arrival and clear the one-shot destination selection."""
    if scene_id not in world.get("scenes", {}):
        raise ValueError("Unknown scene transition target")
    session.setdefault("player_state", {})["current_scene"] = scene_id
    session["selected_scene_id"] = None
    ensure_scene_state(session, world)

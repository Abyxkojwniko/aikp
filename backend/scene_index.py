# -*- coding: utf-8 -*-
"""AIKP Scene Index — Pre-built O(1) lookup indices for runtime use.

Built once per world book load, stored in memory.
Avoids vector/RAG search for 95% of turns.
"""

from __future__ import annotations


def build_scene_index(world: dict) -> dict[str, list[str]]:
    """Build scene_id → list of entity_ids in that scene.
    O(1) lookup at runtime — no search needed.
    """
    index: dict[str, list[str]] = {}
    entities = world.get("entities", {})

    for eid, entity in entities.items():
        # An entity's presence is multi-scene: mobile NPCs (climbing companions,
        # traveling party) appear across many scenes. The parser captures this in
        # `all_scenes`; `scene` is just the first/home scene. Index into EVERY
        # scene the entity is actually present in, so reference resolution finds
        # 尾金 on the climbing path, not only in the hut. Falls back to the
        # singular `scene` when the parser didn't compute all_scenes.
        present_in = entity.get("all_scenes") or []
        if not isinstance(present_in, list):
            present_in = []
        home = entity.get("scene", "")
        if home and home not in present_in:
            present_in = [home, *present_in]

        # Supplement for MOBILE NPCs only (those the parser already judged
        # multi-scene): also count any scene whose source text names them. The
        # parser's all_scenes can miss a scene the module text clearly places
        # them in (e.g. 尾金 in 清晨出发). A fixed NPC with a single home scene is
        # never spread by a stray mention — this only widens already-mobile ones.
        is_mobile_npc = (entity.get("type") == "npc" and len(present_in) > 1)
        if is_mobile_npc:
            ename = entity.get("name", "")
            if ename:
                for sid, scene in world.get("scenes", {}).items():
                    if sid in present_in:
                        continue
                    # source_text is often empty (Pass 1.7 gap); the scene's
                    # desc/description holds the actual prose mentioning NPCs.
                    stext = (scene.get("source_text", "")
                             or scene.get("desc", "")
                             or scene.get("description", "") or "")
                    if ename in stext:
                        present_in.append(sid)

        for scene_id in present_in:
            if scene_id:
                lst = index.setdefault(scene_id, [])
                if eid not in lst:
                    lst.append(eid)

    # Also include any entities mentioned in scene items/clues/npcs (parser format compat)
    for sid, scene in world.get("scenes", {}).items():
        if sid not in index:
            index[sid] = []

        def _add_ids(source_list):
            for entry in source_list:
                eid = entry.get("id", entry) if isinstance(entry, dict) else entry
                if eid not in index[sid]:
                    index[sid].append(eid)

        _add_ids(scene.get("items", []))
        _add_ids(scene.get("clues", []))
        _add_ids(scene.get("npcs", []))

    return index


def build_entity_index(world: dict) -> dict[str, dict]:
    """Build entity_id → {type, scene, name} for O(1) metadata lookup."""
    index: dict[str, dict] = {}
    entities = world.get("entities", {})

    for eid, entity in entities.items():
        index[eid] = {
            "type": entity.get("type", "?"),
            "scene": entity.get("scene", ""),
            "name": entity.get("name", eid),
        }

    # Legacy compat: build entries from old npcs/items/clues if entities is empty
    if not index:
        for npc_id, npc in world.get("npcs", {}).items():
            index[npc_id] = {
                "type": "npc",
                "scene": npc.get("scene", ""),
                "name": npc.get("name", npc_id),
            }
        for item_id, item in world.get("items", {}).items():
            index[item_id] = {
                "type": "item",
                "scene": item.get("scene", ""),
                "name": item.get("name", item_id),
            }
        for clue_id, clue in world.get("clues", {}).items():
            index[clue_id] = {
                "type": "clue",
                "scene": clue.get("scene", ""),
                "name": clue.get("name", clue_id),
            }

    return index


def get_entities_in_scene(
    scene_id: str,
    scene_index: dict[str, list[str]],
) -> list[str]:
    """O(1) lookup: return entity IDs in a scene."""
    return scene_index.get(scene_id, [])


def get_entity_info(
    entity_id: str,
    entity_index: dict[str, dict],
) -> dict | None:
    """O(1) lookup: return entity metadata."""
    return entity_index.get(entity_id)

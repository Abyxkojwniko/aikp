"""Authoritative fact state and event reducer for world entities.

The legacy ``entity_states`` map remains supported, but it is no longer asked
to encode location, visibility, ownership, and condition in one string. Those
orthogonal facts live in ``session.entity_facts`` and are changed only through
events recorded in ``session.world_events``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


HIDDEN_STATES = frozenset({
    "hidden", "unknown", "undiscovered", "unrevealed", "concealed",
    "secret", "sealed_hidden", "not_found",
})
INVENTORY_STATES = frozenset({"in_inventory", "obtained", "carried", "held"})
REMOVED_STATES = frozenset({
    "removed", "consumed", "gone", "destroyed", "lost", "discarded",
})
NON_OBJECT_TYPES = frozenset({"npc"})


def _state(value: Any) -> str:
    return str(value or "default").strip().lower()


def _initial_location(entity: dict, legacy_state: str) -> dict:
    if legacy_state in INVENTORY_STATES:
        return {"kind": "inventory", "id": "player"}
    if legacy_state in REMOVED_STATES:
        return {"kind": "removed", "id": ""}
    container = str(entity.get("container", "")).strip()
    if container:
        return {"kind": "container", "id": container}
    return {"kind": "scene", "id": str(entity.get("scene", ""))}


def _initial_fact(eid: str, entity: dict, legacy_state: str,
                  opening: str) -> dict:
    hidden = legacy_state in HIDDEN_STATES
    name = str(entity.get("name", ""))
    entity_type = str(entity.get("type", "item"))
    portable_default = entity_type in {"item", "clue"}
    return {
        "entity_id": eid,
        "location": _initial_location(entity, legacy_state),
        "known": bool(name and name in opening) or not hidden,
        "visible": not hidden and legacy_state not in REMOVED_STATES,
        "exists": legacy_state not in REMOVED_STATES,
        "portable": bool(entity.get("portable", portable_default)),
        "condition": "intact",
        "open": legacy_state in {"open", "opened"},
        "locked": legacy_state in {"locked", "sealed", "sealed_hidden"},
        "legacy_state": legacy_state,
    }


def _sync_fact_from_external_legacy(fact: dict, entity: dict,
                                    legacy_state: str) -> None:
    """Honor old save files and tests that still mutate entity_states directly."""
    previous = _state(fact.get("legacy_state"))
    if previous == legacy_state:
        return
    fact["legacy_state"] = legacy_state
    if legacy_state in HIDDEN_STATES:
        fact["visible"] = False
    elif legacy_state in INVENTORY_STATES:
        fact.update({
            "location": {"kind": "inventory", "id": "player"},
            "known": True,
            "visible": True,
            "exists": True,
        })
    elif legacy_state in REMOVED_STATES:
        fact.update({
            "location": {"kind": "removed", "id": ""},
            "visible": False,
            "exists": False,
        })
    else:
        fact["exists"] = True
        if legacy_state in {"present", "visible", "available", "revealed",
                            "found", "read", "opened", "used"}:
            fact["known"] = True
            fact["visible"] = True
        if legacy_state in {"open", "opened"}:
            fact["open"] = True
        if legacy_state in {"closed"}:
            fact["open"] = False
        if legacy_state in {"locked", "sealed"}:
            fact["locked"] = True
        if legacy_state in {"unlocked"}:
            fact["locked"] = False
        # A direct legacy reset from inventory to present puts the object home.
        if (fact.get("location", {}).get("kind") == "inventory"
                and legacy_state in {"present", "visible", "available"}):
            fact["location"] = {
                "kind": "scene", "id": str(entity.get("scene", ""))}


def ensure_fact_state(session: dict, world: dict) -> dict[str, dict]:
    facts = session.setdefault("entity_facts", {})
    legacy_states = session.setdefault("entity_states", {})
    opening = str(world.get("opening", ""))
    inventory_ids = session.setdefault("inventory_entity_ids", [])

    for eid, entity in world.get("entities", {}).items():
        if not isinstance(entity, dict):
            continue
        legacy = _state(legacy_states.get(eid, entity.get("initial_state")))
        legacy_states.setdefault(eid, legacy)
        if eid not in facts or not isinstance(facts[eid], dict):
            facts[eid] = _initial_fact(eid, entity, legacy, opening)
        else:
            _sync_fact_from_external_legacy(facts[eid], entity, legacy)

        location = facts[eid].get("location", {})
        if location.get("kind") == "inventory":
            if eid not in inventory_ids:
                inventory_ids.append(eid)
        elif eid in inventory_ids:
            inventory_ids.remove(eid)

    _sync_inventory_names(session, world)
    return facts


def _sync_inventory_names(session: dict, world: dict) -> None:
    names = []
    entities = world.get("entities", {})
    for eid in session.get("inventory_entity_ids", []):
        entity = entities.get(eid, {})
        name = str(entity.get("name", eid))
        if name not in names:
            names.append(name)
    session.setdefault("player_state", {})["inventory"] = names


def fact_for(session: dict, world: dict, entity_id: str) -> dict:
    return ensure_fact_state(session, world).get(entity_id, {})


def entity_is_visible(entity_id: str, world: dict, session: dict) -> bool:
    fact = fact_for(session, world, entity_id)
    entity = world.get("entities", {}).get(entity_id, {})
    if not fact or not fact.get("exists", True) or not fact.get("visible", False):
        return False
    if entity.get("type") == "clue":
        return bool(
            fact.get("known")
            or entity_id in session.get("discovered_clues", [])
            or f"{entity_id}_discovered" in session.get("flags", [])
        )
    return True


def scene_entity_ids(session: dict, world: dict, scene_index: dict,
                     scene_id: str) -> list[str]:
    facts = ensure_fact_state(session, world)
    ordered = list(scene_index.get(scene_id, []))
    for eid, fact in facts.items():
        location = fact.get("location", {})
        if location.get("kind") == "scene" and location.get("id") == scene_id:
            if eid not in ordered:
                ordered.append(eid)

    result = []
    for eid in ordered:
        fact = facts.get(eid)
        if not fact:
            continue
        location = fact.get("location", {})
        if location.get("kind") == "scene" and location.get("id") == scene_id:
            result.append(eid)
    return result


def inventory_entity_ids(session: dict, world: dict) -> list[str]:
    ensure_fact_state(session, world)
    return list(session.get("inventory_entity_ids", []))


def _capabilities(entity: dict, fact: dict) -> list[str]:
    location_kind = fact.get("location", {}).get("kind")
    entity_type = entity.get("type", "item")
    caps = ["inspect"]
    if location_kind == "scene" and fact.get("portable"):
        caps.append("take")
    if location_kind == "inventory":
        caps.extend(["use", "drop"])
    if entity_type in {"door", "container"}:
        caps.append("close" if fact.get("open") else "open")
        if fact.get("locked"):
            caps.append("unlock")
    if entity_type in {"item", "clue", "document"}:
        caps.append("read")
    return list(dict.fromkeys(caps))


def list_interactable_objects(session: dict, world: dict, scene_index: dict,
                              entity_index: dict) -> list[dict]:
    scene_id = session.get("player_state", {}).get("current_scene", "")
    facts = ensure_fact_state(session, world)
    candidates = scene_entity_ids(session, world, scene_index, scene_id)
    candidates += [eid for eid in inventory_entity_ids(session, world)
                   if eid not in candidates]
    selected = session.get("selected_object_id")
    result = []
    for eid in candidates:
        entity = world.get("entities", {}).get(eid, {})
        if not isinstance(entity, dict) or entity.get("type") in NON_OBJECT_TYPES:
            continue
        if not entity_is_visible(eid, world, session):
            continue
        fact = facts[eid]
        result.append({
            "id": eid,
            "label": str(entity.get("name", eid)),
            "type": str(entity.get("type", "object")),
            "location": deepcopy(fact.get("location", {})),
            "state": str(session.get("entity_states", {}).get(eid, "default")),
            "capabilities": _capabilities(entity, fact),
            "selected": eid == selected,
        })
    return result


def reconcile_object_target(session: dict, roster: list[dict]) -> bool:
    selected = session.get("selected_object_id")
    if not selected or selected in {item["id"] for item in roster}:
        return False
    session["selected_object_id"] = None
    focus = session.setdefault("conversation_focus", {})
    if focus.get("object") == selected:
        focus.pop("object", None)
    return True


def select_object_target(session: dict, object_id: str | None, world: dict,
                         scene_index: dict, entity_index: dict) -> dict:
    if not object_id:
        session["selected_object_id"] = None
        session.setdefault("conversation_focus", {}).pop("object", None)
        return {"selected_object_id": None}
    roster = list_interactable_objects(
        session, world, scene_index, entity_index)
    if object_id not in {item["id"] for item in roster}:
        raise ValueError("Object is not visible or carried by the player")
    session["selected_object_id"] = object_id
    session.setdefault("conversation_focus", {})["object"] = object_id
    return {"selected_object_id": object_id}


def _set_legacy_state(session: dict, entity_id: str, state: str) -> None:
    session.setdefault("entity_states", {})[entity_id] = state
    fact = session.setdefault("entity_facts", {}).setdefault(entity_id, {})
    fact["legacy_state"] = state


def apply_world_event(session: dict, world: dict, event: dict) -> dict:
    """Apply one validated event to facts and legacy compatibility fields."""
    ensure_fact_state(session, world)
    eid = str(event.get("entity_id", ""))
    if not eid or eid not in world.get("entities", {}):
        raise ValueError("world event references an unknown entity")
    fact = session["entity_facts"][eid]
    event_type = str(event.get("type", ""))
    scene_id = str(event.get("scene_id") or
                   session.get("player_state", {}).get("current_scene", ""))

    if event_type == "entity_discovered":
        fact.update({"known": True, "visible": True, "exists": True})
        _set_legacy_state(session, eid, str(event.get("state", "revealed")))
        if world.get("entities", {}).get(eid, {}).get("type") == "clue":
            discovered = session.setdefault("discovered_clues", [])
            if eid not in discovered:
                discovered.append(eid)
            flag = f"{eid}_discovered"
            flags = session.setdefault("flags", [])
            if flag not in flags:
                flags.append(flag)
    elif event_type == "npc_name_disclosed":
        entity = world.get("entities", {}).get(eid, {})
        if entity.get("type") != "npc":
            raise ValueError("npc_name_disclosed requires an NPC entity")
        fact["known"] = True
        name = str(entity.get("name", "")).strip()
        npc_state = session.setdefault("npc_states", {}).get(name)
        if isinstance(npc_state, dict):
            npc_state.setdefault("dynamic", {}).setdefault(
                "disclosure", {})["name"] = True
    elif event_type == "item_picked_up":
        fact.update({
            "location": {"kind": "inventory", "id": "player"},
            "known": True, "visible": True, "exists": True,
        })
        _set_legacy_state(session, eid, "in_inventory")
    elif event_type == "item_dropped":
        fact.update({
            "location": {"kind": "scene", "id": scene_id},
            "known": True, "visible": True, "exists": True,
        })
        _set_legacy_state(session, eid, "present")
    elif event_type == "item_transferred":
        owner_id = str(event.get("owner_id", ""))
        owner = world.get("entities", {}).get(owner_id, {})
        if not owner_id or not isinstance(owner, dict) or owner.get("type") != "npc":
            raise ValueError("item transfer requires a known NPC owner")
        fact.update({
            "location": {"kind": "entity", "id": owner_id},
            "known": True, "visible": True, "exists": True,
        })
        _set_legacy_state(session, eid, "transferred")
    elif event_type == "item_used":
        fact["known"] = True
        if event.get("consumed"):
            fact.update({
                "location": {"kind": "removed", "id": ""},
                "visible": False, "exists": False, "condition": "consumed",
            })
            _set_legacy_state(session, eid, "consumed")
        else:
            fact["condition"] = str(event.get("condition", "used"))
            _set_legacy_state(session, eid, "used")
    elif event_type == "entity_moved":
        location = event.get("location", {})
        if not isinstance(location, dict):
            raise ValueError("entity movement requires a structured location")
        location_kind = str(location.get("kind", ""))
        location_id = str(location.get("id", ""))
        if location_kind == "scene":
            if location_id not in world.get("scenes", {}):
                raise ValueError("entity movement references an unknown scene")
        elif location_kind == "entity":
            if location_id not in world.get("entities", {}):
                raise ValueError("entity movement references an unknown owner")
        else:
            raise ValueError("entity movement requires a scene or entity location")
        fact["location"] = {"kind": location_kind, "id": location_id}
    elif event_type == "entity_removed":
        fact.update({
            "location": {"kind": "removed", "id": ""},
            "visible": False, "exists": False,
            "condition": str(event.get("condition", "removed")),
        })
        _set_legacy_state(session, eid, str(event.get("state", "removed")))
    elif event_type == "object_opened":
        fact.update({"open": True, "known": True})
        _set_legacy_state(session, eid, "opened")
    elif event_type == "object_closed":
        fact["open"] = False
        _set_legacy_state(session, eid, "closed")
    elif event_type == "object_unlocked":
        fact.update({"locked": False, "known": True})
        _set_legacy_state(session, eid, "unlocked")
    elif event_type == "object_locked":
        fact["locked"] = True
        _set_legacy_state(session, eid, "locked")
    elif event_type == "entity_damaged":
        fact["condition"] = str(event.get("condition", "damaged"))
    else:
        raise ValueError(f"unsupported world event type: {event_type}")

    inventory_ids = session.setdefault("inventory_entity_ids", [])
    if fact.get("location", {}).get("kind") == "inventory":
        if eid not in inventory_ids:
            inventory_ids.append(eid)
    elif eid in inventory_ids:
        inventory_ids.remove(eid)
    _sync_inventory_names(session, world)
    return event


def append_world_event(session: dict, world: dict, event: dict) -> dict:
    payload = deepcopy(event)
    seq = int(session.get("world_event_seq", 0)) + 1
    session["world_event_seq"] = seq
    payload.setdefault("event_id", f"evt-{seq:06d}")
    payload.setdefault("turn", int(session.get("current_turn", 0)) + 1)
    apply_world_event(session, world, payload)
    session.setdefault("world_events", []).append(payload)
    return payload


def sync_legacy_transition(session: dict, world: dict, entity_id: str,
                           old_state: str, new_state: str) -> list[dict]:
    """Translate a legacy state-machine transition into fact events."""
    old = _state(old_state)
    new = _state(new_state)
    if old == new:
        return []
    if new in INVENTORY_STATES:
        event = {"type": "item_picked_up", "entity_id": entity_id,
                 "source": "legacy_state_machine"}
    elif new in {"used"}:
        event = {"type": "item_used", "entity_id": entity_id,
                 "source": "legacy_state_machine"}
    elif new in REMOVED_STATES:
        event = {"type": "entity_removed", "entity_id": entity_id,
                 "state": new, "source": "legacy_state_machine"}
    elif new == "opened":
        event_types = (["object_unlocked", "object_opened"]
                       if old == "locked" else ["object_opened"])
        committed_events = [
            append_world_event(session, world, {
                "type": event_type, "entity_id": entity_id,
                "source": "legacy_state_machine",
            })
            for event_type in event_types
        ]
        session.setdefault("entity_states", {})[entity_id] = new_state
        session.setdefault("entity_facts", {}).setdefault(
            entity_id, {})["legacy_state"] = new_state
        return committed_events
    elif new in {"found", "read", "revealed", "visible"}:
        event = {"type": "entity_discovered", "entity_id": entity_id,
                 "state": new, "source": "legacy_state_machine"}
    else:
        ensure_fact_state(session, world)
        fact = session["entity_facts"].get(entity_id, {})
        fact["legacy_state"] = new
        return []
    committed = append_world_event(session, world, event)
    # Preserve the module's exact legacy state name for its next transition;
    # the generic event reducer may otherwise normalize "obtained" to
    # "in_inventory" and make a following legacy state unreachable.
    session.setdefault("entity_states", {})[entity_id] = new_state
    session.setdefault("entity_facts", {}).setdefault(
        entity_id, {})["legacy_state"] = new_state
    return [committed]

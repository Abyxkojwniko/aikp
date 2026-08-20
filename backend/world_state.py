"""Authoritative fact state and event reducer for world entities.

The legacy ``entity_states`` map remains supported, but it is no longer asked
to encode location, visibility, ownership, and condition in one string. Those
orthogonal facts live in ``session.entity_facts`` and are changed only through
events recorded in ``session.world_events``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


HIDDEN_STATES = frozenset({
    "hidden", "unknown", "undiscovered", "unrevealed", "concealed",
    "secret", "sealed_hidden", "not_found",
})
INVENTORY_STATES = frozenset({"in_inventory", "obtained", "carried", "held"})
REMOVED_STATES = frozenset({
    "removed", "consumed", "gone", "destroyed", "lost", "discarded",
})
OFFSTAGE_STATES = frozenset({
    "absent", "departed", "fled", "escaped", "vanished", "missing", "offstage",
    "离场", "离开", "已离开", "逃离", "已逃离", "消失", "已消失", "失踪",
})
INCAPACITATED_STATES = frozenset({
    "dead", "defeated", "unconscious", "incapacitated", "restrained",
    "死亡", "已死亡", "被击败", "昏迷", "失去意识", "失能", "被制服",
})
ACTIVE_STATES = frozenset({
    "default", "present", "visible", "available", "active", "alive",
    "restored", "revived", "resurrected", "在场", "存活", "复活", "已复活",
})
NON_OBJECT_TYPES = frozenset({"npc"})

_CANONICAL_STATE_KEYS = (
    "player_state",
    "entity_states",
    "entity_states_cooldown",
    "flags",
    "npc_dispositions",
    "npc_states",
    "discovered_clues",
    "current_beat_id",
    "completed_beats",
    "unlocked_scenes",
    "companions",
    "clocks",
    "entity_facts",
    "inventory_entity_ids",
    "discovered_scene_ids",
    "visited_scene_ids",
    "current_scenario_id",
)


def _state(value: Any) -> str:
    return str(value or "default").strip().lower()


def _initial_location(entity: dict, legacy_state: str) -> dict:
    if legacy_state in INVENTORY_STATES:
        return {"kind": "inventory", "id": "player"}
    if legacy_state in REMOVED_STATES:
        return {"kind": "removed", "id": ""}
    if legacy_state in OFFSTAGE_STATES:
        return {"kind": "offstage", "id": ""}
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
        "visible": not hidden and legacy_state not in (
            REMOVED_STATES | OFFSTAGE_STATES),
        "exists": legacy_state not in REMOVED_STATES,
        "portable": bool(entity.get("portable", portable_default)),
        "condition": (
            legacy_state if legacy_state in (
                INCAPACITATED_STATES | OFFSTAGE_STATES) else "intact"
        ),
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
    elif legacy_state in OFFSTAGE_STATES:
        fact.update({
            "location": {"kind": "offstage", "id": ""},
            "visible": False,
            "exists": True,
            "condition": legacy_state,
        })
    else:
        fact["exists"] = True
        if legacy_state in INCAPACITATED_STATES:
            fact["condition"] = legacy_state
        elif previous in (INCAPACITATED_STATES | OFFSTAGE_STATES):
            fact["condition"] = "intact"
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
        elif (fact.get("location", {}).get("kind") == "offstage"
              and legacy_state in {"present", "visible", "available", "revealed"}):
            fact.update({
                "location": {"kind": "scene", "id": str(entity.get("scene", ""))},
                "visible": True,
            })


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


def canonical_state_hash(session: dict) -> str:
    """Return a stable hash of gameplay facts, excluding logs and UI focus."""
    snapshot = {
        key: session.get(key)
        for key in _CANONICAL_STATE_KEYS
    }
    encoded = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_world_invariants(session: dict, world: dict) -> list[str]:
    """Check cross-field facts after a proposed transaction.

    Event schema checks happen in ``apply_world_event``. These checks cover
    constraints that only become visible after multiple deltas are composed.
    """
    facts = ensure_fact_state(session, world)
    entities = world.get("entities", {})
    scenes = world.get("scenes", {})
    issues: list[str] = []

    current_scene = str(
        session.get("player_state", {}).get("current_scene", ""))
    if current_scene and current_scene not in scenes:
        issues.append(f"player references unknown scene: {current_scene}")

    inventory = list(session.get("inventory_entity_ids", []))
    if len(inventory) != len(set(inventory)):
        issues.append("inventory contains duplicate entity ids")

    inventory_from_facts: set[str] = set()
    for eid, fact in facts.items():
        if eid not in entities:
            issues.append(f"fact references unknown entity: {eid}")
            continue
        location = fact.get("location", {})
        if not isinstance(location, dict):
            issues.append(f"{eid} has an invalid location record")
            continue
        kind = str(location.get("kind", ""))
        location_id = str(location.get("id", ""))
        exists = bool(fact.get("exists", True))
        visible = bool(fact.get("visible", False))

        if kind == "scene" and location_id not in scenes:
            issues.append(f"{eid} references unknown scene: {location_id}")
        elif kind in {"entity", "container"} and location_id not in entities:
            issues.append(f"{eid} references unknown entity location: {location_id}")
        elif kind == "inventory":
            if location_id != "player":
                issues.append(f"{eid} has a non-player inventory owner")
            inventory_from_facts.add(eid)
        elif kind == "removed":
            if exists or visible:
                issues.append(f"removed entity {eid} still exists or is visible")
        elif kind == "offstage":
            if not exists or visible:
                issues.append(f"offstage entity {eid} must exist and be invisible")
        elif kind not in {
                "scene", "entity", "container", "inventory", "removed", "offstage"}:
            issues.append(f"{eid} has unsupported location kind: {kind or '<empty>'}")

        if not exists and (visible or kind != "removed"):
            issues.append(f"nonexistent entity {eid} remains visible or located")
        if kind == "inventory" and (not exists or not visible):
            issues.append(f"inventory entity {eid} is not present and visible")

    if set(inventory) != inventory_from_facts:
        issues.append("inventory index disagrees with entity locations")
    return issues


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


def _authored_pickup_transition_is_valid(session: dict, entity: dict,
                                         event: dict) -> bool:
    """Allow conditionally portable objects only through their authored edge."""
    from_state = _state(event.get("from_state"))
    to_state = _state(event.get("state"))
    if (not from_state or to_state not in INVENTORY_STATES
            or _state(session.get("entity_states", {}).get(
                event.get("entity_id"))) != from_state):
        return False
    state_def = entity.get("states", {}).get(from_state, {})
    if not isinstance(state_def, dict):
        return False
    outcomes = [
        state_def.get("on_trigger"), state_def.get("on_pass"),
        state_def.get("on_fail"),
    ]
    return any(
        isinstance(outcome, dict)
        and _state(outcome.get("to_state")) == to_state
        for outcome in outcomes
    )


def apply_world_event(session: dict, world: dict, event: dict) -> dict:
    """Apply one validated event to facts and legacy compatibility fields."""
    ensure_fact_state(session, world)
    event_type = str(event.get("type", ""))
    if event_type == "scene_entered":
        scene_id = str(event.get("scene_id", ""))
        if not scene_id or scene_id not in world.get("scenes", {}):
            raise ValueError("scene transition references an unknown scene")
        session.setdefault("player_state", {})["current_scene"] = scene_id
        session["selected_scene_id"] = None
        for field in ("discovered_scene_ids", "visited_scene_ids"):
            scene_ids = session.setdefault(field, [])
            if scene_id not in scene_ids:
                scene_ids.append(scene_id)
        applied = session.setdefault("applied_scene_entry_events", [])
        for event_key in event.get("entry_event_keys", []) or []:
            key = str(event_key)
            if key and key not in applied:
                applied.append(key)
        return event

    eid = str(event.get("entity_id", ""))
    if not eid or eid not in world.get("entities", {}):
        raise ValueError("world event references an unknown entity")
    fact = session["entity_facts"][eid]
    entity = world.get("entities", {}).get(eid, {})
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
        conditionally_portable = _authored_pickup_transition_is_valid(
            session, entity, event)
        if (entity.get("type") == "npc"
                or (not fact.get("portable", False)
                    and not conditionally_portable)):
            raise ValueError("item pickup requires a portable non-NPC entity")
        if not fact.get("exists", True) or not fact.get("visible", False):
            raise ValueError("item pickup requires an existing visible entity")
        fact.update({
            "location": {"kind": "inventory", "id": "player"},
            "known": True, "visible": True, "exists": True,
        })
        _set_legacy_state(
            session, eid, str(event.get("state", "in_inventory")))
    elif event_type == "item_dropped":
        if fact.get("location", {}).get("kind") != "inventory":
            raise ValueError("item drop requires player inventory ownership")
        fact.update({
            "location": {"kind": "scene", "id": scene_id},
            "known": True, "visible": True, "exists": True,
        })
        _set_legacy_state(session, eid, str(event.get("state", "present")))
    elif event_type == "item_transferred":
        if fact.get("location", {}).get("kind") != "inventory":
            raise ValueError("item transfer requires player inventory ownership")
        owner_id = str(event.get("owner_id", ""))
        owner = world.get("entities", {}).get(owner_id, {})
        if not owner_id or not isinstance(owner, dict) or owner.get("type") != "npc":
            raise ValueError("item transfer requires a known NPC owner")
        owner_fact = session.get("entity_facts", {}).get(owner_id, {})
        owner_location = owner_fact.get("location", {})
        owner_present = (
            owner_fact.get("exists", True)
            and owner_fact.get("visible", False)
            and (
                (owner_location.get("kind") == "scene"
                 and str(owner_location.get("id", "")) == scene_id)
                or owner_id in set(session.get("companions", []))
            )
        )
        from reference_resolver import npc_is_interactable
        if not owner_present or not npc_is_interactable(owner_id, world, session):
            raise ValueError("item transfer requires a present active NPC owner")
        fact.update({
            "location": {"kind": "entity", "id": owner_id},
            "known": True, "visible": True, "exists": True,
        })
        _set_legacy_state(
            session, eid, str(event.get("state", "transferred")))
    elif event_type == "item_used":
        if fact.get("location", {}).get("kind") != "inventory":
            raise ValueError("item use requires player inventory ownership")
        fact["known"] = True
        if event.get("consumed"):
            fact.update({
                "location": {"kind": "removed", "id": ""},
                "visible": False, "exists": False, "condition": "consumed",
            })
            _set_legacy_state(
                session, eid, str(event.get("state", "consumed")))
        else:
            fact["condition"] = str(event.get("condition", "used"))
            _set_legacy_state(
                session, eid, str(event.get("state", "used")))
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
        visible = event.get("visible")
        if visible is None:
            visible = (
                entity.get("type") == "npc"
                or bool(fact.get("visible", False))
            )
        fact.update({
            "location": {"kind": location_kind, "id": location_id},
            "exists": True,
            "visible": bool(visible),
        })
    elif event_type == "entity_departed":
        fact.update({
            "location": {"kind": "offstage", "id": ""},
            "visible": False,
            "exists": True,
            "condition": str(event.get("condition", "offstage")),
        })
        _set_legacy_state(session, eid, str(event.get("state", "offstage")))
    elif event_type == "entity_restored":
        restored_scene = str(event.get("scene_id") or entity.get("scene", ""))
        if restored_scene not in world.get("scenes", {}):
            raise ValueError("entity restoration references an unknown scene")
        fact.update({
            "location": {"kind": "scene", "id": restored_scene},
            "visible": True,
            "exists": True,
            "condition": "intact",
        })
        _set_legacy_state(session, eid, str(event.get("state", "present")))
    elif event_type == "entity_removed":
        fact.update({
            "location": {"kind": "removed", "id": ""},
            "visible": False, "exists": False,
            "condition": str(event.get("condition", "removed")),
        })
        _set_legacy_state(session, eid, str(event.get("state", "removed")))
    elif event_type == "object_opened":
        fact.update({"open": True, "known": True})
        _set_legacy_state(session, eid, str(event.get("state", "opened")))
    elif event_type == "object_closed":
        fact["open"] = False
        _set_legacy_state(session, eid, str(event.get("state", "closed")))
    elif event_type == "object_unlocked":
        fact.update({"locked": False, "known": True})
        _set_legacy_state(session, eid, str(event.get("state", "unlocked")))
    elif event_type == "object_locked":
        fact["locked"] = True
        _set_legacy_state(session, eid, str(event.get("state", "locked")))
    elif event_type == "entity_damaged":
        fact["condition"] = str(event.get("condition", "damaged"))
        if event.get("state"):
            _set_legacy_state(session, eid, str(event["state"]))
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


def commit_world_events(session: dict, world: dict, events: list[dict],
                        *, actor: str = "world", source: str = "") -> list[dict]:
    """Validate and atomically commit a batch of world events.

    All deltas are first applied to an isolated copy. A bad later delta cannot
    leave earlier mutations, event ids, or inventory indexes behind. Successful
    commits form a hash-linked journal suitable for replay and debugging.
    """
    if not events:
        return []
    if not all(isinstance(event, dict) for event in events):
        raise ValueError("world transaction events must be objects")

    staged = deepcopy(session)
    ensure_fact_state(staged, world)
    before_hash = canonical_state_hash(staged)
    seq = int(staged.get("world_event_seq", 0))
    turn = int(staged.get("current_turn", 0)) + 1
    committed: list[dict] = []

    for raw_event in events:
        seq += 1
        payload = deepcopy(raw_event)
        payload.setdefault("event_id", f"evt-{seq:06d}")
        payload.setdefault("turn", turn)
        if source:
            payload.setdefault("source", source)
        apply_world_event(staged, world, payload)
        committed.append(payload)

    issues = validate_world_invariants(staged, world)
    if issues:
        raise ValueError("world transaction violates invariants: " + "; ".join(issues))

    staged["world_event_seq"] = seq
    after_hash = canonical_state_hash(staged)
    digest_input = json.dumps({
        "before": before_hash,
        "after": after_hash,
        "events": committed,
        "actor": actor,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    commit_id = "tx-" + hashlib.sha256(
        digest_input.encode("utf-8")).hexdigest()[:20]
    for payload in committed:
        payload["commit_id"] = commit_id
    staged.setdefault("world_events", []).extend(committed)
    staged.setdefault("world_commits", []).append({
        "commit_id": commit_id,
        "turn": turn,
        "actor": actor,
        "source": source,
        "event_ids": [payload["event_id"] for payload in committed],
        "before_hash": before_hash,
        "after_hash": after_hash,
    })
    staged["world_state_hash"] = after_hash

    session.clear()
    session.update(staged)
    return committed


def append_world_event(session: dict, world: dict, event: dict) -> dict:
    """Backward-compatible single-event atomic transaction."""
    return commit_world_events(
        session, world, [event],
        actor=str(event.get("actor", "world")),
        source=str(event.get("source", "")),
    )[0]


def sync_legacy_transition(session: dict, world: dict, entity_id: str,
                           old_state: str, new_state: str) -> list[dict]:
    """Translate a legacy state-machine transition into fact events."""
    old = _state(old_state)
    new = _state(new_state)
    if old == new:
        return []
    if new in INVENTORY_STATES:
        event = {"type": "item_picked_up", "entity_id": entity_id,
                 "state": new_state, "from_state": old_state,
                 "source": "legacy_state_machine"}
    elif new in {"used"}:
        event = {"type": "item_used", "entity_id": entity_id,
                 "state": new_state, "source": "legacy_state_machine"}
    elif new in REMOVED_STATES:
        event = {"type": "entity_removed", "entity_id": entity_id,
                 "state": new, "source": "legacy_state_machine"}
    elif new in OFFSTAGE_STATES:
        event = {"type": "entity_departed", "entity_id": entity_id,
                 "state": new, "condition": new,
                 "source": "legacy_state_machine"}
    elif new in INCAPACITATED_STATES:
        event = {"type": "entity_damaged", "entity_id": entity_id,
                 "state": new, "condition": new,
                 "source": "legacy_state_machine"}
    elif (old in (INCAPACITATED_STATES | OFFSTAGE_STATES)
          and new in ACTIVE_STATES):
        event = {"type": "entity_restored", "entity_id": entity_id,
                 "state": new_state, "source": "legacy_state_machine"}
    elif new == "opened":
        raw_events = ([
            {"type": "object_unlocked", "entity_id": entity_id,
             "source": "legacy_state_machine"},
            {"type": "object_opened", "entity_id": entity_id,
             "state": new_state, "source": "legacy_state_machine"},
        ] if old == "locked" else [{
            "type": "object_opened", "entity_id": entity_id,
            "state": new_state, "source": "legacy_state_machine",
        }])
        return commit_world_events(
            session, world, raw_events, actor="rules",
            source="legacy_state_machine")
    elif new in {"found", "read", "revealed", "visible"}:
        event = {"type": "entity_discovered", "entity_id": entity_id,
                 "state": new, "source": "legacy_state_machine"}
    else:
        ensure_fact_state(session, world)
        session.setdefault("entity_states", {})[entity_id] = new_state
        fact = session["entity_facts"].get(entity_id, {})
        fact["legacy_state"] = new_state
        session["world_state_hash"] = canonical_state_hash(session)
        return []
    committed = append_world_event(session, world, event)
    return [committed]

"""Turn-level narrative contracts and consistency auditing.

The narrator is an observation generator, not a state writer. This module
projects canonical world state into explicit commitments and checks prose
against those commitments before it reaches the player or long-term memory.
"""

from __future__ import annotations

import json
import re
from typing import Any

from reference_resolver import npc_is_interactable
from world_state import ensure_fact_state


AUDIT_KINDS = frozenset({
    "fact_conflict",
    "commitment_conflict",
    "player_input_conflict",
    "hidden_information_leak",
    "unsupported_entity",
    "unsupported_location",
    "unsupported_mutation",
})

_NPC_ACTION_RE = re.compile(
    r"(?:说(?:道|着)?|回答|答道|问道|喊道|叫道|点头|摇头|走来|走近|"
    r"站起|递给|接过|攻击|扑向|开口)|"
    r"\b(?:say|says|said|speak|speaks|reply|replies|ask|asks|nod|nods|"
    r"walk|walks|stand|stands|hand|hands|take|takes|attack|attacks)\b",
    re.IGNORECASE,
)
_PRESENCE_RE = re.compile(
    r"(?:就在|仍在|还在|站在|坐在|躺在|走来|出现|映入眼帘)|"
    r"\b(?:is|are)\s+(?:here|present)|\b(?:stands|sits|lies|appears)\b",
    re.IGNORECASE,
)
_OPEN_RE = re.compile(
    r"(?:敞开|打开着|已经打开|应声打开)|\b(?:is|stands|swings)\s+open\b",
    re.IGNORECASE,
)
_LOCKED_RE = re.compile(
    r"(?:仍然?锁着|已经锁上|处于锁定)|\b(?:is|remains)\s+locked\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"(?:不|没|没有|并未|无法|不能|尸体|遗体|曾经|过去)|"
    r"\b(?:not|never|cannot|can't|couldn't|corpse|body|formerly|used to)\b",
    re.IGNORECASE,
)
_CARRIED_RE = re.compile(
    r"(?:手中|手里|身上|背包|口袋|行囊|携带|拿着|握着|收好)|"
    r"\b(?:hand|hands|backpack|pack|pocket|inventory|carried|carrying|"
    r"holding|held|pocketed)\b",
    re.IGNORECASE,
)


def _location(fact: dict) -> dict[str, str]:
    raw = fact.get("location", {})
    if not isinstance(raw, dict):
        return {"kind": "", "id": ""}
    return {"kind": str(raw.get("kind", "")), "id": str(raw.get("id", ""))}


def _source_evidence(state: dict, relevant_ids: set[str]) -> str:
    world = state["world"]
    current = state.get("current_scene", {})
    parts = [
        str(current.get("desc", "") or current.get("description", "")),
        str(current.get("atmosphere", "")),
    ]
    for eid in sorted(relevant_ids):
        entity = world.get("entities", {}).get(eid, {})
        if not isinstance(entity, dict):
            continue
        parts.extend(str(entity.get(key, "")) for key in (
            "description", "appearance", "personality", "public_label"))
    return "\n".join(part for part in parts if part)[:8000]


def _active_story_commitments(session: dict, world: dict,
                              target_scene: str) -> list[dict]:
    """Keep source-backed setups alive after their source node leaves context."""
    commitments = [
        dict(row)
        for row in session.get("narrative_commitments", [])
        if isinstance(row, dict)
        and str(row.get("status", "pending")) not in {"fulfilled", "cancelled"}
    ]
    details = [
        row for row in world.get("detailed_story_nodes", [])
        if isinstance(row, dict) and row.get("node_id")
    ]
    if not details:
        return commitments

    selected_scenario = str(session.get("current_scenario_id", ""))
    if selected_scenario:
        details = [
            row for row in details
            if not row.get("scenario_id")
            or str(row.get("scenario_id")) == selected_scenario
        ]
    node_scenes = {
        str(detail["node_id"]): {
            str(scene.get("id", ""))
            for scene in detail.get("scenes", [])
            if isinstance(scene, dict) and scene.get("id")
        }
        for detail in details
    }
    visited = set(session.get("visited_scene_ids", []))
    completed_nodes = set(session.get("completed_beats", []))
    active_nodes = set(completed_nodes)
    current_node = str(session.get("current_beat_id", ""))
    if current_node:
        active_nodes.add(current_node)
    active_nodes.update(
        node_id for node_id, scene_ids in node_scenes.items()
        if scene_ids & visited
    )
    due_nodes = {
        node_id for node_id, scene_ids in node_scenes.items()
        if target_scene in scene_ids
    }
    if current_node:
        due_nodes.add(current_node)

    seen_ids = {str(row.get("id", "")) for row in commitments}
    for detail in details:
        node_id = str(detail["node_id"])
        for index, promise in enumerate(detail.get("promises_payoffs", [])):
            if not isinstance(promise, dict):
                continue
            setup = str(promise.get("setup", "")).strip()
            payoff = str(promise.get("payoff", "")).strip()
            relation = str(promise.get("relation", "setup")).strip().lower()
            linked_node = str(promise.get("linked_node_id", "")).strip()
            if relation == "payoff" or not (setup or payoff) or not linked_node:
                continue
            if linked_node in completed_nodes:
                continue
            opened = node_id in active_nodes
            due = linked_node in due_nodes
            if not opened and not due:
                continue
            commitment_id = f"story:{node_id}:{index:03d}"
            if commitment_id in seen_ids:
                continue
            commitments.append({
                "id": commitment_id,
                "status": "due" if due else "active",
                "setup": setup,
                "payoff": payoff,
                "opened_at": node_id,
                "due_at": linked_node,
                "source_ref": promise.get("source_ref"),
            })
            seen_ids.add(commitment_id)
    return commitments


def build_narrative_contract(state: dict) -> dict:
    """Project the smallest relevant canonical state for one narration."""
    session = state["session"]
    world = state["world"]
    facts = ensure_fact_state(session, world)
    current_scene = str(
        session.get("player_state", {}).get("current_scene", ""))
    target_scene = str(state.get("movement_target") or current_scene)

    relevant_ids = set(state.get("scene_entities", []))
    relevant_ids.update(session.get("inventory_entity_ids", []))
    relevant_ids.update(session.get("companions", []))
    relevant_ids.update(event.get("entity_id", "")
                        for event in state.get("_action_events", []))
    relevant_ids.update(value for value in (
        session.get("selected_npc_id"), session.get("selected_object_id"))
                        if value)
    relevant_ids.update(
        str(row.get("entity_id", ""))
        for row in session.get("entity_mentions", [])[-20:]
        if isinstance(row, dict) and row.get("entity_id"))

    # Initial scene prose is allowed evidence, but it may contain objects that
    # have since moved. Keep those entities in the contract so dynamic facts win.
    for eid, entity in world.get("entities", {}).items():
        if not isinstance(entity, dict):
            continue
        home = str(entity.get("home_scene", entity.get("scene", "")))
        if home in {current_scene, target_scene}:
            relevant_ids.add(eid)

    entities = []
    for eid in sorted(relevant_ids):
        entity = world.get("entities", {}).get(eid)
        fact = facts.get(eid)
        if not isinstance(entity, dict) or not isinstance(fact, dict):
            continue
        location = _location(fact)
        exists = bool(fact.get("exists", True))
        carried = bool(
            exists and location["kind"] == "inventory"
        )
        present = bool(
            exists and (
                (location["kind"] == "scene" and location["id"] == target_scene)
                or eid in set(session.get("companions", []))
            )
        )
        entity_type = str(entity.get("type", "object"))
        can_act = bool(
            entity_type == "npc" and present
            and npc_is_interactable(eid, world, session)
        )
        entities.append({
            "id": eid,
            "name": str(entity.get("name", eid)),
            "type": entity_type,
            "state": str(session.get("entity_states", {}).get(
                eid, entity.get("initial_state", "default"))),
            "location": location,
            "exists": exists,
            "known": bool(fact.get("known", False)),
            "visible": bool(fact.get("visible", False)),
            "present": present,
            "carried": carried,
            "condition": str(fact.get("condition", "intact")),
            "open": bool(fact.get("open", False)),
            "locked": bool(fact.get("locked", False)),
            "can_act": can_act,
        })

    node = {}
    for detail in world.get("detailed_story_nodes", []):
        if not isinstance(detail, dict):
            continue
        if any(isinstance(scene, dict) and str(scene.get("id", "")) == target_scene
               for scene in detail.get("scenes", [])):
            node = {
                "node_id": str(detail.get("node_id", "")),
                "state_transitions": detail.get("state_transitions", []),
                "knowledge_changes": detail.get("knowledge_changes", []),
                "promises_payoffs": detail.get("promises_payoffs", []),
            }
            break

    return {
        "turn": int(state.get(
            "_contract_turn", int(session.get("current_turn", 0)) + 1)),
        "player_scene": target_scene,
        "movement_is_committed": bool(state.get("movement_target")),
        "applied_events": [
            {key: event.get(key) for key in (
                "type", "entity_id", "owner_id", "scene_id", "condition")
             if event.get(key) not in (None, "")}
            for event in state.get("_action_events", [])
        ],
        "entities": entities,
        "story_commitments": _active_story_commitments(
            session, world, target_scene),
        "current_story_node": node,
        "source_evidence": _source_evidence(state, relevant_ids),
    }


def render_contract_block(contract: dict) -> str:
    compact = dict(contract)
    compact.pop("source_evidence", None)
    return (
        "=== NARRATIVE COMMITMENTS (CANONICAL, END OF THIS TURN) ===\n"
        "Treat these as binding facts. Prose is an observation of this state, "
        "not permission to change it. Only APPLIED_EVENTS may establish a state "
        "change. A non-present or inactive NPC cannot act or speak. Pending story "
        "commitments must remain satisfiable; do not skip their prerequisites.\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    )


def build_audit_prompt(contract: dict, player_input: str, narrative: str) -> str:
    return (
        "Audit one RPG narration against explicit canon. A contradiction exists "
        "only when canon explicitly establishes a fact, the narration explicitly "
        "establishes a conflicting claim about the same entity and time, and no "
        "listed applied event reconciles them. Also flag invented entities, places, "
        "clues, success, or knowledge absent from source evidence; a dead, absent, "
        "or inactive NPC acting; skipped story prerequisites; and failure to honor "
        "the player's declared action as at least an attempt. Do not flag style, "
        "minor sensory wording, uncertainty, or a failed attempt. Use the smallest "
        "supported violation set. Return JSON only: "
        '{"valid":true,"violations":[]} or '
        '{"valid":false,"violations":[{"kind":"fact_conflict",'
        '"entity_id":"known id or empty","evidence":"exact short span",'
        '"reason":"concise"}]}.'
        " Allowed kinds: " + ", ".join(sorted(AUDIT_KINDS)) + ".\n\n"
        "CANONICAL CONTRACT:\n"
        + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
        + "\n\nPLAYER INPUT:\n" + player_input
        + "\n\nNARRATION:\n" + narrative
    )


def parse_audit_response(raw: str, allowed_ids: set[str]) -> list[dict]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                      flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    result = []
    for raw_item in payload.get("violations", []) or []:
        if not isinstance(raw_item, dict):
            continue
        kind = str(raw_item.get("kind", "fact_conflict"))
        if kind not in AUDIT_KINDS:
            kind = "fact_conflict"
        entity_id = str(raw_item.get("entity_id", ""))
        if entity_id not in allowed_ids:
            entity_id = ""
        result.append({
            "kind": kind,
            "entity_id": entity_id,
            "evidence": str(raw_item.get("evidence", ""))[:240],
            "reason": str(raw_item.get("reason", ""))[:400],
            "source": "semantic_auditor",
        })
    if payload.get("valid") is False and not result:
        result.append({
            "kind": "fact_conflict", "entity_id": "", "evidence": "",
            "reason": "Auditor rejected the narration without a typed detail.",
            "source": "semantic_auditor",
        })
    return result


def audit_response_is_structured(raw: str) -> bool:
    """Return whether an auditor response satisfies the required JSON envelope."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                      flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return bool(
        isinstance(payload, dict)
        and isinstance(payload.get("valid"), bool)
        and isinstance(payload.get("violations"), list)
    )


def _windows(content: str, name: str) -> list[str]:
    return [
        content[max(0, match.start() - 80):min(len(content), match.end() + 80)]
        for match in re.finditer(re.escape(name), content, flags=re.IGNORECASE)
    ]


def deterministic_violations(content: str, contract: dict) -> list[dict]:
    """Catch high-confidence lifecycle/location conflicts without another model."""
    violations = []
    for entity in contract.get("entities", []):
        name = str(entity.get("name", "")).strip()
        if len(name) < 2 or not re.search(re.escape(name), content, re.IGNORECASE):
            continue
        windows = _windows(content, name)
        if entity.get("type") == "npc" and not entity.get("can_act"):
            if any(_NPC_ACTION_RE.search(window) and not _NEGATION_RE.search(window)
                   for window in windows):
                violations.append({
                    "kind": "fact_conflict", "entity_id": entity["id"],
                    "evidence": name,
                    "reason": "An inactive or absent NPC is narrated as acting.",
                    "source": "deterministic_guard",
                })
                continue
        if not entity.get("present") and entity.get("type") != "npc":
            if any(
                    _PRESENCE_RE.search(window)
                    and not _NEGATION_RE.search(window)
                    and not (entity.get("carried") and _CARRIED_RE.search(window))
                    for window in windows):
                violations.append({
                    "kind": "fact_conflict", "entity_id": entity["id"],
                    "evidence": name,
                    "reason": "An object is narrated at the wrong location.",
                    "source": "deterministic_guard",
                })
                continue
        if entity.get("type") in {"door", "container"}:
            if (not entity.get("open")
                    and any(_OPEN_RE.search(window) and not _NEGATION_RE.search(window)
                            for window in windows)):
                violations.append({
                    "kind": "fact_conflict", "entity_id": entity["id"],
                    "evidence": name,
                    "reason": "A closed object is narrated as open.",
                    "source": "deterministic_guard",
                })
            if (not entity.get("locked")
                    and any(_LOCKED_RE.search(window) and not _NEGATION_RE.search(window)
                            for window in windows)):
                violations.append({
                    "kind": "fact_conflict", "entity_id": entity["id"],
                    "evidence": name,
                    "reason": "An unlocked object is narrated as locked.",
                    "source": "deterministic_guard",
                })
    return violations


def build_repair_prompt(contract: dict, player_input: str, narrative: str,
                        violations: list[dict]) -> str:
    return (
        "Rewrite the narration once. Preserve supported content and tone, honor "
        "the player's action as an attempt, and remove only claims that conflict "
        "with canon. Do not add entities, locations, clues, outcomes, knowledge, "
        "or state changes. Output player-facing narration only, with no analysis "
        "and no JSON.\n\nCONTRACT:\n"
        + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
        + "\n\nVIOLATIONS:\n"
        + json.dumps(violations, ensure_ascii=False, separators=(",", ":"))
        + "\n\nPLAYER INPUT:\n" + player_input
        + "\n\nORIGINAL NARRATION:\n" + narrative
    )


def grounded_fallback(state: dict) -> str:
    """Render a minimal observation exclusively from committed outcomes."""
    world = state["world"]
    entities = world.get("entities", {})
    explicit_fallback = str(state.get("_grounded_fallback", "")).strip()
    if explicit_fallback:
        return explicit_fallback
    override = str(state.get("_narration_override") or "").strip()
    if override:
        return override
    if state.get("_pending_roll"):
        check = state.get("session", {}).get("pending_check", {})
        skill = str(check.get("skill", "") or "相应技能")
        return f"这项行动的结果尚不确定，请先进行〈{skill}〉检定。"
    if state.get("movement_target"):
        scene_id = str(state["movement_target"])
        name = str(world.get("scenes", {}).get(scene_id, {}).get("name", scene_id))
        return f"你沿着当前可行的路径抵达{name}。"

    lines = []
    for event in state.get("_action_events", []):
        eid = str(event.get("entity_id", ""))
        name = str(entities.get(eid, {}).get("name", eid or "该对象"))
        event_type = str(event.get("type", ""))
        if event_type == "item_picked_up":
            lines.append(f"你取得了{name}。")
        elif event_type == "item_dropped":
            lines.append(f"你将{name}留在当前地点。")
        elif event_type == "item_transferred":
            owner_id = str(event.get("owner_id", ""))
            owner = str(entities.get(owner_id, {}).get("name", "对方"))
            lines.append(f"{owner}接过了{name}。")
        elif event_type == "item_used":
            lines.append(f"你使用了{name}。")
        elif event_type == "object_opened":
            lines.append(f"{name}现在打开了。")
        elif event_type == "object_closed":
            lines.append(f"{name}现在关闭了。")
        elif event_type == "object_unlocked":
            lines.append(f"{name}已经解锁。")
        elif event_type == "object_locked":
            lines.append(f"{name}已经锁上。")
        elif event_type == "entity_discovered":
            lines.append(f"你确认发现了{name}。")
        elif event_type == "entity_removed":
            lines.append(f"{name}已不再能参与当前场景。")
    return "\n".join(lines) or "这次行动没有产生可以确认的额外世界变化。"

"""Hybrid action understanding and deterministic validation.

Natural language may be interpreted by a model, but the model only selects
from a closed list of visible/carried entity ids. Code owns preconditions and
the world events that mutate authoritative facts.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from world_state import fact_for, list_interactable_objects


ACTION_ALIASES = {
    "take": (
        "拿", "拿起", "拾起", "捡", "捡起", "取走", "带走", "收起",
        "夺取", "pick up", "take", "grab", "collect", "seize",
    ),
    "drop": ("放下", "丢下", "扔下", "drop", "put down", "leave behind"),
    "give": (
        "交给", "递给", "交还", "移交", "给他", "给她", "给对方",
        "give", "hand over",
        "turn over",
    ),
    "use": ("使用", "用", "喝", "服用", "use", "drink", "apply"),
    "open": ("打开", "开启", "掀开", "open"),
    "close": ("关上", "关闭", "合上", "close", "shut"),
    "unlock": ("解锁", "开锁", "unlock"),
    "lock": ("上锁", "锁上", "lock"),
    "read": ("阅读", "读", "查看文字", "read"),
    "inspect": (
        "检查", "查看", "观察", "调查", "搜索", "搜查", "翻找", "inspect",
        "examine", "search", "investigate", "look at",
    ),
    "break": (
        "攻击", "破坏", "砸", "砸坏", "踹", "撬", "烧", "烧毁", "点燃",
        "炸", "炸毁", "爆破", "摧毁", "毁掉", "拆", "拆毁", "杀", "杀死",
        "弄死", "break", "smash", "force", "attack", "burn", "destroy",
        "demolish", "ignite", "kill", "murder", "blow up", "set fire",
    ),
}

MUTATING_ACTIONS = frozenset({
    "take", "drop", "give", "use", "open", "close", "unlock", "lock", "break",
})
TARGET_REQUIRED_ACTIONS = frozenset({
    "take", "drop", "give", "use", "unlock", "lock", "break",
})

_ABSTRACT_ACTION_PATTERNS = (
    r"\btake\s+(?:a\s+|the\s+)?(?:[a-z-]+\s+){0,2}"
    r"(?:look|moment|breath|break|road|path|route|passage|corridor|hallway|"
    r"stairs?|stairway|staircase|ladder|bridgeway|gangway|trail|tunnel|"
    r"doorway|exit|turn)\b",
    r"\buse\s+(?:the\s+)?(?:hidden\s+)?(?:plan|strategy|trick|force|violence|skill|chance|time|road|path|route)\b",
    r"\bopen\s+fire\b",
    r"\bdrop\s+(?:the\s+)?subject\b",
    r"\bgive\s+(?:the\s+)?(?:rats?|people|crowd|others?|them|him|her)\s+"
    r"(?:enough\s+)?(?:room|space|way)\b",
    r"用[^\s，。！？]{0,24}(?:力量|暴力|方法|方式|计划|策略)",
)

_NAVIGATION_TAKE_RE = re.compile(
    r"\btake\s+(?:a\s+|the\s+)?(?:[a-z-]+\s+){0,3}"
    r"(?:road|path|route|passage|corridor|hallway|stairs?|stairway|"
    r"staircase|ladder|bridgeway|gangway|trail|tunnel|doorway|exit|turn)\b",
    re.IGNORECASE,
)
_CHINESE_NAVIGATION_RE = re.compile(
    r"(?:走|去|前往|进入|穿过|沿着|通过|爬|上|下|返回).{0,16}"
    r"(?:楼梯|梯子|走廊|通道|桥道|栈桥|道路|小径|隧道|门口|出口)"
)


def looks_like_navigation_action(player_input: str) -> bool:
    """Recognize route-taking phrases before stale object selection can bind."""
    text = (player_input or "").lower()
    return bool(_NAVIGATION_TAKE_RE.search(text)
                or _CHINESE_NAVIGATION_RE.search(text))


def looks_like_abstract_action(player_input: str) -> bool:
    text = (player_input or "").lower()
    return any(re.search(pattern, text) for pattern in _ABSTRACT_ACTION_PATTERNS)


def normalize(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", (value or "").lower(), flags=re.UNICODE)


def detect_action(player_input: str) -> str:
    lowered = (player_input or "").lower()
    if re.search(
            r"\b(?:kill(?:ed|ing)?|murder(?:ed|ing)?|burn(?:ed|ing)?|"
            r"destroy(?:ed|ing)?|demolish(?:ed|ing)?|ignit(?:e|ed|ing)|"
            r"attack(?:ed|ing)?|smash(?:ed|ing)?)\b",
            lowered):
        return "break"
    if re.search(r"\bpick\b.{0,40}\bup\b", lowered):
        return "take"
    if re.search(r"\bput\b.{0,40}\bdown\b", lowered):
        return "drop"
    if re.search(r"\b(?:hand|turn)\b.{0,40}\bover\b", lowered):
        return "give"
    if re.search(r"把.{1,40}给.{1,40}", lowered):
        return "give"
    hits = []
    for intent, aliases in ACTION_ALIASES.items():
        for alias in aliases:
            if alias.isascii():
                pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
                match = re.search(pattern, lowered)
                pos = match.start() if match else -1
            else:
                pos = lowered.find(alias)
            if pos >= 0:
                hits.append((pos, -len(alias), intent))
    concrete = [hit for hit in hits if hit[2] != "use"]
    return min(concrete or hits)[2] if hits else ""


def _candidate_forms(item: dict, world: dict) -> list[str]:
    entity = world.get("entities", {}).get(item["id"], {})
    forms = [item.get("label", "")]
    forms.extend(str(alias) for alias in (entity.get("aliases", []) or []))
    return [form for form in forms if normalize(form)]


def _explicit_targets(player_input: str, candidates: list[dict],
                      world: dict) -> list[dict]:
    needle = normalize(player_input)
    scored = []
    for item in candidates:
        matches = [form for form in _candidate_forms(item, world)
                   if normalize(form) in needle]
        if matches:
            scored.append((max(len(normalize(form)) for form in matches), item))
    if not scored:
        return []
    best = max(score for score, _item in scored)
    return [item for score, item in scored if score == best]


def build_action_planner_prompt(player_input: str, intent: str,
                                candidates: list[dict]) -> str:
    safe_candidates = [{
        "id": item["id"],
        "label": item["label"],
        "type": item["type"],
        "location": item["location"],
        "capabilities": item.get("capabilities", []),
    } for item in candidates]
    return (
        "Interpret one player action. You may only use entity ids from the "
        "candidate list. Never invent an entity, location, success, clue, or "
        "state change. Return JSON only with keys intent, target_id, tool_id, "
        "confidence, ambiguous. intent must be one of take/drop/use/open/close/"
        "unlock/lock/give/read/inspect/break/none. Use an empty id if unresolved.\n\n"
        f"Detected verb hint: {intent or 'none'}\n"
        f"Player: {player_input}\n"
        f"Candidates: {json.dumps(safe_candidates, ensure_ascii=False)}"
    )


def parse_ai_proposal(raw: str, allowed_ids: set[str]) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                      flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    intent = str(data.get("intent", "none")).lower()
    if intent not in {*ACTION_ALIASES, "none"}:
        intent = "none"
    target_id = str(data.get("target_id", ""))
    tool_id = str(data.get("tool_id", ""))
    if target_id not in allowed_ids:
        target_id = ""
    if tool_id not in allowed_ids:
        tool_id = ""
    try:
        confidence = max(0.0, min(float(data.get("confidence", 0)), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "intent": intent,
        "target_id": target_id,
        "tool_id": tool_id,
        "confidence": confidence,
        "ambiguous": bool(data.get("ambiguous", False)),
        "source": "ai_closed_world",
    }


def plan_action(player_input: str, session: dict, world: dict,
                scene_index: dict, entity_index: dict,
                ai_planner: Optional[Callable[[str], str]] = None) -> dict:
    candidates = list_interactable_objects(
        session, world, scene_index, entity_index)
    intent = detect_action(player_input)
    selected = session.get("selected_object_id")
    by_id = {item["id"]: item for item in candidates}
    explicit = _explicit_targets(player_input, candidates, world)
    navigation_action = looks_like_navigation_action(player_input)
    abstract_action = looks_like_abstract_action(player_input)

    target_id = ""
    source = "none"
    ambiguous = False
    if len(explicit) == 1:
        target_id = explicit[0]["id"]
        source = "exact_visible_reference"
    elif len(explicit) > 1:
        ambiguous = True
        source = "ambiguous_visible_reference"
    elif (selected in by_id and intent
          and not navigation_action and not abstract_action):
        target_id = selected
        source = "selected_object"

    proposal = {
        "intent": intent or "none",
        "target_id": target_id,
        "tool_id": "",
        "confidence": 1.0 if target_id else 0.0,
        "ambiguous": ambiguous,
        "source": source,
        "candidate_ids": list(by_id),
        "player_input": player_input,
    }

    # The model is useful only for semantic resolution that deterministic name
    # matching could not settle. It receives no hidden entities.
    if (intent and not target_id and not navigation_action
            and ai_planner and candidates):
        prompt = build_action_planner_prompt(player_input, intent, candidates)
        ai = parse_ai_proposal(ai_planner(prompt), set(by_id))
        if ai:
            if ai.get("intent") == "none":
                ai["intent"] = intent
            ai["candidate_ids"] = list(by_id)
            ai["player_input"] = player_input
            proposal = ai
    return proposal


def _location(fact: dict) -> tuple[str, str]:
    location = fact.get("location", {})
    return str(location.get("kind", "")), str(location.get("id", ""))


def validate_action(proposal: dict, session: dict, world: dict) -> dict:
    intent = str(proposal.get("intent", "none"))
    target_id = str(proposal.get("target_id", ""))
    if intent == "none":
        return {"status": "passthrough", "events": []}
    if proposal.get("ambiguous"):
        return {
            "status": "ambiguous", "events": [],
            "message": "这个动作可能指向多个对象，请先在对象列表中选择一个。",
        }
    # State-changing object verbs cannot be narrated without a grounded target.
    # A small abstract/movement whitelist keeps phrases such as "take the road"
    # and "use the plan" in the open narrative path.
    if not target_id:
        text = str(proposal.get("player_input", "")).lower()
        is_abstract = looks_like_abstract_action(text)
        if intent in TARGET_REQUIRED_ACTIONS and not is_abstract:
            if intent == "break":
                return {
                    "status": "ambiguous", "events": [],
                    "message": "破坏、攻击或杀伤必须先选择明确目标，并按模组规则或检定处理。",
                }
            return {
                "status": "ambiguous", "events": [],
                "message": "未能把这个动作对应到当前可见或背包中的对象，请先选择对象或明确名称。",
            }
        return {"status": "passthrough", "events": []}
    entity = world.get("entities", {}).get(target_id)
    if not isinstance(entity, dict) or entity.get("type") == "npc":
        return {
            "status": "blocked", "events": [],
            "message": "当前选择不是可交互对象。",
        }
    fact = fact_for(session, world, target_id)
    current_scene = session.get("player_state", {}).get("current_scene", "")
    kind, location_id = _location(fact)
    accessible = (kind == "inventory" or
                  (kind == "scene" and location_id == current_scene))
    if not fact.get("exists", True) or not fact.get("visible", False) or not accessible:
        return {
            "status": "blocked", "events": [],
            "message": "这个对象当前不在你能够接触的位置。",
        }

    base = {
        "status": "accepted", "events": [], "target_id": target_id,
        "intent": intent, "requires_adjudication": False,
    }
    if intent == "take":
        if kind == "inventory":
            return {**base, "status": "blocked", "message": "这个对象已经在你的背包中。"}
        if not fact.get("portable", False):
            return {**base, "status": "blocked", "message": "这个对象无法被直接带走。"}
        base["events"] = [{"type": "item_picked_up", "entity_id": target_id}]
    elif intent == "drop":
        if kind != "inventory":
            return {**base, "status": "blocked", "message": "你并没有携带这个对象。"}
        base["events"] = [{
            "type": "item_dropped", "entity_id": target_id,
            "scene_id": current_scene,
        }]
    elif intent == "give":
        if kind != "inventory":
            return {**base, "status": "blocked", "message": "你并没有携带这个对象。"}
        recipient_id = str(session.get("selected_npc_id", ""))
        recipient = world.get("entities", {}).get(recipient_id, {})
        if not recipient_id or recipient.get("type") != "npc":
            return {
                **base, "status": "blocked",
                "message": "请先选择一名在场人物作为接收者。",
            }
        from reference_resolver import npc_is_interactable
        recipient_fact = fact_for(session, world, recipient_id)
        recipient_location = recipient_fact.get("location", {})
        recipient_present = (
            recipient_location.get("kind") == "scene"
            and str(recipient_location.get("id", "")) == current_scene
        ) or recipient_id in set(session.get("companions", []))
        if (not recipient_present
                or not recipient_fact.get("visible", False)
                or not npc_is_interactable(recipient_id, world, session)):
            return {
                **base, "status": "blocked",
                "message": "所选人物当前不在场或无法接收物品。",
            }
        base["events"] = [{
            "type": "item_transferred", "entity_id": target_id,
            "owner_id": recipient_id,
        }]
    elif intent == "use":
        if kind != "inventory":
            return {**base, "status": "blocked", "message": "你必须先取得这个对象才能使用它。"}
        base["events"] = [{
            "type": "item_used", "entity_id": target_id,
            "consumed": bool(entity.get("consumable", False)),
        }]
    elif intent == "open":
        if fact.get("locked"):
            return {**base, "status": "blocked", "message": "这个对象仍然锁着。"}
        if fact.get("open"):
            return {**base, "status": "blocked", "message": "这个对象已经打开了。"}
        base["events"] = [{"type": "object_opened", "entity_id": target_id}]
    elif intent == "close":
        if not fact.get("open"):
            return {**base, "status": "blocked", "message": "这个对象目前并未打开。"}
        base["events"] = [{"type": "object_closed", "entity_id": target_id}]
    elif intent == "unlock":
        if not fact.get("locked"):
            return {**base, "status": "blocked", "message": "这个对象并没有上锁。"}
        required_key = str(entity.get("requires_key", ""))
        tool_id = str(proposal.get("tool_id", ""))
        carried = set(session.get("inventory_entity_ids", []))
        if required_key and required_key not in carried:
            return {**base, "status": "blocked", "message": "你没有能够打开这把锁的钥匙。"}
        if tool_id and tool_id not in carried:
            return {**base, "status": "blocked", "message": "你并没有携带所选工具。"}
        base["events"] = [{
            "type": "object_unlocked", "entity_id": target_id,
            "tool_id": tool_id or required_key,
        }]
    elif intent == "lock":
        if fact.get("locked"):
            return {**base, "status": "blocked", "message": "这个对象已经锁上了。"}
        base["events"] = [{"type": "object_locked", "entity_id": target_id}]
    elif intent == "break":
        return {
            **base,
            "status": "blocked",
            "message": "破坏或攻击不能仅凭描述直接生效；请选择明确目标，并按模组规则或检定处理。",
        }
    elif intent in {"inspect", "read"}:
        base["requires_adjudication"] = True
    return base


def legacy_state_matches_action(entity: dict, current_state: str,
                                proposal: dict, player_input: str) -> bool:
    state_def = entity.get("states", {}).get(current_state, {})
    if not isinstance(state_def, dict) or not state_def:
        return False
    lowered = (player_input or "").lower()
    triggers = [str(value).lower() for value in state_def.get("triggers", [])]
    if any(trigger and trigger in lowered for trigger in triggers):
        return True
    intent = proposal.get("intent", "none")
    aliases = ACTION_ALIASES.get(intent, ())
    return any(any(alias in trigger for alias in aliases) for trigger in triggers)


def legacy_action_requirements(state_def: dict, session: dict) -> list[str]:
    """Return unmet closed-world prerequisites for an authored state trigger."""
    missing: list[str] = []
    inventory = set(session.get("inventory_entity_ids", []))
    flags = set(session.get("flags", []))
    entity_states = session.get("entity_states", {})

    required_inventory = state_def.get("requires_inventory", [])
    if isinstance(required_inventory, str):
        required_inventory = [required_inventory]
    for entity_id in required_inventory if isinstance(required_inventory, list) else []:
        entity_id = str(entity_id).strip()
        if entity_id and entity_id not in inventory:
            missing.append(f"inventory:{entity_id}")

    required_flags = state_def.get("requires_flags", [])
    if isinstance(required_flags, str):
        required_flags = [required_flags]
    for flag in required_flags if isinstance(required_flags, list) else []:
        flag = str(flag).strip()
        if flag and flag not in flags:
            missing.append(f"flag:{flag}")

    any_flags = state_def.get("requires_any_flags", [])
    if isinstance(any_flags, str):
        any_flags = [any_flags]
    if isinstance(any_flags, list):
        choices = [str(flag).strip() for flag in any_flags if str(flag).strip()]
        if choices and not any(flag in flags for flag in choices):
            missing.append("any_flag:" + "|".join(choices))

    required_states = state_def.get("requires_entity_states", {})
    if isinstance(required_states, dict):
        for entity_id, expected in required_states.items():
            allowed = expected if isinstance(expected, list) else [expected]
            allowed = {str(value) for value in allowed}
            if str(entity_states.get(str(entity_id), "")) not in allowed:
                missing.append(f"state:{entity_id}")
    return missing

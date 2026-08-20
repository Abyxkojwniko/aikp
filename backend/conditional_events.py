"""Source-grounded conditional perception and chained-check runtime."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Callable, Optional

from world_state import commit_world_events


SkillResolver = Callable[[str, dict, str], Optional[tuple[str, int, int, int]]]

_SUCCESS_RANK = {
    "critical_success": 4,
    "extreme_success": 3,
    "hard_success": 2,
    "success": 1,
    "luck_auto_success": 4,
}
_DIFFICULTY_RANK = {"regular": 1, "normal": 1, "hard": 2, "extreme": 3}

_CLOCK_MILESTONE_RE = re.compile(
    r"◇\s*当\s*(?P<name>[^\n]{0,20}?轮次)\s*到达\s*"
    r"[\[［【]?\s*(?P<at>\d+)\s*[\]］】]?\s*时"
)
_CONDITIONAL_SKILLS = (
    "图书馆使用", "克苏鲁神话", "心理学", "神秘学", "灵感", "智力",
    "侦查", "聆听", "幸运", "教育", "体质", "意志", "力量", "敏捷",
    "外貌", "体型", "医学", "急救", "潜行", "导航", "攀爬", "闪避",
    "说服", "恐吓", "话术", "艺术", "英语", "INT", "POW", "CON",
    "STR", "DEX", "APP", "SIZ", "EDU",
)
_SKILL_PATTERN = "|".join(
    re.escape(skill) for skill in sorted(_CONDITIONAL_SKILLS, key=len, reverse=True))
_CHECK_RE = re.compile(
    rf"(?:(?P<difficulty>普通|困难|极难)\s*(?:难度)?\s*的?\s*)?"
    rf"(?P<skill>{_SKILL_PATTERN})\s*(?:技能)?\s*检定",
    flags=re.IGNORECASE,
)
_SAN_RE = re.compile(
    r"(?:SAN|理智(?:值)?)\s*(?:检定)?\s*"
    r"(?P<loss>\d*d?\d+(?:[+\-]\d+)?\s*/\s*\d*d?\d+(?:[+\-]\d+)?)",
    flags=re.IGNORECASE,
)


def _difficulty_name(raw: str) -> str:
    return {"困难": "hard", "极难": "extreme"}.get(raw, "regular")


def _outcome_observation(section: str, outcome: str) -> str:
    marker = "成功" if outcome == "success" else "失败"
    match = re.search(
        rf"(?:若|如果)?\s*(?:检定)?{marker}\s*(?:则|后)?\s*"
        r"(?P<text>[^。；）)]{1,180})",
        section,
        flags=re.DOTALL,
    )
    if not match:
        return ""
    text = re.sub(r"\s+", "", match.group("text")).strip("，, ：:")
    text = re.split(
        r"[,，]?(?:并|再|随后|之后)?\s*(?:进行|作|做).{0,20}"
        r"(?:SAN|理智).*$",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip("，, 。")
    return text


def extract_clock_conditional_events(text: str, action_clocks: dict) -> list[dict]:
    """Recover explicit milestone -> check -> observation/follow-up chains."""
    if not text or not isinstance(action_clocks, dict):
        return []
    matches = list(_CLOCK_MILESTONE_RE.finditer(text))
    events = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end():next_start]
        structural = re.search(r"(?m)^■", section)
        if structural:
            section = section[:structural.start()]
        check_matches = list(_CHECK_RE.finditer(section))
        # Multi-skill chains can carry penalties and branch-specific endings.
        # Leave those to the structured reconstruction rather than flattening
        # them into an incorrect single deterministic check.
        if len(check_matches) != 1:
            continue
        check_match = check_matches[0]
        at = int(match.group("at"))
        clock_id = next((
            candidate_id for candidate_id, definition in action_clocks.items()
            if isinstance(definition, dict)
            and str(definition.get("name", "")).strip() == match.group("name").strip()
        ), next(iter(action_clocks), ""))
        if not clock_id:
            continue
        mechanic = re.search(
            r"[（(][^）)]{0,600}" + re.escape(check_match.group(0))
            + r"[^）)]{0,600}[）)]",
            section,
            flags=re.DOTALL | re.IGNORECASE,
        )
        source_quote = (mechanic.group(0) if mechanic else section[
            max(0, check_match.start() - 80):check_match.end() + 240]).strip()
        event_id = f"{clock_id}:{at}:conditional-perception"
        success: dict = {}
        failure: dict = {}
        success_text = _outcome_observation(source_quote, "success")
        failure_text = _outcome_observation(source_quote, "failure")
        if success_text:
            success["observations"] = [{
                "id": f"{event_id}:success",
                "text": success_text,
                "source_ref": text.count("\n", 0, match.start()),
            }]
        if failure_text:
            failure["observations"] = [{
                "id": f"{event_id}:failure",
                "text": failure_text,
                "source_ref": text.count("\n", 0, match.start()),
            }]
        san_match = _SAN_RE.search(source_quote)
        if san_match:
            success["followup_checks"] = [{
                "type": "san",
                "loss": re.sub(r"\s+", "", san_match.group("loss")),
            }]
        events.append({
            "id": event_id,
            "observer_scope": "active_character",
            "once_per_observer": True,
            "when": {
                "type": "clock_at_least",
                "clock_id": clock_id,
                "value": at,
            },
            "check": {
                "type": "skill",
                "skill": check_match.group("skill"),
                "difficulty": _difficulty_name(check_match.group("difficulty") or ""),
            },
            "outcomes": {"success": success, "failure": failure},
            "source_quote": source_quote,
            "source_line": text.count("\n", 0, match.start()),
            "source_verified": bool(source_quote and source_quote in text),
        })
    return events


def augment_world_conditional_events(world: dict) -> list[dict]:
    """Add deterministic recoveries without replacing reconstructed events."""
    existing = [
        dict(row) for row in world.get("conditional_events", []) or []
        if isinstance(row, dict) and row.get("id")
    ]
    source_parts = []
    for scene in world.get("scenes", {}).values():
        if not isinstance(scene, dict):
            continue
        source = str(scene.get("source_text", "") or "")
        if source and source not in source_parts:
            source_parts.append(source)
    recovered = extract_clock_conditional_events(
        "\n\n".join(source_parts), world.get("action_clocks", {}))
    by_id = {str(row["id"]): row for row in existing}
    for row in recovered:
        by_id.setdefault(str(row["id"]), row)
    world["conditional_events"] = list(by_id.values())
    return world["conditional_events"]


def active_observer_id(session: dict) -> str:
    player = session.setdefault("player_state", {})
    observer_id = str(
        player.get("active_character_id")
        or session.get("active_character_id")
        or "player"
    ).strip()
    return observer_id or "player"


def ensure_conditional_state(session: dict) -> None:
    session.setdefault("knowledge_facts", {})
    session.setdefault("knowledge_events", [])
    session.setdefault("knowledge_event_seq", 0)
    session.setdefault("conditional_event_states", {})


def observer_knowledge(session: dict, observer_id: str = "") -> list[dict]:
    ensure_conditional_state(session)
    observer_id = observer_id or active_observer_id(session)
    facts = session.get("knowledge_facts", {})
    combined = {}
    for scope in ("public", observer_id):
        rows = facts.get(scope, {})
        if isinstance(rows, dict):
            combined.update(rows)
    return [dict(combined[key]) for key in sorted(combined)]


def render_observer_knowledge(session: dict, observer_id: str = "") -> str:
    observer_id = observer_id or active_observer_id(session)
    rows = observer_knowledge(session, observer_id)
    if not rows:
        return ""
    lines = [
        "=== ACTIVE INVESTIGATOR KNOWLEDGE (CANONICAL) ===",
        f"Observer: {observer_id}",
        "Only this observer's facts and public facts may be treated as perceived. "
        "Never reveal another observer's private result.",
    ]
    lines.extend(f"- {row['text']}" for row in rows if row.get("text"))
    return "\n".join(lines)


def _record_observation(
    session: dict,
    event_id: str,
    outcome: str,
    observer_id: str,
    raw: object,
    index: int,
) -> str:
    if isinstance(raw, str):
        observation = {"text": raw}
    elif isinstance(raw, dict):
        observation = dict(raw)
    else:
        return ""
    text = str(
        observation.get("text")
        or observation.get("fact")
        or observation.get("description")
        or ""
    ).strip()
    if not text:
        return ""
    fact_id = str(observation.get("id") or f"{event_id}:{outcome}:{index}")
    scope = "public" if observation.get("public") is True else observer_id
    fact = {
        "id": fact_id,
        "text": text,
        "observer_id": scope,
        "event_id": event_id,
        "outcome": outcome,
        "turn": int(session.get("current_turn", 0)),
        "source_ref": observation.get("source_ref"),
    }
    session.setdefault("knowledge_facts", {}).setdefault(scope, {})[fact_id] = fact
    session["knowledge_event_seq"] = int(session.get("knowledge_event_seq", 0)) + 1
    session.setdefault("knowledge_events", []).append({
        **fact,
        "seq": session["knowledge_event_seq"],
        "type": "observation_recorded",
    })
    return text


def _render_observation(text: str) -> str:
    """Turn terse authored outcome labels into minimal player-facing prose."""
    match = re.fullmatch(
        r"(?:调查员)?(?:会|将|能够|能)?"
        r"(?P<verb>看到|看见|听到|发现|意识到|察觉到)(?P<object>.+)",
        text.strip().rstrip("。！？!?"),
    )
    if not match:
        return text
    completed = {
        "看到": "看到了", "看见": "看见了", "听到": "听到了",
        "发现": "发现了", "意识到": "意识到了", "察觉到": "察觉到了",
    }[match.group("verb")]
    return f"你{completed}{match.group('object')}。"


def _compare(left: object, operator: str, right: object) -> bool:
    if operator in {"eq", "=="}:
        return left == right
    if operator in {"ne", "!="}:
        return left != right
    try:
        lhs, rhs = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return {
        "gte": lhs >= rhs,
        ">=": lhs >= rhs,
        "gt": lhs > rhs,
        ">": lhs > rhs,
        "lte": lhs <= rhs,
        "<=": lhs <= rhs,
        "lt": lhs < rhs,
        "<": lhs < rhs,
    }.get(operator, False)


def condition_satisfied(
    condition: object, session: dict, observer_id: str = "",
) -> bool:
    if not isinstance(condition, dict):
        return False
    if isinstance(condition.get("all"), list):
        return all(condition_satisfied(row, session, observer_id)
                   for row in condition["all"])
    if isinstance(condition.get("any"), list):
        return any(condition_satisfied(row, session, observer_id)
                   for row in condition["any"])
    if isinstance(condition.get("not"), dict):
        return not condition_satisfied(condition["not"], session, observer_id)

    kind = str(condition.get("type", "")).strip().lower()
    observer_id = observer_id or active_observer_id(session)
    if kind in {"always", "immediate"}:
        return True
    if kind in {"clock_at_least", "clock_reaches"}:
        clock_id = str(condition.get("clock_id", ""))
        target = condition.get("value", condition.get("at", 0))
        return _compare(session.get("clocks", {}).get(clock_id, 0), "gte", target)
    if kind == "flag_present":
        return str(condition.get("flag", "")) in session.get("flags", [])
    if kind == "scene_is":
        return str(session.get("player_state", {}).get("current_scene", "")) == str(
            condition.get("scene_id", ""))
    if kind == "entity_state_is":
        return str(session.get("entity_states", {}).get(
            str(condition.get("entity_id", "")), "")) == str(condition.get("state", ""))
    if kind == "player_stat":
        from perception import observer_player_state
        player = observer_player_state(session, observer_id)
        name = str(condition.get("name", ""))
        canonical_name = {
            "灵感": "智力", "idea": "智力", "inspiration": "智力",
            "int": "智力", "pow": "意志", "con": "体质", "str": "力量",
            "dex": "敏捷", "app": "外貌", "siz": "体型", "edu": "教育",
        }.get(name.casefold(), name)
        value = player.get(name, player.get(canonical_name))
        if value is None:
            value = player.get("attributes", {}).get(
                name, player.get("attributes", {}).get(canonical_name))
        if value is None:
            value = player.get("skills", {}).get(
                name, player.get("skills", {}).get(canonical_name))
        return _compare(value, str(condition.get("operator", "gte")), condition.get("value"))
    if kind == "knowledge_fact_present":
        fact_id = str(condition.get("fact_id", ""))
        facts = session.get("knowledge_facts", {})
        return bool(
            fact_id and (
                fact_id in facts.get("public", {})
                or fact_id in facts.get(observer_id, {})
            )
        )
    if kind == "conditional_outcome_is":
        event_id = str(condition.get("event_id", ""))
        expected = str(condition.get("outcome", ""))
        state = session.get("conditional_event_states", {}).get(
            event_id, {}).get(observer_id, {})
        return bool(event_id and expected and str(state.get("outcome", "")) == expected)
    return False


def _event_by_id(world: dict, event_id: str) -> dict:
    for event in world.get("conditional_events", []) or []:
        if isinstance(event, dict) and str(event.get("id", "")) == event_id:
            return event
    return {}


def _event_state(session: dict, event_id: str, observer_id: str) -> dict:
    ensure_conditional_state(session)
    return session["conditional_event_states"].setdefault(
        event_id, {}).setdefault(observer_id, {})


def check_passed(pending: dict, result: dict) -> bool:
    if not result.get("success"):
        return False
    required = str(pending.get("_required_success_level", "regular")).lower()
    return _SUCCESS_RANK.get(str(result.get("verdict", "success")), 1) >= _DIFFICULTY_RANK.get(
        required, 1)


def _build_pending_check(
    event: dict,
    observer_id: str,
    check: object,
    session: dict,
    resolve_skill: SkillResolver,
    *,
    stage: str,
    remaining: Optional[list] = None,
) -> Optional[dict]:
    spec = {"skill": check} if isinstance(check, str) else (
        dict(check) if isinstance(check, dict) else {})
    check_type = str(spec.get("type", "skill")).lower()
    common = {
        "entity_id": "",
        "state": "",
        "rule_system": session.get("rule_system", "coc"),
        "scene": session.get("player_state", {}).get("current_scene", ""),
        "dynamic": False,
        "_conditional_event_id": str(event.get("id", "")),
        "_conditional_observer_id": observer_id,
        "_conditional_stage": stage,
        "_conditional_check_spec": deepcopy(spec),
        "_conditional_remaining_checks": deepcopy(remaining or []),
    }
    if check_type == "san":
        loss = str(spec.get("loss") or spec.get("san_check") or "").strip()
        if not loss:
            return None
        return {
            **common,
            "skill": "",
            "skill_value": 0,
            "effective": 0,
            "dc": 0,
            "san_check": loss,
        }

    skill_text = str(spec.get("skill") or spec.get("name") or "").strip()
    from perception import observer_player_state
    resolved = resolve_skill(
        skill_text,
        observer_player_state(session, observer_id),
        str(session.get("rule_system", "coc")),
    )
    if not resolved:
        return None
    display, skill_value, effective, dc = resolved
    return {
        **common,
        "skill": display,
        "skill_value": skill_value,
        "effective": effective,
        "dc": dc,
        "san_check": "",
        "_required_success_level": str(spec.get("difficulty", "regular")).lower(),
    }


def _payload_followups(payload: dict) -> list:
    followups = list(payload.get("followup_checks", []) or [])
    san_check = str(payload.get("san_check", "") or "").strip()
    if san_check:
        followups.append({"type": "san", "loss": san_check})
    return followups


def _apply_payload(
    session: dict,
    world: dict,
    event: dict,
    observer_id: str,
    outcome_name: str,
    payload: object,
) -> list[str]:
    if not isinstance(payload, dict):
        payload = {}
    event_id = str(event.get("id", ""))
    lines = []
    explicit = str(payload.get("narration", "") or "").strip()
    if explicit:
        lines.append(explicit)
    observations = payload.get("observations", payload.get("observation", []))
    if not isinstance(observations, list):
        observations = [observations]
    for index, observation in enumerate(observations):
        text = _record_observation(
            session, event_id, outcome_name, observer_id, observation, index)
        rendered = _render_observation(text) if text else ""
        if rendered and rendered not in lines:
            lines.append(rendered)

    for flag in payload.get("flags", []) or []:
        flag = str(flag).strip()
        if flag and flag not in session.setdefault("flags", []):
            session["flags"].append(flag)
    from perception import activate_perception_layer, deactivate_perception_layer
    activation_ids = payload.get("activate_perception_layers", []) or []
    if isinstance(activation_ids, str):
        activation_ids = [activation_ids]
    for layer_id in activation_ids:
        activate_perception_layer(
            session, world, str(layer_id), observer_id,
            source=f"conditional_event:{event_id}:{outcome_name}")
    deactivation_ids = payload.get("deactivate_perception_layers", []) or []
    if isinstance(deactivation_ids, str):
        deactivation_ids = [deactivation_ids]
    for layer_id in deactivation_ids:
        deactivate_perception_layer(
            session, world, str(layer_id), observer_id,
            source=f"conditional_event:{event_id}:{outcome_name}")
    raw_events = [
        dict(row) for row in payload.get("events", []) or []
        if isinstance(row, dict)
    ]
    if raw_events:
        for row in raw_events:
            row.setdefault("source", f"conditional_event:{event_id}:{outcome_name}")
        commit_world_events(
            session, world, raw_events, actor="rules",
            source=f"conditional_event:{event_id}:{outcome_name}")
    return lines


def _continue_or_finish(
    session: dict,
    world: dict,
    event: dict,
    observer_id: str,
    followups: list,
    resolve_skill: SkillResolver,
) -> Optional[dict]:
    event_id = str(event.get("id", ""))
    state = _event_state(session, event_id, observer_id)
    while followups:
        current, *remaining = followups
        pending = _build_pending_check(
            event, observer_id, current, session, resolve_skill,
            stage="followup", remaining=remaining)
        if pending:
            state["status"] = "awaiting_followup"
            session["pending_check"] = pending
            return pending
        followups = remaining
    state["status"] = "resolved"
    state["resolved_turn"] = int(session.get("current_turn", 0))
    session["pending_check"] = None
    return None


def arm_conditional_events(
    session: dict,
    world: dict,
    resolve_skill: SkillResolver,
) -> dict:
    ensure_conditional_state(session)
    if session.get("pending_check"):
        return {}
    observer_id = active_observer_id(session)
    selected_scenario = str(session.get("current_scenario_id", ""))
    immediate_narration = []
    for event in world.get("conditional_events", []) or []:
        if not isinstance(event, dict) or not event.get("id"):
            continue
        if (selected_scenario and event.get("scenario_id")
                and str(event.get("scenario_id")) != selected_scenario):
            continue
        event_id = str(event["id"])
        state = _event_state(session, event_id, observer_id)
        if state.get("status") in {"pending", "awaiting_followup", "resolved", "invalid"}:
            continue
        if not condition_satisfied(
                event.get("when", event.get("trigger")), session, observer_id):
            continue
        state.update({
            "status": "triggered",
            "triggered_turn": int(session.get("current_turn", 0)),
        })
        check = event.get("check")
        if check:
            pending = _build_pending_check(
                event, observer_id, check, session, resolve_skill, stage="primary")
            if not pending:
                state.update({"status": "invalid", "reason": "unresolvable_check"})
                continue
            state["status"] = "pending"
            session["pending_check"] = pending
            return {"event": event, "pending_check": pending,
                    "narration": immediate_narration}

        payload = (event.get("outcomes", {}) or {}).get(
            "success", event.get("outcome", {}))
        immediate_narration.extend(_apply_payload(
            session, world, event, observer_id, "success", payload))
        followups = _payload_followups(payload if isinstance(payload, dict) else {})
        pending = _continue_or_finish(
            session, world, event, observer_id, followups, resolve_skill)
        if pending:
            return {"event": event, "pending_check": pending,
                    "narration": immediate_narration}
    return {"narration": immediate_narration} if immediate_narration else {}


def resolve_primary_outcome(
    session: dict,
    world: dict,
    pending: dict,
    passed: bool,
    verdict: str,
    resolve_skill: SkillResolver,
) -> dict:
    event_id = str(pending.get("_conditional_event_id", ""))
    observer_id = str(pending.get("_conditional_observer_id", ""))
    event = _event_by_id(world, event_id)
    if not event or not observer_id:
        raise ValueError("Conditional check references an unknown event or observer")
    outcome_name = "success" if passed else "failure"
    payload = (event.get("outcomes", {}) or {}).get(outcome_name, {})
    state = _event_state(session, event_id, observer_id)
    state.update({"outcome": outcome_name, "verdict": verdict})
    lines = _apply_payload(
        session, world, event, observer_id, outcome_name, payload)
    followups = _payload_followups(payload if isinstance(payload, dict) else {})
    next_check = _continue_or_finish(
        session, world, event, observer_id, followups, resolve_skill)
    return {"outcome": outcome_name, "narration": lines, "next_check": next_check}


def resolve_followup_outcome(
    session: dict,
    world: dict,
    pending: dict,
    passed: bool,
    verdict: str,
    resolve_skill: SkillResolver,
) -> dict:
    event_id = str(pending.get("_conditional_event_id", ""))
    observer_id = str(pending.get("_conditional_observer_id", ""))
    event = _event_by_id(world, event_id)
    if not event or not observer_id:
        raise ValueError("Conditional follow-up references an unknown event or observer")
    spec = pending.get("_conditional_check_spec", {})
    branch_name = "success" if passed else "failure"
    branch = {}
    if isinstance(spec, dict):
        branch = spec.get("on_success" if passed else "on_failure", {}) or {}
    lines = _apply_payload(
        session, world, event, observer_id,
        f"followup_{branch_name}", branch)
    remaining = list(pending.get("_conditional_remaining_checks", []) or [])
    remaining = _payload_followups(branch if isinstance(branch, dict) else {}) + remaining
    next_check = _continue_or_finish(
        session, world, event, observer_id, remaining, resolve_skill)
    state = _event_state(session, event_id, observer_id)
    state.setdefault("followups", []).append({
        "type": str(spec.get("type", "skill")) if isinstance(spec, dict) else "skill",
        "outcome": branch_name,
        "verdict": verdict,
    })
    return {"outcome": branch_name, "narration": lines, "next_check": next_check}

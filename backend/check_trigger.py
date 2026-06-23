# -*- coding: utf-8 -*-
"""AIKP Check Trigger — dual-path dice check + SAN auto-detection.

Design: docs/npc-design.md

Path A: JSON-fixed
  entity.states[state].check = "will 14"      -> dice.check("will", 14)
  entity.states[state].san_check = "1/1d6"   -> coc_san_loss("1/1d6", current_san)

Path B: LLM dynamic
  GM narration contains "CHECK:San 12"        -> parse -> dice -> inject next turn

SAN auto-detection:
  Engine scans GM narration for horror keywords -> auto 1/1d3 SAN check
"""

from __future__ import annotations
import re


# ── Path A: Fixed Checks ──────────────────────────────────────

def check_fixed(entity: dict, current_state: str) -> dict | None:
    """Execute fixed check from entity state definition.

    Returns dice result dict or None if no check defined.
    """
    states = entity.get("states", {})
    state_def = states.get(current_state, {})

    # Regular skill check
    check_str = state_def.get("check", "")
    if check_str:
        return _parse_and_execute_skill_check(check_str)

    # SAN check
    san_str = state_def.get("san_check", "")
    if san_str:
        return {"type": "san", "san_check": san_str}

    return None


def _parse_and_execute_skill_check(check_str: str, player_skills: dict = None,
                                    rule_system: str = "dnd") -> dict:
    """Parse 'Skill DC' string and execute dice roll."""
    parts = check_str.strip().split()
    if len(parts) < 2:
        return None

    try:
        dc = int(parts[-1])
    except ValueError:
        return None

    skill = " ".join(parts[:-1])
    sv = (player_skills or {}).get(skill, 0)
    from dice import resolve_check
    return resolve_check(rule_system, sv, dc)


# ── Path B: Dynamic Checks ────────────────────────────────────

DYNAMIC_CHECK_RE = re.compile(
    r'CHECK\s*:\s*(\w+(?:\s+\w+)*)\s+(\d+)', re.IGNORECASE
)


def parse_dynamic_checks(text: str) -> list[dict]:
    """Parse CHECK:Skill DC from GM narration text.

    Returns list of parsed check dicts.
    """
    matches = DYNAMIC_CHECK_RE.findall(text)
    results = []
    for skill, dc_str in matches:
        try:
            dc = int(dc_str)
            results.append({"skill": skill.strip(), "dc": dc, "raw": f"CHECK:{skill} {dc}"})
        except ValueError:
            pass
    return results


def execute_dynamic_checks(checks: list[dict], player_skills: dict = None,
                           rule_system: str = "dnd") -> list[dict]:
    """Execute parsed dynamic checks via dice.py."""
    from dice import resolve_check
    results = []
    for chk in checks:
        if chk.get("skill", "").lower() == "san":
            results.append({
                "type": "san",
                "raw": chk["raw"],
                "dc": chk["dc"],
            })
        else:
            sv = (player_skills or {}).get(chk["skill"], 0)
            result = resolve_check(rule_system, sv, chk["dc"])
            results.append({"type": "skill", "result": result, "raw": chk["raw"]})
    return results


# ── SAN Auto-Detection ────────────────────────────────────────

HORROR_KEYWORDS = [
    # Cosmic / Mythos — only extreme, sanity-shattering encounters
    "修普诺斯", "克苏鲁", "旧神", "旧日", "外神",
    "不可名状", "难以名状", "古老存在",
    "理智崩溃", "超越认知", "维度撕裂", "时空错乱",
    "奈亚拉托提普", "莎布·尼古拉斯", "阿撒托斯", "哈斯塔",
    "黄衣之王", "犹格·索托斯",
]

SAN_TIER_MAP = {
    "mild": "0/1",       # corpse, blood — driven by module text, not auto-trigger
    "moderate": "0/1d3",  # supernatural phenomenon
    "severe": "1d3/1d10", # cosmic entity encounter
}

HORROR_PATTERN = re.compile(
    '|'.join(re.escape(kw) for kw in HORROR_KEYWORDS),
    re.IGNORECASE
)

DEFAULT_SAN_STRING = "0/1d3"


def detect_san_trigger(narration: str) -> str | None:
    """Scan GM narration for horror/cosmic keywords.

    Returns the matched keyword or None.
    """
    match = HORROR_PATTERN.search(narration)
    if match:
        return match.group(0)
    return None


def execute_san_check(san_string: str, current_san: int) -> dict:
    """Execute a SAN check via dice.py coc_san_loss.

    Returns SAN result dict.
    """
    from dice import coc_san_loss
    return coc_san_loss(san_string, current_san)


def process_gm_response(gm_response: str, current_san: int) -> dict:
    """Post-turn processing of GM response: detect SAN triggers + dynamic checks.

    Returns dict with san_result, dynamic_check_results, and san_triggered flag.
    """
    result = {
        "san_result": None,
        "dynamic_checks": [],
        "san_triggered": False,
    }

    # Auto SAN detection
    trigger = detect_san_trigger(gm_response)
    if trigger:
        result["san_result"] = execute_san_check(DEFAULT_SAN_STRING, current_san)
        result["san_triggered"] = True

    # Dynamic checks
    checks = parse_dynamic_checks(gm_response)
    if checks:
        result["dynamic_checks"] = execute_dynamic_checks(checks)

    return result

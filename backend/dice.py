# -*- coding: utf-8 -*-
"""AIKP 骰子引擎 —— 与 LLM 完全分离的确定性判定模块"""

import random
from typing import Optional

# ── 通用骰子 ──────────────────────────────────────────

def roll(dice: str) -> int:
    """标准骰子表达式：d20, 2d6, d100, d20+3, 2d6-1"""
    dice = dice.strip().lower()
    bonus = 0

    if "+" in dice:
        dice, bonus_str = dice.split("+", 1)
        bonus = int(bonus_str.strip())
    elif "-" in dice:
        dice, bonus_str = dice.split("-", 1)
        bonus = -int(bonus_str.strip())

    parts = dice.split("d")
    count = int(parts[0]) if parts[0] else 1
    sides = int(parts[1])

    total = sum(random.randint(1, sides) for _ in range(count))
    return total + bonus


# ── 技能判定 ──────────────────────────────────────────

def skill_check(skill_value: int, difficulty: int) -> dict:
    """
    d20 + 技能值 vs 难度值 (DC)
    返回详细结果便于 LLM 叙事
    """
    d20_result = random.randint(1, 20)
    total = d20_result + skill_value
    success = total >= difficulty

    if d20_result == 20:
        verdict = "critical_success"
    elif d20_result == 1:
        verdict = "critical_failure"
    elif success:
        verdict = "success"
    else:
        verdict = "failure"

    return {
        "skill_value": skill_value,
        "difficulty": difficulty,
        "d20": d20_result,
        "total": total,
        "success": success,
        "verdict": verdict,
    }


# ── 理智判定（SAN） ───────────────────────────────────

def sanity_check(current_san: int, loss_on_fail: str) -> dict:
    """SAN 检定，成功无损失，失败扣对应骰子"""
    roll_result = roll("d100")
    success = roll_result <= current_san

    if success:
        loss = 0
    else:
        loss = roll(loss_on_fail)

    return {
        "san_before": current_san,
        "d100": roll_result,
        "success": success,
        "san_loss": loss,
        "san_after": current_san - loss,
    }


def coc_san_loss(sancheck_str: str, current_san: int, pow_stat: int = 50) -> dict:
    """CoC-style SAN check: "0/1d3" = success: no loss, fail: 1d3
    "1/1d5" = success: lose 1, fail: lose 1d5
    "1d3/1d5+1" = success: lose 1d3, fail: lose 1d5+1
    Also handles: "POW*5", "Luck", "DEX*5", "SANcheck 1d3/1d6" etc.
    
    Returns full result dict including temporary/indefinite insanity flags.
    """
    # Normalize: strip prefixes like "SANcheck ", trim whitespace
    clean = sancheck_str.strip()
    for prefix in ("SANcheck", "SANcheck ", "sancheck", "SAN"):
        if clean.lower().startswith(prefix.lower()):
            clean = clean[len(prefix):].strip()
    
    if not clean or clean in ("0", "none", "null", "-"):
        return {"loss": 0, "san_after": current_san, "insanity_temp": False, "insanity_indef": False}

    # Parse: "X/Y" format
    success_loss = "0"
    fail_loss = clean
    
    if "/" in clean:
        parts = clean.split("/")
        success_loss = parts[0].strip()
        fail_loss = parts[1].strip()

    # CoC SAN check: roll d100 vs CURRENT SAN — ≤ current_san succeeds (lose the
    # success amount), > current_san fails (lose the failure dice). Must scale
    # with the investigator's SAN, NOT a fixed value.
    d100 = roll("d100")
    passed = d100 <= current_san
    
    loss = 0
    if passed:
        loss = _parse_san_component(success_loss)
    else:
        loss = _parse_san_component(fail_loss)

    new_san = current_san - loss
    
    # CoC insanity rules
    max_san = 99  # default CoC max
    insanity_temp = False
    insanity_indef = False
    
    # 5+ SAN loss in one roll → temporary insanity
    if loss >= 5:
        insanity_temp = True
    # SAN reaches 0 → indefinite insanity
    if new_san <= 0:
        insanity_indef = True
        new_san = max(0, new_san)

    return {
        "san_before": current_san,
        "d100": d100,
        "pow_stat": pow_stat,
        "passed_pow_check": passed,
        "success_loss_component": success_loss,
        "fail_loss_component": fail_loss,
        "san_loss": loss,
        "san_after": new_san,
        "insanity_temp": insanity_temp,
        "insanity_indef": insanity_indef,
    }


def _parse_san_component(comp: str) -> int:
    """Parse a SAN loss component: "0", "1", "1d3", "1d5+1", "1d6" """
    comp = comp.strip()
    if not comp or comp == "0":
        return 0
    try:
        return int(comp)
    except ValueError:
        return roll(comp)


def coc_attribute_check(stat_value: int, multiplier: int = 5) -> dict:
    """CoC attribute check: roll d100 vs stat*multiplier (e.g. POW*5, DEX*5)."""
    target = stat_value * multiplier
    d100 = roll("d100")
    success = d100 <= target
    return {
        "target": target,
        "d100": d100,
        "success": success,
        "type": "attribute_check",
    }


def coc_luck_check(luck_stat: int) -> dict:
    """CoC Luck roll: d100 <= current Luck."""
    d100 = roll("d100")
    success = d100 <= luck_stat
    return {
        "target": luck_stat,
        "d100": d100,
        "success": success,
        "type": "luck_check",
    }


# ── CoC d100 技能判定 ─────────────────────────────────

def coc_skill_check(skill_value: int) -> dict:
    """CoC d100: roll d100 <= skill_value.
    Critical(01), Extreme(<=skill/5), Hard(<=skill/2),
    Normal(<=skill), Failure(>skill), Fumble(96-100 if skill<50, 100 if skill>=50).
    """
    d100 = random.randint(1, 100)

    extreme_threshold = max(1, skill_value // 5)
    hard_threshold = max(1, skill_value // 2)

    if d100 == 1:
        verdict = "critical_success"
        success = True
    elif d100 <= extreme_threshold:
        verdict = "extreme_success"
        success = True
    elif d100 <= hard_threshold:
        verdict = "hard_success"
        success = True
    elif d100 <= skill_value:
        verdict = "success"
        success = True
    elif (skill_value < 50 and d100 >= 96) or d100 == 100:
        verdict = "fumble"
        success = False
    else:
        verdict = "failure"
        success = False

    return {
        "skill_value": skill_value,
        "d100": d100,
        "success": success,
        "verdict": verdict,
        "extreme_threshold": extreme_threshold,
        "hard_threshold": hard_threshold,
    }


# ── 统一检定分发 ──────────────────────────────────────

def resolve_check(rule_system: str, skill_value: int, difficulty: int = 0) -> dict:
    """Dispatch to correct check based on rule_system.
    'coc' -> d100 <= skill_value; 'dnd' (default) -> d20 + skill vs DC.
    """
    if rule_system == "coc":
        result = coc_skill_check(skill_value)
    else:
        result = skill_check(skill_value, difficulty)
    result["rule_system"] = rule_system
    return result


# ── 测试 ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Dice Engine Test")
    print(f"d20: {roll('d20')}")
    print(f"2d6+3: {roll('2d6+3')}")
    print(f"Skill check (5, DC12): {skill_check(5, 12)}")
    print(f"SAN check (60, fail=1d6): {sanity_check(60, '1d6')}")
    print(f"CoC SAN '1/1d3' (san=60): {coc_san_loss('1/1d3', 60)}")
    print(f"CoC SAN '1d3/1d5+1' (san=30): {coc_san_loss('1d3/1d5+1', 30)}")
    print(f"POW*5 check (POW=50): {coc_attribute_check(50, 5)}")
    print(f"Luck check (luck=60): {coc_luck_check(60)}")
    print(f"CoC skill check (50): {coc_skill_check(50)}")
    print(f"resolve_check coc: {resolve_check('coc', 50)}")
    print(f"resolve_check dnd: {resolve_check('dnd', 5, 12)}")

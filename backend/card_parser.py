# -*- coding: utf-8 -*-
"""AIKP COC7 Character Card Parser.

The robust import path is NOT scraping the Excel template's hundreds of merged
cells — it's the `.st` dice-bot command that every COC card exports (and that
players already know):

    .st 力量60敏捷50意志55体质60外貌50教育70体型55智力65幸运50san55hp12mp11会计5侦查60图书馆使用70...

Format is alternating <name><number>. Names come in many spellings
(力量/STR, 理智/san/san值, 图书馆使用/图书馆) — we normalize via alias groups
and write every synonym to the same value, so whatever spelling the world book's
`check` field uses will resolve.
"""

from __future__ import annotations

import re


# ── Alias groups: every spelling in a group maps to one canonical value ──
# Listed lowercase; matching is case-insensitive and space-insensitive.
# Single-name skills (侦查, 聆听, 急救…) need no entry — kept verbatim.
_ALIAS_GROUPS: list[list[str]] = [
    # Core attributes (Chinese + English abbrev.)
    ["力量", "str"],
    ["敏捷", "dex"],
    ["意志", "pow"],
    ["体质", "con"],
    ["外貌", "app"],
    ["教育", "edu", "知识"],
    ["体型", "siz"],
    ["智力", "int", "灵感"],
    ["幸运", "luck", "运气"],
    # Derived / status
    ["理智", "san", "san值", "理智值", "sanity"],
    ["体力", "hp", "hitpoints"],
    ["魔法", "mp", "magic", "魔法值"],
    # Commonly aliased skills (from the card's own 别名 table)
    ["图书馆使用", "图书馆利用", "图书馆"],
    ["计算机使用", "计算机", "电脑"],
    ["汽车驾驶", "汽车"],
    ["信用评级", "信用", "信誉"],
    ["克苏鲁神话", "克苏鲁", "cm"],
    ["博物学", "自然学"],
    ["导航", "领航"],
    ["操作重型机械", "重型操作", "重型机械", "重型"],
    ["侦查", "侦察"],
    ["闪避", "dodge"],
    ["斗殴", "格斗"],
    ["话术", "快速交谈"],
    ["心理学"],
    ["聆听"],
    ["潜行"],
    ["急救"],
    ["说服"],
    ["恐吓"],
    ["取悦"],
    ["母语"],
]

# Build lookup: normalized spelling -> the group it belongs to
_SPELLING_TO_GROUP: dict[str, list[str]] = {}
for _grp in _ALIAS_GROUPS:
    for _name in _grp:
        _SPELLING_TO_GROUP[_name.lower()] = _grp


# Names that are attributes (used for *5 attribute checks downstream)
ATTRIBUTE_NAMES = {"力量", "敏捷", "意志", "体质", "外貌", "教育",
                   "体型", "智力", "幸运"}


# Matches <name><value>: name = run of CJK/letters, value = digits.
_PAIR_RE = re.compile(r'([一-鿿A-Za-z]+?)\s*(\d+)')


def parse_st_command(text: str) -> dict[str, int]:
    """Parse a COC `.st` command into a {skill_name: value} dict.

    Expands aliases: every spelling in a matched group is written to the same
    value. Unknown skill names are kept verbatim. Later occurrences of the same
    name win (matches dice-bot behavior).
    """
    if not text:
        return {}

    # Strip a leading ".st" / "st" command token if present.
    cleaned = text.strip()
    cleaned = re.sub(r'^[.。]?\s*st\b', '', cleaned, flags=re.IGNORECASE).strip()

    result: dict[str, int] = {}
    for raw_name, raw_val in _PAIR_RE.findall(cleaned):
        name = raw_name.strip()
        try:
            value = int(raw_val)
        except ValueError:
            continue
        group = _SPELLING_TO_GROUP.get(name.lower())
        if group:
            for syn in group:
                result[syn] = value
        else:
            result[name] = value
    return result


def looks_like_st_command(text: str) -> bool:
    """True if the message looks like a `.st` card-import command."""
    if not text:
        return False
    head = text.strip()[:6].lower()
    if not (head.startswith(".st") or head.startswith("。st") or head.startswith("st ")):
        return False
    # Require at least a few name+number pairs to avoid false positives.
    return len(_PAIR_RE.findall(text)) >= 5


def build_player_state_patch(stats: dict[str, int]) -> dict:
    """Turn parsed stats into a player_state patch.

    Returns {"skills": {...}, "san", "max_san", "hp", "max_hp", "mp", "max_mp",
             "attributes": {...}} — only keys that could be derived.
    Skills dict carries BOTH attributes and skills (CoC rolls both as d100<=val).
    """
    patch: dict = {"skills": dict(stats)}

    # Attributes snapshot (canonical Chinese names only)
    attrs = {a: stats[a] for a in ATTRIBUTE_NAMES if a in stats}
    if attrs:
        patch["attributes"] = attrs

    # SAN: explicit 理智 wins; else initial SAN = POW (意志)
    if "理智" in stats:
        patch["san"] = stats["理智"]
    elif "意志" in stats:
        patch["san"] = stats["意志"]
    if "san" in patch or "理智" in stats or "意志" in stats:
        patch["max_san"] = 99

    # HP: explicit 体力 wins; else (CON+SIZ)//10
    if "体力" in stats:
        patch["hp"] = stats["体力"]
        patch["max_hp"] = stats["体力"]
    elif "体质" in stats and "体型" in stats:
        hp = (stats["体质"] + stats["体型"]) // 10
        patch["hp"] = hp
        patch["max_hp"] = hp

    # MP: explicit 魔法 wins; else POW//5
    if "魔法" in stats:
        patch["mp"] = stats["魔法"]
        patch["max_mp"] = stats["魔法"]
    elif "意志" in stats:
        patch["mp"] = stats["意志"] // 5
        patch["max_mp"] = stats["意志"] // 5

    return patch


def apply_patch(player_state: dict, patch: dict) -> None:
    """Apply a parsed-card patch onto a session's player_state, in place.

    Skills are merged (card values override defaults). Derived stats
    (san/hp/mp + maxes) and the attribute snapshot are set when present.
    """
    skills = player_state.setdefault("skills", {})
    skills.update(patch.get("skills", {}))
    if patch.get("attributes"):
        player_state["attributes"] = patch["attributes"]
    for k in ("san", "max_san", "hp", "max_hp", "mp", "max_mp"):
        if k in patch:
            player_state[k] = patch[k]


# Canonical Chinese names of derived/status stats (not real skills)
_DERIVED_NAMES = {"理智", "体力", "魔法"}


def canonical_skills(skills: dict) -> dict[str, int]:
    """Collapse alias spellings to ONE canonical Chinese name per skill,
    dropping attributes and derived stats. For display — the raw skills dict
    keeps every alias so check-matching against the world book stays robust.
    """
    out: dict[str, int] = {}
    seen_groups: set[int] = set()
    for name, val in skills.items():
        if not val:
            continue
        group = _SPELLING_TO_GROUP.get(name.lower())
        if group:
            gid = id(group)
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            canon = next((g for g in group
                          if all('一' <= c <= '鿿' for c in g)), group[0])
            if canon in ATTRIBUTE_NAMES or canon in _DERIVED_NAMES:
                continue
            out[canon] = val
        elif all('一' <= c <= '鿿' for c in name):
            out[name] = val
    return out


def summarize_card(patch: dict) -> str:
    """Human-readable one-block summary for the import confirmation reply."""
    attrs = patch.get("attributes", {})
    order = ["力量", "敏捷", "意志", "体质", "外貌", "教育", "体型", "智力", "幸运"]
    attr_str = " ".join(f"{a}{attrs[a]}" for a in order if a in attrs)
    # Count distinct real skills (alias-collapsed)
    skill_names = canonical_skills(patch.get("skills", {}))
    lines = ["【角色卡已导入】"]
    if attr_str:
        lines.append("属性: " + attr_str)
    san = patch.get("san"); hp = patch.get("hp"); mp = patch.get("mp")
    status = []
    if hp is not None:
        status.append(f"HP {hp}/{patch.get('max_hp', hp)}")
    if san is not None:
        status.append(f"SAN {san}/{patch.get('max_san', 99)}")
    if mp is not None:
        status.append(f"MP {mp}/{patch.get('max_mp', mp)}")
    if status:
        lines.append(" · ".join(status))
    lines.append(f"技能 {len(skill_names)} 项已录入。")
    lines.append("发送「开始游戏」即可带这张卡开团。")
    return "\n".join(lines)


if __name__ == "__main__":
    sample = (".st 力量60敏捷50意志55体质60外貌50教育70体型55智力65幸运50"
              "san55hp12mp11会计5侦查60图书馆使用70聆听45说服40")
    parsed = parse_st_command(sample)
    print("parsed:", parsed)
    print("patch:", build_player_state_patch(parsed))

# -*- coding: utf-8 -*-
"""AIKP NPC trait extractor — runtime, dynamic.

The module's static entity data only has name/appearance/personality/dialogue.
But players refer to NPCs by what they're DOING in the narration ("读书的",
"啐人的", "敲背包的") — behavior that only exists in the KP's narrated text,
not in the static data. We extract those observable traits from narration with
one closed-world LLM call and store them on npc_states[name].dynamic.traits, so
reference resolution can match them. The npc system stays dynamically updated.

Closed-world: the model may only attach traits to NPCs in the supplied roster;
it cannot invent characters.
"""

from __future__ import annotations

import json


_SYSTEM = (
    "你是 TRPG 系统内部的「NPC 特征抽取」模块。你只为给定名单中的角色抽取"
    "『玩家一眼就能观察到、可能用来指代他们的外表/身份/标志行为』。"
    "绝不新增名单外的角色。绝不抽取隐藏身份、秘密、阴谋、真实来历等玩家无法一眼看出的设定。"
)


def extract_npc_traits(narrative: str, roster: list[dict],
                       api_key: str, base_url: str, model: str) -> dict:
    """Return {npc_name: [trait, ...]} extracted from `narrative`.

    roster: [{"name", "appearance", "personality"}]. On any failure returns {}.
    Traits are short 2-4 char descriptors (读书 / 啐人 / 高个子 / 敲背包).
    """
    if not narrative or not roster:
        return {}

    roster_str = "\n".join(
        f"- {n['name']}：外貌 {n.get('appearance','') or '未知'}；"
        f"性格 {n.get('personality','') or '未知'}"
        for n in roster
    )
    user = (
        f"下面是一段开场/场景叙事。请为名单中的每个角色，抽取他们在这段叙事里"
        f"【可观察的标志行为或外观特征】——也就是玩家事后可能拿来指代他们的说法，"
        f"例如「读书的」「啐人的」「敲背包的」「高个子」「在注射的」。\n"
        f"用台词、点名、动作、排除法把行为对应到正确的人。每人 2-4 个简短特征。\n\n"
        f"角色名单（只能用这些，禁止新增）：\n{roster_str}\n\n"
        f"叙事：\n{narrative}\n\n"
        f'只输出 JSON：{{"角色名": ["特征1","特征2"], ...}}。'
        f"特征用 2-4 字短词（如「读书」「啐人」「高个子」「银发」「女仆装」），不要整句。"
        f"【只抽玩家一眼能看出的外表/身份/行为；绝不抽隐藏身份、秘密、阴谋、真实来历"
        f"这类设定——例如「哈斯塔化身」「人偶」「邪教徒」「真实身份」一律不准抽。】"
        f"没有明显特征的角色可以省略或给空数组。"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        print(f"[TRAIT] extract failed: {e}", flush=True)
        return {}

    # Keep only roster names; coerce to short string lists.
    names = {n["name"] for n in roster}
    out: dict = {}
    for k, v in data.items():
        if k in names and isinstance(v, list):
            traits = [str(t).strip()[:8] for t in v if str(t).strip()]
            if traits:
                out[k] = traits[:5]
    return out

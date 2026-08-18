#!/usr/bin/env python3
"""Build source-grounded local worlds without using a configured API key.

This is deliberately conservative: a previously tested route graph supplies
semantic scene identity, while descriptions and explicit icon-marked entities
are rebound from the original PDF/DOCX. Unmatched structure is reported rather
than guessed into the playable map.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from parser import (  # noqa: E402
    ModuleParser, _extract_docx_text_text, _extract_opening,
    _extract_pdf_text_text, _recover_source_marked_entities,
    _scene_coverage_candidates, _source_contains_text, _split_text_segments,
    save_world_book,
)


FIXTURES = Path("/home/lonpyer/aikp_eval_data/manual")
DOWNLOADS = Path("/home/lonpyer/下载")

MODULES = (
    {
        "id": "reparsed_dawn",
        "source": DOWNLOADS / "黎明之盏 1.0.docx",
        "base": FIXTURES / "dawn_exhaustive_world.json",
        "supplement": FIXTURES / "dawn_goblet_world.json",
        "scene_remap": {
            "mansion_hall": "day1", "greenhouse": "greenhouse",
            "dream_banquet": "banquet",
        },
        "headings": {
            "intro": ["导入"],
            "day1": ["主线：委托", "【马车夫所知情报/书房可调查】",
                     "【四位受赠者的信息】", "【卡珊所知情报】", "HO3：验尸",
                     "【女仆长所知情报】", "HO1：亲人", "HO2：笔友",
                     "【缇亚所知情报】", "探索：公馆", "【珍妮所知情报】",
                     "【马修所知情报】"],
            "greenhouse": ["主线：温室"],
            "tia_dead": ["分支：缇亚死亡"],
            "tia_alive": ["分支：缇亚存活"],
            "dream1": ["HO1：海岸", "HO2：谜题", "HO3：安眠"],
            "day2": ["主线：园丁", "【莉娜所知情报】", "【女仆长所知情报】"],
            "town": ["探索：小镇", "【史蒂芬所知的情报】",
                     "【和工会代表对话的内容如下】：", "【和卡德对话的内容如下】："],
            "market": ["探索：集市", "【马特所知情报】"],
            "dream2": ["HO1：汇报", "HO2：沙盘", "HO3：睡前故事",
                       "HO1：湖畔", "HO2：迷宫", "HO3：埋葬"],
            "ridge": ["探索：圣岭", "【蒂凡尼所知情报】"],
            "banquet": ["主线：晚宴（幻）"],
            "dream_mansion": [
                "KP信息：PC们此时已进入幻梦境，接下来看到的都是梦中的场景。幻梦境中的黎明公馆部分可探索场景有所变化，详情见自由探索。",
                "【尼科特所知的情报】", "探索：公馆（幻）"],
            "knight": ["主线：骑士（幻）"],
            "day4": ["主线：香槟", "【卡德所知情报】"],
            "cellar": ["主线：香槟"],
            "nodes": ["探索：节点", "【蒂凡尼所知情报】", "【雨果所知情报】"],
            "dream4": ["HO1：长河", "HO2：秘密", "HO3：纪念"],
            "day5": ["主线：演讲"],
            "speech": ["主线：演讲"],
            "windmill": ["分支：风车"],
            "rescue": ["分支：救疗"],
            "discuss": ["分支：商讨"],
            "despair": ["分支：绝境"],
            "finale": ["主线：决战"],
            "rivercross": ["主线：决战"],
            "end1": ["结局"], "end2": ["结局"],
            "end3": ["结局"], "end4": ["结局"],
        },
    },
    {
        "id": "reparsed_mountain",
        "source": DOWNLOADS / "为何不可攀登此山（译）.docx",
        "base": FIXTURES / "mountain_exhaustive_world.json",
        "supplement": FIXTURES / "climb_mountain_world.json",
        "scene_remap": {
            "starting_cabin": "cabin", "mountain_path": "event1",
            "summit": "summit",
        },
        "headings": {
            "cabin": ["■开始", "■对话"],
            "event1": ["【事件① 诀别】"], "event2": ["【事件② 影像】"],
            "event3": ["【事件③ 遭遇】"], "event4": ["【事件④ 山山山】"],
            "fork": ["【事件④ 山山山】"], "water": ["【事件④ 山山山】"],
            "rest_cabin": ["【事件④ 山山山】"],
            "event6": ["【事件⑥家人】"], "event7": ["【事件⑦遗体】"],
            "event8": ["【事件⑧继续攀登】"],
            "event9": ["【事件⑨山中小屋】", "◆与女子的对话"],
            "event10": ["【事件⑩山是存在的】"],
            "event11": ["【事件⑪黄金】"], "event12": ["【事件⑫疑问】"],
            "summit": ["【高潮阶段　登顶】"],
            "end_a": ["END:A"], "end_a_lost": ["END:A lost"],
            "end_b": ["END:B"], "descent": ["【事件④ 山山山】"],
        },
    },
    {
        "id": "reparsed_honeybucket",
        "source": DOWNLOADS / "绝境行者-Honeybucket：奶与蜜的黄金乡.pdf",
        "base": FIXTURES / "honey_exhaustive_world.json",
        "supplement": FIXTURES / "honeybucket_world.json",
        "scene_remap": {"toilet_interior": "toilet"},
        "headings": {
            "toilet": ["■调查员的导入", "■蜂蜜桶 HoneyBucket", "◆“蜂蜜桶”内部：",
                       "■齿轮开始转动", "◆随着轮次的增加……", "■直☆面☆粪☆怪",
                       "【胜天半纸】", "【沧海涓滴】", "【去你的吧】",
                       "【屎里逃生】", "【谁入地狱】"],
        },
    },
)


def _read_source(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return _extract_docx_text_text(str(path))
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text_text(str(path))
    return path.read_text(encoding="utf-8", errors="replace")


def _remap_entity(entity: dict, scene_remap: dict[str, str]) -> dict:
    result = copy.deepcopy(entity)
    result["scene"] = scene_remap.get(result.get("scene", ""), result.get("scene", ""))
    if result.get("all_scenes"):
        result["all_scenes"] = list(dict.fromkeys(
            scene_remap.get(scene_id, scene_id)
            for scene_id in result["all_scenes"]
        ))
    return result


def _apply_heading_bindings(scene_list: list[dict], source: str,
                            heading_map: dict[str, list[str]]) -> None:
    by_title = {}
    for segment in _split_text_segments(source):
        by_title.setdefault(segment["title"], []).append(segment)
    by_id = {scene["id"]: scene for scene in scene_list}
    for scene_id, headings in heading_map.items():
        scene = by_id.get(scene_id)
        if not scene:
            continue
        selected = [
            segment
            for heading in headings
            for segment in by_title.get(heading, [])
        ]
        if not selected:
            continue
        scene["source_text"] = "\n\n".join(
            segment["text"] for segment in selected)
        scene["source_sections"] = [segment["text"] for segment in selected]
        scene["source_starts"] = [segment["start"] for segment in selected]
        scene["source_start"] = selected[0]["start"]
        scene["source_binding"] = "audited_heading_map"


def reparse_module(spec: dict) -> dict:
    source_path = spec["source"]
    source = _read_source(source_path)
    world = json.loads(spec["base"].read_text(encoding="utf-8"))
    supplement = json.loads(spec["supplement"].read_text(encoding="utf-8"))
    world["name"] = spec["id"]
    world["version"] = "source-reparse-1"

    scene_list = []
    for scene_id, scene in world.get("scenes", {}).items():
        payload = copy.deepcopy(scene)
        payload["id"] = scene_id
        # Placeholder fixture text must never count as source provenance.
        payload.pop("source_text", None)
        scene_list.append(payload)

    pass1 = {
        "scenes": scene_list,
        "npcs": [],
        "items": [],
        "clues": [],
        "events": [],
    }
    for entity_id, entity in supplement.get("entities", {}).items():
        payload = _remap_entity(entity, spec["scene_remap"])
        payload["id"] = entity_id
        collection = "npcs" if payload.get("type") == "npc" else "items"
        pass1[collection].append(payload)

    # Call only the code-based source binder, bypassing ModuleParser.__init__
    # and therefore creating no OpenAI client or network request.
    parser = ModuleParser.__new__(ModuleParser)
    parser.pass1_7_bind_source_text([pass1], source)
    _apply_heading_bindings(pass1["scenes"], source, spec.get("headings", {}))
    recovered = _recover_source_marked_entities(pass1, source)

    rebound = {}
    for scene in pass1["scenes"]:
        scene_id = scene.pop("id")
        rebound[scene_id] = scene
    world["scenes"] = rebound

    entities = {}
    for entity in pass1["npcs"] + pass1["items"]:
        entity_id = entity.pop("id")
        if entity.get("scene") in rebound:
            entities[entity_id] = entity
    world["entities"] = entities

    opening = world.get("opening", "")
    if not _source_contains_text(source, opening):
        grounded_opening = _extract_opening(source)
        if grounded_opening:
            world["opening"] = grounded_opening
    bound_ids = []
    for scene_id, scene in rebound.items():
        sections = scene.get("source_sections", [])
        if sections:
            verified = all(_source_contains_text(source, part) for part in sections)
        else:
            verified = bool(scene.get("source_text") and _source_contains_text(
                source, scene["source_text"]))
        scene["source_verified"] = verified
        if verified:
            bound_ids.append(scene_id)
    candidates = _scene_coverage_candidates(
        source,
        [dict(scene, id=scene_id) for scene_id, scene in rebound.items()],
    )
    world["_source_reparse"] = {
        "source_file": source_path.name,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_chars": len(source),
        "bound_scene_ids": bound_ids,
        "unbound_scene_ids": sorted(set(rebound) - set(bound_ids)),
        "recovered_marked_entities": recovered,
        "unclassified_source_candidates": [
            {"start": item["start"], "name": item["name"]}
            for item in candidates
        ],
        "network_model_used": False,
    }
    path = save_world_book(world["name"], world)
    return {
        "world": world["name"],
        "path": path,
        "scenes": len(rebound),
        "bound_scenes": len(bound_ids),
        "entities": len(entities),
        "recovered": recovered,
        "unclassified_candidates": len(candidates),
    }


def main() -> int:
    reports = [reparse_module(spec) for spec in MODULES]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

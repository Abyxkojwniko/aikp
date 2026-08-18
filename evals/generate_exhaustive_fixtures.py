#!/usr/bin/env python3
"""Generate exhaustive no-key route fixtures for the three local modules."""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path("/home/lonpyer/aikp_eval_data/manual")


def write_json(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_route(stem: str, module: str, world_fixture: str,
                steps: list[dict]) -> None:
    turns = []
    responses = []
    for turn_number, step in enumerate(steps, start=1):
        response = step["response"]
        target = step.get("target")
        if target:
            response += f"\n\n〔前往：{target}〕"
        expect = {
            "turn": turn_number,
            "response_contains": step.get("contains", []),
        }
        if step.get("scene"):
            expect["scene"] = step["scene"]
        if target:
            expect["scene"] = target
        expect.update(step.get("expect", {}))
        turn = {
            "input": step["input"],
            "intent": step.get("intent", ""),
            "covers": step.get("covers", []),
            "expect": expect,
        }
        if step.get("roll_verdict"):
            turn["roll_verdict"] = step["roll_verdict"]
        for field in ("select_npc", "select_object", "select_scene",
                      "set_entity_states"):
            if field in step:
                turn[field] = step[field]
        turns.append(turn)
        if not step.get("code_response"):
            responses.append(response)
    write_json(f"{stem}_case.json", {
        "name": stem.replace("_", "-"),
        "module": module,
        "world_fixture": world_fixture,
        "turns": turns,
    })
    write_json(f"{stem}_responses.json", {"responses": responses})


def movement_step(player_input: str, response: str, target: str,
                  covers: list[str], contains: list[str]) -> dict:
    return {
        "input": player_input,
        "response": response,
        "target": target,
        "covers": covers,
        "contains": contains,
    }


def generate_mountain() -> None:
    order = [
        "cabin", "event1", "event2", "event3", "event4", "fork", "water",
        "rest_cabin", "event6", "event7", "event8", "event9", "event10",
        "event11", "event12", "summit",
    ]
    names = {
        "cabin": "最初的山中小屋", "event1": "诀别", "event2": "影像",
        "event3": "遭遇", "event4": "山山山", "fork": "岔路",
        "water": "山中水源", "rest_cabin": "破败小屋", "event6": "家人",
        "event7": "遗体", "event8": "继续攀登", "event9": "山顶前小屋",
        "event10": "山是存在的", "event11": "黄金", "event12": "疑问",
        "summit": "山顶神像", "end_a": "END A", "end_b": "END B",
        "end_a_lost": "END A LOST", "descent": "中途下山",
    }
    scenes = {}
    for index, scene_id in enumerate(order):
        exits = {}
        if index + 1 < len(order):
            exits[f"继续前往{names[order[index + 1]]}"] = order[index + 1]
        scenes[scene_id] = {
            "name": names[scene_id],
            "desc": f"《为何不可攀登此山》的{names[scene_id]}阶段。",
            "source_text": f"原文顺序事件：{names[scene_id]}。",
            "exits": exits,
        }
    scenes["cabin"]["exits"]["中途下山"] = "descent"
    scenes["summit"]["exits"] = {
        "回答是山": "end_a", "回答不是山": "end_b",
        "相信山并陷入幻觉": "end_a_lost",
    }
    for ending in ("end_a", "end_b", "end_a_lost", "descent"):
        scenes[ending] = {
            "name": names[ending], "desc": names[ending],
            "source_text": names[ending], "exits": {},
        }
    write_json("mountain_exhaustive_world.json", {
        "name": "eval_mountain_exhaustive", "rule_system": "coc",
        "starting_scene": "cabin",
        "opening": "调查员与山登隙造、尾金星杉、四间管从山中小屋开始攀登。",
        "scenes": scenes, "entities": {},
    })
    write_json("mountain_summit_world.json", {
        "name": "eval_mountain_summit", "rule_system": "coc",
        "starting_scene": "summit",
        "opening": "调查员抵达山顶，头颅神像正在等待最终回答。",
        "scenes": {
            "summit": scenes["summit"],
            "end_a": scenes["end_a"],
            "end_b": scenes["end_b"],
            "end_a_lost": scenes["end_a_lost"],
        },
        "entities": {},
    })

    event_responses = {
        "event1": "方格旗旁躺着一具白骨，笔记记录十人队伍如何分裂和迷失。",
        "event2": "摄像机中的年轻人在登山影像末尾尖叫着坠落，画面随即消失。",
        "event3": "一群面容完全相同的登山者整齐走来，又对你们视而不见地下山。",
        "event4": "扭曲的登山者反复喊着这座山疯了，四肢反折后倒地死亡。",
        "fork": "落石过后道路分岔，导航结果决定你们是否看见来路崩塌的幻觉。",
        "water": "山中水源在仔细观察下显出血液本相；误认者会把它直接饮下。",
        "rest_cabin": "破败小屋可以生火、找到红茶和日记，但失败的幸运检定会揭示小屋根本不存在。",
        "event6": "老妇人以妻子美智子引诱四间管走向悬崖，阻挡的力量让任何人都无法救下他。",
        "event7": "风中倒着一具与调查员完全相同的遗体，一个无形声音说等候已久。",
        "event8": "碎石实为骸骨，树根缠住山登；雪地与求救女子继续侵蚀众人的现实感。",
        "event9": "临近山顶的小屋里尾金和围裙女子已经等候，书中写着信与不信者皆将得救。",
        "event10": "山登念着山是存在的走入悬崖上方，踏着不存在的道路升空，再也没有回来。",
        "event11": "尾金把白骨当成黄金，留在坍塌洞穴中；入口封死后所有人都知道他已经死亡。",
        "event12": "连续的灵感让你怀疑山是幻觉；一名坠落者留下这里是梦境的最后一句话。",
        "summit": "山顶小屋中的头颅神像直接在脑中发问：你认为这座山真的是山吗？",
    }

    def full_route(stem: str, fail: bool, ending: str) -> None:
        steps = [{
            "input": "开始游戏", "response": "调查员在山中小屋与三名同行者会合。",
            "scene": "cabin", "contains": ["山中小屋"],
            "covers": [f"mountain_seg_{i:02d}" for i in range(6)],
        }]
        next_ids = order[1:]
        for scene_id in next_ids:
            covers = []
            segment_map = {
                "event1": 6, "event2": 7, "event3": 8, "event4": 9,
                "event6": 10, "event7": 11, "event8": 12, "event9": 13,
                "event10": 15, "event11": 16, "event12": 17, "summit": 18,
            }
            if scene_id in segment_map:
                covers.append(f"mountain_seg_{segment_map[scene_id]:02d}")
            branch_covers = {
                "fork": ["rockfall_fail" if fail else "rockfall_pass",
                         "navigation_fail" if fail else "navigation_pass"],
                "water": ["water_spot_fail" if fail else "water_spot_pass"],
                "rest_cabin": ["cabin_luck_fail" if fail else "cabin_luck_pass"],
                "event8": ["scree_fail" if fail else "scree_pass",
                           "snow_nav_fail" if fail else "snow_nav_pass",
                           "woman_pow_fail" if fail else "woman_pow_pass"],
                "event9": ["late_cabin_search_fail" if fail else "late_cabin_search_pass",
                           "mountain_seg_14"],
                "event10": ["answer_disbelieve" if fail else "answer_believe"],
                "event11": ["sand_spot_fail" if fail else "sand_spot_pass"],
                "event12": ["insight_fail" if fail else "insight_pass",
                            "listen_fail" if fail else "listen_pass"],
            }
            covers.extend(branch_covers.get(scene_id, []))
            response = event_responses[scene_id]
            branch_responses = {
                "fork": (
                    "落石令你失去意识；醒来后错误导航让你看见来路已经崩塌，恐惧继续加深。"
                    if fail else
                    "你躲过落石，并以正确导航确认来路仍然存在，队伍继续向上。"
                ),
                "water": (
                    "你没看出水源的异常，把实际的血液饮了下去。"
                    if fail else
                    "你察觉所谓水源实际是血液，阻止队伍饮用。"
                ),
                "rest_cabin": (
                    "休息后你突然发现红茶、日记和整座小屋都从未存在。"
                    if fail else
                    "你在真实的破败小屋中找到红茶和日记，短暂休整。"
                ),
                "event8": (
                    "你把骸骨认作碎石，在雪地迷路，并相信求救女子的幻觉。"
                    if fail else
                    "你认出碎石是骸骨，保持正确方向，也抵抗了求救女子的幻觉。"
                ),
                "event9": (
                    "你没有搜到小屋里的关键记录，只能在不完整信息下继续。"
                    if fail else
                    "你搜出书中记录，得知信与不信者都将以不同方式获救。"
                ),
                "event10": (
                    "你不相信山登脚下存在道路，只能看着他走入空中消失。"
                    if fail else
                    "你一度相信山登脚下存在道路，目送他沿空中的山路离开。"
                ),
                "event11": (
                    "你没能看破黄金幻觉，直到洞穴坍塌才确认尾金已经死亡。"
                    if fail else
                    "你认出所谓黄金全是白骨，却仍来不及阻止洞穴坍塌夺走尾金。"
                ),
                "event12": (
                    "灵感与聆听都失败，你无法确认山的本质，也没听清坠落者的遗言。"
                    if fail else
                    "灵感与聆听都成功：你怀疑山是幻觉，并听清坠落者说这里是梦境。"
                ),
            }
            response = branch_responses.get(scene_id, response)
            steps.append(movement_step(
                f"继续推进到{names[scene_id]}", response, scene_id,
                covers, ["神像" if scene_id == "summit" else response[:4]],
            ))
        if ending == "end_a":
            response = "你回答这里是山。神像认可信念，山从视野中消失，而你永远困在梦境中。"
            covers = ["mountain_seg_19", "end_a"]
        else:
            response = "你回答这里不是山。意识断裂后你在最初小屋醒来，三名同行者都已停止呼吸。"
            covers = ["mountain_seg_21", "end_b"]
        steps.append(movement_step(
            "我回答神像的问题", response, ending, covers,
            ["梦境" if ending == "end_a" else "醒来"],
        ))
        write_route(stem, "mountain", "mountain_exhaustive", steps)

    full_route("mountain_full_route_a", False, "end_a")
    full_route("mountain_full_route_b", True, "end_b")
    write_route("mountain_branch_lost", "mountain", "mountain_summit", [
        {"input": "开始游戏", "response": "调查员抵达山顶，头颅神像等待回答。", "scene": "summit",
         "contains": ["山顶"], "covers": []},
        movement_step("我的理智已降到最大值五分之一，并回答这里是山。",
                      "山体从感知中消失，持续坠落的幻觉将你永久吞没。",
                      "end_a_lost", ["mountain_seg_20", "end_a_lost"], ["永久"]),
    ])
    write_route("mountain_branch_descend", "mountain", "mountain_exhaustive", [
        {"input": "开始游戏", "response": "调查员从山中小屋开始。", "scene": "cabin",
         "contains": ["小屋"], "covers": []},
        movement_step("我决定中途下山。",
                      "疯狂幻觉覆盖下山路；理智未归零的调查员最终活着离开。",
                      "descent", ["descend_survive"], ["活着离开"]),
    ])
    write_route("mountain_branch_descend_san_zero", "mountain", "mountain_exhaustive", [
        {"input": "开始游戏", "response": "调查员从山中小屋开始。", "scene": "cabin",
         "contains": ["小屋"], "covers": []},
        movement_step("我的理智已经归零，仍决定中途下山。",
                      "疯狂幻觉彻底吞没意识；理智归零的调查员没能活着走出山区。",
                      "descent", ["descend_san_zero"], ["没能活着"]),
    ])
    required = [f"mountain_seg_{i:02d}" for i in range(22)] + [
        "rockfall_pass", "rockfall_fail", "navigation_pass", "navigation_fail",
        "water_spot_pass", "water_spot_fail", "cabin_luck_pass", "cabin_luck_fail",
        "scree_pass", "scree_fail", "snow_nav_pass", "snow_nav_fail",
        "woman_pow_pass", "woman_pow_fail", "late_cabin_search_pass",
        "late_cabin_search_fail", "answer_believe", "answer_disbelieve",
        "sand_spot_pass", "sand_spot_fail", "insight_pass", "insight_fail",
        "listen_pass", "listen_fail", "end_a", "end_b", "end_a_lost",
        "descend_survive", "descend_san_zero",
    ]
    write_json("mountain_coverage.json", {"module": "mountain", "required": required})


def honey_world() -> dict:
    actions = [
        ("stand", ["站起", "起身"], 0), ("door", ["开门", "门把"], 1),
        ("flush", ["冲水", "冲水按钮"], 2), ("fan", ["排气扇"], 1),
        ("rack", ["读物架", "报纸"], 2), ("shake", ["晃动厕所"], 3),
        ("device", ["手机", "手表"], 0),
        ("contact", ["打电话", "发送信息"], 1),
    ]
    return {
        "name": "eval_honey_exhaustive", "rule_system": "coc",
        "starting_scene": "toilet", "opening": "调查员选择一座蜂蜜桶并在内部恢复意识。",
        "scenes": {"toilet": {
            "name": "蜂蜜桶内部", "desc": "门板、排气扇、读物架、便池与冲水按钮都在这里。",
            "source_text": "发展轮次按原文行动成本推进，4、8、12触发事件。", "exits": {},
        }}, "entities": {},
        "action_clocks": {"development_round": {
            "name": "发展轮次", "initial": 0, "default_increment": 0,
            "actions": [{"label": key, "triggers": triggers, "increment": increment}
                        for key, triggers, increment in actions],
            "outcome_actions": [{
                "label": "暴力行为",
                "triggers": ["暴力", "攻击", "破坏", "砸", "踹", "撞", "撬", "射击", "开枪"],
                "outcome_increments": {
                    "critical_failure": 3, "fumble": 3, "failure": 0,
                    "success": 1, "hard_success": 1,
                    "extreme_success": 2, "critical_success": 3,
                },
            }],
            "milestones": [
                {"at": 4, "flag": "honey_4", "narration": "恶臭增强并带来饥饿错觉。"},
                {"at": 8, "flag": "honey_8", "narration": "墙壁爬满脏污手印。"},
                {"at": 12, "flag": "honey_12", "narration": "秽物怪物从便池中涌出。"},
            ],
        }},
    }


def generate_honey() -> None:
    write_json("honey_exhaustive_world.json", honey_world())

    def start(entrance: str, extra: list[str] | None = None) -> dict:
        return {
            "input": "开始游戏", "response": f"你选择{entrance}蜂蜜桶，随后在内部恢复意识。",
            "scene": "toilet", "contains": [entrance],
            "covers": ["honey_seg_00", "honey_seg_01", "honey_seg_02",
                       "honey_seg_03", "honey_seg_04", "honey_seg_05",
                       *(extra or [])],
        }

    write_route("honey_route_early_escape", "honey", "honey_exhaustive", [
        start("2号", ["entrance_2", "bird_hole"]),
        {"input": "我查看排气扇。", "response": "排气扇仍转动，外面却只有黑暗，也没有气流。",
         "contains": ["黑暗"], "covers": ["action_fan", "honey_seg_06"]},
        {"input": "我给索菲亚发送信息。", "response": "索菲亚只回复：注意卫生。",
         "contains": ["注意卫生"], "covers": ["contact_sophia"]},
        {"input": "我把一只手伸进便池并按下冲水按钮。",
         "response": "身体与灵魂被下水口拉长，最终被厕所吐回午夜地面。",
         "contains": ["午夜"], "covers": ["ending_escape", "honey_seg_13"]},
    ])

    def monster_setup(entrance: str, entrance_tag: str) -> list[dict]:
        return [
            start(entrance, [entrance_tag]),
            {"input": "我晃动厕所。", "response": "厕所剧烈摇晃。", "contains": ["摇晃"],
             "covers": ["action_shake"]},
            {"input": "我查看读物架上的报纸。", "response": "报纸记录了三名失踪者的通讯反馈。",
             "contains": ["失踪者"], "covers": ["action_rack", "contact_emily", "contact_james"]},
            {"input": "我再晃动厕所。", "response": "厕所继续摇晃。", "contains": ["摇晃"],
             "covers": ["clock_4"]},
            {"input": "我压下门把开门。", "response": "门依旧打不开。", "contains": ["打不开"],
             "covers": ["action_door"]},
            {"input": "我按下冲水按钮。", "response": "污水反向涌起，蛆虫在其中浮动。",
             "contains": ["蛆虫"], "covers": ["action_flush", "clock_8"]},
            {"input": "我再次晃动厕所。", "response": "异样存在越来越接近。",
             "contains": ["接近"], "covers": ["clock_12", "honey_seg_07", "honey_seg_08"]},
        ]

    routes = {
        "honey_route_combat_win": (
            "1号", "entrance_1", "我与怪物战斗并获胜。",
            "你战胜怪物，漩涡将残躯吸回便池，黎明时终于重获自由。",
            ["ending_combat_win", "honey_seg_09", "honey_seg_10"]),
        "honey_route_combat_loss": (
            "3号", "entrance_3", "我与怪物战斗但落败。",
            "秽物将你淹没并拖入黑色漩涡，你的精神被污秽同化。",
            ["ending_filth", "honey_seg_11", "combat_loss"]),
        "honey_route_button_escape": (
            "1号", "entrance_1", "我未被吞噬，立刻按下冲水按钮。",
            "怪物被吸回便池；你在正常厕所中醒来，手还按着按钮。",
            ["ending_flush", "honey_seg_12"]),
        "honey_route_sacrifice": (
            "2号", "entrance_2", "我已被吞噬，抱着同归于尽的想法按下按钮。",
            "你与怪物一同坠入幽暗深处，以精神束缚它不再重现人间。",
            ["ending_sacrifice", "honey_seg_14"]),
        "honey_route_fire": (
            "3号", "entrance_3", "我点燃明火攻击怪物。",
            "沼气爆燃席卷厕所；污染仍将你拖入黑色漩涡。",
            ["fire_explosion", "achievement_childhood", "ending_filth"]),
    }
    for stem, (entrance, entrance_tag, final_input, response, covers) in routes.items():
        steps = monster_setup(entrance, entrance_tag)
        steps.append({"input": final_input, "response": response,
                      "contains": [response[:4]], "covers": covers})
        write_route(stem, "honey", "honey_exhaustive", steps)

    # Explicitly exercise zero-cost actions and every authored violence cost.
    write_route("honey_route_action_costs", "honey", "honey_exhaustive", [
        start("2号", ["entrance_2"]),
        {"input": "我站起。", "response": "你轻松站起。", "contains": ["站起"],
         "covers": ["action_stand", "cost_zero"]},
        {"input": "我查看手机时间。", "response": "设备显示当前发展轮次。", "contains": ["发展轮次"],
         "covers": ["action_device"]},
        {"input": "我用力量攻击墙板。", "response": "请进行力量检定。〔检定：力量〕",
         "contains": ["力量检定"], "covers": ["violence_fail_cost_0"],
         "roll_verdict": "failure", "expect": {"clocks": {"development_round": 0}}},
        {"input": "我再次攻击墙板。", "response": "请进行力量检定。〔检定：力量〕",
         "contains": ["力量检定"], "covers": ["violence_success_cost_1"],
         "roll_verdict": "success", "expect": {"clocks": {"development_round": 1}}},
        {"input": "我继续砸墙板。", "response": "请进行力量检定。〔检定：力量〕",
         "contains": ["力量检定"], "covers": ["violence_hard_cost_1"],
         "roll_verdict": "hard_success", "expect": {"clocks": {"development_round": 2}}},
        {"input": "我猛烈撞击墙板。", "response": "请进行力量检定。〔检定：力量〕",
         "contains": ["力量检定"], "covers": ["violence_extreme_cost_2"],
         "roll_verdict": "extreme_success", "expect": {"clocks": {"development_round": 4}}},
        {"input": "我用工具破坏墙板。", "response": "请进行力量检定。〔检定：力量〕",
         "contains": ["力量检定"], "covers": ["violence_critical_cost_3"],
         "roll_verdict": "critical_success", "expect": {"clocks": {"development_round": 7}}},
        {"input": "我踹击墙板。", "response": "请进行力量检定。〔检定：力量〕",
         "contains": ["力量检定"], "covers": ["violence_fumble_cost_3"],
         "roll_verdict": "fumble", "expect": {"clocks": {"development_round": 10}}},
    ])
    required = [f"honey_seg_{i:02d}" for i in range(15)] + [
        "entrance_1", "entrance_2", "entrance_3", "bird_hole",
        "action_stand", "action_door", "action_flush", "action_fan",
        "action_rack", "action_shake", "action_device", "contact_emily",
        "contact_james", "contact_sophia", "clock_4", "clock_8", "clock_12",
        "violence_fumble_cost_3", "violence_fail_cost_0",
        "violence_success_cost_1", "violence_hard_cost_1", "violence_extreme_cost_2",
        "violence_critical_cost_3", "fire_explosion", "achievement_childhood",
        "ending_combat_win", "ending_filth", "ending_flush", "ending_escape",
        "ending_sacrifice",
    ]
    write_json("honey_coverage.json", {"module": "honey", "required": required})


def generate_dawn() -> None:
    scene_names = {
        "intro": "导入", "day1": "第一日委托与公馆", "greenhouse": "温室",
        "tia_dead": "缇亚死亡分支", "tia_alive": "缇亚存活分支",
        "dream1": "第一日梦境", "day2": "第二日园丁", "town": "小镇探索",
        "market": "集市探索", "dream2": "第二日梦境", "ridge": "第三日圣岭",
        "banquet": "幻之晚宴", "dream_mansion": "幻之公馆",
        "knight": "幻之骑士", "day4": "第四日香槟", "cellar": "香槟酒窖",
        "nodes": "节点探索", "dream4": "第四日梦境", "day5": "第五日",
        "speech": "集市演讲", "windmill": "风车分支", "rescue": "救疗分支",
        "discuss": "商讨分支", "despair": "绝境分支", "finale": "无名湖决战",
        "rivercross": "河曲渡口", "end1": "END 1", "end2": "END 2",
        "end3": "END 3", "end4": "END 4",
    }
    graph = {
        "intro": ["day1"], "day1": ["greenhouse"],
        "greenhouse": ["tia_dead", "tia_alive"],
        "tia_dead": ["dream1"], "tia_alive": ["dream1"],
        "dream1": ["day2"], "day2": ["town"], "town": ["market"],
        "market": ["dream2"], "dream2": ["ridge"], "ridge": ["banquet"],
        "banquet": ["dream_mansion"], "dream_mansion": ["knight"],
        "knight": ["day4"], "day4": ["cellar"], "cellar": ["nodes"],
        "nodes": ["dream4"], "dream4": ["day5"], "day5": ["speech"],
        "speech": ["windmill", "rescue"], "windmill": ["discuss", "despair"],
        "rescue": ["discuss", "despair"], "discuss": ["finale"],
        "despair": ["finale"],
        "finale": ["end1", "end2", "end3", "rivercross"],
        "rivercross": ["end4", "end1"],
    }
    scenes = {}
    for scene_id, name in scene_names.items():
        scenes[scene_id] = {
            "name": name, "desc": f"《黎明之盏》的{name}阶段。",
            "source_text": f"原文章节：{name}。",
            "exits": {scene_names[target]: target for target in graph.get(scene_id, [])},
        }
    write_json("dawn_exhaustive_world.json", {
        "name": "eval_dawn_exhaustive", "rule_system": "coc",
        "starting_scene": "intro", "opening": "三名调查员分别以客人、友人和送葬人的身份抵达维丽。",
        "scenes": scenes, "entities": {},
    })

    responses = {
        "day1": "委托、验尸、亲人和笔友三条个人线索在公馆汇合，调查员完成书房与佣人情报调查。",
        "greenhouse": "温室中缇亚作画，持剑的雨果幻影正在接近；是否及时阻止将决定她的生死分支。",
        "tia_dead": "缇亚被腰斩后尸体消失，随后又与夫人一同出现，死亡与复活的矛盾无法被演员说破。",
        "tia_alive": "调查员击退雨果幻影保住缇亚，女仆和宪兵却只看见调查员对着空气战斗。",
        "dream1": "海岸、谜题和安眠三个梦境分别推进HO1、HO2、HO3的转化与记忆。",
        "day2": "第二日从园丁莉娜开始，怀表与公馆的新情报继续推动调查。",
        "town": "调查员走访小镇、工厂与居民，史蒂芬和工会信息揭示集体行动的异常。",
        "market": "集市、肉铺和马特的情报把失踪案、黄金与仪式联系起来。",
        "dream2": "汇报、沙盘、睡前故事以及湖畔、迷宫、埋葬梦境分别推进三条HO。",
        "ridge": "第三日踏遍圣岭、墓园和各HO支线，接触佩乐林、蒂凡尼及幻梦境入口。",
        "banquet": "幻之晚宴让调查员经历乔装、交涉、力量、敏捷、幸运与随机事件。",
        "dream_mansion": "幻之公馆以火焰、窒息等死亡幻觉揭示演员被舞台操纵。",
        "knight": "雨果与尼科特完成骑士对决，尼科特被长剑贯穿，死亡仍不是演员的终点。",
        "day4": "第四日香槟案揭开八具工人尸体与人体炼金，怀表获得进入镜面世界的方法。",
        "cellar": "酒窖的十桶香槟、八具尸体和拖拽痕迹指向团伙作案及黄金节点。",
        "nodes": "调查员检查河曲、黄金乡、墓园、香槟集市和铸金厂等节点，区分真节点与误导。",
        "dream4": "长河、秘密和纪念三个梦境完成第四日的HO推进。",
        "day5": "第五日伯爵演讲传闻引导调查员前往游行现场。",
        "speech": "人群试图指定HO3扮演伯爵，追踪黄衣工人后发现黄金球像阵眼。",
        "windmill": "阵眼破坏失败，调查员被绑入仓库并由雨果带往黄金乡。",
        "rescue": "阵眼破坏成功，反噬令游行人群倒下，伤者在莉娜与施密特处接受救疗。",
        "discuss": "持有黎明之盏时，古代缇亚、尼科特与雨果商讨进入仪式并跳入湖中的计划。",
        "despair": "未取得黎明之盏时，舞台没有完整退路，三条HO只能面对各自有限的脱离方案。",
        "finale": "无名湖V形法阵启动，佩乐林将雕花小刀交给HO3，仪式进入最终选择。",
    }

    segment_tags = {
        "intro": list(range(0, 9)), "day1": list(range(9, 24)),
        "greenhouse": [24], "tia_dead": [25], "tia_alive": [26],
        "dream1": [27, 28, 29], "day2": [30, 31, 32, 33],
        "town": [34, 35, 36, 37], "market": [38, 39],
        "dream2": [40, 41, 42, 43, 44, 45], "ridge": [46, 47, 48],
        "banquet": [49, 50], "dream_mansion": [51], "knight": [52],
        "day4": [53, 54, 55], "cellar": [], "nodes": [56, 57, 58],
        "dream4": [59, 60, 61], "day5": [62], "speech": [63],
        "windmill": [64], "rescue": [65], "discuss": [66], "despair": [67],
        "finale": [68, 69],
    }

    def full_dawn_route(stem: str, tia: str, battle: str,
                        planning: str, ending: str) -> None:
        steps = [{
            "input": "开始游戏", "response": "三名调查员分别作为客人、律师友人与送葬人抵达维丽。",
            "scene": "intro", "contains": ["三名调查员"],
            "covers": [f"dawn_seg_{i:02d}" for i in segment_tags["intro"]]
                      + ["escape_attempt_blocked", "actor_resurrection"],
        }]
        route = ["day1", "greenhouse", tia, "dream1", "day2", "town", "market",
                 "dream2", "ridge", "banquet", "dream_mansion", "knight", "day4",
                 "cellar", "nodes", "dream4", "day5", "speech", battle, planning,
                 "finale"]
        previous = "intro"
        for scene_id in route:
            covers = [f"dawn_seg_{i:02d}" for i in segment_tags.get(scene_id, [])]
            extras = {
                "tia_dead": ["tia_dead", "tia_resurrected"],
                "tia_alive": ["tia_alive"], "banquet": ["banquet_all_random_events"],
                "knight": ["nikot_killed", "nikot_resurrectable"],
                "nodes": ["node_riverbend", "node_golden_land", "node_graveyard",
                          "node_market_dream", "node_gold_factory"],
                "windmill": ["array_fail", "windmill_strength_pass",
                             "windmill_will_pass", "hugo_negotiation"],
                "rescue": ["array_success", "medical_rescue"],
                "discuss": ["has_dawn_goblet"],
                "despair": ["missing_dawn_goblet"],
            }
            covers.extend(extras.get(scene_id, []))
            response = responses[scene_id]
            if scene_id == "speech":
                if ending in ("end1", "end3"):
                    response = (
                        "HO3从人群指定中脱身，说服骚乱者让出道路，并成功发现、追踪黄衣工人。")
                    covers.extend([
                        "speech_escape_pass", "speech_persuasion", "crowd_spot_pass"])
                else:
                    response = (
                        "HO3未能摆脱人群指定，也没在拥挤中看见黄衣工人，线索因此中断。")
                    covers.extend(["speech_escape_fail", "crowd_spot_fail"])
            steps.append(movement_step(
                f"从{scene_names[previous]}推进到{scene_names[scene_id]}",
                response, scene_id, covers, [response[:4]],
            ))
            previous = scene_id

        if ending == "end1":
            text = "HO3接受交易自尽，召唤失败但舞台继续轮回，HO2被送走而HO3失踪。"
            covers = ["end1", "ho3_suicide"]
            target = "end1"
        elif ending == "end2":
            text = "调查员保留舞台并以黎明之盏重构许愿机，哈斯塔未降临但舞台只是被扭曲。"
            covers = ["end2", "goblet_accept_condition"]
            target = "end2"
        elif ending == "end3":
            text = "金像未能被破坏，哈斯塔召唤成功；影厅银幕上的金黄存在升起。"
            covers = ["end3", "summoning_success", "hand_over_puppet_with_goblet"]
            target = "end3"
        else:
            text = "调查员破坏金像、拒绝邪教徒条件并获胜，随后携黎明之盏跳入湖中进入河曲。"
            covers = ["summoning_stopped", "refuse_condition", "cultist_win"]
            target = "rivercross"
        steps.append(movement_step("我执行最终选择。", text, target, covers, [text[:4]]))
        if ending == "end4":
            for ho in (1, 2, 3):
                steps.append({
                    "input": f"HO{ho}连续通过三次意志并且不回头。",
                    "response": f"HO{ho}顶住三次回望过去的诱惑，继续走向河对岸。",
                    "contains": [f"HO{ho}"], "covers": [f"ho{ho}_three_will_pass"],
                })
            steps.append(movement_step(
                "所有人继续前进。",
                "调查员抵达未知小镇的火车站，黎明之盏的长夜终于结束。",
                "end4", ["end4", "riverbend_escape"], ["火车站"],
            ))
        write_route(stem, "dawn", "dawn_exhaustive", steps)

    full_dawn_route("dawn_full_end1", "tia_dead", "windmill", "despair", "end1")
    full_dawn_route("dawn_full_end2", "tia_alive", "rescue", "discuss", "end2")
    full_dawn_route("dawn_full_end3", "tia_dead", "rescue", "discuss", "end3")
    full_dawn_route("dawn_full_end4", "tia_alive", "windmill", "discuss", "end4")

    # Individual failures branch from END 4's river crossing.
    write_json("dawn_rivercross_world.json", {
        "name": "eval_dawn_rivercross", "rule_system": "coc",
        "starting_scene": "rivercross", "opening": "调查员携黎明之盏抵达河曲的最终渡口。",
        "scenes": {
            "rivercross": {"name": "河曲渡口", "desc": "三次意志决定是否回头。",
                           "source_text": "HO1、HO2、HO3各有意志失败结局。",
                           "exits": {"HO1回头": "ho1_end", "HO2回头": "ho2_end", "HO3回头": "ho3_end"}},
            "ho1_end": {"name": "一幅油画", "desc": "HO1个人结局", "source_text": "一幅油画", "exits": {}},
            "ho2_end": {"name": "他的遗作", "desc": "HO2个人结局", "source_text": "他的遗作", "exits": {}},
            "ho3_end": {"name": "盘中之物", "desc": "HO3个人结局", "source_text": "盘中之物", "exits": {}},
        }, "entities": {},
    })
    for ho, target, title in ((1, "ho1_end", "一幅油画"),
                              (2, "ho2_end", "他的遗作"),
                              (3, "ho3_end", "盘中之物")):
        write_route(f"dawn_ho{ho}_failure", "dawn", "dawn_rivercross", [
            {"input": "开始游戏", "response": "调查员携灯来到河曲渡口。", "scene": "rivercross",
             "contains": ["河曲"], "covers": []},
            movement_step(
                f"HO{ho}的意志检定失败并回头。",
                f"HO{ho}没能走出过去，进入个人结局《{title}》。",
                target, [f"ho{ho}_will_fail", f"ho{ho}_ending"], [title],
            ),
        ])

    required = [f"dawn_seg_{i:02d}" for i in range(70)] + [
        "escape_attempt_blocked", "actor_resurrection", "tia_dead", "tia_alive",
        "tia_resurrected", "banquet_all_random_events", "nikot_killed",
        "nikot_resurrectable", "node_riverbend", "node_golden_land",
        "node_graveyard", "node_market_dream", "node_gold_factory",
        "speech_escape_pass", "speech_escape_fail", "speech_persuasion",
        "crowd_spot_pass", "crowd_spot_fail", "array_fail", "array_success",
        "windmill_strength_pass", "windmill_will_pass", "hugo_negotiation",
        "medical_rescue", "has_dawn_goblet", "missing_dawn_goblet",
        "end1", "end2", "end3", "end4", "ho3_suicide",
        "goblet_accept_condition", "summoning_success",
        "hand_over_puppet_with_goblet", "summoning_stopped", "refuse_condition",
        "cultist_win", "riverbend_escape", "ho1_three_will_pass",
        "ho2_three_will_pass", "ho3_three_will_pass", "ho1_will_fail",
        "ho2_will_fail", "ho3_will_fail", "ho1_ending", "ho2_ending",
        "ho3_ending",
    ]
    write_json("dawn_coverage.json", {"module": "dawn", "required": required})


def main() -> None:
    generate_mountain()
    generate_honey()
    generate_dawn()
    print(f"Generated exhaustive fixtures in {OUT}")


if __name__ == "__main__":
    main()

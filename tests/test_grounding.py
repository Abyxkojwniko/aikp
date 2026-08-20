import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from chunker import Chunk
from engine import (
    advance_validated_action_clocks,
    _apply_pending_clock_outcome,
    _advance_action_clocks,
    _generate_narration,
    _find_hidden_entity_action,
    _find_unavailable_scene_move,
    _fit_context_budget,
    _extract_interaction_target,
    _is_dialogue_intent,
    _known_absent_npc,
    _maybe_arm_dynamic_check,
    _maybe_apply_movement,
    _redact_unrevealed_entities,
    _redact_unrevealed_names,
    _unlock_names_player_knows,
    _try_arm_scene_clue,
    assemble_context,
    narrate,
    narrative_audit_provider,
    narration_provider,
    resolve_entity,
    run_gm_turn,
)
from narrative_guard import build_narrative_contract
from models import create_session
from npc_context import build_scene_layer
from parser import (
    _bind_chunk_provenance,
    _extract_action_clocks,
    _iter_text_windows,
    _source_contains_text,
    _split_text_segments,
)
from plot_pusher import generate_push
from rag import hybrid_search
from reference_resolver import (
    bind_player_alias,
    list_interactable_npcs,
    reconcile_interaction_target,
    resolve_known_reference,
    select_interaction_target,
)
from state_manager import compute_state_snapshot
from server import app


class ContextBudgetTests(unittest.TestCase):
    def test_budget_never_drops_current_turn_controls(self):
        reference = "R" * 12000
        controls = (
            "=== DICE RESULT ===\nSUCCESS\n"
            "=== AVAILABLE EXITS ===\nfront_door\n"
            "CRITICAL: You are a Game Master."
        )

        fitted = _fit_context_budget(
            [reference, controls],
            [(reference, 300, "left")],
            max_chars=1800,
        )

        self.assertIn("=== DICE RESULT ===", fitted)
        self.assertIn("=== AVAILABLE EXITS ===", fitted)
        self.assertIn("CRITICAL: You are a Game Master.", fitted)
        self.assertNotIn("R" * 2000, fitted)

    def test_recent_context_keeps_newest_turns(self):
        recent = "OLD_TURN\n" + ("history " * 1500) + "\nNEWEST_TURN"
        controls = "CURRENT_TURN_CONTROL"

        fitted = _fit_context_budget(
            [recent, controls],
            [(recent, 400, "right")],
            max_chars=900,
        )

        self.assertIn("NEWEST_TURN", fitted)
        self.assertIn("CURRENT_TURN_CONTROL", fitted)
        self.assertNotIn("OLD_TURN", fitted)


class NarrationProviderTests(unittest.TestCase):
    def test_manual_provider_receives_prompt_without_openai_call(self):
        requests = []

        def provider(request):
            requests.append(request)
            return "manual narration"

        with narration_provider(provider), patch(
                "engine.OpenAI", side_effect=AssertionError("network model used")):
            response = _generate_narration(
                messages=[{"role": "user", "content": "look"}],
                api_key="unused", temperature=0.5, max_tokens=100,
                kind="turn", metadata={"scene": "study"})

        self.assertEqual("manual narration", response)
        self.assertEqual("turn", requests[0]["kind"])
        self.assertEqual("study", requests[0]["metadata"]["scene"])

    @staticmethod
    def _dynamic_check_state(player_input, resolution=None):
        return {
            "player_input": player_input,
            "world": {},
            "action_resolution": resolution or {
                "status": "passthrough", "events": [],
            },
            "session": {
                "rule_system": "coc", "pending_check": None,
                "player_state": {
                    "current_scene": "room",
                    "skills": {"侦查": 55, "攀爬": 40},
                },
            },
        }

    def test_dynamic_check_rejects_unknown_skill(self):
        state = self._dynamic_check_state("I search the desk.")

        content = _maybe_arm_dynamic_check(
            state, "Try it. 〔检定：God Mode〕")

        self.assertNotIn("检定", content)
        self.assertIsNone(state["session"]["pending_check"])

    def test_dynamic_check_rejects_routine_observation(self):
        state = self._dynamic_check_state("I look around the room.")

        content = _maybe_arm_dynamic_check(
            state, "Look carefully. 〔检定：Spot Hidden〕")

        self.assertNotIn("检定", content)
        self.assertIsNone(state["session"]["pending_check"])

    def test_dynamic_check_accepts_known_skill_for_uncertain_action(self):
        state = self._dynamic_check_state(
            "I search the desk.",
            {"status": "accepted", "requires_adjudication": True},
        )

        content = _maybe_arm_dynamic_check(
            state, "Search it. 〔检定：Spot Hidden〕")

        self.assertNotIn("检定", content)
        self.assertEqual("侦查", state["session"]["pending_check"]["skill"])
        self.assertEqual(55, state["session"]["pending_check"]["effective"])
        self.assertTrue(state["_dynamic_check_armed"])

    @staticmethod
    def _audit_state():
        world = {
            "name": "Narrative Guard",
            "scenes": {"room": {
                "name": "Room", "desc": "A witness lies motionless.",
                "exits": {},
            }},
            "entities": {"witness": {
                "type": "npc", "name": "Witness", "scene": "room",
                "initial_state": "present", "known_to_player": True,
            }},
        }
        session = create_session("narrative-guard", world["name"])
        session["player_state"]["current_scene"] = "room"
        session["entity_states"]["witness"] = "dead"
        return {
            "context_prompt": "Only describe the room.",
            "player_input": "I look around.",
            "stream": False,
            "api_key": "manual-provider-no-api-key",
            "chat_id": session["chat_id"],
            "model": world["name"],
            "world": world,
            "session": session,
            "entity_index": {"witness": {
                "type": "npc", "name": "Witness", "scene": "room"}},
            "scene_index": {"room": ["witness"]},
            "current_scene": world["scenes"]["room"],
            "scene_entities": ["witness"],
            "movement_target": None,
            "turn_summary": {},
            "_action_events": [],
            "_narration_override": None,
            "_pending_roll": False,
            "_clock_events": [],
        }

    def test_conflicting_narration_is_audited_and_repaired_once(self):
        state = self._audit_state()
        narration_calls = []
        audit_phases = []

        def narrator(request):
            narration_calls.append(request["kind"])
            if request["kind"] == "turn":
                return "Witness stands up and says hello."
            return "The Witness's body remains still."

        def auditor(request):
            audit_phases.append(request["phase"])
            if request["phase"] == "initial":
                return (
                    '{"valid":false,"violations":[{'
                    '"kind":"fact_conflict","entity_id":"witness",'
                    '"evidence":"stands up","reason":"dead NPC acts"}]}')
            return '{"valid":true,"violations":[]}'

        with narration_provider(narrator), narrative_audit_provider(auditor):
            result = narrate(state)

        self.assertEqual("The Witness's body remains still.", result["gm_response"])
        self.assertEqual(["turn", "narrative_repair"], narration_calls)
        self.assertEqual(["initial", "repair_confirmation"], audit_phases)
        self.assertEqual(0, result["_trust_signal"])
        self.assertTrue(result["_narrative_audit"]["repaired"])
        self.assertFalse(result["_narrative_audit"]["fallback_used"])

    def test_rejected_narration_cannot_leave_dynamic_roll_lock(self):
        state = self._audit_state()
        state["player_input"] = "I search the room."
        state["action_resolution"] = {
            "status": "accepted", "requires_adjudication": True,
        }
        state["session"]["player_state"]["skills"] = {"侦查": 55}

        def narrator(request):
            if request["kind"] == "turn":
                return "Witness stands up. 〔检定：Spot Hidden〕"
            return "The Witness's body remains still."

        def auditor(request):
            if request["phase"] == "initial":
                return (
                    '{"valid":false,"violations":[{'
                    '"kind":"fact_conflict","entity_id":"witness",'
                    '"evidence":"stands up","reason":"dead NPC acts"}]}')
            return '{"valid":true,"violations":[]}'

        with narration_provider(narrator), narrative_audit_provider(auditor):
            result = narrate(state)

        self.assertIsNone(result["session"]["pending_check"])
        self.assertNotIn("_dynamic_check_armed", result)
        self.assertNotIn("检定", result["gm_response"])

    def test_failed_repair_falls_back_to_committed_facts(self):
        state = self._audit_state()

        def narrator(request):
            return "Witness stands up and talks." if request["kind"] == "turn" \
                else "Witness walks out of the room."

        invalid = (
            '{"valid":false,"violations":[{'
            '"kind":"fact_conflict","entity_id":"witness",'
            '"evidence":"acts","reason":"dead NPC acts"}]}')
        with narration_provider(narrator), narrative_audit_provider(
                lambda _request: invalid):
            result = narrate(state)

        self.assertNotIn("stands", result["gm_response"])
        self.assertNotIn("walks", result["gm_response"])
        self.assertTrue(result["_narrative_audit"]["fallback_used"])

    def test_strict_audit_fails_closed_on_invalid_auditor_json(self):
        state = self._audit_state()

        with patch("engine.NARRATIVE_AUDIT_MODE", "strict"), \
                narration_provider(lambda _request: (
                    "The Witness's body remains motionless.")), \
                narrative_audit_provider(lambda _request: "not-json"):
            result = narrate(state)

        self.assertEqual(
            "这次行动没有产生可以确认的额外世界变化。",
            result["gm_response"],
        )
        self.assertTrue(result["_narrative_audit"]["fallback_used"])
        self.assertEqual(
            "unavailable", result["_narrative_audit"]["verification_status"])

    def test_story_setup_remains_due_in_a_later_linked_node(self):
        state = self._audit_state()
        state["world"]["scenes"]["payoff"] = {
            "name": "Payoff Room", "desc": "The promised bell hangs here.",
            "exits": {},
        }
        state["world"]["detailed_story_nodes"] = [
            {
                "node_id": "setup_node",
                "scenes": [{"id": "room"}],
                "promises_payoffs": [{
                    "setup": "The cracked seal must matter later.",
                    "payoff": "The bell answers the broken seal.",
                    "relation": "setup",
                    "linked_node_id": "payoff_node",
                    "source_ref": 120,
                }],
            },
            {
                "node_id": "payoff_node",
                "scenes": [{"id": "payoff"}],
                "promises_payoffs": [],
            },
        ]
        session = state["session"]
        session["visited_scene_ids"] = ["room", "payoff"]
        session["player_state"]["current_scene"] = "payoff"
        session["current_beat_id"] = "payoff_node"
        state["current_scene"] = state["world"]["scenes"]["payoff"]
        state["scene_entities"] = []

        contract = build_narrative_contract(state)

        commitment = next(
            row for row in contract["story_commitments"]
            if row["id"] == "story:setup_node:000")
        self.assertEqual("due", commitment["status"])
        self.assertEqual("payoff_node", commitment["due_at"])

    def test_scene_transition_advances_exactly_one_turn(self):
        world = {
            "name": "Movement Turn",
            "scenes": {
                "outside": {
                    "name": "Outside", "desc": "At the door.",
                    "exits": {"inside": "inside"},
                },
                "inside": {"name": "Inside", "desc": "A bare room.", "exits": {}},
            },
            "entities": {},
        }
        session = create_session("movement-turn", "Movement Turn")
        session["player_state"]["current_scene"] = "outside"
        result = {
            "movement_target": "inside",
            "gm_response": "You arrive inside.",
            "player_input": "I go inside.",
            "dice_result": None,
            "_matched_npc_ids": [],
        }

        with patch("engine.load_world", return_value=world), \
                patch("engine.get_indices", return_value=({"outside": [], "inside": []}, {})), \
                patch("engine.get_session", return_value=session), \
                patch("engine.gm_agent.invoke", return_value=result), \
                patch("engine.save_session"), \
                patch("engine._extract_arrival_traits"), \
                patch("engine._advance_plot_phase"), \
                patch("npc_context.compress_story", return_value=""):
            response = run_gm_turn(
                [{"role": "user", "content": "I go inside."}],
                model="Movement Turn", chat_id="movement-turn", api_key="unused")

        self.assertEqual("You arrive inside.", response)
        self.assertEqual(1, session["current_turn"])
        self.assertEqual("inside", session["player_state"]["current_scene"])
        self.assertEqual(1, session["turn_log"][-1]["turn"])

    def test_dead_selected_npc_is_blocked_through_compiled_graph(self):
        world = {
            "name": "Dead Target Graph",
            "rule_system": "coc",
            "scenes": {"room": {
                "name": "Room", "desc": "A quiet room.", "exits": {},
            }},
            "entities": {"witness": {
                "type": "npc", "name": "The Witness", "scene": "room",
                "initial_state": "present", "public_label": "witness",
            }},
        }
        session = create_session("dead-target-graph", "Dead Target Graph")
        session["player_state"]["current_scene"] = "room"
        session["entity_states"] = {"witness": "dead"}
        session["selected_npc_id"] = "witness"
        session["conversation_focus"] = {"npc": "witness"}
        indices = (
            {"room": ["witness"]},
            {"witness": {"type": "npc", "name": "The Witness", "scene": "room"}},
        )

        def forbidden_provider(_request):
            raise AssertionError("dead NPC dialogue reached narration provider")

        with patch("engine.load_world", return_value=world), \
                patch("engine.get_indices", return_value=indices), \
                patch("engine.get_session", return_value=session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                narration_provider(forbidden_provider):
            response = run_gm_turn(
                [{"role": "user", "content": "I ask him what happened."}],
                model="Dead Target Graph", chat_id="dead-target-graph",
                api_key="unused",
            )

        self.assertEqual("当前没有可交谈的在场人物。", response)
        self.assertIsNone(session["selected_npc_id"])
        self.assertNotIn("npc", session["conversation_focus"])
        self.assertEqual(1, session["current_turn"])

    def test_entity_transition_to_dead_blocks_dialogue_in_same_turn(self):
        world = {
            "name": "Lifecycle Transition",
            "scenes": {"room": {"name": "Room", "desc": "Quiet.", "exits": {}}},
            "entities": {"witness": {
                "type": "npc", "name": "Witness", "scene": "room",
                "initial_state": "present", "public_label": "witness",
                "states": {"present": {
                    "triggers": ["poison"],
                    "on_trigger": {"to_state": "dead"},
                }},
            }},
        }
        session = create_session("lifecycle-transition", "Lifecycle Transition")
        session["player_state"]["current_scene"] = "room"
        session["entity_states"] = {"witness": "present"}
        session["selected_npc_id"] = "witness"
        session["conversation_focus"] = {"npc": "witness"}
        entity_index = {
            "witness": {"type": "npc", "name": "Witness", "scene": "room"}}
        state = {
            "world": world,
            "session": session,
            "scene_index": {"room": ["witness"]},
            "entity_index": entity_index,
            "scene_entities": ["witness"],
            "player_input": "I ask the poisoned witness a question",
            "api_key": "",
            "matched_entity": {
                "id": "witness", "current_state": "present",
                "state_def": world["entities"]["witness"]["states"]["present"],
            },
            "dice_result": None,
        }

        resolved = resolve_entity(state)
        with patch("rag.hybrid_search", return_value=[]):
            assembled = assemble_context(resolved)

        self.assertEqual("dead", session["entity_states"]["witness"])
        self.assertEqual("present→dead", assembled["turn_summary"]
                         ["entity_state_changes"]["witness"])
        self.assertIsNone(session["selected_npc_id"])
        self.assertTrue(assembled["_npc_selection_required"])
        self.assertEqual("当前没有可交谈的在场人物。", narrate(assembled)["gm_response"])

    def test_full_assembly_keeps_source_and_controls_for_large_module(self):
        world = {
            "name": "Long Module",
            "description": "background " * 5000,
            "starting_scene": "shore",
            "scenes": {
                "shore": {
                    "name": "Rocky Shore",
                    "desc": "Rain lashes the rocks.",
                    "source_text": (
                        "CANONICAL_VISIBLE_FACT: the lighthouse door is uphill.\n"
                        + ("authoritative scene detail " * 500)
                    ),
                    "exits": {"climb to lighthouse": "lighthouse"},
                },
                "lighthouse": {"name": "Lighthouse", "desc": "A dark tower."},
            },
            "entities": {},
        }
        session = create_session("context-test", "Long Module")
        session["player_state"]["current_scene"] = "shore"
        state = {
            "world": world,
            "session": session,
            "entity_index": {},
            "scene_index": {"shore": [], "lighthouse": []},
            "api_key": "",
            "player_input": "我去灯塔",
            "scene_entities": [],
        }

        with patch("rag.hybrid_search", return_value=[]):
            result = assemble_context(state)

        prompt = result["context_prompt"]
        self.assertIn("=== CANONICAL CURRENT-SCENE SOURCE ===", prompt)
        self.assertIn("CANONICAL_VISIBLE_FACT", prompt)
        self.assertIn("climb to lighthouse", prompt)
        self.assertIn("【移动提示·重要】", prompt)
        self.assertIn("CRITICAL: You are a Game Master.", prompt)
        self.assertIn("=== NARRATIVE COMMITMENTS", prompt)


class ProvenanceTests(unittest.TestCase):
    def test_action_clock_is_extracted_from_explicit_costs_and_milestones(self):
        text = (
            "□尝试站起（0）\n可以站起来。\n"
            "□尝试开门（1）\n门打不开。\n"
            "□尝试冲水（2）\n污水涌上来。\n"
            "◇当发展轮次到达[4]时\n恶臭变得更加浓烈。\n"
            "（此时让调查员进行一次体质检定和SAN检定。）\n"
            "◇当发展轮次到达[8]时\n墙上爬满手印。\n"
        )

        clocks = _extract_action_clocks(text)
        clock = clocks["development_round"]

        self.assertEqual("发展轮次", clock["name"])
        self.assertEqual([0, 1, 2], [a["increment"] for a in clock["actions"]])
        self.assertEqual([4, 8], [m["at"] for m in clock["milestones"]])
        self.assertNotIn("检定", clock["milestones"][0]["narration"])
        self.assertNotIn("SAN", clock["milestones"][0]["narration"])

    def test_dialogue_intent_does_not_match_words_containing_say_or_ask(self):
        for text in ("继续推进到疑问", "我去集市演讲现场", "我回答神像的问题",
                     "这是一个传说", "阅读小说"):
            with self.subTest(text=text):
                self.assertFalse(_is_dialogue_intent(text))

    def test_dialogue_intent_keeps_actual_conversation(self):
        for text in ("我问房东发生了什么", "我对她说你好", "和山登交谈",
                     "我呼喊四间管", "去问房东", "和山登说", "找老板聊",
                     "I ask him", "I convince Brinn to release us",
                     "I negotiate with the captain"):
            with self.subTest(text=text):
                self.assertTrue(_is_dialogue_intent(text))

    def test_roll_dependent_clock_cost_uses_actual_verdict(self):
        world = {"action_clocks": {"development_round": {
            "name": "发展轮次", "initial": 0,
            "milestones": [{"at": 4, "flag": "round_4", "narration": "恶臭增强。"}],
        }}}
        session = {"clocks": {}, "flags": []}
        pending = {
            "_outcome_clock_id": "development_round",
            "_outcome_clock_name": "发展轮次",
            "_outcome_clock_action": "暴力行为",
            "_outcome_clock_increments": {
                "failure": 0, "hard_success": 1,
                "extreme_success": 2, "fumble": 3,
            },
        }

        self.assertEqual(0, _apply_pending_clock_outcome(
            session, world, pending, "failure")["increment"])
        self.assertEqual(1, _apply_pending_clock_outcome(
            session, world, pending, "hard_success")["new"])
        self.assertEqual(3, _apply_pending_clock_outcome(
            session, world, pending, "extreme_success")["new"])
        event = _apply_pending_clock_outcome(session, world, pending, "fumble")
        self.assertEqual(6, event["new"])
        self.assertEqual([4], [item["at"] for item in event["milestones"]])

    def test_action_clock_uses_action_cost_not_chat_turn_count(self):
        world = {"action_clocks": {"development_round": {
            "name": "发展轮次", "initial": 0, "default_increment": 0,
            "actions": [
                {"label": "站起", "triggers": ["站起"], "increment": 0},
                {"label": "开门", "triggers": ["开门"], "increment": 1},
                {"label": "冲水", "triggers": ["冲水"], "increment": 2},
            ],
            "milestones": [
                {"at": 4, "flag": "round_4", "narration": "恶臭增强。"},
            ],
        }}}
        session = {"clocks": {"development_round": 0}, "flags": []}

        self.assertEqual([], _advance_action_clocks("我站起", session, world))
        _advance_action_clocks("我开门", session, world)
        _advance_action_clocks("我冲水", session, world)
        events = _advance_action_clocks("我开门", session, world)

        self.assertEqual(4, session["clocks"]["development_round"])
        self.assertEqual(["round_4"], session["flags"])
        self.assertEqual(4, events[0]["milestones"][0]["at"])

    def test_blocked_action_does_not_advance_action_clock(self):
        world = {"action_clocks": {"round": {
            "name": "Round", "initial": 0, "default_increment": 0,
            "actions": [{
                "label": "Open", "triggers": ["open the door"],
                "increment": 1,
            }],
        }}}
        session = {"clocks": {"round": 0}, "flags": []}
        state = {
            "player_input": "I open the door.", "world": world,
            "session": session, "_action_block": "It is locked.",
            "action_resolution": {"status": "blocked"},
        }

        advance_validated_action_clocks(state)

        self.assertEqual(0, session["clocks"]["round"])
        self.assertEqual([], state["_clock_events"])

    def test_validated_action_advances_action_clock(self):
        world = {"action_clocks": {"round": {
            "name": "Round", "initial": 0, "default_increment": 0,
            "actions": [{
                "label": "Open", "triggers": ["open the door"],
                "increment": 1,
            }],
        }}}
        session = {"clocks": {"round": 0}, "flags": []}
        state = {
            "player_input": "I open the door.", "world": world,
            "session": session, "_action_block": "",
            "action_resolution": {"status": "accepted"},
        }

        advance_validated_action_clocks(state)

        self.assertEqual(1, session["clocks"]["round"])
        self.assertEqual(1, state["_clock_events"][0]["increment"])

    def test_chinese_campaign_structure_beats_inline_checks(self):
        text = (
            "正文\n导入\n开场内容。\n"
            "第一日\n主线：委托\n抵达公馆。\n"
            "【侦查】女仆长的裙尾带着泥土。\n"
            "【艺术】：这是一幅抽象作品。\n"
            "分支：缇亚死亡\n分支内容。\n"
            "探索：公馆\n探索内容。"
        )

        segments = _split_text_segments(text)
        titles = [segment["title"] for segment in segments]

        self.assertIn("正文", titles)
        self.assertIn("导入", titles)
        self.assertIn("第一日", titles)
        self.assertIn("主线：委托", titles)
        self.assertIn("分支：缇亚死亡", titles)
        self.assertIn("探索：公馆", titles)
        self.assertNotIn("【侦查】女仆长的裙尾带着泥土。", titles)
        self.assertNotIn("【艺术】：这是一幅抽象作品。", titles)

    def test_toc_dot_leaders_and_merged_ho_page_numbers_are_not_headings(self):
        text = (
            "目录\n■模组简介 ..1\nHO114\nHO214\n主线：委托16\n"
            "正文\nHO1\n真实的个人导入。\n主线：委托\n"
            "HO2：这是一句很长的角色台词，并不是章节标题。\n"
            "■模组简介\n真实的模组正文。"
        )

        titles = [segment["title"] for segment in _split_text_segments(text)]

        self.assertNotIn("■模组简介 ..1", titles)
        self.assertNotIn("HO114", titles)
        self.assertNotIn("HO214", titles)
        self.assertNotIn("主线：委托16", titles)
        self.assertNotIn("HO2：这是一句很长的角色台词，并不是章节标题。", titles)
        self.assertIn("HO1", titles)
        self.assertIn("主线：委托", titles)
        self.assertIn("■模组简介", titles)

    def test_english_toc_and_title_case_scene_headings_are_split(self):
        text = (
            "Contents\n"
            "Introduction ........ 5\n"
            "Tracking the Monster ........ 14\n"
            "RUNNING TITLE\n"
            "Introduction\nOpening material.\n"
            "The Castle of the Crane\nCastle material.\n"
            "Scene One: The Storm\nStorm material.\n"
            "Tracking The Monster\nTracking material.\n"
            "RUNNING TITLE\nPage body.\nRUNNING TITLE\nMore body.\n"
        )

        titles = [segment["title"] for segment in _split_text_segments(text)]

        self.assertIn("Introduction", titles)
        self.assertIn("The Castle of the Crane", titles)
        self.assertIn("Scene One: The Storm", titles)
        self.assertIn("Tracking The Monster", titles)
        self.assertNotIn("Introduction ........ 5", titles)
        self.assertNotIn("RUNNING TITLE", titles)

    def test_solo_numeric_sections_exclude_pdf_page_numbers(self):
        text = (
            "INTRODUCTION\nNow go to 1.\n"
            "1\nThe journey begins.\nGo to 263.\n"
            "ALONE AGAINST THE FLAMES\n3\nALONE AGAINST THE FLAMES\n"
            "4\nr\nd\nf\n"
            "2\nYou wake in darkness.\nGo to 108.\n"
            + "\n".join(f"Go to {n}." for n in range(10, 25))
        )

        titles = [segment["title"] for segment in _split_text_segments(text)]

        self.assertIn("1", titles)
        self.assertIn("2", titles)
        self.assertNotIn("3", titles)
        self.assertNotIn("4", titles)

    def test_invalid_llm_quote_is_replaced_by_real_source_excerpt(self):
        chunk = Chunk(
            index=3,
            title="Study",
            text="The investigators enter the study. A brass key rests under the ledger.",
            start_line=10,
            end_line=11,
            char_count=73,
        )
        result = {
            "clues": [{
                "id": "brass_key",
                "name": "brass key",
                "source_quote": "A silver key floats above the desk.",
            }]
        }

        _bind_chunk_provenance(result, chunk)

        clue = result["clues"][0]
        self.assertTrue(clue["source_verified"])
        self.assertIn("brass key", clue["source_quote"])
        self.assertNotIn("silver key", clue["source_quote"])
        self.assertEqual(3, clue["source_chunk"])

    def test_unverifiable_entry_is_marked_unverified(self):
        chunk = Chunk(0, "Room", "An empty room.", 0, 1, 14)
        result = {"items": [{"id": "idol", "name": "Golden Idol",
                              "source_quote": "A golden idol gleams."}]}

        _bind_chunk_provenance(result, chunk)

        item = result["items"][0]
        self.assertFalse(item["source_verified"])
        self.assertNotIn("source_quote", item)

    def test_full_module_windows_include_the_ending(self):
        text = "BEGIN\n\n" + ("middle paragraph\n\n" * 4000) + "FINAL_CLIMAX"

        windows = list(_iter_text_windows(
            text,
            window_chars=5000,
            overlap_chars=300,
        ))

        self.assertGreater(len(windows), 2)
        self.assertTrue(windows[0].startswith("BEGIN"))
        self.assertTrue(windows[-1].endswith("FINAL_CLIMAX"))

    def test_english_location_headings_split_source(self):
        text = (
            "THE ADVENTURE\nIntro text.\n"
            "LOCATION 1:\nTHE OLD HOUSE\nDust covers the hall.\n"
            "LOCATION 2:\nCITY ARCHIVES\nA clerk guards the files."
        )

        segments = _split_text_segments(text)
        titles = [segment["title"] for segment in segments]

        self.assertIn("LOCATION 1:", titles)
        self.assertIn("THE OLD HOUSE", titles)
        self.assertIn("LOCATION 2:", titles)
        self.assertIn("CITY ARCHIVES", titles)

    def test_repeated_pdf_running_header_is_not_a_scene_boundary(self):
        text = (
            "THE LIGHTLESS BEACON\nFront matter.\n"
            "LOCATION 1:\nROCKY SHORE\nWaves break here.\n"
            "THE LIGHTLESS BEACON\nMore shore text.\n"
            "THE LIGHTLESS BEACON\nEnd of shore text.\n"
            "LOCATION 2:\nLIGHTHOUSE COTTAGE\nA door is open."
        )

        segments = _split_text_segments(text)
        titles = [segment["title"] for segment in segments]

        self.assertNotIn("THE LIGHTLESS BEACON", titles)
        shore = next(s for s in segments if s["title"] == "ROCKY SHORE")
        self.assertIn("End of shore text", shore["text"])

    def test_character_stat_line_is_not_a_scene_boundary(self):
        text = (
            "MONSTER PROFILES\nCreature details.\n"
            "STR 100 CON 40 SIZ 25 DEX 80 INT 10\n"
            "STR50 CON45 SIZ60 DEX60 INT45 POW80\n"
            "Its claws cause terrible wounds."
        )

        titles = [segment["title"] for segment in _split_text_segments(text)]

        self.assertNotIn("STR 100 CON 40 SIZ 25 DEX 80 INT 10", titles)

    def test_opening_must_exist_in_source_allowing_pdf_line_wraps(self):
        source = "The storm bends the trees.\nThe lighthouse is completely dark."

        self.assertTrue(_source_contains_text(
            source,
            "The storm bends the trees. The lighthouse is completely dark.",
        ))
        self.assertFalse(_source_contains_text(
            source,
            "A friendly keeper welcomes everyone into a warm lighthouse.",
        ))


class RetrievalScopeTests(unittest.TestCase):
    def test_hybrid_search_is_scoped_to_current_scene(self):
        with patch("rag.query", return_value=[]) as query_mock:
            hybrid_search("module", "hidden door", "study", n_results=4)

        query_mock.assert_called_once_with(
            "module",
            "hidden door",
            n_results=4,
            filter_scene="study",
        )


class NpcTargetApiTests(unittest.TestCase):
    def setUp(self):
        self.world = {
            "name": "Roster API",
            "scenes": {"dock": {
                "name": "Dock", "exits": {},
                "source_text": "A wounded sailor waits beside the pier.",
            }},
            "entities": {
                "sailor": {
                    "type": "npc", "name": "Elias", "scene": "dock",
                    "initial_state": "present", "public_label": "wounded sailor",
                },
                "stalker": {
                    "type": "npc", "name": "Stalker", "scene": "dock",
                    "initial_state": "hidden", "public_label": "stalker",
                },
            },
        }
        self.session = create_session("api-roster", "Roster API")
        self.session["model"] = "Roster API"
        self.session["player_state"]["current_scene"] = "dock"
        self.session["entity_states"] = {"sailor": "present", "stalker": "hidden"}
        self.session["turn_log"] = [{
            "turn": 0, "scene": "dock", "player_input": "[game start]",
            "gm_response": "A wounded sailor waits beside the pier.",
        }]
        self.scene_index = {"dock": ["sailor", "stalker"]}
        self.entity_index = {
            eid: {"type": "npc", "name": e["name"], "scene": "dock"}
            for eid, e in self.world["entities"].items()
        }
        self.context = (
            self.session, self.world, self.scene_index, self.entity_index)
        self.client = TestClient(app)

    def test_roster_endpoint_and_target_validation(self):
        with patch("server._npc_roster_context", return_value=self.context), \
                patch("state_manager.save_session"):
            roster = self.client.get("/api/session/api-roster/npcs")
            selected = self.client.post(
                "/api/session/api-roster/npc-target", json={"npc_id": "sailor"})
            rejected = self.client.post(
                "/api/session/api-roster/npc-target", json={"npc_id": "stalker"})

        self.assertEqual(200, roster.status_code)
        self.assertEqual(["sailor"], [n["id"] for n in roster.json()["npcs"]])
        self.assertEqual(200, selected.status_code)
        self.assertEqual("sailor", selected.json()["selected_npc_id"])
        self.assertEqual(400, rejected.status_code)

    def test_roster_endpoint_clears_a_stale_hidden_selection(self):
        self.session["selected_npc_id"] = "stalker"
        self.session["conversation_focus"] = {"npc": "stalker"}
        with patch("server._npc_roster_context", return_value=self.context), \
                patch("state_manager.save_session") as save_mock:
            response = self.client.get("/api/session/api-roster/npcs")

        self.assertEqual(200, response.status_code)
        self.assertIsNone(response.json()["selected_npc_id"])
        self.assertNotIn("npc", self.session["conversation_focus"])
        save_mock.assert_called_once_with(self.session)

    def test_dead_npc_disappears_and_stale_api_selection_is_cleared(self):
        self.session["selected_npc_id"] = "sailor"
        self.session["conversation_focus"] = {"npc": "sailor"}
        self.session["entity_states"]["sailor"] = "dead"

        with patch("server._npc_roster_context", return_value=self.context), \
                patch("state_manager.save_session") as save_mock:
            response = self.client.get("/api/session/api-roster/npcs")

        self.assertEqual(200, response.status_code)
        self.assertEqual([], response.json()["npcs"])
        self.assertIsNone(response.json()["selected_npc_id"])
        self.assertNotIn("npc", self.session["conversation_focus"])
        save_mock.assert_called_once_with(self.session)


class ScenarioTargetApiTests(unittest.TestCase):
    def setUp(self):
        self.world = {
            "name": "Anthology API", "rule_system": "coc",
            "scenarios": [
                {"id": "first", "title": "First", "starting_scene": "first::room"},
                {"id": "second", "title": "Second", "starting_scene": "second::room"},
            ],
            "scenes": {
                "first::room": {"name": "First Room", "scenario_id": "first",
                                "exits": {}},
                "second::room": {"name": "Second Room", "scenario_id": "second",
                                 "exits": {}},
            },
            "entities": {}, "story_beats": [],
        }
        self.session = create_session("scenario-api", "Anthology API")
        self.session["model"] = "Anthology API"
        self.session["current_scenario_id"] = "first"
        self.context = (self.session, self.world, {}, {})
        self.client = TestClient(app)

    def test_list_and_switch_scenario_contract(self):
        with patch("server._npc_roster_context", return_value=self.context), \
                patch("state_manager.save_session") as save_mock, \
                patch.dict("engine._session_cache", {}, clear=True):
            listed = self.client.get("/api/session/scenario-api/scenarios")
            switched = self.client.post(
                "/api/session/scenario-api/scenario-target",
                json={"scenario_id": "second"},
            )

        self.assertEqual(200, listed.status_code)
        self.assertEqual("first", listed.json()["selected_scenario_id"])
        self.assertEqual(200, switched.status_code)
        self.assertEqual("second", switched.json()["selected_scenario_id"])
        self.assertEqual("second::room", switched.json()["starting_scene"])
        save_mock.assert_called_once()

    def test_unknown_scenario_is_rejected(self):
        with patch("server._npc_roster_context", return_value=self.context):
            response = self.client.post(
                "/api/session/scenario-api/scenario-target",
                json={"scenario_id": "missing"},
            )
        self.assertEqual(400, response.status_code)

class PlayerKnowledgeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.world = {
            "name": "Closed World",
            "scenes": {
                "study": {
                    "name": "Study",
                    "desc": "A desk stands beneath the window.",
                    "source_text": (
                        "A desk stands beneath the window. A lighthouse keeper "
                        "waits beside it."
                    ),
                    "exits": {"hall": "hall"},
                },
                "hall": {"name": "Hall", "desc": "A narrow hall."},
                "basement": {"name": "Secret Basement", "desc": "The finale."},
            },
            "entities": {
                "desk": {
                    "type": "item", "name": "wooden desk", "scene": "study",
                    "initial_state": "present", "description": "An old desk.",
                },
                "silver_key": {
                    "type": "item", "name": "silver key", "scene": "study",
                    "initial_state": "hidden", "description": "Under the ledger.",
                },
                "blood_note": {
                    "type": "clue", "name": "bloody note", "scene": "study",
                    "initial_state": "hidden", "description": "Names the culprit.",
                },
                "lurker": {
                    "type": "npc", "name": "Marsh Lurker", "scene": "study",
                    "initial_state": "hidden", "description": "Waiting outside.",
                },
                "keeper": {
                    "type": "npc", "name": "George Cassidy", "scene": "study",
                    "initial_state": "present", "public_label": "lighthouse keeper",
                    "description": "A tired keeper.",
                },
            },
        }
        self.scene_index = {
            "study": ["desk", "silver_key", "blood_note", "lurker", "keeper"],
            "hall": [], "basement": []}
        self.entity_index = {
            eid: {"type": e["type"], "name": e["name"], "scene": e["scene"]}
            for eid, e in self.world["entities"].items()
        }
        self.session = create_session("knowledge", "Closed World")
        self.session["player_state"]["current_scene"] = "study"
        self.session["entity_states"] = {
            "desk": "present", "silver_key": "hidden", "blood_note": "hidden",
            "lurker": "hidden", "keeper": "present"}
        self.session["turn_log"] = [{
            "turn": 0, "scene": "study", "player_input": "[game start]",
            "gm_response": "A lighthouse keeper waits beside the desk.",
        }]
        self.session["npc_states"]["George Cassidy"] = {
            "static": {"name": "George Cassidy", "dialogue": {}},
            "dynamic": {
                "disclosure": {"name": False, "background": False},
                "traits": [], "nicknames": [], "trust": 0, "mood": "neutral",
            },
        }

    def test_scene_and_snapshot_hide_undiscovered_entities(self):
        scene = build_scene_layer(
            "study", self.world, self.scene_index, self.entity_index, self.session)
        snapshot = compute_state_snapshot(
            self.session, self.world, self.scene_index, self.entity_index)

        for player_context in (scene, snapshot):
            self.assertIn("wooden desk", player_context)
            self.assertNotIn("silver key", player_context)
            self.assertNotIn("bloody note", player_context)
            self.assertNotIn("Names the culprit", player_context)

    def test_direct_action_on_guessed_hidden_item_is_blocked(self):
        target = _find_hidden_entity_action(
            "I pick up the silver key", self.session, self.world)
        self.assertEqual("silver key", target)

        self.session["entity_states"]["silver_key"] = "obtained"
        target = _find_hidden_entity_action(
            "I use the silver key", self.session, self.world)
        self.assertEqual("", target)

    def test_generated_output_redacts_hidden_entity_names(self):
        self.session["npc_states"]["Marsh Lurker"] = {
            "dynamic": {"disclosure": {"name": False}, "traits": []}}
        output = _redact_unrevealed_entities(
            "I am Marsh Lurker. The bloody note is beside the silver key.",
            self.session,
            self.world,
        )
        output = _redact_unrevealed_names(output, self.session, self.entity_index)
        self.assertNotIn("bloody note", output)
        self.assertNotIn("silver key", output)
        self.assertNotIn("Marsh Lurker", output)
        self.assertFalse(
            self.session["npc_states"]["Marsh Lurker"]["dynamic"]
            ["disclosure"]["name"])

    def test_publicly_named_hidden_item_is_known_but_cannot_be_taken(self):
        self.world["opening"] = "Your task is to recover the silver key."

        output = _redact_unrevealed_entities(
            "The silver key is the object named in your task.",
            self.session,
            self.world,
        )

        self.assertIn("silver key", output)
        self.assertEqual(
            "silver key",
            _find_hidden_entity_action(
                "I take the silver key", self.session, self.world),
        )

    def test_english_self_introduction_reveals_canonical_name(self):
        output = _redact_unrevealed_names(
            "The keeper says, 'My name is George Cassidy.'",
            self.session,
            self.entity_index,
        )

        self.assertIn("George Cassidy", output)
        self.assertTrue(
            self.session["npc_states"]["George Cassidy"]["dynamic"]
            ["disclosure"]["name"])

    def test_unrevealed_english_surname_is_redacted(self):
        output = _redact_unrevealed_names(
            "Cassidy's body lies in the lantern room.",
            self.session,
            self.entity_index,
        )

        self.assertNotIn("Cassidy", output)
        self.assertFalse(
            self.session["npc_states"]["George Cassidy"]["dynamic"]
            ["disclosure"]["name"])

    def test_public_place_surname_is_not_redacted_but_full_name_stays_hidden(self):
        self.session["npc_states"]["Walter Corbitt"] = {
            "dynamic": {"disclosure": {"name": False}, "traits": []}}
        self.entity_index["corbitt"] = {
            "type": "npc", "name": "Walter Corbitt",
            "public_label": "wizened body", "scene": "cellar",
        }
        world = dict(self.world)
        world["opening"] = "You have been hired to inspect the Corbitt House."

        output = _redact_unrevealed_names(
            "The Corbitt House was owned by Walter Corbitt.",
            self.session, self.entity_index, world)

        self.assertIn("Corbitt House", output)
        self.assertNotIn("Walter Corbitt", output)
        self.assertIn("wizened body", output)

    def test_territorial_title_does_not_redact_place_or_faction_name(self):
        self.session["npc_states"]["King Uriens of Gorre"] = {
            "dynamic": {"disclosure": {"name": False}, "traits": []}}
        self.entity_index["uriens"] = {
            "type": "npc", "name": "King Uriens of Gorre",
            "public_label": "king of Gorre", "scene": "study",
        }

        output = _redact_unrevealed_names(
            "The Blue Team charges the Knights of Gorre.",
            self.session,
            self.entity_index,
        )

        self.assertEqual(
            "The Blue Team charges the Knights of Gorre.", output)

    def test_surname_unlock_requires_an_allowed_current_or_public_npc(self):
        _unlock_names_player_knows(
            "I inspect Cassidy's coat.", self.session, {"George Cassidy"})

        self.assertTrue(
            self.session["npc_states"]["George Cassidy"]["dynamic"]
            ["disclosure"]["name"])

    def test_role_name_and_shared_surname_are_not_over_redacted(self):
        self.session["npc_states"] = {
            "The Landlord": {"dynamic": {"disclosure": {"name": False}}},
            "May Ledbetter": {"dynamic": {"disclosure": {"name": True}}},
            "Ruth Ledbetter": {"dynamic": {"disclosure": {"name": False}}},
        }
        self.entity_index["landlord"] = {
            "type": "npc", "name": "The Landlord",
            "public_label": "landlord", "scene": "study",
        }

        output = _redact_unrevealed_names(
            "The landlord says May Ledbetter is at the Ledbetter house, but "
            "Ruth Ledbetter is absent.",
            self.session,
            self.entity_index,
        )

        self.assertIn("landlord", output.lower())
        self.assertIn("May Ledbetter", output)
        self.assertIn("Ledbetter house", output)
        self.assertNotIn("Ruth Ledbetter", output)

    def test_stall_push_does_not_name_hidden_entity_or_invent_time(self):
        self.session["current_turn"] = 8
        self.session["turn_log"] = [
            {"player_input": "look", "entity_state_changes": {}, "new_flags": [],
             "items_obtained": []}
            for _ in range(5)
        ]

        push = generate_push(
            self.session, self.scene_index, self.entity_index, self.world)

        self.assertNotIn("silver key", push)
        self.assertNotIn("bloody note", push)
        self.assertNotIn("late", push.lower())

    def test_player_cannot_jump_to_a_non_adjacent_real_scene(self):
        blocked = _find_unavailable_scene_move(
            "I go to the Secret Basement", "study", self.world)
        self.assertEqual("Secret Basement", blocked)

        allowed = _find_unavailable_scene_move("I go to the Hall", "study", self.world)
        self.assertEqual("", allowed)

    def test_player_cannot_use_alias_to_bypass_gated_exit(self):
        self.world["scenes"]["study"]["exits"]["open sealed stairs"] = {
            "target": "basement", "requires_flag": "stairs_unsealed",
        }
        self.world["scenes"]["basement"]["aliases"] = ["cellar"]

        blocked = _find_unavailable_scene_move(
            "I enter the cellar", "study", self.world, self.session)

        self.assertEqual("cellar", blocked)

    def test_numbered_module_heading_resolves_from_natural_scene_name(self):
        world = {
            "scenes": {
                "rec": {"name": "A2. Rec Room", "exits": {"cargo": "cargo"}},
                "cargo": {"name": "A4. Lower Cargo Hold", "exits": {}},
                "bridge": {"name": "B5. Bridge", "exits": {}},
            },
            "entities": {},
        }

        blocked = _find_unavailable_scene_move(
            "We teleport directly to the bridge.", "rec", world)

        self.assertEqual("Bridge", blocked)

    def test_player_cannot_enter_a_storybook_setting(self):
        world = dict(self.world)
        world["embedded_settings"] = [{
            "scope": {"id": "storybook", "navigable": False},
            "scenes": [{
                "id": "book_castle", "name": "Castle in the Storybook",
                "aliases": ["Paper Castle"], "navigable": False,
            }],
            "entities": [],
        }]

        self.assertEqual(
            "Castle in the Storybook",
            _find_unavailable_scene_move(
                "I enter the Castle in the Storybook", "study", world),
        )
        self.assertEqual(
            "Paper Castle",
            _find_unavailable_scene_move(
                "I go to the Paper Castle", "study", world),
        )

    def test_english_adverbial_move_cannot_skip_scenes(self):
        blocked = _find_unavailable_scene_move(
            "I ride directly to the Secret Basement", "study", self.world)

        self.assertEqual("Secret Basement", blocked)

    def test_internal_scene_id_is_not_a_player_facing_destination(self):
        world = {
            "scenes": {
                "castle": {
                    "name": "Castle of the Crane",
                    "exits": {"journey west": "road"},
                },
                "road": {"name": "The Journey West", "exits": {}},
                "return": {"name": "Homecoming", "exits": {}},
            },
        }

        blocked = _find_unavailable_scene_move(
            "We return to the road to reconsider", "castle", world)

        self.assertEqual("", blocked)

    def test_movement_marker_cannot_bypass_scene_graph(self):
        state = {
            "world": self.world,
            "current_scene": self.world["scenes"]["study"],
            "player_input": "I go to the Secret Basement",
            "movement_target": None,
        }

        text = _maybe_apply_movement(state, "You arrive. 〔前往：basement〕")

        self.assertNotIn("〔前往", text)
        self.assertIsNone(state["movement_target"])

    def test_scene_clue_search_really_pauses_for_player_roll(self):
        self.world["scenes"]["study"]["clues"] = [{
            "id": "desk_scratches",
            "desc": "Scratches show the desk was moved.",
            "check": "Spot Hidden",
        }]
        state = {"world": self.world, "session": self.session}

        _try_arm_scene_clue(state, "I search the study")

        self.assertTrue(state["_pending_roll"])
        self.assertEqual(
            "desk_scratches",
            self.session["pending_check"]["_scene_clue_id"],
        )

    def test_unresolved_roll_blocks_next_action_without_advancing_turn(self):
        self.session["pending_check"] = {
            "skill": "Spot Hidden", "effective": 50, "rule_system": "coc"}
        self.session["current_turn"] = 4

        with patch("engine.load_world", return_value=self.world), \
                patch("engine.get_indices",
                      return_value=(self.scene_index, self.entity_index)), \
                patch("engine.get_session", return_value=self.session), \
                patch("engine.gm_agent.invoke",
                      side_effect=AssertionError("pending roll reached graph")):
            response = run_gm_turn(
                [{"role": "user", "content": "I leave the room"}],
                "Closed World", "pending-roll-test", "unused", stream=False)

        self.assertIn("先完成", response)
        self.assertEqual(4, self.session["current_turn"])
        self.assertIsNotNone(self.session["pending_check"])

    def test_long_english_npc_reference_is_parsed_and_absence_is_deterministic(self):
        self.world["entities"]["george"] = {
            "type": "npc", "name": "George Cassidy", "scene": "basement",
            "aliases": ["Cassidy"],
        }
        self.entity_index["george"] = {
            "type": "npc", "name": "George Cassidy", "scene": "basement"}

        target = _extract_interaction_target(
            "I ask George Cassidy about the lighthouse")
        absent = _known_absent_npc(
            target, self.scene_index["study"], self.entity_index, self.world)

        self.assertEqual("George Cassidy", target)
        self.assertEqual("george", absent)

    def test_npc_roster_uses_safe_labels_and_server_validates_selection(self):
        roster = list_interactable_npcs(
            self.session, self.world, self.scene_index, self.entity_index)

        self.assertEqual(["keeper"], [item["id"] for item in roster])
        self.assertEqual("lighthouse keeper", roster[0]["label"])
        self.assertNotIn("George Cassidy", roster[0]["label"])
        self.assertNotIn("lurker", [item["id"] for item in roster])

        select_interaction_target(
            self.session, "keeper", self.world, self.scene_index, self.entity_index)
        self.assertEqual("keeper", self.session["selected_npc_id"])
        with self.assertRaises(ValueError):
            select_interaction_target(
                self.session, "lurker", self.world, self.scene_index,
                self.entity_index)

    def test_roster_rejects_ungrounded_label_and_unavailable_npc(self):
        self.world["entities"]["keeper"]["public_label"] = "secret cult leader"
        roster = list_interactable_npcs(
            self.session, self.world, self.scene_index, self.entity_index)
        self.assertEqual("在场人物", roster[0]["label"])

        self.session["entity_states"]["keeper"] = "unconscious"
        roster = list_interactable_npcs(
            self.session, self.world, self.scene_index, self.entity_index)
        self.assertEqual([], roster)
        with self.assertRaises(ValueError):
            select_interaction_target(
                self.session, "keeper", self.world, self.scene_index,
                self.entity_index)

    def test_vanished_npc_is_removed_but_can_return_after_resurrection(self):
        for unavailable_state in ("disappeared", "vanished", "已消失"):
            with self.subTest(state=unavailable_state):
                self.session["entity_states"]["keeper"] = unavailable_state
                self.assertEqual([], list_interactable_npcs(
                    self.session, self.world, self.scene_index,
                    self.entity_index))

        # Some modules explicitly resurrect actors. Availability follows the
        # current authoritative state rather than retaining a permanent death.
        self.session["entity_states"]["keeper"] = "present"
        roster = list_interactable_npcs(
            self.session, self.world, self.scene_index, self.entity_index)
        self.assertEqual(["keeper"], [item["id"] for item in roster])

    def test_dead_selected_npc_dialogue_is_blocked_before_narration(self):
        select_interaction_target(
            self.session, "keeper", self.world, self.scene_index,
            self.entity_index)
        self.session["entity_states"]["keeper"] = "dead"
        state = self._context_state("I ask him what happened")

        with patch("rag.hybrid_search", return_value=[]):
            result = assemble_context(state)

        self.assertTrue(result["_npc_selection_required"])
        self.assertEqual(0, result["_available_npc_count"])
        self.assertIsNone(self.session["selected_npc_id"])
        self.assertNotIn("npc", self.session["conversation_focus"])
        with narration_provider(
                lambda request: (_ for _ in ()).throw(
                    AssertionError("dead NPC dialogue reached narration provider"))):
            narrated = narrate(result)
        self.assertEqual("当前没有可交谈的在场人物。", narrated["gm_response"])

    def test_authored_dialogue_can_finish_when_it_disables_target(self):
        self.world["entities"]["keeper"]["states"] = {
            "present": {},
            "restrained": {"interactable": False},
        }
        select_interaction_target(
            self.session, "keeper", self.world, self.scene_index,
            self.entity_index)
        self.session["entity_states"]["keeper"] = "restrained"
        state = self._context_state("I ask him what happened")
        state["matched_entity"] = {
            "id": "keeper", "current_state": "present", "state_def": {},
        }
        state["action_resolution"] = {"status": "accepted"}

        with patch("rag.hybrid_search", return_value=[]):
            result = assemble_context(state)

        self.assertFalse(result.get("_npc_selection_required", False))
        self.assertEqual(["keeper"], result["_matched_npc_ids"])
        self.assertIsNone(self.session["selected_npc_id"])
        with narration_provider(lambda _request: "He gives one final warning."):
            narrated = narrate(result)
        self.assertEqual("He gives one final warning.", narrated["gm_response"])

    def test_reconcile_keeps_valid_target_and_clears_only_stale_focus(self):
        self.session["selected_npc_id"] = "keeper"
        self.session["conversation_focus"] = {"npc": "keeper", "item": "desk"}
        roster = list_interactable_npcs(
            self.session, self.world, self.scene_index, self.entity_index)

        self.assertFalse(reconcile_interaction_target(self.session, roster))
        self.session["entity_states"]["keeper"] = "absent"
        self.assertTrue(reconcile_interaction_target(self.session, []))
        self.assertEqual({"item": "desk"}, self.session["conversation_focus"])

    def test_player_alias_resolves_to_stable_visible_id(self):
        bind_player_alias(self.session, "灯塔大叔", "keeper")

        result = resolve_known_reference(
            "我观察灯塔大叔", self.scene_index["study"], self.entity_index,
            self.world, self.session)

        self.assertEqual("keeper", result["entity_id"])

    def _context_state(self, player_input):
        return {
            "world": self.world,
            "session": self.session,
            "entity_index": self.entity_index,
            "scene_index": self.scene_index,
            "api_key": "",
            "player_input": player_input,
            "scene_entities": self.scene_index["study"],
            "stream": False,
        }

    def test_dialogue_requires_click_selection_without_calling_narration_model(self):
        for player_input in (
                "I ask him about the light",
                "I keep asking the dead keeper what happened",
                "I continue questioning her",
                "你好",
                "我喊道「等一下」",
        ):
            with self.subTest(player_input=player_input):
                state = self._context_state(player_input)
                with patch("rag.hybrid_search", return_value=[]):
                    result = assemble_context(state)

                self.assertTrue(result["_npc_selection_required"])
                narrated = narrate(result)
                self.assertIn("选择", narrated["gm_response"])

    def test_player_can_name_public_third_party_across_scenes(self):
        self.world["scenes"]["hospital"] = {
            "name": "Hospital", "desc": "A hospital ward."}
        self.world["entities"]["old_gurteen"] = {
            "type": "npc", "name": "Old Gurteen", "scene": "hospital",
            "initial_state": "present", "description": "An elderly patient.",
            "known_to_player": True,
        }
        self.entity_index["old_gurteen"] = {
            "type": "npc", "name": "Old Gurteen", "scene": "hospital"}
        self.session["entity_states"]["old_gurteen"] = "present"
        self.session["npc_states"]["Old Gurteen"] = {
            "static": {"name": "Old Gurteen", "dialogue": {}},
            "dynamic": {
                "disclosure": {"name": False, "background": False},
                "traits": [], "nicknames": [], "trust": 0, "mood": "neutral",
            },
        }
        self.session["selected_npc_id"] = "keeper"
        state = self._context_state(
            "I ask the keeper where Old Gurteen lives.")

        with patch("rag.hybrid_search", return_value=[]):
            assemble_context(state)

        self.assertTrue(
            self.session["npc_states"]["Old Gurteen"]["dynamic"]
            ["disclosure"]["name"])
        self.assertIn(
            "Old Gurteen",
            _redact_unrevealed_names(
                "Old Gurteen's cottage is nearby.", self.session,
                self.entity_index),
        )

    def test_player_cannot_unlock_unencountered_name_by_guessing(self):
        self.world["entities"]["elias_marsh"] = {
            "type": "npc", "name": "Elias Marsh", "scene": "lighthouse",
            "initial_state": "present", "description": "A distant keeper.",
        }
        self.entity_index["elias_marsh"] = dict(
            self.world["entities"]["elias_marsh"])
        self.session["entity_states"]["elias_marsh"] = "present"
        self.session["npc_states"]["Elias Marsh"] = {
            "static": {"name": "Elias Marsh", "dialogue": {}},
            "dynamic": {
                "disclosure": {"name": False, "background": False},
                "traits": [], "nicknames": [], "trust": 0, "mood": "neutral",
            },
        }
        self.session["selected_npc_id"] = "keeper"

        with patch("rag.hybrid_search", return_value=[]):
            assemble_context(self._context_state(
                "I ask whether Elias Marsh is at the lighthouse."))

        self.assertFalse(
            self.session["npc_states"]["Elias Marsh"]["dynamic"]
            ["disclosure"]["name"])

    def test_player_cannot_unlock_hidden_npc_by_guessing_name(self):
        self.session["npc_states"]["Marsh Lurker"] = {
            "static": {"name": "Marsh Lurker", "dialogue": {}},
            "dynamic": {
                "disclosure": {"name": False, "background": False},
                "traits": [], "nicknames": [], "trust": 0, "mood": "neutral",
            },
        }
        self.session["selected_npc_id"] = "keeper"
        state = self._context_state(
            "I ask the keeper whether Marsh Lurker is nearby.")

        with patch("rag.hybrid_search", return_value=[]):
            assemble_context(state)

        self.assertFalse(
            self.session["npc_states"].get("Marsh Lurker", {})
            .get("dynamic", {}).get("disclosure", {}).get("name", False))

    def test_selected_npc_id_is_the_only_dialogue_target(self):
        select_interaction_target(
            self.session, "keeper", self.world, self.scene_index, self.entity_index)
        state = self._context_state("I ask him about the light")

        with patch("rag.hybrid_search", return_value=[]):
            result = assemble_context(state)

        self.assertEqual(["keeper"], result["_matched_npc_ids"])
        self.assertFalse(result.get("_npc_selection_required", False))
        self.assertIn("ACTIVE NPC TARGET", result["context_prompt"])
        self.assertIn("'keeper'", result["context_prompt"])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from action_system import detect_action, parse_ai_proposal, plan_action, validate_action
from engine import narration_provider, run_gm_turn
from models import create_session
from parser import (
    _apply_numbered_scene_edges, _apply_scene_coverage_repair,
    _recover_source_marked_entities, _scene_coverage_candidates,
)
from scene_system import list_available_scenes, select_scene_target
from scene_index import build_entity_index, build_scene_index
from server import app
from state_manager import initialize_session_from_world
from world_state import (
    append_world_event,
    list_interactable_objects,
    select_object_target,
)


def build_world():
    return {
        "name": "Object Facts",
        "rule_system": "coc",
        "starting_scene": "study",
        "opening": "A brass coin rests on the desk.",
        "scenes": {
            "study": {
                "name": "Study", "desc": "A small study.",
                "exits": {"hall": "hall"},
            },
            "hall": {
                "name": "Hall", "desc": "A narrow hall.",
                "exits": {"study": "study"},
            },
        },
        "entities": {
            "coin": {
                "type": "item", "name": "brass coin", "scene": "study",
                "initial_state": "present", "aliases": ["coin"],
            },
            "potion": {
                "type": "item", "name": "red potion", "scene": "study",
                "initial_state": "obtained", "consumable": True,
            },
            "door": {
                "type": "door", "name": "cellar door", "scene": "hall",
                "initial_state": "locked", "portable": False,
                "requires_key": "coin",
            },
            "secret": {
                "type": "item", "name": "black crown", "scene": "study",
                "initial_state": "hidden",
            },
        },
    }


class FactEventTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world()
        self.scene_index = build_scene_index(self.world)
        self.entity_index = build_entity_index(self.world)
        self.session = create_session("object-facts", self.world["name"])
        initialize_session_from_world(self.session, self.world)

    def roster(self):
        return list_interactable_objects(
            self.session, self.world, self.scene_index, self.entity_index)

    def test_hidden_objects_are_not_candidates(self):
        ids = {item["id"] for item in self.roster()}

        self.assertIn("coin", ids)
        self.assertIn("potion", ids)
        self.assertNotIn("secret", ids)

    def test_action_words_do_not_match_inside_names_or_other_verbs(self):
        for text in (
            "go to the lighthouse", "return to Sir Servause",
            "refuse the quest", "accuse the villagers",
        ):
            with self.subTest(text=text):
                self.assertEqual("", detect_action(text))

    def test_specific_attack_overrides_generic_use(self):
        self.assertEqual("break", detect_action("我用力量攻击墙板"))

    def test_pickup_removes_object_from_scene_and_carries_across_scenes(self):
        append_world_event(self.session, self.world, {
            "type": "item_picked_up", "entity_id": "coin",
        })
        self.session["player_state"]["current_scene"] = "hall"

        roster = {item["id"]: item for item in self.roster()}

        self.assertEqual("inventory", roster["coin"]["location"]["kind"])
        self.assertIn("coin", self.session["inventory_entity_ids"])
        self.assertEqual(["red potion", "brass coin"],
                         self.session["player_state"]["inventory"])
        self.session["player_state"]["current_scene"] = "study"
        study_scene_objects = [
            item["id"] for item in self.roster()
            if item["location"]["kind"] == "scene"]
        self.assertNotIn("coin", study_scene_objects)

    def test_drop_moves_object_to_current_scene(self):
        append_world_event(self.session, self.world, {
            "type": "item_picked_up", "entity_id": "coin",
        })
        self.session["player_state"]["current_scene"] = "hall"
        append_world_event(self.session, self.world, {
            "type": "item_dropped", "entity_id": "coin", "scene_id": "hall",
        })

        coin = next(item for item in self.roster() if item["id"] == "coin")
        self.assertEqual({"kind": "scene", "id": "hall"}, coin["location"])
        self.assertNotIn("coin", self.session["inventory_entity_ids"])

    def test_consumed_item_leaves_inventory_and_roster(self):
        append_world_event(self.session, self.world, {
            "type": "item_used", "entity_id": "potion", "consumed": True,
        })

        self.assertNotIn("potion", self.session["inventory_entity_ids"])
        self.assertNotIn("potion", {item["id"] for item in self.roster()})
        self.assertEqual("consumed", self.session["entity_states"]["potion"])

    def test_selected_object_resolves_pronoun_without_ai(self):
        select_object_target(
            self.session, "coin", self.world, self.scene_index,
            self.entity_index)

        proposal = plan_action(
            "pick it up", self.session, self.world, self.scene_index,
            self.entity_index)
        resolution = validate_action(proposal, self.session, self.world)

        self.assertEqual("coin", proposal["target_id"])
        self.assertEqual("selected_object", proposal["source"])
        self.assertEqual("accepted", resolution["status"])
        self.assertEqual("item_picked_up", resolution["events"][0]["type"])

    def test_ai_cannot_select_an_id_outside_closed_candidates(self):
        proposal = parse_ai_proposal(
            '{"intent":"take","target_id":"invented_relic",'
            '"tool_id":"secret","confidence":1,"ambiguous":false}',
            {"coin", "potion"},
        )

        self.assertEqual("", proposal["target_id"])
        self.assertEqual("", proposal["tool_id"])

    def test_unresolved_mutating_action_is_blocked_before_narration(self):
        proposal = plan_action(
            "take the laser rifle", self.session, self.world,
            self.scene_index, self.entity_index,
            ai_planner=lambda _prompt: (
                '{"intent":"take","target_id":"laser_rifle",'
                '"tool_id":"","confidence":1,"ambiguous":false}'),
        )

        resolution = validate_action(proposal, self.session, self.world)

        self.assertEqual("", proposal["target_id"])
        self.assertEqual("ambiguous", resolution["status"])

    def test_scene_wide_destruction_requires_a_grounded_target(self):
        for text in (
            "我已经烧毁了整座公馆", "我杀死这里所有人", "I blow up the building",
            "I already murdered everyone in the room",
        ):
            with self.subTest(text=text):
                proposal = plan_action(
                    text, self.session, self.world,
                    self.scene_index, self.entity_index)
                resolution = validate_action(proposal, self.session, self.world)
                self.assertEqual("break", proposal["intent"])
                self.assertIn(resolution["status"], {"ambiguous", "blocked"})

    def test_selected_object_cannot_be_destroyed_by_narration_alone(self):
        select_object_target(
            self.session, "coin", self.world, self.scene_index,
            self.entity_index)

        proposal = plan_action(
            "smash it", self.session, self.world,
            self.scene_index, self.entity_index)
        resolution = validate_action(proposal, self.session, self.world)

        self.assertEqual("coin", proposal["target_id"])
        self.assertEqual("blocked", resolution["status"])

    def test_abstract_action_is_left_for_open_adjudication(self):
        proposal = plan_action(
            "use the plan against the guards", self.session, self.world,
            self.scene_index, self.entity_index)

        resolution = validate_action(proposal, self.session, self.world)

        self.assertEqual("passthrough", resolution["status"])

    def test_ambiguous_alias_requires_selection(self):
        self.world["entities"]["silver_coin"] = {
            "type": "item", "name": "silver token", "scene": "study",
            "initial_state": "present", "aliases": ["coin"],
        }
        self.scene_index = build_scene_index(self.world)
        self.entity_index = build_entity_index(self.world)

        proposal = plan_action(
            "take the coin", self.session, self.world, self.scene_index,
            self.entity_index)
        resolution = validate_action(proposal, self.session, self.world)

        self.assertTrue(proposal["ambiguous"])
        self.assertEqual("ambiguous", resolution["status"])


class ObjectTargetApiTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world()
        self.scene_index = build_scene_index(self.world)
        self.entity_index = build_entity_index(self.world)
        self.session = create_session("object-api", self.world["name"])
        initialize_session_from_world(self.session, self.world)
        self.context = (
            self.session, self.world, self.scene_index, self.entity_index)
        self.client = TestClient(app)

    def test_object_roster_and_target_validation(self):
        with patch("server._npc_roster_context", return_value=self.context), \
                patch("state_manager.save_session"):
            roster = self.client.get("/api/session/object-api/objects")
            selected = self.client.post(
                "/api/session/object-api/object-target",
                json={"object_id": "coin"})
            rejected = self.client.post(
                "/api/session/object-api/object-target",
                json={"object_id": "secret"})

        self.assertEqual(200, roster.status_code)
        self.assertEqual({"coin", "potion"},
                         {item["id"] for item in roster.json()["objects"]})
        self.assertEqual("coin", selected.json()["selected_object_id"])
        self.assertEqual(400, rejected.status_code)


class SceneSystemTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world()
        self.world["scenes"]["study"]["exits"]["secret stairs"] = {
            "target": "cellar", "hidden": True,
        }
        self.world["scenes"]["cellar"] = {
            "name": "Secret Cellar", "desc": "A hidden cellar.", "exits": {},
        }
        self.scene_index = build_scene_index(self.world)
        self.entity_index = build_entity_index(self.world)
        self.session = create_session("scene-system", self.world["name"])
        initialize_session_from_world(self.session, self.world)

    def test_roster_exposes_adjacent_scene_but_not_hidden_exit(self):
        roster = list_available_scenes(self.session, self.world)

        self.assertEqual(["hall"], [scene["id"] for scene in roster])
        self.assertIn("study", self.session["visited_scene_ids"])
        self.assertIn("hall", self.session["discovered_scene_ids"])
        self.assertNotIn("cellar", self.session["discovered_scene_ids"])

    def test_hidden_exit_appears_after_unlock(self):
        self.session["unlocked_scenes"].append("cellar")

        roster = list_available_scenes(self.session, self.world)

        self.assertEqual({"hall", "cellar"}, {scene["id"] for scene in roster})

    def test_scene_target_api_rejects_non_adjacent_scene(self):
        context = (self.session, self.world, self.scene_index, self.entity_index)
        client = TestClient(app)
        with patch("server._npc_roster_context", return_value=context), \
                patch("state_manager.save_session"):
            roster = client.get("/api/session/scene-system/scenes")
            selected = client.post(
                "/api/session/scene-system/scene-target", json={"scene_id": "hall"})
            rejected = client.post(
                "/api/session/scene-system/scene-target", json={"scene_id": "cellar"})

        self.assertEqual(["hall"], [item["id"] for item in roster.json()["scenes"]])
        self.assertEqual("hall", selected.json()["selected_scene_id"])
        self.assertEqual(400, rejected.status_code)

    def test_explicit_scene_target_moves_without_narrator_marker(self):
        select_scene_target(self.session, "hall", self.world)
        with patch("engine.load_world", return_value=self.world), \
                patch("engine.get_indices", return_value=(
                    self.scene_index, self.entity_index)), \
                patch("engine.get_session", return_value=self.session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                patch("npc_context.compress_story"), \
                narration_provider(lambda _request: "The hall lies beyond the door."):
            run_gm_turn(
                [{"role": "user", "content": "go there"}],
                model=self.world["name"], chat_id=self.session["chat_id"],
                api_key="manual-provider-no-api-key",
            )

        self.assertEqual("hall", self.session["player_state"]["current_scene"])
        self.assertIn("hall", self.session["visited_scene_ids"])
        self.assertIsNone(self.session["selected_scene_id"])

    def test_authored_aggressive_exit_is_not_blocked_as_ungrounded_damage(self):
        self.world["scenes"]["study"]["exits"] = {
            "attack the guards": "hall",
            "secret stairs": {"target": "cellar", "hidden": True},
        }
        with patch("engine.load_world", return_value=self.world), \
                patch("engine.get_indices", return_value=(
                    self.scene_index, self.entity_index)), \
                patch("engine.get_session", return_value=self.session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                patch("engine.get_or_compress_conversation_summary",
                      return_value=""), \
                narration_provider(lambda _request: (
                    "The confrontation begins.\n〔前往：hall〕")):
            response = run_gm_turn(
                [{"role": "user", "content": (
                    "We accuse the guards and attack.")}],
                model=self.world["name"], chat_id=self.session["chat_id"],
                api_key="manual-provider-no-api-key",
            )

        self.assertEqual("hall", self.session["player_state"]["current_scene"])
        self.assertIn("confrontation", response)

    def test_narrator_marker_cannot_reveal_hidden_exit(self):
        with patch("engine.load_world", return_value=self.world), \
                patch("engine.get_indices", return_value=(
                    self.scene_index, self.entity_index)), \
                patch("engine.get_session", return_value=self.session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                narration_provider(
                    lambda _request: "A stair opens. 〔前往：cellar〕"):
            response = run_gm_turn(
                [{"role": "user", "content": "I go through the secret stairs."}],
                model=self.world["name"], chat_id=self.session["chat_id"],
                api_key="manual-provider-no-api-key",
            )

        self.assertEqual("study", self.session["player_state"]["current_scene"])
        self.assertNotIn("cellar", self.session["visited_scene_ids"])
        self.assertNotIn("〔前往", response)

    def test_narration_cannot_claim_arrival_in_invented_scene(self):
        with patch("engine.load_world", return_value=self.world), \
                patch("engine.get_indices", return_value=(
                    self.scene_index, self.entity_index)), \
                patch("engine.get_session", return_value=self.session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                narration_provider(lambda _request: (
                    "You arrive inside the invented underground command room.")):
            response = run_gm_turn(
                [{"role": "user", "content": (
                    "I walk through the wall into the underground command room.")}],
                model=self.world["name"], chat_id=self.session["chat_id"],
                api_key="manual-provider-no-api-key",
            )

        self.assertEqual("study", self.session["player_state"]["current_scene"])
        self.assertIn("当前可达场景", response)
        self.assertNotIn("command room", response)


class SceneCoverageRepairTests(unittest.TestCase):
    def test_repair_uses_only_source_candidate_content(self):
        source = (
            "SCENE 1: STUDY\n" + "Dust covers the study walls. " * 8
            + "\nSCENE 2: OLD CELLAR\n" + "Cold water drips in the old cellar. " * 8
        )
        study_end = source.index("SCENE 2: OLD CELLAR")
        existing = [{
            "id": "study", "name": "STUDY",
            "source_text": source[:study_end].strip(),
        }]
        candidates = _scene_coverage_candidates(source, existing)
        cellar = next(item for item in candidates if "CELLAR" in item["name"])

        additions = _apply_scene_coverage_repair(existing, candidates, {
            "additions": [
                {"candidate_start": cellar["start"], "name": "Invented Palace"},
                {"candidate_start": 999999, "name": "Fabricated Scene"},
            ],
        })

        self.assertEqual(1, len(additions))
        self.assertEqual("OLD CELLAR", additions[0]["name"])
        self.assertIn("Cold water drips", additions[0]["source_text"])
        self.assertNotIn("Invented Palace", str(additions[0]))

    def test_numbered_solo_nodes_are_eligible_for_grounded_repair(self):
        choices = " ".join(f"go to {number}" for number in range(1, 12))
        source = (
            f"Introduction\n{choices}\n"
            "1\nYou wake beside the road and may enter the village. "
            + "The choice remains uncertain. " * 5
            + "\n2\nA coach waits beside the next crossroads. "
            + "The driver watches the road. " * 5
        )

        candidates = _scene_coverage_candidates(source, [])

        numbered = {item["name"] for item in candidates if item["kind_hint"] == "event"}
        self.assertIn("Section 1", numbered)
        self.assertIn("Section 2", numbered)

    def test_numbered_solo_edges_come_from_source_references(self):
        scenes = {
            "section_1": {
                "source_section_number": 1,
                "source_text": "If you enter the coach, go to 2. Otherwise turn to 3.",
                "exits": {},
            },
            "section_2": {
                "source_section_number": 2, "source_text": "The coach leaves.",
            },
            "section_3": {
                "source_section_number": 3, "source_text": "You remain behind.",
            },
        }

        _apply_numbered_scene_edges(scenes)

        self.assertEqual({
            "Section 2": "section_2", "Section 3": "section_3",
        }, scenes["section_1"]["exits"])

    def test_explicit_source_markers_recover_entities_without_invention(self):
        source = (
            "【公馆大厅】\n" + "大厅里回荡着脚步声。" * 8
            + "\n👤 珍妮：正在擦洗墙面的女仆。"
            + "\n📖 铁制钥匙：可以打开后门。"
        )
        pass1 = {
            "scenes": [{
                "id": "hall", "name": "公馆大厅", "source_start": 0,
                "source_text": source,
            }],
            "npcs": [], "items": [],
        }

        counts = _recover_source_marked_entities(pass1, source)

        self.assertEqual({"npcs": 1, "items": 1}, counts)
        self.assertEqual("珍妮", pass1["npcs"][0]["name"])
        self.assertEqual("hall", pass1["npcs"][0]["scene"])
        self.assertEqual("hidden", pass1["items"][0]["initial_state"])
        self.assertIn("铁制钥匙", pass1["items"][0]["source_quote"])


class HybridEngineActionTests(unittest.TestCase):
    def test_narrator_cannot_invent_catastrophic_world_mutation(self):
        world = build_world()
        scene_index = build_scene_index(world)
        entity_index = build_entity_index(world)
        session = create_session("hostile-catastrophe", world["name"])
        initialize_session_from_world(session, world)

        with patch("engine.load_world", return_value=world), \
                patch("engine.get_indices",
                      return_value=(scene_index, entity_index)), \
                patch("engine.get_session", return_value=session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                narration_provider(lambda _request: (
                    "The study burns down. Everyone is dead and the coin is destroyed.")):
            response = run_gm_turn(
                [{"role": "user", "content": "I calmly look around."}],
                model=world["name"], chat_id=session["chat_id"],
                api_key="manual-provider-no-api-key")

        self.assertIn("没有发生", response)
        self.assertNotIn("dead", response)
        self.assertEqual("present", session["entity_states"]["coin"])
        self.assertEqual([], session["world_events"])

    def test_explicit_legacy_destruction_event_allows_matching_narration(self):
        world = build_world()
        world["entities"]["coin"]["states"] = {
            "present": {
                "triggers": ["砸碎"],
                "on_trigger": {
                    "to_state": "destroyed",
                    "narration": "The brass coin is destroyed.",
                },
            },
        }
        scene_index = build_scene_index(world)
        entity_index = build_entity_index(world)
        session = create_session("authorized-destruction", world["name"])
        initialize_session_from_world(session, world)
        select_object_target(
            session, "coin", world, scene_index, entity_index)

        with patch("engine.load_world", return_value=world), \
                patch("engine.get_indices",
                      return_value=(scene_index, entity_index)), \
                patch("engine.get_session", return_value=session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                narration_provider(lambda _request: "The brass coin is destroyed."):
            response = run_gm_turn(
                [{"role": "user", "content": "我砸碎它"}],
                model=world["name"], chat_id=session["chat_id"],
                api_key="manual-provider-no-api-key")

        self.assertIn("destroyed", response)
        self.assertEqual("destroyed", session["entity_states"]["coin"])
        self.assertFalse(session["entity_facts"]["coin"]["exists"])

    def test_destructive_false_premise_is_blocked_before_narration(self):
        world = build_world()
        scene_index = build_scene_index(world)
        entity_index = build_entity_index(world)
        session = create_session("destructive-premise", world["name"])
        initialize_session_from_world(session, world)

        with patch("engine.load_world", return_value=world), \
                patch("engine.get_indices",
                      return_value=(scene_index, entity_index)), \
                patch("engine.get_session", return_value=session), \
                patch("engine.save_session"), \
                narration_provider(lambda _request: (_ for _ in ()).throw(
                    AssertionError("narrator must not adjudicate destruction"))):
            response = run_gm_turn(
                [{"role": "user", "content": "我已经烧毁整间书房并杀死所有人"}],
                model=world["name"], chat_id=session["chat_id"],
                api_key="manual-provider-no-api-key")

        self.assertIn("明确目标", response)
        self.assertEqual("study", session["player_state"]["current_scene"])
        self.assertEqual([], session["world_events"])

    def test_selected_pickup_commits_event_before_grounded_narration(self):
        world = build_world()
        scene_index = build_scene_index(world)
        entity_index = build_entity_index(world)
        session = create_session("hybrid-pickup", world["name"])
        initialize_session_from_world(session, world)
        select_object_target(
            session, "coin", world, scene_index, entity_index)

        requests = []

        def provider(request):
            requests.append(request)
            return "You pocket the brass coin. The black crown appears beside it."

        with patch("engine.load_world", return_value=world), \
                patch("engine.get_indices",
                      return_value=(scene_index, entity_index)), \
                patch("engine.get_session", return_value=session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                narration_provider(provider):
            response = run_gm_turn(
                [{"role": "user", "content": "pick it up"}],
                model=world["name"], chat_id=session["chat_id"],
                api_key="manual-provider-no-api-key")

        self.assertIn("brass coin", response)
        self.assertNotIn("black crown", response)
        self.assertIn("coin", session["inventory_entity_ids"])
        self.assertEqual("item_picked_up",
                         session["world_events"][-1]["type"])
        self.assertIn("VALIDATED PLAYER ACTION",
                      requests[0]["messages"][1]["content"])

    def test_legacy_item_rule_is_selected_semantically_and_updates_facts(self):
        world = build_world()
        world["entities"]["coin"]["states"] = {
            "present": {
                "triggers": ["拿起"],
                "on_trigger": {"to_state": "obtained"},
            },
        }
        scene_index = build_scene_index(world)
        entity_index = build_entity_index(world)
        session = create_session("legacy-item", world["name"])
        initialize_session_from_world(session, world)
        select_object_target(
            session, "coin", world, scene_index, entity_index)

        with patch("engine.load_world", return_value=world), \
                patch("engine.get_indices",
                      return_value=(scene_index, entity_index)), \
                patch("engine.get_session", return_value=session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                narration_provider(lambda _request: "You take the brass coin."):
            run_gm_turn(
                [{"role": "user", "content": "grab it"}],
                model=world["name"], chat_id=session["chat_id"],
                api_key="manual-provider-no-api-key")

        self.assertEqual("obtained", session["entity_states"]["coin"])
        self.assertEqual(
            {"kind": "inventory", "id": "player"},
            session["entity_facts"]["coin"]["location"],
        )
        self.assertEqual(1, len([
            event for event in session["world_events"]
            if event["entity_id"] == "coin"]),
        )

    def test_legacy_check_still_waits_for_player_roll(self):
        world = build_world()
        world["entities"]["coin"]["states"] = {
            "present": {
                "triggers": ["检查"],
                "check": "侦查",
                "on_pass": {"to_state": "found"},
                "on_fail": {"to_state": "present"},
            },
        }
        scene_index = build_scene_index(world)
        entity_index = build_entity_index(world)
        session = create_session("legacy-check", world["name"])
        initialize_session_from_world(session, world)
        select_object_target(
            session, "coin", world, scene_index, entity_index)

        with patch("engine.load_world", return_value=world), \
                patch("engine.get_indices",
                      return_value=(scene_index, entity_index)), \
                patch("engine.get_session", return_value=session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                narration_provider(lambda _request: "Make a Spot Hidden check."):
            run_gm_turn(
                [{"role": "user", "content": "examine it"}],
                model=world["name"], chat_id=session["chat_id"],
                api_key="manual-provider-no-api-key")

        self.assertIsNotNone(session["pending_check"])
        self.assertEqual("coin", session["pending_check"]["entity_id"])
        self.assertEqual([], session["world_events"])


if __name__ == "__main__":
    unittest.main()

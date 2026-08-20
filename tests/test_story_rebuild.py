import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from action_system import plan_action, validate_action
from engine import _build_current_story_node_block, narration_provider, run_gm_turn
from models import create_session
from parser import (
    ModuleParser, _assemble_world_book, _bind_document_provenance,
    _bind_perception_entity_instances,
    _catalog_windows, _detect_rule_profile, _focused_source_excerpt,
    _merge_story_node_details, _prepare_full_rebuild,
    _normalize_story_node_kinds,
    _score_global_reconstruction, _score_story_node_detail,
    _select_node_evidence,
)
from scene_index import build_entity_index, build_scene_index
from server import _completed_parse_job
from state_manager import initialize_session_from_world
from world_state import append_world_event


def reconstructed_module():
    return {
        "overview": {
            "title": "Nested Test",
            "mystery": "A missing letter.",
            "opening": "You enter the study.",
            "starting_scene": "study",
            "starting_node": "study_node",
            "rule_system": "coc",
        },
        "story_spine": {
            "premise": "Find the letter.",
            "truth": "It is inside the bookshelf.",
            "timeline": ["The letter was hidden."],
            "invariants": ["The storybook is fiction inside the module."],
        },
        "narrative_scopes": [
            {"id": "physical", "kind": "physical", "navigable": True},
            {"id": "storybook", "kind": "document", "parent_scope": "physical",
             "navigable": False},
        ],
        "entity_registry": [],
        "story_tree": {
            "root_id": "root",
            "nodes": [
                {"id": "root", "parent_id": "", "children": [
                    "study_node", "hall_node", "book_node"],
                 "title": "Root", "kind": "root", "scope_id": "physical",
                 "playable": False, "summary": "Whole story.",
                 "source_refs": [0], "preconditions": [], "outcomes": [],
                 "successors": [], "expected_facets": {}},
                {"id": "study_node", "parent_id": "root", "children": [],
                 "title": "Search the study", "kind": "scene",
                 "scope_id": "physical", "playable": True,
                 "summary": "Search for the letter.", "source_refs": [0],
                 "preconditions": [], "outcomes": ["study searched"],
                 "successors": ["hall_node"],
                 "expected_facets": {"scenes": 1, "objects": 2,
                                     "state_changes": 1}},
                {"id": "hall_node", "parent_id": "root", "children": [],
                 "title": "Enter the hall", "kind": "scene",
                 "scope_id": "physical", "playable": True,
                 "summary": "Continue into the hall.", "source_refs": [0],
                 "preconditions": ["study searched"], "outcomes": [],
                 "successors": [], "expected_facets": {"scenes": 1}},
                {"id": "book_node", "parent_id": "root", "children": [],
                 "title": "Story inside the book", "kind": "event",
                 "scope_id": "storybook", "playable": False,
                 "summary": "Embedded fiction only.", "source_refs": [0],
                 "preconditions": [], "outcomes": [], "successors": [],
                 "expected_facets": {}},
            ],
            "relations": [
                {"from": "study_node", "to": "hall_node", "type": "before"}
            ],
        },
        "scenes": [
            {"id": "study", "name": "Study", "scope_id": "physical",
             "navigable": True, "desc": "A study.", "source_quote": "You enter the study."},
            {"id": "hall", "name": "Hall", "scope_id": "physical",
             "navigable": True, "desc": "A hall.", "source_quote": "The hall is quiet."},
            {"id": "black_forest", "name": "Black Forest", "scope_id": "storybook",
             "navigable": False, "desc": "A castle rises in the book.",
             "source_quote": "In the story, a castle rises in the Black Forest."},
        ],
        "npcs": [],
        "objects": [
            {"id": "door", "name": "wooden door", "type": "door", "scene": "study",
             "scope_id": "physical", "source_start": 10,
             "source_quote": "A wooden door leads to the hall."},
            {"id": "door", "name": "wooden door", "type": "door", "scene": "hall",
             "scope_id": "physical", "source_start": 80,
             "source_quote": "Another wooden door closes the hall."},
            {"id": "bookshelf", "name": "bookshelf", "type": "object", "scene": "study",
             "scope_id": "physical", "portable": False,
             "interactions": {"inspect": "Dust hides a narrow gap."},
             "source_quote": "A bookshelf covers the west wall."},
            {"id": "castle_gate", "name": "castle gate", "type": "door",
             "scene": "black_forest", "scope_id": "storybook",
             "source_quote": "The hero opens the castle gate."},
        ],
        "clues": [],
        "events": [],
        "scene_graph": {
            "study": {"exits": {"hall": "hall", "enter the book": "black_forest"}},
            "hall": {"exits": {"study": "study", "invented": "moon_base"}},
        },
        "story_beats": [{
            "id": "search", "name": "Search", "scenes": ["study", "black_forest"],
            "critical_clues": [], "optional_clues": [], "advance_when": "visited",
            "unlocks_scenes": ["hall", "black_forest"],
        }],
    }


def detailed_story_nodes():
    return [
        {
            "node_id": "study_node",
            "node_summary": "The investigators search the study and establish the route to the hall.",
            "scenes": [{"id": "study", "name": "Study", "scope_id": "physical",
                        "navigable": True, "desc": "A study.", "source_ref": 0,
                        "source_quote": "You enter the study."}],
            "npcs": [],
            "objects": [
                {"id": "door", "name": "wooden door", "type": "door",
                 "scene": "study", "scope_id": "physical", "source_ref": 0,
                 "source_quote": "A wooden door leads to the hall."},
                {"id": "bookshelf", "name": "bookshelf", "type": "object",
                 "scene": "study", "scope_id": "physical", "portable": False,
                 "interactions": {"inspect": "Dust hides a narrow gap."},
                 "source_ref": 0,
                 "source_quote": "A bookshelf covers the west wall."},
            ],
            "clues": [], "events": [],
            "state_transitions": [{"subject_id": "study_node",
                                   "dimension": "condition", "before": "unsearched",
                                   "after": "searched", "condition": "search completed",
                                   "source_ref": 0,
                                   "source_quote": "A bookshelf covers the west wall."}],
            "knowledge_changes": [], "promises_payoffs": [], "branch_edges": [],
        },
        {
            "node_id": "hall_node",
            "node_summary": "The investigators enter the quiet hall after leaving the study.",
            "scenes": [{"id": "hall", "name": "Hall", "scope_id": "physical",
                        "navigable": True, "desc": "A hall.", "source_ref": 0,
                        "source_quote": "The hall is quiet."}],
            "npcs": [],
            "objects": [{"id": "door", "name": "wooden door", "type": "door",
                         "scene": "hall", "scope_id": "physical", "source_ref": 0,
                         "source_quote": "Another wooden door closes the hall."}],
            "clues": [], "events": [], "state_transitions": [],
            "knowledge_changes": [], "promises_payoffs": [], "branch_edges": [],
        },
    ]


class FullStoryRebuildTests(unittest.TestCase):
    def test_long_document_maps_every_window_before_global_synthesis(self):
        parser = object.__new__(ModuleParser)
        parser.model = "mock"
        parser._last_error = None
        mapped_source = []

        def fake_llm(system, user, **kwargs):
            if system.startswith("You are mapping ONE ordered window"):
                payload = json.loads(user.split("\n\n", 1)[0])
                mapped_source.append(user)
                ref = payload["allowed_source_refs"][0]
                return json.dumps({
                    "document_roles": [{"source_refs": [ref], "kind": "adventure",
                                        "scenario_id": "test", "summary": "part"}],
                    "story_fragments": [{"scenario_id": "test", "title": "Test",
                                         "scope_id": "physical", "source_refs": [ref],
                                         "summary": "part", "preconditions": [],
                                         "outcomes": [], "choices": [], "links_to": []}],
                    "entity_mentions": [], "scope_mentions": [],
                    "unresolved_links": [],
                })
            rebuilt = reconstructed_module()
            rebuilt["scenarios"] = [{"id": "test", "title": "Test",
                                      "root_node_id": "root",
                                      "starting_node": "study_node"}]
            for node in rebuilt["story_tree"]["nodes"]:
                node["scenario_id"] = "test"
            return json.dumps(rebuilt)

        parser._llm = fake_llm
        text = (
            "SCENE 1: Opening\n" + "opening evidence " * 300
            + "\nSCENE 2: Middle\n" + "middle evidence " * 300
            + "\nSCENE 3: Ending\n" + "ending evidence " * 300
            + "FINAL_LONG_DOCUMENT_PAYOFF"
        )
        with patch("parser.FULL_REBUILD_MAX_CHARS", 5000), \
                patch("parser.LONG_DOCUMENT_WINDOW_CHARS", 4000):
            result = parser.pass_full_rebuild(text)

        self.assertEqual("hierarchical_document_map",
                         result["_planning"]["method"])
        self.assertGreater(result["_planning"]["window_count"], 1)
        self.assertTrue(any("FINAL_LONG_DOCUMENT_PAYOFF" in row
                            for row in mapped_source))
        later_payload = json.loads(mapped_source[1].split("\n\n", 1)[0])
        self.assertEqual("test", later_payload["known_scenarios"][0]["id"])

    def test_long_document_rejects_invented_synthesis_source_ref(self):
        parser = object.__new__(ModuleParser)
        parser.model = "mock"
        parser._last_error = None

        def fake_llm(system, user, **kwargs):
            if system.startswith("You are mapping ONE ordered window"):
                payload = json.loads(user.split("\n\n", 1)[0])
                ref = payload["allowed_source_refs"][0]
                return json.dumps({
                    "document_roles": [{"source_refs": [ref], "kind": "adventure"}],
                    "story_fragments": [], "entity_mentions": [],
                    "scope_mentions": [], "unresolved_links": [],
                })
            result = reconstructed_module()
            result["story_tree"]["nodes"][0]["source_refs"] = [999999]
            return json.dumps(result)

        parser._llm = fake_llm
        with patch("parser.LONG_DOCUMENT_WINDOW_CHARS", 4000):
            result = parser.pass_long_document_rebuild(
                "SCENE 1: Test\n" + "source " * 1000)

        self.assertEqual({}, result)

    def test_catalog_windows_preserve_oversized_section_content(self):
        body = "0123456789" * 1000
        windows = _catalog_windows(
            [{"start": 7, "title": "One giant section", "text": body}], 4000)

        self.assertGreater(len(windows), 1)
        self.assertEqual(body, "".join(
            segment["text"] for window in windows for segment in window))
        self.assertEqual({7}, {
            segment["start"] for window in windows for segment in window})

    def test_ruleset_detection_distinguishes_percentile_games(self):
        self.assertEqual(
            {"ruleset": "runequest", "dice_system": "d100"},
            _detect_rule_profile("RuneQuest Quickstart Rules and Adventure"),
        )
        self.assertEqual(
            {"ruleset": "brp", "dice_system": "d100"},
            _detect_rule_profile("Basic Roleplaying Universal Game Engine"),
        )
        self.assertEqual(
            {"ruleset": "pendragon", "dice_system": "d20"},
            _detect_rule_profile("Pendragon 6th Edition Quick-Start Scenario\nRoll 1D20"),
        )
        self.assertEqual(
            {"ruleset": "7thsea", "dice_system": "custom"},
            _detect_rule_profile("THE SWORD OF KINGS 7TH SEA ADVENTURES"),
        )
        self.assertEqual(
            {"ruleset": "coriolis", "dice_system": "d6_pool"},
            _detect_rule_profile(
                "QUICKSTART\nCoriolis: The Great Dark\nThe Sky Machine"),
        )
        self.assertEqual(
            {"ruleset": "starfinder", "dice_system": "d20"},
            _detect_rule_profile(
                "STARFINDER SECOND EDITION ADVENTURE\nArmor Class 17\nDC 15"),
        )
        self.assertEqual(
            {"ruleset": "other", "dice_system": "custom"},
            _detect_rule_profile("STR DEX CON are ordinary labels without dice rules"),
        )
        self.assertEqual(
            {"ruleset": "brp", "dice_system": "d100"},
            _detect_rule_profile(
                "BASIC ROLEPLAYING QUICKSTART\nHistory mentions RuneQuest and Call of Cthulhu"),
        )

    def test_story_node_kinds_use_closed_annotation_taxonomy(self):
        blueprint = {"story_tree": {"nodes": [
            {"id": "root", "kind": "root", "playable": False},
            {"id": "old-choice", "kind": "decision"},
            {"id": "old-scene", "kind": "investigation"},
            {"id": "unknown", "kind": "set_piece"},
        ]}}

        _normalize_story_node_kinds(blueprint)

        self.assertEqual(
            ["root", "choice", "event", "event"],
            [row["kind"] for row in blueprint["story_tree"]["nodes"]],
        )

    def test_large_section_evidence_keeps_contract_relevant_middle(self):
        source = "start " * 1000 + "THE HIDDEN CEREMONIAL CORRAL" + " end" * 1000

        excerpt = _focused_source_excerpt(
            source,
            {"title": "Ceremonial Corral", "summary": "recover the cattle",
             "preconditions": [], "outcomes": ["cattle recovered"]},
            2400,
        )

        self.assertLessEqual(len(excerpt), 2400)
        self.assertIn("THE HIDDEN CEREMONIAL CORRAL", excerpt)

    def test_full_rebuild_receives_document_ending_beyond_old_pass0_limit(self):
        parser = object.__new__(ModuleParser)
        parser.model = "mock"
        parser._last_error = None
        captured = {}

        def fake_llm(system, user, **kwargs):
            captured["system"] = system
            captured["user"] = user
            return json.dumps(reconstructed_module())

        parser._llm = fake_llm
        text = "You enter the study.\n" + ("middle text\n" * 6500) + "FINAL_CAUSAL_REVEAL"

        result = parser.pass_full_rebuild(text)

        self.assertTrue(result)
        self.assertIn("FINAL_CAUSAL_REVEAL", captured["user"])
        self.assertIn("COMPLETE MODULE DOCUMENT", captured["user"])
        self.assertIn("causal story spine", captured["system"])

    def test_parse_uses_rebuilt_story_without_segmented_storyline_pass(self):
        parser = object.__new__(ModuleParser)
        parser.model = "mock"
        parser._last_error = None
        source = (
            "You enter the study. A wooden door leads to the hall. "
            "A bookshelf covers the west wall. The hall is quiet. "
            "Another wooden door closes the hall. In the story, a castle "
            "rises in the Black Forest. The hero opens the castle gate."
        )
        rebuilt = reconstructed_module()
        _bind_document_provenance(rebuilt, source)
        parser.pass_full_rebuild = Mock(return_value=rebuilt)
        parser.rebuild_story_tree_nodes = Mock(return_value=(
            detailed_story_nodes(),
            {"method": "test", "threshold": 82, "overall": 100,
             "passed": True, "node_count": 2, "failed_node_ids": [], "nodes": []},
        ))
        parser.pass_game_mechanics = Mock()
        parser.pass_npc_storylines = Mock()
        parser.pass_npc_style = Mock()

        world = parser.parse(source)

        self.assertEqual("hierarchical_story_tree_rebuild", world["_parser_mode"])
        self.assertIn("story_spine", world)
        self.assertTrue(world["reconstruction_quality"]["passed"])
        self.assertNotIn("black_forest", world["scenes"])
        self.assertEqual("hall", world["scenes"]["study"]["exits"]["继续：Enter the hall"])
        parser.pass_npc_storylines.assert_not_called()

    def test_embedded_story_scene_never_becomes_runtime_map(self):
        overview, pass1, pass2, embedded = _prepare_full_rebuild(reconstructed_module())
        world = _assemble_world_book(pass1, pass2, overview)

        self.assertEqual({"study", "hall"}, set(world["scenes"]))
        self.assertNotIn("black_forest", world["scenes"])
        self.assertEqual(["hall"], list(world["scenes"]["study"]["exits"].values()))
        self.assertEqual(["study"], list(world["scenes"]["hall"]["exits"].values()))
        embedded_scenes = [
            scene["id"]
            for setting in embedded["embedded_settings"]
            for scene in setting["scenes"]
        ]
        self.assertIn("black_forest", embedded_scenes)
        self.assertEqual(["study"], world["story_beats"][0]["scenes"])
        self.assertEqual(["hall"], world["story_beats"][0]["unlocks_scenes"])

    def test_same_named_doors_are_distinct_scene_instances(self):
        overview, pass1, pass2, _embedded = _prepare_full_rebuild(reconstructed_module())
        world = _assemble_world_book(pass1, pass2, overview)
        doors = {
            eid: entity for eid, entity in world["entities"].items()
            if entity.get("name") == "wooden door"
        }

        self.assertEqual(2, len(doors))
        self.assertEqual({"study", "hall"}, {door["home_scene"] for door in doors.values()})
        self.assertEqual(2, len(set(doors)))

    def test_two_same_room_doors_with_different_evidence_both_survive(self):
        rebuilt = reconstructed_module()
        rebuilt["objects"].append({
            "id": "door", "name": "wooden door", "type": "door", "scene": "study",
            "scope_id": "physical", "source_start": 35,
            "source_quote": "A second wooden door opens to the garden.",
        })
        overview, pass1, pass2, _embedded = _prepare_full_rebuild(rebuilt)
        world = _assemble_world_book(pass1, pass2, overview)

        study_doors = [
            eid for eid, entity in world["entities"].items()
            if entity.get("name") == "wooden door" and entity.get("scene") == "study"
        ]
        self.assertEqual(2, len(study_doors))

    def test_explicit_unique_object_keeps_one_continuous_identity(self):
        rebuilt = reconstructed_module()
        rebuilt["objects"].extend([
            {"id": "red_book", "name": "red book", "type": "document",
             "scene": "study", "scope_id": "physical", "unique_identity": True,
             "source_start": 100, "source_quote": "The red book starts in the study."},
            {"id": "red_book", "name": "red book", "type": "document",
             "scene": "hall", "scope_id": "physical", "unique_identity": True,
             "source_start": 200, "source_quote": "Later, the red book is in the hall."},
        ])
        overview, pass1, pass2, _embedded = _prepare_full_rebuild(rebuilt)
        world = _assemble_world_book(pass1, pass2, overview)
        books = [
            entity for entity in world["entities"].values()
            if entity.get("name") == "red book"
        ]

        self.assertEqual(1, len(books))
        self.assertEqual(["study", "hall"], books[0]["all_scenes"])
        self.assertEqual("red_book", books[0]["continuity_id"])

    def test_bookshelf_is_a_visible_inspectable_object(self):
        overview, pass1, pass2, _embedded = _prepare_full_rebuild(reconstructed_module())
        world = _assemble_world_book(pass1, pass2, overview)
        world.update({"name": "Nested Test", "opening": "You enter the study."})
        scene_index = build_scene_index(world)
        entity_index = build_entity_index(world)
        session = create_session("bookshelf", world["name"])
        initialize_session_from_world(session, world)

        proposal = plan_action(
            "检查bookshelf", session, world, scene_index, entity_index)
        resolution = validate_action(proposal, session, world)

        self.assertEqual("bookshelf", proposal["target_id"])
        self.assertEqual("inspect", proposal["intent"])
        self.assertEqual("accepted", resolution["status"])
        self.assertTrue(resolution["requires_adjudication"])


class HierarchicalReconstructionQualityTests(unittest.TestCase):
    def setUp(self):
        self.source = (
            "## Study\nYou enter the study. A bookshelf covers the west wall. "
            "Searching it reveals a letter.\n"
            "## Hall\nThe hall is quiet."
        )
        self.contract = {
            "id": "study_node", "title": "Search the study", "scope_id": "physical",
            "source_refs": [0], "preconditions": [], "outcomes": ["letter found"],
            "successors": [],
            "expected_facets": {"scenes": 1, "objects": 1, "state_changes": 1},
        }

    def _good_detail(self, evidence):
        source_ref = evidence["source_refs"][0]
        return {
            "node_id": "study_node",
            "node_summary": (
                "The study is searched methodically. The west-wall bookshelf is the "
                "focus of the investigation and finding the concealed letter changes "
                "the investigation state for the following node."
            ),
            "scenes": [{
                "id": "study", "name": "Study", "scope_id": "physical",
                "desc": (
                    "The investigators enter the study and can inspect the west-wall "
                    "bookshelf, where the module places the actionable discovery."
                ),
                "source_ref": source_ref, "source_quote": "You enter the study.",
                "source_verified": True,
            }],
            "npcs": [],
            "objects": [{
                "id": "bookshelf", "name": "bookshelf", "scene": "study",
                "source_ref": source_ref,
                "source_quote": "A bookshelf covers the west wall.",
                "source_verified": True,
            }],
            "clues": [], "events": [],
            "state_transitions": [{
                "subject_id": "letter", "dimension": "knowledge",
                "before": "hidden", "after": "found", "condition": "bookshelf searched",
                "source_ref": source_ref,
                "source_quote": "Searching it reveals a letter.",
                "source_verified": True,
            }],
            "knowledge_changes": [], "promises_payoffs": [], "branch_edges": [],
        }

    def test_node_evidence_uses_closed_catalog_and_reports_bad_refs(self):
        contract = dict(self.contract)
        contract["source_refs"] = [999999]

        evidence = _select_node_evidence(self.source, contract)

        self.assertEqual([999999], evidence["invalid_requested_refs"])
        self.assertNotIn(999999, evidence["source_refs"])
        self.assertTrue(evidence["source_refs"])

    def test_deterministic_score_rejects_unsupported_records(self):
        evidence = _select_node_evidence(self.source, self.contract)
        detail = self._good_detail(evidence)
        detail["objects"][0]["source_verified"] = False

        report = _score_story_node_detail(
            self.contract, detail, evidence, {"study_node"})

        self.assertLess(report["overall"], 100)
        self.assertTrue(any("unverified detailed records" in error
                            for error in report["hard_errors"]))

    def test_deterministic_score_rejects_broken_conditional_chain(self):
        evidence = _select_node_evidence(self.source, self.contract)
        detail = self._good_detail(evidence)
        detail["conditional_events"] = [{
            "id": "idea-gate", "observer_scope": "active_character",
            "when": {"type": "clock_at_least", "value": 8},
            "check": {"type": "skill", "difficulty": "impossible"},
            "outcomes": {
                "success": {"followup_checks": [{"type": "san"}]},
            },
            "source_ref": evidence["source_refs"][0],
            "source_quote": "You enter the study.",
            "source_verified": True,
        }]

        report = _score_story_node_detail(
            self.contract, detail, evidence, {"study_node"})

        self.assertTrue(any("trigger is missing clock_id" in error
                            for error in report["hard_errors"]))
        self.assertTrue(any("is missing skill" in error
                            for error in report["hard_errors"]))
        self.assertTrue(any("unsupported difficulty" in error
                            for error in report["hard_errors"]))
        self.assertTrue(any("is missing SAN loss" in error
                            for error in report["hard_errors"]))

    def test_deterministic_score_rejects_broken_scene_projection(self):
        evidence = _select_node_evidence(self.source, self.contract)
        detail = self._good_detail(evidence)
        detail["perception_layers"] = [{
            "id": "bad-view", "activation": "condition",
            "description_mode": "merge", "description": "",
            "when": {"type": "player_stat", "name": "灵感"},
            "visible_entity_ids": ["door"], "hidden_entity_ids": ["door"],
            "source_ref": evidence["source_refs"][0],
            "source_quote": "You enter the study.", "source_verified": True,
        }]
        detail["conditional_events"] = [{
            "id": "idea", "when": {"type": "always"},
            "check": {"type": "skill", "skill": "Idea"},
            "outcomes": {"success": {
                "activate_perception_layers": ["missing-view"],
            }},
            "source_ref": evidence["source_refs"][0],
            "source_quote": "You enter the study.", "source_verified": True,
        }]

        report = _score_story_node_detail(
            self.contract, detail, evidence, {"study_node"})

        errors = "\n".join(report["hard_errors"])
        self.assertIn("missing scene_id", errors)
        self.assertIn("unsupported description mode", errors)
        self.assertIn("condition is missing value", errors)
        self.assertIn("both reveals and hides", errors)
        self.assertIn("unknown perception layers", errors)

    def test_low_scoring_node_is_rebuilt_with_targeted_feedback(self):
        parser = object.__new__(ModuleParser)
        parser.model = "mock"
        parser._last_error = None
        evidence = _select_node_evidence(self.source, self.contract)
        bad = {"node_id": "study_node", "scenes": [], "objects": [],
               "clues": [], "events": [], "state_transitions": [],
               "knowledge_changes": [], "promises_payoffs": [], "branch_edges": []}
        good = self._good_detail(evidence)
        blueprint = {
            "story_spine": {}, "narrative_scopes": [], "entity_registry": [],
            "story_tree": {"root_id": "study_node", "nodes": [self.contract],
                           "relations": []},
        }
        parser.pass_node_rebuild = Mock(side_effect=[bad, good])
        parser.pass_node_evaluation = Mock(return_value={
            "scores": {"source_fidelity": 100, "causal_completeness": 100,
                       "detail_completeness": 100, "state_tracking": 100,
                       "branch_completeness": 100, "scope_consistency": 100},
            "unsupported_claims": [], "missing_details": [],
            "contradictions": [], "repair_instructions": [],
        })

        details, quality = parser.rebuild_story_tree_nodes(self.source, blueprint)

        self.assertEqual(2, parser.pass_node_rebuild.call_count)
        self.assertIsNotNone(parser.pass_node_rebuild.call_args_list[1].kwargs["feedback"])
        self.assertTrue(quality["passed"])
        self.assertEqual(2, details[0]["_quality"]["attempts"])

    def test_global_quality_rejects_dangling_successor(self):
        blueprint = reconstructed_module()
        blueprint["story_tree"]["nodes"][1]["successors"] = ["missing_node"]
        quality = {"overall": 100, "passed": True, "node_count": 2}

        report = _score_global_reconstruction(blueprint, quality)

        self.assertFalse(report["passed"])
        self.assertLess(report["global_dimensions"]["story_graph_closure"], 100)
        self.assertTrue(any("missing_node" in error for error in report["graph_errors"]))

    def test_global_quality_rejects_cross_scenario_story_edge(self):
        blueprint = reconstructed_module()
        blueprint["scenarios"] = [
            {"id": "first", "root_node_id": "study_node",
             "starting_node": "study_node"},
            {"id": "second", "root_node_id": "hall_node",
             "starting_node": "hall_node"},
        ]
        for node in blueprint["story_tree"]["nodes"]:
            node["scenario_id"] = "first" if node["id"] != "hall_node" else "second"
        quality = {"overall": 100, "passed": True, "node_count": 2}

        report = _score_global_reconstruction(blueprint, quality)

        self.assertFalse(report["passed"])
        self.assertEqual(0, report["global_dimensions"]["scenario_isolation"])
        self.assertTrue(any("cross-scenario" in error
                            for error in report["graph_errors"]))

    def test_anthology_runtime_ids_are_namespaced_per_scenario(self):
        blueprint = {
            "overview": {"title": "Two Stories", "starting_node": "a_node"},
            "story_spine": {}, "narrative_scopes": [], "entity_registry": [],
            "scenarios": [
                {"id": "a", "starting_node": "a_node"},
                {"id": "b", "starting_node": "b_node"},
            ],
            "story_tree": {"root_id": "", "relations": [], "nodes": [
                {"id": "a_node", "scenario_id": "a", "playable": True,
                 "successors": []},
                {"id": "b_node", "scenario_id": "b", "playable": True,
                 "successors": []},
            ]},
        }
        details = [
            {"node_id": "a_node", "scenes": [{"id": "study", "name": "Study"}],
             "npcs": [{"id": "keeper", "name": "Keeper", "scene": "study"}],
             "objects": [{"id": "door", "name": "Door", "scene": "study"}]},
            {"node_id": "b_node", "scenes": [{"id": "study", "name": "Study"}],
             "npcs": [{"id": "keeper", "name": "Keeper", "scene": "study"}],
             "objects": [{"id": "door", "name": "Door", "scene": "study"}]},
        ]

        rebuilt = _merge_story_node_details(blueprint, details, {"passed": True})
        overview, pass1, pass2, _embedded = _prepare_full_rebuild(rebuilt)
        world = _assemble_world_book(pass1, pass2, overview)

        self.assertEqual({"a::study", "b::study"}, set(world["scenes"]))
        self.assertEqual({"a::keeper", "b::keeper"}, {
            npc["id"] for npc in rebuilt["npcs"]})
        self.assertEqual("a::study", rebuilt["scenarios"][0]["starting_scene"])
        self.assertEqual("b::study", rebuilt["scenarios"][1]["starting_scene"])

    def test_merge_distinguishes_critical_and_optional_clues(self):
        blueprint = reconstructed_module()
        details = detailed_story_nodes()
        details[0]["clues"] = [
            {"id": "letter", "critical": True},
            {"id": "dust", "critical": False},
        ]

        rebuilt = _merge_story_node_details(blueprint, details, {"passed": True})
        beat = next(row for row in rebuilt["story_beats"]
                    if row["id"] == "study_node")

        self.assertEqual(["letter"], beat["critical_clues"])
        self.assertEqual(["dust"], beat["optional_clues"])

    def test_merge_links_conditional_outcome_to_perception_layer(self):
        blueprint = reconstructed_module()
        details = detailed_story_nodes()
        details[0]["perception_layers"] = [{
            "id": "hidden-study", "scene_id": "study", "priority": 10,
            "activation": "conditional_outcome", "description_mode": "replace",
            "description": "The study becomes an endless archive.",
            "visible_entity_ids": ["bookshelf"], "hidden_entity_ids": [],
        }]
        details[0]["conditional_events"] = [{
            "id": "idea-check", "when": {"type": "always"},
            "check": {"type": "skill", "skill": "Idea"},
            "outcomes": {"success": {
                "activate_perception_layers": ["hidden-study"],
            }},
        }]

        rebuilt = _merge_story_node_details(blueprint, details, {"passed": True})

        layer = rebuilt["perception_layers"][0]
        event = rebuilt["conditional_events"][0]
        self.assertEqual("study_node::hidden-study", layer["id"])
        self.assertEqual(
            ["study_node::hidden-study"],
            event["outcomes"]["success"]["activate_perception_layers"],
        )

    def test_perception_layer_binds_the_correct_same_named_door_instance(self):
        blueprint = reconstructed_module()
        details = detailed_story_nodes()
        details[0]["perception_layers"] = [{
            "id": "study-view", "scene_id": "study",
            "activation": "condition", "when": {"type": "always"},
            "description": "The study door appears black.",
            "visible_entity_ids": ["door"],
        }]
        details[1]["perception_layers"] = [{
            "id": "hall-view", "scene_id": "hall",
            "activation": "condition", "when": {"type": "always"},
            "description": "The hall door appears white.",
            "visible_entity_ids": ["door"],
        }]
        rebuilt = _merge_story_node_details(blueprint, details, {"passed": True})
        overview, pass1, pass2, _embedded = _prepare_full_rebuild(rebuilt)
        world = _assemble_world_book(pass1, pass2, overview)
        world["perception_layers"] = rebuilt["perception_layers"]
        _bind_perception_entity_instances(world)

        study_id = world["perception_layers"][0]["visible_entity_ids"][0]
        hall_id = world["perception_layers"][1]["visible_entity_ids"][0]
        self.assertNotEqual(study_id, hall_id)
        self.assertEqual("study", world["entities"][study_id]["scene"])
        self.assertEqual("hall", world["entities"][hall_id]["scene"])

    def test_runtime_initializes_only_selected_anthology_scenario(self):
        world = {
            "name": "Anthology",
            "rule_system": "coc",
            "starting_scene": "a_room",
            "scenarios": [
                {"id": "a", "title": "A", "starting_scene": "a_room"},
                {"id": "b", "title": "B", "starting_scene": "b_room"},
            ],
            "scenes": {
                "a_room": {"name": "A Room", "scenario_id": "a", "exits": {}},
                "b_room": {"name": "B Room", "scenario_id": "b", "exits": {}},
            },
            "entities": {
                "a_npc": {"id": "a_npc", "name": "A NPC", "type": "npc",
                          "scenario_id": "a", "scene": "a_room"},
                "b_npc": {"id": "b_npc", "name": "B NPC", "type": "npc",
                          "scenario_id": "b", "scene": "b_room"},
            },
            "story_beats": [
                {"id": "a_beat", "scenario_id": "a", "scenes": ["a_room"]},
                {"id": "b_beat", "scenario_id": "b", "scenes": ["b_room"]},
            ],
        }
        session = create_session("anthology-test", "Anthology")

        initialize_session_from_world(session, world, "b")

        self.assertEqual("b", session["current_scenario_id"])
        self.assertEqual("b_room", session["player_state"]["current_scene"])
        self.assertEqual("b_beat", session["current_beat_id"])
        self.assertNotIn("a_npc", session["entity_states"])
        self.assertIn("b_npc", session["entity_states"])

    def test_parse_status_exposes_failed_quality_gate(self):
        world = {
            "name": "Low Quality", "scenes": {"study": {}}, "entities": {},
            "reconstruction_quality": {
                "overall": 71, "passed": False,
                "failed_node_ids": ["study_node"],
                "global_dimensions": {"story_graph_closure": 100},
            },
            "_validation_issues": ["Story reconstruction quality gate failed"],
        }

        result = _completed_parse_job(world, "/tmp/world.json")

        self.assertEqual("done", result["status"])
        self.assertFalse(result["quality_passed"])
        self.assertEqual(71, result["quality_score"])
        self.assertEqual(["study_node"], result["failed_node_ids"])

    def test_runtime_context_selects_only_current_detailed_node(self):
        world = reconstructed_module()
        world["detailed_story_nodes"] = detailed_story_nodes()

        block = _build_current_story_node_block(world, "study")

        self.assertIn("study_node", block)
        self.assertIn("search completed", block)
        self.assertNotIn("investigators enter the quiet hall", block.lower())


class ObjectLocationNarrationTests(unittest.TestCase):
    def _run_carried_object_turn(self, narration):
        world = {
            "name": "Location Ledger", "rule_system": "coc",
            "starting_scene": "study", "opening": "桌上放着铜币。",
            "scenes": {"study": {
                "name": "书房", "desc": "一间书房。", "source_text": "桌上放着铜币。",
                "exits": {},
            }},
            "entities": {"coin": {
                "type": "item", "name": "铜币", "scene": "study",
                "home_scene": "study", "initial_state": "present",
            }},
        }
        scene_index = build_scene_index(world)
        entity_index = build_entity_index(world)
        session = create_session("wrong-location", world["name"])
        initialize_session_from_world(session, world)
        append_world_event(session, world, {"type": "item_picked_up", "entity_id": "coin"})

        with patch("engine.load_world", return_value=world), \
                patch("engine.get_indices", return_value=(scene_index, entity_index)), \
                patch("engine.get_session", return_value=session), \
                patch("engine.save_session"), \
                patch("rag.hybrid_search", return_value=[]), \
                patch("engine.get_or_compress_conversation_summary", return_value=""), \
                narration_provider(lambda _request: narration):
            response = run_gm_turn(
                [{"role": "user", "content": "我看看桌面"}],
                model=world["name"], chat_id=session["chat_id"],
                api_key="manual-provider-no-api-key",
            )
        return response, session

    def test_static_scene_source_cannot_restore_carried_object(self):
        response, session = self._run_carried_object_turn(
            "你看到桌上仍放着铜币。")

        self.assertIn("没有在当前场景看到", response)
        self.assertNotIn("桌上仍放着", response)
        self.assertIn("coin", session["inventory_entity_ids"])

    def test_english_scene_claim_cannot_restore_carried_object(self):
        response, session = self._run_carried_object_turn(
            "The 铜币 lies on the table beside you.")

        self.assertIn("没有在当前场景看到", response)
        self.assertNotIn("lies on the table", response)
        self.assertIn("coin", session["inventory_entity_ids"])

    def test_explicit_carried_object_claim_remains_allowed(self):
        response, session = self._run_carried_object_turn(
            "The 铜币 remains secure in your backpack.")

        self.assertIn("backpack", response)
        self.assertIn("coin", session["inventory_entity_ids"])


if __name__ == "__main__":
    unittest.main()

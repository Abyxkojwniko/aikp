import sys
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from story_graph_benchmark import optimal_match, score_world


class StoryGraphBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.gold = {
            "module_id": "synthetic",
            "nodes": [
                {"id": "study", "label": "Study Search", "playable": True,
                 "scenario_id": "main"},
                {"id": "ending", "label": "Escape", "playable": True,
                 "scenario_id": "main"},
            ],
            "edges": [{"from": "study", "to": "ending", "type": "branches_to"}],
            "entities": [{"id": "keeper", "name": "The Keeper", "type": "npc"}],
            "forbidden_navigable_scenes": [
                {"id": "book_castle", "label": "Castle in the storybook"}],
        }

    def test_perfect_graph_scores_one(self):
        world = {
            "story_tree": {"nodes": [
                {"id": "search_room", "title": "Study Search", "playable": True,
                 "scenario_id": "generated-main"},
                {"id": "get_away", "title": "Escape", "playable": True,
                 "scenario_id": "generated-main"},
            ], "relations": [
                {"from": "search_room", "to": "get_away", "type": "branches_to"}
            ]},
            "scenarios": [{"id": "generated-main", "title": "Main"}],
            "entity_registry": [
                {"id": "npc_1", "name": "The Keeper", "type": "npc",
                 "scenario_id": "generated-main"}
            ],
            "scenes": {"study": {"name": "Study"}},
        }

        report = score_world(self.gold, world)

        self.assertEqual(1.0, report["nodes"]["f1"])
        self.assertEqual(1.0, report["typed_nodes"]["f1"])
        self.assertEqual(1.0, report["typed_edges"]["f1"])
        self.assertEqual(1.0, report["entities"]["f1"])
        self.assertEqual(1.0, report["narrative_scope"]["accuracy"])
        self.assertEqual(1.0, report["scenario_assignment"]["accuracy"])
        self.assertFalse(report["multi_scenario_assignment"]["applicable"])
        self.assertEqual(1.0, report["entity_scenario_assignment"]["accuracy"])

    def test_missing_branch_and_embedded_scene_are_penalized(self):
        world = {
            "story_tree": {"nodes": [
                {"id": "study", "title": "Study Search", "playable": True},
            ], "relations": []},
            "scenes": {"castle": {"name": "Castle in the storybook"}},
            "entities": {},
        }

        report = score_world(self.gold, world)

        self.assertLess(report["nodes"]["recall"], 1.0)
        self.assertEqual(0.0, report["typed_edges"]["recall"])
        self.assertEqual(0.0, report["narrative_scope"]["accuracy"])

    def test_cross_scenario_edge_is_reported(self):
        gold = dict(self.gold)
        gold["nodes"] = [dict(self.gold["nodes"][0], scenario_id="one"),
                         dict(self.gold["nodes"][1], scenario_id="two")]
        world = {
            "story_tree": {"nodes": [
                {"id": "study", "title": "Study Search", "playable": True,
                 "scenario_id": "pred-one"},
                {"id": "escape", "title": "Escape", "playable": True,
                 "scenario_id": "pred-two"},
            ], "relations": [
                {"from": "study", "to": "escape", "type": "branches_to"}
            ]},
            "scenarios": [{"id": "pred-one"}, {"id": "pred-two"}],
            "scenes": {}, "entities": {},
        }

        report = score_world(gold, world)

        self.assertEqual(1, report["scenario_isolation"]["cross_scenario_edges"])
        self.assertEqual(0.0, report["scenario_isolation"]["accuracy"])

    def test_missing_anthology_membership_cannot_score_as_isolated(self):
        gold = dict(self.gold)
        gold["nodes"] = [dict(self.gold["nodes"][0], scenario_id="one"),
                         dict(self.gold["nodes"][1], scenario_id="two")]
        world = {
            "story_tree": {"nodes": [
                {"id": "study", "title": "Study Search", "playable": True},
                {"id": "escape", "title": "Escape", "playable": True},
            ], "relations": []},
            "scenes": {}, "entities": {},
        }

        report = score_world(gold, world)

        self.assertEqual(0.0, report["scenario_assignment"]["assignment_coverage"])
        self.assertEqual(0.0, report["scenario_assignment"]["accuracy"])

    def test_scenario_clustering_allows_arbitrary_generated_ids(self):
        gold = dict(self.gold)
        gold["nodes"] = [
            {"id": "a1", "label": "Alpha Start", "scenario_id": "alpha"},
            {"id": "a2", "label": "Alpha End", "scenario_id": "alpha"},
            {"id": "b1", "label": "Beta Start", "scenario_id": "beta"},
            {"id": "b2", "label": "Beta End", "scenario_id": "beta"},
        ]
        gold["edges"] = []
        world = {
            "story_tree": {"nodes": [
                {"id": "p1", "title": "Alpha Start", "scenario_id": "story-x"},
                {"id": "p2", "title": "Alpha End", "scenario_id": "story-x"},
                {"id": "p3", "title": "Beta Start", "scenario_id": "story-y"},
                {"id": "p4", "title": "Beta End", "scenario_id": "story-y"},
            ], "relations": []},
            "scenarios": [{"id": "story-x"}, {"id": "story-y"}],
            "scenes": {}, "entities": {},
        }

        report = score_world(gold, world)

        self.assertEqual(1.0, report["scenario_assignment"]["pairwise"]["f1"])
        self.assertEqual(1.0, report["scenario_assignment"]["accuracy"])
        self.assertTrue(report["multi_scenario_assignment"]["applicable"])

    def test_edge_to_unmatched_hallucinated_node_is_false_positive(self):
        world = {
            "story_tree": {"nodes": [
                {"id": "study", "title": "Study Search", "scenario_id": "main"},
                {"id": "ending", "title": "Escape", "scenario_id": "main"},
                {"id": "moon", "title": "Invented Moon Base", "scenario_id": "main"},
            ], "relations": [
                {"from": "study", "to": "ending", "type": "branches_to"},
                {"from": "study", "to": "moon", "type": "branches_to"},
            ]},
            "scenarios": [{"id": "main"}], "scenes": {}, "entities": {},
        }

        report = score_world(self.gold, world)

        self.assertEqual(1, report["typed_edges"]["tp"])
        self.assertEqual(1, report["typed_edges"]["fp"])
        self.assertEqual(1, report["typed_edges"]["unmatched_endpoint_edges"])

    def test_matching_maximizes_cardinality_before_similarity(self):
        gold = [{"id": "g1"}, {"id": "g2"}]
        predicted = [{"id": "p1"}, {"id": "p2"}]
        scores = {
            ("g1", "p1"): 0.95,
            ("g1", "p2"): 0.80,
            ("g2", "p1"): 0.75,
            ("g2", "p2"): 0.0,
        }

        with patch("story_graph_benchmark.label_similarity",
                   side_effect=lambda left, right: scores[(left["id"], right["id"])]):
            matches, trace = optimal_match(gold, predicted)

        self.assertEqual({"g1": "p2", "g2": "p1"}, matches)
        self.assertEqual(2, len(trace))

    def test_matching_is_invariant_to_input_order(self):
        gold = [{"id": "g2", "label": "Room"},
                {"id": "g1", "label": "Room"}]
        predicted = [{"id": "p2", "label": "Room"},
                     {"id": "p1", "label": "Room"}]

        first, _ = optimal_match(gold, predicted)
        second, _ = optimal_match(list(reversed(gold)), list(reversed(predicted)))

        self.assertEqual(first, second)

    def test_wrong_node_kind_reduces_typed_node_f1_only(self):
        world = {
            "story_tree": {"nodes": [
                {"id": "study", "title": "Study Search", "kind": "ending"},
                {"id": "escape", "title": "Escape", "kind": "ending"},
            ], "relations": []},
            "scenes": {}, "entities": {},
        }

        report = score_world(self.gold, world)

        self.assertEqual(1.0, report["nodes"]["f1"])
        self.assertLess(report["typed_nodes"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

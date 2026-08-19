import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from validate_gold import validate_directory, validate_gold


class GoldValidationTests(unittest.TestCase):
    def test_repository_gold_is_structurally_valid(self):
        report = validate_directory(ROOT / "evals" / "gold")
        self.assertTrue(report["valid"], report)
        self.assertEqual(10, report["module_count"])

    def test_cross_scenario_edge_and_unknown_entity_scope_are_rejected(self):
        gold = {
            "module_id": "bad",
            "scenarios": [{"id": "a"}, {"id": "b"}],
            "nodes": [
                {"id": "one", "label": "One", "kind": "event",
                 "scenario_id": "a"},
                {"id": "two", "label": "Two", "kind": "event",
                 "scenario_id": "b"},
            ],
            "edges": [{"from": "one", "to": "two", "type": "before"}],
            "entities": [
                {"id": "npc", "name": "NPC", "type": "npc",
                 "scenario_id": "missing"},
            ],
            "source": {"download_url": "https://example.invalid/test.pdf"},
        }

        report = validate_gold(gold, "bad.json")

        self.assertFalse(report["valid"])
        self.assertTrue(any("crosses scenarios" in error for error in report["errors"]))
        self.assertTrue(any("invalid scenario_id" in error for error in report["errors"]))

    def test_missing_or_open_ended_node_kind_is_rejected(self):
        gold = {
            "module_id": "bad",
            "scenarios": [{"id": "main"}],
            "nodes": [{"id": "one", "label": "One", "scenario_id": "main",
                       "kind": "investigation"}],
            "edges": [], "entities": [],
            "source": {"download_url": "https://example.invalid/test.pdf"},
        }

        report = validate_gold(gold, "bad.json")

        self.assertFalse(report["valid"])
        self.assertTrue(any("invalid kind" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()

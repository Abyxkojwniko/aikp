import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from build_paper_tables import build_summary, markdown


class PaperTableTests(unittest.TestCase):
    def test_summary_preserves_denominators_and_limitations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold = root / "gold"
            gold.mkdir()
            (gold / "sample.json").write_text(json.dumps({
                "scenarios": [{"id": "main"}], "nodes": [{"id": "a"}],
                "edges": [], "entities": [{"id": "x"}],
                "forbidden_navigable_scenes": [{"id": "dream"}],
            }), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"documents": [{
                "id": "sample", "annotation": "gold-v1", "split": "dev",
            }]}), encoding="utf-8")
            report = root / "interactive.json"
            report.write_text(json.dumps({"aggregate": {
                "case_count": 1, "run_count": 2,
                "min_runs_per_case": 2, "max_runs_per_case": 2,
                "turns": 10, "task_success_rate": 1.0,
                "macro_branch_coverage": 0.5,
                "unsupported_world_mutations": {
                    "violations": 1, "eligible_turns": 4, "per_100_turns": 25.0},
                "hidden_information_leaks": {
                    "violations": 0, "eligible_turns": 3, "per_100_turns": 0.0},
                "location_continuity_violations": {
                    "violations": 0, "eligible_turns": 2, "per_100_turns": 0.0},
                "valid_action_false_rejection": {"eligible_turns": 7},
                "invalid_action_acceptance": {"eligible_turns": 3},
                "pass_power_k_by_case": {
                    "sample": {"runs": 2, "k": 3,
                               "pass_power_k": None,
                               "sufficient_runs": False}},
            }}), encoding="utf-8")
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({
                "experiment": {"baseline_id": "heading_chain_v1"},
                "aggregate": {
                    "nodes": {"macro_f1": 0.25},
                    "typed_nodes": {"macro_f1": 0.2},
                    "typed_edges": {"macro_f1": 0.1},
                    "entities": {"macro_f1": 0.0},
                    "narrative_scope": {"macro": 0.5},
                    "multi_scenario_assignment": {"macro": 0.4},
                },
            }), encoding="utf-8")

            summary = build_summary(
                gold, manifest, report, baselines=[baseline])
            output = markdown(summary)

            self.assertEqual(3, summary["interactive"]
                             ["hidden_information_leaks"]["eligible_turns"])
            self.assertIn("| Hidden-information leak | 0 | 3 | 0.0 |", output)
            self.assertIn("| 1 | 2 | 2-2 | 10 | 1.0 | 0.5 | 7 | 3 |", output)
            self.assertTrue(any("fewer than k" in item
                                for item in summary["limitations"]))
            self.assertTrue(any("second annotation" in item
                                for item in summary["limitations"]))
            self.assertTrue(any("No real-provider" in item
                                for item in summary["limitations"]))
            self.assertEqual(
                "heading_chain_v1", summary["parser_baselines"][0]["baseline_id"])
            self.assertIn(
                "| heading_chain_v1 | 0.25 | 0.2 | 0.1 | 0.0 | 0.5 | 0.4 |",
                output,
            )
            self.assertTrue(any("structural lower bound" in item
                                for item in summary["limitations"]))


if __name__ == "__main__":
    unittest.main()

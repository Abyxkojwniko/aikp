import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from interactive_benchmark import aggregate_reports, score_transcript


class InteractiveBenchmarkTests(unittest.TestCase):
    def test_unannotated_risk_metrics_are_null_not_zero(self):
        report = score_transcript({
            "case": "plain", "turns": [{"index": 1}], "failures": [],
            "coverage": ["a"], "required_coverage": ["a"],
        })

        self.assertTrue(report["task_success"])
        self.assertEqual(1.0, report["branch_coverage"])
        self.assertIsNone(report["hidden_information_leaks"]["per_100_turns"])
        self.assertIsNone(report["valid_action_false_rejection"]["rate"])

    def test_explicit_risk_annotations_produce_rates(self):
        report = score_transcript({
            "case": "risk", "failures": [], "turns": [
                {"evaluation": {"action_validity": "valid",
                                "expected_outcome": "accepted"},
                 "observed": {"action_outcome": "blocked",
                              "hidden_information_leak": False,
                              "unsupported_world_mutation": True}},
                {"evaluation": {"action_validity": "invalid",
                                "expected_outcome": "blocked"},
                 "observed": {"action_outcome": "accepted",
                              "hidden_information_leak": True,
                              "unsupported_world_mutation": False}},
            ],
        })

        self.assertEqual(50.0, report["hidden_information_leaks"]["per_100_turns"])
        self.assertEqual(50.0, report["unsupported_world_mutations"]["per_100_turns"])
        self.assertEqual(1.0, report["valid_action_false_rejection"]["rate"])
        self.assertEqual(1.0, report["invalid_action_acceptance"]["rate"])

    def test_pass_power_k_is_grouped_by_case(self):
        reports = [
            score_transcript({"case": "same", "turns": [], "failures": []}),
            score_transcript({"case": "same", "turns": [], "failures": ["bad"]}),
        ]

        aggregate = aggregate_reports(reports, pass_k=3)

        self.assertEqual(0.5, aggregate["task_success_rate"])
        self.assertEqual(1, aggregate["case_count"])
        self.assertEqual(2, aggregate["min_runs_per_case"])
        self.assertIsNone(
            aggregate["pass_power_k_by_case"]["same"]["pass_power_k"])
        self.assertFalse(
            aggregate["pass_power_k_by_case"]["same"]["sufficient_runs"])

        reports.append(score_transcript({
            "case": "same", "turns": [], "failures": [],
        }))
        aggregate = aggregate_reports(reports, pass_k=3)
        self.assertEqual(
            0.2963,
            aggregate["pass_power_k_by_case"]["same"]["pass_power_k"],
        )
        self.assertEqual(1, aggregate["pass_power_k_eligible_cases"])

    def test_coverage_is_unioned_across_routes_of_one_module(self):
        reports = [
            score_transcript({"case": "one", "module": "m", "turns": [],
                              "failures": [], "coverage": ["a"]}),
            score_transcript({"case": "two", "module": "m", "turns": [],
                              "failures": [], "coverage": ["b"]}),
        ]

        aggregate = aggregate_reports(reports, coverage_manifests={"m": {"a", "b"}})

        self.assertEqual(1.0, aggregate["macro_branch_coverage"])
        self.assertEqual(2, aggregate["branch_coverage_by_module"]["m"]["covered"])


if __name__ == "__main__":
    unittest.main()

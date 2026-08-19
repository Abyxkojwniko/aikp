import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from validate_interactive_cases import validate_case


class InteractiveCaseValidationTests(unittest.TestCase):
    def test_valid_annotated_case(self):
        case = {"name": "case", "turns": [{
            "input": "open the door", "covers": ["door"],
            "evaluation": {"action_validity": "valid",
                           "expected_outcome": "accepted"},
        }]}
        self.assertEqual([], validate_case(
            case, Path("case.json"), True, {"door"}))

    def test_missing_annotation_is_rejected_but_extra_coverage_is_allowed(self):
        case = {"name": "case", "turns": [{
            "input": "teleport", "covers": ["unknown"],
        }]}
        errors = validate_case(case, Path("case.json"), True, {"known"})
        self.assertTrue(any("missing evaluation" in error for error in errors))
        self.assertFalse(any("unknown coverage" in error for error in errors))

    def test_post_run_risk_labels_are_rejected_from_expectations(self):
        case = {"name": "case", "turns": [{
            "input": "guess secret", "covers": [],
            "evaluation": {
                "action_validity": "invalid", "expected_outcome": "blocked",
                "hidden_information_leak": False,
            },
        }]}
        errors = validate_case(case, Path("case.json"))
        self.assertTrue(any("post-run risk labels" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

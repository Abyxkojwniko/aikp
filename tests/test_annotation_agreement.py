import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from annotation_agreement import compare_annotations, score_annotations


def annotation(prefix: str = "a") -> dict:
    return {
        "module_id": "sample",
        "scenarios": [{"id": f"{prefix}-one"}, {"id": f"{prefix}-two"}],
        "nodes": [
            {"id": f"{prefix}-start", "label": "Opening", "kind": "opening",
             "scenario_id": f"{prefix}-one"},
            {"id": f"{prefix}-end", "label": "Ending", "kind": "ending",
             "scenario_id": f"{prefix}-one"},
            {"id": f"{prefix}-other", "label": "Other Story", "kind": "event",
             "scenario_id": f"{prefix}-two"},
        ],
        "edges": [{"from": f"{prefix}-start", "to": f"{prefix}-end",
                   "type": "before"}],
        "entities": [{"id": f"{prefix}-keeper", "name": "Keeper", "type": "npc",
                      "scenario_id": f"{prefix}-one"}],
        "forbidden_navigable_scenes": [
            {"id": f"{prefix}-dream", "label": "Castle in a dream"}],
    }


class AnnotationAgreementTests(unittest.TestCase):
    def test_permutation_of_ids_preserves_full_agreement(self):
        report = score_annotations(annotation("a"), annotation("b"))

        self.assertEqual(1.0, report["nodes"]["f1"])
        self.assertEqual(1.0, report["typed_edges"]["f1"])
        self.assertEqual(1.0, report["entities"]["f1"])
        self.assertEqual(1.0, report["scenario_assignment"]["accuracy"])

    def test_missing_node_and_changed_relation_reduce_agreement(self):
        compared = annotation("b")
        compared["nodes"].pop()
        compared["edges"][0]["type"] = "causes"

        report = score_annotations(annotation("a"), compared)

        self.assertLess(report["nodes"]["recall"], 1.0)
        self.assertEqual(0.0, report["typed_edges"]["f1"])

    def test_two_empty_optional_collections_are_not_disagreement(self):
        left, right = annotation("a"), annotation("b")
        left["forbidden_navigable_scenes"] = []
        right["forbidden_navigable_scenes"] = []

        report = score_annotations(left, right)

        self.assertFalse(report["forbidden_scopes"]["applicable"])
        self.assertEqual(1.0, report["forbidden_scopes"]["f1"])

    def test_missing_module_makes_directory_comparison_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right = root / "left", root / "right"
            left.mkdir()
            right.mkdir()
            (left / "sample.json").write_text(
                json.dumps(annotation("a")), encoding="utf-8")
            second = annotation("extra")
            second["module_id"] = "extra"
            (left / "extra.json").write_text(json.dumps(second), encoding="utf-8")
            (right / "sample.json").write_text(
                json.dumps(annotation("b")), encoding="utf-8")

            report = compare_annotations(left, right)

            self.assertFalse(report["valid"])
            self.assertEqual(["extra"], report["missing_from_b"])

    def test_source_root_enforces_both_annotation_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.txt"
            source.write_text("source", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            left, right = root / "left", root / "right"
            left.mkdir()
            right.mkdir()
            for directory, prefix, expected in (
                    (left, "a", digest), (right, "b", "0" * 64)):
                payload = annotation(prefix)
                payload["source"] = {
                    "local_filename": "source.txt", "sha256": expected,
                    "download_url": "https://example.test/source",
                }
                (directory / "sample.json").write_text(
                    json.dumps(payload), encoding="utf-8")

            report = compare_annotations(left, right, root)

            self.assertFalse(report["valid"])
            invalid = [row for row in report["validation"] if not row["valid"]]
            self.assertEqual(1, len(invalid))
            self.assertEqual("b", invalid[0]["annotator"])


if __name__ == "__main__":
    unittest.main()

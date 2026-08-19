import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from annotation_packets import collect_submission, prepare_packets


class AnnotationPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root = self.root / "sources"
        self.source_root.mkdir()
        self.source = self.source_root / "module.txt"
        self.source.write_text("A complete adventure source.", encoding="utf-8")
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"documents": [{
            "id": "real_module", "local_filename": "module.txt",
            "sha256": digest, "ruleset": "coc", "split": "public-dev",
            "annotation": "gold-v1", "source_url": "https://example.test/module",
            "publisher": "Example Publisher",
        }]}), encoding="utf-8")
        self.guide = self.root / "guide.md"
        self.guide.write_text("Independent instructions", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_is_blind_deterministic_and_annotator_specific(self):
        output = self.root / "packet"
        packet = prepare_packets(
            self.manifest, self.source_root, output, ["a", "b"],
            guide_path=self.guide,
        )

        a = json.loads((output / "a/assignment.json").read_text())
        b = json.loads((output / "b/assignment.json").read_text())
        a_task, b_task = a["tasks"][0]["task_id"], b["tasks"][0]["task_id"]
        self.assertNotEqual(a_task, b_task)
        self.assertNotIn("real_module", (output / "a/assignment.json").read_text())
        self.assertNotIn("real_module", (output / "a/annotations" / f"{a_task}.json").read_text())
        self.assertEqual("real_module", packet["annotators"]["a"]["tasks"][0]["module_id"])

        with self.assertRaisesRegex(ValueError, "not empty"):
            prepare_packets(
                self.manifest, self.source_root, output, ["a", "b"],
                guide_path=self.guide,
            )

    def test_collect_validates_then_unblinds_and_freezes_hash(self):
        packet_dir = self.root / "packet"
        prepare_packets(
            self.manifest, self.source_root, packet_dir, ["reader"],
            copy_sources=True, guide_path=self.guide,
        )
        assignment = json.loads((packet_dir / "reader/assignment.json").read_text())
        task = assignment["tasks"][0]["task_id"]
        annotation_path = packet_dir / "reader/annotations" / f"{task}.json"
        annotation = json.loads(annotation_path.read_text())
        annotation.update({
            "title": "Independent title",
            "scenarios": [{"id": "s1", "title": "Scenario"}],
            "nodes": [{"id": "n1", "label": "Opening", "kind": "opening",
                       "scenario_id": "s1"}],
            "entities": [{"id": "e1", "name": "Keeper", "type": "npc",
                          "scenario_id": "s1"}],
        })
        annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

        collected = self.root / "collected"
        report = collect_submission(
            packet_dir / "coordinator_packet.json", "reader",
            annotation_path.parent, collected,
        )

        result = json.loads((collected / "annotations/real_module.json").read_text())
        self.assertEqual("real_module", result["module_id"])
        self.assertEqual("reader", result["annotation_provenance"]["annotator_id"])
        self.assertEqual(
            report["annotations"][0]["annotation_sha256"],
            hashlib.sha256((collected / "annotations/real_module.json").read_bytes()).hexdigest(),
        )

    def test_collect_rejects_incomplete_annotation(self):
        packet_dir = self.root / "packet"
        prepare_packets(
            self.manifest, self.source_root, packet_dir, ["reader"],
            guide_path=self.guide,
        )
        assignment = json.loads((packet_dir / "reader/assignment.json").read_text())
        task = assignment["tasks"][0]["task_id"]
        with self.assertRaisesRegex(ValueError, "nodes must be a non-empty list"):
            collect_submission(
                packet_dir / "coordinator_packet.json", "reader",
                packet_dir / "reader/annotations", self.root / "rejected",
            )


if __name__ == "__main__":
    unittest.main()

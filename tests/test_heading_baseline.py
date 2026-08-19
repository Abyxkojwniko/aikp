import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from run_heading_baseline import build_heading_world, run_baseline


class HeadingBaselineTests(unittest.TestCase):
    def test_world_is_single_scenario_sequential_heading_chain(self):
        world = build_heading_world(
            "INTRODUCTION\nArrival text.\n\nSTUDY SEARCH\nA desk.\n\nENDING\nEscape.",
            "sample",
        )

        nodes = world["story_tree"]["nodes"]
        edges = world["story_tree"]["relations"]
        self.assertGreaterEqual(len(nodes), 3)
        self.assertEqual(len(nodes) - 1, len(edges))
        self.assertEqual({"document"}, {row["scenario_id"] for row in nodes})
        self.assertEqual({}, world["entities"])

    def test_runner_hashes_predictions_and_scores_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.txt"
            source.write_text("INTRODUCTION\nStart.\nENDING\nFinish.", encoding="utf-8")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"documents": [{
                "id": "sample", "local_filename": "source.txt",
                "sha256": source_hash, "annotation": "gold-v1",
            }]}), encoding="utf-8")
            gold = root / "gold"
            gold.mkdir()
            (gold / "sample.json").write_text(json.dumps({
                "module_id": "sample",
                "source": {"local_filename": "source.txt", "sha256": source_hash},
                "scenarios": [{"id": "main"}],
                "nodes": [{"id": "start", "label": "INTRODUCTION",
                           "scenario_id": "main"}],
                "edges": [], "entities": [], "forbidden_navigable_scenes": [],
            }), encoding="utf-8")

            output = root / "result"
            report = run_baseline(manifest, root, gold, output)

            self.assertEqual(1, report["aggregate"]["module_count"])
            prediction = output / "predictions/sample.json"
            self.assertTrue(prediction.exists())
            run = report["experiment"]["runs"][0]
            self.assertEqual(
                run["prediction_sha256"], hashlib.sha256(prediction.read_bytes()).hexdigest())
            self.assertTrue(report["experiment"]["gold_tree_sha256"])


if __name__ == "__main__":
    unittest.main()

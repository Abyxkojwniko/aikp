import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from compare_parser_conditions import compare_conditions
from run_parser_matrix import build_jobs, run_matrix


class FakeParser:
    calls = 0

    def __init__(self, api_key, base_url, model, ablation):
        self.ablation = ablation

    def parse(self, text):
        FakeParser.calls += 1
        return {
            "name": "prediction",
            "_parser_mode": "fake",
            "_parser_ablation": self.ablation,
            "story_tree": {"nodes": [
                {"id": "start", "title": "Start", "scenario_id": "main"},
            ], "relations": []},
            "scenarios": [{"id": "main"}],
            "entity_registry": [],
            "scenes": {},
        }

    def experiment_usage(self):
        return {"calls": 1, "failures": 0, "prompt_chars": 10,
                "response_chars": 5, "prompt_tokens": 2,
                "completion_tokens": 1}


class ProviderFailureParser(FakeParser):
    def experiment_usage(self):
        usage = super().experiment_usage()
        usage["failures"] = 1
        return usage


def benchmark(module, node_f1):
    return {
        "module_id": module,
        "nodes": {"f1": node_f1},
        "typed_nodes": {"f1": node_f1},
        "typed_edges": {"f1": node_f1},
        "entities": {"f1": node_f1},
        "narrative_scope": {"accuracy": node_f1},
        "scenario_isolation": {"accuracy": node_f1},
        "scenario_assignment": {"accuracy": node_f1},
        "entity_scenario_assignment": {"accuracy": node_f1},
        "provenance": {"coverage": node_f1},
        "graph_closure": {"coverage": node_f1},
    }


class ParserExperimentTests(unittest.TestCase):
    def test_job_order_is_blocked_randomized_and_reproducible(self):
        documents = [{"id": "a"}, {"id": "b"}]
        conditions = ["legacy", "full", "no_node_repair"]
        first = build_jobs(documents, conditions, 2, Path("/tmp/out"), 17)
        second = build_jobs(documents, conditions, 2, Path("/tmp/out"), 17)
        changed = build_jobs(documents, conditions, 2, Path("/tmp/out"), 18)

        identity = lambda rows: [
            (row["repeat"], row["module_id"], row["condition"])
            for row in rows]
        self.assertEqual(identity(first), identity(second))
        self.assertNotEqual(identity(first), identity(changed))
        self.assertEqual(list(range(1, 13)), [row["job_sequence"] for row in first])
        for offset in range(0, len(first), len(conditions)):
            block = first[offset:offset + len(conditions)]
            self.assertEqual(1, len({(row["repeat"], row["module_id"])
                                     for row in block}))
            self.assertEqual(set(conditions), {row["condition"] for row in block})

    def test_matrix_writes_predictions_metadata_benchmark_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.txt"
            source.write_text("Start", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"documents": [{
                "id": "sample", "local_filename": "source.txt",
                "sha256": digest, "annotation": "gold-v1",
            }]}), encoding="utf-8")
            gold = root / "gold"
            gold.mkdir()
            (gold / "sample.json").write_text(json.dumps({
                "module_id": "sample",
                "nodes": [{"id": "start", "label": "Start",
                           "scenario_id": "gold-main"}],
                "edges": [], "entities": [],
                "forbidden_navigable_scenes": [],
            }), encoding="utf-8")
            output = root / "experiment"
            FakeParser.calls = 0
            dry_report = run_matrix(
                manifest, root, output, ["full"], 2, "fake-model",
                "http://invalid.local", "", gold_dir=gold, dry_run=True,
                parser_factory=FakeParser,
            )
            self.assertTrue(dry_report["dry_run"])
            report = run_matrix(
                manifest, root, output, ["full"], 2, "fake-model",
                "http://invalid.local", "offline-test", gold_dir=gold,
                parser_factory=FakeParser,
            )

            self.assertEqual(0, report["failures"])
            self.assertFalse(report["dry_run"])
            experiment = json.loads((output / "experiment.json").read_text())
            self.assertFalse(experiment["dry_run"])
            self.assertTrue(experiment["manifest_sha256"])
            self.assertTrue(experiment["source_tree_sha256"])
            self.assertTrue(experiment["gold_tree_sha256"])
            self.assertTrue(experiment["job_order_sha256"])
            self.assertEqual(2, FakeParser.calls)
            self.assertTrue((output / "full/repeat-001/benchmark.json").exists())
            metadata = json.loads((
                output / "full/repeat-001/runs/sample.json").read_text())
            self.assertEqual(1, metadata["telemetry"]["calls"])
            self.assertTrue(metadata["prediction_sha256"])
            self.assertIsInstance(metadata["job_sequence"], int)

            run_matrix(
                manifest, root, output, ["full"], 2, "fake-model",
                "http://invalid.local", "offline-test", gold_dir=gold,
                parser_factory=FakeParser,
            )
            self.assertEqual(2, FakeParser.calls)

            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                run_matrix(
                    manifest, root, output, ["full"], 2, "different-model",
                    "http://invalid.local", "offline-test", gold_dir=gold,
                    parser_factory=FakeParser,
                )

    def test_provider_failure_invalidates_repeat_and_skips_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.txt"
            source.write_text("Start", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"documents": [{
                "id": "sample", "local_filename": "source.txt",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "annotation": "gold-v1",
            }]}), encoding="utf-8")
            gold = root / "gold"
            gold.mkdir()
            (gold / "sample.json").write_text(json.dumps({
                "module_id": "sample", "nodes": [], "edges": [],
                "entities": [], "forbidden_navigable_scenes": [],
            }), encoding="utf-8")
            output = root / "experiment"
            report = run_matrix(
                manifest, root, output, ["full"], 1, "fake-model",
                "http://invalid.local", "offline-test", gold_dir=gold,
                parser_factory=ProviderFailureParser,
            )

            self.assertEqual(1, report["failures"])
            self.assertFalse((output / "full/repeat-001/benchmark.json").exists())
            status = json.loads((
                output / "full/repeat-001/benchmark_status.json").read_text())
            self.assertEqual("invalid", status["status"])

            stale = output / "full/repeat-001/benchmark.json"
            stale.write_text(json.dumps({
                "experiment": {"condition": "full", "repeat": 1},
                "modules": [benchmark("sample", 1.0)],
            }), encoding="utf-8")
            reference = output / "legacy/repeat-001"
            reference.mkdir(parents=True)
            (reference / "benchmark.json").write_text(json.dumps({
                "experiment": {"condition": "legacy", "repeat": 1},
                "modules": [benchmark("sample", 0.0)],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both conditions"):
                compare_conditions(output, "legacy", "full", 10, seed=7)

    def test_module_clustered_bootstrap_uses_paired_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for condition, offset in (("legacy", 0.0), ("full", 0.2)):
                for repeat in (1, 2):
                    target = root / condition / f"repeat-{repeat:03d}"
                    target.mkdir(parents=True)
                    payload = {
                        "experiment": {"condition": condition, "repeat": repeat},
                        "modules": [
                            benchmark("a", 0.4 + offset),
                            benchmark("b", 0.6 + offset),
                        ],
                    }
                    (target / "benchmark.json").write_text(
                        json.dumps(payload), encoding="utf-8")

            report = compare_conditions(root, "legacy", "full", 500, seed=7)
            nodes = report["metrics"]["node_f1"]
            self.assertEqual(2, nodes["module_count"])
            self.assertEqual(4, nodes["paired_run_count"])
            self.assertAlmostEqual(0.2, nodes["paired_delta"])
            self.assertEqual([0.2, 0.2], nodes["bootstrap_95_ci"])

            payload = json.loads((
                root / "full/repeat-002/benchmark.json").read_text())
            payload["modules"][0]["nodes"] = {}
            (root / "full/repeat-002/benchmark.json").write_text(
                json.dumps(payload), encoding="utf-8")
            report = compare_conditions(root, "legacy", "full", 100, seed=7)
            self.assertEqual(1, report["metrics"]["node_f1"]["missing_pairs"])


if __name__ == "__main__":
    unittest.main()

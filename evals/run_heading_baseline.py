#!/usr/bin/env python3
"""Run a deterministic heading-chain lower bound for story graph extraction."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "evals"))

from parser import _split_text_segments
from run_parser_matrix import directory_sha256, sha256, source_tree_sha256
from story_graph_benchmark import evaluate_directory


BASELINE_ID = "heading_chain_v1"


def clean_label(segment: dict, module_id: str) -> str:
    title = str(segment.get("title", "")).strip()
    if title and title != "(preamble)":
        return title[:160]
    for line in str(segment.get("text", "")).splitlines():
        candidate = re.sub(r"\s+", " ", line).strip()
        if candidate:
            return candidate[:160]
    return module_id


def build_heading_world(text: str, module_id: str) -> dict:
    segments = [row for row in _split_text_segments(text)
                if isinstance(row, dict) and str(row.get("text", "")).strip()]
    if not segments:
        segments = [{"title": "(preamble)", "text": text, "start": 0}]
    nodes = []
    scenes = {}
    for index, segment in enumerate(segments):
        node_id = f"heading_{index:04d}"
        label = clean_label(segment, module_id)
        nodes.append({
            "id": node_id,
            "title": label,
            "kind": "event",
            "playable": True,
            "scenario_id": "document",
            "source_start": int(segment.get("start", 0)),
        })
        scenes[node_id] = {
            "name": label,
            "navigable": True,
            "scenario_id": "document",
        }
    relations = [
        {"from": nodes[index]["id"], "to": nodes[index + 1]["id"],
         "type": "before"}
        for index in range(len(nodes) - 1)
    ]
    return {
        "name": module_id,
        "_baseline": {
            "id": BASELINE_ID,
            "assumptions": [
                "every detected heading is a playable node",
                "all nodes share one scenario",
                "adjacent nodes have a before relation",
                "no entities or embedded scopes are inferred",
            ],
        },
        "scenarios": [{"id": "document", "title": "Document"}],
        "story_tree": {"nodes": nodes, "relations": relations},
        "entity_registry": [],
        "entities": {},
        "scenes": scenes,
    }


def run_baseline(manifest_path: Path, source_root: Path, gold_dir: Path,
                 output_dir: Path, annotation: str = "gold-v1",
                 module_ids: set[str] | None = None) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = [row for row in manifest.get("documents", [])
                 if isinstance(row, dict)]
    if module_ids:
        documents = [row for row in documents if str(row.get("id")) in module_ids]
        missing = module_ids - {str(row.get("id")) for row in documents}
        if missing:
            raise ValueError("unknown module ids: " + ", ".join(sorted(missing)))
    else:
        documents = [row for row in documents
                     if row.get("annotation") == annotation]
    documents.sort(key=lambda row: str(row.get("id", "")))
    if not documents:
        raise ValueError("no corpus documents selected")

    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for document in documents:
        module_id = str(document["id"])
        source_path = source_root / str(document["local_filename"])
        expected_hash = str(document.get("sha256", ""))
        if not source_path.exists() or sha256(source_path) != expected_hash:
            raise ValueError(
                f"source hash mismatch or missing file: {document['local_filename']}")
        world = build_heading_world(
            source_path.read_text(encoding="utf-8", errors="replace"), module_id)
        prediction_path = prediction_dir / f"{module_id}.json"
        prediction_path.write_text(
            json.dumps(world, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        runs.append({
            "module_id": module_id,
            "source_sha256": expected_hash,
            "prediction_sha256": sha256(prediction_path),
            "predicted_nodes": len(world["story_tree"]["nodes"]),
            "predicted_edges": len(world["story_tree"]["relations"]),
        })

    report = evaluate_directory(gold_dir, prediction_dir, source_root)
    report["experiment"] = {
        "baseline_id": BASELINE_ID,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "gold_dir": str(gold_dir),
        "gold_tree_sha256": directory_sha256(gold_dir),
        "source_tree_sha256": source_tree_sha256(),
        "modules": [str(row["id"]) for row in documents],
        "runs": runs,
    }
    report_path = output_dir / "benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--manifest", type=Path,
                     default=ROOT / "evals" / "corpus_manifest.json")
    cli.add_argument("--source-root", type=Path,
                     default=Path("/home/lonpyer/aikp_eval_data"))
    cli.add_argument("--gold-dir", type=Path, default=ROOT / "evals" / "gold")
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--annotation", default="gold-v1")
    cli.add_argument("--module-id", action="append")
    args = cli.parse_args()
    try:
        report = run_baseline(
            args.manifest, args.source_root, args.gold_dir, args.output_dir,
            args.annotation, set(args.module_id or []) or None,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    aggregate = report["aggregate"]
    print(json.dumps({
        "baseline_id": BASELINE_ID,
        "output_dir": str(args.output_dir),
        "module_count": aggregate["module_count"],
        "node_macro_f1": aggregate["nodes"]["macro_f1"],
        "typed_edge_macro_f1": aggregate["typed_edges"]["macro_f1"],
        "entity_macro_f1": aggregate["entities"]["macro_f1"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate structural gold annotations before running paper experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EDGE_TYPES = {"before", "causes", "enables", "branches_to", "reveals", "pays_off"}
ENTITY_TYPES = {"npc", "object", "clue", "concept"}
NODE_KINDS = {"opening", "event", "choice", "clue", "encounter", "ending"}


def duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def validate_gold(gold: dict, filename: str = "", source_root: Path | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    module_id = str(gold.get("module_id", ""))
    if not module_id:
        errors.append("missing module_id")
    if filename and module_id and Path(filename).stem != module_id:
        errors.append(f"module_id {module_id!r} does not match filename {filename!r}")

    scenarios = [row for row in gold.get("scenarios", []) if isinstance(row, dict)]
    scenario_ids = [str(row.get("id", "")) for row in scenarios if row.get("id")]
    if not scenario_ids:
        errors.append("missing scenarios registry")
    if duplicate := duplicates(scenario_ids):
        errors.append("duplicate scenario ids: " + ", ".join(duplicate))
    scenario_set = set(scenario_ids)

    nodes = [row for row in gold.get("nodes", []) if isinstance(row, dict)]
    node_ids = [str(row.get("id", "")) for row in nodes if row.get("id")]
    if not nodes:
        errors.append("nodes must be a non-empty list")
    if len(node_ids) != len(nodes):
        errors.append("every node must have an id")
    if duplicate := duplicates(node_ids):
        errors.append("duplicate node ids: " + ", ".join(duplicate))
    node_set = set(node_ids)
    scenario_by_node = {}
    for row in nodes:
        node_id = str(row.get("id", ""))
        scenario_id = str(row.get("scenario_id", ""))
        scenario_by_node[node_id] = scenario_id
        if not str(row.get("label", "")).strip():
            errors.append(f"node {node_id!r} has no label")
        if str(row.get("kind", "")) not in NODE_KINDS:
            errors.append(f"node {node_id!r} has invalid kind {row.get('kind')!r}")
        if scenario_id not in scenario_set:
            errors.append(f"node {node_id!r} has invalid scenario_id {scenario_id!r}")

    edges = [row for row in gold.get("edges", []) if isinstance(row, dict)]
    edge_keys = []
    for index, row in enumerate(edges):
        source, target = str(row.get("from", "")), str(row.get("to", ""))
        edge_type = str(row.get("type", ""))
        edge_keys.append(f"{source}|{target}|{edge_type}")
        if source not in node_set:
            errors.append(f"edge {index} has unknown source {source!r}")
        if target not in node_set:
            errors.append(f"edge {index} has unknown target {target!r}")
        if edge_type not in EDGE_TYPES:
            errors.append(f"edge {index} has invalid type {edge_type!r}")
        if source == target:
            errors.append(f"edge {index} is a self-loop on {source!r}")
        left, right = scenario_by_node.get(source), scenario_by_node.get(target)
        if left and right and left != right:
            errors.append(
                f"edge {index} crosses scenarios: {source!r} ({left}) -> "
                f"{target!r} ({right})")
    if duplicate := duplicates(edge_keys):
        errors.append("duplicate typed edges: " + ", ".join(duplicate))

    entities = [row for row in gold.get("entities", []) if isinstance(row, dict)]
    entity_ids = [str(row.get("id", "")) for row in entities if row.get("id")]
    if len(entity_ids) != len(entities):
        errors.append("every entity must have an id")
    if duplicate := duplicates(entity_ids):
        errors.append("duplicate entity ids: " + ", ".join(duplicate))
    for row in entities:
        entity_id = str(row.get("id", ""))
        if not str(row.get("name", "")).strip():
            errors.append(f"entity {entity_id!r} has no name")
        if str(row.get("type", "")) not in ENTITY_TYPES:
            errors.append(f"entity {entity_id!r} has invalid type {row.get('type')!r}")
        scenario_id = str(row.get("scenario_id", ""))
        if scenario_id not in scenario_set:
            errors.append(f"entity {entity_id!r} has invalid scenario_id {scenario_id!r}")

    source_hash_matches = None
    source = gold.get("source", {})
    if source_root and isinstance(source, dict):
        relative, expected = source.get("local_filename"), source.get("sha256")
        if not relative or not expected:
            errors.append("source must declare local_filename and sha256")
        else:
            path = source_root / str(relative)
            source_hash_matches = path.exists() and (
                hashlib.sha256(path.read_bytes()).hexdigest() == str(expected)
            )
            if not source_hash_matches:
                errors.append(f"source hash mismatch or missing file: {relative}")
    if not isinstance(source, dict) or not source.get("download_url"):
        warnings.append("source has no publisher download_url")

    return {
        "module_id": module_id,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "scenarios": len(scenario_ids), "nodes": len(nodes),
            "edges": len(edges), "entities": len(entities),
            "forbidden_scenes": len(gold.get("forbidden_navigable_scenes", [])),
        },
        "source_hash_matches": source_hash_matches,
    }


def validate_directory(gold_dir: Path, source_root: Path | None = None) -> dict:
    reports = []
    for path in sorted(gold_dir.glob("*.json")):
        try:
            gold = json.loads(path.read_text(encoding="utf-8"))
            reports.append(validate_gold(gold, path.name, source_root))
        except (OSError, json.JSONDecodeError) as exc:
            reports.append({"module_id": path.stem, "valid": False,
                            "errors": [str(exc)], "warnings": []})
    return {
        "valid": bool(reports) and all(row["valid"] for row in reports),
        "module_count": len(reports),
        "modules": reports,
    }


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--gold-dir", type=Path, default=Path(__file__).with_name("gold"))
    cli.add_argument("--source-root", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    report = validate_directory(args.gold_dir, args.source_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

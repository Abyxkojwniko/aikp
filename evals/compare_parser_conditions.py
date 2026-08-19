#!/usr/bin/env python3
"""Compare parser conditions with module-clustered paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


METRICS = {
    "node_f1": ("nodes", "f1"),
    "typed_node_f1": ("typed_nodes", "f1"),
    "typed_edge_f1": ("typed_edges", "f1"),
    "entity_f1": ("entities", "f1"),
    "narrative_scope_accuracy": ("narrative_scope", "accuracy"),
    "scenario_isolation_accuracy": ("scenario_isolation", "accuracy"),
    "scenario_assignment_accuracy": ("scenario_assignment", "accuracy"),
    "multi_scenario_assignment_accuracy": (
        "multi_scenario_assignment", "accuracy"),
    "entity_scenario_assignment_accuracy": (
        "entity_scenario_assignment", "accuracy"),
    "provenance_coverage": ("provenance", "coverage"),
    "graph_closure_coverage": ("graph_closure", "coverage"),
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def load_condition(root: Path, condition: str) -> dict[tuple[int, str], dict]:
    rows = {}
    for path in sorted((root / condition).glob("repeat-*/benchmark.json")):
        status_path = path.parent / "benchmark_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "success":
                continue
        report = json.loads(path.read_text(encoding="utf-8"))
        repeat = int(report.get("experiment", {}).get(
            "repeat", path.parent.name.removeprefix("repeat-")))
        for module in report.get("modules", []):
            module_id = str(module.get("module_id", ""))
            if module_id:
                rows[(repeat, module_id)] = module
    return rows


def metric_value(report: dict, path: tuple[str, str]) -> float | None:
    row = report.get(path[0], {})
    if isinstance(row, dict) and row.get("applicable") is False:
        return None
    value = row.get(path[1]) if isinstance(row, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def paired_metric(reference: dict[tuple[int, str], dict],
                  compared: dict[tuple[int, str], dict],
                  metric_path: tuple[str, str], bootstrap_samples: int,
                  seed: int) -> dict:
    common = sorted(set(reference) & set(compared))
    pairs = []
    by_module: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key in common:
        left = metric_value(reference[key], metric_path)
        right = metric_value(compared[key], metric_path)
        if left is None or right is None:
            continue
        pairs.append((left, right))
        by_module[key[1]].append((left, right))
    module_rows = {
        module: (mean([a for a, _ in values]), mean([b for _, b in values]))
        for module, values in by_module.items()
    }
    module_deltas = [right - left for left, right in module_rows.values()]
    point_delta = mean(module_deltas)
    generator = random.Random(seed)
    modules = sorted(module_rows)
    draws = []
    if modules:
        for _ in range(bootstrap_samples):
            sample = [module_rows[generator.choice(modules)] for _ in modules]
            draws.append(mean([right - left for left, right in sample]))
    draws.sort()
    tolerance = 1e-12
    wins = sum(delta > tolerance for delta in module_deltas)
    losses = sum(delta < -tolerance for delta in module_deltas)
    return {
        "module_count": len(modules),
        "paired_run_count": len(pairs),
        "reference_mean": round(mean([left for left, _ in module_rows.values()]), 6),
        "compared_mean": round(mean([right for _, right in module_rows.values()]), 6),
        "paired_delta": round(point_delta, 6),
        "module_delta_std": round(
            statistics.stdev(module_deltas), 6) if len(module_deltas) > 1 else 0.0,
        "bootstrap_95_ci": [
            round(percentile(draws, 0.025), 6),
            round(percentile(draws, 0.975), 6),
        ],
        "bootstrap_samples": bootstrap_samples,
        "wins": wins,
        "ties": len(module_deltas) - wins - losses,
        "losses": losses,
        "missing_pairs": (
            len(set(reference) ^ set(compared)) + len(common) - len(pairs)
        ),
    }


def telemetry_summary(root: Path, condition: str) -> dict:
    rows = []
    for path in sorted((root / condition).glob("repeat-*/runs/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "success":
            rows.append(payload)
    fields = (
        "calls", "failures", "prompt_chars", "response_chars",
        "prompt_tokens", "completion_tokens",
    )
    result = {"successful_runs": len(rows)}
    for field in fields:
        values = [float(row.get("telemetry", {}).get(field, 0)) for row in rows]
        result[f"mean_{field}"] = round(mean(values), 4) if values else None
        result[f"total_{field}"] = int(sum(values))
    elapsed = [float(row.get("elapsed_seconds", 0)) for row in rows]
    result["mean_elapsed_seconds"] = round(mean(elapsed), 4) if elapsed else None
    result["total_elapsed_seconds"] = round(sum(elapsed), 4)
    return result


def compare_conditions(root: Path, reference_condition: str,
                       compared_condition: str, bootstrap_samples: int = 10000,
                       seed: int = 20260818) -> dict:
    reference = load_condition(root, reference_condition)
    compared = load_condition(root, compared_condition)
    if not reference or not compared:
        raise ValueError("both conditions need at least one benchmark report")
    return {
        "schema_version": 1,
        "reference_condition": reference_condition,
        "compared_condition": compared_condition,
        "bootstrap_unit": "module",
        "repeat_aggregation": "mean_within_module",
        "seed": seed,
        "metrics": {
            name: paired_metric(
                reference, compared, path, bootstrap_samples, seed + index)
            for index, (name, path) in enumerate(METRICS.items())
        },
        "telemetry": {
            reference_condition: telemetry_summary(root, reference_condition),
            compared_condition: telemetry_summary(root, compared_condition),
        },
    }


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--experiment-dir", type=Path, required=True)
    cli.add_argument("--reference", default="legacy")
    cli.add_argument("--compared", default="full")
    cli.add_argument("--bootstrap-samples", type=int, default=10000)
    cli.add_argument("--seed", type=int, default=20260818)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.bootstrap_samples < 1:
        cli.error("--bootstrap-samples must be positive")
    try:
        report = compare_conditions(
            args.experiment_dir, args.reference, args.compared,
            args.bootstrap_samples, args.seed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

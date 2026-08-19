#!/usr/bin/env python3
"""Aggregate final-state interactive evaluation from AIKP playtest transcripts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


RISK_FIELDS = {
    "unsupported_world_mutation": "unsupported_world_mutations",
    "hidden_information_leak": "hidden_information_leaks",
    "location_continuity_violation": "location_continuity_violations",
}


def rate(numerator: int, denominator: int, scale: int = 1) -> float | None:
    return round(scale * numerator / denominator, 4) if denominator else None


def score_transcript(transcript: dict, path: str = "") -> dict:
    turns = [row for row in transcript.get("turns", []) if isinstance(row, dict)]
    failures = list(transcript.get("failures", []))
    required = {str(value) for value in transcript.get("required_coverage", [])}
    covered = {str(value) for value in transcript.get("coverage", [])}
    metrics = {
        name: {"violations": 0, "eligible_turns": 0}
        for name in RISK_FIELDS.values()
    }
    valid_actions = false_rejections = 0
    invalid_actions = invalid_acceptances = 0
    annotated_turns = 0
    for turn in turns:
        evaluation = turn.get("evaluation", {})
        if not isinstance(evaluation, dict) or not evaluation:
            continue
        annotated_turns += 1
        observed = turn.get("observed", {})
        if not isinstance(observed, dict):
            observed = {}
        for source, name in RISK_FIELDS.items():
            if source not in observed:
                continue
            metrics[name]["eligible_turns"] += 1
            metrics[name]["violations"] += int(bool(observed[source]))
        validity = str(evaluation.get("action_validity", ""))
        outcome = str(observed.get("action_outcome", "")) \
            if observed else ""
        if validity == "valid":
            valid_actions += 1
            false_rejections += int(outcome == "blocked")
        elif validity == "invalid":
            invalid_actions += 1
            invalid_acceptances += int(outcome == "accepted")

    for row in metrics.values():
        row["per_100_turns"] = rate(row["violations"], row["eligible_turns"], 100)
    coverage = rate(len(required & covered), len(required))
    return {
        "path": path,
        "case": str(transcript.get("case", "")),
        "module": str(transcript.get("module", "")),
        "task_success": not failures,
        "turns": len(turns),
        "annotated_turns": annotated_turns,
        "assertion_failures": len(failures),
        "branch_coverage": coverage,
        "covered_points": len(covered),
        "coverage_points": sorted(covered),
        "required_points": len(required),
        **metrics,
        "valid_action_false_rejection": {
            "violations": false_rejections,
            "eligible_turns": valid_actions,
            "rate": rate(false_rejections, valid_actions),
        },
        "invalid_action_acceptance": {
            "violations": invalid_acceptances,
            "eligible_turns": invalid_actions,
            "rate": rate(invalid_acceptances, invalid_actions),
        },
    }


def discover_transcripts(root: Path, latest_per_case: bool = False) -> list[Path]:
    paths = sorted(root.glob("**/transcript.json"))
    if not latest_per_case:
        return paths
    latest = {}
    for path in paths:
        try:
            case = str(json.loads(path.read_text(encoding="utf-8")).get("case", ""))
        except (OSError, json.JSONDecodeError):
            case = ""
        key = case or path.parent.name
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path
    return sorted(latest.values())


def load_coverage_manifests(fixtures_dir: Path | None) -> dict[str, set[str]]:
    manifests = {}
    if not fixtures_dir:
        return manifests
    for path in sorted(fixtures_dir.glob("*_coverage.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests[str(payload["module"])] = {
            str(value) for value in payload.get("required", [])}
    return manifests


def aggregate_reports(
    reports: list[dict],
    pass_k: int = 3,
    coverage_manifests: dict[str, set[str]] | None = None,
) -> dict:
    turns = sum(row["turns"] for row in reports)
    successes = sum(row["task_success"] for row in reports)

    def aggregate_rate(name: str, field: str) -> dict:
        violations = sum(row[name]["violations"] for row in reports)
        eligible = sum(row[name]["eligible_turns"] for row in reports)
        return {"violations": violations, "eligible_turns": eligible,
                field: rate(violations, eligible, 100 if field == "per_100_turns" else 1)}

    by_case: dict[str, list[bool]] = defaultdict(list)
    for row in reports:
        by_case[row["case"]].append(bool(row["task_success"]))
    repeated = {}
    for case, values in sorted(by_case.items()):
        success_rate = sum(values) / len(values)
        repeated[case] = {
            "runs": len(values), "success_rate": round(success_rate, 4),
            "pass_power_k": (
                round(success_rate ** pass_k, 4)
                if len(values) >= pass_k else None
            ),
            "k": pass_k,
            "sufficient_runs": len(values) >= pass_k,
        }

    coverage_rows = [row["branch_coverage"] for row in reports
                     if row["branch_coverage"] is not None]
    coverage_by_module = {}
    manifests = coverage_manifests or {}
    for module, required_points in sorted(manifests.items()):
        covered_points = {
            point for row in reports if row["module"] == module
            for point in row.get("coverage_points", [])
        }
        matched = required_points & covered_points
        coverage_by_module[module] = {
            "covered": len(matched), "required": len(required_points),
            "coverage": rate(len(matched), len(required_points)),
            "missing": sorted(required_points - covered_points),
        }
    module_coverage_values = [row["coverage"] for row in coverage_by_module.values()
                              if row["coverage"] is not None]
    return {
        "case_count": len(by_case),
        "run_count": len(reports),
        "min_runs_per_case": min(map(len, by_case.values()), default=0),
        "max_runs_per_case": max(map(len, by_case.values()), default=0),
        "pass_power_k_eligible_cases": sum(
            len(values) >= pass_k for values in by_case.values()),
        "turns": turns,
        "task_successes": successes,
        "task_success_rate": rate(successes, len(reports)),
        "macro_branch_coverage": round(
            sum(module_coverage_values) / len(module_coverage_values), 4
        ) if module_coverage_values else (
            round(sum(coverage_rows) / len(coverage_rows), 4)
            if coverage_rows else None),
        "branch_coverage_by_module": coverage_by_module,
        "annotated_turns": sum(row["annotated_turns"] for row in reports),
        "unsupported_world_mutations": aggregate_rate(
            "unsupported_world_mutations", "per_100_turns"),
        "hidden_information_leaks": aggregate_rate(
            "hidden_information_leaks", "per_100_turns"),
        "location_continuity_violations": aggregate_rate(
            "location_continuity_violations", "per_100_turns"),
        "valid_action_false_rejection": aggregate_rate(
            "valid_action_false_rejection", "rate"),
        "invalid_action_acceptance": aggregate_rate(
            "invalid_action_acceptance", "rate"),
        "pass_power_k_by_case": repeated,
    }


def evaluate_runs(root: Path, latest_per_case: bool = False, pass_k: int = 3,
                  coverage_dir: Path | None = None) -> dict:
    reports = []
    invalid = []
    for path in discover_transcripts(root, latest_per_case):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            reports.append(score_transcript(payload, str(path)))
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    return {"aggregate": aggregate_reports(
                reports, pass_k, load_coverage_manifests(coverage_dir)),
            "runs": reports, "invalid_transcripts": invalid}


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--runs-dir", type=Path, required=True)
    cli.add_argument("--latest-per-case", action="store_true")
    cli.add_argument("--pass-k", type=int, default=3)
    cli.add_argument("--coverage-dir", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    report = evaluate_runs(args.runs_dir, args.latest_per_case, max(1, args.pass_k),
                           args.coverage_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if report["invalid_transcripts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

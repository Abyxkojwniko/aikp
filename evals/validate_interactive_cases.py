#!/usr/bin/env python3
"""Validate offline interactive cases and their evaluation annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VALIDITY = {"valid", "invalid", "ambiguous"}
OUTCOMES = {"accepted", "blocked"}
def validate_case(case: dict, path: Path, require_annotations: bool = False,
                  known_coverage: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if not str(case.get("name", "")).strip():
        errors.append("missing non-empty name")
    turns = case.get("turns")
    if not isinstance(turns, list) or not turns:
        return errors + ["turns must be a non-empty list"]
    for index, turn in enumerate(turns, start=1):
        prefix = f"turn {index}"
        if not isinstance(turn, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        if not str(turn.get("input", "")).strip():
            errors.append(f"{prefix}: missing non-empty input")
        covers = turn.get("covers", [])
        if not isinstance(covers, list) or any(not isinstance(v, str) for v in covers):
            errors.append(f"{prefix}: covers must be a string list")
        evaluation = turn.get("evaluation")
        if not evaluation:
            if require_annotations:
                errors.append(f"{prefix}: missing evaluation annotation")
            continue
        if not isinstance(evaluation, dict):
            errors.append(f"{prefix}: evaluation must be an object")
            continue
        validity = evaluation.get("action_validity")
        if validity not in VALIDITY:
            errors.append(f"{prefix}: invalid action_validity {validity!r}")
        expected = evaluation.get("expected_outcome")
        if expected not in OUTCOMES:
            errors.append(f"{prefix}: invalid expected_outcome {expected!r}")
        forbidden_risk_labels = sorted({
            "unsupported_world_mutation", "hidden_information_leak",
            "location_continuity_violation",
        } & evaluation.keys())
        if forbidden_risk_labels:
            errors.append(
                f"{prefix}: post-run risk labels cannot appear in evaluation: "
                + ", ".join(forbidden_risk_labels)
            )
    return errors


def load_coverage(fixtures_dir: Path) -> dict[str, set[str]]:
    manifests: dict[str, set[str]] = {}
    for path in fixtures_dir.glob("*_coverage.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests[str(payload.get("module", ""))] = {
            str(value) for value in payload.get("required", [])
        }
    return manifests


def load_annotation_overlay(path: Path | None) -> dict[str, dict[int, dict]]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(case): {int(index): value for index, value in turns.items()}
        for case, turns in payload.get("cases", {}).items()
    }


def apply_annotation_overlay(case: dict,
                             overlay: dict[str, dict[int, dict]]) -> dict:
    annotations = overlay.get(str(case.get("name", "")), {})
    for index, turn in enumerate(case.get("turns", []), start=1):
        if isinstance(turn, dict) and not turn.get("evaluation") and index in annotations:
            turn["evaluation"] = annotations[index]
    return case


def validate_directory(fixtures_dir: Path, require_annotations: bool = False,
                       annotations_path: Path | None = None) -> dict:
    manifests = load_coverage(fixtures_dir)
    overlay = load_annotation_overlay(annotations_path)
    reports = []
    annotated_turns = total_turns = 0
    for path in sorted(fixtures_dir.glob("*_case.json")):
        try:
            case = apply_annotation_overlay(
                json.loads(path.read_text(encoding="utf-8")), overlay)
        except (OSError, json.JSONDecodeError) as exc:
            reports.append({"path": str(path), "errors": [str(exc)]})
            continue
        turns = case.get("turns", [])
        total_turns += len(turns) if isinstance(turns, list) else 0
        annotated_turns += sum(
            bool(turn.get("evaluation")) for turn in turns if isinstance(turn, dict)
        ) if isinstance(turns, list) else 0
        module = str(case.get("module", ""))
        errors = validate_case(
            case, path, require_annotations,
            manifests.get(module) if module in manifests else None,
        )
        reports.append({"path": str(path), "errors": errors})
    failures = sum(bool(row["errors"]) for row in reports)
    return {
        "case_count": len(reports),
        "failed_cases": failures,
        "turn_count": total_turns,
        "annotated_turns": annotated_turns,
        "annotation_coverage": (
            round(annotated_turns / total_turns, 4) if total_turns else None
        ),
        "cases": reports,
    }


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--fixtures-dir", type=Path, required=True)
    cli.add_argument("--require-annotations", action="store_true")
    cli.add_argument(
        "--annotations", type=Path,
        default=Path(__file__).with_name("interactive_annotations.json"),
    )
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    report = validate_directory(
        args.fixtures_dir, args.require_annotations, args.annotations)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if report["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

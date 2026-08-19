#!/usr/bin/env python3
"""Compute deterministic agreement between two independent gold annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_graph_benchmark import optimal_match, prf, scenario_assignment_metric
from validate_gold import validate_gold


def rows_by_module(path: Path) -> dict[str, tuple[dict, str]]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    result = {}
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        result[str(payload.get("module_id", file.stem))] = (payload, file.name)
    return result


def typed_edge_agreement(
    reference_edges: list[dict],
    compared_edges: list[dict],
    compared_to_reference: dict[str, str],
) -> dict:
    reference = {
        (str(row.get("from")), str(row.get("to")), str(row.get("type", "before")))
        for row in reference_edges if isinstance(row, dict)
    }
    compared = {
        (compared_to_reference.get(str(row.get("from")),
                                   f"__unmatched_b__:{row.get('from')}"),
         compared_to_reference.get(str(row.get("to")),
                                   f"__unmatched_b__:{row.get('to')}"),
         str(row.get("type", "before")))
        for row in compared_edges if isinstance(row, dict)
    }
    score = prf(len(reference & compared), len(compared - reference),
                len(reference - compared))
    score["applicable"] = bool(reference or compared)
    if not score["applicable"]:
        score.update({"precision": 1.0, "recall": 1.0, "f1": 1.0})
    score["unmatched_endpoint_edges"] = sum(
        source.startswith("__unmatched_b__:") or target.startswith("__unmatched_b__:")
        for source, target, _edge_type in compared
    )
    return score


def collection_agreement(reference: list[dict], compared: list[dict],
                         require_type: bool = False,
                         require_kind: bool = False) -> tuple[dict, dict[str, str]]:
    matches, _trace = optimal_match(
        reference, compared, require_type=require_type, require_kind=require_kind)
    score = prf(len(matches), len(compared) - len(matches),
                len(reference) - len(matches))
    score["applicable"] = bool(reference or compared)
    if not score["applicable"]:
        score.update({"precision": 1.0, "recall": 1.0, "f1": 1.0})
    return score, matches


def score_annotations(reference: dict, compared: dict) -> dict:
    reference_nodes = [row for row in reference.get("nodes", []) if isinstance(row, dict)]
    compared_nodes = [row for row in compared.get("nodes", []) if isinstance(row, dict)]
    node_score, node_matches = collection_agreement(reference_nodes, compared_nodes)
    typed_node_score, _typed_node_matches = collection_agreement(
        reference_nodes, compared_nodes, require_kind=True)
    compared_to_reference = {value: key for key, value in node_matches.items()}

    reference_entities = [row for row in reference.get("entities", [])
                          if isinstance(row, dict)]
    compared_entities = [row for row in compared.get("entities", [])
                         if isinstance(row, dict)]
    entity_score, entity_matches = collection_agreement(
        reference_entities, compared_entities, require_type=True)
    scope_score, _scope_matches = collection_agreement(
        [row for row in reference.get("forbidden_navigable_scenes", [])
         if isinstance(row, dict)],
        [row for row in compared.get("forbidden_navigable_scenes", [])
         if isinstance(row, dict)],
    )
    compared_world = {"scenarios": compared.get("scenarios", [])}
    scenario_assignment = scenario_assignment_metric(
        reference_nodes, compared_nodes, node_matches, compared_world)
    multi_scenario_assignment = dict(scenario_assignment)
    multi_scenario_assignment["applicable"] = (
        scenario_assignment["expected_scenario_count"] > 1)
    entity_scenario_assignment = scenario_assignment_metric(
        reference_entities, compared_entities, entity_matches, compared_world)
    multi_scenario_entity_assignment = dict(entity_scenario_assignment)
    multi_scenario_entity_assignment["applicable"] = (
        entity_scenario_assignment["expected_scenario_count"] > 1)
    return {
        "module_id": reference.get("module_id", compared.get("module_id", "")),
        "nodes": node_score,
        "typed_nodes": typed_node_score,
        "typed_edges": typed_edge_agreement(
            reference.get("edges", []), compared.get("edges", []),
            compared_to_reference),
        "entities": entity_score,
        "forbidden_scopes": scope_score,
        "scenario_assignment": scenario_assignment,
        "multi_scenario_assignment": multi_scenario_assignment,
        "entity_scenario_assignment": entity_scenario_assignment,
        "multi_scenario_entity_assignment": multi_scenario_entity_assignment,
    }


def macro(reports: list[dict], metric: str, field: str = "f1") -> float:
    rows = [row[metric] for row in reports if row[metric].get("applicable", True)]
    return round(sum(row[field] for row in rows) / len(rows), 4) if rows else 0.0


def compare_annotations(reference_path: Path, compared_path: Path,
                        source_root: Path | None = None) -> dict:
    reference = rows_by_module(reference_path)
    compared = rows_by_module(compared_path)
    common = sorted(set(reference) & set(compared))
    reports = []
    validation = []
    for module_id in common:
        left, left_name = reference[module_id]
        right, right_name = compared[module_id]
        validation.extend([
            {"annotator": "a", **validate_gold(
                left, left_name, source_root)},
            {"annotator": "b", **validate_gold(
                right, right_name, source_root)},
        ])
        reports.append(score_annotations(left, right))
    missing_from_a = sorted(set(compared) - set(reference))
    missing_from_b = sorted(set(reference) - set(compared))
    return {
        "valid": (
            bool(common)
            and not missing_from_a
            and not missing_from_b
            and all(row["valid"] for row in validation)
        ),
        "module_count": len(common),
        "missing_from_a": missing_from_a,
        "missing_from_b": missing_from_b,
        "aggregate": {
            "node_macro_f1": macro(reports, "nodes"),
            "typed_node_macro_f1": macro(reports, "typed_nodes"),
            "typed_edge_macro_f1": macro(reports, "typed_edges"),
            "entity_macro_f1": macro(reports, "entities"),
            "forbidden_scope_macro_f1": macro(reports, "forbidden_scopes"),
            "scenario_assignment_macro": macro(
                reports, "scenario_assignment", "accuracy"),
            "multi_scenario_assignment_macro": macro(
                reports, "multi_scenario_assignment", "accuracy"),
            "entity_scenario_assignment_macro": macro(
                reports, "entity_scenario_assignment", "accuracy"),
            "multi_scenario_entity_assignment_macro": macro(
                reports, "multi_scenario_entity_assignment", "accuracy"),
        },
        "modules": reports,
        "validation": validation,
    }


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--annotator-a", type=Path, required=True)
    cli.add_argument("--annotator-b", type=Path, required=True)
    cli.add_argument("--source-root", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    report = compare_annotations(
        args.annotator_a, args.annotator_b, args.source_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

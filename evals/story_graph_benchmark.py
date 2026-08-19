#!/usr/bin/env python3
"""Deterministic structural benchmark for parsed AIKP world books.

Gold files contain derived labels and graph annotations only. Source module prose and
PDFs remain outside the repository. The scorer does not call an LLM or load an API key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


def normalize(value: object) -> str:
    return "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", str(value or "").lower()))


def tokens(value: object) -> set[str]:
    raw = str(value or "").lower()
    words = set(re.findall(r"[a-z0-9]+", raw))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", raw))
    if chinese:
        words.update(chinese[index:index + 2]
                     for index in range(max(1, len(chinese) - 1)))
    return {word for word in words if word}


def labels(row: dict) -> list[str]:
    values = [row.get("name"), row.get("title"), row.get("label"), row.get("id")]
    values.extend(row.get("aliases", []) if isinstance(row.get("aliases"), list) else [])
    return [str(value) for value in values if value]


def label_similarity(left: dict, right: dict) -> float:
    best = 0.0
    for a in labels(left):
        na, ta = normalize(a), tokens(a)
        for b in labels(right):
            nb, tb = normalize(b), tokens(b)
            if not na or not nb:
                continue
            if na == nb:
                score = 1.0
            elif min(len(na), len(nb)) >= 4 and (na in nb or nb in na):
                score = 0.88
            else:
                union = ta | tb
                score = len(ta & tb) / len(union) if union else 0.0
            best = max(best, score)
    return best


def canonical_type(value: object) -> str:
    kind = normalize(value)
    if kind in {"npc", "character", "monster", "person"}:
        return "npc"
    if kind in {"item", "object", "door", "container", "document", "uniqueobject"}:
        return "object"
    if kind in {"clue", "evidence"}:
        return "clue"
    return kind


def canonical_node_kind(value: object) -> str:
    kind = normalize(value)
    aliases = {
        "decision": "choice",
        "revelation": "clue",
        "climax": "encounter",
        "resolution": "ending",
        "scene": "event",
        "investigation": "event",
        "transition": "event",
        "optional": "event",
        "conditional": "event",
        "outcome": "event",
        "travel": "event",
    }
    return aliases.get(kind, kind)


def optimal_match(gold: list[dict], predicted: list[dict], threshold: float = 0.55,
                  require_type: bool = False,
                  require_kind: bool = False) -> tuple[dict[str, str], list[dict]]:
    """Return a maximum-cardinality, maximum-similarity bipartite matching."""
    gold_rows = sorted(gold, key=lambda row: str(row["id"]))
    predicted_rows = sorted(predicted, key=lambda row: str(row["id"]))
    candidates: list[tuple[int, int, float]] = []
    for gold_index, gold_row in enumerate(gold_rows):
        for pred_index, pred_row in enumerate(predicted_rows):
            if require_type and canonical_type(gold_row.get("type")) != canonical_type(
                    pred_row.get("type")):
                continue
            if require_kind and canonical_node_kind(
                    gold_row.get("kind")) != canonical_node_kind(pred_row.get("kind")):
                continue
            score = label_similarity(gold_row, pred_row)
            if score >= threshold:
                candidates.append((gold_index, pred_index, score))

    # Successive shortest augmenting paths find maximum cardinality first. Negative
    # similarity costs then maximize total similarity among matchings of that size.
    source = 0
    gold_offset = 1
    pred_offset = gold_offset + len(gold_rows)
    sink = pred_offset + len(predicted_rows)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int, cost: int) -> list[int]:
        forward = [end, len(graph[end]), capacity, cost]
        backward = [start, len(graph[start]), 0, -cost]
        graph[start].append(forward)
        graph[end].append(backward)
        return forward

    for index in range(len(gold_rows)):
        add_edge(source, gold_offset + index, 1, 0)
    for index in range(len(predicted_rows)):
        add_edge(pred_offset + index, sink, 1, 0)
    candidate_edges: list[tuple[int, int, float, list[int]]] = []
    for gold_index, pred_index, score in candidates:
        edge = add_edge(
            gold_offset + gold_index,
            pred_offset + pred_index,
            1,
            -round(score * 1_000_000),
        )
        candidate_edges.append((gold_index, pred_index, score, edge))

    node_count = len(graph)
    while True:
        distance: list[int | None] = [None] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distance[source] = 0
        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distance[node] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge[2] <= 0:
                        continue
                    candidate_distance = distance[node] + edge[3]
                    if (distance[edge[0]] is None
                            or candidate_distance < distance[edge[0]]):
                        distance[edge[0]] = candidate_distance
                        previous[edge[0]] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = parent

    selected = [row for row in candidate_edges if row[3][2] == 0]
    matches = {
        str(gold_rows[gold_index]["id"]): str(predicted_rows[pred_index]["id"])
        for gold_index, pred_index, _score, _edge in selected
    }
    trace = [
        {
            "gold_id": str(gold_rows[gold_index]["id"]),
            "prediction_id": str(predicted_rows[pred_index]["id"]),
            "similarity": round(score, 4),
        }
        for gold_index, pred_index, score, _edge in selected
    ]
    trace.sort(key=lambda row: (row["gold_id"], row["prediction_id"]))
    return matches, trace


def greedy_match(gold: list[dict], predicted: list[dict], threshold: float = 0.55,
                 require_type: bool = False,
                 require_kind: bool = False) -> tuple[dict[str, str], list[dict]]:
    """Backward-compatible name for the optimal matcher."""
    return optimal_match(gold, predicted, threshold, require_type, require_kind)


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def extract_nodes(world: dict) -> list[dict]:
    tree_nodes = world.get("story_tree", {}).get("nodes", [])
    if isinstance(tree_nodes, list) and tree_nodes:
        return [dict(row) for row in tree_nodes if isinstance(row, dict) and row.get("id")
                and row.get("playable", row.get("kind") not in {"root", "act"})]
    beats = world.get("story_beats", [])
    if isinstance(beats, list) and beats:
        return [dict(row) for row in beats if isinstance(row, dict) and row.get("id")]
    return [dict(row, id=scene_id) for scene_id, row in world.get("scenes", {}).items()
            if isinstance(row, dict)]


def extract_edges(world: dict, nodes: list[dict]) -> list[dict]:
    result = []
    for row in world.get("story_tree", {}).get("relations", []):
        if isinstance(row, dict) and row.get("from") and row.get("to"):
            result.append({"from": str(row["from"]), "to": str(row["to"]),
                           "type": str(row.get("type", "before"))})
    existing = {(row["from"], row["to"]) for row in result}
    for node in nodes:
        for target in node.get("successors", []):
            edge = (str(node["id"]), str(target))
            if edge not in existing:
                result.append({"from": edge[0], "to": edge[1], "type": "before"})
    return result


def extract_entities(world: dict) -> list[dict]:
    result: dict[str, dict] = {}
    registry = world.get("entity_registry", [])
    if isinstance(registry, list):
        for row in registry:
            if isinstance(row, dict) and row.get("id"):
                result[str(row["id"])] = dict(row)
    for entity_id, row in world.get("entities", {}).items():
        if not isinstance(row, dict):
            continue
        merged = dict(result.get(str(entity_id), {}))
        merged.update(row)
        merged["id"] = str(entity_id)
        result[str(entity_id)] = merged
    return list(result.values())


def extract_navigable_scenes(world: dict) -> list[dict]:
    result = []
    for scene_id, row in world.get("scenes", {}).items():
        if isinstance(row, dict) and row.get("navigable", True):
            result.append(dict(row, id=str(scene_id)))
    return result


def provenance_metric(world: dict) -> dict:
    verified = total = 0
    collections = ("scenes", "npcs", "objects", "items", "clues", "events",
                   "state_transitions", "knowledge_changes", "promises_payoffs",
                   "branch_edges")
    for node in world.get("detailed_story_nodes", []):
        if not isinstance(node, dict):
            continue
        for key in collections:
            for row in node.get(key, []):
                if isinstance(row, dict):
                    total += 1
                    verified += int(row.get("source_verified") is True)
    return {"verified": verified, "total": total,
            "coverage": round(verified / total, 4) if total else 0.0}


def graph_closure(nodes: list[dict], edges: list[dict]) -> dict:
    ids = {str(row["id"]) for row in nodes}
    closed = sum(1 for row in edges if row["from"] in ids and row["to"] in ids)
    return {"closed": closed, "total": len(edges),
            "coverage": round(closed / len(edges), 4) if edges else 1.0}


def scenario_assignment_metric(
    gold_rows: list[dict],
    predicted_rows: list[dict],
    matches: dict[str, str],
    world: dict,
) -> dict:
    """Score scenario clustering without requiring identical generated ids."""
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    pred_by_id = {str(row["id"]): row for row in predicted_rows}
    paired = [
        (gold_id, pred_id, str(gold_by_id[gold_id].get("scenario_id", "")),
         str(pred_by_id[pred_id].get("scenario_id", "")))
        for gold_id, pred_id in matches.items()
    ]
    assigned = sum(bool(pred_scenario) for _, _, _, pred_scenario in paired)
    coverage = assigned / len(paired) if paired else 0.0
    expected_count = len({str(row.get("scenario_id")) for row in gold_rows
                          if row.get("scenario_id")})
    if not expected_count:
        return {
            "applicable": False,
            "matched_rows": len(paired),
            "assigned_rows": assigned,
            "assignment_coverage": 1.0,
            "pairwise": {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0,
                         "recall": 1.0, "f1": 1.0},
            "pairwise_accuracy": 1.0,
            "expected_scenario_count": 0,
            "predicted_scenario_count": 0,
            "scenario_count_accuracy": 1.0,
            "accuracy": 1.0,
            "gold_scenarios_in_matches": 0,
            "predicted_scenarios_in_matches": 0,
        }

    tp = fp = fn = agreement = pair_total = 0
    for index, left in enumerate(paired):
        for right in paired[index + 1:]:
            gold_same = bool(left[2] and right[2] and left[2] == right[2])
            # Missing membership is treated as an unassigned singleton, not a shared id.
            pred_same = bool(left[3] and right[3] and left[3] == right[3])
            tp += int(gold_same and pred_same)
            fp += int(not gold_same and pred_same)
            fn += int(gold_same and not pred_same)
            agreement += int(gold_same == pred_same)
            pair_total += 1
    pair_score = prf(tp, fp, fn)
    if not (tp or fp or fn):
        pair_score.update({"precision": 1.0, "recall": 1.0, "f1": 1.0})
    pair_accuracy = agreement / pair_total if pair_total else 1.0

    gold_scenarios = {scenario for _, _, scenario, _ in paired if scenario}
    predicted_scenarios = {scenario for _, _, _, scenario in paired if scenario}
    declared_registry = {
        str(row.get("id")) for row in world.get("scenarios", [])
        if isinstance(row, dict) and row.get("id")
    }
    predicted_count = len(declared_registry or predicted_scenarios)
    count_accuracy = (
        min(expected_count, predicted_count) / max(expected_count, predicted_count)
        if expected_count or predicted_count else 1.0
    )
    accuracy = coverage * pair_accuracy * count_accuracy
    return {
        "applicable": True,
        "matched_rows": len(paired),
        "assigned_rows": assigned,
        "assignment_coverage": round(coverage, 4),
        "pairwise": pair_score,
        "pairwise_accuracy": round(pair_accuracy, 4),
        "expected_scenario_count": expected_count,
        "predicted_scenario_count": predicted_count,
        "scenario_count_accuracy": round(count_accuracy, 4),
        "accuracy": round(accuracy, 4),
        "gold_scenarios_in_matches": len(gold_scenarios),
        "predicted_scenarios_in_matches": len(predicted_scenarios),
    }


def score_world(gold: dict, world: dict) -> dict:
    gold_nodes = [row for row in gold.get("nodes", []) if row.get("playable", True)]
    predicted_nodes = extract_nodes(world)
    node_matches, node_trace = optimal_match(gold_nodes, predicted_nodes)
    node_score = prf(len(node_matches), len(predicted_nodes) - len(node_matches),
                     len(gold_nodes) - len(node_matches))
    typed_node_matches, typed_node_trace = optimal_match(
        gold_nodes, predicted_nodes, require_kind=True)
    typed_node_score = prf(
        len(typed_node_matches),
        len(predicted_nodes) - len(typed_node_matches),
        len(gold_nodes) - len(typed_node_matches),
    )

    predicted_edges = extract_edges(world, predicted_nodes)
    pred_to_gold = {pred_id: gold_id for gold_id, pred_id in node_matches.items()}
    gold_edge_set = {(str(row["from"]), str(row["to"]), str(row.get("type", "before")))
                     for row in gold.get("edges", [])}
    scored_predicted_edges = {
        (pred_to_gold.get(row["from"], f"__unmatched_pred__:{row['from']}"),
         pred_to_gold.get(row["to"], f"__unmatched_pred__:{row['to']}"),
         row["type"])
        for row in predicted_edges
    }
    mapped_edges = {
        edge for edge in scored_predicted_edges
        if not edge[0].startswith("__unmatched_pred__:")
        and not edge[1].startswith("__unmatched_pred__:")
    }
    edge_score = prf(
        len(gold_edge_set & scored_predicted_edges),
        len(scored_predicted_edges - gold_edge_set),
        len(gold_edge_set - scored_predicted_edges),
    )
    edge_score["unmatched_endpoint_edges"] = sum(
        edge[0].startswith("__unmatched_pred__:")
        or edge[1].startswith("__unmatched_pred__:")
        for edge in scored_predicted_edges
    )
    scenario_by_node = {
        str(row["id"]): str(row.get("scenario_id", "")) for row in gold_nodes
    }
    gold_cross_edges = {
        (source, target, edge_type)
        for source, target, edge_type in mapped_edges
        if scenario_by_node.get(source) and scenario_by_node.get(target)
        and scenario_by_node[source] != scenario_by_node[target]
    }
    predicted_scenario_by_node = {
        str(row["id"]): str(row.get("scenario_id", "")) for row in predicted_nodes
    }
    declared_cross_edges = {
        (row["from"], row["to"], row["type"])
        for row in predicted_edges
        if predicted_scenario_by_node.get(row["from"])
        and predicted_scenario_by_node.get(row["to"])
        and predicted_scenario_by_node[row["from"]]
        != predicted_scenario_by_node[row["to"]]
    }
    violations = [
        {"from": source, "to": target, "type": edge_type, "basis": "gold_membership"}
        for source, target, edge_type in sorted(gold_cross_edges)
    ] + [
        {"from": source, "to": target, "type": edge_type,
         "basis": "predicted_membership"}
        for source, target, edge_type in sorted(declared_cross_edges)
    ]
    violating_predicted_edges = {
        (row["from"], row["to"], row["type"])
        for row in predicted_edges
        if (row["from"], row["to"], row["type"]) in declared_cross_edges
        or (pred_to_gold.get(row["from"]), pred_to_gold.get(row["to"]), row["type"])
        in gold_cross_edges
    }
    edge_isolation_accuracy = (
        1 - len(violating_predicted_edges) / len(predicted_edges)
        if predicted_edges else 1.0
    )
    scenario_isolation = {
        "cross_scenario_edges": len(violating_predicted_edges),
        "predicted_edge_total": len(predicted_edges),
        "mapped_edge_total": len(mapped_edges),
        "accuracy": round(edge_isolation_accuracy, 4),
        "violations": violations,
    }
    scenario_assignment = scenario_assignment_metric(
        gold_nodes, predicted_nodes, node_matches, world)
    multi_scenario_assignment = dict(scenario_assignment)
    multi_scenario_assignment["applicable"] = (
        scenario_assignment["expected_scenario_count"] > 1)

    gold_entities = gold.get("entities", [])
    predicted_entities = extract_entities(world)
    entity_matches, entity_trace = optimal_match(
        gold_entities, predicted_entities, require_type=True)
    entity_score = prf(len(entity_matches), len(predicted_entities) - len(entity_matches),
                       len(gold_entities) - len(entity_matches))
    entity_scenario_assignment = scenario_assignment_metric(
        gold_entities, predicted_entities, entity_matches, world)
    multi_scenario_entity_assignment = dict(entity_scenario_assignment)
    multi_scenario_entity_assignment["applicable"] = (
        entity_scenario_assignment["expected_scenario_count"] > 1)

    scenes = extract_navigable_scenes(world)
    forbidden_hits = []
    for forbidden in gold.get("forbidden_navigable_scenes", []):
        best = max((label_similarity(forbidden, scene) for scene in scenes), default=0.0)
        if best >= 0.55:
            forbidden_hits.append({"id": forbidden["id"], "similarity": round(best, 4)})
    forbidden_total = len(gold.get("forbidden_navigable_scenes", []))
    scope = {"false_positives": len(forbidden_hits), "forbidden_total": forbidden_total,
             "accuracy": round(1 - len(forbidden_hits) / forbidden_total, 4)
             if forbidden_total else 1.0, "hits": forbidden_hits}

    return {
        "module_id": gold["module_id"],
        "nodes": node_score, "typed_nodes": typed_node_score,
        "typed_edges": edge_score, "entities": entity_score,
        "narrative_scope": scope, "scenario_isolation": scenario_isolation,
        "scenario_assignment": scenario_assignment,
        "multi_scenario_assignment": multi_scenario_assignment,
        "entity_scenario_assignment": entity_scenario_assignment,
        "multi_scenario_entity_assignment": multi_scenario_entity_assignment,
        "provenance": provenance_metric(world),
        "graph_closure": graph_closure(predicted_nodes, predicted_edges),
        "matches": {"nodes": node_trace, "typed_nodes": typed_node_trace,
                    "entities": entity_trace},
    }


def aggregate(reports: list[dict]) -> dict:
    result = {"module_count": len(reports)}
    for metric in ("nodes", "typed_nodes", "typed_edges", "entities"):
        tp = sum(row[metric]["tp"] for row in reports)
        fp = sum(row[metric]["fp"] for row in reports)
        fn = sum(row[metric]["fn"] for row in reports)
        result[metric] = prf(tp, fp, fn)
        result[metric]["macro_f1"] = round(
            sum(row[metric]["f1"] for row in reports) / len(reports), 4) if reports else 0.0
    for metric, field in (("narrative_scope", "accuracy"),
                          ("scenario_isolation", "accuracy"),
                          ("scenario_assignment", "accuracy"),
                          ("multi_scenario_assignment", "accuracy"),
                          ("entity_scenario_assignment", "accuracy"),
                          ("multi_scenario_entity_assignment", "accuracy"),
                          ("provenance", "coverage"),
                          ("graph_closure", "coverage")):
        applicable = [row for row in reports
                      if row[metric].get("applicable", True)]
        result[metric] = {
            "macro": round(
                sum(row[metric][field] for row in applicable) / len(applicable), 4
            ) if applicable else 0.0,
            "module_count": len(applicable),
        }
    return result


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def source_hash_matches(gold: dict, source_root: Path | None) -> bool | None:
    if not source_root:
        return None
    relative = gold.get("source", {}).get("local_filename")
    expected = gold.get("source", {}).get("sha256")
    if not relative or not expected:
        return None
    path = source_root / relative
    if not path.exists():
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == expected


def evaluate_directory(gold_dir: Path, prediction_dir: Path,
                       source_root: Path | None = None) -> dict:
    reports = []
    missing = []
    for gold_path in sorted(gold_dir.glob("*.json")):
        gold = load_json(gold_path)
        prediction_path = prediction_dir / f"{gold['module_id']}.json"
        if not prediction_path.exists():
            missing.append(gold["module_id"])
            continue
        report = score_world(gold, load_json(prediction_path))
        report["source_hash_matches"] = source_hash_matches(gold, source_root)
        reports.append(report)
    return {"aggregate": aggregate(reports), "modules": reports,
            "missing_predictions": missing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path,
                        default=Path(__file__).with_name("gold"))
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_directory(args.gold_dir, args.predictions_dir, args.source_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

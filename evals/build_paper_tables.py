#!/usr/bin/env python3
"""Build paper-ready tables from versioned evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


RISK_METRICS = (
    ("Unsupported world mutation", "unsupported_world_mutations"),
    ("Hidden-information leak", "hidden_information_leaks"),
    ("Location-continuity violation", "location_continuity_violations"),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(gold_dir: Path, corpus_manifest: Path,
                  interactive_report: Path,
                  comparisons: list[Path] | None = None,
                  baselines: list[Path] | None = None) -> dict:
    manifest = load_json(corpus_manifest)
    documents = [row for row in manifest.get("documents", [])
                 if isinstance(row, dict)]
    annotations = Counter(str(row.get("annotation", "unspecified"))
                          for row in documents)
    splits = Counter(str(row.get("split", "unspecified")) for row in documents)

    gold_rows = [load_json(path) for path in sorted(gold_dir.glob("*.json"))]
    gold_counts = {
        "modules": len(gold_rows),
        "scenarios": sum(len(row.get("scenarios", [])) for row in gold_rows),
        "nodes": sum(len(row.get("nodes", [])) for row in gold_rows),
        "edges": sum(len(row.get("edges", [])) for row in gold_rows),
        "entities": sum(len(row.get("entities", [])) for row in gold_rows),
        "forbidden_scenes": sum(
            len(row.get("forbidden_navigable_scenes", [])) for row in gold_rows),
    }

    interactive = load_json(interactive_report).get("aggregate", {})
    parser_comparisons = []
    for path in comparisons or []:
        report = load_json(path)
        parser_comparisons.append({
            "path": str(path),
            "reference": report.get("reference_condition", ""),
            "compared": report.get("compared_condition", ""),
            "bootstrap_unit": report.get("bootstrap_unit", ""),
            "metrics": report.get("metrics", {}),
        })

    parser_baselines = []
    for path in baselines or []:
        report = load_json(path)
        parser_baselines.append({
            "path": str(path),
            "baseline_id": report.get("experiment", {}).get(
                "baseline_id", path.stem),
            "aggregate": report.get("aggregate", {}),
        })

    limitations = [
        "Gold-v1 is preliminary until an independent second annotation and adjudication are complete.",
        "Recorded-narration playtests isolate deterministic runtime behavior; they do not measure open-ended narrator fluency or total hallucination rate.",
    ]
    repeated = interactive.get("pass_power_k_by_case", {})
    if repeated and any(not row.get("sufficient_runs", False)
                        for row in repeated.values()):
        limitations.append(
            "Some trajectories have fewer than k runs; pass^k is null for those cases.")
    if not parser_comparisons:
        limitations.append(
            "No real-provider parser comparison was supplied, so no model ablation effect is reported.")
    if parser_baselines and not parser_comparisons:
        limitations.append(
            "The deterministic parser result is a no-model structural lower bound, not a substitute for a real-provider baseline or ablation.")

    return {
        "schema_version": 1,
        "corpus": {
            "documents": len(documents),
            "annotation_counts": dict(sorted(annotations.items())),
            "split_counts": dict(sorted(splits.items())),
        },
        "gold": gold_counts,
        "interactive": interactive,
        "parser_baselines": parser_baselines,
        "parser_comparisons": parser_comparisons,
        "limitations": limitations,
    }


def markdown(summary: dict) -> str:
    corpus, gold = summary["corpus"], summary["gold"]
    interactive = summary["interactive"]
    lines = [
        "# AIKP Evaluation Snapshot",
        "",
        "## Dataset",
        "",
        "| Documents | Gold modules | Scenarios | Nodes | Typed edges | Entities | Forbidden scenes |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (f"| {corpus['documents']} | {gold['modules']} | {gold['scenarios']} | "
         f"{gold['nodes']} | {gold['edges']} | {gold['entities']} | "
         f"{gold['forbidden_scenes']} |"),
        "",
        "Annotation states: " + ", ".join(
            f"`{key}`={value}"
            for key, value in corpus["annotation_counts"].items()),
        "",
        "## Interactive Runtime",
        "",
        "| Cases | Runs | Runs/case | Turns | Task success | Macro branch coverage | Valid actions | Invalid actions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (f"| {interactive.get('case_count', interactive.get('run_count', 0))} | "
         f"{interactive.get('run_count', 0)} | "
         f"{interactive.get('min_runs_per_case', 1)}-"
         f"{interactive.get('max_runs_per_case', 1)} | "
         f"{interactive.get('turns', 0)} | "
         f"{interactive.get('task_success_rate')} | "
         f"{interactive.get('macro_branch_coverage')} | "
         f"{interactive.get('valid_action_false_rejection', {}).get('eligible_turns', 0)} | "
         f"{interactive.get('invalid_action_acceptance', {}).get('eligible_turns', 0)} |"),
        "",
        "| Risk | Violations | Eligible turns | Per 100 turns |",
        "|---|---:|---:|---:|",
    ]
    for label, key in RISK_METRICS:
        row = interactive.get(key, {})
        lines.append(
            f"| {label} | {row.get('violations', 0)} | "
            f"{row.get('eligible_turns', 0)} | {row.get('per_100_turns')} |")

    if summary.get("parser_baselines"):
        lines.extend([
            "", "## Deterministic Parser Lower Bounds", "",
            "| Baseline | Node macro F1 | Typed-node macro F1 | Typed-edge macro F1 | Entity macro F1 | Narrative-scope macro | Multi-scenario assignment macro |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for baseline in summary["parser_baselines"]:
            aggregate = baseline["aggregate"]
            lines.append(
                f"| {baseline['baseline_id']} | "
                f"{aggregate.get('nodes', {}).get('macro_f1')} | "
                f"{aggregate.get('typed_nodes', {}).get('macro_f1')} | "
                f"{aggregate.get('typed_edges', {}).get('macro_f1')} | "
                f"{aggregate.get('entities', {}).get('macro_f1')} | "
                f"{aggregate.get('narrative_scope', {}).get('macro')} | "
                f"{aggregate.get('multi_scenario_assignment', {}).get('macro')} |")

    for comparison in summary["parser_comparisons"]:
        lines.extend([
            "", f"## Parser: {comparison['reference']} vs {comparison['compared']}", "",
            "| Metric | Paired delta | 95% CI | Modules | Paired runs |",
            "|---|---:|---:|---:|---:|",
        ])
        for name, row in comparison["metrics"].items():
            interval = row.get("bootstrap_95_ci", [None, None])
            lines.append(
                f"| {name} | {row.get('paired_delta')} | "
                f"[{interval[0]}, {interval[1]}] | {row.get('module_count')} | "
                f"{row.get('paired_run_count')} |")

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cli = argparse.ArgumentParser()
    cli.add_argument("--gold-dir", type=Path, default=root / "evals" / "gold")
    cli.add_argument("--corpus-manifest", type=Path,
                     default=root / "evals" / "corpus_manifest.json")
    cli.add_argument("--interactive-report", type=Path, required=True)
    cli.add_argument("--parser-comparison", type=Path, action="append", default=[])
    cli.add_argument("--parser-baseline", type=Path, action="append", default=[])
    cli.add_argument("--output-json", type=Path, required=True)
    cli.add_argument("--output-markdown", type=Path, required=True)
    args = cli.parse_args()
    summary = build_summary(
        args.gold_dir, args.corpus_manifest, args.interactive_report,
        args.parser_comparison, args.parser_baseline)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({
        "json": str(args.output_json), "markdown": str(args.output_markdown),
        "documents": summary["corpus"]["documents"],
        "gold_modules": summary["gold"]["modules"],
        "interactive_cases": summary["interactive"].get(
            "case_count", summary["interactive"].get("run_count", 0)),
        "interactive_runs": summary["interactive"].get("run_count", 0),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

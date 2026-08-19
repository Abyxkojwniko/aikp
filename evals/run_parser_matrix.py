#!/usr/bin/env python3
"""Run reproducible parser baselines and ablations over the corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parser import ModuleParser, PARSER_ABLATIONS
from story_graph_benchmark import evaluate_directory


DEFAULT_CONDITIONS = (
    "legacy", "no_document_map", "no_semantic_judge", "no_node_repair", "full",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(root: Path, pattern: str = "*.json") -> str:
    """Hash file names and bytes so evaluation labels are immutable per run."""
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"no files matching {pattern!r} under {root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def source_tree_sha256() -> str:
    """Fingerprint executable experiment code, including uncommitted files."""
    digest = hashlib.sha256()
    patterns = (
        "backend/**/*.py", "evals/*.py", "requirements*.txt", "pyproject.toml",
    )
    paths = {
        path for pattern in patterns for path in ROOT.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def git_state() -> dict:
    def command(*args: str) -> str:
        try:
            return subprocess.run(
                args, cwd=ROOT, check=True, capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "dirty": bool(command("git", "status", "--porcelain")),
    }


def select_documents(manifest: dict, module_ids: set[str] | None = None,
                     annotation: str = "gold-v1") -> list[dict]:
    documents = [row for row in manifest.get("documents", []) if isinstance(row, dict)]
    if module_ids:
        documents = [row for row in documents if str(row.get("id")) in module_ids]
        missing = module_ids - {str(row.get("id")) for row in documents}
        if missing:
            raise ValueError("unknown module ids: " + ", ".join(sorted(missing)))
    elif annotation:
        documents = [row for row in documents if row.get("annotation") == annotation]
    return sorted(documents, key=lambda row: str(row.get("id", "")))


def build_jobs(documents: list[dict], conditions: list[str], repeats: int,
               output_dir: Path, job_order_seed: int = 20260818) -> list[dict]:
    """Interleave conditions within repeat/module blocks to reduce time drift."""
    jobs = []
    sequence = 0
    for repeat in range(1, repeats + 1):
        for document in documents:
            block = list(conditions)
            random.Random(
                f"{job_order_seed}|{repeat}|{document['id']}"
            ).shuffle(block)
            for condition in block:
                sequence += 1
                jobs.append({
                    "module_id": str(document["id"]),
                    "document": document,
                    "condition": condition,
                    "repeat": repeat,
                    "job_sequence": sequence,
                    "run_dir": output_dir / condition / f"repeat-{repeat:03d}",
                })
    return jobs


def job_order_sha256(jobs: list[dict]) -> str:
    payload = [
        [job["job_sequence"], job["repeat"], job["module_id"], job["condition"]]
        for job in jobs
    ]
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def execute_job(job: dict, source_root: Path, model: str, base_url: str,
                api_key: str, parser_factory: Callable[..., object] = ModuleParser,
                overwrite: bool = False) -> dict:
    run_dir: Path = job["run_dir"]
    module_id = job["module_id"]
    prediction_path = run_dir / "predictions" / f"{module_id}.json"
    metadata_path = run_dir / "runs" / f"{module_id}.json"
    if metadata_path.exists() and prediction_path.exists() and not overwrite:
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        prediction_hash = sha256(prediction_path)
        if (prior.get("status") == "success"
                and prior.get("prediction_sha256") == prediction_hash):
            return {**prior, "resumed": True}
        if prior.get("status") == "success":
            return {
                **prior,
                "status": "failed",
                "error": "prediction artifact hash mismatch; rerun with --overwrite",
                "resumed": False,
            }

    source_path = source_root / job["document"]["local_filename"]
    expected_hash = str(job["document"].get("sha256", ""))
    actual_hash = sha256(source_path) if source_path.exists() else ""
    metadata = {
        "schema_version": 1,
        "module_id": module_id,
        "condition": job["condition"],
        "repeat": job["repeat"],
        "job_sequence": job.get("job_sequence"),
        "model": model,
        "base_url": base_url,
        "source_file": str(source_path),
        "source_sha256": actual_hash,
        "source_hash_matches": bool(expected_hash and actual_hash == expected_hash),
        "started_at": utc_now(),
        "status": "running",
    }
    atomic_json(metadata_path, metadata)
    started = time.perf_counter()
    try:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not metadata["source_hash_matches"]:
            raise ValueError(f"source hash mismatch for {module_id}")
        text = source_path.read_text(encoding="utf-8", errors="replace")
        parser = parser_factory(
            api_key=api_key, base_url=base_url, model=model,
            ablation=job["condition"],
        )
        world = parser.parse(text)
        world["_experiment"] = {
            "condition": job["condition"], "repeat": job["repeat"],
            "module_id": module_id, "source_sha256": actual_hash,
        }
        atomic_json(prediction_path, world)
        telemetry = (
            parser.experiment_usage() if hasattr(parser, "experiment_usage") else {})
        provider_failures = int(telemetry.get("failures", 0) or 0)
        metadata.update({
            "status": "failed" if provider_failures else "success",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "parser_mode": world.get("_parser_mode", ""),
            "parser_ablation": world.get("_parser_ablation", ""),
            "validation_issue_count": len(world.get("_validation_issues", [])),
            "telemetry": telemetry,
            "prediction_sha256": sha256(prediction_path),
            "finished_at": utc_now(),
        })
        if provider_failures:
            metadata["error"] = (
                f"provider failures during parse: {provider_failures}; "
                "prediction excluded from evaluation"
            )
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
            "finished_at": utc_now(),
        })
    atomic_json(metadata_path, metadata)
    return metadata


def score_completed_repeats(output_dir: Path, gold_dir: Path,
                            source_root: Path, conditions: list[str],
                            repeats: int, expected_modules: list[str]) -> list[dict]:
    summaries = []
    for condition in conditions:
        for repeat in range(1, repeats + 1):
            run_dir = output_dir / condition / f"repeat-{repeat:03d}"
            prediction_dir = run_dir / "predictions"
            if not prediction_dir.exists():
                continue
            run_metadata = {}
            for module_id in expected_modules:
                metadata_path = run_dir / "runs" / f"{module_id}.json"
                if metadata_path.exists():
                    run_metadata[module_id] = json.loads(
                        metadata_path.read_text(encoding="utf-8"))
            invalid = [
                module_id for module_id in expected_modules
                if run_metadata.get(module_id, {}).get("status") != "success"
            ]
            if invalid:
                status = {
                    "experiment": {"condition": condition, "repeat": repeat},
                    "status": "invalid",
                    "invalid_or_missing_runs": invalid,
                }
                atomic_json(run_dir / "benchmark_status.json", status)
                summaries.append({
                    "condition": condition, "repeat": repeat,
                    "status": "invalid", "invalid_or_missing_runs": invalid,
                })
                continue
            report = evaluate_directory(gold_dir, prediction_dir, source_root)
            report["experiment"] = {"condition": condition, "repeat": repeat}
            atomic_json(run_dir / "benchmark.json", report)
            atomic_json(run_dir / "benchmark_status.json", {
                "experiment": {"condition": condition, "repeat": repeat},
                "status": "success",
                "modules_scored": report["aggregate"]["module_count"],
            })
            summaries.append({
                "condition": condition,
                "repeat": repeat,
                "status": "success",
                "modules_scored": report["aggregate"]["module_count"],
                "missing_predictions": report["missing_predictions"],
            })
    return summaries


def run_matrix(manifest_path: Path, source_root: Path, output_dir: Path,
               conditions: list[str], repeats: int, model: str, base_url: str,
               api_key: str, module_ids: set[str] | None = None,
               annotation: str = "gold-v1", gold_dir: Path | None = None,
               dry_run: bool = False, overwrite: bool = False,
               parser_factory: Callable[..., object] = ModuleParser,
               job_order_seed: int = 20260818) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = select_documents(manifest, module_ids, annotation)
    if not documents:
        raise ValueError("no corpus documents selected")
    invalid = sorted(set(conditions) - PARSER_ABLATIONS)
    if invalid:
        raise ValueError("unknown conditions: " + ", ".join(invalid))
    gold_root = gold_dir or ROOT / "evals" / "gold"
    jobs = build_jobs(
        documents, conditions, repeats, output_dir, job_order_seed)
    experiment = {
        "schema_version": 1,
        "created_at": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "conditions": conditions,
        "repeats": repeats,
        "model": model,
        "base_url": base_url,
        "modules": [row["id"] for row in documents],
        "job_count": len(jobs),
        "job_order_policy": "repeat-module-blocked-condition-shuffle",
        "job_order_seed": job_order_seed,
        "job_order_sha256": job_order_sha256(jobs),
        "gold_dir": str(gold_root.resolve()),
        "gold_tree_sha256": directory_sha256(gold_root),
        "python": platform.python_version(),
        "git": git_state(),
        "source_tree_sha256": source_tree_sha256(),
        "dry_run": dry_run,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = output_dir / "experiment.json"
    if experiment_path.exists() and not overwrite:
        existing = json.loads(experiment_path.read_text(encoding="utf-8"))
        identity_fields = (
            "manifest", "manifest_sha256", "source_root", "conditions",
            "repeats", "model", "base_url", "modules", "job_count",
            "source_tree_sha256", "gold_dir", "gold_tree_sha256",
            "job_order_policy", "job_order_seed", "job_order_sha256",
        )
        changed = [
            field for field in identity_fields
            if existing.get(field) != experiment.get(field)
        ]
        if changed:
            raise ValueError(
                "experiment directory configuration mismatch: "
                + ", ".join(changed)
                + "; use a new directory or --overwrite"
            )
        experiment = {**existing, "dry_run": dry_run}
        atomic_json(experiment_path, experiment)
    else:
        atomic_json(experiment_path, experiment)
    if dry_run:
        return {**experiment, "runs": [], "failures": 0}
    if not api_key:
        raise ValueError("AIKP_EVAL_API_KEY is required unless --dry-run is used")

    runs = [
        execute_job(job, source_root, model, base_url, api_key,
                    parser_factory, overwrite)
        for job in jobs
    ]
    benchmarks = score_completed_repeats(
        output_dir, gold_root,
        source_root, conditions, repeats,
        [str(row["id"]) for row in documents],
    )
    result = {
        **experiment,
        "runs": runs,
        "benchmarks": benchmarks,
        "failures": sum(row.get("status") != "success" for row in runs),
    }
    atomic_json(output_dir / "matrix_report.json", result)
    return result


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--manifest", type=Path,
                     default=ROOT / "evals" / "corpus_manifest.json")
    cli.add_argument("--source-root", type=Path,
                     default=Path("/home/lonpyer/aikp_eval_data"))
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--condition", action="append", choices=sorted(PARSER_ABLATIONS))
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--module-id", action="append")
    cli.add_argument("--annotation", default="gold-v1")
    cli.add_argument("--gold-dir", type=Path, default=ROOT / "evals" / "gold")
    cli.add_argument("--model", default=os.environ.get("AIKP_EVAL_MODEL", "deepseek-chat"))
    cli.add_argument("--base-url", default=os.environ.get(
        "AIKP_EVAL_BASE_URL", "https://api.deepseek.com/v1"))
    cli.add_argument("--api-key-env", default="AIKP_EVAL_API_KEY")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--overwrite", action="store_true")
    cli.add_argument("--job-order-seed", type=int, default=20260818)
    args = cli.parse_args()
    if args.repeats < 1:
        cli.error("--repeats must be positive")
    try:
        report = run_matrix(
            args.manifest, args.source_root, args.output_dir,
            args.condition or list(DEFAULT_CONDITIONS), args.repeats,
            args.model, args.base_url, os.environ.get(args.api_key_env, ""),
            set(args.module_id or []) or None, args.annotation,
            args.gold_dir, args.dry_run, args.overwrite,
            job_order_seed=args.job_order_seed,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "job_count": report["job_count"],
        "failures": report["failures"],
        "dry_run": report["dry_run"],
    }, ensure_ascii=False, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

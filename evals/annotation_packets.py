#!/usr/bin/env python3
"""Prepare and collect reproducible blind story-graph annotation packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from validate_gold import validate_gold


ROOT = Path(__file__).resolve().parents[1]
PACKET_VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n").encode("utf-8")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(payload))
    temporary.replace(path)


def task_id(module_id: str, annotator_id: str, seed: int) -> str:
    digest = hashlib.sha256(
        f"aikp-annotation-v{PACKET_VERSION}|{seed}|{annotator_id}|{module_id}"
        .encode("utf-8")
    ).hexdigest()[:12]
    return f"task-{digest}"


def selected_documents(manifest: dict, annotation: str,
                       module_ids: set[str] | None = None) -> list[dict]:
    rows = [row for row in manifest.get("documents", []) if isinstance(row, dict)]
    if module_ids:
        rows = [row for row in rows if str(row.get("id")) in module_ids]
        missing = module_ids - {str(row.get("id")) for row in rows}
        if missing:
            raise ValueError("unknown module ids: " + ", ".join(sorted(missing)))
    elif annotation:
        rows = [row for row in rows if row.get("annotation") == annotation]
    return sorted(rows, key=lambda row: str(row.get("id", "")))


def annotation_template(task: dict) -> dict:
    return {
        "benchmark_version": "1.0",
        "module_id": task["task_id"],
        "title": "",
        "source": {
            "publisher": task.get("publisher", ""),
            "download_url": task.get("source_url", ""),
            "local_filename": task["source_file"],
            "sha256": task["source_sha256"],
        },
        "scenarios": [],
        "nodes": [],
        "edges": [],
        "entities": [],
        "forbidden_navigable_scenes": [],
        "annotator_notes": [],
    }


def packet_fingerprint(payload: dict) -> str:
    stable = dict(payload)
    stable.pop("packet_fingerprint", None)
    return hashlib.sha256(canonical_json(stable)).hexdigest()


def prepare_packets(
    manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    annotator_ids: list[str],
    annotation: str = "gold-v1",
    module_ids: set[str] | None = None,
    seed: int = 20260818,
    copy_sources: bool = False,
    guide_path: Path = ROOT / "evals" / "ANNOTATION_GUIDE.md",
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    if len(set(annotator_ids)) != len(annotator_ids) or not annotator_ids:
        raise ValueError("annotator ids must be non-empty and unique")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = selected_documents(manifest, annotation, module_ids)
    if not documents:
        raise ValueError("no corpus documents selected")
    if not guide_path.exists():
        raise FileNotFoundError(guide_path)

    source_rows = []
    for document in documents:
        module_id = str(document.get("id", ""))
        relative = str(document.get("local_filename", ""))
        expected = str(document.get("sha256", ""))
        path = source_root / relative
        if not module_id or not relative or not expected:
            raise ValueError(f"incomplete manifest row: {module_id or '<unknown>'}")
        if not path.exists() or sha256(path) != expected:
            raise ValueError(f"source hash mismatch or missing file: {relative}")
        source_rows.append((document, path))

    coordinator = {
        "packet_version": PACKET_VERSION,
        "seed": seed,
        "annotation_filter": annotation,
        "manifest_sha256": sha256(manifest_path),
        "guide_sha256": sha256(guide_path),
        "source_root": str(source_root.resolve()),
        "copy_sources": copy_sources,
        "annotators": {},
    }
    for annotator_id in annotator_ids:
        annotator_dir = output_dir / annotator_id
        annotation_dir = annotator_dir / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(guide_path, annotator_dir / "ANNOTATION_GUIDE.md")
        tasks = []
        coordinator_tasks = []
        for document, source_path in source_rows:
            module_id = str(document["id"])
            anonymous_id = task_id(module_id, annotator_id, seed)
            source_file = (
                f"sources/{anonymous_id}{source_path.suffix or '.txt'}"
                if copy_sources else str(source_path.resolve())
            )
            task = {
                "task_id": anonymous_id,
                "ruleset": document.get("ruleset", ""),
                "split": document.get("split", ""),
                "source_file": source_file,
                "source_sha256": str(document["sha256"]),
                "source_url": document.get("source_url", ""),
                "publisher": document.get("publisher", ""),
            }
            tasks.append(task)
            coordinator_tasks.append({
                **task,
                "module_id": module_id,
                "original_local_filename": str(document["local_filename"]),
            })
            if copy_sources:
                destination = annotator_dir / source_file
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
            atomic_json(annotation_dir / f"{anonymous_id}.json",
                        annotation_template(task))

        assignment = {
            "packet_version": PACKET_VERSION,
            "annotator_id": annotator_id,
            "manifest_sha256": coordinator["manifest_sha256"],
            "guide_sha256": coordinator["guide_sha256"],
            "task_count": len(tasks),
            "tasks": tasks,
        }
        assignment["assignment_fingerprint"] = packet_fingerprint(assignment)
        atomic_json(annotator_dir / "assignment.json", assignment)
        (annotator_dir / "README.md").write_text(
            "# Blind annotation packet\n\n"
            "Read `ANNOTATION_GUIDE.md` before opening a source. Complete every JSON "
            "file under `annotations/` without consulting AIKP parser output, existing "
            "gold annotations, or another annotator. Do not rename task IDs.\n",
            encoding="utf-8",
        )
        coordinator["annotators"][annotator_id] = {
            "assignment_fingerprint": assignment["assignment_fingerprint"],
            "tasks": coordinator_tasks,
        }

    coordinator["packet_fingerprint"] = packet_fingerprint(coordinator)
    atomic_json(output_dir / "coordinator_packet.json", coordinator)
    return coordinator


def collect_submission(packet_path: Path, annotator_id: str,
                       annotation_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("packet_fingerprint") != packet_fingerprint(packet):
        raise ValueError("coordinator packet fingerprint mismatch")
    annotator = packet.get("annotators", {}).get(annotator_id)
    if not isinstance(annotator, dict):
        raise ValueError(f"unknown annotator id: {annotator_id}")
    source_root = Path(str(packet.get("source_root", "")))
    tasks = annotator.get("tasks", [])
    annotations_out = output_dir / "annotations"
    records = []
    errors = []
    for task in tasks:
        anonymous_id = str(task["task_id"])
        module_id = str(task["module_id"])
        input_path = annotation_dir / f"{anonymous_id}.json"
        if not input_path.exists():
            errors.append(f"missing annotation: {anonymous_id}")
            continue
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{anonymous_id}: invalid JSON: {exc}")
            continue
        validation = validate_gold(payload, input_path.name)
        if not validation["valid"]:
            errors.extend(f"{anonymous_id}: {item}" for item in validation["errors"])
            continue
        original_source = source_root / str(task["original_local_filename"])
        if (not original_source.exists()
                or sha256(original_source) != str(task["source_sha256"])):
            errors.append(f"{anonymous_id}: original source hash mismatch")
            continue
        result = dict(payload)
        result["module_id"] = module_id
        result["source"] = {
            "publisher": task.get("publisher", ""),
            "download_url": task.get("source_url", ""),
            "local_filename": task["original_local_filename"],
            "sha256": task["source_sha256"],
        }
        result["annotation_provenance"] = {
            "packet_fingerprint": packet["packet_fingerprint"],
            "assignment_fingerprint": annotator["assignment_fingerprint"],
            "annotator_id": annotator_id,
            "blind_task_id": anonymous_id,
        }
        output_path = annotations_out / f"{module_id}.json"
        atomic_json(output_path, result)
        records.append({
            "module_id": module_id,
            "blind_task_id": anonymous_id,
            "annotation_sha256": sha256(output_path),
            "source_sha256": task["source_sha256"],
        })

    if errors:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise ValueError("submission rejected:\n" + "\n".join(errors))
    manifest = {
        "packet_version": PACKET_VERSION,
        "packet_fingerprint": packet["packet_fingerprint"],
        "assignment_fingerprint": annotator["assignment_fingerprint"],
        "annotator_id": annotator_id,
        "annotation_count": len(records),
        "annotations": records,
    }
    manifest["submission_fingerprint"] = packet_fingerprint(manifest)
    atomic_json(output_dir / "submission_manifest.json", manifest)
    return manifest


def main() -> int:
    cli = argparse.ArgumentParser()
    subparsers = cli.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path,
                         default=ROOT / "evals" / "corpus_manifest.json")
    prepare.add_argument("--source-root", type=Path,
                         default=Path("/home/lonpyer/aikp_eval_data"))
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--annotator", action="append", required=True)
    prepare.add_argument("--annotation", default="gold-v1")
    prepare.add_argument("--module-id", action="append")
    prepare.add_argument("--seed", type=int, default=20260818)
    prepare.add_argument("--copy-sources", action="store_true")
    prepare.add_argument("--guide", type=Path,
                         default=ROOT / "evals" / "ANNOTATION_GUIDE.md")

    collect = subparsers.add_parser("collect")
    collect.add_argument("--packet", type=Path, required=True)
    collect.add_argument("--annotator", required=True)
    collect.add_argument("--annotation-dir", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_packets(
                args.manifest, args.source_root, args.output_dir, args.annotator,
                args.annotation, set(args.module_id or []) or None, args.seed,
                args.copy_sources, args.guide,
            )
            summary = {
                "packet_fingerprint": result["packet_fingerprint"],
                "annotators": sorted(result["annotators"]),
                "task_count_per_annotator": len(next(iter(
                    result["annotators"].values()))["tasks"]),
                "output_dir": str(args.output_dir),
            }
        else:
            result = collect_submission(
                args.packet, args.annotator, args.annotation_dir, args.output_dir)
            summary = {
                "submission_fingerprint": result["submission_fingerprint"],
                "annotator": result["annotator_id"],
                "annotation_count": result["annotation_count"],
                "output_dir": str(args.output_dir),
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

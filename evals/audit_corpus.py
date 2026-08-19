#!/usr/bin/env python3
"""Verify external evaluation sources and offline parser preconditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config import LONG_DOCUMENT_WINDOW_CHARS
from parser import _catalog_windows, _detect_rule_profile, _split_text_segments


def audit(manifest: dict, source_root: Path) -> dict:
    rows = []
    for declared in manifest.get("documents", []):
        path = source_root / declared["local_filename"]
        row = {"id": declared["id"], "path": str(path), "exists": path.exists(),
               "expected_ruleset": declared["ruleset"], "split": declared["split"],
               "annotation": declared["annotation"]}
        if path.exists():
            data = path.read_bytes()
            text = data.decode("utf-8", errors="replace")
            segments = _split_text_segments(text)
            windows = _catalog_windows(segments, LONG_DOCUMENT_WINDOW_CHARS)
            source_chars = sum(len(str(segment.get("text", ""))) for segment in segments)
            mapped_chars = sum(len(str(segment.get("text", "")))
                               for window in windows for segment in window)
            profile = _detect_rule_profile(text)
            row.update({
                "sha256_matches": hashlib.sha256(data).hexdigest() == declared["sha256"],
                "chars": len(text), "segments": len(segments), "windows": len(windows),
                "window_coverage": source_chars == mapped_chars,
                "detected_ruleset": profile["ruleset"],
                "ruleset_matches": profile["ruleset"] == declared["ruleset"],
                "dice_system": profile["dice_system"],
            })
        rows.append(row)
    complete = all(row.get("exists") and row.get("sha256_matches")
                   and row.get("window_coverage") and row.get("ruleset_matches")
                   for row in rows)
    return {"complete": complete, "document_count": len(rows), "documents": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("corpus_manifest.json"))
    parser.add_argument("--source-root", type=Path,
                        default=Path("/home/lonpyer/aikp_eval_data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = audit(manifest, args.source_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.strict and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

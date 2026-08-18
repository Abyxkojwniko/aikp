#!/usr/bin/env python3
"""Run every no-key manual playtest fixture in a directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--fixtures-dir", required=True, type=Path)
    cli.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/lonpyer/aikp_eval_runs"),
    )
    args = cli.parse_args()

    runner = Path(__file__).with_name("run_manual_playtest.py")
    cases = sorted(args.fixtures_dir.glob("*_case.json"))
    if not cases:
        print(f"No *_case.json fixtures found in {args.fixtures_dir}")
        return 2

    failures = []
    passed_cases = 0
    module_coverage: dict[str, set[str]] = {}
    for case_path in cases:
        stem = case_path.name.removesuffix("_case.json")
        case = json.loads(case_path.read_text(encoding="utf-8"))
        world_stem = case.get("world_fixture", stem)
        response_stem = case.get("responses_fixture", stem)
        world_path = args.fixtures_dir / f"{world_stem}_world.json"
        responses_path = args.fixtures_dir / f"{response_stem}_responses.json"
        missing = [path for path in (world_path, responses_path) if not path.exists()]
        if missing:
            failures.append(f"{stem}: missing {', '.join(map(str, missing))}")
            continue

        print(f"\n{'=' * 72}\nRunning {stem}\n{'=' * 72}", flush=True)
        result = subprocess.run(
            [
                sys.executable, str(runner),
                "--world", str(world_path),
                "--case", str(case_path),
                "--responses", str(responses_path),
                "--output-dir", str(args.output_dir),
            ],
            check=False,
        )
        if result.returncode:
            failures.append(f"{stem}: exit {result.returncode}")
        else:
            passed_cases += 1
            module = case.get("module", "")
            if module:
                covered = module_coverage.setdefault(module, set())
                for turn in case.get("turns", []):
                    covered.update(str(point) for point in turn.get("covers", []))

    for coverage_path in sorted(args.fixtures_dir.glob("*_coverage.json")):
        manifest = json.loads(coverage_path.read_text(encoding="utf-8"))
        module = manifest["module"]
        required = {str(point) for point in manifest.get("required", [])}
        covered = module_coverage.get(module, set())
        missing = sorted(required - covered)
        print(
            f"Coverage {module}: {len(required) - len(missing)}/"
            f"{len(required)} required points")
        if missing:
            failures.append(
                f"{module}: missing coverage {', '.join(missing)}")

    print(f"\nManual suite: {passed_cases}/{len(cases)} cases passed")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

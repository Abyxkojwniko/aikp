#!/usr/bin/env python3
"""Parse a real module and run a repeatable adversarial player transcript."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from config import DEEPSEEK_API_KEY
from parser import ModuleParser, save_world_book
from state_manager import load_session


def _snapshot(session: dict) -> dict:
    return {
        "turn": session.get("current_turn", 0),
        "scene": session.get("player_state", {}).get("current_scene", ""),
        "pending_check": session.get("pending_check"),
        "discovered_clues": session.get("discovered_clues", []),
        "flags": session.get("flags", []),
        "companions": session.get("companions", []),
    }


def _load_or_parse_world(case: dict, data_dir: Path, reuse: bool) -> str:
    world_id = case["world_id"]
    world_path = ROOT / "models" / world_id / f"{world_id}.json"
    if reuse and world_path.exists():
        return world_id

    module_path = data_dir / case["module_file"]
    text = module_path.read_text(encoding="utf-8", errors="replace")
    parser = ModuleParser(api_key=DEEPSEEK_API_KEY)
    world = parser.parse(text)
    world["name"] = world_id
    world["eval_source_file"] = module_path.name
    save_world_book(world_id, world)
    return world_id


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--case", required=True, type=Path)
    cli.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/lonpyer/aikp_eval_data"),
    )
    cli.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/lonpyer/aikp_eval_runs"),
    )
    cli.add_argument("--reuse-world", action="store_true")
    args = cli.parse_args()

    if not DEEPSEEK_API_KEY:
        print("DEEPSEEK_API_KEY is missing. Configure it in the project .env.", file=sys.stderr)
        return 2

    case = json.loads(args.case.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"{case['name']}-{stamp}"
    trace_dir = run_dir / "traces"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AIKP_TRACE_DIR"] = str(trace_dir)

    world_id = _load_or_parse_world(case, args.data_dir, args.reuse_world)

    from engine import invalidate_world_cache, run_gm_turn

    invalidate_world_cache(world_id)
    chat_id = f"eval-{case['name']}-{stamp}"
    transcript = {
        "case": case["name"],
        "world_id": world_id,
        "chat_id": chat_id,
        "turns": [],
    }

    for index, turn in enumerate(case["turns"], start=1):
        player_input = turn["input"]
        print(f"\n[Player {index}] {player_input}", flush=True)
        response = run_gm_turn(
            messages=[{"role": "user", "content": player_input}],
            model=world_id,
            chat_id=chat_id,
            api_key=DEEPSEEK_API_KEY,
            stream=False,
        )
        print(f"[KP {index}] {response}", flush=True)
        session = load_session(chat_id)
        transcript["turns"].append({
            "index": index,
            "player": player_input,
            "kp": response,
            "intent": turn.get("intent", ""),
            "state": _snapshot(session),
        })
        (run_dir / "transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\nSaved playtest to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

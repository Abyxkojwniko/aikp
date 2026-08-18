#!/usr/bin/env python3
"""Run a no-key playtest with recorded prompts and manually authored replies."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manual_provider import ReplayNarrationProvider
from config import SESSIONS_DIR, WORLD_BOOK_DIR
from parser import save_world_book
from state_manager import load_session


def _snapshot(session: dict) -> dict:
    return {
        "turn": session.get("current_turn", 0),
        "scene": session.get("player_state", {}).get("current_scene", ""),
        "selected_npc_id": session.get("selected_npc_id"),
        "pending_check": session.get("pending_check"),
        "discovered_clues": session.get("discovered_clues", []),
        "flags": session.get("flags", []),
        "entity_states": session.get("entity_states", {}),
        "entity_facts": session.get("entity_facts", {}),
        "inventory_entity_ids": session.get("inventory_entity_ids", []),
        "world_event_types": [
            event.get("type", "") for event in session.get("world_events", [])],
        "clocks": session.get("clocks", {}),
    }


def _check_expectations(expect: dict, response: str, state: dict) -> list[str]:
    failures = []
    for key in ("turn", "scene", "selected_npc_id"):
        if key in expect and state.get(key) != expect[key]:
            failures.append(
                f"{key}: expected {expect[key]!r}, got {state.get(key)!r}")
    response_lower = response.lower()
    for needle in expect.get("response_contains", []):
        if str(needle).lower() not in response_lower:
            failures.append(f"response missing {needle!r}")
    for needle in expect.get("response_not_contains", []):
        if str(needle).lower() in response_lower:
            failures.append(f"response unexpectedly contains {needle!r}")
    for entity_id, expected_state in expect.get("entity_states", {}).items():
        actual_state = state.get("entity_states", {}).get(entity_id)
        if actual_state != expected_state:
            failures.append(
                f"entity_states.{entity_id}: expected {expected_state!r}, "
                f"got {actual_state!r}")
    for clock_id, expected_value in expect.get("clocks", {}).items():
        actual_value = state.get("clocks", {}).get(clock_id)
        if actual_value != expected_value:
            failures.append(
                f"clocks.{clock_id}: expected {expected_value!r}, "
                f"got {actual_value!r}")
    for entity_id in expect.get("inventory_contains", []):
        if entity_id not in state.get("inventory_entity_ids", []):
            failures.append(f"inventory missing {entity_id!r}")
    for entity_id, expected_location in expect.get("object_locations", {}).items():
        actual = state.get("entity_facts", {}).get(
            entity_id, {}).get("location", {})
        if actual != expected_location:
            failures.append(
                f"object_locations.{entity_id}: expected "
                f"{expected_location!r}, got {actual!r}")
    event_types = state.get("world_event_types", [])
    for event_type in expect.get("world_event_types_contains", []):
        if event_type not in event_types:
            failures.append(f"world events missing {event_type!r}")
    return failures


_FORCED_COC_ROLLS = {
    "critical_success": {"d100": 1, "success": True, "verdict": "critical_success"},
    "extreme_success": {"d100": 5, "success": True, "verdict": "extreme_success"},
    "hard_success": {"d100": 20, "success": True, "verdict": "hard_success"},
    "success": {"d100": 40, "success": True, "verdict": "success"},
    "failure": {"d100": 80, "success": False, "verdict": "failure"},
    "fumble": {"d100": 100, "success": False, "verdict": "fumble"},
}


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--world", required=True, type=Path)
    cli.add_argument("--case", required=True, type=Path)
    cli.add_argument("--responses", required=True, type=Path)
    cli.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/lonpyer/aikp_eval_runs"),
    )
    cli.add_argument("--max-turns", type=int, default=0)
    cli.add_argument("--keep-runtime-state", action="store_true")
    args = cli.parse_args()

    world = json.loads(args.world.read_text(encoding="utf-8"))
    world_id = world["name"]
    save_world_book(world_id, world)

    case = json.loads(args.case.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"manual-{case['name']}-{stamp}"
    requests_dir = run_dir / "requests"
    run_dir.mkdir(parents=True, exist_ok=True)
    provider = ReplayNarrationProvider(args.responses, requests_dir)

    import engine
    import npc_context
    from reference_resolver import select_interaction_target
    from scene_system import select_scene_target
    from world_state import select_object_target

    # Trait extraction is a separate LLM enrichment call, not narration. The
    # manual run deliberately disables it so no hidden API request can occur.
    engine._extract_arrival_traits = lambda *args, **kwargs: None
    npc_context.compress_story = (
        lambda session, **kwargs: session.get("_story_summary", ""))
    engine.get_or_compress_conversation_summary = (
        lambda session, **kwargs: session.get("_cached_summary", ""))
    engine.invalidate_world_cache(world_id)
    chat_id = f"manual-{case['name']}-{stamp}"
    transcript = {
        "case": case["name"],
        "module": case.get("module", ""),
        "world_id": world_id,
        "chat_id": chat_id,
        "responses_file": str(args.responses),
        "turns": [],
        "failures": [],
        "coverage": [],
    }
    covered = set()

    turns = case["turns"][:args.max_turns or None]
    with engine.narration_provider(provider):
        for index, turn in enumerate(turns, start=1):
            state_changes = turn.get("set_entity_states", {})
            if state_changes:
                session = engine.get_session(chat_id, world_id)
                session.setdefault("entity_states", {}).update(state_changes)
                engine.save_session(session)

            npc_id = turn.get("select_npc")
            if npc_id:
                session = engine.get_session(chat_id, world_id)
                scene_index, entity_index = engine.get_indices(world_id)
                select_interaction_target(
                    session, npc_id, world, scene_index, entity_index)
                engine.save_session(session)

            object_id = turn.get("select_object")
            if object_id:
                session = engine.get_session(chat_id, world_id)
                scene_index, entity_index = engine.get_indices(world_id)
                select_object_target(
                    session, object_id, world, scene_index, entity_index)
                engine.save_session(session)

            scene_id = turn.get("select_scene")
            if scene_id:
                session = engine.get_session(chat_id, world_id)
                select_scene_target(session, scene_id, world)
                engine.save_session(session)

            player_input = turn["input"]
            response = engine.run_gm_turn(
                messages=[{"role": "user", "content": player_input}],
                model=world_id,
                chat_id=chat_id,
                api_key="manual-provider-no-api-key",
                stream=False,
            )
            roll_result = None
            forced_verdict = turn.get("roll_verdict")
            if forced_verdict:
                forced = _FORCED_COC_ROLLS.get(forced_verdict)
                if not forced:
                    raise ValueError(f"unsupported forced verdict: {forced_verdict}")
                import server
                with patch("dice.coc_skill_check", return_value=dict(forced)):
                    roll_result = server.roll_check(chat_id)
            session = load_session(chat_id)
            snapshot = _snapshot(session)
            failures = _check_expectations(
                turn.get("expect", {}), response, snapshot)
            transcript["failures"].extend(
                f"turn {index}: {failure}" for failure in failures)
            transcript["turns"].append({
                "index": index,
                "player": player_input,
                "selected_npc": npc_id,
                "selected_object": object_id,
                "selected_scene": scene_id,
                "kp": response,
                "roll": roll_result,
                "intent": turn.get("intent", ""),
                "state": snapshot,
                "failures": failures,
                "covers": turn.get("covers", []),
            })
            covered.update(str(point) for point in turn.get("covers", []))
            print(f"\n[Player {index}] {player_input}")
            print(f"[KP {index}] {response}")

    if provider.index != len(provider.responses):
        transcript["failures"].append(
            f"manual responses consumed: expected {len(provider.responses)}, "
            f"got {provider.index}")
    transcript["coverage"] = sorted(covered)
    required_coverage = {
        str(point) for point in case.get("required_coverage", [])}
    missing_coverage = sorted(required_coverage - covered)
    if missing_coverage:
        transcript["failures"].append(
            "missing coverage: " + ", ".join(missing_coverage))
    (run_dir / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved manual playtest to {run_dir}")
    if not args.keep_runtime_state:
        shutil.rmtree(Path(WORLD_BOOK_DIR) / world_id, ignore_errors=True)
        (Path(SESSIONS_DIR) / f"{chat_id}.json").unlink(missing_ok=True)
        engine.invalidate_world_cache(world_id)
    if transcript["failures"]:
        for failure in transcript["failures"]:
            print(f"FAIL: {failure}")
        return 1
    print("All manual playtest expectations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

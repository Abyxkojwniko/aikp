#!/usr/bin/env python3
"""Generate a synthetic end-to-end identity and narrative-scope stress case."""

from __future__ import annotations

from generate_exhaustive_fixtures import movement_step, write_json, write_route


INVENTORY = {"kind": "inventory", "id": "player"}
ARCHIVE = {"kind": "scene", "id": "archive"}


def carried(expect: dict | None = None) -> dict:
    result = dict(expect or {})
    result["inventory_contains"] = ["brass_key"]
    result["object_locations"] = {"brass_key": INVENTORY}
    return result


def generate() -> None:
    world = {
        "name": "eval_identity_scope_stress",
        "rule_system": "coc",
        "starting_scene": "study",
        "opening": "You enter a study containing a bookshelf and a wooden door.",
        "scenes": {
            "study": {
                "name": "Study",
                "desc": "A bookshelf, a blue drawer, and a wooden door occupy the study.",
                "source_text": "The study opens into the hall through its wooden door.",
                "exits": {"enter the hall": "hall"},
            },
            "hall": {
                "name": "Hall",
                "desc": "A second wooden door stands at the far end of the hall.",
                "source_text": "The hall connects the study and the archive.",
                "exits": {"return to the study": "study", "enter the archive": "archive"},
            },
            "archive": {
                "name": "Archive",
                "desc": "The archivist works beside locked document cabinets.",
                "source_text": "The archive connects only to the hall.",
                "exits": {"return to the hall": "hall"},
            },
        },
        "entities": {
            "study_door": {
                "type": "door", "name": "wooden door", "scene": "study",
                "home_scene": "study", "initial_state": "present",
                "portable": False, "description": "A wooden door with a brass handle.",
            },
            "hall_door": {
                "type": "door", "name": "wooden door", "scene": "hall",
                "home_scene": "hall", "initial_state": "present",
                "portable": False, "description": "A wooden door with iron bands.",
            },
            "bookshelf": {
                "type": "object", "name": "bookshelf", "scene": "study",
                "initial_state": "present", "portable": False,
            },
            "storybook": {
                "type": "document", "name": "storybook", "scene": "study",
                "initial_state": "present", "portable": True,
                "description": "A story about an ink castle and an obsidian throne.",
            },
            "brass_key": {
                "type": "item", "name": "brass key", "scene": "study",
                "initial_state": "hidden", "portable": True,
                "description": "The brass key begins inside the blue drawer.",
            },
            "archivist": {
                "type": "npc", "name": "Mara Venn", "scene": "archive",
                "initial_state": "present", "public_label": "archivist",
            },
        },
        "narrative_scopes": [
            {"id": "physical", "kind": "physical", "navigable": True},
            {"id": "storybook", "kind": "document", "parent_scope": "physical",
             "navigable": False},
        ],
        "embedded_settings": [{
            "scope": {"id": "storybook", "kind": "document", "navigable": False},
            "scenes": [{
                "id": "ink_castle", "name": "Ink Castle",
                "aliases": ["castle in the storybook"],
                "scope_id": "storybook", "navigable": False,
            }],
            "entities": [{
                "id": "obsidian_throne", "name": "obsidian throne",
                "scene": "ink_castle", "scope_id": "storybook",
            }],
        }],
    }
    write_json("identity_scope_stress_world.json", world)

    steps = [
        {
            "input": "开始游戏",
            "response": "You stand in the study beside a bookshelf and its wooden door.",
            "scene": "study", "contains": ["study"], "covers": ["scope_opening"],
        },
        {
            "input": "I inspect the wooden door in this room.",
            "response": "This wooden door has a brass handle and leads toward the hall.",
            "select_object": "study_door", "contains": ["brass handle"],
            "covers": ["study_door_local_identity"],
        },
        {
            "input": "I enter the Ink Castle from the storybook.",
            "response": "The embedded setting is not a physical destination.",
            "code_response": True, "contains": ["不能直接越过"],
            "covers": ["embedded_scene_move_block"],
            "expect": {"response_not_contains": ["obsidian throne"]},
        },
        {
            "input": "I take the brass key before searching anything.",
            "response": "The hidden key cannot be used yet.",
            "code_response": True, "contains": ["没有发现"],
            "covers": ["hidden_key_block"],
            "expect": {"response_not_contains": ["blue drawer"]},
        },
        {
            "input": "I search the bookshelf and the nearby drawer.",
            "response": "Searching the study reveals a brass key inside the blue drawer.",
            "select_object": "bookshelf", "contains": ["brass key"],
            "set_entity_states": {"brass_key": "revealed"},
            "covers": ["key_discovered"],
        },
        {
            "input": "I pick up the brass key.",
            "response": "You take the brass key from the drawer.",
            "select_object": "brass_key", "contains": ["take"],
            "covers": ["key_picked_up"],
            "expect": {
                "inventory_contains": ["brass_key"],
                "object_locations": {"brass_key": INVENTORY},
                "world_event_types_contains": ["item_picked_up"],
            },
        },
        {
            **movement_step(
                "I go through the door into the hall.",
                "You carry the key into the hall.", "hall",
                ["key_carried_to_hall"], ["hall"],
            ),
            "expect": carried(),
        },
        {
            "input": "I inspect the wooden door here.",
            "response": "The hall's wooden door is reinforced with iron bands.",
            "select_object": "hall_door", "contains": ["iron bands"],
            "covers": ["hall_door_local_identity"], "expect": carried(),
        },
        {
            **movement_step(
                "I enter the archive.", "You enter the archive with the key.",
                "archive", ["key_carried_to_archive"], ["archive"],
            ),
            "expect": carried(),
        },
        {
            "input": "I ask the archivist about the cabinets.",
            "response": "The archivist says the cabinets hold municipal records.",
            "select_npc": "archivist", "contains": ["municipal records"],
            "covers": ["archivist_dialogue"], "expect": carried(),
        },
        {
            "input": "I keep asking the dead archivist for the combination.",
            "response": "A dead character cannot answer.",
            "set_entity_states": {"archivist": "dead"},
            "code_response": True, "contains": ["没有可交谈"],
            "covers": ["dead_archivist_block"],
            "expect": {
                **carried(),
                "response_not_contains": ["archive combination"],
            },
        },
        {
            "input": "I drop the brass key here.",
            "response": "You leave the brass key on an archive table.",
            "select_object": "brass_key", "contains": ["archive table"],
            "covers": ["key_dropped_in_archive"],
            "expect": {
                "object_locations": {"brass_key": ARCHIVE},
                "world_event_types_contains": ["item_dropped"],
            },
        },
        {
            **movement_step(
                "I return to the hall.",
                "You return to the hall; the key remains in the archive.",
                "hall", ["backtrack_after_drop"], ["hall"],
            ),
            "expect": {"object_locations": {"brass_key": ARCHIVE}},
        },
        {
            **movement_step(
                "I continue back to the study.",
                "You return to the study without the key.",
                "study", ["return_old_map_without_item"], ["study"],
            ),
            "expect": {"object_locations": {"brass_key": ARCHIVE}},
        },
        {
            "input": "I use the brass key from here.",
            "response": "The remote object cannot be used.",
            "code_response": True, "contains": ["未能把这个动作对应"],
            "covers": ["remote_key_use_block"],
            "expect": {"object_locations": {"brass_key": ARCHIVE}},
        },
    ]
    write_route(
        "identity_scope_stress", "identity_scope_stress",
        "identity_scope_stress", steps,
    )
    write_json("identity_scope_stress_coverage.json", {
        "module": "identity_scope_stress",
        "required": [
            "scope_opening", "study_door_local_identity",
            "embedded_scene_move_block", "hidden_key_block", "key_discovered",
            "key_picked_up", "key_carried_to_hall", "hall_door_local_identity",
            "key_carried_to_archive", "archivist_dialogue",
            "dead_archivist_block", "key_dropped_in_archive",
            "backtrack_after_drop", "return_old_map_without_item",
            "remote_key_use_block",
        ],
    })


if __name__ == "__main__":
    generate()
    print("Generated identity and narrative-scope stress fixture.")

#!/usr/bin/env python3
"""Generate source-grounded destructive runtime cases for Battle for Nova Rush."""

from __future__ import annotations

from generate_exhaustive_fixtures import movement_step, write_json, write_route


INVENTORY = {"kind": "inventory", "id": "player"}


def scene(name: str, desc: str, exits: dict[str, str] | None = None) -> dict:
    return {"name": name, "desc": desc, "source_text": desc,
            "exits": exits or {}}


def npc(name: str, at: str, label: str, states: dict | None = None) -> dict:
    row = {
        "type": "npc", "name": name, "scene": at,
        "initial_state": "present", "public_label": label,
    }
    if states:
        row["states"] = states
    return row


def stateful_object(name: str, at: str, initial: str, triggers: list[str],
                    target: str, narration: str) -> dict:
    return {
        "type": "object", "name": name, "scene": at,
        "initial_state": initial, "portable": False,
        "states": {
            initial: {
                "triggers": triggers,
                "on_trigger": {"to_state": target, "narration": narration},
            },
            target: {},
        },
    }


def build_world() -> dict:
    return {
        "name": "eval_starfinder_battle_for_nova_rush",
        "rule_system": "starfinder",
        "ruleset": "starfinder",
        "dice_system": "d20",
        "automatic_check_adapter": True,
        "starting_scene": "brig",
        "opening": "The captured crew begins inside the Nova Rush brig.",
        "scenes": {
            "brig": scene(
                "A1. Brig", "A force barrier encloses the prisoners and their gear.",
                {"leave the brig": "rec_room"}),
            "rec_room": scene(
                "A2. Rec Room", "Pirates defend the lounge during the attack.",
                {"enter the lower cargo hold": "lower_cargo",
                 "enter the medbay": "medbay",
                 "crawl into munitions": "munitions"}),
            "munitions": scene(
                "A6. Munitions", "Brinn waits among ammunition coils.",
                {"return to the rec room": "rec_room"}),
            "medbay": scene(
                "A7. Medbay", "The unused medical bay remains stocked.",
                {"return to the rec room": "rec_room"}),
            "lower_cargo": scene(
                "A4. Lower Cargo Hold", "A catwalk crosses the central cargo hold.",
                {"enter reactor access": "reactor",
                 "enter the control room": "control_room",
                 "take the stairs up": "upper_cargo",
                 "return to the rec room": "rec_room"}),
            "control_room": scene(
                "A5. Control Room", "Cameras show the jammed starboard launcher.",
                {"return to lower cargo": "lower_cargo"}),
            "reactor": scene(
                "A8. Reactor Access", "The reactor sparks while it overloads.",
                {"return to lower cargo": "lower_cargo"}),
            "upper_cargo": scene(
                "B1. Upper Cargo Hold", "Polly faces a spiritual sinkwell.",
                {"take the starboard bridgeway": "bridgeway",
                 "enter the mess": "mess",
                 "enter the bedrooms": "bedrooms",
                 "enter the bridge": "bridge",
                 "take the stairs down": "lower_cargo"}),
            "bridgeway": scene(
                "B2. Starboard Bridgeway", "The jammed launcher is exposed to vacuum.",
                {"return to upper cargo": "upper_cargo"}),
            "mess": scene(
                "B3. Mess", "A Y-shaped table stands between the food bars.",
                {"return to upper cargo": "upper_cargo"}),
            "bedrooms": scene(
                "B4. Bedrooms", "Captain Firestorm's private room is locked.",
                {"return to upper cargo": "upper_cargo"}),
            "bridge": scene(
                "B5. Bridge", "Captain Firestorm and her crew defend the controls.",
                {"return to upper cargo": "upper_cargo"}),
        },
        "entities": {
            "brinn": npc("Brinn", "brig", "bitter android mechanic"),
            "concierge": npc("Captain Concierge", "rec_room", "shipboard VI"),
            "polly": npc("Polly", "upper_cargo", "snarky electrovore"),
            "firestorm": npc(
                "Captain Phaedra Firestorm", "bridge", "android pirate captain",
                {
                    "present": {
                        "triggers": ["defeat Captain Firestorm", "defeat Firestorm"],
                        "on_trigger": {
                            "to_state": "defeated",
                            "narration": "Captain Firestorm is defeated and loses the bridge.",
                        },
                    },
                    "defeated": {"interactable": False},
                    "fled": {"interactable": False},
                }),
            "bridge_pilot": npc("Bridge pilot", "bridge", "noncombatant pirate pilot"),
            "magboots": {
                "type": "item", "name": "commercial magboots", "scene": "brig",
                "initial_state": "present", "portable": True,
            },
            "smuggling_cache": {
                "type": "object", "name": "smuggling compartment", "scene": "brig",
                "initial_state": "hidden", "portable": False,
            },
            "reactor": stateful_object(
                "overloading reactor", "reactor", "overloading",
                ["repair the reactor", "fix the reactor"], "repaired",
                "The reactor stabilizes and stops overloading."),
            "launcher": stateful_object(
                "jammed missile launcher", "bridgeway", "jammed",
                ["repair the missile launcher", "unjam the launcher"], "repaired",
                "The missile launcher cycles back into service."),
            "quarters_door": stateful_object(
                "locked door", "bedrooms", "locked",
                ["unlock Firestorm's door", "open the quarters door", "open it"],
                "unlocked", "Firestorm's private door unlocks."),
            "bridge_airlock": stateful_object(
                "airlock door", "bridge", "closed",
                ["open the airlock door", "open it"], "open",
                "The bridge airlock opens."),
            "holoskin": {
                "type": "item", "name": "Vidia's commercial holoskin",
                "scene": "bedrooms", "initial_state": "present", "portable": True,
            },
            "arc_pistol": {
                "type": "item", "name": "tactical arc pistol",
                "scene": "bedrooms", "initial_state": "present", "portable": True,
            },
            "escape_pods": {
                "type": "object", "name": "deactivated escape pods",
                "scene": "brig", "initial_state": "deactivated", "portable": False,
            },
            "sinkwell": {
                "type": "object", "name": "spiritual sinkwell",
                "scene": "upper_cargo", "initial_state": "active", "portable": False,
            },
        },
        "narrative_scopes": [
            {"id": "physical", "kind": "physical", "navigable": True},
            {"id": "character_biographies", "kind": "backstory",
             "parent_scope": "physical", "navigable": False},
        ],
        "embedded_settings": [{
            "scope": {"id": "character_biographies", "kind": "backstory",
                      "navigable": False},
            "scenes": [
                {"id": "lorespire", "name": "Lorespire Complex",
                 "scope_id": "character_biographies", "navigable": False},
                {"id": "fullbright", "name": "Fullbright Market",
                 "scope_id": "character_biographies", "navigable": False},
                {"id": "vesk_quarter", "name": "Vesk Quarter",
                 "scope_id": "character_biographies", "navigable": False},
            ],
            "entities": [],
        }],
    }


def generate() -> None:
    write_json("starfinder_battle_for_nova_rush_world.json", build_world())

    repair_route = [
        {"input": "开始游戏", "response": "The crew wakes behind the brig force barrier.",
         "scene": "brig", "contains": ["brig"], "covers": ["brig_opening"]},
        {"input": "I convince Brinn to release us.",
         "response": "Brinn releases the prisoners and points out the concealed cache.",
         "select_npc": "brinn", "contains": ["releases"], "covers": ["brinn_escape"]},
        {"input": "I pick up the commercial magboots.",
         "response": "You secure the magboots with the recovered equipment.",
         "select_object": "magboots", "contains": ["magboots"],
         "covers": ["brig_equipment"],
         "expect": {"inventory_contains": ["magboots"],
                    "object_locations": {"magboots": INVENTORY}}},
        movement_step("We leave the brig for the rec room.",
                      "The crew enters the rec room as pirates rush in.",
                      "rec_room", ["rec_room"], ["pirates"]),
        {"input": "We defeat the four pirates.",
         "response": "The last pirate falls and the holoprojector activates.",
         "contains": ["holoprojector"], "covers": ["rec_room_battle"]},
        {"input": "I ask Captain Concierge how to save the ship.",
         "response": "Captain Concierge identifies the reactor, launcher, allies, and bridge.",
         "select_npc": "concierge", "contains": ["reactor"],
         "covers": ["concierge_plan"]},
        {"input": "I enter the Lorespire Complex from Chk Chk's memories.",
         "response": "A biography location is not physically reachable from this ship.",
         "code_response": True, "contains": ["不能直接越过"],
         "covers": ["biography_scope_block"],
         "expect": {"response_not_contains": ["Absalom Station skyline"]}},
        movement_step("We enter the lower cargo hold.", "The crew crosses the cargo hold.",
                      "lower_cargo", ["lower_cargo"], ["cargo"]),
        movement_step("We enter reactor access.", "The overloading reactor throws sparks.",
                      "reactor", ["reactor_access"], ["reactor"]),
        {"input": "I repair the reactor.",
         "response": "The reactor stabilizes and stops overloading.",
         "select_object": "reactor", "contains": ["stabilizes"],
         "covers": ["reactor_repaired"],
         "expect": {"entity_states": {"reactor": "repaired"}}},
        movement_step("We return to lower cargo.", "The crew leaves reactor access.",
                      "lower_cargo", [], ["leaves"]),
        movement_step("We take the stairs to upper cargo.",
                      "A snarky electrovore faces a spreading spiritual sinkwell.",
                      "upper_cargo", ["upper_cargo", "polly_encounter"],
                      ["snarky electrovore"]),
        {"input": "I disable the sinkwell and ask Polly to help.",
         "response": "The sinkwell collapses; Polly agrees to absorb the reactor's charge.",
         "select_npc": "polly", "contains": ["agrees"],
         "covers": ["polly_saved", "polly_aid"]},
        movement_step("We take the starboard bridgeway.",
                      "The crew steps onto the vacuum bridgeway in magboots.",
                      "bridgeway", ["bridgeway"], ["magboots"]),
        {"input": "I repair the missile launcher.",
         "response": "The missile launcher cycles back into service.",
         "select_object": "launcher", "contains": ["service"],
         "covers": ["launcher_repaired"],
         "expect": {"entity_states": {"launcher": "repaired"},
                    "inventory_contains": ["magboots"]}},
        movement_step("We return to upper cargo.", "The crew reenters the upper hold.",
                      "upper_cargo", [], ["upper hold"]),
        movement_step("We enter the bridge.",
                      "The android pirate captain defends the bridge.",
                      "bridge", ["bridge"], ["android pirate captain"]),
        {"input": "We defeat Captain Firestorm.",
         "response": "Captain Firestorm is defeated and loses the bridge.",
         "select_npc": "firestorm", "contains": ["defeated"],
         "covers": ["firestorm_defeated", "bridge_secured"],
         "expect": {"entity_states": {"firestorm": "defeated"}}},
        {"input": "I ask the defeated captain to help us escape.",
         "response": "The defeated captain is no longer available for conversation.",
         "select_npc": "firestorm", "code_response": True,
         "contains": ["无法交谈"], "covers": ["defeated_npc_block"]},
        {"input": "We pilot Nova Rush away while using the repaired systems.",
         "response": "With the reactor stable and launcher online, Nova Rush escapes intact.",
         "contains": ["escapes intact"], "covers": ["clean_escape"],
         "expect": {"entity_states": {"reactor": "repaired", "launcher": "repaired",
                                      "firestorm": "defeated"}}},
    ]
    write_route("starfinder_repair_and_escape", "starfinder_battle_for_nova_rush",
                "starfinder_battle_for_nova_rush", repair_route)

    destructive_route = [
        {"input": "开始游戏", "response": "The prisoners remain trapped as battle begins.",
         "scene": "brig", "contains": ["trapped"], "covers": ["brig_opening"]},
        {"input": "I take everything from the hidden smuggling compartment.",
         "response": "The compartment has not been discovered.",
         "select_object": "smuggling_cache", "code_response": True,
         "contains": ["没有发现"], "covers": ["hidden_cache_block"]},
        movement_step("The attack shorts the cell and we leave the brig.",
                      "A blast shorts the barrier and the crew reaches the rec room.",
                      "rec_room", ["attack_release"], ["shorts"]),
        {"input": "We teleport directly to Captain Firestorm's bridge.",
         "response": "The bridge cannot be reached directly from the rec room.",
         "code_response": True, "contains": ["不能直接越过"],
         "covers": ["bridge_jump_block"]},
        movement_step("We fight through the rec room and enter lower cargo.",
                      "The pirates are driven back and the crew enters lower cargo.",
                      "lower_cargo", ["rec_room_battle"], ["lower cargo"]),
        movement_step("We take the stairs to upper cargo.",
                      "Polly is caught beside the active sinkwell.",
                      "upper_cargo", ["polly_encounter"], ["sinkwell"]),
        {"input": "We flee from the sinkwell and abandon Polly.",
         "response": "The crew retreats; the sinkwell kills Polly.",
         "set_entity_states": {"polly": "dead"}, "contains": ["kills Polly"],
         "covers": ["polly_killed"]},
        {"input": "I ask Polly to guide us now.",
         "response": "Polly is dead and cannot answer.",
         "select_npc": "polly", "code_response": True,
         "contains": ["交谈"], "covers": ["dead_polly_block"]},
        {"input": "We travel to the Fullbright market from Chk Chk's biography.",
         "response": "The market is only part of a character biography.",
         "code_response": True, "contains": ["不能直接越过"],
         "covers": ["biography_scope_block"],
         "expect": {"response_not_contains": ["market vendors"]}},
        movement_step("We enter the bridge without making repairs.",
                      "The damaged ship rocks as the crew reaches the bridge.",
                      "bridge", ["bridge"], ["damaged ship"]),
        {"input": "Captain Firestorm flees to an escape pod.",
         "response": "Firestorm abandons the bridge and flees toward an escape pod.",
         "set_entity_states": {"firestorm": "fled"}, "contains": ["flees"],
         "covers": ["firestorm_flees", "bridge_secured"]},
        {"input": "I continue negotiating with Firestorm on the bridge.",
         "response": "Firestorm has fled and is no longer on the bridge.",
         "select_npc": "firestorm", "code_response": True,
         "contains": ["交谈"], "covers": ["fled_npc_block"]},
        {"input": "We escape without repairing anything.",
         "response": "Nova Rush escapes the Corpse Fleet but suffers severe damage.",
         "contains": ["severe damage"], "covers": ["damaged_escape"],
         "expect": {"entity_states": {"polly": "dead", "firestorm": "fled",
                                      "reactor": "overloading", "launcher": "jammed"}}},
    ]
    write_route("starfinder_destructive_failure", "starfinder_battle_for_nova_rush",
                "starfinder_battle_for_nova_rush", destructive_route)

    object_route = [
        {"input": "开始游戏", "response": "The crew has already broken out of the brig.",
         "scene": "brig", "contains": ["brig"], "covers": ["brig_opening"]},
        movement_step("We leave the brig.", "The crew enters the rec room.",
                      "rec_room", [], ["rec room"]),
        movement_step("We cross into lower cargo.", "The crew enters lower cargo.",
                      "lower_cargo", ["lower_cargo"], ["lower cargo"]),
        movement_step("We take the stairs up.", "The crew reaches upper cargo.",
                      "upper_cargo", ["upper_cargo"], ["upper cargo"]),
        movement_step("We enter Captain Firestorm's bedroom area.",
                      "The locked private door stands among the bedrooms.",
                      "bedrooms", ["firestorm_quarters"], ["locked"]),
        {"input": "I open it.", "response": "Firestorm's private door unlocks.",
         "select_object": "quarters_door", "contains": ["unlocks"],
         "covers": ["quarters_door_unlock"],
         "expect": {"entity_states": {"quarters_door": "unlocked",
                                      "bridge_airlock": "closed"}}},
        {"input": "I take Vidia's commercial holoskin.",
         "response": "You take Vidia's commercial holoskin from the private room.",
         "select_object": "holoskin", "contains": ["holoskin"],
         "covers": ["holoskin_loot"],
         "expect": {"inventory_contains": ["holoskin"],
                    "object_locations": {"holoskin": INVENTORY}}},
        {"input": "I take the tactical arc pistol.",
         "response": "You secure the tactical arc pistol.",
         "select_object": "arc_pistol", "contains": ["arc pistol"],
         "covers": ["arc_pistol_loot"],
         "expect": {"inventory_contains": ["holoskin", "arc_pistol"],
                    "object_locations": {"arc_pistol": INVENTORY}}},
        movement_step("We return to upper cargo.",
                      "The crew returns to the snarky electrovore.",
                      "upper_cargo", [], ["snarky electrovore"]),
        {"input": "I give the holoskin to Polly.",
         "response": "Polly accepts the holoskin and keeps it.",
         "select_npc": "polly", "select_object": "holoskin",
         "contains": ["accepts"], "covers": ["holoskin_transfer"],
         "expect": {"inventory_contains": ["arc_pistol"],
                    "object_locations": {"holoskin": {"kind": "entity", "id": "polly"}},
                    "world_event_types_contains": ["item_transferred"]}},
        movement_step("We enter the bridge.", "The crew enters the bridge.",
                      "bridge", ["bridge"], ["bridge"]),
        {"input": "I use Vidia's holoskin here.",
         "response": "The holoskin belongs to Polly and is not in your possession.",
         "code_response": True, "contains": ["未能把这个动作对应"],
         "covers": ["transferred_holoskin_reuse_block"],
         "expect": {"object_locations": {"holoskin": {"kind": "entity", "id": "polly"}},
                    "response_not_contains": ["bedrooms"]}},
        {"input": "I open it.", "response": "The bridge airlock opens.",
         "select_object": "bridge_airlock", "contains": ["airlock opens"],
         "covers": ["bridge_airlock_open"],
         "expect": {"entity_states": {"quarters_door": "unlocked",
                                      "bridge_airlock": "open"},
                    "inventory_contains": ["arc_pistol"]}},
    ]
    write_route("starfinder_object_continuity", "starfinder_battle_for_nova_rush",
                "starfinder_battle_for_nova_rush", object_route)

    required = sorted({
        point for route in (repair_route, destructive_route, object_route)
        for step in route for point in step.get("covers", [])
    })
    write_json("starfinder_battle_for_nova_rush_coverage.json", {
        "module": "starfinder_battle_for_nova_rush",
        "required": required,
    })


if __name__ == "__main__":
    generate()

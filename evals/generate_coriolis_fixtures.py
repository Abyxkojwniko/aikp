#!/usr/bin/env python3
"""Generate source-grounded runtime stress cases for The Sky Machine."""

from __future__ import annotations

from generate_exhaustive_fixtures import movement_step, write_json, write_route


INVENTORY = {"kind": "inventory", "id": "player"}


def scene(name: str, desc: str, exits: dict[str, str] | None = None) -> dict:
    return {"name": name, "desc": desc, "source_text": desc, "exits": exits or {}}


def npc(name: str, at: str, label: str) -> dict:
    return {
        "type": "npc", "name": name, "scene": at,
        "initial_state": "present", "public_label": label,
    }


def build_main_world() -> dict:
    return {
        "name": "eval_coriolis_sky_machine",
        "rule_system": "coriolis",
        "dice_system": "d6_pool",
        "automatic_check_adapter": False,
        "starting_scene": "landing",
        "opening": (
            "The Explorers land at Gilen's Point on Moubarra 4 for a rescue "
            "mission concerning five missing prospectors."
        ),
        "scenes": {
            "landing": scene("Moubarra 4 Landing Pad", "Drenk Zabo meets the shuttle.", {"meet the chief": "chief"}),
            "chief": scene("Station Chief's Office", "Lia Kalvanetes gives the rescue briefing.", {"visit the canteen": "canteen", "visit the medical ward": "medical", "attend supper": "observatory"}),
            "canteen": scene("Gilen's Point Canteen", "A supposed technician crew keeps apart from the prospectors.", {"return to the chief": "chief"}),
            "medical": scene("Medical Ward", "Rez Autreb lies isolated after escaping the fissure.", {"return to the chief": "chief"}),
            "observatory": scene("Observatory Dome", "The darkening supper brings competing offers for any artifact.", {"prepare the delve": "loading"}),
            "loading": scene("Loading Dock", "The crew checks its delving equipment and receives keepsakes.", {"depart in the rover": "surface"}),
            "surface": scene("Moubarra 4 Surface", "The old rover crosses unstable ground in darkness.", {"continue to the fissure": "fissure"}),
            "fissure": scene("The Fissure", "The crew scans for Blight and descends while tracking Supply.", {"enter the side tunnel": "auxiliary", "enter the structure": "antechamber"}),
            "auxiliary": scene("Auxiliary Chamber", "Rov Anker's body lies among three points of Supply.", {"continue to the structure": "antechamber"}),
            "antechamber": scene("The Antechamber", "A sealed Cathedral Vault lies beyond a circular glyph.", {"take the western passage": "western", "take the eastern passage": "eastern"}),
            "western": scene("Western Passage", "Floating dust and distant scraping lead onward.", {"enter the Memory Chamber": "memory"}),
            "memory": scene("Memory Chamber", "Jahamala Kaiff draws a city while a memory pattern pulses.", {"continue south": "key"}),
            "eastern": scene("Eastern Passage", "Dark frost covers a passage infested by Blight Crawlers.", {"continue to the Key Chamber": "key"}),
            "key": scene("Key Chamber", "Blight vines surround the mechanism that opens the vault.", {"open the Cathedral Vault": "vault"}),
            "vault": scene("Cathedral Vault", "Birsa Lovada, Nev Ringa's body, and an active fractured sphere occupy the vault.", {"leave the ruin": "aftermath"}),
            "aftermath": scene("The Fissure After the Delve", "Ytreppo waits outside while the crew decides the artifact's fate.", {"return it to the Guild": "guild", "meet Ytreppo": "ytreppo", "return to Chief Kalvanetes": "chief_end", "return without it": "lost"}),
            "guild": scene("Explorers Guild Return", "A Guild quartermaster receives recovered artifacts."),
            "ytreppo": scene("Quassar's Dream Ramp", "Ytreppo repeats his offer beside his ship."),
            "chief_end": scene("Gilen's Point Chief's Office", "Kalvanetes makes one last plea for the artifact."),
            "lost": scene("Mission End Without the Sphere", "The Black Toad has escaped with the major artifact."),
        },
        "entities": {
            "drenk": npc("Drenk Zabo", "landing", "laconic prospector"),
            "lia": npc("Lia Kalvanetes", "chief", "station chief"),
            "rez": npc("Rez Autreb", "medical", "isolated survivor"),
            "kaiff": npc("Jahamala Kaiff", "memory", "Blight-ridden prospector"),
            "lovada": npc("Birsa Lovada", "vault", "wounded prospector"),
            "zera": npc("Zera Vandao", "vault", "Black Toad leader"),
            "quartermaster": npc("Explorers Guild quartermaster", "guild", "Guild quartermaster"),
            "ytreppo": npc("Ytreppo Ashur mir-Mira", "ytreppo", "masked Coriolite"),
            "lia_end": npc("Lia Kalvanetes", "chief_end", "station chief"),
            "equipment": {
                "type": "object", "name": "delving equipment", "scene": "loading",
                "initial_state": "present", "portable": False,
            },
            "anker_body": {
                "type": "object", "name": "Rov Anker's body", "scene": "auxiliary",
                "initial_state": "present", "portable": False,
            },
            "ringa_body": {
                "type": "object", "name": "Nev Ringa's body", "scene": "vault",
                "initial_state": "present", "portable": False,
            },
            "memory_disc": {
                "type": "item", "name": "frost-encased memory disc", "scene": "eastern",
                "initial_state": "hidden", "portable": True,
            },
            "sphere": {
                "type": "item", "name": "fractured sphere", "scene": "vault",
                "initial_state": "active", "portable": False,
                "states": {
                    "active": {
                        "triggers": [
                            "return the fractured sphere to its pedestal",
                            "place the sphere back on the pedestal"
                        ],
                        "on_trigger": {
                            "to_state": "deactivated",
                            "narration": "The pedestal deactivates the gravity waves."
                        }
                    },
                    "deactivated": {
                        "triggers": ["pick up", "take the fractured sphere"],
                        "on_trigger": {
                            "to_state": "obtained",
                            "narration": "The deactivated sphere can now be carried."
                        }
                    }
                }
            }
        },
        "narrative_scopes": [
            {"id": "physical", "kind": "physical", "navigable": True},
            {"id": "builder_vision", "kind": "vision", "parent_scope": "physical", "navigable": False}
        ],
        "embedded_settings": [{
            "scope": {"id": "builder_vision", "kind": "vision", "navigable": False},
            "scenes": [{
                "id": "floating_city", "name": "Builder Floating City",
                "aliases": ["the City in the memory pattern"],
                "scope_id": "builder_vision", "navigable": False
            }],
            "entities": []
        }]
    }


def common_opening() -> list[dict]:
    return [
        {"input": "开始游戏", "response": "The shuttle lands at Gilen's Point beneath the immense gas giant.", "scene": "landing", "contains": ["Gilen's Point"], "covers": ["arrival"]},
        movement_step("We meet the station chief.", "Lia Kalvanetes briefs the crew about the five missing prospectors.", "chief", ["chief_briefing"], ["five missing"]),
    ]


def generate() -> None:
    write_json("coriolis_sky_machine_world.json", build_main_world())

    rescue = common_opening() + [
        movement_step("We inspect the supposed technicians in the canteen.", "The cheerful newcomers are Zera Vandao's disguised Black Toad crew.", "canteen", ["canteen"], ["Black Toad"]),
        movement_step("We return to Chief Kalvanetes.", "The crew returns to the chief's office.", "chief", [], ["chief"]),
        movement_step("We visit the survivor in the medical ward.", "Rez Autreb lies in isolation, consumed by Blight.", "medical", ["medical_ward"], ["Blight"]),
        {"input": "I ask Rez Autreb what he saw.", "response": "Autreb spasms and asks whether you have seen the City.", "select_npc": "rez", "contains": ["City"], "covers": ["city_clue"]},
        {"input": "I keep questioning the unconscious survivor.", "response": "Autreb cannot answer while unconscious.", "set_entity_states": {"rez": "unconscious"}, "code_response": True, "contains": ["没有可交谈"], "covers": ["unconscious_npc_block"]},
        movement_step("We return to the chief before supper.", "The crew leaves the ward and returns to the office.", "chief", [], ["office"]),
        movement_step("We attend the darkening supper.", "At supper Ytreppo and Kalvanetes separately offer money for any artifact.", "observatory", ["dark_supper", "competing_offers"], ["offer"]),
        movement_step("We prepare the delving pod.", "At the loading dock the crew receives keepsakes and notices sabotaged equipment.", "loading", ["equipment_check", "sabotage_clue", "blessings"], ["sabotaged"]),
        movement_step("We drive toward the fissure and clear the spreading crevice.", "The driver keeps the rover clear of the collapsing ground.", "surface", ["rover_shakes", "rover_safe"], ["clear"]),
        movement_step("We continue to the fissure.", "The crew scans the fissure and descends while tracking Blight and Supply.", "fissure", ["fissure_scan", "fissure_delving"], ["Supply"]),
        movement_step("We enter the obstructed side tunnel.", "The side tunnel opens into an auxiliary chamber.", "auxiliary", ["side_tunnel"], ["auxiliary"]),
        {"input": "I inspect Rov Anker's body.", "response": "Anker died after dressing a broken leg; three points of Supply remain nearby.", "select_object": "anker_body", "contains": ["three points"], "covers": ["auxiliary_chamber"]},
        movement_step("We continue into the Builder structure.", "The crew enters the antechamber before the sealed vault.", "antechamber", ["antechamber"], ["sealed vault"]),
        movement_step("We take the western passage.", "Floating dust leads through the western passage.", "western", ["western_route"], ["dust"]),
        movement_step("We follow the scraping into the Memory Chamber.", "Jahamala Kaiff sits beneath the pulsing memory pattern.", "memory", [], ["memory pattern"]),
        {"input": "I enter the Builder Floating City shown by the pattern.", "response": "A vision is not a physical destination.", "code_response": True, "contains": ["不能直接越过"], "covers": ["memory_vision_scope_block"]},
        {"input": "I persuade Jahamala Kaiff to come with us.", "response": "Kaiff responds and agrees to leave with support from an Explorer.", "select_npc": "kaiff", "contains": ["agrees"], "covers": ["kaiff_rescue", "memory_vision"]},
        movement_step("We continue south to the mechanism.", "The crew crosses the southern passage and reaches the vine-covered key mechanism.", "key", ["key_chamber"], ["mechanism"]),
        movement_step("We press the mechanism and enter the Cathedral Vault.", "The key mechanism opens the Cathedral Vault and a Blight surge meets the crew.", "vault", ["unlock_vault", "cathedral_vault"], ["Blight surge"]),
        {"input": "I take the active fractured sphere.", "response": "The active sphere cannot yet be carried.", "select_object": "sphere", "code_response": True, "contains": ["无法被直接带走"], "covers": ["active_sphere_take_block"]},
        {"input": "I return the fractured sphere to its pedestal.", "response": "The pedestal deactivates the gravity waves.", "select_object": "sphere", "contains": ["deactivates"], "covers": ["sphere_hazard", "deactivate_sphere"]},
        {"input": "I pick up the fractured sphere.", "response": "The deactivated sphere can now be carried.", "select_object": "sphere", "contains": ["carried"], "covers": [], "expect": {"inventory_contains": ["sphere"], "object_locations": {"sphere": INVENTORY}, "world_event_types_contains": ["item_picked_up"]}},
        {"input": "I ask Birsa Lovada what happened.", "response": "After receiving water, Lovada explains that removing the sphere began the disaster.", "select_npc": "lovada", "contains": ["removing"], "covers": ["prospector_truth", "lovada_rescue"]},
        {"input": "I refuse Zera Vandao and keep the sphere.", "response": "Vandao attacks, but the Explorers drive the Black Toad crew away and retain the sphere.", "select_npc": "zera", "contains": ["retain"], "covers": ["toad_ambush", "fight_toads"], "expect": {"inventory_contains": ["sphere"], "object_locations": {"sphere": INVENTORY}}},
        movement_step("We leave the ruin with the sphere.", "The crew pays the return Supply and emerges into Jumuah's light.", "aftermath", ["leave_ruin"], ["Jumuah"]),
        movement_step("We return the artifact to the Explorers Guild.", "The crew returns home to the Explorers Guild.", "guild", [], ["Guild"]),
        {"input": "I hand the fractured sphere over to the quartermaster.", "response": "The quartermaster accepts the major artifact for Master Moska.", "select_npc": "quartermaster", "select_object": "sphere", "contains": ["accepts"], "covers": ["artifact_to_guild"], "expect": {"object_locations": {"sphere": {"kind": "entity", "id": "quartermaster"}}, "world_event_types_contains": ["item_transferred"]}},
        {"input": "I use the sphere again after handing it over.", "response": "The sphere is no longer in the player's possession.", "code_response": True, "contains": ["未能把这个动作对应"], "covers": ["transferred_item_reuse_block"], "expect": {"object_locations": {"sphere": {"kind": "entity", "id": "quartermaster"}}}},
    ]
    write_route("coriolis_rescue_guild", "coriolis_sky_machine", "coriolis_sky_machine", rescue)

    surrender = common_opening() + [
        {"input": "I take the frost-encased memory disc before entering the fissure.", "response": "The undiscovered artifact cannot be taken.", "code_response": True, "contains": ["没有发现"], "covers": ["hidden_disc_block"], "expect": {"response_not_contains": ["eastern passage"]}},
        {"input": "I go directly to the Cathedral Vault.", "response": "The sealed vault cannot be reached from here.", "code_response": True, "contains": ["不能直接越过"], "covers": ["vault_jump_block"]},
        movement_step("We attend supper before departing.", "Ytreppo and Kalvanetes make competing offers during the darkening supper.", "observatory", ["dark_supper", "competing_offers"], ["offers"]),
        movement_step("We rush through the equipment check.", "The loading dock crew gives the Explorers keepsakes.", "loading", ["equipment_check", "blessings"], ["keepsakes"]),
        movement_step("We drive out, fail to avoid the crevice, and continue on foot.", "The rover crashes; the crew loses Supply and walks onward.", "surface", ["rover_shakes", "rover_crash"], ["crashes"]),
        movement_step("We reach and descend the fissure.", "The crew scans and descends through Blight events.", "fissure", ["fissure_scan", "fissure_delving"], ["Blight"]),
        movement_step("We skip the side tunnel and enter the structure.", "The crew enters the antechamber directly.", "antechamber", ["side_tunnel", "antechamber"], ["antechamber"]),
        movement_step("We take the eastern passage.", "Dark frost and Blight Crawlers fill the eastern passage.", "eastern", ["eastern_route"], ["Crawlers"]),
        {"input": "I search the dark frost.", "response": "The search reveals a frost-encased memory disc before the Crawlers attack.", "set_entity_states": {"memory_disc": "revealed"}, "contains": ["memory disc"], "covers": ["disc_and_crawlers"]},
        {"input": "I pick up the memory disc.", "response": "You secure the lesser artifact in your equipment.", "select_object": "memory_disc", "contains": ["lesser artifact"], "expect": {"inventory_contains": ["memory_disc"], "object_locations": {"memory_disc": INVENTORY}}},
        movement_step("We continue to the Key Chamber.", "The crew reaches the key mechanism behind the vines.", "key", ["key_chamber"], ["key mechanism"]),
        movement_step("We activate it and enter the Cathedral Vault.", "The open vault contains survivors, a body, and the active sphere.", "vault", ["unlock_vault", "cathedral_vault", "prospector_truth"], ["active sphere"]),
        {"input": "I ask Lia Kalvanetes to explain the sphere here.", "response": "The station chief is not present in the vault.", "code_response": True, "contains": ["选择一名在场人物"], "covers": ["absent_npc_block"]},
        {"input": "I place the sphere back on the pedestal.", "response": "The pedestal deactivates the gravity waves.", "select_object": "sphere", "contains": ["deactivates"], "covers": ["sphere_hazard", "deactivate_sphere"]},
        {"input": "I pick up the fractured sphere.", "response": "The deactivated sphere can now be carried.", "select_object": "sphere", "contains": ["carried"], "expect": {"inventory_contains": ["sphere", "memory_disc"], "object_locations": {"sphere": INVENTORY, "memory_disc": INVENTORY}}},
        {"input": "I hand the fractured sphere over to Zera Vandao.", "response": "Zera accepts the sphere and orders her crew to retreat.", "select_npc": "zera", "select_object": "sphere", "contains": ["accepts"], "covers": ["toad_ambush", "surrender_sphere"], "expect": {"inventory_contains": ["memory_disc"], "object_locations": {"sphere": {"kind": "entity", "id": "zera"}, "memory_disc": INVENTORY}, "world_event_types_contains": ["item_transferred"]}},
        movement_step("We leave the ruin after the Black Toad.", "The crew returns through the ruin without the sphere.", "aftermath", ["leave_ruin"], ["without the sphere"]),
        {"input": "I take the fractured sphere back from here.", "response": "The sphere is with Zera, not at the fissure.", "code_response": True, "contains": ["未能把这个动作对应"], "covers": ["remote_sphere_retake_block"], "expect": {"object_locations": {"sphere": {"kind": "entity", "id": "zera"}}, "response_not_contains": ["Cathedral Vault"]}},
        movement_step("We report that the major artifact was lost.", "The mission ends without the sphere, though the memory disc remains with the crew.", "lost", ["artifact_lost"], ["without the sphere"]),
    ]
    write_route("coriolis_surrender_toad", "coriolis_sky_machine", "coriolis_sky_machine", surrender)

    ending_world = {
        "name": "eval_coriolis_aftermath", "rule_system": "coriolis",
        "dice_system": "d6_pool", "automatic_check_adapter": False,
        "starting_scene": "aftermath",
        "opening": "The crew emerges from the fissure carrying the deactivated fractured sphere.",
        "scenes": {
            "aftermath": scene("The Fissure After the Delve", "Ytreppo waits beside Quassar's Dream.", {"meet Ytreppo": "ytreppo", "return to Kalvanetes": "chief"}),
            "ytreppo": scene("Quassar's Dream Ramp", "Ytreppo offers six thousand dirham for the sphere."),
            "chief": scene("Gilen's Point Chief's Office", "Kalvanetes offers to split a buyer's payment."),
        },
        "entities": {
            "sphere": {"type": "item", "name": "fractured sphere", "scene": "aftermath", "initial_state": "obtained", "portable": True},
            "ytreppo": npc("Ytreppo Ashur mir-Mira", "ytreppo", "masked Coriolite"),
            "lia": npc("Lia Kalvanetes", "chief", "station chief"),
        },
    }
    write_json("coriolis_aftermath_world.json", ending_world)
    write_route("coriolis_sell_ytreppo", "coriolis_sky_machine", "coriolis_aftermath", [
        {"input": "开始游戏", "response": "The crew emerges with the deactivated sphere.", "scene": "aftermath", "contains": ["sphere"], "covers": []},
        movement_step("We meet Ytreppo at his ship.", "Ytreppo repeats his offer of six thousand dirham.", "ytreppo", ["ytreppo_final_offer"], ["six thousand"]),
        {"input": "I hand the fractured sphere over to Ytreppo.", "response": "Ytreppo accepts the sphere and offers the crew a ride.", "select_npc": "ytreppo", "select_object": "sphere", "contains": ["accepts"], "covers": ["artifact_to_ytreppo"], "expect": {"object_locations": {"sphere": {"kind": "entity", "id": "ytreppo"}}, "world_event_types_contains": ["item_transferred"]}},
    ])
    write_route("coriolis_sell_kalvanetes", "coriolis_sky_machine", "coriolis_aftermath", [
        {"input": "开始游戏", "response": "The crew emerges with the deactivated sphere.", "scene": "aftermath", "contains": ["sphere"], "covers": []},
        movement_step("We reject Ytreppo and return to Kalvanetes.", "Kalvanetes makes one final plea to arrange a buyer.", "chief", ["ytreppo_final_offer"], ["buyer"]),
        {"input": "I give the fractured sphere to Lia Kalvanetes.", "response": "Kalvanetes takes custody of the sphere for the Navigators Guild buyer.", "select_npc": "lia", "select_object": "sphere", "contains": ["custody"], "covers": ["artifact_to_kalvanetes"], "expect": {"object_locations": {"sphere": {"kind": "entity", "id": "lia"}}, "world_event_types_contains": ["item_transferred"]}},
    ])

    write_json("coriolis_sky_machine_coverage.json", {
        "module": "coriolis_sky_machine",
        "required": [
            "arrival", "chief_briefing", "canteen", "medical_ward", "city_clue",
            "unconscious_npc_block", "dark_supper", "competing_offers",
            "equipment_check", "sabotage_clue", "blessings", "rover_shakes",
            "rover_safe", "rover_crash", "fissure_scan", "fissure_delving",
            "side_tunnel", "auxiliary_chamber", "antechamber", "western_route",
            "memory_vision_scope_block", "kaiff_rescue", "memory_vision",
            "eastern_route", "disc_and_crawlers", "hidden_disc_block",
            "vault_jump_block", "key_chamber", "unlock_vault", "cathedral_vault",
            "prospector_truth", "lovada_rescue", "active_sphere_take_block",
            "sphere_hazard", "deactivate_sphere", "toad_ambush", "fight_toads",
            "surrender_sphere", "absent_npc_block", "leave_ruin",
            "transferred_item_reuse_block", "remote_sphere_retake_block",
            "artifact_to_guild", "artifact_to_ytreppo", "artifact_to_kalvanetes",
            "artifact_lost"
        ]
    })


if __name__ == "__main__":
    generate()
    print("Generated Coriolis: The Sky Machine runtime fixtures.")

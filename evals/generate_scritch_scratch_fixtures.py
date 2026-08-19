#!/usr/bin/env python3
"""Generate source-grounded sandbox and destructive cases for Scritch Scratch."""

from __future__ import annotations

from generate_exhaustive_fixtures import movement_step, write_json, write_route


INVENTORY = {"kind": "inventory", "id": "player"}


def scene(name: str, desc: str, exits: dict | None = None,
          aliases: list[str] | None = None) -> dict:
    row = {"name": name, "desc": desc, "source_text": desc,
           "exits": exits or {}}
    if aliases:
        row["aliases"] = aliases
    return row


def discover(*entity_ids: str) -> list[dict]:
    return [{"type": "entity_discovered", "entity_id": eid} for eid in entity_ids]


def npc(name: str, at: str, label: str, states: dict | None = None) -> dict:
    row = {"type": "npc", "name": name, "scene": at,
           "initial_state": "present", "public_label": label}
    if states:
        row["states"] = states
    return row


def build_world() -> dict:
    village_exits = {
        "Old Gurteen's cottage": "gurteen_cottage",
        "St. Gertrude's church": "church",
        "drive to Appleford": "appleford",
        "hospital": "hospital",
    }
    internal_rooms = {
        "front hall and parlor": "parlor", "dining room": "dining",
        "kitchen": "kitchen", "woods": "woods",
        "return to Lucy's Tea Shoppe": "tea_shop", "hospital": "hospital",
    }
    cottage_exits = {
        "woods": "woods", "return to Lucy's Tea Shoppe": "tea_shop",
        "hospital": "hospital",
        "enter using the Council keys": {
            "target": "parlor",
            "requires_entity_states": {"front_door": "opened"},
        },
    }
    appleford_exits = {
        "Golden Ram": "golden_ram", "Library and Museum": "library",
        "Community Hospital": "hospital", "return to Muscoby": "tea_shop",
    }
    return {
        "name": "eval_manual_scritch_scratch",
        "description": "Investigators clear Old Gurteen's cottage and uncover Muscoby's rat-catching tribute.",
        "rule_system": "coc", "ruleset": "coc", "dice_system": "d100",
        "automatic_check_adapter": True,
        "starting_scene": "tea_shop",
        "opening": (
            "Rain falls over Muscoby. Lucy Albright appears behind the counter "
            "of her nearly empty tea shop. The Council has supplied keys to "
            "clear Old Gurteen's cottage while he recovers in hospital."),
        "scenes": {
            "tea_shop": scene("Lucy's Tea Shoppe",
                "Lucy Albright runs the former Post Office as a struggling tea shop.",
                village_exits),
            "gurteen_cottage": scene("Old Gurteen's Cottage",
                "The empty cottage stands at the lane's end with woods behind it.",
                cottage_exits, ["Gurteen cottage", "the cottage"]),
            "parlor": scene("Gurteen Cottage Parlor",
                "A black family Bible rests on the coffee table in the formal parlor.",
                internal_rooms, ["parlor", "front hall"]),
            "dining": scene("Gurteen Cottage Dining Room",
                "Rodent scrapbooks and a Green Man painting dominate the dining room.",
                internal_rooms, ["dining room"]),
            "kitchen": scene("Gurteen Cottage Kitchen",
                "An unsafe gas cooker, matches, and a torch remain in the old kitchen.",
                {**internal_rooms, "courtyard": "courtyard"}, ["kitchen"]),
            "courtyard": scene("Gurteen Cottage Courtyard",
                "Outbuildings contain a chemical store, potting shed, and workshop.",
                {"kitchen": "kitchen", "workshop": "workshop", "woods": "woods"},
                ["courtyard"]),
            "workshop": scene("Gurteen's Workshop",
                "Animal traps, a forge, tools, and an attic ladder fill the workshop.",
                {"courtyard": "courtyard", "climb into the attic": "attic"}),
            "attic": scene("Gurteen Cottage Attic",
                "Dusty cases, old clothing, and unexplained scratching occupy the attic.",
                {"climb down to the workshop": "workshop"}),
            "woods": scene("Woods Behind Gurteen's Cottage",
                "Dozens of rat and mouse corpses hang by their tails between moving trees.",
                {"return to the cottage": "gurteen_cottage",
                 "follow the woodland path to church": "church"}),
            "church": scene("St. Gertrude's Church",
                "Ancient mouse carvings and an enormous Green Man watch the nave.",
                {"return through the woods": "woods", "return to the tea shop": "tea_shop",
                 "drive to Appleford": "appleford",
                 "face the awakened Green Man": {
                     "target": "confrontation", "requires_flag": "end_game_ready_discovered"}}),
            "appleford": scene("Appleford Marketplace",
                "The Golden Ram faces the Library and Museum; the hospital lies nearby.",
                appleford_exits),
            "golden_ram": scene("The Golden Ram",
                "Young George tends the bar while Old George waits for a pint.",
                {"marketplace": "appleford", "return to Muscoby": "tea_shop"}),
            "library": scene("Appleford Library",
                "Joyce Deakins guards local history and the locked museum downstairs.",
                {"marketplace": "appleford", "enter the museum": {
                    "target": "museum", "requires_inventory": ["museum_key"]}}),
            "museum": scene("Appleford Museum",
                "Reginald Gurteen's Victorian rodent dioramas stand against the far wall.",
                {"return to the library": "library"}),
            "hospital": scene("Appleford Community Hospital",
                "Old Gurteen is a distressed patient who repeatedly asks to go home.",
                {"marketplace": "appleford", "return to the cottage": "gurteen_cottage"}),
            "confrontation": scene("The Green Man's Manifestation",
                "The Green Man lashes out while a rat horde blocks escape.",
                {"burn the woodland tether": "burned_ending",
                 "drive away from Muscoby": "fled_ending",
                 "the tribute takes us": "death_ending"}),
            "burned_ending": scene("The Woods Burn",
                "Fire drives off the Green Man and may break his tether."),
            "fled_ending": scene("Escape from Muscoby",
                "The investigators escape, but the cycle remains unresolved."),
            "death_ending": scene("Tribute to the Green Man",
                "Deaths briefly satiate the ancient spirit."),
        },
        "entities": {
            "lucy_albright": npc("Lucy Albright", "tea_shop", "tea shop owner"),
            "old_george": npc("Old George", "golden_ram", "elderly storyteller", {
                "present": {"triggers": ["ask Old George about the rhyme"],
                    "on_trigger": {"to_state": "warned", "events": discover("old_george_history")}},
                "warned": {}}),
            "young_george": npc("Young George", "golden_ram", "pub landlord"),
            "joyce_deakins": npc("Joyce Deakins", "library", "head librarian", {
                "present": {"triggers": ["ask Joyce Deakins for the museum key"],
                    "on_trigger": {"to_state": "helped", "events": discover("museum_key")}},
                "helped": {}, "hospitalized": {"interactable": False}}),
            "old_gurteen": npc("Old Gurteen", "hospital", "distressed patient", {
                "present": {"triggers": ["ask Old Gurteen why the trees need feeding"],
                    "on_trigger": {"to_state": "restrained", "events": discover("gurteen_warning")}},
                "restrained": {"interactable": False}, "dead": {"interactable": False}}),
            "green_man": npc("The Green Man", "confrontation", "writhing foliage figure", {
                "present": {}, "driven_off": {"interactable": False}}),
            "rat_horde": npc("rat horde", "confrontation", "swarming rats"),
            "cottage_keys": {"type": "item", "name": "keys to Old Gurteen's cottage",
                "scene": "tea_shop", "initial_state": "obtained", "portable": True},
            "front_door": {"type": "door", "name": "dark green cottage front door",
                "scene": "gurteen_cottage", "initial_state": "locked", "portable": False,
                "states": {"locked": {"triggers": ["unlock the cottage front door"],
                    "requires_inventory": ["cottage_keys"],
                    "on_trigger": {"to_state": "opened", "events": discover("cottage_access")}},
                    "opened": {}}},
            "cottage_access": {"type": "clue", "name": "access to Gurteen's cottage",
                "scene": "gurteen_cottage", "initial_state": "hidden", "portable": False},
            "family_bible": {"type": "clue", "name": "Gurteen family Bible",
                "scene": "parlor", "initial_state": "present", "portable": False,
                "states": {"present": {"triggers": ["read the Gurteen family Bible"],
                    "on_trigger": {"to_state": "found"}}, "found": {}}},
            "scrapbooks": {"type": "clue", "name": "rodent and Green Man scrapbooks",
                "scene": "dining", "initial_state": "present", "portable": False},
            "green_man_painting": {"type": "object", "name": "Green Man painting",
                "scene": "dining", "initial_state": "present", "portable": False},
            "matches": {"type": "item", "name": "large box of matches",
                "scene": "kitchen", "initial_state": "present", "portable": True},
            "chemicals": {"type": "item", "name": "flammable poisons and herbicides",
                "scene": "courtyard", "initial_state": "hidden", "portable": True},
            "chemical_store": {"type": "container", "name": "chemical storage shelves",
                "scene": "courtyard", "initial_state": "closed", "portable": False,
                "states": {"closed": {"triggers": ["inspect the chemical storage shelves"],
                    "on_trigger": {"to_state": "opened", "events": discover("chemicals")}},
                    "opened": {}}},
            "rat_bunting": {"type": "clue", "name": "rat corpse bunting",
                "scene": "woods", "initial_state": "present", "portable": False,
                "states": {"present": {"triggers": ["inspect the rat corpse bunting"],
                    "on_trigger": {"to_state": "found"}}, "found": {}}},
            "stone_watchers": {"type": "clue", "name": "church Green Man stone watchers",
                "scene": "church", "initial_state": "present", "portable": False,
                "states": {"present": {"triggers": ["study the church Green Man carvings"],
                    "on_trigger": {"to_state": "found"}}, "found": {}}},
            "old_george_history": {"type": "clue", "name": "Old George's rat-catcher history",
                "scene": "golden_ram", "initial_state": "hidden", "portable": False},
            "museum_key": {"type": "item", "name": "Appleford museum key",
                "scene": "library", "initial_state": "hidden", "portable": True},
            "museum_history": {"type": "clue", "name": "Reginald Gurteen museum history",
                "scene": "museum", "initial_state": "hidden", "portable": False},
            "dioramas": {"type": "container", "name": "Victorian rodent dioramas",
                "scene": "museum", "initial_state": "present", "portable": False,
                "states": {"present": {"triggers": ["inspect Reginald Gurteen's dioramas"],
                    "on_trigger": {"to_state": "examined", "events": discover("museum_history")}},
                    "examined": {}}},
            "gurteen_warning": {"type": "clue", "name": "Gurteen's warning to burn the woods",
                "scene": "hospital", "initial_state": "hidden", "portable": False},
            "end_game_gate": {"type": "object", "name": "church Green Man avatar",
                "scene": "church", "initial_state": "dormant", "portable": False,
                "states": {"dormant": {
                    "triggers": ["prepare for the Green Man's attack"],
                    "requires_flags": ["family_bible_discovered", "stone_watchers_discovered"],
                    "requires_any_flags": ["old_george_history_discovered", "museum_history_discovered"],
                    "on_trigger": {"to_state": "awakened", "events": discover("end_game_ready")}},
                    "awakened": {}}},
            "end_game_ready": {"type": "clue", "name": "assembled evidence awakens the Green Man",
                "scene": "church", "initial_state": "hidden", "portable": False},
            "woods_tether": {"type": "object", "name": "Green Man's woodland tether",
                "scene": "confrontation", "initial_state": "intact", "portable": False,
                "states": {"intact": {"triggers": ["burn the woodland tether with the chemicals and matches"],
                    "requires_inventory": ["matches", "chemicals"],
                    "on_trigger": {"to_state": "destroyed", "events": [{
                        "type": "entity_removed", "entity_id": "green_man",
                        "state": "driven_off", "condition": "driven off by fire"}]}},
                    "destroyed": {}}},
        },
        "embedded_settings": [
            {"scope": {"id": "bible_story", "kind": "story", "navigable": False},
             "scenes": [{"id": "ark_story", "name": "Philistine Ark Story", "navigable": False}], "entities": []},
            {"scope": {"id": "museum_dioramas", "kind": "depiction", "navigable": False},
             "scenes": [{"id": "victorian_mouse_town", "name": "Victorian Mouse Town", "navigable": False}], "entities": []},
            {"scope": {"id": "future_hook", "kind": "future_hook", "navigable": False},
             "scenes": [{"id": "drowned_valley", "name": "Drowned Muscoby", "navigable": False}], "entities": []},
        ],
    }


def generate() -> None:
    write_json("scritch_scratch_world.json", build_world())

    burn = [
        {"input": "开始游戏", "response": "Lucy waits in the warm tea shop while rain falls over Muscoby.", "scene": "tea_shop", "contains": ["Lucy"], "covers": ["assignment", "arrive_muscoby", "sandbox_choice"]},
        movement_step("We go to Old Gurteen's cottage.", "The cottage stands empty beside the woods.", "gurteen_cottage", ["cottage_exterior"], ["empty"]),
        {"input": "We unlock the cottage front door.", "response": "The Council keys unlock the dark green door.", "select_object": "front_door", "contains": ["unlock"], "covers": ["enter_cottage"]},
        movement_step("We enter the front hall and parlor.", "A black family Bible rests in the formal parlor.", "parlor", [], ["Bible"]),
        {"input": "We read the Gurteen family Bible.", "response": "The family tree names Gilbert Oswin Gurteen and generations of rat catchers.", "select_object": "family_bible", "contains": ["Gilbert"], "covers": ["family_bible"]},
        movement_step("We go to the kitchen.", "An unsafe gas cooker and a large box of matches remain in the kitchen.", "kitchen", ["kitchen_resources"], ["matches"]),
        {"input": "I take the large box of matches.", "response": "You take the matches.", "select_object": "matches", "contains": ["matches"], "expect": {"inventory_contains": ["matches"], "object_locations": {"matches": INVENTORY}}},
        movement_step("We enter the courtyard.", "The courtyard outbuildings include a chemical store and workshop.", "courtyard", [], ["chemical"]),
        {"input": "We inspect the chemical storage shelves.", "response": "The shelves reveal illegal poisons and flammable herbicides.", "select_object": "chemical_store", "contains": ["herbicides"], "covers": ["courtyard_chemicals"]},
        {"input": "I take the flammable poisons and herbicides.", "response": "You secure the flammable chemicals.", "select_object": "chemicals", "contains": ["chemicals"], "expect": {"inventory_contains": ["matches", "chemicals"]}},
        movement_step("We go into the woods.", "Countless rat and mouse corpses hang between the trees.", "woods", ["rat_bunting"], ["corpses"]),
        {"input": "We inspect the rat corpse bunting.", "response": "The hanging rodents form a deliberate tribute.", "select_object": "rat_bunting", "contains": ["tribute"], "covers": ["creeping_woods"]},
        movement_step("We return to the cottage.", "The woods have crept closer to the cottage hedge.", "gurteen_cottage", [], ["closer"]),
        movement_step("We return to Lucy's Tea Shoppe.", "Lucy points toward the Golden Ram and Appleford museum.", "tea_shop", ["tea_shop", "lucy_rat_catcher", "lucy_appleford_leads"], ["Golden Ram"]),
        movement_step("We drive to Appleford.", "The market town offers a pub, museum, and hospital.", "appleford", ["appleford_choice"], ["museum"]),
        movement_step("We enter the Golden Ram.", "An elderly storyteller waits at the bar beside the pub landlord.", "golden_ram", ["golden_ram"], ["elderly storyteller"]),
        {"input": "I ask Old George about the rhyme.", "response": "Old George chants the rhyme and warns that the woods are waking.", "select_npc": "old_george", "contains": ["woods are waking"], "covers": ["old_george_rhyme", "young_george_warning"]},
        movement_step("We return to Muscoby.", "Rain still falls over Muscoby.", "tea_shop", [], ["Muscoby"]),
        movement_step("We continue to St. Gertrude's church.", "Gurteen graves surround the ancient church.", "church", ["churchyard", "apollo_stones", "mouse_connection"], ["Gurteen"]),
        {"input": "We study the church Green Man carvings.", "response": "The vine-covered font and huge oak-leaf face watch the nave.", "select_object": "stone_watchers", "contains": ["oak-leaf"], "covers": ["stone_watchers"]},
        {"input": "We prepare for the Green Man's attack.", "response": "The assembled history makes the stone face begin to move.", "contains": ["stone face"], "covers": ["history_threshold", "manifestation_choice"]},
        movement_step("We face the awakened Green Man.", "The Green Man attacks while rats close every path of retreat.", "confrontation", ["green_man_attack", "rat_blockade", "final_response"], ["rats"]),
        {"input": "We burn the woodland tether with the chemicals and matches.", "response": "Fire races through the tether and drives the Green Man away.", "contains": ["drives"], "covers": ["drive_off"], "expect": {"entity_states": {"woods_tether": "destroyed", "green_man": "driven_off"}, "inventory_contains": ["matches", "chemicals"]}},
        movement_step("We burn the woodland tether.", "The burning woods may finally break the ancient tether.", "burned_ending", ["burn_woods", "rewards"], ["tether"]),
    ]
    write_route("scritch_burn_victory", "scritch_scratch", "scritch_scratch", burn)

    museum = [
        {"input": "开始游戏", "response": "Lucy waits behind the tea shop counter.", "scene": "tea_shop", "contains": ["Lucy"], "covers": []},
        {"input": "I enter the Victorian Mouse Town inside a diorama.", "response": "The miniature display is not a physical destination.", "code_response": True, "contains": ["不能直接越过"], "covers": ["diorama_scope_block"]},
        movement_step("We go to Old Gurteen's cottage.", "The cottage waits at the lane's end.", "gurteen_cottage", [], ["cottage"]),
        {"input": "We unlock the cottage front door.", "response": "The keys open the front door.", "select_object": "front_door", "contains": ["open"], "covers": []},
        movement_step("We enter the parlor.", "The family Bible lies on the coffee table.", "parlor", [], ["Bible"]),
        {"input": "We read the Gurteen family Bible.", "response": "The book traces the Gurteen line of rat catchers.", "select_object": "family_bible", "contains": ["rat catchers"], "covers": []},
        movement_step("We go to the woods.", "Rat corpses hang in deliberate rows.", "woods", [], ["corpses"]),
        {"input": "We inspect the rat corpse bunting.", "response": "The corpses are offerings rather than ordinary pest disposal.", "select_object": "rat_bunting", "contains": ["offerings"], "covers": []},
        movement_step("We follow the woodland path to church.", "The path reaches St. Gertrude's church.", "church", [], ["church"]),
        {"input": "We study the church Green Man carvings.", "response": "An enormous leafy face glowers over the altar.", "select_object": "stone_watchers", "contains": ["leafy"], "covers": []},
        {"input": "We prepare for the Green Man's attack.", "response": "The evidence is still incomplete.", "code_response": True, "contains": ["尚未满足"], "covers": ["premature_endgame_block"]},
        movement_step("We drive to Appleford.", "The marketplace lies between the Golden Ram and library.", "appleford", [], ["marketplace"]),
        movement_step("We visit the Library and Museum.", "The head librarian sits above the locked museum.", "library", ["library", "deakins_history"], ["head librarian"]),
        {"input": "I ask Joyce Deakins for the museum key.", "response": "Joyce lends you the museum key.", "select_npc": "joyce_deakins", "contains": ["museum key"], "covers": ["museum_access"]},
        {"input": "I take the Appleford museum key.", "response": "You take temporary custody of the key.", "select_object": "museum_key", "contains": ["key"], "expect": {"inventory_contains": ["museum_key"]}},
        movement_step("We enter the museum.", "Victorian rodent dioramas stand against the wall.", "museum", [], ["dioramas"]),
        {"input": "We inspect Reginald Gurteen's dioramas.", "response": "Their labels connect Reginald Gurteen to Muscoby in the 1860s.", "select_object": "dioramas", "contains": ["1860s"], "covers": ["gurteen_dioramas"]},
        movement_step("We return to the library.", "The investigators leave the museum.", "library", [], ["museum"]),
        movement_step("We return to the marketplace.", "The investigators return outside.", "appleford", [], ["outside"]),
        movement_step("We return to Muscoby.", "Rain still falls over Muscoby.", "tea_shop", [], ["rain"]),
        movement_step("We go to St. Gertrude's church.", "The Green Man carving waits over the altar.", "church", [], ["carving"]),
        {"input": "We prepare for the Green Man's attack.", "response": "The combined evidence awakens the Green Man avatar.", "contains": ["awakens"], "covers": []},
        movement_step("We face the awakened Green Man.", "Vines strike as rats gather around the church.", "confrontation", [], ["rats"]),
        movement_step("We drive away from Muscoby.", "The investigators escape by car, leaving the cycle unresolved.", "fled_ending", ["flee_muscoby", "reservoir_epilogue", "rewards"], ["unresolved"]),
    ]
    write_route("scritch_museum_escape", "scritch_scratch", "scritch_scratch", museum)

    destructive = [
        {"input": "开始游戏", "response": "Lucy waits behind the counter.", "scene": "tea_shop", "contains": ["Lucy"], "covers": []},
        {"input": "I take the hidden flammable chemicals now.", "response": "The chemicals have not been found.", "select_object": "chemicals", "code_response": True, "contains": ["当前可见或背包"], "covers": ["hidden_chemical_block"]},
        {"input": "I teleport to The Green Man's Manifestation.", "response": "The confrontation is not reachable.", "code_response": True, "contains": ["不能直接越过"], "covers": ["confrontation_jump_block"]},
        {"input": "I enter the Philistine Ark Story in the Bible.", "response": "The Bible story is not a physical destination.", "code_response": True, "contains": ["不能直接越过"], "covers": ["bible_scope_block"]},
        {"input": "I ask Old George to warn us from Lucy's shop.", "response": "Old George is not present.", "select_npc": "old_george", "code_response": True, "contains": ["无法交谈"], "covers": ["absent_old_george_block"]},
        movement_step("We go to Old Gurteen's cottage.", "The locked cottage stands beside the woods.", "gurteen_cottage", [], ["locked"]),
        {"input": "We enter the parlor without unlocking the door.", "response": "The locked door prevents entry.", "code_response": True, "contains": ["不能直接越过"], "covers": ["locked_cottage_bypass_block"]},
        movement_step("We return to Lucy's Tea Shoppe.", "Lucy remains behind the counter.", "tea_shop", [], ["Lucy"]),
        movement_step("We drive to Appleford.", "The investigators reach Appleford marketplace.", "appleford", [], ["Appleford"]),
        movement_step("We go to the Community Hospital.", "Old Gurteen waits in a private hospital room.", "hospital", ["hospital_gate"], ["Gurteen"]),
        {"input": "I ask Old Gurteen why the trees need feeding.", "response": "Gurteen warns that the trees must be fed and tries to escape.", "select_npc": "old_gurteen", "contains": ["tries to escape"], "covers": ["gurteen_warning", "gurteen_escape"]},
        {"input": "I keep questioning the restrained Old Gurteen.", "response": "Staff have restrained him and he cannot continue.", "select_npc": "old_gurteen", "code_response": True, "contains": ["没有可交谈"], "covers": ["restrained_gurteen_block"]},
    ]
    write_route("scritch_scope_and_lifecycle", "scritch_scratch", "scritch_scratch", destructive)

    required = sorted({point for route in (burn, museum, destructive)
                       for turn in route for point in turn.get("covers", [])})
    write_json("scritch_scratch_coverage.json",
               {"module": "scritch_scratch", "required": required})


if __name__ == "__main__":
    generate()

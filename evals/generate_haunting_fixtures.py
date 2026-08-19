#!/usr/bin/env python3
"""Generate source-grounded destructive runtime cases for The Haunting."""

from __future__ import annotations

from generate_exhaustive_fixtures import movement_step, write_json, write_route


INVENTORY = {"kind": "inventory", "id": "player"}


def scene(name: str, desc: str, exits: dict | None = None) -> dict:
    return {
        "name": name, "desc": desc, "source_text": desc,
        "exits": exits or {},
    }


def npc(name: str, at: str, label: str,
        states: dict | None = None) -> dict:
    row = {
        "type": "npc", "name": name, "scene": at,
        "initial_state": "present", "public_label": label,
    }
    if states:
        row["states"] = states
    return row


def discover(*entity_ids: str) -> list[dict]:
    return [
        {"type": "entity_discovered", "entity_id": entity_id}
        for entity_id in entity_ids
    ]


def build_world() -> dict:
    research_exits = {
        "return to Mr. Knott": "knott_briefing",
        "visit the Corbitt House": "house_exterior",
    }
    return {
        "name": "eval_manual_haunting",
        "description": (
            "Investigators are hired to examine the troubled Corbitt House "
            "in 1920s Boston."),
        "rule_system": "coc",
        "ruleset": "coc",
        "dice_system": "d100",
        "automatic_check_adapter": True,
        "starting_scene": "knott_briefing",
        "opening": (
            "A landlord, Mr. Knott, asks you to examine the Corbitt House. "
            "He gives you the keys, address, twenty dollars in advance, and "
            "offers twenty dollars per day. You may research at the Boston "
            "Globe, Central Library, Hall of Records, or visit the house."),
        "scenes": {
            "knott_briefing": scene(
                "Meeting with Mr. Knott",
                "Mr. Knott commissions the investigation and provides the house keys.",
                {
                    "Boston Globe": "boston_globe",
                    "Central Library": "central_library",
                    "Hall of Records": "hall_of_records",
                    "ask around the neighborhood": "neighborhood",
                    "Corbitt House": "house_exterior",
                }),
            "boston_globe": scene(
                "The Boston Globe",
                "Arty Wilmot guards access to the newspaper clipping morgue.",
                research_exits),
            "central_library": scene(
                "The Central Library",
                "Each successful half-day of research yields the next house record.",
                research_exits),
            "hall_of_records": scene(
                "Hall of Records",
                "Civil and church registers contain Corbitt's estate records.",
                {
                    **research_exits,
                    "follow the restricted record trail": {
                        "target": "police_records",
                        "requires_flag": "chapel_record_discovered",
                    },
                }),
            "police_records": scene(
                "Higher Courts and Central Police Station",
                "Restricted files contain the suppressed account of the 1912 raid.",
                {"return to the Hall of Records": "hall_of_records",
                 "visit the Corbitt House": "house_exterior"}),
            "neighborhood": scene(
                "The Corbitt House Neighborhood",
                "Mr. Dooley sells cigars and newspapers near the lone old residence.",
                {
                    **research_exits,
                    "visit Roxbury Sanitarium": {
                        "target": "sanitarium",
                        "requires_flag": "macario_lead_discovered",
                    },
                    "visit the Chapel ruins": {
                        "target": "chapel",
                        "requires_flag": "chapel_lead_discovered",
                    },
                }),
            "sanitarium": scene(
                "Roxbury Sanitarium",
                "Vittorio clutches a bible while Gabriela can describe the presence.",
                {"return to the neighborhood": "neighborhood",
                 "visit the Corbitt House": "house_exterior"}),
            "chapel": scene(
                "Ruined Chapel of Contemplation",
                "Fresh triple-Y eye signs overlook fire-damaged ruins and weak flooring.",
                {"explore the weak floorboards": "chapel_basement",
                 "return to the neighborhood": "neighborhood",
                 "visit the Corbitt House": "house_exterior"}),
            "chapel_basement": scene(
                "Sealed Chapel Basement",
                "Two robed skeletons and a cabinet remain below the collapsed floor.",
                {"climb out to the Chapel ruins": "chapel",
                 "continue to the Corbitt House": "house_exterior"}),
            "house_exterior": scene(
                "Corbitt House Exterior",
                "The brick house has blank curtains, nailed windows, and new inner bolts.",
                {"enter the house": "ground_floor",
                 "return to Mr. Knott": "knott_briefing"}),
            "ground_floor": scene(
                "Corbitt House Ground Floor",
                "Storage rooms, Catholic artifacts, spoiled soup, and rat-eaten food fill the floor.",
                {"go outside": "house_exterior", "go upstairs": "upper_floor",
                 "descend to the basement": "basement"}),
            "upper_floor": scene(
                "Corbitt House Upper Floor",
                "The Macario bedrooms flank Corbitt's old spare room and its bare bed frame.",
                {"go downstairs": "ground_floor",
                 "the investigation ends in death or madness": "conclusion_loss"}),
            "basement": scene(
                "Corbitt House Basement Storage",
                "Moving stairs descend to clutter and hollow, closely fitted wall boards.",
                {"go upstairs": "ground_floor",
                 "enter the opened crawlspace": "crawlspace"}),
            "crawlspace": scene(
                "Corbitt's Hidden Crawlspace",
                "Rats nest between the two wooden basement walls.",
                {"return to basement storage": "basement",
                 "enter Corbitt's hiding place": "hiding_place"}),
            "hiding_place": scene(
                "Corbitt's Hiding Place",
                "Walter Corbitt's wizened body lies on a pallet beside crumbling papers.",
                {"return to the crawlspace": "crawlspace",
                 "conclude the solved case": "conclusion_victory"}),
            "conclusion_victory": scene(
                "Corbitt Destroyed",
                "Mr. Knott pays the investigators after Corbitt is overthrown."),
            "conclusion_loss": scene(
                "Death or Madness",
                "The investigation ends with investigators dead, insane, or fleeing."),
        },
        "entities": {
            "mr_knott": npc(
                "Mr. Knott", "knott_briefing", "landlord",
                {
                    "present": {
                        "triggers": ["file a false clean report"],
                        "on_trigger": {
                            "to_state": "dead",
                            "narration": (
                                "Knott accepts the false report, visits the house, "
                                "and is later stabbed to death in the basement."),
                        },
                    },
                    "dead": {"interactable": False},
                }),
            "arty_wilmot": npc("Arty Wilmot", "boston_globe", "pompous editor"),
            "ruth_blake": npc("Ruth Blake", "boston_globe", "records keeper"),
            "mr_dooley": npc(
                "Mr. Dooley", "neighborhood", "cigar and newspaper vendor",
                {
                    "present": {
                        "triggers": ["ask Mr. Dooley about the house and chapel"],
                        "on_trigger": {
                            "to_state": "helped",
                            "narration": (
                                "Dooley describes the Macarios and points out the "
                                "ruined Chapel of Contemplation."),
                            "events": discover("macario_lead", "chapel_lead"),
                        },
                    },
                    "helped": {},
                }),
            "vittorio": npc("Vittorio Macario", "sanitarium", "distressed patient"),
            "gabriela": npc("Gabriela Macario", "sanitarium", "conscious patient"),
            "corbitt": {
                "type": "npc", "name": "Walter Corbitt",
                "aliases": ["the wizened body", "undead fiend"],
                "scene": "hiding_place", "initial_state": "dormant",
                "public_label": "wizened body",
                "states": {
                    "dormant": {
                        "interactable": False,
                        "triggers": ["confront Walter Corbitt using his own dagger"],
                        "requires_inventory": ["magic_dagger"],
                        "on_trigger": {
                            "to_state": "dead",
                            "narration": (
                                "Corbitt rises, but his own dagger destroys the "
                                "undead body after the final struggle."),
                        },
                    },
                    "dead": {"interactable": False},
                },
            },
            "house_keys": {
                "type": "item", "name": "keys to the Corbitt House",
                "scene": "knott_briefing", "initial_state": "obtained",
                "portable": True,
            },
            "globe_clipping": {
                "type": "clue", "name": "unpublished 1918 clipping",
                "scene": "boston_globe", "initial_state": "hidden",
                "portable": False,
                "states": {
                    "hidden": {
                        "triggers": ["search the clipping files"],
                        "on_trigger": {
                            "to_state": "found",
                            "narration": (
                                "The clipping records violent accidents and the "
                                "Macario family's abrupt flight."),
                        },
                    },
                    "found": {},
                },
            },
            "library_records": {
                "type": "clue", "name": "successive Corbitt house records",
                "scene": "central_library", "initial_state": "hidden",
                "portable": False,
                "states": {
                    "hidden": {
                        "triggers": ["research the first library record"],
                        "on_trigger": {
                            "to_state": "merchant_sale",
                            "narration": "The builder sold the house to Walter Corbitt in 1835.",
                            "events": [{
                                "type": "npc_name_disclosed", "entity_id": "corbitt",
                            }],
                        },
                    },
                    "merchant_sale": {
                        "triggers": ["continue the library research"],
                        "on_trigger": {"to_state": "neighbor_suit",
                                       "narration": "Neighbors sued to force Corbitt from the area in 1852."},
                    },
                    "neighbor_suit": {
                        "triggers": ["finish the library research"],
                        "on_trigger": {"to_state": "burial_will",
                                       "narration": "Corbitt's obituary records his demand for basement burial."},
                    },
                    "burial_will": {},
                },
            },
            "chapel_record": {
                "type": "clue", "name": "Corbitt estate and chapel record",
                "scene": "hall_of_records", "initial_state": "hidden",
                "portable": False,
                "states": {
                    "hidden": {
                        "triggers": ["research Corbitt's will at the hall"],
                        "on_trigger": {
                            "to_state": "found",
                            "narration": (
                                "The will names Reverend Michael Thomas of the "
                                "Chapel of Contemplation as executor."),
                        },
                    },
                    "found": {},
                },
            },
            "raid_file": {
                "type": "clue", "name": "suppressed Chapel raid file",
                "scene": "police_records", "initial_state": "hidden",
                "portable": False,
                "states": {
                    "hidden": {
                        "triggers": ["obtain the chapel raid file"],
                        "on_trigger": {
                            "to_state": "found",
                            "narration": (
                                "The restricted file describes the deadly 1912 "
                                "raid, cover-up, and Thomas's later escape."),
                        },
                    },
                    "found": {},
                },
            },
            "macario_lead": {
                "type": "clue", "name": "Roxbury Sanitarium lead",
                "scene": "neighborhood", "initial_state": "hidden",
                "portable": False,
            },
            "chapel_lead": {
                "type": "clue", "name": "location of the Chapel ruins",
                "scene": "neighborhood", "initial_state": "hidden",
                "portable": False,
            },
            "chapel_cabinet": {
                "type": "container", "name": "moldering chapel cabinet",
                "scene": "chapel_basement", "initial_state": "closed",
                "portable": False,
                "states": {
                    "closed": {
                        "triggers": ["search the chapel cabinet"],
                        "on_trigger": {
                            "to_state": "opened",
                            "narration": (
                                "The cabinet reveals a cult journal and a "
                                "worm-eaten Latin tome."),
                            "events": discover("cult_journal", "liber_ivonis"),
                        },
                    },
                    "opened": {},
                },
            },
            "cult_journal": {
                "type": "clue", "name": "Chapel cult journal",
                "scene": "chapel_basement", "initial_state": "hidden",
                "portable": False,
            },
            "liber_ivonis": {
                "type": "item", "name": "Liber Ivonis",
                "aliases": ["worm-eaten Latin tome"],
                "scene": "chapel_basement", "initial_state": "hidden",
                "portable": True,
            },
            "ground_cupboard": {
                "type": "container", "name": "boarded storage-room cupboard",
                "scene": "ground_floor", "initial_state": "closed",
                "portable": False,
                "states": {
                    "closed": {
                        "triggers": ["search the boarded cupboard"],
                        "on_trigger": {
                            "to_state": "opened",
                            "narration": "The opened cupboard contains Corbitt's three bound diaries.",
                            "events": discover("corbitt_diaries"),
                        },
                    },
                    "opened": {},
                },
            },
            "corbitt_diaries": {
                "type": "clue", "name": "Corbitt diaries",
                "aliases": ["three bound diaries"],
                "scene": "ground_floor", "initial_state": "hidden",
                "portable": False,
            },
            "animated_bed": {
                "type": "object", "name": "bare spare-room bed",
                "scene": "upper_floor", "initial_state": "present",
                "portable": False,
            },
            "basement_clutter": {
                "type": "container", "name": "basement clutter",
                "scene": "basement", "initial_state": "present",
                "portable": False,
                "states": {
                    "present": {
                        "triggers": ["search the basement clutter"],
                        "on_trigger": {
                            "to_state": "searched",
                            "narration": "The search reveals an ornate knife crusted with dried blood.",
                            "events": discover("magic_dagger"),
                        },
                    },
                    "searched": {},
                },
            },
            "magic_dagger": {
                "type": "item", "name": "Corbitt's magic dagger",
                "aliases": ["floating knife", "ornate knife", "his own dagger"],
                "scene": "basement", "initial_state": "hidden",
                "portable": True,
            },
            "hollow_boards": {
                "type": "object", "name": "hollow basement wall boards",
                "scene": "basement", "initial_state": "intact",
                "portable": False,
                "states": {
                    "intact": {
                        "triggers": ["remove the hollow boards"],
                        "on_trigger": {"to_state": "open",
                                       "narration": "The removed boards expose a foul crawlspace."},
                    },
                    "open": {},
                },
            },
            "rat_pack": npc("rat pack", "crawlspace", "wall-nesting rats"),
        },
        "narrative_scopes": [
            {"id": "physical", "kind": "physical", "navigable": True},
            {"id": "insanity_delusions", "kind": "delusion",
             "parent_scope": "physical", "navigable": False},
            {"id": "future_extension", "kind": "future_hook",
             "parent_scope": "physical", "navigable": False},
        ],
        "embedded_settings": [
            {
                "scope": {"id": "insanity_delusions", "kind": "delusion",
                          "navigable": False},
                "scenes": [{
                    "id": "ancestry_photograph", "name": "Corbitt Ancestry Photograph",
                    "scope_id": "insanity_delusions", "navigable": False,
                }],
                "entities": [],
            },
            {
                "scope": {"id": "future_extension", "kind": "future_hook",
                          "navigable": False},
                "scenes": [{
                    "id": "future_conspiracy", "name": "Future Chapel Conspiracy",
                    "scope_id": "future_extension", "navigable": False,
                }],
                "entities": [],
            },
        ],
    }


def generate() -> None:
    write_json("haunting_world.json", build_world())

    victory = [
        {"input": "开始游戏",
         "response": "Mr. Knott offers twenty dollars per day and hands over the house keys.",
         "scene": "knott_briefing", "contains": ["twenty dollars"],
         "covers": ["commission", "initial_choice"]},
        movement_step("We research at the Boston Globe.",
                      "The investigators enter the newspaper office and meet its pompous editor.",
                      "boston_globe", ["globe_gate"], ["pompous editor"]),
        {"input": "I persuade Arty Wilmot to grant access to the morgue.",
         "response": "Arty reluctantly grants access to the clipping morgue.",
         "select_npc": "arty_wilmot", "contains": ["grants access"], "covers": []},
        {"input": "We search the clipping files.",
         "response": "The clipping records violent accidents and the Macario family's abrupt flight.",
         "contains": ["violent accidents"], "covers": ["globe_clipping"],
         "expect": {"entity_states": {"globe_clipping": "found"}}},
        movement_step("We return to Mr. Knott.", "The investigators return to the briefing point.",
                      "knott_briefing", [], ["briefing"]),
        movement_step("We continue research at the Central Library.",
                      "The library's historical collections await a methodical search.",
                      "central_library", ["library_research"], ["historical"]),
        {"input": "We research the first library record.",
         "response": "The builder sold the house to Walter Corbitt in 1835.",
         "contains": ["1835"], "covers": ["merchant_sale"]},
        {"input": "We continue the library research.",
         "response": "Neighbors sued to force Corbitt from the area in 1852.",
         "contains": ["1852"], "covers": ["neighbor_suit"]},
        {"input": "We finish the library research.",
         "response": "Corbitt's obituary records his demand for burial in his basement.",
         "contains": ["burial"], "covers": ["burial_will"],
         "expect": {"entity_states": {"library_records": "burial_will"}}},
        movement_step("We return to Mr. Knott.", "The investigators leave the library.",
                      "knott_briefing", [], ["leave"]),
        movement_step("We visit the Hall of Records.",
                      "Civil and church registers are available for research.",
                      "hall_of_records", ["hall_records"], ["registers"]),
        {"input": "We research Corbitt's will at the hall.",
         "response": "The will names Reverend Michael Thomas of the Chapel as executor.",
         "contains": ["Michael Thomas"], "covers": ["chapel_executor"],
         "expect": {"entity_states": {"chapel_record": "found"}}},
        movement_step("We follow the restricted record trail.",
                      "The investigators obtain access to the higher-court and police archive.",
                      "police_records", ["higher_records_gate"], ["police archive"]),
        {"input": "We obtain the chapel raid file.",
         "response": "The file describes the deadly 1912 raid, cover-up, and Thomas's escape.",
         "contains": ["1912"], "covers": ["chapel_raid"]},
        movement_step("We return to the Hall of Records.", "The investigators leave the restricted archive.",
                      "hall_of_records", [], ["archive"]),
        movement_step("We return to Mr. Knott.", "The investigators regroup before further inquiries.",
                      "knott_briefing", [], ["regroup"]),
        movement_step("We go to the Corbitt House neighborhood.",
                      "The lone old house stands near Mr. Dooley's newspaper stall.",
                      "neighborhood", ["neighborhood_inquiry"], ["newspaper stall"]),
        {"input": "I ask Mr. Dooley about the house and chapel.",
         "response": "Dooley describes the Macarios and points out the ruined Chapel of Contemplation.",
         "select_npc": "mr_dooley", "contains": ["Macarios", "Chapel"],
         "covers": ["dooley_interview", "macario_history"],
         "expect": {"world_event_types_contains": ["entity_discovered"]}},
        movement_step("We visit Roxbury Sanitarium.",
                      "Vittorio clutches a bible while Gabriela waits nearby.",
                      "sanitarium", ["sanitarium_visit"], ["bible"]),
        {"input": "I ask Vittorio what can defeat the presence.",
         "response": "Vittorio insists that the devil is worsted by his own weapon.",
         "select_npc": "vittorio", "contains": ["own weapon"],
         "covers": ["vittorio_weapon"]},
        {"input": "I ask Gabriela what happened inside the house.",
         "response": "Gabriela describes an evil presence and objects flying through rooms.",
         "select_npc": "gabriela", "contains": ["evil presence"],
         "covers": ["gabriela_haunting"]},
        movement_step("We return to the neighborhood.",
                      "The investigators return from the sanitarium to Dooley's block.",
                      "neighborhood", [], ["Dooley"]),
        movement_step("We visit the Chapel ruins.",
                      "The ruined Chapel bears a freshly painted triple-Y staring eye.",
                      "chapel", ["chapel_ruins", "chapel_symbol"], ["triple-Y"]),
        movement_step("We explore the weak floorboards and fall into the sealed basement.",
                      "The floor collapses above two robed skeletons and an old cabinet.",
                      "chapel_basement", ["chapel_collapse"], ["skeletons"]),
        {"input": "We search the chapel cabinet.",
         "response": "The cabinet reveals a cult journal and a worm-eaten Latin tome.",
         "select_object": "chapel_cabinet", "contains": ["cult journal", "Latin tome"],
         "covers": ["cult_journal", "liber_ivonis"],
         "expect": {"world_event_types_contains": ["entity_discovered"]}},
        {"input": "I read the Chapel cult journal.",
         "response": "The journal confirms that Corbitt was buried beneath his own house.",
         "select_object": "cult_journal", "contains": ["buried"], "covers": []},
        {"input": "I take the Liber Ivonis.",
         "response": "You carry the worm-eaten Liber Ivonis out of the sealed room.",
         "select_object": "liber_ivonis", "contains": ["Liber Ivonis"],
         "expect": {"inventory_contains": ["liber_ivonis"],
                    "object_locations": {"liber_ivonis": INVENTORY}}},
        movement_step("We continue to the Corbitt House.",
                      "The investigators arrive before the shadowed brick house.",
                      "house_exterior", ["house_exterior"], ["brick house"]),
        movement_step("We enter the house using Knott's keys.",
                      "The keys open a ground floor filled with Catholic artifacts and spoiled soup.",
                      "ground_floor", ["ground_floor_choice"], ["Catholic"]),
        {"input": "We search the boarded cupboard.",
         "response": "The opened cupboard contains Corbitt's three bound diaries.",
         "select_object": "ground_cupboard", "contains": ["diaries"],
         "covers": ["corbitt_diaries"]},
        {"input": "I read the Corbitt diaries.",
         "response": "The diaries describe Corbitt's occult experiments and summoning spell.",
         "select_object": "corbitt_diaries", "contains": ["occult experiments"],
         "covers": []},
        movement_step("We go upstairs.",
                      "The investigators search the bedrooms and Corbitt's former spare room.",
                      "upper_floor", ["upper_floor"], ["spare room"]),
        {"input": "We investigate the thumping, blood, and scratching.",
         "response": "The manifestations draw attention toward the spare-room window.",
         "contains": ["spare-room"], "covers": ["haunting_manifestations"]},
        {"input": "I inspect the rattling window beside the bed.",
         "response": "The animated bed lunges toward the window, forcing a desperate dodge.",
         "select_object": "animated_bed", "contains": ["animated bed"],
         "covers": ["bed_attack"]},
        movement_step("We go downstairs.", "The investigators return to the ground floor.",
                      "ground_floor", [], ["ground floor"]),
        movement_step("We descend to the basement.",
                      "The moving stairs lead into dark storage lined with hollow boards.",
                      "basement", ["basement_descent"], ["hollow boards"]),
        {"input": "We search the basement clutter.",
         "response": "The search reveals an ornate knife crusted with dried blood.",
         "select_object": "basement_clutter", "contains": ["ornate knife"],
         "covers": ["knife_search"]},
        {"input": "I take Corbitt's magic dagger.",
         "response": "You seize the knife as it twists and tries to tear itself free.",
         "select_object": "magic_dagger", "contains": ["twists"],
         "covers": ["floating_knife"],
         "expect": {"inventory_contains": ["liber_ivonis", "magic_dagger"],
                    "object_locations": {"magic_dagger": INVENTORY}}},
        movement_step("We remove the hollow boards and enter the opened crawlspace.",
                      "The boards come away, exposing a foul crawlspace full of rats.",
                      "crawlspace", ["hidden_wall", "rat_pack"], ["rats"]),
        movement_step("We give the rats room to escape and enter Corbitt's hiding place.",
                      "The rats flee, revealing a wizened body on a pallet beyond them.",
                      "hiding_place", ["hiding_place"], ["wizened body"]),
        {"input": "I ask the motionless Walter Corbitt to explain himself.",
         "response": "The dormant body cannot be interviewed.",
         "select_npc": "corbitt", "code_response": True,
         "contains": ["没有可交谈"], "covers": ["dormant_corbitt_block"]},
        {"input": "We confront Walter Corbitt using his own dagger.",
         "response": "Corbitt rises, but his own dagger destroys the undead body after the final struggle.",
         "contains": ["destroys"],
         "covers": ["corbitt_rises", "final_strategy", "destroy_corbitt"],
         "expect": {"entity_states": {"corbitt": "dead"},
                    "inventory_contains": ["liber_ivonis", "magic_dagger"]}},
        {"input": "I question Walter Corbitt after his destruction.",
         "response": "The destroyed undead body cannot answer.",
         "select_npc": "corbitt", "code_response": True,
         "contains": ["没有可交谈"], "covers": ["dead_corbitt_block"]},
        movement_step("We conclude the solved case.",
                      "Knott pays the fee and bonus; the survivors retain the Liber Ivonis.",
                      "conclusion_victory", ["rewards"], ["bonus"]),
    ]
    write_route("haunting_research_victory", "the_haunting", "haunting", victory)

    false_report = [
        {"input": "开始游戏", "response": "Mr. Knott gives the investigators the house keys.",
         "scene": "knott_briefing", "contains": ["keys"], "covers": []},
        {"input": "I take Corbitt's magic dagger before visiting the house.",
         "response": "The undiscovered dagger cannot be taken.",
         "select_object": "magic_dagger", "code_response": True,
         "contains": ["没有发现"], "covers": ["hidden_dagger_block"]},
        {"input": "I teleport directly to Corbitt's Hiding Place.",
         "response": "The hidden room cannot be reached from the briefing.",
         "code_response": True, "contains": ["不能直接越过"],
         "covers": ["hiding_place_jump_block"]},
        movement_step("We go straight to the Corbitt House.",
                      "The investigators arrive at the bolted brick house.",
                      "house_exterior", [], ["brick house"]),
        movement_step("We enter the house.", "The keys admit the group to the ground floor.",
                      "ground_floor", [], ["ground floor"]),
        movement_step("We descend to the basement without researching.",
                      "The investigators descend the moving basement stairs.",
                      "basement", [], ["moving"]),
        movement_step("We remove the hollow boards and enter the opened crawlspace.",
                      "The wall opens into the rat-filled crawlspace.",
                      "crawlspace", [], ["rat"]),
        movement_step("We force past the rats and enter Corbitt's hiding place.",
                      "The investigators find the wizened body beyond the rats.",
                      "hiding_place", [], ["wizened"]),
        {"input": "We confront Walter Corbitt using his own dagger.",
         "response": "The required weapon is not in the investigators' possession.",
         "code_response": True, "contains": ["尚未满足"],
         "covers": ["missing_dagger_requirement"],
         "expect": {"entity_states": {"corbitt": "dormant"}}},
        movement_step("We return to the crawlspace.", "The investigators retreat from the body.",
                      "crawlspace", [], ["retreat"]),
        movement_step("We return to basement storage.", "The group leaves the crawlspace.",
                      "basement", [], ["leaves"]),
        movement_step("We go upstairs.", "The investigators climb back to the ground floor.",
                      "ground_floor", [], ["ground floor"]),
        movement_step("We leave the house.", "The group exits the Corbitt House.",
                      "house_exterior", [], ["exits"]),
        movement_step("We return to Mr. Knott.", "The investigators bring Knott an incomplete report.",
                      "knott_briefing", [], ["report"]),
        {"input": "We file a false clean report: nothing is wrong.",
         "response": "Knott accepts the false report, visits the house, and is later stabbed to death in the basement.",
         "contains": ["stabbed to death"],
         "covers": ["leave_unresolved", "knott_death"],
         "expect": {"entity_states": {"mr_knott": "dead"}}},
        {"input": "I ask Mr. Knott to reconsider after his death.",
         "response": "Knott is dead and cannot answer.",
         "select_npc": "mr_knott", "code_response": True,
         "contains": ["没有可交谈"], "covers": ["dead_knott_block"]},
    ]
    write_route("haunting_false_report", "the_haunting", "haunting", false_report)

    scope_loss = [
        {"input": "开始游戏", "response": "Mr. Knott gives the investigators the house keys.",
         "scene": "knott_briefing", "contains": ["keys"], "covers": []},
        {"input": "I go directly to the Ruined Chapel of Contemplation.",
         "response": "The Chapel's location has not been established from here.",
         "code_response": True, "contains": ["不能直接越过"],
         "covers": ["chapel_jump_block"]},
        {"input": "I enter the Corbitt Ancestry Photograph from the madness table.",
         "response": "An insanity delusion is not a physical destination.",
         "code_response": True, "contains": ["不能直接越过"],
         "covers": ["delusion_scope_block"]},
        {"input": "I ask Gabriela Macario what is in the basement.",
         "response": "Gabriela is not present at the briefing.",
         "select_npc": "gabriela", "code_response": True,
         "contains": ["无法交谈"], "covers": ["absent_gabriela_block"]},
        movement_step("We go straight to the Corbitt House.",
                      "The investigators reach the shadowed house.",
                      "house_exterior", [], ["house"]),
        movement_step("We enter the house.", "The investigators enter the ground floor.",
                      "ground_floor", [], ["ground floor"]),
        movement_step("We go upstairs.", "The spare room waits beyond the Macario bedrooms.",
                      "upper_floor", [], ["spare room"]),
        {"input": "I remain beside the window when the bed attacks.",
         "response": "The animated bed hurls the investigator through the window.",
         "select_object": "animated_bed", "contains": ["through the window"],
         "covers": ["bed_attack"]},
        movement_step("The investigation ends in death or madness.",
                      "The failed investigation ends in death, madness, and flight.",
                      "conclusion_loss", ["investigator_loss"], ["death"]),
    ]
    write_route("haunting_scope_and_loss", "the_haunting", "haunting", scope_loss)

    required = sorted({
        point for route in (victory, false_report, scope_loss)
        for turn in route for point in turn.get("covers", [])
    })
    write_json("haunting_coverage.json", {
        "module": "the_haunting", "required": required,
    })


if __name__ == "__main__":
    generate()

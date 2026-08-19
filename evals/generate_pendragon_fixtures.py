#!/usr/bin/env python3
"""Generate source-grounded runtime cases for The Sword Tournament."""

from __future__ import annotations

from generate_exhaustive_fixtures import movement_step, write_json, write_route


def scene(name: str, desc: str, exits: dict[str, str] | None = None,
          entry_events: list[dict] | None = None) -> dict:
    row = {
        "name": name, "desc": desc, "source_text": desc,
        "exits": exits or {},
    }
    if entry_events:
        row["entry_events"] = entry_events
    return row


def move(entity_id: str, scene_id: str) -> dict:
    return {
        "type": "entity_moved", "entity_id": entity_id,
        "location": {"kind": "scene", "id": scene_id},
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


def build_world() -> dict:
    tournament_entries = [
        move("arthur", "tournament"),
        move("ector", "tournament"),
    ]
    revelation_entries = [
        move(entity_id, "revelation") for entity_id in (
            "arthur", "ector", "merlin", "leodegrance", "lot", "uriens",
            "sword_stone", "anvil_stone",
        )
    ]
    return {
        "name": "eval_pendragon_sword_tournament",
        "rule_system": "pendragon",
        "ruleset": "pendragon",
        "dice_system": "d20",
        "automatic_check_adapter": True,
        "starting_scene": "camp",
        "opening": (
            "In Year 510, landless knights camp outside Londinium before the "
            "New Year's Tournament that is meant to choose a High King."
        ),
        "scenes": {
            "camp": scene(
                "Tournament Camp", "Knights lodge in a tent town outside Londinium.",
                {"sightsee in Londinium": "streets",
                 "attend the tournament": "tournament"}),
            "streets": scene(
                "Londinium Streets", "The knights explore the crowded winter city.",
                {"visit St. Paul's": "st_pauls_before",
                 "visit an ale house": "alehouse",
                 "return to camp": "camp"}),
            "st_pauls_before": scene(
                "St. Paul's Courtyard Before the Tournament",
                "A sword stands through an iron anvil atop an inscribed stone.",
                {"return to the streets": "streets"}),
            "alehouse": scene(
                "Londinium Ale House",
                "A young squire stops a beggar cutting a knight's purse while Sir Ector watches.",
                {"return to the streets": "streets"}),
            "tournament": scene(
                "Tournament Melee Field",
                "King Leodegrance leads the Blue Team while Gorre and Lothian conrois assemble.",
                {"follow the great commotion": "fleet_street"},
                tournament_entries),
            "fleet_street": scene(
                "Fleet Street and Ludgate",
                "The melee ends and mounted knights race through the crowded gate.",
                {"reach St. Paul's": "revelation"}),
            "revelation": scene(
                "St. Paul's Cathedral Revelation",
                "Arthur demonstrates the sword's claim before the assembled kings.",
                {"hail Arthur": "conclusion"},
                revelation_entries),
            "conclusion": scene(
                "King Arthur Acclaimed",
                "The crowd kneels, Lot retreats, and the knights tally Glory."),
        },
        "entities": {
            "arthur": npc("Arthur", "alehouse", "young squire"),
            "ector": npc("Sir Ector", "alehouse", "aging hedge knight"),
            "merlin": npc("Merlin", "streets", "hooded old enchanter"),
            "leodegrance": npc(
                "King Leodegrance of Cameliard", "tournament",
                "aging king of the Blue Team"),
            "lot": npc(
                "King Lot of Lothian", "tournament", "powerful northern king",
                {
                    "present": {
                        "triggers": [
                            "intercept King Lot", "block King Lot",
                            "fight King Lot at the cathedral",
                        ],
                        "on_trigger": {
                            "to_state": "retreated",
                            "narration": (
                                "After the clash and Arthur's intervention, "
                                "King Lot rejects the claim and retreats."
                            ),
                        },
                    },
                    "retreated": {"interactable": False},
                }),
            "uriens": npc(
                "King Uriens of Gorre", "tournament", "king of Gorre"),
            "dubricus": npc(
                "Archbishop Dubricus", "revelation", "solemn archbishop"),
            "sword_stone": {
                "type": "object", "name": "Sword in the Stone",
                "aliases": ["the sword", "the blade"],
                "scene": "st_pauls_before", "initial_state": "fixed",
                "portable": False,
                "states": {
                    "fixed": {
                        "triggers": [
                            "try to pull the sword", "attempt to draw the sword",
                        ],
                        "on_trigger": {
                            "to_state": "fixed",
                            "narration": "The sword does not move for the Player-knight.",
                        },
                    },
                },
            },
            "anvil_stone": {
                "type": "object", "name": "iron anvil and inscribed stone",
                "aliases": ["inscription", "the Stone"],
                "scene": "st_pauls_before", "initial_state": "present",
                "portable": False,
            },
            "forgotten_sword": {
                "type": "item", "name": "forgotten tournament sword",
                "aliases": ["sword back at the inn"],
                "scene": "alehouse", "initial_state": "hidden",
                "portable": True,
            },
        },
        "narrative_scopes": [
            {"id": "physical", "kind": "physical", "navigable": True},
            {"id": "knight_biographies", "kind": "backstory",
             "parent_scope": "physical", "navigable": False},
            {"id": "londinium_legend", "kind": "legend",
             "parent_scope": "physical", "navigable": False},
        ],
        "embedded_settings": [
            {
                "scope": {"id": "knight_biographies", "kind": "backstory",
                          "navigable": False},
                "scenes": [
                    {"id": "broceliande", "name": "Forest of Broceliande",
                     "scope_id": "knight_biographies", "navigable": False},
                    {"id": "cambria", "name": "Hills of Cambria",
                     "scope_id": "knight_biographies", "navigable": False},
                ],
                "entities": [],
            },
            {
                "scope": {"id": "londinium_legend", "kind": "legend",
                          "navigable": False},
                "scenes": [
                    {"id": "ancient_troy", "name": "Ancient Troy",
                     "scope_id": "londinium_legend", "navigable": False},
                ],
                "entities": [],
            },
        ],
    }


def generate() -> None:
    write_json("pendragon_sword_tournament_world.json", build_world())

    loyal_route = [
        {"input": "开始游戏",
         "response": "The knights wait in the tournament camp outside Londinium.",
         "scene": "camp", "contains": ["tournament camp"],
         "covers": ["londinium_arrival"]},
        movement_step(
            "We sightsee in Londinium.",
            "The knights enter the crowded streets of Londinium.",
            "streets", ["city_exploration", "londinium_event"], ["Londinium"]),
        movement_step(
            "We visit St. Paul's courtyard.",
            "A sword pierces an iron anvil atop an inscribed stone.",
            "st_pauls_before", ["sword_stone"], ["inscribed stone"]),
        {"input": "I inspect the inscription.",
         "response": "It promises Britain to whoever pulls the sword from the stone.",
         "select_object": "anvil_stone", "contains": ["whoever pulls"],
         "covers": ["inscription"]},
        {"input": "I try to pull the sword.",
         "response": "The sword does not move for the Player-knight.",
         "contains": ["does not move"], "covers": ["failed_sword_attempt"],
         "expect": {"entity_states": {"sword_stone": "fixed"}}},
        movement_step(
            "We return to the streets.", "The knights resume their city excursion.",
            "streets", [], ["city"]),
        movement_step(
            "We visit an ale house.",
            "A young squire catches a beggar cutting a knight's purse.",
            "alehouse", ["alehouse_theft"], ["young squire"]),
        {"input": "I ask the young squire his name.",
         "response": "The helpful young squire answers, \"I am Arthur.\"",
         "select_npc": "arthur", "contains": ["Arthur"],
         "covers": ["arthur_name"]},
        movement_step(
            "We return through the streets.", "The knights leave the ale house.",
            "streets", [], ["leave"]),
        movement_step(
            "We return to the tournament camp.",
            "The knights prepare for the next morning's melee.",
            "camp", [], ["prepare"]),
        movement_step(
            "We attend the tournament.",
            "The Blue Team assembles under King Leodegrance as Arthur searches for a forgotten sword.",
            "tournament", ["tournament_morning", "forgotten_sword"],
            ["forgotten sword"]),
        {"input": "We choose Honor as our conroi's Passion.",
         "response": "Honor sets the conroi's Morale for the melee.",
         "contains": ["Morale"], "covers": ["morale_passion"]},
        {"input": "Our Battle rolls leave us facing the Knights of Gorre.",
         "response": "The Blue Team charges the Knights of Gorre.",
         "contains": ["Knights of Gorre"],
         "covers": ["opponent_choice", "gorre_encounter", "melee_resolution"]},
        movement_step(
            "We follow the shout that the sword was drawn.",
            "The melee ends and Leodegrance leads a race along Fleet Street.",
            "fleet_street", ["great_commotion", "race_cathedral"], ["melee ends"]),
        movement_step(
            "We ride at the king's side and reach St. Paul's first.",
            "The front-row knights see Arthur draw the sword while Lot and Uriens fail.",
            "revelation", ["front_row", "arthur_draws", "rival_tests",
                           "leodegrance_claim"], ["Arthur draw"]),
        {"input": "We intercept King Lot before he reaches Arthur.",
         "response": "After the clash and Arthur's intervention, King Lot rejects the claim and retreats.",
         "select_npc": "lot", "contains": ["retreats"],
         "covers": ["intervention_choice", "intercept_lot", "arthur_kills_guard",
                    "lot_retreats"],
         "expect": {"entity_states": {"lot": "retreated"},
                    "object_locations": {
                        "arthur": {"kind": "scene", "id": "revelation"},
                        "sword_stone": {"kind": "scene", "id": "revelation"}
                    },
                    "world_event_types_contains": ["entity_moved"]}},
        {"input": "I ask King Lot to reconsider.",
         "response": "Lot has retreated and cannot be addressed here.",
         "select_npc": "lot", "code_response": True,
         "contains": ["交谈"], "covers": ["retreated_lot_block"]},
        movement_step(
            "We are among the first to hail Arthur as king.",
            "The knights swear early loyalty; the crowd hails King Arthur and Glory is tallied.",
            "conclusion", ["loyalty_choice", "early_oath", "arthur_acclaimed",
                           "glory_awards"], ["King Arthur"]),
    ]
    write_route(
        "pendragon_loyal_intervention", "pendragon_sword_tournament",
        "pendragon_sword_tournament", loyal_route)

    alternate_route = [
        {"input": "开始游戏",
         "response": "The knights wait in the tournament camp outside Londinium.",
         "scene": "camp", "contains": ["tournament camp"], "covers": []},
        movement_step(
            "We skip sightseeing and attend the tournament.",
            "The Blue Team assembles beneath Leodegrance's banner.",
            "tournament", ["tournament_morning"], ["Blue Team"]),
        {"input": "We invoke Fealty to establish Morale.",
         "response": "The shared Passion strengthens the conroi's Morale.",
         "contains": ["Morale"], "covers": ["morale_passion"]},
        {"input": "A critical Battle result pairs our conroi against King Lot directly.",
         "response": "The conroi takes the rare opportunity to challenge King Lot.",
         "select_npc": "lot", "contains": ["King Lot"],
         "covers": ["opponent_choice", "lot_opportunity", "melee_resolution"]},
        movement_step(
            "We follow the commotion but fall behind at Ludgate.",
            "The late riders struggle through the press toward St. Paul's.",
            "fleet_street", ["great_commotion", "race_cathedral"], ["press"]),
        movement_step(
            "We reach the cathedral behind the crowd.",
            "From behind the crowd, the knights see Arthur prove the claim and others block Lot.",
            "revelation", ["late_arrival", "arthur_draws", "rival_tests",
                           "leodegrance_claim", "intervention_choice",
                           "decline_intervention", "arthur_kills_guard"],
            ["behind the crowd"]),
        {"input": "We wait until the crowd kneels before following.",
         "response": "The knights follow the crowd rather than claiming the earliest oath.",
         "contains": ["follow the crowd"],
         "covers": ["loyalty_choice", "follow_crowd"]},
        {
            **movement_step(
                "We hail Arthur after Lot rides away.",
                "Lot retreats and the assembled crowd hails King Arthur.",
                "conclusion",
                ["lot_retreats", "arthur_acclaimed", "glory_awards"],
                ["King Arthur"]),
            "set_entity_states": {"lot": "retreated"},
            "expect": {"entity_states": {"lot": "retreated"}},
        },
    ]
    write_route(
        "pendragon_late_lot_opportunity", "pendragon_sword_tournament",
        "pendragon_sword_tournament", alternate_route)

    destructive_route = [
        {"input": "开始游戏",
         "response": "The knights wait in the tournament camp outside Londinium.",
         "scene": "camp", "contains": ["tournament camp"], "covers": []},
        {"input": "I ask Arthur what the sword prophecy means.",
         "response": "Arthur is not present in the tournament camp.",
         "select_npc": "arthur", "code_response": True,
         "contains": ["交谈"], "covers": ["absent_arthur_block"]},
        {"input": "I take the forgotten tournament sword from the inn.",
         "response": "The forgotten sword has not been revealed.",
         "select_object": "forgotten_sword", "code_response": True,
         "contains": ["没有发现"], "covers": ["hidden_forgotten_sword_block"]},
        {"input": "I teleport directly to St. Paul's Cathedral Revelation.",
         "response": "The final gathering cannot be reached from camp.",
         "code_response": True, "contains": ["不能直接越过"],
         "covers": ["revelation_jump_block"]},
        {"input": "I enter the Forest of Broceliande from a knight's biography.",
         "response": "The biography is not a physical route from Londinium.",
         "code_response": True, "contains": ["不能直接越过"],
         "covers": ["biography_scope_block"]},
        movement_step(
            "We sightsee in Londinium.", "The knights enter Londinium.",
            "streets", [], ["Londinium"]),
        movement_step(
            "We visit St. Paul's courtyard.",
            "The immovable sword stands in its anvil and stone.",
            "st_pauls_before", ["sword_stone"], ["immovable"]),
        {"input": "I take the Sword in the Stone with me.",
         "response": "The fixed sword cannot be carried away.",
         "select_object": "sword_stone", "code_response": True,
         "contains": ["无法被直接带走"], "covers": ["sword_take_block"],
         "expect": {"object_locations": {
             "sword_stone": {"kind": "scene", "id": "st_pauls_before"}}}},
        movement_step(
            "We return to the streets.", "The knights leave the cathedral courtyard.",
            "streets", [], ["leave"]),
        movement_step(
            "We visit the ale house.", "The young squire and Sir Ector are lodging here.",
            "alehouse", ["alehouse_theft"], ["young squire"]),
        {"input": "I ask the young squire his name and what happened.",
         "response": "The young squire says, \"I am Arthur,\" then explains the purse cutting.",
         "select_npc": "arthur", "contains": ["Arthur", "purse"],
         "covers": ["arthur_name"]},
        movement_step(
            "We return to the streets.", "The knights leave Arthur at the inn.",
            "streets", [], ["leave Arthur"]),
        movement_step(
            "We return to camp.", "The knights prepare for the melee.",
            "camp", [], ["melee"]),
        {"input": "I continue talking to the young squire from the inn.",
         "response": "The squire is still at the inn and cannot answer from camp.",
         "select_npc": "arthur", "code_response": True,
         "contains": ["交谈"], "covers": ["stale_arthur_block"]},
        movement_step(
            "We attend the tournament.",
            "Arthur and Sir Ector arrive as the Blue Team assembles.",
            "tournament", ["tournament_morning", "forgotten_sword"], ["Arthur"]),
        {"input": "I ask Arthur about the forgotten sword now.",
         "response": "Arthur admits he must recover his knight's forgotten sword.",
         "select_npc": "arthur", "contains": ["forgotten sword"],
         "covers": ["recurring_arthur_at_tournament"],
         "expect": {"object_locations": {
             "arthur": {"kind": "scene", "id": "tournament"}}}},
        movement_step(
            "We follow the commotion.", "The knights race toward St. Paul's.",
            "fleet_street", ["great_commotion"], ["St. Paul's"]),
        movement_step(
            "We reach the cathedral.",
            "Arthur, the rival kings, and the same Sword in the Stone are in the courtyard.",
            "revelation", ["arthur_draws", "rival_tests"], ["same Sword"]),
        {"input": "I ask Arthur whether this is the same sword.",
         "response": "Arthur stands beside the same sword and answers from the courtyard.",
         "select_npc": "arthur", "contains": ["same sword"],
         "covers": ["recurring_arthur_at_revelation"],
         "expect": {"object_locations": {
             "arthur": {"kind": "scene", "id": "revelation"},
             "sword_stone": {"kind": "scene", "id": "revelation"}}}},
    ]
    write_route(
        "pendragon_identity_and_scope", "pendragon_sword_tournament",
        "pendragon_sword_tournament", destructive_route)

    required = sorted({
        point for route in (loyal_route, alternate_route, destructive_route)
        for turn in route for point in turn.get("covers", [])
    })
    write_json("pendragon_sword_tournament_coverage.json", {
        "module": "pendragon_sword_tournament",
        "required": required,
    })


if __name__ == "__main__":
    generate()

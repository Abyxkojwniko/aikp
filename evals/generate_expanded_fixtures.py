#!/usr/bin/env python3
"""Generate adversarial no-key fixtures for five official free adventures."""

from __future__ import annotations

from generate_exhaustive_fixtures import movement_step, write_json, write_route


def scene(name: str, desc: str, exits: dict[str, str] | None = None) -> dict:
    return {
        "name": name,
        "desc": desc,
        "source_text": desc,
        "exits": exits or {},
    }


def npc(name: str, at: str, label: str,
        all_scenes: list[str] | None = None) -> dict:
    value = {
        "type": "npc",
        "name": name,
        "scene": at,
        "initial_state": "present",
        "public_label": label,
    }
    if all_scenes:
        value["all_scenes"] = all_scenes
    return value


def generate_alone() -> None:
    scenes = {
        "bus_stop": scene(
            "Osborn's Drug Store Bus Stop",
            "The investigator waits alone for the coach to Arkham.",
            {"board the coach": "coach"}),
        "coach": scene(
            "Coach to Arkham",
            "Silas drives toward Arkham before an unexpected stop.",
            {"continue the journey": "emberhead"}),
        "emberhead": scene(
            "Emberhead Arrival",
            "The broken coach leaves the investigator stranded in Emberhead.",
            {"Ledbetter house": "ledbetter", "village center": "village"}),
        "ledbetter": scene(
            "Ledbetter House",
            "May offers lodging while Ruth privately warns about the festival.",
            {"village center": "village"}),
        "village": scene(
            "Emberhead Village",
            "The village hall, general store, ruined church, roads, and Beacon are reachable.",
            {"Ledbetter house": "ledbetter", "ruined church": "church",
             "blocked roads": "roads", "festival square": "festival"}),
        "church": scene(
            "Ruined Church",
            "Records and the ruined church reveal that the festival is not harmless.",
            {"village center": "village", "hidden route": "roads"}),
        "roads": scene(
            "Guarded Roads",
            "Villagers block the ordinary routes out of Emberhead.",
            {"village center": "village", "escape route": "escape"}),
        "festival": scene(
            "Emberhead Festival",
            "The investigator is seized for sacrifice beneath the Beacon.",
            {"the Beacon": "beacon"}),
        "beacon": scene(
            "The Beacon",
            "Fire climbs the metal structure while the investigator struggles free.",
            {"escape": "escape", "burned alive": "death"}),
        "escape": scene("Road out of Emberhead", "The investigator escapes Emberhead."),
        "death": scene("The End: Burned in the Beacon", "The sacrifice ends in death."),
    }
    write_json("expanded_alone_world.json", {
        "name": "eval_expanded_alone",
        "rule_system": "coc",
        "starting_scene": "bus_stop",
        "opening": "The investigator waits at a hot roadside bus stop, leaving home for Arkham.",
        "scenes": scenes,
        "entities": {
            "silas": npc("Silas", "coach", "coach driver"),
            "may": npc("May Ledbetter", "ledbetter", "landlady"),
            "ruth": npc("Ruth Ledbetter", "ledbetter", "young girl"),
            "ritual_knife": {
                "type": "item", "name": "ritual knife", "scene": "festival",
                "initial_state": "hidden", "description": "A concealed ritual implement.",
            },
        },
    })

    write_route("expanded_alone_escape", "expanded_alone", "expanded_alone", [
        {"input": "开始游戏", "response": "You wait alone at the bus stop for the Arkham coach.",
         "scene": "bus_stop", "contains": ["bus stop"], "covers": ["opening"]},
        {"input": "I go there and board.",
         "response": "The driver says, 'I am Silas,' then loads your cases and drives north.",
         "select_scene": "coach", "scene": "coach",
         "covers": ["coach", "explicit_scene_target"], "contains": ["Silas"]},
        {"input": "I ask Silas why we stopped.", "response": "The coach driver blames engine trouble.",
         "select_npc": "silas", "contains": ["engine"], "covers": ["selected_dialogue"]},
        movement_step("I continue with the journey.", "The coach leaves you stranded in Emberhead.",
                      "emberhead", ["stranded"], ["Emberhead"]),
        movement_step("I accept the offer of lodging.", "May leads you into the Ledbetter house.",
                      "ledbetter", ["lodging"], ["Ledbetter"]),
        {"input": "I ask Ruth what she whispered.", "response": "Ruth urgently warns you to leave before the festival.",
         "select_npc": "ruth", "contains": ["leave", "festival"], "covers": ["ruth_warning"]},
        movement_step("I investigate the village.", "You enter the center of Emberhead.",
                      "village", ["village"], ["center"]),
        movement_step("I search the ruined church.", "Church records connect the Beacon to sacrifice.",
                      "church", ["church_clue"], ["sacrifice"]),
        movement_step("I use the hidden route out.", "You avoid the guarded main roads.",
                      "roads", ["alternate_escape"], ["guarded"]),
        movement_step("I keep moving away from Emberhead.", "You reach the road beyond the village alive.",
                      "escape", ["escape_ending"], ["alive"]),
    ])
    write_route("expanded_alone_adversarial", "expanded_alone", "expanded_alone", [
        {"input": "开始游戏", "response": "You wait alone at the bus stop for the Arkham coach.",
         "scene": "bus_stop", "contains": ["bus stop"], "covers": []},
        {"input": "I go directly to The Beacon.", "response": "不能跳过中间区域。",
         "code_response": True, "contains": ["不能直接越过"], "covers": ["blocked_jump"]},
        {"input": "I take the ritual knife.", "response": "尚未发现。",
         "code_response": True, "contains": ["没有发现"], "covers": ["hidden_item_block"]},
        {"input": "I take the laser pistol from the bus stop.", "response": "不存在的物品。",
         "code_response": True, "contains": ["未能把这个动作对应"],
         "covers": ["invented_object_block"]},
        {"input": "I ask May Ledbetter about the sacrifice.", "response": "没有在场人物。",
         "code_response": True, "contains": ["没有可交谈"], "covers": ["absent_npc_block"]},
        movement_step("I board the coach.", "The driver says, 'I am Silas,' then loads your cases and drives north.",
                      "coach", [], ["Silas"]),
        movement_step("I continue to Emberhead.", "The coach leaves you stranded in Emberhead.",
                      "emberhead", [], ["Emberhead"]),
        movement_step("I go to the Ledbetter house.", "May welcomes you inside.",
                      "ledbetter", [], ["May"]),
        {"input": "I ask May about the festival.", "response": "May calls it an old local celebration.",
         "select_npc": "may", "contains": ["celebration"], "covers": ["may_selected"]},
        {"input": "I keep questioning the dead landlady.", "response": "无法继续交谈。",
         "set_entity_states": {"may": "dead"}, "code_response": True,
         "contains": ["无法交谈"], "covers": ["dead_npc_block"]},
    ])
    write_route("expanded_alone_death", "expanded_alone", "expanded_alone", [
        {"input": "开始游戏", "response": "You wait alone at the bus stop for the Arkham coach.",
         "scene": "bus_stop", "contains": ["bus stop"], "covers": []},
        movement_step("I board and continue.", "The coach carries you toward Emberhead.",
                      "coach", [], ["coach"]),
        movement_step("I remain with the coach.", "You are stranded in Emberhead.",
                      "emberhead", [], ["stranded"]),
        movement_step("I enter the village.", "The festival preparations surround the village center.",
                      "village", [], ["festival"]),
        movement_step("I wait for the festival instead of escaping.", "Villagers seize you for the ceremony.",
                      "festival", ["captured"], ["seize"]),
        movement_step("I do not resist.", "You are chained high inside the burning Beacon.",
                      "beacon", ["beacon"], ["burning"]),
        movement_step("I fail to escape the flames.", "The Beacon burns you alive. The adventure ends.",
                      "death", ["death_ending"], ["ends"]),
    ])
    write_json("expanded_alone_coverage.json", {
        "module": "expanded_alone",
        "required": [
            "opening", "coach", "selected_dialogue", "stranded", "lodging",
            "ruth_warning", "village", "church_clue", "alternate_escape",
            "escape_ending", "blocked_jump", "hidden_item_block",
            "absent_npc_block", "may_selected", "dead_npc_block", "captured",
            "beacon", "death_ending", "invented_object_block",
        ],
    })


def generate_great_hunt() -> None:
    scenes = {
        "lodge": scene(
            "Sir Servause's Hunting Lodge",
            "Sir Ector asks the knights to save a noble knight from the Berwyn dragon.",
            {"plan the hunt": "planning", "refuse the quest": "refusal"}),
        "planning": scene(
            "Great Hunt Planning",
            "The knights must obtain six animals, especially a panther.",
            {"gather the menagerie": "menagerie", "ride directly to Berwyn": "berwyn"}),
        "menagerie": scene(
            "The Traveling Menagerie",
            "The crane, stag, eagle, lion cub, mouse, unicorn, and panther plans converge.",
            {"ride to Berwyn": "berwyn", "return to the lodge": "lodge"}),
        "berwyn": scene(
            "Berwyn",
            "The dragon threatens the countryside and its victim.",
            {"confront the dragon": "dragon"}),
        "dragon": scene(
            "The Dragon of Berwyn",
            "The knights may fight or use the panther's belch.",
            {"use the panther": "victory", "fight recklessly": "defeat"}),
        "victory": scene("Great Hunt Victory", "The dragon is defeated without needless slaughter."),
        "defeat": scene("Great Hunt Defeat", "The direct assault ends in disaster."),
        "refusal": scene("Quest Refused", "The knights refuse Sir Ector's appeal."),
    }
    write_json("expanded_great_hunt_world.json", {
        "name": "eval_expanded_great_hunt",
        "rule_system": "dnd",
        "starting_scene": "lodge",
        "opening": "The knights are guests of Sir Servause when Sir Ector arrives seeking aid.",
        "scenes": scenes,
        "entities": {
            "servause": npc("Sir Servause", "lodge", "elderly host"),
            "ector": npc("Sir Ector", "lodge", "honored knight"),
            "dragon": {
                "type": "monster", "name": "Dragon of Berwyn", "scene": "dragon",
                "initial_state": "present",
            },
            "panther": {
                "type": "item", "name": "captured panther", "scene": "menagerie",
                "initial_state": "hidden", "aliases": ["panther"],
            },
        },
    })
    write_route("expanded_great_hunt_clever", "expanded_great_hunt", "expanded_great_hunt", [
        {"input": "开始游戏", "response": "Sir Ector arrives at the hunting lodge asking for aid.",
         "scene": "lodge", "contains": ["Sir Ector"], "covers": ["hunt_opening"]},
        {"input": "I ask Sir Ector who must be saved.", "response": "He explains that the Berwyn dragon threatens a noble knight.",
         "select_npc": "ector", "contains": ["dragon"], "covers": ["ector_dialogue"]},
        {"input": "We agree and go there to plan the hunt.",
         "response": "Sir Servause outlines the animal lore behind the plan.",
         "select_scene": "planning", "scene": "planning",
         "covers": ["accept_quest", "explicit_scene_target"], "contains": ["animal"]},
        movement_step("We divide the party and gather every animal.", "The knights assemble the unusual traveling menagerie.",
                      "menagerie", ["gather_animals", "party_split"], ["menagerie"]),
        {**movement_step(
            "We return to Sir Servause before leaving.",
            "The party returns to the lodge to confirm the plan.",
            "lodge", ["return_old_map"], ["lodge"]),
         "set_entity_states": {"panther": "obtained"},
         "expect": {
             "inventory_contains": ["panther"],
             "object_locations": {
                 "panther": {"kind": "inventory", "id": "player"}},
         }},
        movement_step("We resume the plan.", "The animal plan is ready.",
                      "planning", [], ["ready"]),
        movement_step("We bring the menagerie to Berwyn.", "The knights reach Berwyn with the animals.",
                      "menagerie", [], ["animals"]),
        movement_step("We ride to Berwyn.", "The dragon's attack shakes the countryside.",
                      "berwyn", ["reach_berwyn"], ["dragon"]),
        movement_step("We confront the dragon.", "The knights draw the dragon toward the prepared panther.",
                      "dragon", [], ["panther"]),
        {**movement_step(
            "We use the panther rather than charge.",
            "The panther's belch overcomes the dragon.",
            "victory", ["clever_victory", "carried_object_use"], ["overcomes"]),
         "select_object": "panther",
         "expect": {
             "inventory_contains": ["panther"],
             "world_event_types_contains": ["item_used"],
         }},
    ])
    write_route("expanded_great_hunt_reckless", "expanded_great_hunt", "expanded_great_hunt", [
        {"input": "开始游戏", "response": "Sir Ector arrives at the hunting lodge asking for aid.",
         "scene": "lodge", "contains": ["Sir Ector"], "covers": []},
        {"input": "I ride directly to The Dragon of Berwyn.", "response": "路径被阻止。",
         "code_response": True, "contains": ["不能直接越过"], "covers": ["dragon_jump_block"]},
        {"input": "I take the captured panther from my inventory.", "response": "未发现该物品。",
         "code_response": True, "contains": ["没有发现"], "covers": ["panther_guess_block"]},
        {"input": "I use the dragon horn in my pack.", "response": "不存在的物品。",
         "code_response": True, "contains": ["未能把这个动作对应"],
         "covers": ["invented_object_block"]},
        movement_step("We plan the hunt but reject the animal trick.", "The knights prepare a direct ride to Berwyn.",
                      "planning", ["reject_clever_plan"], ["direct"]),
        movement_step("We ride directly to Berwyn.", "The knights reach the threatened countryside.",
                      "berwyn", [], ["countryside"]),
        movement_step("We confront the dragon with steel.", "The dragon descends on the charging knights.",
                      "dragon", ["direct_combat"], ["charging"]),
        movement_step("We continue the reckless attack and fail.", "The assault ends in defeat and grievous loss.",
                      "defeat", ["combat_defeat"], ["defeat"]),
    ])
    write_route("expanded_great_hunt_refusal", "expanded_great_hunt", "expanded_great_hunt", [
        {"input": "开始游戏", "response": "Sir Ector arrives at the hunting lodge asking for aid.",
         "scene": "lodge", "contains": ["Sir Ector"], "covers": []},
        {"input": "I speak to Sir Ector without selecting him.", "response": "要求选择人物。",
         "code_response": True, "contains": ["选择一名"], "covers": ["selection_required"]},
        {"input": "I ask Sir Ector whether refusal is allowed.", "response": "He says the choice is yours, though lives are at stake.",
         "select_npc": "ector", "contains": ["choice"], "covers": ["refusal_discussed"]},
        movement_step("We refuse the quest.", "The knights remain behind while the danger continues.",
                      "refusal", ["refusal_ending"], ["remain"]),
    ])
    write_json("expanded_great_hunt_coverage.json", {
        "module": "expanded_great_hunt",
        "required": [
            "hunt_opening", "ector_dialogue", "accept_quest", "gather_animals",
            "party_split", "return_old_map", "reach_berwyn", "clever_victory",
            "dragon_jump_block", "panther_guess_block", "reject_clever_plan",
            "direct_combat", "combat_defeat", "selection_required",
            "carried_object_use", "invented_object_block",
            "refusal_discussed", "refusal_ending",
        ],
    })


def generate_red_blade() -> None:
    scenes = {
        "briefing": scene(
            "Sir Gregor's Briefing",
            "Sir Gregor asks the knights to recover the Red Death Blade.",
            {"journey west": "road"}),
        "road": scene(
            "The Journey West",
            "The knights travel toward the Castle of the Crane.",
            {"Castle of the Crane": "crane"}),
        "crane": scene(
            "The Castle of the Crane",
            "King Garan may help if the knights swear not to harm his people.",
            {"Castle of the Kite": "kite", "journey west": "road"}),
        "kite": scene(
            "The Castle of the Kite",
            "King Cadwalader guards the Red Death Blade and Pig Boy.",
            {"royal hall": "hall", "dungeon": "dungeon"}),
        "hall": scene(
            "King Cadwalader's Hall",
            "The blade, the king, his household, and Pig Boy are present.",
            {"seize the blade": "flight", "reveal the mission": "dungeon"}),
        "flight": scene(
            "The Flight North",
            "The knights flee with the blade while choosing whether to save Pig Boy.",
            {"save Pig Boy": "feast", "abandon Pig Boy": "return"}),
        "feast": scene(
            "The Terrible Feast",
            "Pig Boy faces sacrifice unless rescued.",
            {"return home": "return"}),
        "dungeon": scene("Cadwalader's Dungeon", "Revealing the theft brings imprisonment."),
        "return": scene("Return from Meirionydd", "The surviving knights return with the blade."),
    }
    write_json("expanded_red_blade_world.json", {
        "name": "eval_expanded_red_blade",
        "rule_system": "dnd",
        "starting_scene": "briefing",
        "opening": "Sir Gregor charges the knights with recovering the Red Death Blade.",
        "scenes": scenes,
        "entities": {
            "gregor": npc("Sir Gregor", "briefing", "elderly knight"),
            "garan": npc("King Garan", "crane", "lord of the Crane"),
            "cadwalader": npc("King Cadwalader", "hall", "king of Meirionydd"),
            "pig_boy": npc("Pig Boy", "hall", "young captive", ["hall", "flight", "feast"]),
            "red_blade": {
                "type": "item", "name": "Red Death Blade", "scene": "hall",
                "initial_state": "hidden", "aliases": ["blade", "sword"],
            },
        },
    })
    write_route("expanded_red_blade_honorable", "expanded_red_blade", "expanded_red_blade", [
        {"input": "开始游戏", "response": "Sir Gregor explains the mission to recover the Red Death Blade.",
         "scene": "briefing", "contains": ["mission"], "covers": ["blade_opening"]},
        {"input": "I ask Sir Gregor what matters besides the sword.", "response": "He asks the knights to preserve their honor.",
         "select_npc": "gregor", "contains": ["honor"], "covers": ["gregor_dialogue"]},
        {"input": "We choose that road and go there.",
         "response": "The company rides toward Cambria.",
         "select_scene": "road", "scene": "road",
         "covers": ["explicit_scene_target"], "contains": ["Cambria"]},
        movement_step("We seek hospitality at the Castle of the Crane.", "The lord says, 'I am King Garan,' and receives the company.",
                      "crane", ["garan_meeting"], ["Garan"]),
        {"input": "I swear not to harm Garan's people.", "response": "King Garan accepts the oath and offers guidance.",
         "select_npc": "garan", "contains": ["oath"], "covers": ["garan_oath"]},
        movement_step("We continue to the Castle of the Kite.", "The company reaches Cadwalader's stronghold.",
                      "kite", [], ["stronghold"]),
        {**movement_step(
            "We enter the royal hall without revealing the theft.",
            "King Cadwalader receives the disguised visitors.",
            "hall", ["conceal_goal"], ["disguised"]),
         "set_entity_states": {"red_blade": "revealed"}},
        {**movement_step(
            "We seize the blade and take Pig Boy with us.",
            "The knights escape north with Pig Boy and the blade.",
            "flight", ["seize_blade", "blade_carried_across_scene"], ["Pig Boy"]),
         "select_object": "red_blade",
         "expect": {
             "inventory_contains": ["red_blade"],
             "object_locations": {
                 "red_blade": {"kind": "inventory", "id": "player"}},
             "world_event_types_contains": ["item_picked_up"],
         }},
        movement_step("We turn back to save Pig Boy from the feast.", "The rescue prevents Pig Boy's sacrifice.",
                      "feast", ["save_pig_boy"], ["rescue"]),
        movement_step("We return home together.", "The knights return with the blade and the rescued boy.",
                      "return", ["honorable_return"], ["rescued"]),
    ])
    write_route("expanded_red_blade_betrayal", "expanded_red_blade", "expanded_red_blade", [
        {"input": "开始游戏", "response": "Sir Gregor explains the mission to recover the Red Death Blade.",
         "scene": "briefing", "contains": ["mission"], "covers": []},
        {"input": "I go directly to King Cadwalader's Hall.", "response": "不能跳到终局。",
         "code_response": True, "contains": ["不能直接越过"], "covers": ["hall_jump_block"]},
        {"input": "I use the Red Death Blade now.", "response": "隐藏物品不可用。",
         "code_response": True, "contains": ["没有发现"], "covers": ["blade_guess_block"]},
        {"input": "I take the invisibility cloak.", "response": "不存在的物品。",
         "code_response": True, "contains": ["未能把这个动作对应"],
         "covers": ["invented_object_block"]},
        movement_step("We travel west.", "The company begins the western journey.",
                      "road", [], ["journey"]),
        movement_step("We pass through the Castle of the Crane.", "The lord says, 'I am King Garan,' and permits the company to continue.",
                      "crane", [], ["Garan"]),
        movement_step("We travel to the Castle of the Kite.", "The company reaches Cadwalader's castle.",
                      "kite", [], ["castle"]),
        movement_step("We enter the royal hall.", "King Cadwalader questions the visitors.",
                      "hall", [], ["questions"]),
        {**movement_step(
            "I tell Cadwalader we came to steal his sword.",
            "The admission brings immediate arrest.",
            "dungeon", ["revealed_goal", "dungeon_ending"], ["arrest"]),
         "select_npc": "cadwalader"},
    ])
    write_route("expanded_red_blade_abandon", "expanded_red_blade", "expanded_red_blade", [
        {"input": "开始游戏", "response": "Sir Gregor explains the mission to recover the Red Death Blade.",
         "scene": "briefing", "contains": ["mission"], "covers": []},
        movement_step("We travel west.", "The company begins the journey.",
                      "road", [], ["journey"]),
        movement_step("We go to the Castle of the Crane.", "The lord says, 'I am King Garan,' and receives the company.",
                      "crane", [], ["Garan"]),
        movement_step("We return to the road to reconsider.", "The company backtracks from the castle.",
                      "road", ["backtrack"], ["backtracks"]),
        movement_step("We go back to the Castle of the Crane.", "King Garan receives the returning company.",
                      "crane", [], ["returning"]),
        movement_step("We continue to the Castle of the Kite.", "The company reaches the stronghold.",
                      "kite", [], ["stronghold"]),
        {**movement_step(
            "We enter Cadwalader's hall.",
            "The king displays the Red Death Blade.",
            "hall", [], ["displays"]),
         "set_entity_states": {"red_blade": "revealed"}},
        {**movement_step(
            "We seize the blade.", "The knights flee north with the weapon.",
            "flight", ["blade_carried_across_scene"], ["flee"]),
         "select_object": "red_blade",
         "expect": {
             "inventory_contains": ["red_blade"],
             "object_locations": {
                 "red_blade": {"kind": "inventory", "id": "player"}},
             "world_event_types_contains": ["item_picked_up"],
         }},
        movement_step("We abandon Pig Boy and keep running.", "The knights save themselves and leave the captive behind.",
                      "return", ["abandon_pig_boy", "selfish_return"], ["leave"]),
    ])
    write_json("expanded_red_blade_coverage.json", {
        "module": "expanded_red_blade",
        "required": [
            "blade_opening", "gregor_dialogue", "garan_meeting", "garan_oath",
            "conceal_goal", "seize_blade", "save_pig_boy", "honorable_return",
            "hall_jump_block", "blade_guess_block", "revealed_goal",
            "dungeon_ending", "backtrack", "abandon_pig_boy", "selfish_return",
            "blade_carried_across_scene", "invented_object_block",
        ],
    })


def generate_sword_kings() -> None:
    scenes = {
        "storm": scene(
            "Scene One: The Storm",
            "The Heroes fight Cormac's crew aboard two ships in a supernatural storm.",
            {"reach the island free": "island", "arrive in chains": "captured"}),
        "captured": scene(
            "Captured by Cormac",
            "The defeated Heroes reach the island in chains.",
            {"escape onto the island": "island"}),
        "island": scene(
            "Scene Two: The Island",
            "A local tribe protects the cave and can be fought or negotiated with.",
            {"negotiate passage": "cave", "fight the tribe": "cave"}),
        "cave": scene(
            "Cave of the Three Swords",
            "Three swords present different consequences.",
            {"first sword": "first", "second sword": "second", "third sword": "third"}),
        "first": scene("First Sword Ending", "The first sword is chosen."),
        "second": scene("Second Sword Ending", "The second sword is chosen."),
        "third": scene("Third Sword Ending", "The third sword is chosen."),
    }
    write_json("expanded_sword_kings_world.json", {
        "name": "eval_expanded_sword_kings",
        "rule_system": "dnd",
        "starting_scene": "storm",
        "opening": "The Heroes begin amid a supernatural storm and boarding action.",
        "scenes": scenes,
        "entities": {
            "cormac": npc("Cormac McDougal", "storm", "enemy captain"),
            "tribal_leader": npc("Whakapau Hina", "island", "island guardian"),
            "third_sword": {
                "type": "item", "name": "Third Sword", "scene": "cave",
                "initial_state": "hidden",
            },
        },
    })

    def sword_route(stem: str, storm_outcome: str, approach: str,
                    ending: str, tags: list[str]) -> None:
        first_target = "captured" if storm_outcome == "loss" else "island"
        first_text = (
            "Cormac's crew defeats and chains the Heroes."
            if storm_outcome == "loss" else
            "The Heroes repel the boarders and reach the island free."
        )
        steps = [
            {"input": "开始游戏", "response": "A storm lashes both ships as Cormac's crew boards.",
             "scene": "storm", "contains": ["storm"], "covers": ["sword_opening"]},
            movement_step(
                "We lose the boarding action." if storm_outcome == "loss"
                else "We repel the boarding party.",
                first_text, first_target,
                ["storm_loss" if storm_outcome == "loss" else "storm_win"],
                ["chains" if storm_outcome == "loss" else "free"]),
        ]
        if first_target == "captured":
            steps.append(movement_step(
                "We escape our chains on the island.",
                "The Heroes slip away from Cormac and enter the jungle.",
                "island", ["chain_escape"], ["jungle"]))
        steps.append(movement_step(
            "We negotiate with the island guardians." if approach == "talk"
            else "We fight through the island guardians.",
            "Whakapau Hina grants passage after negotiation."
            if approach == "talk" else
            "The Heroes force a costly path through the defenders.",
            "cave", ["island_negotiate" if approach == "talk" else "island_violence"],
            ["passage" if approach == "talk" else "costly"]))
        steps.append(movement_step(
            f"We choose the {ending} sword.",
            f"The Heroes accept the consequences of the {ending} sword.",
            ending, tags, [ending]))
        write_route(stem, "expanded_sword_kings", "expanded_sword_kings", steps)

    sword_route("expanded_sword_first", "win", "talk", "first", ["first_ending"])
    sword_route("expanded_sword_second", "loss", "violence", "second", ["second_ending"])
    write_route("expanded_sword_third_adversarial", "expanded_sword_kings", "expanded_sword_kings", [
        {"input": "开始游戏", "response": "A storm lashes both ships as Cormac's crew boards.",
         "scene": "storm", "contains": ["storm"], "covers": []},
        {"input": "I ask Cormac to surrender without selecting him.", "response": "需要选择。",
         "code_response": True, "contains": ["选择一名"], "covers": ["cormac_selection_required"]},
        {"input": "I take the Third Sword from my bag.", "response": "尚未取得。",
         "code_response": True, "contains": ["没有发现"], "covers": ["third_sword_guess_block"]},
        {"input": "I use the magic compass in my pack.", "response": "不存在的物品。",
         "code_response": True, "contains": ["未能把这个动作对应"],
         "covers": ["invented_object_block"]},
        {"input": "I go directly to the Cave of the Three Swords.", "response": "不能跳过岛屿。",
         "code_response": True, "contains": ["不能直接越过"], "covers": ["cave_jump_block"]},
        movement_step("We win the storm battle.", "The Heroes reach the island free.",
                      "island", [], ["free"]),
        movement_step("We negotiate passage.", "The island guardians permit entry to the cave.",
                      "cave", [], ["permit"]),
        movement_step("We choose the third sword.", "The third sword imposes its dramatic consequence.",
                      "third", ["third_ending"], ["third"]),
    ])
    write_json("expanded_sword_kings_coverage.json", {
        "module": "expanded_sword_kings",
        "required": [
            "sword_opening", "storm_win", "storm_loss", "chain_escape",
            "island_negotiate", "island_violence", "first_ending", "second_ending",
            "third_ending", "cormac_selection_required", "third_sword_guess_block",
            "cave_jump_block", "invented_object_block",
        ],
    })


def generate_rattling_wind() -> None:
    scenes = {
        "tavern": scene(
            "New Stone Tavern",
            "Jorgrin asks the adventurers to protect Farfield from the Rattling Wind.",
            {"question the village": "farfield", "wait for the attack": "attack",
             "leave Farfield": "destruction"}),
        "farfield": scene(
            "Farfield Hamlet",
            "The villagers hide their connection to Beleros; pond weed points toward the lake.",
            {"return to the tavern": "tavern", "wait for the attack": "attack",
             "track toward Drakemere": "tracks"}),
        "attack": scene(
            "The Monster Attacks",
            "An undead bronze chariot charges through Farfield.",
            {"track the monster": "tracks", "stand and fight": "fight"}),
        "fight": scene(
            "Fight the Bone Chariot",
            "The adventurers attempt to destroy the chariot in Farfield.",
            {"victory": "victory", "failure": "destruction"}),
        "tracks": scene(
            "Tracking the Monster",
            "The chariot tracks lead toward Drakemere.",
            {"Drakemere": "drakemere"}),
        "drakemere": scene(
            "Drakemere",
            "Darvyn's people deny causing the attacks and reveal the lake's recent clearing.",
            {"enter the lake": "lake", "attack the ducks": "duck_conflict"}),
        "lake": scene(
            "Into the Lake",
            "The remains of Beleros and the bronze chariot lie below.",
            {"lay Beleros to rest": "victory", "disturb the remains": "final_attack"}),
        "final_attack": scene(
            "Fury From the Lake",
            "The enraged chariot races toward Farfield.",
            {"stop the chariot": "victory", "fail": "destruction"}),
        "duck_conflict": scene("War with Drakemere", "The false accusation creates needless bloodshed."),
        "victory": scene("The Rattling Wind Ended", "Beleros is laid to rest and Farfield survives."),
        "destruction": scene("Farfield Destroyed", "The unchecked chariot devastates the hamlet."),
    }
    write_json("expanded_rattling_wind_world.json", {
        "name": "eval_expanded_rattling_wind",
        "rule_system": "dnd",
        "starting_scene": "tavern",
        "opening": "Jorgrin and Farfield's frightened villagers ask for protection before the next attack.",
        "scenes": scenes,
        "entities": {
            "jorgrin": npc("Jorgrin", "tavern", "tavern-keeper", ["tavern", "farfield"]),
            "orvald": npc("Orvald", "farfield", "limping ostler"),
            "viborna": npc("Viborna", "farfield", "scarred redsmith"),
            "darvyn": npc("Darvyn Blackfeather", "drakemere", "Drakemere leader"),
            "kopis": {
                "type": "item", "name": "Beleros' fine kopis", "scene": "farfield",
                "initial_state": "hidden",
            },
            "pond_weed": {
                "type": "clue", "name": "pond weed", "scene": "farfield",
                "initial_state": "hidden", "description": "Wet weed from a victim's wound.",
            },
            "bone_chariot": {
                "type": "monster", "name": "Bone Chariot", "scene": "attack",
                "initial_state": "present",
            },
        },
    })
    write_route("expanded_rattling_truth", "expanded_rattling_wind", "expanded_rattling_wind", [
        {"input": "开始游戏", "response": "Jorgrin asks for protection from the weekly Rattling Wind.",
         "scene": "tavern", "contains": ["Jorgrin"], "covers": ["rattling_opening"]},
        {"input": "I ask Jorgrin about earlier attacks.", "response": "He describes three victims but avoids the distant past.",
         "select_npc": "jorgrin", "contains": ["three"], "covers": ["jorgrin_selected"]},
        {"input": "I go there to investigate.",
         "response": "The villagers reveal crushed bodies and wet pond weed.",
         "select_scene": "farfield", "scene": "farfield",
         "covers": ["victim_clue", "explicit_scene_target"],
         "contains": ["pond weed"],
         "set_entity_states": {"pond_weed": "revealed"}},
        {"input": "I ask Orvald where his fine horses came from.", "response": "The limping ostler gives an evasive account.",
         "select_npc": "orvald", "contains": ["evasive"], "covers": ["orvald_lie"]},
        movement_step("We track toward Drakemere.", "Tracks and pond weed lead toward the lake settlement.",
                      "tracks", ["track_clue"], ["lake"]),
        movement_step("We follow the tracks to Drakemere.", "Darvyn's people confront the adventurers.",
                      "drakemere", ["meet_darvyn"], ["Darvyn"]),
        {"input": "I ask Darvyn what changed in the lake.", "response": "He explains that his people recently cleared the lily pads.",
         "select_npc": "darvyn", "contains": ["lily pads"], "covers": ["lake_cleared"]},
        movement_step("We enter the lake and inspect the remains.", "The submerged chariot and Beleros' remains reveal the old crime.",
                      "lake", ["discover_beleros"], ["old crime"]),
        movement_step("We lay Beleros to rest.", "The vengeance ends and Farfield survives.",
                      "victory", ["rest_ending"], ["survives"]),
    ])
    write_route("expanded_rattling_wrong_target", "expanded_rattling_wind", "expanded_rattling_wind", [
        {"input": "开始游戏", "response": "Jorgrin asks for protection from the weekly Rattling Wind.",
         "scene": "tavern", "contains": ["Jorgrin"], "covers": []},
        {"input": "I take Beleros' fine kopis from the stable.", "response": "隐藏物品被阻止。",
         "code_response": True, "contains": ["没有发现"], "covers": ["kopis_guess_block"]},
        {"input": "I take the silver wand from the stable.", "response": "不存在的物品。",
         "code_response": True, "contains": ["未能把这个动作对应"],
         "covers": ["invented_object_block"]},
        {"input": "I go directly to Drakemere.", "response": "不能跳过追踪。",
         "code_response": True, "contains": ["不能直接越过"], "covers": ["lake_jump_block"]},
        movement_step("I go into Farfield to investigate.", "The villagers blame their secretive neighbors.",
                      "farfield", [], ["neighbors"]),
        movement_step("We track toward Drakemere.", "The party follows the trail toward the lake.",
                      "tracks", [], ["trail"]),
        movement_step("We continue to Drakemere.", "Darvyn denies responsibility.",
                      "drakemere", [], ["denies"]),
        movement_step("We accuse the ducks and attack.", "The false accusation starts a needless conflict.",
                      "duck_conflict", ["false_accusation", "duck_conflict"], ["needless"]),
    ])
    write_route("expanded_rattling_wait", "expanded_rattling_wind", "expanded_rattling_wind", [
        {"input": "开始游戏", "response": "Jorgrin asks for protection from the weekly Rattling Wind.",
         "scene": "tavern", "contains": ["Jorgrin"], "covers": []},
        {"input": "I tell Jorgrin we will simply wait.", "response": "He reluctantly agrees to wait and prepares the tavern for another attack.",
         "select_npc": "jorgrin", "contains": ["wait"], "covers": ["wait_choice"]},
        movement_step("We wait for the monster.", "The undead bronze chariot charges into Farfield.",
                      "attack", ["attack_arrives"], ["chariot"]),
        {"input": "I keep asking the dead Jorgrin what to do.", "response": "死亡人物不能回答。",
         "set_entity_states": {"jorgrin": "dead"}, "code_response": True,
         "contains": ["没有可交谈"], "covers": ["jorgrin_death_block"]},
        movement_step("We stand and fight the chariot.", "The party tries to stop it in the village.",
                      "fight", ["stand_fight"], ["stop"]),
        movement_step("We fail to stop it.", "The unchecked chariot devastates Farfield.",
                      "destruction", ["destruction_ending"], ["devastates"]),
    ])
    write_json("expanded_rattling_wind_coverage.json", {
        "module": "expanded_rattling_wind",
        "required": [
            "rattling_opening", "jorgrin_selected", "victim_clue", "orvald_lie",
            "track_clue", "meet_darvyn", "lake_cleared", "discover_beleros",
            "rest_ending", "kopis_guess_block", "lake_jump_block",
            "false_accusation", "duck_conflict", "wait_choice", "attack_arrives",
            "jorgrin_death_block", "stand_fight", "destruction_ending",
            "invented_object_block",
        ],
    })


def main() -> None:
    generate_alone()
    generate_great_hunt()
    generate_red_blade()
    generate_sword_kings()
    generate_rattling_wind()
    print("Generated expanded adversarial fixtures.")


if __name__ == "__main__":
    main()

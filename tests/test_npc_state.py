import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from backend.npc_state import merge_npcs
from reference_resolver import npc_is_interactable


class NpcDisclosureInitializationTests(unittest.TestCase):
    def test_only_opening_or_explicitly_public_names_start_revealed(self):
        world = {
            "opening": "Lucy Albright waits behind the counter.",
            "entities": {
                "lucy": {
                    "type": "npc", "name": "Lucy Albright", "scene": "shop",
                },
                "gurteen": {
                    "type": "npc", "name": "Old Gurteen", "scene": "hospital",
                    "known_to_player": True,
                },
                "cassidy": {
                    "type": "npc", "name": "George Cassidy", "scene": "light",
                },
            },
        }

        states = merge_npcs(world)

        self.assertTrue(states["Lucy Albright"]["dynamic"]["disclosure"]["name"])
        self.assertTrue(states["Old Gurteen"]["dynamic"]["disclosure"]["name"])
        self.assertFalse(states["George Cassidy"]["dynamic"]["disclosure"]["name"])

    def test_authored_state_can_disable_dialogue_without_reserved_name(self):
        world = {"entities": {"captain": {
            "type": "npc", "name": "Captain", "initial_state": "present",
            "states": {
                "present": {},
                "defeated": {"interactable": False},
                "captured": {"interactable": True},
            },
        }}}

        self.assertFalse(npc_is_interactable(
            "captain", world, {"entity_states": {"captain": "defeated"}}))
        self.assertTrue(npc_is_interactable(
            "captain", world, {"entity_states": {"captain": "captured"}}))


if __name__ == "__main__":
    unittest.main()

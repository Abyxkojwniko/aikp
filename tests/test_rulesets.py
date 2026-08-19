import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dice import resolve_check


class RulesetDiceTests(unittest.TestCase):
    @patch("dice.random.randint", return_value=10)
    def test_runequest_uses_special_success_band(self, _roll):
        result = resolve_check("runequest", 50)

        self.assertEqual(10, result["d100"])
        self.assertEqual("special_success", result["verdict"])
        self.assertTrue(result["success"])

    @patch("dice.random.randint", return_value=6)
    def test_brp_critical_is_one_tenth_of_skill(self, _roll):
        result = resolve_check("brp", 60)

        self.assertEqual("critical_success", result["verdict"])

    @patch("dice.random.randint", return_value=12)
    def test_dragonbane_is_d20_roll_under(self, _roll):
        result = resolve_check("dragonbane", 14)

        self.assertEqual(12, result["d20"])
        self.assertTrue(result["success"])
        self.assertNotIn("total", result)

    @patch("dice.random.randint", return_value=15)
    def test_pendragon_exact_target_is_critical(self, _roll):
        result = resolve_check("pendragon", 15)

        self.assertEqual("critical_success", result["verdict"])
        self.assertTrue(result["success"])

    @patch("dice.random.randint", return_value=12)
    def test_starfinder_uses_d20_plus_modifier_against_dc(self, _roll):
        result = resolve_check("starfinder", 4, 15)

        self.assertEqual(16, result["total"])
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()

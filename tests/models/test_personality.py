# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for CommanderPersonality model."""

import pytest

from tritium_lib.models.personality import (
    CommanderPersonality,
    PRESET_PERSONALITIES,
    PATROL_PERSONALITY,
    BATTLE_PERSONALITY,
    STEALTH_PERSONALITY,
    OBSERVER_PERSONALITY,
)


class TestCommanderPersonality:
    """Tests for CommanderPersonality dataclass."""

    def test_default_values(self):
        p = CommanderPersonality()
        assert p.aggression == 0.5
        assert p.curiosity == 0.5
        assert p.verbosity == 0.5
        assert p.caution == 0.5
        assert p.initiative == 0.5

    def test_custom_values(self):
        p = CommanderPersonality(aggression=0.8, curiosity=0.2)
        assert p.aggression == 0.8
        assert p.curiosity == 0.2

    def test_clamping(self):
        p = CommanderPersonality(aggression=1.5, curiosity=-0.3)
        assert p.aggression == 1.0
        assert p.curiosity == 0.0

    def test_to_dict(self):
        p = CommanderPersonality(aggression=0.7, curiosity=0.3)
        d = p.to_dict()
        assert d["aggression"] == 0.7
        assert d["curiosity"] == 0.3
        assert d["verbosity"] == 0.5
        assert len(d) == 5

    def test_from_dict(self):
        data = {"aggression": 0.9, "curiosity": 0.1, "verbosity": 0.6}
        p = CommanderPersonality.from_dict(data)
        assert p.aggression == 0.9
        assert p.curiosity == 0.1
        assert p.verbosity == 0.6
        assert p.caution == 0.5  # default
        assert p.initiative == 0.5  # default

    def test_from_dict_empty(self):
        p = CommanderPersonality.from_dict({})
        assert p.aggression == 0.5

    def test_roundtrip(self):
        original = CommanderPersonality(aggression=0.75, curiosity=0.25, verbosity=0.9, caution=0.1, initiative=0.6)
        d = original.to_dict()
        restored = CommanderPersonality.from_dict(d)
        assert restored.aggression == original.aggression
        assert restored.curiosity == original.curiosity
        assert restored.verbosity == original.verbosity
        assert restored.caution == original.caution
        assert restored.initiative == original.initiative

    def test_profile_label_balanced(self):
        p = CommanderPersonality()
        assert p.profile_label == "balanced"

    def test_profile_label_aggressive(self):
        p = CommanderPersonality(aggression=0.9, curiosity=0.3)
        assert p.profile_label == "aggressive"

    def test_profile_label_cautious(self):
        p = CommanderPersonality(caution=0.9, aggression=0.2, curiosity=0.2)
        assert p.profile_label == "cautious"

    def test_profile_label_restrained(self):
        p = CommanderPersonality(aggression=0.1, curiosity=0.1, verbosity=0.1, caution=0.2, initiative=0.1)
        assert p.profile_label == "restrained"


class TestPresets:
    """Tests for preset personality profiles."""

    def test_presets_exist(self):
        assert "default" in PRESET_PERSONALITIES
        assert "patrol" in PRESET_PERSONALITIES
        assert "battle" in PRESET_PERSONALITIES
        assert "stealth" in PRESET_PERSONALITIES
        assert "observer" in PRESET_PERSONALITIES

    def test_patrol_is_curious(self):
        assert PATROL_PERSONALITY.curiosity > 0.5
        assert PATROL_PERSONALITY.aggression < 0.5

    def test_battle_is_aggressive(self):
        assert BATTLE_PERSONALITY.aggression > 0.7
        assert BATTLE_PERSONALITY.initiative > 0.7

    def test_stealth_is_quiet(self):
        assert STEALTH_PERSONALITY.verbosity < 0.2
        assert STEALTH_PERSONALITY.caution > 0.7

    def test_observer_is_curious(self):
        assert OBSERVER_PERSONALITY.curiosity > 0.8
        assert OBSERVER_PERSONALITY.aggression < 0.2

    def test_all_presets_valid_range(self):
        for name, preset in PRESET_PERSONALITIES.items():
            for attr in ("aggression", "curiosity", "verbosity", "caution", "initiative"):
                val = getattr(preset, attr)
                assert 0.0 <= val <= 1.0, f"{name}.{attr} = {val} out of range"

    def test_all_presets_serializable(self):
        for name, preset in PRESET_PERSONALITIES.items():
            d = preset.to_dict()
            assert len(d) == 5, f"{name} has {len(d)} keys"
            restored = CommanderPersonality.from_dict(d)
            assert restored.aggression == preset.aggression

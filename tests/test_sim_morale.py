# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the morale and psychology simulation module."""

import pytest

from tritium_lib.sim_engine.morale import (
    COMMANDER_AURA_RADIUS,
    RECOVERY_DELAY,
    RECOVERY_RATE,
    MoraleEngine,
    MoraleEvent,
    MoraleEventType,
    MoraleState,
    UnitMorale,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> MoraleEngine:
    """A fresh MoraleEngine."""
    return MoraleEngine()


@pytest.fixture
def populated_engine() -> MoraleEngine:
    """Engine with two squads of units registered."""
    me = MoraleEngine()
    for i in range(4):
        me.register_unit(f"f_{i}", alliance="friendly", starting_morale=75.0)
    me.register_unit("f_cmd", alliance="friendly", starting_morale=80.0, is_commander=True)
    for i in range(4):
        me.register_unit(f"h_{i}", alliance="hostile", starting_morale=70.0)
    return me


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_unit(self, engine: MoraleEngine) -> None:
        um = engine.register_unit("u1", alliance="friendly", starting_morale=60.0)
        assert um.unit_id == "u1"
        assert um.morale == 60.0
        assert um.alliance == "friendly"
        assert um.is_alive is True

    def test_register_commander(self, engine: MoraleEngine) -> None:
        um = engine.register_unit("cmd", is_commander=True)
        assert um.is_commander is True

    def test_register_clamps_morale(self, engine: MoraleEngine) -> None:
        um = engine.register_unit("high", starting_morale=150.0)
        assert um.morale == 100.0
        um2 = engine.register_unit("low", starting_morale=-10.0)
        assert um2.morale == 0.0

    def test_remove_unit(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1")
        engine.remove_unit("u1")
        assert engine.get_morale("u1") == 0.0

    def test_mark_dead(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.mark_dead("u1")
        assert engine.get_morale("u1") == 0.0
        assert engine.get_state("u1") == MoraleState.ROUTED


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_took_damage_lowers_morale(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.apply_event(MoraleEvent(unit_id="u1", event_type=MoraleEventType.TOOK_DAMAGE))
        engine.tick(0.1)
        assert engine.get_morale("u1") < 75.0

    def test_enemy_killed_raises_morale(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=50.0)
        engine.apply_event(MoraleEvent(unit_id="u1", event_type=MoraleEventType.ENEMY_KILLED))
        engine.tick(0.1)
        assert engine.get_morale("u1") > 50.0

    def test_ally_killed_lowers_morale(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.apply_event(MoraleEvent(unit_id="u1", event_type=MoraleEventType.ALLY_KILLED))
        engine.tick(0.1)
        assert engine.get_morale("u1") < 75.0

    def test_suppression_lowers_morale(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.apply_event(MoraleEvent(unit_id="u1", event_type=MoraleEventType.SUPPRESSED))
        engine.tick(0.1)
        assert engine.get_morale("u1") < 75.0

    def test_engagement_won_raises_morale(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=50.0)
        engine.apply_event(MoraleEvent(unit_id="u1", event_type=MoraleEventType.ENGAGEMENT_WON))
        engine.tick(0.1)
        assert engine.get_morale("u1") > 50.0

    def test_magnitude_multiplier(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.register_unit("u2", starting_morale=75.0)
        engine.apply_event(MoraleEvent(unit_id="u1", event_type=MoraleEventType.TOOK_DAMAGE, magnitude=1.0))
        engine.apply_event(MoraleEvent(unit_id="u2", event_type=MoraleEventType.TOOK_DAMAGE, magnitude=2.0))
        engine.tick(0.1)
        # u2 should be lower than u1 because magnitude is higher
        assert engine.get_morale("u2") < engine.get_morale("u1")

    def test_string_event_type(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.apply_event(MoraleEvent(unit_id="u1", event_type="took_damage"))
        engine.tick(0.1)
        assert engine.get_morale("u1") < 75.0

    def test_events_via_tick_param(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        events = [MoraleEvent(unit_id="u1", event_type=MoraleEventType.TOOK_DAMAGE)]
        engine.tick(0.1, events=events)
        assert engine.get_morale("u1") < 75.0

    def test_morale_clamped_at_zero(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=10.0)
        for _ in range(10):
            engine.apply_event(MoraleEvent(
                unit_id="u1",
                event_type=MoraleEventType.ENGAGEMENT_LOST,
                magnitude=2.0,
            ))
        engine.tick(0.1)
        assert engine.get_morale("u1") == 0.0

    def test_morale_clamped_at_100(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=95.0)
        for _ in range(10):
            engine.apply_event(MoraleEvent(
                unit_id="u1",
                event_type=MoraleEventType.REINFORCEMENTS,
                magnitude=2.0,
            ))
        engine.tick(0.1)
        assert engine.get_morale("u1") == 100.0

    def test_dead_unit_ignores_events(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.mark_dead("u1")
        engine.apply_event(MoraleEvent(unit_id="u1", event_type=MoraleEventType.REINFORCEMENTS))
        engine.tick(0.1)
        assert engine.get_morale("u1") == 0.0


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_high_morale_state(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=80.0)
        engine.tick(0.1)
        assert engine.get_state("u1") == MoraleState.HIGH

    def test_fanatical_state(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=95.0)
        engine.tick(0.1)
        assert engine.get_state("u1") == MoraleState.FANATICAL

    def test_shaken_state(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=40.0)
        engine.tick(0.1)
        assert engine.get_state("u1") == MoraleState.SHAKEN

    def test_broken_state(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=20.0)
        engine.tick(0.1)
        assert engine.get_state("u1") == MoraleState.BROKEN

    def test_routed_state(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=5.0)
        engine.tick(0.1)
        assert engine.get_state("u1") == MoraleState.ROUTED

    def test_state_change_notification(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.tick(0.1)  # sets initial state to HIGH
        # Now break morale
        for _ in range(5):
            engine.apply_event(MoraleEvent(
                unit_id="u1",
                event_type=MoraleEventType.ENGAGEMENT_LOST,
                magnitude=1.5,
            ))
        notifications = engine.tick(0.1)
        # Should have at least one morale_change notification
        assert any(n["type"] == "morale_change" for n in notifications)


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------

class TestModifiers:
    def test_accuracy_modifier_high(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=80.0)
        engine.tick(0.1)
        assert engine.get_accuracy_modifier("u1") == 1.0

    def test_accuracy_modifier_shaken(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=40.0)
        engine.tick(0.1)
        assert engine.get_accuracy_modifier("u1") == 0.7

    def test_speed_modifier_broken(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=20.0)
        engine.tick(0.1)
        # Broken troops run faster (fleeing)
        assert engine.get_speed_modifier("u1") > 1.0

    def test_should_retreat(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=20.0)
        engine.tick(0.1)
        assert engine.should_retreat("u1") is True

    def test_should_not_retreat_high_morale(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=80.0)
        engine.tick(0.1)
        assert engine.should_retreat("u1") is False

    def test_should_surrender(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=5.0)
        engine.tick(0.1)
        assert engine.should_surrender("u1") is True


# ---------------------------------------------------------------------------
# Commander aura
# ---------------------------------------------------------------------------

class TestCommanderAura:
    def test_commander_boosts_nearby(self, engine: MoraleEngine) -> None:
        engine.register_unit("cmd", alliance="friendly", starting_morale=80.0, is_commander=True)
        engine.register_unit("u1", alliance="friendly", starting_morale=50.0)
        positions = {"cmd": (100.0, 100.0), "u1": (110.0, 100.0)}
        # Tick several times to accumulate bonus
        for _ in range(50):
            engine.tick(0.1, unit_positions=positions)
        assert engine.get_morale("u1") > 50.0

    def test_commander_no_effect_on_far_unit(self, engine: MoraleEngine) -> None:
        engine.register_unit("cmd", alliance="friendly", starting_morale=80.0, is_commander=True)
        engine.register_unit("u1", alliance="friendly", starting_morale=50.0)
        # Place unit far from commander
        positions = {"cmd": (100.0, 100.0), "u1": (500.0, 500.0)}
        engine.tick(0.1, unit_positions=positions)
        # No change (no recovery yet either since time_since_contact is 0.1 < RECOVERY_DELAY)
        assert engine.get_morale("u1") == 50.0


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_morale_recovers_after_delay(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.apply_event(MoraleEvent(unit_id="u1", event_type=MoraleEventType.TOOK_DAMAGE))
        engine.tick(0.1)
        low_morale = engine.get_morale("u1")
        assert low_morale < 75.0
        # Tick past the recovery delay
        for _ in range(int((RECOVERY_DELAY + 5.0) / 0.1)):
            engine.tick(0.1)
        assert engine.get_morale("u1") > low_morale

    def test_recovery_stops_at_base_morale(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.apply_event(MoraleEvent(unit_id="u1", event_type=MoraleEventType.TOOK_DAMAGE))
        engine.tick(0.1)
        # Tick a very long time
        for _ in range(5000):
            engine.tick(0.1)
        # Should recover to base morale but not exceed it
        assert engine.get_morale("u1") <= 75.0


# ---------------------------------------------------------------------------
# Alliance queries
# ---------------------------------------------------------------------------

class TestAllianceQueries:
    def test_alliance_average_morale(self, populated_engine: MoraleEngine) -> None:
        avg = populated_engine.alliance_average_morale("friendly")
        # 4 units at 75 + 1 commander at 80 = (4*75 + 80)/5 = 76
        assert 75.0 <= avg <= 80.0

    def test_alliance_average_empty(self, engine: MoraleEngine) -> None:
        assert engine.alliance_average_morale("friendly") == 0.0


# ---------------------------------------------------------------------------
# Three.js visualization
# ---------------------------------------------------------------------------

class TestThreeJS:
    def test_to_three_js_structure(self, populated_engine: MoraleEngine) -> None:
        populated_engine.tick(0.1)
        viz = populated_engine.to_three_js()
        assert "units" in viz
        assert "alliance_averages" in viz
        assert len(viz["units"]) > 0

    def test_to_three_js_unit_fields(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.tick(0.1)
        viz = engine.to_three_js()
        unit = viz["units"][0]
        assert unit["id"] == "u1"
        assert "morale" in unit
        assert "state" in unit
        assert "aura_color" in unit
        assert "accuracy_mod" in unit
        assert "speed_mod" in unit
        assert "retreating" in unit

    def test_to_three_js_serializable(self, populated_engine: MoraleEngine) -> None:
        populated_engine.tick(0.1)
        import json
        viz = populated_engine.to_three_js()
        # Should be JSON serializable
        serialized = json.dumps(viz)
        assert len(serialized) > 0

    def test_dead_units_excluded_from_viz(self, engine: MoraleEngine) -> None:
        engine.register_unit("u1", starting_morale=75.0)
        engine.register_unit("u2", starting_morale=75.0)
        engine.mark_dead("u1")
        engine.tick(0.1)
        viz = engine.to_three_js()
        ids = [u["id"] for u in viz["units"]]
        assert "u1" not in ids
        assert "u2" in ids

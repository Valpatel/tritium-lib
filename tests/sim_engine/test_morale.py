# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.morale — unit psychology and morale state machine.

Covers: morale events, state transitions, commander aura, natural recovery,
accuracy/speed modifiers, alliance averages, dead units, edge cases.
"""

import pytest

from tritium_lib.sim_engine.morale import (
    COMMANDER_AURA_RADIUS,
    MoraleEngine,
    MoraleEvent,
    MoraleEventType,
    MoraleState,
    RECOVERY_DELAY,
    RECOVERY_RATE,
    UnitMorale,
)


# ---------------------------------------------------------------------------
# Registration and basic queries
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_register_unit(self):
        eng = MoraleEngine()
        um = eng.register_unit("u1", alliance="blue", starting_morale=75.0)
        assert um.unit_id == "u1"
        assert um.alliance == "blue"
        assert um.morale == 75.0

    def test_register_unit_clamps_morale(self):
        """Starting morale is clamped to 0-100."""
        eng = MoraleEngine()
        high = eng.register_unit("u1", starting_morale=150.0)
        low = eng.register_unit("u2", starting_morale=-50.0)
        assert high.morale == 100.0
        assert low.morale == 0.0

    def test_remove_unit(self):
        eng = MoraleEngine()
        eng.register_unit("u1")
        eng.remove_unit("u1")
        assert eng.get_morale("u1") == 0.0

    def test_remove_nonexistent_unit(self):
        """Removing a unit that was never registered is a no-op."""
        eng = MoraleEngine()
        eng.remove_unit("ghost")  # should not raise

    def test_get_morale_unknown_unit(self):
        eng = MoraleEngine()
        assert eng.get_morale("unknown") == 0.0

    def test_get_state_unknown_unit(self):
        eng = MoraleEngine()
        assert eng.get_state("unknown") == MoraleState.ROUTED

    def test_mark_dead(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=80.0)
        eng.mark_dead("u1")
        um = eng.units["u1"]
        assert not um.is_alive
        assert um.morale == 0.0
        assert um.state == MoraleState.ROUTED


# ---------------------------------------------------------------------------
# Morale events and deltas
# ---------------------------------------------------------------------------

class TestMoraleEvents:

    def test_took_damage_reduces_morale(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=75.0)
        eng.apply_event(MoraleEvent(unit_id="u1", event_type="took_damage"))
        eng.tick(0.01)
        assert eng.get_morale("u1") < 75.0

    def test_enemy_killed_increases_morale(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=60.0)
        eng.apply_event(MoraleEvent(unit_id="u1", event_type="enemy_killed"))
        eng.tick(0.01)
        assert eng.get_morale("u1") > 60.0

    def test_magnitude_scales_delta(self):
        """Magnitude multiplier scales the effect."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=75.0)
        eng.register_unit("u2", starting_morale=75.0)
        eng.apply_event(MoraleEvent(unit_id="u1", event_type="took_damage", magnitude=1.0))
        eng.apply_event(MoraleEvent(unit_id="u2", event_type="took_damage", magnitude=2.0))
        eng.tick(0.01)
        # u2 should have taken a bigger morale hit
        assert eng.get_morale("u2") < eng.get_morale("u1")

    def test_event_enum_works(self):
        """Events can use the MoraleEventType enum."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=50.0)
        eng.apply_event(MoraleEvent(
            unit_id="u1",
            event_type=MoraleEventType.REINFORCEMENTS,
        ))
        eng.tick(0.01)
        assert eng.get_morale("u1") > 50.0

    def test_morale_clamped_at_zero(self):
        """Morale cannot go below zero."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=5.0)
        eng.apply_event(MoraleEvent(unit_id="u1", event_type="engagement_lost", magnitude=5.0))
        eng.tick(0.01)
        assert eng.get_morale("u1") == 0.0

    def test_morale_clamped_at_hundred(self):
        """Morale cannot exceed 100."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=98.0)
        eng.apply_event(MoraleEvent(unit_id="u1", event_type="reinforcements", magnitude=5.0))
        eng.tick(0.01)
        assert eng.get_morale("u1") == 100.0

    def test_event_on_dead_unit_ignored(self):
        """Events on dead units have no effect."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=50.0)
        eng.mark_dead("u1")
        eng.apply_event(MoraleEvent(unit_id="u1", event_type="reinforcements"))
        eng.tick(0.01)
        assert eng.get_morale("u1") == 0.0

    def test_event_on_unknown_unit_ignored(self):
        """Events for non-registered units are silently ignored."""
        eng = MoraleEngine()
        eng.apply_event(MoraleEvent(unit_id="ghost", event_type="took_damage"))
        eng.tick(0.01)  # should not raise

    def test_recent_events_history_capped(self):
        """Recent events list is capped at 10."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=90.0)
        for _ in range(15):
            eng.apply_event(MoraleEvent(unit_id="u1", event_type="enemy_killed"))
        eng.tick(0.01)
        assert len(eng.units["u1"].recent_events) == 10


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:

    def test_high_morale_state(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=75.0)
        eng.tick(0.01)
        assert eng.get_state("u1") == MoraleState.HIGH

    def test_fanatical_state(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=95.0)
        eng.tick(0.01)
        assert eng.get_state("u1") == MoraleState.FANATICAL

    def test_shaken_state(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=40.0)
        eng.tick(0.01)
        assert eng.get_state("u1") == MoraleState.SHAKEN

    def test_broken_state(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=20.0)
        eng.tick(0.01)
        assert eng.get_state("u1") == MoraleState.BROKEN

    def test_routed_state(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=5.0)
        eng.tick(0.01)
        assert eng.get_state("u1") == MoraleState.ROUTED

    def test_state_change_notification(self):
        """Transitioning state emits a notification."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=72.0)
        eng.tick(0.01)  # establishes HIGH state
        # Big hit drops to SHAKEN
        eng.apply_event(MoraleEvent(unit_id="u1", event_type="engagement_lost", magnitude=2.0))
        notifications = eng.tick(0.01)
        changes = [n for n in notifications if n["type"] == "morale_change"]
        assert len(changes) == 1
        assert changes[0]["old_state"] == "high"
        assert changes[0]["new_state"] in ("shaken", "broken", "routed")


# ---------------------------------------------------------------------------
# Modifiers — accuracy and speed
# ---------------------------------------------------------------------------

class TestModifiers:

    def test_high_morale_normal_modifiers(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=75.0)
        eng.tick(0.01)
        assert eng.get_accuracy_modifier("u1") == 1.0
        assert eng.get_speed_modifier("u1") == 1.0

    def test_broken_morale_bad_accuracy_fast_speed(self):
        """Broken troops have terrible accuracy but run fast."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=20.0)
        eng.tick(0.01)
        assert eng.get_accuracy_modifier("u1") == 0.4
        assert eng.get_speed_modifier("u1") == 1.2

    def test_fanatical_morale_bonuses(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=95.0)
        eng.tick(0.01)
        assert eng.get_accuracy_modifier("u1") == 1.1
        assert eng.get_speed_modifier("u1") == 1.1

    def test_unknown_unit_modifiers(self):
        eng = MoraleEngine()
        assert eng.get_accuracy_modifier("unknown") == 0.0
        assert eng.get_speed_modifier("unknown") == 1.0


# ---------------------------------------------------------------------------
# Commander aura
# ---------------------------------------------------------------------------

class TestCommanderAura:

    def test_commander_aura_boosts_nearby_unit(self):
        """Units near a commander gain morale over time."""
        eng = MoraleEngine()
        eng.register_unit("cmd", alliance="blue", starting_morale=90.0, is_commander=True)
        eng.register_unit("troop", alliance="blue", starting_morale=50.0)
        positions = {
            "cmd": (100.0, 100.0),
            "troop": (110.0, 100.0),  # 10m away, within aura
        }
        for _ in range(50):
            eng.tick(0.1, unit_positions=positions)
        assert eng.get_morale("troop") > 50.0

    def test_commander_aura_does_not_affect_far_units(self):
        """Units outside the aura radius get no commander bonus."""
        eng = MoraleEngine()
        eng.register_unit("cmd", alliance="blue", starting_morale=90.0, is_commander=True)
        eng.register_unit("troop", alliance="blue", starting_morale=50.0)
        positions = {
            "cmd": (0.0, 0.0),
            "troop": (1000.0, 1000.0),  # far away
        }
        eng.tick(1.0, unit_positions=positions)
        # Without aura and without recovery (time_since_contact only 1s < 10s delay),
        # morale should not change
        assert eng.get_morale("troop") == pytest.approx(50.0, abs=0.1)

    def test_commander_aura_ignores_enemy_alliance(self):
        """Commander does not boost enemy units."""
        eng = MoraleEngine()
        eng.register_unit("cmd", alliance="blue", starting_morale=90.0, is_commander=True)
        eng.register_unit("enemy", alliance="red", starting_morale=50.0)
        positions = {
            "cmd": (100.0, 100.0),
            "enemy": (105.0, 100.0),  # 5m away
        }
        eng.tick(1.0, unit_positions=positions)
        # Enemy morale should not increase from blue commander
        assert eng.get_morale("enemy") == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

class TestRecovery:

    def test_recovery_after_delay(self):
        """Morale recovers toward base after RECOVERY_DELAY seconds without contact."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=75.0)
        # Drop morale
        eng.apply_event(MoraleEvent(unit_id="u1", event_type="took_damage", magnitude=2.0))
        eng.tick(0.01)
        low_morale = eng.get_morale("u1")
        assert low_morale < 75.0
        # Tick past recovery delay
        for _ in range(200):
            eng.tick(0.1)  # 20 seconds total
        assert eng.get_morale("u1") > low_morale

    def test_no_recovery_during_contact(self):
        """Continuous combat events prevent recovery."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=50.0)
        eng.apply_event(MoraleEvent(unit_id="u1", event_type="suppressed"))
        eng.tick(0.01)
        morale_after_hit = eng.get_morale("u1")
        # Keep applying events to reset time_since_contact
        for _ in range(50):
            eng.apply_event(MoraleEvent(unit_id="u1", event_type="suppressed", magnitude=0.01))
            eng.tick(0.2)
        # Morale should not have recovered above where it was
        # (it may have dropped further due to suppressed events)
        assert eng.get_morale("u1") <= morale_after_hit + 1.0  # small tolerance for rounding


# ---------------------------------------------------------------------------
# Retreat and surrender queries
# ---------------------------------------------------------------------------

class TestRetreatSurrender:

    def test_should_retreat_when_broken(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=15.0)
        eng.tick(0.01)
        assert eng.should_retreat("u1")

    def test_should_not_retreat_when_steady(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=60.0)
        eng.tick(0.01)
        assert not eng.should_retreat("u1")

    def test_should_surrender_when_routed(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=5.0)
        eng.tick(0.01)
        assert eng.should_surrender("u1")

    def test_should_not_surrender_when_broken(self):
        """Broken but not routed — retreats but does not surrender."""
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=20.0)
        eng.tick(0.01)
        assert eng.should_retreat("u1")
        assert not eng.should_surrender("u1")


# ---------------------------------------------------------------------------
# Alliance average
# ---------------------------------------------------------------------------

class TestAllianceAverage:

    def test_average_morale(self):
        eng = MoraleEngine()
        eng.register_unit("u1", alliance="blue", starting_morale=80.0)
        eng.register_unit("u2", alliance="blue", starting_morale=60.0)
        eng.register_unit("u3", alliance="red", starting_morale=90.0)
        avg = eng.alliance_average_morale("blue")
        assert avg == pytest.approx(70.0)

    def test_average_excludes_dead(self):
        eng = MoraleEngine()
        eng.register_unit("u1", alliance="blue", starting_morale=80.0)
        eng.register_unit("u2", alliance="blue", starting_morale=60.0)
        eng.mark_dead("u2")
        avg = eng.alliance_average_morale("blue")
        assert avg == pytest.approx(80.0)

    def test_average_empty_alliance(self):
        eng = MoraleEngine()
        assert eng.alliance_average_morale("ghost") == 0.0


# ---------------------------------------------------------------------------
# Three.js export
# ---------------------------------------------------------------------------

class TestThreeJsExport:

    def test_export_structure(self):
        eng = MoraleEngine()
        eng.register_unit("u1", alliance="blue", starting_morale=75.0)
        eng.tick(0.01)
        export = eng.to_three_js()
        assert "units" in export
        assert "alliance_averages" in export
        assert len(export["units"]) == 1
        unit = export["units"][0]
        assert unit["id"] == "u1"
        assert unit["state"] == "high"
        assert "aura_color" in unit

    def test_export_excludes_dead(self):
        eng = MoraleEngine()
        eng.register_unit("u1", alliance="blue", starting_morale=75.0)
        eng.mark_dead("u1")
        export = eng.to_three_js()
        assert len(export["units"]) == 0

    def test_export_retreating_flag(self):
        eng = MoraleEngine()
        eng.register_unit("u1", starting_morale=15.0)
        eng.tick(0.01)
        export = eng.to_three_js()
        assert export["units"][0]["retreating"] is True

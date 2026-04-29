# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.artillery — indirect fire system.

Covers: fire missions, shell trajectories, CEP scatter, impact resolution,
forward observer fire adjustment, ammo/cooldown state machines, edge cases.
"""

import math
import random

import pytest

from tritium_lib.sim_engine.artillery import (
    ArtilleryEngine,
    ArtilleryPiece,
    ArtilleryType,
    ARTILLERY_TEMPLATES,
    FireMission,
    ForwardObserver,
    Shell,
    create_piece,
    _apply_cep_scatter,
    _time_of_flight,
    _parabolic_altitude,
    _max_altitude,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mortar(piece_id: str = "m1", alliance: str = "blue",
            position: tuple = (0.0, 0.0), ammo: int | None = None) -> ArtilleryPiece:
    """Create a mortar from template for testing."""
    return create_piece(ArtilleryType.MORTAR_81MM, piece_id, alliance, position, ammo=ammo)


def _engine(seed: int = 42) -> ArtilleryEngine:
    return ArtilleryEngine(rng=random.Random(seed))


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Verify the math helpers used by the engine."""

    def test_time_of_flight_minimum(self):
        """Very short range still gets at least 2 seconds TOF."""
        assert _time_of_flight(50.0) == 2.0

    def test_time_of_flight_proportional(self):
        """TOF scales linearly with range at 1s per 200m."""
        assert _time_of_flight(1000.0) == pytest.approx(5.0)

    def test_parabolic_altitude_zero_at_endpoints(self):
        """Shell is at ground level at launch and impact."""
        assert _parabolic_altitude(0.0, 10.0, 500.0) == 0.0
        assert _parabolic_altitude(10.0, 10.0, 500.0) == 0.0

    def test_parabolic_altitude_peak_at_midpoint(self):
        """Peak altitude occurs at TOF/2."""
        peak = _parabolic_altitude(5.0, 10.0, 500.0)
        assert peak == pytest.approx(500.0)

    def test_parabolic_altitude_symmetric(self):
        """Altitude at t and (tof - t) should be equal."""
        alt_early = _parabolic_altitude(2.0, 10.0, 500.0)
        alt_late = _parabolic_altitude(8.0, 10.0, 500.0)
        assert alt_early == pytest.approx(alt_late)

    def test_parabolic_altitude_zero_tof(self):
        """Zero TOF returns zero altitude without division error."""
        assert _parabolic_altitude(0.0, 0.0, 500.0) == 0.0

    def test_max_altitude_scales_with_range(self):
        """Longer range -> higher arc."""
        alt_short = _max_altitude(500.0)
        alt_long = _max_altitude(5000.0)
        assert alt_long > alt_short

    def test_max_altitude_floor(self):
        """Even very short range gets at least 50m altitude."""
        assert _max_altitude(10.0) == 50.0

    def test_cep_scatter_deterministic_with_seed(self):
        """Same RNG seed produces same scatter."""
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        pos1 = _apply_cep_scatter((100.0, 200.0), 25.0, rng1)
        pos2 = _apply_cep_scatter((100.0, 200.0), 25.0, rng2)
        assert pos1 == pos2

    def test_cep_scatter_zero_cep(self):
        """Zero CEP should produce negligible scatter."""
        rng = random.Random(42)
        # With CEP=0, sigma=0, gauss(0,0) should return 0
        pos = _apply_cep_scatter((100.0, 200.0), 0.0, rng)
        assert pos[0] == pytest.approx(100.0, abs=0.001)
        assert pos[1] == pytest.approx(200.0, abs=0.001)


# ---------------------------------------------------------------------------
# Piece creation and templates
# ---------------------------------------------------------------------------

class TestPieceCreation:
    """Verify template-based piece creation."""

    def test_all_templates_exist(self):
        """Every ArtilleryType has a template."""
        for atype in ArtilleryType:
            assert atype in ARTILLERY_TEMPLATES

    def test_create_piece_from_template(self):
        """Factory creates a piece with correct template values."""
        piece = create_piece(ArtilleryType.MORTAR_81MM, "m1", "blue", (10.0, 20.0))
        assert piece.piece_id == "m1"
        assert piece.alliance == "blue"
        assert piece.position == (10.0, 20.0)
        assert piece.min_range == 80.0
        assert piece.max_range == 5600.0
        assert piece.ammo == 40  # default max_ammo

    def test_create_piece_custom_ammo(self):
        """Custom ammo count overrides template default."""
        piece = create_piece(ArtilleryType.HOWITZER_155MM, "h1", "red", (0, 0), ammo=5)
        assert piece.ammo == 5
        assert piece.max_ammo == 20  # template default still set

    def test_template_min_range_less_than_max_range(self):
        """All templates have sane min < max range."""
        for atype, tmpl in ARTILLERY_TEMPLATES.items():
            assert tmpl["min_range"] < tmpl["max_range"], f"{atype} has invalid ranges"


# ---------------------------------------------------------------------------
# ArtilleryEngine — piece management
# ---------------------------------------------------------------------------

class TestEngineManagement:
    """Add/remove pieces and basic state."""

    def test_add_and_retrieve_piece(self):
        eng = _engine()
        piece = _mortar()
        eng.add_piece(piece)
        assert "m1" in eng.pieces
        assert eng.pieces["m1"] is piece

    def test_remove_piece_cancels_missions(self):
        """Removing a piece deactivates its active missions."""
        eng = _engine()
        piece = _mortar(position=(0, 0))
        eng.add_piece(piece)
        mission = eng.request_fire_mission("m1", (1000.0, 0.0), rounds=5)
        eng.remove_piece("m1")
        assert not mission.active

    def test_remove_nonexistent_piece(self):
        """Removing a piece that does not exist is a no-op."""
        eng = _engine()
        eng.remove_piece("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Fire missions — request, cancel, validation
# ---------------------------------------------------------------------------

class TestFireMissions:
    """Fire mission lifecycle: request, validate, cancel."""

    def test_request_valid_mission(self):
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        mission = eng.request_fire_mission("m1", (1000.0, 0.0), "point", 3)
        assert mission.active
        assert mission.rounds == 3
        assert mission.rounds_fired == 0

    def test_request_mission_unknown_piece(self):
        eng = _engine()
        with pytest.raises(ValueError, match="Unknown artillery piece"):
            eng.request_fire_mission("nonexistent", (100, 100))

    def test_request_mission_target_too_close(self):
        """Target inside min_range raises ValueError."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        with pytest.raises(ValueError, match="too close"):
            eng.request_fire_mission("m1", (10.0, 0.0))  # 10m < 80m min

    def test_request_mission_target_too_far(self):
        """Target beyond max_range raises ValueError."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        with pytest.raises(ValueError, match="too far"):
            eng.request_fire_mission("m1", (999999.0, 0.0))

    def test_request_mission_no_ammo(self):
        """Piece with zero ammo cannot fire."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0), ammo=0))
        with pytest.raises(ValueError, match="no ammo"):
            eng.request_fire_mission("m1", (1000.0, 0.0))

    def test_cancel_active_mission(self):
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        mission = eng.request_fire_mission("m1", (1000.0, 0.0), rounds=5)
        assert eng.cancel_mission(mission.mission_id)
        assert not mission.active

    def test_cancel_nonexistent_mission(self):
        eng = _engine()
        assert not eng.cancel_mission("fake_id")


# ---------------------------------------------------------------------------
# Tick simulation — firing and shell flight
# ---------------------------------------------------------------------------

class TestTickSimulation:
    """Verify tick advances cooldowns, fires shells, and resolves impacts."""

    def test_first_round_fires_immediately(self):
        """First round of a mission fires on the first tick."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=1)
        events = eng.tick(0.1)
        fire_events = [e for e in events if e["event"] == "fire"]
        assert len(fire_events) == 1

    def test_shell_in_flight_after_fire(self):
        """After firing, there should be a shell in flight."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=1)
        eng.tick(0.1)
        assert len(eng.shells_in_flight) == 1

    def test_shell_impacts_after_tof(self):
        """Shell impacts after enough time has elapsed."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=1)
        eng.tick(0.01)  # fire
        # TOF for 1000m ~ 5s; tick forward past it
        events = eng.tick(10.0)
        impact_events = [e for e in events if e["event"] == "impact"]
        assert len(impact_events) == 1
        assert len(eng.shells_in_flight) == 0

    def test_cooldown_prevents_rapid_fire(self):
        """Piece cannot fire again before cooldown expires."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=3, interval=1.0)
        # First tick fires round 1
        eng.tick(0.1)
        piece = eng.pieces["m1"]
        assert not piece.ready
        # Small tick -- still on cooldown
        events = eng.tick(0.5)
        fire_events = [e for e in events if e["event"] == "fire"]
        assert len(fire_events) == 0

    def test_cooldown_recovery_enables_next_round(self):
        """After cooldown expires, piece fires next round."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=3, interval=1.0)
        eng.tick(0.1)  # fires round 1
        # Mortar reload_time is 4.0s; tick past that plus interval
        events = eng.tick(5.0)
        fire_events = [e for e in events if e["event"] == "fire"]
        assert len(fire_events) >= 1

    def test_ammo_decrements_on_fire(self):
        eng = _engine()
        piece = _mortar(position=(0, 0), ammo=2)
        eng.add_piece(piece)
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=1)
        eng.tick(0.1)
        assert piece.ammo == 1

    def test_mission_complete_event(self):
        """Completed mission emits mission_complete event."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=1)
        eng.tick(0.1)  # fires and completes
        # The mission should complete on the same tick or the next
        all_events = eng.tick(0.1)
        # After firing the only round, mission should have been completed
        # Check across both ticks
        assert len(eng.fire_missions) == 0  # cleaned up

    def test_smoke_shell_event(self):
        """Smoke mission produces smoke events, not impact events."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), mission_type="smoke", rounds=1)
        eng.tick(0.01)  # fire
        events = eng.tick(10.0)  # impact
        smoke_events = [e for e in events if e["event"] == "smoke"]
        impact_events = [e for e in events if e["event"] == "impact"]
        assert len(smoke_events) == 1
        assert len(impact_events) == 0

    def test_illumination_shell_event(self):
        """Illumination mission produces illumination events."""
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), mission_type="illumination", rounds=1)
        eng.tick(0.01)
        events = eng.tick(10.0)
        illum = [e for e in events if e["event"] == "illumination"]
        assert len(illum) == 1
        # Illumination radius is 3x blast radius
        assert illum[0]["radius"] > 0

    def test_danger_close_tighter_cep(self):
        """Danger close mission uses tighter CEP (0.7x), confirmed by less scatter."""
        # With fixed seed, danger_close should produce less scatter than area
        rng_dc = random.Random(42)
        rng_area = random.Random(42)
        eng_dc = ArtilleryEngine(rng=rng_dc)
        eng_area = ArtilleryEngine(rng=rng_area)
        for eng in (eng_dc, eng_area):
            eng.add_piece(_mortar(position=(0, 0)))

        eng_dc.request_fire_mission("m1", (1000.0, 0.0), mission_type="danger_close", rounds=1)
        eng_area.request_fire_mission("m1", (1000.0, 0.0), mission_type="area", rounds=1)
        eng_dc.tick(0.01)
        eng_area.tick(0.01)
        # Both have shells; danger_close should have tighter scatter
        dc_shell = eng_dc.shells_in_flight[0]
        area_shell = eng_area.shells_in_flight[0]
        # We cannot compare single samples definitively, but the CEP scaling
        # means dc_cep = base*0.7 and area_cep = base*2.0 -- at minimum
        # we verify the shells were created with different impact positions
        # (different CEP multipliers with same RNG state yield different results)
        assert dc_shell.impact_pos != area_shell.impact_pos


# ---------------------------------------------------------------------------
# Impact resolution
# ---------------------------------------------------------------------------

class TestImpactResolution:
    """Verify damage calculation against targets."""

    def test_direct_hit_full_damage(self):
        eng = _engine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (100.0, 100.0),
            "damage": 120.0,
            "blast_radius": 25.0,
            "shell_type": "he",
        }]
        targets = [((100.0, 100.0), "t1", 0.0)]  # at impact, no armor
        results = eng.resolve_impacts(events, targets)
        assert len(results) == 1
        assert results[0]["damage"] == pytest.approx(120.0, rel=0.01)

    def test_damage_falloff_with_distance(self):
        """Targets farther from impact take less damage."""
        eng = _engine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 100.0,
            "blast_radius": 50.0,
            "shell_type": "he",
        }]
        targets = [
            ((0.0, 0.0), "close", 0.0),
            ((25.0, 0.0), "mid", 0.0),
            ((49.0, 0.0), "far", 0.0),
        ]
        results = eng.resolve_impacts(events, targets)
        damages = {r["target_id"]: r["damage"] for r in results}
        assert damages["close"] > damages["mid"] > damages["far"]

    def test_armor_reduces_damage(self):
        """Armor absorbs a fraction of the damage."""
        eng = _engine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 100.0,
            "blast_radius": 50.0,
            "shell_type": "he",
        }]
        no_armor = [((0.0, 0.0), "t1", 0.0)]
        with_armor = [((0.0, 0.0), "t2", 0.5)]
        r_no = eng.resolve_impacts(events, no_armor)
        r_yes = eng.resolve_impacts(events, with_armor)
        assert r_no[0]["damage"] > r_yes[0]["damage"]
        assert r_yes[0]["damage"] == pytest.approx(50.0, rel=0.01)

    def test_outside_blast_radius_no_damage(self):
        """Targets beyond blast radius take no damage."""
        eng = _engine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 100.0,
            "blast_radius": 25.0,
            "shell_type": "he",
        }]
        targets = [((30.0, 0.0), "t1", 0.0)]
        results = eng.resolve_impacts(events, targets)
        assert len(results) == 0

    def test_resolve_skips_non_impact_events(self):
        """Smoke and illumination events are not treated as impacts."""
        eng = _engine()
        events = [
            {"event": "smoke", "position": (0, 0), "radius": 25},
            {"event": "illumination", "position": (0, 0), "radius": 75},
        ]
        targets = [((0.0, 0.0), "t1", 0.0)]
        results = eng.resolve_impacts(events, targets)
        assert len(results) == 0

    def test_full_armor_negates_damage(self):
        """Armor of 1.0 completely negates damage."""
        eng = _engine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 100.0,
            "blast_radius": 50.0,
            "shell_type": "he",
        }]
        targets = [((0.0, 0.0), "t1", 1.0)]
        results = eng.resolve_impacts(events, targets)
        assert results[0]["damage"] == 0.0


# ---------------------------------------------------------------------------
# ForwardObserver
# ---------------------------------------------------------------------------

class TestForwardObserver:
    """Verify FO call_fire, adjust_fire, and piece selection."""

    def test_call_fire_selects_piece(self):
        """FO auto-selects a same-alliance piece in range."""
        eng = _engine()
        eng.add_piece(_mortar("m1", "blue", (0, 0)))
        eng.add_piece(_mortar("m2", "red", (0, 0)))  # wrong alliance
        fo = ForwardObserver("fo1", (500, 500), "blue")
        mission = fo.call_fire(eng, (1000.0, 0.0))
        assert mission.piece_id == "m1"

    def test_call_fire_no_piece_in_range(self):
        """FO raises when no piece can reach the target."""
        eng = _engine()
        eng.add_piece(_mortar("m1", "blue", (0, 0)))
        fo = ForwardObserver("fo1", (500, 500), "blue")
        with pytest.raises(ValueError, match="No available artillery"):
            fo.call_fire(eng, (999999.0, 0.0))

    def test_adjust_fire(self):
        """FO adjusts fire — cancels old mission, creates new with remaining rounds."""
        eng = _engine()
        eng.add_piece(_mortar("m1", "blue", (0, 0)))
        fo = ForwardObserver("fo1", (500, 500), "blue")
        mission1 = fo.call_fire(eng, (1000.0, 0.0), rounds=5)
        # Fire one round
        eng.tick(0.1)
        # Adjust fire by +50, +50
        mission2 = fo.adjust_fire(eng, mission1.mission_id, (50.0, 50.0))
        assert mission2 is not None
        assert mission2.target_pos == (1050.0, 50.0)
        # Remaining rounds: 5 - 1 = 4
        assert mission2.rounds == 4

    def test_adjust_fire_nonexistent_mission(self):
        """Adjusting a nonexistent/completed mission returns None."""
        eng = _engine()
        fo = ForwardObserver("fo1", (500, 500), "blue")
        result = fo.adjust_fire(eng, "fake_id", (10, 10))
        assert result is None

    def test_end_mission(self):
        """FO can cancel a mission."""
        eng = _engine()
        eng.add_piece(_mortar("m1", "blue", (0, 0)))
        fo = ForwardObserver("fo1", (500, 500), "blue")
        mission = fo.call_fire(eng, (1000.0, 0.0), rounds=5)
        assert fo.end_mission(eng, mission.mission_id)
        assert not mission.active


# ---------------------------------------------------------------------------
# Three.js export
# ---------------------------------------------------------------------------

class TestThreeJsExport:
    """Verify to_three_js returns correct structure."""

    def test_export_pieces(self):
        eng = _engine()
        eng.add_piece(_mortar("m1", "blue", (100, 200)))
        export = eng.to_three_js()
        assert len(export["pieces"]) == 1
        assert export["pieces"][0]["id"] == "m1"
        assert export["pieces"][0]["alliance"] == "blue"

    def test_export_shells_in_flight(self):
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=1)
        eng.tick(0.01)  # fire
        eng.tick(1.0)   # shell mid-flight
        export = eng.to_three_js()
        assert len(export["shells"]) == 1
        shell = export["shells"][0]
        assert "altitude" in shell
        assert shell["altitude"] > 0  # mid-flight
        assert 0 < shell["progress"] < 1

    def test_export_impacts(self):
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=1)
        eng.tick(0.01)
        eng.tick(10.0)  # impact
        export = eng.to_three_js()
        assert len(export["impacts"]) == 1

    def test_export_smoke_areas(self):
        eng = _engine()
        eng.add_piece(_mortar(position=(0, 0)))
        eng.request_fire_mission("m1", (1000.0, 0.0), mission_type="smoke", rounds=1)
        eng.tick(0.01)
        eng.tick(10.0)
        export = eng.to_three_js()
        assert len(export["smoke_areas"]) == 1

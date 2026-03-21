# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for artillery.py — ArtilleryEngine fire/reload cycle, ForwardObserver,
and CEP scatter."""

import math
import random

import pytest
from tritium_lib.sim_engine.artillery import (
    ArtilleryEngine,
    ArtilleryPiece,
    ArtilleryType,
    FireMission,
    ForwardObserver,
    Shell,
    ARTILLERY_TEMPLATES,
    create_piece,
    _apply_cep_scatter,
    _time_of_flight,
    _parabolic_altitude,
    _max_altitude,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mortar(piece_id: str = "m1", position=(100.0, 100.0)) -> ArtilleryPiece:
    """Create an 81mm mortar for testing (range 80–5600 m)."""
    return create_piece(ArtilleryType.MORTAR_81MM, piece_id, "blue", position)


def _engine_with_mortar(piece_id: str = "m1", position=(100.0, 100.0)):
    eng = ArtilleryEngine(rng=random.Random(42))
    piece = _mortar(piece_id, position)
    eng.add_piece(piece)
    return eng, piece


# ---------------------------------------------------------------------------
# create_piece / ARTILLERY_TEMPLATES
# ---------------------------------------------------------------------------

class TestCreatePiece:
    def test_mortar_81mm_stats(self):
        piece = _mortar()
        assert piece.artillery_type == ArtilleryType.MORTAR_81MM
        assert piece.min_range == 80.0
        assert piece.max_range == 5600.0
        assert piece.damage == 120.0
        assert piece.blast_radius == 25.0
        assert piece.crew == 3
        assert piece.ready is True

    def test_howitzer_155mm_has_more_range(self):
        hw = create_piece(ArtilleryType.HOWITZER_155MM, "hw1", "red", (0.0, 0.0))
        assert hw.max_range > 20000.0

    def test_mlrs_minimum_range(self):
        ml = create_piece(ArtilleryType.MLRS, "ml1", "blue", (0.0, 0.0))
        assert ml.min_range >= 10000.0

    def test_custom_ammo_override(self):
        piece = create_piece(ArtilleryType.MORTAR_60MM, "p1", "blue", (0.0, 0.0), ammo=5)
        assert piece.ammo == 5

    def test_templates_has_all_types(self):
        for art_type in ArtilleryType:
            assert art_type in ARTILLERY_TEMPLATES


# ---------------------------------------------------------------------------
# ArtilleryEngine — piece management
# ---------------------------------------------------------------------------

class TestArtilleryEngineManagement:
    def test_add_piece(self):
        eng = ArtilleryEngine()
        piece = _mortar()
        eng.add_piece(piece)
        assert "m1" in eng.pieces

    def test_remove_piece(self):
        eng, _ = _engine_with_mortar()
        eng.remove_piece("m1")
        assert "m1" not in eng.pieces

    def test_remove_nonexistent_piece_no_error(self):
        eng = ArtilleryEngine()
        eng.remove_piece("nobody")  # should not raise

    def test_remove_piece_cancels_missions(self):
        eng, piece = _engine_with_mortar()
        target = (100.0, 1000.0)  # 900 m away — within mortar range
        mission = eng.request_fire_mission("m1", target, rounds=5)
        eng.remove_piece("m1")
        assert mission.active is False


# ---------------------------------------------------------------------------
# request_fire_mission validation
# ---------------------------------------------------------------------------

class TestFireMissionValidation:
    def test_unknown_piece_raises(self):
        eng = ArtilleryEngine()
        with pytest.raises(ValueError, match="Unknown artillery piece"):
            eng.request_fire_mission("no_such_piece", (500.0, 500.0))

    def test_target_too_close_raises(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        # mortar min_range = 80 m
        with pytest.raises(ValueError, match="too close"):
            eng.request_fire_mission("m1", (10.0, 0.0))  # 10 m away

    def test_target_too_far_raises(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        # mortar max_range = 5600 m
        with pytest.raises(ValueError, match="too far"):
            eng.request_fire_mission("m1", (10000.0, 0.0))

    def test_no_ammo_raises(self):
        eng = ArtilleryEngine()
        piece = create_piece(ArtilleryType.MORTAR_81MM, "m1", "blue", (0.0, 0.0), ammo=0)
        eng.add_piece(piece)
        with pytest.raises(ValueError, match="no ammo"):
            eng.request_fire_mission("m1", (1000.0, 0.0))

    def test_valid_mission_returns_fire_mission(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        mission = eng.request_fire_mission("m1", (1000.0, 0.0))
        assert isinstance(mission, FireMission)
        assert mission.active is True
        assert mission.rounds == 1


# ---------------------------------------------------------------------------
# cancel_mission
# ---------------------------------------------------------------------------

class TestCancelMission:
    def test_cancel_active_mission(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        mission = eng.request_fire_mission("m1", (1000.0, 0.0), rounds=5)
        result = eng.cancel_mission(mission.mission_id)
        assert result is True
        assert mission.active is False

    def test_cancel_nonexistent_returns_false(self):
        eng = ArtilleryEngine()
        assert eng.cancel_mission("nonexistent_id") is False


# ---------------------------------------------------------------------------
# tick — fire/reload cycle
# ---------------------------------------------------------------------------

class TestArtilleryTick:
    def test_first_round_fires_immediately(self):
        eng, piece = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=1)
        events = eng.tick(0.001)  # tiny dt — first round fires at interval_acc=0
        fire_events = [e for e in events if e["event"] == "fire"]
        assert len(fire_events) == 1

    def test_fire_event_has_expected_keys(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0))
        events = eng.tick(0.001)
        fire = next((e for e in events if e["event"] == "fire"), None)
        assert fire is not None
        assert "piece_id" in fire
        assert "shell_id" in fire
        assert "target" in fire
        assert "impact_pos" in fire

    def test_piece_not_ready_after_firing(self):
        eng, piece = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=5)
        eng.tick(0.001)
        assert piece.ready is False

    def test_ammo_decrements_on_fire(self):
        eng, piece = _engine_with_mortar(position=(0.0, 0.0))
        initial_ammo = piece.ammo
        eng.request_fire_mission("m1", (1000.0, 0.0))
        eng.tick(0.001)
        assert piece.ammo == initial_ammo - 1

    def test_shell_in_flight_after_fire(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0))
        eng.tick(0.001)
        assert len(eng.shells_in_flight) == 1

    def test_piece_reloads_after_cooldown(self):
        eng, piece = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0))
        eng.tick(0.001)  # fire
        assert piece.ready is False
        # MORTAR_81MM reload_time = 4.0 s
        eng.tick(4.5)  # advance past reload
        assert piece.ready is True

    def test_mission_complete_event_fires(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=1)
        all_events = []
        for _ in range(200):
            all_events.extend(eng.tick(0.1))
        completed = [e for e in all_events if e.get("event") == "mission_complete"]
        assert len(completed) >= 1

    def test_impact_event_fires_after_flight(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0))
        all_events = []
        for _ in range(300):  # tick until shell lands
            all_events.extend(eng.tick(0.1))
        impacts = [e for e in all_events if e.get("event") == "impact"]
        assert len(impacts) >= 1

    def test_impact_event_has_damage(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0))
        all_events = []
        for _ in range(300):
            all_events.extend(eng.tick(0.1))
        impact = next((e for e in all_events if e.get("event") == "impact"), None)
        if impact:
            assert impact["damage"] > 0.0
            assert "blast_radius" in impact

    def test_multi_round_mission(self):
        eng, piece = _engine_with_mortar(position=(0.0, 0.0))
        piece.ammo = 5
        eng.request_fire_mission("m1", (1000.0, 0.0), rounds=3, interval=0.5)
        all_events = []
        for _ in range(500):
            all_events.extend(eng.tick(0.1))
        fire_events = [e for e in all_events if e.get("event") == "fire"]
        assert len(fire_events) == 3

    def test_smoke_mission_type(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0), mission_type="smoke")
        all_events = []
        for _ in range(300):
            all_events.extend(eng.tick(0.1))
        smoke = [e for e in all_events if e.get("event") == "smoke"]
        assert len(smoke) >= 1

    def test_illumination_mission_type(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0), mission_type="illumination")
        all_events = []
        for _ in range(300):
            all_events.extend(eng.tick(0.1))
        illum = [e for e in all_events if e.get("event") == "illumination"]
        assert len(illum) >= 1

    def test_shells_cleared_after_impact(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0))
        for _ in range(300):
            eng.tick(0.1)
        # All shells should have landed
        assert len(eng.shells_in_flight) == 0


# ---------------------------------------------------------------------------
# Parabolic shell physics helpers
# ---------------------------------------------------------------------------

class TestShellPhysics:
    def test_time_of_flight_minimum(self):
        assert _time_of_flight(100.0) >= 2.0

    def test_time_of_flight_scales_with_range(self):
        t1 = _time_of_flight(500.0)
        t2 = _time_of_flight(2000.0)
        assert t2 > t1

    def test_parabolic_altitude_zero_at_start(self):
        alt = _parabolic_altitude(0.0, 10.0, 200.0)
        assert alt == pytest.approx(0.0)

    def test_parabolic_altitude_zero_at_end(self):
        alt = _parabolic_altitude(10.0, 10.0, 200.0)
        assert alt == pytest.approx(0.0)

    def test_parabolic_altitude_max_at_midpoint(self):
        tof = 10.0
        max_alt = 100.0
        mid_alt = _parabolic_altitude(5.0, tof, max_alt)
        assert mid_alt == pytest.approx(max_alt, abs=0.1)

    def test_parabolic_altitude_non_negative(self):
        for t in range(11):
            alt = _parabolic_altitude(float(t), 10.0, 50.0)
            assert alt >= 0.0

    def test_max_altitude_positive(self):
        assert _max_altitude(1000.0) > 0.0

    def test_max_altitude_scales_with_range(self):
        assert _max_altitude(5000.0) > _max_altitude(1000.0)


# ---------------------------------------------------------------------------
# CEP scatter
# ---------------------------------------------------------------------------

class TestCEPScatter:
    def test_cep_scatter_returns_tuple(self):
        result = _apply_cep_scatter((0.0, 0.0), 25.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cep_scatter_is_random(self):
        results = set()
        rng = random.Random(1)
        for _ in range(10):
            r = _apply_cep_scatter((100.0, 100.0), 30.0, rng)
            results.add(r)
        # Should not all be identical
        assert len(results) > 1

    def test_cep_50_pct_within_radius(self):
        """Statistically, ~50% of shots should land within CEP radius."""
        rng = random.Random(99)
        cep = 50.0
        target = (0.0, 0.0)
        n = 1000
        within = sum(
            1 for _ in range(n)
            if math.hypot(*_apply_cep_scatter(target, cep, rng)) <= cep
        )
        ratio = within / n
        # CEP is 50th percentile — should be roughly 50% within
        assert 0.40 <= ratio <= 0.65


# ---------------------------------------------------------------------------
# resolve_impacts
# ---------------------------------------------------------------------------

class TestResolveImpacts:
    def test_resolve_target_in_blast_radius(self):
        eng = ArtilleryEngine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (100.0, 100.0),
            "damage": 120.0,
            "blast_radius": 25.0,
            "shell_type": "he",
        }]
        targets = [((105.0, 100.0), "t1", 0.0)]
        results = eng.resolve_impacts(events, targets)
        assert len(results) == 1
        assert results[0]["target_id"] == "t1"
        assert results[0]["damage"] > 0.0

    def test_resolve_target_outside_blast_radius(self):
        eng = ArtilleryEngine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 120.0,
            "blast_radius": 25.0,
            "shell_type": "he",
        }]
        targets = [((200.0, 200.0), "t1", 0.0)]
        results = eng.resolve_impacts(events, targets)
        assert len(results) == 0

    def test_resolve_armor_reduces_damage(self):
        eng = ArtilleryEngine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (100.0, 100.0),
            "damage": 100.0,
            "blast_radius": 25.0,
            "shell_type": "he",
        }]
        # Direct hit (same position), armor 0.5
        targets_no_armor = [((100.0, 100.0), "t1", 0.0)]
        targets_with_armor = [((100.0, 100.0), "t2", 0.5)]
        r_no = eng.resolve_impacts(events, targets_no_armor)
        r_armored = eng.resolve_impacts(events, targets_with_armor)
        assert r_armored[0]["damage"] < r_no[0]["damage"]

    def test_damage_linear_falloff(self):
        eng = ArtilleryEngine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 100.0,
            "blast_radius": 50.0,
            "shell_type": "he",
        }]
        close = [((5.0, 0.0), "close", 0.0)]
        far = [((40.0, 0.0), "far", 0.0)]
        r_close = eng.resolve_impacts(events, close)
        r_far = eng.resolve_impacts(events, far)
        assert r_close[0]["damage"] > r_far[0]["damage"]


# ---------------------------------------------------------------------------
# to_three_js export
# ---------------------------------------------------------------------------

class TestArtilleryToThreeJs:
    def test_to_three_js_structure(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0))
        eng.tick(0.1)  # fire one shell
        result = eng.to_three_js()
        assert "shells" in result
        assert "impacts" in result
        assert "smoke_areas" in result
        assert "illumination_areas" in result
        assert "pieces" in result

    def test_shell_in_flight_appears_in_three_js(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0))
        eng.tick(0.001)
        result = eng.to_three_js()
        assert len(result["shells"]) == 1

    def test_shell_has_altitude_and_progress(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        eng.request_fire_mission("m1", (1000.0, 0.0))
        eng.tick(0.001)
        shell_data = eng.to_three_js()["shells"][0]
        assert "altitude" in shell_data
        assert "progress" in shell_data
        assert 0.0 <= shell_data["progress"] <= 1.0


# ---------------------------------------------------------------------------
# ForwardObserver
# ---------------------------------------------------------------------------

class TestForwardObserver:
    def test_fo_creation(self):
        fo = ForwardObserver("fo1", (300.0, 300.0), "blue")
        assert fo.observer_id == "fo1"
        assert fo.alliance == "blue"

    def test_fo_call_fire_returns_mission(self):
        eng, _ = _engine_with_mortar(position=(0.0, 0.0))
        fo = ForwardObserver("fo1", (0.0, 0.0), "blue")
        mission = fo.call_fire(eng, (1000.0, 0.0), "point", 3)
        assert isinstance(mission, FireMission)
        assert mission.rounds == 3

    def test_fo_call_fire_no_suitable_piece_raises(self):
        eng = ArtilleryEngine()
        fo = ForwardObserver("fo1", (0.0, 0.0), "blue")
        with pytest.raises(ValueError, match="No available artillery piece"):
            fo.call_fire(eng, (1000.0, 0.0))

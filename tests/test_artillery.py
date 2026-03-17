# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the artillery and indirect fire system."""

from __future__ import annotations

import math
import random

import pytest

from tritium_lib.sim_engine.artillery import (
    ARTILLERY_TEMPLATES,
    ArtilleryEngine,
    ArtilleryPiece,
    ArtilleryType,
    FireMission,
    ForwardObserver,
    Shell,
    _apply_cep_scatter,
    _max_altitude,
    _parabolic_altitude,
    _time_of_flight,
    create_piece,
)
from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mortar(piece_id: str = "m1", alliance: str = "blue",
            pos: Vec2 = (0.0, 0.0), ammo: int | None = None) -> ArtilleryPiece:
    """Create a standard 81mm mortar for testing."""
    return create_piece(ArtilleryType.MORTAR_81MM, piece_id, alliance, pos, ammo=ammo)


def _engine(seed: int = 42) -> ArtilleryEngine:
    return ArtilleryEngine(rng=random.Random(seed))


def _target_in_range(piece: ArtilleryPiece) -> Vec2:
    """Return a target position within the piece's range envelope."""
    mid = (piece.min_range + piece.max_range) / 2.0
    return (piece.position[0] + mid, piece.position[1])


# ---------------------------------------------------------------------------
# ArtilleryType enum
# ---------------------------------------------------------------------------

class TestArtilleryType:
    def test_all_six_types(self):
        assert len(ArtilleryType) == 6

    def test_mortar_60mm_value(self):
        assert ArtilleryType.MORTAR_60MM.value == "mortar_60mm"

    def test_mortar_81mm_value(self):
        assert ArtilleryType.MORTAR_81MM.value == "mortar_81mm"

    def test_howitzer_105mm_value(self):
        assert ArtilleryType.HOWITZER_105MM.value == "howitzer_105mm"

    def test_howitzer_155mm_value(self):
        assert ArtilleryType.HOWITZER_155MM.value == "howitzer_155mm"

    def test_mlrs_value(self):
        assert ArtilleryType.MLRS.value == "mlrs"

    def test_naval_gun_value(self):
        assert ArtilleryType.NAVAL_GUN.value == "naval_gun"


# ---------------------------------------------------------------------------
# ARTILLERY_TEMPLATES
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_six_templates(self):
        assert len(ARTILLERY_TEMPLATES) == 6

    def test_all_types_present(self):
        for t in ArtilleryType:
            assert t in ARTILLERY_TEMPLATES

    def test_mortar_60mm_range(self):
        t = ARTILLERY_TEMPLATES[ArtilleryType.MORTAR_60MM]
        assert t["min_range"] < t["max_range"]

    def test_howitzer_155_has_large_blast(self):
        t = ARTILLERY_TEMPLATES[ArtilleryType.HOWITZER_155MM]
        assert t["blast_radius"] >= 50.0

    def test_mlrs_long_range(self):
        t = ARTILLERY_TEMPLATES[ArtilleryType.MLRS]
        assert t["max_range"] >= 50000.0

    def test_templates_have_required_keys(self):
        required = {"min_range", "max_range", "damage", "blast_radius",
                     "reload_time", "max_ammo", "accuracy_cep", "crew"}
        for typ, tmpl in ARTILLERY_TEMPLATES.items():
            assert required.issubset(tmpl.keys()), f"{typ} missing keys"


# ---------------------------------------------------------------------------
# create_piece
# ---------------------------------------------------------------------------

class TestCreatePiece:
    def test_creates_mortar(self):
        p = create_piece(ArtilleryType.MORTAR_81MM, "m1", "blue", (10, 20))
        assert p.piece_id == "m1"
        assert p.artillery_type == ArtilleryType.MORTAR_81MM
        assert p.alliance == "blue"
        assert p.position == (10, 20)

    def test_default_ammo_equals_max(self):
        p = create_piece(ArtilleryType.MORTAR_60MM, "m1", "red", (0, 0))
        assert p.ammo == p.max_ammo

    def test_custom_ammo(self):
        p = create_piece(ArtilleryType.HOWITZER_105MM, "h1", "blue", (0, 0), ammo=5)
        assert p.ammo == 5
        assert p.max_ammo == 30

    def test_all_types_create(self):
        for t in ArtilleryType:
            p = create_piece(t, f"test_{t.value}", "blue", (0, 0))
            assert p.artillery_type == t
            assert p.ready is True
            assert p.cooldown == 0.0


# ---------------------------------------------------------------------------
# ArtilleryPiece dataclass
# ---------------------------------------------------------------------------

class TestArtilleryPiece:
    def test_defaults(self):
        p = _mortar()
        assert p.ready is True
        assert p.cooldown == 0.0
        assert p.crew == 3

    def test_heading(self):
        p = create_piece(ArtilleryType.MORTAR_81MM, "m1", "blue", (0, 0), heading=1.57)
        assert abs(p.heading - 1.57) < 0.01


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_cep_scatter_center_tendency(self):
        """Average scatter should be near zero over many samples."""
        rng = random.Random(123)
        dx_sum = 0.0
        dy_sum = 0.0
        n = 10000
        for _ in range(n):
            pos = _apply_cep_scatter((100.0, 200.0), 25.0, rng)
            dx_sum += pos[0] - 100.0
            dy_sum += pos[1] - 200.0
        assert abs(dx_sum / n) < 2.0
        assert abs(dy_sum / n) < 2.0

    def test_cep_scatter_zero_cep(self):
        pos = _apply_cep_scatter((50.0, 50.0), 0.0, random.Random(1))
        # With CEP=0, sigma=0, all shots land at target
        assert pos == (50.0, 50.0)

    def test_time_of_flight_minimum(self):
        assert _time_of_flight(0.0) == 2.0
        assert _time_of_flight(100.0) == 2.0

    def test_time_of_flight_scales(self):
        assert _time_of_flight(1000.0) == 5.0
        assert _time_of_flight(4000.0) == 20.0

    def test_parabolic_altitude_endpoints(self):
        assert _parabolic_altitude(0.0, 10.0, 100.0) == 0.0
        assert _parabolic_altitude(10.0, 10.0, 100.0) == 0.0

    def test_parabolic_altitude_peak(self):
        alt = _parabolic_altitude(5.0, 10.0, 100.0)
        assert abs(alt - 100.0) < 0.01

    def test_max_altitude_minimum(self):
        assert _max_altitude(0.0) == 50.0

    def test_max_altitude_scales(self):
        assert _max_altitude(1000.0) == 300.0


# ---------------------------------------------------------------------------
# ArtilleryEngine — add/remove
# ---------------------------------------------------------------------------

class TestEngineManagement:
    def test_add_piece(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        assert "m1" in e.pieces

    def test_remove_piece(self):
        e = _engine()
        e.add_piece(_mortar())
        e.remove_piece("m1")
        assert "m1" not in e.pieces

    def test_remove_nonexistent(self):
        e = _engine()
        e.remove_piece("nope")  # should not raise

    def test_remove_cancels_missions(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        m = e.request_fire_mission("m1", target, "point", 5)
        e.remove_piece("m1")
        assert m.active is False


# ---------------------------------------------------------------------------
# ArtilleryEngine — fire missions
# ---------------------------------------------------------------------------

class TestFireMissions:
    def test_request_fire_mission(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        m = e.request_fire_mission("m1", target, "point", 3)
        assert isinstance(m, FireMission)
        assert m.rounds == 3
        assert m.rounds_fired == 0
        assert m.active is True

    def test_target_too_close(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        with pytest.raises(ValueError, match="too close"):
            e.request_fire_mission("m1", (10.0, 0.0), "point", 1)

    def test_target_too_far(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        with pytest.raises(ValueError, match="too far"):
            e.request_fire_mission("m1", (999999.0, 0.0), "point", 1)

    def test_unknown_piece(self):
        e = _engine()
        with pytest.raises(ValueError, match="Unknown"):
            e.request_fire_mission("nope", (100.0, 0.0), "point", 1)

    def test_no_ammo(self):
        e = _engine()
        p = _mortar(ammo=0)
        e.add_piece(p)
        target = _target_in_range(p)
        with pytest.raises(ValueError, match="no ammo"):
            e.request_fire_mission("m1", target, "point", 1)

    def test_cancel_mission(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        m = e.request_fire_mission("m1", target, "point", 5)
        assert e.cancel_mission(m.mission_id) is True
        assert m.active is False

    def test_cancel_nonexistent(self):
        e = _engine()
        assert e.cancel_mission("nope") is False

    def test_mission_types(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        for mt in ("point", "area", "barrage", "smoke", "illumination", "danger_close"):
            # Reset ammo for each
            p.ammo = p.max_ammo
            m = e.request_fire_mission("m1", target, mt, 1)
            assert m.mission_type == mt
            # Tick to complete
            for _ in range(100):
                e.tick(0.5)
            p.ready = True
            p.cooldown = 0.0


# ---------------------------------------------------------------------------
# ArtilleryEngine — tick
# ---------------------------------------------------------------------------

class TestEngineTick:
    def test_tick_fires_first_round_immediately(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "point", 1)
        events = e.tick(0.1)
        fire_events = [ev for ev in events if ev["event"] == "fire"]
        assert len(fire_events) == 1

    def test_tick_spawns_shell(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "point", 1)
        e.tick(0.1)
        assert len(e.shells_in_flight) == 1

    def test_shell_impacts_after_tof(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "point", 1)
        e.tick(0.1)  # fire
        assert len(e.shells_in_flight) == 1
        # Tick until impact
        impact_events = []
        for _ in range(200):
            evts = e.tick(0.5)
            impact_events.extend(ev for ev in evts if ev["event"] == "impact")
            if impact_events:
                break
        assert len(impact_events) == 1

    def test_cooldown_prevents_rapid_fire(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "barrage", 3, interval=0.1)
        events = e.tick(0.1)
        fire_events = [ev for ev in events if ev["event"] == "fire"]
        # Should fire only 1 round (cooldown blocks second)
        assert len(fire_events) == 1
        assert p.ready is False

    def test_cooldown_expires(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "point", 2, interval=0.1)
        e.tick(0.1)  # fires round 1
        assert p.ready is False
        # Tick past reload time
        for _ in range(50):
            evts = e.tick(0.5)
            fire_events = [ev for ev in evts if ev["event"] == "fire"]
            if fire_events:
                break
        assert p.ammo == p.max_ammo - 2

    def test_ammo_decrements(self):
        e = _engine()
        p = _mortar()
        initial_ammo = p.ammo
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "point", 1)
        e.tick(0.1)
        assert p.ammo == initial_ammo - 1

    def test_mission_completes(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        m = e.request_fire_mission("m1", target, "point", 1)
        complete_events = []
        for _ in range(200):
            evts = e.tick(0.5)
            complete_events.extend(ev for ev in evts if ev["event"] == "mission_complete")
            if complete_events:
                break
        assert len(complete_events) >= 1
        assert complete_events[0]["mission_id"] == m.mission_id

    def test_smoke_event(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "smoke", 1)
        smoke_events = []
        for _ in range(200):
            evts = e.tick(0.5)
            smoke_events.extend(ev for ev in evts if ev["event"] == "smoke")
            if smoke_events:
                break
        assert len(smoke_events) == 1
        assert "radius" in smoke_events[0]

    def test_illumination_event(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "illumination", 1)
        illum_events = []
        for _ in range(200):
            evts = e.tick(0.5)
            illum_events.extend(ev for ev in evts if ev["event"] == "illumination")
            if illum_events:
                break
        assert len(illum_events) == 1
        # Illumination radius is 3x blast radius
        assert illum_events[0]["radius"] == p.blast_radius * 3.0

    def test_shell_altitude_rises_and_falls(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "point", 1)
        e.tick(0.1)  # fire
        shell = e.shells_in_flight[0]
        altitudes = []
        for _ in range(100):
            e.tick(0.2)
            altitudes.append(shell.altitude)
            if shell not in e.shells_in_flight:
                break
        # Should rise then fall
        max_idx = altitudes.index(max(altitudes))
        assert max_idx > 0  # not at start
        assert max(altitudes) > 0

    def test_empty_tick(self):
        e = _engine()
        events = e.tick(1.0)
        assert events == []


# ---------------------------------------------------------------------------
# ArtilleryEngine — resolve_impacts
# ---------------------------------------------------------------------------

class TestResolveImpacts:
    def test_damage_at_center(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (100.0, 100.0),
            "damage": 120.0,
            "blast_radius": 25.0,
            "shell_type": "he",
        }]
        targets = [((100.0, 100.0), "t1", 0.0)]
        results = e.resolve_impacts(events, targets)
        assert len(results) == 1
        assert results[0]["damage"] == 120.0

    def test_damage_falloff(self):
        e = _engine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 100.0,
            "blast_radius": 20.0,
            "shell_type": "he",
        }]
        # Target at half blast radius
        targets = [((10.0, 0.0), "t1", 0.0)]
        results = e.resolve_impacts(events, targets)
        assert len(results) == 1
        assert results[0]["damage"] == 50.0

    def test_out_of_blast_radius(self):
        e = _engine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 100.0,
            "blast_radius": 10.0,
            "shell_type": "he",
        }]
        targets = [((20.0, 0.0), "t1", 0.0)]
        results = e.resolve_impacts(events, targets)
        assert len(results) == 0

    def test_armor_reduces_damage(self):
        e = _engine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 100.0,
            "blast_radius": 20.0,
            "shell_type": "he",
        }]
        targets = [((0.0, 0.0), "t1", 0.5)]
        results = e.resolve_impacts(events, targets)
        assert results[0]["damage"] == 50.0

    def test_ignores_non_impact_events(self):
        e = _engine()
        events = [
            {"event": "fire", "piece_id": "m1"},
            {"event": "smoke", "position": (0, 0), "radius": 10},
        ]
        results = e.resolve_impacts(events, [((0, 0), "t1", 0.0)])
        assert len(results) == 0

    def test_multiple_targets(self):
        e = _engine()
        events = [{
            "event": "impact",
            "shell_id": "s1",
            "position": (0.0, 0.0),
            "damage": 100.0,
            "blast_radius": 50.0,
            "shell_type": "he",
        }]
        targets = [
            ((0.0, 0.0), "t1", 0.0),
            ((10.0, 0.0), "t2", 0.0),
            ((100.0, 0.0), "t3", 0.0),  # out of range
        ]
        results = e.resolve_impacts(events, targets)
        assert len(results) == 2
        ids = {r["target_id"] for r in results}
        assert ids == {"t1", "t2"}


# ---------------------------------------------------------------------------
# ArtilleryEngine — to_three_js
# ---------------------------------------------------------------------------

class TestToThreeJs:
    def test_empty_state(self):
        e = _engine()
        d = e.to_three_js()
        assert d["shells"] == []
        assert d["impacts"] == []
        assert d["smoke_areas"] == []
        assert d["illumination_areas"] == []
        assert d["pieces"] == []

    def test_pieces_exported(self):
        e = _engine()
        e.add_piece(_mortar())
        d = e.to_three_js()
        assert len(d["pieces"]) == 1
        assert d["pieces"][0]["id"] == "m1"
        assert d["pieces"][0]["type"] == "mortar_81mm"

    def test_shells_in_flight_exported(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "point", 1)
        e.tick(0.1)
        d = e.to_three_js()
        assert len(d["shells"]) == 1
        assert "altitude" in d["shells"][0]
        assert "progress" in d["shells"][0]

    def test_impact_exported(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "point", 1)
        # Run until impact
        for _ in range(200):
            evts = e.tick(0.5)
            if any(ev["event"] == "impact" for ev in evts):
                break
        d = e.to_three_js()
        assert len(d["impacts"]) >= 1

    def test_smoke_area_exported(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "smoke", 1)
        for _ in range(200):
            evts = e.tick(0.5)
            if any(ev["event"] == "smoke" for ev in evts):
                break
        d = e.to_three_js()
        assert len(d["smoke_areas"]) >= 1

    def test_illumination_area_exported(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        e.request_fire_mission("m1", target, "illumination", 1)
        for _ in range(200):
            evts = e.tick(0.5)
            if any(ev["event"] == "illumination" for ev in evts):
                break
        d = e.to_three_js()
        assert len(d["illumination_areas"]) >= 1


# ---------------------------------------------------------------------------
# ForwardObserver
# ---------------------------------------------------------------------------

class TestForwardObserver:
    def test_call_fire(self):
        e = _engine()
        p = _mortar(alliance="blue")
        e.add_piece(p)
        fo = ForwardObserver("fo1", (0.0, 0.0), "blue")
        target = _target_in_range(p)
        m = fo.call_fire(e, target, "point", 3, piece_id="m1")
        assert isinstance(m, FireMission)
        assert m.rounds == 3

    def test_auto_select_piece(self):
        e = _engine()
        p = _mortar(alliance="blue")
        e.add_piece(p)
        fo = ForwardObserver("fo1", (0.0, 0.0), "blue")
        target = _target_in_range(p)
        m = fo.call_fire(e, target, "point", 2)
        assert m.piece_id == "m1"

    def test_auto_select_no_piece_available(self):
        e = _engine()
        # Only red piece, blue FO
        p = _mortar(alliance="red")
        e.add_piece(p)
        fo = ForwardObserver("fo1", (0.0, 0.0), "blue")
        target = _target_in_range(p)
        with pytest.raises(ValueError, match="No available"):
            fo.call_fire(e, target)

    def test_adjust_fire(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        fo = ForwardObserver("fo1", (0, 0), "blue")
        target = _target_in_range(p)
        m = fo.call_fire(e, target, "point", 5, piece_id="m1")
        # Adjust before any rounds fire
        new_m = fo.adjust_fire(e, m.mission_id, (50.0, 30.0))
        assert new_m is not None
        assert new_m.target_pos == (target[0] + 50.0, target[1] + 30.0)
        assert new_m.rounds == 5  # all rounds remaining

    def test_adjust_fire_reduces_rounds(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        fo = ForwardObserver("fo1", (0, 0), "blue")
        target = _target_in_range(p)
        m = fo.call_fire(e, target, "point", 5, piece_id="m1")
        e.tick(0.1)  # fires 1 round
        new_m = fo.adjust_fire(e, m.mission_id, (10.0, 0.0))
        assert new_m is not None
        assert new_m.rounds == 4  # 5 - 1 fired

    def test_adjust_nonexistent_mission(self):
        e = _engine()
        fo = ForwardObserver("fo1", (0, 0), "blue")
        result = fo.adjust_fire(e, "nope", (10.0, 0.0))
        assert result is None

    def test_end_mission(self):
        e = _engine()
        p = _mortar()
        e.add_piece(p)
        fo = ForwardObserver("fo1", (0, 0), "blue")
        target = _target_in_range(p)
        m = fo.call_fire(e, target, "point", 5, piece_id="m1")
        assert fo.end_mission(e, m.mission_id) is True
        assert m.active is False

    def test_end_nonexistent_mission(self):
        e = _engine()
        fo = ForwardObserver("fo1", (0, 0), "blue")
        assert fo.end_mission(e, "nope") is False

    def test_fo_selects_best_piece(self):
        """FO should prefer the piece whose midpoint range is closest to target distance."""
        e = _engine()
        mortar = _mortar(piece_id="mortar", alliance="blue")
        howitzer = create_piece(ArtilleryType.HOWITZER_105MM, "howitzer", "blue", (0, 0))
        e.add_piece(mortar)
        e.add_piece(howitzer)
        # Target at 3000m — mortar midrange ~2840, howitzer midrange ~6000
        target = (3000.0, 0.0)
        fo = ForwardObserver("fo1", (0, 0), "blue")
        m = fo.call_fire(e, target, "point", 1)
        assert m.piece_id == "mortar"


# ---------------------------------------------------------------------------
# Shell dataclass
# ---------------------------------------------------------------------------

class TestShell:
    def test_shell_fields(self):
        s = Shell(
            shell_id="s1",
            origin=(0, 0),
            target=(1000, 0),
            impact_pos=(1005, 3),
            altitude=50.0,
            time_of_flight=5.0,
            damage=120.0,
            blast_radius=25.0,
            shell_type="he",
        )
        assert s.shell_id == "s1"
        assert s.elapsed == 0.0
        assert s.shell_type == "he"

    def test_shell_defaults(self):
        s = Shell(
            shell_id="s2",
            origin=(0, 0),
            target=(500, 0),
            impact_pos=(500, 0),
            altitude=0.0,
            time_of_flight=3.0,
        )
        assert s.damage == 0.0
        assert s.blast_radius == 0.0
        assert s.shell_type == "he"


# ---------------------------------------------------------------------------
# FireMission dataclass
# ---------------------------------------------------------------------------

class TestFireMission:
    def test_fire_mission_fields(self):
        m = FireMission(
            mission_id="fm1",
            piece_id="m1",
            target_pos=(500, 300),
            mission_type="barrage",
            rounds=6,
            interval=1.5,
        )
        assert m.mission_id == "fm1"
        assert m.rounds_fired == 0
        assert m.active is True


# ---------------------------------------------------------------------------
# Integration: full fire mission lifecycle
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_mission_lifecycle(self):
        """Fire 3 rounds, all impact, mission completes."""
        e = _engine(seed=99)
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)
        m = e.request_fire_mission("m1", target, "point", 3, interval=0.5)

        all_events: list[dict] = []
        for _ in range(500):
            evts = e.tick(0.5)
            all_events.extend(evts)
            if not e.fire_missions and not e.shells_in_flight:
                break

        fire_events = [ev for ev in all_events if ev["event"] == "fire"]
        impact_events = [ev for ev in all_events if ev["event"] == "impact"]
        complete_events = [ev for ev in all_events if ev["event"] == "mission_complete"]

        assert len(fire_events) == 3
        assert len(impact_events) == 3
        assert len(complete_events) >= 1
        assert p.ammo == p.max_ammo - 3

    def test_multiple_pieces_independent(self):
        """Two pieces can fire independently."""
        e = _engine(seed=77)
        p1 = _mortar(piece_id="m1")
        p2 = _mortar(piece_id="m2")
        e.add_piece(p1)
        e.add_piece(p2)
        t1 = _target_in_range(p1)
        t2 = (t1[0], t1[1] + 200.0)
        e.request_fire_mission("m1", t1, "point", 1)
        e.request_fire_mission("m2", t2, "point", 1)

        all_events: list[dict] = []
        for _ in range(200):
            evts = e.tick(0.5)
            all_events.extend(evts)
            if not e.fire_missions and not e.shells_in_flight:
                break

        fire_events = [ev for ev in all_events if ev["event"] == "fire"]
        assert len(fire_events) == 2
        piece_ids = {ev["piece_id"] for ev in fire_events}
        assert piece_ids == {"m1", "m2"}

    def test_fo_call_adjust_end(self):
        """FO calls fire, adjusts, then ends."""
        e = _engine(seed=55)
        p = _mortar()
        e.add_piece(p)
        fo = ForwardObserver("fo1", (0, 0), "blue")
        target = _target_in_range(p)

        m = fo.call_fire(e, target, "point", 10, piece_id="m1")
        e.tick(0.1)  # fire first round
        assert m.rounds_fired == 1

        # Adjust fire
        new_m = fo.adjust_fire(e, m.mission_id, (20.0, 10.0))
        assert new_m is not None
        assert new_m.rounds == 9

        # End mission
        fo.end_mission(e, new_m.mission_id)
        # No more active missions
        active = [fm for fm in e.fire_missions if fm.active]
        assert len(active) == 0

    def test_danger_close_tighter_cep(self):
        """Danger close missions should cluster closer to target."""
        rng = random.Random(42)
        e = ArtilleryEngine(rng=rng)
        p = _mortar()
        e.add_piece(p)
        target = _target_in_range(p)

        # Fire 20 danger_close rounds and 20 area rounds, compare scatter
        dc_scatters = []
        area_scatters = []

        for _ in range(20):
            p.ammo = p.max_ammo
            p.ready = True
            p.cooldown = 0.0
            m = e.request_fire_mission("m1", target, "danger_close", 1)
            e.tick(0.1)
            if e.shells_in_flight:
                shell = e.shells_in_flight[-1]
                dc_scatters.append(distance(target, shell.impact_pos))

        for _ in range(20):
            p.ammo = p.max_ammo
            p.ready = True
            p.cooldown = 0.0
            m = e.request_fire_mission("m1", target, "area", 1)
            e.tick(0.1)
            if e.shells_in_flight:
                shell = e.shells_in_flight[-1]
                area_scatters.append(distance(target, shell.impact_pos))

        avg_dc = sum(dc_scatters) / len(dc_scatters) if dc_scatters else 999
        avg_area = sum(area_scatters) / len(area_scatters) if area_scatters else 0
        assert avg_dc < avg_area

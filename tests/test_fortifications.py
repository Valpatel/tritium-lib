# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the fortification and engineering module."""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.fortifications import (
    EngineeringEngine,
    Fortification,
    FortificationType,
    FORTIFICATION_TEMPLATES,
    Mine,
    MINE_TEMPLATES,
    _angle_diff,
    _in_cone,
)


# ---------------------------------------------------------------------------
# FortificationType enum
# ---------------------------------------------------------------------------

class TestFortificationType:
    def test_all_values(self):
        names = {e.name for e in FortificationType}
        assert "SANDBAG" in names
        assert "BUNKER" in names
        assert "TRENCH" in names
        assert "BARRICADE" in names
        assert "WATCHTOWER" in names
        assert "MINEFIELD" in names
        assert "WIRE" in names
        assert "CHECKPOINT" in names
        assert "FOXHOLE" in names

    def test_count(self):
        assert len(FortificationType) == 9

    def test_string_values(self):
        assert FortificationType.BUNKER.value == "bunker"
        assert FortificationType.TRENCH.value == "trench"


# ---------------------------------------------------------------------------
# Fortification dataclass
# ---------------------------------------------------------------------------

class TestFortification:
    def _make(self, **kw) -> Fortification:
        defaults = dict(
            fort_id="f1",
            fort_type=FortificationType.BUNKER,
            position=(50.0, 50.0),
            facing=0.0,
            width=6.0, depth=4.0, height=2.5,
            health=500.0, max_health=500.0,
            cover_value=0.95, concealment=0.9,
            capacity=4,
        )
        defaults.update(kw)
        return Fortification(**defaults)

    def test_is_complete_default(self):
        f = self._make()
        assert f.is_complete

    def test_is_complete_partial(self):
        f = self._make(build_progress=0.5)
        assert not f.is_complete

    def test_health_pct_full(self):
        f = self._make()
        assert f.health_pct == pytest.approx(1.0)

    def test_health_pct_half(self):
        f = self._make(health=250.0)
        assert f.health_pct == pytest.approx(0.5)

    def test_health_pct_zero_max(self):
        f = self._make(max_health=0.0)
        assert f.health_pct == 0.0

    def test_effective_cover_full(self):
        f = self._make()
        assert f.effective_cover == pytest.approx(0.95)

    def test_effective_cover_half_built(self):
        f = self._make(build_progress=0.5)
        assert f.effective_cover == pytest.approx(0.95 * 0.5)

    def test_effective_cover_damaged(self):
        f = self._make(health=250.0)
        assert f.effective_cover == pytest.approx(0.95 * 0.5)

    def test_effective_cover_destroyed(self):
        f = self._make(is_destroyed=True)
        assert f.effective_cover == 0.0

    def test_occupants_default_empty(self):
        f = self._make()
        assert f.occupants == []

    def test_occupants_independent(self):
        f1 = self._make(fort_id="f1")
        f2 = self._make(fort_id="f2")
        f1.occupants.append("u1")
        assert "u1" not in f2.occupants


# ---------------------------------------------------------------------------
# Mine dataclass
# ---------------------------------------------------------------------------

class TestMine:
    def test_defaults(self):
        m = Mine(
            mine_id="m1", position=(10.0, 20.0),
            mine_type="anti_personnel", damage=40.0,
            blast_radius=3.0, trigger_radius=2.0,
            alliance="friendly",
        )
        assert m.is_armed
        assert not m.is_triggered
        assert m.cone_angle == 360.0

    def test_claymore_cone(self):
        m = Mine(
            mine_id="m2", position=(0.0, 0.0),
            mine_type="claymore", damage=60.0,
            blast_radius=5.0, trigger_radius=3.0,
            alliance="friendly",
            facing=0.0, cone_angle=60.0,
        )
        assert m.cone_angle == 60.0


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_fort_template_keys(self):
        expected = {"sandbag", "foxhole", "trench", "bunker", "watchtower",
                    "barricade", "wire", "checkpoint"}
        assert set(FORTIFICATION_TEMPLATES.keys()) == expected

    def test_mine_template_keys(self):
        expected = {"anti_personnel", "anti_vehicle", "claymore"}
        assert set(MINE_TEMPLATES.keys()) == expected

    def test_sandbag_quick_build(self):
        t = FORTIFICATION_TEMPLATES["sandbag"]
        assert t["build_time"] == 10.0

    def test_bunker_slow_build(self):
        t = FORTIFICATION_TEMPLATES["bunker"]
        assert t["build_time"] == 300.0

    def test_trench_capacity(self):
        assert FORTIFICATION_TEMPLATES["trench"]["capacity"] == 6

    def test_bunker_high_cover(self):
        assert FORTIFICATION_TEMPLATES["bunker"]["cover_value"] == 0.95

    def test_wire_movement_penalty(self):
        assert FORTIFICATION_TEMPLATES["wire"].get("movement_penalty", 0) == 0.7

    def test_barricade_blocks_vehicles(self):
        assert FORTIFICATION_TEMPLATES["barricade"].get("blocks_vehicles") is True

    def test_watchtower_detection_bonus(self):
        assert FORTIFICATION_TEMPLATES["watchtower"].get("detection_bonus", 0) == 0.5

    def test_anti_personnel_damage(self):
        assert MINE_TEMPLATES["anti_personnel"]["damage"] == 40.0

    def test_anti_vehicle_damage(self):
        assert MINE_TEMPLATES["anti_vehicle"]["damage"] == 200.0

    def test_claymore_cone(self):
        assert MINE_TEMPLATES["claymore"]["cone_angle"] == 60.0

    def test_anti_vehicle_weight_threshold(self):
        assert MINE_TEMPLATES["anti_vehicle"]["weight_threshold"] == 500.0

    def test_all_templates_have_required_keys(self):
        required = {"fort_type", "width", "depth", "height", "max_health",
                     "cover_value", "concealment", "capacity", "build_time"}
        for name, tmpl in FORTIFICATION_TEMPLATES.items():
            for key in required:
                assert key in tmpl, f"{name} missing {key}"


# ---------------------------------------------------------------------------
# EngineeringEngine — build & construction
# ---------------------------------------------------------------------------

class TestBuild:
    def test_build_instant(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (10.0, 20.0))
        assert fort.is_complete
        assert fort.build_progress == 1.0
        assert fort.fort_id in eng.fortifications

    def test_build_with_builder(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (10.0, 20.0), builder_id="engineer_1")
        assert fort.build_progress == 0.0
        assert not fort.is_complete
        assert len(eng.construction_queue) == 1

    def test_build_unknown_type(self):
        eng = EngineeringEngine()
        with pytest.raises(ValueError, match="Unknown fortification"):
            eng.build("laser_turret", (0.0, 0.0))

    def test_build_uses_template_values(self):
        eng = EngineeringEngine()
        fort = eng.build("trench", (5.0, 5.0))
        assert fort.width == 10.0
        assert fort.capacity == 6
        assert fort.cover_value == 0.8

    def test_build_facing(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0), facing=1.57)
        assert fort.facing == pytest.approx(1.57)

    def test_multiple_builds(self):
        eng = EngineeringEngine()
        f1 = eng.build("sandbag", (0.0, 0.0))
        f2 = eng.build("bunker", (10.0, 10.0))
        assert f1.fort_id != f2.fort_id
        assert len(eng.fortifications) == 2


class TestConstruction:
    def test_advance_construction(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0), builder_id="e1")
        # sandbag build_time = 10s, 1 engineer -> 0.1/s
        eng.advance_construction(fort.fort_id, dt=5.0)
        assert fort.build_progress == pytest.approx(0.5)

    def test_advance_construction_complete(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0), builder_id="e1")
        result = eng.advance_construction(fort.fort_id, dt=10.0)
        assert result == pytest.approx(1.0)
        assert fort.is_complete

    def test_advance_construction_clamps(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0), builder_id="e1")
        eng.advance_construction(fort.fort_id, dt=100.0)
        assert fort.build_progress == 1.0

    def test_more_engineers_faster(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0), builder_id="e1")
        # bunker build_time=300s, 3 engineers -> 3/300 = 0.01/s
        eng.advance_construction(fort.fort_id, dt=10.0, engineers=3)
        assert fort.build_progress == pytest.approx(0.1)

    def test_advance_unknown_fort(self):
        eng = EngineeringEngine()
        with pytest.raises(KeyError):
            eng.advance_construction("nonexistent", dt=1.0)

    def test_advance_destroyed_noop(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0), builder_id="e1")
        fort.is_destroyed = True
        result = eng.advance_construction(fort.fort_id, dt=10.0)
        assert result == 0.0

    def test_advance_already_complete(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0))
        result = eng.advance_construction(fort.fort_id, dt=10.0)
        assert result == 1.0


# ---------------------------------------------------------------------------
# EngineeringEngine — mines
# ---------------------------------------------------------------------------

class TestMines:
    def test_place_mine(self):
        eng = EngineeringEngine()
        mine = eng.place_mine((10.0, 20.0), "anti_personnel", "friendly")
        assert mine.is_armed
        assert mine.damage == 40.0
        assert len(eng.minefields) == 1

    def test_place_mine_unknown_type(self):
        eng = EngineeringEngine()
        with pytest.raises(ValueError, match="Unknown mine type"):
            eng.place_mine((0.0, 0.0), "nuclear", "friendly")

    def test_place_mine_anti_vehicle(self):
        eng = EngineeringEngine()
        mine = eng.place_mine((0.0, 0.0), "anti_vehicle", "hostile")
        assert mine.damage == 200.0
        assert mine.weight_threshold == 500.0

    def test_place_claymore(self):
        eng = EngineeringEngine()
        mine = eng.place_mine((0.0, 0.0), "claymore", "friendly", facing=1.0)
        assert mine.cone_angle == 60.0
        assert mine.facing == 1.0

    def test_clear_mines_enemy(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "hostile")
        eng.place_mine((12.0, 10.0), "anti_personnel", "hostile")
        eng.place_mine((50.0, 50.0), "anti_personnel", "hostile")
        cleared = eng.clear_mines((10.0, 10.0), radius=5.0, alliance="friendly")
        assert cleared == 2
        assert len(eng.minefields) == 1

    def test_clear_mines_skips_friendly(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        cleared = eng.clear_mines((10.0, 10.0), radius=5.0, alliance="friendly")
        assert cleared == 0
        assert len(eng.minefields) == 1

    def test_clear_mines_none_in_range(self):
        eng = EngineeringEngine()
        eng.place_mine((100.0, 100.0), "anti_personnel", "hostile")
        cleared = eng.clear_mines((0.0, 0.0), radius=5.0, alliance="friendly")
        assert cleared == 0


# ---------------------------------------------------------------------------
# EngineeringEngine — occupancy & cover
# ---------------------------------------------------------------------------

class TestOccupancy:
    def test_enter_fortification(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0))
        assert eng.enter_fortification(fort.fort_id, "u1")
        assert "u1" in fort.occupants

    def test_enter_full(self):
        eng = EngineeringEngine()
        fort = eng.build("foxhole", (0.0, 0.0))  # capacity 2
        eng.enter_fortification(fort.fort_id, "u1")
        eng.enter_fortification(fort.fort_id, "u2")
        assert not eng.enter_fortification(fort.fort_id, "u3")

    def test_enter_incomplete(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0), builder_id="e1")
        assert not eng.enter_fortification(fort.fort_id, "u1")

    def test_enter_destroyed(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0))
        fort.is_destroyed = True
        assert not eng.enter_fortification(fort.fort_id, "u1")

    def test_enter_nonexistent(self):
        eng = EngineeringEngine()
        assert not eng.enter_fortification("nope", "u1")

    def test_enter_idempotent(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0))
        eng.enter_fortification(fort.fort_id, "u1")
        assert eng.enter_fortification(fort.fort_id, "u1")
        assert fort.occupants.count("u1") == 1

    def test_exit_fortification(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0))
        eng.enter_fortification(fort.fort_id, "u1")
        eng.exit_fortification(fort.fort_id, "u1")
        assert "u1" not in fort.occupants

    def test_exit_nonexistent_fort(self):
        eng = EngineeringEngine()
        eng.exit_fortification("nope", "u1")  # should not raise

    def test_exit_unit_not_inside(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0))
        eng.exit_fortification(fort.fort_id, "u1")  # should not raise

    def test_get_cover_bonus(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0))
        eng.enter_fortification(fort.fort_id, "u1")
        assert eng.get_cover_bonus("u1") == pytest.approx(0.95)

    def test_get_cover_bonus_not_inside(self):
        eng = EngineeringEngine()
        assert eng.get_cover_bonus("u1") == 0.0

    def test_get_detection_bonus_watchtower(self):
        eng = EngineeringEngine()
        fort = eng.build("watchtower", (0.0, 0.0))
        eng.enter_fortification(fort.fort_id, "u1")
        assert eng.get_detection_bonus("u1") == 0.5

    def test_get_detection_bonus_bunker(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0))
        eng.enter_fortification(fort.fort_id, "u1")
        assert eng.get_detection_bonus("u1") == 0.0


# ---------------------------------------------------------------------------
# EngineeringEngine — damage
# ---------------------------------------------------------------------------

class TestDamageFortification:
    def test_damage_reduces_health(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0))
        eng.damage_fortification(fort.fort_id, 20.0)
        assert fort.health == pytest.approx(30.0)

    def test_damage_destroys(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0))
        destroyed = eng.damage_fortification(fort.fort_id, 100.0)
        assert destroyed
        assert fort.is_destroyed
        assert fort.health == 0.0

    def test_damage_clears_occupants(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0))
        eng.enter_fortification(fort.fort_id, "u1")
        eng.damage_fortification(fort.fort_id, 600.0)
        assert fort.occupants == []

    def test_damage_nonexistent(self):
        eng = EngineeringEngine()
        assert not eng.damage_fortification("nope", 10.0)

    def test_damage_already_destroyed(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0))
        fort.is_destroyed = True
        assert not eng.damage_fortification(fort.fort_id, 10.0)


# ---------------------------------------------------------------------------
# EngineeringEngine — tick
# ---------------------------------------------------------------------------

class TestTick:
    def test_mine_triggers_on_enemy(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        units = {"e1": ((10.5, 10.5), "hostile")}
        events = eng.tick(0.1, units)
        assert len(events) == 1
        assert events[0]["type"] == "mine_triggered"
        assert events[0]["triggered_by"] == "e1"

    def test_mine_skips_friendly(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        units = {"f1": ((10.0, 10.0), "friendly")}
        events = eng.tick(0.1, units)
        assert len(events) == 0

    def test_mine_out_of_range(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        units = {"e1": ((100.0, 100.0), "hostile")}
        events = eng.tick(0.1, units)
        assert len(events) == 0

    def test_anti_vehicle_ignores_infantry(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_vehicle", "friendly")
        # Default weight 80kg < threshold 500kg
        units = {"e1": ((10.0, 10.0), "hostile")}
        events = eng.tick(0.1, units)
        assert len(events) == 0

    def test_anti_vehicle_triggers_on_vehicle(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_vehicle", "friendly")
        units = {"v1": ((10.0, 10.0), "hostile", 2000.0)}
        events = eng.tick(0.1, units)
        assert len(events) == 1

    def test_mine_casualties_include_blast_radius(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        # e1 triggers, e2 is in blast radius (3m)
        units = {
            "e1": ((10.0, 10.0), "hostile"),
            "e2": ((12.0, 10.0), "hostile"),
        }
        events = eng.tick(0.1, units)
        assert len(events) == 1
        casualties = events[0]["casualties"]
        uids = [c["unit_id"] for c in casualties]
        assert "e1" in uids
        assert "e2" in uids

    def test_mine_damage_falloff(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        units = {
            "e1": ((10.0, 10.0), "hostile"),  # distance 0 -> full damage
            "e2": ((11.5, 10.0), "hostile"),   # distance 1.5 -> half damage
        }
        events = eng.tick(0.1, units)
        cas = {c["unit_id"]: c["damage"] for c in events[0]["casualties"]}
        assert cas["e1"] > cas["e2"]
        assert cas["e1"] == pytest.approx(40.0, abs=0.1)
        assert cas["e2"] == pytest.approx(20.0, abs=0.5)

    def test_claymore_directional(self):
        eng = EngineeringEngine()
        # Claymore facing east (0 rad), 60 degree cone
        eng.place_mine((10.0, 10.0), "claymore", "friendly", facing=0.0)
        # e1 is east (in cone), e2 is north (out of cone)
        units = {
            "e1": ((12.0, 10.0), "hostile"),
            "e2": ((10.0, 12.0), "hostile"),
        }
        events = eng.tick(0.1, units)
        assert len(events) == 1
        assert events[0]["triggered_by"] == "e1"

    def test_construction_completes_via_tick(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (0.0, 0.0), builder_id="e1")
        events = eng.tick(10.0, {})
        assert fort.is_complete
        complete_events = [e for e in events if e["type"] == "construction_complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["fort_id"] == fort.fort_id

    def test_construction_partial_tick(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0), builder_id="e1")
        events = eng.tick(10.0, {})
        assert not fort.is_complete
        assert len([e for e in events if e["type"] == "construction_complete"]) == 0
        assert len(eng.construction_queue) == 1

    def test_construction_removed_from_queue(self):
        eng = EngineeringEngine()
        eng.build("sandbag", (0.0, 0.0), builder_id="e1")
        eng.tick(10.0, {})
        assert len(eng.construction_queue) == 0

    def test_triggered_mine_not_retrigger(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        units = {"e1": ((10.0, 10.0), "hostile")}
        eng.tick(0.1, units)
        events2 = eng.tick(0.1, units)
        mine_events = [e for e in events2 if e["type"] == "mine_triggered"]
        assert len(mine_events) == 0


# ---------------------------------------------------------------------------
# EngineeringEngine — to_three_js
# ---------------------------------------------------------------------------

class TestToThreeJs:
    def test_empty(self):
        eng = EngineeringEngine()
        data = eng.to_three_js()
        assert data["fortifications"] == []
        assert data["mines"] == []
        assert data["effects"] == []

    def test_fortification_fields(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (50.0, 60.0), facing=1.0)
        data = eng.to_three_js()
        f = data["fortifications"][0]
        assert f["id"] == fort.fort_id
        assert f["type"] == "bunker"
        assert f["x"] == 50.0
        assert f["y"] == 60.0
        assert f["facing"] == 1.0
        assert f["w"] == 6.0
        assert f["d"] == 4.0
        assert f["h"] == 2.5
        assert f["health_pct"] == 1.0
        assert f["build_progress"] == 1.0

    def test_mine_fields(self):
        eng = EngineeringEngine()
        mine = eng.place_mine((80.0, 60.0), "anti_personnel", "friendly")
        data = eng.to_three_js()
        m = data["mines"][0]
        assert m["id"] == mine.mine_id
        assert m["type"] == "anti_personnel"
        assert m["armed"] is True
        assert m["alliance"] == "friendly"

    def test_triggered_mine_excluded(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        eng.tick(0.1, {"e1": ((10.0, 10.0), "hostile")})
        data = eng.to_three_js()
        assert len(data["mines"]) == 0

    def test_effects_from_explosion(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        eng.tick(0.1, {"e1": ((10.0, 10.0), "hostile")})
        data = eng.to_three_js()
        assert len(data["effects"]) == 1
        assert data["effects"][0]["type"] == "mine_explosion"

    def test_effects_cleared_after_read(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        eng.tick(0.1, {"e1": ((10.0, 10.0), "hostile")})
        eng.to_three_js()  # consumes effects
        data2 = eng.to_three_js()
        assert data2["effects"] == []


# ---------------------------------------------------------------------------
# EngineeringEngine — query helpers
# ---------------------------------------------------------------------------

class TestQueryHelpers:
    def test_get_fortification(self):
        eng = EngineeringEngine()
        fort = eng.build("bunker", (0.0, 0.0))
        assert eng.get_fortification(fort.fort_id) is fort

    def test_get_fortification_missing(self):
        eng = EngineeringEngine()
        assert eng.get_fortification("nope") is None

    def test_get_fortifications_near(self):
        eng = EngineeringEngine()
        eng.build("sandbag", (10.0, 10.0))
        eng.build("sandbag", (100.0, 100.0))
        near = eng.get_fortifications_near((10.0, 10.0), 5.0)
        assert len(near) == 1

    def test_get_fortifications_near_excludes_destroyed(self):
        eng = EngineeringEngine()
        fort = eng.build("sandbag", (10.0, 10.0))
        fort.is_destroyed = True
        near = eng.get_fortifications_near((10.0, 10.0), 5.0)
        assert len(near) == 0

    def test_get_mines_near(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        eng.place_mine((100.0, 100.0), "anti_personnel", "friendly")
        near = eng.get_mines_near((10.0, 10.0), 5.0)
        assert len(near) == 1

    def test_get_mines_near_excludes_triggered(self):
        eng = EngineeringEngine()
        eng.place_mine((10.0, 10.0), "anti_personnel", "friendly")
        eng.tick(0.1, {"e1": ((10.0, 10.0), "hostile")})
        near = eng.get_mines_near((10.0, 10.0), 5.0)
        assert len(near) == 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_angle_diff_same(self):
        assert _angle_diff(0.0, 0.0) == pytest.approx(0.0)

    def test_angle_diff_quarter(self):
        assert _angle_diff(0.0, math.pi / 2) == pytest.approx(math.pi / 2)

    def test_angle_diff_wrap(self):
        # Going from just below pi to just above -pi
        assert abs(_angle_diff(3.0, -3.0)) < math.pi

    def test_in_cone_ahead(self):
        assert _in_cone((0.0, 0.0), 0.0, 90.0, (5.0, 0.0))

    def test_in_cone_behind(self):
        assert not _in_cone((0.0, 0.0), 0.0, 90.0, (-5.0, 0.0))

    def test_in_cone_edge(self):
        # 45 degrees from facing, with 90 degree cone -> on edge
        target = (5.0, 5.0)  # 45 degrees
        assert _in_cone((0.0, 0.0), 0.0, 90.0, target)

    def test_in_cone_omnidirectional(self):
        assert _in_cone((0.0, 0.0), 0.0, 360.0, (-5.0, -5.0))

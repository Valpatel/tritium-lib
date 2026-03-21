# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the destruction and fire propagation system."""

from __future__ import annotations

import math
import random

import pytest

from tritium_lib.sim_engine.destruction import (
    MATERIAL_PROPERTIES,
    DamageLevel,
    Debris,
    DestructionEngine,
    Fire,
    Structure,
    StructureType,
    _DEBRIS_SIZE_MAP,
    _fire_color,
    _health_to_damage_level,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**kw) -> DestructionEngine:
    return DestructionEngine(rng=random.Random(42), **kw)


def _make_structure(
    sid: str = "s1",
    material: str = "concrete",
    health: float = 100.0,
    max_health: float = 100.0,
    pos: tuple[float, float] = (0.0, 0.0),
    size: tuple[float, float, float] = (10.0, 10.0, 5.0),
    stype: StructureType = StructureType.BUILDING,
) -> Structure:
    return Structure(
        structure_id=sid,
        structure_type=stype,
        position=pos,
        size=size,
        health=health,
        max_health=max_health,
        material=material,
    )


# ===========================================================================
# StructureType enum
# ===========================================================================

class TestStructureType:
    def test_all_values_exist(self):
        expected = {"building", "wall", "barrier", "vehicle_wreck", "bridge", "tower", "fence"}
        assert {e.value for e in StructureType} == expected

    def test_count(self):
        assert len(StructureType) == 7


# ===========================================================================
# DamageLevel enum
# ===========================================================================

class TestDamageLevel:
    def test_all_values_exist(self):
        expected = {"intact", "light_damage", "heavy_damage", "critical", "destroyed", "collapsed"}
        assert {e.value for e in DamageLevel} == expected

    def test_count(self):
        assert len(DamageLevel) == 6


# ===========================================================================
# Health -> DamageLevel mapping
# ===========================================================================

class TestHealthToDamageLevel:
    def test_full_health_is_intact(self):
        assert _health_to_damage_level(1.0) == DamageLevel.INTACT

    def test_zero_health_is_collapsed(self):
        assert _health_to_damage_level(0.0) == DamageLevel.COLLAPSED

    def test_75_pct_is_light(self):
        assert _health_to_damage_level(0.74) == DamageLevel.LIGHT_DAMAGE

    def test_50_pct_is_heavy(self):
        assert _health_to_damage_level(0.49) == DamageLevel.HEAVY_DAMAGE

    def test_25_pct_is_critical(self):
        assert _health_to_damage_level(0.24) == DamageLevel.CRITICAL

    def test_5_pct_is_destroyed(self):
        assert _health_to_damage_level(0.04) == DamageLevel.DESTROYED

    def test_transition_order(self):
        """Damage levels progress in order as health drops."""
        levels = []
        for pct in [1.0, 0.8, 0.7, 0.4, 0.2, 0.03, 0.0]:
            levels.append(_health_to_damage_level(pct))
        assert levels == [
            DamageLevel.INTACT,
            DamageLevel.INTACT,
            DamageLevel.LIGHT_DAMAGE,
            DamageLevel.HEAVY_DAMAGE,
            DamageLevel.CRITICAL,
            DamageLevel.DESTROYED,
            DamageLevel.COLLAPSED,
        ]


# ===========================================================================
# Structure dataclass
# ===========================================================================

class TestStructure:
    def test_defaults(self):
        s = _make_structure()
        assert s.damage_level == DamageLevel.INTACT
        assert s.is_on_fire is False
        assert s.fire_intensity == 0.0
        assert s.provides_cover is True
        assert s.debris == []
        assert s.holes == []

    def test_custom_material(self):
        s = _make_structure(material="wood")
        assert s.material == "wood"


# ===========================================================================
# Material properties
# ===========================================================================

class TestMaterialProperties:
    def test_all_five_materials(self):
        assert set(MATERIAL_PROPERTIES.keys()) == {"concrete", "wood", "metal", "brick", "glass"}

    @pytest.mark.parametrize("mat", MATERIAL_PROPERTIES.keys())
    def test_required_keys(self, mat):
        props = MATERIAL_PROPERTIES[mat]
        for key in ("health", "fire_resistance", "debris_size", "color", "burn_rate"):
            assert key in props, f"{mat} missing {key}"

    @pytest.mark.parametrize("mat", MATERIAL_PROPERTIES.keys())
    def test_fire_resistance_range(self, mat):
        r = MATERIAL_PROPERTIES[mat]["fire_resistance"]
        assert 0.0 <= r <= 1.0

    def test_glass_low_health(self):
        assert MATERIAL_PROPERTIES["glass"]["health"] == 20.0

    def test_concrete_high_health(self):
        assert MATERIAL_PROPERTIES["concrete"]["health"] == 200.0

    def test_wood_low_fire_resistance(self):
        assert MATERIAL_PROPERTIES["wood"]["fire_resistance"] == 0.1

    def test_concrete_high_fire_resistance(self):
        assert MATERIAL_PROPERTIES["concrete"]["fire_resistance"] == 0.9

    @pytest.mark.parametrize("mat", MATERIAL_PROPERTIES.keys())
    def test_valid_debris_size(self, mat):
        assert MATERIAL_PROPERTIES[mat]["debris_size"] in _DEBRIS_SIZE_MAP

    @pytest.mark.parametrize("mat", MATERIAL_PROPERTIES.keys())
    def test_color_is_hex(self, mat):
        color = MATERIAL_PROPERTIES[mat]["color"]
        assert color.startswith("#") and len(color) == 7


# ===========================================================================
# DestructionEngine — structure damage
# ===========================================================================

class TestDamageStructure:
    def test_damage_reduces_health(self):
        eng = _make_engine()
        s = _make_structure(health=100)
        eng.add_structure(s)
        eng.damage_structure("s1", 30, (0, 0))
        assert s.health == 70.0

    def test_health_cannot_go_negative(self):
        eng = _make_engine()
        s = _make_structure(health=10)
        eng.add_structure(s)
        eng.damage_structure("s1", 50, (0, 0))
        assert s.health == 0.0

    def test_damage_level_transitions(self):
        eng = _make_engine()
        s = _make_structure(health=100, max_health=100)
        eng.add_structure(s)

        eng.damage_structure("s1", 10, (0, 0))
        assert s.damage_level == DamageLevel.INTACT  # 90% health

        eng.damage_structure("s1", 20, (0, 0))
        assert s.damage_level == DamageLevel.LIGHT_DAMAGE  # 70%

        eng.damage_structure("s1", 25, (0, 0))
        assert s.damage_level == DamageLevel.HEAVY_DAMAGE  # 45%

        eng.damage_structure("s1", 25, (0, 0))
        assert s.damage_level == DamageLevel.CRITICAL  # 20%

        eng.damage_structure("s1", 17, (0, 0))
        assert s.damage_level == DamageLevel.DESTROYED  # 3%

        eng.damage_structure("s1", 5, (0, 0))
        assert s.damage_level == DamageLevel.COLLAPSED  # 0%

    def test_zero_damage_no_change(self):
        eng = _make_engine()
        s = _make_structure(health=100)
        eng.add_structure(s)
        result = eng.damage_structure("s1", 0, (0, 0))
        assert s.health == 100.0
        assert result["damage"] == 0

    def test_returns_event_dict(self):
        eng = _make_engine()
        eng.add_structure(_make_structure())
        result = eng.damage_structure("s1", 20, (5, 5))
        assert result["event"] == "damage"
        assert result["structure_id"] == "s1"
        assert "new_health" in result
        assert "damage_level" in result
        assert "debris_spawned" in result

    def test_unknown_structure(self):
        eng = _make_engine()
        result = eng.damage_structure("nonexistent", 10, (0, 0))
        assert result["error"] == "not_found"

    def test_generates_debris(self):
        eng = _make_engine()
        eng.add_structure(_make_structure())
        eng.damage_structure("s1", 40, (0, 0))
        assert len(eng.debris_list) > 0

    def test_hole_at_impact(self):
        eng = _make_engine()
        eng.add_structure(_make_structure())
        eng.damage_structure("s1", 20, (3, 4))
        assert len(eng.structures[0].holes) == 1
        assert eng.structures[0].holes[0]["x"] == 3
        assert eng.structures[0].holes[0]["y"] == 4


# ===========================================================================
# Wood + fire
# ===========================================================================

class TestWoodFire:
    def test_wood_catches_fire_at_heavy_damage(self):
        """Wood structures with heavy damage should eventually ignite."""
        # Run multiple trials to overcome randomness
        ignited = False
        for seed in range(100):
            eng = DestructionEngine(rng=random.Random(seed))
            s = _make_structure(material="wood", health=100, max_health=100)
            eng.add_structure(s)
            # One big hit to reach heavy damage
            eng.damage_structure("s1", 60, (0, 0))
            if s.is_on_fire:
                ignited = True
                break
        assert ignited, "Wood never ignited across 100 trials at heavy damage"

    def test_concrete_does_not_self_ignite(self):
        """Concrete should never self-ignite from damage alone."""
        for seed in range(50):
            eng = DestructionEngine(rng=random.Random(seed))
            s = _make_structure(material="concrete", health=100, max_health=100)
            eng.add_structure(s)
            eng.damage_structure("s1", 90, (0, 0))
            assert not s.is_on_fire


# ===========================================================================
# Glass
# ===========================================================================

class TestGlass:
    def test_glass_shatters_quickly(self):
        eng = _make_engine()
        s = _make_structure(material="glass", health=20, max_health=20)
        eng.add_structure(s)
        eng.damage_structure("s1", 25, (0, 0))
        assert s.health == 0.0
        assert s.damage_level == DamageLevel.COLLAPSED


# ===========================================================================
# Collapse
# ===========================================================================

class TestCollapse:
    def test_collapse_generates_rubble(self):
        eng = _make_engine()
        s = _make_structure(health=10, max_health=100, size=(20, 20, 10))
        eng.add_structure(s)
        initial_debris = len(eng.debris_list)
        eng.damage_structure("s1", 100, (0, 0))
        assert len(eng.debris_list) > initial_debris + 5  # collapse generates many pieces

    def test_collapse_removes_cover(self):
        eng = _make_engine()
        s = _make_structure(health=5, max_health=100)
        eng.add_structure(s)
        assert s.provides_cover is True
        eng.damage_structure("s1", 10, (0, 0))
        assert s.provides_cover is False

    def test_rubble_blocks_movement(self):
        eng = _make_engine()
        s = _make_structure(health=5, max_health=100, pos=(5, 5), size=(4, 4, 5))
        eng.add_structure(s)
        eng.damage_structure("s1", 100, (5, 5))
        blocked = eng.get_blocked_positions(cell_size=1.0)
        assert len(blocked) > 0
        # Structure center cell should be blocked
        assert (5, 5) in blocked


# ===========================================================================
# Fire system
# ===========================================================================

class TestFire:
    def test_start_fire(self):
        eng = _make_engine()
        f = eng.start_fire((10, 10), radius=3.0, intensity=0.7, fuel=30.0)
        assert len(eng.fires) == 1
        assert f.intensity == 0.7
        assert f.fuel_remaining == 30.0

    def test_fire_fuel_depletes(self):
        eng = _make_engine()
        eng.start_fire((0, 0), radius=2.0, intensity=1.0, fuel=5.0)
        # Tick enough for fuel to run out
        for _ in range(100):
            eng.tick(0.1)
        assert len(eng.fires) == 0

    def test_fire_intensity_zero_dies(self):
        eng = _make_engine()
        f = eng.start_fire((0, 0), intensity=0.0, fuel=10.0)
        eng.tick(0.1)
        assert len(eng.fires) == 0

    def test_fire_damages_wood_structure(self):
        eng = _make_engine()
        s = _make_structure(material="wood", health=80, max_health=80, pos=(0, 0))
        eng.add_structure(s)
        eng.start_fire((0, 0), radius=5.0, intensity=0.8, fuel=60.0)
        for _ in range(50):
            eng.tick(0.1)
        assert s.health < 80.0

    def test_fire_concrete_resists(self):
        eng = _make_engine()
        s = _make_structure(material="concrete", health=200, max_health=200, pos=(0, 0))
        eng.add_structure(s)
        eng.start_fire((0, 0), radius=5.0, intensity=0.8, fuel=10.0)
        initial = s.health
        for _ in range(20):
            eng.tick(0.1)
        # Concrete should take very little fire damage
        assert s.health > initial * 0.95

    def test_fire_spreads_toward_wind(self):
        eng = _make_engine(wind_direction=0.0, wind_speed=5.0)
        f = eng.start_fire((10, 10), radius=2.0, intensity=0.7, fuel=60.0)
        initial_x = f.position[0]
        for _ in range(20):
            eng.tick(0.5)
        # Fire should have drifted in the wind direction (east = +x)
        assert f.position[0] > initial_x

    def test_fire_ignites_nearby_wood(self):
        """A strong fire near a wood structure should eventually ignite it."""
        ignited = False
        for seed in range(50):
            eng = DestructionEngine(
                wind_direction=0.0, wind_speed=0.0,
                rng=random.Random(seed),
            )
            s = _make_structure(sid="w1", material="wood", pos=(3, 0))
            eng.add_structure(s)
            eng.start_fire((0, 0), radius=4.0, intensity=0.9, fuel=60.0)
            for _ in range(200):
                eng.tick(0.1)
            if s.is_on_fire:
                ignited = True
                break
        assert ignited, "Fire never spread to nearby wood structure"


# ===========================================================================
# Smoke / LOS
# ===========================================================================

class TestSmoke:
    def test_smoke_blocks_los(self):
        eng = _make_engine()
        eng.start_fire((10, 10), radius=3.0, intensity=0.8, fuel=60.0)
        blockers = eng.get_los_blockers()
        assert len(blockers) == 1
        assert blockers[0]["height"] > 0
        assert blockers[0]["radius"] > 0

    def test_no_smoke_from_weak_fire(self):
        eng = _make_engine()
        eng.start_fire((0, 0), radius=1.0, intensity=0.05, fuel=10.0)
        blockers = eng.get_los_blockers()
        assert len(blockers) == 0

    def test_smoke_drifts_with_wind(self):
        eng = _make_engine(wind_direction=0.0, wind_speed=10.0)
        eng.start_fire((0, 0), radius=2.0, intensity=0.8, fuel=60.0)
        blockers = eng.get_los_blockers()
        assert len(blockers) == 1
        # Smoke should be offset in the wind direction
        assert blockers[0]["x"] > 0


# ===========================================================================
# Debris physics
# ===========================================================================

class TestDebris:
    def test_debris_has_ballistic_trajectory(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(health=100))
        eng.damage_structure("s1", 50, (0, 0))
        d = eng.debris_list[0]
        initial_pos = d.position
        eng.tick(0.1)
        assert d.position != initial_pos

    def test_debris_falls_with_gravity(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(health=100))
        eng.damage_structure("s1", 50, (0, 0))
        d = eng.debris_list[0]
        initial_vz = d.vz
        eng.tick(0.1)
        # vz should decrease due to gravity
        assert d.vz < initial_vz

    def test_debris_settles_after_lifetime(self):
        eng = _make_engine()
        eng.add_structure(_make_structure())
        eng.damage_structure("s1", 50, (0, 0))
        assert any(d.is_active for d in eng.debris_list)
        # Tick past debris lifetime
        for _ in range(40):
            eng.tick(0.1)
        assert all(not d.is_active for d in eng.debris_list)

    def test_debris_z_does_not_go_negative(self):
        eng = _make_engine()
        eng.add_structure(_make_structure())
        eng.damage_structure("s1", 50, (0, 0))
        for _ in range(50):
            eng.tick(0.1)
        for d in eng.debris_list:
            assert d.z >= 0.0


# ===========================================================================
# to_three_js output
# ===========================================================================

class TestToThreeJs:
    def test_empty_engine(self):
        eng = _make_engine()
        out = eng.to_three_js()
        assert out == {"structures": [], "fires": [], "debris": [], "smoke": []}

    def test_includes_structures(self):
        eng = _make_engine()
        eng.add_structure(_make_structure())
        out = eng.to_three_js()
        assert len(out["structures"]) == 1
        s = out["structures"][0]
        assert s["id"] == "s1"
        assert s["material"] == "concrete"
        assert s["damage"] == "intact"
        assert s["health_pct"] == 1.0

    def test_includes_fires(self):
        eng = _make_engine()
        eng.start_fire((5, 5), intensity=0.6)
        out = eng.to_three_js()
        assert len(out["fires"]) == 1
        f = out["fires"][0]
        assert "color" in f
        assert "emitter" in f
        assert f["emitter"]["rate"] > 0
        assert len(f["emitter"]["colors"]) == 3

    def test_includes_debris(self):
        eng = _make_engine()
        eng.add_structure(_make_structure())
        eng.damage_structure("s1", 50, (0, 0))
        out = eng.to_three_js()
        assert len(out["debris"]) > 0
        d = out["debris"][0]
        for key in ("id", "x", "y", "z", "vx", "vy", "vz", "size", "material", "rotation"):
            assert key in d, f"Missing key {key} in debris"

    def test_includes_smoke_for_active_fire(self):
        eng = _make_engine()
        eng.start_fire((5, 5), intensity=0.7)
        out = eng.to_three_js()
        assert len(out["smoke"]) == 1
        sm = out["smoke"][0]
        for key in ("x", "y", "radius", "height", "opacity", "color"):
            assert key in sm

    def test_no_smoke_for_weak_fire(self):
        eng = _make_engine()
        eng.start_fire((5, 5), intensity=0.01)
        out = eng.to_three_js()
        assert len(out["smoke"]) == 0

    def test_structure_damage_reflected(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(health=100, max_health=100))
        eng.damage_structure("s1", 60, (2, 2))
        out = eng.to_three_js()
        s = out["structures"][0]
        assert s["damage"] == "heavy_damage"
        assert s["health_pct"] == pytest.approx(0.4)
        assert len(s["holes"]) == 1

    def test_settled_debris_excluded(self):
        """Inactive (settled) debris should not appear in Three.js output."""
        eng = _make_engine()
        eng.add_structure(_make_structure())
        eng.damage_structure("s1", 50, (0, 0))
        # Settle all debris
        for _ in range(40):
            eng.tick(0.1)
        out = eng.to_three_js()
        assert len(out["debris"]) == 0

    def test_structure_has_frontend_field_names(self):
        """Frontend ensureBuildings() expects width/depth/height/destroyed."""
        eng = _make_engine()
        eng.add_structure(_make_structure(size=(20.0, 15.0, 10.0)))
        out = eng.to_three_js()
        s = out["structures"][0]
        # Full names for the Three.js frontend
        assert s["width"] == 20.0
        assert s["depth"] == 15.0
        assert s["height"] == 10.0
        # Short aliases kept for backward compat
        assert s["w"] == 20.0
        assert s["d"] == 15.0
        assert s["h"] == 10.0

    def test_destroyed_flag_false_when_intact(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(health=100, max_health=100))
        out = eng.to_three_js()
        assert out["structures"][0]["destroyed"] is False

    def test_destroyed_flag_true_when_destroyed(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(health=100, max_health=100))
        eng.damage_structure("s1", 99, (0, 0))
        out = eng.to_three_js()
        s = out["structures"][0]
        assert s["damage"] in ("destroyed", "collapsed", "critical")
        # destroyed flag is True only for destroyed/collapsed
        if s["damage"] in ("destroyed", "collapsed"):
            assert s["destroyed"] is True

    def test_destroyed_flag_true_when_collapsed(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(health=100, max_health=100))
        eng.damage_structure("s1", 200, (0, 0))  # overkill
        out = eng.to_three_js()
        s = out["structures"][0]
        assert s["destroyed"] is True


# ===========================================================================
# get_blocked_positions
# ===========================================================================

class TestBlockedPositions:
    def test_intact_structure_not_blocked(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(pos=(5, 5), size=(4, 4, 5)))
        blocked = eng.get_blocked_positions()
        assert len(blocked) == 0

    def test_destroyed_structure_blocked(self):
        eng = _make_engine()
        s = _make_structure(health=1, max_health=100, pos=(5, 5), size=(4, 4, 5))
        eng.add_structure(s)
        eng.damage_structure("s1", 100, (5, 5))
        blocked = eng.get_blocked_positions()
        assert len(blocked) > 0

    def test_cell_size_changes_grid(self):
        eng = _make_engine()
        s = _make_structure(health=1, max_health=100, pos=(10, 10), size=(6, 6, 5))
        eng.add_structure(s)
        eng.damage_structure("s1", 100, (10, 10))
        blocked_small = eng.get_blocked_positions(cell_size=1.0)
        blocked_large = eng.get_blocked_positions(cell_size=5.0)
        assert len(blocked_small) > len(blocked_large)


# ===========================================================================
# get_los_blockers
# ===========================================================================

class TestLosBlockers:
    def test_no_fires_no_blockers(self):
        eng = _make_engine()
        assert eng.get_los_blockers() == []

    def test_fire_produces_blocker(self):
        eng = _make_engine()
        eng.start_fire((10, 10), intensity=0.8)
        blockers = eng.get_los_blockers()
        assert len(blockers) == 1
        b = blockers[0]
        assert b["height"] > 0
        assert b["radius"] > 0
        assert 0 < b["opacity"] <= 1.0


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_fire_at_zero_intensity(self):
        eng = _make_engine()
        eng.start_fire((0, 0), intensity=0.0)
        eng.tick(0.1)
        assert len(eng.fires) == 0

    def test_damage_zero_does_nothing(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(health=50))
        eng.damage_structure("s1", 0, (0, 0))
        assert eng.structures[0].health == 50.0

    def test_negative_damage_treated_as_zero(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(health=50))
        result = eng.damage_structure("s1", -10, (0, 0))
        assert eng.structures[0].health == 50.0
        assert result["damage"] == 0

    def test_multiple_structures(self):
        eng = _make_engine()
        eng.add_structure(_make_structure(sid="a", pos=(0, 0)))
        eng.add_structure(_make_structure(sid="b", pos=(50, 50)))
        eng.damage_structure("a", 30, (0, 0))
        assert eng._structure_map["a"].health == 70.0
        assert eng._structure_map["b"].health == 100.0

    def test_tick_with_no_entities(self):
        eng = _make_engine()
        events = eng.tick(0.1)
        assert events["fires_spread"] == []
        assert events["debris_settled"] == 0

    def test_fire_color_ranges(self):
        assert _fire_color(0.1) == "#ff4400"
        assert _fire_color(0.4) == "#ff6600"
        assert _fire_color(0.7) == "#ff8800"
        assert _fire_color(0.9) == "#ffcc00"

    def test_structure_type_values(self):
        for st in StructureType:
            assert isinstance(st.value, str)

    def test_fire_radius_clamped(self):
        eng = _make_engine()
        f = eng.start_fire((0, 0), radius=-5)
        assert f.radius >= 0.1

    def test_fire_intensity_clamped(self):
        eng = _make_engine()
        f = eng.start_fire((0, 0), intensity=2.0)
        assert f.intensity <= 1.0

    def test_debris_dataclass(self):
        d = Debris(
            debris_id="d1",
            position=(1, 2),
            velocity=(3, 4),
            angular_velocity=1.0,
            size=0.5,
            material="concrete",
        )
        assert d.is_active is True
        assert d.time_alive == 0.0

    def test_fire_dataclass(self):
        f = Fire(
            fire_id="f1",
            position=(1, 2),
            radius=3.0,
            intensity=0.5,
            fuel_remaining=30.0,
        )
        assert f.spread_rate == 0.5
        assert f.temperature == 800.0

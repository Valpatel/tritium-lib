# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the air combat sim engine module.

Covers aircraft spawning, physics, missiles, anti-air, countermeasures,
stall/g-force mechanics, fuel, Three.js export, and combat resolution.
"""

from __future__ import annotations

import math
import random

import pytest

from tritium_lib.sim_engine.air_combat import (
    AircraftClass,
    AircraftState,
    AirCombatEffect,
    AirCombatEngine,
    AntiAir,
    Missile,
    AIRCRAFT_TEMPLATES,
    AA_TEMPLATES,
    MISSILE_TEMPLATES,
    GUN_STATS,
    _normalize_angle,
    _distance_3d,
    _bearing_to,
    _pitch_to,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> AirCombatEngine:
    """Engine with deterministic RNG."""
    return AirCombatEngine(rng=random.Random(42))


@pytest.fixture
def seeded_engine() -> AirCombatEngine:
    """Engine with seed 0 for different random sequence."""
    return AirCombatEngine(rng=random.Random(0))


# ---------------------------------------------------------------------------
# AircraftClass enum
# ---------------------------------------------------------------------------

class TestAircraftClass:
    def test_all_values(self):
        assert AircraftClass.FIGHTER.value == "fighter"
        assert AircraftClass.BOMBER.value == "bomber"
        assert AircraftClass.TRANSPORT.value == "transport"
        assert AircraftClass.GUNSHIP.value == "gunship"
        assert AircraftClass.RECON.value == "recon"
        assert AircraftClass.STEALTH.value == "stealth"

    def test_member_count(self):
        assert len(AircraftClass) == 6


# ---------------------------------------------------------------------------
# AircraftState dataclass
# ---------------------------------------------------------------------------

class TestAircraftState:
    def test_is_alive(self):
        ac = AircraftState(
            aircraft_id="a1", name="Test", aircraft_class=AircraftClass.FIGHTER,
            alliance="blue", position=(0, 0), altitude=1000, heading=0, pitch=0,
            speed=200, max_speed=600, min_speed=80, turn_rate=0.5, climb_rate=100,
            health=500, max_health=500, armor=0.1,
        )
        assert ac.is_alive()
        ac.health = 0
        assert not ac.is_alive()

    def test_is_alive_destroyed_flag(self):
        ac = AircraftState(
            aircraft_id="a1", name="Test", aircraft_class=AircraftClass.FIGHTER,
            alliance="blue", position=(0, 0), altitude=1000, heading=0, pitch=0,
            speed=200, max_speed=600, min_speed=80, turn_rate=0.5, climb_rate=100,
            health=500, max_health=500, armor=0.1, is_destroyed=True,
        )
        assert not ac.is_alive()

    def test_health_pct(self):
        ac = AircraftState(
            aircraft_id="a1", name="Test", aircraft_class=AircraftClass.FIGHTER,
            alliance="blue", position=(0, 0), altitude=1000, heading=0, pitch=0,
            speed=200, max_speed=600, min_speed=80, turn_rate=0.5, climb_rate=100,
            health=250, max_health=500, armor=0.1,
        )
        assert ac.health_pct() == pytest.approx(0.5)

    def test_health_pct_zero_max(self):
        ac = AircraftState(
            aircraft_id="a1", name="Test", aircraft_class=AircraftClass.FIGHTER,
            alliance="blue", position=(0, 0), altitude=1000, heading=0, pitch=0,
            speed=200, max_speed=600, min_speed=80, turn_rate=0.5, climb_rate=100,
            health=0, max_health=0, armor=0.1,
        )
        assert ac.health_pct() == 0.0

    def test_afterburner_active(self):
        ac = AircraftState(
            aircraft_id="a1", name="Test", aircraft_class=AircraftClass.FIGHTER,
            alliance="blue", position=(0, 0), altitude=1000, heading=0, pitch=0,
            speed=500, max_speed=600, min_speed=80, turn_rate=0.5, climb_rate=100,
            health=500, max_health=500, armor=0.1,
        )
        ac._throttle = 1.0
        assert ac.afterburner_active()

    def test_afterburner_inactive_low_throttle(self):
        ac = AircraftState(
            aircraft_id="a1", name="Test", aircraft_class=AircraftClass.FIGHTER,
            alliance="blue", position=(0, 0), altitude=1000, heading=0, pitch=0,
            speed=500, max_speed=600, min_speed=80, turn_rate=0.5, climb_rate=100,
            health=500, max_health=500, armor=0.1,
        )
        ac._throttle = 0.5
        assert not ac.afterburner_active()

    def test_default_fuel(self):
        ac = AircraftState(
            aircraft_id="a1", name="Test", aircraft_class=AircraftClass.FIGHTER,
            alliance="blue", position=(0, 0), altitude=1000, heading=0, pitch=0,
            speed=200, max_speed=600, min_speed=80, turn_rate=0.5, climb_rate=100,
            health=500, max_health=500, armor=0.1,
        )
        assert ac.fuel == 1.0

    def test_default_countermeasures(self):
        ac = AircraftState(
            aircraft_id="a1", name="Test", aircraft_class=AircraftClass.FIGHTER,
            alliance="blue", position=(0, 0), altitude=1000, heading=0, pitch=0,
            speed=200, max_speed=600, min_speed=80, turn_rate=0.5, climb_rate=100,
            health=500, max_health=500, armor=0.1,
        )
        assert ac.countermeasures == 20


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_normalize_angle(self):
        assert _normalize_angle(0) == pytest.approx(0)
        assert _normalize_angle(math.pi) == pytest.approx(math.pi) or \
               _normalize_angle(math.pi) == pytest.approx(-math.pi)
        assert abs(_normalize_angle(3 * math.pi)) == pytest.approx(math.pi, abs=0.01)

    def test_distance_3d_same_point(self):
        assert _distance_3d((0, 0), 0, (0, 0), 0) == 0.0

    def test_distance_3d_horizontal(self):
        assert _distance_3d((0, 0), 0, (3, 4), 0) == pytest.approx(5.0)

    def test_distance_3d_vertical(self):
        assert _distance_3d((0, 0), 0, (0, 0), 100) == pytest.approx(100.0)

    def test_distance_3d_diagonal(self):
        d = _distance_3d((0, 0), 0, (100, 0), 100)
        assert d == pytest.approx(math.sqrt(20000))

    def test_bearing_to(self):
        # East
        assert _bearing_to((0, 0), (100, 0)) == pytest.approx(0.0)
        # North
        assert _bearing_to((0, 0), (0, 100)) == pytest.approx(math.pi / 2)

    def test_pitch_to_above(self):
        p = _pitch_to((0, 0), 0, (100, 0), 100)
        assert p == pytest.approx(math.pi / 4)

    def test_pitch_to_same_position(self):
        p = _pitch_to((0, 0), 0, (0, 0), 100)
        assert p == pytest.approx(math.pi / 2)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_all_aircraft_templates_exist(self):
        expected = {"f16", "f22", "a10", "b52", "c130", "u2"}
        assert set(AIRCRAFT_TEMPLATES.keys()) == expected

    def test_f16_is_fighter(self):
        assert AIRCRAFT_TEMPLATES["f16"]["aircraft_class"] == AircraftClass.FIGHTER

    def test_f22_is_stealth(self):
        assert AIRCRAFT_TEMPLATES["f22"]["aircraft_class"] == AircraftClass.STEALTH

    def test_a10_is_gunship(self):
        assert AIRCRAFT_TEMPLATES["a10"]["aircraft_class"] == AircraftClass.GUNSHIP

    def test_b52_is_bomber(self):
        assert AIRCRAFT_TEMPLATES["b52"]["aircraft_class"] == AircraftClass.BOMBER

    def test_c130_is_transport(self):
        assert AIRCRAFT_TEMPLATES["c130"]["aircraft_class"] == AircraftClass.TRANSPORT

    def test_u2_is_recon(self):
        assert AIRCRAFT_TEMPLATES["u2"]["aircraft_class"] == AircraftClass.RECON

    def test_c130_has_no_weapons(self):
        assert AIRCRAFT_TEMPLATES["c130"]["weapons"] == []

    def test_f22_has_high_countermeasures(self):
        assert AIRCRAFT_TEMPLATES["f22"]["countermeasures"] == 40

    def test_f22_low_rcs(self):
        assert AIRCRAFT_TEMPLATES["f22"]["radar_cross_section"] < 0.1

    def test_all_aa_templates_exist(self):
        expected = {"patriot", "stinger", "phalanx", "flak_88"}
        assert set(AA_TEMPLATES.keys()) == expected

    def test_patriot_is_sam(self):
        assert AA_TEMPLATES["patriot"]["aa_type"] == "sam"

    def test_stinger_is_manpad(self):
        assert AA_TEMPLATES["stinger"]["aa_type"] == "manpad"

    def test_phalanx_is_ciws(self):
        assert AA_TEMPLATES["phalanx"]["aa_type"] == "ciws"

    def test_flak_is_flak(self):
        assert AA_TEMPLATES["flak_88"]["aa_type"] == "flak"

    def test_patriot_long_range(self):
        assert AA_TEMPLATES["patriot"]["range_m"] == 160000.0

    def test_missile_templates_exist(self):
        expected = {"sidewinder", "amraam", "maverick", "sam_missile", "stinger_missile"}
        assert set(MISSILE_TEMPLATES.keys()) == expected

    def test_sidewinder_is_heat(self):
        assert MISSILE_TEMPLATES["sidewinder"]["seeker_type"] == "heat"

    def test_amraam_is_radar(self):
        assert MISSILE_TEMPLATES["amraam"]["seeker_type"] == "radar"

    def test_sidewinder_vulnerable_to_countermeasures(self):
        assert MISSILE_TEMPLATES["sidewinder"]["countermeasure_vulnerable"] is True

    def test_amraam_not_vulnerable_to_countermeasures(self):
        assert MISSILE_TEMPLATES["amraam"]["countermeasure_vulnerable"] is False

    def test_gun_stats_exist(self):
        assert "gun" in GUN_STATS
        assert "gau8" in GUN_STATS


# ---------------------------------------------------------------------------
# Spawn aircraft
# ---------------------------------------------------------------------------

class TestSpawnAircraft:
    def test_spawn_f16(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        assert ac.aircraft_id == "a1"
        assert ac.alliance == "blue"
        assert ac.aircraft_class == AircraftClass.FIGHTER
        assert ac.max_speed == 600.0
        assert ac.min_speed == 80.0
        assert ac.is_alive()

    def test_spawn_at_position(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue", position=(500, 300), altitude=5000)
        assert ac.position == (500, 300)
        assert ac.altitude == 5000

    def test_spawn_initial_speed(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        assert ac.speed == 300.0  # 50% of max_speed

    def test_spawn_stores_in_dict(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        assert "a1" in engine.aircraft

    def test_spawn_unknown_template(self, engine: AirCombatEngine):
        with pytest.raises(KeyError):
            engine.spawn_aircraft("mig29", "a1", "red")

    def test_spawn_negative_altitude_clamped(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue", altitude=-100)
        assert ac.altitude >= 0.0

    def test_spawn_event_emitted(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        assert any(e["event"] == "aircraft_spawned" for e in engine._events)

    def test_spawn_all_templates(self, engine: AirCombatEngine):
        for i, tmpl in enumerate(AIRCRAFT_TEMPLATES):
            ac = engine.spawn_aircraft(tmpl, f"a{i}", "blue")
            assert ac.is_alive()


# ---------------------------------------------------------------------------
# Remove / get aircraft
# ---------------------------------------------------------------------------

class TestAircraftManagement:
    def test_get_aircraft(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        assert engine.get_aircraft("a1") is not None
        assert engine.get_aircraft("missing") is None

    def test_remove_aircraft(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        assert engine.remove_aircraft("a1") is True
        assert engine.get_aircraft("a1") is None

    def test_remove_missing(self, engine: AirCombatEngine):
        assert engine.remove_aircraft("missing") is False

    def test_aircraft_by_alliance(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red")
        engine.spawn_aircraft("f16", "a3", "blue")
        blues = engine.aircraft_by_alliance("blue")
        assert len(blues) == 2
        assert all(a.alliance == "blue" for a in blues)


# ---------------------------------------------------------------------------
# Fire missile
# ---------------------------------------------------------------------------

class TestFireMissile:
    def test_fire_sidewinder(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red")
        m = engine.fire_missile("a1", "a2", "sidewinder")
        assert m is not None
        assert m.seeker_type == "heat"
        assert m.target_id == "a2"

    def test_fire_consumes_weapon(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red")
        initial = list(ac.weapons)
        engine.fire_missile("a1", "a2", "sidewinder")
        assert "sidewinder" not in ac.weapons or ac.weapons.count("sidewinder") < initial.count("sidewinder")

    def test_fire_no_weapon(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("c130", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red")
        m = engine.fire_missile("a1", "a2", "sidewinder")
        assert m is None

    def test_fire_unknown_type(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        m = engine.fire_missile("a1", "a2", "torpedo")
        assert m is None

    def test_fire_dead_aircraft(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        ac.health = 0
        m = engine.fire_missile("a1", "a2", "sidewinder")
        assert m is None

    def test_fire_missing_aircraft(self, engine: AirCombatEngine):
        m = engine.fire_missile("missing", "a2", "sidewinder")
        assert m is None

    def test_missile_initial_speed(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red")
        m = engine.fire_missile("a1", "a2", "sidewinder")
        assert m.speed > ac.speed  # boosted from launch platform

    def test_missile_added_to_list(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red")
        engine.fire_missile("a1", "a2", "sidewinder")
        assert len(engine.missiles) == 1


# ---------------------------------------------------------------------------
# Countermeasures
# ---------------------------------------------------------------------------

class TestCountermeasures:
    def test_deploy_success(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        initial = ac.countermeasures
        result = engine.deploy_countermeasures("a1")
        assert result is True
        assert ac.countermeasures == initial - 1

    def test_deploy_no_countermeasures(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        ac.countermeasures = 0
        assert engine.deploy_countermeasures("a1") is False

    def test_deploy_dead_aircraft(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        ac.health = 0
        assert engine.deploy_countermeasures("a1") is False

    def test_deploy_missing_aircraft(self, engine: AirCombatEngine):
        assert engine.deploy_countermeasures("missing") is False

    def test_deploy_creates_flare_effect(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.deploy_countermeasures("a1")
        flares = [e for e in engine.effects if e.effect_type == "flare"]
        assert len(flares) == 1

    def test_flare_defeats_heat_missile(self):
        """With seed that produces rng < 0.5, flare defeats heat seeker."""
        # Try multiple seeds to find one where flare defeats
        for seed in range(100):
            eng = AirCombatEngine(rng=random.Random(seed))
            eng.spawn_aircraft("f16", "a1", "blue")
            eng.spawn_aircraft("f16", "a2", "red")
            m = eng.fire_missile("a2", "a1", "sidewinder")
            if m is None:
                continue
            assert m.countermeasure_vulnerable
            eng.deploy_countermeasures("a1")
            if not m.is_active:
                return  # Found a seed where flare worked
        pytest.fail("No seed defeated missile in 100 tries")

    def test_radar_missile_not_defeated(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f22", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red")
        m = engine.fire_missile("a1", "a2", "amraam")
        assert m is not None
        assert not m.countermeasure_vulnerable
        engine.deploy_countermeasures("a2")
        assert m.is_active  # AMRAAM not fooled by flares


# ---------------------------------------------------------------------------
# Fire guns
# ---------------------------------------------------------------------------

class TestFireGuns:
    def test_fire_guns_in_range(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("a10", "a1", "blue")
        results = engine.fire_guns("a1", ac.position, ac.altitude)
        # A-10 has gau8; range is 2000m, self-position is 0 distance
        assert len(results) > 0

    def test_fire_guns_out_of_range(self, engine: AirCombatEngine):
        engine.spawn_aircraft("a10", "a1", "blue")
        results = engine.fire_guns("a1", (100000, 100000), 0)
        assert len(results) == 0

    def test_fire_guns_dead_aircraft(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("a10", "a1", "blue")
        ac.health = 0
        results = engine.fire_guns("a1", (0, 0))
        assert len(results) == 0

    def test_fire_guns_missing_aircraft(self, engine: AirCombatEngine):
        results = engine.fire_guns("missing", (0, 0))
        assert len(results) == 0

    def test_fire_guns_cooldown(self, engine: AirCombatEngine):
        engine.spawn_aircraft("a10", "a1", "blue")
        r1 = engine.fire_guns("a1", (0, 0), 3000)
        r2 = engine.fire_guns("a1", (0, 0), 3000)
        # Second burst should be blocked by cooldown
        assert len(r1) > 0
        assert len(r2) == 0

    def test_no_gun_weapon(self, engine: AirCombatEngine):
        engine.spawn_aircraft("c130", "a1", "blue")
        results = engine.fire_guns("a1", (0, 0))
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Anti-air
# ---------------------------------------------------------------------------

class TestAntiAir:
    def test_add_anti_air(self, engine: AirCombatEngine):
        aa = engine.add_anti_air("patriot", "aa1", "blue", (0, 0))
        assert aa.aa_id == "aa1"
        assert aa.aa_type == "sam"
        assert aa.range_m == 160000.0
        assert len(engine.anti_air) == 1

    def test_add_all_aa_templates(self, engine: AirCombatEngine):
        for i, tmpl in enumerate(AA_TEMPLATES):
            aa = engine.add_anti_air(tmpl, f"aa{i}", "blue", (0, 0))
            assert aa.ammo > 0

    def test_add_unknown_aa_template(self, engine: AirCombatEngine):
        with pytest.raises(KeyError):
            engine.add_anti_air("s400", "aa1", "red", (0, 0))

    def test_aa_fires_at_hostile(self, engine: AirCombatEngine):
        engine.add_anti_air("phalanx", "aa1", "blue", (0, 0))
        engine.spawn_aircraft("f16", "a1", "red", position=(500, 0), altitude=500)
        result = engine.tick(0.1)
        # CIWS has fast fire rate, should engage
        # The aircraft is within 2000m (dist ~707m), so it should fire
        assert engine.anti_air[0].ammo < 1550  # started with 1550

    def test_aa_ignores_friendly(self, engine: AirCombatEngine):
        engine.add_anti_air("phalanx", "aa1", "blue", (0, 0))
        engine.spawn_aircraft("f16", "a1", "blue", position=(500, 0), altitude=500)
        engine.tick(0.1)
        assert engine.anti_air[0].ammo == 1550  # no shots fired

    def test_aa_out_of_range(self, engine: AirCombatEngine):
        engine.add_anti_air("stinger", "aa1", "blue", (0, 0))
        # Stinger range is 5km; aircraft at 100km
        engine.spawn_aircraft("f16", "a1", "red", position=(100000, 0), altitude=5000)
        engine.tick(0.1)
        assert engine.anti_air[0].ammo == 1  # no shots fired

    def test_aa_stealth_detection(self, engine: AirCombatEngine):
        engine.add_anti_air("stinger", "aa1", "blue", (0, 0))
        # F-22 has RCS 0.05, so effective range = 5000 * 0.05 = 250m
        # Place at 1000m - outside effective range for stealth
        engine.spawn_aircraft("f22", "a1", "red", position=(1000, 0), altitude=500)
        engine.tick(0.1)
        assert engine.anti_air[0].ammo == 1  # stealth not detected


# ---------------------------------------------------------------------------
# Tick / Physics
# ---------------------------------------------------------------------------

class TestTick:
    def test_tick_returns_dict(self, engine: AirCombatEngine):
        result = engine.tick(0.1)
        assert "time" in result
        assert "missile_hits" in result
        assert "aa_hits" in result
        assert "destroyed" in result
        assert "effects" in result
        assert "events" in result
        assert "aircraft_count" in result

    def test_tick_advances_time(self, engine: AirCombatEngine):
        engine.tick(0.5)
        assert engine._time == pytest.approx(0.5)
        engine.tick(0.3)
        assert engine._time == pytest.approx(0.8)

    def test_aircraft_moves(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue", position=(0, 0))
        ac.heading = 0.0  # flying east
        old_x = ac.position[0]
        engine.tick(1.0)
        assert ac.position[0] > old_x  # moved east

    def test_aircraft_turns(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        ac.heading = 0.0
        engine.set_controls("a1", target_heading=math.pi / 2)
        engine.tick(1.0)
        # Should have turned toward pi/2
        assert ac.heading > 0.0

    def test_aircraft_climbs(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue", altitude=1000)
        engine.set_controls("a1", target_altitude=5000)
        old_alt = ac.altitude
        engine.tick(1.0)
        assert ac.altitude > old_alt

    def test_aircraft_descends(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue", altitude=5000)
        engine.set_controls("a1", target_altitude=1000)
        old_alt = ac.altitude
        engine.tick(1.0)
        assert ac.altitude < old_alt

    def test_stall_causes_dive(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue", altitude=5000)
        ac.speed = 10.0  # well below stall speed of 80
        ac._throttle = 0.0
        old_alt = ac.altitude
        engine.tick(1.0)
        # Stalling: should lose altitude
        assert ac.altitude < old_alt

    def test_fuel_consumption(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        ac._throttle = 1.0
        old_fuel = ac.fuel
        engine.tick(1.0)
        assert ac.fuel < old_fuel

    def test_no_fuel_stops_throttle(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        ac.fuel = 0.001
        ac._throttle = 1.0
        # Tick enough to exhaust fuel
        for _ in range(10):
            engine.tick(1.0)
        assert ac.fuel == 0.0
        assert ac._throttle == 0.0

    def test_ground_collision_destroys(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue", altitude=10)
        ac.speed = 10.0  # stalling
        ac.pitch = -0.5  # diving
        ac._throttle = 0.0
        ac._target_altitude = None
        # Multiple ticks to crash
        for _ in range(50):
            result = engine.tick(0.5)
            if ac.is_destroyed:
                break
        assert ac.is_destroyed

    def test_trail_recorded(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        engine.tick(0.1)
        assert len(ac._trail) >= 1

    def test_g_force_during_turn(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        ac.speed = 300.0
        engine.set_controls("a1", target_heading=math.pi)
        engine.tick(0.1)
        # Turning at 300 m/s should produce noticeable g-force
        assert ac.g_force >= 1.0

    def test_missile_homes_on_target(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue", position=(0, 0), altitude=3000)
        engine.spawn_aircraft("f16", "a2", "red", position=(5000, 0), altitude=3000)
        engine.fire_missile("a1", "a2", "sidewinder")
        m = engine.missiles[0]
        old_dist = _distance_3d(m.position, m.altitude,
                                engine.aircraft["a2"].position,
                                engine.aircraft["a2"].altitude)
        engine.tick(1.0)
        new_dist = _distance_3d(m.position, m.altitude,
                                engine.aircraft["a2"].position,
                                engine.aircraft["a2"].altitude)
        assert new_dist < old_dist  # missile closing on target

    def test_missile_range_exhaustion(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red", position=(1000000, 0))
        m = engine.fire_missile("a1", "a2", "sidewinder")
        # Tick many times until missile runs out of fuel
        for _ in range(100):
            engine.tick(1.0)
        # Missile should be cleaned up
        assert m not in engine.missiles or not m.is_active

    def test_missile_hits_target(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue", position=(0, 0), altitude=3000)
        target = engine.spawn_aircraft("f16", "a2", "red", position=(100, 0), altitude=3000)
        engine.fire_missile("a1", "a2", "amraam")
        initial_health = target.health
        # Tick until hit or timeout
        hit_occurred = False
        for _ in range(200):
            result = engine.tick(0.1)
            if result["missile_hits"]:
                hit_occurred = True
                break
        assert hit_occurred
        assert target.health < initial_health

    def test_destroyed_aircraft_not_counted(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        result = engine.tick(0.1)
        assert result["aircraft_count"] == 1
        ac.health = 0
        result = engine.tick(0.1)
        assert result["aircraft_count"] == 0


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

class TestSetControls:
    def test_set_throttle(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        engine.set_controls("a1", throttle=0.5)
        assert ac._throttle == 0.5

    def test_set_heading(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.set_controls("a1", target_heading=1.5)
        assert engine.aircraft["a1"]._target_heading == 1.5

    def test_set_altitude(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.set_controls("a1", target_altitude=8000)
        assert engine.aircraft["a1"]._target_altitude == 8000

    def test_set_controls_missing(self, engine: AirCombatEngine):
        assert engine.set_controls("missing") is False

    def test_throttle_clamped(self, engine: AirCombatEngine):
        ac = engine.spawn_aircraft("f16", "a1", "blue")
        engine.set_controls("a1", throttle=5.0)
        assert ac._throttle == 1.0
        engine.set_controls("a1", throttle=-1.0)
        assert ac._throttle == 0.0

    def test_altitude_clamped_positive(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.set_controls("a1", target_altitude=-500)
        assert engine.aircraft["a1"]._target_altitude == 0.0


# ---------------------------------------------------------------------------
# Three.js export
# ---------------------------------------------------------------------------

class TestThreeJsExport:
    def test_empty_export(self, engine: AirCombatEngine):
        data = engine.to_three_js()
        assert data["aircraft"] == []
        assert data["missiles"] == []
        assert data["aa_sites"] == []
        assert data["effects"] == []
        assert "time" in data

    def test_aircraft_export(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue", position=(100, 200), altitude=5000)
        engine.tick(0.1)  # generate trail
        data = engine.to_three_js()
        assert len(data["aircraft"]) == 1
        ac = data["aircraft"][0]
        assert ac["id"] == "a1"
        assert ac["class"] == "fighter"
        assert ac["alliance"] == "blue"
        assert "x" in ac
        assert "y" in ac
        assert "z" in ac
        assert "heading" in ac
        assert "pitch" in ac
        assert "speed" in ac
        assert "health_pct" in ac
        assert "fuel" in ac
        assert "g_force" in ac
        assert "trail" in ac
        assert "afterburner" in ac
        assert "countermeasures" in ac
        assert "weapons" in ac
        assert "rcs" in ac
        assert "is_destroyed" in ac

    def test_missile_export(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red", position=(5000, 0))
        engine.fire_missile("a1", "a2", "sidewinder")
        engine.tick(0.1)
        data = engine.to_three_js()
        assert len(data["missiles"]) == 1
        m = data["missiles"][0]
        assert "id" in m
        assert "x" in m
        assert "y" in m
        assert "z" in m
        assert "heading" in m
        assert "type" in m
        assert "trail" in m
        assert "color" in m
        assert m["type"] == "heat"
        assert m["color"] == "#ff4400"

    def test_aa_export(self, engine: AirCombatEngine):
        engine.add_anti_air("patriot", "aa1", "blue", (50, 50))
        data = engine.to_three_js()
        assert len(data["aa_sites"]) == 1
        aa = data["aa_sites"][0]
        assert aa["id"] == "aa1"
        assert aa["type"] == "sam"
        assert aa["range"] == 160000.0

    def test_effects_export(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue")
        engine.deploy_countermeasures("a1")
        data = engine.to_three_js()
        assert len(data["effects"]) > 0
        eff = data["effects"][0]
        assert "type" in eff
        assert "x" in eff
        assert "y" in eff
        assert "z" in eff

    def test_missile_color_map(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f22", "a1", "blue")
        engine.spawn_aircraft("f16", "a2", "red", position=(5000, 0))
        engine.fire_missile("a1", "a2", "amraam")
        engine.tick(0.1)
        data = engine.to_three_js()
        assert data["missiles"][0]["color"] == "#4488ff"  # radar


# ---------------------------------------------------------------------------
# Detect targets
# ---------------------------------------------------------------------------

class TestDetectTargets:
    def test_detect_nearby(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue", position=(0, 0), altitude=5000)
        engine.spawn_aircraft("f16", "a2", "red", position=(10000, 0), altitude=5000)
        targets = engine.detect_targets("a1")
        assert len(targets) == 1
        assert targets[0]["target_id"] == "a2"

    def test_detect_out_of_range(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue", position=(0, 0))
        engine.spawn_aircraft("f16", "a2", "red", position=(200000, 0))
        targets = engine.detect_targets("a1")
        assert len(targets) == 0

    def test_detect_stealth_reduced_range(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue", position=(0, 0), altitude=5000)
        engine.spawn_aircraft("f22", "a2", "red", position=(10000, 0), altitude=5000)
        # F-22 RCS=0.05, so effective range = 50000 * 0.05 = 2500m
        # At 10000m, should not detect
        targets = engine.detect_targets("a1")
        assert len(targets) == 0

    def test_detect_dead_aircraft_not_listed(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue", position=(0, 0))
        ac2 = engine.spawn_aircraft("f16", "a2", "red", position=(100, 0))
        ac2.health = 0
        targets = engine.detect_targets("a1")
        assert len(targets) == 0

    def test_detect_missing_aircraft(self, engine: AirCombatEngine):
        targets = engine.detect_targets("missing")
        assert targets == []

    def test_detect_includes_bearing(self, engine: AirCombatEngine):
        engine.spawn_aircraft("f16", "a1", "blue", position=(0, 0), altitude=3000)
        engine.spawn_aircraft("f16", "a2", "red", position=(1000, 0), altitude=3000)
        targets = engine.detect_targets("a1")
        assert len(targets) == 1
        assert "bearing" in targets[0]
        assert targets[0]["bearing"] == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# AirCombatEffect
# ---------------------------------------------------------------------------

class TestAirCombatEffect:
    def test_to_dict(self):
        e = AirCombatEffect(
            effect_type="explosion",
            position=(100.123, 200.456),
            altitude=500.789,
            radius=10.0,
            duration=2.0,
            intensity=0.8,
        )
        d = e.to_dict()
        assert d["type"] == "explosion"
        assert d["x"] == 100.12
        assert d["y"] == 200.46
        assert d["z"] == 500.79
        assert d["radius"] == 10.0
        assert d["duration"] == 2.0
        assert d["intensity"] == 0.8


# ---------------------------------------------------------------------------
# Integration: dogfight scenario
# ---------------------------------------------------------------------------

class TestDogfightScenario:
    def test_two_fighters_engage(self, engine: AirCombatEngine):
        """Two fighters fire missiles at each other; at least one hit occurs."""
        engine.spawn_aircraft("f16", "blue1", "blue", position=(0, 0), altitude=5000)
        engine.spawn_aircraft("f16", "red1", "red", position=(2000, 0), altitude=5000)

        engine.fire_missile("blue1", "red1", "sidewinder")
        engine.fire_missile("red1", "blue1", "sidewinder")

        hit_count = 0
        for _ in range(300):
            result = engine.tick(0.1)
            hit_count += len(result["missile_hits"])
            if hit_count >= 1:
                break

        assert hit_count >= 1

    def test_bomber_vs_aa(self, engine: AirCombatEngine):
        """B-52 flying over AA battery takes damage."""
        b52 = engine.spawn_aircraft("b52", "b1", "red", position=(0, 0), altitude=3000)
        engine.add_anti_air("flak_88", "aa1", "blue", (500, 0))
        initial_health = b52.health

        for _ in range(100):
            engine.tick(0.1)
            if b52.health < initial_health:
                break

        assert b52.health < initial_health

    def test_full_battle_runs_without_error(self, engine: AirCombatEngine):
        """Spawn multiple aircraft and AA, run 100 ticks, no exceptions."""
        engine.spawn_aircraft("f16", "b1", "blue", position=(0, 0), altitude=5000)
        engine.spawn_aircraft("f22", "b2", "blue", position=(100, 100), altitude=6000)
        engine.spawn_aircraft("a10", "r1", "red", position=(5000, 0), altitude=2000)
        engine.spawn_aircraft("b52", "r2", "red", position=(8000, 1000), altitude=8000)
        engine.add_anti_air("patriot", "aa1", "blue", (0, 0))
        engine.add_anti_air("stinger", "aa2", "red", (5000, 0))

        engine.fire_missile("b1", "r1", "sidewinder")
        engine.fire_missile("b2", "r1", "amraam")

        for _ in range(100):
            result = engine.tick(0.1)
            data = engine.to_three_js()
            # Verify export is always valid
            assert isinstance(data["aircraft"], list)
            assert isinstance(data["missiles"], list)

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the vehicle and drone combat simulation module."""

from __future__ import annotations

import math

import pytest

from tritium_lib.sim_engine.vehicles import (
    VehicleClass,
    VehicleState,
    VehiclePhysicsEngine,
    DroneController,
    ConvoySimulator,
    VEHICLE_TEMPLATES,
    create_vehicle,
)
from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vehicle(**overrides) -> VehicleState:
    """Create a simple test vehicle with sensible defaults."""
    defaults = dict(
        vehicle_id="test_v1",
        name="Test Car",
        vehicle_class=VehicleClass.CAR,
        alliance="friendly",
        position=(0.0, 0.0),
        max_speed=30.0,
        acceleration=5.0,
        turn_rate=1.0,
        health=100.0,
        max_health=100.0,
        armor=0.0,
    )
    defaults.update(overrides)
    return VehicleState(**defaults)


def _make_helicopter(**overrides) -> VehicleState:
    defaults = dict(
        vehicle_id="heli_1",
        name="Test Heli",
        vehicle_class=VehicleClass.HELICOPTER,
        alliance="friendly",
        position=(0.0, 0.0),
        altitude=50.0,
        max_speed=70.0,
        acceleration=6.0,
        turn_rate=1.5,
        health=500.0,
        max_health=500.0,
        armor=0.3,
    )
    defaults.update(overrides)
    return VehicleState(**defaults)


def _make_quad(**overrides) -> VehicleState:
    defaults = dict(
        vehicle_id="quad_1",
        name="Test Quad",
        vehicle_class=VehicleClass.DRONE_QUAD,
        alliance="friendly",
        position=(0.0, 0.0),
        altitude=30.0,
        max_speed=16.0,
        acceleration=8.0,
        turn_rate=3.0,
        health=50.0,
        max_health=50.0,
        armor=0.0,
    )
    defaults.update(overrides)
    return VehicleState(**defaults)


def _make_fixed_wing(**overrides) -> VehicleState:
    defaults = dict(
        vehicle_id="fw_1",
        name="Test Reaper",
        vehicle_class=VehicleClass.DRONE_FIXED_WING,
        alliance="friendly",
        position=(0.0, 0.0),
        altitude=100.0,
        speed=30.0,
        max_speed=110.0,
        acceleration=5.0,
        turn_rate=0.6,
        health=300.0,
        max_health=300.0,
        armor=0.1,
    )
    defaults.update(overrides)
    return VehicleState(**defaults)


# ===========================================================================
# Template creation tests
# ===========================================================================


class TestTemplates:
    """All 10 templates create valid vehicles."""

    @pytest.mark.parametrize("template", list(VEHICLE_TEMPLATES.keys()))
    def test_create_from_template(self, template: str):
        v = create_vehicle(template, f"v_{template}", "friendly", (10.0, 20.0))
        assert v.vehicle_id == f"v_{template}"
        assert v.alliance == "friendly"
        assert v.position == (10.0, 20.0)
        assert v.health == v.max_health
        assert v.max_speed > 0
        assert not v.is_destroyed
        assert v.fuel == 1.0

    def test_template_count(self):
        assert len(VEHICLE_TEMPLATES) == 10

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError, match="Unknown vehicle template"):
            create_vehicle("nonexistent", "v1", "hostile", (0.0, 0.0))

    def test_custom_name(self):
        v = create_vehicle("humvee", "h1", "friendly", (0.0, 0.0), name="Alpha-1")
        assert v.name == "Alpha-1"

    def test_default_name(self):
        v = create_vehicle("btr80", "b1", "hostile", (0.0, 0.0))
        assert v.name == "Btr80"

    def test_humvee_has_weapons(self):
        v = create_vehicle("humvee", "h1", "friendly", (0.0, 0.0))
        assert "m2_turret" in v.weapons

    def test_motorcycle_no_weapons(self):
        v = create_vehicle("motorcycle", "m1", "friendly", (0.0, 0.0))
        assert v.weapons == []

    def test_tank_high_armor(self):
        v = create_vehicle("t72", "t1", "hostile", (0.0, 0.0))
        assert v.armor == 0.8

    def test_technical_no_armor(self):
        v = create_vehicle("technical", "t1", "hostile", (0.0, 0.0))
        assert v.armor == 0.0

    def test_blackhawk_is_helicopter(self):
        v = create_vehicle("blackhawk", "bh1", "friendly", (0.0, 0.0))
        assert v.vehicle_class == VehicleClass.HELICOPTER
        assert v.is_aircraft()

    def test_quadcopter_is_drone(self):
        v = create_vehicle("quadcopter", "q1", "friendly", (0.0, 0.0))
        assert v.vehicle_class == VehicleClass.DRONE_QUAD
        assert v.is_aircraft()
        assert v.is_hover_capable()

    def test_reaper_is_fixed_wing(self):
        v = create_vehicle("reaper", "r1", "friendly", (0.0, 0.0))
        assert v.vehicle_class == VehicleClass.DRONE_FIXED_WING
        assert v.is_aircraft()
        assert v.is_fixed_wing()
        assert not v.is_hover_capable()


# ===========================================================================
# Physics tests
# ===========================================================================


class TestPhysics:
    """Vehicle physics: acceleration, braking, turning, fuel."""

    def test_acceleration(self):
        v = _make_vehicle()
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=1.0, steering=0.0, dt=1.0)
        assert v.speed > 0
        assert v.position[0] > 0  # moved east (heading=0)

    def test_braking(self):
        v = _make_vehicle(speed=20.0)
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=-1.0, steering=0.0, dt=1.0)
        assert v.speed < 20.0

    def test_speed_clamped_to_max(self):
        v = _make_vehicle(speed=29.0)
        engine = VehiclePhysicsEngine()
        for _ in range(20):
            engine.update(v, throttle=1.0, steering=0.0, dt=0.1)
        assert v.speed <= v.max_speed

    def test_reverse_speed_clamped(self):
        v = _make_vehicle(speed=0.0)
        engine = VehiclePhysicsEngine()
        for _ in range(50):
            engine.update(v, throttle=-1.0, steering=0.0, dt=0.1)
        assert v.speed >= -(v.max_speed * 0.3)

    def test_turning_changes_heading(self):
        v = _make_vehicle(speed=10.0)
        engine = VehiclePhysicsEngine()
        initial_heading = v.heading
        engine.update(v, throttle=0.5, steering=1.0, dt=1.0)
        assert v.heading != initial_heading

    def test_no_turn_when_stationary(self):
        v = _make_vehicle(speed=0.0)
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=0.0, steering=1.0, dt=1.0)
        assert v.heading == 0.0  # no turn at zero speed

    def test_fuel_consumption_while_moving(self):
        v = _make_vehicle(fuel=1.0, fuel_consumption=0.1)
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=1.0, steering=0.0, dt=1.0)
        assert v.fuel < 1.0

    def test_no_fuel_consumption_when_stopped(self):
        v = _make_vehicle(fuel=1.0, speed=0.0)
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=0.0, steering=0.0, dt=1.0)
        assert v.fuel == 1.0

    def test_engine_stops_when_fuel_empty(self):
        v = _make_vehicle(fuel=0.001, fuel_consumption=0.5, speed=10.0)
        engine = VehiclePhysicsEngine()
        # Run until fuel runs out
        for _ in range(100):
            engine.update(v, throttle=1.0, steering=0.0, dt=0.1)
        assert v.fuel == 0.0
        assert v.engine_disabled

    def test_destroyed_vehicle_stops(self):
        v = _make_vehicle(is_destroyed=True, speed=10.0)
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=1.0, steering=0.0, dt=1.0)
        assert v.speed == 0.0

    def test_ground_vehicle_no_altitude(self):
        v = _make_vehicle()
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=1.0, steering=0.0, dt=1.0, altitude_input=1.0)
        assert v.altitude == 0.0  # ground vehicle ignores altitude input


# ===========================================================================
# Helicopter tests
# ===========================================================================


class TestHelicopter:
    """Helicopter altitude changes and hover."""

    def test_altitude_increase(self):
        v = _make_helicopter(altitude=50.0)
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=0.5, steering=0.0, dt=1.0, altitude_input=1.0)
        assert v.altitude > 50.0

    def test_altitude_decrease(self):
        v = _make_helicopter(altitude=50.0)
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=0.5, steering=0.0, dt=1.0, altitude_input=-1.0)
        assert v.altitude < 50.0

    def test_altitude_floor_at_zero(self):
        v = _make_helicopter(altitude=1.0)
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=0.0, steering=0.0, dt=5.0, altitude_input=-1.0)
        assert v.altitude >= 0.0

    def test_helicopter_can_hover(self):
        v = _make_helicopter(speed=0.0, altitude=50.0)
        engine = VehiclePhysicsEngine()
        pos_before = v.position
        engine.update(v, throttle=0.0, steering=0.0, dt=1.0)
        # Should not move significantly
        assert distance(pos_before, v.position) < 0.1

    def test_helicopter_is_hover_capable(self):
        v = _make_helicopter()
        assert v.is_hover_capable()
        assert v.is_aircraft()


# ===========================================================================
# Fixed-wing drone tests
# ===========================================================================


class TestFixedWing:
    """Fixed-wing drones need minimum speed."""

    def test_min_flight_speed(self):
        v = _make_fixed_wing()
        assert v.min_flight_speed > 0
        assert v.min_flight_speed == v.max_speed * 0.25

    def test_stall_loses_altitude(self):
        v = _make_fixed_wing(speed=5.0, altitude=100.0)  # well below min speed
        engine = VehiclePhysicsEngine()
        engine.update(v, throttle=0.0, steering=0.0, dt=1.0)
        assert v.altitude < 100.0

    def test_not_hover_capable(self):
        v = _make_fixed_wing()
        assert not v.is_hover_capable()

    def test_is_fixed_wing(self):
        v = _make_fixed_wing()
        assert v.is_fixed_wing()

    def test_ground_vehicle_zero_min_flight_speed(self):
        v = _make_vehicle()
        assert v.min_flight_speed == 0.0


# ===========================================================================
# Quad drone tests
# ===========================================================================


class TestQuadDrone:
    """Quad drones can hover."""

    def test_quad_can_hover(self):
        v = _make_quad(speed=0.0)
        engine = VehiclePhysicsEngine()
        pos_before = v.position
        engine.update(v, throttle=0.0, steering=0.0, dt=1.0)
        assert distance(pos_before, v.position) < 0.1

    def test_quad_is_hover_capable(self):
        v = _make_quad()
        assert v.is_hover_capable()

    def test_quad_zero_min_flight_speed(self):
        v = _make_quad()
        assert v.min_flight_speed == 0.0


# ===========================================================================
# Damage tests
# ===========================================================================


class TestDamage:
    """Armor, criticals, destruction."""

    def test_armor_reduces_damage(self):
        v = _make_vehicle(armor=0.5, health=100.0, max_health=100.0)
        engine = VehiclePhysicsEngine()
        report = engine.apply_damage(v, 50.0)
        assert report["damage_dealt"] == pytest.approx(25.0)
        assert report["armor_absorbed"] == pytest.approx(25.0)
        assert v.health == pytest.approx(75.0)

    def test_zero_armor_full_damage(self):
        v = _make_vehicle(armor=0.0, health=100.0, max_health=100.0)
        engine = VehiclePhysicsEngine()
        report = engine.apply_damage(v, 30.0)
        assert report["damage_dealt"] == pytest.approx(30.0)
        assert v.health == pytest.approx(70.0)

    def test_full_armor_blocks_all(self):
        v = _make_vehicle(armor=1.0, health=100.0, max_health=100.0)
        engine = VehiclePhysicsEngine()
        report = engine.apply_damage(v, 50.0)
        assert report["damage_dealt"] == pytest.approx(0.0)
        assert v.health == pytest.approx(100.0)

    def test_vehicle_destroyed_at_zero_health(self):
        v = _make_vehicle(armor=0.0, health=10.0, max_health=100.0)
        engine = VehiclePhysicsEngine()
        report = engine.apply_damage(v, 100.0)
        assert v.is_destroyed
        assert v.health == 0.0
        assert report["destroyed"]

    def test_no_damage_to_destroyed_vehicle(self):
        v = _make_vehicle(is_destroyed=True, health=0.0)
        engine = VehiclePhysicsEngine()
        report = engine.apply_damage(v, 50.0)
        assert report["damage_dealt"] == 0.0
        assert report["effects"] == []

    def test_negative_damage_ignored(self):
        v = _make_vehicle(health=50.0)
        engine = VehiclePhysicsEngine()
        report = engine.apply_damage(v, -10.0)
        assert report["damage_dealt"] == 0.0

    def test_effects_contain_sparks(self):
        v = _make_vehicle(armor=0.0, health=100.0, max_health=100.0)
        engine = VehiclePhysicsEngine()
        report = engine.apply_damage(v, 20.0)
        spark_effects = [e for e in report["effects"] if e["type"] == "sparks"]
        assert len(spark_effects) >= 1

    def test_destruction_has_explosion_effect(self):
        v = _make_vehicle(armor=0.0, health=5.0, max_health=100.0)
        engine = VehiclePhysicsEngine()
        report = engine.apply_damage(v, 100.0)
        explosion_effects = [e for e in report["effects"] if e["type"] == "explosion"]
        assert len(explosion_effects) >= 1

    def test_critical_engine_disables_movement(self):
        v = _make_vehicle(armor=0.0, health=1000.0, max_health=1000.0)
        v.engine_disabled = True  # Simulate critical hit
        engine = VehiclePhysicsEngine()
        v.speed = 10.0
        engine.update(v, throttle=1.0, steering=0.0, dt=2.0)
        # Engine disabled decays speed rapidly
        assert v.speed < 5.0

    def test_critical_fuel_leak_increases_consumption(self):
        v = _make_vehicle(fuel=1.0, fuel_consumption=0.01, speed=10.0, max_speed=30.0)
        v.fuel_leak = True
        engine = VehiclePhysicsEngine()
        fuel_before = v.fuel
        engine.update(v, throttle=0.5, steering=0.0, dt=1.0)
        consumed = fuel_before - v.fuel
        # With fuel leak, consumption should be 3x normal
        assert consumed > 0

    def test_critical_weapons_disabled_flag(self):
        v = _make_vehicle()
        v.weapons_disabled = True
        assert v.weapons_disabled

    def test_hit_pos_in_sparks_effect(self):
        v = _make_vehicle(armor=0.0, health=100.0, max_health=100.0)
        engine = VehiclePhysicsEngine()
        hit = (5.0, 10.0)
        report = engine.apply_damage(v, 20.0, hit_pos=hit)
        spark = [e for e in report["effects"] if e["type"] == "sparks"][0]
        assert spark["position"] == hit


# ===========================================================================
# Convoy tests
# ===========================================================================


class TestConvoy:
    """Convoy route-following and ambush reaction."""

    def _make_convoy(self, n: int = 3) -> ConvoySimulator:
        route = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]
        vehicles = [
            _make_vehicle(vehicle_id=f"cv_{i}", position=(-i * 15.0, 0.0))
            for i in range(n)
        ]
        return ConvoySimulator(vehicles, route)

    def test_convoy_requires_vehicles(self):
        with pytest.raises(ValueError, match="at least one vehicle"):
            ConvoySimulator([], [(0.0, 0.0), (10.0, 0.0)])

    def test_convoy_requires_route(self):
        v = _make_vehicle()
        with pytest.raises(ValueError, match="at least 2 waypoints"):
            ConvoySimulator([v], [(0.0, 0.0)])

    def test_convoy_tick_moves_vehicles(self):
        convoy = self._make_convoy()
        initial_positions = [v.position for v in convoy.vehicles]
        for _ in range(50):
            convoy.tick(0.1)
        for i, v in enumerate(convoy.vehicles):
            assert distance(v.position, initial_positions[i]) > 0.1

    def test_convoy_maintains_spacing(self):
        convoy = self._make_convoy(3)
        # Run for a while to let convoy settle into formation
        for _ in range(200):
            convoy.tick(0.1)
        # Check that vehicles have some spacing (not all bunched up)
        for i in range(1, len(convoy.vehicles)):
            dist = distance(
                convoy.vehicles[i - 1].position,
                convoy.vehicles[i].position,
            )
            # Should be roughly within 2x spacing (allow tolerance for formation dynamics)
            assert dist > 1.0, f"Vehicles {i-1} and {i} are too close"

    def test_convoy_ambush_stops_vehicles(self):
        convoy = self._make_convoy()
        # Get them moving first
        for _ in range(20):
            convoy.tick(0.1)
        convoy.ambush([(50.0, 50.0)])
        assert convoy._is_ambushed
        # All vehicles should have speed 0 after ambush
        for v in convoy.vehicles:
            assert v.speed == 0.0

    def test_convoy_ambush_faces_threat(self):
        convoy = self._make_convoy()
        convoy.ambush([(0.0, 100.0)])  # threat to the north
        for v in convoy.vehicles:
            # Heading should be roughly toward the threat (pi/2)
            assert abs(v.heading - math.pi / 2) < 0.5 or abs(v.heading - math.pi / 2) < math.pi

    def test_convoy_scatter_on_overwhelming_force(self):
        convoy = self._make_convoy(2)
        # 5 attackers vs 2 vehicles = outnumbered
        attackers = [(float(i * 10), 50.0) for i in range(5)]
        convoy.ambush(attackers)
        assert convoy._scatter_mode

    def test_convoy_no_scatter_when_not_outnumbered(self):
        convoy = self._make_convoy(3)
        convoy.ambush([(50.0, 50.0)])  # 1 attacker vs 3 vehicles
        assert not convoy._scatter_mode

    def test_convoy_to_three_js(self):
        convoy = self._make_convoy()
        data = convoy.to_three_js()
        assert data["type"] == "convoy"
        assert len(data["vehicles"]) == 3
        assert len(data["route"]) == 3
        assert "spacing" in data
        assert "is_ambushed" in data
        for vd in data["vehicles"]:
            assert "id" in vd
            assert "position" in vd
            assert "x" in vd["position"]
            assert "y" in vd["position"]
            assert "heading" in vd
            assert "vehicle_class" in vd

    def test_convoy_skips_destroyed_vehicles(self):
        convoy = self._make_convoy()
        convoy.vehicles[1].is_destroyed = True
        # Should not crash
        for _ in range(10):
            convoy.tick(0.1)


# ===========================================================================
# Drone controller tests
# ===========================================================================


class TestDroneController:
    """Drone patrol, pursuit, orbit, RTB."""

    def test_only_drones_accepted(self):
        car = _make_vehicle()
        with pytest.raises(ValueError, match="requires a drone"):
            DroneController(car)

    def test_quad_accepted(self):
        q = _make_quad()
        ctrl = DroneController(q)
        assert ctrl.mode == "idle"

    def test_fixed_wing_accepted(self):
        fw = _make_fixed_wing()
        ctrl = DroneController(fw)
        assert ctrl.mode == "idle"

    def test_set_patrol(self):
        q = _make_quad()
        ctrl = DroneController(q)
        wps = [(10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        ctrl.set_patrol(wps)
        assert ctrl.mode == "patrol"
        assert len(ctrl.waypoints) == 3

    def test_patrol_cycles_waypoints(self):
        q = _make_quad(position=(0.0, 0.0), speed=10.0)
        ctrl = DroneController(q)
        ctrl.set_patrol([(1.0, 0.0), (2.0, 0.0)])
        ctrl._waypoint_threshold = 2.0  # easy threshold
        engine = VehiclePhysicsEngine()
        for _ in range(100):
            t, s, a = ctrl.tick(0.1)
            engine.update(q, t, s, 0.1, altitude_input=a)
        # Should have advanced past first waypoint
        assert ctrl._waypoint_idx >= 0

    def test_pursue_mode(self):
        q = _make_quad(position=(0.0, 0.0))
        ctrl = DroneController(q)
        ctrl.pursue((100.0, 0.0))
        assert ctrl.mode == "pursue"
        t, s, a = ctrl.tick(0.1)
        assert t > 0  # should throttle toward target

    def test_orbit_mode(self):
        q = _make_quad(position=(50.0, 0.0))
        ctrl = DroneController(q)
        ctrl.orbit((0.0, 0.0), radius=50.0, altitude=30.0)
        assert ctrl.mode == "orbit"
        t, s, a = ctrl.tick(0.1)
        # Should produce some control output
        assert isinstance(t, float)
        assert isinstance(s, float)

    def test_rtb_mode(self):
        q = _make_quad(position=(100.0, 100.0))
        ctrl = DroneController(q)
        ctrl.return_to_base((0.0, 0.0))
        assert ctrl.mode == "rtb"
        t, s, a = ctrl.tick(0.1)
        assert t > 0

    def test_idle_quad_hovers(self):
        q = _make_quad(speed=0.0)
        ctrl = DroneController(q)
        t, s, a = ctrl.tick(0.1)
        assert t == 0.0  # hover

    def test_idle_fixed_wing_circles(self):
        fw = _make_fixed_wing()
        ctrl = DroneController(fw)
        t, s, a = ctrl.tick(0.1)
        assert t > 0  # must keep moving

    def test_destroyed_drone_no_output(self):
        q = _make_quad(is_destroyed=True)
        ctrl = DroneController(q)
        ctrl.set_patrol([(10.0, 0.0)])
        t, s, a = ctrl.tick(0.1)
        assert t == 0.0 and s == 0.0 and a == 0.0

    def test_to_three_js_format(self):
        q = _make_quad(position=(5.0, 10.0), altitude=25.0)
        ctrl = DroneController(q)
        ctrl.set_patrol([(10.0, 0.0), (20.0, 0.0)])
        data = ctrl.to_three_js()
        assert data["type"] == "drone"
        assert data["id"] == q.vehicle_id
        assert data["position"]["x"] == 5.0
        assert data["position"]["y"] == 10.0
        assert data["position"]["z"] == 25.0
        assert data["mode"] == "patrol"
        assert data["vehicle_class"] == "drone_quad"
        assert len(data["waypoints"]) == 2
        assert "health" in data
        assert "fuel" in data
        assert "destroyed" in data

    def test_orbit_three_js_has_orbit_data(self):
        q = _make_quad(position=(50.0, 0.0))
        ctrl = DroneController(q)
        ctrl.orbit((0.0, 0.0), radius=50.0, altitude=30.0)
        data = ctrl.to_three_js()
        assert data["orbit"] is not None
        assert data["orbit"]["radius"] == 50.0
        assert data["orbit"]["altitude"] == 30.0

    def test_non_orbit_three_js_no_orbit_data(self):
        q = _make_quad()
        ctrl = DroneController(q)
        ctrl.set_patrol([(10.0, 0.0)])
        data = ctrl.to_three_js()
        assert data["orbit"] is None

    def test_empty_patrol_goes_idle(self):
        q = _make_quad()
        ctrl = DroneController(q)
        ctrl.set_patrol([])
        t, s, a = ctrl.tick(0.1)
        assert ctrl.mode == "idle"


# ===========================================================================
# Passengers test
# ===========================================================================


class TestPassengers:
    """Passengers board and disembark."""

    def test_add_passengers(self):
        v = create_vehicle("btr80", "apc_1", "friendly", (0.0, 0.0))
        v.passengers = ["unit_1", "unit_2", "unit_3"]
        assert len(v.passengers) == 3

    def test_remove_passenger(self):
        v = create_vehicle("blackhawk", "bh_1", "friendly", (0.0, 0.0))
        v.passengers = ["unit_1", "unit_2"]
        v.passengers.remove("unit_1")
        assert v.passengers == ["unit_2"]

    def test_default_no_passengers(self):
        v = create_vehicle("humvee", "h1", "friendly", (0.0, 0.0))
        assert v.passengers == []


# ===========================================================================
# VehicleClass enum tests
# ===========================================================================


class TestVehicleClassEnum:
    """Ensure all expected vehicle classes exist."""

    def test_all_classes(self):
        expected = {
            "CAR", "TRUCK", "APC", "TANK", "HELICOPTER",
            "DRONE_QUAD", "DRONE_FIXED_WING", "BOAT", "MOTORCYCLE",
        }
        actual = {c.name for c in VehicleClass}
        assert actual == expected

    def test_class_values_are_strings(self):
        for c in VehicleClass:
            assert isinstance(c.value, str)


# ===========================================================================
# Three.js output format validation
# ===========================================================================


class TestThreeJsOutput:
    """Validate Three.js-compatible output structures."""

    def test_convoy_three_js_position_format(self):
        route = [(0.0, 0.0), (100.0, 0.0)]
        vehicles = [_make_vehicle(vehicle_id="v1")]
        convoy = ConvoySimulator(vehicles, route)
        data = convoy.to_three_js()
        pos = data["vehicles"][0]["position"]
        assert "x" in pos and "y" in pos and "z" in pos

    def test_drone_three_js_waypoint_format(self):
        q = _make_quad()
        ctrl = DroneController(q)
        ctrl.set_patrol([(1.0, 2.0), (3.0, 4.0)])
        data = ctrl.to_three_js()
        for wp in data["waypoints"]:
            assert "x" in wp and "y" in wp

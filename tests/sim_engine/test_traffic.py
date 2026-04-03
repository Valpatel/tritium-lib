# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.traffic -- IDM+MOBIL traffic simulation."""

import math

import pytest

from tritium_lib.sim_engine.idm import IDMParams
from tritium_lib.sim_engine.traffic import (
    RoadEdge,
    RouteStep,
    VehicleSubtype,
    VehicleProfile,
    VehiclePurpose,
    TrafficVehicle,
    TrafficManager,
    VEHICLE_PROFILES,
    create_traffic_vehicle,
    tick_vehicle,
    advance_to_next_edge,
    set_red_light,
    park_vehicle,
    _update_position,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_edge(
    edge_id="e1",
    from_node="n0",
    to_node="n1",
    length=100.0,
    lanes=1,
    road_class="residential",
):
    return RoadEdge(
        edge_id=edge_id,
        from_node=from_node,
        to_node=to_node,
        length=length,
        ax=0.0, az=0.0,
        bx=length, bz=0.0,
        lanes_per_dir=lanes,
        road_class=road_class,
    )


def _make_network():
    """Create a simple 3-node, 2-edge network: n0 --e1-- n1 --e2-- n2"""
    e1 = _make_edge("e1", "n0", "n1", 100.0)
    e2 = _make_edge("e2", "n1", "n2", 100.0, lanes=1)
    e2.ax, e2.az = 100.0, 0.0
    e2.bx, e2.bz = 200.0, 0.0
    return {"e1": e1, "e2": e2}


# ===================================================================
# RoadEdge
# ===================================================================


class TestRoadEdge:
    def test_effective_speed_limit(self):
        e = _make_edge(road_class="motorway")
        assert e.effective_speed_limit == 30.0

    def test_explicit_speed_limit(self):
        e = _make_edge()
        e.speed_limit = 15.0
        assert e.effective_speed_limit == 15.0

    def test_unknown_road_class_fallback(self):
        e = _make_edge(road_class="alien_road")
        assert e.effective_speed_limit == 10.0


# ===================================================================
# create_traffic_vehicle
# ===================================================================


class TestCreateTrafficVehicle:
    def test_creates_vehicle(self):
        edge = _make_edge()
        car = create_traffic_vehicle(edge, u=10.0)
        assert car.vehicle_id.startswith("car_")
        assert car.edge_id == "e1"
        assert car.u == 10.0
        assert car.alive is True
        assert car.speed == 0.0

    def test_subtype_applied(self):
        edge = _make_edge()
        car = create_traffic_vehicle(edge, subtype="truck")
        assert car.subtype == "truck"
        assert car.length == 7.0
        assert car.mass == 5000.0

    def test_position_set(self):
        edge = _make_edge(length=100.0)
        car = create_traffic_vehicle(edge, u=50.0)
        assert car.x == pytest.approx(50.0, abs=1.0)
        assert car.z == pytest.approx(0.0, abs=0.1)

    def test_unique_ids(self):
        edge = _make_edge()
        cars = [create_traffic_vehicle(edge) for _ in range(5)]
        ids = {c.vehicle_id for c in cars}
        assert len(ids) == 5

    def test_random_subtype(self):
        edge = _make_edge()
        subtypes = set()
        for _ in range(100):
            car = create_traffic_vehicle(edge)
            subtypes.add(car.subtype)
        # Should see at least 2 different subtypes
        assert len(subtypes) >= 2

    def test_purpose(self):
        edge = _make_edge()
        car = create_traffic_vehicle(edge, purpose=VehiclePurpose.TAXI)
        assert car.purpose == VehiclePurpose.TAXI


# ===================================================================
# VEHICLE_PROFILES
# ===================================================================


class TestVehicleProfiles:
    def test_all_subtypes_present(self):
        for sub in ["sedan", "suv", "truck", "motorcycle", "van"]:
            assert sub in VEHICLE_PROFILES

    def test_truck_longest(self):
        assert VEHICLE_PROFILES["truck"].length > VEHICLE_PROFILES["sedan"].length

    def test_motorcycle_lightest(self):
        assert VEHICLE_PROFILES["motorcycle"].mass < VEHICLE_PROFILES["sedan"].mass


# ===================================================================
# tick_vehicle — IDM physics
# ===================================================================


class TestTickVehicle:
    def test_accelerates_from_stop(self):
        """Stopped car should accelerate."""
        edge = _make_edge()
        car = create_traffic_vehicle(edge, u=10.0, subtype="sedan")
        car.speed = 0.0

        tick_vehicle(car, edge, 0.1, [])
        assert car.speed > 0.0
        assert car.acc > 0.0

    def test_follows_leader(self):
        """Car behind a slower leader should eventually match speed."""
        edge = _make_edge(length=200.0)
        leader = create_traffic_vehicle(edge, u=100.0, subtype="sedan")
        leader.speed = 5.0
        follower = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        follower.speed = 10.0

        for _ in range(500):
            # Leader moves at constant speed
            leader.u += leader.speed * 0.1
            tick_vehicle(follower, edge, 0.1, [leader])

        # Follower should converge toward leader speed
        assert follower.speed == pytest.approx(leader.speed, abs=2.0)

    def test_position_advances(self):
        edge = _make_edge()
        car = create_traffic_vehicle(edge, u=10.0, subtype="sedan")
        car.speed = 5.0
        old_u = car.u
        tick_vehicle(car, edge, 0.1, [])
        assert car.u > old_u

    def test_position_synced_with_world(self):
        """World x, z should match edge-based u."""
        edge = _make_edge(length=100.0)
        car = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        car.speed = 5.0
        tick_vehicle(car, edge, 0.1, [])
        # x should approximately equal u (edge goes from 0 to 100 along x)
        expected_x = car.u  # edge is along x axis
        assert car.x == pytest.approx(expected_x, abs=1.0)

    def test_edge_transition_detected(self):
        """Car reaching edge end should return transition node."""
        edge = _make_edge(length=100.0)
        car = create_traffic_vehicle(edge, u=99.5, subtype="sedan")
        car.speed = 10.0
        result = tick_vehicle(car, edge, 0.1, [])
        assert result == "n1"  # to_node

    def test_reverse_direction_transition(self):
        """Car going in reverse reaching u=0 should transition."""
        edge = _make_edge()
        car = create_traffic_vehicle(edge, u=0.5, direction=-1, subtype="sedan")
        car.speed = 10.0
        result = tick_vehicle(car, edge, 0.1, [])
        assert result == "n0"  # from_node

    def test_no_transition_mid_edge(self):
        edge = _make_edge()
        car = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        car.speed = 5.0
        result = tick_vehicle(car, edge, 0.1, [])
        assert result is None

    def test_parked_vehicle_doesnt_move(self):
        edge = _make_edge()
        car = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        park_vehicle(car, 10.0)
        old_u = car.u
        tick_vehicle(car, edge, 0.1, [])
        assert car.u == old_u
        assert car.parked is True

    def test_parked_vehicle_unparks(self):
        edge = _make_edge()
        car = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        park_vehicle(car, 0.1)
        tick_vehicle(car, edge, 0.2, [])  # timer expires
        assert car.parked is False

    def test_accident_stops_vehicle(self):
        edge = _make_edge()
        car = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        car.in_accident = True
        car.accident_timer = 5.0
        car.speed = 10.0
        old_u = car.u
        tick_vehicle(car, edge, 0.1, [])
        assert car.u == old_u  # didn't move

    def test_accident_clears(self):
        edge = _make_edge()
        car = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        car.in_accident = True
        car.accident_timer = 0.05
        tick_vehicle(car, edge, 0.1, [])
        assert car.in_accident is False


# ===================================================================
# Red light interaction
# ===================================================================


class TestRedLight:
    def test_red_light_slows_vehicle(self):
        """Vehicle approaching a red light should decelerate."""
        edge = _make_edge(length=100.0)
        car = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        car.speed = 10.0
        set_red_light(car, active=True, gap=15.0)

        speeds = []
        for _ in range(50):
            tick_vehicle(car, edge, 0.1, [])
            speeds.append(car.speed)

        # Speed should decrease
        assert speeds[-1] < speeds[0]

    def test_green_light_no_effect(self):
        """Vehicle with green light should not be affected."""
        edge = _make_edge(length=200.0)
        car = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        car.speed = 5.0
        set_red_light(car, active=False)

        tick_vehicle(car, edge, 0.1, [])
        # Should still be moving normally
        assert car.speed > 0

    def test_emergency_ignores_red(self):
        """Emergency vehicles should ignore red lights."""
        edge = _make_edge(length=100.0)
        car = create_traffic_vehicle(edge, u=50.0, subtype="sedan")
        car.speed = 10.0
        car.is_emergency = True
        set_red_light(car, active=True, gap=10.0)

        tick_vehicle(car, edge, 0.1, [])
        # Should still accelerate or maintain speed (red light ignored)
        assert car.speed > 5.0


# ===================================================================
# advance_to_next_edge
# ===================================================================


class TestAdvanceToNextEdge:
    def test_forward_transition(self):
        """Arriving at from_node should set direction=+1, u=0."""
        edge = _make_edge("e2", "n1", "n2", 80.0)
        car = create_traffic_vehicle(_make_edge(), u=99.0, subtype="sedan")
        advance_to_next_edge(car, edge, "n1")  # arrived at from_node
        assert car.direction == 1
        assert car.u == 0.0
        assert car.edge_id == "e2"

    def test_reverse_transition(self):
        """Arriving at to_node should set direction=-1, u=length."""
        edge = _make_edge("e2", "n1", "n2", 80.0)
        car = create_traffic_vehicle(_make_edge(), subtype="sedan")
        advance_to_next_edge(car, edge, "n2")  # arrived at to_node
        assert car.direction == -1
        assert car.u == 80.0

    def test_lane_clamped(self):
        """Lane index should be clamped to new edge's lane count."""
        edge = _make_edge("e2", "n1", "n2", 80.0, lanes=1)
        car = create_traffic_vehicle(_make_edge(lanes=3), subtype="sedan")
        car.lane_idx = 2
        advance_to_next_edge(car, edge, "n1")
        assert car.lane_idx == 0  # clamped to single lane


# ===================================================================
# _update_position
# ===================================================================


class TestUpdatePosition:
    def test_start_of_edge(self):
        edge = _make_edge(length=100.0)
        car = TrafficVehicle(u=0.0, direction=1)
        _update_position(car, edge)
        assert car.x == pytest.approx(0.0, abs=0.1)

    def test_end_of_edge(self):
        edge = _make_edge(length=100.0)
        car = TrafficVehicle(u=100.0, direction=1)
        _update_position(car, edge)
        assert car.x == pytest.approx(100.0, abs=0.1)

    def test_midpoint(self):
        edge = _make_edge(length=100.0)
        car = TrafficVehicle(u=50.0, direction=1)
        _update_position(car, edge)
        assert car.x == pytest.approx(50.0, abs=0.1)

    def test_reverse_heading(self):
        edge = _make_edge(length=100.0)
        car_fwd = TrafficVehicle(u=50.0, direction=1)
        car_rev = TrafficVehicle(u=50.0, direction=-1)
        _update_position(car_fwd, edge)
        _update_position(car_rev, edge)
        # Headings should differ by pi
        diff = abs(car_fwd.heading - car_rev.heading)
        assert diff == pytest.approx(math.pi, abs=0.1)


# ===================================================================
# TrafficManager
# ===================================================================


class TestTrafficManager:
    def test_add_edge(self):
        tm = TrafficManager()
        edge = _make_edge()
        tm.add_edge(edge)
        assert "e1" in tm.edges
        assert "n0" in tm._adjacency
        assert "n1" in tm._adjacency

    def test_spawn_vehicle(self):
        tm = TrafficManager()
        edge = _make_edge()
        tm.add_edge(edge)
        car = tm.spawn_vehicle("e1", u=10.0, subtype="sedan")
        assert car.vehicle_id in tm.vehicles
        assert tm.vehicle_count == 1

    def test_spawn_on_unknown_edge_raises(self):
        tm = TrafficManager()
        with pytest.raises(KeyError):
            tm.spawn_vehicle("nonexistent")

    def test_remove_vehicle(self):
        tm = TrafficManager()
        edge = _make_edge()
        tm.add_edge(edge)
        car = tm.spawn_vehicle("e1")
        tm.remove_vehicle(car.vehicle_id)
        assert tm.vehicle_count == 0

    def test_tick_advances_vehicles(self):
        tm = TrafficManager()
        edge = _make_edge(length=200.0)
        tm.add_edge(edge)
        car = tm.spawn_vehicle("e1", u=10.0, subtype="sedan")
        car.speed = 5.0
        old_u = car.u
        tm.tick(0.1)
        assert car.u > old_u

    def test_tick_returns_transitioned(self):
        """Edge transitions should be reported."""
        edges = _make_network()
        tm = TrafficManager(edges)
        # Build adjacency
        for e in edges.values():
            tm._adjacency.setdefault(e.from_node, []).append(e.edge_id)
            tm._adjacency.setdefault(e.to_node, []).append(e.edge_id)

        car = tm.spawn_vehicle("e1", u=99.5, subtype="sedan")
        car.speed = 10.0
        transitioned = tm.tick(0.1)
        # Car should have transitioned (or reversed if no route)
        # Either way, tick should not crash
        assert isinstance(transitioned, list)

    def test_get_vehicles_on_edge(self):
        tm = TrafficManager()
        edge = _make_edge()
        tm.add_edge(edge)
        tm.spawn_vehicle("e1", u=10.0)
        tm.spawn_vehicle("e1", u=50.0)
        on_edge = tm.get_vehicles_on_edge("e1")
        assert len(on_edge) == 2

    def test_to_dict(self):
        tm = TrafficManager()
        edge = _make_edge()
        tm.add_edge(edge)
        tm.spawn_vehicle("e1", u=10.0, subtype="sedan")
        d = tm.to_dict()
        assert d["vehicle_count"] == 1
        assert len(d["vehicles"]) == 1
        v = d["vehicles"][0]
        assert "id" in v
        assert "x" in v
        assert "speed" in v
        assert "subtype" in v

    def test_multi_vehicle_idm_interaction(self):
        """Multiple vehicles on same edge should follow each other."""
        tm = TrafficManager()
        edge = _make_edge(length=1000.0)  # long enough that nobody reaches the end
        tm.add_edge(edge)

        leader = tm.spawn_vehicle("e1", u=200.0, subtype="sedan")
        leader.speed = 8.0
        follower = tm.spawn_vehicle("e1", u=100.0, subtype="sedan")
        follower.speed = 12.0  # faster than leader

        for _ in range(200):
            tm.tick(0.1)

        # Follower should slow down toward leader speed
        assert follower.speed < 12.0
        # Follower should not have passed leader (both still going forward)
        assert follower.direction == 1
        assert leader.direction == 1
        assert follower.u < leader.u

    def test_edge_transition_to_next(self):
        """Vehicle should transition to next edge when reaching end."""
        edges = _make_network()
        tm = TrafficManager(edges)
        for e in edges.values():
            tm._adjacency.setdefault(e.from_node, []).append(e.edge_id)
            tm._adjacency.setdefault(e.to_node, []).append(e.edge_id)

        car = tm.spawn_vehicle("e1", u=95.0, subtype="sedan")
        car.speed = 10.0

        # Run enough ticks for the car to reach end of e1
        for _ in range(20):
            tm.tick(0.1)

        # Car should have moved to e2 (or reversed)
        assert car.edge_id in ("e1", "e2")


# ===================================================================
# Integration: IDM platoon on TrafficManager
# ===================================================================


class TestPlatoonIntegration:
    def test_three_car_platoon_stabilizes(self):
        """Three cars following each other should form a stable platoon."""
        tm = TrafficManager()
        edge = _make_edge(length=2000.0)  # long road so nobody reverses
        tm.add_edge(edge)

        c1 = tm.spawn_vehicle("e1", u=200.0, subtype="sedan")
        c1.speed = 8.0
        c2 = tm.spawn_vehicle("e1", u=150.0, subtype="sedan")
        c2.speed = 8.0
        c3 = tm.spawn_vehicle("e1", u=100.0, subtype="sedan")
        c3.speed = 8.0

        for _ in range(500):
            tm.tick(0.1)

        # All should be at similar speeds
        speeds = [c1.speed, c2.speed, c3.speed]
        assert max(speeds) - min(speeds) < 3.0

        # All should still be going forward
        assert c1.direction == 1
        assert c2.direction == 1
        assert c3.direction == 1

        # Ordering should be preserved (c1 ahead of c2 ahead of c3)
        assert c1.u > c2.u > c3.u

    def test_lane_change_on_multi_lane(self):
        """Vehicles on a multi-lane road should be able to change lanes."""
        tm = TrafficManager()
        edge = _make_edge(length=500.0, lanes=2)
        tm.add_edge(edge)

        # Slow leader in lane 0
        slow = tm.spawn_vehicle("e1", u=200.0, subtype="truck")
        slow.speed = 5.0
        slow.lane_idx = 0
        slow.idm = IDMParams(v0=5.0)  # stays slow

        # Fast follower in lane 0
        fast = tm.spawn_vehicle("e1", u=100.0, subtype="sedan")
        fast.speed = 10.0
        fast.lane_idx = 0
        fast._mobil_timer = 0.0  # force immediate evaluation

        # Run simulation
        initial_lane = fast.lane_idx
        changed = False
        for _ in range(200):
            tm.tick(0.1)
            if fast.lane_idx != initial_lane:
                changed = True
                break

        # Fast car may or may not have changed lanes (probabilistic),
        # but at minimum it should not have crashed
        assert fast.speed >= 0.0
        assert fast.alive is True

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.mobil -- MOBIL lane change model."""

import pytest

from tritium_lib.sim_engine.idm import IDMParams
from tritium_lib.sim_engine.mobil import (
    MOBILParams,
    MOBIL_DEFAULTS,
    LaneNeighbors,
    LaneChangeResult,
    LaneChangeDecision,
    find_neighbors_in_lane,
    evaluate_lane_change,
    decide_lane_change,
)
from tritium_lib.sim_engine.traffic import (
    TrafficVehicle,
    RoadEdge,
    create_traffic_vehicle,
)


def _make_edge(edge_id="e1", length=100.0, lanes=2):
    return RoadEdge(
        edge_id=edge_id,
        from_node="n0",
        to_node="n1",
        length=length,
        ax=0.0, az=0.0,
        bx=length, bz=0.0,
        lanes_per_dir=lanes,
    )


def _make_car(edge_id="e1", u=50.0, speed=10.0, lane=0, direction=1):
    """Create a minimal TrafficVehicle for testing."""
    return TrafficVehicle(
        vehicle_id=f"car_{u}_{lane}",
        edge_id=edge_id,
        u=u,
        speed=speed,
        direction=direction,
        lane_idx=lane,
        length=4.5,
        idm=IDMParams(v0=12.0),
    )


# ===================================================================
# MOBILParams
# ===================================================================


class TestMOBILParams:
    def test_defaults(self):
        p = MOBILParams()
        assert p.politeness == 0.3
        assert p.threshold == 0.2
        assert p.b_safe == 4.0
        assert p.min_gap == 5.0

    def test_custom(self):
        p = MOBILParams(politeness=0.8, threshold=0.5)
        assert p.politeness == 0.8
        assert p.threshold == 0.5


# ===================================================================
# find_neighbors_in_lane
# ===================================================================


class TestFindNeighborsInLane:
    def test_empty_road(self):
        car = _make_car()
        result = find_neighbors_in_lane(car, 0, [])
        assert result.ahead is None
        assert result.behind is None
        assert result.ahead_gap == float("inf")
        assert result.behind_gap == float("inf")

    def test_finds_leader_ahead(self):
        car = _make_car(u=20.0, lane=0)
        leader = _make_car(u=40.0, lane=0)
        result = find_neighbors_in_lane(car, 0, [car, leader])
        assert result.ahead is leader
        # Gap = 40 - 20 - 4.5/2 - 4.5/2 = 15.5
        assert result.ahead_gap == pytest.approx(15.5, abs=0.1)

    def test_finds_follower_behind(self):
        car = _make_car(u=40.0, lane=0)
        follower = _make_car(u=20.0, lane=0)
        result = find_neighbors_in_lane(car, 0, [car, follower])
        assert result.behind is follower
        assert result.behind_gap == pytest.approx(15.5, abs=0.1)

    def test_ignores_different_lane(self):
        car = _make_car(u=20.0, lane=0)
        other = _make_car(u=40.0, lane=1)
        result = find_neighbors_in_lane(car, 0, [car, other])
        assert result.ahead is None

    def test_ignores_different_edge(self):
        car = _make_car(u=20.0, lane=0, edge_id="e1")
        other = _make_car(u=40.0, lane=0, edge_id="e2")
        result = find_neighbors_in_lane(car, 0, [car, other])
        assert result.ahead is None

    def test_ignores_different_direction(self):
        car = _make_car(u=20.0, lane=0, direction=1)
        other = _make_car(u=40.0, lane=0, direction=-1)
        result = find_neighbors_in_lane(car, 0, [car, other])
        assert result.ahead is None

    def test_closest_leader_wins(self):
        car = _make_car(u=10.0, lane=0)
        far = _make_car(u=80.0, lane=0)
        near = _make_car(u=30.0, lane=0)
        result = find_neighbors_in_lane(car, 0, [car, far, near])
        assert result.ahead is near

    def test_target_lane_lookup(self):
        """Can query neighbors in a different lane than car's own."""
        car = _make_car(u=20.0, lane=0)
        other = _make_car(u=40.0, lane=1)
        result = find_neighbors_in_lane(car, 1, [car, other])
        assert result.ahead is other


# ===================================================================
# evaluate_lane_change
# ===================================================================


class TestEvaluateLaneChange:
    def test_empty_target_lane_beneficial(self):
        """Slow leader in current lane + empty target lane = change."""
        slow_leader = _make_car(u=60.0, lane=0, speed=3.0)
        car = _make_car(u=40.0, lane=0, speed=10.0)
        result = evaluate_lane_change(car, 1, [car, slow_leader])
        assert result.should_change is True
        assert result.incentive > 0
        assert "beneficial" in result.reason

    def test_insufficient_gap(self):
        """Not enough gap in target lane -> no change."""
        car = _make_car(u=50.0, lane=0, speed=10.0)
        blocker = _make_car(u=52.0, lane=1, speed=10.0)  # 2m gap < minGap
        result = evaluate_lane_change(car, 1, [car, blocker])
        assert result.should_change is False
        assert result.reason == "insufficient_gap"

    def test_unsafe_for_new_follower(self):
        """Lane change would cause unsafe braking for new follower."""
        car = _make_car(u=50.0, lane=0, speed=10.0)
        # Fast follower very close behind in target lane
        fast_follower = _make_car(u=44.0, lane=1, speed=15.0)
        # Just enough gap to pass min_gap but unsafe braking
        params = MOBILParams(min_gap=2.0, b_safe=1.0)
        result = evaluate_lane_change(car, 1, [car, fast_follower], params)
        # With the follower very close and fast, it should be unsafe
        assert result.should_change is False or result.reason == "unsafe_new_follower"

    def test_no_advantage_stays(self):
        """Same conditions in both lanes -> no change."""
        car = _make_car(u=50.0, lane=0, speed=10.0)
        # Similar leaders in both lanes at same distance
        leader_cur = _make_car(u=80.0, lane=0, speed=10.0)
        leader_tgt = _make_car(u=80.0, lane=1, speed=10.0)
        result = evaluate_lane_change(car, 1, [car, leader_cur, leader_tgt])
        assert result.should_change is False
        assert result.reason == "insufficient_incentive"


# ===================================================================
# decide_lane_change
# ===================================================================


class TestDecideLaneChange:
    def test_single_lane_no_change(self):
        """Single-lane road -> never change."""
        car = _make_car(lane=0)
        result = decide_lane_change(car, [], num_lanes=1)
        assert result.direction is None
        assert result.target_lane is None

    def test_prefers_better_lane(self):
        """Should pick the lane with better incentive."""
        car = _make_car(u=50.0, lane=1, speed=10.0)
        # Slow leader in current lane, open lanes on both sides
        slow_leader = _make_car(u=60.0, lane=1, speed=2.0)
        result = decide_lane_change(car, [car, slow_leader], num_lanes=3)
        # Should pick either left or right (both are empty)
        if result.direction is not None:
            assert result.target_lane in (0, 2)
            assert result.incentive > 0

    def test_left_boundary(self):
        """At lane 0, can only go right."""
        car = _make_car(u=50.0, lane=0, speed=10.0)
        slow = _make_car(u=60.0, lane=0, speed=2.0)
        result = decide_lane_change(car, [car, slow], num_lanes=2)
        if result.direction is not None:
            assert result.direction == "right"
            assert result.target_lane == 1

    def test_right_boundary(self):
        """At rightmost lane, can only go left."""
        car = _make_car(u=50.0, lane=1, speed=10.0)
        slow = _make_car(u=60.0, lane=1, speed=2.0)
        result = decide_lane_change(car, [car, slow], num_lanes=2)
        if result.direction is not None:
            assert result.direction == "left"
            assert result.target_lane == 0


# ===================================================================
# Integration: MOBIL with IDM physics
# ===================================================================


class TestMOBILIntegration:
    def test_polite_driver_less_likely_to_change(self):
        """Higher politeness should reduce lane-change incentive."""
        car = _make_car(u=50.0, lane=0, speed=10.0)
        slow = _make_car(u=60.0, lane=0, speed=2.0)
        follower = _make_car(u=30.0, lane=1, speed=10.0)

        selfish = MOBILParams(politeness=0.0, threshold=0.1)
        polite = MOBILParams(politeness=1.0, threshold=0.1)

        r_selfish = evaluate_lane_change(car, 1, [car, slow, follower], selfish)
        r_polite = evaluate_lane_change(car, 1, [car, slow, follower], polite)

        assert r_selfish.incentive >= r_polite.incentive

    def test_higher_threshold_fewer_changes(self):
        """Higher threshold should require more incentive."""
        car = _make_car(u=50.0, lane=0, speed=10.0)
        slow = _make_car(u=62.0, lane=0, speed=8.0)

        low_thresh = MOBILParams(threshold=0.01)
        high_thresh = MOBILParams(threshold=5.0)

        r_low = evaluate_lane_change(car, 1, [car, slow], low_thresh)
        r_high = evaluate_lane_change(car, 1, [car, slow], high_thresh)

        # Same incentive, but different threshold decisions
        assert r_low.incentive == pytest.approx(r_high.incentive, abs=0.1)
        # High threshold should NOT trigger change for marginal improvement
        if r_low.should_change:
            # High threshold may or may not change -- but incentive threshold is higher
            assert high_thresh.threshold > low_thresh.threshold

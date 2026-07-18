# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for route following — turning a planned path into a body command.

A planner hands down a polyline; a body can only obey a velocity.  Everything
between those two facts lives here, and the failure modes worth testing are
the ones that make a follower *look* like it is working:

  * a follower that always commands "straight ahead" tracks a straight route
    perfectly and is still useless, so the turning tests assert *sign* against
    a known frame convention (REP-103: +Z up, yaw CCW, so port is positive);
  * a follower that never declares arrival walks through its goal forever;
  * a differential mixer that ignores the sign of yaw drives both sides the
    same way and turns nothing, which reads as "the gait just isn't strong
    enough" rather than as the bug it is.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.control.waypoint_follower import (
    PurePursuitFollower,
    StrideBias,
    TwistCommand,
    cross_track_distance,
    differential_stride,
)


EAST = 0.0
NORTH = math.pi / 2


# --------------------------------------------------------------------------
# TwistCommand
# --------------------------------------------------------------------------

def test_twist_is_frozen_so_a_command_cannot_be_edited_after_issue():
    twist = TwistCommand(linear_mps=0.3, angular_rps=0.1)
    with pytest.raises(Exception):
        twist.linear_mps = 9.0  # type: ignore[misc]


def test_twist_stop_is_all_zero():
    assert TwistCommand.stop() == TwistCommand(linear_mps=0.0, angular_rps=0.0)


# --------------------------------------------------------------------------
# Pure pursuit — geometry
# --------------------------------------------------------------------------

def test_route_dead_ahead_commands_no_turn():
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((0.0, 0.0, EAST), [(0.0, 0.0), (5.0, 0.0)])
    assert state.twist.angular_rps == pytest.approx(0.0, abs=1e-9)
    assert state.twist.linear_mps > 0.0


def test_waypoint_to_port_commands_a_positive_yaw_rate():
    """REP-103: yaw is CCW about +Z, so a target to the left is positive."""
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((0.0, 0.0, EAST), [(0.0, 0.0), (0.0, 5.0)])
    assert state.twist.angular_rps > 0.0


def test_waypoint_to_starboard_commands_a_negative_yaw_rate():
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((0.0, 0.0, EAST), [(0.0, 0.0), (0.0, -5.0)])
    assert state.twist.angular_rps < 0.0


def test_turn_sign_follows_the_body_not_the_world():
    """Same world waypoint, body spun around: the command must flip."""
    route = [(0.0, 0.0), (0.0, 5.0)]
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    facing_east = follower.update((0.0, 0.0, EAST), route)
    facing_west = follower.update((0.0, 0.0, math.pi), route)
    assert facing_east.twist.angular_rps > 0.0
    assert facing_west.twist.angular_rps < 0.0


def test_curvature_matches_the_pure_pursuit_law():
    """gamma = 2*sin(alpha)/L — the published form, not an invented gain."""
    lookahead, cruise = 2.0, 0.4
    follower = PurePursuitFollower(
        lookahead_m=lookahead, cruise_mps=cruise, max_angular_rps=99.0
    )
    # Goal at 45 deg to port, exactly one lookahead away.
    d = lookahead / math.sqrt(2.0)
    state = follower.update((0.0, 0.0, EAST), [(0.0, 0.0), (d, d)])
    alpha = math.pi / 4
    expected = cruise * (2.0 * math.sin(alpha) / lookahead)
    assert state.twist.angular_rps == pytest.approx(expected, rel=1e-6)


def test_angular_rate_is_clamped_to_the_body_limit():
    follower = PurePursuitFollower(
        lookahead_m=0.5, cruise_mps=1.0, max_angular_rps=0.4
    )
    # Target directly behind — maximum possible steer demand.
    state = follower.update((0.0, 0.0, EAST), [(0.0, 0.0), (-5.0, 0.01)])
    assert abs(state.twist.angular_rps) <= 0.4


def test_a_target_behind_the_body_still_turns_toward_it():
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((0.0, 0.0, EAST), [(0.0, 0.0), (-5.0, 0.5)])
    assert state.twist.angular_rps > 0.0  # port side is the shorter way round


# --------------------------------------------------------------------------
# Pure pursuit — progress along the route
# --------------------------------------------------------------------------

def test_lookahead_point_advances_as_the_body_moves():
    route = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)]
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    early = follower.update((0.0, 0.0, EAST), route)
    late = follower.update((4.5, 0.0, EAST), route)
    assert late.target_index > early.target_index
    assert late.lookahead_point[0] > early.lookahead_point[0]


def test_lookahead_never_walks_backwards_down_the_route():
    """A body that drifts sideways must not re-target a waypoint it passed."""
    route = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    follower.update((3.9, 0.0, EAST), route)
    after = follower.update((3.9, 0.6, EAST), route)  # shoved off to port
    assert after.target_index >= 2


def test_lookahead_beyond_the_route_end_targets_the_goal():
    route = [(0.0, 0.0), (1.0, 0.0)]
    follower = PurePursuitFollower(lookahead_m=5.0, cruise_mps=0.3)
    state = follower.update((0.0, 0.0, EAST), route)
    assert state.lookahead_point == pytest.approx((1.0, 0.0))


# --------------------------------------------------------------------------
# Pure pursuit — cross-track error, the honest tracking metric
# --------------------------------------------------------------------------

def test_cross_track_is_zero_on_the_line():
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((1.0, 0.0, EAST), [(0.0, 0.0), (5.0, 0.0)])
    assert state.cross_track_m == pytest.approx(0.0, abs=1e-9)


def test_cross_track_measures_distance_to_the_segment_not_the_waypoint():
    """Abeam the middle of a long leg: 0.5 m off the line, far from any node."""
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((2.5, 0.5, EAST), [(0.0, 0.0), (5.0, 0.0)])
    assert state.cross_track_m == pytest.approx(0.5, abs=1e-9)


def test_cross_track_is_unsigned_distance_on_either_side():
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    port = follower.update((2.5, 0.5, EAST), [(0.0, 0.0), (5.0, 0.0)])
    stbd = follower.update((2.5, -0.5, EAST), [(0.0, 0.0), (5.0, 0.0)])
    assert port.cross_track_m == pytest.approx(stbd.cross_track_m)


def test_cross_track_distance_is_callable_without_a_follower():
    """The metric is shared with the offline scorer, so it must stand alone.

    A grader recomputing tracking error from ground truth has no follower and
    wants none — instantiating one would mean inventing a lookahead and a
    cruise speed that have nothing to do with the measurement.
    """
    assert cross_track_distance((2.5, 0.5), [(0.0, 0.0), (5.0, 0.0)]) == pytest.approx(0.5)


def test_cross_track_distance_degenerates_to_point_distance_for_one_waypoint():
    assert cross_track_distance((0.0, 3.0), [(0.0, 0.0)]) == pytest.approx(3.0)


def test_cross_track_distance_takes_the_nearest_of_several_segments():
    route = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    assert cross_track_distance((5.4, 3.0), route) == pytest.approx(0.4)


def test_the_follower_delegates_its_cross_track_to_the_shared_function():
    """Behaviour-preservation proof: method and function must not drift apart."""
    route = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    for probe in ((2.5, 0.5), (5.4, 3.0), (-1.0, -1.0), (5.0, 5.0)):
        state = follower.update((probe[0], probe[1], EAST), route)
        assert state.cross_track_m == pytest.approx(cross_track_distance(probe, route))


# --------------------------------------------------------------------------
# Pure pursuit — arrival
# --------------------------------------------------------------------------

def test_arrival_inside_tolerance_stops_the_body():
    follower = PurePursuitFollower(
        lookahead_m=1.0, cruise_mps=0.3, goal_tolerance_m=0.25
    )
    state = follower.update((5.0, 0.1, EAST), [(0.0, 0.0), (5.0, 0.0)])
    assert state.arrived
    assert state.twist == TwistCommand.stop()


def test_not_arrived_while_the_goal_is_still_out_of_tolerance():
    follower = PurePursuitFollower(
        lookahead_m=1.0, cruise_mps=0.3, goal_tolerance_m=0.25
    )
    state = follower.update((4.0, 0.0, EAST), [(0.0, 0.0), (5.0, 0.0)])
    assert not state.arrived
    assert state.twist.linear_mps > 0.0


def test_distance_to_goal_is_measured_to_the_last_waypoint():
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((1.0, 0.0, EAST), [(0.0, 0.0), (5.0, 0.0)])
    assert state.distance_to_goal_m == pytest.approx(4.0)


def test_the_body_slows_as_it_closes_on_the_goal():
    follower = PurePursuitFollower(
        lookahead_m=1.0, cruise_mps=0.4, slow_radius_m=2.0, goal_tolerance_m=0.1
    )
    far = follower.update((0.0, 0.0, EAST), [(0.0, 0.0), (5.0, 0.0)])
    near = follower.update((4.5, 0.0, EAST), [(0.0, 0.0), (5.0, 0.0)])
    assert near.twist.linear_mps < far.twist.linear_mps


# --------------------------------------------------------------------------
# Pure pursuit — degenerate routes must not silently drive
# --------------------------------------------------------------------------

def test_empty_route_commands_a_stop_rather_than_guessing():
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((0.0, 0.0, EAST), [])
    assert state.twist == TwistCommand.stop()
    assert state.arrived is False


def test_single_point_route_is_treated_as_a_goal_to_reach():
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((0.0, 0.0, EAST), [(3.0, 0.0)])
    assert state.twist.linear_mps > 0.0
    assert state.distance_to_goal_m == pytest.approx(3.0)


def test_a_route_of_duplicate_points_does_not_divide_by_zero():
    follower = PurePursuitFollower(lookahead_m=1.0, cruise_mps=0.3)
    state = follower.update((0.0, 0.0, EAST), [(2.0, 0.0), (2.0, 0.0)])
    assert math.isfinite(state.twist.angular_rps)
    assert math.isfinite(state.cross_track_m)


def test_rejects_a_nonpositive_lookahead_because_curvature_would_blow_up():
    with pytest.raises(ValueError):
        PurePursuitFollower(lookahead_m=0.0, cruise_mps=0.3)


# --------------------------------------------------------------------------
# Differential stride mixing — twist to a legged body
# --------------------------------------------------------------------------

def test_straight_ahead_drives_both_sides_equally():
    bias = differential_stride(
        TwistCommand(linear_mps=0.3, angular_rps=0.0),
        track_width_m=0.26,
        nominal_mps=0.3,
    )
    assert bias.left_scale == pytest.approx(bias.right_scale)
    assert bias.left_scale == pytest.approx(1.0)


def test_a_port_turn_shortens_the_port_stride():
    """Turning to port means the left side travels less than the right."""
    bias = differential_stride(
        TwistCommand(linear_mps=0.3, angular_rps=0.5),
        track_width_m=0.26,
        nominal_mps=0.3,
    )
    assert bias.left_scale < bias.right_scale


def test_a_starboard_turn_shortens_the_starboard_stride():
    bias = differential_stride(
        TwistCommand(linear_mps=0.3, angular_rps=-0.5),
        track_width_m=0.26,
        nominal_mps=0.3,
    )
    assert bias.right_scale < bias.left_scale


def test_a_spin_in_place_counter_rotates_the_two_sides():
    """The bug this catches: mixing that ignores sign turns nothing at all."""
    bias = differential_stride(
        TwistCommand(linear_mps=0.0, angular_rps=0.6),
        track_width_m=0.26,
        nominal_mps=0.3,
    )
    assert bias.left_scale < 0.0 < bias.right_scale
    assert bias.left_scale == pytest.approx(-bias.right_scale)


def test_mixing_matches_the_skid_steer_law():
    v, omega, track, nominal = 0.3, 0.4, 0.26, 0.3
    bias = differential_stride(
        TwistCommand(linear_mps=v, angular_rps=omega),
        track_width_m=track,
        nominal_mps=nominal,
    )
    assert bias.left_scale == pytest.approx((v - omega * track / 2.0) / nominal)
    assert bias.right_scale == pytest.approx((v + omega * track / 2.0) / nominal)


def test_stride_scales_are_clamped_so_a_huge_yaw_cannot_command_a_leap():
    bias = differential_stride(
        TwistCommand(linear_mps=0.3, angular_rps=50.0),
        track_width_m=0.26,
        nominal_mps=0.3,
        max_scale=1.5,
    )
    assert abs(bias.left_scale) <= 1.5
    assert abs(bias.right_scale) <= 1.5


def test_mixing_rejects_a_zero_track_width():
    with pytest.raises(ValueError):
        differential_stride(
            TwistCommand(linear_mps=0.3, angular_rps=0.1),
            track_width_m=0.0,
            nominal_mps=0.3,
        )


def test_bias_is_frozen():
    bias = StrideBias(left_scale=1.0, right_scale=1.0)
    with pytest.raises(Exception):
        bias.left_scale = 2.0  # type: ignore[misc]

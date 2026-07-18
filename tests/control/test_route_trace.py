# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for grading a ground-truth pose trace against a planned route.

The module under test exists to stop a run from grading itself.  So these
tests are mostly about the ways a route score can be *flattered*: progress
accrued by walking sideways, progress accrued by walking the same leg twice,
a goal "reached" by driving through a wall, a tracking error measured against
the nearest waypoint instead of the nearest segment.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.control import route_trace
from tritium_lib.control.route_trace import RouteScore, score_route_trace
from tritium_lib.planning import scene_costmap
from tritium_lib.planning.scene_costmap import SceneObstacle


def _wall(x: float, y: float, half=(0.5, 0.5, 0.5), z: float = 0.3) -> SceneObstacle:
    """A body-height box centred at ``(x, y)``."""
    return SceneObstacle(prim_path=f"/World/wall_{x}_{y}", center=(x, y, z), half_extents=half)


STRAIGHT = [(0.0, 0.0), (10.0, 0.0)]


# --------------------------------------------------------------------------
# Degenerate traces
# --------------------------------------------------------------------------

def test_empty_trace_is_no_trace_not_a_failure_to_reach():
    score = score_route_trace([], STRAIGHT)
    assert score.verdict == "NO_TRACE"
    assert score.samples == 0
    assert score.reached_goal is False
    assert score.progress_ratio == 0.0


def test_single_sample_trace_is_no_trace():
    # One pose is a snapshot, not a walk — nothing can be said about tracking.
    score = score_route_trace([(0.0, 0.0)], STRAIGHT)
    assert score.verdict == "NO_TRACE"
    assert score.samples == 1


def test_empty_route_is_no_trace_even_with_a_full_pose_log():
    score = score_route_trace([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], [])
    assert score.verdict == "NO_TRACE"


def test_score_is_frozen_so_a_grade_cannot_be_edited_after_the_fact():
    score = score_route_trace([(0.0, 0.0), (10.0, 0.0)], STRAIGHT)
    assert isinstance(score, RouteScore)
    with pytest.raises(Exception):
        score.verdict = "REACHED"  # type: ignore[misc]


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------

def test_perfect_straight_line_follow_reaches_with_zero_error():
    positions = [(x * 0.5, 0.0) for x in range(21)]  # 0.0 .. 10.0
    score = score_route_trace(positions, STRAIGHT)
    assert score.verdict == "REACHED"
    assert score.reached_goal is True
    assert score.final_gap_m == pytest.approx(0.0)
    assert score.max_cross_track_m == pytest.approx(0.0)
    assert score.rms_cross_track_m == pytest.approx(0.0)
    assert score.progress_ratio == pytest.approx(1.0)
    assert score.samples == 21
    assert score.collided is False
    assert math.isinf(score.min_clearance_m)


def test_a_wobbly_but_arriving_run_still_reaches_and_reports_its_wobble():
    positions = [(x * 0.5, 0.2 if x % 2 else -0.2) for x in range(20)]
    positions.append((10.0, 0.0))
    score = score_route_trace(positions, STRAIGHT)
    assert score.verdict == "REACHED"
    assert score.max_cross_track_m == pytest.approx(0.2)
    assert 0.0 < score.rms_cross_track_m <= 0.2


# --------------------------------------------------------------------------
# SHORT
# --------------------------------------------------------------------------

def test_a_body_that_stops_halfway_is_short_not_reached():
    positions = [(x * 0.5, 0.0) for x in range(11)]  # 0.0 .. 5.0
    score = score_route_trace(positions, STRAIGHT)
    assert score.verdict == "SHORT"
    assert score.reached_goal is False
    assert score.final_gap_m == pytest.approx(5.0)
    assert score.progress_ratio == pytest.approx(0.5)


def test_goal_tolerance_decides_reached_versus_short():
    positions = [(0.0, 0.0), (9.6, 0.0)]
    # The final gap is 0.4 m, so the tolerance alone flips the verdict.
    assert score_route_trace(positions, STRAIGHT, goal_tolerance_m=0.5).final_gap_m == pytest.approx(0.4)
    assert score_route_trace(positions, STRAIGHT, goal_tolerance_m=0.5).verdict == "REACHED"
    assert score_route_trace(positions, STRAIGHT, goal_tolerance_m=0.45).verdict == "REACHED"
    assert score_route_trace(positions, STRAIGHT, goal_tolerance_m=0.3).verdict == "SHORT"
    assert score_route_trace(positions, STRAIGHT, goal_tolerance_m=0.3).progress_ratio == pytest.approx(0.96)


# --------------------------------------------------------------------------
# COLLIDED dominates
# --------------------------------------------------------------------------

def test_reaching_the_goal_through_an_obstacle_is_collided_not_reached():
    # This is the whole ranking rule: the failure outranks the success.
    positions = [(x * 0.5, 0.0) for x in range(21)]
    score = score_route_trace(positions, STRAIGHT, [_wall(5.0, 0.0)])
    assert score.verdict == "COLLIDED"
    assert score.collided is True
    assert score.reached_goal is True  # the fact is still reported, honestly
    assert score.min_clearance_m < 0.0


def test_stopping_short_inside_an_obstacle_is_also_collided():
    positions = [(x * 0.5, 0.0) for x in range(11)]
    score = score_route_trace(positions, STRAIGHT, [_wall(5.0, 0.0)])
    assert score.verdict == "COLLIDED"
    assert score.reached_goal is False


def test_min_clearance_is_the_closest_approach_over_all_samples_and_boxes():
    positions = [(0.0, 0.0), (5.0, 3.0), (10.0, 0.0)]
    obstacles = [_wall(5.0, 5.0), _wall(0.0, 9.0)]
    score = score_route_trace(positions, STRAIGHT, obstacles)
    # Sample (5, 3) sits 1.5 m below the box spanning y in [4.5, 5.5].
    assert score.min_clearance_m == pytest.approx(1.5)
    assert score.collided is False
    assert score.verdict == "REACHED"


def test_clearance_threshold_inflates_the_footprint():
    positions = [(0.0, 0.0), (5.0, 3.0), (10.0, 0.0)]
    obstacles = [_wall(5.0, 5.0)]
    assert score_route_trace(positions, STRAIGHT, obstacles, clearance_m=1.0).collided is False
    hit = score_route_trace(positions, STRAIGHT, obstacles, clearance_m=2.0)
    assert hit.collided is True
    assert hit.verdict == "COLLIDED"


def test_a_box_purely_overhead_is_not_a_collision_for_a_walking_body():
    # A gantry at z = 3 m is something a quadruped walks under.  Counting it
    # would make every indoor scene un-walkable (ceilings are boxes too).
    gantry = SceneObstacle(
        prim_path="/World/gantry",
        center=(5.0, 0.0, 3.0),
        half_extents=(0.5, 0.5, 0.5),
    )
    positions = [(x * 0.5, 0.0) for x in range(21)]
    score = score_route_trace(positions, STRAIGHT, [gantry])
    assert score.collided is False
    assert score.verdict == "REACHED"
    assert math.isinf(score.min_clearance_m)


def test_the_ground_slab_underfoot_is_not_a_collision():
    slab = SceneObstacle(
        prim_path="/World/ground",
        center=(5.0, 0.0, -0.5),
        half_extents=(50.0, 50.0, 0.5),
    )
    positions = [(x * 0.5, 0.0) for x in range(21)]
    assert score_route_trace(positions, STRAIGHT, [slab]).verdict == "REACHED"


def test_a_world_scale_ground_slab_inside_the_body_band_is_terrain_not_a_wall():
    # THE regression that makes a live run uniformly useless.  A real stage
    # returned a ground mesh with 1519 m half-extents spanning z=-24..+33 --
    # that box passes the body-band test, so without a footprint cap EVERY
    # sample sits inside it and EVERY run scores COLLIDED.  The planner already
    # rejects this geometry via DEFAULT_MAX_FOOTPRINT_M; the scorer must agree,
    # or a route is planned through terrain the scorer then fails you for.
    terrain = SceneObstacle(
        prim_path="/World/terrain",
        center=(0.0, 0.0, 4.5),
        half_extents=(1519.0, 1519.0, 28.5),
    )
    positions = [(x * 0.5, 0.0) for x in range(21)]
    score = score_route_trace(positions, STRAIGHT, [terrain])
    assert score.collided is False
    assert score.verdict == "REACHED"
    assert math.isinf(score.min_clearance_m)


def test_a_normal_sized_box_still_collides_after_the_cap_is_applied():
    # The cap must not become a blanket amnesty — a wall is still a wall.
    positions = [(x * 0.5, 0.0) for x in range(21)]
    assert score_route_trace(positions, STRAIGHT, [_wall(5.0, 0.0)]).verdict == "COLLIDED"


def test_terrain_is_excluded_from_min_clearance_not_merely_from_the_verdict():
    # If terrain were only suppressed at the verdict, min_clearance_m would
    # still read as deeply negative and any caller thresholding on that number
    # would draw the same wrong conclusion one layer down.
    terrain = SceneObstacle(
        prim_path="/World/terrain",
        center=(0.0, 0.0, 4.5),
        half_extents=(1519.0, 1519.0, 28.5),
    )
    positions = [(0.0, 0.0), (5.0, 3.0), (10.0, 0.0)]
    score = score_route_trace(positions, STRAIGHT, [terrain, _wall(5.0, 5.0)])
    assert score.min_clearance_m == pytest.approx(1.5)  # the wall, not the terrain


def test_the_cap_matches_the_planners_default():
    # Scorer and planner must draw the "world vs obstacle" line in the same
    # place, or a route is planned around boxes the scorer ignores.
    assert route_trace.DEFAULT_MAX_FOOTPRINT_M == scene_costmap.DEFAULT_MAX_FOOTPRINT_M


def test_a_box_exactly_at_the_cap_is_still_an_obstacle():
    # The planner rejects on `edge > cap`, so the boundary box is kept. Same here.
    edge = route_trace.DEFAULT_MAX_FOOTPRINT_M / 2.0
    box = SceneObstacle(
        prim_path="/World/big", center=(5.0, 0.0, 0.3), half_extents=(edge, edge, 0.5)
    )
    assert score_route_trace([(0.0, 0.0), (5.0, 0.0)], STRAIGHT, [box]).collided is True


def test_the_cap_is_measured_on_the_larger_footprint_edge():
    # A long thin wall -- short in x, world-scale in y -- is still world-scale.
    sliver = SceneObstacle(
        prim_path="/World/sliver", center=(5.0, 0.0, 0.3), half_extents=(0.5, 800.0, 0.5)
    )
    assert score_route_trace([(0.0, 0.0), (5.0, 0.0)], STRAIGHT, [sliver]).collided is False


def test_passing_none_opts_out_of_the_cap_and_restores_raw_box_scoring():
    # The opt-out has to be real: a caller that has already curated its
    # obstacle list must be able to say "grade every box I gave you".
    terrain = SceneObstacle(
        prim_path="/World/terrain",
        center=(0.0, 0.0, 4.5),
        half_extents=(1519.0, 1519.0, 28.5),
    )
    positions = [(x * 0.5, 0.0) for x in range(21)]
    score = score_route_trace(positions, STRAIGHT, [terrain], max_footprint_m=None)
    assert score.verdict == "COLLIDED"
    assert score.min_clearance_m < 0.0


def test_infinity_opts_out_of_the_cap_too():
    terrain = SceneObstacle(
        prim_path="/World/terrain",
        center=(0.0, 0.0, 4.5),
        half_extents=(1519.0, 1519.0, 28.5),
    )
    positions = [(x * 0.5, 0.0) for x in range(21)]
    assert score_route_trace(
        positions, STRAIGHT, [terrain], max_footprint_m=math.inf
    ).verdict == "COLLIDED"


def test_the_cap_can_be_tightened_below_the_default():
    positions = [(0.0, 0.0), (5.0, 0.0)]
    wall = _wall(5.0, 0.0)  # a 1.0 m box
    assert score_route_trace(positions, STRAIGHT, [wall], max_footprint_m=0.5).collided is False
    assert score_route_trace(positions, STRAIGHT, [wall], max_footprint_m=2.0).collided is True


def test_a_negative_cap_is_rejected():
    with pytest.raises(ValueError):
        score_route_trace([(0.0, 0.0), (1.0, 0.0)], STRAIGHT, max_footprint_m=-1.0)


def test_body_band_is_configurable_so_a_taller_body_hits_the_gantry():
    gantry = SceneObstacle(
        prim_path="/World/gantry",
        center=(5.0, 0.0, 3.0),
        half_extents=(0.5, 0.5, 0.5),
    )
    positions = [(x * 0.5, 0.0) for x in range(21)]
    score = score_route_trace(positions, STRAIGHT, [gantry], body_band=(0.1, 4.0))
    assert score.verdict == "COLLIDED"


def test_a_yawed_box_is_measured_in_its_own_frame():
    # A 45-degree box: the corner reaches further along +x than the half-extent.
    box = SceneObstacle(
        prim_path="/World/diamond",
        center=(5.0, 0.0, 0.3),
        half_extents=(1.0, 1.0, 0.5),
        yaw_deg=45.0,
    )
    # (5 + 1.3, 0) is outside an axis-aligned 1.0 box but inside the rotated
    # one, whose corner lies at x = 5 + sqrt(2) ~= 6.414.
    score = score_route_trace([(0.0, 0.0), (6.3, 0.0)], STRAIGHT, [box])
    assert score.min_clearance_m < 0.0


# --------------------------------------------------------------------------
# progress_ratio must not be gameable
# --------------------------------------------------------------------------

def test_walking_past_the_goal_and_back_cannot_exceed_full_progress():
    positions = [(x * 0.5, 0.0) for x in range(30)]  # runs out to 14.5 m
    positions += [(14.5 - x * 0.5, 0.0) for x in range(10)]
    score = score_route_trace(positions, STRAIGHT)
    assert score.progress_ratio == pytest.approx(1.0)
    assert score.progress_ratio <= 1.0


def test_oscillating_along_the_route_does_not_accumulate_progress():
    # Back and forth over the first 2 m, forty times.  A path-length-based
    # metric would call this an 80 m journey and score 1.0.
    positions = []
    for _ in range(40):
        positions += [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 0.0)]
    score = score_route_trace(positions, STRAIGHT)
    assert score.progress_ratio == pytest.approx(0.2)
    assert score.verdict == "SHORT"


def test_moving_only_sideways_off_route_accrues_no_progress():
    positions = [(0.0, y * 0.5) for y in range(21)]  # straight off to port
    score = score_route_trace(positions, STRAIGHT)
    assert score.progress_ratio == pytest.approx(0.0)
    assert score.verdict == "SHORT"
    assert score.max_cross_track_m == pytest.approx(10.0)


def test_progress_is_monotone_so_a_body_shoved_backwards_keeps_its_ground():
    forward = [(x * 0.5, 0.0) for x in range(15)]  # out to 7.0 m
    shoved = [(7.0 - x * 0.5, 0.0) for x in range(8)]  # back to 3.5 m
    score = score_route_trace(forward + shoved, STRAIGHT)
    assert score.progress_ratio == pytest.approx(0.7)
    assert score.final_gap_m == pytest.approx(6.5)


def test_progress_counts_arclength_along_a_multi_leg_route():
    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]  # 20 m total
    positions = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 5.0)]
    score = score_route_trace(positions, route)
    assert score.progress_ratio == pytest.approx(0.75)


# --------------------------------------------------------------------------
# Cross-track is measured against the SEGMENT, not the nearest waypoint
# --------------------------------------------------------------------------

def test_cross_track_uses_the_segment_not_the_nearest_waypoint():
    # Mid-leg of a 10 m straight, 0.5 m off the line.  Distance to the nearest
    # *waypoint* is ~5.02 m; distance to the *segment* is 0.5 m.  A scorer that
    # measured to waypoints would report a ten-times-worse follower.
    score = score_route_trace([(0.0, 0.0), (5.0, 0.5), (10.0, 0.0)], STRAIGHT)
    assert score.max_cross_track_m == pytest.approx(0.5)


def test_cross_track_on_a_curved_route_uses_the_nearest_segment():
    # An arc sampled coarsely: chords cut inside the circle, so a body flying
    # the true arc is off the *polyline* by the sagitta, not by the distance to
    # a node.  The two differ materially here, which pins the definition.
    route = [
        (math.cos(math.radians(a)) * 5.0, math.sin(math.radians(a)) * 5.0)
        for a in range(0, 91, 30)
    ]
    # A point on the true arc, halfway between the first two route nodes.
    on_arc = (math.cos(math.radians(15)) * 5.0, math.sin(math.radians(15)) * 5.0)
    to_nearest_waypoint = min(math.dist(on_arc, w) for w in route)
    score = score_route_trace([route[0], on_arc, route[-1]], route)
    assert score.max_cross_track_m < to_nearest_waypoint * 0.5
    assert score.max_cross_track_m == pytest.approx(0.1704, abs=1e-3)


def test_a_single_waypoint_route_degenerates_to_point_distance():
    score = score_route_trace([(0.0, 0.0), (0.0, 3.0)], [(0.0, 0.0)])
    assert score.max_cross_track_m == pytest.approx(3.0)
    assert score.verdict == "SHORT"
    assert score.final_gap_m == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Input shapes
# --------------------------------------------------------------------------

def test_three_component_poses_are_accepted_and_z_is_ignored_for_tracking():
    flat = [(x * 0.5, 0.0) for x in range(21)]
    tall = [(x, y, 0.31) for x, y in flat]
    assert score_route_trace(tall, STRAIGHT).verdict == "REACHED"
    assert score_route_trace(tall, STRAIGHT).max_cross_track_m == pytest.approx(0.0)


def test_rms_is_the_quadratic_mean_not_the_average():
    positions = [(0.0, 0.0), (5.0, 3.0), (10.0, 0.0)]
    score = score_route_trace(positions, STRAIGHT)
    assert score.rms_cross_track_m == pytest.approx(math.sqrt(9.0 / 3.0))


def test_negative_goal_tolerance_is_rejected():
    with pytest.raises(ValueError):
        score_route_trace([(0.0, 0.0), (1.0, 0.0)], STRAIGHT, goal_tolerance_m=-1.0)


def test_an_inverted_body_band_is_rejected():
    with pytest.raises(ValueError):
        score_route_trace([(0.0, 0.0), (1.0, 0.0)], STRAIGHT, body_band=(0.6, 0.1))

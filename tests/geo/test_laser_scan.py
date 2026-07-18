# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.laser_scan — a LaserScan turned into map obstacles.

A LiDAR does not hand anyone obstacles.  It hands over a flat array of
distances and two angles, and every consumer that wants to draw something on a
map has to rebuild the same polar-to-Cartesian step, drop the same no-return
beams, and apply the same body pose.  Done twice, it is done with two different
sign conventions, and the obstacles land mirrored on one of the two maps.

The failures pinned here are the ones that survive a careless test suite:

* **Frame convention.**  REP-103: +X forward, +Y left, yaw positive
  counter-clockwise — the same convention ``tritium_lib.control`` steers in.  A
  clockwise-yaw implementation is correct at yaw 0 and mirrored everywhere
  else, so every pose test below uses a NON-zero yaw.
* **Rotate-then-translate.**  Adding the sensor's position before rotating is
  right only at the origin.  So the rotation tests place the sensor away from
  the origin.
* **No-return beams.**  A max-range reading means "nothing out there", not "a
  wall at exactly range_max".  Keeping those paints a phantom ring of obstacles
  around the robot at its own sensor range.
* **Segmentation adjacency.**  Range-gap segmentation walks the beams IN ORDER.
  Sorting or set-ifying the points first destroys the adjacency the method
  depends on and merges everything into one blob.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tritium_lib.geo.laser_scan import (
    Obstacle,
    cluster_centroids,
    cluster_extents,
    cluster_points,
    scan_obstacles,
    scan_to_body_points,
    scan_to_world_points,
)


# --- polar -> Cartesian in the sensor frame ---------------------------------


def test_single_beam_straight_ahead_lands_on_x_axis():
    """angle 0 is dead ahead: +X forward, zero lateral offset."""
    pts = scan_to_body_points([5.0], angle_min=0.0, angle_increment=0.0)
    assert pts.shape == (1, 2)
    assert pts[0] == pytest.approx((5.0, 0.0), abs=1e-9)


def test_single_beam_at_90_degrees_lands_to_port():
    """+90 deg is to the LEFT (+Y), not the right — REP-103, not compass."""
    pts = scan_to_body_points([2.0], angle_min=math.pi / 2, angle_increment=0.0)
    assert pts[0] == pytest.approx((0.0, 2.0), abs=1e-9)


def test_beam_at_minus_90_degrees_lands_to_starboard():
    pts = scan_to_body_points([2.0], angle_min=-math.pi / 2, angle_increment=0.0)
    assert pts[0] == pytest.approx((0.0, -2.0), abs=1e-9)


def test_angle_increment_walks_the_beams_counter_clockwise():
    """Beam i sits at angle_min + i * angle_increment, increasing CCW."""
    pts = scan_to_body_points(
        [1.0, 1.0, 1.0], angle_min=0.0, angle_increment=math.pi / 2
    )
    assert pts[0] == pytest.approx((1.0, 0.0), abs=1e-9)
    assert pts[1] == pytest.approx((0.0, 1.0), abs=1e-9)
    assert pts[2] == pytest.approx((-1.0, 0.0), abs=1e-9)


def test_diagonal_beam_uses_the_range_as_hypotenuse():
    r = 4.0
    pts = scan_to_body_points([r], angle_min=math.radians(45.0), angle_increment=0.0)
    leg = r / math.sqrt(2.0)
    assert pts[0] == pytest.approx((leg, leg), abs=1e-9)


# --- no-return beams --------------------------------------------------------


def test_beams_at_or_beyond_range_max_are_dropped():
    """A max-range return is 'nothing there', not a wall at range_max."""
    pts = scan_to_body_points(
        [1.0, 10.0, 12.0, 2.0],
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
    )
    assert pts.shape == (2, 2)
    assert np.linalg.norm(pts, axis=1) == pytest.approx([1.0, 2.0], abs=1e-9)


def test_beams_at_or_below_range_min_are_dropped():
    pts = scan_to_body_points(
        [0.0, 0.05, 3.0],
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
    )
    assert pts.shape == (1, 2)
    assert np.linalg.norm(pts[0]) == pytest.approx(3.0, abs=1e-9)


def test_nan_and_inf_beams_are_dropped():
    """ROS drivers publish NaN and +inf for 'no return'; both must vanish."""
    pts = scan_to_body_points(
        [float("nan"), 3.0, float("inf"), float("-inf"), 4.0],
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
    )
    assert pts.shape == (2, 2)
    assert np.isfinite(pts).all()


def test_empty_scan_returns_empty_two_column_array():
    pts = scan_to_body_points([], angle_min=0.0, angle_increment=0.1)
    assert pts.shape == (0, 2)


def test_all_no_return_scan_returns_empty_and_does_not_crash():
    pts = scan_to_body_points(
        [float("inf")] * 64, angle_min=-1.0, angle_increment=0.03, range_max=10.0
    )
    assert pts.shape == (0, 2)
    assert cluster_points(pts, gap_m=0.5) == []
    assert cluster_centroids(cluster_points(pts, gap_m=0.5)) == []


# --- body -> world ----------------------------------------------------------


def test_identity_pose_is_a_no_op():
    ranges = [1.0, 2.0, 3.0, 4.0]
    body = scan_to_body_points(ranges, angle_min=-0.5, angle_increment=0.25)
    world = scan_to_world_points(
        ranges,
        angle_min=-0.5,
        angle_increment=0.25,
        sensor_x=0.0,
        sensor_y=0.0,
        sensor_yaw_deg=0.0,
    )
    assert world == pytest.approx(body, abs=1e-12)


def test_pure_translation_shifts_every_point_equally():
    ranges = [1.0, 2.0, 3.0, 4.0]
    body = scan_to_body_points(ranges, angle_min=-0.5, angle_increment=0.25)
    world = scan_to_world_points(
        ranges,
        angle_min=-0.5,
        angle_increment=0.25,
        sensor_x=10.0,
        sensor_y=-4.0,
        sensor_yaw_deg=0.0,
    )
    assert (world - body) == pytest.approx(
        np.tile([10.0, -4.0], (len(body), 1)), abs=1e-12
    )


def test_yaw_90_rotates_forward_beam_to_world_north():
    """Yaw is CCW positive: a body yawed +90 deg has its nose along +Y."""
    world = scan_to_world_points(
        [5.0],
        angle_min=0.0,
        angle_increment=0.0,
        sensor_x=0.0,
        sensor_y=0.0,
        sensor_yaw_deg=90.0,
    )
    assert world[0] == pytest.approx((0.0, 5.0), abs=1e-9)


def test_negative_yaw_rotates_clockwise():
    world = scan_to_world_points(
        [5.0],
        angle_min=0.0,
        angle_increment=0.0,
        sensor_x=0.0,
        sensor_y=0.0,
        sensor_yaw_deg=-90.0,
    )
    assert world[0] == pytest.approx((0.0, -5.0), abs=1e-9)


def test_rotation_is_applied_before_translation():
    """The classic bug: translate-then-rotate is right only at the origin.

    Sensor at (10, 0) yawed +90 deg, one beam 5 m dead ahead.  Correct answer
    is (10, 5).  Translating first and then rotating the sum gives (-5, 10).
    """
    world = scan_to_world_points(
        [5.0],
        angle_min=0.0,
        angle_increment=0.0,
        sensor_x=10.0,
        sensor_y=0.0,
        sensor_yaw_deg=90.0,
    )
    assert world[0] == pytest.approx((10.0, 5.0), abs=1e-9)


def test_transform_preserves_pairwise_distances():
    """A rigid transform cannot change the shape of the scan."""
    ranges = [2.0, 2.5, 3.0, 3.5, 4.0]
    body = scan_to_body_points(ranges, angle_min=-0.4, angle_increment=0.2)
    world = scan_to_world_points(
        ranges,
        angle_min=-0.4,
        angle_increment=0.2,
        sensor_x=-7.5,
        sensor_y=3.25,
        sensor_yaw_deg=37.0,
    )
    body_d = np.linalg.norm(np.diff(body, axis=0), axis=1)
    world_d = np.linalg.norm(np.diff(world, axis=0), axis=1)
    assert world_d == pytest.approx(body_d, abs=1e-9)


# --- clustering -------------------------------------------------------------


def _two_wall_points() -> np.ndarray:
    """Two 1 m wall segments, 5 m apart, each sampled every 0.1 m."""
    left = np.array([[3.0, y] for y in np.arange(0.0, 1.01, 0.1)])
    right = np.array([[3.0, y] for y in np.arange(6.0, 7.01, 0.1)])
    return np.vstack([left, right])


def test_two_separated_walls_cluster_into_exactly_two():
    clusters = cluster_points(_two_wall_points(), gap_m=0.5)
    assert len(clusters) == 2
    assert [len(c) for c in clusters] == [11, 11]


def test_two_wall_centroids_land_where_expected():
    clusters = cluster_points(_two_wall_points(), gap_m=0.5)
    centroids = cluster_centroids(clusters)
    assert centroids[0] == pytest.approx((3.0, 0.5), abs=1e-6)
    assert centroids[1] == pytest.approx((3.0, 6.5), abs=1e-6)


def test_raising_gap_merges_the_two_walls():
    clusters = cluster_points(_two_wall_points(), gap_m=6.0)
    assert len(clusters) == 1
    assert len(clusters[0]) == 22


def test_lowering_gap_splits_a_single_wall_into_every_point():
    clusters = cluster_points(_two_wall_points(), gap_m=0.05)
    assert len(clusters) == 22
    assert all(len(c) == 1 for c in clusters)


def test_clustering_walks_beams_in_order_not_sorted():
    """Points revisiting an earlier region must not merge with it.

    Beam order is the only adjacency a LaserScan has.  An implementation that
    sorts or spatially groups the points would fold these three runs into two.
    """
    pts = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0], [0.2, 0.0], [0.3, 0.0]]
    )
    clusters = cluster_points(pts, gap_m=0.5)
    assert len(clusters) == 3


def test_single_point_scan_is_one_cluster():
    clusters = cluster_points(np.array([[1.0, 2.0]]), gap_m=0.5)
    assert len(clusters) == 1
    assert cluster_centroids(clusters)[0] == pytest.approx((1.0, 2.0))


def test_non_positive_gap_is_rejected():
    with pytest.raises(ValueError):
        cluster_points(_two_wall_points(), gap_m=0.0)


# --- extents ----------------------------------------------------------------


def test_extent_of_a_wall_is_half_its_length():
    """A 1 m segment fits a marker of radius 0.5 m about its centroid."""
    clusters = cluster_points(_two_wall_points(), gap_m=0.5)
    extents = cluster_extents(clusters)
    assert extents == pytest.approx([0.5, 0.5], abs=1e-6)


def test_extent_of_a_single_point_is_zero():
    assert cluster_extents([np.array([[4.0, 4.0]])]) == pytest.approx([0.0])


# --- the whole pipeline -----------------------------------------------------


def test_scan_obstacles_end_to_end_places_two_markers_in_world_frame():
    """One call from raw ranges to map-ready obstacles.

    A sensor at (2, 2) yawed +90 deg sees two beams 3 m out, 0.4 rad apart,
    plus a no-return between them.  Yawed +90, 'ahead' is world +Y.
    """
    obstacles = scan_obstacles(
        [3.0, float("inf"), 3.0],
        angle_min=-0.2,
        angle_increment=0.2,
        sensor_x=2.0,
        sensor_y=2.0,
        sensor_yaw_deg=90.0,
        gap_m=0.5,
    )
    assert len(obstacles) == 2
    assert all(isinstance(o, Obstacle) for o in obstacles)
    assert all(o.point_count == 1 for o in obstacles)
    assert all(o.radius_m == pytest.approx(0.0) for o in obstacles)
    # Both hits are 3 m from the sensor, which the transform must preserve.
    for o in obstacles:
        assert math.hypot(o.x - 2.0, o.y - 2.0) == pytest.approx(3.0, abs=1e-9)
    # Yaw +90 puts the boresight along +Y, so both land north of the sensor.
    assert all(o.y > 2.0 for o in obstacles)


def test_scan_obstacles_on_an_empty_scan_returns_nothing():
    assert scan_obstacles([], angle_min=0.0, angle_increment=0.1) == []


def test_obstacle_min_points_rejects_speckle():
    """A single stray return is noise, not an obstacle worth drawing."""
    # One lone return, then a wide no-return sector, then a 3-beam surface.
    # The dropped sector spans ~1.3 m at 3 m out, comfortably over the gap.
    pts_scan = [3.0] + [float("inf")] * 20 + [3.0, 3.0, 3.0]
    kept = scan_obstacles(
        pts_scan, angle_min=0.0, angle_increment=0.02, gap_m=0.5, min_points=2
    )
    assert len(kept) == 1
    assert kept[0].point_count == 3


class TestAzimuthSeamWrap:
    """A 360 deg sweep is a RING; the array holding it is a line.

    Found by driving the SC /api/sighting lidar route live: three known boxes
    produced FOUR obstacle tracks.  The extra one was a box sitting at dead
    ahead, whose returns straddle the 0/360 seam -- ``np.diff`` compares
    beam i to i+1 and so never compares the LAST beam to the FIRST, splitting
    one surface into two obstacles at opposite ends of the array.

    On the map that is a wall directly in front of the robot rendering as two
    contacts that a planner must then thread between.
    """

    def test_seam_straddling_surface_is_one_cluster(self):
        """The live three-box geometry that exposed this (Isaac, tick 22).

        The OTHER two boxes are what make this bite: they sit between the
        seam's two halves in the array, so the last kept point is far from
        the first and the run cannot be rescued by luck.  A single seam
        surface on an otherwise empty sweep does NOT reproduce the defect --
        its endpoints stay within gap_m of each other in array order.
        """
        n = 360
        ranges = [float("inf")] * n
        for b in (359, 0, 1):          # box A, straddling the seam
            ranges[b % n] = 3.597
        for b in (89, 90, 91):         # box B
            ranges[b] = 3.098
        for b in (188, 189, 190):      # box C
            ranges[b] = 3.649

        obstacles = scan_obstacles(
            ranges,
            angle_min=0.0,
            angle_increment=2.0 * math.pi / n,
            range_max=30.0,
            gap_m=0.5,
        )

        assert len(obstacles) == 3, (
            f"seam split three boxes into {len(obstacles)} obstacles"
        )
        assert sorted(o.point_count for o in obstacles) == [3, 3, 3]

    def test_partial_fov_does_not_wrap(self):
        """A 180 deg scanner's two ends are NOT neighbours -- never join them.

        This is why wrapping must be derived from the angular span rather than
        applied unconditionally: a forward-facing lidar seeing a wall hard left
        and another hard right would otherwise fuse them into one phantom
        obstacle straight through the robot.
        """
        n = 180
        ranges = [float("inf")] * n
        ranges[0] = 4.0      # hard right
        ranges[n - 1] = 4.0  # hard left, 180 deg away

        obstacles = scan_obstacles(
            ranges,
            angle_min=0.0,
            angle_increment=math.pi / n,  # spans pi, not 2pi
            range_max=30.0,
            gap_m=0.5,
        )

        assert len(obstacles) == 2

    def test_full_circle_with_a_real_gap_still_splits(self):
        """Wrapping must not merge ends that are genuinely far apart."""
        n = 360
        ranges = [float("inf")] * n
        ranges[0] = 4.0
        ranges[180] = 4.0

        obstacles = scan_obstacles(
            ranges,
            angle_min=0.0,
            angle_increment=2.0 * math.pi / n,
            range_max=30.0,
            gap_m=0.5,
        )

        assert len(obstacles) == 2

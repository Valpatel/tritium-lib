# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fire-control geometry: where the barrel points, and what the ray reaches.

Each test below is a failure mode that has shipped in somebody's weapon code,
not a restatement of the implementation.
"""

import math

import pytest

from tritium_lib.geo.camera_mount import CameraMount
from tritium_lib.geo.hitscan import (
    BoxTarget,
    Muzzle,
    SphereTarget,
    muzzle_from_body,
    ray_aabb,
    ray_sphere,
    resolve_shot,
)
from tritium_lib.geo.isaac_frame import LocalPose


# --- the barrel's own pose ------------------------------------------------


def test_muzzle_sits_a_barrel_ahead_of_the_body_nose():
    """Facing north, the muzzle is offset along +north, not along +east."""
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.4, heading_deg=0.0)
    mount = CameraMount(forward_m=0.3, up_m=0.1)
    muzzle = muzzle_from_body(body, mount, barrel_m=0.2)

    assert muzzle.north_m == pytest.approx(0.5)  # 0.3 mount + 0.2 barrel
    assert muzzle.east_m == pytest.approx(0.0)
    assert muzzle.up_m == pytest.approx(0.5)
    assert muzzle.heading_deg == pytest.approx(0.0)


def test_barrel_offset_rotates_with_the_body():
    """The bug this catches: adding metres of 'forward' onto north always.

    That is correct facing north and wrong at every other heading, which is
    exactly why it survives a test suite that only ever faces north.
    """
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.0, heading_deg=90.0)
    muzzle = muzzle_from_body(body, CameraMount(forward_m=0.3), barrel_m=0.2)

    assert muzzle.east_m == pytest.approx(0.5)
    assert muzzle.north_m == pytest.approx(0.0)


def test_turret_pan_steers_the_boresight_without_moving_the_body():
    """Pan is positive to the LEFT, matching CameraMount and REP-103."""
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.0, heading_deg=0.0)
    muzzle = muzzle_from_body(body, CameraMount(pan_deg=90.0), barrel_m=1.0)

    assert muzzle.heading_deg == pytest.approx(270.0)
    assert muzzle.east_m == pytest.approx(-1.0)  # barrel now points west
    assert muzzle.north_m == pytest.approx(0.0)


def test_tilt_lifts_the_muzzle_tip_and_the_boresight():
    body = LocalPose(east_m=0.0, north_m=0.0, up_m=0.0, heading_deg=0.0)
    muzzle = muzzle_from_body(body, CameraMount(tilt_deg=90.0), barrel_m=2.0)

    assert muzzle.up_m == pytest.approx(2.0)
    assert muzzle.elevation_deg == pytest.approx(90.0)
    assert muzzle.north_m == pytest.approx(0.0)


# --- ray vs sphere --------------------------------------------------------


def test_ray_sphere_returns_the_entry_distance_not_the_centre_distance():
    """A shot registers on the SURFACE; grading on centre range over-reports."""
    hit = ray_sphere((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 10.0, 0.0), 1.5)
    assert hit == pytest.approx(8.5)


def test_ray_sphere_rejects_a_target_behind_the_muzzle():
    """The quadratic has two roots and BOTH are negative here.

    Naive code takes the smaller root unconditionally and cheerfully shoots
    backwards through its own operator.
    """
    assert ray_sphere((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -10.0, 0.0), 1.5) is None


def test_ray_sphere_hits_from_inside_the_sphere():
    """Muzzle inside the target: one root behind, one ahead. Range 0, a hit."""
    hit = ray_sphere((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.0), 2.0)
    assert hit is not None
    assert hit == pytest.approx(0.0)


def test_ray_sphere_hits_point_blank_when_the_centre_is_behind_the_muzzle():
    """Muzzle inside the target AND past its centre -- still contact.

    Rejecting on a negative projection ALONE looks right and passes every
    test where the centre happens to be in front.  It fails exactly when a
    body walks into its target and fires: the muzzle is buried in the target,
    the centre is behind the barrel tip, and the weapon reports a clean miss.
    """
    hit = ray_sphere((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -0.5, 0.0), 2.0)
    assert hit == pytest.approx(0.0)


def test_ray_sphere_misses_when_the_aim_is_wider_than_the_target():
    assert ray_sphere((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (3.0, 10.0, 0.0), 1.5) is None


def test_ray_sphere_grazes_at_exactly_the_radius():
    """The tangent case: discriminant is zero, and it is a hit, not a miss."""
    hit = ray_sphere((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.5, 10.0, 0.0), 1.5)
    assert hit == pytest.approx(10.0)


def test_ray_sphere_rejects_a_degenerate_direction():
    with pytest.raises(ValueError):
        ray_sphere((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 10.0, 0.0), 1.0)


# --- ray vs box -----------------------------------------------------------


def test_ray_aabb_hits_the_near_face():
    hit = ray_aabb((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 4.0, -1.0), (1.0, 6.0, 1.0))
    assert hit == pytest.approx(4.0)


def test_ray_aabb_parallel_ray_outside_the_slab_is_a_miss():
    """The slab method's signature bug: a zero component gives 0/0 -> NaN.

    NaN compares false against everything, so an unguarded implementation
    silently reports a HIT for a ray running parallel to and completely
    outside the box.
    """
    assert ray_aabb((5.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 4.0, -1.0), (1.0, 6.0, 1.0)) is None


def test_ray_aabb_parallel_ray_exactly_on_the_slab_face_still_hits():
    """The true 0/0 case: origin ON the plane, direction parallel to it.

    Every other parallel ray divides a non-zero numerator by zero and gets a
    signed infinity, which compares correctly and gives the right answer by
    luck.  Only when the origin sits exactly on the boundary does the naive
    form produce NaN -- and NaN loses every comparison, so the clip silently
    does nothing and the box reports a hit or a miss at random.
    """
    hit = ray_aabb((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 4.0, -1.0), (1.0, 6.0, 1.0))
    assert hit == pytest.approx(4.0)


def test_ray_aabb_parallel_ray_inside_the_slab_still_hits():
    hit = ray_aabb((0.5, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 4.0, -1.0), (1.0, 6.0, 1.0))
    assert hit == pytest.approx(4.0)


def test_ray_aabb_rejects_a_box_behind_the_muzzle():
    assert ray_aabb((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, -6.0, -1.0), (1.0, -4.0, 1.0)) is None


# --- resolving a shot -----------------------------------------------------


def _muzzle(heading_deg=0.0, elevation_deg=0.0):
    return Muzzle(
        east_m=0.0, north_m=0.0, up_m=0.0,
        heading_deg=heading_deg, elevation_deg=elevation_deg,
    )


def test_shot_hits_a_target_dead_ahead():
    target = SphereTarget("dummy-a", east_m=0.0, north_m=10.0, up_m=0.0, radius_m=0.5)
    shot = resolve_shot(_muzzle(), [target], max_range_m=50.0)

    assert shot.hit is True
    assert shot.target_id == "dummy-a"
    assert shot.range_m == pytest.approx(9.5)
    assert shot.impact_east_m == pytest.approx(0.0)
    assert shot.impact_north_m == pytest.approx(9.5)


def test_heading_ninety_shoots_east_not_north():
    """Compass convention, shared with CameraMount: 0 = north, 90 = east."""
    east_target = SphereTarget("east", east_m=10.0, north_m=0.0, up_m=0.0, radius_m=0.5)
    assert resolve_shot(_muzzle(heading_deg=90.0), [east_target], 50.0).hit is True
    assert resolve_shot(_muzzle(heading_deg=0.0), [east_target], 50.0).hit is False


def test_the_nearest_target_stops_the_ray():
    """You cannot shoot through the thing in front. Occlusion is the point."""
    near = SphereTarget("near", east_m=0.0, north_m=5.0, up_m=0.0, radius_m=0.5)
    far = SphereTarget("far", east_m=0.0, north_m=20.0, up_m=0.0, radius_m=0.5)

    shot = resolve_shot(_muzzle(), [far, near], max_range_m=50.0)
    assert shot.target_id == "near"


def test_beyond_max_range_is_a_miss_even_though_the_ray_intersects():
    target = SphereTarget("far", east_m=0.0, north_m=100.0, up_m=0.0, radius_m=0.5)
    shot = resolve_shot(_muzzle(), [target], max_range_m=50.0)

    assert shot.hit is False
    assert shot.target_id is None


def test_a_miss_reports_closest_approach_for_the_near_miss_call():
    """Distance to the line, not to the muzzle -- 'how badly did I miss'."""
    target = SphereTarget("wide", east_m=3.0, north_m=10.0, up_m=0.0, radius_m=0.5)
    shot = resolve_shot(_muzzle(), [target], max_range_m=50.0)

    assert shot.hit is False
    assert shot.miss_distance_m == pytest.approx(2.5)  # 3.0 centre - 0.5 radius


def test_closest_approach_ignores_targets_behind_the_muzzle():
    """A target at your back is not a near miss; it was never in the cone."""
    behind = SphereTarget("behind", east_m=0.1, north_m=-10.0, up_m=0.0, radius_m=0.5)
    wide = SphereTarget("wide", east_m=5.0, north_m=10.0, up_m=0.0, radius_m=0.5)
    shot = resolve_shot(_muzzle(), [behind, wide], max_range_m=50.0)

    assert shot.hit is False
    assert shot.miss_distance_m == pytest.approx(4.5)


def test_elevation_is_needed_to_reach_a_raised_target():
    high = SphereTarget("high", east_m=0.0, north_m=10.0, up_m=10.0, radius_m=0.5)
    assert resolve_shot(_muzzle(elevation_deg=0.0), [high], 50.0).hit is False
    assert resolve_shot(_muzzle(elevation_deg=45.0), [high], 50.0).hit is True


def test_boxes_and_spheres_compete_in_the_same_ray():
    """Mixed geometry: the wall in front beats the dummy behind it."""
    wall = BoxTarget(
        "wall",
        min_east_m=-5.0, min_north_m=3.0, min_up_m=-1.0,
        max_east_m=5.0, max_north_m=3.5, max_up_m=3.0,
    )
    dummy = SphereTarget("dummy", east_m=0.0, north_m=10.0, up_m=0.0, radius_m=0.5)

    shot = resolve_shot(_muzzle(), [dummy, wall], max_range_m=50.0)
    assert shot.target_id == "wall"
    assert shot.range_m == pytest.approx(3.0)


def test_no_targets_is_a_clean_miss_not_a_crash():
    shot = resolve_shot(_muzzle(), [], max_range_m=50.0)
    assert shot.hit is False
    assert shot.target_id is None
    assert shot.miss_distance_m is None


def test_negative_range_is_rejected():
    with pytest.raises(ValueError):
        resolve_shot(_muzzle(), [], max_range_m=-1.0)


def test_a_shot_is_reproducible_from_its_own_record():
    """The trace must carry the muzzle, so a run can be re-graded offline."""
    target = SphereTarget("dummy-a", east_m=0.0, north_m=10.0, up_m=0.0, radius_m=0.5)
    shot = resolve_shot(_muzzle(), [target], max_range_m=50.0)

    record = shot.to_dict()
    assert record["hit"] is True
    assert record["target_id"] == "dummy-a"
    assert record["muzzle"]["heading_deg"] == pytest.approx(0.0)
    assert record["range_m"] == pytest.approx(9.5)


def test_muzzle_direction_is_a_unit_vector_at_every_attitude():
    for heading in (0.0, 37.0, 90.0, 180.0, 359.0):
        for elevation in (-60.0, 0.0, 12.0, 80.0):
            d = _muzzle(heading, elevation).direction()
            assert math.sqrt(sum(c * c for c in d)) == pytest.approx(1.0)

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.camera_mount — a camera carried BY a moving body.

Why this module exists: every camera Tritium has consumed so far is bolted to
a wall.  Its position is a fixed lat/lon and its heading never changes, so
"where is this camera looking" is a constant the operator types in once.  A
camera mounted on a robot is a different problem: the mount is fixed in the
BODY's frame, and the body rotates.  A camera on the dog's nose looks north
when the dog faces north and east when the dog faces east, and the lens itself
physically moves 0.3 m every time the dog turns in place.

The failure this pins down is the one everybody writes first: adding the mount
offset in WORLD axes instead of body axes.  That is invisible in every test
where the body faces north (the two frames coincide) and wrong at every other
heading.  So each offset test below is checked at north AND at a rotated
heading, where a world-frame implementation gives a different answer.

Conventions, matching the rest of ``tritium_lib.geo``:

* **Body frame**: +forward out the nose, +left out the port side, +up.  This is
  the ROS ``base_link`` convention (REP-103), so a real robot's URDF mount
  transform drops in without re-deriving signs.
* **Tritium local**: +X east, +Y north, +Z up, metres; heading 0 = north,
  increasing CLOCKWISE.
* **Pan** is measured in the body frame the way a turret operator means it:
  positive pan slews LEFT (counter-clockwise seen from above).  Compass
  heading increases clockwise, so ``camera_heading = body_heading - pan`` --
  a subtraction.  Getting this backwards still passes a pan=0 test and puts
  the FOV cone on the wrong side of the robot for every non-zero pan, so both
  signs are pinned.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.geo.camera_mount import CameraMount
from tritium_lib.geo.isaac_frame import LocalPose


def _body(east=0.0, north=0.0, up=0.0, heading=0.0) -> LocalPose:
    return LocalPose(east_m=east, north_m=north, up_m=up, heading_deg=heading)


# --------------------------------------------------------------------------
# Mount offsets rotate WITH the body
# --------------------------------------------------------------------------


def test_zero_mount_puts_camera_at_the_body_origin():
    """A camera with no offset is at the body's own position and heading."""
    cam = CameraMount().world_pose(_body(east=10.0, north=20.0, heading=137.0))
    assert cam.east_m == pytest.approx(0.0 + 10.0)
    assert cam.north_m == pytest.approx(20.0)
    assert cam.up_m == pytest.approx(0.0)
    assert cam.heading_deg == pytest.approx(137.0)


def test_forward_offset_points_north_when_the_body_faces_north():
    """Body facing north: nose-mounted camera is displaced +north."""
    cam = CameraMount(forward_m=2.0).world_pose(_body(heading=0.0))
    assert cam.east_m == pytest.approx(0.0, abs=1e-9)
    assert cam.north_m == pytest.approx(2.0)


def test_forward_offset_points_east_when_the_body_faces_east():
    """The same nose mount, body rotated 90 deg CW -- now displaced +east.

    A world-frame implementation would still report north here.  This is the
    test that separates a correct body-frame rotation from the naive version.
    """
    cam = CameraMount(forward_m=2.0).world_pose(_body(heading=90.0))
    assert cam.east_m == pytest.approx(2.0)
    assert cam.north_m == pytest.approx(0.0, abs=1e-9)


def test_left_offset_points_west_when_the_body_faces_north():
    """+left is port side: facing north, port is west (negative east)."""
    cam = CameraMount(left_m=1.5).world_pose(_body(heading=0.0))
    assert cam.east_m == pytest.approx(-1.5)
    assert cam.north_m == pytest.approx(0.0, abs=1e-9)


def test_left_offset_points_north_when_the_body_faces_east():
    """Facing east, port side is north."""
    cam = CameraMount(left_m=1.5).world_pose(_body(heading=90.0))
    assert cam.east_m == pytest.approx(0.0, abs=1e-9)
    assert cam.north_m == pytest.approx(1.5)


def test_up_offset_is_unaffected_by_heading():
    """Yaw does not tilt the body, so the up offset is heading-invariant."""
    for heading in (0.0, 45.0, 180.0, 271.0):
        cam = CameraMount(up_m=0.4).world_pose(_body(up=1.0, heading=heading))
        assert cam.up_m == pytest.approx(1.4)


def test_combined_offset_at_a_diagonal_heading():
    """Forward+left mount at heading 45 -- both components rotate together."""
    cam = CameraMount(forward_m=1.0, left_m=1.0).world_pose(_body(heading=45.0))
    # Forward at 45deg = (+.707, +.707); left is 45-90 = -45deg = (-.707, +.707).
    assert cam.east_m == pytest.approx(0.0, abs=1e-9)
    assert cam.north_m == pytest.approx(math.sqrt(2.0))


def test_offset_magnitude_is_preserved_under_rotation():
    """A rigid mount cannot change how far the lens is from the body origin."""
    mount = CameraMount(forward_m=0.3, left_m=-0.1)
    expected = math.hypot(0.3, -0.1)
    for heading in range(0, 360, 17):
        cam = mount.world_pose(_body(heading=float(heading)))
        assert math.hypot(cam.east_m, cam.north_m) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Pan: the sign trap
# --------------------------------------------------------------------------


def test_pan_slews_left_which_DECREASES_compass_heading():
    """Body facing north, camera panned 90 deg left, looks west (270)."""
    cam = CameraMount(pan_deg=90.0).world_pose(_body(heading=0.0))
    assert cam.heading_deg == pytest.approx(270.0)


def test_negative_pan_slews_right_which_INCREASES_compass_heading():
    """Panning starboard from north gives east."""
    cam = CameraMount(pan_deg=-90.0).world_pose(_body(heading=0.0))
    assert cam.heading_deg == pytest.approx(90.0)


def test_pan_composes_with_body_heading_and_wraps():
    cam = CameraMount(pan_deg=-45.0).world_pose(_body(heading=340.0))
    assert cam.heading_deg == pytest.approx(25.0)


def test_camera_heading_stays_in_range():
    for heading in range(0, 360, 23):
        for pan in (-350.0, -90.0, 0.0, 90.0, 350.0):
            cam = CameraMount(pan_deg=pan).world_pose(_body(heading=float(heading)))
            assert 0.0 <= cam.heading_deg < 360.0


# --------------------------------------------------------------------------
# Ground footprint -- what the operator actually sees drawn on the map
# --------------------------------------------------------------------------


def test_level_camera_footprint_reaches_the_range_limit():
    """A camera looking at the horizon never hits the ground, so the far edge
    is the sensor's own useful range, not a geometric intersection."""
    mount = CameraMount(up_m=1.0, tilt_deg=0.0, vfov_deg=0.0, range_m=25.0)
    poly = mount.ground_footprint(_body(heading=0.0))
    far = max(math.hypot(e, n) for e, n in poly)
    assert far == pytest.approx(25.0)


def test_downward_tilt_pulls_the_far_edge_in_to_the_ground_intersection():
    """Camera 2 m up tilted 45 deg down: the centre ray hits ground at 2 m.

    With a zero vertical FOV the whole footprint collapses to that ring, so
    this is a closed-form check of the intersection, not a regression blob.
    """
    mount = CameraMount(up_m=2.0, tilt_deg=-45.0, vfov_deg=0.0, range_m=100.0)
    poly = mount.ground_footprint(_body(heading=0.0))
    far = max(math.hypot(e, n) for e, n in poly)
    assert far == pytest.approx(2.0)


def test_steeper_tilt_brings_the_footprint_closer():
    """Monotonic: looking further down sees nearer ground."""
    def far_at(tilt):
        m = CameraMount(up_m=2.0, tilt_deg=tilt, vfov_deg=0.0, range_m=100.0)
        return max(math.hypot(e, n) for e, n in m.ground_footprint(_body()))

    assert far_at(-60.0) < far_at(-45.0) < far_at(-20.0)


def test_footprint_is_centred_on_the_camera_heading():
    """The sector's bearing spread brackets where the camera looks."""
    mount = CameraMount(up_m=2.0, tilt_deg=-30.0, hfov_deg=60.0, range_m=50.0)
    poly = mount.ground_footprint(_body(heading=90.0))
    bearings = [math.degrees(math.atan2(e, n)) % 360.0 for e, n in poly]
    assert min(bearings) == pytest.approx(60.0, abs=1e-6)
    assert max(bearings) == pytest.approx(120.0, abs=1e-6)


def test_footprint_translates_with_the_body():
    """Move the robot 100 m east; the footprint moves with it, unchanged."""
    mount = CameraMount(up_m=2.0, tilt_deg=-30.0, range_m=40.0)
    at_origin = mount.ground_footprint(_body())
    moved = mount.ground_footprint(_body(east=100.0, north=-50.0))
    for (e0, n0), (e1, n1) in zip(at_origin, moved):
        assert e1 == pytest.approx(e0 + 100.0)
        assert n1 == pytest.approx(n0 - 50.0)


def test_footprint_rotates_with_the_body():
    """The same mount on a body facing east is the north footprint rotated."""
    mount = CameraMount(up_m=2.0, tilt_deg=-30.0, hfov_deg=40.0, range_m=40.0)
    north = mount.ground_footprint(_body(heading=0.0))
    east = mount.ground_footprint(_body(heading=90.0))
    for (e0, n0), (e1, n1) in zip(north, east):
        # Heading +90 (clockwise) maps (east, north) -> (north, -east).
        assert e1 == pytest.approx(n0)
        assert n1 == pytest.approx(-e0)


def test_footprint_is_a_closed_ring_of_points():
    poly = CameraMount(up_m=2.0, tilt_deg=-30.0).ground_footprint(_body())
    assert len(poly) >= 4
    assert poly[0] == pytest.approx(poly[-1])


def test_upward_tilt_yields_no_ground_footprint():
    """A camera aimed at the sky sees no ground; say so rather than inventing
    a sector behind the robot, which is what a naive tan() would produce."""
    mount = CameraMount(up_m=2.0, tilt_deg=60.0, vfov_deg=20.0, range_m=50.0)
    assert mount.ground_footprint(_body()) == []


# --------------------------------------------------------------------------
# Isaac hand-off: the same mount as a USD-stage translation
# --------------------------------------------------------------------------


def test_stage_offset_maps_body_axes_onto_a_z_up_stage():
    """Isaac Z-up: body +forward is stage +X, +left is +Y, +up is +Z."""
    mount = CameraMount(forward_m=0.3, left_m=0.1, up_m=0.2)
    assert mount.stage_offset(up_axis="Z") == pytest.approx((0.3, 0.1, 0.2))


def test_stage_offset_maps_body_axes_onto_a_y_up_stage():
    """Y-up: ground plane is XZ and north is -Z, matching IsaacFrame."""
    mount = CameraMount(forward_m=0.3, left_m=0.1, up_m=0.2)
    assert mount.stage_offset(up_axis="Y") == pytest.approx((0.3, 0.2, -0.1))


def test_stage_offset_scales_to_stage_units():
    """A centimetre-authored stage wants centimetres, not metres."""
    mount = CameraMount(forward_m=0.3)
    x, _, _ = mount.stage_offset(up_axis="Z", meters_per_unit=0.01)
    assert x == pytest.approx(30.0)


def test_stage_offset_rejects_an_unknown_up_axis():
    with pytest.raises(ValueError, match="up_axis"):
        CameraMount().stage_offset(up_axis="X")


# --------------------------------------------------------------------------
# Validation -- refuse nonsense rather than drawing a nonsense cone
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hfov", [0.0, -10.0, 360.0])
def test_rejects_an_impossible_horizontal_fov(hfov):
    with pytest.raises(ValueError, match="hfov_deg"):
        CameraMount(hfov_deg=hfov)


def test_rejects_a_negative_range():
    with pytest.raises(ValueError, match="range_m"):
        CameraMount(range_m=-1.0)


def test_rejects_an_out_of_range_tilt():
    with pytest.raises(ValueError, match="tilt_deg"):
        CameraMount(tilt_deg=120.0)

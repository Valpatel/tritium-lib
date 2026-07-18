# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.isaac_frame — Isaac/USD stage <-> Tritium local ENU.

Why this module exists: a simulated body's pose is only useful to the operator
if the icon on the tactical map lands where the robot actually is in the
simulator.  That mapping was previously a prose comment inside the Isaac addon
(``isaac_quadruped_server.py`` "FRAME MAPPING"), applied by hand at two call
sites and never tested.  Prose does not catch a sign error.  These tests pin
the contract so any consumer -- the Isaac pose bridge, a ROS2 odom relay, a
future rover/aerial body -- converts identically.

The two conventions being bridged:

* **Tritium local**: +X east, +Y north, +Z up, metres, heading 0 = north and
  increasing CLOCKWISE (this is what ``tritium_lib.geo`` already assumes).
* **USD/Isaac stage**: author-defined up axis (Z-up or Y-up), author-defined
  ``metersPerUnit``, yaw measured COUNTER-CLOCKWISE from +X.

The headline identity is ``heading = 90 - yaw`` (mod 360) -- a reflection, not
a rotation, because the two frames disagree about handedness of the angle.  A
plain offset would pass a north test and fail every east/west test, so the
tests below check all four cardinals in both directions.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.geo.isaac_frame import IsaacFrame, quat_to_yaw_deg


# --------------------------------------------------------------------------
# Yaw <-> heading: the reflection, checked on all four cardinals
# --------------------------------------------------------------------------

# (isaac yaw CCW from +X/east, tritium heading CW from north)
CARDINALS = [
    (0.0, 90.0),      # facing +X = east      -> heading 90
    (90.0, 0.0),      # facing +Y = north     -> heading 0
    (180.0, 270.0),   # facing -X = west      -> heading 270
    (270.0, 180.0),   # facing -Y = south     -> heading 180
]


@pytest.mark.parametrize("yaw,heading", CARDINALS)
def test_yaw_to_heading_cardinals(yaw, heading):
    assert IsaacFrame().yaw_to_heading(yaw) == pytest.approx(heading)


@pytest.mark.parametrize("yaw,heading", CARDINALS)
def test_heading_to_yaw_cardinals(yaw, heading):
    assert IsaacFrame().heading_to_yaw(heading) == pytest.approx(yaw)


def test_yaw_heading_is_its_own_inverse():
    """heading = 90 - yaw is an involution; round-tripping must be exact."""
    frame = IsaacFrame()
    for yaw in range(-720, 721, 17):
        back = frame.heading_to_yaw(frame.yaw_to_heading(float(yaw)))
        assert back == pytest.approx(float(yaw) % 360.0, abs=1e-9)


def test_headings_are_normalised_to_0_360():
    frame = IsaacFrame()
    for yaw in (-450.0, -90.0, 450.0, 1000.0):
        h = frame.yaw_to_heading(yaw)
        assert 0.0 <= h < 360.0


def test_reflection_not_rotation():
    """Guards the classic bug: heading = yaw + 90 also maps north correctly."""
    frame = IsaacFrame()
    # A body turning CCW in Isaac (yaw increasing) must have DECREASING
    # Tritium heading, because Tritium heading is clockwise.
    assert frame.yaw_to_heading(10.0) < frame.yaw_to_heading(0.0) % 360.0 or \
        frame.yaw_to_heading(10.0) == pytest.approx(80.0)
    assert frame.yaw_to_heading(10.0) == pytest.approx(80.0)


# --------------------------------------------------------------------------
# Position: up-axis handling, unit scaling, stage origin offset
# --------------------------------------------------------------------------

def test_z_up_position_maps_one_to_one():
    frame = IsaacFrame(up_axis="Z")
    assert frame.stage_to_local((3.0, 4.0, 5.0)) == pytest.approx((3.0, 4.0, 5.0))


def test_y_up_stage_remaps_axes():
    """USD Y-up: +Y is up and the ground plane is XZ with north = -Z."""
    frame = IsaacFrame(up_axis="Y")
    east, north, up = frame.stage_to_local((3.0, 5.0, -4.0))
    assert (east, north, up) == pytest.approx((3.0, 4.0, 5.0))


def test_meters_per_unit_scales_position_not_heading():
    """A centimetre-authored stage still reports metres to the operator."""
    frame = IsaacFrame(meters_per_unit=0.01)
    assert frame.stage_to_local((300.0, 400.0, 500.0)) == pytest.approx((3.0, 4.0, 5.0))
    # Angles are unit-free.
    assert frame.yaw_to_heading(0.0) == pytest.approx(90.0)


def test_origin_offset_is_applied_in_stage_units_before_scaling():
    """The Tritium origin can sit anywhere in the stage (e.g. a city block)."""
    frame = IsaacFrame(meters_per_unit=0.01, origin_offset=(100.0, 200.0, 0.0))
    assert frame.stage_to_local((400.0, 600.0, 0.0)) == pytest.approx((3.0, 4.0, 0.0))


@pytest.mark.parametrize("up_axis", ["Z", "Y"])
@pytest.mark.parametrize("mpu", [1.0, 0.01])
def test_position_round_trip(up_axis, mpu):
    frame = IsaacFrame(up_axis=up_axis, meters_per_unit=mpu, origin_offset=(7.0, -3.0, 1.0))
    stage = (123.0, 45.0, -67.0)
    assert frame.local_to_stage(frame.stage_to_local(stage)) == pytest.approx(stage)


def test_rejects_unknown_up_axis():
    with pytest.raises(ValueError, match="up_axis"):
        IsaacFrame(up_axis="X")


def test_rejects_nonpositive_meters_per_unit():
    with pytest.raises(ValueError, match="meters_per_unit"):
        IsaacFrame(meters_per_unit=0.0)


# --------------------------------------------------------------------------
# Quaternions: what Isaac actually hands you off a prim
# --------------------------------------------------------------------------

def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    """Right-handed rotation of yaw_deg about +Z, as (w, x, y, z)."""
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


@pytest.mark.parametrize("yaw", [0.0, 45.0, 90.0, 180.0, 270.0, 359.0])
def test_quat_to_yaw_recovers_z_rotation(yaw):
    assert quat_to_yaw_deg(_yaw_quat(yaw)) % 360.0 == pytest.approx(yaw % 360.0, abs=1e-6)


@pytest.mark.parametrize("yaw,heading", CARDINALS)
def test_pose_to_local_end_to_end(yaw, heading):
    """The call the pose bridge actually makes: prim pose -> operator pose."""
    frame = IsaacFrame()
    pose = frame.pose_to_local((10.0, 20.0, 0.4), _yaw_quat(yaw))
    assert pose.east_m == pytest.approx(10.0)
    assert pose.north_m == pytest.approx(20.0)
    assert pose.up_m == pytest.approx(0.4)
    assert pose.heading_deg == pytest.approx(heading, abs=1e-6)


def test_pose_to_local_survives_unnormalised_quaternion():
    """Isaac hands back float32; a quat off unit length must not skew heading."""
    frame = IsaacFrame()
    w, x, y, z = _yaw_quat(90.0)
    fat = (w * 3.0, x * 3.0, y * 3.0, z * 3.0)
    assert frame.pose_to_local((0.0, 0.0, 0.0), fat).heading_deg == pytest.approx(0.0, abs=1e-6)


def test_pose_to_local_ignores_roll_and_pitch():
    """A quadruped's torso pitches and rolls while walking; heading must not wobble."""
    frame = IsaacFrame()
    # 30 deg roll about +X composed with 90 deg yaw about +Z.
    yaw_h, roll_h = math.radians(90.0) / 2.0, math.radians(30.0) / 2.0
    qw = math.cos(yaw_h) * math.cos(roll_h)
    qx = math.cos(yaw_h) * math.sin(roll_h)
    qy = math.sin(yaw_h) * math.sin(roll_h)
    qz = math.sin(yaw_h) * math.cos(roll_h)
    heading = frame.pose_to_local((0.0, 0.0, 0.0), (qw, qx, qy, qz)).heading_deg
    assert heading == pytest.approx(0.0, abs=1e-6)


def test_zero_quaternion_is_rejected_not_silently_north():
    """A dropped/zeroed pose must raise, not masquerade as a valid heading."""
    with pytest.raises(ValueError, match="quaternion"):
        quat_to_yaw_deg((0.0, 0.0, 0.0, 0.0))


# --------------------------------------------------------------------------
# The stage-metadata constructor -- how a live bridge builds the frame
# --------------------------------------------------------------------------

def test_from_stage_metadata_reads_isaac_health_payload():
    """Shape matches the MCP bridge /health result verbatim."""
    frame = IsaacFrame.from_stage_metadata({"up_axis": "Z", "meters_per_unit": 1.0})
    assert frame.up_axis == "Z"
    assert frame.meters_per_unit == pytest.approx(1.0)


def test_from_stage_metadata_defaults_are_isaac_defaults():
    frame = IsaacFrame.from_stage_metadata({})
    assert frame.up_axis == "Z"
    assert frame.meters_per_unit == pytest.approx(1.0)

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.body_attitude — has the body fallen over?

Written against a specific failure that a full run of honest-looking numbers
did not catch.  A Go2 was driven through a trot in a live Newton sim and the
score card read: displacement 1.27 m, `height_retained` 0.89, `collapsed`
False, verdict MOVED.  Every one of those is true.  The rendered frame at
t=4.5 s shows the robot lying **on its back** with its legs in the air.

The metric missed it because it watched height, and an inverted quadruped
occupies almost exactly the same height as a standing one — the body is a
similar distance off the floor either way.  Height cannot see rotation.  So a
tumbling robot that skitters along on its shoulders scores as a walking robot,
and displacement alone will happily certify a gait that does not work.

The fix is to measure the thing that actually changed: where the body's own up
axis points.  Upright is 0 deg from world up, on its side is ~90, on its back
is ~180 — a quantity no amount of sliding or bouncing can fake.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.geo.body_attitude import (
    is_upright,
    tilt_from_upright_deg,
)

# Quaternions are (w, x, y, z), matching tritium_lib.geo.isaac_frame.
LEVEL = (1.0, 0.0, 0.0, 0.0)
ROLL_90 = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)      # on its side
ROLL_180 = (0.0, 1.0, 0.0, 0.0)                            # on its back
PITCH_90 = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)      # nose down
YAW_90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))        # turned, still level


def test_level_body_has_zero_tilt() -> None:
    assert tilt_from_upright_deg(LEVEL) == pytest.approx(0.0, abs=1e-9)


def test_yaw_alone_is_not_tilt() -> None:
    """Turning to face a new heading must not read as falling over.

    This is the case that makes tilt the right metric rather than a generic
    "how far from the identity rotation" distance: a walking robot yaws
    constantly and that is not a failure.
    """
    assert tilt_from_upright_deg(YAW_90) == pytest.approx(0.0, abs=1e-9)


def test_body_on_its_side_reads_90_degrees() -> None:
    assert tilt_from_upright_deg(ROLL_90) == pytest.approx(90.0, abs=1e-6)


def test_body_on_its_back_reads_180_degrees() -> None:
    """The exact case the height metric certified as a healthy walk."""
    assert tilt_from_upright_deg(ROLL_180) == pytest.approx(180.0, abs=1e-6)


def test_pitched_nose_down_reads_90_degrees() -> None:
    assert tilt_from_upright_deg(PITCH_90) == pytest.approx(90.0, abs=1e-6)


def test_tilt_is_never_negative_and_never_exceeds_180() -> None:
    for quat in (LEVEL, ROLL_90, ROLL_180, PITCH_90, YAW_90):
        assert 0.0 <= tilt_from_upright_deg(quat) <= 180.0


def test_unnormalised_quaternion_is_still_measured() -> None:
    """Solver-read quaternions drift off unit length; that is not an error."""
    scaled = tuple(2.5 * c for c in ROLL_90)
    assert tilt_from_upright_deg(scaled) == pytest.approx(90.0, abs=1e-6)


def test_zero_quaternion_is_rejected() -> None:
    """A zero quat means the read failed — silently calling it upright hides that."""
    with pytest.raises(ValueError, match="zero"):
        tilt_from_upright_deg((0.0, 0.0, 0.0, 0.0))


def test_is_upright_accepts_a_level_body() -> None:
    assert is_upright(LEVEL) is True


def test_is_upright_rejects_the_body_on_its_back() -> None:
    assert is_upright(ROLL_180) is False


def test_is_upright_rejects_the_body_on_its_side() -> None:
    assert is_upright(ROLL_90) is False


def test_is_upright_tolerates_normal_gait_lean() -> None:
    """A trotting body pitches and rolls a few degrees every stride.

    The threshold has to sit above that or every real gait fails; the default
    45 deg is far above stride lean and far below any fall.
    """
    lean = math.radians(10.0)
    quat = (math.cos(lean / 2), math.sin(lean / 2), 0.0, 0.0)
    assert is_upright(quat) is True


def test_is_upright_threshold_is_tunable() -> None:
    lean = math.radians(30.0)
    quat = (math.cos(lean / 2), math.sin(lean / 2), 0.0, 0.0)
    assert is_upright(quat, max_tilt_deg=20.0) is False
    assert is_upright(quat, max_tilt_deg=40.0) is True

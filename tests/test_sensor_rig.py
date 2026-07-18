# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Sensor-rig bring-up -> Command Center registration.

The seam under test turns "these sensors came up healthy" into "this is what
the operator must be told", and — the part that actually earns its keep —
refuses to call a rig healthy when it isn't.
"""

from __future__ import annotations

import pytest

from tritium_lib.fleet.sensor_rig import (
    RegistrationCall,
    RigSensor,
    RigBringupReport,
    registration_plan,
    summarize_bringup,
)


def _cam(**kw):
    base = dict(role="camera", host="rtx4090", port=8100, ready=True)
    base.update(kw)
    return RigSensor(**base)


# --------------------------------------------------------------------------- #
# registration_plan: healthy sensors -> the calls that put them on the map
# --------------------------------------------------------------------------- #


def test_camera_becomes_an_isaac_rgb_feed_registration():
    (call,) = registration_plan([_cam()])
    assert isinstance(call, RegistrationCall)
    assert call.method == "POST"
    assert call.path == "/api/camera-feeds/isaac"
    assert call.payload["stream"] == "rgb"
    assert call.payload["host"] == "rtx4090"
    assert call.payload["port"] == 8100


def test_depth_and_stereo_map_to_their_own_streams_on_the_same_server():
    calls = registration_plan(
        [_cam(role="depth"), _cam(role="stereo_right")]
    )
    streams = [c.payload["stream"] for c in calls]
    assert streams == ["depth16", "right"]
    # Same physical server, distinct source ids -- otherwise the second
    # registration would collide with the first.  The ids are asserted by
    # VALUE, not just for distinctness: they must match the `isaac_{stream}`
    # convention SC's own route defaults to, or a rig-registered feed and a
    # hand-registered one become two feeds for one camera.
    ids = [c.payload["source_id"] for c in calls]
    assert ids == ["isaac_depth16", "isaac_right"]


def test_rgb_feed_id_matches_the_route_default():
    (call,) = registration_plan([_cam()])
    assert call.payload["source_id"] == "isaac_rgb"


def test_a_role_the_rig_does_not_know_gets_no_feed():
    """Unknown roles fail closed.

    A future sensor type must not be silently registered as a camera feed
    just because it came up healthy.
    """
    assert registration_plan([_cam(role="thermal_v2")]) == []
    assert registration_plan([_cam(role="body", port=18973)]) == []


def test_an_unready_sensor_produces_no_registration_at_all():
    """A feed tile that can never render is worse than no tile."""
    assert registration_plan([_cam(ready=False)]) == []


def test_lidar_gets_no_camera_feed_call():
    """LiDAR has no feed surface -- it streams sightings, not frames.

    Inventing a camera-feed registration for it would put a permanently
    black tile on the operator's wall.
    """
    calls = registration_plan([_cam(role="lidar", port=8110)])
    assert [c.path for c in calls] == []


def test_mount_is_passed_through_only_when_the_rig_knows_it():
    (bound,) = registration_plan([_cam(attach_to="robot_go2")])
    assert bound.payload["attach_to"] == "robot_go2"

    (unbound,) = registration_plan([_cam()])
    # Deliberately ABSENT, not null -- a null would overwrite whatever the
    # camera server advertises about its own mount.
    assert "attach_to" not in unbound.payload


def test_plan_is_deterministic_for_the_same_rig():
    rig = [_cam(role="depth"), _cam(), _cam(role="stereo_right")]
    assert registration_plan(rig) == registration_plan(rig)


# --------------------------------------------------------------------------- #
# summarize_bringup: the honesty gate
# --------------------------------------------------------------------------- #


def test_all_registered_is_ok():
    report = summarize_bringup(
        [("isaac_rgb", "registered"), ("isaac_depth16", "already_registered")]
    )
    assert isinstance(report, RigBringupReport)
    assert report.ok
    assert report.registered == 1
    assert report.already == 1
    assert report.failed == 0


def test_one_failure_sinks_the_whole_rig():
    report = summarize_bringup(
        [("isaac_rgb", "registered"), ("isaac_depth16", "failed")]
    )
    assert not report.ok
    assert report.failed == 1
    assert "isaac_depth16" in report.detail


def test_an_empty_rig_is_not_a_healthy_rig():
    """Zero sensors registered must never read as green.

    This is the failure mode the gate exists for: a bring-up where every
    server died still returns an empty result set, and an `all(...)` over
    an empty list is True.
    """
    assert not summarize_bringup([]).ok


def test_report_names_what_failed_so_the_operator_can_act():
    report = summarize_bringup(
        [("isaac_rgb", "failed"), ("isaac_depth16", "failed")]
    )
    assert not report.ok
    assert report.failed == 2
    assert "isaac_rgb" in report.detail and "isaac_depth16" in report.detail


def test_unknown_outcome_is_rejected_rather_than_silently_counted_ok():
    with pytest.raises(ValueError):
        summarize_bringup([("isaac_rgb", "probably_fine")])

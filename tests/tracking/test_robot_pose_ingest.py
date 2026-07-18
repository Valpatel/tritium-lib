"""Robot-pose ingest — an externally-driven body's pose reaching the map.

This is the last mile of Isaac capability 2 ("position/heading AGREES with
the tactical map").  The Isaac->Tritium frame conversion is proven live at
0.000 m / 0.000 deg error (``tritium_lib.geo.isaac_frame``), but the pose had
nowhere to land: every ``TargetTracker.update_from_*`` method models a
*sensor observing something else*, and none models *a body reporting where it
itself is*.

The distinction is not cosmetic.  A robot pose is self-reported and, when it
comes out of a simulator, it is GROUND TRUTH — the one position in the whole
system that is known exactly.  So this path:

  * takes heading directly (a sensor track has to infer heading from motion),
  * carries confidence 1.0 for ground truth and a lower value for a real
    robot's onboard estimate (odometry/GPS drifts),
  * does NOT run the velocity/teleport integrity gate on ground truth, since
    an operator repositioning a sim body is legitimate, not a spoof.

Copyright (c) Matthew Valancy / Valpatel Software LLC. AGPL-3.0.
"""

import time

import pytest

from tritium_lib.tracking.target_tracker import TargetTracker


@pytest.fixture
def tracker():
    return TargetTracker()


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_creates_target_from_pose(tracker):
    tid = tracker.update_from_robot_pose({
        "target_id": "isaac_go2_01",
        "name": "Go2",
        "position": {"x": 10.0, "y": -4.0},
        "heading": 90.0,
    })
    assert tid == "isaac_go2_01"
    t = tracker.get_target("isaac_go2_01")
    assert t is not None
    assert t.position == (10.0, -4.0)
    assert t.heading == 90.0
    assert t.source == "robot_pose"
    assert "robot_pose" in t.confirming_sources


def test_missing_target_id_is_rejected(tracker):
    assert tracker.update_from_robot_pose({"position": {"x": 1, "y": 2}}) is None
    assert len(tracker.get_all()) == 0


def test_position_accepts_sequence_form(tracker):
    """The bridge may emit ``position`` as [x, y] as well as {x, y}."""
    tracker.update_from_robot_pose({"target_id": "r1", "position": [3.0, 4.0]})
    assert tracker.get_target("r1").position == (3.0, 4.0)


def test_position_accepts_flat_xy_form(tracker):
    """The shape the Isaac pose bridge and robot MQTT telemetry already emit."""
    tracker.update_from_robot_pose({"target_id": "r1", "x": 8.0, "y": -2.0, "z": 0.4})
    assert tracker.get_target("r1").position == (8.0, -2.0)


def test_defaults_are_friendly_robot(tracker):
    tracker.update_from_robot_pose({"target_id": "r1", "position": {"x": 0, "y": 0}})
    t = tracker.get_target("r1")
    assert t.alliance == "friendly"
    assert t.asset_type == "robot"
    assert t.name == "r1"


def test_asset_type_and_alliance_honoured(tracker):
    tracker.update_from_robot_pose({
        "target_id": "q1",
        "position": {"x": 0, "y": 0},
        "asset_type": "quadruped",
        "alliance": "hostile",
    })
    t = tracker.get_target("q1")
    assert t.asset_type == "quadruped"
    assert t.alliance == "hostile"


# ---------------------------------------------------------------------------
# Ground truth vs onboard estimate — the confidence distinction
# ---------------------------------------------------------------------------

def test_ground_truth_is_full_confidence(tracker):
    tracker.update_from_robot_pose({
        "target_id": "sim1",
        "position": {"x": 1, "y": 1},
        "ground_truth": True,
    })
    t = tracker.get_target("sim1")
    assert t.position_source == "sim_truth"
    assert t.position_confidence == 1.0


def test_onboard_estimate_is_not_full_confidence(tracker):
    """A real robot's own odometry/GPS drifts — it must not claim truth."""
    tracker.update_from_robot_pose({
        "target_id": "real1",
        "position": {"x": 1, "y": 1},
    })
    t = tracker.get_target("real1")
    assert t.position_source == "onboard"
    assert 0.0 < t.position_confidence < 1.0


# ---------------------------------------------------------------------------
# Update path
# ---------------------------------------------------------------------------

def test_second_pose_updates_in_place(tracker):
    tracker.update_from_robot_pose({"target_id": "r1", "position": {"x": 0, "y": 0}, "heading": 0})
    tracker.update_from_robot_pose({"target_id": "r1", "position": {"x": 5, "y": 5}, "heading": 180})
    assert len(tracker.get_all()) == 1
    t = tracker.get_target("r1")
    assert t.position == (5.0, 5.0)
    assert t.heading == 180.0
    assert t.signal_count == 2


def test_heading_is_normalised_to_0_360(tracker):
    tracker.update_from_robot_pose({"target_id": "r1", "position": {"x": 0, "y": 0}, "heading": -90.0})
    assert tracker.get_target("r1").heading == 270.0
    tracker.update_from_robot_pose({"target_id": "r1", "position": {"x": 0, "y": 0}, "heading": 450.0})
    assert tracker.get_target("r1").heading == 90.0


def test_speed_and_battery_carried(tracker):
    tracker.update_from_robot_pose({
        "target_id": "r1",
        "position": {"x": 0, "y": 0},
        "speed": 1.4,
        "battery": 0.62,
    })
    t = tracker.get_target("r1")
    assert t.speed == pytest.approx(1.4)
    assert t.battery == pytest.approx(0.62)


def test_last_seen_advances(tracker):
    tracker.update_from_robot_pose({"target_id": "r1", "position": {"x": 0, "y": 0}})
    first = tracker.get_target("r1").last_seen
    time.sleep(0.01)
    tracker.update_from_robot_pose({"target_id": "r1", "position": {"x": 0, "y": 0}})
    assert tracker.get_target("r1").last_seen > first


# ---------------------------------------------------------------------------
# Integrity gate — deliberately NOT applied to ground truth
# ---------------------------------------------------------------------------

def test_ground_truth_teleport_is_not_flagged_suspicious(tracker):
    """Repositioning a sim body is legitimate; a spoof flag here is a lie."""
    tracker.update_from_robot_pose({
        "target_id": "sim1", "position": {"x": 0, "y": 0}, "ground_truth": True})
    tracker.update_from_robot_pose({
        "target_id": "sim1", "position": {"x": 9000, "y": 9000}, "ground_truth": True})
    assert tracker.get_target("sim1").velocity_suspicious is False


def test_onboard_teleport_is_flagged_suspicious(tracker):
    """A real robot claiming a 9 km jump between ticks is not to be trusted."""
    tracker.update_from_robot_pose({"target_id": "real1", "position": {"x": 0, "y": 0}})
    tracker.update_from_robot_pose({"target_id": "real1", "position": {"x": 9000, "y": 9000}})
    assert tracker.get_target("real1").velocity_suspicious is True


# ---------------------------------------------------------------------------
# Operator alliance pinning — the standing tracker invariant
# ---------------------------------------------------------------------------

def test_operator_pinned_alliance_survives_pose_updates(tracker):
    tracker.update_from_robot_pose({
        "target_id": "r1", "position": {"x": 0, "y": 0}, "alliance": "friendly"})
    t = tracker.get_target("r1")
    t.alliance = "hostile"
    t.alliance_source = "operator"

    tracker.update_from_robot_pose({
        "target_id": "r1", "position": {"x": 1, "y": 1}, "alliance": "friendly"})
    t = tracker.get_target("r1")
    assert t.alliance == "hostile"
    assert t.alliance_source == "operator"


# ---------------------------------------------------------------------------
# Trail — the operator sees where the body has been
# ---------------------------------------------------------------------------

def test_pose_updates_record_a_trail(tracker):
    for i in range(4):
        tracker.update_from_robot_pose({
            "target_id": "r1", "position": {"x": float(i), "y": 0.0}})
    trail = tracker.history.get_trail_dicts("r1", max_points=10)
    assert len(trail) >= 4


# ---------------------------------------------------------------------------
# The capability-2 round trip: isaac_frame -> tracker -> map payload
# ---------------------------------------------------------------------------

def test_isaac_frame_pose_round_trips_to_map_payload(tracker):
    """The whole point: an Isaac stage pose lands on the map unchanged.

    Uses the SAME conversion library the live bridge uses, so a sign error
    introduced in either half fails here rather than on the GPU box.
    """
    from tritium_lib.geo.isaac_frame import IsaacFrame

    frame = IsaacFrame()
    # Isaac stage pose: 12 m east, 7 m north, facing Isaac-yaw 0 (== east).
    east, north, _up = frame.stage_to_local((12.0, 7.0, 0.45))
    heading = frame.yaw_to_heading(0.0)

    tracker.update_from_robot_pose({
        "target_id": "isaac_go2_01",
        "name": "Go2",
        "asset_type": "quadruped",
        "position": {"x": east, "y": north},
        "heading": heading,
        "ground_truth": True,
    })

    d = tracker.get_target("isaac_go2_01").to_dict()
    assert d["position"]["x"] == pytest.approx(12.0)
    assert d["position"]["y"] == pytest.approx(7.0)
    assert d["heading"] == pytest.approx(90.0)  # Isaac yaw 0 == east == 90 deg
    assert d["position_source"] == "sim_truth"

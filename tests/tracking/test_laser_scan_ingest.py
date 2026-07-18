"""LiDAR ingest — a laser sweep becoming operator-visible contacts.

``tritium_lib.geo.laser_scan`` turns a raw LaserScan into world-frame
``Obstacle`` clusters, but nothing consumed it: the Command Center had no
LiDAR ingest of any kind, so a live sweep out of Isaac had nowhere to land.
This is that landing point.

The hard part is NOT the geometry (that is already tested in
``tests/geo/test_laser_scan.py``) — it is **identity across sweeps**.  A
cluster index is not a track: a sweep that picks up one extra speckle
renumbers every obstacle behind it, so naive index-based IDs make a static
wall look like a stream of new contacts appearing and vanishing.  This ingest
therefore associates by POSITION within a gate — global nearest neighbour,
the standard LaserScan tracking association — so a stationary obstacle keeps
one target id sweep after sweep.

Two further properties matter operationally:

  * obstacles are **unknown** contacts, not friendlies.  A LiDAR return says
    "something is there", never what or whose it is.
  * confidence is **below** a self-reported robot pose.  A range cluster is
    weaker evidence than a body telling you where it is.

Copyright (c) Matthew Valancy / Valpatel Software LLC. AGPL-3.0.
"""

import math

import pytest

from tritium_lib.tracking.target_tracker import TargetTracker


@pytest.fixture
def tracker():
    return TargetTracker()


def _sweep(ranges, **kw):
    """A minimal /scan-shaped payload, matching the lidar_server contract."""
    payload = {
        "lidar_id": "isaac-lidar-01",
        "ranges": list(ranges),
        "angle_min": -math.pi,
        "angle_increment": 2.0 * math.pi / len(ranges),
        "range_min": 0.1,
        "range_max": 30.0,
    }
    payload.update(kw)
    return payload


def _one_obstacle_sweep(beam: int, distance: float, num_beams: int = 360, **kw):
    """All no-return except a short run of beams around ``beam``."""
    ranges = [30.0] * num_beams
    for b in (beam - 1, beam, beam + 1):
        ranges[b % num_beams] = distance
    return _sweep(ranges, **kw)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_sweep_creates_a_target_per_obstacle(tracker):
    ids = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    assert len(ids) == 1
    t = tracker.get_target(ids[0])
    assert t is not None
    assert t.source == "lidar"


def test_empty_sweep_creates_nothing(tracker):
    # Every beam at range_max is "no return", not an obstacle ring.
    assert tracker.update_from_laser_scan(_sweep([30.0] * 360)) == []


def test_obstacle_is_unknown_not_friendly(tracker):
    """A range return says something is there — never what, never whose."""
    ids = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    t = tracker.get_target(ids[0])
    assert t.alliance == "unknown"


def test_confidence_is_below_a_self_reported_robot_pose(tracker):
    ids = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    lidar_conf = tracker.get_target(ids[0]).position_confidence
    tracker.update_from_robot_pose(
        {"target_id": "robot_go2", "position": {"x": 0, "y": 0}, "ground_truth": True}
    )
    assert lidar_conf < tracker.get_target("robot_go2").position_confidence


# ---------------------------------------------------------------------------
# Geometry — the sensor pose is applied, not ignored
# ---------------------------------------------------------------------------

def test_obstacle_lands_in_world_frame_using_sensor_pose(tracker):
    """Beam 270 of a [-pi, pi) sweep points +y in the body frame; with the
    sensor translated to (10, 0) the obstacle must land near (10, 4), NOT
    near (0, 4).  This is what catches an ingest that drops the pose."""
    ids = tracker.update_from_laser_scan(
        _one_obstacle_sweep(270, 4.0, sensor_x=10.0, sensor_y=0.0, sensor_yaw_deg=0.0)
    )
    x, y = tracker.get_target(ids[0]).position
    assert x == pytest.approx(10.0, abs=0.2)
    assert y == pytest.approx(4.0, abs=0.2)


def test_two_separated_obstacles_yield_two_targets(tracker):
    ranges = [30.0] * 360
    for b in (89, 90, 91):
        ranges[b] = 4.0
    for b in (269, 270, 271):
        ranges[b] = 4.0
    assert len(tracker.update_from_laser_scan(_sweep(ranges))) == 2


# ---------------------------------------------------------------------------
# Identity across sweeps — the property that makes this a TRACK, not a blip
# ---------------------------------------------------------------------------

def test_static_obstacle_keeps_its_id_across_sweeps(tracker):
    first = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    second = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    assert first == second
    assert len(tracker.get_all()) == 1


def test_extra_speckle_does_not_renumber_the_real_obstacle(tracker):
    """The regression this whole design exists for.  Sweep 2 adds a nearer
    return at a LOWER beam index, which shifts every cluster index behind it.
    Index-based ids would rename the real obstacle; position association must
    not."""
    first = tracker.update_from_laser_scan(_one_obstacle_sweep(200, 4.0))

    ranges = [30.0] * 360
    for b in (9, 10, 11):          # a new cluster at a lower index
        ranges[b] = 2.0
    for b in (199, 200, 201):      # the SAME obstacle as sweep 1
        ranges[b] = 4.0
    second = tracker.update_from_laser_scan(_sweep(ranges))

    assert first[0] in second, "the original obstacle was renumbered"
    assert len(second) == 2


def test_obstacle_beyond_the_gate_becomes_a_new_track(tracker):
    """Association must not teleport a track across the map to keep an id."""
    first = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    second = tracker.update_from_laser_scan(
        _one_obstacle_sweep(90, 4.0, sensor_x=25.0, association_gate_m=1.0)
    )
    assert second != first
    assert len(tracker.get_all()) == 2


def test_moving_obstacle_within_the_gate_keeps_its_id_and_moves(tracker):
    first = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    before = tracker.get_target(first[0]).position
    second = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.5))
    assert second == first
    after = tracker.get_target(first[0]).position
    assert after != before


def test_two_lidars_do_not_share_track_ids(tracker):
    """Different sensors at the same spot are still different sensors — an
    id collision here would silently fuse two rooms into one."""
    a = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    b = tracker.update_from_laser_scan(
        _one_obstacle_sweep(90, 4.0, lidar_id="isaac-lidar-02")
    )
    assert set(a).isdisjoint(set(b))


# ---------------------------------------------------------------------------
# Payload robustness — a connector should not be able to crash the tracker
# ---------------------------------------------------------------------------

def test_missing_ranges_is_ignored_not_fatal(tracker):
    assert tracker.update_from_laser_scan({"lidar_id": "x"}) == []


def test_min_points_filters_speckle(tracker):
    """A single stray return is a dust mote; drawing it puts a phantom
    obstacle in front of a planner that then refuses to move."""
    ranges = [30.0] * 360
    ranges[90] = 4.0                      # exactly one return
    assert tracker.update_from_laser_scan(_sweep(ranges, min_points=3)) == []
    assert len(tracker.update_from_laser_scan(_sweep(ranges, min_points=1))) == 1


def test_signal_count_accumulates_on_reobservation(tracker):
    ids = tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    tracker.update_from_laser_scan(_one_obstacle_sweep(90, 4.0))
    assert tracker.get_target(ids[0]).signal_count == 2

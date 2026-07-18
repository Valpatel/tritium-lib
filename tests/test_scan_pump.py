# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""LiDAR sweep -> Command Center sighting: the forwarding decision.

``sensor_rig`` documents that LiDAR "streams *sightings*, not frames" and is
therefore given no camera-feed registration.  Nothing implemented that path,
so the rig's LiDAR came up healthy and reached the operator's map never.

The seam under test is the decision half of that pump: given a polled sweep,
should it be forwarded, and as what payload.  The refusals are the point — a
LiDAR that has stopped scanning keeps answering ``/scan`` with its last sweep,
and forwarding that repeatedly refreshes a track's ``last_seen``, painting a
live contact on the operator's map from a dead sensor.
"""

from __future__ import annotations

import pytest

from tritium_lib.fleet.scan_pump import ScanDecision, ScanPump


def _sweep(ranges=None, **kw):
    """A ``/scan``-shaped payload as ``lidar_server`` serves it."""
    base = {
        "angle_min": -3.1416,
        "angle_increment": 0.01745,
        "range_min": 0.1,
        "range_max": 30.0,
        "ranges": ranges if ranges is not None else _wall(),
    }
    base.update(kw)
    return base


def _wall(n=360, hit_at=(90, 91, 92), hit_range=4.0, max_range=30.0):
    """A sweep with a small cluster of returns and the rest no-return."""
    r = [max_range] * n
    for i in hit_at:
        r[i] = hit_range
    return r


# --------------------------------------------------------------------------- #
# The happy path: a real sweep becomes a lidar sighting payload
# --------------------------------------------------------------------------- #


def test_a_sweep_with_returns_is_forwarded_as_a_lidar_sighting():
    pump = ScanPump(lidar_id="isaac-lidar-01")
    decision = pump.offer(_sweep())

    assert isinstance(decision, ScanDecision)
    assert decision.forward is True
    assert decision.reason == "forward"
    assert decision.payload["source"] == "lidar"
    assert decision.payload["lidar_id"] == "isaac-lidar-01"


def test_the_sweep_geometry_is_carried_through_untouched():
    """Dropping the geometry would place every obstacle at the wrong bearing.

    The tracker converts polar -> world using exactly these fields; a payload
    that omits them silently falls back to defaults, and a sweep whose
    ``angle_min`` is 0 rather than -pi lands its obstacles half a turn away.
    """
    pump = ScanPump(lidar_id="l")
    payload = pump.offer(_sweep()).payload

    assert payload["angle_min"] == pytest.approx(-3.1416)
    assert payload["angle_increment"] == pytest.approx(0.01745)
    assert payload["range_min"] == pytest.approx(0.1)
    assert payload["range_max"] == pytest.approx(30.0)
    assert payload["ranges"] == _wall()


def test_the_sensor_pose_rides_along_so_returns_land_in_world_space():
    pump = ScanPump(lidar_id="l", sensor_x=12.5, sensor_y=-3.25,
                    sensor_yaw_deg=90.0)
    payload = pump.offer(_sweep()).payload

    assert payload["sensor_x"] == pytest.approx(12.5)
    assert payload["sensor_y"] == pytest.approx(-3.25)
    assert payload["sensor_yaw_deg"] == pytest.approx(90.0)


def test_a_moving_body_updates_the_pose_between_sweeps():
    """A body-mounted LiDAR moves; the pose is per-sweep, not per-pump."""
    pump = ScanPump(lidar_id="l")
    pump.set_sensor_pose(1.0, 2.0, 45.0)
    payload = pump.offer(_sweep()).payload

    assert payload["sensor_x"] == pytest.approx(1.0)
    assert payload["sensor_y"] == pytest.approx(2.0)
    assert payload["sensor_yaw_deg"] == pytest.approx(45.0)


# --------------------------------------------------------------------------- #
# The refusals — the reason this is a seam and not a one-line POST
# --------------------------------------------------------------------------- #


def test_an_unchanged_sweep_is_refused_as_stale():
    """A frozen LiDAR answers /scan forever with its last sweep.

    Forwarding it refreshes the track's last_seen every poll, so a dead
    sensor is indistinguishable from a static wall in front of a live one.
    """
    pump = ScanPump(lidar_id="l")
    assert pump.offer(_sweep()).forward is True

    repeat = pump.offer(_sweep())
    assert repeat.forward is False
    assert repeat.reason == "stale"
    assert repeat.payload is None


def test_a_changed_sweep_after_a_stale_one_forwards_again():
    pump = ScanPump(lidar_id="l")
    pump.offer(_sweep())
    pump.offer(_sweep())                       # stale
    moved = _sweep(ranges=_wall(hit_at=(120, 121, 122)))

    assert pump.offer(moved).forward is True


def test_a_sweep_with_no_returns_at_all_is_refused():
    """Every beam at range_max carries no obstacle information.

    An open field and an unplugged sensor produce byte-identical sweeps, so
    forwarding one buys nothing and grants a dead LiDAR a heartbeat.
    """
    pump = ScanPump(lidar_id="l")
    decision = pump.offer(_sweep(ranges=[30.0] * 360))

    assert decision.forward is False
    assert decision.reason == "no_returns"


def test_a_malformed_or_empty_sweep_is_refused_not_raised():
    """A connector must never be able to crash the pump."""
    pump = ScanPump(lidar_id="l")

    for bad in ({}, {"ranges": []}, {"ranges": None}):
        decision = pump.offer(bad)
        assert decision.forward is False
        assert decision.reason == "malformed"


# --------------------------------------------------------------------------- #
# The breaker: a wedged Command Center must not be hammered forever
# --------------------------------------------------------------------------- #


def test_consecutive_failures_trip_the_breaker_and_stop_forwarding():
    pump = ScanPump(lidar_id="l", max_failures=3)
    for i in range(3):
        assert pump.offer(_sweep(ranges=_wall(hit_at=(i, i + 1, i + 2)))).forward
        pump.record_result(False)

    assert pump.tripped is True
    blocked = pump.offer(_sweep(ranges=_wall(hit_at=(200, 201, 202))))
    assert blocked.forward is False
    assert blocked.reason == "tripped"


def test_a_success_resets_the_failure_run():
    """Intermittent failures are normal; only a *run* of them means dead.

    The run must be long enough to trip *without* the reset, or the test
    passes against a pump that never resets at all — five alternating results
    never stack three deep either way.
    """
    pump = ScanPump(lidar_id="l", max_failures=3)
    results = [False, False, True, False, False]
    for i, ok in enumerate(results):
        pump.offer(_sweep(ranges=_wall(hit_at=(i, i + 1, i + 2))))
        pump.record_result(ok)

    # Five results, four of them failures: without the reset the run reaches
    # 4 >= 3 and the breaker trips.  It must not.
    assert pump.tripped is False
    assert pump.stats()["accepted"] == 1


def test_a_tripped_refusal_is_counted_like_any_other():
    """A breaker that silently drops sweeps looks identical to an idle LiDAR.

    The refusal counters are how an operator learns the pump gave up rather
    than the scene going quiet, so ``tripped`` must show up in them.
    """
    pump = ScanPump(lidar_id="l", max_failures=2)
    for i in range(2):
        pump.offer(_sweep(ranges=_wall(hit_at=(i, i + 1, i + 2))))
        pump.record_result(False)

    pump.offer(_sweep(ranges=_wall(hit_at=(50, 51, 52))))
    stats = pump.stats()

    assert stats["tripped"] is True
    assert stats["refusals"]["tripped"] == 1
    assert stats["refused"] == 1


def test_stats_report_forwarded_and_refused_counts_honestly():
    pump = ScanPump(lidar_id="l")
    pump.offer(_sweep())
    pump.record_result(True)
    pump.offer(_sweep())                       # stale
    pump.offer(_sweep(ranges=[30.0] * 360))    # no returns

    stats = pump.stats()
    assert stats["forwarded"] == 1
    assert stats["accepted"] == 1
    assert stats["refused"] == 2
    assert stats["refusals"]["stale"] == 1
    assert stats["refusals"]["no_returns"] == 1

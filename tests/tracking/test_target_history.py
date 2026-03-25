# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.target_history."""

import math
import threading

from tritium_lib.tracking.target_history import TargetHistory, PositionRecord


# --- PositionRecord ---

def test_position_record_fields():
    r = PositionRecord(x=1.0, y=2.0, timestamp=100.0)
    assert r.x == 1.0
    assert r.y == 2.0
    assert r.timestamp == 100.0


# --- TargetHistory ---

def test_empty_history():
    th = TargetHistory()
    assert th.tracked_count == 0
    assert th.get_trail("nonexistent") == []
    assert th.get_speed("nonexistent") == 0.0
    assert th.get_heading("nonexistent") == 0.0


def test_record_and_trail():
    th = TargetHistory()
    th.record("t-1", (10.0, 20.0), timestamp=100.0)
    th.record("t-1", (30.0, 40.0), timestamp=101.0)
    trail = th.get_trail("t-1")
    assert len(trail) == 2
    assert trail[0] == (10.0, 20.0, 100.0)
    assert trail[1] == (30.0, 40.0, 101.0)


def test_tracked_count():
    th = TargetHistory()
    th.record("t-1", (0, 0), timestamp=100.0)
    th.record("t-2", (0, 0), timestamp=100.0)
    assert th.tracked_count == 2


def test_trail_max_points():
    th = TargetHistory()
    for i in range(10):
        th.record("t-1", (float(i), 0.0), timestamp=100.0 + i)
    trail = th.get_trail("t-1", max_points=3)
    assert len(trail) == 3
    # Should be the 3 most recent
    assert trail[0][0] == 7.0
    assert trail[2][0] == 9.0


def test_speed_calculation():
    th = TargetHistory()
    th.record("t-1", (0.0, 0.0), timestamp=100.0)
    th.record("t-1", (10.0, 0.0), timestamp=101.0)
    speed = th.get_speed("t-1")
    assert abs(speed - 10.0) < 0.01  # 10 units/second


def test_speed_stationary():
    th = TargetHistory()
    th.record("t-1", (5.0, 5.0), timestamp=100.0)
    th.record("t-1", (5.0, 5.0), timestamp=101.0)
    assert th.get_speed("t-1") == 0.0


def test_speed_single_record():
    th = TargetHistory()
    th.record("t-1", (0, 0), timestamp=100.0)
    assert th.get_speed("t-1") == 0.0


def test_heading_north():
    th = TargetHistory()
    th.record("t-1", (0.0, 0.0), timestamp=100.0)
    th.record("t-1", (0.0, 10.0), timestamp=101.0)
    heading = th.get_heading("t-1")
    assert abs(heading) < 1.0 or abs(heading - 360) < 1.0  # ~0 degrees (north)


def test_heading_east():
    th = TargetHistory()
    th.record("t-1", (0.0, 0.0), timestamp=100.0)
    th.record("t-1", (10.0, 0.0), timestamp=101.0)
    heading = th.get_heading("t-1")
    assert abs(heading - 90.0) < 1.0  # ~90 degrees (east)


def test_heading_stationary():
    th = TargetHistory()
    th.record("t-1", (5.0, 5.0), timestamp=100.0)
    th.record("t-1", (5.0, 5.0), timestamp=101.0)
    assert th.get_heading("t-1") == 0.0


def test_get_trail_dicts():
    th = TargetHistory()
    th.record("t-1", (1.0, 2.0), timestamp=100.0)
    dicts = th.get_trail_dicts("t-1")
    assert len(dicts) == 1
    assert dicts[0]["x"] == 1.0
    assert dicts[0]["y"] == 2.0
    assert dicts[0]["t"] == 100.0


def test_ring_buffer_limit():
    th = TargetHistory()
    for i in range(1100):
        th.record("t-1", (float(i), 0.0), timestamp=100.0 + i)
    trail = th.get_trail("t-1", max_points=2000)
    assert len(trail) == 1000  # MAX_RECORDS_PER_TARGET


def test_clear_specific_target():
    th = TargetHistory()
    th.record("t-1", (0, 0), timestamp=100.0)
    th.record("t-2", (0, 0), timestamp=100.0)
    th.clear("t-1")
    assert th.tracked_count == 1
    assert th.get_trail("t-1") == []
    assert len(th.get_trail("t-2")) == 1


def test_clear_all():
    th = TargetHistory()
    th.record("t-1", (0, 0), timestamp=100.0)
    th.record("t-2", (0, 0), timestamp=100.0)
    th.clear()
    assert th.tracked_count == 0


def test_prune_stale():
    th = TargetHistory()
    # Record with old timestamp — will be stale
    th.record("old", (0, 0), timestamp=0.0)
    # Record with recent timestamp
    import time
    th.record("new", (0, 0), timestamp=time.monotonic())
    th.prune_stale()
    assert th.get_trail("old") == []
    assert len(th.get_trail("new")) == 1


def test_thread_safety():
    th = TargetHistory()
    errors = []

    def writer(tid):
        try:
            for i in range(50):
                th.record(tid, (float(i), float(i)), timestamp=100.0 + i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(f"t-{n}",)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert th.tracked_count == 4


# --- Deepened tests ---

def test_speed_diagonal_movement():
    """Speed for diagonal movement should use pythagorean distance."""
    th = TargetHistory()
    th.record("t-1", (0.0, 0.0), timestamp=100.0)
    th.record("t-1", (3.0, 4.0), timestamp=101.0)
    speed = th.get_speed("t-1")
    assert abs(speed - 5.0) < 0.01  # 3-4-5 triangle

def test_speed_with_sample_count():
    """Speed should only use the last N samples."""
    th = TargetHistory()
    th.record("t-1", (0.0, 0.0), timestamp=100.0)
    th.record("t-1", (100.0, 0.0), timestamp=101.0)  # fast
    th.record("t-1", (101.0, 0.0), timestamp=102.0)   # slow
    th.record("t-1", (102.0, 0.0), timestamp=103.0)   # slow
    speed = th.get_speed("t-1", sample_count=2)
    assert abs(speed - 1.0) < 0.01

def test_heading_south():
    th = TargetHistory()
    th.record("t-1", (0.0, 10.0), timestamp=100.0)
    th.record("t-1", (0.0, 0.0), timestamp=101.0)
    heading = th.get_heading("t-1")
    assert abs(heading - 180.0) < 1.0

def test_heading_west():
    th = TargetHistory()
    th.record("t-1", (10.0, 0.0), timestamp=100.0)
    th.record("t-1", (0.0, 0.0), timestamp=101.0)
    heading = th.get_heading("t-1")
    assert abs(heading - 270.0) < 1.0

def test_heading_with_sample_count():
    """Heading should use first and last of sampled points."""
    th = TargetHistory()
    th.record("t-1", (0.0, 0.0), timestamp=100.0)
    th.record("t-1", (10.0, 0.0), timestamp=101.0)  # east
    th.record("t-1", (10.0, 10.0), timestamp=102.0)  # then north
    heading = th.get_heading("t-1", sample_count=2)
    assert abs(heading) < 1.0 or abs(heading - 360.0) < 1.0  # north

def test_prune_stale_with_custom_timeout():
    """Prune uses PRUNE_TIMEOUT relative to time.monotonic()."""
    import time
    th = TargetHistory()
    now = time.monotonic()
    th.record("recent", (0, 0), timestamp=now - 10)  # 10s ago
    th.record("ancient", (0, 0), timestamp=now - 99999)  # very old
    th.prune_stale()
    assert th.get_trail("recent") != []
    assert th.get_trail("ancient") == []

def test_multiple_targets_independent():
    """Records for different targets should be independent."""
    th = TargetHistory()
    th.record("t-1", (1.0, 1.0), timestamp=100.0)
    th.record("t-2", (2.0, 2.0), timestamp=100.0)
    trail1 = th.get_trail("t-1")
    trail2 = th.get_trail("t-2")
    assert trail1[0][0] == 1.0
    assert trail2[0][0] == 2.0

def test_speed_zero_time_delta():
    """Same timestamp records should return 0 speed."""
    th = TargetHistory()
    th.record("t-1", (0.0, 0.0), timestamp=100.0)
    th.record("t-1", (10.0, 0.0), timestamp=100.0)
    assert th.get_speed("t-1") == 0.0

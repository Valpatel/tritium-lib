# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Wave 200: motion-aware YOLO matching.

The Wave 200 perf audit found 306 YOLO targets accumulating in 2 minutes,
with no targets ever pruned (last_seen kept getting refreshed by nearby
detections).  Root cause: the matcher used a fixed 3 m radius, so a fast
target moving >3 m between camera frames spawned a new ID while the old
ID was kept alive by other detections nearby.

These tests pin the new behaviour:
  - first-match becomes closest-match (no premature break),
  - the radius scales with the time gap × YOLO_MAX_TRACK_SPEED,
  - a slow target near a still ghost still matches the ghost.
"""

from __future__ import annotations

import time

from tritium_lib.tracking.target_tracker import TargetTracker


def _detection(cls: str, x: float, y: float, conf: float = 0.9) -> dict:
    return {"class_name": cls, "center_x": x, "center_y": y, "confidence": conf}


def test_fast_target_does_not_split_into_new_id_each_frame() -> None:
    tracker = TargetTracker()
    tracker.update_from_detection(_detection("car", 0.0, 0.0))
    initial_count = len(tracker._targets)

    # Car at 25 m/s, camera fires every 200 ms → 5 m per frame, beyond the
    # 3 m base radius.  Motion-aware matching should reuse the same id.
    # We backdate last_seen to simulate the 200 ms frame interval without
    # actually sleeping (test must stay fast).
    tid = next(iter(tracker._targets))
    for step in range(1, 6):
        tracker._targets[tid].last_seen = time.monotonic() - 0.2
        x = step * 5.0
        tracker.update_from_detection(_detection("car", x, 0.0))

    assert len(tracker._targets) == initial_count, (
        f"Fast target spawned new IDs: {list(tracker._targets.keys())}"
    )


def test_slow_target_near_ghost_matches_ghost_not_new_id() -> None:
    tracker = TargetTracker()
    tracker.update_from_detection(_detection("person", 10.0, 10.0))
    ghost_id = next(iter(tracker._targets))

    time.sleep(0.05)
    # 0.5 m offset, which is well inside the base radius for a fresh target.
    tracker.update_from_detection(_detection("person", 10.5, 10.0))

    assert len(tracker._targets) == 1
    assert next(iter(tracker._targets)) == ghost_id


def test_distinct_classes_remain_separate() -> None:
    tracker = TargetTracker()
    tracker.update_from_detection(_detection("person", 5.0, 5.0))
    tracker.update_from_detection(_detection("car", 5.0, 5.0))

    assert len(tracker._targets) == 2


def test_match_picks_closest_not_first() -> None:
    tracker = TargetTracker()
    # Place two existing targets.  Without the closest-match fix the
    # iteration order would pick whichever the dict yielded first; with
    # the fix the nearer one wins regardless.
    tracker.update_from_detection(_detection("person", 10.0, 10.0))
    tracker.update_from_detection(_detection("person", 13.0, 10.0))
    assert len(tracker._targets) == 2

    a_id, b_id = list(tracker._targets.keys())
    a_pos = tracker._targets[a_id].position
    b_pos = tracker._targets[b_id].position

    # Detection adjacent to the *second* target.
    tracker.update_from_detection(_detection("person", 13.1, 10.0))
    closer_id = b_id if abs(b_pos[0] - 13.1) < abs(a_pos[0] - 13.1) else a_id

    assert tracker._targets[closer_id].position == (13.1, 10.0)
    assert len(tracker._targets) == 2


def test_motion_budget_does_not_match_truly_distant() -> None:
    tracker = TargetTracker()
    tracker.update_from_detection(_detection("person", 0.0, 0.0))

    # 2 s gap × 30 m/s = 60 m budget.  A detection 200 m away should still
    # spawn a new ID, not collapse into the old one.
    time.sleep(0.05)
    tracker._targets[next(iter(tracker._targets))].last_seen = time.monotonic() - 2.0
    tracker.update_from_detection(_detection("person", 200.0, 200.0))

    assert len(tracker._targets) == 2

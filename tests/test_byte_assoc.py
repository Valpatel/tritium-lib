# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for BYTE two-stage data association (fusion/byte_assoc.py).

The core promise of BYTE: a low-confidence detection that a single-stage
0.4 confidence cut (fusion/engine.py:431) would HARD-DROP is recovered in
the second association round, while clean high-confidence detections still
associate normally in the first round.
"""

from __future__ import annotations

from tritium_lib.fusion.byte_assoc import (
    Detection,
    Track,
    byte_associate,
    iou,
)


def _box(cx: float, cy: float, w: float = 2.0, h: float = 2.0) -> tuple:
    """Center-form box -> (x1, y1, x2, y2)."""
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_iou_identical_is_one():
    b = _box(0.0, 0.0)
    assert iou(b, b) == 1.0


def test_iou_disjoint_is_zero():
    assert iou(_box(0.0, 0.0), _box(100.0, 100.0)) == 0.0


def test_iou_half_overlap():
    # Two 2x2 boxes offset by 1 in x -> overlap area 1*2=2, union 4+4-2=6.
    a = _box(0.0, 0.0)
    b = _box(1.0, 0.0)
    assert abs(iou(a, b) - (2.0 / 6.0)) < 1e-9


def test_high_conf_clean_association():
    """A clean high-confidence detection on top of a track associates in round 1."""
    tracks = [Track(track_id="t1", box=_box(0.0, 0.0))]
    dets = [Detection(box=_box(0.1, 0.0), score=0.9)]

    result = byte_associate(tracks, dets)

    assert len(result.matches) == 1
    track_idx, det_idx, stage = result.matches[0]
    assert tracks[track_idx].track_id == "t1"
    assert det_idx == 0
    assert stage == "high"  # matched in the first (high-confidence) round
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []


def test_low_conf_detection_recovered_by_second_round():
    """The headline test: a 0.3-score detection that the <0.4 cut would drop
    is recovered by BYTE's second association round."""
    tracks = [Track(track_id="t1", box=_box(0.0, 0.0))]
    # Single detection, below the 0.4 hard-drop threshold.
    dets = [Detection(box=_box(0.2, 0.0), score=0.3)]

    # Sanity: single-stage gate with the engine's 0.4 cut would drop it.
    single_stage_survivors = [d for d in dets if d.score >= 0.4]
    assert single_stage_survivors == []

    # BYTE recovers it: it survives because it still overlaps an active track.
    result = byte_associate(
        tracks, dets, high_thresh=0.5, low_thresh=0.1
    )

    assert len(result.matches) == 1
    track_idx, det_idx, stage = result.matches[0]
    assert tracks[track_idx].track_id == "t1"
    assert det_idx == 0
    assert stage == "low"  # recovered in the second (low-confidence) round
    assert result.unmatched_tracks == []


def test_low_conf_below_floor_is_dropped():
    """Detections under low_thresh are genuine noise and stay dropped."""
    tracks = [Track(track_id="t1", box=_box(0.0, 0.0))]
    dets = [Detection(box=_box(0.0, 0.0), score=0.05)]

    result = byte_associate(tracks, dets, high_thresh=0.5, low_thresh=0.1)

    assert result.matches == []
    assert result.unmatched_tracks == [0]
    # Detection was below the low floor -> not even a candidate, not "unmatched".
    assert result.unmatched_detections == []


def test_high_then_low_two_tracks():
    """One high-conf det locks track A in round 1; a low-conf det then
    recovers against the still-unmatched track B in round 2."""
    tracks = [
        Track(track_id="A", box=_box(0.0, 0.0)),
        Track(track_id="B", box=_box(50.0, 50.0)),
    ]
    dets = [
        Detection(box=_box(0.0, 0.0), score=0.95),   # high -> A
        Detection(box=_box(50.0, 50.0), score=0.25),  # low -> B (recovered)
    ]

    result = byte_associate(tracks, dets, high_thresh=0.5, low_thresh=0.1)

    matched = {tracks[ti].track_id: (di, stage) for ti, di, stage in result.matches}
    assert matched["A"] == (0, "high")
    assert matched["B"] == (1, "low")
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []


def test_distance_metric_mode():
    """BYTE also supports a centroid-distance gate for point detections
    (no boxes), used by RF/mesh modalities in the fusion engine."""
    tracks = [Track(track_id="t1", point=(0.0, 0.0))]
    dets = [Detection(point=(0.3, 0.0), score=0.3)]

    result = byte_associate(
        tracks,
        dets,
        metric="distance",
        high_thresh=0.5,
        low_thresh=0.1,
        max_distance=2.0,
    )

    assert len(result.matches) == 1
    assert result.matches[0][2] == "low"


def test_distance_gate_rejects_far_detection():
    tracks = [Track(track_id="t1", point=(0.0, 0.0))]
    dets = [Detection(point=(100.0, 0.0), score=0.9)]

    result = byte_associate(
        tracks, dets, metric="distance", max_distance=2.0
    )

    assert result.matches == []
    assert result.unmatched_tracks == [0]
    assert result.unmatched_detections == [0]


def test_empty_inputs():
    assert byte_associate([], []).matches == []
    r = byte_associate([Track(track_id="t1", box=_box(0, 0))], [])
    assert r.unmatched_tracks == [0]
    r2 = byte_associate([], [Detection(box=_box(0, 0), score=0.9)])
    assert r2.unmatched_detections == [0]

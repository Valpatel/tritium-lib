# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BYTE two-stage data association (algorithm-only).

Implements the association core of ByteTrack (Zhang et al., ECCV 2022):
instead of discarding every low-confidence detection, BYTE keeps them and
runs a *second* association round against the tracks that the first
(high-confidence) round left unmatched.  Occluded / blurred / partially
detected targets that a single confidence gate would throw away are
recovered as long as they still overlap an active track.

This matters here because ``fusion/engine.py`` (``ingest_camera``,
line ~431) HARD-DROPS any detection with ``confidence < 0.4``.  That is a
single-stage gate: a temporarily dim target simply vanishes from the
tactical picture, fragmenting its track and costing an identity switch
when it reappears.  BYTE's second round is the standard, license-clean
(MIT) remedy.

This module is deliberately decoupled from the fusion engine: it operates
over plain ``Track`` / ``Detection`` dataclasses and returns an
assignment plan.  Wiring it into ``FusionEngine`` is a documented
follow-up (left out here to avoid colliding with concurrent edits to
``engine.py``).

Two cost metrics are supported:

* ``metric="iou"``    — 1 - IoU between axis-aligned boxes (vision/YOLO).
* ``metric="distance"`` — Euclidean centroid distance (RF / mesh / point
  modalities that have a position but no bounding box).

The optimal assignment within each round uses the Hungarian algorithm via
:func:`scipy.optimize.linear_sum_assignment`.

Reference: Zhang, Y. et al. "ByteTrack: Multi-Object Tracking by
Associating Every Detection Box." ECCV 2022. Algorithm is MIT-licensed;
this is an independent reimplementation, no upstream code copied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)
Point = tuple[float, float]
Metric = Literal["iou", "distance"]

# Stage label is carried through to the caller so it can tell which
# detections were recovered by the second (low-confidence) round.
Stage = Literal["high", "low"]


@dataclass
class Track:
    """A confirmed/active track to associate detections against.

    Provide ``box`` for IoU matching or ``point`` for distance matching.
    """

    track_id: str
    box: Box | None = None
    point: Point | None = None


@dataclass
class Detection:
    """A single detection emitted by a sensor for one frame.

    ``score`` is the detector confidence in ``[0, 1]``.  BYTE splits the
    detection set on this value rather than discarding the low tail.
    """

    score: float
    box: Box | None = None
    point: Point | None = None


@dataclass
class AssociationResult:
    """Outcome of a BYTE association pass.

    Attributes:
        matches: list of ``(track_index, detection_index, stage)`` triples,
            where ``stage`` is ``"high"`` (first round) or ``"low"``
            (recovered in the second round).
        unmatched_tracks: indices of tracks that received no detection.
        unmatched_detections: indices of detections (above the low floor)
            that matched no track.  Detections below ``low_thresh`` are
            treated as noise and are *not* listed here.
    """

    matches: list[tuple[int, int, Stage]] = field(default_factory=list)
    unmatched_tracks: list[int] = field(default_factory=list)
    unmatched_detections: list[int] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Geometry primitives
# ----------------------------------------------------------------------------
def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two axis-aligned boxes ``(x1,y1,x2,y2)``.

    Returns 0.0 for disjoint or degenerate (zero-area) boxes, 1.0 for
    identical boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _euclidean(p_a: Point, p_b: Point) -> float:
    return math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])


# ----------------------------------------------------------------------------
# Cost matrix + single Hungarian round
# ----------------------------------------------------------------------------
def _cost_matrix(
    tracks: list[Track],
    track_idx: list[int],
    dets: list[Detection],
    det_idx: list[int],
    metric: Metric,
) -> np.ndarray:
    """Build a cost matrix (rows = tracks subset, cols = dets subset).

    Lower cost == better match.  IoU cost is ``1 - IoU``; distance cost is
    the raw Euclidean distance.
    """
    n = len(track_idx)
    m = len(det_idx)
    cost = np.zeros((n, m), dtype=float)
    for i, ti in enumerate(track_idx):
        trk = tracks[ti]
        for j, dj in enumerate(det_idx):
            det = dets[dj]
            if metric == "iou":
                if trk.box is None or det.box is None:
                    cost[i, j] = 1.0
                else:
                    cost[i, j] = 1.0 - iou(trk.box, det.box)
            else:  # distance
                if trk.point is None or det.point is None:
                    cost[i, j] = math.inf
                else:
                    cost[i, j] = _euclidean(trk.point, det.point)
    return cost


def _associate_round(
    tracks: list[Track],
    track_idx: list[int],
    dets: list[Detection],
    det_idx: list[int],
    metric: Metric,
    max_cost: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Run one Hungarian assignment round over the given subsets.

    Args:
        track_idx / det_idx: indices (into the full lists) participating in
            this round.
        max_cost: gate — a Hungarian pairing whose cost exceeds this is
            rejected (the pair is split back into the unmatched pools).

    Returns:
        (matches, remaining_track_idx, remaining_det_idx) where ``matches``
        is a list of ``(track_index, detection_index)`` into the full lists.
    """
    if not track_idx or not det_idx:
        return [], list(track_idx), list(det_idx)

    cost = _cost_matrix(tracks, track_idx, dets, det_idx, metric)
    rows, cols = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_cost:
            ti = track_idx[r]
            dj = det_idx[c]
            matches.append((ti, dj))
            matched_tracks.add(ti)
            matched_dets.add(dj)

    remaining_tracks = [ti for ti in track_idx if ti not in matched_tracks]
    remaining_dets = [dj for dj in det_idx if dj not in matched_dets]
    return matches, remaining_tracks, remaining_dets


# ----------------------------------------------------------------------------
# Public BYTE entry point
# ----------------------------------------------------------------------------
def byte_associate(
    tracks: list[Track],
    detections: list[Detection],
    *,
    metric: Metric = "iou",
    high_thresh: float = 0.5,
    low_thresh: float = 0.1,
    iou_thresh: float = 0.1,
    max_distance: float = 5.0,
) -> AssociationResult:
    """BYTE two-stage association.

    Round 1 (high): split detections at ``high_thresh``; Hungarian-match the
    high-confidence set against all tracks.

    Round 2 (low): take the detections in ``[low_thresh, high_thresh)`` —
    exactly the band that ``fusion/engine.py``'s ``confidence < 0.4`` gate
    would discard — and Hungarian-match them against the tracks left
    unmatched by round 1.  Survivors are flagged ``stage="low"``.

    Detections below ``low_thresh`` are noise: they never become candidates
    and never appear in ``unmatched_detections``.

    Args:
        tracks: active tracks (need ``box`` for ``metric="iou"`` or
            ``point`` for ``metric="distance"``).
        detections: this frame's detections with confidence ``score``.
        metric: ``"iou"`` (boxes) or ``"distance"`` (points).
        high_thresh: confidence boundary between the two rounds.
        low_thresh: floor below which a detection is discarded as noise.
        iou_thresh: minimum IoU to accept an IoU pairing (gate).
        max_distance: maximum centroid distance to accept a distance pairing.

    Returns:
        AssociationResult with stage-tagged matches and leftovers.
    """
    result = AssociationResult()

    # Gate value: a pairing is accepted only if its cost <= max_cost.
    if metric == "iou":
        max_cost = 1.0 - iou_thresh
    else:
        max_cost = max_distance

    # Partition detection indices by confidence band.
    high_dets = [i for i, d in enumerate(detections) if d.score >= high_thresh]
    low_dets = [
        i
        for i, d in enumerate(detections)
        if low_thresh <= d.score < high_thresh
    ]
    # (Detections with score < low_thresh are dropped entirely — noise.)

    all_track_idx = list(range(len(tracks)))

    # -- Round 1: high-confidence detections vs all tracks ----------------
    high_matches, rem_tracks, rem_high = _associate_round(
        tracks, all_track_idx, detections, high_dets, metric, max_cost
    )
    for ti, dj in high_matches:
        result.matches.append((ti, dj, "high"))

    # -- Round 2: low-confidence detections vs leftover tracks ------------
    # This is the recovery step BYTE adds over a single-stage gate.
    low_matches, rem_tracks, rem_low = _associate_round(
        tracks, rem_tracks, detections, low_dets, metric, max_cost
    )
    for ti, dj in low_matches:
        result.matches.append((ti, dj, "low"))

    result.unmatched_tracks = sorted(rem_tracks)
    # Unmatched detections: only those that were *candidates* (>= low_thresh)
    # but found no track.  High-confidence leftovers are genuine new-target
    # candidates; low-confidence leftovers are reported too so the caller can
    # decide (typically discard).
    result.unmatched_detections = sorted(rem_high + rem_low)
    return result


__all__ = [
    "Box",
    "Point",
    "Metric",
    "Stage",
    "Track",
    "Detection",
    "AssociationResult",
    "iou",
    "byte_associate",
]

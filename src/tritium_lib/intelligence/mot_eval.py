# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MOT scorecard — thin wrapper over py-motmetrics (MIT).

Turns a stream of per-frame ``(ground-truth ids, hypothesis ids,
distance matrix)`` into the standard CLEAR-MOT / IDF1 scorecard used to
grade the tracker (and, by the North Star, the simulator that exercises
it):

* **MOTA**  — Multiple Object Tracking Accuracy (1 - (FN+FP+IDSW)/GT).
* **MOTP**  — Multiple Object Tracking Precision (avg matched distance).
* **IDF1**  — identity F1; rewards keeping the *same* id on the *same*
  object across the whole sequence.
* **id switches / fragmentations / misses / false positives**.

A perfect track scores MOTA == IDF1 == 1.0 with zero switches; an identity
swap (the failure BYTE recovery is meant to prevent) shows up as
``num_switches``.

``py-motmetrics`` is MIT-licensed and already installed, but this wrapper
imports it defensively: if it is ever missing the scorecard comes back
with ``available=False`` and a reason, never an exception, so callers
(reporting / AAR / dashboards) degrade gracefully.

Distances are pre-computed by the caller (e.g. ``1 - IoU`` or Euclidean
metres) and gated by ``max_distance``; pairs over the gate are treated as
non-matches.  Use :data:`float('nan')` in the distance matrix for pairs
that should never match.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

try:  # pragma: no cover - exercised via monkeypatch in tests
    import motmetrics as mm  # type: ignore

    _HAVE_MOTMETRICS = True
except Exception:  # pragma: no cover
    mm = None  # type: ignore
    _HAVE_MOTMETRICS = False


@dataclass
class MOTScorecard:
    """Computed multi-object-tracking metrics for one sequence.

    When ``available`` is ``False`` (motmetrics missing) every numeric
    field is ``None`` and ``reason`` explains why.
    """

    available: bool = True
    reason: str = ""
    mota: float | None = None
    motp: float | None = None
    idf1: float | None = None
    num_switches: int = 0
    num_fragmentations: int = 0
    num_misses: int = 0
    num_false_positives: int = 0
    num_objects: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (NaN floats coerced to ``None``)."""
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and math.isnan(v):
                d[k] = None
        return d


def _unavailable(reason: str) -> MOTScorecard:
    return MOTScorecard(
        available=False,
        reason=reason,
        mota=None,
        motp=None,
        idf1=None,
    )


def _gate_distances(
    distances: list[list[float]] | None,
    max_distance: float | None,
) -> Any:
    """Convert a nested distance list into the form motmetrics wants.

    Pairs whose distance exceeds ``max_distance`` (or that are already NaN)
    become NaN, which motmetrics reads as "cannot match".
    """
    if not distances:
        return distances
    gated: list[list[float]] = []
    for row in distances:
        if row is None:
            gated.append([])
            continue
        new_row: list[float] = []
        for val in row:
            if val is None:
                new_row.append(float("nan"))
            elif max_distance is not None and val > max_distance:
                new_row.append(float("nan"))
            else:
                new_row.append(float(val))
        gated.append(new_row)
    return gated


def score_mot(
    frames: list[dict[str, Any]],
    *,
    max_distance: float | None = None,
) -> MOTScorecard:
    """Score a sequence of frames into a :class:`MOTScorecard`.

    Args:
        frames: ordered list of per-frame dicts, each with::

            {
                "gt_ids":    [hashable, ...],   # ground-truth object ids
                "hyp_ids":   [hashable, ...],   # tracker hypothesis ids
                "distances": [[float, ...], ...]  # row per gt, col per hyp
            }

            ``distances[i][j]`` is the cost of matching ``gt_ids[i]`` to
            ``hyp_ids[j]`` (NaN => disallowed).  Empty lists are fine
            (e.g. a frame with no detections).
        max_distance: optional gate; distances above it are treated as NaN.

    Returns:
        MOTScorecard.  ``available=False`` if motmetrics is unavailable.
    """
    if not _HAVE_MOTMETRICS or mm is None:
        return _unavailable(
            "py-motmetrics is not installed; install the 'testing' extra "
            "or `pip install motmetrics` to enable MOTA/IDF1 scoring."
        )

    acc = mm.MOTAccumulator(auto_id=True)
    for frame in frames:
        gt_ids = list(frame.get("gt_ids", []))
        hyp_ids = list(frame.get("hyp_ids", []))
        distances = _gate_distances(frame.get("distances"), max_distance)
        acc.update(gt_ids, hyp_ids, distances)

    metric_names = [
        "mota",
        "motp",
        "idf1",
        "num_switches",
        "num_fragmentations",
        "num_misses",
        "num_false_positives",
        "num_objects",
    ]
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metric_names, name="seq")
    row = summary.to_dict("records")[0]

    def _f(key: str) -> float | None:
        val = row.get(key)
        if val is None:
            return None
        fval = float(val)
        return None if math.isnan(fval) else fval

    def _i(key: str) -> int:
        val = row.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 0
        return int(val)

    return MOTScorecard(
        available=True,
        reason="",
        mota=_f("mota"),
        motp=_f("motp"),
        idf1=_f("idf1"),
        num_switches=_i("num_switches"),
        num_fragmentations=_i("num_fragmentations"),
        num_misses=_i("num_misses"),
        num_false_positives=_i("num_false_positives"),
        num_objects=_i("num_objects"),
    )


__all__ = ["MOTScorecard", "score_mot"]

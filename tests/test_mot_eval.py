# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the MOT scorecard wrapper (intelligence/mot_eval.py).

Thin wrapper over py-motmetrics (MIT).  A perfect track must score
MOTA ~= 1.0 / IDF1 ~= 1.0 with zero switches; an injected identity
switch must surface as num_switches == 1.  When motmetrics is absent the
wrapper must degrade gracefully (available=False), never raise.
"""

from __future__ import annotations

import pytest

from tritium_lib.intelligence.mot_eval import MOTScorecard, score_mot

motmetrics = pytest.importorskip(
    "motmetrics",
    reason="py-motmetrics not installed; MOT scoring degrades gracefully",
)


def test_perfect_track_scores_one():
    """GT id 1 tracked by hyp id 100 across 3 frames, always co-located."""
    frames = [
        {"gt_ids": [1], "hyp_ids": [100], "distances": [[0.0]]},
        {"gt_ids": [1], "hyp_ids": [100], "distances": [[0.0]]},
        {"gt_ids": [1], "hyp_ids": [100], "distances": [[0.0]]},
    ]

    card = score_mot(frames)

    assert isinstance(card, MOTScorecard)
    assert card.available is True
    assert card.mota == pytest.approx(1.0, abs=1e-6)
    assert card.idf1 == pytest.approx(1.0, abs=1e-6)
    assert card.num_switches == 0
    assert card.num_fragmentations == 0


def test_injected_id_switch_counts_one():
    """Same GT object, but the hypothesis id flips 100 -> 200 mid-track.
    motmetrics must charge exactly one identity switch."""
    frames = [
        {"gt_ids": [1], "hyp_ids": [100], "distances": [[0.0]]},
        {"gt_ids": [1], "hyp_ids": [100], "distances": [[0.0]]},
        {"gt_ids": [1], "hyp_ids": [200], "distances": [[0.0]]},  # switch
        {"gt_ids": [1], "hyp_ids": [200], "distances": [[0.0]]},
    ]

    card = score_mot(frames)

    assert card.available is True
    assert card.num_switches == 1


def test_misses_lower_mota():
    """Two frames with no hypothesis -> misses -> MOTA below 1.0."""
    frames = [
        {"gt_ids": [1], "hyp_ids": [100], "distances": [[0.0]]},
        {"gt_ids": [1], "hyp_ids": [], "distances": [[]]},
        {"gt_ids": [1], "hyp_ids": [], "distances": [[]]},
    ]

    card = score_mot(frames)

    assert card.available is True
    assert card.mota < 1.0
    assert card.num_misses == 2


def test_false_positive_counted():
    """A hypothesis with no matching GT is a false positive."""
    frames = [
        {"gt_ids": [1], "hyp_ids": [100], "distances": [[0.0]]},
        {
            "gt_ids": [1],
            "hyp_ids": [100, 999],
            "distances": [[0.0, float("nan")], None],
        },
    ]
    # Second frame: gt 1 matches hyp 100 (dist 0), hyp 999 has no gt within gate.
    frames[1]["distances"] = [[0.0, 5.0]]  # 999 too far from gt 1 -> FP

    card = score_mot(frames, max_distance=1.0)

    assert card.available is True
    assert card.num_false_positives >= 1


def test_to_dict_roundtrip():
    frames = [{"gt_ids": [1], "hyp_ids": [100], "distances": [[0.0]]}]
    card = score_mot(frames)
    d = card.as_dict()
    assert d["available"] is True
    assert "mota" in d and "idf1" in d and "num_switches" in d


def test_empty_input_is_available_but_zeroed():
    card = score_mot([])
    # No data is not an error; metrics are simply empty/NaN-safe.
    assert card.available is True
    assert card.num_switches == 0


def test_unavailable_path_when_motmetrics_missing(monkeypatch):
    """Force the import-failure branch and confirm graceful degradation."""
    import tritium_lib.intelligence.mot_eval as mod

    monkeypatch.setattr(mod, "_HAVE_MOTMETRICS", False)
    monkeypatch.setattr(mod, "mm", None)

    frames = [{"gt_ids": [1], "hyp_ids": [100], "distances": [[0.0]]}]
    card = mod.score_mot(frames)

    assert card.available is False
    assert card.reason  # a human-readable explanation is present
    assert card.mota is None
    d = card.as_dict()
    assert d["available"] is False

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""TargetTracker.tactical_brief() — concise live SA grounding for cognition.

`summary()` is combat-oriented (proximity alerts + hostile sectors) and
crucially OMITS the `neutral` alliance, so civil_unrest civilians are
invisible to anything grounded on it.  `tactical_brief()` is a
state-of-the-board inventory that grounds an operator/Amy question
regardless of game state: counts by alliance INCLUDING neutral, plus a
breakdown by target classification (person/vehicle/phone/animal — the
operational mission's target taxonomy).
"""
from __future__ import annotations

import pytest

from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget

pytestmark = pytest.mark.unit


def _add(tracker, tid, alliance, classification="unknown", asset_type="person"):
    """Inject a target directly.  source='manual' is never auto-pruned, so
    the brief is deterministic regardless of wall-clock during the test."""
    t = TrackedTarget(
        target_id=tid, name=tid, alliance=alliance,
        asset_type=asset_type, classification=classification, source="manual",
    )
    with tracker._lock:
        tracker._targets[tid] = t


class TestTacticalBrief:
    def test_empty_tracker_returns_empty_string(self):
        assert TargetTracker().tactical_brief() == ""

    def test_counts_all_alliances_including_neutral(self):
        tr = TargetTracker()
        _add(tr, "f1", "friendly")
        _add(tr, "h1", "hostile")
        _add(tr, "h2", "hostile")
        _add(tr, "n1", "neutral")  # civilian
        _add(tr, "n2", "neutral")
        _add(tr, "n3", "neutral")
        _add(tr, "u1", "unknown")
        brief = tr.tactical_brief()
        assert "7 target(s)" in brief
        assert "1 friendly" in brief
        assert "2 hostile" in brief
        assert "3 neutral" in brief  # the key fix — civilians are visible
        assert "1 unknown" in brief

    def test_neutral_is_grounded_where_combat_summary_drops_it(self):
        """Regression guard: combat summary() never mentions neutral, but the
        grounding brief MUST (civil_unrest civilians spawn neutral)."""
        tr = TargetTracker()
        _add(tr, "civ", "neutral", classification="person")
        assert "neutral" in tr.tactical_brief()
        assert "neutral" not in tr.summary()  # documents the divergence

    def test_classification_breakdown(self):
        tr = TargetTracker()
        _add(tr, "p1", "neutral", classification="person")
        _add(tr, "p2", "neutral", classification="person")
        _add(tr, "v1", "hostile", classification="vehicle")
        _add(tr, "ph", "neutral", classification="phone")
        brief = tr.tactical_brief()
        assert "Types:" in brief
        assert "2 person" in brief
        assert "1 vehicle" in brief
        assert "1 phone" in brief

    def test_falls_back_to_asset_type_when_classification_unknown(self):
        tr = TargetTracker()
        _add(tr, "d1", "friendly", classification="unknown", asset_type="drone")
        assert "1 drone" in tr.tactical_brief()

    def test_no_types_line_when_all_types_unknown(self):
        tr = TargetTracker()
        _add(tr, "x", "hostile", classification="unknown", asset_type="unknown")
        brief = tr.tactical_brief()
        assert "1 hostile" in brief
        assert "Types:" not in brief  # nothing meaningful to say

    def test_works_without_any_game_active(self):
        """tactical_brief is a pure tracker read — no game_mode dependency, so
        it grounds Amy in monitor mode where build_tactical_context returns ''."""
        tr = TargetTracker()
        _add(tr, "x", "hostile")
        assert tr.tactical_brief() != ""

    def test_type_breakdown_is_deterministic_count_desc(self):
        tr = TargetTracker()
        _add(tr, "v1", "hostile", classification="vehicle")
        _add(tr, "v2", "hostile", classification="vehicle")
        _add(tr, "v3", "hostile", classification="vehicle")
        _add(tr, "p1", "neutral", classification="person")
        brief = tr.tactical_brief()
        types_line = [ln for ln in brief.splitlines() if ln.startswith("Types:")][0]
        # vehicle (3) must be listed before person (1)
        assert types_line.index("vehicle") < types_line.index("person")

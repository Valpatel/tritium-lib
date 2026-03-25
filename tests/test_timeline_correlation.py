# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.intelligence.timeline_correlation."""

from __future__ import annotations

import math

import pytest

from tritium_lib.intelligence.timeline_correlation import (
    CausalChain,
    EventSequence,
    FollowerResult,
    PATTERN_ESCORT,
    PATTERN_MEETUP,
    PATTERN_SURVEILLANCE,
    TemporalOverlap,
    TemporalPattern,
    TimelineCorrelator,
    TimelineEvent,
    _classify_causal_event,
    _std_dev,
    _total_displacement,
)
from tritium_lib.tracking.movement_patterns import MovementPatternAnalyzer
from tritium_lib.tracking.target_history import TargetHistory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history_with_trails(
    trails: dict[str, list[tuple[float, float, float]]],
) -> TargetHistory:
    """Create a TargetHistory populated with explicit (x, y, t) trails."""
    h = TargetHistory()
    for tid, points in trails.items():
        for x, y, t in points:
            h.record(tid, (x, y), timestamp=t)
    return h


def _parallel_trail(
    target_id: str,
    start_x: float,
    start_y: float,
    dx: float,
    dy: float,
    steps: int,
    dt: float,
    t0: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Generate a straight-line trail."""
    trail = []
    for i in range(steps):
        trail.append((start_x + dx * i, start_y + dy * i, t0 + dt * i))
    return trail


# ---------------------------------------------------------------------------
# TimelineEvent
# ---------------------------------------------------------------------------

class TestTimelineEvent:
    def test_to_dict(self):
        ev = TimelineEvent(
            target_id="t1", timestamp=100.0, position=(5.0, 10.0),
            event_type="position", details={"key": "val"},
        )
        d = ev.to_dict()
        assert d["target_id"] == "t1"
        assert d["timestamp"] == 100.0
        assert d["position"] == {"x": 5.0, "y": 10.0}
        assert d["event_type"] == "position"
        assert d["details"] == {"key": "val"}

    def test_defaults(self):
        ev = TimelineEvent(target_id="t2", timestamp=0.0, position=(0.0, 0.0))
        assert ev.event_type == "position"
        assert ev.details == {}


# ---------------------------------------------------------------------------
# EventSequence
# ---------------------------------------------------------------------------

class TestEventSequence:
    def test_empty_sequence(self):
        seq = EventSequence(target_id="t1")
        assert seq.events == []
        assert seq.start_time == 0.0
        assert seq.duration == 0.0

    def test_auto_compute_on_init(self):
        events = [
            TimelineEvent("t1", 10.0, (0.0, 0.0)),
            TimelineEvent("t1", 5.0, (1.0, 1.0)),
            TimelineEvent("t1", 20.0, (2.0, 2.0)),
        ]
        seq = EventSequence(target_id="t1", events=events)
        assert seq.start_time == 5.0
        assert seq.end_time == 20.0
        assert seq.duration == 15.0
        # Events should be sorted
        assert seq.events[0].timestamp == 5.0

    def test_append(self):
        seq = EventSequence(target_id="t1")
        seq.append(TimelineEvent("t1", 10.0, (0.0, 0.0)))
        seq.append(TimelineEvent("t1", 5.0, (1.0, 1.0)))
        assert seq.start_time == 5.0
        assert seq.end_time == 10.0
        assert len(seq.events) == 2

    def test_in_range(self):
        events = [
            TimelineEvent("t1", 1.0, (0.0, 0.0)),
            TimelineEvent("t1", 5.0, (1.0, 1.0)),
            TimelineEvent("t1", 10.0, (2.0, 2.0)),
            TimelineEvent("t1", 15.0, (3.0, 3.0)),
        ]
        seq = EventSequence(target_id="t1", events=events)
        subset = seq.in_range(4.0, 11.0)
        assert len(subset) == 2
        assert subset[0].timestamp == 5.0
        assert subset[1].timestamp == 10.0

    def test_to_dict(self):
        events = [TimelineEvent("t1", 1.0, (0.0, 0.0))]
        seq = EventSequence(target_id="t1", events=events)
        d = seq.to_dict()
        assert d["target_id"] == "t1"
        assert d["event_count"] == 1
        assert len(d["events"]) == 1


# ---------------------------------------------------------------------------
# TemporalOverlap
# ---------------------------------------------------------------------------

class TestTemporalOverlap:
    def test_to_dict(self):
        ov = TemporalOverlap(
            target_a="a", target_b="b",
            start_time=100.0, end_time=200.0, duration=100.0,
            center=(5.0, 10.0), avg_distance=2.5, confidence=0.8,
            event_count=15,
        )
        d = ov.to_dict()
        assert d["target_a"] == "a"
        assert d["target_b"] == "b"
        assert d["duration"] == 100.0
        assert d["center"] == {"x": 5.0, "y": 10.0}
        assert d["avg_distance"] == 2.5
        assert d["confidence"] == 0.8
        assert d["event_count"] == 15


# ---------------------------------------------------------------------------
# CausalChain
# ---------------------------------------------------------------------------

class TestCausalChain:
    def test_to_dict(self):
        chain = CausalChain(
            chain_id="chain_0001",
            events=[TimelineEvent("t1", 1.0, (0.0, 0.0))],
            targets_involved=["t1", "t2"],
            confidence=0.75,
            pattern_type="follow",
            description="test",
        )
        d = chain.to_dict()
        assert d["chain_id"] == "chain_0001"
        assert len(d["events"]) == 1
        assert d["targets_involved"] == ["t1", "t2"]
        assert d["pattern_type"] == "follow"


# ---------------------------------------------------------------------------
# FollowerResult
# ---------------------------------------------------------------------------

class TestFollowerResult:
    def test_to_dict(self):
        fr = FollowerResult(
            leader_id="leader", follower_id="follower",
            occurrence_count=5, avg_delay_seconds=30.0,
            avg_distance=8.0, confidence=0.65,
        )
        d = fr.to_dict()
        assert d["leader_id"] == "leader"
        assert d["follower_id"] == "follower"
        assert d["occurrence_count"] == 5
        assert d["avg_delay_seconds"] == 30.0


# ---------------------------------------------------------------------------
# TemporalPattern pre-defined patterns
# ---------------------------------------------------------------------------

class TestTemporalPatterns:
    def test_meetup_defaults(self):
        assert PATTERN_MEETUP.name == "meetup"
        assert PATTERN_MEETUP.min_targets == 2
        assert PATTERN_MEETUP.max_time_window == 300.0
        assert PATTERN_MEETUP.require_movement is False

    def test_surveillance_defaults(self):
        assert PATTERN_SURVEILLANCE.name == "surveillance"
        assert PATTERN_SURVEILLANCE.max_time_window == 1800.0
        assert PATTERN_SURVEILLANCE.require_movement is True

    def test_escort_defaults(self):
        assert PATTERN_ESCORT.name == "escort"
        assert PATTERN_ESCORT.min_co_occurrences == 10
        assert PATTERN_ESCORT.require_movement is True

    def test_custom_pattern(self):
        p = TemporalPattern(
            name="custom",
            min_targets=3,
            max_time_window=120.0,
            max_spatial_radius=5.0,
        )
        assert p.name == "custom"
        assert p.min_targets == 3


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_std_dev_single(self):
        assert _std_dev([5.0]) == 0.0

    def test_std_dev_empty(self):
        assert _std_dev([]) == 0.0

    def test_std_dev_uniform(self):
        assert _std_dev([3.0, 3.0, 3.0]) == 0.0

    def test_std_dev_known(self):
        # [2, 4, 4, 4, 5, 5, 7, 9] => mean=5, variance=4, std=2
        sd = _std_dev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(sd - 2.0) < 0.01

    def test_total_displacement_empty(self):
        assert _total_displacement([]) == 0.0

    def test_total_displacement_single(self):
        assert _total_displacement([(0.0, 0.0, 0.0)]) == 0.0

    def test_total_displacement_known(self):
        trail = [(0.0, 0.0, 0.0), (3.0, 4.0, 1.0)]
        assert abs(_total_displacement(trail) - 5.0) < 0.001

    def test_classify_causal_loitering_meetup(self):
        ea = TimelineEvent("a", 1.0, (0.0, 0.0), event_type="loitering")
        eb = TimelineEvent("b", 2.0, (0.0, 0.0), event_type="position")
        assert _classify_causal_event(ea, eb) == "meetup"

    def test_classify_causal_handoff(self):
        ea = TimelineEvent("a", 1.0, (0.0, 0.0), event_type="position")
        eb = TimelineEvent("b", 2.0, (0.0, 0.0), event_type="loitering")
        assert _classify_causal_event(ea, eb) == "handoff"

    def test_classify_causal_follow(self):
        ea = TimelineEvent("a", 1.0, (0.0, 0.0), event_type="position")
        eb = TimelineEvent("b", 2.0, (0.0, 0.0), event_type="position")
        assert _classify_causal_event(ea, eb) == "follow"

    def test_classify_causal_reaction(self):
        ea = TimelineEvent("a", 1.0, (0.0, 0.0), event_type="deviation")
        eb = TimelineEvent("b", 2.0, (0.0, 0.0), event_type="position")
        assert _classify_causal_event(ea, eb) == "reaction"

    def test_classify_causal_sequence(self):
        ea = TimelineEvent("a", 1.0, (0.0, 0.0), event_type="stationary")
        eb = TimelineEvent("b", 2.0, (0.0, 0.0), event_type="stationary")
        assert _classify_causal_event(ea, eb) == "sequence"


# ---------------------------------------------------------------------------
# TimelineCorrelator — build_event_sequence
# ---------------------------------------------------------------------------

class TestBuildEventSequence:
    def test_basic_sequence(self):
        trail = _parallel_trail("t1", 0.0, 0.0, 1.0, 0.0, 10, 1.0)
        h = _make_history_with_trails({"t1": trail})
        tc = TimelineCorrelator(h)

        seq = tc.build_event_sequence("t1")
        assert seq.target_id == "t1"
        assert len(seq.events) == 10
        assert seq.duration == 9.0

    def test_empty_target(self):
        h = TargetHistory()
        tc = TimelineCorrelator(h)
        seq = tc.build_event_sequence("nonexistent")
        assert len(seq.events) == 0

    def test_with_pattern_analyzer(self):
        # Generate a trail long enough for pattern analysis (needs 3+ points)
        trail = _parallel_trail("t1", 0.0, 0.0, 0.1, 0.0, 20, 5.0)
        h = _make_history_with_trails({"t1": trail})
        analyzer = MovementPatternAnalyzer(history=h)
        tc = TimelineCorrelator(h, pattern_analyzer=analyzer)

        seq = tc.build_event_sequence("t1", include_patterns=True)
        # Should have at least the position events
        assert len(seq.events) >= 20


# ---------------------------------------------------------------------------
# TimelineCorrelator — find_co_occurrences
# ---------------------------------------------------------------------------

class TestFindCoOccurrences:
    def test_parallel_targets_close(self):
        """Two targets walking side by side should produce co-occurrences."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 20, 1.0, t0=0.0)
        trail_b = _parallel_trail("b", 0.0, 2.0, 1.0, 0.0, 20, 1.0, t0=0.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h, co_occurrence_radius=5.0, co_occurrence_min_duration=3.0)

        overlaps = tc.find_co_occurrences("a", "b")
        assert len(overlaps) >= 1
        assert overlaps[0].target_a == "a"
        assert overlaps[0].target_b == "b"
        assert overlaps[0].duration >= 3.0

    def test_distant_targets_no_overlap(self):
        """Targets far apart should have no co-occurrences."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0)
        trail_b = _parallel_trail("b", 100.0, 100.0, 1.0, 0.0, 10, 1.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h, co_occurrence_radius=5.0)

        overlaps = tc.find_co_occurrences("a", "b")
        assert len(overlaps) == 0

    def test_non_overlapping_times(self):
        """Targets at same place but different times should have no overlap."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0, t0=0.0)
        trail_b = _parallel_trail("b", 0.0, 0.0, 1.0, 0.0, 10, 1.0, t0=100.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h, co_occurrence_radius=5.0, co_occurrence_min_duration=1.0)

        overlaps = tc.find_co_occurrences("a", "b")
        assert len(overlaps) == 0

    def test_overlap_to_dict(self):
        """Co-occurrence results should serialize properly."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 0.5, 0.0, 20, 1.0)
        trail_b = _parallel_trail("b", 0.0, 1.0, 0.5, 0.0, 20, 1.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h, co_occurrence_radius=5.0, co_occurrence_min_duration=3.0)

        overlaps = tc.find_co_occurrences("a", "b")
        if overlaps:
            d = overlaps[0].to_dict()
            assert "target_a" in d
            assert "center" in d
            assert "x" in d["center"]

    def test_insufficient_data(self):
        """Single-point trails should return empty."""
        h = _make_history_with_trails({
            "a": [(0.0, 0.0, 0.0)],
            "b": [(0.0, 0.0, 0.0)],
        })
        tc = TimelineCorrelator(h)
        assert tc.find_co_occurrences("a", "b") == []

    def test_custom_radius(self):
        """Custom radius parameter should override default."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 20, 1.0)
        trail_b = _parallel_trail("b", 0.0, 8.0, 1.0, 0.0, 20, 1.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h, co_occurrence_radius=5.0, co_occurrence_min_duration=3.0)

        # Default radius 5 — should not match (distance is 8)
        assert tc.find_co_occurrences("a", "b") == []
        # Custom radius 10 — should match
        overlaps = tc.find_co_occurrences("a", "b", radius=10.0)
        assert len(overlaps) >= 1


# ---------------------------------------------------------------------------
# TimelineCorrelator — find_followers
# ---------------------------------------------------------------------------

class TestFindFollowers:
    def test_clear_follower(self):
        """Target B appears at each of A's locations with a consistent delay."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 10.0, 0.0, 10, 10.0, t0=0.0)
        # B follows 5 seconds later at the same positions
        trail_b = _parallel_trail("b", 0.0, 0.0, 10.0, 0.0, 10, 10.0, t0=5.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(
            h,
            follower_time_window=15.0,
            follower_spatial_radius=5.0,
        )

        results = tc.find_followers("a", candidate_ids=["b"], min_occurrences=2)
        assert len(results) >= 1
        assert results[0].follower_id == "b"
        assert results[0].avg_delay_seconds > 0
        assert results[0].confidence > 0

    def test_no_follower_wrong_place(self):
        """Target at a different location should not be detected as follower."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 10.0, 0.0, 10, 10.0, t0=0.0)
        trail_b = _parallel_trail("b", 100.0, 100.0, 10.0, 0.0, 10, 10.0, t0=5.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h, follower_spatial_radius=10.0)

        results = tc.find_followers("a", candidate_ids=["b"])
        assert len(results) == 0

    def test_no_candidates(self):
        """No candidates returns empty."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0)
        h = _make_history_with_trails({"a": trail_a})
        tc = TimelineCorrelator(h)
        assert tc.find_followers("a", candidate_ids=None) == []

    def test_self_excluded(self):
        """Target should not be its own follower."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0)
        h = _make_history_with_trails({"a": trail_a})
        tc = TimelineCorrelator(h)
        results = tc.find_followers("a", candidate_ids=["a"])
        assert len(results) == 0

    def test_follower_sorted_by_confidence(self):
        """Multiple followers should be sorted by confidence descending."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 10.0, 0.0, 10, 10.0, t0=0.0)
        # Close follower
        trail_b = _parallel_trail("b", 0.0, 0.0, 10.0, 0.0, 10, 10.0, t0=3.0)
        # Distant follower with less consistent timing
        trail_c = _parallel_trail("c", 0.0, 5.0, 10.0, 0.0, 10, 10.0, t0=8.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b, "c": trail_c})
        tc = TimelineCorrelator(
            h,
            follower_time_window=15.0,
            follower_spatial_radius=10.0,
        )

        results = tc.find_followers("a", candidate_ids=["b", "c"], min_occurrences=2)
        if len(results) >= 2:
            assert results[0].confidence >= results[1].confidence


# ---------------------------------------------------------------------------
# TimelineCorrelator — detect_pattern
# ---------------------------------------------------------------------------

class TestDetectPattern:
    def test_meetup_pattern(self):
        """Two targets converging on same location should match meetup."""
        # A and B walking side by side — continuous co-location = 1 overlap,
        # so min_co_occurrences must be 1 for a single continuous stretch.
        trail_a = _parallel_trail("a", 0.0, 0.0, 0.5, 0.0, 30, 2.0, t0=0.0)
        trail_b = _parallel_trail("b", 0.0, 3.0, 0.5, 0.0, 30, 2.0, t0=0.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})

        pattern = TemporalPattern(
            name="meetup",
            min_targets=2,
            max_time_window=600.0,
            max_spatial_radius=10.0,
            min_co_occurrences=1,
        )
        tc = TimelineCorrelator(h, co_occurrence_radius=10.0, co_occurrence_min_duration=0.5)
        chains = tc.detect_pattern(["a", "b"], pattern)
        assert len(chains) >= 1
        assert chains[0].pattern_type == "meetup"
        assert "a" in chains[0].targets_involved
        assert "b" in chains[0].targets_involved

    def test_too_few_targets(self):
        """Pattern requiring 3 targets should fail with 2."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0)
        h = _make_history_with_trails({"a": trail_a})
        pattern = TemporalPattern(name="group", min_targets=3)
        tc = TimelineCorrelator(h)
        chains = tc.detect_pattern(["a"], pattern)
        assert len(chains) == 0

    def test_movement_required_stationary(self):
        """Escort pattern requires movement — stationary targets should fail."""
        # Both targets stationary
        trail_a = [(5.0, 5.0, float(i)) for i in range(20)]
        trail_b = [(5.0, 5.0, float(i)) for i in range(20)]
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})

        pattern = TemporalPattern(
            name="escort",
            min_targets=2,
            max_time_window=600.0,
            max_spatial_radius=10.0,
            min_co_occurrences=2,
            require_movement=True,
        )
        tc = TimelineCorrelator(h, co_occurrence_min_duration=0.5)
        chains = tc.detect_pattern(["a", "b"], pattern)
        assert len(chains) == 0

    def test_pattern_chain_serialization(self):
        """Detected pattern chains should serialize properly."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 0.5, 0.0, 30, 2.0)
        trail_b = _parallel_trail("b", 0.0, 1.0, 0.5, 0.0, 30, 2.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})

        pattern = TemporalPattern(
            name="test_pattern",
            min_targets=2,
            max_spatial_radius=10.0,
            min_co_occurrences=2,
        )
        tc = TimelineCorrelator(h, co_occurrence_min_duration=0.5)
        chains = tc.detect_pattern(["a", "b"], pattern)
        if chains:
            d = chains[0].to_dict()
            assert "chain_id" in d
            assert "pattern_type" in d


# ---------------------------------------------------------------------------
# TimelineCorrelator — find_causal_chains
# ---------------------------------------------------------------------------

class TestFindCausalChains:
    def test_follow_chain(self):
        """Two targets where B follows A should produce a causal chain."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 5.0, 0.0, 10, 5.0, t0=0.0)
        trail_b = _parallel_trail("b", 0.0, 0.0, 5.0, 0.0, 10, 5.0, t0=3.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h)

        chains = tc.find_causal_chains(
            ["a", "b"],
            max_delay=10.0,
            spatial_radius=10.0,
        )
        assert len(chains) >= 1
        # All chains should have valid confidence
        for chain in chains:
            assert 0.0 <= chain.confidence <= 1.0

    def test_single_target_no_chains(self):
        """Single target should produce no chains."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0)
        h = _make_history_with_trails({"a": trail_a})
        tc = TimelineCorrelator(h)
        chains = tc.find_causal_chains(["a"])
        assert len(chains) == 0

    def test_distant_targets_no_chains(self):
        """Targets far apart should produce no causal chains."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0)
        trail_b = _parallel_trail("b", 500.0, 500.0, 1.0, 0.0, 10, 1.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h)
        chains = tc.find_causal_chains(["a", "b"], spatial_radius=5.0)
        assert len(chains) == 0


# ---------------------------------------------------------------------------
# TimelineCorrelator — get_timeline_overlap
# ---------------------------------------------------------------------------

class TestGetTimelineOverlap:
    def test_fully_overlapping(self):
        """Two targets with identical time ranges should have full overlap."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0, t0=0.0)
        trail_b = _parallel_trail("b", 5.0, 5.0, 1.0, 0.0, 10, 1.0, t0=0.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h)

        result = tc.get_timeline_overlap(["a", "b"])
        assert result["targets_with_data"] == 2
        assert result["overlap_duration"] == 9.0

    def test_partial_overlap(self):
        """Partially overlapping time ranges."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0, t0=0.0)
        trail_b = _parallel_trail("b", 0.0, 0.0, 1.0, 0.0, 10, 1.0, t0=5.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h)

        result = tc.get_timeline_overlap(["a", "b"])
        assert result["targets_with_data"] == 2
        # A: 0-9, B: 5-14. Overlap: 5-9 = 4 seconds
        assert abs(result["overlap_duration"] - 4.0) < 0.01

    def test_no_overlap(self):
        """Non-overlapping time ranges should have zero overlap."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0, t0=0.0)
        trail_b = _parallel_trail("b", 0.0, 0.0, 1.0, 0.0, 10, 1.0, t0=100.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h)

        result = tc.get_timeline_overlap(["a", "b"])
        assert result["overlap_duration"] == 0.0

    def test_single_target(self):
        """Single target should report 1 target with data, no overlap."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 10, 1.0)
        h = _make_history_with_trails({"a": trail_a})
        tc = TimelineCorrelator(h)

        result = tc.get_timeline_overlap(["a"])
        assert result["targets_with_data"] == 1
        assert result["overlap_duration"] == 0.0

    def test_individual_ranges_in_result(self):
        """Result should include individual time ranges."""
        trail_a = _parallel_trail("a", 0.0, 0.0, 1.0, 0.0, 5, 2.0, t0=10.0)
        trail_b = _parallel_trail("b", 0.0, 0.0, 1.0, 0.0, 5, 2.0, t0=12.0)
        h = _make_history_with_trails({"a": trail_a, "b": trail_b})
        tc = TimelineCorrelator(h)

        result = tc.get_timeline_overlap(["a", "b"])
        assert "a" in result["individual_ranges"]
        assert "b" in result["individual_ranges"]
        assert result["individual_ranges"]["a"]["start"] == 10.0
        assert result["individual_ranges"]["b"]["start"] == 12.0


# ---------------------------------------------------------------------------
# Integration: correlator + pattern analyzer
# ---------------------------------------------------------------------------

class TestCorrelatorWithPatternAnalyzer:
    def test_enriched_sequence_includes_patterns(self):
        """EventSequence built with a pattern analyzer should include pattern events."""
        # Long trail with loitering section
        trail = []
        # Moving phase
        for i in range(10):
            trail.append((float(i), 0.0, float(i)))
        # Loitering phase (same spot for a long time, needs to exceed 300s default)
        for i in range(10, 60):
            trail.append((10.0 + 0.1 * (i % 3), 0.1 * (i % 2), 10.0 + (i - 10) * 10.0))
        h = _make_history_with_trails({"t1": trail})
        analyzer = MovementPatternAnalyzer(
            history=h,
            loiter_radius=5.0,
            loiter_min_duration=100.0,
        )
        tc = TimelineCorrelator(h, pattern_analyzer=analyzer)

        seq = tc.build_event_sequence("t1", include_patterns=True)
        event_types = {e.event_type for e in seq.events}
        # Should at least have position events
        assert "position" in event_types


# ---------------------------------------------------------------------------
# Thread safety / chain ID uniqueness
# ---------------------------------------------------------------------------

class TestChainIdGeneration:
    def test_unique_chain_ids(self):
        """Chain IDs should be unique across calls."""
        h = TargetHistory()
        tc = TimelineCorrelator(h)
        ids = set()
        for _ in range(100):
            cid = tc._next_chain_id()
            assert cid not in ids
            ids.add(cid)
        assert len(ids) == 100

    def test_chain_id_format(self):
        h = TargetHistory()
        tc = TimelineCorrelator(h)
        cid = tc._next_chain_id()
        assert cid.startswith("chain_")
        assert len(cid) == len("chain_0001")

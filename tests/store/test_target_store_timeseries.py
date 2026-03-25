# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for TargetStore time-series query capabilities.

Covers get_trajectory, get_activity_timeline, get_co_located, and
get_hourly_counts.
"""

import pytest

from tritium_lib.store.targets import TargetStore


@pytest.fixture
def store():
    """Create an in-memory TargetStore for testing."""
    s = TargetStore(":memory:")
    yield s
    s.close()


# ── get_trajectory ─────────────────────────────────────────────────


class TestGetTrajectory:
    """Tests for get_trajectory() — ordered position history in a time window."""

    def test_full_trajectory(self, store):
        """All history returned chronologically when no time bounds given."""
        store.record_sighting("t1", position_x=0.0, position_y=0.0, timestamp=1000.0)
        store.record_sighting("t1", position_x=5.0, position_y=5.0, timestamp=2000.0)
        store.record_sighting("t1", position_x=10.0, position_y=10.0, timestamp=3000.0)
        traj = store.get_trajectory("t1")
        assert len(traj) == 3
        # Oldest first (chronological)
        assert traj[0]["timestamp"] == pytest.approx(1000.0)
        assert traj[1]["timestamp"] == pytest.approx(2000.0)
        assert traj[2]["timestamp"] == pytest.approx(3000.0)

    def test_trajectory_with_start_time(self, store):
        """Only positions after start_time are returned."""
        store.record_sighting("t1", position_x=0.0, position_y=0.0, timestamp=1000.0)
        store.record_sighting("t1", position_x=5.0, position_y=5.0, timestamp=2000.0)
        store.record_sighting("t1", position_x=10.0, position_y=10.0, timestamp=3000.0)
        traj = store.get_trajectory("t1", start_time=1500.0)
        assert len(traj) == 2
        assert traj[0]["timestamp"] == pytest.approx(2000.0)

    def test_trajectory_with_end_time(self, store):
        """Only positions before end_time are returned."""
        store.record_sighting("t1", position_x=0.0, position_y=0.0, timestamp=1000.0)
        store.record_sighting("t1", position_x=5.0, position_y=5.0, timestamp=2000.0)
        store.record_sighting("t1", position_x=10.0, position_y=10.0, timestamp=3000.0)
        traj = store.get_trajectory("t1", end_time=2500.0)
        assert len(traj) == 2
        assert traj[-1]["timestamp"] == pytest.approx(2000.0)

    def test_trajectory_with_both_bounds(self, store):
        """Time window filters to the middle segment."""
        for i in range(5):
            store.record_sighting(
                "t1",
                position_x=float(i),
                position_y=float(i),
                timestamp=1000.0 + i * 1000,
            )
        traj = store.get_trajectory("t1", start_time=2000.0, end_time=4000.0)
        assert len(traj) == 3
        assert traj[0]["x"] == pytest.approx(1.0)
        assert traj[-1]["x"] == pytest.approx(3.0)

    def test_trajectory_empty_for_unknown_target(self, store):
        """Unknown target returns empty list."""
        assert store.get_trajectory("nonexistent") == []

    def test_trajectory_includes_source(self, store):
        """Source field is included in each trajectory point."""
        store.record_sighting("t1", position_x=1.0, position_y=2.0, source="ble", timestamp=1000.0)
        store.record_sighting("t1", position_x=3.0, position_y=4.0, source="wifi", timestamp=2000.0)
        traj = store.get_trajectory("t1")
        assert traj[0]["source"] == "ble"
        assert traj[1]["source"] == "wifi"

    def test_trajectory_no_history_positions(self, store):
        """Target with no position data returns empty trajectory."""
        store.record_sighting("t1", name="No Position")
        assert store.get_trajectory("t1") == []


# ── get_activity_timeline ──────────────────────────────────────────


class TestGetActivityTimeline:
    """Tests for get_activity_timeline() — target activity summary."""

    def test_basic_timeline(self, store):
        """Activity timeline has expected fields and values."""
        store.record_sighting("t1", position_x=0.0, position_y=0.0, source="ble", timestamp=1000.0)
        store.record_sighting("t1", position_x=10.0, position_y=10.0, source="wifi", timestamp=2000.0)
        timeline = store.get_activity_timeline("t1")
        assert timeline is not None
        assert timeline["target_id"] == "t1"
        assert timeline["first_seen"] == pytest.approx(1000.0)
        assert timeline["last_seen"] == pytest.approx(2000.0)
        assert timeline["total_time"] == pytest.approx(1000.0)
        assert timeline["sighting_count"] == 2

    def test_timeline_nonexistent_target(self, store):
        """Returns None for unknown target."""
        assert store.get_activity_timeline("fake") is None

    def test_timeline_zones_visited(self, store):
        """Zones are grid-snapped distinct positions."""
        # Two sightings near (10.3, 20.7) should snap to same zone (10, 20)
        store.record_sighting("t1", position_x=10.3, position_y=20.7, timestamp=1000.0)
        store.record_sighting("t1", position_x=10.8, position_y=20.2, timestamp=2000.0)
        # One sighting at a different zone
        store.record_sighting("t1", position_x=50.0, position_y=60.0, timestamp=3000.0)
        timeline = store.get_activity_timeline("t1")
        assert len(timeline["zones_visited"]) == 2

    def test_timeline_sources_used(self, store):
        """Distinct source strings are collected."""
        store.record_sighting("t1", position_x=0.0, position_y=0.0, source="ble", timestamp=1000.0)
        store.record_sighting("t1", position_x=1.0, position_y=1.0, source="wifi", timestamp=2000.0)
        store.record_sighting("t1", position_x=2.0, position_y=2.0, source="ble", timestamp=3000.0)
        timeline = store.get_activity_timeline("t1")
        assert sorted(timeline["sources_used"]) == ["ble", "wifi"]

    def test_timeline_no_history(self, store):
        """Target with no position history has zero sighting_count and empty zones."""
        store.record_sighting("t1", name="Ghost")
        timeline = store.get_activity_timeline("t1")
        assert timeline is not None
        assert timeline["sighting_count"] == 0
        assert timeline["zones_visited"] == []
        assert timeline["total_time"] == pytest.approx(0.0)

    def test_timeline_single_sighting(self, store):
        """Single-sighting target has total_time of zero."""
        store.record_sighting("t1", position_x=5.0, position_y=5.0, source="cam", timestamp=1000.0)
        timeline = store.get_activity_timeline("t1")
        assert timeline["total_time"] == pytest.approx(0.0)
        assert timeline["sighting_count"] == 1
        assert len(timeline["zones_visited"]) == 1


# ── get_co_located ─────────────────────────────────────────────────


class TestGetCoLocated:
    """Tests for get_co_located() — finding nearby targets."""

    def test_basic_co_location(self, store):
        """Two targets at same position and time are co-located."""
        store.record_sighting("ref", position_x=10.0, position_y=10.0, timestamp=1000.0)
        store.record_sighting("near", position_x=11.0, position_y=10.0, timestamp=1000.0)
        results = store.get_co_located("ref", radius=5.0, time_window=60.0)
        assert len(results) == 1
        assert results[0]["target_id"] == "near"
        assert results[0]["encounter_count"] == 1
        assert results[0]["min_distance"] == pytest.approx(1.0)

    def test_co_located_outside_radius(self, store):
        """Target beyond radius is not returned."""
        store.record_sighting("ref", position_x=0.0, position_y=0.0, timestamp=1000.0)
        store.record_sighting("far", position_x=100.0, position_y=100.0, timestamp=1000.0)
        results = store.get_co_located("ref", radius=5.0, time_window=60.0)
        assert len(results) == 0

    def test_co_located_outside_time_window(self, store):
        """Target nearby but at a different time is not returned."""
        store.record_sighting("ref", position_x=10.0, position_y=10.0, timestamp=1000.0)
        store.record_sighting("late", position_x=10.0, position_y=10.0, timestamp=9000.0)
        results = store.get_co_located("ref", radius=5.0, time_window=60.0)
        assert len(results) == 0

    def test_co_located_multiple_encounters(self, store):
        """Multiple co-location events are counted and tracked."""
        store.record_sighting("ref", position_x=10.0, position_y=10.0, timestamp=1000.0)
        store.record_sighting("ref", position_x=20.0, position_y=20.0, timestamp=2000.0)
        store.record_sighting("buddy", position_x=11.0, position_y=10.0, timestamp=1000.0)
        store.record_sighting("buddy", position_x=21.0, position_y=20.0, timestamp=2000.0)
        results = store.get_co_located("ref", radius=5.0, time_window=60.0)
        assert len(results) == 1
        assert results[0]["encounter_count"] == 2
        assert results[0]["first_encounter"] == pytest.approx(1000.0)
        assert results[0]["last_encounter"] == pytest.approx(2000.0)

    def test_co_located_sorted_by_encounters(self, store):
        """Results sorted by encounter_count descending."""
        store.record_sighting("ref", position_x=10.0, position_y=10.0, timestamp=1000.0)
        store.record_sighting("ref", position_x=10.0, position_y=10.0, timestamp=2000.0)
        # 'frequent' appears at both times
        store.record_sighting("frequent", position_x=11.0, position_y=10.0, timestamp=1000.0)
        store.record_sighting("frequent", position_x=11.0, position_y=10.0, timestamp=2000.0)
        # 'rare' appears only once
        store.record_sighting("rare", position_x=10.0, position_y=11.0, timestamp=1000.0)
        results = store.get_co_located("ref", radius=5.0, time_window=60.0)
        assert len(results) == 2
        assert results[0]["target_id"] == "frequent"
        assert results[0]["encounter_count"] == 2
        assert results[1]["target_id"] == "rare"
        assert results[1]["encounter_count"] == 1

    def test_co_located_empty_history(self, store):
        """Target with no history returns empty results."""
        store.record_sighting("ref", name="No Position")
        results = store.get_co_located("ref", radius=5.0, time_window=60.0)
        assert results == []

    def test_co_located_euclidean_not_manhattan(self, store):
        """Diagonal distance correctly uses Euclidean metric, not Manhattan."""
        # Distance from (0,0) to (3,4) = 5.0 exactly
        store.record_sighting("ref", position_x=0.0, position_y=0.0, timestamp=1000.0)
        store.record_sighting("edge", position_x=3.0, position_y=4.0, timestamp=1000.0)
        # Radius exactly 5.0 should include it
        results = store.get_co_located("ref", radius=5.0, time_window=60.0)
        assert len(results) == 1
        assert results[0]["min_distance"] == pytest.approx(5.0)
        # Radius just under 5.0 should exclude it
        results = store.get_co_located("ref", radius=4.9, time_window=60.0)
        assert len(results) == 0

    def test_co_located_min_distance_tracked(self, store):
        """min_distance reflects the closest encounter, not the last."""
        store.record_sighting("ref", position_x=10.0, position_y=10.0, timestamp=1000.0)
        store.record_sighting("ref", position_x=10.0, position_y=10.0, timestamp=2000.0)
        # First encounter: distance 3.0
        store.record_sighting("buddy", position_x=13.0, position_y=10.0, timestamp=1000.0)
        # Second encounter: distance 1.0
        store.record_sighting("buddy", position_x=11.0, position_y=10.0, timestamp=2000.0)
        results = store.get_co_located("ref", radius=5.0, time_window=60.0)
        assert results[0]["min_distance"] == pytest.approx(1.0)


# ── get_hourly_counts ──────────────────────────────────────────────


class TestGetHourlyCounts:
    """Tests for get_hourly_counts() — target counts per hour bucket."""

    def test_single_hour_bucket(self, store):
        """Sightings in the same hour produce one bucket."""
        # All within the same hour (3600-second bucket)
        base = 3600.0 * 100  # an arbitrary hour-aligned start
        store.record_sighting("t1", position_x=0.0, position_y=0.0, timestamp=base + 100)
        store.record_sighting("t2", position_x=1.0, position_y=1.0, timestamp=base + 200)
        counts = store.get_hourly_counts()
        assert len(counts) == 1
        assert counts[0]["count"] == 2
        assert counts[0]["hour_start"] == base

    def test_multiple_hour_buckets(self, store):
        """Sightings in different hours produce separate buckets."""
        base = 3600.0 * 100
        store.record_sighting("t1", position_x=0.0, position_y=0.0, timestamp=base + 100)
        store.record_sighting("t2", position_x=1.0, position_y=1.0, timestamp=base + 3700)
        counts = store.get_hourly_counts()
        assert len(counts) == 2
        assert counts[0]["hour_start"] < counts[1]["hour_start"]

    def test_hourly_counts_distinct_targets(self, store):
        """Same target seen multiple times in an hour counts as 1."""
        base = 3600.0 * 100
        store.record_sighting("t1", position_x=0.0, position_y=0.0, timestamp=base + 100)
        store.record_sighting("t1", position_x=1.0, position_y=1.0, timestamp=base + 200)
        store.record_sighting("t1", position_x=2.0, position_y=2.0, timestamp=base + 300)
        counts = store.get_hourly_counts()
        assert len(counts) == 1
        assert counts[0]["count"] == 1  # one distinct target

    def test_hourly_counts_with_start_time(self, store):
        """start_time filters out earlier buckets."""
        base = 3600.0 * 100
        store.record_sighting("t1", position_x=0.0, position_y=0.0, timestamp=base + 100)
        store.record_sighting("t2", position_x=1.0, position_y=1.0, timestamp=base + 3700)
        counts = store.get_hourly_counts(start_time=base + 3600)
        assert len(counts) == 1
        assert counts[0]["hour_start"] == base + 3600

    def test_hourly_counts_with_end_time(self, store):
        """end_time filters out later buckets."""
        base = 3600.0 * 100
        store.record_sighting("t1", position_x=0.0, position_y=0.0, timestamp=base + 100)
        store.record_sighting("t2", position_x=1.0, position_y=1.0, timestamp=base + 3700)
        counts = store.get_hourly_counts(end_time=base + 3599)
        assert len(counts) == 1
        assert counts[0]["hour_start"] == base

    def test_hourly_counts_empty(self, store):
        """No history produces no buckets."""
        assert store.get_hourly_counts() == []

    def test_hourly_counts_chronological_order(self, store):
        """Buckets are returned in chronological order."""
        base = 3600.0 * 100
        # Insert out of order
        store.record_sighting("t1", position_x=0.0, position_y=0.0, timestamp=base + 7300)
        store.record_sighting("t2", position_x=1.0, position_y=1.0, timestamp=base + 100)
        store.record_sighting("t3", position_x=2.0, position_y=2.0, timestamp=base + 3700)
        counts = store.get_hourly_counts()
        assert len(counts) == 3
        assert counts[0]["hour_start"] < counts[1]["hour_start"] < counts[2]["hour_start"]

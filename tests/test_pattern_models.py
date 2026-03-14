# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for behavioral pattern learning models (models/pattern.py)."""

import time
from datetime import datetime, timezone

import pytest

from tritium_lib.models.pattern import (
    BehaviorPattern,
    CoPresenceRelationship,
    DeviationType,
    LocationCluster,
    PatternAlert,
    PatternAnomaly,
    PatternStatus,
    PatternType,
    TimeSlot,
    compute_temporal_correlation,
    detect_time_regularity,
)


class TestTimeSlot:
    def test_contains_time_basic(self):
        slot = TimeSlot(hour_start=8, hour_end=10, minute_start=0, minute_end=0,
                        days_of_week=[0, 1, 2, 3, 4])
        # Monday 9:00
        dt = datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc)  # Monday
        assert slot.contains_time(dt)

    def test_rejects_wrong_day(self):
        slot = TimeSlot(hour_start=8, hour_end=10, days_of_week=[0, 1, 2, 3, 4])
        # Saturday
        dt = datetime(2026, 3, 14, 9, 0, tzinfo=timezone.utc)  # Saturday
        assert not slot.contains_time(dt)

    def test_rejects_wrong_time(self):
        slot = TimeSlot(hour_start=8, hour_end=10, days_of_week=list(range(7)))
        dt = datetime(2026, 3, 16, 15, 0, tzinfo=timezone.utc)
        assert not slot.contains_time(dt)


class TestBehaviorPattern:
    def test_create_pattern(self):
        p = BehaviorPattern(
            pattern_id="pat_test",
            target_id="ble_aabbccdd",
            pattern_type=PatternType.DAILY_ROUTINE,
            confidence=0.5,
        )
        assert p.pattern_id == "pat_test"
        assert p.pattern_type == PatternType.DAILY_ROUTINE
        assert not p.is_established

    def test_reinforce_increases_confidence(self):
        p = BehaviorPattern(
            pattern_id="pat_test",
            target_id="ble_aabbccdd",
            confidence=0.3,
            observation_count=3,
        )
        p.reinforce()
        assert p.confidence == 0.35
        assert p.observation_count == 4

    def test_becomes_established(self):
        p = BehaviorPattern(
            pattern_id="pat_test",
            target_id="ble_aabbccdd",
            confidence=0.68,
            observation_count=4,
        )
        p.reinforce()  # conf -> 0.73, obs -> 5
        assert p.is_established
        assert p.status == PatternStatus.ESTABLISHED

    def test_age_days(self):
        p = BehaviorPattern(
            pattern_id="pat_test",
            target_id="ble_x",
            first_seen=time.time() - 3 * 86400,
        )
        assert 2.9 < p.age_days < 3.1


class TestPatternAnomaly:
    def test_create_anomaly(self):
        a = PatternAnomaly(
            anomaly_id="anom_test",
            target_id="ble_aabbccdd",
            pattern_id="pat_test",
            deviation_type=DeviationType.MISSING,
            deviation_score=0.8,
            expected_behavior="Expected at Zone A 8-9am",
            actual_behavior="Not seen",
        )
        assert a.deviation_type == DeviationType.MISSING
        assert a.deviation_score == 0.8
        assert not a.acknowledged


class TestCoPresenceRelationship:
    def test_compute_confidence(self):
        rel = CoPresenceRelationship(
            target_a="ble_a",
            target_b="ble_b",
            temporal_correlation=0.9,
            spatial_correlation=0.8,
            co_occurrence_count=25,
            total_observations=30,
        )
        conf = rel.compute_confidence()
        assert conf > 0.7
        assert rel.confidence == conf

    def test_low_observations_zero_confidence(self):
        rel = CoPresenceRelationship(
            target_a="ble_a",
            target_b="ble_b",
            temporal_correlation=1.0,
            total_observations=2,
        )
        conf = rel.compute_confidence()
        assert conf == 0.0


class TestPatternAlert:
    def test_can_fire(self):
        alert = PatternAlert(
            alert_id="palert_test",
            pattern_id="pat_test",
            target_id="ble_x",
            enabled=True,
            cooldown_seconds=60.0,
            last_fired=0.0,
        )
        assert alert.can_fire()

    def test_cooldown_blocks_fire(self):
        alert = PatternAlert(
            alert_id="palert_test",
            pattern_id="pat_test",
            enabled=True,
            cooldown_seconds=3600.0,
            last_fired=time.time(),
        )
        assert not alert.can_fire()

    def test_disabled_cannot_fire(self):
        alert = PatternAlert(
            alert_id="palert_test",
            pattern_id="pat_test",
            enabled=False,
        )
        assert not alert.can_fire()

    def test_fire_increments_count(self):
        alert = PatternAlert(
            alert_id="palert_test",
            pattern_id="pat_test",
            enabled=True,
            cooldown_seconds=0,
        )
        alert.fire()
        assert alert.fire_count == 1
        assert alert.last_fired > 0


class TestComputeTemporalCorrelation:
    def test_perfect_correlation(self):
        base = time.time()
        times_a = [base + i * 60 for i in range(10)]
        times_b = [base + i * 60 + 5 for i in range(10)]  # 5s offset
        corr = compute_temporal_correlation(times_a, times_b, window_s=60.0)
        assert corr >= 0.9

    def test_no_correlation(self):
        base = time.time()
        times_a = [base + i * 60 for i in range(10)]
        times_b = [base + 100000 + i * 60 for i in range(10)]  # far apart
        corr = compute_temporal_correlation(times_a, times_b, window_s=60.0)
        assert corr == 0.0

    def test_empty_inputs(self):
        assert compute_temporal_correlation([], [1.0, 2.0]) == 0.0
        assert compute_temporal_correlation([1.0], []) == 0.0


class TestDetectTimeRegularity:
    def test_regular_times(self):
        # Generate timestamps all around 9:00 AM UTC
        base_date = datetime(2026, 3, 9, 9, 0, tzinfo=timezone.utc)  # Monday
        timestamps = []
        for day_offset in range(5):
            dt = base_date.replace(day=9 + day_offset, minute=day_offset * 3)
            timestamps.append(dt.timestamp())

        slot = detect_time_regularity(timestamps, tolerance_minutes=30)
        assert slot is not None
        assert 8 <= slot.hour_start <= 9
        assert 9 <= slot.hour_end <= 10

    def test_irregular_times_returns_none(self):
        # Timestamps spread across different hours
        base = datetime(2026, 3, 9, tzinfo=timezone.utc)
        timestamps = [
            base.replace(hour=3).timestamp(),
            base.replace(hour=9).timestamp(),
            base.replace(hour=18).timestamp(),
            base.replace(hour=23).timestamp(),
        ]
        slot = detect_time_regularity(timestamps, tolerance_minutes=30)
        assert slot is None

    def test_too_few_timestamps(self):
        assert detect_time_regularity([1.0, 2.0]) is None


class TestLocationCluster:
    def test_create(self):
        c = LocationCluster(
            center_lat=40.7128,
            center_lng=-74.0060,
            radius_m=100.0,
            visit_count=10,
        )
        assert c.visit_count == 10


class TestModelSerialization:
    def test_behavior_pattern_roundtrip(self):
        p = BehaviorPattern(
            pattern_id="pat_test",
            target_id="ble_x",
            pattern_type=PatternType.COMMUTE,
            confidence=0.85,
        )
        d = p.model_dump()
        p2 = BehaviorPattern(**d)
        assert p2.pattern_id == p.pattern_id
        assert p2.pattern_type == PatternType.COMMUTE

    def test_anomaly_roundtrip(self):
        a = PatternAnomaly(
            anomaly_id="anom_test",
            target_id="ble_x",
            deviation_type=DeviationType.LATE,
            deviation_score=0.6,
        )
        d = a.model_dump()
        a2 = PatternAnomaly(**d)
        assert a2.deviation_type == DeviationType.LATE

    def test_co_presence_roundtrip(self):
        r = CoPresenceRelationship(
            target_a="ble_a",
            target_b="ble_b",
            temporal_correlation=0.9,
        )
        d = r.model_dump()
        r2 = CoPresenceRelationship(**d)
        assert r2.temporal_correlation == 0.9

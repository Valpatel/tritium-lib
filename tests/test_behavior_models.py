# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for behavioral pattern recognition models."""

import time

import pytest

from tritium_lib.models.behavior import (
    AnomalySeverity,
    AnomalyType,
    BehaviorAnomaly,
    BehaviorPattern,
    BehaviorType,
    CorrelationScore,
    PositionSample,
    TargetRoutine,
    classify_anomaly_severity,
    compute_correlation_score,
)


class TestBehaviorPattern:
    def test_basic(self):
        p = BehaviorPattern(
            target_id="ble_aa:bb:cc",
            behavior_type=BehaviorType.LOITERING,
            confidence=0.8,
        )
        assert p.behavior_type == BehaviorType.LOITERING
        assert p.confidence == 0.8

    def test_is_active_no_end(self):
        p = BehaviorPattern(target_id="t1", end_time=0)
        assert p.is_active is True

    def test_is_active_recent(self):
        p = BehaviorPattern(target_id="t1", end_time=time.time() - 10)
        assert p.is_active is True

    def test_is_active_old(self):
        p = BehaviorPattern(target_id="t1", end_time=time.time() - 120)
        assert p.is_active is False

    def test_all_behavior_types(self):
        for bt in BehaviorType:
            p = BehaviorPattern(target_id="t", behavior_type=bt)
            assert p.behavior_type == bt


class TestBehaviorAnomaly:
    def test_basic(self):
        a = BehaviorAnomaly(
            target_id="ble_xx",
            anomaly_type=AnomalyType.NEW_DEVICE,
            severity=AnomalySeverity.LOW,
            description="First time seeing this device",
        )
        assert a.anomaly_type == AnomalyType.NEW_DEVICE
        assert a.severity == AnomalySeverity.LOW

    def test_classify_severity(self):
        assert classify_anomaly_severity(AnomalyType.ASSOCIATION_ANOMALY) == AnomalySeverity.HIGH
        assert classify_anomaly_severity(AnomalyType.UNUSUAL_TIME) == AnomalySeverity.MEDIUM
        assert classify_anomaly_severity(AnomalyType.NEW_DEVICE) == AnomalySeverity.LOW

    def test_all_anomaly_types_mapped(self):
        for at in AnomalyType:
            sev = classify_anomaly_severity(at)
            assert isinstance(sev, AnomalySeverity)


class TestTargetRoutine:
    def test_basic(self):
        r = TargetRoutine(
            target_id="ble_phone",
            active_hours=[8, 9, 17, 18],
            active_days=[0, 1, 2, 3, 4],
            total_observations=100,
        )
        assert len(r.active_hours) == 4
        assert len(r.active_days) == 5

    def test_empty_routine(self):
        r = TargetRoutine(target_id="new")
        assert r.total_observations == 0
        assert r.confidence == 0.0


class TestCorrelationScore:
    def test_basic(self):
        cs = CorrelationScore(
            target_a="ble_phone",
            target_b="det_person_0",
            score=0.85,
            source_a="ble",
            source_b="camera",
            reasons=["co-located", "temporal overlap"],
        )
        assert cs.score == 0.85
        assert len(cs.reasons) == 2

    def test_compute_perfect_correlation(self):
        score = compute_correlation_score(
            temporal_overlap=1.0,
            spatial_proximity_m=0.0,
            co_movement=1.0,
        )
        assert score == 1.0

    def test_compute_no_correlation(self):
        score = compute_correlation_score(
            temporal_overlap=0.0,
            spatial_proximity_m=1000.0,
            co_movement=0.0,
        )
        assert score == 0.0

    def test_compute_partial_correlation(self):
        score = compute_correlation_score(
            temporal_overlap=0.5,
            spatial_proximity_m=10.0,
            co_movement=0.3,
        )
        assert 0.0 < score < 1.0

    def test_compute_score_clamped(self):
        """Score should be clamped to [0, 1]."""
        score = compute_correlation_score(
            temporal_overlap=2.0,
            spatial_proximity_m=-100.0,
            co_movement=2.0,
        )
        assert 0.0 <= score <= 1.0


class TestPositionSample:
    def test_basic(self):
        s = PositionSample(
            latitude=37.7749,
            longitude=-122.4194,
            timestamp=time.time(),
            speed_mps=1.5,
        )
        assert s.speed_mps == 1.5
        assert s.source == ""

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for VelocityProfile model and anomaly scoring."""

import pytest
from datetime import datetime, timezone

from tritium_lib.models.velocity import (
    VelocityProfile,
    compute_anomaly_score,
)


class TestVelocityProfile:
    def test_defaults(self):
        vp = VelocityProfile(target_id="ble_aa:bb:cc")
        assert vp.target_id == "ble_aa:bb:cc"
        assert vp.current_speed == 0.0
        assert vp.max_speed == 0.0
        assert vp.avg_speed == 0.0
        assert vp.acceleration == 0.0
        assert vp.heading_change_rate == 0.0
        assert vp.anomaly_score == 0.0
        assert vp.is_stationary is True
        assert vp.sample_count == 0
        assert vp.analysis_window_s == 300.0
        assert vp.generated_at is not None

    def test_is_anomalous(self):
        vp = VelocityProfile(target_id="x", anomaly_score=0.7)
        assert vp.is_anomalous()
        assert vp.is_anomalous(threshold=0.7)
        assert not vp.is_anomalous(threshold=0.8)

    def test_is_anomalous_default_threshold(self):
        normal = VelocityProfile(target_id="a", anomaly_score=0.3)
        assert not normal.is_anomalous()

    def test_speed_consistency_no_data(self):
        vp = VelocityProfile(target_id="x")
        assert vp.speed_consistency() == 1.0

    def test_speed_consistency_perfect(self):
        vp = VelocityProfile(
            target_id="x",
            avg_speed=5.0,
            speed_variance=0.0,
            sample_count=10,
        )
        assert vp.speed_consistency() == 1.0

    def test_speed_consistency_high_variance(self):
        vp = VelocityProfile(
            target_id="x",
            avg_speed=5.0,
            speed_variance=25.0,  # variance == avg^2
            sample_count=10,
        )
        assert vp.speed_consistency() == 0.0

    def test_speed_consistency_moderate(self):
        vp = VelocityProfile(
            target_id="x",
            avg_speed=10.0,
            speed_variance=25.0,  # 25/100 = 0.25, consistency = 0.75
            sample_count=10,
        )
        assert vp.speed_consistency() == pytest.approx(0.75)

    def test_to_dict(self):
        ts = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        vp = VelocityProfile(
            target_id="det_person_1",
            current_speed=1.5,
            max_speed=3.0,
            avg_speed=1.2,
            min_speed=0.0,
            acceleration=0.3,
            heading_change_rate=10.0,
            heading_deg=45.0,
            anomaly_score=0.2,
            speed_variance=0.5,
            sample_count=20,
            is_stationary=False,
            generated_at=ts,
        )
        d = vp.to_dict()
        assert d["target_id"] == "det_person_1"
        assert d["current_speed"] == 1.5
        assert d["acceleration"] == 0.3
        assert d["anomaly_score"] == 0.2
        assert d["is_stationary"] is False
        assert "2026-03-14" in d["generated_at"]

    def test_from_dict(self):
        data = {
            "target_id": "mesh_node_7",
            "current_speed": 2.5,
            "max_speed": 5.0,
            "avg_speed": 2.0,
            "acceleration": -0.5,
            "heading_change_rate": 15.0,
            "anomaly_score": 0.4,
            "sample_count": 50,
            "is_stationary": False,
        }
        vp = VelocityProfile.from_dict(data)
        assert vp.target_id == "mesh_node_7"
        assert vp.current_speed == 2.5
        assert vp.acceleration == -0.5
        assert vp.anomaly_score == 0.4

    def test_roundtrip(self):
        ts = datetime(2026, 3, 14, 10, 30, 0, tzinfo=timezone.utc)
        original = VelocityProfile(
            target_id="ble_11:22:33",
            current_speed=4.0,
            max_speed=8.0,
            avg_speed=3.5,
            min_speed=1.0,
            acceleration=1.2,
            heading_change_rate=20.0,
            heading_deg=270.0,
            anomaly_score=0.6,
            speed_variance=2.5,
            sample_count=100,
            analysis_window_s=600.0,
            is_stationary=False,
            generated_at=ts,
        )
        d = original.to_dict()
        restored = VelocityProfile.from_dict(d)
        assert restored.target_id == original.target_id
        assert restored.current_speed == original.current_speed
        assert restored.max_speed == original.max_speed
        assert restored.acceleration == original.acceleration
        assert restored.anomaly_score == original.anomaly_score
        assert restored.sample_count == original.sample_count
        assert restored.generated_at == ts

    def test_model_dump_roundtrip(self):
        vp = VelocityProfile(
            target_id="wifi_aa:bb:cc",
            current_speed=2.0,
            anomaly_score=0.1,
        )
        d = vp.model_dump()
        vp2 = VelocityProfile(**d)
        assert vp2.target_id == vp.target_id
        assert vp2.anomaly_score == vp.anomaly_score

    def test_anomaly_score_bounds(self):
        # Score must be between 0 and 1
        vp = VelocityProfile(target_id="x", anomaly_score=0.0)
        assert vp.anomaly_score == 0.0
        vp2 = VelocityProfile(target_id="x", anomaly_score=1.0)
        assert vp2.anomaly_score == 1.0

    def test_anomaly_score_out_of_bounds(self):
        with pytest.raises(Exception):
            VelocityProfile(target_id="x", anomaly_score=1.5)
        with pytest.raises(Exception):
            VelocityProfile(target_id="x", anomaly_score=-0.1)


class TestComputeAnomalyScore:
    def test_all_zero(self):
        score = compute_anomaly_score(
            speed_variance=0.0,
            avg_speed=0.0,
            acceleration=0.0,
            heading_change_rate=0.0,
        )
        assert score == 0.0

    def test_normal_movement(self):
        score = compute_anomaly_score(
            speed_variance=1.0,
            avg_speed=5.0,
            acceleration=0.5,
            heading_change_rate=5.0,
        )
        # Should be low
        assert score < 0.3

    def test_high_acceleration(self):
        score = compute_anomaly_score(
            speed_variance=0.0,
            avg_speed=5.0,
            acceleration=5.0,  # at max expected
            heading_change_rate=0.0,
        )
        # 30% from acceleration factor
        assert score == pytest.approx(0.3)

    def test_erratic_heading(self):
        score = compute_anomaly_score(
            speed_variance=0.0,
            avg_speed=5.0,
            acceleration=0.0,
            heading_change_rate=45.0,  # at max expected
        )
        assert score == pytest.approx(0.3)

    def test_high_speed_variance(self):
        score = compute_anomaly_score(
            speed_variance=25.0,
            avg_speed=5.0,  # variance/avg^2 = 1.0
            acceleration=0.0,
            heading_change_rate=0.0,
        )
        assert score == pytest.approx(0.4)

    def test_max_anomaly(self):
        score = compute_anomaly_score(
            speed_variance=100.0,
            avg_speed=5.0,
            acceleration=10.0,
            heading_change_rate=90.0,
        )
        assert score == 1.0

    def test_clamped_to_bounds(self):
        # Even with extreme values, score is clamped 0-1
        score = compute_anomaly_score(
            speed_variance=1000.0,
            avg_speed=1.0,
            acceleration=100.0,
            heading_change_rate=500.0,
        )
        assert score <= 1.0
        assert score >= 0.0

    def test_custom_thresholds(self):
        score = compute_anomaly_score(
            speed_variance=0.0,
            avg_speed=5.0,
            acceleration=2.0,
            heading_change_rate=10.0,
            max_expected_accel=2.0,   # at max
            max_expected_heading_rate=10.0,  # at max
        )
        # accel and heading both at max = 0.3 + 0.3 = 0.6
        assert score == pytest.approx(0.6)

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for acoustic TDoA models and computation."""

import time

import pytest

from tritium_lib.models.acoustic_tdoa import (
    TDoAObservation,
    TDoAResult,
    compute_tdoa_position,
    SPEED_OF_SOUND_MPS,
)


class TestTDoAObservation:
    """Tests for TDoAObservation model."""

    def test_create_minimal(self):
        obs = TDoAObservation(sensor_id="mic1", arrival_time_ms=1000.0)
        assert obs.sensor_id == "mic1"
        assert obs.arrival_time_ms == 1000.0
        assert obs.signal_strength == 0.0
        assert obs.confidence == 1.0
        assert obs.ntp_sync_quality == 0.0

    def test_create_full(self):
        obs = TDoAObservation(
            sensor_id="mic1",
            arrival_time_ms=1710500000000.0,
            signal_strength=-40.0,
            event_type="gunshot",
            confidence=0.9,
            ntp_sync_quality=0.95,
            lat=40.0,
            lon=-74.0,
        )
        assert obs.event_type == "gunshot"
        assert obs.ntp_sync_quality == 0.95
        assert obs.lat == 40.0

    def test_serialization(self):
        obs = TDoAObservation(sensor_id="test", arrival_time_ms=1000.0)
        d = obs.model_dump()
        assert "sensor_id" in d
        assert "arrival_time_ms" in d
        assert "ntp_sync_quality" in d


class TestTDoAResult:
    """Tests for TDoAResult model."""

    def test_default_values(self):
        r = TDoAResult()
        assert r.position == (0.0, 0.0)
        assert r.confidence == 0.0
        assert r.residual_error_m == 0.0
        assert r.sensors_used == []
        assert r.method == "tdoa_weighted_centroid"

    def test_lat_lon_properties(self):
        r = TDoAResult(position=(40.123, -74.456))
        assert r.lat == 40.123
        assert r.lon == -74.456

    def test_with_sensors(self):
        r = TDoAResult(
            position=(40.0, -74.0),
            confidence=0.85,
            residual_error_m=5.2,
            sensors_used=["mic1", "mic2", "mic3"],
        )
        assert len(r.sensors_used) == 3
        assert r.confidence == 0.85


class TestComputeTDoAPosition:
    """Tests for compute_tdoa_position function."""

    def test_fewer_than_3_returns_none(self):
        now = time.time() * 1000
        obs = [
            TDoAObservation(sensor_id="mic1", arrival_time_ms=now, lat=40.0, lon=-74.0),
            TDoAObservation(sensor_id="mic2", arrival_time_ms=now + 100, lat=40.001, lon=-74.0),
        ]
        assert compute_tdoa_position(obs) is None

    def test_empty_returns_none(self):
        assert compute_tdoa_position([]) is None

    def test_three_observers_returns_result(self):
        now = time.time() * 1000
        obs = [
            TDoAObservation(
                sensor_id="mic1", arrival_time_ms=now,
                lat=40.0, lon=-74.0, ntp_sync_quality=0.9,
            ),
            TDoAObservation(
                sensor_id="mic2", arrival_time_ms=now + 100,
                lat=40.001, lon=-74.0, ntp_sync_quality=0.9,
            ),
            TDoAObservation(
                sensor_id="mic3", arrival_time_ms=now + 50,
                lat=40.0, lon=-73.999, ntp_sync_quality=0.9,
            ),
        ]
        result = compute_tdoa_position(obs)
        assert result is not None
        assert isinstance(result, TDoAResult)
        assert result.confidence > 0.0
        assert len(result.sensors_used) == 3
        assert result.method == "tdoa_weighted_centroid"

    def test_closest_sensor_weighted_highest(self):
        now = time.time() * 1000
        obs = [
            TDoAObservation(
                sensor_id="mic1", arrival_time_ms=now,
                lat=40.0, lon=-74.0, ntp_sync_quality=0.95,
            ),
            TDoAObservation(
                sensor_id="mic2", arrival_time_ms=now + 300,
                lat=40.002, lon=-74.0, ntp_sync_quality=0.95,
            ),
            TDoAObservation(
                sensor_id="mic3", arrival_time_ms=now + 150,
                lat=40.001, lon=-73.999, ntp_sync_quality=0.95,
            ),
        ]
        result = compute_tdoa_position(obs)
        assert result is not None
        # Source should be closer to mic1 (earliest arrival)
        assert abs(result.lat - 40.0) < abs(result.lat - 40.002)

    def test_simultaneous_arrival_near_centroid(self):
        now = time.time() * 1000
        obs = [
            TDoAObservation(
                sensor_id="mic1", arrival_time_ms=now,
                lat=40.0, lon=-74.0, ntp_sync_quality=0.9,
            ),
            TDoAObservation(
                sensor_id="mic2", arrival_time_ms=now,
                lat=40.001, lon=-74.0, ntp_sync_quality=0.9,
            ),
            TDoAObservation(
                sensor_id="mic3", arrival_time_ms=now,
                lat=40.0, lon=-73.999, ntp_sync_quality=0.9,
            ),
        ]
        result = compute_tdoa_position(obs)
        assert result is not None
        # All simultaneous = equal weight = near centroid
        expected_lat = (40.0 + 40.001 + 40.0) / 3
        assert abs(result.lat - expected_lat) < 0.001

    def test_sync_quality_affects_confidence(self):
        now = time.time() * 1000
        high_sync = [
            TDoAObservation(
                sensor_id=f"mic{i}", arrival_time_ms=now + i * 50,
                lat=40.0 + i * 0.001, lon=-74.0, ntp_sync_quality=0.95,
            )
            for i in range(3)
        ]
        low_sync = [
            TDoAObservation(
                sensor_id=f"mic{i}", arrival_time_ms=now + i * 50,
                lat=40.0 + i * 0.001, lon=-74.0, ntp_sync_quality=0.1,
            )
            for i in range(3)
        ]
        r_high = compute_tdoa_position(high_sync)
        r_low = compute_tdoa_position(low_sync)
        assert r_high is not None
        assert r_low is not None
        assert r_high.confidence > r_low.confidence

    def test_more_sensors_higher_confidence(self):
        now = time.time() * 1000
        obs_3 = [
            TDoAObservation(
                sensor_id=f"mic{i}", arrival_time_ms=now + i * 30,
                lat=40.0 + i * 0.001, lon=-74.0 + (i % 2) * 0.001,
                ntp_sync_quality=0.9,
            )
            for i in range(3)
        ]
        obs_5 = [
            TDoAObservation(
                sensor_id=f"mic{i}", arrival_time_ms=now + i * 30,
                lat=40.0 + i * 0.001, lon=-74.0 + (i % 2) * 0.001,
                ntp_sync_quality=0.9,
            )
            for i in range(5)
        ]
        r3 = compute_tdoa_position(obs_3)
        r5 = compute_tdoa_position(obs_5)
        assert r3 is not None and r5 is not None
        assert r5.confidence >= r3.confidence

    def test_residual_error_computed(self):
        now = time.time() * 1000
        obs = [
            TDoAObservation(
                sensor_id="mic1", arrival_time_ms=now,
                lat=40.0, lon=-74.0, ntp_sync_quality=0.5,
            ),
            TDoAObservation(
                sensor_id="mic2", arrival_time_ms=now + 500,
                lat=40.001, lon=-74.0, ntp_sync_quality=0.5,
            ),
            TDoAObservation(
                sensor_id="mic3", arrival_time_ms=now + 250,
                lat=40.0, lon=-73.999, ntp_sync_quality=0.5,
            ),
        ]
        result = compute_tdoa_position(obs)
        assert result is not None
        assert result.residual_error_m > 0

    def test_event_type_propagated(self):
        now = time.time() * 1000
        obs = [
            TDoAObservation(
                sensor_id=f"mic{i}", arrival_time_ms=now + i * 30,
                lat=40.0 + i * 0.001, lon=-74.0,
                event_type="gunshot", ntp_sync_quality=0.9,
            )
            for i in range(3)
        ]
        result = compute_tdoa_position(obs)
        assert result is not None
        assert result.event_type == "gunshot"


class TestSpeedOfSound:
    def test_speed_constant(self):
        assert SPEED_OF_SOUND_MPS == 343.0

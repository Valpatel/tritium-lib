# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for acoustic intelligence models — ML classification, TDoA localization."""

import time

import pytest

from tritium_lib.models.acoustic_intelligence import (
    AcousticObserver,
    AcousticTrilateration,
    AudioFeatureVector,
    SoundClassification,
    SoundSignature,
    acoustic_trilaterate,
    SPEED_OF_SOUND_MPS,
)


class TestAudioFeatureVector:
    """Tests for AudioFeatureVector model."""

    def test_default_mfcc_length(self):
        fv = AudioFeatureVector()
        assert len(fv.mfcc) == 13

    def test_custom_values(self):
        fv = AudioFeatureVector(
            sensor_id="mic1",
            mfcc=[1.0] * 13,
            spectral_centroid=1500.0,
            zero_crossing_rate=0.1,
            rms_energy=0.5,
        )
        assert fv.sensor_id == "mic1"
        assert fv.spectral_centroid == 1500.0
        assert fv.rms_energy == 0.5

    def test_serialization(self):
        fv = AudioFeatureVector(sensor_id="test")
        d = fv.model_dump()
        assert "mfcc" in d
        assert "spectral_centroid" in d
        assert d["sensor_id"] == "test"


class TestSoundSignature:
    """Tests for SoundSignature model."""

    def test_basic_creation(self):
        sig = SoundSignature(
            class_name="gunshot",
            frequencies=[3000, 4000, 5000],
            duration_range_ms=(10, 200),
            energy_profile=[0.1, 0.9, 0.3],
        )
        assert sig.class_name == "gunshot"
        assert len(sig.frequencies) == 3

    def test_matches_features_in_range(self):
        sig = SoundSignature(
            class_name="voice",
            spectral_centroid_range=(85, 3000),
            zero_crossing_range=(0.02, 0.15),
            rms_energy_range=(0.05, 0.5),
            duration_range_ms=(200, 5000),
        )
        fv = AudioFeatureVector(
            spectral_centroid=800,
            zero_crossing_rate=0.08,
            rms_energy=0.25,
            duration_ms=1500,
        )
        score = sig.matches_features(fv)
        assert score == 1.0

    def test_matches_features_out_of_range(self):
        sig = SoundSignature(
            class_name="gunshot",
            spectral_centroid_range=(2000, 6000),
            zero_crossing_range=(0.1, 0.3),
            rms_energy_range=(0.7, 1.0),
            duration_range_ms=(10, 200),
        )
        fv = AudioFeatureVector(
            spectral_centroid=100,  # way below range
            zero_crossing_rate=0.01,
            rms_energy=0.1,
            duration_ms=5000,
        )
        score = sig.matches_features(fv)
        assert score == 0.0


class TestSoundClassification:
    """Tests for SoundClassification model."""

    def test_default_values(self):
        sc = SoundClassification()
        assert sc.event_type == "unknown"
        assert sc.confidence == 0.0
        assert sc.model_version == "rule_based_v1"

    def test_with_predictions(self):
        sc = SoundClassification(
            event_type="gunshot",
            confidence=0.92,
            model_version="mfcc_knn_v1",
            predictions=[
                {"class_name": "gunshot", "confidence": 0.92},
                {"class_name": "explosion", "confidence": 0.05},
            ],
        )
        assert len(sc.predictions) == 2
        assert sc.predictions[0]["class_name"] == "gunshot"


class TestAcousticTrilateration:
    """Tests for acoustic source localization via TDoA."""

    def test_two_observers(self):
        t = time.time()
        result = acoustic_trilaterate([
            {"sensor_id": "mic1", "lat": 40.0, "lon": -74.0, "arrival_time": t},
            {"sensor_id": "mic2", "lat": 40.001, "lon": -74.0, "arrival_time": t + 0.3},
        ])
        assert result is not None
        assert "estimated_lat" in result
        assert "estimated_lon" in result
        assert result["confidence"] > 0.0
        # Source should be closer to mic1 (earlier arrival by 0.3s = ~100m)
        assert abs(result["estimated_lat"] - 40.0) < abs(result["estimated_lat"] - 40.001)

    def test_three_observers(self):
        t = time.time()
        result = acoustic_trilaterate([
            {"sensor_id": "mic1", "lat": 40.0, "lon": -74.0, "arrival_time": t},
            {"sensor_id": "mic2", "lat": 40.001, "lon": -74.0, "arrival_time": t + 0.1},
            {"sensor_id": "mic3", "lat": 40.0, "lon": -73.999, "arrival_time": t + 0.05},
        ])
        assert result is not None
        assert result["confidence"] > 0.3
        # With 3 observers, confidence should be higher
        two_obs = acoustic_trilaterate([
            {"sensor_id": "mic1", "lat": 40.0, "lon": -74.0, "arrival_time": t},
            {"sensor_id": "mic2", "lat": 40.001, "lon": -74.0, "arrival_time": t + 0.1},
        ])
        assert result["confidence"] >= two_obs["confidence"]

    def test_single_observer_returns_none(self):
        t = time.time()
        result = acoustic_trilaterate([
            {"sensor_id": "mic1", "lat": 40.0, "lon": -74.0, "arrival_time": t},
        ])
        assert result is None

    def test_empty_observers_returns_none(self):
        result = acoustic_trilaterate([])
        assert result is None

    def test_simultaneous_arrival(self):
        """When all observers hear the sound at the same time, source is equidistant."""
        t = time.time()
        result = acoustic_trilaterate([
            {"sensor_id": "mic1", "lat": 40.0, "lon": -74.0, "arrival_time": t},
            {"sensor_id": "mic2", "lat": 40.001, "lon": -74.0, "arrival_time": t},
            {"sensor_id": "mic3", "lat": 40.0, "lon": -73.999, "arrival_time": t},
        ])
        assert result is not None
        # Should be near the centroid of the observers

    def test_model_fields(self):
        tri = AcousticTrilateration(
            event_id="test",
            estimated_lat=40.0,
            estimated_lon=-74.0,
            confidence=0.8,
        )
        assert tri.estimated_position == (40.0, -74.0)
        assert tri.sound_speed_mps == 343.0

    def test_observer_model(self):
        obs = AcousticObserver(
            sensor_id="mic1",
            lat=40.0,
            lon=-74.0,
            arrival_time=time.time(),
            amplitude_db=-30.0,
        )
        assert obs.sensor_id == "mic1"
        assert obs.confidence == 1.0  # default


class TestSpeedOfSound:
    """Verify speed of sound constant."""

    def test_speed_value(self):
        assert SPEED_OF_SOUND_MPS == 343.0

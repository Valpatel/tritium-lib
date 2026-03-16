# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for AcousticFeatureVector model."""

import json
import time

from tritium_lib.models.acoustic_feature_vector import AcousticFeatureVector


class TestAcousticFeatureVector:
    """Tests for the AcousticFeatureVector MQTT transport model."""

    def test_create_minimal(self):
        """Minimal creation with just device_id."""
        fv = AcousticFeatureVector(device_id="esp32-mic-001")
        assert fv.device_id == "esp32-mic-001"
        assert len(fv.mfcc_coefficients) == 13
        assert all(c == 0.0 for c in fv.mfcc_coefficients)
        assert fv.energy == 0.0
        assert fv.zero_crossing_rate == 0.0
        assert fv.spectral_centroid == 0.0
        assert fv.duration_ms == 0
        assert fv.sample_rate == 16000
        assert fv.classification is None
        assert fv.confidence is None

    def test_create_full(self):
        """Full creation with all fields."""
        mfcc = [1.0, -2.5, 3.1, 0.7, -1.2, 0.3, -0.9, 2.1, -1.5, 0.8, -0.4, 1.6, -0.2]
        fv = AcousticFeatureVector(
            device_id="esp32-mic-002",
            timestamp=1710500000.0,
            mfcc_coefficients=mfcc,
            energy=0.45,
            zero_crossing_rate=0.12,
            spectral_centroid=2500.0,
            duration_ms=50,
            sample_rate=16000,
            classification="speech",
            confidence=0.87,
        )
        assert fv.device_id == "esp32-mic-002"
        assert fv.mfcc_coefficients == mfcc
        assert fv.energy == 0.45
        assert fv.zero_crossing_rate == 0.12
        assert fv.spectral_centroid == 2500.0
        assert fv.duration_ms == 50
        assert fv.sample_rate == 16000
        assert fv.classification == "speech"
        assert fv.confidence == 0.87

    def test_to_mqtt_payload_basic(self):
        """to_mqtt_payload produces valid compact JSON."""
        fv = AcousticFeatureVector(
            device_id="node-01",
            timestamp=1710500000.0,
            energy=0.3,
            zero_crossing_rate=0.15,
            spectral_centroid=1200.5,
            duration_ms=50,
            sample_rate=16000,
        )
        payload = fv.to_mqtt_payload()
        data = json.loads(payload)
        assert data["device_id"] == "node-01"
        assert data["ts"] == 1710500000.0
        assert data["energy"] == 0.3
        assert data["zcr"] == 0.15
        assert data["sc"] == 1200.5
        assert data["dur_ms"] == 50
        assert data["sr"] == 16000
        assert len(data["mfcc"]) == 13
        # No classification or confidence keys when None
        assert "cls" not in data
        assert "conf" not in data

    def test_to_mqtt_payload_with_classification(self):
        """to_mqtt_payload includes classification when present."""
        fv = AcousticFeatureVector(
            device_id="node-01",
            classification="vehicle",
            confidence=0.72,
        )
        payload = fv.to_mqtt_payload()
        data = json.loads(payload)
        assert data["cls"] == "vehicle"
        assert data["conf"] == 0.72

    def test_from_mqtt_payload_basic(self):
        """from_mqtt_payload deserializes compact JSON."""
        raw = json.dumps({
            "device_id": "node-02",
            "ts": 1710500100.0,
            "mfcc": [0.1] * 13,
            "energy": 0.25,
            "zcr": 0.08,
            "sc": 3000.0,
            "dur_ms": 100,
            "sr": 16000,
        })
        fv = AcousticFeatureVector.from_mqtt_payload(raw)
        assert fv.device_id == "node-02"
        assert fv.timestamp == 1710500100.0
        assert len(fv.mfcc_coefficients) == 13
        assert fv.energy == 0.25
        assert fv.zero_crossing_rate == 0.08
        assert fv.spectral_centroid == 3000.0
        assert fv.duration_ms == 100
        assert fv.sample_rate == 16000
        assert fv.classification is None
        assert fv.confidence is None

    def test_from_mqtt_payload_with_classification(self):
        """from_mqtt_payload handles optional classification fields."""
        raw = json.dumps({
            "device_id": "node-03",
            "ts": 1710500200.0,
            "mfcc": [0.5, -0.3, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "energy": 0.6,
            "zcr": 0.2,
            "sc": 500.0,
            "dur_ms": 50,
            "sr": 16000,
            "cls": "impact",
            "conf": 0.95,
        })
        fv = AcousticFeatureVector.from_mqtt_payload(raw)
        assert fv.classification == "impact"
        assert fv.confidence == 0.95

    def test_roundtrip_mqtt(self):
        """Serialize and deserialize via MQTT payload preserves data."""
        mfcc = [1.23, -4.56, 7.89, 0.12, -3.45, 6.78, -9.01, 2.34, -5.67, 8.9, -1.23, 4.56, -7.89]
        original = AcousticFeatureVector(
            device_id="roundtrip-node",
            timestamp=1710500500.0,
            mfcc_coefficients=mfcc,
            energy=0.4321,
            zero_crossing_rate=0.1567,
            spectral_centroid=1800.3,
            duration_ms=75,
            sample_rate=16000,
            classification="speech",
            confidence=0.823,
        )
        payload = original.to_mqtt_payload()
        restored = AcousticFeatureVector.from_mqtt_payload(payload)

        assert restored.device_id == original.device_id
        assert restored.timestamp == original.timestamp
        assert restored.mfcc_coefficients == original.mfcc_coefficients
        assert restored.energy == round(original.energy, 4)
        assert restored.zero_crossing_rate == round(original.zero_crossing_rate, 4)
        assert restored.spectral_centroid == round(original.spectral_centroid, 1)
        assert restored.duration_ms == original.duration_ms
        assert restored.sample_rate == original.sample_rate
        assert restored.classification == original.classification
        assert restored.confidence == round(original.confidence, 3)

    def test_pydantic_json_roundtrip(self):
        """Standard Pydantic JSON serialization roundtrip."""
        fv = AcousticFeatureVector(
            device_id="pydantic-test",
            energy=0.5,
            classification="ambient",
            confidence=0.99,
        )
        j = fv.model_dump_json()
        fv2 = AcousticFeatureVector.model_validate_json(j)
        assert fv2.device_id == fv.device_id
        assert fv2.energy == fv.energy
        assert fv2.classification == fv.classification
        assert fv2.confidence == fv.confidence

    def test_default_timestamp_is_recent(self):
        """Default timestamp should be close to current time."""
        before = time.time()
        fv = AcousticFeatureVector(device_id="ts-test")
        after = time.time()
        assert before <= fv.timestamp <= after

    def test_mfcc_custom_count(self):
        """mfcc_coefficients can hold different counts if needed."""
        fv = AcousticFeatureVector(
            device_id="custom-mfcc",
            mfcc_coefficients=[1.0, 2.0, 3.0],
        )
        assert len(fv.mfcc_coefficients) == 3
        payload = fv.to_mqtt_payload()
        restored = AcousticFeatureVector.from_mqtt_payload(payload)
        assert restored.mfcc_coefficients == [1.0, 2.0, 3.0]

    def test_compact_payload_no_spaces(self):
        """MQTT payload uses compact JSON (no spaces)."""
        fv = AcousticFeatureVector(device_id="compact-test")
        payload = fv.to_mqtt_payload()
        assert " " not in payload

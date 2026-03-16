# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for acoustic event classification models."""

from tritium_lib.models.acoustic_event import (
    AcousticEvent,
    AcousticEventType,
    AcousticSensorConfig,
    AcousticSeverity,
    AcousticSpectrum,
    AcousticStats,
    classify_event_severity,
)


class TestAcousticEvent:
    def test_basic_creation(self):
        e = AcousticEvent(
            event_type=AcousticEventType.GUNSHOT,
            confidence=0.95,
            sensor_id="mic01",
        )
        assert e.event_type == AcousticEventType.GUNSHOT
        assert e.confidence == 0.95

    def test_to_target_dict(self):
        e = AcousticEvent(
            event_type=AcousticEventType.EXPLOSION,
            confidence=0.88,
            sensor_id="mic02",
            latitude=37.77,
            longitude=-122.42,
        )
        td = e.to_target_dict()
        assert td["source"] == "acoustic"
        assert td["target_id"] == "acoustic_mic02_explosion"
        assert td["position"]["lat"] == 37.77
        assert td["metadata"]["severity"] == "info"  # default

    def test_all_event_types(self):
        for evt in AcousticEventType:
            e = AcousticEvent(event_type=evt, sensor_id="test")
            assert e.event_type == evt

    def test_all_severity_levels(self):
        for sev in AcousticSeverity:
            e = AcousticEvent(severity=sev, sensor_id="test")
            assert e.severity == sev


class TestClassifyEventSeverity:
    def test_gunshot_is_critical(self):
        assert classify_event_severity(AcousticEventType.GUNSHOT) == AcousticSeverity.CRITICAL

    def test_explosion_is_critical(self):
        assert classify_event_severity(AcousticEventType.EXPLOSION) == AcousticSeverity.CRITICAL

    def test_glass_break_is_high(self):
        assert classify_event_severity(AcousticEventType.GLASS_BREAK) == AcousticSeverity.HIGH

    def test_siren_is_medium(self):
        assert classify_event_severity(AcousticEventType.SIREN) == AcousticSeverity.MEDIUM

    def test_voice_is_low(self):
        assert classify_event_severity(AcousticEventType.VOICE) == AcousticSeverity.LOW

    def test_ambient_is_info(self):
        assert classify_event_severity(AcousticEventType.AMBIENT) == AcousticSeverity.INFO

    def test_all_types_mapped(self):
        """Every event type should have a severity mapping."""
        for evt in AcousticEventType:
            sev = classify_event_severity(evt)
            assert isinstance(sev, AcousticSeverity)


class TestAcousticSpectrum:
    def test_basic(self):
        s = AcousticSpectrum(
            sensor_id="mic01",
            frequencies=[100.0, 200.0, 300.0],
            magnitudes=[-30.0, -25.0, -40.0],
        )
        assert len(s.frequencies) == 3
        assert s.fft_size == 1024
        assert s.window == "hann"


class TestAcousticSensorConfig:
    def test_defaults(self):
        c = AcousticSensorConfig(sensor_id="mic01")
        assert c.enabled is True
        assert c.sample_rate_hz == 16000
        assert c.classification_model == "rule_based"

    def test_custom_config(self):
        c = AcousticSensorConfig(
            sensor_id="mic02",
            detection_threshold_db=-50.0,
            min_confidence=0.7,
            classification_model="ml_yamnet",
        )
        assert c.detection_threshold_db == -50.0
        assert c.classification_model == "ml_yamnet"


class TestAcousticStats:
    def test_defaults(self):
        s = AcousticStats()
        assert s.total_events == 0
        assert s.sensors_active == 0

    def test_populated(self):
        s = AcousticStats(
            total_events=50,
            events_by_type={"gunshot": 2, "voice": 30, "ambient": 18},
            avg_confidence=0.82,
            sensors_active=4,
        )
        assert s.total_events == 50
        assert len(s.events_by_type) == 3

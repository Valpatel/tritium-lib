# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for fleet event correlation models."""

from datetime import datetime, timezone

from tritium_lib.models.correlation import (
    CorrelationType,
    CorrelationEvent,
    CorrelationSummary,
    classify_correlation_severity,
    summarize_correlations,
)


def _utc(year=2026, month=3, day=7, hour=12, minute=0, second=0):
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


class TestCorrelationEvent:
    def test_create_basic(self):
        e = CorrelationEvent(
            type=CorrelationType.SYNCHRONIZED_REBOOT,
            description="3 devices rebooted within 60s",
            devices_involved=["dev-a", "dev-b", "dev-c"],
            confidence=0.85,
            timestamp=_utc(),
        )
        assert e.type == CorrelationType.SYNCHRONIZED_REBOOT
        assert len(e.devices_involved) == 3
        assert e.confidence == 0.85

    def test_confidence_bounds(self):
        e = CorrelationEvent(
            type=CorrelationType.ENVIRONMENTAL,
            description="test",
            confidence=0.0,
        )
        assert e.confidence == 0.0

        e2 = CorrelationEvent(
            type=CorrelationType.ENVIRONMENTAL,
            description="test",
            confidence=1.0,
        )
        assert e2.confidence == 1.0

    def test_all_types(self):
        for t in CorrelationType:
            e = CorrelationEvent(type=t, description=f"test {t.value}")
            assert e.type == t

    def test_serialization(self):
        e = CorrelationEvent(
            type=CorrelationType.CASCADING_FAILURE,
            description="WiFi error propagated",
            devices_involved=["dev-1", "dev-2"],
            confidence=0.72,
            timestamp=_utc(),
        )
        d = e.model_dump()
        assert d["type"] == "cascading_failure"
        assert d["confidence"] == 0.72
        assert len(d["devices_involved"]) == 2

    def test_json_roundtrip(self):
        e = CorrelationEvent(
            type=CorrelationType.PERIODIC_FAILURE,
            description="Failures at noon",
            devices_involved=["dev-x"],
            confidence=0.5,
            timestamp=_utc(),
        )
        json_str = e.model_dump_json()
        e2 = CorrelationEvent.model_validate_json(json_str)
        assert e2.type == e.type
        assert e2.description == e.description
        assert e2.devices_involved == e.devices_involved


class TestCorrelationSummary:
    def _make_events(self):
        return [
            CorrelationEvent(
                type=CorrelationType.SYNCHRONIZED_REBOOT,
                description="Mass reboot",
                devices_involved=["a", "b", "c", "d", "e"],
                confidence=0.9,
                timestamp=_utc(hour=10),
            ),
            CorrelationEvent(
                type=CorrelationType.CASCADING_FAILURE,
                description="WiFi cascade",
                devices_involved=["a", "b"],
                confidence=0.75,
                timestamp=_utc(hour=11),
            ),
            CorrelationEvent(
                type=CorrelationType.ENVIRONMENTAL,
                description="I2C errors",
                devices_involved=["c", "d"],
                confidence=0.6,
                timestamp=_utc(hour=12),
            ),
            CorrelationEvent(
                type=CorrelationType.PERIODIC_FAILURE,
                description="Noon failures",
                devices_involved=["a"],
                confidence=0.4,
                timestamp=_utc(hour=12),
            ),
        ]

    def test_total_events(self):
        summary = CorrelationSummary(events=self._make_events())
        assert summary.total_events == 4

    def test_high_confidence_filter(self):
        summary = CorrelationSummary(events=self._make_events())
        high = summary.high_confidence_events
        assert len(high) == 2  # 0.9 and 0.75

    def test_events_by_type(self):
        summary = CorrelationSummary(events=self._make_events())
        reboots = summary.events_by_type(CorrelationType.SYNCHRONIZED_REBOOT)
        assert len(reboots) == 1
        env = summary.events_by_type(CorrelationType.ENVIRONMENTAL)
        assert len(env) == 1

    def test_affected_devices(self):
        summary = CorrelationSummary(events=self._make_events())
        devices = summary.affected_devices()
        assert devices == {"a", "b", "c", "d", "e"}

    def test_empty_summary(self):
        summary = CorrelationSummary()
        assert summary.total_events == 0
        assert summary.high_confidence_events == []
        assert summary.affected_devices() == set()

    def test_with_window(self):
        summary = CorrelationSummary(
            events=self._make_events(),
            snapshot_count=50,
            device_count=10,
            window_start=_utc(hour=0),
            window_end=_utc(hour=23, minute=59),
        )
        assert summary.snapshot_count == 50
        assert summary.device_count == 10


class TestClassifySeverity:
    def test_mass_reboot_critical(self):
        e = CorrelationEvent(
            type=CorrelationType.SYNCHRONIZED_REBOOT,
            description="test",
            devices_involved=["a", "b", "c", "d", "e"],
            confidence=0.9,
        )
        assert classify_correlation_severity(e) == "critical"

    def test_small_reboot_warning(self):
        e = CorrelationEvent(
            type=CorrelationType.SYNCHRONIZED_REBOOT,
            description="test",
            devices_involved=["a", "b", "c"],
            confidence=0.7,
        )
        assert classify_correlation_severity(e) == "warning"

    def test_cascade_high_conf_critical(self):
        e = CorrelationEvent(
            type=CorrelationType.CASCADING_FAILURE,
            description="test",
            confidence=0.8,
        )
        assert classify_correlation_severity(e) == "critical"

    def test_cascade_low_conf_warning(self):
        e = CorrelationEvent(
            type=CorrelationType.CASCADING_FAILURE,
            description="test",
            confidence=0.5,
        )
        assert classify_correlation_severity(e) == "warning"

    def test_environmental_warning(self):
        e = CorrelationEvent(
            type=CorrelationType.ENVIRONMENTAL,
            description="test",
            confidence=0.9,
        )
        assert classify_correlation_severity(e) == "warning"

    def test_periodic_info(self):
        e = CorrelationEvent(
            type=CorrelationType.PERIODIC_FAILURE,
            description="test",
            confidence=0.9,
        )
        assert classify_correlation_severity(e) == "info"


class TestSummarizeCorrelations:
    def test_summary(self):
        events = [
            CorrelationEvent(
                type=CorrelationType.SYNCHRONIZED_REBOOT,
                description="test",
                devices_involved=["a", "b", "c"],
                confidence=0.9,
            ),
            CorrelationEvent(
                type=CorrelationType.CASCADING_FAILURE,
                description="test",
                devices_involved=["b", "d"],
                confidence=0.3,
            ),
        ]
        s = summarize_correlations(events)
        assert s["total"] == 2
        assert s["high_confidence"] == 1
        assert s["by_type"]["synchronized_reboot"] == 1
        assert s["by_type"]["cascading_failure"] == 1
        assert s["affected_devices"] == 4  # a, b, c, d

    def test_empty(self):
        s = summarize_correlations([])
        assert s["total"] == 0
        assert s["high_confidence"] == 0
        assert s["affected_devices"] == 0

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.quality — data quality monitoring."""

import time

import pytest

from tritium_lib.events import EventBus
from tritium_lib.quality import (
    DataQualityMonitor,
    QualityAlert,
    QualityDimension,
    QualityMetric,
    QualityReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def monitor(bus):
    return DataQualityMonitor(event_bus=bus, alert_cooldown=0.0)


# ---------------------------------------------------------------------------
# QualityDimension
# ---------------------------------------------------------------------------

class TestQualityDimension:
    def test_all_five_dimensions_exist(self):
        dims = list(QualityDimension)
        assert len(dims) == 5
        assert QualityDimension.COMPLETENESS in dims
        assert QualityDimension.TIMELINESS in dims
        assert QualityDimension.ACCURACY in dims
        assert QualityDimension.CONSISTENCY in dims
        assert QualityDimension.FRESHNESS in dims

    def test_dimension_values_are_strings(self):
        assert QualityDimension.COMPLETENESS.value == "completeness"
        assert QualityDimension.FRESHNESS.value == "freshness"


# ---------------------------------------------------------------------------
# QualityMetric
# ---------------------------------------------------------------------------

class TestQualityMetric:
    def test_metric_creation(self):
        m = QualityMetric(
            dimension=QualityDimension.ACCURACY,
            source_id="sensor_1",
            score=0.95,
            detail="All values in range",
        )
        assert m.dimension == QualityDimension.ACCURACY
        assert m.source_id == "sensor_1"
        assert m.score == 0.95
        assert m.detail == "All values in range"
        assert m.timestamp > 0

    def test_metric_is_immutable(self):
        m = QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            source_id="s1",
            score=1.0,
        )
        with pytest.raises(AttributeError):
            m.score = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# QualityAlert
# ---------------------------------------------------------------------------

class TestQualityAlert:
    def test_alert_creation(self):
        a = QualityAlert(
            alert_id="abc-123",
            source_id="cam_01",
            dimension=QualityDimension.TIMELINESS,
            score=0.3,
            threshold=0.5,
            message="Too slow",
        )
        assert a.alert_id == "abc-123"
        assert a.dimension == QualityDimension.TIMELINESS
        assert a.score == 0.3
        assert a.threshold == 0.5

    def test_alert_is_immutable(self):
        a = QualityAlert(
            alert_id="x",
            source_id="y",
            dimension=QualityDimension.ACCURACY,
            score=0.1,
            threshold=0.5,
            message="bad",
        )
        with pytest.raises(AttributeError):
            a.score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------

class TestQualityReport:
    def test_report_export(self):
        report = QualityReport(source_id="ble_01", overall_score=0.85, sample_count=100)
        report.metrics[QualityDimension.COMPLETENESS] = QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            source_id="ble_01",
            score=0.9,
            detail="test",
        )
        exported = report.export()
        assert exported["source_id"] == "ble_01"
        assert exported["overall_score"] == 0.85
        assert exported["sample_count"] == 100
        assert "completeness" in exported["metrics"]
        assert exported["metrics"]["completeness"]["score"] == 0.9

    def test_report_export_empty(self):
        report = QualityReport(source_id="empty")
        exported = report.export()
        assert exported["source_id"] == "empty"
        assert exported["metrics"] == {}
        assert exported["alerts"] == []


# ---------------------------------------------------------------------------
# DataQualityMonitor — Registration
# ---------------------------------------------------------------------------

class TestMonitorRegistration:
    def test_register_source(self, monitor):
        monitor.register_source("s1", expected_fields=["a", "b"])
        assert "s1" in monitor.get_source_ids()

    def test_unregister_source(self, monitor):
        monitor.register_source("s1")
        assert monitor.unregister_source("s1") is True
        assert "s1" not in monitor.get_source_ids()

    def test_unregister_nonexistent_source(self, monitor):
        assert monitor.unregister_source("ghost") is False

    def test_auto_register_on_record(self, monitor):
        monitor.record("auto_source", {"x": 1})
        assert "auto_source" in monitor.get_source_ids()


# ---------------------------------------------------------------------------
# DataQualityMonitor — Completeness
# ---------------------------------------------------------------------------

class TestCompleteness:
    def test_complete_data_scores_one(self, monitor):
        monitor.register_source("s1", expected_fields=["mac", "rssi"])
        monitor.record("s1", {"mac": "AA:BB", "rssi": -65})
        report = monitor.get_report("s1")
        assert report is not None
        comp = report.metrics[QualityDimension.COMPLETENESS]
        assert comp.score == 1.0

    def test_missing_field_lowers_score(self, monitor):
        monitor.register_source("s1", expected_fields=["mac", "rssi", "timestamp"])
        monitor.record("s1", {"mac": "AA:BB"})  # missing rssi and timestamp
        report = monitor.get_report("s1")
        assert report is not None
        comp = report.metrics[QualityDimension.COMPLETENESS]
        assert comp.score == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_none_values_count_as_missing(self, monitor):
        monitor.register_source("s1", expected_fields=["a", "b"])
        monitor.record("s1", {"a": "hello", "b": None})
        report = monitor.get_report("s1")
        assert report is not None
        comp = report.metrics[QualityDimension.COMPLETENESS]
        assert comp.score == 0.5

    def test_no_expected_fields_always_complete(self, monitor):
        monitor.register_source("s1")  # no expected fields
        monitor.record("s1", {"whatever": 42})
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.COMPLETENESS].score == 1.0


# ---------------------------------------------------------------------------
# DataQualityMonitor — Timeliness
# ---------------------------------------------------------------------------

class TestTimeliness:
    def test_realtime_data_scores_one(self, monitor):
        monitor.register_source("s1", max_latency_seconds=10.0)
        now = time.time()
        monitor.record("s1", {"x": 1}, data_timestamp=now)
        report = monitor.get_report("s1")
        assert report is not None
        time_score = report.metrics[QualityDimension.TIMELINESS].score
        assert time_score >= 0.95  # nearly perfect

    def test_old_data_scores_low(self, monitor):
        monitor.register_source("s1", max_latency_seconds=10.0)
        old_time = time.time() - 15.0  # 15 seconds old, max is 10
        monitor.record("s1", {"x": 1}, data_timestamp=old_time)
        report = monitor.get_report("s1")
        assert report is not None
        time_score = report.metrics[QualityDimension.TIMELINESS].score
        assert time_score == 0.0

    def test_no_timestamp_assumes_timely(self, monitor):
        monitor.register_source("s1")
        monitor.record("s1", {"x": 1})  # no data_timestamp
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.TIMELINESS].score == 1.0


# ---------------------------------------------------------------------------
# DataQualityMonitor — Accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_in_range_scores_one(self, monitor):
        monitor.register_source("s1", value_ranges={"rssi": (-100.0, 0.0)})
        monitor.record("s1", {"rssi": -65})
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.ACCURACY].score == 1.0

    def test_out_of_range_scores_zero(self, monitor):
        monitor.register_source("s1", value_ranges={"rssi": (-100.0, 0.0)})
        monitor.record("s1", {"rssi": 50})  # out of range
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.ACCURACY].score == 0.0

    def test_multiple_ranged_fields(self, monitor):
        monitor.register_source(
            "s1",
            value_ranges={
                "rssi": (-100.0, 0.0),
                "temp": (0.0, 50.0),
            },
        )
        monitor.record("s1", {"rssi": -65, "temp": 100.0})  # temp out of range
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.ACCURACY].score == 0.5

    def test_no_ranges_always_accurate(self, monitor):
        monitor.register_source("s1")  # no value_ranges
        monitor.record("s1", {"anything": 999999})
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.ACCURACY].score == 1.0

    def test_non_numeric_value_counts_as_out_of_range(self, monitor):
        monitor.register_source("s1", value_ranges={"rssi": (-100.0, 0.0)})
        monitor.record("s1", {"rssi": "not_a_number"})
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.ACCURACY].score == 0.0


# ---------------------------------------------------------------------------
# DataQualityMonitor — Consistency
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_agreeing_sensors_score_high(self, monitor):
        monitor.register_source("s1")
        monitor.register_source("s2")
        monitor.register_consistency_group("temp_group", ["s1", "s2"])

        monitor.record("s1", {"temp": 22.0})
        monitor.record("s2", {"temp": 22.5})  # within 20%

        report = monitor.get_report("s1")
        assert report is not None
        cons = report.metrics[QualityDimension.CONSISTENCY]
        assert cons.score >= 0.9

    def test_disagreeing_sensors_score_low(self, monitor):
        monitor.register_source("s1")
        monitor.register_source("s2")
        monitor.register_consistency_group("temp_group", ["s1", "s2"])

        monitor.record("s1", {"temp": 20.0})
        monitor.record("s2", {"temp": 100.0})  # way off

        report = monitor.get_report("s1")
        assert report is not None
        cons = report.metrics[QualityDimension.CONSISTENCY]
        assert cons.score < 0.5

    def test_no_group_defaults_to_one(self, monitor):
        monitor.register_source("s1")
        monitor.record("s1", {"temp": 22.0})

        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.CONSISTENCY].score == 1.0


# ---------------------------------------------------------------------------
# DataQualityMonitor — Freshness
# ---------------------------------------------------------------------------

class TestFreshness:
    def test_recent_data_is_fresh(self, monitor):
        monitor.register_source("s1", max_staleness_seconds=60.0)
        monitor.record("s1", {"x": 1})
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.FRESHNESS].score == 1.0

    def test_never_seen_data_is_stale(self, monitor):
        monitor.register_source("s1")
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.FRESHNESS].score == 0.0


# ---------------------------------------------------------------------------
# DataQualityMonitor — Alerts
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_low_completeness_triggers_alert(self, monitor):
        monitor.register_source(
            "s1",
            expected_fields=["a", "b", "c", "d"],
            thresholds={QualityDimension.COMPLETENESS: 0.8},
        )
        alerts = monitor.record("s1", {"a": 1})  # only 1/4 = 0.25
        assert len(alerts) >= 1
        assert any(a.dimension == QualityDimension.COMPLETENESS for a in alerts)

    def test_alert_cooldown_prevents_spam(self):
        monitor = DataQualityMonitor(alert_cooldown=9999.0)
        monitor.register_source(
            "s1",
            expected_fields=["a", "b", "c", "d"],
            thresholds={QualityDimension.COMPLETENESS: 0.8},
        )
        alerts1 = monitor.record("s1", {"a": 1})
        alerts2 = monitor.record("s1", {"a": 1})
        # First record triggers, second is suppressed by cooldown
        assert len(alerts1) >= 1
        assert len(alerts2) == 0

    def test_alert_published_on_event_bus(self, bus):
        received = []
        bus.subscribe("quality.alert", lambda e: received.append(e))

        monitor = DataQualityMonitor(event_bus=bus, alert_cooldown=0.0)
        monitor.register_source(
            "s1",
            expected_fields=["a", "b", "c"],
            thresholds={QualityDimension.COMPLETENESS: 0.9},
        )
        monitor.record("s1", {"a": 1})  # 1/3 < 0.9

        assert len(received) == 1
        assert received[0].data["source_id"] == "s1"
        assert received[0].data["dimension"] == "completeness"

    def test_get_alerts_filtered(self, monitor):
        monitor.register_source(
            "s1",
            expected_fields=["a", "b"],
            thresholds={QualityDimension.COMPLETENESS: 0.9},
        )
        monitor.register_source(
            "s2",
            expected_fields=["x", "y"],
            thresholds={QualityDimension.COMPLETENESS: 0.9},
        )
        monitor.record("s1", {})
        monitor.record("s2", {})

        all_alerts = monitor.get_alerts()
        s1_alerts = monitor.get_alerts("s1")
        s2_alerts = monitor.get_alerts("s2")

        assert len(all_alerts) >= 2
        assert all(a.source_id == "s1" for a in s1_alerts)
        assert all(a.source_id == "s2" for a in s2_alerts)

    def test_clear_alerts(self, monitor):
        monitor.register_source("s1", expected_fields=["a", "b"],
                                thresholds={QualityDimension.COMPLETENESS: 0.9})
        monitor.record("s1", {})
        assert len(monitor.get_alerts()) >= 1
        cleared = monitor.clear_alerts()
        assert cleared >= 1
        assert len(monitor.get_alerts()) == 0

    def test_clear_alerts_by_source(self, monitor):
        monitor.register_source("s1", expected_fields=["a"],
                                thresholds={QualityDimension.COMPLETENESS: 0.9})
        monitor.register_source("s2", expected_fields=["b"],
                                thresholds={QualityDimension.COMPLETENESS: 0.9})
        monitor.record("s1", {})
        monitor.record("s2", {})
        monitor.clear_alerts("s1")
        remaining = monitor.get_alerts()
        assert all(a.source_id != "s1" for a in remaining)


# ---------------------------------------------------------------------------
# DataQualityMonitor — Reports
# ---------------------------------------------------------------------------

class TestReports:
    def test_report_for_unknown_source_is_none(self, monitor):
        assert monitor.get_report("nonexistent") is None

    def test_report_overall_score(self, monitor):
        monitor.register_source("s1", expected_fields=["a", "b"])
        monitor.record("s1", {"a": 1, "b": 2})
        report = monitor.get_report("s1")
        assert report is not None
        # All dimensions should be perfect with complete, timely, in-range data
        assert report.overall_score >= 0.8

    def test_get_all_reports(self, monitor):
        monitor.register_source("s1")
        monitor.register_source("s2")
        monitor.record("s1", {"x": 1})
        monitor.record("s2", {"y": 2})
        reports = monitor.get_all_reports()
        assert len(reports) == 2
        ids = {r.source_id for r in reports}
        assert ids == {"s1", "s2"}

    def test_sample_count_tracks_records(self, monitor):
        monitor.register_source("s1")
        for i in range(10):
            monitor.record("s1", {"i": i})
        report = monitor.get_report("s1")
        assert report is not None
        assert report.sample_count == 10


# ---------------------------------------------------------------------------
# DataQualityMonitor — Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_structure(self, monitor):
        monitor.register_source("s1")
        monitor.record("s1", {"a": 1})
        exported = monitor.export()
        assert "sources" in exported
        assert "alert_count" in exported
        assert "source_count" in exported
        assert "timestamp" in exported
        assert exported["source_count"] == 1

    def test_export_with_multiple_sources(self, monitor):
        for i in range(5):
            sid = f"sensor_{i}"
            monitor.register_source(sid)
            monitor.record(sid, {"val": i})
        exported = monitor.export()
        assert exported["source_count"] == 5
        assert len(exported["sources"]) == 5


# ---------------------------------------------------------------------------
# DataQualityMonitor — Dimension weights
# ---------------------------------------------------------------------------

class TestDimensionWeights:
    def test_custom_weights(self, bus):
        weights = {
            QualityDimension.COMPLETENESS: 3.0,
            QualityDimension.TIMELINESS: 1.0,
            QualityDimension.ACCURACY: 1.0,
            QualityDimension.CONSISTENCY: 0.5,
            QualityDimension.FRESHNESS: 0.5,
        }
        monitor = DataQualityMonitor(
            event_bus=bus,
            dimension_weights=weights,
            alert_cooldown=0.0,
        )
        monitor.register_source("s1", expected_fields=["a", "b"])
        # Send complete data — completeness=1.0 with high weight
        monitor.record("s1", {"a": 1, "b": 2})
        report = monitor.get_report("s1")
        assert report is not None
        # With complete data the overall should be high
        assert report.overall_score >= 0.8


# ---------------------------------------------------------------------------
# DataQualityMonitor — Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_record_with_empty_data(self, monitor):
        monitor.register_source("s1", expected_fields=["a"])
        alerts = monitor.record("s1", {})
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.COMPLETENESS].score == 0.0

    def test_accuracy_with_missing_ranged_field(self, monitor):
        """If a ranged field is not present in data, it's not checked."""
        monitor.register_source("s1", value_ranges={"rssi": (-100.0, 0.0)})
        monitor.record("s1", {"other_field": 42})  # rssi not present
        report = monitor.get_report("s1")
        assert report is not None
        # rssi wasn't in data, so nothing to check -> score 1.0
        assert report.metrics[QualityDimension.ACCURACY].score == 1.0

    def test_timeliness_with_future_timestamp(self, monitor):
        """Future timestamp should give perfect timeliness."""
        monitor.register_source("s1", max_latency_seconds=10.0)
        future = time.time() + 100
        monitor.record("s1", {"x": 1}, data_timestamp=future)
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.TIMELINESS].score == 1.0

    def test_boundary_value_in_range(self, monitor):
        """Boundary values (exactly min or max) should be in range."""
        monitor.register_source("s1", value_ranges={"val": (0.0, 100.0)})
        monitor.record("s1", {"val": 0.0})
        report = monitor.get_report("s1")
        assert report is not None
        assert report.metrics[QualityDimension.ACCURACY].score == 1.0

        monitor.record("s1", {"val": 100.0})
        report2 = monitor.get_report("s1")
        assert report2 is not None
        assert report2.metrics[QualityDimension.ACCURACY].score == 1.0

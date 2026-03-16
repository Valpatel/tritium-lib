# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for diagnostic and telemetry models."""

from datetime import datetime, timezone

from tritium_lib.models.diagnostics import (
    Anomaly,
    AnomalyType,
    DiagEvent,
    DiagLogBatch,
    DiagLogEntry,
    DiagLogSummary,
    FleetHealthSummary,
    HeapTrend,
    HealthSnapshot,
    I2cSlaveHealth,
    MeshPeer,
    NodeDiagReport,
    Severity,
    aggregate_fleet_health,
    analyze_heap_trends,
    classify_node_health,
    detect_fleet_anomalies,
    summarize_diag_log,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc)


def _make_health(
    node_id: str = "esp32-001",
    free_heap: int = 200_000,
    min_free_heap: int = 180_000,
    free_psram: int = 4_000_000,
    largest_free_block: int = 150_000,
    wifi_connected: bool = True,
    wifi_rssi: int = -50,
    display_initialized: bool = True,
    reboot_count: int = 0,
    i2c_errors: int = 0,
    **kwargs,
) -> HealthSnapshot:
    return HealthSnapshot(
        timestamp=_NOW,
        node_id=node_id,
        free_heap=free_heap,
        min_free_heap=min_free_heap,
        free_psram=free_psram,
        largest_free_block=largest_free_block,
        wifi_connected=wifi_connected,
        wifi_rssi=wifi_rssi,
        display_initialized=display_initialized,
        reboot_count=reboot_count,
        i2c_errors=i2c_errors,
        **kwargs,
    )


def _make_report(
    node_id: str = "esp32-001",
    anomalies: list[Anomaly] | None = None,
    **health_kwargs,
) -> NodeDiagReport:
    return NodeDiagReport(
        node_id=node_id,
        board_type="touch-lcd-35bc",
        firmware_version="1.0.0",
        current_health=_make_health(node_id=node_id, **health_kwargs),
        active_anomalies=anomalies or [],
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestSeverity:
    def test_values(self):
        assert Severity.TRACE.value == "trace"
        assert Severity.FATAL.value == "fatal"

    def test_all_values(self):
        expected = {"trace", "debug", "info", "warn", "error", "fatal"}
        assert {s.value for s in Severity} == expected


class TestAnomalyType:
    def test_values(self):
        assert AnomalyType.MEMORY_LEAK.value == "memory_leak"
        assert AnomalyType.REBOOT_LOOP.value == "reboot_loop"


class TestDiagEvent:
    def test_create_minimal(self):
        evt = DiagEvent(
            timestamp=_NOW,
            node_id="esp32-001",
            severity=Severity.WARN,
            subsystem="memory",
            message="Heap below threshold",
        )
        assert evt.severity == Severity.WARN
        assert evt.subsystem == "memory"
        assert evt.value is None

    def test_create_with_range(self):
        evt = DiagEvent(
            timestamp=_NOW,
            node_id="esp32-001",
            severity=Severity.ERROR,
            subsystem="power",
            message="Battery voltage low",
            value=3.1,
            expected_min=3.3,
            expected_max=4.2,
        )
        assert evt.value == 3.1
        assert evt.expected_min == 3.3

    def test_json_roundtrip(self):
        evt = DiagEvent(
            timestamp=_NOW,
            node_id="esp32-001",
            severity=Severity.INFO,
            subsystem="wifi",
            message="Connected",
        )
        evt2 = DiagEvent.model_validate_json(evt.model_dump_json())
        assert evt2.severity == Severity.INFO
        assert evt2.node_id == "esp32-001"


class TestHealthSnapshot:
    def test_create_minimal(self):
        h = _make_health()
        assert h.free_heap == 200_000
        assert h.wifi_connected is True
        assert h.battery_voltage is None

    def test_create_full(self):
        h = _make_health(
            battery_voltage=3.85,
            battery_percent=72.0,
            power_source="battery",
            cpu_temp_c=45.2,
            display_fps=30.0,
            uptime_s=3600,
            reset_reason="power_on",
        )
        assert h.battery_voltage == 3.85
        assert h.cpu_temp_c == 45.2
        assert h.uptime_s == 3600

    def test_json_roundtrip(self):
        h = _make_health(wifi_rssi=-65, free_heap=100_000)
        h2 = HealthSnapshot.model_validate_json(h.model_dump_json())
        assert h2.wifi_rssi == -65
        assert h2.free_heap == 100_000


class TestAnomaly:
    def test_create(self):
        a = Anomaly(
            timestamp=_NOW,
            node_id="esp32-001",
            anomaly_type=AnomalyType.MEMORY_LEAK,
            subsystem="memory",
            description="Heap declining steadily",
            severity_score=0.7,
        )
        assert a.anomaly_type == AnomalyType.MEMORY_LEAK
        assert a.severity_score == 0.7

    def test_severity_score_bounds(self):
        """severity_score must be in [0.0, 1.0]."""
        import pytest
        with pytest.raises(Exception):
            Anomaly(
                timestamp=_NOW,
                node_id="n1",
                anomaly_type=AnomalyType.MEMORY_LEAK,
                subsystem="memory",
                description="bad",
                severity_score=1.5,
            )
        with pytest.raises(Exception):
            Anomaly(
                timestamp=_NOW,
                node_id="n1",
                anomaly_type=AnomalyType.MEMORY_LEAK,
                subsystem="memory",
                description="bad",
                severity_score=-0.1,
            )

    def test_json_roundtrip(self):
        a = Anomaly(
            timestamp=_NOW,
            node_id="esp32-001",
            anomaly_type=AnomalyType.WIFI_DEGRADATION,
            subsystem="wifi",
            description="RSSI dropping",
            severity_score=0.5,
        )
        a2 = Anomaly.model_validate_json(a.model_dump_json())
        assert a2.anomaly_type == AnomalyType.WIFI_DEGRADATION
        assert a2.severity_score == 0.5


class TestNodeDiagReport:
    def test_create(self):
        report = _make_report()
        assert report.node_id == "esp32-001"
        assert report.recent_events == []
        assert report.active_anomalies == []

    def test_with_events_and_anomalies(self):
        evt = DiagEvent(
            timestamp=_NOW,
            node_id="esp32-001",
            severity=Severity.WARN,
            subsystem="memory",
            message="Low heap",
        )
        anomaly = Anomaly(
            timestamp=_NOW,
            node_id="esp32-001",
            anomaly_type=AnomalyType.MEMORY_LEAK,
            subsystem="memory",
            description="Heap declining",
            severity_score=0.6,
        )
        report = NodeDiagReport(
            node_id="esp32-001",
            board_type="touch-lcd-35bc",
            firmware_version="1.0.0",
            current_health=_make_health(),
            recent_events=[evt],
            active_anomalies=[anomaly],
        )
        assert len(report.recent_events) == 1
        assert len(report.active_anomalies) == 1

    def test_json_roundtrip(self):
        report = _make_report()
        report2 = NodeDiagReport.model_validate_json(report.model_dump_json())
        assert report2.node_id == report.node_id
        assert report2.current_health.free_heap == report.current_health.free_heap


class TestFleetHealthSummary:
    def test_create(self):
        summary = FleetHealthSummary(
            total_nodes=3, healthy_nodes=2, warning_nodes=1, critical_nodes=0,
        )
        assert summary.total_nodes == 3

    def test_health_score(self):
        summary = FleetHealthSummary(
            total_nodes=4, healthy_nodes=3, warning_nodes=1, critical_nodes=0,
        )
        assert summary.health_score == 0.75

    def test_health_score_empty(self):
        summary = FleetHealthSummary(
            total_nodes=0, healthy_nodes=0, warning_nodes=0, critical_nodes=0,
        )
        assert summary.health_score == 1.0

    def test_health_score_all_critical(self):
        summary = FleetHealthSummary(
            total_nodes=5, healthy_nodes=0, warning_nodes=0, critical_nodes=5,
        )
        assert summary.health_score == 0.0


# ---------------------------------------------------------------------------
# classify_node_health tests
# ---------------------------------------------------------------------------

class TestClassifyNodeHealth:
    def test_healthy_node(self):
        report = _make_report()
        assert classify_node_health(report) == "healthy"

    def test_critical_low_heap(self):
        report = _make_report(free_heap=15_000)
        assert classify_node_health(report) == "critical"

    def test_critical_display_down(self):
        report = _make_report(display_initialized=False)
        assert classify_node_health(report) == "critical"

    def test_critical_reboot_loop(self):
        report = _make_report(reboot_count=3)
        assert classify_node_health(report) == "critical"

    def test_critical_i2c_errors(self):
        report = _make_report(i2c_errors=25)
        assert classify_node_health(report) == "critical"

    def test_critical_high_severity_anomaly(self):
        anomaly = Anomaly(
            timestamp=_NOW,
            node_id="esp32-001",
            anomaly_type=AnomalyType.MEMORY_LEAK,
            subsystem="memory",
            description="Critical leak",
            severity_score=0.9,
        )
        report = _make_report(anomalies=[anomaly])
        assert classify_node_health(report) == "critical"

    def test_warning_medium_anomaly(self):
        anomaly = Anomaly(
            timestamp=_NOW,
            node_id="esp32-001",
            anomaly_type=AnomalyType.PERFORMANCE_DROP,
            subsystem="performance",
            description="Slow loops",
            severity_score=0.5,
        )
        report = _make_report(anomalies=[anomaly])
        assert classify_node_health(report) == "warning"

    def test_warning_low_heap(self):
        report = _make_report(free_heap=40_000)
        assert classify_node_health(report) == "warning"

    def test_warning_wifi_disconnected(self):
        report = _make_report(wifi_connected=False)
        assert classify_node_health(report) == "warning"

    def test_warning_i2c_errors(self):
        report = _make_report(i2c_errors=8)
        assert classify_node_health(report) == "warning"


# ---------------------------------------------------------------------------
# aggregate_fleet_health tests
# ---------------------------------------------------------------------------

class TestAggregateFleetHealth:
    def test_empty(self):
        summary = aggregate_fleet_health([])
        assert summary.total_nodes == 0
        assert summary.health_score == 1.0

    def test_all_healthy(self):
        reports = [_make_report(node_id=f"n{i}") for i in range(3)]
        summary = aggregate_fleet_health(reports)
        assert summary.total_nodes == 3
        assert summary.healthy_nodes == 3
        assert summary.warning_nodes == 0
        assert summary.critical_nodes == 0
        assert summary.health_score == 1.0

    def test_mixed(self):
        reports = [
            _make_report(node_id="n1"),  # healthy
            _make_report(node_id="n2", wifi_connected=False),  # warning
            _make_report(node_id="n3", free_heap=10_000),  # critical
        ]
        summary = aggregate_fleet_health(reports)
        assert summary.total_nodes == 3
        assert summary.healthy_nodes == 1
        assert summary.warning_nodes == 1
        assert summary.critical_nodes == 1

    def test_nodes_included(self):
        reports = [_make_report(node_id="n1"), _make_report(node_id="n2")]
        summary = aggregate_fleet_health(reports)
        assert len(summary.nodes) == 2
        assert summary.nodes[0].node_id == "n1"


# ---------------------------------------------------------------------------
# detect_fleet_anomalies tests
# ---------------------------------------------------------------------------

class TestDetectFleetAnomalies:
    def test_empty(self):
        assert detect_fleet_anomalies([]) == []

    def test_no_anomalies_healthy_fleet(self):
        reports = [_make_report(node_id=f"n{i}") for i in range(4)]
        anomalies = detect_fleet_anomalies(reports)
        assert anomalies == []

    def test_wifi_infrastructure_issue(self):
        """More than half nodes with bad WiFi triggers fleet anomaly."""
        reports = [
            _make_report(node_id="n1", wifi_connected=False),
            _make_report(node_id="n2", wifi_connected=False),
            _make_report(node_id="n3", wifi_connected=False),
            _make_report(node_id="n4"),  # healthy
        ]
        anomalies = detect_fleet_anomalies(reports)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.WIFI_DEGRADATION
        assert anomalies[0].node_id == "fleet"
        assert "3/4" in anomalies[0].description

    def test_wifi_bad_rssi(self):
        """Nodes connected but with very weak signal count as degraded."""
        reports = [
            _make_report(node_id="n1", wifi_rssi=-85),
            _make_report(node_id="n2", wifi_rssi=-90),
            _make_report(node_id="n3"),  # good
        ]
        anomalies = detect_fleet_anomalies(reports)
        wifi_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.WIFI_DEGRADATION]
        assert len(wifi_anomalies) == 1

    def test_widespread_reboots(self):
        reports = [
            _make_report(node_id="n1", reboot_count=3),
            _make_report(node_id="n2", reboot_count=5),
            _make_report(node_id="n3", reboot_count=2),
            _make_report(node_id="n4", reboot_count=0),
        ]
        anomalies = detect_fleet_anomalies(reports)
        reboot_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.REBOOT_LOOP]
        assert len(reboot_anomalies) == 1
        assert "3/4" in reboot_anomalies[0].description

    def test_i2c_failures(self):
        reports = [
            _make_report(node_id="n1", i2c_errors=10),
            _make_report(node_id="n2", i2c_errors=15),
            _make_report(node_id="n3", i2c_errors=0),
        ]
        anomalies = detect_fleet_anomalies(reports)
        i2c_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.I2C_FAILURE]
        assert len(i2c_anomalies) == 1

    def test_multiple_fleet_anomalies(self):
        """A really bad fleet can trigger multiple anomaly types."""
        reports = [
            _make_report(node_id="n1", wifi_connected=False, reboot_count=5, i2c_errors=20),
            _make_report(node_id="n2", wifi_connected=False, reboot_count=3, i2c_errors=10),
            _make_report(node_id="n3", wifi_connected=False, reboot_count=4, i2c_errors=8),
        ]
        anomalies = detect_fleet_anomalies(reports)
        types = {a.anomaly_type for a in anomalies}
        assert AnomalyType.WIFI_DEGRADATION in types
        assert AnomalyType.REBOOT_LOOP in types
        assert AnomalyType.I2C_FAILURE in types

    def test_below_threshold_no_anomaly(self):
        """Exactly half (not more than) should not trigger."""
        reports = [
            _make_report(node_id="n1", wifi_connected=False),
            _make_report(node_id="n2"),
        ]
        anomalies = detect_fleet_anomalies(reports)
        wifi_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.WIFI_DEGRADATION]
        assert len(wifi_anomalies) == 0

    def test_severity_score_capped(self):
        """Severity score should not exceed 1.0."""
        reports = [
            _make_report(node_id=f"n{i}", wifi_connected=False)
            for i in range(10)
        ]
        anomalies = detect_fleet_anomalies(reports)
        for a in anomalies:
            assert 0.0 <= a.severity_score <= 1.0


# ── DiagLogEntry ──────────────────────────────────────────────────────────


class TestDiagLogEntry:
    """DiagLogEntry — persistent diagnostic log entry model."""

    def test_create_minimal(self):
        entry = DiagLogEntry(
            timestamp=1709836800,
            severity=Severity.INFO,
            subsystem="wifi",
        )
        assert entry.timestamp == 1709836800
        assert entry.severity == Severity.INFO
        assert entry.subsystem == "wifi"
        assert entry.code == 0
        assert entry.message == ""
        assert entry.value == 0.0

    def test_create_full(self):
        entry = DiagLogEntry(
            timestamp=1709836800,
            severity=Severity.ERROR,
            subsystem="i2c",
            code=3,
            message="Bus timeout on addr 0x34",
            value=52.0,
        )
        assert entry.code == 3
        assert entry.message == "Bus timeout on addr 0x34"
        assert entry.value == 52.0

    def test_severity_values(self):
        for sev in Severity:
            entry = DiagLogEntry(
                timestamp=0, severity=sev, subsystem="test"
            )
            assert entry.severity == sev


class TestDiagLogBatch:
    """DiagLogBatch — batch upload model."""

    def test_empty_batch(self):
        batch = DiagLogBatch(device_id="node-1")
        assert batch.device_id == "node-1"
        assert batch.boot_count == 0
        assert batch.events == []

    def test_batch_with_events(self):
        events = [
            DiagLogEntry(timestamp=100, severity=Severity.WARN, subsystem="wifi", message="RSSI low"),
            DiagLogEntry(timestamp=200, severity=Severity.ERROR, subsystem="i2c", code=1, message="Timeout"),
        ]
        batch = DiagLogBatch(device_id="node-2", boot_count=5, events=events)
        assert len(batch.events) == 2
        assert batch.boot_count == 5


class TestSummarizeDiagLog:
    """summarize_diag_log — aggregation function."""

    def test_empty(self):
        summary = summarize_diag_log([])
        assert summary.total_events == 0
        assert summary.total_devices == 0
        assert summary.events_by_severity == {}
        assert summary.events_by_subsystem == {}

    def test_single_device(self):
        entries = [
            DiagLogEntry(timestamp=100, severity=Severity.WARN, subsystem="wifi"),
            DiagLogEntry(timestamp=200, severity=Severity.ERROR, subsystem="i2c", code=1),
            DiagLogEntry(timestamp=300, severity=Severity.WARN, subsystem="wifi"),
        ]
        summary = summarize_diag_log(entries, device_ids=["node-1", "node-1", "node-1"])
        assert summary.total_events == 3
        assert summary.total_devices == 1
        assert summary.events_by_severity["warn"] == 2
        assert summary.events_by_severity["error"] == 1
        assert summary.events_by_subsystem["wifi"] == 2
        assert summary.events_by_subsystem["i2c"] == 1

    def test_multiple_devices(self):
        entries = [
            DiagLogEntry(timestamp=100, severity=Severity.INFO, subsystem="memory"),
            DiagLogEntry(timestamp=200, severity=Severity.ERROR, subsystem="power"),
        ]
        summary = summarize_diag_log(entries, device_ids=["node-1", "node-2"])
        assert summary.total_devices == 2
        assert summary.total_events == 2

    def test_most_frequent_codes(self):
        entries = [
            DiagLogEntry(timestamp=i, severity=Severity.WARN, subsystem="i2c", code=1)
            for i in range(10)
        ] + [
            DiagLogEntry(timestamp=100, severity=Severity.ERROR, subsystem="wifi", code=2)
        ]
        summary = summarize_diag_log(entries)
        assert len(summary.most_frequent_codes) >= 1
        assert summary.most_frequent_codes[0]["count"] == 10
        assert summary.most_frequent_codes[0]["subsystem_code"] == "i2c:1"

    def test_top_codes_limited_to_10(self):
        entries = [
            DiagLogEntry(timestamp=i, severity=Severity.INFO, subsystem=f"sub{i}", code=i)
            for i in range(20)
        ]
        summary = summarize_diag_log(entries)
        assert len(summary.most_frequent_codes) <= 10


# ---------------------------------------------------------------------------
# I2cSlaveHealth model tests
# ---------------------------------------------------------------------------

class TestI2cSlaveHealth:
    def test_defaults(self):
        s = I2cSlaveHealth(addr="0x34")
        assert s.nack_count == 0
        assert s.timeout_count == 0
        assert s.success_count == 0
        assert s.last_latency_us == 0

    def test_total_transactions(self):
        s = I2cSlaveHealth(addr="0x6B", success_count=90, nack_count=5, timeout_count=5)
        assert s.total_transactions == 100

    def test_success_rate(self):
        s = I2cSlaveHealth(addr="0x6B", success_count=90, nack_count=5, timeout_count=5)
        assert s.success_rate == 0.9

    def test_success_rate_no_transactions(self):
        s = I2cSlaveHealth(addr="0x34")
        assert s.success_rate == 1.0

    def test_error_count(self):
        s = I2cSlaveHealth(addr="0x18", nack_count=3, timeout_count=7)
        assert s.error_count == 10

    def test_perfect_health(self):
        s = I2cSlaveHealth(addr="0x51", success_count=1000)
        assert s.success_rate == 1.0
        assert s.error_count == 0


# ---------------------------------------------------------------------------
# Per-slave I2C in classify_node_health
# ---------------------------------------------------------------------------

class TestClassifyNodeHealthI2cSlaves:
    def test_healthy_with_good_slaves(self):
        health = _make_health(i2c_slaves=[
            I2cSlaveHealth(addr="0x34", success_count=100, nack_count=1),
            I2cSlaveHealth(addr="0x6B", success_count=100, nack_count=2),
        ])
        report = NodeDiagReport(
            node_id="esp32-001", board_type="touch-lcd-35bc",
            firmware_version="1.0.0", current_health=health,
        )
        assert classify_node_health(report) == "healthy"

    def test_critical_slave_below_90_percent(self):
        health = _make_health(i2c_slaves=[
            I2cSlaveHealth(addr="0x34", success_count=80, nack_count=15, timeout_count=5),
        ])
        report = NodeDiagReport(
            node_id="esp32-001", board_type="touch-lcd-35bc",
            firmware_version="1.0.0", current_health=health,
        )
        assert classify_node_health(report) == "critical"

    def test_warning_slave_below_95_percent(self):
        health = _make_health(i2c_slaves=[
            I2cSlaveHealth(addr="0x6B", success_count=93, nack_count=4, timeout_count=3),
        ])
        report = NodeDiagReport(
            node_id="esp32-001", board_type="touch-lcd-35bc",
            firmware_version="1.0.0", current_health=health,
        )
        assert classify_node_health(report) == "warning"

    def test_slave_with_few_transactions_ignored(self):
        """Slaves with <= 10 transactions don't trigger thresholds."""
        health = _make_health(i2c_slaves=[
            I2cSlaveHealth(addr="0x34", success_count=1, nack_count=5, timeout_count=4),
        ])
        report = NodeDiagReport(
            node_id="esp32-001", board_type="touch-lcd-35bc",
            firmware_version="1.0.0", current_health=health,
        )
        assert classify_node_health(report) == "healthy"

    def test_display_timing_fields(self):
        health = _make_health(display_frame_us=5000, display_max_frame_us=12000)
        assert health.display_frame_us == 5000
        assert health.display_max_frame_us == 12000

    def test_display_timing_defaults(self):
        health = _make_health()
        assert health.display_frame_us is None
        assert health.display_max_frame_us is None


# ---------------------------------------------------------------------------
# Heap trend analysis tests
# ---------------------------------------------------------------------------

class TestAnalyzeHeapTrends:
    def test_empty_input(self):
        assert analyze_heap_trends([]) == []

    def test_single_sample_ignored(self):
        """Need at least 2 samples per device."""
        result = analyze_heap_trends([
            {"device_id": "n1", "free_heap": 100000, "uptime_s": 0},
        ])
        assert result == []

    def test_stable_heap(self):
        snapshots = [
            {"device_id": "n1", "free_heap": 100000, "uptime_s": 0},
            {"device_id": "n1", "free_heap": 99000, "uptime_s": 3600},
        ]
        result = analyze_heap_trends(snapshots)
        assert len(result) == 1
        assert result[0].device_id == "n1"
        assert result[0].delta == -1000
        assert not result[0].leak_suspected

    def test_leak_detected(self):
        snapshots = [
            {"device_id": "n1", "free_heap": 200000, "uptime_s": 0},
            {"device_id": "n1", "free_heap": 180000, "uptime_s": 3600},
        ]
        result = analyze_heap_trends(snapshots)
        assert len(result) == 1
        assert result[0].delta == -20000
        assert result[0].delta_per_hour == -20000.0
        assert result[0].leak_suspected  # -20000/h < -5000/h threshold

    def test_multiple_devices(self):
        snapshots = [
            {"device_id": "n1", "free_heap": 100000, "uptime_s": 0},
            {"device_id": "n2", "free_heap": 200000, "uptime_s": 0},
            {"device_id": "n1", "free_heap": 95000, "uptime_s": 7200},
            {"device_id": "n2", "free_heap": 150000, "uptime_s": 7200},
        ]
        result = analyze_heap_trends(snapshots)
        assert len(result) == 2
        by_id = {r.device_id: r for r in result}
        assert not by_id["n1"].leak_suspected  # -2500/h
        assert by_id["n2"].leak_suspected      # -25000/h

    def test_custom_threshold(self):
        snapshots = [
            {"device_id": "n1", "free_heap": 100000, "uptime_s": 0},
            {"device_id": "n1", "free_heap": 98000, "uptime_s": 3600},
        ]
        # With default threshold (-5000/h) this is fine
        result = analyze_heap_trends(snapshots)
        assert not result[0].leak_suspected
        # With stricter threshold (-1000/h) this is a leak
        result = analyze_heap_trends(snapshots, leak_threshold_per_hour=-1000)
        assert result[0].leak_suspected

    def test_min_heap_tracked(self):
        snapshots = [
            {"device_id": "n1", "free_heap": 100000, "uptime_s": 0},
            {"device_id": "n1", "free_heap": 50000, "uptime_s": 1800},
            {"device_id": "n1", "free_heap": 90000, "uptime_s": 3600},
        ]
        result = analyze_heap_trends(snapshots)
        assert result[0].min_heap == 50000
        assert result[0].samples == 3


# ---------------------------------------------------------------------------
# Touch / NTP field tests
# ---------------------------------------------------------------------------

class TestHealthSnapshotExtended:
    def test_touch_default(self):
        h = _make_health()
        assert h.touch_available is False

    def test_touch_available(self):
        h = _make_health(touch_available=True)
        assert h.touch_available is True

    def test_ntp_default(self):
        h = _make_health()
        assert h.ntp_synced is False
        assert h.ntp_last_sync_age_s == 0

    def test_ntp_synced(self):
        h = _make_health(ntp_synced=True, ntp_last_sync_age_s=120)
        assert h.ntp_synced is True
        assert h.ntp_last_sync_age_s == 120


# ---------------------------------------------------------------------------
# MeshPeer model tests
# ---------------------------------------------------------------------------

class TestMeshPeer:
    def test_create(self):
        p = MeshPeer(mac="AA:BB:CC:DD:EE:FF", rssi=-45, hops=0)
        assert p.mac == "AA:BB:CC:DD:EE:FF"
        assert p.rssi == -45
        assert p.hops == 0

    def test_defaults(self):
        p = MeshPeer(mac="11:22:33:44:55:66")
        assert p.rssi == 0
        assert p.hops == 0

    def test_json_roundtrip(self):
        p = MeshPeer(mac="AA:BB:CC:DD:EE:FF", rssi=-60, hops=1)
        p2 = MeshPeer.model_validate_json(p.model_dump_json())
        assert p2.mac == "AA:BB:CC:DD:EE:FF"
        assert p2.rssi == -60
        assert p2.hops == 1

    def test_mesh_peer_list_in_health(self):
        peers = [
            MeshPeer(mac="AA:BB:CC:DD:EE:FF", rssi=-45, hops=0),
            MeshPeer(mac="11:22:33:44:55:66", rssi=-70, hops=1),
        ]
        h = _make_health(mesh_peer_list=peers, mesh_peers=2)
        assert len(h.mesh_peer_list) == 2
        assert h.mesh_peer_list[0].mac == "AA:BB:CC:DD:EE:FF"
        assert h.mesh_peers == 2

    def test_mesh_peer_list_default_empty(self):
        h = _make_health()
        assert h.mesh_peer_list == []

    def test_health_with_mesh_peers_json_roundtrip(self):
        peers = [MeshPeer(mac="AA:BB:CC:DD:EE:FF", rssi=-45, hops=0)]
        h = _make_health(mesh_peer_list=peers, mesh_peers=1)
        h2 = HealthSnapshot.model_validate_json(h.model_dump_json())
        assert len(h2.mesh_peer_list) == 1
        assert h2.mesh_peer_list[0].mac == "AA:BB:CC:DD:EE:FF"


# ---------------------------------------------------------------------------
# Mesh isolation in classify_node_health
# ---------------------------------------------------------------------------

class TestClassifyNodeHealthMeshIsolation:
    def test_mesh_active_no_peers_is_warning(self):
        """A node that has mesh traffic but zero peers is isolated."""
        report = _make_report(mesh_tx=50, mesh_rx=10, mesh_peers=0)
        assert classify_node_health(report) == "warning"

    def test_mesh_active_with_peers_is_healthy(self):
        """A node with mesh traffic and peers is fine."""
        peers = [MeshPeer(mac="AA:BB:CC:DD:EE:FF", rssi=-45, hops=0)]
        report = _make_report(mesh_tx=50, mesh_rx=10, mesh_peers=1,
                              mesh_peer_list=peers)
        assert classify_node_health(report) == "healthy"

    def test_mesh_inactive_no_peers_still_healthy(self):
        """A node not using mesh at all (no tx/rx/routes) stays healthy."""
        report = _make_report(mesh_tx=0, mesh_rx=0, mesh_routes=0, mesh_peers=0)
        assert classify_node_health(report) == "healthy"

    def test_mesh_routes_but_no_peers_is_warning(self):
        """A node with routes configured but no peers is isolated."""
        report = _make_report(mesh_routes=3, mesh_peers=0)
        assert classify_node_health(report) == "warning"

    def test_mesh_peer_list_counts_as_peers(self):
        """mesh_peers=0 but mesh_peer_list populated should not be warning."""
        peers = [MeshPeer(mac="AA:BB:CC:DD:EE:FF", rssi=-45, hops=0)]
        report = _make_report(mesh_tx=10, mesh_rx=5, mesh_peers=0,
                              mesh_peer_list=peers)
        assert classify_node_health(report) == "healthy"

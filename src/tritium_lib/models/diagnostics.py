# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Diagnostic and telemetry models for fleet health monitoring.

These models support the remote diagnostics system. ESP32 nodes send
health snapshots, diagnostic events, and anomaly reports to the fleet
server. Both the ESP32 JSON serializer and the Python server use these
shared types.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Log severity level for diagnostic events."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


class AnomalyType(str, Enum):
    """Categories of auto-detected anomalies."""
    MEMORY_LEAK = "memory_leak"
    BATTERY_DRAIN = "battery_drain"
    WIFI_DEGRADATION = "wifi_degradation"
    PERFORMANCE_DROP = "performance_drop"
    I2C_FAILURE = "i2c_failure"
    DISPLAY_FAILURE = "display_failure"
    TEMPERATURE_HIGH = "temperature_high"
    REBOOT_LOOP = "reboot_loop"


class DiagEvent(BaseModel):
    """A single diagnostic event from a node."""
    timestamp: datetime
    node_id: str
    severity: Severity
    subsystem: str  # display, power, imu, wifi, ble, memory, etc.
    message: str
    value: Optional[float] = None
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None


class CrashInfo(BaseModel):
    """Crash info from a previous device boot, stored in NVS."""
    epoch_time: int = 0
    uptime_ms: int = 0
    free_heap: int = 0
    reset_reason: str = ""
    message: str = ""
    task_name: str = ""


class DiagLogEntry(BaseModel):
    """A single persistent diagnostic log entry from firmware ring buffer.

    This is the on-wire format used when the ESP32 uploads its diagnostic
    event log to the fleet server.  Uses epoch timestamps (int) since the
    firmware stores them as uint32_t.
    """
    timestamp: int  # Unix epoch seconds
    severity: Severity
    subsystem: str  # i2c, wifi, power, display, memory, spi, etc.
    code: int = 0  # Subsystem-specific event code
    message: str = ""  # Human-readable description (max ~80 chars on device)
    value: float = 0.0  # Optional numeric value (e.g., heap bytes, voltage)


class DiagLogBatch(BaseModel):
    """Batch upload of diagnostic log events from a device."""
    device_id: str
    boot_count: int = 0
    events: list[DiagLogEntry] = Field(default_factory=list)


class DiagLogSummary(BaseModel):
    """Fleet-wide diagnostic log summary statistics."""
    total_events: int = 0
    total_devices: int = 0
    events_by_severity: dict[str, int] = Field(default_factory=dict)
    events_by_subsystem: dict[str, int] = Field(default_factory=dict)
    devices_with_criticals: list[str] = Field(default_factory=list)
    most_frequent_codes: list[dict] = Field(default_factory=list)


def summarize_diag_log(
    entries: list[DiagLogEntry],
    device_ids: list[str] | None = None,
) -> DiagLogSummary:
    """Build a summary from a collection of diagnostic log entries.

    Args:
        entries: All log entries to summarize.
        device_ids: Optional list of device IDs that contributed entries.

    Returns:
        Aggregated summary statistics.
    """
    by_severity: dict[str, int] = {}
    by_subsystem: dict[str, int] = {}
    code_counts: dict[str, int] = {}

    for e in entries:
        sev = e.severity.value if isinstance(e.severity, Severity) else str(e.severity)
        by_severity[sev] = by_severity.get(sev, 0) + 1
        by_subsystem[e.subsystem] = by_subsystem.get(e.subsystem, 0) + 1
        key = f"{e.subsystem}:{e.code}"
        code_counts[key] = code_counts.get(key, 0) + 1

    # Top 10 most frequent event codes
    top_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    most_frequent = [
        {"subsystem_code": k, "count": v} for k, v in top_codes
    ]

    return DiagLogSummary(
        total_events=len(entries),
        total_devices=len(set(device_ids)) if device_ids else 0,
        events_by_severity=by_severity,
        events_by_subsystem=by_subsystem,
        most_frequent_codes=most_frequent,
    )


class I2cSlaveHealth(BaseModel):
    """Per-slave I2C bus health metrics."""
    addr: str  # I2C address as hex string (e.g., "0x34")
    nack_count: int = 0
    timeout_count: int = 0
    success_count: int = 0
    last_latency_us: int = 0

    @property
    def total_transactions(self) -> int:
        return self.success_count + self.nack_count + self.timeout_count

    @property
    def success_rate(self) -> float:
        total = self.total_transactions
        return self.success_count / total if total > 0 else 1.0

    @property
    def error_count(self) -> int:
        return self.nack_count + self.timeout_count


class HealthSnapshot(BaseModel):
    """Periodic hardware health snapshot from a node."""
    timestamp: datetime
    node_id: str
    # Memory
    free_heap: int
    min_free_heap: int
    free_psram: int
    largest_free_block: int
    # Power
    battery_voltage: Optional[float] = None
    battery_percent: Optional[float] = None
    power_source: Optional[str] = None
    # Temperature
    cpu_temp_c: Optional[float] = None
    pmic_temp_c: Optional[float] = None
    # Display
    display_initialized: bool = True
    display_fps: Optional[float] = None
    display_frame_us: Optional[int] = None
    display_max_frame_us: Optional[int] = None
    # Connectivity
    wifi_rssi: Optional[int] = None
    wifi_connected: bool = False
    wifi_disconnects: int = 0
    # I2C
    i2c_devices_found: int = 0
    i2c_errors: int = 0
    i2c_slaves: list[I2cSlaveHealth] = Field(default_factory=list)
    # Camera
    camera_available: bool = False
    camera_frames: int = 0
    camera_fails: int = 0
    camera_last_us: int = 0
    camera_max_us: int = 0
    camera_avg_fps: float = 0.0
    # Touch
    touch_available: bool = False
    # NTP
    ntp_synced: bool = False
    ntp_last_sync_age_s: int = 0
    # Mesh networking
    mesh_peers: int = 0
    mesh_routes: int = 0
    mesh_tx: int = 0
    mesh_rx: int = 0
    mesh_tx_fail: int = 0
    mesh_relayed: int = 0
    # Performance
    loop_time_us: int = 0
    max_loop_time_us: int = 0
    uptime_s: int = 0
    reboot_count: int = 0
    reset_reason: Optional[str] = None


class Anomaly(BaseModel):
    """Auto-detected anomaly from health trend analysis."""
    timestamp: datetime
    node_id: str
    anomaly_type: AnomalyType
    subsystem: str
    description: str
    severity_score: float = Field(ge=0.0, le=1.0)


class NodeDiagReport(BaseModel):
    """Full diagnostic report from a single node."""
    node_id: str
    board_type: str
    firmware_version: str
    current_health: HealthSnapshot
    recent_events: list[DiagEvent] = Field(default_factory=list)
    active_anomalies: list[Anomaly] = Field(default_factory=list)


class FleetHealthSummary(BaseModel):
    """Aggregated fleet health across all nodes."""
    total_nodes: int
    healthy_nodes: int
    warning_nodes: int
    critical_nodes: int
    nodes: list[NodeDiagReport] = Field(default_factory=list)

    @property
    def health_score(self) -> float:
        if self.total_nodes == 0:
            return 1.0
        return self.healthy_nodes / self.total_nodes


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Thresholds for node health classification
_CRITICAL_HEAP_BYTES = 20_000
_WARNING_HEAP_BYTES = 50_000
_CRITICAL_ANOMALY_SCORE = 0.8
_WARNING_ANOMALY_SCORE = 0.4
_WARNING_I2C_ERRORS = 5
_CRITICAL_I2C_ERRORS = 20
_REBOOT_LOOP_THRESHOLD = 3


def classify_node_health(report: NodeDiagReport) -> str:
    """Classify a node as ``"healthy"``, ``"warning"``, or ``"critical"``.

    Classification rules (first match wins):

    * **critical** — any anomaly with severity_score >= 0.8, free heap below
      20 KB, display not initialized, reboot_count >= 3, or I2C errors >= 20.
    * **warning** — any anomaly with severity_score >= 0.4, free heap below
      50 KB, WiFi disconnected, or I2C errors >= 5.
    * **healthy** — everything else.
    """
    health = report.current_health

    # --- Critical checks ---
    for anomaly in report.active_anomalies:
        if anomaly.severity_score >= _CRITICAL_ANOMALY_SCORE:
            return "critical"

    if health.free_heap < _CRITICAL_HEAP_BYTES:
        return "critical"

    if not health.display_initialized:
        return "critical"

    if health.reboot_count >= _REBOOT_LOOP_THRESHOLD:
        return "critical"

    if health.i2c_errors >= _CRITICAL_I2C_ERRORS:
        return "critical"

    # Per-slave I2C health: any slave below 90% success rate is critical
    for slave in health.i2c_slaves:
        if slave.total_transactions > 10 and slave.success_rate < 0.90:
            return "critical"

    # --- Warning checks ---
    for anomaly in report.active_anomalies:
        if anomaly.severity_score >= _WARNING_ANOMALY_SCORE:
            return "warning"

    if health.free_heap < _WARNING_HEAP_BYTES:
        return "warning"

    if not health.wifi_connected:
        return "warning"

    if health.i2c_errors >= _WARNING_I2C_ERRORS:
        return "warning"

    # Per-slave I2C health: any slave below 95% success rate is warning
    for slave in health.i2c_slaves:
        if slave.total_transactions > 10 and slave.success_rate < 0.95:
            return "warning"

    return "healthy"


def aggregate_fleet_health(
    reports: list[NodeDiagReport],
) -> FleetHealthSummary:
    """Build a :class:`FleetHealthSummary` from individual node reports."""
    healthy = warning = critical = 0
    for report in reports:
        status = classify_node_health(report)
        if status == "healthy":
            healthy += 1
        elif status == "warning":
            warning += 1
        else:
            critical += 1

    return FleetHealthSummary(
        total_nodes=len(reports),
        healthy_nodes=healthy,
        warning_nodes=warning,
        critical_nodes=critical,
        nodes=reports,
    )


def detect_fleet_anomalies(
    reports: list[NodeDiagReport],
) -> list[Anomaly]:
    """Detect cross-node anomalies that indicate infrastructure issues.

    Current detectors:

    * **WiFi infrastructure** — if more than half the nodes have WiFi
      disconnected or RSSI below -80, emit a single WIFI_DEGRADATION
      anomaly attributed to infrastructure rather than individual nodes.
    * **Widespread reboots** — if more than half the nodes have
      ``reboot_count >= 2``, flag as a potential power or firmware issue.
    * **I2C bus failures** — if more than half report I2C errors, flag as
      a possible environmental/electrical issue.
    """
    if not reports:
        return []

    anomalies: list[Anomaly] = []
    n = len(reports)
    threshold = n / 2

    # Use the latest timestamp from reports for anomaly timestamps
    latest_ts = max(r.current_health.timestamp for r in reports)

    # --- WiFi infrastructure degradation ---
    wifi_bad = sum(
        1 for r in reports
        if not r.current_health.wifi_connected
        or (r.current_health.wifi_rssi is not None and r.current_health.wifi_rssi < -80)
    )
    if wifi_bad > threshold:
        anomalies.append(Anomaly(
            timestamp=latest_ts,
            node_id="fleet",
            anomaly_type=AnomalyType.WIFI_DEGRADATION,
            subsystem="wifi",
            description=(
                f"{wifi_bad}/{n} nodes have degraded WiFi — "
                "likely infrastructure issue, not individual nodes"
            ),
            severity_score=min(1.0, wifi_bad / n),
        ))

    # --- Widespread reboots ---
    reboot_nodes = sum(
        1 for r in reports if r.current_health.reboot_count >= 2
    )
    if reboot_nodes > threshold:
        anomalies.append(Anomaly(
            timestamp=latest_ts,
            node_id="fleet",
            anomaly_type=AnomalyType.REBOOT_LOOP,
            subsystem="power",
            description=(
                f"{reboot_nodes}/{n} nodes rebooting frequently — "
                "possible power or firmware issue"
            ),
            severity_score=min(1.0, reboot_nodes / n),
        ))

    # --- I2C bus failures ---
    i2c_bad = sum(
        1 for r in reports if r.current_health.i2c_errors >= 5
    )
    if i2c_bad > threshold:
        anomalies.append(Anomaly(
            timestamp=latest_ts,
            node_id="fleet",
            anomaly_type=AnomalyType.I2C_FAILURE,
            subsystem="i2c",
            description=(
                f"{i2c_bad}/{n} nodes reporting I2C errors — "
                "possible environmental or electrical issue"
            ),
            severity_score=min(1.0, i2c_bad / n),
        ))

    return anomalies


# ---------------------------------------------------------------------------
# Heap trend analysis
# ---------------------------------------------------------------------------

class HeapTrend(BaseModel):
    """Heap usage trend for a single device over time."""
    device_id: str
    samples: int = 0
    first_heap: int = 0
    last_heap: int = 0
    min_heap: int = 0
    delta: int = 0          # last - first (negative = shrinking)
    delta_per_hour: float = 0.0  # Normalized rate
    leak_suspected: bool = False


def analyze_heap_trends(
    snapshots: list[dict],
    leak_threshold_per_hour: int = -5000,
) -> list[HeapTrend]:
    """Analyze heap usage trends across device snapshots.

    Args:
        snapshots: List of dicts with keys: device_id, free_heap, uptime_s.
            Should be sorted chronologically per device.
        leak_threshold_per_hour: Heap delta per hour below which a leak is
            suspected (negative value, e.g. -5000 = losing 5KB/hour).

    Returns:
        Per-device heap trend analysis.
    """
    from collections import defaultdict

    # Group by device
    by_device: dict[str, list[dict]] = defaultdict(list)
    for s in snapshots:
        did = s.get("device_id", "")
        if did:
            by_device[did].append(s)

    results = []
    for device_id, samples in by_device.items():
        if len(samples) < 2:
            continue

        first = samples[0]
        last = samples[-1]
        first_heap = first.get("free_heap", 0)
        last_heap = last.get("free_heap", 0)
        min_heap = min(s.get("free_heap", 0) for s in samples)
        delta = last_heap - first_heap

        # Calculate time span
        first_uptime = first.get("uptime_s", 0)
        last_uptime = last.get("uptime_s", 0)
        elapsed_h = (last_uptime - first_uptime) / 3600.0 if last_uptime > first_uptime else 0.0

        delta_per_hour = delta / elapsed_h if elapsed_h > 0.5 else 0.0

        results.append(HeapTrend(
            device_id=device_id,
            samples=len(samples),
            first_heap=first_heap,
            last_heap=last_heap,
            min_heap=min_heap,
            delta=delta,
            delta_per_hour=round(delta_per_hour, 1),
            leak_suspected=delta_per_hour < leak_threshold_per_hour,
        ))

    return results

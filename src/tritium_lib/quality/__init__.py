# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.quality — data quality monitoring for sensor feeds.

Tracks five quality dimensions for each sensor/source:
  - **Completeness**: are all expected fields present in incoming data?
  - **Timeliness**: is data arriving within expected latency bounds?
  - **Accuracy**: are values within expected ranges?
  - **Consistency**: are different sensors reporting agreeing values?
  - **Freshness**: has the sensor gone stale (no data for too long)?

Core classes:
  - ``QualityDimension`` — enum of the five quality dimensions
  - ``QualityMetric`` — a single quality measurement for one dimension
  - ``QualityAlert`` — fired when quality drops below a threshold
  - ``QualityReport`` — periodic quality assessment for one source
  - ``DataQualityMonitor`` — the main monitor; tracks all sources,
    evaluates incoming data, emits alerts via EventBus

Quick start::

    from tritium_lib.quality import DataQualityMonitor, QualityDimension
    from tritium_lib.events import EventBus

    bus = EventBus()
    monitor = DataQualityMonitor(event_bus=bus)

    # Define expected schema for a BLE sensor
    monitor.register_source("ble_sensor_01", expected_fields=["mac", "rssi", "timestamp"])

    # Record an incoming reading
    monitor.record(
        source_id="ble_sensor_01",
        data={"mac": "AA:BB:CC:DD:EE:FF", "rssi": -65, "timestamp": 1711330000.0},
    )

    # Get a quality report
    report = monitor.get_report("ble_sensor_01")
    print(report.overall_score)  # 0.0–1.0
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("quality")


# ---------------------------------------------------------------------------
# QualityDimension — the five quality axes
# ---------------------------------------------------------------------------

class QualityDimension(str, Enum):
    """The five data quality dimensions tracked by the monitor."""
    COMPLETENESS = "completeness"
    TIMELINESS = "timeliness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    FRESHNESS = "freshness"


# ---------------------------------------------------------------------------
# QualityMetric — a single measurement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityMetric:
    """A single quality measurement for one dimension of one source.

    Attributes:
        dimension: Which quality dimension this measures.
        source_id: The sensor/source this metric is for.
        score: Quality score from 0.0 (terrible) to 1.0 (perfect).
        detail: Human-readable explanation of the score.
        timestamp: When this metric was computed.
    """
    dimension: QualityDimension
    source_id: str
    score: float
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# QualityAlert — triggered when quality drops below threshold
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityAlert:
    """Fired when a source's quality drops below the configured threshold.

    Attributes:
        alert_id: Unique identifier for this alert instance.
        source_id: The sensor/source whose quality triggered the alert.
        dimension: Which quality dimension failed.
        score: The score that triggered the alert.
        threshold: The threshold that was violated.
        message: Human-readable alert description.
        timestamp: When the alert was created.
    """
    alert_id: str
    source_id: str
    dimension: QualityDimension
    score: float
    threshold: float
    message: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# QualityReport — periodic quality assessment for one source
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Periodic quality assessment for a single source.

    Attributes:
        source_id: The sensor/source assessed.
        metrics: Per-dimension quality metrics (latest).
        overall_score: Weighted average of dimension scores (0.0–1.0).
        sample_count: How many data records were evaluated in this period.
        alerts: Any alerts generated during this assessment.
        timestamp: When this report was generated.
    """
    source_id: str
    metrics: dict[QualityDimension, QualityMetric] = field(default_factory=dict)
    overall_score: float = 1.0
    sample_count: int = 0
    alerts: list[QualityAlert] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def export(self) -> dict[str, Any]:
        """Export as a JSON-serializable dict."""
        return {
            "source_id": self.source_id,
            "overall_score": round(self.overall_score, 4),
            "sample_count": self.sample_count,
            "metrics": {
                dim.value: {
                    "score": round(m.score, 4),
                    "detail": m.detail,
                    "timestamp": m.timestamp,
                }
                for dim, m in self.metrics.items()
            },
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "dimension": a.dimension.value,
                    "score": round(a.score, 4),
                    "threshold": a.threshold,
                    "message": a.message,
                    "timestamp": a.timestamp,
                }
                for a in self.alerts
            ],
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# _SourceTracker — internal per-source state
# ---------------------------------------------------------------------------

@dataclass
class _SourceTracker:
    """Internal bookkeeping for one monitored source."""
    source_id: str
    expected_fields: list[str] = field(default_factory=list)
    value_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    max_latency_seconds: float = 30.0
    max_staleness_seconds: float = 300.0

    # Rolling buffers
    arrival_times: deque = field(default_factory=lambda: deque(maxlen=500))
    completeness_scores: deque = field(default_factory=lambda: deque(maxlen=500))
    accuracy_scores: deque = field(default_factory=lambda: deque(maxlen=500))
    timeliness_scores: deque = field(default_factory=lambda: deque(maxlen=500))

    # Latest values for consistency checks
    latest_values: dict[str, Any] = field(default_factory=dict)

    # Counters
    total_records: int = 0
    last_arrival_time: float = 0.0
    last_alert_time: float = 0.0  # cooldown


# ---------------------------------------------------------------------------
# DataQualityMonitor — the main monitor
# ---------------------------------------------------------------------------

class DataQualityMonitor:
    """Monitors data quality across all registered sensor sources.

    Evaluates incoming data against expected schemas, latency bounds,
    value ranges, and cross-sensor consistency. Emits QualityAlerts
    via the EventBus when quality drops below configured thresholds.

    Thread-safe. All public methods can be called from any thread.

    Args:
        event_bus: Optional EventBus for publishing alerts on topic
            ``quality.alert``.
        default_threshold: Default quality score threshold below which
            alerts are generated (0.0–1.0, default 0.5).
        alert_cooldown: Minimum seconds between alerts for the same
            source (default 60).
        dimension_weights: Optional dict mapping QualityDimension to
            a float weight for computing overall_score. Defaults to
            equal weights.
    """

    # Default topic for alert events
    ALERT_TOPIC = "quality.alert"

    def __init__(
        self,
        event_bus: Any | None = None,
        default_threshold: float = 0.5,
        alert_cooldown: float = 60.0,
        dimension_weights: dict[QualityDimension, float] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._default_threshold = max(0.0, min(1.0, default_threshold))
        self._alert_cooldown = alert_cooldown
        self._lock = threading.Lock()
        self._sources: dict[str, _SourceTracker] = {}
        self._thresholds: dict[str, dict[QualityDimension, float]] = {}
        self._consistency_groups: dict[str, list[str]] = {}
        self._alerts: list[QualityAlert] = []

        if dimension_weights is not None:
            self._dimension_weights = dict(dimension_weights)
        else:
            self._dimension_weights = {d: 1.0 for d in QualityDimension}

    # -- Registration ---------------------------------------------------------

    def register_source(
        self,
        source_id: str,
        expected_fields: list[str] | None = None,
        value_ranges: dict[str, tuple[float, float]] | None = None,
        max_latency_seconds: float = 30.0,
        max_staleness_seconds: float = 300.0,
        thresholds: dict[QualityDimension, float] | None = None,
    ) -> None:
        """Register a source for quality monitoring.

        Args:
            source_id: Unique identifier for the sensor/source.
            expected_fields: Fields that must be present in every record.
            value_ranges: Map of field name to (min, max) expected range.
            max_latency_seconds: Max acceptable arrival delay.
            max_staleness_seconds: Max time without data before stale.
            thresholds: Per-dimension alert thresholds (overrides default).
        """
        with self._lock:
            self._sources[source_id] = _SourceTracker(
                source_id=source_id,
                expected_fields=expected_fields or [],
                value_ranges=value_ranges or {},
                max_latency_seconds=max_latency_seconds,
                max_staleness_seconds=max_staleness_seconds,
            )
            if thresholds:
                self._thresholds[source_id] = dict(thresholds)

    def unregister_source(self, source_id: str) -> bool:
        """Unregister a source. Returns True if it existed."""
        with self._lock:
            existed = source_id in self._sources
            self._sources.pop(source_id, None)
            self._thresholds.pop(source_id, None)
            return existed

    def register_consistency_group(self, group_name: str, source_ids: list[str]) -> None:
        """Register a group of sources that should agree with each other.

        When consistency is checked, values from sources in the same group
        are compared. Large deviations lower the consistency score.

        Args:
            group_name: A name for this consistency group.
            source_ids: List of source IDs in the group.
        """
        with self._lock:
            self._consistency_groups[group_name] = list(source_ids)

    # -- Data ingestion -------------------------------------------------------

    def record(
        self,
        source_id: str,
        data: dict[str, Any],
        data_timestamp: float | None = None,
    ) -> list[QualityAlert]:
        """Record incoming data and evaluate quality.

        If the source has not been registered, it is auto-registered with
        no schema constraints (completeness will always be 1.0).

        Args:
            source_id: Sensor/source that produced this data.
            data: The data record as a dict.
            data_timestamp: When the data was generated (for timeliness).
                If None, timeliness is not evaluated.

        Returns:
            List of any QualityAlerts generated by this record.
        """
        now = time.time()
        alerts: list[QualityAlert] = []

        with self._lock:
            # Auto-register unknown sources
            if source_id not in self._sources:
                self._sources[source_id] = _SourceTracker(source_id=source_id)

            tracker = self._sources[source_id]
            tracker.total_records += 1
            tracker.arrival_times.append(now)
            tracker.last_arrival_time = now

            # Store latest values for consistency
            for k, v in data.items():
                tracker.latest_values[k] = v

            # -- Completeness --
            comp_score = self._eval_completeness(tracker, data)
            tracker.completeness_scores.append(comp_score)

            # -- Timeliness --
            time_score = self._eval_timeliness(tracker, data_timestamp, now)
            tracker.timeliness_scores.append(time_score)

            # -- Accuracy --
            acc_score = self._eval_accuracy(tracker, data)
            tracker.accuracy_scores.append(acc_score)

            # Check thresholds and generate alerts
            thresholds = self._thresholds.get(source_id, {})
            for dim, score in [
                (QualityDimension.COMPLETENESS, comp_score),
                (QualityDimension.TIMELINESS, time_score),
                (QualityDimension.ACCURACY, acc_score),
            ]:
                threshold = thresholds.get(dim, self._default_threshold)
                if score < threshold:
                    alert = self._maybe_create_alert(
                        tracker, dim, score, threshold, now
                    )
                    if alert is not None:
                        alerts.append(alert)

        return alerts

    # -- Quality evaluation ---------------------------------------------------

    def _eval_completeness(self, tracker: _SourceTracker, data: dict[str, Any]) -> float:
        """Evaluate completeness: fraction of expected fields present."""
        if not tracker.expected_fields:
            return 1.0
        present = sum(1 for f in tracker.expected_fields if f in data and data[f] is not None)
        return present / len(tracker.expected_fields)

    def _eval_timeliness(
        self, tracker: _SourceTracker, data_timestamp: float | None, now: float
    ) -> float:
        """Evaluate timeliness: how close to real-time the data arrived."""
        if data_timestamp is None:
            return 1.0  # Cannot evaluate without a data timestamp
        latency = now - data_timestamp
        if latency <= 0:
            return 1.0
        if latency >= tracker.max_latency_seconds:
            return 0.0
        # Linear decay from 1.0 at 0 latency to 0.0 at max_latency
        return 1.0 - (latency / tracker.max_latency_seconds)

    def _eval_accuracy(self, tracker: _SourceTracker, data: dict[str, Any]) -> float:
        """Evaluate accuracy: fraction of ranged fields within expected bounds."""
        if not tracker.value_ranges:
            return 1.0
        checked = 0
        in_range = 0
        for field_name, (lo, hi) in tracker.value_ranges.items():
            if field_name in data and data[field_name] is not None:
                checked += 1
                try:
                    val = float(data[field_name])
                    if lo <= val <= hi:
                        in_range += 1
                except (TypeError, ValueError):
                    pass  # Non-numeric value counts as out of range
        if checked == 0:
            return 1.0
        return in_range / checked

    def _eval_consistency(self, source_id: str) -> float:
        """Evaluate consistency: how well this source agrees with its group peers.

        Must be called with self._lock held.
        """
        tracker = self._sources.get(source_id)
        if tracker is None:
            return 1.0

        # Find which groups this source belongs to
        peer_sources: list[str] = []
        for _group_name, members in self._consistency_groups.items():
            if source_id in members:
                for m in members:
                    if m != source_id and m in self._sources:
                        peer_sources.append(m)

        if not peer_sources:
            return 1.0  # No peers to compare

        # Compare shared numeric fields
        agreements = 0
        comparisons = 0
        for peer_id in peer_sources:
            peer = self._sources[peer_id]
            for field_name, val in tracker.latest_values.items():
                if field_name in peer.latest_values:
                    peer_val = peer.latest_values[field_name]
                    try:
                        v1 = float(val)
                        v2 = float(peer_val)
                        comparisons += 1
                        # Within 20% of each other = agreement
                        denom = max(abs(v1), abs(v2), 1e-9)
                        deviation = abs(v1 - v2) / denom
                        if deviation <= 0.2:
                            agreements += 1
                        elif deviation <= 0.5:
                            agreements += 0.5
                        # else: 0 agreement
                    except (TypeError, ValueError):
                        # Non-numeric: exact match
                        comparisons += 1
                        if val == peer_val:
                            agreements += 1

        if comparisons == 0:
            return 1.0
        return agreements / comparisons

    def _eval_freshness(self, tracker: _SourceTracker, now: float) -> float:
        """Evaluate freshness: has data arrived recently enough?

        Returns 1.0 if data arrived within the last max_staleness_seconds,
        decaying linearly to 0.0 at 2x max_staleness.
        """
        if tracker.last_arrival_time <= 0:
            return 0.0  # Never seen data
        age = now - tracker.last_arrival_time
        if age <= tracker.max_staleness_seconds:
            return 1.0
        # Linear decay from 1.0 to 0.0 over another max_staleness window
        overage = age - tracker.max_staleness_seconds
        if overage >= tracker.max_staleness_seconds:
            return 0.0
        return 1.0 - (overage / tracker.max_staleness_seconds)

    # -- Alert generation -----------------------------------------------------

    def _maybe_create_alert(
        self,
        tracker: _SourceTracker,
        dimension: QualityDimension,
        score: float,
        threshold: float,
        now: float,
    ) -> QualityAlert | None:
        """Create an alert if the cooldown has elapsed.

        Must be called with self._lock held.
        """
        if now - tracker.last_alert_time < self._alert_cooldown:
            return None

        tracker.last_alert_time = now

        message = (
            f"Source '{tracker.source_id}' {dimension.value} quality "
            f"dropped to {score:.2f} (threshold: {threshold:.2f})"
        )

        alert = QualityAlert(
            alert_id=str(uuid.uuid4()),
            source_id=tracker.source_id,
            dimension=dimension,
            score=score,
            threshold=threshold,
            message=message,
        )

        self._alerts.append(alert)
        logger.warning("Quality alert: %s", message)

        # Emit on EventBus
        if self._event_bus is not None:
            self._event_bus.publish(
                self.ALERT_TOPIC,
                data={
                    "alert_id": alert.alert_id,
                    "source_id": alert.source_id,
                    "dimension": alert.dimension.value,
                    "score": alert.score,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                },
                source="quality_monitor",
            )

        return alert

    # -- Reports --------------------------------------------------------------

    def get_report(self, source_id: str) -> QualityReport | None:
        """Generate a quality report for a specific source.

        Returns None if the source is not registered.
        """
        now = time.time()
        with self._lock:
            tracker = self._sources.get(source_id)
            if tracker is None:
                return None

            metrics: dict[QualityDimension, QualityMetric] = {}

            # Completeness — average of recent scores
            if tracker.completeness_scores:
                avg_comp = sum(tracker.completeness_scores) / len(tracker.completeness_scores)
            else:
                avg_comp = 1.0
            metrics[QualityDimension.COMPLETENESS] = QualityMetric(
                dimension=QualityDimension.COMPLETENESS,
                source_id=source_id,
                score=avg_comp,
                detail=f"Avg completeness over {len(tracker.completeness_scores)} records",
                timestamp=now,
            )

            # Timeliness — average of recent scores
            if tracker.timeliness_scores:
                avg_time = sum(tracker.timeliness_scores) / len(tracker.timeliness_scores)
            else:
                avg_time = 1.0
            metrics[QualityDimension.TIMELINESS] = QualityMetric(
                dimension=QualityDimension.TIMELINESS,
                source_id=source_id,
                score=avg_time,
                detail=f"Avg timeliness over {len(tracker.timeliness_scores)} records",
                timestamp=now,
            )

            # Accuracy — average of recent scores
            if tracker.accuracy_scores:
                avg_acc = sum(tracker.accuracy_scores) / len(tracker.accuracy_scores)
            else:
                avg_acc = 1.0
            metrics[QualityDimension.ACCURACY] = QualityMetric(
                dimension=QualityDimension.ACCURACY,
                source_id=source_id,
                score=avg_acc,
                detail=f"Avg accuracy over {len(tracker.accuracy_scores)} records",
                timestamp=now,
            )

            # Consistency — live evaluation
            cons_score = self._eval_consistency(source_id)
            metrics[QualityDimension.CONSISTENCY] = QualityMetric(
                dimension=QualityDimension.CONSISTENCY,
                source_id=source_id,
                score=cons_score,
                detail="Cross-sensor consistency check",
                timestamp=now,
            )

            # Freshness — live evaluation
            fresh_score = self._eval_freshness(tracker, now)
            metrics[QualityDimension.FRESHNESS] = QualityMetric(
                dimension=QualityDimension.FRESHNESS,
                source_id=source_id,
                score=fresh_score,
                detail=f"Last data {now - tracker.last_arrival_time:.1f}s ago" if tracker.last_arrival_time > 0 else "No data received",
                timestamp=now,
            )

            # Overall score: weighted average
            overall = self._compute_overall(metrics)

            # Collect alerts for this source
            source_alerts = [a for a in self._alerts if a.source_id == source_id]

            return QualityReport(
                source_id=source_id,
                metrics=metrics,
                overall_score=overall,
                sample_count=tracker.total_records,
                alerts=source_alerts,
                timestamp=now,
            )

    def get_all_reports(self) -> list[QualityReport]:
        """Generate quality reports for all registered sources."""
        with self._lock:
            source_ids = list(self._sources.keys())
        # Call get_report outside the lock to avoid re-entry issues
        reports = []
        for sid in source_ids:
            report = self.get_report(sid)
            if report is not None:
                reports.append(report)
        return reports

    def get_alerts(self, source_id: str | None = None) -> list[QualityAlert]:
        """Return all alerts, optionally filtered by source_id."""
        with self._lock:
            if source_id is not None:
                return [a for a in self._alerts if a.source_id == source_id]
            return list(self._alerts)

    def clear_alerts(self, source_id: str | None = None) -> int:
        """Clear alerts, optionally only for a specific source.

        Returns the number of alerts cleared.
        """
        with self._lock:
            if source_id is None:
                count = len(self._alerts)
                self._alerts.clear()
                return count
            before = len(self._alerts)
            self._alerts = [a for a in self._alerts if a.source_id != source_id]
            return before - len(self._alerts)

    def get_source_ids(self) -> list[str]:
        """Return all registered source IDs."""
        with self._lock:
            return list(self._sources.keys())

    # -- Internals ------------------------------------------------------------

    def _compute_overall(self, metrics: dict[QualityDimension, QualityMetric]) -> float:
        """Compute weighted average of dimension scores."""
        total_weight = 0.0
        weighted_sum = 0.0
        for dim, metric in metrics.items():
            weight = self._dimension_weights.get(dim, 1.0)
            weighted_sum += metric.score * weight
            total_weight += weight
        if total_weight <= 0:
            return 1.0
        return weighted_sum / total_weight

    def export(self) -> dict[str, Any]:
        """Export full monitor state as a JSON-serializable dict."""
        reports = self.get_all_reports()
        return {
            "sources": [r.export() for r in reports],
            "alert_count": len(self._alerts),
            "source_count": len(self._sources),
            "timestamp": time.time(),
        }


__all__ = [
    "DataQualityMonitor",
    "QualityAlert",
    "QualityDimension",
    "QualityMetric",
    "QualityReport",
]

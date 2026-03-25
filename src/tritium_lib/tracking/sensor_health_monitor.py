# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SensorHealthMonitor — tracks per-sensor sighting rates and flags
when a sensor goes quiet (possible failure, obstruction, or tampering).

Provides health status data and publishes alerts when sighting rate
drops >50% from baseline.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("sensor-health")


@dataclass
class _SensorRecord:
    """Internal record tracking a sensor's sighting timestamps."""
    sensor_id: str
    sighting_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    baseline_rate: float = 0.0  # sightings/min learned over time
    baseline_samples: int = 0   # number of rate samples used for baseline
    last_seen: float = 0.0
    last_alert_time: float = 0.0  # prevent alert spam


class SensorHealthMonitor:
    """Monitors per-sensor sighting rates and detects anomalies.

    Usage:
        monitor = SensorHealthMonitor()
        monitor.record_sighting("node-01")  # call on each sighting
        health = monitor.get_health()  # returns list of sensor health dicts
    """

    # How far back to measure current rate
    RATE_WINDOW_SECONDS = 300.0  # 5 minutes
    # How many samples before baseline is considered learned
    BASELINE_MIN_SAMPLES = 5
    # Exponential moving average alpha for baseline learning
    BASELINE_ALPHA = 0.1
    # Alert cooldown — don't re-alert on same sensor within this window
    ALERT_COOLDOWN_SECONDS = 600.0  # 10 minutes
    # Offline threshold — no sightings for this long means offline
    OFFLINE_THRESHOLD_SECONDS = 300.0

    def __init__(self, event_bus=None) -> None:
        self._sensors: dict[str, _SensorRecord] = {}
        self._lock = threading.Lock()
        self._event_bus = event_bus

    def record_sighting(self, sensor_id: str) -> None:
        """Record that a sensor reported a sighting."""
        now = time.monotonic()
        with self._lock:
            if sensor_id not in self._sensors:
                self._sensors[sensor_id] = _SensorRecord(
                    sensor_id=sensor_id,
                    last_seen=now,
                )
            rec = self._sensors[sensor_id]
            rec.sighting_times.append(now)
            rec.last_seen = now

            # Update baseline via EMA
            rate = self._compute_rate(rec, now)
            if rate > 0:
                if rec.baseline_samples < self.BASELINE_MIN_SAMPLES:
                    # Still bootstrapping — use simple average
                    rec.baseline_rate = (
                        (rec.baseline_rate * rec.baseline_samples + rate)
                        / (rec.baseline_samples + 1)
                    )
                    rec.baseline_samples += 1
                else:
                    # EMA update
                    rec.baseline_rate = (
                        self.BASELINE_ALPHA * rate
                        + (1 - self.BASELINE_ALPHA) * rec.baseline_rate
                    )
                    rec.baseline_samples += 1

    def _compute_rate(self, rec: _SensorRecord, now: float) -> float:
        """Compute sightings per minute in the recent window."""
        cutoff = now - self.RATE_WINDOW_SECONDS
        count = sum(1 for t in rec.sighting_times if t >= cutoff)
        return count / (self.RATE_WINDOW_SECONDS / 60.0)

    def get_health(self) -> list[dict]:
        """Return health status for all known sensors."""
        now = time.monotonic()
        results = []
        with self._lock:
            for sensor_id, rec in self._sensors.items():
                rate = self._compute_rate(rec, now)
                seconds_since = now - rec.last_seen if rec.last_seen > 0 else None

                # Classify
                if seconds_since is not None and seconds_since >= self.OFFLINE_THRESHOLD_SECONDS:
                    status = "offline"
                elif rec.baseline_samples < self.BASELINE_MIN_SAMPLES:
                    status = "unknown"
                elif rec.baseline_rate <= 0:
                    status = "unknown"
                else:
                    deviation = (rate - rec.baseline_rate) / rec.baseline_rate * 100.0
                    if deviation >= -25.0:
                        status = "healthy"
                    elif deviation >= -50.0:
                        status = "degraded"
                    else:
                        status = "critical"

                deviation_pct = 0.0
                if rec.baseline_rate > 0:
                    deviation_pct = (rate - rec.baseline_rate) / rec.baseline_rate * 100.0

                alert_msg = None
                if status == "critical":
                    alert_msg = (
                        f"Sensor {sensor_id} sighting rate dropped {abs(deviation_pct):.0f}% "
                        f"below baseline ({rate:.1f}/min vs {rec.baseline_rate:.1f}/min baseline)"
                    )
                    self._maybe_emit_alert(rec, alert_msg, now)
                elif status == "offline":
                    alert_msg = (
                        f"Sensor {sensor_id} offline — no sightings for "
                        f"{seconds_since:.0f}s"
                    )
                    self._maybe_emit_alert(rec, alert_msg, now)

                results.append({
                    "sensor_id": sensor_id,
                    "sighting_rate": round(rate, 2),
                    "baseline_rate": round(rec.baseline_rate, 2),
                    "deviation_pct": round(deviation_pct, 1),
                    "status": status,
                    "last_seen_seconds_ago": round(seconds_since, 1) if seconds_since is not None else None,
                    "sighting_count": len(rec.sighting_times),
                    "baseline_samples": rec.baseline_samples,
                    "alert_message": alert_msg,
                })

        return results

    def get_sensor_health(self, sensor_id: str) -> Optional[dict]:
        """Return health for a specific sensor."""
        all_health = self.get_health()
        for h in all_health:
            if h["sensor_id"] == sensor_id:
                return h
        return None

    def _maybe_emit_alert(self, rec: _SensorRecord, message: str, now: float) -> None:
        """Emit an alert via EventBus if cooldown has elapsed."""
        if now - rec.last_alert_time < self.ALERT_COOLDOWN_SECONDS:
            return
        rec.last_alert_time = now
        log.warning("Sensor health alert: %s", message)
        if self._event_bus is not None:
            self._event_bus.publish("sensor:health_alert", data={
                "sensor_id": rec.sensor_id,
                "message": message,
                "timestamp": time.time(),
            })

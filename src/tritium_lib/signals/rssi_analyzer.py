# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""RSSI time-series analysis for BLE and WiFi signals.

Provides Kalman-filtered RSSI smoothing, log-distance path loss distance
estimation, motion detection via RSSI variance analysis, and time-windowed
statistics.  All algorithms are pure Python (stdlib math only).

This module builds on the low-level RSSIFilter from tritium_lib.models but
adds higher-level analytics: sliding windows, trend detection, motion
classification, and multi-device tracking in a single analyzer instance.

Usage::

    from tritium_lib.signals import RSSIAnalyzer

    analyzer = RSSIAnalyzer()
    analyzer.add_reading("ble_AA:BB:CC:DD:EE:FF", -65.0, 1000.0)
    analyzer.add_reading("ble_AA:BB:CC:DD:EE:FF", -68.0, 1001.0)
    analyzer.add_reading("ble_AA:BB:CC:DD:EE:FF", -63.0, 1002.0)

    smoothed = analyzer.get_smoothed("ble_AA:BB:CC:DD:EE:FF")
    distance = analyzer.estimate_distance("ble_AA:BB:CC:DD:EE:FF")
    moving = analyzer.detect_motion("ble_AA:BB:CC:DD:EE:FF")
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Kalman filter for 1D RSSI smoothing
# ---------------------------------------------------------------------------

class _KalmanRSSI:
    """Minimal 1D Kalman filter tuned for RSSI data.

    Separate from models.trilateration.RSSIFilter so this module has
    zero intra-project imports and can be tested in isolation.
    """

    __slots__ = ("x", "p", "q", "r")

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 3.0,
        initial: float = -70.0,
    ) -> None:
        self.q = process_noise
        self.r = measurement_noise
        self.x = initial
        self.p = measurement_noise

    def update(self, measurement: float) -> float:
        """Feed one raw RSSI reading; return filtered value."""
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1.0 - k)
        return self.x

    def reset(self, initial: float = -70.0) -> None:
        self.x = initial
        self.p = self.r


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RSSIReading:
    """A single timestamped RSSI measurement."""
    rssi_dbm: float
    timestamp: float
    smoothed_dbm: float = 0.0


@dataclass
class MotionResult:
    """Result of motion detection analysis."""
    is_moving: bool
    variance: float
    trend_dbm_per_sec: float
    confidence: float
    classification: str  # "stationary", "slow", "fast", "erratic"

    def to_dict(self) -> dict:
        return {
            "is_moving": self.is_moving,
            "variance": round(self.variance, 3),
            "trend_dbm_per_sec": round(self.trend_dbm_per_sec, 4),
            "confidence": round(self.confidence, 3),
            "classification": self.classification,
        }


@dataclass
class RSSIStats:
    """Statistical summary over a time window."""
    count: int
    mean: float
    std_dev: float
    min_val: float
    max_val: float
    range_val: float
    latest: float
    smoothed: float

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": round(self.mean, 2),
            "std_dev": round(self.std_dev, 2),
            "min": round(self.min_val, 2),
            "max": round(self.max_val, 2),
            "range": round(self.range_val, 2),
            "latest": round(self.latest, 2),
            "smoothed": round(self.smoothed, 2),
        }


# ---------------------------------------------------------------------------
# Per-device tracking state
# ---------------------------------------------------------------------------

@dataclass
class _DeviceState:
    """Internal tracking state for one device."""
    device_id: str
    kalman: _KalmanRSSI
    readings: deque  # of RSSIReading
    first_seen: float = 0.0
    last_seen: float = 0.0

    def __init__(
        self,
        device_id: str,
        window_size: int = 100,
        process_noise: float = 0.5,
        measurement_noise: float = 3.0,
    ) -> None:
        self.device_id = device_id
        self.kalman = _KalmanRSSI(process_noise, measurement_noise)
        self.readings = deque(maxlen=window_size)
        self.first_seen = 0.0
        self.last_seen = 0.0


# ---------------------------------------------------------------------------
# RSSIAnalyzer
# ---------------------------------------------------------------------------

class RSSIAnalyzer:
    """Multi-device RSSI time-series analyzer.

    Tracks RSSI readings from multiple BLE/WiFi devices simultaneously,
    providing Kalman-smoothed values, distance estimation, and motion
    detection for each.

    Args:
        window_size: Max readings kept per device (sliding window).
        process_noise: Kalman Q parameter — how fast true RSSI changes.
        measurement_noise: Kalman R parameter — expected RSSI jitter.
        tx_power: Reference RSSI at 1 metre (default -59 dBm for BLE).
        path_loss_exponent: Environment factor for log-distance model.
            2.0 = free space, 2.5 = typical indoor, 3.0-4.0 = obstructed.
        motion_variance_threshold: RSSI variance above which a device is
            considered to be in motion.
    """

    def __init__(
        self,
        window_size: int = 100,
        process_noise: float = 0.5,
        measurement_noise: float = 3.0,
        tx_power: float = -59.0,
        path_loss_exponent: float = 2.5,
        motion_variance_threshold: float = 4.0,
    ) -> None:
        self._window_size = window_size
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._tx_power = tx_power
        self._path_loss_exponent = path_loss_exponent
        self._motion_var_threshold = motion_variance_threshold
        self._devices: dict[str, _DeviceState] = {}

    # -- Ingestion ----------------------------------------------------------

    def add_reading(
        self,
        device_id: str,
        rssi_dbm: float,
        timestamp: float | None = None,
    ) -> float:
        """Add an RSSI measurement for a device.

        Args:
            device_id: Unique device identifier (e.g. ``ble_AA:BB:CC``).
            rssi_dbm: Raw RSSI in dBm.
            timestamp: Epoch seconds. Uses ``time.time()`` if omitted.

        Returns:
            Kalman-smoothed RSSI value.
        """
        if timestamp is None:
            timestamp = time.time()

        state = self._get_or_create(device_id)

        if state.first_seen == 0.0:
            state.first_seen = timestamp
            state.kalman.reset(rssi_dbm)
        state.last_seen = timestamp

        smoothed = state.kalman.update(rssi_dbm)
        reading = RSSIReading(
            rssi_dbm=rssi_dbm,
            timestamp=timestamp,
            smoothed_dbm=smoothed,
        )
        state.readings.append(reading)
        return smoothed

    # -- Queries ------------------------------------------------------------

    def get_smoothed(self, device_id: str) -> float | None:
        """Return current Kalman-smoothed RSSI, or None if unknown device."""
        state = self._devices.get(device_id)
        if state is None:
            return None
        return state.kalman.x

    def estimate_distance(
        self,
        device_id: str,
        tx_power: float | None = None,
        path_loss_exponent: float | None = None,
    ) -> float | None:
        """Estimate distance to a device using the log-distance path loss model.

        Uses the smoothed RSSI for a more stable distance estimate.

        Args:
            device_id: Device identifier.
            tx_power: Override default tx_power for this call.
            path_loss_exponent: Override default path loss exponent.

        Returns:
            Distance in metres, or None if device unknown. Clamped >= 0.1.
        """
        smoothed = self.get_smoothed(device_id)
        if smoothed is None:
            return None
        tp = tx_power if tx_power is not None else self._tx_power
        n = path_loss_exponent if path_loss_exponent is not None else self._path_loss_exponent
        return _rssi_to_distance(smoothed, tp, n)

    def detect_motion(
        self,
        device_id: str,
        window_seconds: float = 10.0,
    ) -> MotionResult | None:
        """Analyse recent RSSI variance to detect whether a device is moving.

        Computes variance and linear trend over the last ``window_seconds``
        of smoothed readings.  Higher variance and stronger trend indicate
        motion.

        Returns:
            MotionResult, or None if device is unknown or has < 3 readings.
        """
        state = self._devices.get(device_id)
        if state is None or len(state.readings) < 3:
            return None

        now = state.last_seen
        cutoff = now - window_seconds
        recent = [r for r in state.readings if r.timestamp >= cutoff]

        if len(recent) < 3:
            return None

        # Variance of smoothed RSSI
        values = [r.smoothed_dbm for r in recent]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n

        # Linear trend (least-squares slope) on smoothed RSSI vs. time
        times = [r.timestamp - recent[0].timestamp for r in recent]
        t_mean = sum(times) / n
        numerator = sum((times[i] - t_mean) * (values[i] - mean) for i in range(n))
        denominator = sum((times[i] - t_mean) ** 2 for i in range(n))
        trend = numerator / denominator if denominator > 1e-9 else 0.0

        # Classification thresholds
        is_moving = variance > self._motion_var_threshold
        abs_trend = abs(trend)

        if variance < self._motion_var_threshold * 0.5:
            classification = "stationary"
            confidence = min(1.0, 1.0 - variance / self._motion_var_threshold)
        elif variance < self._motion_var_threshold:
            # Borderline — could be minor movement or noise
            classification = "stationary"
            confidence = 0.5
        elif abs_trend > 2.0:
            classification = "fast"
            confidence = min(1.0, 0.6 + abs_trend * 0.1)
        elif variance > self._motion_var_threshold * 3.0:
            classification = "erratic"
            confidence = min(1.0, 0.5 + variance / (self._motion_var_threshold * 10.0))
        else:
            classification = "slow"
            confidence = min(1.0, 0.5 + variance / (self._motion_var_threshold * 4.0))

        return MotionResult(
            is_moving=is_moving,
            variance=variance,
            trend_dbm_per_sec=trend,
            confidence=min(1.0, max(0.0, confidence)),
            classification=classification,
        )

    def get_stats(self, device_id: str) -> RSSIStats | None:
        """Return statistical summary for a device's RSSI readings.

        Uses all readings currently in the sliding window.
        """
        state = self._devices.get(device_id)
        if state is None or len(state.readings) == 0:
            return None

        values = [r.rssi_dbm for r in state.readings]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std_dev = math.sqrt(variance)

        return RSSIStats(
            count=n,
            mean=mean,
            std_dev=std_dev,
            min_val=min(values),
            max_val=max(values),
            range_val=max(values) - min(values),
            latest=values[-1],
            smoothed=state.kalman.x,
        )

    def get_readings(
        self,
        device_id: str,
        max_count: int = 0,
    ) -> list[dict]:
        """Return recent readings as serializable dicts."""
        state = self._devices.get(device_id)
        if state is None:
            return []
        readings = list(state.readings)
        if max_count > 0:
            readings = readings[-max_count:]
        return [
            {
                "rssi_dbm": round(r.rssi_dbm, 2),
                "smoothed_dbm": round(r.smoothed_dbm, 2),
                "timestamp": r.timestamp,
            }
            for r in readings
        ]

    def get_tracked_devices(self) -> list[str]:
        """Return list of all tracked device IDs."""
        return list(self._devices.keys())

    def remove_device(self, device_id: str) -> bool:
        """Stop tracking a device. Returns True if it was tracked."""
        return self._devices.pop(device_id, None) is not None

    def clear(self) -> None:
        """Remove all tracked devices."""
        self._devices.clear()

    def get_status(self) -> dict:
        """Return analyzer status summary."""
        return {
            "tracked_devices": len(self._devices),
            "window_size": self._window_size,
            "tx_power": self._tx_power,
            "path_loss_exponent": self._path_loss_exponent,
            "motion_variance_threshold": self._motion_var_threshold,
        }

    # -- Internal -----------------------------------------------------------

    def _get_or_create(self, device_id: str) -> _DeviceState:
        state = self._devices.get(device_id)
        if state is None:
            state = _DeviceState(
                device_id=device_id,
                window_size=self._window_size,
                process_noise=self._process_noise,
                measurement_noise=self._measurement_noise,
            )
            self._devices[device_id] = state
        return state


# ---------------------------------------------------------------------------
# Standalone distance function
# ---------------------------------------------------------------------------

def _rssi_to_distance(
    rssi_dbm: float,
    tx_power: float = -59.0,
    path_loss_exponent: float = 2.5,
) -> float:
    """Log-distance path loss model: d = 10^((tx_power - rssi) / (10 * n)).

    Returns distance in metres, clamped to >= 0.1.
    """
    if path_loss_exponent <= 0:
        raise ValueError("path_loss_exponent must be positive")
    exponent = (tx_power - rssi_dbm) / (10.0 * path_loss_exponent)
    return max(0.1, 10.0 ** exponent)

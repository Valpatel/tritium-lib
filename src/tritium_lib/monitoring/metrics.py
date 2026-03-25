# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MetricsCollector — pure-Python performance metrics for Tritium subsystems.

Collects latency, throughput, queue depth, and gauge values with rolling
time-window statistics.  Thread-safe.  No external dependencies.

Metric types:
  - **counter**: monotonically increasing value (e.g., total events processed)
  - **gauge**: point-in-time value (e.g., current queue depth)
  - **latency**: duration samples (e.g., fusion pipeline latency)

Usage::

    mc = MetricsCollector(window_seconds=300)
    mc.record_latency("fusion.pipeline", 0.023)
    mc.increment("events.processed")
    mc.set_gauge("tracker.target_count", 42)

    stats = mc.get_stats("fusion.pipeline")
    # {"count": 150, "mean": 0.021, "p50": 0.019, "p95": 0.035, ...}
"""

from __future__ import annotations

import bisect
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricSample:
    """A single timestamped metric sample."""

    value: float
    timestamp: float


class MetricWindow:
    """Rolling time-window buffer for metric samples.

    Maintains samples within a configurable time window and provides
    statistical aggregations.  Thread-safe — callers are expected to
    hold the parent MetricsCollector lock.
    """

    def __init__(self, window_seconds: float = 300.0, max_samples: int = 50000) -> None:
        self._window = window_seconds
        self._max_samples = max_samples
        self._samples: list[MetricSample] = []
        self._total: float = 0.0
        self._count: int = 0

    def add(self, value: float, timestamp: float | None = None) -> None:
        """Add a sample to the window."""
        ts = timestamp if timestamp is not None else time.time()
        sample = MetricSample(value=value, timestamp=ts)
        self._samples.append(sample)
        self._total += value
        self._count += 1

        # Evict old samples
        self._prune(ts)

        # Hard cap to prevent unbounded memory growth
        if len(self._samples) > self._max_samples:
            removed = self._samples[: len(self._samples) - self._max_samples]
            self._samples = self._samples[-self._max_samples:]
            for s in removed:
                self._total -= s.value
                self._count -= 1

    def _prune(self, now: float | None = None) -> None:
        """Remove samples older than the window."""
        if now is None:
            now = time.time()
        cutoff = now - self._window
        while self._samples and self._samples[0].timestamp < cutoff:
            removed = self._samples.pop(0)
            self._total -= removed.value
            self._count -= 1

    @property
    def count(self) -> int:
        """Number of samples in the current window."""
        return max(0, self._count)

    @property
    def total(self) -> float:
        """Sum of all sample values in the current window."""
        return self._total

    def get_values(self) -> list[float]:
        """Return a copy of current window values (pruned first)."""
        self._prune()
        return [s.value for s in self._samples]

    def get_stats(self) -> dict[str, Any]:
        """Compute summary statistics for the current window.

        Returns a dict with: count, total, mean, min, max, p50, p90, p95, p99.
        Returns zeroes when no samples are available.
        """
        self._prune()
        values = [s.value for s in self._samples]
        n = len(values)
        if n == 0:
            return {
                "count": 0,
                "total": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        sorted_vals = sorted(values)
        return {
            "count": n,
            "total": round(self._total, 6),
            "mean": round(statistics.mean(values), 6),
            "min": round(sorted_vals[0], 6),
            "max": round(sorted_vals[-1], 6),
            "p50": round(self._percentile(sorted_vals, 50), 6),
            "p90": round(self._percentile(sorted_vals, 90), 6),
            "p95": round(self._percentile(sorted_vals, 95), 6),
            "p99": round(self._percentile(sorted_vals, 99), 6),
        }

    @staticmethod
    def _percentile(sorted_vals: list[float], pct: float) -> float:
        """Compute a percentile from pre-sorted values using linear interpolation."""
        n = len(sorted_vals)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_vals[0]
        k = (pct / 100.0) * (n - 1)
        f = int(k)
        c = f + 1
        if c >= n:
            return sorted_vals[-1]
        d = k - f
        return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])

    def clear(self) -> None:
        """Remove all samples."""
        self._samples.clear()
        self._total = 0.0
        self._count = 0


class MetricsCollector:
    """Thread-safe performance metrics collector for Tritium subsystems.

    Tracks three metric types:
      - **latency**: duration samples with rolling-window statistics
      - **counter**: monotonically increasing counters
      - **gauge**: point-in-time values

    All operations are thread-safe.

    Usage::

        mc = MetricsCollector(window_seconds=300)
        mc.record_latency("fusion.pipeline", 0.023)
        mc.increment("events.processed", 5)
        mc.set_gauge("tracker.target_count", 42)

        stats = mc.get_stats("fusion.pipeline")
        gauges = mc.get_all_gauges()
        counters = mc.get_all_counters()
    """

    def __init__(self, window_seconds: float = 300.0) -> None:
        self._lock = threading.RLock()
        self._window = window_seconds
        self._latencies: dict[str, MetricWindow] = {}
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._gauge_timestamps: dict[str, float] = {}

    # -- Latency ---------------------------------------------------------------

    def record_latency(self, name: str, duration: float) -> None:
        """Record a latency sample (in seconds).

        Parameters
        ----------
        name:
            Dotted metric name (e.g., "fusion.pipeline", "tracker.update").
        duration:
            Duration in seconds.
        """
        with self._lock:
            if name not in self._latencies:
                self._latencies[name] = MetricWindow(
                    window_seconds=self._window,
                )
            self._latencies[name].add(duration)

    def get_stats(self, name: str) -> dict[str, Any]:
        """Get rolling-window statistics for a latency metric.

        Returns dict with count, total, mean, min, max, p50, p90, p95, p99.
        """
        with self._lock:
            window = self._latencies.get(name)
            if window is None:
                return {
                    "count": 0,
                    "total": 0.0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }
            return window.get_stats()

    def get_all_latency_names(self) -> list[str]:
        """Return all registered latency metric names."""
        with self._lock:
            return list(self._latencies.keys())

    # -- Counters --------------------------------------------------------------

    def increment(self, name: str, amount: float = 1.0) -> float:
        """Increment a counter and return its new value.

        Parameters
        ----------
        name:
            Dotted counter name (e.g., "events.processed").
        amount:
            Amount to increment by (default 1).
        """
        with self._lock:
            self._counters[name] += amount
            return self._counters[name]

    def get_counter(self, name: str) -> float:
        """Get the current value of a counter."""
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_all_counters(self) -> dict[str, float]:
        """Return a snapshot of all counters."""
        with self._lock:
            return dict(self._counters)

    def reset_counter(self, name: str) -> None:
        """Reset a counter to zero."""
        with self._lock:
            self._counters[name] = 0.0

    # -- Gauges ----------------------------------------------------------------

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to a specific value.

        Parameters
        ----------
        name:
            Dotted gauge name (e.g., "tracker.target_count").
        value:
            The current value.
        """
        with self._lock:
            self._gauges[name] = value
            self._gauge_timestamps[name] = time.time()

    def get_gauge(self, name: str) -> float:
        """Get the current value of a gauge.  Returns 0.0 if not set."""
        with self._lock:
            return self._gauges.get(name, 0.0)

    def get_gauge_with_timestamp(self, name: str) -> tuple[float, float]:
        """Get a gauge's value and when it was last set.

        Returns (value, timestamp).  Returns (0.0, 0.0) if not set.
        """
        with self._lock:
            return (
                self._gauges.get(name, 0.0),
                self._gauge_timestamps.get(name, 0.0),
            )

    def get_all_gauges(self) -> dict[str, float]:
        """Return a snapshot of all gauge values."""
        with self._lock:
            return dict(self._gauges)

    # -- Bulk ------------------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """Export all metrics as a JSON-serializable dict.

        Returns::

            {
                "counters": {"events.processed": 150, ...},
                "gauges": {"tracker.target_count": 42, ...},
                "latencies": {
                    "fusion.pipeline": {"count": 100, "mean": 0.021, ...},
                    ...
                },
                "timestamp": 1711330000.0,
            }
        """
        with self._lock:
            latencies = {}
            for name, window in self._latencies.items():
                latencies[name] = window.get_stats()
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latencies": latencies,
                "timestamp": time.time(),
            }

    def clear(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latencies.clear()
            self._counters.clear()
            self._gauges.clear()
            self._gauge_timestamps.clear()

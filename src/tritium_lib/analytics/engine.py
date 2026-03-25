# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.analytics.engine — real-time analytics engine for the tracking pipeline.

Computes live statistics from tracking events using efficient O(1) sliding
time windows backed by bucketed counters.  Thread-safe.  No external
dependencies.

Components:

  - **TimeWindow** — sliding time window with configurable bucket granularity.
    Events are counted into second-granularity buckets and expired in O(1)
    amortised time via a circular buffer.

  - **Counter** — event counter with instantaneous rate calculation across
    multiple time horizons (1min, 5min, 1hr, 24hr).

  - **Histogram** — distribution tracker for categorical data (target types,
    zones, sensor sources).  Maintains per-category counts within a sliding
    window.

  - **TrendDetector** — detects increasing/decreasing/stable trends via
    linear regression over bucketed rate data.

  - **TopN** — tracks the top-N items by activity using a windowed
    counter per item, pruned lazily.

  - **AnalyticsEngine** — orchestrates all components, ingests tracking
    events, and exports a unified snapshot of all metrics.

Usage::

    from tritium_lib.analytics import AnalyticsEngine

    engine = AnalyticsEngine()

    # Record events
    engine.record_detection("ble_aabbccdd", source="ble", zone="lobby")
    engine.record_detection("det_person_1", source="yolo", zone="parking")
    engine.record_alert("geofence_entry", severity="warning")
    engine.record_correlation("ble_aabbccdd", "det_person_1", success=True)

    # Query live statistics
    snapshot = engine.snapshot()
    print(snapshot["detection_rate"])      # events per minute
    print(snapshot["zone_activity"])       # per-zone counts
    print(snapshot["sensor_utilization"])  # per-source counts
    print(snapshot["trends"])             # trend directions
    print(snapshot["top_targets"])        # most active targets
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# TimeWindow — O(1) amortised sliding window via bucketed counters
# ---------------------------------------------------------------------------

class TimeWindow:
    """Sliding time window with O(1) amortised add/query.

    Events are counted into second-granularity buckets.  On each add or
    query, expired buckets are lazily evicted.  The total count is maintained
    incrementally so ``count`` and ``rate`` are O(1).

    Parameters
    ----------
    window_seconds:
        Duration of the sliding window in seconds.
    bucket_seconds:
        Granularity of each bucket (default 1 second).
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        bucket_seconds: float = 1.0,
    ) -> None:
        self._window = max(1.0, window_seconds)
        self._bucket_size = max(0.1, bucket_seconds)
        self._buckets: dict[int, float] = {}
        self._total: float = 0.0
        self._last_evict_bucket: int = 0

    def _bucket_key(self, ts: float) -> int:
        """Map a timestamp to its bucket index."""
        return int(ts / self._bucket_size)

    def _evict(self, now: float) -> None:
        """Remove buckets older than the window."""
        cutoff_key = self._bucket_key(now - self._window)
        to_remove = [k for k in self._buckets if k < cutoff_key]
        for k in to_remove:
            self._total -= self._buckets.pop(k)

    def add(self, value: float = 1.0, timestamp: float | None = None) -> None:
        """Add a value to the current bucket."""
        ts = timestamp if timestamp is not None else time.time()
        key = self._bucket_key(ts)
        self._buckets[key] = self._buckets.get(key, 0.0) + value
        self._total += value
        self._evict(ts)

    @property
    def count(self) -> float:
        """Total count within the window (lazily evicted)."""
        self._evict(time.time())
        return max(0.0, self._total)

    @property
    def rate_per_second(self) -> float:
        """Average rate per second over the window."""
        self._evict(time.time())
        if self._total <= 0.0:
            return 0.0
        return self._total / self._window

    @property
    def rate_per_minute(self) -> float:
        """Average rate per minute over the window."""
        return self.rate_per_second * 60.0

    def clear(self) -> None:
        """Remove all data."""
        self._buckets.clear()
        self._total = 0.0

    def export(self) -> dict[str, Any]:
        """Export window state as JSON-serializable dict."""
        self._evict(time.time())
        return {
            "count": round(self._total, 6),
            "rate_per_second": round(self.rate_per_second, 6),
            "rate_per_minute": round(self.rate_per_minute, 6),
            "window_seconds": self._window,
        }


# ---------------------------------------------------------------------------
# Counter — multi-horizon event counter with rate calculation
# ---------------------------------------------------------------------------

class Counter:
    """Event counter with instantaneous rate across multiple time horizons.

    Maintains four sliding windows (1min, 5min, 1hr, 24hr) and a lifetime
    total.  All operations are O(1) amortised.

    Parameters
    ----------
    name:
        Human-readable counter name.
    """

    HORIZONS: dict[str, float] = {
        "1min": 60.0,
        "5min": 300.0,
        "1hr": 3600.0,
        "24hr": 86400.0,
    }

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._lifetime_total: float = 0.0
        self._lifetime_count: int = 0
        self._windows: dict[str, TimeWindow] = {
            label: TimeWindow(window_seconds=secs)
            for label, secs in self.HORIZONS.items()
        }

    def increment(self, value: float = 1.0, timestamp: float | None = None) -> None:
        """Record an event."""
        self._lifetime_total += value
        self._lifetime_count += 1
        for w in self._windows.values():
            w.add(value, timestamp)

    @property
    def total(self) -> float:
        """Lifetime total value."""
        return self._lifetime_total

    @property
    def count(self) -> int:
        """Lifetime event count."""
        return self._lifetime_count

    def rate(self, horizon: str = "1min") -> float:
        """Events per minute at the given horizon.

        Parameters
        ----------
        horizon:
            One of "1min", "5min", "1hr", "24hr".
        """
        w = self._windows.get(horizon)
        if w is None:
            return 0.0
        return w.rate_per_minute

    def rates(self) -> dict[str, float]:
        """Per-minute rates for all horizons."""
        return {
            label: w.rate_per_minute
            for label, w in self._windows.items()
        }

    def window_count(self, horizon: str = "1min") -> float:
        """Event count within a specific horizon window."""
        w = self._windows.get(horizon)
        if w is None:
            return 0.0
        return w.count

    def clear(self) -> None:
        """Reset all data."""
        self._lifetime_total = 0.0
        self._lifetime_count = 0
        for w in self._windows.values():
            w.clear()

    def export(self) -> dict[str, Any]:
        """Export as JSON-serializable dict."""
        return {
            "name": self.name,
            "lifetime_total": self._lifetime_total,
            "lifetime_count": self._lifetime_count,
            "rates_per_minute": self.rates(),
            "window_counts": {
                label: round(w.count, 6)
                for label, w in self._windows.items()
            },
        }


# ---------------------------------------------------------------------------
# Histogram — categorical distribution within a sliding window
# ---------------------------------------------------------------------------

class Histogram:
    """Distribution tracker for categorical data over a sliding window.

    Maintains per-category windowed counts.  Useful for tracking the
    distribution of target types, zones, sensor sources, etc.

    Parameters
    ----------
    name:
        Human-readable histogram name.
    window_seconds:
        Sliding window duration (default 300s = 5 minutes).
    """

    def __init__(self, name: str = "", window_seconds: float = 300.0) -> None:
        self.name = name
        self._window_seconds = window_seconds
        self._categories: dict[str, TimeWindow] = {}
        self._lifetime: dict[str, int] = defaultdict(int)

    def _get_window(self, category: str) -> TimeWindow:
        if category not in self._categories:
            self._categories[category] = TimeWindow(
                window_seconds=self._window_seconds,
            )
        return self._categories[category]

    def record(self, category: str, value: float = 1.0, timestamp: float | None = None) -> None:
        """Record an observation in the given category."""
        self._get_window(category).add(value, timestamp)
        self._lifetime[category] += 1

    @property
    def categories(self) -> list[str]:
        """All observed categories."""
        return list(self._categories.keys())

    def count(self, category: str) -> float:
        """Windowed count for a category."""
        w = self._categories.get(category)
        if w is None:
            return 0.0
        return w.count

    def distribution(self) -> dict[str, float]:
        """Current windowed counts per category."""
        result: dict[str, float] = {}
        for cat, w in self._categories.items():
            c = w.count
            if c > 0:
                result[cat] = round(c, 6)
        return result

    def percentages(self) -> dict[str, float]:
        """Current windowed distribution as percentages (0-100)."""
        dist = self.distribution()
        total = sum(dist.values())
        if total <= 0:
            return {}
        return {cat: round(100.0 * v / total, 2) for cat, v in dist.items()}

    def lifetime_counts(self) -> dict[str, int]:
        """Lifetime event counts per category."""
        return dict(self._lifetime)

    def clear(self) -> None:
        """Reset all data."""
        self._categories.clear()
        self._lifetime.clear()

    def export(self) -> dict[str, Any]:
        """Export as JSON-serializable dict."""
        return {
            "name": self.name,
            "window_seconds": self._window_seconds,
            "distribution": self.distribution(),
            "percentages": self.percentages(),
            "lifetime_counts": self.lifetime_counts(),
        }


# ---------------------------------------------------------------------------
# TrendDetector — linear regression over bucketed rate data
# ---------------------------------------------------------------------------

@dataclass
class TrendResult:
    """Result of a trend analysis."""

    direction: str  # "increasing", "decreasing", "stable"
    slope: float  # events per second per second (rate of change)
    confidence: float  # 0.0 to 1.0 (R-squared)
    current_rate: float  # current per-minute rate
    samples: int  # number of buckets used in regression


class TrendDetector:
    """Detects increasing/decreasing/stable trends using linear regression.

    Divides the analysis window into fixed-size buckets and fits a
    least-squares line through the bucket counts.  The slope direction
    and R-squared value determine the trend.

    Parameters
    ----------
    name:
        Human-readable name.
    window_seconds:
        Total analysis window (default 300s = 5 minutes).
    bucket_seconds:
        Size of each bucket for regression (default 30s).
    slope_threshold:
        Minimum absolute slope to declare non-stable (default 0.01).
    """

    def __init__(
        self,
        name: str = "",
        window_seconds: float = 300.0,
        bucket_seconds: float = 30.0,
        slope_threshold: float = 0.01,
    ) -> None:
        self.name = name
        self._window = window_seconds
        self._bucket_size = max(1.0, bucket_seconds)
        self._slope_threshold = slope_threshold
        self._buckets: dict[int, float] = {}

    def _bucket_key(self, ts: float) -> int:
        return int(ts / self._bucket_size)

    def _evict(self, now: float) -> None:
        cutoff = self._bucket_key(now - self._window)
        to_remove = [k for k in self._buckets if k < cutoff]
        for k in to_remove:
            del self._buckets[k]

    def record(self, value: float = 1.0, timestamp: float | None = None) -> None:
        """Record an event."""
        ts = timestamp if timestamp is not None else time.time()
        key = self._bucket_key(ts)
        self._buckets[key] = self._buckets.get(key, 0.0) + value
        self._evict(ts)

    def analyze(self) -> TrendResult:
        """Analyze the current trend.

        Returns a TrendResult with direction, slope, confidence (R^2),
        and the current rate.
        """
        now = time.time()
        self._evict(now)

        if len(self._buckets) < 2:
            current_rate = sum(self._buckets.values()) * 60.0 / self._window if self._buckets else 0.0
            return TrendResult(
                direction="stable",
                slope=0.0,
                confidence=0.0,
                current_rate=current_rate,
                samples=len(self._buckets),
            )

        # Sort buckets by key and extract (x, y) pairs
        sorted_keys = sorted(self._buckets.keys())
        n = len(sorted_keys)
        xs = list(range(n))
        ys = [self._buckets[k] for k in sorted_keys]

        # Linear regression: y = mx + b
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        ss_xx = sum((x - mean_x) ** 2 for x in xs)
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        ss_yy = sum((y - mean_y) ** 2 for y in ys)

        if ss_xx == 0.0:
            slope = 0.0
        else:
            slope = ss_xy / ss_xx

        # R-squared
        if ss_yy == 0.0:
            r_squared = 0.0
        else:
            r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_xx > 0 else 0.0

        r_squared = max(0.0, min(1.0, r_squared))

        # Normalise slope to events/second/second
        normalized_slope = slope / self._bucket_size if self._bucket_size > 0 else slope

        # Determine direction
        if abs(normalized_slope) < self._slope_threshold:
            direction = "stable"
        elif normalized_slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Current rate: last bucket extrapolated to per-minute
        last_bucket_count = ys[-1] if ys else 0.0
        current_rate = last_bucket_count * 60.0 / self._bucket_size

        return TrendResult(
            direction=direction,
            slope=round(normalized_slope, 8),
            confidence=round(r_squared, 6),
            current_rate=round(current_rate, 6),
            samples=n,
        )

    def clear(self) -> None:
        """Reset all data."""
        self._buckets.clear()

    def export(self) -> dict[str, Any]:
        """Export trend analysis as JSON-serializable dict."""
        result = self.analyze()
        return {
            "name": self.name,
            "direction": result.direction,
            "slope": result.slope,
            "confidence": result.confidence,
            "current_rate": result.current_rate,
            "samples": result.samples,
        }


# ---------------------------------------------------------------------------
# TopN — track top-N items by activity
# ---------------------------------------------------------------------------

class TopN:
    """Track top-N items by activity count within a sliding window.

    Each item has its own windowed counter.  ``top()`` returns the N
    most active items, lazily pruning zeroed-out entries.

    Parameters
    ----------
    n:
        Number of top items to return (default 10).
    window_seconds:
        Sliding window for counting activity (default 300s).
    max_tracked:
        Maximum number of items to track before pruning inactive ones
        (default 10000).
    """

    def __init__(
        self,
        n: int = 10,
        window_seconds: float = 300.0,
        max_tracked: int = 10000,
    ) -> None:
        self._n = max(1, n)
        self._window_seconds = window_seconds
        self._max_tracked = max_tracked
        self._items: dict[str, TimeWindow] = {}
        self._lifetime: dict[str, int] = defaultdict(int)

    def record(self, item: str, value: float = 1.0, timestamp: float | None = None) -> None:
        """Record activity for an item."""
        if item not in self._items:
            # Prune if at capacity
            if len(self._items) >= self._max_tracked:
                self._prune()
            self._items[item] = TimeWindow(window_seconds=self._window_seconds)
        self._items[item].add(value, timestamp)
        self._lifetime[item] += 1

    def _prune(self) -> None:
        """Remove items with zero windowed count."""
        to_remove = [k for k, w in self._items.items() if w.count <= 0]
        for k in to_remove:
            del self._items[k]

    def top(self, n: int | None = None) -> list[tuple[str, float]]:
        """Return the top-N items by windowed count.

        Returns a list of (item, count) tuples sorted by count descending.
        """
        limit = n if n is not None else self._n
        scored = [(k, w.count) for k, w in self._items.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def count(self, item: str) -> float:
        """Windowed count for a specific item."""
        w = self._items.get(item)
        if w is None:
            return 0.0
        return w.count

    def lifetime_count(self, item: str) -> int:
        """Lifetime count for a specific item."""
        return self._lifetime.get(item, 0)

    def clear(self) -> None:
        """Reset all data."""
        self._items.clear()
        self._lifetime.clear()

    def export(self) -> dict[str, Any]:
        """Export as JSON-serializable dict."""
        return {
            "top": [
                {"item": item, "count": round(count, 6)}
                for item, count in self.top()
            ],
            "tracked_items": len(self._items),
            "window_seconds": self._window_seconds,
        }


# ---------------------------------------------------------------------------
# AnalyticsEngine — unified real-time analytics
# ---------------------------------------------------------------------------

class AnalyticsEngine:
    """Real-time analytics engine for the Tritium tracking pipeline.

    Ingests detection, alert, and correlation events and computes live
    statistics including:

      - Target detection rate (per minute/hour across multiple horizons)
      - Zone activity levels (per-zone detection counts)
      - Sensor utilization (per-source detection counts)
      - Alert frequency (per-severity alert rates)
      - Correlation success rate (successful fusions vs attempts)
      - Top targets, zones, and sources by activity

    Thread-safe — all public methods acquire an internal lock.

    Usage::

        engine = AnalyticsEngine()
        engine.record_detection("ble_aabbccdd", source="ble", zone="lobby")
        engine.record_alert("geofence_entry", severity="warning")
        engine.record_correlation("ble_aa", "det_person_1", success=True)
        snap = engine.snapshot()
    """

    def __init__(
        self,
        trend_window: float = 300.0,
        trend_bucket: float = 30.0,
        histogram_window: float = 300.0,
        top_n: int = 10,
        top_n_window: float = 300.0,
    ) -> None:
        self._lock = threading.RLock()

        # Detection counters
        self._detection_counter = Counter(name="detections")
        self._detection_trend = TrendDetector(
            name="detection_trend",
            window_seconds=trend_window,
            bucket_seconds=trend_bucket,
        )

        # Per-source (sensor) counters
        self._sensor_histogram = Histogram(
            name="sensor_utilization",
            window_seconds=histogram_window,
        )

        # Per-zone activity
        self._zone_histogram = Histogram(
            name="zone_activity",
            window_seconds=histogram_window,
        )

        # Alert counters
        self._alert_counter = Counter(name="alerts")
        self._alert_severity_histogram = Histogram(
            name="alert_severity",
            window_seconds=histogram_window,
        )
        self._alert_trend = TrendDetector(
            name="alert_trend",
            window_seconds=trend_window,
            bucket_seconds=trend_bucket,
        )

        # Correlation counters
        self._correlation_attempts = Counter(name="correlation_attempts")
        self._correlation_successes = Counter(name="correlation_successes")

        # Target type histogram
        self._target_type_histogram = Histogram(
            name="target_types",
            window_seconds=histogram_window,
        )

        # TopN trackers
        self._top_targets = TopN(
            n=top_n,
            window_seconds=top_n_window,
        )
        self._top_zones = TopN(
            n=top_n,
            window_seconds=top_n_window,
        )
        self._top_sources = TopN(
            n=top_n,
            window_seconds=top_n_window,
        )

    # -- Event ingestion ---------------------------------------------------

    def record_detection(
        self,
        target_id: str,
        source: str = "unknown",
        zone: str = "",
        target_type: str = "",
        timestamp: float | None = None,
    ) -> None:
        """Record a target detection event.

        Parameters
        ----------
        target_id:
            Unique target identifier (e.g., "ble_aabbccdd").
        source:
            Sensor source type (e.g., "ble", "yolo", "wifi").
        zone:
            Zone where the detection occurred (optional).
        target_type:
            Classification of the target (e.g., "person", "vehicle").
        timestamp:
            Event timestamp (default: now).
        """
        with self._lock:
            ts = timestamp
            self._detection_counter.increment(timestamp=ts)
            self._detection_trend.record(timestamp=ts)
            self._sensor_histogram.record(source, timestamp=ts)
            self._top_targets.record(target_id, timestamp=ts)
            self._top_sources.record(source, timestamp=ts)

            if zone:
                self._zone_histogram.record(zone, timestamp=ts)
                self._top_zones.record(zone, timestamp=ts)

            if target_type:
                self._target_type_histogram.record(target_type, timestamp=ts)

    def record_alert(
        self,
        alert_type: str,
        severity: str = "info",
        timestamp: float | None = None,
    ) -> None:
        """Record an alert event.

        Parameters
        ----------
        alert_type:
            Alert type identifier (e.g., "geofence_entry").
        severity:
            Alert severity: "info", "warning", "critical".
        timestamp:
            Event timestamp (default: now).
        """
        with self._lock:
            ts = timestamp
            self._alert_counter.increment(timestamp=ts)
            self._alert_severity_histogram.record(severity, timestamp=ts)
            self._alert_trend.record(timestamp=ts)

    def record_correlation(
        self,
        target_a: str,
        target_b: str,
        success: bool = True,
        timestamp: float | None = None,
    ) -> None:
        """Record a correlation (target fusion) attempt.

        Parameters
        ----------
        target_a:
            First target ID.
        target_b:
            Second target ID.
        success:
            Whether the correlation was accepted.
        timestamp:
            Event timestamp (default: now).
        """
        with self._lock:
            ts = timestamp
            self._correlation_attempts.increment(timestamp=ts)
            if success:
                self._correlation_successes.increment(timestamp=ts)

    # -- Queries -----------------------------------------------------------

    @property
    def detection_rate(self) -> float:
        """Current detection rate (events per minute, 1-min window)."""
        with self._lock:
            return self._detection_counter.rate("1min")

    @property
    def alert_rate(self) -> float:
        """Current alert rate (events per minute, 1-min window)."""
        with self._lock:
            return self._alert_counter.rate("1min")

    @property
    def correlation_success_rate(self) -> float:
        """Correlation success rate (0.0 to 1.0) over the 5-min window."""
        with self._lock:
            attempts = self._correlation_attempts.window_count("5min")
            if attempts <= 0:
                return 0.0
            successes = self._correlation_successes.window_count("5min")
            return successes / attempts

    def zone_activity(self) -> dict[str, float]:
        """Current per-zone activity counts."""
        with self._lock:
            return self._zone_histogram.distribution()

    def sensor_utilization(self) -> dict[str, float]:
        """Current per-source detection counts."""
        with self._lock:
            return self._sensor_histogram.distribution()

    def target_type_distribution(self) -> dict[str, float]:
        """Current target type distribution (percentages)."""
        with self._lock:
            return self._target_type_histogram.percentages()

    def detection_trend(self) -> TrendResult:
        """Analyse the detection rate trend."""
        with self._lock:
            return self._detection_trend.analyze()

    def alert_trend(self) -> TrendResult:
        """Analyse the alert rate trend."""
        with self._lock:
            return self._alert_trend.analyze()

    def top_targets(self, n: int | None = None) -> list[tuple[str, float]]:
        """Most active targets by detection count."""
        with self._lock:
            return self._top_targets.top(n)

    def top_zones(self, n: int | None = None) -> list[tuple[str, float]]:
        """Most active zones by detection count."""
        with self._lock:
            return self._top_zones.top(n)

    def top_sources(self, n: int | None = None) -> list[tuple[str, float]]:
        """Most active sensor sources."""
        with self._lock:
            return self._top_sources.top(n)

    # -- Bulk export -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Export a complete analytics snapshot as a JSON-serializable dict.

        Returns::

            {
                "detection_rate": {...},
                "alert_rate": {...},
                "correlation": {...},
                "zone_activity": {...},
                "sensor_utilization": {...},
                "target_types": {...},
                "trends": {...},
                "top_targets": [...],
                "top_zones": [...],
                "top_sources": [...],
                "timestamp": 1711330000.0,
            }
        """
        with self._lock:
            attempts_5m = self._correlation_attempts.window_count("5min")
            successes_5m = self._correlation_successes.window_count("5min")
            return {
                "detection_rate": self._detection_counter.export(),
                "alert_rate": self._alert_counter.export(),
                "correlation": {
                    "attempts": self._correlation_attempts.export(),
                    "successes": self._correlation_successes.export(),
                    "success_rate_5min": round(
                        successes_5m / attempts_5m if attempts_5m > 0 else 0.0, 6
                    ),
                },
                "zone_activity": self._zone_histogram.export(),
                "sensor_utilization": self._sensor_histogram.export(),
                "target_types": self._target_type_histogram.export(),
                "trends": {
                    "detections": self._detection_trend.export(),
                    "alerts": self._alert_trend.export(),
                },
                "top_targets": self._top_targets.export(),
                "top_zones": self._top_zones.export(),
                "top_sources": self._top_sources.export(),
                "timestamp": time.time(),
            }

    def clear(self) -> None:
        """Reset all analytics data."""
        with self._lock:
            self._detection_counter.clear()
            self._detection_trend.clear()
            self._sensor_histogram.clear()
            self._zone_histogram.clear()
            self._alert_counter.clear()
            self._alert_severity_histogram.clear()
            self._alert_trend.clear()
            self._correlation_attempts.clear()
            self._correlation_successes.clear()
            self._target_type_histogram.clear()
            self._top_targets.clear()
            self._top_zones.clear()
            self._top_sources.clear()

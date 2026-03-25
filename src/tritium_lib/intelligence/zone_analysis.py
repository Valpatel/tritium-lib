# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ZoneAnalyzer — zone-level intelligence: activity patterns, hotspots, predictions.

Analyses geofence zones for activity patterns (entry/exit counts, dwell times,
peak hours), predicts future activity based on historical trends, identifies
spatial hotspots on a grid, and compares activity levels across zones.

Integrates with:
  - :class:`~tritium_lib.tracking.heatmap.HeatmapEngine` for spatial event data
  - :class:`~tritium_lib.store.targets.TargetStore` for persistent target history
  - :class:`~tritium_lib.tracking.geofence.GeofenceEngine` for zone event logs

Usage::

    from tritium_lib.intelligence.zone_analysis import ZoneAnalyzer
    from tritium_lib.tracking.heatmap import HeatmapEngine
    from tritium_lib.store.targets import TargetStore

    store = TargetStore(":memory:")
    heatmap = HeatmapEngine()
    analyzer = ZoneAnalyzer(target_store=store, heatmap_engine=heatmap)

    # Record zone events, then analyse
    analyzer.record_zone_event("zone_lobby", "ble_aa", "enter", timestamp=t)
    report = analyzer.analyze_zone("zone_lobby", time_range=(t_start, t_end))
    hotspots = analyzer.find_hotspots(area=(0, 0, 100, 100), resolution=20)
    prediction = analyzer.predict_activity("zone_lobby", future_hours=4)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ZoneEvent:
    """A recorded zone transition event."""

    zone_id: str
    target_id: str
    event_type: str  # "enter", "exit"
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "target_id": self.target_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
        }


@dataclass
class ZoneReport:
    """Aggregated activity report for a single zone over a time range.

    Fields
    ------
    zone_id : str
    time_range : tuple[float, float]
        (start, end) unix timestamps.
    entry_count : int
    exit_count : int
    unique_targets : int
    avg_dwell_seconds : float
        Mean dwell time (seconds) for targets that both entered and exited.
    max_dwell_seconds : float
    min_dwell_seconds : float
    peak_hours : list[int]
        Hours of day (0-23) with the highest entry counts.
    hourly_entries : dict[int, int]
        Entry count per hour of day.
    targets_currently_inside : int
    """

    zone_id: str = ""
    time_range: tuple[float, float] = (0.0, 0.0)
    entry_count: int = 0
    exit_count: int = 0
    unique_targets: int = 0
    avg_dwell_seconds: float = 0.0
    max_dwell_seconds: float = 0.0
    min_dwell_seconds: float = 0.0
    peak_hours: list[int] = field(default_factory=list)
    hourly_entries: dict[int, int] = field(default_factory=dict)
    targets_currently_inside: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "time_range": list(self.time_range),
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "unique_targets": self.unique_targets,
            "avg_dwell_seconds": round(self.avg_dwell_seconds, 2),
            "max_dwell_seconds": round(self.max_dwell_seconds, 2),
            "min_dwell_seconds": round(self.min_dwell_seconds, 2),
            "peak_hours": self.peak_hours,
            "hourly_entries": self.hourly_entries,
            "targets_currently_inside": self.targets_currently_inside,
        }


@dataclass
class ActivityPrediction:
    """Predicted future activity for a zone.

    Fields
    ------
    zone_id : str
    predicted_counts : list[dict]
        One entry per future hour with ``{"hour": int, "predicted_entries": float,
        "confidence": float}``.
    trend : str
        ``"increasing"``, ``"decreasing"``, or ``"stable"``.
    avg_predicted_entries : float
    """

    zone_id: str = ""
    predicted_counts: list[dict] = field(default_factory=list)
    trend: str = "stable"
    avg_predicted_entries: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "predicted_counts": self.predicted_counts,
            "trend": self.trend,
            "avg_predicted_entries": round(self.avg_predicted_entries, 2),
        }


@dataclass
class Hotspot:
    """A high-activity grid cell identified by hotspot analysis."""

    x: float
    y: float
    intensity: float  # normalised 0-1
    event_count: int
    cell_row: int
    cell_col: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "intensity": round(self.intensity, 4),
            "event_count": self.event_count,
            "cell_row": self.cell_row,
            "cell_col": self.cell_col,
        }


@dataclass
class ZoneComparison:
    """Result of comparing activity levels across zones."""

    zone_ids: list[str] = field(default_factory=list)
    rankings: list[dict] = field(default_factory=list)
    busiest_zone: str = ""
    quietest_zone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_ids": self.zone_ids,
            "rankings": self.rankings,
            "busiest_zone": self.busiest_zone,
            "quietest_zone": self.quietest_zone,
        }


# ---------------------------------------------------------------------------
# ZoneAnalyzer
# ---------------------------------------------------------------------------

class ZoneAnalyzer:
    """Analyses geofence zones for activity patterns and predictions.

    Thread-safe.  All public methods acquire the internal lock where
    needed for zone-event storage.  Integration with TargetStore and
    HeatmapEngine is read-only and does not require locking.

    Parameters
    ----------
    target_store : optional
        A :class:`~tritium_lib.store.targets.TargetStore` for querying
        persistent target history.  May be ``None``.
    heatmap_engine : optional
        A :class:`~tritium_lib.tracking.heatmap.HeatmapEngine` for spatial
        event data.  May be ``None``.
    max_events : int
        Maximum zone events retained in memory (FIFO ring buffer).
    """

    def __init__(
        self,
        target_store=None,
        heatmap_engine=None,
        max_events: int = 50_000,
    ) -> None:
        self._target_store = target_store
        self._heatmap_engine = heatmap_engine
        self._max_events = max_events
        self._lock = threading.Lock()

        # Zone events: zone_id -> list[ZoneEvent]
        self._events: dict[str, list[ZoneEvent]] = defaultdict(list)

        # Per-zone hourly baseline: zone_id -> hour -> list[entry_counts]
        # Each count represents entries observed during that hour in a day.
        self._hourly_baselines: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_zone_event(
        self,
        zone_id: str,
        target_id: str,
        event_type: str,
        timestamp: float | None = None,
    ) -> ZoneEvent:
        """Record a zone enter/exit event.

        Parameters
        ----------
        zone_id : str
            The zone identifier.
        target_id : str
            The target that entered/exited.
        event_type : str
            ``"enter"`` or ``"exit"``.
        timestamp : float, optional
            Unix timestamp.  Defaults to ``time.time()``.

        Returns
        -------
        ZoneEvent
        """
        ts = timestamp if timestamp is not None else time.time()
        evt = ZoneEvent(
            zone_id=zone_id,
            target_id=target_id,
            event_type=event_type,
            timestamp=ts,
        )
        with self._lock:
            self._events[zone_id].append(evt)
            # Trim per-zone list
            if len(self._events[zone_id]) > self._max_events:
                self._events[zone_id] = self._events[zone_id][-self._max_events:]

            # Update hourly baseline on enter events
            if event_type == "enter":
                hour = time.localtime(ts).tm_hour
                self._hourly_baselines[zone_id][hour].append(1.0)
                # Trim hourly list
                if len(self._hourly_baselines[zone_id][hour]) > 5000:
                    self._hourly_baselines[zone_id][hour] = (
                        self._hourly_baselines[zone_id][hour][-5000:]
                    )

        return evt

    def record_zone_events_batch(
        self, events: list[dict[str, Any]]
    ) -> int:
        """Bulk-record zone events.

        Each dict should have keys: ``zone_id``, ``target_id``,
        ``event_type``, and optional ``timestamp``.

        Returns the number of events recorded.
        """
        count = 0
        for e in events:
            zone_id = e.get("zone_id", "")
            target_id = e.get("target_id", "")
            event_type = e.get("event_type", "")
            if not zone_id or not target_id or event_type not in ("enter", "exit"):
                continue
            self.record_zone_event(
                zone_id, target_id, event_type, e.get("timestamp")
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    # Zone analysis
    # ------------------------------------------------------------------

    def analyze_zone(
        self,
        zone_id: str,
        time_range: tuple[float, float] | None = None,
    ) -> ZoneReport:
        """Analyse a zone for activity patterns.

        Parameters
        ----------
        zone_id : str
            Zone to analyse.
        time_range : tuple[float, float], optional
            ``(start, end)`` unix timestamps.  Defaults to last 24 hours.

        Returns
        -------
        ZoneReport
        """
        now = time.time()
        if time_range is None:
            time_range = (now - 86400, now)
        start, end = time_range

        with self._lock:
            events = [
                e for e in self._events.get(zone_id, [])
                if start <= e.timestamp <= end
            ]

        entries = [e for e in events if e.event_type == "enter"]
        exits = [e for e in events if e.event_type == "exit"]

        unique_targets = set()
        for e in events:
            unique_targets.add(e.target_id)

        # Hourly entry counts
        hourly_entries: dict[int, int] = defaultdict(int)
        for e in entries:
            hour = time.localtime(e.timestamp).tm_hour
            hourly_entries[hour] += 1

        # Peak hours (top 3 or all tied at max)
        peak_hours: list[int] = []
        if hourly_entries:
            max_count = max(hourly_entries.values())
            peak_hours = sorted(
                [h for h, c in hourly_entries.items() if c == max_count]
            )

        # Dwell times: pair enter/exit per target to compute dwell
        dwell_times = self._compute_dwell_times(events)

        avg_dwell = 0.0
        max_dwell = 0.0
        min_dwell = 0.0
        if dwell_times:
            avg_dwell = sum(dwell_times) / len(dwell_times)
            max_dwell = max(dwell_times)
            min_dwell = min(dwell_times)

        # Targets currently inside: entered but not exited within the range
        targets_inside = self._count_targets_inside(events)

        return ZoneReport(
            zone_id=zone_id,
            time_range=time_range,
            entry_count=len(entries),
            exit_count=len(exits),
            unique_targets=len(unique_targets),
            avg_dwell_seconds=avg_dwell,
            max_dwell_seconds=max_dwell,
            min_dwell_seconds=min_dwell,
            peak_hours=peak_hours,
            hourly_entries=dict(hourly_entries),
            targets_currently_inside=targets_inside,
        )

    # ------------------------------------------------------------------
    # Activity prediction
    # ------------------------------------------------------------------

    def predict_activity(
        self,
        zone_id: str,
        future_hours: int = 4,
    ) -> ActivityPrediction:
        """Predict future target entry counts for a zone.

        Uses per-hour-of-day historical averages to project future
        activity.  If insufficient data exists, returns a flat prediction
        with zero confidence.

        Parameters
        ----------
        zone_id : str
        future_hours : int
            Number of hours into the future to predict (1-48).

        Returns
        -------
        ActivityPrediction
        """
        future_hours = max(1, min(48, future_hours))
        now = time.time()
        current_hour = time.localtime(now).tm_hour

        with self._lock:
            hourly_data = dict(self._hourly_baselines.get(zone_id, {}))

        predicted_counts: list[dict] = []
        total_predicted = 0.0

        for offset in range(future_hours):
            hour = (current_hour + offset + 1) % 24
            values = hourly_data.get(hour, [])

            if values:
                mean = sum(values) / len(values)
                confidence = min(1.0, len(values) / 50.0)
            else:
                mean = 0.0
                confidence = 0.0

            predicted_counts.append({
                "hour": hour,
                "predicted_entries": round(mean, 2),
                "confidence": round(confidence, 2),
            })
            total_predicted += mean

        avg_predicted = total_predicted / future_hours if future_hours > 0 else 0.0

        # Determine trend from the first half vs second half
        trend = "stable"
        if len(predicted_counts) >= 2:
            mid = len(predicted_counts) // 2
            first_half = sum(p["predicted_entries"] for p in predicted_counts[:mid])
            second_half = sum(p["predicted_entries"] for p in predicted_counts[mid:])
            # Account for possibly unequal halves
            first_avg = first_half / mid if mid > 0 else 0
            second_avg = second_half / (len(predicted_counts) - mid) if (len(predicted_counts) - mid) > 0 else 0
            if second_avg > first_avg * 1.2:
                trend = "increasing"
            elif second_avg < first_avg * 0.8:
                trend = "decreasing"

        return ActivityPrediction(
            zone_id=zone_id,
            predicted_counts=predicted_counts,
            trend=trend,
            avg_predicted_entries=avg_predicted,
        )

    # ------------------------------------------------------------------
    # Hotspot detection
    # ------------------------------------------------------------------

    def find_hotspots(
        self,
        area: tuple[float, float, float, float],
        resolution: int = 20,
        time_window_minutes: float = 60,
        threshold: float = 0.3,
    ) -> list[Hotspot]:
        """Identify high-activity grid cells within a rectangular area.

        If a :class:`HeatmapEngine` is attached, uses its events.
        Otherwise falls back to zone events with position data (from
        the TargetStore).

        Parameters
        ----------
        area : tuple
            ``(min_x, min_y, max_x, max_y)`` bounding rectangle.
        resolution : int
            Grid size (resolution x resolution cells).
        time_window_minutes : float
            Look-back window in minutes.
        threshold : float
            Minimum normalised intensity (0-1) to qualify as a hotspot.

        Returns
        -------
        list[Hotspot]
            Hotspot cells sorted by intensity descending.
        """
        resolution = max(2, min(200, resolution))
        min_x, min_y, max_x, max_y = area

        if max_x <= min_x or max_y <= min_y:
            return []

        # Try to use heatmap engine first
        if self._heatmap_engine is not None:
            heatmap = self._heatmap_engine.get_heatmap(
                time_window_minutes=time_window_minutes,
                resolution=resolution,
                layer="all",
            )
            grid = heatmap.get("grid", [])
            max_value = heatmap.get("max_value", 0.0)
            if max_value <= 0.0:
                return []

            hotspots: list[Hotspot] = []
            range_x = max_x - min_x
            range_y = max_y - min_y

            for row_idx in range(len(grid)):
                for col_idx in range(len(grid[row_idx])):
                    val = grid[row_idx][col_idx]
                    if val <= 0:
                        continue
                    intensity = val / max_value
                    if intensity < threshold:
                        continue
                    # Map grid cell back to world coordinates
                    cx = min_x + (col_idx + 0.5) / resolution * range_x
                    cy = min_y + (row_idx + 0.5) / resolution * range_y
                    hotspots.append(Hotspot(
                        x=cx,
                        y=cy,
                        intensity=intensity,
                        event_count=int(val),
                        cell_row=row_idx,
                        cell_col=col_idx,
                    ))

            hotspots.sort(key=lambda h: h.intensity, reverse=True)
            return hotspots

        # Fallback: use target store position history to build a grid
        if self._target_store is not None:
            cutoff = time.time() - time_window_minutes * 60
            return self._hotspots_from_store(
                area, resolution, cutoff, threshold
            )

        return []

    # ------------------------------------------------------------------
    # Zone comparison
    # ------------------------------------------------------------------

    def compare_zones(
        self,
        zone_ids: list[str],
        time_range: tuple[float, float] | None = None,
    ) -> ZoneComparison:
        """Compare activity levels across multiple zones.

        Parameters
        ----------
        zone_ids : list[str]
            Zone identifiers to compare.
        time_range : tuple[float, float], optional
            ``(start, end)`` unix timestamps.  Defaults to last 24 hours.

        Returns
        -------
        ZoneComparison
        """
        if not zone_ids:
            return ZoneComparison()

        rankings: list[dict] = []
        for zid in zone_ids:
            report = self.analyze_zone(zid, time_range=time_range)
            rankings.append({
                "zone_id": zid,
                "entry_count": report.entry_count,
                "exit_count": report.exit_count,
                "unique_targets": report.unique_targets,
                "avg_dwell_seconds": round(report.avg_dwell_seconds, 2),
                "peak_hours": report.peak_hours,
            })

        # Sort by entry_count descending
        rankings.sort(key=lambda r: r["entry_count"], reverse=True)

        busiest = rankings[0]["zone_id"] if rankings else ""
        quietest = rankings[-1]["zone_id"] if rankings else ""

        return ZoneComparison(
            zone_ids=zone_ids,
            rankings=rankings,
            busiest_zone=busiest,
            quietest_zone=quietest,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_zone_events(
        self,
        zone_id: str,
        limit: int = 100,
        event_type: str | None = None,
    ) -> list[ZoneEvent]:
        """Retrieve recorded zone events.

        Parameters
        ----------
        zone_id : str
        limit : int
        event_type : str, optional
            Filter to ``"enter"`` or ``"exit"``.

        Returns
        -------
        list[ZoneEvent]
            Most recent events first.
        """
        with self._lock:
            events = list(self._events.get(zone_id, []))

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return list(reversed(events[-limit:]))

    def get_all_zone_ids(self) -> list[str]:
        """Return all zone IDs that have recorded events."""
        with self._lock:
            return list(self._events.keys())

    def get_stats(self) -> dict[str, Any]:
        """Return engine-wide statistics."""
        with self._lock:
            total_events = sum(len(v) for v in self._events.values())
            zone_count = len(self._events)
        return {
            "total_events": total_events,
            "zone_count": zone_count,
            "max_events_per_zone": self._max_events,
        }

    def clear(self, zone_id: str | None = None) -> None:
        """Clear stored events.  If zone_id is given, clears only that zone."""
        with self._lock:
            if zone_id:
                self._events.pop(zone_id, None)
                self._hourly_baselines.pop(zone_id, None)
            else:
                self._events.clear()
                self._hourly_baselines.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dwell_times(events: list[ZoneEvent]) -> list[float]:
        """Pair enter/exit events per target and compute dwell durations.

        For each target, matches the earliest unmatched enter with the
        next exit to produce a dwell time in seconds.
        """
        # Group by target, sort by time
        by_target: dict[str, list[ZoneEvent]] = defaultdict(list)
        for e in events:
            by_target[e.target_id].append(e)

        dwell_times: list[float] = []
        for tid, tevents in by_target.items():
            tevents.sort(key=lambda e: e.timestamp)
            enter_time: float | None = None
            for ev in tevents:
                if ev.event_type == "enter":
                    enter_time = ev.timestamp
                elif ev.event_type == "exit" and enter_time is not None:
                    dwell = ev.timestamp - enter_time
                    if dwell >= 0:
                        dwell_times.append(dwell)
                    enter_time = None

        return dwell_times

    @staticmethod
    def _count_targets_inside(events: list[ZoneEvent]) -> int:
        """Count targets that entered but have not exited in the event list."""
        inside: set[str] = set()
        # Process chronologically
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        for ev in sorted_events:
            if ev.event_type == "enter":
                inside.add(ev.target_id)
            elif ev.event_type == "exit":
                inside.discard(ev.target_id)
        return len(inside)

    def _hotspots_from_store(
        self,
        area: tuple[float, float, float, float],
        resolution: int,
        cutoff: float,
        threshold: float,
    ) -> list[Hotspot]:
        """Build hotspot grid from TargetStore history data."""
        min_x, min_y, max_x, max_y = area
        range_x = max_x - min_x
        range_y = max_y - min_y

        grid: list[list[int]] = [
            [0] * resolution for _ in range(resolution)
        ]

        try:
            hourly = self._target_store.get_hourly_counts(start_time=cutoff)
        except Exception:
            hourly = []

        # If the store has trajectory data, use it for spatial hotspots
        try:
            all_targets = self._target_store.get_all_targets(since=cutoff)
        except Exception:
            all_targets = []

        for target in all_targets:
            tid = target.get("target_id", "")
            px = target.get("position_x")
            py = target.get("position_y")
            if px is None or py is None:
                continue
            if not (min_x <= px <= max_x and min_y <= py <= max_y):
                continue

            col = int((px - min_x) / range_x * (resolution - 1))
            row = int((py - min_y) / range_y * (resolution - 1))
            col = max(0, min(resolution - 1, col))
            row = max(0, min(resolution - 1, row))
            grid[row][col] += 1

        max_value = max(
            (grid[r][c] for r in range(resolution) for c in range(resolution)),
            default=0,
        )
        if max_value <= 0:
            return []

        hotspots: list[Hotspot] = []
        for row_idx in range(resolution):
            for col_idx in range(resolution):
                val = grid[row_idx][col_idx]
                if val <= 0:
                    continue
                intensity = val / max_value
                if intensity < threshold:
                    continue
                cx = min_x + (col_idx + 0.5) / resolution * range_x
                cy = min_y + (row_idx + 0.5) / resolution * range_y
                hotspots.append(Hotspot(
                    x=cx,
                    y=cy,
                    intensity=intensity,
                    event_count=val,
                    cell_row=row_idx,
                    cell_col=col_idx,
                ))

        hotspots.sort(key=lambda h: h.intensity, reverse=True)
        return hotspots

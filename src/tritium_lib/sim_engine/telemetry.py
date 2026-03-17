# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Session telemetry and performance monitoring for the Tritium sim engine.

Records all performance data and game events during a simulation session
for post-session analysis — like InfluxDB for the sim engine. Supports
time-series metric recording, game event logging, statistical analysis,
CSV/JSON export, HTML report generation, and performance budget grading.

Usage::

    from tritium_lib.sim_engine.telemetry import (
        TelemetrySession, TelemetryDashboard, PerformanceBudget,
        MetricType, TelemetryPoint, GameEvent,
    )

    session = TelemetrySession(metadata={"preset": "urban_riot"})
    session.record_frame(fps=60, frame_time=16.6, draw_calls=80,
                         triangles=50000, entity_count=120,
                         particle_count=30, projectile_count=5)
    session.record_event("kill", {"attacker": "u1", "victim": "u2"})

    stats = session.get_statistics("fps")
    drops = session.find_drops("fps", threshold=30)
    session.save_json("/tmp/session.json")

    budget = PerformanceBudget()
    violations = budget.check(session)
    grade = budget.grade(session)

    dashboard = TelemetryDashboard()
    html = dashboard.generate_report(session)
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# MetricType enum
# ---------------------------------------------------------------------------


class MetricType(Enum):
    """Standard metric categories for simulation telemetry."""

    FPS = "fps"
    FRAME_TIME = "frame_time"
    SIM_TIME = "sim_time"
    DRAW_CALLS = "draw_calls"
    TRIANGLES = "triangles"
    ENTITIES = "entities"
    PARTICLES = "particles"
    PROJECTILES = "projectiles"
    MEMORY = "memory"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# TelemetryPoint dataclass
# ---------------------------------------------------------------------------


@dataclass
class TelemetryPoint:
    """A single metric measurement at a point in time.

    Attributes:
        timestamp: Wall-clock seconds since session start.
        sim_time: In-simulation time in seconds.
        tick: Simulation tick number.
        metric: Name of the metric (e.g. "fps", "entities").
        value: The measured value.
        tags: Optional key-value tags for filtering (e.g. {"phase": "riot"}).
    """

    timestamp: float
    sim_time: float
    tick: int
    metric: str
    value: float
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "timestamp": self.timestamp,
            "sim_time": self.sim_time,
            "tick": self.tick,
            "metric": self.metric,
            "value": self.value,
            "tags": dict(self.tags),
        }


# ---------------------------------------------------------------------------
# GameEvent dataclass
# ---------------------------------------------------------------------------


@dataclass
class GameEvent:
    """A discrete game event (kill, spawn, explosion, phase change, etc.).

    Attributes:
        timestamp: Wall-clock seconds since session start.
        sim_time: In-simulation time in seconds.
        tick: Simulation tick number.
        event_type: Category string (e.g. "kill", "spawn", "explosion").
        data: Arbitrary event payload.
    """

    timestamp: float
    sim_time: float
    tick: int
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "timestamp": self.timestamp,
            "sim_time": self.sim_time,
            "tick": self.tick,
            "event_type": self.event_type,
            "data": dict(self.data),
        }


# ---------------------------------------------------------------------------
# TelemetrySession — core session recorder
# ---------------------------------------------------------------------------


class TelemetrySession:
    """Records and queries telemetry data for a single simulation session.

    Attributes:
        session_id: Unique identifier for this session.
        start_time: Wall-clock time when session started (``time.time()``).
        metrics: Per-metric time-series data.
        events: Chronological list of game events.
        metadata: Arbitrary session metadata (preset, config, etc.).
    """

    def __init__(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex[:12]
        self.start_time: float = time.time()
        self.metrics: dict[str, list[TelemetryPoint]] = {}
        self.events: list[GameEvent] = []
        self.metadata: dict[str, Any] = metadata or {}
        self._tick: int = 0
        self._sim_time: float = 0.0

    # -- internal helpers --------------------------------------------------

    def _elapsed(self) -> float:
        """Wall-clock seconds since session start."""
        return time.time() - self.start_time

    def set_tick(self, tick: int, sim_time: float) -> None:
        """Advance the internal tick/sim_time counters."""
        self._tick = tick
        self._sim_time = sim_time

    # -- recording ---------------------------------------------------------

    def record_metric(
        self,
        metric: str,
        value: float,
        tags: dict[str, Any] | None = None,
    ) -> TelemetryPoint:
        """Append a metric measurement to the time series.

        Args:
            metric: Metric name (use ``MetricType.value`` or any custom string).
            value: Measured value.
            tags: Optional tags for the data point.

        Returns:
            The recorded ``TelemetryPoint``.
        """
        point = TelemetryPoint(
            timestamp=self._elapsed(),
            sim_time=self._sim_time,
            tick=self._tick,
            metric=metric,
            value=value,
            tags=tags or {},
        )
        self.metrics.setdefault(metric, []).append(point)
        return point

    def record_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> GameEvent:
        """Log a discrete game event.

        Args:
            event_type: Event category (e.g. "kill", "spawn", "phase_change").
            data: Arbitrary event payload.

        Returns:
            The recorded ``GameEvent``.
        """
        event = GameEvent(
            timestamp=self._elapsed(),
            sim_time=self._sim_time,
            tick=self._tick,
            event_type=event_type,
            data=data or {},
        )
        self.events.append(event)
        return event

    def record_frame(
        self,
        fps: float,
        frame_time: float,
        draw_calls: int = 0,
        triangles: int = 0,
        entity_count: int = 0,
        particle_count: int = 0,
        projectile_count: int = 0,
    ) -> None:
        """Convenience method to record all standard per-frame metrics at once.

        Args:
            fps: Frames per second.
            frame_time: Time for this frame in milliseconds.
            draw_calls: Number of draw calls.
            triangles: Number of triangles rendered.
            entity_count: Active entity count.
            particle_count: Active particle count.
            projectile_count: Active projectile count.
        """
        self.record_metric(MetricType.FPS.value, fps)
        self.record_metric(MetricType.FRAME_TIME.value, frame_time)
        self.record_metric(MetricType.DRAW_CALLS.value, float(draw_calls))
        self.record_metric(MetricType.TRIANGLES.value, float(triangles))
        self.record_metric(MetricType.ENTITIES.value, float(entity_count))
        self.record_metric(MetricType.PARTICLES.value, float(particle_count))
        self.record_metric(MetricType.PROJECTILES.value, float(projectile_count))

    # -- queries -----------------------------------------------------------

    def get_series(
        self,
        metric: str,
        start: float | None = None,
        end: float | None = None,
    ) -> list[TelemetryPoint]:
        """Return the time series for *metric*, optionally filtered by time range.

        Args:
            metric: Metric name.
            start: Minimum timestamp (inclusive). ``None`` = no lower bound.
            end: Maximum timestamp (inclusive). ``None`` = no upper bound.

        Returns:
            List of ``TelemetryPoint`` objects in chronological order.
        """
        series = self.metrics.get(metric, [])
        if start is not None:
            series = [p for p in series if p.timestamp >= start]
        if end is not None:
            series = [p for p in series if p.timestamp <= end]
        return series

    def get_events(
        self,
        event_type: str | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> list[GameEvent]:
        """Return events, optionally filtered by type and/or time range.

        Args:
            event_type: Filter to this event type. ``None`` = all types.
            start: Minimum timestamp (inclusive).
            end: Maximum timestamp (inclusive).

        Returns:
            List of ``GameEvent`` objects in chronological order.
        """
        result = self.events
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        if start is not None:
            result = [e for e in result if e.timestamp >= start]
        if end is not None:
            result = [e for e in result if e.timestamp <= end]
        return result

    def get_statistics(self, metric: str) -> dict[str, float]:
        """Compute descriptive statistics for *metric*.

        Returns:
            Dict with keys: min, max, avg, p50, p95, p99, std_dev, count.
            Returns empty dict if the metric has no data.
        """
        series = self.metrics.get(metric, [])
        if not series:
            return {}
        values = [p.value for p in series]
        values_sorted = sorted(values)
        n = len(values_sorted)

        def _percentile(pct: float) -> float:
            """Compute *pct*-th percentile (0-100) from sorted values."""
            if n == 1:
                return values_sorted[0]
            k = (pct / 100.0) * (n - 1)
            f = math.floor(k)
            c = min(f + 1, n - 1)
            d = k - f
            return values_sorted[f] + d * (values_sorted[c] - values_sorted[f])

        avg = sum(values) / n
        std_dev = statistics.pstdev(values) if n > 1 else 0.0

        return {
            "min": values_sorted[0],
            "max": values_sorted[-1],
            "avg": avg,
            "p50": _percentile(50),
            "p95": _percentile(95),
            "p99": _percentile(99),
            "std_dev": std_dev,
            "count": float(n),
        }

    def get_timeline(self) -> list[dict[str, Any]]:
        """Return a merged chronological timeline of metrics and events.

        Each entry has a ``"type"`` key of ``"metric"`` or ``"event"`` plus
        the corresponding serialized data.

        Returns:
            Sorted list of dicts.
        """
        entries: list[dict[str, Any]] = []
        for series in self.metrics.values():
            for point in series:
                entries.append({"type": "metric", **point.to_dict()})
        for event in self.events:
            entries.append({"type": "event", **event.to_dict()})
        entries.sort(key=lambda e: e["timestamp"])
        return entries

    def find_drops(
        self,
        metric: str,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Find contiguous regions where *metric* dropped below *threshold*.

        Returns:
            List of dicts with ``start_timestamp``, ``end_timestamp``,
            ``duration``, ``min_value``, ``avg_value``, ``samples``.
        """
        series = self.metrics.get(metric, [])
        if not series:
            return []

        drops: list[dict[str, Any]] = []
        in_drop = False
        drop_points: list[TelemetryPoint] = []

        for point in series:
            if point.value < threshold:
                if not in_drop:
                    in_drop = True
                    drop_points = []
                drop_points.append(point)
            else:
                if in_drop:
                    vals = [p.value for p in drop_points]
                    drops.append({
                        "start_timestamp": drop_points[0].timestamp,
                        "end_timestamp": drop_points[-1].timestamp,
                        "duration": drop_points[-1].timestamp - drop_points[0].timestamp,
                        "min_value": min(vals),
                        "avg_value": sum(vals) / len(vals),
                        "samples": len(drop_points),
                    })
                    in_drop = False
                    drop_points = []

        # Close any open drop at end of series
        if in_drop and drop_points:
            vals = [p.value for p in drop_points]
            drops.append({
                "start_timestamp": drop_points[0].timestamp,
                "end_timestamp": drop_points[-1].timestamp,
                "duration": drop_points[-1].timestamp - drop_points[0].timestamp,
                "min_value": min(vals),
                "avg_value": sum(vals) / len(vals),
                "samples": len(drop_points),
            })

        return drops

    # -- export ------------------------------------------------------------

    def save_json(self, filepath: str) -> None:
        """Export the full session as a JSON file.

        Args:
            filepath: Path to the output JSON file.
        """
        data = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "metadata": self.metadata,
            "metrics": {
                name: [p.to_dict() for p in series]
                for name, series in self.metrics.items()
            },
            "events": [e.to_dict() for e in self.events],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def save_csv(
        self,
        filepath: str,
        metrics: list[str] | None = None,
    ) -> None:
        """Export metrics as CSV (InfluxDB line-protocol compatible).

        Args:
            filepath: Path to the output CSV file.
            metrics: List of metric names to export. ``None`` = all metrics.
        """
        metric_names = metrics or list(self.metrics.keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "sim_time", "tick", "metric", "value", "tags",
            ])
            for name in metric_names:
                for point in self.metrics.get(name, []):
                    tag_str = ",".join(
                        f"{k}={v}" for k, v in sorted(point.tags.items())
                    ) if point.tags else ""
                    writer.writerow([
                        f"{point.timestamp:.6f}",
                        f"{point.sim_time:.6f}",
                        point.tick,
                        point.metric,
                        f"{point.value:.6f}",
                        tag_str,
                    ])

    def summary(self) -> dict[str, Any]:
        """Return a high-level session overview.

        Returns:
            Dict with duration, avg_fps, min_fps, total_events, peak_entities,
            total_metrics, session_id, and metadata.
        """
        fps_stats = self.get_statistics(MetricType.FPS.value)
        entity_series = self.metrics.get(MetricType.ENTITIES.value, [])
        peak_entities = max((p.value for p in entity_series), default=0)

        total_points = sum(len(s) for s in self.metrics.values())

        # Duration from the latest timestamp across all data
        latest = 0.0
        for series in self.metrics.values():
            if series:
                latest = max(latest, series[-1].timestamp)
        for event in self.events:
            latest = max(latest, event.timestamp)

        return {
            "session_id": self.session_id,
            "duration": latest,
            "avg_fps": fps_stats.get("avg", 0.0),
            "min_fps": fps_stats.get("min", 0.0),
            "total_events": len(self.events),
            "peak_entities": peak_entities,
            "total_metric_points": total_points,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# TelemetryDashboard — HTML report generator
# ---------------------------------------------------------------------------


class TelemetryDashboard:
    """Generates a standalone HTML performance report from a telemetry session.

    The output is a single self-contained HTML file using inline SVG charts
    and CSS styling (no external dependencies).
    """

    def generate_report(self, session: TelemetrySession) -> str:
        """Generate a standalone HTML report for *session*.

        Args:
            session: The telemetry session to report on.

        Returns:
            A complete HTML document as a string.
        """
        summary = session.summary()
        fps_stats = session.get_statistics(MetricType.FPS.value)
        frame_time_stats = session.get_statistics(MetricType.FRAME_TIME.value)
        entity_stats = session.get_statistics(MetricType.ENTITIES.value)
        draw_call_stats = session.get_statistics(MetricType.DRAW_CALLS.value)

        # Build SVG charts
        fps_chart = self._svg_line_chart(
            session.get_series(MetricType.FPS.value),
            title="FPS Over Time",
            color="#00f0ff",
            threshold=30,
        )
        entity_chart = self._svg_line_chart(
            session.get_series(MetricType.ENTITIES.value),
            title="Entity Count Over Time",
            color="#05ffa1",
        )
        draw_call_chart = self._svg_line_chart(
            session.get_series(MetricType.DRAW_CALLS.value),
            title="Draw Calls Over Time",
            color="#fcee0a",
        )

        # Event timeline
        event_rows = ""
        for event in session.events[-50:]:  # last 50 events
            event_rows += (
                f"<tr><td>{event.timestamp:.2f}s</td>"
                f"<td>{event.sim_time:.2f}</td>"
                f"<td class='event-type'>{_esc(event.event_type)}</td>"
                f"<td>{_esc(json.dumps(event.data, default=str))}</td></tr>\n"
            )

        # FPS drops
        drops = session.find_drops(MetricType.FPS.value, 30)
        drop_rows = ""
        for drop in drops:
            # Find concurrent events for context
            concurrent = session.get_events(
                start=drop["start_timestamp"] - 0.5,
                end=drop["end_timestamp"] + 0.5,
            )
            context_str = ", ".join(e.event_type for e in concurrent[:5]) or "none"
            drop_rows += (
                f"<tr><td>{drop['start_timestamp']:.2f}s</td>"
                f"<td>{drop['duration']:.2f}s</td>"
                f"<td>{drop['min_value']:.1f}</td>"
                f"<td>{drop['avg_value']:.1f}</td>"
                f"<td>{_esc(context_str)}</td></tr>\n"
            )

        # Stats table helper
        def _stats_row(name: str, stats: dict[str, float]) -> str:
            if not stats:
                return f"<tr><td>{_esc(name)}</td>" + "<td>-</td>" * 6 + "</tr>"
            return (
                f"<tr><td>{_esc(name)}</td>"
                f"<td>{stats['min']:.1f}</td>"
                f"<td>{stats['max']:.1f}</td>"
                f"<td>{stats['avg']:.1f}</td>"
                f"<td>{stats['p50']:.1f}</td>"
                f"<td>{stats['p95']:.1f}</td>"
                f"<td>{stats['p99']:.1f}</td></tr>"
            )

        stats_rows = (
            _stats_row("FPS", fps_stats)
            + _stats_row("Frame Time (ms)", frame_time_stats)
            + _stats_row("Entities", entity_stats)
            + _stats_row("Draw Calls", draw_call_stats)
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Telemetry Report — {_esc(session.session_id)}</title>
<style>
  body {{ background: #0a0a0f; color: #ccc; font-family: 'Courier New', monospace; margin: 20px; }}
  h1 {{ color: #00f0ff; border-bottom: 1px solid #00f0ff; padding-bottom: 8px; }}
  h2 {{ color: #ff2a6d; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; }}
  th {{ background: #1a1a2e; color: #00f0ff; }}
  tr:nth-child(even) {{ background: #0e0e14; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 15px 0; }}
  .summary-card {{ background: #12121a; border: 1px solid #333; padding: 12px; border-radius: 4px; }}
  .summary-card .label {{ color: #888; font-size: 0.85em; }}
  .summary-card .value {{ color: #05ffa1; font-size: 1.4em; font-weight: bold; }}
  .chart-container {{ margin: 15px 0; background: #0e0e14; padding: 10px; border: 1px solid #333; border-radius: 4px; }}
  .event-type {{ color: #fcee0a; }}
  svg {{ width: 100%; }}
</style>
</head>
<body>
<h1>Session Telemetry Report</h1>

<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Session ID</div>
    <div class="value" style="font-size:1em">{_esc(session.session_id)}</div>
  </div>
  <div class="summary-card">
    <div class="label">Duration</div>
    <div class="value">{summary['duration']:.1f}s</div>
  </div>
  <div class="summary-card">
    <div class="label">Avg FPS</div>
    <div class="value">{summary['avg_fps']:.1f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Min FPS</div>
    <div class="value">{summary['min_fps']:.1f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Peak Entities</div>
    <div class="value">{summary['peak_entities']:.0f}</div>
  </div>
  <div class="summary-card">
    <div class="label">Total Events</div>
    <div class="value">{summary['total_events']}</div>
  </div>
</div>

<h2>FPS Over Time</h2>
<div class="chart-container">{fps_chart}</div>

<h2>Entity Count</h2>
<div class="chart-container">{entity_chart}</div>

<h2>Draw Calls</h2>
<div class="chart-container">{draw_call_chart}</div>

<h2>Performance Statistics</h2>
<table>
<tr><th>Metric</th><th>Min</th><th>Max</th><th>Avg</th><th>P50</th><th>P95</th><th>P99</th></tr>
{stats_rows}
</table>

<h2>FPS Drops (&lt;30)</h2>
<table>
<tr><th>Start</th><th>Duration</th><th>Min FPS</th><th>Avg FPS</th><th>Context</th></tr>
{drop_rows if drop_rows else "<tr><td colspan='5'>No FPS drops detected</td></tr>"}
</table>

<h2>Event Timeline (last 50)</h2>
<table>
<tr><th>Time</th><th>Sim Time</th><th>Type</th><th>Data</th></tr>
{event_rows if event_rows else "<tr><td colspan='4'>No events recorded</td></tr>"}
</table>

<p style="color:#555; margin-top:30px; font-size:0.8em">
Generated by Tritium Sim Engine Telemetry &mdash; Copyright 2026 Valpatel Software LLC
</p>
</body>
</html>"""
        return html

    def _svg_line_chart(
        self,
        series: list[TelemetryPoint],
        title: str = "",
        color: str = "#00f0ff",
        width: int = 800,
        height: int = 200,
        threshold: float | None = None,
    ) -> str:
        """Render a simple inline SVG line chart.

        Args:
            series: Data points to chart.
            title: Chart title.
            color: Line stroke color.
            width: SVG width.
            height: SVG height.
            threshold: Optional horizontal threshold line.

        Returns:
            SVG markup string.
        """
        if not series:
            return f'<svg viewBox="0 0 {width} {height}"><text x="50%" y="50%" text-anchor="middle" fill="#555">No data</text></svg>'

        margin_left = 50
        margin_right = 10
        margin_top = 25
        margin_bottom = 25
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        values = [p.value for p in series]
        timestamps = [p.timestamp for p in series]
        v_min = min(values)
        v_max = max(values)
        t_min = timestamps[0]
        t_max = timestamps[-1]

        # Avoid division by zero
        v_range = v_max - v_min if v_max != v_min else 1.0
        t_range = t_max - t_min if t_max != t_min else 1.0

        def _x(t: float) -> float:
            return margin_left + ((t - t_min) / t_range) * plot_w

        def _y(v: float) -> float:
            return margin_top + plot_h - ((v - v_min) / v_range) * plot_h

        # Build polyline points
        points_str = " ".join(
            f"{_x(t):.1f},{_y(v):.1f}" for t, v in zip(timestamps, values)
        )

        # Threshold line
        threshold_line = ""
        if threshold is not None and v_min <= threshold <= v_max:
            ty = _y(threshold)
            threshold_line = (
                f'<line x1="{margin_left}" y1="{ty:.1f}" '
                f'x2="{width - margin_right}" y2="{ty:.1f}" '
                f'stroke="#ff2a6d" stroke-dasharray="4,4" stroke-width="1"/>'
                f'<text x="{width - margin_right - 2}" y="{ty - 3:.1f}" '
                f'fill="#ff2a6d" font-size="10" text-anchor="end">{threshold}</text>'
            )

        # Y-axis labels
        y_labels = ""
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            val = v_min + frac * v_range
            yy = _y(val)
            y_labels += (
                f'<text x="{margin_left - 4}" y="{yy + 3:.1f}" '
                f'fill="#555" font-size="9" text-anchor="end">{val:.0f}</text>'
                f'<line x1="{margin_left}" y1="{yy:.1f}" '
                f'x2="{width - margin_right}" y2="{yy:.1f}" '
                f'stroke="#222" stroke-width="0.5"/>'
            )

        title_el = ""
        if title:
            title_el = (
                f'<text x="{width / 2}" y="14" fill="{color}" '
                f'font-size="12" text-anchor="middle" font-weight="bold">'
                f'{_esc(title)}</text>'
            )

        return (
            f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            f'{title_el}'
            f'{y_labels}'
            f'{threshold_line}'
            f'<polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="1.5"/>'
            f'</svg>'
        )


# ---------------------------------------------------------------------------
# PerformanceBudget — budget checking and grading
# ---------------------------------------------------------------------------


class PerformanceBudget:
    """Define performance budgets and check/grade sessions against them.

    Attributes:
        fps_target: Minimum acceptable FPS (default 60).
        max_entities: Maximum entity count budget (default 500).
        max_particles: Maximum particle count budget (default 200).
        max_draw_calls: Maximum draw call budget (default 100).
        max_frame_time: Maximum frame time in ms (default 16.67, i.e. 60 FPS).
    """

    def __init__(
        self,
        fps_target: float = 60.0,
        max_entities: float = 500.0,
        max_particles: float = 200.0,
        max_draw_calls: float = 100.0,
        max_frame_time: float = 16.67,
    ) -> None:
        self.fps_target = fps_target
        self.max_entities = max_entities
        self.max_particles = max_particles
        self.max_draw_calls = max_draw_calls
        self.max_frame_time = max_frame_time

    def check(self, session: TelemetrySession) -> list[dict[str, Any]]:
        """Check *session* against all budgets and return violations.

        Returns:
            List of dicts, each describing a budget violation with keys:
            ``metric``, ``budget``, ``violations`` (count of samples exceeding
            budget), ``total_samples``, ``worst_value``, ``pct_violated``.
        """
        budgets = [
            (MetricType.FPS.value, self.fps_target, "below"),
            (MetricType.ENTITIES.value, self.max_entities, "above"),
            (MetricType.PARTICLES.value, self.max_particles, "above"),
            (MetricType.DRAW_CALLS.value, self.max_draw_calls, "above"),
            (MetricType.FRAME_TIME.value, self.max_frame_time, "above"),
        ]

        violations: list[dict[str, Any]] = []
        for metric_name, budget_val, direction in budgets:
            series = session.metrics.get(metric_name, [])
            if not series:
                continue

            values = [p.value for p in series]
            total = len(values)

            if direction == "below":
                bad = [v for v in values if v < budget_val]
                worst = min(values) if bad else budget_val
            else:
                bad = [v for v in values if v > budget_val]
                worst = max(values) if bad else budget_val

            if bad:
                violations.append({
                    "metric": metric_name,
                    "budget": budget_val,
                    "direction": direction,
                    "violations": len(bad),
                    "total_samples": total,
                    "worst_value": worst,
                    "pct_violated": (len(bad) / total) * 100.0,
                })

        return violations

    def grade(self, session: TelemetrySession) -> str:
        """Grade *session* based on budget adherence.

        Grading scale:
            A — 0% violations across all budgets
            B — <5% violations
            C — <15% violations
            D — <30% violations
            F — 30%+ violations

        Returns:
            A letter grade string ("A", "B", "C", "D", or "F").
        """
        violations = self.check(session)
        if not violations:
            # Check we actually have data
            total_samples = sum(len(s) for s in session.metrics.values())
            if total_samples == 0:
                return "A"  # No data, no violations
            return "A"

        # Compute overall violation percentage (weighted by sample count)
        total_violated = sum(v["violations"] for v in violations)
        total_samples = sum(v["total_samples"] for v in violations)
        if total_samples == 0:
            return "A"

        pct = (total_violated / total_samples) * 100.0

        if pct < 5.0:
            return "B"
        elif pct < 15.0:
            return "C"
        elif pct < 30.0:
            return "D"
        else:
            return "F"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Minimal HTML escaping for safe embedding."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

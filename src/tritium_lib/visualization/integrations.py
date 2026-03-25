# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Integration helpers — bridge existing Tritium data sources to visualization
data structures.

Each function takes an existing Tritium object (TargetHistory, HeatmapEngine,
AuditTrail, StatsTracker) and converts it to the corresponding visualization
type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .timeline import Timeline, TimelineEvent
from .heatmap_data import HeatmapData, HeatmapBounds
from .chart_series import ChartSeries

if TYPE_CHECKING:
    from tritium_lib.tracking.target_history import TargetHistory
    from tritium_lib.tracking.heatmap import HeatmapEngine
    from tritium_lib.audit.trail import AuditTrail
    from tritium_lib.sim_engine.game.stats import StatsTracker


def timeline_from_target_history(
    history: TargetHistory,
    target_id: str,
    max_points: int = 100,
    title: str | None = None,
) -> Timeline:
    """Convert a target's position history into a :class:`Timeline`.

    Each position record becomes a timeline event with coordinates in
    the metadata.

    Parameters
    ----------
    history : TargetHistory
        The position history tracker.
    target_id : str
        Which target's trail to extract.
    max_points : int
        Maximum trail points to include.
    title : str, optional
        Timeline title; defaults to ``"Target {target_id} Movement"``.
    """
    tl = Timeline(title=title or f"Target {target_id} Movement")
    trail = history.get_trail(target_id, max_points=max_points)

    for i, (x, y, ts) in enumerate(trail):
        tl.add_event(
            timestamp=ts,
            label=f"Position ({x:.1f}, {y:.1f})",
            category="motion",
            metadata={"x": x, "y": y, "index": i},
        )

    return tl


def heatmap_data_from_engine(
    engine: HeatmapEngine,
    layer: str = "all",
    time_window_minutes: float = 60,
    resolution: int = 50,
    title: str | None = None,
) -> HeatmapData:
    """Convert a :class:`HeatmapEngine` grid into a :class:`HeatmapData`.

    Calls ``engine.get_heatmap()`` and wraps the result.

    Parameters
    ----------
    engine : HeatmapEngine
        The heatmap engine with accumulated events.
    layer : str
        Layer to extract (``"all"``, ``"ble_activity"``, etc.).
    time_window_minutes : float
        Time window for event inclusion.
    resolution : int
        Grid resolution.
    title : str, optional
        Display title; defaults to ``"Activity Heatmap ({layer})"``.
    """
    raw = engine.get_heatmap(
        time_window_minutes=time_window_minutes,
        resolution=resolution,
        layer=layer,
    )

    bounds_data = raw.get("bounds", {})
    bounds = HeatmapBounds(
        min_x=bounds_data.get("min_x", 0.0),
        max_x=bounds_data.get("max_x", 1.0),
        min_y=bounds_data.get("min_y", 0.0),
        max_y=bounds_data.get("max_y", 1.0),
    )

    hm = HeatmapData(
        title=title or f"Activity Heatmap ({layer})",
        resolution=raw.get("resolution", resolution),
        bounds=bounds,
    )

    grid = raw.get("grid")
    if grid:
        hm.set_grid(grid)

    return hm


def timeline_from_audit_trail(
    trail: AuditTrail,
    start_time: float | None = None,
    end_time: float | None = None,
    actions: list[str] | None = None,
    limit: int = 500,
    title: str = "Audit Trail",
) -> Timeline:
    """Convert audit trail entries into a :class:`Timeline`.

    Each audit entry becomes a timeline event with severity and actor
    in the metadata.

    Parameters
    ----------
    trail : AuditTrail
        The audit trail instance.
    start_time : float, optional
        Include entries at or after this epoch timestamp.
    end_time : float, optional
        Include entries at or before this epoch timestamp.
    actions : list[str], optional
        Only include these action types.
    limit : int
        Maximum entries.
    title : str
        Timeline title.
    """
    entries = trail.export(
        start_time=start_time,
        end_time=end_time,
        actions=actions,
        limit=limit,
    )

    tl = Timeline(title=title)
    for entry in entries:
        tl.add_event(
            timestamp=entry.get("timestamp", 0.0),
            label=f"{entry.get('action', 'unknown')}: {entry.get('details', '')}".strip(": "),
            category="audit",
            metadata={
                "actor": entry.get("actor", ""),
                "severity": entry.get("severity", "info"),
                "resource": entry.get("resource", ""),
                "resource_id": entry.get("resource_id", ""),
                "entry_id": entry.get("entry_id", ""),
            },
        )

    return tl


def chart_series_from_stats_tracker(
    tracker: StatsTracker,
    metric: str = "kills",
    title: str | None = None,
) -> ChartSeries:
    """Convert per-wave stats into a :class:`ChartSeries`.

    Each completed wave becomes a data point.

    Parameters
    ----------
    tracker : StatsTracker
        The stats tracker with wave data.
    metric : str
        Which wave metric to plot.  Supported:
        ``"kills"`` (hostiles_eliminated), ``"damage_dealt"``,
        ``"damage_taken"``, ``"shots_fired"``, ``"shots_hit"``,
        ``"score"``, ``"duration"``, ``"friendly_losses"``,
        ``"hostiles_spawned"``, ``"hostiles_escaped"``.
    title : str, optional
        Chart title; defaults to ``"{metric} per Wave"``.
    """
    _METRIC_MAP = {
        "kills": "hostiles_eliminated",
        "damage_dealt": "total_damage_dealt",
        "damage_taken": "total_damage_taken",
        "shots_fired": "total_shots_fired",
        "shots_hit": "total_shots_hit",
        "score": "score_earned",
        "duration": "duration",
        "friendly_losses": "friendly_losses",
        "hostiles_spawned": "hostiles_spawned",
        "hostiles_escaped": "hostiles_escaped",
    }

    attr_name = _METRIC_MAP.get(metric, metric)
    series = ChartSeries(
        title=title or f"{metric.replace('_', ' ').title()} per Wave",
        x_label="Wave",
        y_label=metric.replace("_", " ").title(),
        chart_type="bar",
        color="#ff2a6d" if "damage" in metric or "loss" in metric else "#05ffa1",
    )

    for wave in tracker.get_wave_stats():
        wave_dict = wave.to_dict()
        value = wave_dict.get(attr_name, 0)
        series.add_point(
            x=float(wave.wave_number),
            y=float(value),
            label=wave.wave_name,
        )

    return series

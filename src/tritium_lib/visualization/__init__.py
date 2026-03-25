# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.visualization — pure-data structures for charts, timelines,
heatmaps, and network graphs.

All classes are renderer-agnostic: they hold structured data and can export
to Vega-Lite JSON specs or simple SVG strings.  No matplotlib, plotly, or
any rendering library is required.

Data flow
---------
    TargetHistory  -> Timeline      (target movement events)
    HeatmapEngine  -> HeatmapData   (spatial activity grid)
    AuditTrail     -> Timeline      (action history)
    StatsTracker   -> ChartSeries   (combat statistics over waves)

Usage
-----
    from tritium_lib.visualization import (
        Timeline, TimelineEvent, HeatmapData, ChartSeries,
        NetworkGraph, GraphNode, GraphEdge,
    )

    tl = Timeline(title="Target Alpha Movement")
    tl.add_event(1000.0, "First sighting", category="ble")
    tl.add_event(1005.0, "Moved to zone B", category="motion")
    print(tl.to_vega_lite())
    print(tl.to_svg())

    series = ChartSeries(title="Kills per Wave")
    series.add_point(1, 5)
    series.add_point(2, 12)
    print(series.to_vega_lite())

    hm = HeatmapData(title="BLE Activity", resolution=10)
    hm.set_cell(3, 4, 7.5)
    print(hm.to_vega_lite())

    graph = NetworkGraph(title="Entity Relationships")
    graph.add_node("ble_aa", label="Phone")
    graph.add_node("det_person_0", label="Person")
    graph.add_edge("ble_aa", "det_person_0", label="carried_by")
    print(graph.to_vega_lite())
"""

from __future__ import annotations

from .timeline import Timeline, TimelineEvent
from .heatmap_data import HeatmapData
from .chart_series import ChartSeries, DataPoint
from .network_graph import NetworkGraph, GraphNode, GraphEdge
from .integrations import (
    timeline_from_target_history,
    heatmap_data_from_engine,
    timeline_from_audit_trail,
    chart_series_from_stats_tracker,
)

__all__ = [
    "Timeline",
    "TimelineEvent",
    "HeatmapData",
    "ChartSeries",
    "DataPoint",
    "NetworkGraph",
    "GraphNode",
    "GraphEdge",
    "timeline_from_target_history",
    "heatmap_data_from_engine",
    "timeline_from_audit_trail",
    "chart_series_from_stats_tracker",
]

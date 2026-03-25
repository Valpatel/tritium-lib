# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.visualization — 20+ tests covering all data
structures, serialization, Vega-Lite export, SVG generation, and
integration bridges.
"""

from __future__ import annotations

import json

import pytest

from tritium_lib.visualization import (
    Timeline,
    TimelineEvent,
    HeatmapData,
    ChartSeries,
    DataPoint,
    NetworkGraph,
    GraphNode,
    GraphEdge,
    timeline_from_target_history,
    heatmap_data_from_engine,
    timeline_from_audit_trail,
    chart_series_from_stats_tracker,
)
from tritium_lib.visualization.heatmap_data import HeatmapBounds


# ============================================================================
# Timeline Tests
# ============================================================================


class TestTimeline:
    def test_create_empty(self) -> None:
        tl = Timeline(title="Test")
        assert len(tl) == 0
        assert not tl
        assert tl.title == "Test"
        assert tl.start is None
        assert tl.end is None
        assert tl.duration == 0.0

    def test_add_events_sorted(self) -> None:
        tl = Timeline()
        tl.add_event(300.0, "Third")
        tl.add_event(100.0, "First")
        tl.add_event(200.0, "Second")
        assert len(tl) == 3
        assert tl
        labels = [e.label for e in tl.events]
        assert labels == ["First", "Second", "Third"]

    def test_start_end_duration(self) -> None:
        tl = Timeline()
        tl.add_event(10.0, "A")
        tl.add_event(30.0, "B")
        assert tl.start == 10.0
        assert tl.end == 30.0
        assert tl.duration == 20.0

    def test_categories(self) -> None:
        tl = Timeline()
        tl.add_event(1.0, "a", category="ble")
        tl.add_event(2.0, "b", category="wifi")
        tl.add_event(3.0, "c", category="ble")
        assert tl.categories == ["ble", "wifi"]

    def test_filter_by_category(self) -> None:
        tl = Timeline()
        tl.add_event(1.0, "a", category="ble")
        tl.add_event(2.0, "b", category="wifi")
        tl.add_event(3.0, "c", category="ble")
        filtered = tl.filter(category="ble")
        assert len(filtered) == 2
        assert all(e.category == "ble" for e in filtered.events)

    def test_filter_by_time_range(self) -> None:
        tl = Timeline()
        tl.add_event(10.0, "a")
        tl.add_event(20.0, "b")
        tl.add_event(30.0, "c")
        filtered = tl.filter(start=15.0, end=25.0)
        assert len(filtered) == 1
        assert filtered.events[0].label == "b"

    def test_to_dict_round_trip(self) -> None:
        tl = Timeline(title="RT Test")
        tl.add_event(1.0, "ev1", category="ble", metadata={"k": "v"})
        tl.add_event(2.0, "ev2", category="wifi")
        data = tl.to_dict()
        restored = Timeline.from_dict(data)
        assert restored.title == "RT Test"
        assert len(restored) == 2
        assert restored.events[0].metadata == {"k": "v"}

    def test_vega_lite_spec_structure(self) -> None:
        tl = Timeline(title="VL Test")
        tl.add_event(1.0, "a", category="ble")
        tl.add_event(2.0, "b", category="wifi")
        spec = tl.to_vega_lite()
        assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite")
        assert spec["title"] == "VL Test"
        assert spec["mark"]["type"] == "tick"
        assert len(spec["data"]["values"]) == 2

    def test_vega_lite_json_is_valid(self) -> None:
        tl = Timeline()
        tl.add_event(1.0, "a")
        json_str = tl.to_vega_lite_json()
        parsed = json.loads(json_str)
        assert "$schema" in parsed

    def test_svg_output(self) -> None:
        tl = Timeline(title="SVG Test")
        tl.add_event(1.0, "a", category="ble")
        tl.add_event(5.0, "b", category="camera")
        svg = tl.to_svg()
        assert svg.startswith("<svg")
        assert "SVG Test" in svg
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg

    def test_svg_empty(self) -> None:
        tl = Timeline()
        svg = tl.to_svg()
        assert "No events" in svg

    def test_add_existing_event(self) -> None:
        tl = Timeline()
        evt = TimelineEvent(timestamp=5.0, label="manual", category="system")
        tl.add(evt)
        assert len(tl) == 1
        assert tl.events[0].label == "manual"

    def test_clear(self) -> None:
        tl = Timeline()
        tl.add_event(1.0, "a")
        tl.clear()
        assert len(tl) == 0


# ============================================================================
# TimelineEvent Tests
# ============================================================================


class TestTimelineEvent:
    def test_to_dict(self) -> None:
        evt = TimelineEvent(timestamp=10.0, label="test", category="ble", metadata={"x": 1})
        d = evt.to_dict()
        assert d["timestamp"] == 10.0
        assert d["label"] == "test"
        assert d["metadata"] == {"x": 1}

    def test_from_dict(self) -> None:
        data = {"timestamp": 5.0, "label": "restored", "category": "wifi"}
        evt = TimelineEvent.from_dict(data)
        assert evt.timestamp == 5.0
        assert evt.label == "restored"
        assert evt.category == "wifi"


# ============================================================================
# HeatmapData Tests
# ============================================================================


class TestHeatmapData:
    def test_create_default(self) -> None:
        hm = HeatmapData(resolution=10)
        assert hm.resolution == 10
        assert hm.max_value == 0.0
        assert hm.nonzero_count == 0

    def test_set_and_get_cell(self) -> None:
        hm = HeatmapData(resolution=5)
        hm.set_cell(2, 3, 7.5)
        assert hm.get_cell(2, 3) == 7.5
        assert hm.max_value == 7.5
        assert hm.nonzero_count == 1

    def test_add_to_cell(self) -> None:
        hm = HeatmapData(resolution=5)
        hm.set_cell(1, 1, 3.0)
        hm.add_to_cell(1, 1, 2.0)
        assert hm.get_cell(1, 1) == 5.0

    def test_clear(self) -> None:
        hm = HeatmapData(resolution=5)
        hm.set_cell(0, 0, 10.0)
        hm.clear()
        assert hm.max_value == 0.0

    def test_total(self) -> None:
        hm = HeatmapData(resolution=3)
        hm.set_cell(0, 0, 1.0)
        hm.set_cell(1, 1, 2.0)
        hm.set_cell(2, 2, 3.0)
        assert hm.total == 6.0

    def test_set_grid(self) -> None:
        hm = HeatmapData(resolution=2)
        hm.set_grid([[1.0, 2.0], [3.0, 4.0]])
        assert hm.get_cell(0, 0) == 1.0
        assert hm.get_cell(1, 1) == 4.0

    def test_set_grid_wrong_size(self) -> None:
        hm = HeatmapData(resolution=2)
        with pytest.raises(ValueError, match="rows"):
            hm.set_grid([[1.0]])
        with pytest.raises(ValueError, match="cols"):
            hm.set_grid([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_bounds(self) -> None:
        bounds = HeatmapBounds(min_x=-10, max_x=10, min_y=-5, max_y=5)
        hm = HeatmapData(resolution=5, bounds=bounds)
        assert hm.bounds.min_x == -10
        assert hm.bounds.max_y == 5

    def test_out_of_range_get(self) -> None:
        hm = HeatmapData(resolution=5)
        assert hm.get_cell(-1, -1) == 0.0
        assert hm.get_cell(99, 99) == 0.0

    def test_to_dict_round_trip(self) -> None:
        hm = HeatmapData(title="Test HM", resolution=3)
        hm.set_cell(1, 2, 5.0)
        data = hm.to_dict()
        restored = HeatmapData.from_dict(data)
        assert restored.title == "Test HM"
        assert restored.get_cell(1, 2) == 5.0

    def test_vega_lite_spec(self) -> None:
        hm = HeatmapData(title="VL HM", resolution=3)
        hm.set_cell(0, 0, 1.0)
        spec = hm.to_vega_lite()
        assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite")
        assert spec["mark"] == "rect"
        assert len(spec["data"]["values"]) == 1  # Only nonzero cells

    def test_svg_output(self) -> None:
        hm = HeatmapData(title="SVG HM", resolution=5)
        hm.set_cell(2, 2, 10.0)
        svg = hm.to_svg()
        assert svg.startswith("<svg")
        assert "SVG HM" in svg
        assert "<rect" in svg


# ============================================================================
# ChartSeries Tests
# ============================================================================


class TestChartSeries:
    def test_create_empty(self) -> None:
        cs = ChartSeries(title="Empty")
        assert len(cs) == 0
        assert not cs
        assert cs.y_mean == 0.0

    def test_add_points(self) -> None:
        cs = ChartSeries()
        cs.add_point(1, 10)
        cs.add_point(2, 20)
        cs.add_point(3, 30)
        assert len(cs) == 3
        assert cs
        assert cs.x_values == [1, 2, 3]
        assert cs.y_values == [10, 20, 30]

    def test_add_points_bulk(self) -> None:
        cs = ChartSeries()
        cs.add_points([(1, 10), (2, 20)])
        assert len(cs) == 2

    def test_min_max_mean(self) -> None:
        cs = ChartSeries()
        cs.add_points([(1, 10), (2, 20), (3, 30)])
        assert cs.x_min == 1
        assert cs.x_max == 3
        assert cs.y_min == 10
        assert cs.y_max == 30
        assert cs.y_mean == 20.0

    def test_sort_by_x(self) -> None:
        cs = ChartSeries()
        cs.add_point(3, 30)
        cs.add_point(1, 10)
        cs.add_point(2, 20)
        cs.sort_by_x()
        assert cs.x_values == [1, 2, 3]

    def test_chart_type_validation(self) -> None:
        cs = ChartSeries(chart_type="invalid")
        assert cs.chart_type == "line"  # Falls back to line

    def test_to_dict_round_trip(self) -> None:
        cs = ChartSeries(title="RT", x_label="Wave", y_label="Kills", chart_type="bar")
        cs.add_point(1, 5, label="W1")
        data = cs.to_dict()
        restored = ChartSeries.from_dict(data)
        assert restored.title == "RT"
        assert restored.chart_type == "bar"
        assert len(restored) == 1
        assert restored.points[0].label == "W1"

    def test_vega_lite_line(self) -> None:
        cs = ChartSeries(title="Line", chart_type="line")
        cs.add_points([(1, 10), (2, 20)])
        spec = cs.to_vega_lite()
        assert spec["mark"]["type"] == "line"
        assert spec["mark"]["point"] is True

    def test_vega_lite_bar(self) -> None:
        cs = ChartSeries(title="Bar", chart_type="bar")
        cs.add_points([(1, 10), (2, 20)])
        spec = cs.to_vega_lite()
        assert spec["mark"]["type"] == "bar"

    def test_svg_line(self) -> None:
        cs = ChartSeries(title="SVG Line", chart_type="line")
        cs.add_points([(1, 10), (2, 20), (3, 15)])
        svg = cs.to_svg()
        assert svg.startswith("<svg")
        assert "<path" in svg
        assert "<circle" in svg

    def test_svg_bar(self) -> None:
        cs = ChartSeries(title="SVG Bar", chart_type="bar")
        cs.add_points([(1, 10), (2, 20)])
        svg = cs.to_svg()
        assert "<rect" in svg

    def test_svg_empty(self) -> None:
        cs = ChartSeries()
        svg = cs.to_svg()
        assert "No data" in svg

    def test_clear(self) -> None:
        cs = ChartSeries()
        cs.add_point(1, 10)
        cs.clear()
        assert len(cs) == 0


# ============================================================================
# NetworkGraph Tests
# ============================================================================


class TestNetworkGraph:
    def test_create_empty(self) -> None:
        g = NetworkGraph(title="G")
        assert g.node_count == 0
        assert g.edge_count == 0
        assert not g

    def test_add_nodes_and_edges(self) -> None:
        g = NetworkGraph()
        g.add_node("a", label="Alpha", group="ble")
        g.add_node("b", label="Beta", group="camera")
        g.add_edge("a", "b", label="detected_with")
        assert g.node_count == 2
        assert g.edge_count == 1
        assert g

    def test_neighbors(self) -> None:
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert g.neighbors("a") == ["b", "c"]
        assert g.neighbors("b") == ["a"]

    def test_neighbors_directed(self) -> None:
        g = NetworkGraph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        assert g.neighbors("a") == ["b"]
        assert g.neighbors("b") == []  # Directed: no back-link

    def test_degree(self) -> None:
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert g.degree("a") == 2
        assert g.degree("b") == 1

    def test_add_edge_missing_node(self) -> None:
        g = NetworkGraph()
        g.add_node("a")
        with pytest.raises(ValueError, match="not found"):
            g.add_edge("a", "nonexistent")

    def test_remove_node(self) -> None:
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.remove_node("a")
        assert g.node_count == 1
        assert g.edge_count == 0

    def test_remove_edge(self) -> None:
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.remove_edge("a", "b")
        assert g.edge_count == 0
        assert g.node_count == 2

    def test_groups(self) -> None:
        g = NetworkGraph()
        g.add_node("a", group="ble")
        g.add_node("b", group="camera")
        g.add_node("c", group="ble")
        assert g.groups == ["ble", "camera"]

    def test_circular_layout(self) -> None:
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.circular_layout(radius=50, cx=100, cy=100)
        # All nodes should have non-zero positions near (100,100)
        for n in g.nodes:
            dist = ((n.x - 100) ** 2 + (n.y - 100) ** 2) ** 0.5
            assert abs(dist - 50) < 0.1

    def test_to_dict_round_trip(self) -> None:
        g = NetworkGraph(title="RT Graph", directed=True)
        g.add_node("a", label="Alpha", group="ble")
        g.add_node("b", label="Beta")
        g.add_edge("a", "b", label="linked", weight=0.9)
        data = g.to_dict()
        restored = NetworkGraph.from_dict(data)
        assert restored.title == "RT Graph"
        assert restored.directed is True
        assert restored.node_count == 2
        assert restored.edge_count == 1
        assert restored.edges[0].weight == 0.9

    def test_vega_lite_spec(self) -> None:
        g = NetworkGraph(title="VL Graph")
        g.add_node("a", group="ble")
        g.add_node("b", group="camera")
        g.add_edge("a", "b")
        spec = g.to_vega_lite()
        assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite")
        assert len(spec["layer"]) == 3  # edges, nodes, labels

    def test_svg_output(self) -> None:
        g = NetworkGraph(title="SVG Graph")
        g.add_node("a", group="ble")
        g.add_node("b", group="camera")
        g.add_edge("a", "b", label="link")
        svg = g.to_svg()
        assert svg.startswith("<svg")
        assert "SVG Graph" in svg
        assert "<circle" in svg
        assert "<line" in svg

    def test_svg_empty(self) -> None:
        g = NetworkGraph()
        svg = g.to_svg()
        assert "No nodes" in svg

    def test_clear(self) -> None:
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.clear()
        assert g.node_count == 0
        assert g.edge_count == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegrationTargetHistory:
    def test_timeline_from_target_history(self) -> None:
        from tritium_lib.tracking.target_history import TargetHistory

        h = TargetHistory()
        h.record("tgt1", (10.0, 20.0), timestamp=100.0)
        h.record("tgt1", (15.0, 25.0), timestamp=105.0)
        h.record("tgt1", (20.0, 30.0), timestamp=110.0)

        tl = timeline_from_target_history(h, "tgt1")
        assert len(tl) == 3
        assert tl.title == "Target tgt1 Movement"
        assert tl.events[0].metadata["x"] == 10.0
        assert tl.events[2].timestamp == 110.0

    def test_timeline_from_empty_history(self) -> None:
        from tritium_lib.tracking.target_history import TargetHistory

        h = TargetHistory()
        tl = timeline_from_target_history(h, "nonexistent")
        assert len(tl) == 0


class TestIntegrationHeatmap:
    def test_heatmap_data_from_engine(self) -> None:
        from tritium_lib.tracking.heatmap import HeatmapEngine

        engine = HeatmapEngine()
        engine.record_event("ble_activity", 5.0, 5.0, weight=3.0)
        engine.record_event("ble_activity", 5.1, 5.1, weight=2.0)
        engine.record_event("ble_activity", 8.0, 8.0, weight=1.0)

        hm = heatmap_data_from_engine(engine, layer="ble_activity", resolution=10)
        assert hm.resolution == 10
        assert hm.title == "Activity Heatmap (ble_activity)"
        assert hm.max_value > 0

    def test_heatmap_from_empty_engine(self) -> None:
        from tritium_lib.tracking.heatmap import HeatmapEngine

        engine = HeatmapEngine()
        hm = heatmap_data_from_engine(engine, resolution=5)
        assert hm.max_value == 0.0


class TestIntegrationAuditTrail:
    def test_timeline_from_audit_trail(self) -> None:
        from tritium_lib.audit import AuditTrail

        trail = AuditTrail(":memory:")
        trail.record(
            actor="user:analyst",
            action="target_accessed",
            details="Viewed dossier",
            timestamp=1000.0,
        )
        trail.record(
            actor="user:admin",
            action="config_changed",
            details="Updated broker URL",
            timestamp=1005.0,
        )

        tl = timeline_from_audit_trail(trail, title="Audit Events")
        assert len(tl) == 2
        assert tl.title == "Audit Events"
        assert tl.events[0].category == "audit"
        assert tl.events[0].metadata["actor"] == "user:analyst"
        trail.close()


class TestIntegrationStatsTracker:
    def test_chart_series_from_stats_tracker(self) -> None:
        from tritium_lib.sim_engine.game.stats import StatsTracker

        tracker = StatsTracker()
        tracker.on_wave_start(1, "Wave 1", 10)
        tracker.on_kill("u1", "h1")
        tracker.on_kill("u1", "h2")
        tracker.on_wave_complete(100)

        tracker.on_wave_start(2, "Wave 2", 15)
        tracker.on_kill("u1", "h3")
        tracker.on_kill("u1", "h4")
        tracker.on_kill("u1", "h5")
        tracker.on_wave_complete(200)

        cs = chart_series_from_stats_tracker(tracker, metric="kills")
        assert len(cs) == 2
        assert cs.chart_type == "bar"
        assert cs.points[0].x == 1.0
        assert cs.points[0].y == 2.0  # Wave 1: 2 kills
        assert cs.points[1].y == 3.0  # Wave 2: 3 kills

    def test_chart_series_score_metric(self) -> None:
        from tritium_lib.sim_engine.game.stats import StatsTracker

        tracker = StatsTracker()
        tracker.on_wave_start(1, "W1", 5)
        tracker.on_wave_complete(100)
        tracker.on_wave_start(2, "W2", 8)
        tracker.on_wave_complete(250)

        cs = chart_series_from_stats_tracker(tracker, metric="score")
        assert cs.points[0].y == 100.0
        assert cs.points[1].y == 250.0

    def test_chart_series_empty_tracker(self) -> None:
        from tritium_lib.sim_engine.game.stats import StatsTracker

        tracker = StatsTracker()
        cs = chart_series_from_stats_tracker(tracker, metric="kills")
        assert len(cs) == 0


# ============================================================================
# DataPoint Tests
# ============================================================================


class TestDataPoint:
    def test_to_dict_with_label(self) -> None:
        dp = DataPoint(x=1.0, y=2.0, label="wave1")
        d = dp.to_dict()
        assert d == {"x": 1.0, "y": 2.0, "label": "wave1"}

    def test_to_dict_without_label(self) -> None:
        dp = DataPoint(x=1.0, y=2.0)
        d = dp.to_dict()
        assert d == {"x": 1.0, "y": 2.0}
        assert "label" not in d


# ============================================================================
# Cross-cutting export tests
# ============================================================================


class TestExportConsistency:
    """Verify that all types produce valid Vega-Lite and valid SVG."""

    def _valid_vega(self, spec: dict) -> None:
        """Basic structural checks on a Vega-Lite spec."""
        assert "$schema" in spec
        assert "data" in spec or "layer" in spec

    def _valid_svg(self, svg: str) -> None:
        """Basic structural checks on SVG output."""
        assert svg.strip().startswith("<svg")
        assert "</svg>" in svg

    def test_all_types_vega(self) -> None:
        tl = Timeline()
        tl.add_event(1.0, "a")
        self._valid_vega(tl.to_vega_lite())

        hm = HeatmapData(resolution=3)
        hm.set_cell(0, 0, 1.0)
        self._valid_vega(hm.to_vega_lite())

        cs = ChartSeries()
        cs.add_point(1, 10)
        self._valid_vega(cs.to_vega_lite())

        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        self._valid_vega(g.to_vega_lite())

    def test_all_types_svg(self) -> None:
        tl = Timeline()
        tl.add_event(1.0, "a")
        self._valid_svg(tl.to_svg())

        hm = HeatmapData(resolution=3)
        hm.set_cell(0, 0, 1.0)
        self._valid_svg(hm.to_svg())

        cs = ChartSeries()
        cs.add_point(1, 10)
        self._valid_svg(cs.to_svg())

        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        self._valid_svg(g.to_svg())

    def test_all_types_json_serializable(self) -> None:
        """Verify that to_vega_lite output is JSON-serializable."""
        tl = Timeline()
        tl.add_event(1.0, "a")
        json.dumps(tl.to_vega_lite())

        hm = HeatmapData(resolution=3)
        json.dumps(hm.to_vega_lite())

        cs = ChartSeries()
        cs.add_point(1, 10)
        json.dumps(cs.to_vega_lite())

        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        json.dumps(g.to_vega_lite())

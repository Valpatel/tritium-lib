# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.visualization.chart_series — chart data structures."""

from tritium_lib.visualization.chart_series import ChartSeries, DataPoint


class TestDataPoint:
    def test_basic(self):
        p = DataPoint(x=1.0, y=2.0, label="A")
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.label == "A"

    def test_to_dict(self):
        p = DataPoint(x=1.0, y=2.0)
        d = p.to_dict()
        assert d == {"x": 1.0, "y": 2.0}

    def test_to_dict_with_label(self):
        p = DataPoint(x=1.0, y=2.0, label="peak")
        d = p.to_dict()
        assert d["label"] == "peak"


class TestChartSeries:
    def test_default(self):
        cs = ChartSeries()
        assert cs.title == "Chart"
        assert cs.chart_type == "line"
        assert len(cs) == 0
        assert bool(cs) is False

    def test_add_point(self):
        cs = ChartSeries()
        pt = cs.add_point(1.0, 2.0, "A")
        assert len(cs) == 1
        assert pt.x == 1.0
        assert bool(cs) is True

    def test_add_points(self):
        cs = ChartSeries()
        cs.add_points([(1, 10), (2, 20), (3, 30)])
        assert len(cs) == 3

    def test_clear(self):
        cs = ChartSeries()
        cs.add_points([(1, 10), (2, 20)])
        cs.clear()
        assert len(cs) == 0

    def test_sort_by_x(self):
        cs = ChartSeries()
        cs.add_points([(3, 30), (1, 10), (2, 20)])
        cs.sort_by_x()
        assert cs.x_values == [1.0, 2.0, 3.0]

    def test_points_copy(self):
        cs = ChartSeries()
        cs.add_point(1.0, 2.0)
        pts = cs.points
        pts.clear()
        assert len(cs) == 1

    def test_x_values(self):
        cs = ChartSeries()
        cs.add_points([(1, 10), (2, 20)])
        assert cs.x_values == [1.0, 2.0]

    def test_y_values(self):
        cs = ChartSeries()
        cs.add_points([(1, 10), (2, 20)])
        assert cs.y_values == [10.0, 20.0]

    def test_x_min_max(self):
        cs = ChartSeries()
        cs.add_points([(5, 0), (1, 0), (10, 0)])
        assert cs.x_min == 1.0
        assert cs.x_max == 10.0

    def test_y_min_max(self):
        cs = ChartSeries()
        cs.add_points([(0, 5), (0, 1), (0, 10)])
        assert cs.y_min == 1.0
        assert cs.y_max == 10.0

    def test_empty_min_max(self):
        cs = ChartSeries()
        assert cs.x_min == 0.0
        assert cs.x_max == 0.0
        assert cs.y_min == 0.0
        assert cs.y_max == 0.0

    def test_y_mean(self):
        cs = ChartSeries()
        cs.add_points([(0, 10), (0, 20), (0, 30)])
        assert cs.y_mean == 20.0

    def test_y_mean_empty(self):
        cs = ChartSeries()
        assert cs.y_mean == 0.0

    def test_invalid_chart_type(self):
        cs = ChartSeries(chart_type="scatter")
        assert cs.chart_type == "line"

    def test_bar_chart_type(self):
        cs = ChartSeries(chart_type="bar")
        assert cs.chart_type == "bar"

    def test_to_dict(self):
        cs = ChartSeries(title="Test", x_label="Time", y_label="Value", color="#ff0000")
        cs.add_point(1.0, 2.0)
        d = cs.to_dict()
        assert d["title"] == "Test"
        assert d["x_label"] == "Time"
        assert d["y_label"] == "Value"
        assert d["color"] == "#ff0000"
        assert len(d["points"]) == 1

    def test_from_dict(self):
        data = {
            "title": "Restored",
            "x_label": "X",
            "y_label": "Y",
            "color": "#00ff00",
            "chart_type": "bar",
            "points": [{"x": 1, "y": 2, "label": "A"}],
        }
        cs = ChartSeries.from_dict(data)
        assert cs.title == "Restored"
        assert cs.chart_type == "bar"
        assert len(cs) == 1
        assert cs.points[0].label == "A"

    def test_roundtrip(self):
        cs = ChartSeries(title="Round", chart_type="line")
        cs.add_points([(1, 10), (2, 20), (3, 30)])
        restored = ChartSeries.from_dict(cs.to_dict())
        assert restored.title == "Round"
        assert len(restored) == 3
        assert restored.y_values == [10.0, 20.0, 30.0]

    def test_to_vega_lite(self):
        cs = ChartSeries(title="VL Test")
        cs.add_points([(1, 10), (2, 20)])
        spec = cs.to_vega_lite()
        assert spec["title"] == "VL Test"
        assert "$schema" in spec
        assert spec["mark"]["type"] == "line"
        assert len(spec["data"]["values"]) == 2

    def test_to_vega_lite_bar(self):
        cs = ChartSeries(chart_type="bar")
        cs.add_point(1, 10)
        spec = cs.to_vega_lite()
        assert spec["mark"]["type"] == "bar"
        assert spec["mark"]["point"] is False

    def test_to_vega_lite_json(self):
        cs = ChartSeries()
        cs.add_point(1, 10)
        j = cs.to_vega_lite_json()
        assert '"$schema"' in j

    def test_to_svg_empty(self):
        cs = ChartSeries()
        svg = cs.to_svg()
        assert "No data" in svg

    def test_to_svg_line(self):
        cs = ChartSeries(title="SVG Line")
        cs.add_points([(1, 10), (2, 20), (3, 30)])
        svg = cs.to_svg()
        assert "<svg" in svg
        assert "SVG Line" in svg
        assert "<path" in svg
        assert "<circle" in svg

    def test_to_svg_bar(self):
        cs = ChartSeries(title="SVG Bar", chart_type="bar")
        cs.add_points([(1, 10), (2, 20)])
        svg = cs.to_svg()
        assert "<rect" in svg

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.visualization.heatmap_data — 2D grid data."""

import pytest

from tritium_lib.visualization.heatmap_data import HeatmapBounds, HeatmapData


class TestHeatmapBounds:
    def test_defaults(self):
        b = HeatmapBounds()
        assert b.min_x == 0.0
        assert b.max_x == 1.0

    def test_to_dict(self):
        b = HeatmapBounds(min_x=-10, max_x=10, min_y=-5, max_y=5)
        d = b.to_dict()
        assert d["min_x"] == -10
        assert d["max_y"] == 5


class TestHeatmapData:
    def test_default(self):
        hm = HeatmapData()
        assert hm.title == "Heatmap"
        assert hm.resolution == 50
        assert hm.max_value == 0.0
        assert hm.total == 0.0
        assert hm.nonzero_count == 0

    def test_set_cell(self):
        hm = HeatmapData(resolution=10)
        hm.set_cell(5, 5, 1.0)
        assert hm.get_cell(5, 5) == 1.0
        assert hm.max_value == 1.0
        assert hm.nonzero_count == 1

    def test_set_cell_clamp(self):
        hm = HeatmapData(resolution=5)
        hm.set_cell(100, 100, 1.0)
        assert hm.get_cell(4, 4) == 1.0

    def test_set_cell_clamp_negative(self):
        hm = HeatmapData(resolution=5)
        hm.set_cell(-1, -1, 2.0)
        assert hm.get_cell(0, 0) == 2.0

    def test_add_to_cell(self):
        hm = HeatmapData(resolution=10)
        hm.set_cell(0, 0, 3.0)
        hm.add_to_cell(0, 0, 2.0)
        assert hm.get_cell(0, 0) == 5.0

    def test_get_cell_out_of_range(self):
        hm = HeatmapData(resolution=5)
        assert hm.get_cell(100, 100) == 0.0
        assert hm.get_cell(-1, -1) == 0.0

    def test_set_grid(self):
        hm = HeatmapData(resolution=3)
        grid = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        hm.set_grid(grid)
        assert hm.get_cell(0, 0) == 1
        assert hm.get_cell(1, 1) == 2
        assert hm.get_cell(2, 2) == 3

    def test_set_grid_wrong_rows(self):
        hm = HeatmapData(resolution=3)
        with pytest.raises(ValueError, match="rows"):
            hm.set_grid([[1, 2]])

    def test_set_grid_wrong_cols(self):
        hm = HeatmapData(resolution=3)
        with pytest.raises(ValueError, match="cols"):
            hm.set_grid([[1, 2], [3, 4], [5, 6]])

    def test_clear(self):
        hm = HeatmapData(resolution=5)
        hm.set_cell(0, 0, 10.0)
        hm.clear()
        assert hm.get_cell(0, 0) == 0.0
        assert hm.total == 0.0

    def test_grid_returns_copy(self):
        hm = HeatmapData(resolution=3)
        hm.set_cell(0, 0, 5.0)
        g = hm.grid
        g[0][0] = 99.0
        assert hm.get_cell(0, 0) == 5.0

    def test_min_value(self):
        hm = HeatmapData(resolution=3)
        hm.set_cell(0, 0, 5.0)
        hm.set_cell(1, 1, -3.0)
        assert hm.min_value == -3.0

    def test_total(self):
        hm = HeatmapData(resolution=3)
        hm.set_cell(0, 0, 1.0)
        hm.set_cell(1, 1, 2.0)
        hm.set_cell(2, 2, 3.0)
        assert hm.total == 6.0

    def test_resolution_clamp(self):
        hm = HeatmapData(resolution=-5)
        assert hm.resolution == 1

    def test_to_dict(self):
        hm = HeatmapData(title="Test", resolution=3)
        hm.set_cell(0, 0, 1.0)
        d = hm.to_dict()
        assert d["title"] == "Test"
        assert d["resolution"] == 3
        assert "bounds" in d
        assert "grid" in d

    def test_from_dict(self):
        data = {
            "title": "Restored",
            "resolution": 3,
            "bounds": {"min_x": 0, "max_x": 100, "min_y": 0, "max_y": 100},
            "grid": [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        }
        hm = HeatmapData.from_dict(data)
        assert hm.title == "Restored"
        assert hm.get_cell(0, 0) == 1
        assert hm.get_cell(2, 2) == 3

    def test_roundtrip(self):
        hm = HeatmapData(title="RT", resolution=5)
        hm.set_cell(2, 3, 7.5)
        restored = HeatmapData.from_dict(hm.to_dict())
        assert restored.get_cell(2, 3) == 7.5
        assert restored.title == "RT"

    def test_to_vega_lite(self):
        hm = HeatmapData(title="VL", resolution=3)
        hm.set_cell(1, 1, 5.0)
        spec = hm.to_vega_lite()
        assert spec["title"] == "VL"
        assert spec["mark"] == "rect"
        assert len(spec["data"]["values"]) == 1

    def test_to_vega_lite_json(self):
        hm = HeatmapData(resolution=3)
        hm.set_cell(0, 0, 1.0)
        j = hm.to_vega_lite_json()
        assert '"mark"' in j

    def test_to_svg(self):
        hm = HeatmapData(title="SVG Heat", resolution=5)
        hm.set_cell(0, 0, 1.0)
        hm.set_cell(2, 2, 2.0)
        svg = hm.to_svg()
        assert "<svg" in svg
        assert "<rect" in svg
        assert "SVG Heat" in svg

    def test_to_svg_empty(self):
        hm = HeatmapData(resolution=3)
        svg = hm.to_svg()
        assert "<svg" in svg

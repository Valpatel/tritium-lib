# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.planning.contours — iso-cost marching-squares curves."""

import math

from tritium_lib.planning import iso_cost_contours
from tritium_lib.planning.costmap import Costmap

LETHAL = float("inf")

PROPERTY_KEYS = {"source", "kind", "cost", "level_index"}


def _make_costmap(grid, resolution=10.0, origin=(0.0, 0.0)):
    """Build a Costmap straight from a grid[row][col] list (row 0 = south)."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    return Costmap(
        origin_x=origin[0],
        origin_y=origin[1],
        resolution=resolution,
        width=width,
        height=height,
        grid=[list(row) for row in grid],
    )


def _coords(feature):
    return feature["geometry"]["coordinates"]


# ---------------------------------------------------------------------------
# Orientation pin (row flip is load-bearing)
# ---------------------------------------------------------------------------

class TestOrientationPin:
    def test_row_gradient_contour_sits_on_row2_cell_centers(self):
        # 3 cols x 5 rows, res 10, origin (0,0); cost[row][col] = row so cost
        # increases NORTHWARD.  The level-2.0 contour must sit exactly on the
        # row-2 cell-center latitude y = 25.0.  Any row-flip bug in the
        # Costmap(south-up) -> ElevationGrid(north-up) adapter moves it.
        cm = _make_costmap([[float(row)] * 3 for row in range(5)])
        result = iso_cost_contours(cm, levels=[2.0])

        assert result["type"] == "FeatureCollection"
        assert result["levels"] == [2.0]
        assert len(result["features"]) == 1
        feature = result["features"][0]
        assert feature["geometry"]["type"] == "LineString"
        for x, y in _coords(feature):
            assert y == 25.0  # origin_y + 2.5 * res, exactly
        assert feature["properties"] == {
            "source": "costmap",
            "kind": "iso_cost",
            "cost": 2.0,
            "level_index": 0,
        }


# ---------------------------------------------------------------------------
# X interpolation pin
# ---------------------------------------------------------------------------

class TestXInterpolation:
    def test_level_midway_between_columns_lands_between_cell_centers(self):
        # 2 cols x 3 rows: col 0 costs 0.0, col 1 costs 4.0, res 10.  Cell
        # centers at x = 5.0 and 15.0; level 2.0 is halfway -> x = 10.0.
        cm = _make_costmap([[0.0, 4.0]] * 3)
        result = iso_cost_contours(cm, levels=[2.0])

        assert len(result["features"]) == 1
        for x, y in _coords(result["features"][0]):
            assert x == 10.0


# ---------------------------------------------------------------------------
# Lethal exclusion (LETHAL -> NoData; contours never cross)
# ---------------------------------------------------------------------------

class TestLethalExclusion:
    def test_contour_breaks_around_lethal_block(self):
        # 6x6 column gradient (cost = col) with a lethal block at rows 2-3,
        # cols 2-3.  The 2.5 contour runs vertically at x = 30 but must skip
        # every marching cell touching the block — no output coordinate may
        # fall inside the block's cell-center bounding box.
        grid = [[float(col) for col in range(6)] for _ in range(6)]
        for row in (2, 3):
            for col in (2, 3):
                grid[row][col] = LETHAL
        cm = _make_costmap(grid)
        result = iso_cost_contours(cm, levels=[2.5])

        assert result["features"], "contour should still exist outside the block"
        # Lethal cell centers span x in [25, 35], y in [25, 35].
        for feature in result["features"]:
            for x, y in _coords(feature):
                assert not (25.0 <= x <= 35.0 and 25.0 <= y <= 35.0)

    def test_all_lethal_costmap_is_empty(self):
        cm = _make_costmap([[LETHAL] * 4 for _ in range(4)])
        result = iso_cost_contours(cm)
        assert result == {"type": "FeatureCollection", "features": [], "levels": []}

    def test_all_lethal_costmap_ignores_explicit_levels(self):
        cm = _make_costmap([[LETHAL] * 4 for _ in range(4)])
        result = iso_cost_contours(cm, levels=[1.0, 2.0])
        assert result["features"] == []
        assert result["levels"] == []


# ---------------------------------------------------------------------------
# Auto levels
# ---------------------------------------------------------------------------

class TestAutoLevels:
    def test_auto_levels_strictly_inside_range_with_features(self):
        # 3 cols x 6 rows row gradient: costs 0..5.
        cm = _make_costmap([[float(row)] * 3 for row in range(6)])
        result = iso_cost_contours(cm, levels=None, n=4)

        assert len(result["levels"]) == 4
        for level in result["levels"]:
            assert 0.0 < level < 5.0
        assert result["features"]
        for feature in result["features"]:
            props = feature["properties"]
            assert set(props.keys()) == PROPERTY_KEYS
            assert props["source"] == "costmap"
            assert props["kind"] == "iso_cost"
            assert props["cost"] == round(result["levels"][props["level_index"]], 3)
            assert isinstance(props["level_index"], int)

    def test_uniform_cost_yields_empty(self):
        cm = _make_costmap([[1.0] * 4 for _ in range(4)])
        result = iso_cost_contours(cm, levels=None)
        assert result == {"type": "FeatureCollection", "features": [], "levels": []}


# ---------------------------------------------------------------------------
# Explicit levels
# ---------------------------------------------------------------------------

class TestExplicitLevels:
    def test_explicit_levels_used_as_is_and_cost_rounded(self):
        cm = _make_costmap([[float(row)] * 3 for row in range(6)])
        levels = [1.2345, 3.5]
        result = iso_cost_contours(cm, levels=levels)

        assert result["levels"] == levels
        seen = {f["properties"]["level_index"] for f in result["features"]}
        assert seen == {0, 1}
        for feature in result["features"]:
            props = feature["properties"]
            assert props["cost"] == round(levels[props["level_index"]], 3)

    def test_empty_levels_list_yields_empty(self):
        cm = _make_costmap([[float(row)] * 3 for row in range(6)])
        result = iso_cost_contours(cm, levels=[])
        assert result == {"type": "FeatureCollection", "features": [], "levels": []}


# ---------------------------------------------------------------------------
# Degenerate grids never raise
# ---------------------------------------------------------------------------

class TestDegenerateGrids:
    def test_single_column_yields_empty(self):
        cm = _make_costmap([[float(row)] for row in range(5)])
        result = iso_cost_contours(cm, levels=[2.0])
        assert result == {"type": "FeatureCollection", "features": [], "levels": []}

    def test_single_row_yields_empty(self):
        cm = _make_costmap([[0.0, 1.0, 2.0]])
        result = iso_cost_contours(cm)
        assert result == {"type": "FeatureCollection", "features": [], "levels": []}

    def test_coordinates_are_finite_local_meters(self):
        # Coordinates are [x, y] local meters inside the costmap bounds.
        cm = _make_costmap([[float(row)] * 3 for row in range(6)])
        result = iso_cost_contours(cm, levels=None, n=3)
        min_x, min_y, max_x, max_y = cm.bounds()
        for feature in result["features"]:
            for x, y in _coords(feature):
                assert math.isfinite(x) and math.isfinite(y)
                assert min_x <= x <= max_x
                assert min_y <= y <= max_y

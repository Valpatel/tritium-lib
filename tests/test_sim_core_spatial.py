# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.core.spatial — SpatialGrid."""

import math
import pytest
from types import SimpleNamespace

from tritium_lib.sim_engine.core.spatial import SpatialGrid


def _make_target(tid, x, y, alliance="friendly", status="active"):
    """Create a minimal target-like object for SpatialGrid."""
    return SimpleNamespace(
        target_id=tid,
        position=(x, y),
        alliance=alliance,
        status=status,
    )


class TestSpatialGridConstruction:
    def test_default_cell_size(self):
        grid = SpatialGrid()
        assert grid._cell_size == 50.0

    def test_custom_cell_size(self):
        grid = SpatialGrid(cell_size=100.0)
        assert grid._cell_size == 100.0

    def test_empty_grid_query(self):
        grid = SpatialGrid()
        result = grid.query_radius((0.0, 0.0), 100.0)
        assert result == []


class TestSpatialGridRebuild:
    def test_rebuild_populates_cells(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [
            _make_target("t1", 10.0, 10.0),
            _make_target("t2", 200.0, 200.0),
        ]
        grid.rebuild(targets)
        # Both should be findable
        r1 = grid.query_radius((10.0, 10.0), 5.0)
        r2 = grid.query_radius((200.0, 200.0), 5.0)
        assert len(r1) == 1
        assert r1[0].target_id == "t1"
        assert len(r2) == 1
        assert r2[0].target_id == "t2"

    def test_rebuild_clears_old_data(self):
        grid = SpatialGrid(cell_size=50.0)
        grid.rebuild([_make_target("old", 5.0, 5.0)])
        grid.rebuild([_make_target("new", 100.0, 100.0)])
        assert len(grid.query_radius((5.0, 5.0), 10.0)) == 0
        assert len(grid.query_radius((100.0, 100.0), 10.0)) == 1

    def test_rebuild_many_targets(self):
        grid = SpatialGrid(cell_size=10.0)
        targets = [_make_target(f"t{i}", float(i), float(i)) for i in range(100)]
        grid.rebuild(targets)
        # All should be findable within a large radius
        all_found = grid.query_radius((50.0, 50.0), 200.0)
        assert len(all_found) == 100


class TestSpatialGridQueryRadius:
    def test_finds_nearby_targets(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [
            _make_target("near", 5.0, 5.0),
            _make_target("far", 500.0, 500.0),
        ]
        grid.rebuild(targets)
        result = grid.query_radius((0.0, 0.0), 20.0)
        assert len(result) == 1
        assert result[0].target_id == "near"

    def test_excludes_out_of_range(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [_make_target("t1", 100.0, 0.0)]
        grid.rebuild(targets)
        result = grid.query_radius((0.0, 0.0), 50.0)
        assert len(result) == 0

    def test_exact_boundary_included(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [_make_target("edge", 10.0, 0.0)]
        grid.rebuild(targets)
        result = grid.query_radius((0.0, 0.0), 10.0)
        assert len(result) == 1

    def test_negative_coordinates(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [_make_target("neg", -30.0, -40.0)]
        grid.rebuild(targets)
        result = grid.query_radius((-30.0, -40.0), 5.0)
        assert len(result) == 1

    def test_cross_cell_boundary(self):
        grid = SpatialGrid(cell_size=10.0)
        # Target at (9, 9), query from (11, 11) — different cell, within range
        targets = [_make_target("cross", 9.0, 9.0)]
        grid.rebuild(targets)
        result = grid.query_radius((11.0, 11.0), 5.0)
        assert len(result) == 1

    def test_zero_radius(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [_make_target("t1", 0.0, 0.0)]
        grid.rebuild(targets)
        result = grid.query_radius((0.0, 0.0), 0.0)
        assert len(result) == 1  # Exact match at same point


class TestSpatialGridQueryRect:
    def test_basic_rect_query(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [
            _make_target("in", 25.0, 25.0),
            _make_target("out", 200.0, 200.0),
        ]
        grid.rebuild(targets)
        result = grid.query_rect((0.0, 0.0), (50.0, 50.0))
        assert len(result) == 1
        assert result[0].target_id == "in"

    def test_rect_boundary_inclusive(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [_make_target("edge", 10.0, 10.0)]
        grid.rebuild(targets)
        result = grid.query_rect((10.0, 10.0), (20.0, 20.0))
        assert len(result) == 1

    def test_rect_no_matches(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [_make_target("t", 100.0, 100.0)]
        grid.rebuild(targets)
        result = grid.query_rect((0.0, 0.0), (10.0, 10.0))
        assert len(result) == 0

    def test_rect_with_negative_coords(self):
        grid = SpatialGrid(cell_size=50.0)
        targets = [_make_target("neg", -25.0, -25.0)]
        grid.rebuild(targets)
        result = grid.query_rect((-50.0, -50.0), (0.0, 0.0))
        assert len(result) == 1

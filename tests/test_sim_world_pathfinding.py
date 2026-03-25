# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.world.pathfinding — plan_path routing."""

import math
import pytest
from types import SimpleNamespace

from tritium_lib.sim_engine.world.pathfinding import (
    plan_path,
    _STATIONARY_TYPES,
    _FLYING_TYPES,
    _ROAD_TYPES,
    _PEDESTRIAN_TYPES,
    _HOSTILE_DIRECT_RANGE,
)


class TestPlanPathStationary:
    """Stationary units should return None (no path)."""

    def test_turret_returns_none(self):
        result = plan_path((0, 0), (100, 100), "turret")
        assert result is None

    def test_heavy_turret_returns_none(self):
        result = plan_path((0, 0), (100, 100), "heavy_turret")
        assert result is None

    def test_missile_turret_returns_none(self):
        result = plan_path((0, 0), (100, 100), "missile_turret")
        assert result is None


class TestPlanPathFlying:
    """Flying units go straight line: [start, end]."""

    def test_drone_straight_line(self):
        result = plan_path((0, 0), (100, 200), "drone")
        assert result == [(0, 0), (100, 200)]

    def test_scout_drone_straight_line(self):
        result = plan_path((10, 20), (300, 400), "scout_drone")
        assert result == [(10, 20), (300, 400)]


class TestPlanPathNoGraph:
    """Without street graph or terrain map, all ground units get direct path."""

    def test_rover_direct_fallback(self):
        result = plan_path((0, 0), (100, 100), "rover")
        assert result is not None
        assert len(result) >= 2
        assert result[0] == (0, 0)
        assert result[-1] == (100, 100)

    def test_person_direct_fallback(self):
        result = plan_path((0, 0), (100, 100), "person", alliance="friendly")
        assert result is not None
        assert result[0] == (0, 0)
        assert result[-1] == (100, 100)

    def test_tank_direct_fallback(self):
        result = plan_path((0, 0), (50, 50), "tank")
        assert result is not None
        assert len(result) >= 2

    def test_vehicle_direct_fallback(self):
        result = plan_path((0, 0), (200, 200), "vehicle")
        assert result is not None

    def test_unknown_type_direct_fallback(self):
        result = plan_path((0, 0), (50, 50), "unknown_type_xyz")
        assert result is not None
        assert result[0] == (0, 0)
        assert result[-1] == (50, 50)


class TestPlanPathHostile:
    """Hostile persons with short distance go direct."""

    def test_hostile_person_short_distance_direct(self):
        # Within _HOSTILE_DIRECT_RANGE — goes direct even with a street graph
        end = (20.0, 0.0)  # 20m away < 30m threshold
        result = plan_path((0, 0), end, "person", alliance="hostile")
        assert result is not None
        assert len(result) >= 2

    def test_hostile_person_no_graph_direct(self):
        result = plan_path((0, 0), (500, 500), "person", alliance="hostile")
        assert result is not None
        assert result[0] == (0, 0)
        assert result[-1] == (500, 500)


class TestPlanPathWithStreetGraph:
    """Test plan_path with a mock street graph."""

    def _mock_street_graph(self, path=None):
        """Create a mock street graph that returns a fixed path."""
        sg = SimpleNamespace()
        sg.graph = True  # Non-None signals graph is loaded
        sg.shortest_path = lambda s, e: path if path else [s, e]
        return sg

    def test_rover_uses_street_graph(self):
        sg = self._mock_street_graph(path=[(0, 0), (50, 0), (50, 50), (100, 50)])
        result = plan_path((0, 0), (100, 50), "rover", street_graph=sg)
        assert result is not None
        assert len(result) == 4  # Uses the street graph path

    def test_apc_uses_street_graph(self):
        sg = self._mock_street_graph(path=[(0, 0), (25, 25), (50, 50)])
        result = plan_path((0, 0), (50, 50), "apc", street_graph=sg)
        assert result is not None
        assert len(result) == 3

    def test_none_graph_attribute_fallback(self):
        sg = SimpleNamespace(graph=None)
        result = plan_path((0, 0), (100, 100), "rover", street_graph=sg)
        assert result is not None
        # Falls back to direct since graph is None
        assert result == [(0, 0), (100, 100)]


class TestPlanPathGraphling:
    """Graphlings always use grid A* (never street graph)."""

    def test_graphling_ignores_street_graph(self):
        sg = SimpleNamespace(graph=True)
        sg.shortest_path = lambda s, e: [(0, 0), (50, 0), (50, 50)]
        result = plan_path((0, 0), (50, 50), "graphling", street_graph=sg)
        # Should NOT use the street graph path (graphlings use grid A*)
        # Without a terrain map, falls back to direct
        assert result == [(0, 0), (50, 50)]


class TestPlanPathTypeConstants:
    """Verify type classification constants are populated."""

    def test_stationary_types_nonempty(self):
        assert len(_STATIONARY_TYPES) >= 3
        assert "turret" in _STATIONARY_TYPES

    def test_flying_types_nonempty(self):
        assert len(_FLYING_TYPES) >= 2
        assert "drone" in _FLYING_TYPES

    def test_road_types_nonempty(self):
        assert len(_ROAD_TYPES) >= 3
        assert "rover" in _ROAD_TYPES
        assert "tank" in _ROAD_TYPES

    def test_pedestrian_types_nonempty(self):
        assert len(_PEDESTRIAN_TYPES) >= 3
        assert "person" in _PEDESTRIAN_TYPES
        assert "civilian" in _PEDESTRIAN_TYPES

    def test_hostile_direct_range(self):
        assert _HOSTILE_DIRECT_RANGE == 30.0

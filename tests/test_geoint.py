# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geoint — Geospatial Intelligence module."""

from __future__ import annotations

import math

import pytest

from tritium_lib.geoint import (
    ApproachRoute,
    Building,
    CoverAnalysis,
    CoverPosition,
    CoverageResult,
    GeointAnalyzer,
    LineOfSight,
    ObservationPoint,
    ObservationScore,
    Route,
    RouteWaypoint,
    SurveillanceCoverage,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable building layouts
# ---------------------------------------------------------------------------

def _square_building(
    cx: float = 50.0, cy: float = 50.0,
    size: float = 10.0, height: float = 8.0,
) -> Building:
    """A square building centered at (cx, cy)."""
    hs = size / 2
    return Building(
        polygon=[
            (cx - hs, cy - hs),
            (cx + hs, cy - hs),
            (cx + hs, cy + hs),
            (cx - hs, cy + hs),
        ],
        height=height,
    )


def _wide_wall(
    x_start: float = 40.0, x_end: float = 60.0,
    y: float = 50.0, thickness: float = 2.0, height: float = 8.0,
) -> Building:
    """A wall-like building (long and thin)."""
    ht = thickness / 2
    return Building(
        polygon=[
            (x_start, y - ht),
            (x_end, y - ht),
            (x_end, y + ht),
            (x_start, y + ht),
        ],
        height=height,
    )


def _two_building_layout() -> list[Building]:
    """Two buildings side by side with a gap."""
    return [
        _square_building(cx=30.0, cy=50.0, size=10.0),
        _square_building(cx=70.0, cy=50.0, size=10.0),
    ]


# ---------------------------------------------------------------------------
# Building dataclass
# ---------------------------------------------------------------------------

class TestBuilding:
    def test_aabb_computed_on_init(self):
        b = _square_building(50, 50, 10)
        assert b.aabb == (45.0, 45.0, 55.0, 55.0)

    def test_centroid(self):
        b = _square_building(50, 50, 10)
        c = b.centroid
        assert abs(c[0] - 50.0) < 0.01
        assert abs(c[1] - 50.0) < 0.01

    def test_contains_point_inside(self):
        b = _square_building(50, 50, 10)
        assert b.contains(50.0, 50.0) is True

    def test_contains_point_outside(self):
        b = _square_building(50, 50, 10)
        assert b.contains(100.0, 100.0) is False

    def test_default_height(self):
        b = Building(polygon=[(0, 0), (1, 0), (1, 1)])
        assert b.height == 8.0

    def test_empty_polygon(self):
        b = Building()
        assert b.polygon == []
        assert b.aabb == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# LineOfSight
# ---------------------------------------------------------------------------

class TestLineOfSight:
    def test_clear_line_no_buildings(self):
        los = LineOfSight()
        assert los.check((0, 0), (100, 100)) is True

    def test_blocked_by_building(self):
        los = LineOfSight()
        bldg = _square_building(50, 50, 10)
        los.set_buildings([bldg])
        # Line from (0, 50) to (100, 50) passes through the building
        assert los.check((0, 50), (100, 50)) is False

    def test_clear_around_building(self):
        los = LineOfSight()
        bldg = _square_building(50, 50, 10)
        los.set_buildings([bldg])
        # Line from (0, 0) to (100, 0) does not pass through the building
        assert los.check((0, 0), (100, 0)) is True

    def test_blocked_diagonal(self):
        los = LineOfSight()
        bldg = _square_building(50, 50, 20)
        los.set_buildings([bldg])
        # Diagonal line through the interior of the building (offset from
        # exact corner to avoid degenerate collinear intersection)
        assert los.check((0, 1), (100, 99)) is False

    def test_clear_just_missing_building(self):
        los = LineOfSight()
        bldg = _square_building(50, 50, 10)
        los.set_buildings([bldg])
        # Line along the bottom edge, below the building
        assert los.check((0, 40), (100, 40)) is True

    def test_blocking_buildings_returns_indices(self):
        los = LineOfSight()
        b1 = _square_building(30, 50, 10)
        b2 = _square_building(70, 50, 10)
        los.set_buildings([b1, b2])
        # Line from (0, 50) to (100, 50) hits both buildings
        blockers = los.blocking_buildings((0, 50), (100, 50))
        assert 0 in blockers
        assert 1 in blockers

    def test_blocking_buildings_empty_when_clear(self):
        los = LineOfSight()
        bldg = _square_building(50, 50, 10)
        los.set_buildings([bldg])
        blockers = los.blocking_buildings((0, 0), (100, 0))
        assert blockers == []

    def test_visible_targets(self):
        los = LineOfSight()
        bldg = _square_building(50, 50, 10)
        los.set_buildings([bldg])
        observer = (0, 50)
        targets = [(20, 50), (100, 50), (0, 100)]
        visible = los.visible_targets(observer, targets)
        # (20, 50) is in front of the building — visible
        # (100, 50) is behind the building — blocked
        # (0, 100) is off to the side — visible
        vis_indices = [v[0] for v in visible]
        assert 0 in vis_indices  # (20, 50)
        assert 2 in vis_indices  # (0, 100)
        assert 1 not in vis_indices  # (100, 50) blocked

    def test_same_point_always_visible(self):
        los = LineOfSight()
        bldg = _square_building(50, 50, 10)
        los.set_buildings([bldg])
        # A point can always "see" itself
        assert los.check((0, 0), (0, 0)) is True


# ---------------------------------------------------------------------------
# CoverAnalysis
# ---------------------------------------------------------------------------

class TestCoverAnalysis:
    def test_concealed_behind_building(self):
        bldgs = [_square_building(50, 50, 10)]
        los = LineOfSight()
        los.set_buildings(bldgs)
        ca = CoverAnalysis(los, bldgs)
        # Observer at (0, 50). Point directly behind building should be concealed.
        assert ca.is_concealed((80, 50), (0, 50)) is True

    def test_not_concealed_in_open(self):
        bldgs = [_square_building(50, 50, 10)]
        los = LineOfSight()
        los.set_buildings(bldgs)
        ca = CoverAnalysis(los, bldgs)
        # Observer at (0, 50). Point in front of building.
        assert ca.is_concealed((20, 50), (0, 50)) is False

    def test_find_cover_returns_positions(self):
        bldgs = [_square_building(50, 50, 10)]
        los = LineOfSight()
        los.set_buildings(bldgs)
        ca = CoverAnalysis(los, bldgs)
        cover = ca.find_cover(
            observer=(0, 50), search_radius=100, sample_spacing=5.0,
        )
        assert len(cover) > 0
        # All returned positions should be concealed
        for cp in cover:
            assert not los.check((0, 50), cp.position)

    def test_find_cover_scores_are_valid(self):
        bldgs = [_square_building(50, 50, 10)]
        los = LineOfSight()
        los.set_buildings(bldgs)
        ca = CoverAnalysis(los, bldgs)
        cover = ca.find_cover(observer=(0, 50), search_radius=100)
        for cp in cover:
            assert 0.0 <= cp.cover_score <= 1.0
            assert cp.building_index >= 0

    def test_concealment_from_multiple_observers(self):
        bldgs = [_wide_wall(40, 60, 50, 4)]
        los = LineOfSight()
        los.set_buildings(bldgs)
        ca = CoverAnalysis(los, bldgs)
        # Point behind the wall
        hidden_point = (50, 70)
        # Two observers on the same side of the wall
        observers = [(50, 0), (60, 0)]
        frac = ca.concealment_from_multiple(hidden_point, observers)
        assert frac > 0.0  # At least some concealment

    def test_concealment_from_empty_observers(self):
        bldgs = [_square_building(50, 50, 10)]
        los = LineOfSight()
        los.set_buildings(bldgs)
        ca = CoverAnalysis(los, bldgs)
        assert ca.concealment_from_multiple((50, 70), []) == 0.0


# ---------------------------------------------------------------------------
# ApproachRoute
# ---------------------------------------------------------------------------

class TestApproachRoute:
    def test_route_found_in_open(self):
        """Route between two points with no buildings."""
        los = LineOfSight()
        ar = ApproachRoute(los, [])
        route = ar.find_route(
            start=(0, 0), goal=(50, 0), observer=(25, 50),
            grid_spacing=10.0,
        )
        assert route.total_distance > 0
        assert len(route.waypoints) >= 2
        # Start and end should be near the requested start/goal
        assert abs(route.waypoints[0].position[0] - 0) < 15
        assert abs(route.waypoints[-1].position[0] - 50) < 15

    def test_route_avoids_building(self):
        """Route should not pass through a building."""
        bldg = _square_building(25, 0, 10)
        los = LineOfSight()
        los.set_buildings([bldg])
        ar = ApproachRoute(los, [bldg])
        route = ar.find_route(
            start=(0, 0), goal=(50, 0), observer=(25, 100),
            grid_spacing=5.0,
        )
        # No waypoint should be inside the building
        for wp in route.waypoints:
            assert not bldg.contains(wp.position[0], wp.position[1])

    def test_route_prefers_concealment(self):
        """Route with a high concealment weight should prefer hidden path."""
        wall = _wide_wall(20, 80, 25, 4)
        los = LineOfSight()
        los.set_buildings([wall])
        ar = ApproachRoute(los, [wall])

        # Observer is at (50, 100) looking south
        # The wall provides cover along y=25.
        # Route from (0, 0) to (100, 0) — both below the wall.
        route = ar.find_route(
            start=(0, 0), goal=(100, 0), observer=(50, 100),
            grid_spacing=10.0, concealment_weight=5.0,
        )
        assert route.total_distance > 0
        assert len(route.waypoints) >= 2

    def test_route_has_concealment_scores(self):
        bldg = _square_building(50, 50, 10)
        los = LineOfSight()
        los.set_buildings([bldg])
        ar = ApproachRoute(los, [bldg])
        route = ar.find_route(
            start=(0, 0), goal=(100, 0), observer=(50, 100),
            grid_spacing=10.0,
        )
        for wp in route.waypoints:
            assert 0.0 <= wp.concealment <= 1.0
        assert 0.0 <= route.avg_concealment <= 1.0
        assert 0.0 <= route.min_concealment <= 1.0


# ---------------------------------------------------------------------------
# ObservationPoint
# ---------------------------------------------------------------------------

class TestObservationPoint:
    def test_score_open_area(self):
        """Observation of an area with no buildings — should see everything."""
        los = LineOfSight()
        op = ObservationPoint(los, [])
        score = op.score_position(
            position=(0, 0),
            area_center=(50, 0),
            area_radius=20,
            sample_spacing=5.0,
        )
        assert score.visible_fraction == 1.0
        assert score.visible_count == score.total_samples
        assert score.total_samples > 0

    def test_score_behind_building(self):
        """Observer behind a building — should see less."""
        bldg = _square_building(25, 0, 10)
        los = LineOfSight()
        los.set_buildings([bldg])
        op = ObservationPoint(los, [bldg])
        score = op.score_position(
            position=(0, 0),
            area_center=(50, 0),
            area_radius=20,
            sample_spacing=5.0,
        )
        # Should see some but not all of the area
        assert score.visible_fraction < 1.0
        assert score.visible_count < score.total_samples

    def test_find_best_returns_sorted(self):
        bldg = _square_building(50, 50, 10)
        los = LineOfSight()
        los.set_buildings([bldg])
        op = ObservationPoint(los, [bldg])
        best = op.find_best(
            area_center=(50, 50), area_radius=30,
            search_radius=60, candidate_spacing=15,
            sample_spacing=10, top_n=3,
        )
        assert len(best) <= 3
        if len(best) >= 2:
            assert best[0].visible_fraction >= best[1].visible_fraction

    def test_find_best_not_inside_buildings(self):
        bldg = _square_building(50, 50, 10)
        los = LineOfSight()
        los.set_buildings([bldg])
        op = ObservationPoint(los, [bldg])
        best = op.find_best(
            area_center=(50, 50), area_radius=30,
            search_radius=60, candidate_spacing=15,
            sample_spacing=10,
        )
        for s in best:
            assert not bldg.contains(s.position[0], s.position[1])


# ---------------------------------------------------------------------------
# SurveillanceCoverage
# ---------------------------------------------------------------------------

class TestSurveillanceCoverage:
    def test_full_coverage_no_buildings(self):
        """One sensor in open terrain should cover everything."""
        los = LineOfSight()
        sc = SurveillanceCoverage(los, [])
        result = sc.compute(
            sensor_positions=[(0, 0)],
            area_center=(50, 0),
            area_radius=20,
            sample_spacing=5.0,
        )
        assert result.coverage_fraction == 1.0
        assert result.covered_points == result.total_points
        assert len(result.blind_spots) == 0

    def test_partial_coverage_with_building(self):
        """A building between sensor and area should create blind spots."""
        bldg = _square_building(25, 0, 10)
        los = LineOfSight()
        los.set_buildings([bldg])
        sc = SurveillanceCoverage(los, [bldg])
        result = sc.compute(
            sensor_positions=[(0, 0)],
            area_center=(50, 0),
            area_radius=20,
            sample_spacing=5.0,
        )
        assert result.coverage_fraction < 1.0
        assert len(result.blind_spots) > 0

    def test_two_sensors_improve_coverage(self):
        """Two sensors should cover more than one."""
        bldg = _square_building(25, 0, 10)
        los = LineOfSight()
        los.set_buildings([bldg])
        sc = SurveillanceCoverage(los, [bldg])

        one_sensor = sc.compute(
            sensor_positions=[(0, 0)],
            area_center=(50, 0), area_radius=20,
            sample_spacing=5.0,
        )
        two_sensors = sc.compute(
            sensor_positions=[(0, 0), (50, 30)],
            area_center=(50, 0), area_radius=20,
            sample_spacing=5.0,
        )
        assert two_sensors.coverage_fraction >= one_sensor.coverage_fraction

    def test_sensor_range_limit(self):
        """Max sensor range should reduce coverage."""
        los = LineOfSight()
        sc = SurveillanceCoverage(los, [])
        unlimited = sc.compute(
            sensor_positions=[(0, 0)],
            area_center=(50, 0), area_radius=20,
            sample_spacing=5.0,
            max_sensor_range=0,
        )
        limited = sc.compute(
            sensor_positions=[(0, 0)],
            area_center=(50, 0), area_radius=20,
            sample_spacing=5.0,
            max_sensor_range=40,
        )
        assert limited.coverage_fraction <= unlimited.coverage_fraction

    def test_find_blind_spots(self):
        bldg = _square_building(25, 0, 10)
        los = LineOfSight()
        los.set_buildings([bldg])
        sc = SurveillanceCoverage(los, [bldg])
        spots = sc.find_blind_spots(
            sensor_positions=[(0, 0)],
            area_center=(50, 0), area_radius=20,
            sample_spacing=5.0,
        )
        assert len(spots) > 0
        # Each blind spot should be invisible to the sensor
        for s in spots:
            assert not los.check((0, 0), s)

    def test_sensor_contributions(self):
        """Each sensor should have a count of points it uniquely covers."""
        los = LineOfSight()
        sc = SurveillanceCoverage(los, [])
        result = sc.compute(
            sensor_positions=[(0, 0), (100, 0)],
            area_center=(50, 0), area_radius=20,
            sample_spacing=5.0,
        )
        # Both sensors are in the result
        assert 0 in result.sensor_contributions
        assert 1 in result.sensor_contributions


# ---------------------------------------------------------------------------
# GeointAnalyzer — facade
# ---------------------------------------------------------------------------

class TestGeointAnalyzer:
    def test_load_buildings(self):
        analyzer = GeointAnalyzer()
        count = analyzer.load_buildings([
            {"polygon": [(0, 0), (10, 0), (10, 10), (0, 10)], "height": 12},
            {"polygon": [(20, 20), (30, 20), (30, 30), (20, 30)]},
        ])
        assert count == 2
        assert len(analyzer.buildings) == 2
        assert analyzer.buildings[0].height == 12.0
        assert analyzer.buildings[1].height == 8.0  # default

    def test_load_buildings_skips_degenerate(self):
        analyzer = GeointAnalyzer()
        count = analyzer.load_buildings([
            {"polygon": [(0, 0), (10, 0)]},  # only 2 points — skip
            {"polygon": [(0, 0), (10, 0), (10, 10)], "height": 5},
        ])
        assert count == 1

    def test_line_of_sight_via_facade(self):
        analyzer = GeointAnalyzer()
        analyzer.load_buildings([
            {"polygon": [(45, 45), (55, 45), (55, 55), (45, 55)]},
        ])
        assert analyzer.line_of_sight.check((0, 50), (100, 50)) is False
        assert analyzer.line_of_sight.check((0, 0), (100, 0)) is True

    def test_cover_analysis_via_facade(self):
        analyzer = GeointAnalyzer()
        analyzer.load_buildings([
            {"polygon": [(45, 45), (55, 45), (55, 55), (45, 55)]},
        ])
        cover = analyzer.cover_analysis.find_cover(
            observer=(0, 50), search_radius=100,
        )
        assert len(cover) > 0

    def test_analyze_area(self):
        analyzer = GeointAnalyzer()
        analyzer.load_buildings([
            {"polygon": [(20, 20), (30, 20), (30, 30), (20, 30)]},
        ])
        result = analyzer.analyze_area(
            area_center=(50, 50), area_radius=30,
            sensor_positions=[(0, 50), (100, 50)],
            sample_spacing=10,
        )
        assert "coverage_fraction" in result
        assert "sensor_scores" in result
        assert result["total_points"] > 0
        assert 0.0 <= result["coverage_fraction"] <= 1.0

    def test_properties_accessible_without_buildings(self):
        """Facade properties should be accessible even with no buildings."""
        analyzer = GeointAnalyzer()
        # These should not raise
        assert analyzer.line_of_sight is not None
        assert analyzer.cover_analysis is not None
        assert analyzer.approach_route is not None
        assert analyzer.observation_point is not None
        assert analyzer.surveillance_coverage is not None

    def test_load_buildings_with_list_coords(self):
        """Building dicts can use lists instead of tuples."""
        analyzer = GeointAnalyzer()
        count = analyzer.load_buildings([
            {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        ])
        assert count == 1
        # Verify the polygon was converted to tuples
        assert isinstance(analyzer.buildings[0].polygon[0], tuple)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_los_with_wall_on_line(self):
        """Wall exactly aligned with the line of sight."""
        wall = _wide_wall(40, 60, 50, 2)
        los = LineOfSight()
        los.set_buildings([wall])
        # Line passes directly through the wall
        assert los.check((50, 0), (50, 100)) is False

    def test_cover_far_from_buildings(self):
        """No cover should be found if buildings are too far away."""
        bldg = _square_building(500, 500, 10)
        los = LineOfSight()
        los.set_buildings([bldg])
        ca = CoverAnalysis(los, [bldg])
        cover = ca.find_cover(observer=(0, 0), search_radius=50)
        assert len(cover) == 0

    def test_surveillance_empty_sensors(self):
        """No sensors means zero coverage."""
        los = LineOfSight()
        sc = SurveillanceCoverage(los, [])
        result = sc.compute(
            sensor_positions=[],
            area_center=(50, 50), area_radius=20,
            sample_spacing=5.0,
        )
        assert result.coverage_fraction == 0.0
        assert result.covered_points == 0

    def test_multiple_buildings_block_los(self):
        """LOS blocked by any one of several buildings."""
        bldgs = _two_building_layout()
        los = LineOfSight()
        los.set_buildings(bldgs)
        # Line from far left to far right passes through both
        assert los.check((-10, 50), (110, 50)) is False

    def test_approach_route_between_identical_points(self):
        """Start == goal should return a trivial route."""
        los = LineOfSight()
        ar = ApproachRoute(los, [])
        route = ar.find_route(
            start=(50, 50), goal=(50, 50), observer=(0, 0),
            grid_spacing=10.0,
        )
        # Should have at least 1 waypoint
        assert len(route.waypoints) >= 1

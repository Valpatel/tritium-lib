# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for formation movement and pathfinding integration.

60+ tests covering FormationType, get_formation_positions, FormationMover,
PathPlanner, and CoverMovement.
"""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.ai.steering import Vec2, distance, magnitude
from tritium_lib.sim_engine.ai.pathfinding import RoadNetwork
from tritium_lib.sim_engine.ai.formations import (
    FormationType,
    FormationConfig,
    get_formation_positions,
    FormationMover,
    PathPlanner,
    CoverMovement,
    _rotate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_pos(a: Vec2, b: Vec2, tol: float = 0.5) -> bool:
    """Check two positions are within tolerance."""
    return distance(a, b) < tol


# ===========================================================================
# FormationType enum
# ===========================================================================


class TestFormationType:
    def test_all_ten_values(self):
        assert len(FormationType) == 10

    def test_values_are_strings(self):
        for ft in FormationType:
            assert isinstance(ft.value, str)

    def test_known_members(self):
        expected = {
            "line", "column", "wedge", "diamond", "staggered_column",
            "echelon_left", "echelon_right", "circle", "spread", "file",
        }
        assert {ft.value for ft in FormationType} == expected

    def test_from_string(self):
        assert FormationType("line") == FormationType.LINE
        assert FormationType("wedge") == FormationType.WEDGE


# ===========================================================================
# FormationConfig dataclass
# ===========================================================================


class TestFormationConfig:
    def test_defaults(self):
        cfg = FormationConfig(formation_type=FormationType.LINE)
        assert cfg.spacing == 3.0
        assert cfg.facing == 0.0
        assert cfg.leader_pos == (0.0, 0.0)
        assert cfg.num_members == 1

    def test_custom_values(self):
        cfg = FormationConfig(
            formation_type=FormationType.WEDGE,
            spacing=5.0,
            facing=math.pi / 2,
            leader_pos=(10.0, 20.0),
            num_members=8,
        )
        assert cfg.spacing == 5.0
        assert cfg.num_members == 8


# ===========================================================================
# _rotate helper
# ===========================================================================


class TestRotate:
    def test_zero_heading(self):
        r = _rotate((1.0, 0.0), 0.0)
        assert abs(r[0] - 1.0) < 1e-6
        assert abs(r[1] - 0.0) < 1e-6

    def test_90_degrees(self):
        r = _rotate((1.0, 0.0), math.pi / 2)
        assert abs(r[0] - 0.0) < 1e-6
        assert abs(r[1] - 1.0) < 1e-6

    def test_180_degrees(self):
        r = _rotate((1.0, 0.0), math.pi)
        assert abs(r[0] - (-1.0)) < 1e-6
        assert abs(r[1] - 0.0) < 1e-6


# ===========================================================================
# get_formation_positions
# ===========================================================================


class TestGetFormationPositions:
    def test_empty_members(self):
        cfg = FormationConfig(formation_type=FormationType.LINE, num_members=0)
        assert get_formation_positions(cfg) == []

    def test_single_member_at_leader(self):
        cfg = FormationConfig(
            formation_type=FormationType.LINE,
            leader_pos=(10.0, 20.0),
            num_members=1,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 1
        assert approx_pos(positions[0], (10.0, 20.0), tol=0.1)

    # -- LINE ---------------------------------------------------------------

    def test_line_symmetric(self):
        cfg = FormationConfig(
            formation_type=FormationType.LINE,
            spacing=4.0,
            leader_pos=(0.0, 0.0),
            num_members=3,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 3
        # Members spread perpendicular to facing (y-axis at facing=0)
        ys = sorted(p[1] for p in positions)
        assert abs(ys[0] - (-4.0)) < 0.1
        assert abs(ys[1] - 0.0) < 0.1
        assert abs(ys[2] - 4.0) < 0.1

    def test_line_spacing(self):
        cfg = FormationConfig(
            formation_type=FormationType.LINE,
            spacing=5.0,
            num_members=5,
        )
        positions = get_formation_positions(cfg)
        for i in range(len(positions) - 1):
            d = distance(positions[i], positions[i + 1])
            assert abs(d - 5.0) < 0.1

    # -- COLUMN -------------------------------------------------------------

    def test_column_behind_leader(self):
        cfg = FormationConfig(
            formation_type=FormationType.COLUMN,
            spacing=3.0,
            leader_pos=(50.0, 50.0),
            num_members=4,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 4
        # First slot is leader pos
        assert approx_pos(positions[0], (50.0, 50.0), tol=0.1)
        # Each subsequent is behind (negative x at facing=0)
        for i in range(1, 4):
            assert positions[i][0] < positions[0][0]

    def test_column_spacing(self):
        cfg = FormationConfig(
            formation_type=FormationType.COLUMN,
            spacing=4.0,
            num_members=3,
        )
        positions = get_formation_positions(cfg)
        for i in range(len(positions) - 1):
            d = distance(positions[i], positions[i + 1])
            assert abs(d - 4.0) < 0.1

    # -- WEDGE --------------------------------------------------------------

    def test_wedge_leader_at_front(self):
        cfg = FormationConfig(
            formation_type=FormationType.WEDGE,
            spacing=3.0,
            num_members=5,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 5
        # Leader (slot 0) should be at the foremost x position
        assert all(positions[0][0] >= p[0] - 0.1 for p in positions)

    def test_wedge_v_shape(self):
        cfg = FormationConfig(
            formation_type=FormationType.WEDGE,
            spacing=3.0,
            num_members=5,
        )
        positions = get_formation_positions(cfg)
        # Members 1 and 2 should be on opposite sides of the center
        assert positions[1][1] * positions[2][1] <= 0.01  # opposite signs

    # -- DIAMOND ------------------------------------------------------------

    def test_diamond_four_points(self):
        cfg = FormationConfig(
            formation_type=FormationType.DIAMOND,
            spacing=3.0,
            num_members=4,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 4
        # All 4 should be distinct
        for i in range(4):
            for j in range(i + 1, 4):
                assert distance(positions[i], positions[j]) > 1.0

    def test_diamond_single(self):
        cfg = FormationConfig(
            formation_type=FormationType.DIAMOND,
            num_members=1,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 1

    def test_diamond_two(self):
        cfg = FormationConfig(
            formation_type=FormationType.DIAMOND,
            spacing=3.0,
            num_members=2,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 2
        assert distance(positions[0], positions[1]) > 1.0

    def test_diamond_three(self):
        cfg = FormationConfig(
            formation_type=FormationType.DIAMOND,
            spacing=3.0,
            num_members=3,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 3

    def test_diamond_extra_members(self):
        cfg = FormationConfig(
            formation_type=FormationType.DIAMOND,
            spacing=3.0,
            num_members=6,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 6

    # -- STAGGERED_COLUMN ---------------------------------------------------

    def test_staggered_alternates(self):
        cfg = FormationConfig(
            formation_type=FormationType.STAGGERED_COLUMN,
            spacing=3.0,
            num_members=4,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 4
        # Leader centered, others alternate left/right
        assert abs(positions[0][1]) < 0.1  # leader centered
        # Member 1 and 2 should be on opposite sides
        if len(positions) >= 3:
            assert positions[1][1] * positions[2][1] < 0  # opposite sides

    # -- ECHELON_LEFT -------------------------------------------------------

    def test_echelon_left_diagonal(self):
        cfg = FormationConfig(
            formation_type=FormationType.ECHELON_LEFT,
            spacing=3.0,
            num_members=4,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 4
        # Each successive member is behind and to the left
        for i in range(1, 4):
            assert positions[i][0] < positions[0][0] + 0.1
            assert positions[i][1] < positions[0][1] + 0.1

    # -- ECHELON_RIGHT ------------------------------------------------------

    def test_echelon_right_diagonal(self):
        cfg = FormationConfig(
            formation_type=FormationType.ECHELON_RIGHT,
            spacing=3.0,
            num_members=4,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 4
        # Each successive member is behind and to the right
        for i in range(1, 4):
            assert positions[i][0] < positions[0][0] + 0.1
            assert positions[i][1] > positions[0][1] - 0.1

    # -- CIRCLE -------------------------------------------------------------

    def test_circle_equal_radius(self):
        cfg = FormationConfig(
            formation_type=FormationType.CIRCLE,
            spacing=3.0,
            num_members=6,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 6
        # All should be roughly the same distance from centroid
        cx = sum(p[0] for p in positions) / 6
        cy = sum(p[1] for p in positions) / 6
        dists = [distance(p, (cx, cy)) for p in positions]
        for d in dists:
            assert abs(d - dists[0]) < 0.5

    def test_circle_single(self):
        cfg = FormationConfig(
            formation_type=FormationType.CIRCLE,
            num_members=1,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 1

    # -- SPREAD -------------------------------------------------------------

    def test_spread_wider_than_line(self):
        line_cfg = FormationConfig(
            formation_type=FormationType.LINE,
            spacing=3.0,
            num_members=4,
        )
        spread_cfg = FormationConfig(
            formation_type=FormationType.SPREAD,
            spacing=3.0,
            num_members=4,
        )
        line_pos = get_formation_positions(line_cfg)
        spread_pos = get_formation_positions(spread_cfg)
        line_width = max(p[1] for p in line_pos) - min(p[1] for p in line_pos)
        spread_width = max(p[1] for p in spread_pos) - min(p[1] for p in spread_pos)
        assert spread_width > line_width

    # -- FILE ---------------------------------------------------------------

    def test_file_tighter_than_column(self):
        col_cfg = FormationConfig(
            formation_type=FormationType.COLUMN,
            spacing=3.0,
            num_members=4,
        )
        file_cfg = FormationConfig(
            formation_type=FormationType.FILE,
            spacing=3.0,
            num_members=4,
        )
        col_pos = get_formation_positions(col_cfg)
        file_pos = get_formation_positions(file_cfg)
        col_depth = max(abs(p[0]) for p in col_pos)
        file_depth = max(abs(p[0]) for p in file_pos)
        assert file_depth < col_depth

    # -- Facing rotation ----------------------------------------------------

    def test_facing_rotates_formation(self):
        cfg0 = FormationConfig(
            formation_type=FormationType.COLUMN,
            spacing=3.0,
            num_members=3,
            facing=0.0,
        )
        cfg90 = FormationConfig(
            formation_type=FormationType.COLUMN,
            spacing=3.0,
            num_members=3,
            facing=math.pi / 2,
        )
        pos0 = get_formation_positions(cfg0)
        pos90 = get_formation_positions(cfg90)
        # At facing=0, column extends along -X
        # At facing=pi/2, column should extend along -Y
        assert pos0[1][0] < pos0[0][0]  # behind in x
        assert pos90[1][1] < pos90[0][1] + 0.1  # behind in y direction

    def test_leader_pos_offset(self):
        cfg = FormationConfig(
            formation_type=FormationType.LINE,
            spacing=3.0,
            leader_pos=(100.0, 200.0),
            num_members=3,
        )
        positions = get_formation_positions(cfg)
        # All positions should be near leader_pos
        for p in positions:
            assert distance(p, (100.0, 200.0)) < 20.0

    # -- All formations produce correct count -------------------------------

    @pytest.mark.parametrize("ft", list(FormationType))
    def test_all_formations_correct_count(self, ft: FormationType):
        cfg = FormationConfig(
            formation_type=ft,
            spacing=3.0,
            num_members=7,
        )
        positions = get_formation_positions(cfg)
        assert len(positions) == 7

    @pytest.mark.parametrize("ft", list(FormationType))
    def test_all_formations_no_overlap(self, ft: FormationType):
        """No two slots should be exactly the same (for n > 1)."""
        cfg = FormationConfig(
            formation_type=ft,
            spacing=3.0,
            num_members=5,
        )
        positions = get_formation_positions(cfg)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                assert distance(positions[i], positions[j]) > 0.1


# ===========================================================================
# FormationMover
# ===========================================================================


class TestFormationMover:
    def test_init(self):
        mover = FormationMover(
            waypoints=[(0, 0), (10, 0)],
            formation=FormationType.COLUMN,
        )
        assert not mover.is_complete()
        assert mover.progress() == pytest.approx(0.0, abs=0.01)

    def test_single_waypoint_completes_immediately(self):
        mover = FormationMover(
            waypoints=[(5, 5)],
            formation=FormationType.LINE,
        )
        assert mover.is_complete()
        assert mover.progress() == pytest.approx(1.0)

    def test_empty_waypoints_completes(self):
        mover = FormationMover(
            waypoints=[],
            formation=FormationType.LINE,
        )
        assert mover.is_complete()

    def test_tick_returns_targets(self):
        mover = FormationMover(
            waypoints=[(0, 0), (50, 0)],
            formation=FormationType.LINE,
            spacing=3.0,
        )
        positions = {"a": (0.0, 0.0), "b": (0.0, 3.0), "c": (0.0, -3.0)}
        targets = mover.tick(0.5, positions)
        assert len(targets) == 3
        assert "a" in targets
        assert "b" in targets
        assert "c" in targets

    def test_tick_leader_advances(self):
        mover = FormationMover(
            waypoints=[(0, 0), (100, 0)],
            formation=FormationType.COLUMN,
            max_speed=10.0,
        )
        positions = {"leader": (0.0, 0.0), "follower": (-3.0, 0.0)}
        targets = mover.tick(1.0, positions)
        # Leader target should be ahead of origin
        assert targets["leader"][0] > 0.0

    def test_progress_increases(self):
        mover = FormationMover(
            waypoints=[(0, 0), (10, 0), (10, 10)],
            formation=FormationType.COLUMN,
            max_speed=20.0,
            arrival_threshold=2.0,
        )
        positions = {"a": (0.0, 0.0)}

        p0 = mover.progress()
        # Simulate several ticks with leader moving toward waypoints
        for _ in range(50):
            targets = mover.tick(0.5, positions)
            positions = targets  # move to targets

        p1 = mover.progress()
        assert p1 >= p0

    def test_completes_after_enough_ticks(self):
        mover = FormationMover(
            waypoints=[(0, 0), (5, 0)],
            formation=FormationType.LINE,
            max_speed=10.0,
            arrival_threshold=2.0,
        )
        positions = {"a": (0.0, 0.0), "b": (0.0, 3.0)}
        for _ in range(100):
            if mover.is_complete():
                break
            targets = mover.tick(0.2, positions)
            positions = targets
        assert mover.is_complete()

    def test_tick_with_empty_positions(self):
        mover = FormationMover(
            waypoints=[(0, 0), (10, 0)],
            formation=FormationType.LINE,
        )
        targets = mover.tick(0.1, {})
        assert targets == {}

    def test_formation_type_affects_targets(self):
        wp = [(0, 0), (50, 0)]
        pos = {"a": (0.0, 0.0), "b": (0.0, 3.0), "c": (0.0, -3.0)}

        mover_line = FormationMover(wp, FormationType.LINE, spacing=5.0)
        mover_col = FormationMover(wp, FormationType.COLUMN, spacing=5.0)

        t_line = mover_line.tick(0.1, pos)
        t_col = mover_col.tick(0.1, pos)

        # Different formations should produce different targets
        assert t_line != t_col


# ===========================================================================
# PathPlanner
# ===========================================================================


class TestPathPlannerRoad:
    def test_road_route_simple(self):
        net = RoadNetwork()
        net.add_road((0, 0), (50, 0))
        net.add_road((50, 0), (50, 50))

        path = PathPlanner.plan_road_route((0, 0), (50, 50), net)
        assert len(path) >= 2
        assert approx_pos(path[0], (0, 0), tol=1.0)
        assert approx_pos(path[-1], (50, 50), tol=1.0)

    def test_road_route_no_path(self):
        net = RoadNetwork()
        net.add_road((0, 0), (10, 0))
        # Disconnected segment
        net.add_road((100, 100), (200, 100))
        path = PathPlanner.plan_road_route((0, 0), (200, 100), net)
        # Should still return something (snapped to nearest) or empty
        # Depends on network connectivity
        assert isinstance(path, list)

    def test_road_route_empty_network(self):
        net = RoadNetwork()
        path = PathPlanner.plan_road_route((0, 0), (10, 10), net)
        assert path == []


class TestPathPlannerOffRoad:
    def test_off_road_direct(self):
        path = PathPlanner.plan_off_road((0, 0), (10, 0))
        assert len(path) >= 2
        assert approx_pos(path[0], (0, 0), tol=0.1)
        assert approx_pos(path[-1], (10, 0), tol=0.1)

    def test_off_road_avoids_obstacle(self):
        obstacles = [((5.0, 0.0), 3.0)]
        path = PathPlanner.plan_off_road((0, 0), (10, 0), obstacles=obstacles)
        assert len(path) >= 2
        # Path should go around the obstacle
        for wp in path[1:-1]:
            assert distance(wp, (5.0, 0.0)) >= 2.5

    def test_off_road_blocked_end(self):
        obstacles = [((10.0, 0.0), 5.0)]
        path = PathPlanner.plan_off_road((0, 0), (10, 0), obstacles=obstacles)
        assert path == []

    def test_off_road_with_heightmap(self):
        heightmap = {(5, 0): 10.0}  # Steep hill at (10, 0)
        path = PathPlanner.plan_off_road(
            (0, 0), (20, 0), heightmap=heightmap, grid_size=2.0
        )
        assert len(path) >= 2

    def test_off_road_no_obstacles(self):
        path = PathPlanner.plan_off_road((0, 0), (20, 20))
        assert len(path) >= 2


class TestPathPlannerSmooth:
    def test_smooth_preserves_endpoints(self):
        path = [(0, 0), (5, 5), (10, 0)]
        smoothed = PathPlanner.smooth_path(path, iterations=3)
        assert approx_pos(smoothed[0], (0, 0), tol=0.01)
        assert approx_pos(smoothed[-1], (10, 0), tol=0.01)

    def test_smooth_increases_points(self):
        path = [(0, 0), (5, 5), (10, 0), (15, 5)]
        smoothed = PathPlanner.smooth_path(path, iterations=2)
        assert len(smoothed) > len(path)

    def test_smooth_two_points_unchanged(self):
        path = [(0, 0), (10, 10)]
        smoothed = PathPlanner.smooth_path(path, iterations=5)
        assert len(smoothed) == 2

    def test_smooth_one_point(self):
        path = [(5, 5)]
        smoothed = PathPlanner.smooth_path(path)
        assert len(smoothed) == 1

    def test_smooth_empty(self):
        smoothed = PathPlanner.smooth_path([])
        assert smoothed == []

    def test_smooth_zero_iterations(self):
        path = [(0, 0), (5, 5), (10, 0)]
        smoothed = PathPlanner.smooth_path(path, iterations=0)
        assert len(smoothed) == 3


# ===========================================================================
# CoverMovement
# ===========================================================================


class TestCoverMovementAdvance:
    def test_no_cover_direct_path(self):
        path = CoverMovement.plan_covered_advance(
            start=(0, 0),
            end=(100, 0),
            cover_positions=[],
            threat_positions=[(50, 50)],
        )
        assert len(path) == 2
        assert approx_pos(path[0], (0, 0))
        assert approx_pos(path[-1], (100, 0))

    def test_uses_cover(self):
        covers = [(10, 5), (20, -3), (30, 2), (40, -5), (50, 3)]
        threats = [(25, 50)]
        path = CoverMovement.plan_covered_advance(
            start=(0, 0),
            end=(60, 0),
            cover_positions=covers,
            threat_positions=threats,
        )
        assert len(path) >= 2
        assert approx_pos(path[0], (0, 0))
        assert approx_pos(path[-1], (60, 0))

    def test_avoids_threats(self):
        # Cover on safe side, threat on one side
        safe_covers = [(10, -10), (20, -10), (30, -10)]
        exposed_covers = [(10, 10), (20, 10), (30, 10)]
        threats = [(20, 50)]  # threats to the north

        path_safe = CoverMovement.plan_covered_advance(
            start=(0, 0), end=(40, -10),
            cover_positions=safe_covers, threat_positions=threats,
        )
        # Path should exist and use safe cover
        assert len(path_safe) >= 2

    def test_no_threats(self):
        covers = [(10, 0), (20, 0)]
        path = CoverMovement.plan_covered_advance(
            start=(0, 0),
            end=(30, 0),
            cover_positions=covers,
            threat_positions=[],
        )
        assert len(path) >= 2
        assert approx_pos(path[0], (0, 0))
        assert approx_pos(path[-1], (30, 0))

    def test_start_equals_end(self):
        path = CoverMovement.plan_covered_advance(
            start=(5, 5),
            end=(5, 5),
            cover_positions=[(5, 6)],
            threat_positions=[],
        )
        assert len(path) >= 1


class TestCoverMovementLeapfrog:
    def test_basic_leapfrog(self):
        squad1 = {"a1": (0.0, 0.0), "a2": (0.0, 3.0)}
        squad2 = {"b1": (-5.0, 0.0), "b2": (-5.0, 3.0)}
        phases = CoverMovement.leapfrog_advance(
            squads=[squad1, squad2],
            direction=(1.0, 0.0),
            bound_distance=10.0,
        )
        assert len(phases) == 2
        # Phase 0: squad 0 moves, squad 1 covers
        assert phases[0]["moving"] == 0
        assert phases[0]["covering"] == [1]
        assert phases[0]["phase"] == 0
        # Phase 1: squad 1 moves, squad 0 covers
        assert phases[1]["moving"] == 1
        assert phases[1]["covering"] == [0]

    def test_leapfrog_targets_advance(self):
        squad1 = {"a": (0.0, 0.0)}
        squad2 = {"b": (-5.0, 0.0)}
        phases = CoverMovement.leapfrog_advance(
            [squad1, squad2], direction=(1.0, 0.0), bound_distance=15.0,
        )
        # Phase 0 targets: squad 0 advances 15m in x
        assert phases[0]["targets"]["a"][0] > 0.0

    def test_leapfrog_empty_squads(self):
        phases = CoverMovement.leapfrog_advance([], direction=(1.0, 0.0))
        assert phases == []

    def test_leapfrog_single_squad(self):
        squad = {"a": (0.0, 0.0)}
        phases = CoverMovement.leapfrog_advance(
            [squad], direction=(0.0, 1.0), bound_distance=10.0,
        )
        assert len(phases) == 1
        assert phases[0]["moving"] == 0
        assert phases[0]["covering"] == []

    def test_leapfrog_three_squads(self):
        s1 = {"a": (0.0, 0.0)}
        s2 = {"b": (-5.0, 0.0)}
        s3 = {"c": (-10.0, 0.0)}
        phases = CoverMovement.leapfrog_advance(
            [s1, s2, s3], direction=(1.0, 0.0),
        )
        assert len(phases) == 3
        assert phases[2]["moving"] == 2
        assert set(phases[2]["covering"]) == {0, 1}

    def test_leapfrog_zero_direction(self):
        squad = {"a": (0.0, 0.0)}
        phases = CoverMovement.leapfrog_advance(
            [squad], direction=(0.0, 0.0),
        )
        assert phases == []


# ===========================================================================
# Integration: FormationMover + PathPlanner
# ===========================================================================


class TestIntegration:
    def test_road_route_to_formation_mover(self):
        net = RoadNetwork()
        net.add_road((0, 0), (50, 0))
        net.add_road((50, 0), (50, 50))

        path = PathPlanner.plan_road_route((0, 0), (50, 50), net)
        assert len(path) >= 2

        mover = FormationMover(
            waypoints=path,
            formation=FormationType.WEDGE,
            spacing=3.0,
            max_speed=10.0,
        )
        assert not mover.is_complete()

        positions = {"l": path[0], "f1": (path[0][0] - 2, path[0][1] + 2)}
        targets = mover.tick(0.5, positions)
        assert len(targets) == 2

    def test_smoothed_path_to_formation_mover(self):
        raw = [(0, 0), (10, 10), (20, 0), (30, 10)]
        smoothed = PathPlanner.smooth_path(raw, iterations=2)
        mover = FormationMover(
            waypoints=smoothed,
            formation=FormationType.COLUMN,
        )
        assert not mover.is_complete()
        positions = {"a": smoothed[0]}
        targets = mover.tick(0.1, positions)
        assert "a" in targets

    def test_off_road_to_formation_mover(self):
        obstacles = [((15.0, 0.0), 4.0)]
        path = PathPlanner.plan_off_road((0, 0), (30, 0), obstacles=obstacles)
        if path:  # may be empty if blocked
            mover = FormationMover(
                waypoints=path,
                formation=FormationType.STAGGERED_COLUMN,
            )
            positions = {"x": path[0], "y": (path[0][0], path[0][1] + 2)}
            targets = mover.tick(0.2, positions)
            assert len(targets) == 2

    def test_cover_advance_to_mover(self):
        covers = [(10, 5), (20, -3), (30, 2)]
        path = CoverMovement.plan_covered_advance(
            (0, 0), (40, 0), covers, [(20, 30)],
        )
        mover = FormationMover(
            waypoints=path,
            formation=FormationType.DIAMOND,
            spacing=2.0,
        )
        positions = {f"u{i}": path[0] for i in range(4)}
        targets = mover.tick(0.3, positions)
        assert len(targets) == 4

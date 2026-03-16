# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for combat AI behaviors."""

from __future__ import annotations

import math

import pytest

from tritium_lib.sim_engine.ai.combat_ai import (
    # Cover
    find_cover,
    is_in_cover,
    rate_cover_position,
    # Flanking
    compute_flank_position,
    is_flanking,
    # Engagement
    optimal_engagement_range,
    should_engage,
    should_retreat,
    # Squad
    formation_positions,
    assign_targets,
    # Suppression
    suppression_cone,
    is_suppressed,
    # Behavior trees
    make_assault_tree,
    make_defender_tree,
    make_sniper_tree,
    make_squad_leader_tree,
)
from tritium_lib.sim_engine.ai.behavior_tree import Status
from tritium_lib.sim_engine.ai.steering import distance


# ---------------------------------------------------------------------------
# Cover system
# ---------------------------------------------------------------------------

class TestFindCover:
    def test_finds_position_behind_obstacle(self):
        pos = (0.0, 0.0)
        threat = (50.0, 0.0)
        obstacles = [((25.0, 0.0), 3.0)]
        cover = find_cover(pos, threat, obstacles)
        assert cover is not None
        # Cover should be on the far side of obstacle from threat
        # i.e., closer to pos than to threat
        assert distance(cover, pos) < distance(cover, threat)

    def test_returns_none_if_no_obstacles(self):
        assert find_cover((0, 0), (50, 0), []) is None

    def test_returns_none_if_out_of_range(self):
        pos = (0.0, 0.0)
        threat = (50.0, 0.0)
        obstacles = [((200.0, 200.0), 3.0)]
        assert find_cover(pos, threat, obstacles, max_range=10.0) is None

    def test_picks_nearest_cover(self):
        pos = (0.0, 0.0)
        threat = (50.0, 0.0)
        obstacles = [
            ((10.0, 0.0), 3.0),  # closer
            ((40.0, 0.0), 3.0),  # farther from pos
        ]
        cover = find_cover(pos, threat, obstacles)
        assert cover is not None
        # Should pick the closer obstacle's cover
        assert distance(cover, pos) < 20.0


class TestIsInCover:
    def test_behind_obstacle_is_cover(self):
        # Position behind obstacle from threat's perspective
        pos = (5.0, 0.0)
        threat = (50.0, 0.0)
        obstacles = [((25.0, 0.0), 5.0)]
        assert is_in_cover(pos, threat, obstacles)

    def test_exposed_position_not_cover(self):
        pos = (0.0, 20.0)
        threat = (50.0, 0.0)
        obstacles = [((25.0, 0.0), 2.0)]
        assert not is_in_cover(pos, threat, obstacles)

    def test_no_obstacles_no_cover(self):
        assert not is_in_cover((0, 0), (50, 0), [])


class TestRateCoverPosition:
    def test_no_obstacles_zero_score(self):
        assert rate_cover_position((0, 0), (50, 0), []) == 0.0

    def test_covered_position_positive_score(self):
        pos = (5.0, 0.0)
        threat = (50.0, 0.0)
        obstacles = [((25.0, 0.0), 5.0)]
        score = rate_cover_position(pos, threat, obstacles)
        assert score > 0.0
        assert score <= 1.0

    def test_exposed_position_zero(self):
        pos = (0.0, 30.0)
        threat = (50.0, 0.0)
        obstacles = [((25.0, 0.0), 2.0)]
        score = rate_cover_position(pos, threat, obstacles)
        assert score == 0.0

    def test_multiple_obstacles_better(self):
        pos = (5.0, 0.0)
        threat = (50.0, 0.0)
        single = [((25.0, 0.0), 5.0)]
        double = [((25.0, 0.0), 5.0), ((15.0, 0.0), 5.0)]
        score_single = rate_cover_position(pos, threat, single)
        score_double = rate_cover_position(pos, threat, double)
        assert score_double >= score_single


# ---------------------------------------------------------------------------
# Flanking
# ---------------------------------------------------------------------------

class TestComputeFlankPosition:
    def test_returns_position_away_from_front(self):
        target = (50.0, 50.0)
        facing = 0.0  # facing +x
        attacker = (0.0, 50.0)
        flank = compute_flank_position(target, facing, attacker, 20.0)
        assert flank is not None
        assert distance(flank, target) == pytest.approx(20.0, abs=1.0)

    def test_flank_not_in_front(self):
        target = (50.0, 50.0)
        facing = 0.0
        attacker = (0.0, 50.0)
        flank = compute_flank_position(target, facing, attacker, 20.0)
        # Flank position should be to the side or rear
        assert is_flanking(flank, target, facing, angle_threshold=45.0)


class TestIsFlanking:
    def test_directly_behind(self):
        attacker = (40.0, 0.0)
        target = (50.0, 0.0)
        facing = 0.0  # facing +x, attacker is behind
        assert is_flanking(attacker, target, facing, 90.0)

    def test_directly_in_front(self):
        attacker = (60.0, 0.0)
        target = (50.0, 0.0)
        facing = 0.0  # facing +x, attacker is ahead
        assert not is_flanking(attacker, target, facing, 90.0)

    def test_perpendicular(self):
        attacker = (50.0, 20.0)
        target = (50.0, 0.0)
        facing = 0.0
        # At 90 degrees, exactly at threshold
        assert is_flanking(attacker, target, facing, 89.0)


# ---------------------------------------------------------------------------
# Engagement decisions
# ---------------------------------------------------------------------------

class TestOptimalEngagementRange:
    def test_default_falloff(self):
        assert optimal_engagement_range(100.0) == pytest.approx(70.0)

    def test_custom_falloff(self):
        assert optimal_engagement_range(50.0, 0.5) == pytest.approx(25.0)


class TestShouldEngage:
    def test_healthy_with_ammo_and_cover(self):
        assert should_engage(30.0, 0.9, 0.8, True, 2)

    def test_no_ammo_never_engage(self):
        assert not should_engage(30.0, 1.0, 0.05, True, 5)

    def test_low_health_no_cover_no_allies(self):
        assert not should_engage(30.0, 0.3, 0.5, False, 0)

    def test_full_resources_engage(self):
        assert should_engage(20.0, 1.0, 1.0, True, 3)


class TestShouldRetreat:
    def test_critical_health(self):
        assert should_retreat(0.1, 1.0, 1, 3)

    def test_no_ammo(self):
        assert should_retreat(0.8, 0.02, 1, 3)

    def test_healthy_with_ammo(self):
        assert not should_retreat(0.9, 0.9, 2, 2)

    def test_outnumbered_and_hurt(self):
        assert should_retreat(0.4, 0.5, 8, 1)

    def test_outnumbered_low_ammo(self):
        assert should_retreat(0.8, 0.15, 5, 1)


# ---------------------------------------------------------------------------
# Squad coordination
# ---------------------------------------------------------------------------

class TestFormationPositions:
    def test_wedge_correct_count(self):
        positions = formation_positions((0, 0), 0.0, 4, "wedge", 5.0)
        assert len(positions) == 4

    def test_line_formation(self):
        positions = formation_positions((0, 0), 0.0, 3, "line", 5.0)
        assert len(positions) == 3
        # Line members should be at same forward offset as leader
        for p in positions:
            assert abs(p[0]) < 0.1  # roughly same x

    def test_column_formation(self):
        positions = formation_positions((0, 0), 0.0, 3, "column", 5.0)
        assert len(positions) == 3
        # Column members should be behind leader
        for p in positions:
            assert p[0] < -0.1

    def test_diamond_formation(self):
        positions = formation_positions((50, 50), 0.0, 4, "diamond", 5.0)
        assert len(positions) == 4

    def test_echelon_formation(self):
        positions = formation_positions((0, 0), 0.0, 3, "echelon", 5.0)
        assert len(positions) == 3

    def test_all_formations_valid(self):
        for form in ["wedge", "line", "column", "diamond", "echelon"]:
            positions = formation_positions((0, 0), 0.0, 4, form, 5.0)
            assert len(positions) == 4, f"Formation {form} returned wrong count"

    def test_extra_members_beyond_offsets(self):
        # More members than predefined offsets
        positions = formation_positions((0, 0), 0.0, 12, "wedge", 5.0)
        assert len(positions) == 12

    def test_heading_rotates_positions(self):
        pos_0 = formation_positions((0, 0), 0.0, 2, "wedge", 5.0)
        pos_90 = formation_positions((0, 0), math.pi / 2, 2, "wedge", 5.0)
        # Positions should be different when heading changes
        assert pos_0[0] != pytest.approx(pos_90[0], abs=0.1)

    def test_unknown_formation_defaults_wedge(self):
        positions = formation_positions((0, 0), 0.0, 3, "unknown", 5.0)
        wedge = formation_positions((0, 0), 0.0, 3, "wedge", 5.0)
        for a, b in zip(positions, wedge):
            assert a == pytest.approx(b, abs=1e-6)


class TestAssignTargets:
    def test_one_to_one(self):
        squad = [(0, 0), (20, 0)]
        enemies = [(3, 0), (18, 0)]
        assignments = assign_targets(squad, enemies)
        assert len(assignments) == 2
        assert assignments[0] == 0  # closer to enemy 0
        assert assignments[1] == 1  # closer to enemy 1

    def test_no_enemies(self):
        assignments = assign_targets([(0, 0), (10, 0)], [])
        assert assignments == [-1, -1]

    def test_no_squad(self):
        assert assign_targets([], [(5, 0)]) == []

    def test_more_squad_than_enemies(self):
        squad = [(0, 0), (5, 0), (10, 0)]
        enemies = [(3, 0)]
        assignments = assign_targets(squad, enemies)
        assert len(assignments) == 3
        assert all(a == 0 for a in assignments)


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

class TestSuppressionCone:
    def test_returns_three_vertices(self):
        cone = suppression_cone((0, 0), (50, 0))
        assert len(cone) == 3
        assert cone[0] == (0, 0)  # shooter position

    def test_cone_extends_toward_target(self):
        cone = suppression_cone((0, 0), (50, 0), range_m=100.0)
        # Both far points should be roughly 100m from shooter
        for p in cone[1:]:
            assert distance((0, 0), p) == pytest.approx(100.0, abs=0.1)


class TestIsSuppressed:
    def test_inside_cone(self):
        cone = suppression_cone((0, 0), (50, 0), cone_half_angle=15.0, range_m=60.0)
        # Point directly between shooter and target
        assert is_suppressed((25, 0), [cone])

    def test_outside_cone(self):
        cone = suppression_cone((0, 0), (50, 0), cone_half_angle=15.0, range_m=60.0)
        # Point far to the side
        assert not is_suppressed((25, 50), [cone])

    def test_no_zones(self):
        assert not is_suppressed((0, 0), [])

    def test_behind_shooter_not_suppressed(self):
        cone = suppression_cone((0, 0), (50, 0), range_m=50.0)
        assert not is_suppressed((-10, 0), [cone])


# ---------------------------------------------------------------------------
# Behavior trees
# ---------------------------------------------------------------------------

class TestAssaultTree:
    def test_retreat_when_critical(self):
        tree = make_assault_tree()
        ctx = {
            "health": 0.1,
            "ammo_ratio": 0.02,
            "enemies_visible": 5,
            "allies_nearby": 0,
            "enemies": True,
            "enemy_in_range": True,
            "in_cover": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "retreat"

    def test_engage_from_cover(self):
        tree = make_assault_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 1,
            "allies_nearby": 2,
            "enemies": True,
            "enemy_in_range": True,
            "in_cover": True,
            "engage_duration": 0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "engage"

    def test_seek_cover_when_exposed(self):
        tree = make_assault_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 1,
            "allies_nearby": 2,
            "enemies": True,
            "enemy_in_range": False,
            "in_cover": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "seek_cover"

    def test_flank_when_stalled(self):
        tree = make_assault_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 1,
            "allies_nearby": 2,
            "enemies": True,
            "enemy_in_range": True,
            "in_cover": False,
            "engage_duration": 15.0,
            "stall_threshold": 10.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "flank"

    def test_push_when_no_enemies(self):
        tree = make_assault_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 0,
            "allies_nearby": 2,
            "enemies": False,
            "enemy_in_range": False,
            "in_cover": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "push"


class TestDefenderTree:
    def test_hold_by_default(self):
        tree = make_defender_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 0,
            "allies_nearby": 2,
            "enemies": False,
            "enemy_in_range": False,
            "in_cover": True,
            "is_suppressed": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "hold"

    def test_engage_enemy_in_range(self):
        tree = make_defender_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 1,
            "allies_nearby": 2,
            "enemies": True,
            "enemy_in_range": True,
            "in_cover": True,
            "is_suppressed": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "engage"

    def test_fall_back_when_suppressed(self):
        tree = make_defender_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 1,
            "allies_nearby": 2,
            "enemies": True,
            "enemy_in_range": False,
            "in_cover": False,
            "is_suppressed": True,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "fall_back"


class TestSniperTree:
    def test_observe_by_default(self):
        tree = make_sniper_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 0,
            "allies_nearby": 0,
            "enemies": False,
            "enemy_in_range": False,
            "in_cover": True,
            "shots_fired": 0,
            "time": 0.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "observe"

    def test_engage_from_cover(self):
        tree = make_sniper_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 1,
            "allies_nearby": 0,
            "enemies": True,
            "enemy_in_range": True,
            "in_cover": True,
            "shots_fired": 0,
            "time": 0.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "engage"

    def test_find_vantage_when_no_cover(self):
        tree = make_sniper_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 0,
            "allies_nearby": 0,
            "enemies": False,
            "enemy_in_range": False,
            "in_cover": False,
            "shots_fired": 0,
            "time": 0.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "find_vantage"


class TestSquadLeaderTree:
    def test_assess_by_default(self):
        tree = make_squad_leader_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 0,
            "allies_nearby": 3,
            "enemies": False,
            "squad_members": False,
            "is_suppressed": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "assess"

    def test_assign_targets_with_squad(self):
        tree = make_squad_leader_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 2,
            "allies_nearby": 3,
            "enemies": True,
            "squad_members": [1, 2, 3],
            "is_suppressed": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "assign_targets"

    def test_advance_squad_no_enemies(self):
        tree = make_squad_leader_tree()
        ctx = {
            "health": 0.9,
            "ammo_ratio": 0.8,
            "enemies_visible": 0,
            "allies_nearby": 3,
            "enemies": False,
            "squad_members": [1, 2, 3],
            "is_suppressed": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "advance_squad"

    def test_order_retreat(self):
        tree = make_squad_leader_tree()
        ctx = {
            "health": 0.1,
            "ammo_ratio": 0.02,
            "enemies_visible": 5,
            "allies_nearby": 0,
            "enemies": True,
            "squad_members": [1, 2],
            "is_suppressed": True,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "order_retreat"

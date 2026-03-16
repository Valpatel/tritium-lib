"""Tests for steering behaviors module.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

import math
import pytest

from tritium_lib.movement.steering import (
    Vec2,
    distance,
    magnitude,
    normalize,
    truncate,
    heading_to_vec,
    seek,
    flee,
    arrive,
    wander,
    pursue,
    evade,
    follow_path,
    avoid_obstacles,
    separate,
    align,
    cohere,
    flock,
    formation_offset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_vec2(v):
    """Assert value is a tuple of two floats."""
    assert isinstance(v, tuple), f"Expected tuple, got {type(v)}"
    assert len(v) == 2, f"Expected 2 elements, got {len(v)}"
    assert isinstance(v[0], (int, float))
    assert isinstance(v[1], (int, float))


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_distance_same_point(self):
        assert distance((0, 0), (0, 0)) == 0.0

    def test_distance_known(self):
        assert distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_normalize_unit(self):
        n = normalize((3, 4))
        assert magnitude(n) == pytest.approx(1.0)

    def test_normalize_zero(self):
        assert normalize((0, 0)) == (0.0, 0.0)

    def test_truncate_short_vector(self):
        v = truncate((1, 0), 5.0)
        assert v == (1, 0)

    def test_truncate_long_vector(self):
        v = truncate((10, 0), 3.0)
        assert magnitude(v) == pytest.approx(3.0)
        assert v[0] == pytest.approx(3.0)

    def test_heading_to_vec_zero(self):
        v = heading_to_vec(0.0)
        assert v[0] == pytest.approx(1.0)
        assert v[1] == pytest.approx(0.0, abs=1e-10)

    def test_heading_to_vec_90(self):
        v = heading_to_vec(math.pi / 2)
        assert v[0] == pytest.approx(0.0, abs=1e-10)
        assert v[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Basic behaviors
# ---------------------------------------------------------------------------

class TestSeek:
    def test_returns_vec2(self):
        assert_vec2(seek((0, 0), (10, 0), 5.0))

    def test_moves_toward_target(self):
        force = seek((0, 0), (10, 0), 5.0)
        assert force[0] > 0, "Should move in +x toward target"

    def test_magnitude_equals_max_speed(self):
        force = seek((0, 0), (10, 5), 3.0)
        assert magnitude(force) == pytest.approx(3.0)

    def test_at_target_returns_zero(self):
        force = seek((5, 5), (5, 5), 3.0)
        assert force == (0.0, 0.0)

    def test_diagonal_direction(self):
        force = seek((0, 0), (1, 1), 1.0)
        assert force[0] > 0 and force[1] > 0


class TestFlee:
    def test_returns_vec2(self):
        assert_vec2(flee((0, 0), (10, 0), 5.0))

    def test_moves_away_from_threat(self):
        force = flee((0, 0), (10, 0), 5.0)
        assert force[0] < 0, "Should move in -x away from threat"

    def test_opposite_of_seek(self):
        s = seek((0, 0), (10, 5), 3.0)
        f = flee((0, 0), (10, 5), 3.0)
        # Directions should be opposite
        assert dot(s, f) < 0


class TestArrive:
    def test_returns_vec2(self):
        assert_vec2(arrive((0, 0), (10, 0), 5.0, 3.0))

    def test_full_speed_outside_radius(self):
        force = arrive((0, 0), (100, 0), 5.0, 3.0)
        assert magnitude(force) == pytest.approx(5.0)

    def test_slows_inside_radius(self):
        force = arrive((0, 0), (1, 0), 10.0, 5.0)
        assert magnitude(force) < 10.0, "Should be slower inside slow_radius"
        assert magnitude(force) > 0, "Should still be moving"

    def test_at_target(self):
        force = arrive((5, 5), (5, 5), 10.0, 3.0)
        assert force == (0.0, 0.0)

    def test_speed_proportional_to_distance(self):
        # At half the slow_radius, speed should be half max_speed
        force = arrive((0, 0), (2.5, 0), 10.0, 5.0)
        assert magnitude(force) == pytest.approx(5.0)


class TestWander:
    def test_returns_vec2(self):
        assert_vec2(wander((0, 0), (1, 0), 2.0, 4.0, 0.5))

    def test_nonzero_force(self):
        force = wander((0, 0), (1, 0), 2.0, 4.0, 0.5)
        assert magnitude(force) > 0

    def test_varies_between_calls(self):
        forces = [wander((0, 0), (1, 0), 2.0, 4.0, 1.0) for _ in range(20)]
        # With jitter, not all forces should be identical
        unique = set(forces)
        assert len(unique) > 1, "Wander should produce varied results"


class TestPursue:
    def test_returns_vec2(self):
        assert_vec2(pursue((0, 0), (1, 0), (10, 0), (1, 0), 5.0))

    def test_leads_moving_target(self):
        # Target moving in +x; pursue should aim ahead of current position
        force = pursue((0, 0), (0, 0), (10, 0), (5, 0), 5.0)
        assert force[0] > 0


class TestEvade:
    def test_returns_vec2(self):
        assert_vec2(evade((0, 0), (1, 0), (10, 0), (0, 0), 5.0))

    def test_moves_away(self):
        force = evade((0, 0), (0, 0), (10, 0), (1, 0), 5.0)
        assert force[0] < 0, "Should flee from predicted threat position"


# ---------------------------------------------------------------------------
# Path following
# ---------------------------------------------------------------------------

class TestFollowPath:
    def test_returns_vec2(self):
        path = [(0, 0), (10, 0), (10, 10)]
        assert_vec2(follow_path((0, 0), (1, 0), path, 1.0, 5.0))

    def test_empty_path(self):
        assert follow_path((0, 0), (1, 0), [], 1.0, 5.0) == (0.0, 0.0)

    def test_progresses_along_waypoints(self):
        path = [(0, 0), (10, 0), (20, 0)]
        # Near first waypoint, should head toward second
        force = follow_path((0.5, 0), (1, 0), path, 2.0, 5.0)
        assert force[0] > 0, "Should progress forward along path"

    def test_arrives_at_final_waypoint(self):
        path = [(0, 0), (10, 0)]
        # Very close to final waypoint — should decelerate (arrive behavior)
        force = follow_path((9.9, 0), (1, 0), path, 1.0, 10.0)
        assert magnitude(force) < 10.0, "Should decelerate near final waypoint"


# ---------------------------------------------------------------------------
# Obstacle avoidance
# ---------------------------------------------------------------------------

class TestAvoidObstacles:
    def test_returns_vec2(self):
        obs = [((5, 0), 1.0)]
        assert_vec2(avoid_obstacles((0, 0), (1, 0), obs, 10.0))

    def test_no_obstacles(self):
        force = avoid_obstacles((0, 0), (1, 0), [], 10.0)
        assert force == (0.0, 0.0)

    def test_zero_velocity(self):
        obs = [((5, 0), 1.0)]
        force = avoid_obstacles((0, 0), (0, 0), obs, 10.0)
        assert force == (0.0, 0.0)

    def test_steers_around_obstacle(self):
        # Obstacle directly ahead — should produce lateral force
        obs = [((5, 0), 1.0)]
        force = avoid_obstacles((0, 0), (1, 0), obs, 10.0)
        assert magnitude(force) > 0, "Should steer to avoid"
        # Lateral force should be perpendicular to heading
        assert abs(force[1]) > 0 or abs(force[0]) > 0

    def test_ignores_obstacle_behind(self):
        obs = [((-5, 0), 1.0)]
        force = avoid_obstacles((0, 0), (1, 0), obs, 10.0)
        assert force == (0.0, 0.0)

    def test_ignores_distant_obstacle(self):
        obs = [((100, 0), 1.0)]
        force = avoid_obstacles((0, 0), (1, 0), obs, 5.0)
        assert force == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Group behaviors
# ---------------------------------------------------------------------------

class TestSeparate:
    def test_returns_vec2(self):
        assert_vec2(separate((0, 0), [(1, 0)], 5.0))

    def test_pushes_away_from_close_neighbor(self):
        force = separate((0, 0), [(1, 0)], 5.0)
        assert force[0] < 0, "Should push away in -x"

    def test_no_neighbors(self):
        force = separate((0, 0), [], 5.0)
        assert force == (0.0, 0.0)

    def test_ignores_distant_neighbors(self):
        force = separate((0, 0), [(100, 0)], 5.0)
        assert force == (0.0, 0.0)


class TestAlign:
    def test_returns_vec2(self):
        assert_vec2(align((1, 0), [(0, 1)]))

    def test_average_velocity(self):
        result = align((1, 0), [(0, 2), (0, 4)])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(3.0)

    def test_no_neighbors(self):
        assert align((1, 0), []) == (0.0, 0.0)


class TestCohere:
    def test_returns_vec2(self):
        assert_vec2(cohere((0, 0), [(10, 10)]))

    def test_moves_toward_centroid(self):
        force = cohere((0, 0), [(10, 0), (10, 10)])
        assert force[0] > 0, "Should move toward centroid in +x"

    def test_no_neighbors(self):
        assert cohere((0, 0), []) == (0.0, 0.0)


class TestFlock:
    def test_returns_vec2(self):
        neighbors = [((5, 0), (1, 0)), ((0, 5), (0, 1))]
        assert_vec2(flock((0, 0), (1, 0), neighbors, 3.0, 5.0))

    def test_no_neighbors(self):
        assert flock((0, 0), (1, 0), [], 3.0, 5.0) == (0.0, 0.0)

    def test_produces_bounded_force(self):
        neighbors = [((1, 0), (1, 0)), ((0, 1), (0, 1)), ((-1, 0), (-1, 0))]
        force = flock((0, 0), (1, 0), neighbors, 3.0, 5.0)
        assert magnitude(force) <= 5.0 + 1e-6, "Force should be bounded by max_speed"

    def test_group_cohesion(self):
        # Distant neighbors should pull agent toward them
        neighbors = [((100, 0), (0, 0)), ((100, 0), (0, 0))]
        force = flock((0, 0), (1, 0), neighbors, 2.0, 10.0)
        assert force[0] > 0, "Should be pulled toward distant group"


# ---------------------------------------------------------------------------
# Formation
# ---------------------------------------------------------------------------

class TestFormationOffset:
    def test_returns_vec2(self):
        assert_vec2(formation_offset((0, 0), 0.0, (-2, 1)))

    def test_zero_heading_no_rotation(self):
        # Heading 0 = facing +x; offset (-2, 0) = 2m behind leader
        pos = formation_offset((10, 5), 0.0, (-2, 0))
        assert pos[0] == pytest.approx(8.0)
        assert pos[1] == pytest.approx(5.0)

    def test_90_degree_heading(self):
        # Heading pi/2 = facing +y; offset (-2, 0) = 2m behind = -y
        pos = formation_offset((0, 0), math.pi / 2, (-2, 0))
        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(-2.0)

    def test_lateral_offset(self):
        # Heading 0; offset (0, 3) = 3m to the left
        pos = formation_offset((0, 0), 0.0, (0, 3))
        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(3.0)

    def test_v_formation(self):
        # Two wingmen at (-2, +-3) behind a leader at origin heading +x
        left = formation_offset((0, 0), 0.0, (-2, 3))
        right = formation_offset((0, 0), 0.0, (-2, -3))
        assert left[0] == pytest.approx(-2.0)
        assert left[1] == pytest.approx(3.0)
        assert right[0] == pytest.approx(-2.0)
        assert right[1] == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# Integration / composability
# ---------------------------------------------------------------------------

class TestComposability:
    def test_forces_add_together(self):
        s = seek((0, 0), (10, 0), 5.0)
        f = flee((0, 0), (0, 10), 3.0)
        combined = (s[0] + f[0], s[1] + f[1])
        assert_vec2(combined)
        # Combined should have +x (seek) and -y (flee) components
        assert combined[0] > 0
        assert combined[1] < 0

    def test_truncated_combined_force(self):
        s = seek((0, 0), (10, 0), 10.0)
        a = separate((0, 0), [(1, 0)], 5.0)
        combined = (s[0] + a[0], s[1] + a[1])
        clamped = truncate(combined, 5.0)
        assert magnitude(clamped) <= 5.0 + 1e-6

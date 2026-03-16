# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the city3d collision system.

Covers: ColliderType, Collider, CollisionResult, SpatialHashGrid,
narrow-phase tests (circle-circle, circle-AABB, AABB-AABB),
CollisionWorld (add/remove/update, rules, check_all, resolve,
raycast, query_area), and default city-sim rules.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.sim_engine.collision import (
    KILL_SPEED_THRESHOLD,
    Collider,
    ColliderType,
    CollisionResult,
    CollisionWorld,
    ResponseType,
    SpatialHashGrid,
    _aabb_aabb,
    _circle_aabb,
    _circle_circle,
    _narrow_phase,
    create_city_world,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _circle(eid: str, x: float, y: float, r: float = 1.0,
            vx: float = 0, vy: float = 0,
            mass: float = 1.0, layer: str = "default",
            static: bool = False) -> Collider:
    return Collider(
        entity_id=eid, collider_type=ColliderType.CIRCLE,
        position=(x, y), velocity=(vx, vy), radius=r,
        mass=mass, layer=layer, is_static=static,
    )


def _box(eid: str, x: float, y: float, hw: float = 2.0, hd: float = 2.0,
         vx: float = 0, vy: float = 0,
         mass: float = 1.0, layer: str = "default",
         static: bool = False) -> Collider:
    return Collider(
        entity_id=eid, collider_type=ColliderType.AABB,
        position=(x, y), velocity=(vx, vy),
        half_width=hw, half_depth=hd,
        mass=mass, layer=layer, is_static=static,
    )


# ===========================================================================
# ColliderType enum
# ===========================================================================

class TestColliderType:
    def test_values(self):
        assert ColliderType.CIRCLE.value == "circle"
        assert ColliderType.AABB.value == "aabb"

    def test_members(self):
        assert set(ColliderType) == {ColliderType.CIRCLE, ColliderType.AABB}


# ===========================================================================
# ResponseType enum
# ===========================================================================

class TestResponseType:
    def test_values(self):
        assert ResponseType.PUSH.value == "push"
        assert ResponseType.DAMAGE.value == "damage"
        assert ResponseType.KILL.value == "kill"
        assert ResponseType.STOP.value == "stop"
        assert ResponseType.IGNORE.value == "ignore"


# ===========================================================================
# Collider dataclass
# ===========================================================================

class TestCollider:
    def test_circle_bounds(self):
        c = _circle("c1", 10, 20, r=3)
        assert c.min_x == pytest.approx(7)
        assert c.max_x == pytest.approx(13)
        assert c.min_y == pytest.approx(17)
        assert c.max_y == pytest.approx(23)

    def test_aabb_bounds(self):
        b = _box("b1", 5, 5, hw=2, hd=3)
        assert b.min_x == pytest.approx(3)
        assert b.max_x == pytest.approx(7)
        assert b.min_y == pytest.approx(2)
        assert b.max_y == pytest.approx(8)

    def test_bounding_radius_circle(self):
        c = _circle("c1", 0, 0, r=5)
        assert c.bounding_radius == pytest.approx(5)

    def test_bounding_radius_aabb(self):
        b = _box("b1", 0, 0, hw=3, hd=4)
        assert b.bounding_radius == pytest.approx(5)  # 3-4-5 triangle

    def test_speed(self):
        c = _circle("c1", 0, 0, vx=3, vy=4)
        assert c.speed == pytest.approx(5)

    def test_default_values(self):
        c = Collider(entity_id="x", collider_type=ColliderType.CIRCLE)
        assert c.position == (0.0, 0.0)
        assert c.velocity == (0.0, 0.0)
        assert c.mass == 1.0
        assert c.is_static is False
        assert c.layer == "default"
        assert c.on_collision is None


# ===========================================================================
# CollisionResult dataclass
# ===========================================================================

class TestCollisionResult:
    def test_fields(self):
        r = CollisionResult("a", "b", 1.5, (1.0, 0.0), 3.0)
        assert r.entity_a == "a"
        assert r.entity_b == "b"
        assert r.overlap == pytest.approx(1.5)
        assert r.normal == (1.0, 0.0)
        assert r.impact_speed == pytest.approx(3.0)


# ===========================================================================
# SpatialHashGrid
# ===========================================================================

class TestSpatialHashGrid:
    def test_insert_and_query_cell(self):
        grid = SpatialHashGrid(cell_size=10)
        c = _circle("c1", 5, 5, r=1)
        grid.insert(c)
        assert "c1" in grid.query((0, 0))

    def test_query_radius(self):
        grid = SpatialHashGrid(cell_size=10)
        grid.insert(_circle("c1", 5, 5, r=1))
        grid.insert(_circle("c2", 50, 50, r=1))
        nearby = grid.query_radius((5, 5), 5)
        assert "c1" in nearby
        assert "c2" not in nearby

    def test_candidate_pairs_overlap(self):
        grid = SpatialHashGrid(cell_size=10)
        grid.insert(_circle("a", 5, 5, r=1))
        grid.insert(_circle("b", 6, 5, r=1))
        pairs = grid.candidate_pairs()
        assert ("a", "b") in pairs or ("b", "a") in pairs

    def test_no_pairs_far_apart(self):
        grid = SpatialHashGrid(cell_size=10)
        grid.insert(_circle("a", 0, 0, r=1))
        grid.insert(_circle("b", 100, 100, r=1))
        pairs = grid.candidate_pairs()
        assert len(pairs) == 0

    def test_clear(self):
        grid = SpatialHashGrid(cell_size=10)
        grid.insert(_circle("a", 5, 5, r=1))
        grid.clear()
        assert len(grid.query((0, 0))) == 0

    def test_aabb_multi_cell_insertion(self):
        grid = SpatialHashGrid(cell_size=5)
        b = _box("b1", 5, 5, hw=6, hd=6)  # spans many cells
        grid.insert(b)
        # Should be in cell containing (0,0) since min is -1
        found = grid.query_radius((0, 0), 2)
        assert "b1" in found


# ===========================================================================
# Narrow phase: circle-circle
# ===========================================================================

class TestCircleCircle:
    def test_overlap(self):
        a = _circle("a", 0, 0, r=2)
        b = _circle("b", 3, 0, r=2)
        r = _circle_circle(a, b)
        assert r is not None
        assert r.overlap == pytest.approx(1.0)
        assert r.normal[0] == pytest.approx(1.0)
        assert r.normal[1] == pytest.approx(0.0)

    def test_no_overlap(self):
        a = _circle("a", 0, 0, r=1)
        b = _circle("b", 5, 0, r=1)
        assert _circle_circle(a, b) is None

    def test_touching(self):
        a = _circle("a", 0, 0, r=1)
        b = _circle("b", 2, 0, r=1)
        assert _circle_circle(a, b) is None  # dist == min_dist

    def test_coincident(self):
        a = _circle("a", 5, 5, r=1)
        b = _circle("b", 5, 5, r=1)
        r = _circle_circle(a, b)
        assert r is not None
        assert r.overlap == pytest.approx(2.0, abs=0.01)

    def test_impact_speed(self):
        a = _circle("a", 0, 0, r=2, vx=5, vy=0)
        b = _circle("b", 3, 0, r=2, vx=-3, vy=0)
        r = _circle_circle(a, b)
        assert r is not None
        assert r.impact_speed == pytest.approx(8.0)

    def test_diagonal_overlap(self):
        a = _circle("a", 0, 0, r=2)
        b = _circle("b", 2, 2, r=2)
        r = _circle_circle(a, b)
        assert r is not None
        assert r.overlap > 0


# ===========================================================================
# Narrow phase: circle-AABB
# ===========================================================================

class TestCircleAABB:
    def test_circle_hits_box_side(self):
        circle = _circle("c", 5, 0, r=2)
        box = _box("b", 0, 0, hw=2, hd=2, static=True)
        r = _circle_aabb(circle, box)
        assert r is not None
        assert r.overlap > 0

    def test_no_overlap(self):
        circle = _circle("c", 10, 0, r=1)
        box = _box("b", 0, 0, hw=2, hd=2)
        assert _circle_aabb(circle, box) is None

    def test_circle_inside_box(self):
        circle = _circle("c", 0.5, 0.5, r=0.5)
        box = _box("b", 0, 0, hw=5, hd=5)
        r = _circle_aabb(circle, box)
        assert r is not None
        assert r.overlap > 0

    def test_circle_corner(self):
        circle = _circle("c", 3.5, 3.5, r=1)
        box = _box("b", 0, 0, hw=3, hd=3)
        r = _circle_aabb(circle, box)
        # Distance from circle center to box corner (3,3): sqrt(0.25+0.25) ~= 0.707
        assert r is not None


# ===========================================================================
# Narrow phase: AABB-AABB
# ===========================================================================

class TestAABBAABB:
    def test_overlap_x(self):
        a = _box("a", 0, 0, hw=2, hd=2)
        b = _box("b", 3, 0, hw=2, hd=2)
        r = _aabb_aabb(a, b)
        assert r is not None
        assert r.overlap == pytest.approx(1.0)
        assert r.normal == (1.0, 0.0)

    def test_overlap_y(self):
        a = _box("a", 0, 0, hw=2, hd=2)
        b = _box("b", 0, 3, hw=2, hd=2)
        r = _aabb_aabb(a, b)
        assert r is not None
        assert r.overlap == pytest.approx(1.0)
        assert r.normal == (0.0, 1.0)

    def test_no_overlap(self):
        a = _box("a", 0, 0, hw=1, hd=1)
        b = _box("b", 10, 10, hw=1, hd=1)
        assert _aabb_aabb(a, b) is None

    def test_touching(self):
        a = _box("a", 0, 0, hw=1, hd=1)
        b = _box("b", 2, 0, hw=1, hd=1)
        # overlap_x = 0 -> no collision
        assert _aabb_aabb(a, b) is None

    def test_negative_direction(self):
        a = _box("a", 0, 0, hw=2, hd=2)
        b = _box("b", -3, 0, hw=2, hd=2)
        r = _aabb_aabb(a, b)
        assert r is not None
        assert r.normal == (-1.0, 0.0)


# ===========================================================================
# Narrow phase dispatch
# ===========================================================================

class TestNarrowPhase:
    def test_dispatches_circle_circle(self):
        a = _circle("a", 0, 0, r=2)
        b = _circle("b", 1, 0, r=2)
        assert _narrow_phase(a, b) is not None

    def test_dispatches_aabb_aabb(self):
        a = _box("a", 0, 0, hw=2, hd=2)
        b = _box("b", 1, 0, hw=2, hd=2)
        assert _narrow_phase(a, b) is not None

    def test_dispatches_circle_aabb(self):
        c = _circle("c", 3, 0, r=2)
        b = _box("b", 0, 0, hw=2, hd=2)
        assert _narrow_phase(c, b) is not None

    def test_dispatches_aabb_circle(self):
        b = _box("b", 0, 0, hw=2, hd=2)
        c = _circle("c", 3, 0, r=2)
        assert _narrow_phase(b, c) is not None


# ===========================================================================
# CollisionWorld: add / remove / update
# ===========================================================================

class TestWorldManagement:
    def test_add_and_count(self):
        w = CollisionWorld()
        w.add(_circle("a", 0, 0))
        w.add(_circle("b", 1, 0))
        assert len(w.colliders) == 2

    def test_remove(self):
        w = CollisionWorld()
        w.add(_circle("a", 0, 0))
        w.remove("a")
        assert "a" not in w.colliders

    def test_remove_static(self):
        w = CollisionWorld()
        w.add(_box("bld", 0, 0, static=True, layer="building"))
        assert len(w.static_colliders) == 1
        w.remove("bld")
        assert len(w.static_colliders) == 0

    def test_update_position(self):
        w = CollisionWorld()
        w.add(_circle("a", 0, 0))
        w.update("a", position=(5, 5))
        assert w.colliders["a"].position == (5, 5)

    def test_update_velocity(self):
        w = CollisionWorld()
        w.add(_circle("a", 0, 0))
        w.update("a", velocity=(1, 2))
        assert w.colliders["a"].velocity == (1, 2)

    def test_update_nonexistent(self):
        w = CollisionWorld()
        w.update("missing", position=(1, 1))  # Should not raise

    def test_remove_nonexistent(self):
        w = CollisionWorld()
        w.remove("missing")  # Should not raise


# ===========================================================================
# CollisionWorld: rules
# ===========================================================================

class TestWorldRules:
    def test_default_rules(self):
        w = CollisionWorld()
        assert w.get_rule("car", "car") == "push"
        assert w.get_rule("car", "pedestrian") == "damage"
        assert w.get_rule("pedestrian", "building") == "push"
        assert w.get_rule("car", "building") == "stop"

    def test_set_rule_symmetric(self):
        w = CollisionWorld()
        w.set_rule("drone", "car", "damage")
        assert w.get_rule("drone", "car") == "damage"
        assert w.get_rule("car", "drone") == "damage"

    def test_unknown_layers_default_push(self):
        w = CollisionWorld()
        assert w.get_rule("alien", "robot") == "push"

    def test_override_rule(self):
        w = CollisionWorld()
        w.set_rule("car", "car", "stop")
        assert w.get_rule("car", "car") == "stop"


# ===========================================================================
# CollisionWorld: check_all
# ===========================================================================

class TestCheckAll:
    def test_two_overlapping_circles(self):
        w = CollisionWorld()
        w.add(_circle("a", 0, 0, r=2, layer="pedestrian"))
        w.add(_circle("b", 3, 0, r=2, layer="pedestrian"))
        results = w.check_all()
        assert len(results) == 1

    def test_no_collision(self):
        w = CollisionWorld()
        w.add(_circle("a", 0, 0, r=1))
        w.add(_circle("b", 100, 100, r=1))
        assert len(w.check_all()) == 0

    def test_skip_static_static(self):
        w = CollisionWorld()
        w.add(_box("b1", 0, 0, hw=5, hd=5, static=True, layer="building"))
        w.add(_box("b2", 3, 0, hw=5, hd=5, static=True, layer="building"))
        assert len(w.check_all()) == 0

    def test_ignore_rule_skips(self):
        w = CollisionWorld()
        w.set_rule("ghost", "wall", "ignore")
        w.add(_circle("g", 0, 0, r=2, layer="ghost"))
        w.add(_box("w", 1, 0, hw=2, hd=2, layer="wall", static=True))
        assert len(w.check_all()) == 0

    def test_mixed_shapes(self):
        w = CollisionWorld()
        w.add(_circle("c", 3, 0, r=2, layer="pedestrian"))
        w.add(_box("b", 0, 0, hw=2, hd=2, layer="building", static=True))
        results = w.check_all()
        assert len(results) == 1


# ===========================================================================
# CollisionWorld: resolve — push
# ===========================================================================

class TestResolvePush:
    def test_push_separates(self):
        w = CollisionWorld()
        a = _circle("a", 0, 0, r=2, layer="pedestrian")
        b = _circle("b", 3, 0, r=2, layer="pedestrian")
        w.add(a)
        w.add(b)
        results = w.check_all()
        assert len(results) == 1
        w.resolve(results)
        # After push, they should be further apart.
        dist = math.hypot(
            w.colliders["b"].position[0] - w.colliders["a"].position[0],
            w.colliders["b"].position[1] - w.colliders["a"].position[1],
        )
        assert dist >= 3.0  # At least as far as before (pushed apart)

    def test_push_static_wall(self):
        w = CollisionWorld()
        ped = _circle("p", 1.5, 0, r=1, layer="pedestrian")
        wall = _box("w", 0, 0, hw=1, hd=10, layer="building", static=True)
        w.add(ped)
        w.add(wall)
        results = w.check_all()
        w.resolve(results)
        # Wall should not have moved.
        assert w.colliders["w"].position == (0, 0)

    def test_mass_based_distribution(self):
        w = CollisionWorld()
        heavy = _circle("h", 0, 0, r=2, mass=100, layer="pedestrian")
        light = _circle("l", 3, 0, r=2, mass=1, layer="pedestrian")
        w.add(heavy)
        w.add(light)
        results = w.check_all()
        old_h_x = heavy.position[0]
        w.resolve(results)
        # Heavy should barely move.
        assert abs(w.colliders["h"].position[0] - old_h_x) < 0.05


# ===========================================================================
# CollisionWorld: resolve — stop
# ===========================================================================

class TestResolveStop:
    def test_car_stops_at_building(self):
        w = CollisionWorld()
        car = _circle("car1", 3, 0, r=1.5, vx=-5, layer="car")
        bld = _box("bld1", 0, 0, hw=2, hd=2, layer="building", static=True)
        w.add(car)
        w.add(bld)
        results = w.check_all()
        w.resolve(results)
        assert w.colliders["car1"].velocity == (0.0, 0.0)


# ===========================================================================
# CollisionWorld: resolve — damage / kill
# ===========================================================================

class TestResolveDamageKill:
    def test_low_speed_damage(self):
        w = CollisionWorld()
        car = _circle("car1", 0, 0, r=2, vx=3, layer="car")
        ped = _circle("ped1", 3, 0, r=0.5, layer="pedestrian")
        w.add(car)
        w.add(ped)
        results = w.check_all()
        assert len(results) == 1
        assert results[0].impact_speed < KILL_SPEED_THRESHOLD
        w.resolve(results)
        # Ped still has some velocity (pushed), not killed.
        # Just check resolve didn't crash.

    def test_high_speed_kill(self):
        w = CollisionWorld()
        car = _circle("car1", 0, 0, r=2, vx=15, layer="car")
        ped = _circle("ped1", 2.5, 0, r=0.5, layer="pedestrian")
        w.add(car)
        w.add(ped)
        results = w.check_all()
        assert len(results) == 1
        assert results[0].impact_speed > KILL_SPEED_THRESHOLD
        w.resolve(results)
        # Ped velocity zeroed (killed).
        assert w.colliders["ped1"].velocity == (0.0, 0.0)

    def test_kill_rule_explicit(self):
        w = CollisionWorld()
        w.set_rule("lava", "pedestrian", "kill")
        lava = _box("lava1", 0, 0, hw=5, hd=5, layer="lava", static=True)
        ped = _circle("ped1", 2, 2, r=0.5, vx=1, vy=1, layer="pedestrian")
        w.add(lava)
        w.add(ped)
        results = w.check_all()
        w.resolve(results)
        assert w.colliders["ped1"].velocity == (0.0, 0.0)


# ===========================================================================
# CollisionWorld: raycast
# ===========================================================================

class TestRaycast:
    def test_hits_circle(self):
        w = CollisionWorld()
        w.add(_circle("t", 10, 0, r=2))
        hit = w.raycast((0, 0), (1, 0), 20)
        assert hit is not None
        assert hit.entity_b == "t"

    def test_misses(self):
        w = CollisionWorld()
        w.add(_circle("t", 10, 10, r=1))
        hit = w.raycast((0, 0), (1, 0), 20)
        assert hit is None

    def test_hits_aabb(self):
        w = CollisionWorld()
        w.add(_box("wall", 10, 0, hw=2, hd=2))
        hit = w.raycast((0, 0), (1, 0), 20)
        assert hit is not None
        assert hit.entity_b == "wall"

    def test_max_dist_limit(self):
        w = CollisionWorld()
        w.add(_circle("far", 50, 0, r=2))
        hit = w.raycast((0, 0), (1, 0), 10)
        assert hit is None

    def test_returns_closest(self):
        w = CollisionWorld()
        w.add(_circle("near", 5, 0, r=1))
        w.add(_circle("far", 15, 0, r=1))
        hit = w.raycast((0, 0), (1, 0), 20)
        assert hit is not None
        assert hit.entity_b == "near"

    def test_zero_direction(self):
        w = CollisionWorld()
        w.add(_circle("t", 5, 0, r=1))
        hit = w.raycast((0, 0), (0, 0), 10)
        assert hit is None


# ===========================================================================
# CollisionWorld: query_area
# ===========================================================================

class TestQueryArea:
    def test_finds_nearby(self):
        w = CollisionWorld()
        w.add(_circle("a", 3, 0, r=1))
        w.add(_circle("b", 100, 100, r=1))
        nearby = w.query_area((0, 0), 10)
        ids = {c.entity_id for c in nearby}
        assert "a" in ids
        assert "b" not in ids

    def test_empty_world(self):
        w = CollisionWorld()
        assert len(w.query_area((0, 0), 10)) == 0

    def test_includes_static(self):
        w = CollisionWorld()
        w.add(_box("bld", 3, 0, hw=2, hd=2, static=True))
        nearby = w.query_area((0, 0), 10)
        assert len(nearby) == 1


# ===========================================================================
# Full integration: city simulation scenarios
# ===========================================================================

class TestCityScenarios:
    def test_car_fender_bender(self):
        """Two cars collide head-on at moderate speed."""
        w = create_city_world()
        w.add(_circle("car_a", 0, 0, r=2, vx=5, mass=1000, layer="car"))
        w.add(_circle("car_b", 5, 0, r=2, vx=-5, mass=1000, layer="car"))
        results = w.check_all()
        assert len(results) == 1
        w.resolve(results)
        # Both cars should have reduced/reversed velocity.

    def test_pedestrian_crowd_push(self):
        """Three pedestrians bunched up push apart."""
        w = create_city_world()
        w.add(_circle("p1", 0, 0, r=0.4, layer="pedestrian"))
        w.add(_circle("p2", 0.5, 0, r=0.4, layer="pedestrian"))
        w.add(_circle("p3", 0.25, 0.4, r=0.4, layer="pedestrian"))
        results = w.check_all()
        assert len(results) >= 2
        w.resolve(results)

    def test_pedestrian_bounces_off_building(self):
        """Pedestrian walking into a building wall."""
        w = create_city_world()
        w.add(_circle("ped", 4, 0, r=0.5, vx=-2, layer="pedestrian"))
        w.add(_box("bld", 0, 0, hw=3, hd=10, layer="building", static=True))
        results = w.check_all()
        assert len(results) == 1
        w.resolve(results)
        # Ped should have bounced (vx direction changed or reduced).
        assert w.colliders["bld"].position == (0, 0)  # Building didn't move

    def test_car_into_building_stops(self):
        """Car drives into a building and stops."""
        w = create_city_world()
        w.add(_circle("car", 4, 0, r=1.5, vx=-10, layer="car"))
        w.add(_box("bld", 0, 0, hw=2, hd=5, layer="building", static=True))
        results = w.check_all()
        w.resolve(results)
        assert w.colliders["car"].velocity == (0.0, 0.0)

    def test_projectile_hits_building(self):
        """Projectile impacts a building (damage rule)."""
        w = create_city_world()
        w.add(_circle("bullet", 4, 0, r=0.1, vx=-20, layer="projectile"))
        w.add(_box("bld", 0, 0, hw=3, hd=3, layer="building", static=True))
        results = w.check_all()
        assert len(results) == 1
        assert results[0].impact_speed > 0

    def test_many_entities_performance(self):
        """Smoke test: 200 entities don't crash or take forever."""
        w = create_city_world(cell_size=5)
        import random
        random.seed(42)
        for i in range(200):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            w.add(_circle(f"e{i}", x, y, r=1, layer="pedestrian"))
        results = w.check_all()
        w.resolve(results)
        # Just verify it completes without error.

    def test_raycast_through_city(self):
        """Raycast should hit the nearest building, not a farther one."""
        w = create_city_world()
        w.add(_box("bld_near", 10, 0, hw=2, hd=5, layer="building", static=True))
        w.add(_box("bld_far", 30, 0, hw=2, hd=5, layer="building", static=True))
        hit = w.raycast((0, 0), (1, 0), 50)
        assert hit is not None
        assert hit.entity_b == "bld_near"

    def test_full_tick_cycle(self):
        """Add, check, resolve, update positions — full cycle."""
        w = create_city_world()
        w.add(_circle("car", 0, 0, r=2, vx=10, layer="car"))
        w.add(_circle("ped", 8, 0, r=0.5, layer="pedestrian"))
        # Simulate a few ticks.
        dt = 0.1
        for _ in range(5):
            # Move entities.
            for eid, col in w.colliders.items():
                if not col.is_static:
                    new_x = col.position[0] + col.velocity[0] * dt
                    new_y = col.position[1] + col.velocity[1] * dt
                    w.update(eid, position=(new_x, new_y))
            results = w.check_all()
            if results:
                w.resolve(results)


# ===========================================================================
# create_city_world convenience
# ===========================================================================

class TestCreateCityWorld:
    def test_returns_world(self):
        w = create_city_world()
        assert isinstance(w, CollisionWorld)

    def test_default_rules_loaded(self):
        w = create_city_world()
        assert w.get_rule("car", "pedestrian") == "damage"

    def test_custom_cell_size(self):
        w = create_city_world(cell_size=20)
        assert w._grid.cell_size == 20

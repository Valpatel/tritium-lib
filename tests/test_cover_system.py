# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for CoverSystem — cover objects, damage reduction, directional cover."""

import math

import pytest

from tritium_lib.sim_engine.world.cover import CoverObject, CoverSystem


# ---------------------------------------------------------------------------
# Fake targets and event bus for testing
# ---------------------------------------------------------------------------

class _FakeTarget:
    def __init__(self, target_id: str, position: tuple, status: str = "active"):
        self.target_id = target_id
        self.position = position
        self.status = status


class _FakeEventBus:
    def __init__(self):
        self.published = []

    def publish(self, event_type: str, data: dict):
        self.published.append((event_type, data))


# ---------------------------------------------------------------------------
# CoverObject
# ---------------------------------------------------------------------------

class TestCoverObject:
    def test_default_values(self):
        co = CoverObject(position=(10.0, 20.0))
        assert co.position == (10.0, 20.0)
        assert co.radius == 2.0
        assert co.cover_value == 0.5

    def test_custom_values(self):
        co = CoverObject(position=(0, 0), radius=5.0, cover_value=0.8)
        assert co.radius == 5.0
        assert co.cover_value == 0.8


# ---------------------------------------------------------------------------
# CoverSystem — basic operations
# ---------------------------------------------------------------------------

class TestCoverSystemBasic:
    def test_construction(self):
        cs = CoverSystem()
        assert cs.get_cover_reduction("any") == 0.0

    def test_add_cover(self):
        cs = CoverSystem()
        co = CoverObject(position=(0, 0), radius=5.0)
        cs.add_cover(co)
        assert len(cs._cover_objects) == 1

    def test_add_cover_point(self):
        cs = CoverSystem()
        cp = cs.add_cover_point((50, 50))
        assert isinstance(cp, CoverObject)
        assert cp.position == (50, 50)
        # Should be in both _cover_points and _cover_objects
        assert cp in cs._cover_points
        assert cp in cs._cover_objects

    def test_clear_cover(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0, 0)))
        cs.add_cover(CoverObject(position=(10, 10)))
        cs.clear_cover()
        assert len(cs._cover_objects) == 0

    def test_reset(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0, 0)))
        cs.add_cover_point((10, 10))
        cs.reset()
        assert len(cs._cover_objects) == 0
        assert len(cs._cover_points) == 0
        assert len(cs._assignments) == 0
        assert len(cs._unit_cover) == 0


# ---------------------------------------------------------------------------
# CoverSystem — tick and damage reduction
# ---------------------------------------------------------------------------

class TestCoverTick:
    def test_unit_near_cover_gets_bonus(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(10.0, 10.0), radius=5.0, cover_value=0.6))
        t = _FakeTarget("u1", position=(10.0, 10.0))
        cs.tick(0.1, {"u1": t})
        bonus = cs.get_cover_reduction("u1")
        assert bonus > 0.0
        assert bonus <= 0.8

    def test_unit_at_center_gets_maximum_bonus(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(10.0, 10.0), radius=5.0, cover_value=0.6))
        t = _FakeTarget("u1", position=(10.0, 10.0))
        cs.tick(0.1, {"u1": t})
        bonus = cs.get_cover_reduction("u1")
        assert bonus == pytest.approx(0.6, abs=0.01)

    def test_unit_far_from_cover_gets_no_bonus(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0.0, 0.0), radius=5.0))
        t = _FakeTarget("u1", position=(100.0, 100.0))
        cs.tick(0.1, {"u1": t})
        bonus = cs.get_cover_reduction("u1")
        assert bonus == 0.0

    def test_cover_bonus_capped_at_08(self):
        cs = CoverSystem()
        # Add very high-value cover
        cs.add_cover(CoverObject(position=(0, 0), radius=10.0, cover_value=1.0))
        t = _FakeTarget("u1", position=(0, 0))
        cs.tick(0.1, {"u1": t})
        bonus = cs.get_cover_reduction("u1")
        assert bonus <= 0.8

    def test_eliminated_units_skipped(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0, 0), radius=5.0))
        t = _FakeTarget("u1", position=(0, 0), status="eliminated")
        cs.tick(0.1, {"u1": t})
        # Dead unit shouldn't get cover
        assert cs.get_cover_reduction("u1") == 0.0

    def test_destroyed_units_skipped(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0, 0), radius=5.0))
        t = _FakeTarget("u1", position=(0, 0), status="destroyed")
        cs.tick(0.1, {"u1": t})
        assert cs.get_cover_reduction("u1") == 0.0

    def test_proximity_falloff(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0.0, 0.0), radius=10.0, cover_value=0.6))
        # At center
        t_center = _FakeTarget("u1", position=(0.0, 0.0))
        # At edge
        t_edge = _FakeTarget("u2", position=(9.0, 0.0))
        cs.tick(0.1, {"u1": t_center, "u2": t_edge})
        # Center should have more cover than edge
        assert cs.get_cover_reduction("u1") > cs.get_cover_reduction("u2")

    def test_best_cover_selected(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0, 0), radius=5.0, cover_value=0.3))
        cs.add_cover(CoverObject(position=(0, 0), radius=5.0, cover_value=0.7))
        t = _FakeTarget("u1", position=(0, 0))
        cs.tick(0.1, {"u1": t})
        # Should pick the better cover
        assert cs.get_cover_reduction("u1") >= 0.7

    def test_multiple_units(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0, 0), radius=5.0, cover_value=0.5))
        t1 = _FakeTarget("u1", position=(0, 0))
        t2 = _FakeTarget("u2", position=(100, 100))
        cs.tick(0.1, {"u1": t1, "u2": t2})
        assert cs.get_cover_reduction("u1") > 0.0
        assert cs.get_cover_reduction("u2") == 0.0


# ---------------------------------------------------------------------------
# CoverSystem — directional cover bonus
# ---------------------------------------------------------------------------

class TestDirectionalCoverBonus:
    def test_cover_between_target_and_attacker(self):
        cs = CoverSystem()
        # Cover at (3, 0) with radius 5, target at (0,0) is 3m from cover center (within radius)
        # Attacker at (10, 0) — cover is between target and attacker
        cs.add_cover(CoverObject(position=(3, 0), radius=5.0, cover_value=0.6))
        bonus = cs.get_cover_bonus(
            target_pos=(0, 0),
            attacker_pos=(10, 0),
        )
        assert bonus > 0.0

    def test_cover_behind_target_no_bonus(self):
        cs = CoverSystem()
        # Cover is at (-5, 0), behind target relative to attacker at (10, 0)
        cs.add_cover(CoverObject(position=(-5, 0), radius=3.0, cover_value=0.6))
        bonus = cs.get_cover_bonus(
            target_pos=(0, 0),
            attacker_pos=(10, 0),
        )
        # Cover is behind target, should give no/minimal bonus
        assert bonus == 0.0

    def test_cover_far_away_no_bonus(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(100, 100), radius=2.0, cover_value=0.6))
        bonus = cs.get_cover_bonus(
            target_pos=(0, 0),
            attacker_pos=(10, 0),
        )
        assert bonus == 0.0

    def test_cached_cover_used_with_target_id(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0, 0), radius=5.0, cover_value=0.5))
        t = _FakeTarget("u1", position=(0, 0))
        cs.tick(0.1, {"u1": t})
        # When target_id is provided and cached, use cached value
        bonus = cs.get_cover_bonus(
            target_pos=(0, 0),
            attacker_pos=(10, 0),
            target_id="u1",
        )
        assert bonus > 0.0

    def test_cover_bonus_capped(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(3, 0), radius=10.0, cover_value=1.0))
        bonus = cs.get_cover_bonus(
            target_pos=(0, 0),
            attacker_pos=(10, 0),
        )
        assert bonus <= 0.8


# ---------------------------------------------------------------------------
# CoverSystem — remove_unit
# ---------------------------------------------------------------------------

class TestCoverRemoveUnit:
    def test_remove_clears_cover(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0, 0), radius=5.0))
        t = _FakeTarget("u1", position=(0, 0))
        cs.tick(0.1, {"u1": t})
        assert cs.get_cover_reduction("u1") > 0.0
        cs.remove_unit("u1")
        assert cs.get_cover_reduction("u1") == 0.0

    def test_remove_nonexistent_unit(self):
        cs = CoverSystem()
        # Should not raise
        cs.remove_unit("nonexistent")


# ---------------------------------------------------------------------------
# CoverSystem — event bus integration
# ---------------------------------------------------------------------------

class TestCoverEventBus:
    def test_publish_cover_state(self):
        bus = _FakeEventBus()
        cs = CoverSystem(event_bus=bus)
        cs.add_cover(CoverObject(position=(10, 20), radius=3.0, cover_value=0.5))
        cs.add_cover(CoverObject(position=(30, 40), radius=4.0, cover_value=0.7))
        cs.publish_cover_state()
        assert len(bus.published) == 1
        event_type, data = bus.published[0]
        assert event_type == "cover_points"
        assert len(data["points"]) == 2

    def test_publish_without_event_bus(self):
        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(0, 0)))
        # Should not raise
        cs.publish_cover_state()

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for UnitMissionSystem — backstories, missions, patrol routes."""

import math

import pytest

from tritium_lib.sim_engine.behavior.unit_missions import (
    UnitMissionSystem,
    _perimeter_patrol,
    _grid_sweep,
    _random_patrol,
    _sector_scout,
)


# ---------------------------------------------------------------------------
# Minimal SimulationTarget stub for testing
# ---------------------------------------------------------------------------

class _FakeTarget:
    """Lightweight stand-in for SimulationTarget."""

    def __init__(
        self,
        target_id: str = "t1",
        name: str = "Unit Alpha",
        asset_type: str = "rover",
        alliance: str = "friendly",
        position: tuple = (0.0, 0.0),
        speed: float = 5.0,
        status: str = "active",
        is_combatant: bool = True,
    ):
        self.target_id = target_id
        self.name = name
        self.asset_type = asset_type
        self.alliance = alliance
        self.position = position
        self.speed = speed
        self.status = status
        self.is_combatant = is_combatant
        self.waypoints = []
        self._waypoint_index = 0


# ---------------------------------------------------------------------------
# Patrol route generators
# ---------------------------------------------------------------------------

class TestPerimeterPatrol:
    def test_generates_correct_count(self):
        wps = _perimeter_patrol((0, 0), radius=40.0, points=6)
        assert len(wps) == 6

    def test_points_at_correct_radius(self):
        wps = _perimeter_patrol((10, 20), radius=50.0, points=8)
        for x, y in wps:
            dist = math.hypot(x - 10, y - 20)
            assert dist == pytest.approx(50.0, abs=0.01)

    def test_single_point(self):
        wps = _perimeter_patrol((0, 0), radius=10.0, points=1)
        assert len(wps) == 1

    def test_center_offset(self):
        wps = _perimeter_patrol((100, 200), radius=30.0, points=4)
        assert len(wps) == 4
        # First point should be at angle=0 -> (center_x + radius, center_y)
        assert wps[0] == pytest.approx((130.0, 200.0), abs=0.01)


class TestGridSweep:
    def test_generates_waypoints(self):
        wps = _grid_sweep((0, 0), size=30.0, step=15.0)
        assert len(wps) > 0

    def test_zigzag_pattern(self):
        wps = _grid_sweep((0, 0), size=30.0, step=15.0)
        # Should have pairs of left-right or right-left points
        assert len(wps) >= 4  # At least 2 rows

    def test_center_matters(self):
        wps_a = _grid_sweep((0, 0), size=20.0)
        wps_b = _grid_sweep((100, 100), size=20.0)
        # Different centers should produce different waypoints
        assert wps_a[0] != wps_b[0]


class TestRandomPatrol:
    def test_correct_count(self):
        wps = _random_patrol(bounds=80.0, points=4)
        assert len(wps) == 4

    def test_within_bounds(self):
        for _ in range(10):
            wps = _random_patrol(bounds=50.0, points=6)
            for x, y in wps:
                assert -50.0 <= x <= 50.0
                assert -50.0 <= y <= 50.0


class TestSectorScout:
    def test_correct_count(self):
        # 3 forward points + 1 return to start = 4
        wps = _sector_scout((0, 0), direction=90, range_=50.0)
        assert len(wps) == 4

    def test_returns_to_start(self):
        start = (10, 20)
        wps = _sector_scout(start, direction=45, range_=30.0)
        assert wps[-1] == start

    def test_points_move_outward(self):
        wps = _sector_scout((0, 0), direction=0, range_=60.0)
        # The first 3 points should be progressively farther from origin
        dists = [math.hypot(x, y) for x, y in wps[:3]]
        assert dists[0] < dists[1] < dists[2]


# ---------------------------------------------------------------------------
# UnitMissionSystem — init and basics
# ---------------------------------------------------------------------------

class TestUnitMissionSystemBasic:
    def test_construction(self):
        ums = UnitMissionSystem(map_bounds=200.0)
        assert ums._map_bounds == 200.0

    def test_get_mission_none_initially(self):
        ums = UnitMissionSystem()
        assert ums.get_mission("nonexistent") is None

    def test_get_backstory_none_initially(self):
        ums = UnitMissionSystem()
        assert ums.get_backstory("nonexistent") is None

    def test_reset_clears_all(self):
        ums = UnitMissionSystem()
        t = _FakeTarget()
        ums.assign_starter_mission(t)
        ums.generate_backstory_scripted(t)
        ums.reset()
        assert ums.get_mission(t.target_id) is None
        assert ums.get_backstory(t.target_id) is None

    def test_remove_unit(self):
        ums = UnitMissionSystem()
        t = _FakeTarget()
        ums.assign_starter_mission(t)
        ums.generate_backstory_scripted(t)
        ums.remove_unit(t.target_id)
        assert ums.get_mission(t.target_id) is None
        assert ums.get_backstory(t.target_id) is None


# ---------------------------------------------------------------------------
# Friendly mission assignment
# ---------------------------------------------------------------------------

class TestFriendlyMissions:
    def test_stationary_gets_hold_mission(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(asset_type="turret", speed=0)
        m = ums.assign_starter_mission(t)
        assert m["type"] == "hold"
        assert "position" in m

    def test_rover_gets_mission_with_type(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(asset_type="rover", speed=5.0)
        m = ums.assign_starter_mission(t)
        assert m["type"] in ("patrol", "escort")
        assert "description" in m

    def test_drone_gets_scouting_mission(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(asset_type="drone", speed=10.0)
        m = ums.assign_starter_mission(t)
        assert m["type"] in ("scout", "sweep")
        assert "description" in m

    def test_mission_stored(self):
        ums = UnitMissionSystem()
        t = _FakeTarget()
        m = ums.assign_starter_mission(t)
        assert ums.get_mission(t.target_id) == m

    def test_mobile_unit_gets_waypoints(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(asset_type="rover", speed=5.0)
        m = ums.assign_starter_mission(t)
        # Mobile unit missions should have waypoints (patrol/escort have them)
        if m["type"] in ("patrol", "escort", "scout", "sweep"):
            assert "waypoints" in m
            assert len(m["waypoints"]) > 0


# ---------------------------------------------------------------------------
# Hostile mission assignment
# ---------------------------------------------------------------------------

class TestHostileMissions:
    def test_hostile_gets_mission(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(alliance="hostile", asset_type="person")
        m = ums.assign_starter_mission(t)
        assert m["type"] in ("assault", "infiltrate", "scout", "advance")
        assert "description" in m


# ---------------------------------------------------------------------------
# Neutral mission assignment
# ---------------------------------------------------------------------------

class TestNeutralMissions:
    def test_neutral_person_gets_civilian_mission(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(alliance="neutral", asset_type="person")
        m = ums.assign_starter_mission(t)
        assert m["type"] in ("commute", "walk", "errand", "wander")

    def test_neutral_vehicle_gets_mission(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(alliance="neutral", asset_type="vehicle")
        m = ums.assign_starter_mission(t)
        assert m["type"] in ("commute", "errand")

    def test_neutral_animal_wanders(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(alliance="neutral", asset_type="animal")
        m = ums.assign_starter_mission(t)
        assert m["type"] == "wander"


# ---------------------------------------------------------------------------
# Backstory generation
# ---------------------------------------------------------------------------

class TestBackstoryGeneration:
    def test_scripted_backstory_friendly(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(asset_type="rover", alliance="friendly")
        story = ums.generate_backstory_scripted(t)
        assert isinstance(story, str)
        assert len(story) > 10
        assert ums.get_backstory(t.target_id) == story

    def test_scripted_backstory_hostile(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(asset_type="person", alliance="hostile")
        story = ums.generate_backstory_scripted(t)
        assert isinstance(story, str)
        assert len(story) > 10

    def test_scripted_backstory_neutral(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(asset_type="person", alliance="neutral")
        story = ums.generate_backstory_scripted(t)
        assert isinstance(story, str)
        assert len(story) > 10

    def test_scripted_backstory_unknown_type_fallback(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(asset_type="unknown_thing", alliance="friendly")
        story = ums.generate_backstory_scripted(t)
        assert isinstance(story, str)
        assert len(story) > 0

    def test_backstory_name_interpolation(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(name="Bravo-7", asset_type="rover", alliance="friendly")
        story = ums.generate_backstory_scripted(t)
        assert "Bravo-7" in story


# ---------------------------------------------------------------------------
# LLM prompt builders
# ---------------------------------------------------------------------------

class TestPromptBuilders:
    def test_build_backstory_prompt(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(name="Unit X", alliance="friendly", asset_type="drone",
                        position=(100.0, 200.0))
        prompt = ums.build_backstory_prompt(t)
        assert "friendly" in prompt
        assert "drone" in prompt
        assert "Unit X" in prompt

    def test_build_scenario_prompt(self):
        ums = UnitMissionSystem()
        prompt = ums.build_scenario_prompt(wave=3, total_waves=10, score=500)
        assert "3" in prompt
        assert "10" in prompt
        assert "500" in prompt

    def test_scenario_prompt_custom_context(self):
        ums = UnitMissionSystem()
        prompt = ums.build_scenario_prompt(context="Downtown skirmish")
        assert "Downtown skirmish" in prompt


# ---------------------------------------------------------------------------
# LLM backstory request
# ---------------------------------------------------------------------------

class TestLLMBackstoryRequest:
    def test_request_llm_backstory_assigns_scripted_fallback(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(asset_type="rover", alliance="friendly")
        ums.request_llm_backstory(t)
        # Should have a scripted backstory as fallback
        assert ums.get_backstory(t.target_id) is not None
        assert t.target_id in ums._pending_backstories

    def test_request_llm_backstory_no_generator(self):
        ums = UnitMissionSystem()
        t = _FakeTarget()
        # Should not crash even without a generator
        ums.request_llm_backstory(t)
        assert ums.get_backstory(t.target_id) is not None


# ---------------------------------------------------------------------------
# Tick (idle unit reassignment)
# ---------------------------------------------------------------------------

class TestTickIdleReassignment:
    def test_tick_assigns_mission_to_idle_friendly(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(
            target_id="rover1",
            asset_type="rover",
            alliance="friendly",
            is_combatant=True,
            speed=5.0,
        )
        t.waypoints = []  # idle
        targets = {"rover1": t}
        # Force idle check to run by setting last check long ago
        ums._last_idle_check = 0.0
        ums.tick(0.1, targets)
        # Should have a mission assigned
        assert ums.get_mission("rover1") is not None

    def test_tick_skips_eliminated_units(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(status="eliminated")
        targets = {"t1": t}
        ums._last_idle_check = 0.0
        ums.tick(0.1, targets)
        assert ums.get_mission("t1") is None

    def test_tick_respects_interval(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(alliance="friendly", is_combatant=True)
        t.waypoints = []
        targets = {"t1": t}
        # Set last check to very recent
        import time
        ums._last_idle_check = time.monotonic()
        ums.tick(0.01, targets)
        # Should NOT have assigned a mission because interval not elapsed
        assert ums.get_mission("t1") is None


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------

class TestRouterIntegration:
    def test_set_router(self):
        ums = UnitMissionSystem()
        called = []

        def fake_router(start, end, unit_type, alliance):
            called.append((start, end))
            return [start, end]

        ums.set_router(fake_router)
        assert ums._router is not None

    def test_route_mission_waypoints_without_router(self):
        ums = UnitMissionSystem()
        t = _FakeTarget(position=(0, 0))
        wps = [(10, 10), (20, 20)]
        result = ums._route_mission_waypoints(t, wps)
        # Without router, should return raw waypoints
        assert result == wps

    def test_route_mission_waypoints_with_router(self):
        ums = UnitMissionSystem()

        def fake_router(start, end, unit_type, alliance):
            return [start, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2), end]

        ums.set_router(fake_router)
        t = _FakeTarget(position=(0, 0))
        wps = [(10, 10), (20, 20)]
        result = ums._route_mission_waypoints(t, wps)
        # Router should add intermediate waypoints
        assert len(result) > len(wps)

    def test_route_mission_waypoints_empty_input(self):
        ums = UnitMissionSystem()
        ums.set_router(lambda *a: [])
        t = _FakeTarget(position=(0, 0))
        result = ums._route_mission_waypoints(t, [])
        assert result == []

    def test_route_mission_waypoints_router_exception(self):
        ums = UnitMissionSystem()

        def bad_router(start, end, unit_type, alliance):
            raise ValueError("Route failed")

        ums.set_router(bad_router)
        t = _FakeTarget(position=(0, 0))
        wps = [(10, 10)]
        result = ums._route_mission_waypoints(t, wps)
        # Should gracefully fall back
        assert (10, 10) in result

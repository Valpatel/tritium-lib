# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Integration tests verifying AI modules are wired into the game demo.

Tests that pathfinding, strategy, and behavior profiles are initialized
in build_full_game and produce output in game_tick frames.
"""

import pytest

from tritium_lib.sim_engine.demos.game_server import build_full_game, game_tick


@pytest.fixture(scope="module")
def game_state():
    """Build a full game once for all tests in this module."""
    return build_full_game()


@pytest.fixture(scope="module")
def frames(game_state):
    """Run 35 ticks and collect all frames."""
    collected = []
    for _ in range(35):
        collected.append(game_tick(game_state))
    return collected


# ---------------------------------------------------------------------------
# Road network (pathfinding) integration
# ---------------------------------------------------------------------------


class TestRoadNetworkIntegration:
    """Verify RoadNetwork is initialized and exported in the frame."""

    def test_road_network_initialized(self, game_state):
        assert game_state.road_network is not None

    def test_road_network_has_nodes(self, game_state):
        assert game_state.road_network.node_count >= 8

    def test_road_network_in_first_frame(self, frames):
        assert "road_network" in frames[0]
        data = frames[0]["road_network"]
        assert data["node_count"] >= 8
        assert len(data["nodes"]) >= 8

    def test_road_network_not_in_later_frames(self, frames):
        # Road network is only sent on tick 1
        assert "road_network" not in frames[1]

    def test_astar_finds_path_through_intersection(self, game_state):
        """A* should find a path through the origin intersection."""
        path = game_state.road_network.find_path((-250.0, 0.0), (70.0, 70.0))
        assert len(path) >= 2, "A* should find a multi-hop path"
        # Path should go through (0, 0) as an intermediate
        has_origin = any(abs(p[0]) < 1 and abs(p[1]) < 1 for p in path)
        assert has_origin, "Path should route through the origin intersection"

    def test_astar_path_ns_to_ew(self, game_state):
        """A* should route from a north-south road node to an east-west node."""
        path = game_state.road_network.find_path((0.0, -250.0), (250.0, 0.0))
        assert len(path) >= 3, "Path through two roads requires 3+ waypoints"


# ---------------------------------------------------------------------------
# Walkable area (pedestrian navigation) integration
# ---------------------------------------------------------------------------


class TestWalkableAreaIntegration:
    """Verify WalkableArea is initialized with building obstacles."""

    def test_walkable_area_initialized(self, game_state):
        assert game_state.walkable_area is not None

    def test_walkable_area_has_obstacles(self, game_state):
        assert len(game_state.walkable_area.obstacles) >= 4

    def test_open_area_is_walkable(self, game_state):
        assert game_state.walkable_area.is_walkable((-200.0, -200.0))

    def test_building_center_not_walkable(self, game_state):
        """Building at (0, 0) should block walking."""
        # Building is 20x15 centered at (0, 0)
        assert not game_state.walkable_area.is_walkable((0.0, 0.0))

    def test_pedestrian_path_avoids_buildings(self, game_state):
        """Path should route around building obstacles."""
        path = game_state.walkable_area.find_path((-50.0, -50.0), (50.0, 50.0))
        assert len(path) >= 2
        # No waypoint should be inside a building
        for wp in path:
            for obs in game_state.walkable_area.obstacles:
                from tritium_lib.geo import point_in_polygon
                assert not point_in_polygon(wp[0], wp[1], obs), (
                    f"Waypoint {wp} is inside an obstacle"
                )


# ---------------------------------------------------------------------------
# Strategic AI integration
# ---------------------------------------------------------------------------


class TestStrategicAIIntegration:
    """Verify StrategicAI produces faction-level plans in the frame."""

    def test_strategic_ai_friendly_initialized(self, game_state):
        assert game_state.strategic_ai_friendly is not None

    def test_strategic_ai_hostile_initialized(self, game_state):
        assert game_state.strategic_ai_hostile is not None

    def test_strategy_in_frame(self, frames):
        """Strategy should appear after tick 31 (runs every 30 ticks)."""
        # Tick 31 is at index 30
        assert "strategy" in frames[30]
        strategy = frames[30]["strategy"]
        assert "friendly" in strategy
        assert "hostile" in strategy

    def test_friendly_strategy_has_required_fields(self, frames):
        strategy = frames[30]["strategy"]["friendly"]
        assert "goal" in strategy
        assert "confidence" in strategy
        assert "reasoning" in strategy
        assert "priority" in strategy
        assert strategy["goal"] in (
            "attack", "defend", "flank", "encircle", "retreat",
            "reinforce", "probe", "ambush", "siege", "patrol",
        )

    def test_hostile_strategy_defensive(self, frames):
        """Hostile starts outnumbered (5 vs 11), should defend."""
        strategy = frames[30]["strategy"]["hostile"]
        assert strategy["goal"] in ("defend", "retreat"), (
            f"Outnumbered hostile should defend or retreat, got {strategy['goal']}"
        )

    def test_friendly_strategy_offensive(self, frames):
        """Friendly has numerical superiority, should attack or flank."""
        strategy = frames[30]["strategy"]["friendly"]
        assert strategy["goal"] in ("attack", "flank", "encircle"), (
            f"Friendly with 2:1 ratio should attack/flank, got {strategy['goal']}"
        )

    def test_strategy_confidence_range(self, frames):
        for faction in ("friendly", "hostile"):
            conf = frames[30]["strategy"][faction]["confidence"]
            assert 0.0 <= conf <= 1.0

    def test_plan_stored_on_game_state(self, game_state):
        assert game_state.current_plan_friendly is not None
        assert game_state.current_plan_hostile is not None


# ---------------------------------------------------------------------------
# Behavior profiles integration
# ---------------------------------------------------------------------------


class TestBehaviorProfilesIntegration:
    """Verify BehaviorEngine assigns profiles and produces decisions."""

    def test_behavior_engine_initialized(self, game_state):
        assert game_state.behavior_engine is not None

    def test_all_units_have_profiles(self, game_state):
        for uid in game_state.world.units:
            profile = game_state.behavior_engine.get_profile(uid)
            assert profile is not None, f"Unit {uid} has no behavior profile"

    def test_sniper_has_sniper_profile(self, game_state):
        from tritium_lib.sim_engine.units import UnitType
        for uid, unit in game_state.world.units.items():
            if unit.unit_type == UnitType.SNIPER:
                profile = game_state.behavior_engine.get_profile(uid)
                assert profile.profile_id == "sniper_patient"

    def test_medic_has_medic_profile(self, game_state):
        from tritium_lib.sim_engine.units import UnitType
        for uid, unit in game_state.world.units.items():
            if unit.unit_type == UnitType.MEDIC:
                profile = game_state.behavior_engine.get_profile(uid)
                assert profile.profile_id == "medic_angel"

    def test_behavior_profiles_in_frame(self, frames):
        """Behavior profiles should appear on tick 10 (every 10 ticks)."""
        # Tick 10 is at index 9
        assert "behavior_profiles" in frames[9]
        profiles = frames[9]["behavior_profiles"]
        assert len(profiles) > 0

    def test_behavior_profile_decision_has_fields(self, frames):
        profiles = frames[9]["behavior_profiles"]
        first_uid = list(profiles.keys())[0]
        decision = profiles[first_uid]
        assert "action" in decision
        assert "reasoning" in decision
        assert "profile" in decision

    def test_behavior_actions_are_valid(self, frames):
        profiles = frames[9]["behavior_profiles"]
        valid_actions = {
            "retreat", "seek_cover", "conserve", "rout", "support",
            "advance", "patrol", "assault", "hold_and_engage", "flank",
            "overwatch", "bound_advance", "engage", "hold",
        }
        for uid, decision in profiles.items():
            assert decision["action"] in valid_actions, (
                f"Unit {uid} got invalid action: {decision['action']}"
            )

    def test_profile_count_matches_units(self, game_state):
        """All alive units should have behavior profile assignments."""
        alive_count = sum(1 for u in game_state.world.units.values() if u.is_alive())
        assigned_count = len(game_state.behavior_engine.unit_profiles)
        # Every unit (alive or dead) should have a profile assigned
        assert assigned_count == len(game_state.world.units)
        # Alive count should be substantial
        assert alive_count >= 5


# ---------------------------------------------------------------------------
# Cross-system integration
# ---------------------------------------------------------------------------


class TestAICrossSystems:
    """Verify the AI modules work together in the game loop."""

    def test_30_ticks_no_crash(self, frames):
        """All 35 ticks should complete without error."""
        assert len(frames) == 35

    def test_ai_behaviors_still_present(self, frames):
        """Original AI behaviors (behavior tree, steering) still work."""
        assert "ai_behaviors" in frames[0]

    def test_frame_has_all_ai_keys(self, frames):
        """After enough ticks, the frame should have all AI data."""
        # Gather keys across all frames
        all_keys = set()
        for f in frames:
            all_keys.update(f.keys())
        assert "ai_behaviors" in all_keys
        assert "strategy" in all_keys
        assert "behavior_profiles" in all_keys
        assert "road_network" in all_keys

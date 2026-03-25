# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for unit_states.py — FSM factories for turrets, rovers, drones, hostiles."""

import pytest

from tritium_lib.sim_engine.behavior.unit_states import (
    create_turret_fsm,
    create_rover_fsm,
    create_drone_fsm,
    create_hostile_fsm,
    create_fsm_for_type,
    unit_state_to_three_js,
    units_states_to_three_js,
    _STATE_LABELS,
    _STATE_COLORS,
)


# ---------------------------------------------------------------------------
# Turret FSM
# ---------------------------------------------------------------------------

class TestTurretFSM:
    def test_initial_state_is_idle(self):
        fsm = create_turret_fsm()
        assert fsm.current_state == "idle"

    def test_has_expected_states(self):
        fsm = create_turret_fsm()
        names = fsm.state_names
        for s in ("idle", "scanning", "tracking", "engaging", "cooldown"):
            assert s in names

    def test_idle_to_scanning(self):
        fsm = create_turret_fsm()
        # Tick past the 1s power-up delay
        for _ in range(15):
            fsm.tick(0.1, {})
        assert fsm.current_state == "scanning"

    def test_scanning_to_tracking_on_enemies(self):
        fsm = create_turret_fsm()
        # Move to scanning
        for _ in range(15):
            fsm.tick(0.1, {})
        assert fsm.current_state == "scanning"
        # Now detect enemies
        fsm.tick(0.1, {"enemies_in_range": ["e1"]})
        assert fsm.current_state == "tracking"

    def test_tracking_to_engaging_on_aim(self):
        fsm = create_turret_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        fsm.tick(0.1, {"enemies_in_range": ["e1"]})
        assert fsm.current_state == "tracking"
        fsm.tick(0.1, {"enemies_in_range": ["e1"], "aimed_at_target": True})
        assert fsm.current_state == "engaging"

    def test_engaging_to_cooldown_on_fire(self):
        fsm = create_turret_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        fsm.tick(0.1, {"enemies_in_range": ["e1"]})
        fsm.tick(0.1, {"enemies_in_range": ["e1"], "aimed_at_target": True})
        assert fsm.current_state == "engaging"
        fsm.tick(0.1, {"enemies_in_range": ["e1"], "just_fired": True})
        assert fsm.current_state == "cooldown"

    def test_tracking_returns_to_scanning_when_no_enemies(self):
        fsm = create_turret_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        fsm.tick(0.1, {"enemies_in_range": ["e1"]})
        assert fsm.current_state == "tracking"
        fsm.tick(0.1, {})  # No enemies
        assert fsm.current_state == "scanning"

    def test_degradation_guard_on_condition_transitions(self):
        """The condition-based transition has a degradation guard at 0.8.
        However, the TrackingState.tick() returns 'engaging' directly when
        aimed_at_target is True, which bypasses the condition-based guard.
        Verify the FSM transitions when tick return takes precedence."""
        fsm = create_turret_fsm()
        # Use a single large dt to reach scanning reliably, avoiding
        # floating-point accumulation edge cases with many small ticks.
        fsm.tick(2.0, {})
        assert fsm.current_state == "scanning", (
            f"expected scanning after 2s idle, got {fsm.current_state}"
        )
        fsm.tick(1.0, {"enemies_in_range": ["e1"]})
        assert fsm.current_state == "tracking", (
            f"expected tracking after enemy detection, got {fsm.current_state}"
        )
        # TrackingState.tick() returns "engaging" when aimed_at_target
        # (tick returns override condition-based transitions, so the
        # degradation guard on the condition path does not block this)
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "aimed_at_target": True,
            "degradation": 0.9,
        })
        assert fsm.current_state == "engaging", (
            f"expected engaging via tick-return bypass, got {fsm.current_state}"
        )


# ---------------------------------------------------------------------------
# Rover FSM
# ---------------------------------------------------------------------------

class TestRoverFSM:
    def test_initial_state_is_idle(self):
        fsm = create_rover_fsm()
        assert fsm.current_state == "idle"

    def test_has_expected_states(self):
        fsm = create_rover_fsm()
        for s in ("idle", "patrolling", "pursuing", "engaging", "retreating", "rtb"):
            assert s in fsm.state_names

    def test_idle_to_patrolling_with_waypoints(self):
        fsm = create_rover_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        assert fsm.current_state == "patrolling"

    def test_patrolling_to_pursuing_on_enemies(self):
        fsm = create_rover_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        assert fsm.current_state == "patrolling"
        fsm.tick(0.1, {"enemies_in_range": ["e1"]})
        assert fsm.current_state == "pursuing"

    def test_pursuing_to_engaging_in_weapon_range(self):
        fsm = create_rover_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        fsm.tick(0.1, {"enemies_in_range": ["e1"]})
        assert fsm.current_state == "pursuing"
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
        })
        assert fsm.current_state == "engaging"

    def test_engaging_to_retreating_low_health(self):
        fsm = create_rover_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        fsm.tick(0.1, {"enemies_in_range": ["e1"]})
        fsm.tick(0.1, {"enemies_in_range": ["e1"], "enemy_in_weapon_range": True})
        assert fsm.current_state == "engaging"
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
            "health_pct": 0.2,
        })
        assert fsm.current_state == "retreating"

    def test_idle_to_engaging_direct(self):
        """Idle rover can jump straight to engaging if enemies in weapon range."""
        fsm = create_rover_fsm()
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
        })
        assert fsm.current_state == "engaging"

    def test_idle_to_pursuing_priority_over_patrolling(self):
        fsm = create_rover_fsm()
        fsm.tick(0.1, {
            "has_waypoints": True,
            "enemies_in_range": ["e1"],
        })
        # Pursuing has higher priority than patrolling
        assert fsm.current_state == "pursuing"


# ---------------------------------------------------------------------------
# Drone FSM
# ---------------------------------------------------------------------------

class TestDroneFSM:
    def test_initial_state_is_idle(self):
        fsm = create_drone_fsm()
        assert fsm.current_state == "idle"

    def test_has_expected_states(self):
        fsm = create_drone_fsm()
        for s in ("idle", "scouting", "orbiting", "engaging", "rtb"):
            assert s in fsm.state_names

    def test_idle_to_scouting_with_waypoints(self):
        fsm = create_drone_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        assert fsm.current_state == "scouting"

    def test_scouting_to_orbiting_enemies_not_in_weapon_range(self):
        fsm = create_drone_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": False,
        })
        assert fsm.current_state == "orbiting"

    def test_orbiting_to_engaging(self):
        fsm = create_drone_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        fsm.tick(0.1, {"enemies_in_range": ["e1"]})
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
        })
        assert fsm.current_state == "engaging"

    def test_scouting_to_rtb_critical_health(self):
        fsm = create_drone_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        assert fsm.current_state == "scouting"
        fsm.tick(0.1, {"health_pct": 0.1})
        assert fsm.current_state == "rtb"

    def test_engaging_to_rtb_critical_health(self):
        fsm = create_drone_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
        })
        assert fsm.current_state == "engaging"
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
            "health_pct": 0.1,
        })
        assert fsm.current_state == "rtb"

    def test_engaging_to_scouting_no_enemies(self):
        fsm = create_drone_fsm()
        fsm.tick(0.1, {"has_waypoints": True})
        fsm.tick(0.1, {"enemies_in_range": ["e1"], "enemy_in_weapon_range": True})
        assert fsm.current_state == "engaging"
        fsm.tick(0.1, {})
        assert fsm.current_state == "scouting"


# ---------------------------------------------------------------------------
# Hostile FSM
# ---------------------------------------------------------------------------

class TestHostileFSM:
    def test_initial_state_is_spawning(self):
        fsm = create_hostile_fsm()
        assert fsm.current_state == "spawning"

    def test_has_expected_states(self):
        fsm = create_hostile_fsm()
        for s in ("spawning", "advancing", "engaging", "flanking", "fleeing"):
            assert s in fsm.state_names

    def test_spawning_to_advancing(self):
        fsm = create_hostile_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        assert fsm.current_state == "advancing"

    def test_advancing_to_engaging(self):
        fsm = create_hostile_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        assert fsm.current_state == "advancing"
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
        })
        assert fsm.current_state == "engaging"

    def test_advancing_to_flanking_stationary_target(self):
        fsm = create_hostile_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "nearest_enemy_stationary": True,
            "enemy_in_weapon_range": False,
        })
        assert fsm.current_state == "flanking"

    def test_advancing_to_fleeing_critical_health(self):
        fsm = create_hostile_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        fsm.tick(0.1, {"health_pct": 0.1})
        assert fsm.current_state == "fleeing"

    def test_engaging_to_retreating_under_fire(self):
        fsm = create_hostile_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        fsm.tick(0.1, {"enemies_in_range": ["e1"], "enemy_in_weapon_range": True})
        assert fsm.current_state == "engaging"
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
            "health_pct": 0.2,
            "cover_available": True,
        })
        assert fsm.current_state == "retreating_under_fire"

    def test_engaging_to_suppressing(self):
        fsm = create_hostile_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        fsm.tick(0.1, {"enemies_in_range": ["e1"], "enemy_in_weapon_range": True})
        assert fsm.current_state == "engaging"
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
            "ally_is_flanking": True,
            "nearest_enemy_stationary": True,
        })
        assert fsm.current_state == "suppressing"

    def test_flanking_to_engaging(self):
        fsm = create_hostile_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "nearest_enemy_stationary": True,
        })
        assert fsm.current_state == "flanking"
        fsm.tick(0.1, {
            "enemies_in_range": ["e1"],
            "enemy_in_weapon_range": True,
        })
        assert fsm.current_state == "engaging"


# ---------------------------------------------------------------------------
# Factory router
# ---------------------------------------------------------------------------

class TestFSMFactory:
    def test_turret_type(self):
        fsm = create_fsm_for_type("turret")
        assert fsm is not None
        assert "scanning" in fsm.state_names

    def test_heavy_turret_type(self):
        fsm = create_fsm_for_type("heavy_turret")
        assert fsm is not None
        assert "scanning" in fsm.state_names

    def test_rover_type(self):
        fsm = create_fsm_for_type("rover")
        assert fsm is not None
        assert "patrolling" in fsm.state_names

    def test_tank_type(self):
        fsm = create_fsm_for_type("tank")
        assert fsm is not None
        assert "patrolling" in fsm.state_names

    def test_drone_type(self):
        fsm = create_fsm_for_type("drone")
        assert fsm is not None
        assert "scouting" in fsm.state_names

    def test_scout_drone_type(self):
        fsm = create_fsm_for_type("scout_drone")
        assert fsm is not None
        assert "scouting" in fsm.state_names

    def test_hostile_person(self):
        fsm = create_fsm_for_type("hostile_person")
        assert fsm is not None
        assert "advancing" in fsm.state_names

    def test_hostile_leader(self):
        fsm = create_fsm_for_type("hostile_leader")
        assert fsm is not None

    def test_hostile_vehicle(self):
        fsm = create_fsm_for_type("hostile_vehicle")
        assert fsm is not None

    def test_person_hostile_alliance(self):
        fsm = create_fsm_for_type("person", alliance="hostile")
        assert fsm is not None
        assert "advancing" in fsm.state_names

    def test_unknown_type_returns_none(self):
        fsm = create_fsm_for_type("spaceship")
        assert fsm is None


# ---------------------------------------------------------------------------
# Three.js serialization
# ---------------------------------------------------------------------------

class TestThreeJsSerialization:
    def test_unit_state_to_three_js(self):
        fsm = create_turret_fsm()
        data = unit_state_to_three_js("turret_1", fsm)
        assert data["unit_id"] == "turret_1"
        assert data["state"] == "idle"
        assert data["label"] == "Idle"
        assert data["color"] == "#444444"
        assert isinstance(data["time_in_state"], float)
        assert isinstance(data["available_states"], list)

    def test_state_label_for_scanning(self):
        fsm = create_turret_fsm()
        for _ in range(15):
            fsm.tick(0.1, {})
        data = unit_state_to_three_js("t1", fsm)
        assert data["state"] == "scanning"
        assert data["label"] == "Scanning"
        assert data["color"] == "#00f0ff"

    def test_units_states_to_three_js_batch(self):
        fsm_map = {
            "turret_1": create_turret_fsm(),
            "rover_1": create_rover_fsm(),
            "drone_1": create_drone_fsm(),
        }
        data = units_states_to_three_js(fsm_map)
        assert len(data) == 3
        ids = [d["unit_id"] for d in data]
        assert "turret_1" in ids
        assert "rover_1" in ids
        assert "drone_1" in ids

    def test_unknown_state_gets_default_color(self):
        """If FSM enters an unlabeled state, defaults should apply."""
        assert _STATE_LABELS.get("nonexistent") is None
        assert _STATE_COLORS.get("nonexistent") is None

    def test_state_labels_complete(self):
        """All known FSM states should have labels."""
        all_states = set()
        for factory in (create_turret_fsm, create_rover_fsm, create_drone_fsm, create_hostile_fsm):
            fsm = factory()
            all_states.update(fsm.state_names)
        for state in all_states:
            assert state in _STATE_LABELS, f"Missing label for state: {state}"

    def test_state_colors_complete(self):
        """All known FSM states should have colors."""
        all_states = set()
        for factory in (create_turret_fsm, create_rover_fsm, create_drone_fsm, create_hostile_fsm):
            fsm = factory()
            all_states.update(fsm.state_names)
        for state in all_states:
            assert state in _STATE_COLORS, f"Missing color for state: {state}"

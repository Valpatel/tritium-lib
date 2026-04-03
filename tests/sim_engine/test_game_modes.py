# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for game mode variety — battle, civil_unrest, drone_swarm.

Verifies that each game mode type produces different gameplay behavior:
different scoring, different win/loss conditions, different state fields,
and different event publishing.
"""

import time
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

import pytest

from tritium_lib.sim_engine.game.game_mode import (
    GameMode,
    WaveConfig,
    WAVE_CONFIGS,
    InfiniteWaveMode,
)


# ---------------------------------------------------------------------------
# Minimal stubs for duck-typed dependencies
# ---------------------------------------------------------------------------


@dataclass
class _FakeTarget:
    target_id: str
    name: str = "Unit"
    alliance: str = "friendly"
    is_combatant: bool = True
    status: str = "active"
    battery: float = 1.0
    health: float = 100.0
    max_health: float = 100.0
    asset_type: str = "person"
    speed: float = 5.0
    position: tuple[float, float] = (0.0, 0.0)


class _FakeEngine:
    """Minimal engine stub for GameMode."""

    def __init__(self):
        self._targets: dict[str, _FakeTarget] = {}
        self._map_bounds = 100.0
        self.hazard_manager = MagicMock()
        self.stats_tracker = MagicMock()

    def get_targets(self):
        return list(self._targets.values())

    def spawn_hostile(self, direction="random"):
        tid = f"hostile_{len(self._targets)}"
        t = _FakeTarget(target_id=tid, alliance="hostile", status="active")
        self._targets[tid] = t
        return t

    def spawn_hostile_typed(self, asset_type="person", speed=1.0, health=100.0,
                           drone_variant=None):
        return self.spawn_hostile()

    def add_target(self, target):
        self._targets[target.target_id] = target

    def set_map_bounds(self, bounds):
        self._map_bounds = bounds

    def add_friendly(self):
        tid = f"friendly_{len(self._targets)}"
        t = _FakeTarget(target_id=tid, alliance="friendly", status="active")
        self._targets[tid] = t
        return t


class _FakeCombat:
    def reset_streaks(self): pass
    def clear(self): pass


class _FakeEventBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def publish(self, event_name, data):
        self.events.append((event_name, data))


def _build_game_mode(game_mode_type="battle", infinite=False):
    """Create a GameMode with stub dependencies and friendlies already alive."""
    bus = _FakeEventBus()
    engine = _FakeEngine()
    combat = _FakeCombat()

    # Add 3 friendly combatants so defeat checks don't trigger immediately
    for _ in range(3):
        engine.add_friendly()

    gm = GameMode(event_bus=bus, engine=engine, combat_system=combat, infinite=infinite)
    gm.game_mode_type = game_mode_type

    return gm, bus, engine


# ===================================================================
# Battle mode (default)
# ===================================================================


class TestBattleMode:
    """Default battle mode: wave-based, score by eliminations."""

    def test_default_mode_type_is_battle(self):
        gm, _, _ = _build_game_mode()
        assert gm.game_mode_type == "battle"

    def test_state_starts_at_setup(self):
        gm, _, _ = _build_game_mode()
        assert gm.state == "setup"

    def test_begin_war_transitions_to_countdown(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        assert gm.state == "countdown"

    def test_score_increments_on_elimination(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        gm.state = "active"  # skip countdown
        gm.on_target_eliminated("hostile_0")
        assert gm.score == 100
        assert gm.total_eliminations == 1

    def test_get_state_has_battle_fields(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        state = gm.get_state()
        assert state["game_mode_type"] == "battle"
        assert "score" in state
        assert "wave" in state
        assert "total_eliminations" in state

    def test_get_state_no_civil_unrest_fields(self):
        """Battle mode should NOT have civil_unrest-specific fields."""
        gm, _, _ = _build_game_mode()
        state = gm.get_state()
        assert "de_escalation_score" not in state
        assert "civilian_harm_count" not in state
        assert "infrastructure_health" not in state

    def test_wave_configs_exist(self):
        """Default battle mode should have 10 hardcoded wave configs."""
        assert len(WAVE_CONFIGS) == 10

    def test_waves_increase_in_difficulty(self):
        """Later waves should have more hostiles and/or higher multipliers."""
        first = WAVE_CONFIGS[0]
        last = WAVE_CONFIGS[-1]
        assert last.count > first.count
        assert last.health_mult >= first.health_mult

    def test_spawn_directions_vary(self):
        """Waves should use different spawn directions for variety."""
        directions = {w.spawn_direction for w in WAVE_CONFIGS}
        assert len(directions) >= 2, (
            f"Expected varied spawn directions, got {directions}"
        )

    def test_later_waves_have_mixed_composition(self):
        """Waves 3+ should have mixed unit composition (vehicles, leaders)."""
        mixed = [w for w in WAVE_CONFIGS if w.composition is not None]
        assert len(mixed) >= 5, "Most waves should have mixed compositions"

    def test_unit_types_in_compositions(self):
        """Compositions should include more than just 'person'."""
        all_types = set()
        for w in WAVE_CONFIGS:
            if w.composition:
                for asset_type, count in w.composition:
                    all_types.add(asset_type)
        assert "person" in all_types
        assert "hostile_vehicle" in all_types
        assert "hostile_leader" in all_types


# ===================================================================
# Civil Unrest mode
# ===================================================================


class TestCivilUnrestMode:
    """Civil unrest: de-escalation scoring, civilian harm limits."""

    def test_mode_type(self):
        gm, _, _ = _build_game_mode("civil_unrest")
        assert gm.game_mode_type == "civil_unrest"

    def test_get_state_has_civil_unrest_fields(self):
        """Civil unrest state should include de-escalation and harm fields."""
        gm, _, _ = _build_game_mode("civil_unrest")
        state = gm.get_state()
        assert "de_escalation_score" in state
        assert "civilian_harm_count" in state
        assert "civilian_harm_limit" in state
        assert "weighted_total_score" in state

    def test_civilian_harm_increases_count(self):
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.on_civilian_harmed()
        assert gm.civilian_harm_count == 1

    def test_civilian_harm_reduces_de_escalation_score(self):
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.de_escalation_score = 1000
        gm.on_civilian_harmed()
        assert gm.de_escalation_score == 500  # -500

    def test_excessive_force_causes_defeat(self):
        """Harming too many civilians should trigger defeat."""
        gm, bus, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.civilian_harm_limit = 3  # low limit for testing

        for _ in range(3):
            gm.on_civilian_harmed()

        assert gm.state == "defeat"
        # Check game_over event was published
        game_over_events = [e for name, e in bus.events if name == "game_over"]
        assert len(game_over_events) >= 1
        assert game_over_events[0]["reason"] == "excessive_force"

    def test_weighted_score_formula(self):
        """Weighted score should be 30% combat + 70% de-escalation."""
        gm, _, _ = _build_game_mode("civil_unrest")
        gm.score = 1000
        gm.de_escalation_score = 2000
        state = gm.get_state()
        expected = int(1000 * 0.3 + 2000 * 0.7)
        assert state["weighted_total_score"] == expected

    def test_game_over_data_includes_civil_unrest_fields(self):
        """Game over data in civil_unrest should include mode-specific fields."""
        gm, bus, _ = _build_game_mode("civil_unrest")
        gm.begin_war()
        gm.state = "active"
        gm.de_escalation_score = 500
        gm.civilian_harm_limit = 1  # trigger defeat on first harm
        gm.on_civilian_harmed()

        game_over_events = [e for name, e in bus.events if name == "game_over"]
        assert len(game_over_events) >= 1
        data = game_over_events[0]
        assert "de_escalation_score" in data
        assert "civilian_harm_count" in data
        assert "weighted_total_score" in data


# ===================================================================
# Drone Swarm mode
# ===================================================================


class TestDroneSwarmMode:
    """Drone swarm: infrastructure defense, health-based defeat."""

    def test_mode_type(self):
        gm, _, _ = _build_game_mode("drone_swarm")
        assert gm.game_mode_type == "drone_swarm"

    def test_get_state_has_infrastructure_fields(self):
        """Drone swarm state should include infrastructure health."""
        gm, _, _ = _build_game_mode("drone_swarm")
        gm.infrastructure_health = 500.0
        gm.infrastructure_max = 1000.0
        state = gm.get_state()
        assert "infrastructure_health" in state
        assert "infrastructure_max" in state

    def test_infrastructure_damage_reduces_health(self):
        gm, _, _ = _build_game_mode("drone_swarm")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 100.0
        gm.infrastructure_max = 100.0
        gm.on_infrastructure_damaged(30.0)
        assert gm.infrastructure_health == pytest.approx(70.0)

    def test_infrastructure_destruction_causes_defeat(self):
        """Infrastructure reaching 0 health should trigger defeat."""
        gm, bus, _ = _build_game_mode("drone_swarm")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 50.0
        gm.infrastructure_max = 100.0

        gm.on_infrastructure_damaged(60.0)  # drops to 0

        assert gm.state == "defeat"
        game_over_events = [e for name, e in bus.events if name == "game_over"]
        assert len(game_over_events) >= 1
        assert game_over_events[0]["reason"] == "infrastructure_destroyed"

    def test_infrastructure_health_clamped_at_zero(self):
        gm, _, _ = _build_game_mode("drone_swarm")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 10.0
        gm.on_infrastructure_damaged(100.0)
        assert gm.infrastructure_health == 0.0

    def test_game_over_data_includes_infrastructure(self):
        gm, bus, _ = _build_game_mode("drone_swarm")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 5.0
        gm.on_infrastructure_damaged(10.0)

        game_over_events = [e for name, e in bus.events if name == "game_over"]
        assert len(game_over_events) >= 1
        data = game_over_events[0]
        assert "infrastructure_health" in data
        assert "infrastructure_max" in data


# ===================================================================
# Infinite mode
# ===================================================================


class TestInfiniteMode:
    """Verify infinite mode generates procedural waves beyond wave 10."""

    def test_infinite_flag_set(self):
        gm, _, _ = _build_game_mode(infinite=True)
        assert gm.infinite is True

    def test_infinite_wave_generator_exists(self):
        gm, _, _ = _build_game_mode(infinite=True)
        assert gm._infinite_wave_mode is not None

    def test_wave_11_generated(self):
        """Beyond wave 10, infinite mode should generate wave configs."""
        iwm = InfiniteWaveMode()
        config = iwm.get_wave_config(11)
        assert config is not None
        assert config.count > 0
        assert config.speed_mult > 1.0
        assert config.name  # should have a name

    def test_wave_count_grows(self):
        """Higher wave numbers should have more hostiles."""
        iwm = InfiniteWaveMode()
        c11 = iwm.get_wave_config(11)
        c30 = iwm.get_wave_config(30)
        assert c30.count > c11.count

    def test_wave_21_has_boss(self):
        """Wave 21 is the first boss wave (>20, every 5 after that)."""
        iwm = InfiniteWaveMode()
        c21 = iwm.get_wave_config(21)
        assert c21.has_boss is True
        assert "BOSS" in c21.name

    def test_wave_10_has_elites(self):
        """Wave 10+ should have elite enemies."""
        iwm = InfiniteWaveMode()
        c15 = iwm.get_wave_config(15)
        assert c15.has_elites is True

    def test_infinite_get_state_total_waves_minus_one(self):
        """In infinite mode, total_waves should be -1."""
        gm, _, _ = _build_game_mode(infinite=True)
        state = gm.get_state()
        assert state["total_waves"] == -1

    def test_score_multiplier_grows(self):
        """Higher waves should have higher score multipliers."""
        iwm = InfiniteWaveMode()
        c11 = iwm.get_wave_config(11)
        c50 = iwm.get_wave_config(50)
        assert c50.score_mult >= c11.score_mult


# ===================================================================
# Cross-mode comparison
# ===================================================================


class TestCrossModeComparison:
    """Verify modes produce meaningfully different gameplay."""

    def test_each_mode_has_unique_state_fields(self):
        """Each mode should export different state keys."""
        states = {}
        for mode in ("battle", "civil_unrest", "drone_swarm"):
            gm, _, _ = _build_game_mode(mode)
            states[mode] = set(gm.get_state().keys())

        # Civil unrest has de_escalation_score, battle doesn't
        assert "de_escalation_score" in states["civil_unrest"]
        assert "de_escalation_score" not in states["battle"]
        assert "de_escalation_score" not in states["drone_swarm"]

        # Drone swarm has infrastructure, battle doesn't
        gm_drone, _, _ = _build_game_mode("drone_swarm")
        gm_drone.infrastructure_health = 100.0
        drone_state = gm_drone.get_state()
        assert "infrastructure_health" in drone_state

    def test_modes_share_common_fields(self):
        """All modes should have common fields like score, wave, state."""
        common = {"state", "wave", "score", "game_mode_type"}
        for mode in ("battle", "civil_unrest", "drone_swarm"):
            gm, _, _ = _build_game_mode(mode)
            state_keys = set(gm.get_state().keys())
            assert common.issubset(state_keys), (
                f"Mode '{mode}' missing common fields: {common - state_keys}"
            )

    def test_reset_clears_mode_to_battle(self):
        """Resetting should return mode_type to 'battle'."""
        for mode in ("civil_unrest", "drone_swarm"):
            gm, _, _ = _build_game_mode(mode)
            gm.begin_war()
            gm.reset()
            assert gm.game_mode_type == "battle"
            assert gm.state == "setup"

    def test_civilian_harm_only_matters_in_civil_unrest(self):
        """Civilian harm events should be ignored in battle mode."""
        gm, bus, _ = _build_game_mode("battle")
        gm.begin_war()
        gm.state = "active"
        gm.on_civilian_harmed()
        # In battle mode, civilian_harm_count still increments (field exists)
        # but it should NOT cause defeat
        assert gm.state == "active"

    def test_infrastructure_damage_only_matters_in_drone_swarm(self):
        """Infrastructure damage should not cause defeat in battle mode."""
        gm, _, _ = _build_game_mode("battle")
        gm.begin_war()
        gm.state = "active"
        gm.infrastructure_health = 50.0
        gm.on_infrastructure_damaged(100.0)
        # In battle mode this still reduces the counter but doesn't end game
        assert gm.state == "active"


# ===================================================================
# Difficulty scaler
# ===================================================================


class TestDifficultyScaler:
    """Verify the adaptive difficulty system works within game modes."""

    def test_difficulty_initialized(self):
        gm, _, _ = _build_game_mode()
        assert gm.difficulty is not None

    def test_difficulty_multiplier_starts_at_one(self):
        gm, _, _ = _build_game_mode()
        assert gm.difficulty.get_multiplier() == pytest.approx(1.0)

    def test_difficulty_in_state(self):
        """Game state should include difficulty_multiplier."""
        gm, _, _ = _build_game_mode()
        state = gm.get_state()
        assert "difficulty_multiplier" in state

    def test_reset_resets_difficulty(self):
        gm, _, _ = _build_game_mode()
        gm.begin_war()
        gm.reset()
        assert gm.difficulty.get_multiplier() == pytest.approx(1.0)

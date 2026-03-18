# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.game — game mode, ambient, crowd,
stats, difficulty, and morale systems."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tritium_lib.sim_engine.game.game_mode import (
    GameMode,
    WaveConfig,
    WAVE_CONFIGS,
    InfiniteWaveMode,
    InstigatorDetector,
)
from tritium_lib.sim_engine.game.difficulty import DifficultyScaler, WaveRecord
from tritium_lib.sim_engine.game.morale import (
    MoraleSystem,
    DEFAULT_MORALE,
    BROKEN_THRESHOLD,
    SUPPRESSED_THRESHOLD,
    EMBOLDENED_THRESHOLD,
)
from tritium_lib.sim_engine.game.stats import StatsTracker, UnitStats, WaveStats
from tritium_lib.sim_engine.game.crowd_density import CrowdDensityTracker
from tritium_lib.sim_engine.game.ambient import (
    AmbientSpawner,
    _generate_street_grid,
    _hour_activity,
)


# ---------------------------------------------------------------------------
# Test helpers — lightweight mocks for engine, event_bus, combat_system
# ---------------------------------------------------------------------------

class FakeEventBus:
    """Records all published events."""
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def publish(self, event_name: str, data: dict) -> None:
        self.events.append((event_name, data))


class FakeCombatSystem:
    def reset_streaks(self):
        pass

    def clear(self):
        pass


class FakeTarget:
    def __init__(self, target_id="t1", alliance="friendly", status="active",
                 is_combatant=True, asset_type="person", position=(0, 0),
                 speed=1.0, health=100.0, max_health=100.0, battery=1.0,
                 weapon_range=0.0, crowd_role="", identified=False, name="Test"):
        self.target_id = target_id
        self.alliance = alliance
        self.status = status
        self.is_combatant = is_combatant
        self.asset_type = asset_type
        self.position = position
        self.speed = speed
        self.health = health
        self.max_health = max_health
        self.battery = battery
        self.weapon_range = weapon_range
        self.crowd_role = crowd_role
        self.identified = identified
        self.name = name


class FakeEngine:
    def __init__(self, targets=None):
        self._targets = targets or []
        self.spawners_paused = False
        self._map_bounds = 200.0

    def get_targets(self):
        return self._targets

    def add_target(self, t):
        self._targets.append(t)

    def spawn_hostile(self, direction="random"):
        t = FakeTarget(
            target_id=f"hostile_{len(self._targets)}",
            alliance="hostile",
            status="active",
            is_combatant=True,
            health=50.0,
            max_health=50.0,
            speed=2.0,
        )
        self._targets.append(t)
        return t

    def spawn_hostile_typed(self, asset_type="person", speed=None, health=None,
                            direction="random", drone_variant=None):
        t = FakeTarget(
            target_id=f"hostile_{len(self._targets)}",
            alliance="hostile",
            status="active",
            is_combatant=True,
            asset_type=asset_type,
            health=health or 50.0,
            max_health=health or 50.0,
            speed=speed or 2.0,
        )
        self._targets.append(t)
        return t

    def set_map_bounds(self, b):
        self._map_bounds = b


# ---------------------------------------------------------------------------
# WaveConfig tests
# ---------------------------------------------------------------------------

class TestWaveConfig:
    def test_basic_creation(self):
        wc = WaveConfig(name="Test", count=5, speed_mult=1.0, health_mult=1.0)
        assert wc.name == "Test"
        assert wc.count == 5
        assert wc.speed_mult == 1.0
        assert wc.composition is None
        assert wc.spawn_direction == "random"

    def test_composition(self):
        wc = WaveConfig(
            name="Mixed",
            count=10,
            speed_mult=1.0,
            health_mult=1.0,
            composition=[("person", 7), ("hostile_vehicle", 3)],
        )
        assert wc.composition is not None
        assert sum(c for _, c in wc.composition) == 10

    def test_wave_configs_list(self):
        assert len(WAVE_CONFIGS) == 10
        assert WAVE_CONFIGS[0].name == "Scout Party"
        assert WAVE_CONFIGS[-1].name == "FINAL STAND"

    def test_infinite_fields(self):
        wc = WaveConfig(
            name="Boss", count=20, speed_mult=2.0, health_mult=3.0,
            has_elites=True, elite_count=3, has_boss=True,
            boss_health_mult=5.0, score_mult=2.5,
        )
        assert wc.has_boss is True
        assert wc.elite_count == 3


# ---------------------------------------------------------------------------
# GameMode state machine tests
# ---------------------------------------------------------------------------

class TestGameModeStateMachine:
    def _make_game_mode(self, friendlies=None):
        bus = FakeEventBus()
        targets = friendlies or [
            FakeTarget(target_id="f1", alliance="friendly", is_combatant=True),
        ]
        engine = FakeEngine(targets=targets)
        combat = FakeCombatSystem()
        gm = GameMode(event_bus=bus, engine=engine, combat_system=combat)
        return gm, bus, engine

    def test_initial_state_is_setup(self):
        gm, _, _ = self._make_game_mode()
        assert gm.state == "setup"
        assert gm.wave == 0
        assert gm.score == 0

    def test_begin_war_transitions_to_countdown(self):
        gm, bus, _ = self._make_game_mode()
        gm.begin_war()
        assert gm.state == "countdown"
        assert gm.wave == 1
        # Should have published game_state_change
        assert any(e[0] == "game_state_change" for e in bus.events)

    def test_begin_war_only_from_setup(self):
        gm, _, _ = self._make_game_mode()
        gm.state = "active"
        gm.begin_war()
        assert gm.state == "active"  # unchanged

    def test_countdown_ticks_to_active(self):
        gm, bus, _ = self._make_game_mode()
        gm.begin_war()
        # Tick through the 5s countdown
        for _ in range(60):  # 6 seconds at 0.1s ticks
            gm.tick(0.1)
        assert gm.state == "active"

    def test_reset_returns_to_setup(self):
        gm, _, _ = self._make_game_mode()
        gm.begin_war()
        gm.tick(10.0)  # get to active
        gm.reset()
        assert gm.state == "setup"
        assert gm.wave == 0
        assert gm.score == 0

    def test_on_target_eliminated_scores_points(self):
        gm, _, _ = self._make_game_mode()
        gm.state = "active"
        gm._wave_hostile_ids.add("h1")
        gm.on_target_eliminated("h1")
        assert gm.total_eliminations == 1
        assert gm.score == 100
        assert gm.wave_eliminations == 1

    def test_on_target_eliminated_non_wave_hostile(self):
        gm, _, _ = self._make_game_mode()
        gm.state = "active"
        gm.on_target_eliminated("random_hostile")
        assert gm.total_eliminations == 1
        assert gm.wave_eliminations == 0  # not in _wave_hostile_ids

    def test_get_state_returns_dict(self):
        gm, _, _ = self._make_game_mode()
        state = gm.get_state()
        assert isinstance(state, dict)
        assert state["state"] == "setup"
        assert state["wave"] == 0
        assert "score" in state
        assert "infinite" in state

    def test_defeat_on_civilian_harm(self):
        gm, bus, _ = self._make_game_mode()
        gm.state = "active"
        gm.game_mode_type = "civil_unrest"
        gm.civilian_harm_limit = 2
        gm.on_civilian_harmed()
        assert gm.state == "active"  # first harm, not at limit
        gm.on_civilian_harmed()
        assert gm.state == "defeat"
        # game_over event published
        game_over_events = [e for e in bus.events if e[0] == "game_over"]
        assert len(game_over_events) == 1
        assert game_over_events[0][1]["reason"] == "excessive_force"

    def test_defeat_on_infrastructure_destroyed(self):
        gm, bus, _ = self._make_game_mode()
        gm.state = "active"
        gm.game_mode_type = "drone_swarm"
        gm.infrastructure_health = 100.0
        gm.on_infrastructure_damaged(100.0)
        assert gm.state == "defeat"


# ---------------------------------------------------------------------------
# DifficultyScaler tests
# ---------------------------------------------------------------------------

class TestDifficultyScaler:
    def test_initial_multiplier_is_1(self):
        ds = DifficultyScaler()
        assert ds.get_multiplier() == 1.0

    def test_perfect_wave_increases_difficulty(self):
        ds = DifficultyScaler()
        ds.record_wave({
            "eliminations": 10,
            "hostiles_spawned": 10,
            "wave_time": 15.0,
            "friendly_damage_taken": 0.0,
            "friendly_max_health": 100.0,
            "escapes": 0,
        })
        assert ds.get_multiplier() > 1.0

    def test_terrible_wave_decreases_difficulty(self):
        ds = DifficultyScaler()
        ds.record_wave({
            "eliminations": 0,
            "hostiles_spawned": 10,
            "wave_time": 90.0,
            "friendly_damage_taken": 100.0,
            "friendly_max_health": 100.0,
            "escapes": 10,
        })
        assert ds.get_multiplier() < 1.0

    def test_multiplier_clamped(self):
        ds = DifficultyScaler()
        # Drive multiplier up repeatedly
        for _ in range(100):
            ds.record_wave({
                "eliminations": 10, "hostiles_spawned": 10,
                "wave_time": 5.0, "friendly_damage_taken": 0.0,
                "friendly_max_health": 100.0, "escapes": 0,
            })
        assert ds.get_multiplier() <= 2.0

        ds2 = DifficultyScaler()
        for _ in range(100):
            ds2.record_wave({
                "eliminations": 0, "hostiles_spawned": 10,
                "wave_time": 120.0, "friendly_damage_taken": 100.0,
                "friendly_max_health": 100.0, "escapes": 10,
            })
        assert ds2.get_multiplier() >= 0.5

    def test_wave_adjustments_at_default(self):
        ds = DifficultyScaler()
        adj = ds.get_wave_adjustments(10)
        assert adj["hostile_count"] == 10
        assert adj["hostile_health_bonus"] == 0.0
        assert adj["hardened"] is False
        assert adj["easy"] is False

    def test_reset(self):
        ds = DifficultyScaler()
        ds.record_wave({
            "eliminations": 10, "hostiles_spawned": 10,
            "wave_time": 10.0, "friendly_damage_taken": 0.0,
            "friendly_max_health": 100.0, "escapes": 0,
        })
        ds.reset()
        assert ds.get_multiplier() == 1.0
        assert len(ds.wave_history) == 0


# ---------------------------------------------------------------------------
# InfiniteWaveMode tests
# ---------------------------------------------------------------------------

class TestInfiniteWaveMode:
    def test_wave_1(self):
        iwm = InfiniteWaveMode()
        wc = iwm.get_wave_config(1)
        assert wc.count >= 1
        assert wc.speed_mult > 1.0
        assert wc.name.startswith("Wave")

    def test_count_grows_with_wave(self):
        iwm = InfiniteWaveMode()
        wc5 = iwm.get_wave_config(5)
        wc50 = iwm.get_wave_config(50)
        assert wc50.count > wc5.count

    def test_boss_waves(self):
        iwm = InfiniteWaveMode()
        wc21 = iwm.get_wave_config(21)
        assert wc21.has_boss is True
        assert "BOSS" in wc21.name

    def test_elite_waves(self):
        iwm = InfiniteWaveMode()
        wc15 = iwm.get_wave_config(15)
        assert wc15.has_elites is True
        assert wc15.elite_count >= 1


# ---------------------------------------------------------------------------
# MoraleSystem tests
# ---------------------------------------------------------------------------

class TestMoraleSystem:
    def test_default_morale_is_1(self):
        ms = MoraleSystem()
        assert ms.get_morale("unknown_unit") == 1.0

    def test_set_and_get(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.5)
        assert ms.get_morale("u1") == 0.5

    def test_clamp_to_bounds(self):
        ms = MoraleSystem()
        ms.set_morale("u1", -0.5)
        assert ms.get_morale("u1") == 0.0
        ms.set_morale("u1", 1.5)
        assert ms.get_morale("u1") == 1.0

    def test_damage_reduces_morale(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.7)
        ms.on_damage_taken("u1", 100.0)
        assert ms.get_morale("u1") < 0.7

    def test_ally_eliminated_reduces_morale(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.7)
        ms.on_ally_eliminated("u1")
        assert ms.get_morale("u1") == pytest.approx(0.55, abs=0.01)

    def test_enemy_eliminated_boosts_morale(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.5)
        ms.on_enemy_eliminated("u1")
        assert ms.get_morale("u1") == pytest.approx(0.6, abs=0.01)

    def test_broken_threshold(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.05)
        assert ms.is_broken("u1") is True
        ms.set_morale("u1", 0.2)
        assert ms.is_broken("u1") is False

    def test_suppressed_threshold(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.2)
        assert ms.is_suppressed("u1") is True

    def test_emboldened_threshold(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.95)
        assert ms.is_emboldened("u1") is True

    def test_recovery_tick(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.3)
        # No recent hit, should recover
        target = FakeTarget(target_id="u1", status="active")
        ms.tick(1.0, {"u1": target})
        assert ms.get_morale("u1") > 0.3

    def test_no_recovery_when_recently_hit(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.3)
        ms.on_damage_taken("u1", 10.0)  # sets last hit time to now
        target = FakeTarget(target_id="u1", status="active")
        ms.tick(0.1, {"u1": target})
        # Should NOT have recovered because hit was < 3s ago
        assert ms.get_morale("u1") <= 0.3

    def test_reset_clears_all(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.5)
        ms.reset()
        assert ms.get_morale("u1") == 1.0  # default

    def test_remove_unit(self):
        ms = MoraleSystem()
        ms.set_morale("u1", 0.5)
        ms.remove_unit("u1")
        assert ms.get_morale("u1") == 1.0  # default


# ---------------------------------------------------------------------------
# StatsTracker tests
# ---------------------------------------------------------------------------

class TestStatsTracker:
    def test_register_and_get(self):
        st = StatsTracker()
        st.register_unit("u1", "Alpha", "friendly", "rover")
        stats = st.get_unit_stats("u1")
        assert stats is not None
        assert stats.name == "Alpha"

    def test_record_shot_and_hit(self):
        st = StatsTracker()
        st.register_unit("u1", "Alpha", "friendly", "rover")
        st.register_unit("h1", "Hostile", "hostile", "person")
        st.on_shot_fired("u1")
        st.on_shot_hit("u1", "h1", 25.0)
        u1 = st.get_unit_stats("u1")
        assert u1.shots_fired == 1
        assert u1.shots_hit == 1
        assert u1.damage_dealt == 25.0
        h1 = st.get_unit_stats("h1")
        assert h1.damage_taken == 25.0

    def test_kill_and_assist(self):
        st = StatsTracker()
        st.register_unit("u1", "Alpha", "friendly", "rover")
        st.register_unit("u2", "Beta", "friendly", "rover")
        st.register_unit("h1", "Hostile", "hostile", "person")
        ts = time.monotonic()
        st.on_shot_hit("u2", "h1", 10.0, timestamp=ts)
        st.on_kill("u1", "h1")
        u1 = st.get_unit_stats("u1")
        u2 = st.get_unit_stats("u2")
        assert u1.kills == 1
        assert u2.assists == 1

    def test_wave_tracking(self):
        st = StatsTracker()
        st.on_wave_start(1, "Scout Party", 5)
        st.on_wave_complete(250)
        waves = st.get_wave_stats()
        assert len(waves) == 1
        assert waves[0].score_earned == 250

    def test_summary(self):
        st = StatsTracker()
        st.register_unit("u1", "Alpha", "friendly", "rover")
        summary = st.get_summary()
        assert "total_kills" in summary
        assert "mvp" in summary
        assert summary["unit_count"] == 1

    def test_reset(self):
        st = StatsTracker()
        st.register_unit("u1", "Alpha", "friendly", "rover")
        st.reset()
        assert st.get_unit_stats("u1") is None

    def test_unit_stats_accuracy(self):
        us = UnitStats(target_id="u1", name="A", alliance="f", asset_type="r")
        assert us.accuracy == 0.0
        us.shots_fired = 10
        us.shots_hit = 7
        assert us.accuracy == pytest.approx(0.7)

    def test_unit_stats_kd_ratio(self):
        us = UnitStats(target_id="u1", name="A", alliance="f", asset_type="r")
        us.kills = 5
        us.deaths = 0
        assert us.kd_ratio == 5.0
        us.deaths = 2
        assert us.kd_ratio == 2.5

    def test_to_dict(self):
        st = StatsTracker()
        st.register_unit("u1", "Alpha", "friendly", "rover")
        d = st.to_dict()
        assert "units" in d
        assert "waves" in d
        assert "summary" in d


# ---------------------------------------------------------------------------
# CrowdDensityTracker tests
# ---------------------------------------------------------------------------

class TestCrowdDensityTracker:
    def test_empty_grid_is_sparse(self):
        cdt = CrowdDensityTracker(bounds=(0, 0, 100, 100))
        assert cdt.get_density_at((50, 50)) == "sparse"

    def test_dense_crowd(self):
        cdt = CrowdDensityTracker(bounds=(0, 0, 100, 100), cell_size=100)
        # Place 8 people in one cell
        targets = {}
        for i in range(8):
            targets[f"p{i}"] = FakeTarget(
                target_id=f"p{i}", asset_type="person",
                status="active", position=(50, 50),
            )
        cdt.tick(targets, 1.0)
        assert cdt.get_density_at((50, 50)) == "dense"
        assert cdt.get_conversion_multiplier((50, 50)) == 2.0

    def test_can_identify_instigator(self):
        cdt = CrowdDensityTracker(bounds=(0, 0, 100, 100), cell_size=100)
        # Sparse -> can identify
        assert cdt.can_identify_instigator((50, 50)) is True

    def test_critical_blocks_identification(self):
        cdt = CrowdDensityTracker(bounds=(0, 0, 100, 100), cell_size=100)
        targets = {}
        for i in range(15):
            targets[f"p{i}"] = FakeTarget(
                target_id=f"p{i}", asset_type="person",
                status="active", position=(50, 50),
            )
        cdt.tick(targets, 1.0)
        assert cdt.get_density_at((50, 50)) == "critical"
        assert cdt.can_identify_instigator((50, 50)) is False

    def test_poi_defeat(self):
        cdt = CrowdDensityTracker(bounds=(0, 0, 100, 100), cell_size=100)
        cdt.add_poi_building((50, 50), "City Hall")
        targets = {}
        for i in range(15):
            targets[f"p{i}"] = FakeTarget(
                target_id=f"p{i}", asset_type="person",
                status="active", position=(50, 50),
            )
        # Tick enough to exceed timeout
        for _ in range(70):
            cdt.tick(targets, 1.0)
        assert cdt.check_poi_defeat(timeout=60.0) is True


# ---------------------------------------------------------------------------
# Ambient helper tests
# ---------------------------------------------------------------------------

class TestAmbientHelpers:
    def test_generate_street_grid(self):
        ns, ew = _generate_street_grid(200.0)
        assert len(ns) > 0
        assert 0.0 in ns
        # Should have streets in both directions
        assert any(x < 0 for x in ns)
        assert any(x > 0 for x in ns)

    def test_hour_activity_returns_tuple(self):
        ambient, hostile = _hour_activity()
        assert 0.0 <= ambient <= 1.0
        assert 0.0 <= hostile <= 1.0


# ---------------------------------------------------------------------------
# Import test — verify re-exports work
# ---------------------------------------------------------------------------

class TestImports:
    def test_game_init_exports(self):
        from tritium_lib.sim_engine.game import (
            GameMode,
            WaveConfig,
            WAVE_CONFIGS,
            InfiniteWaveMode,
            InstigatorDetector,
            AmbientSpawner,
            CrowdDensityTracker,
            StatsTracker,
            UnitStats,
            WaveStats,
            DifficultyScaler,
            WaveRecord,
            MoraleSystem,
            DEFAULT_MORALE,
            BROKEN_THRESHOLD,
            SUPPRESSED_THRESHOLD,
            EMBOLDENED_THRESHOLD,
        )
        # All imports succeeded
        assert GameMode is not None

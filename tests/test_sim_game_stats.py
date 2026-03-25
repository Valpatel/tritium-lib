# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.game.stats — StatsTracker, UnitStats, WaveStats."""

import math
import time
import pytest
from types import SimpleNamespace

from tritium_lib.sim_engine.game.stats import (
    StatsTracker,
    UnitStats,
    WaveStats,
)


# ---------------------------------------------------------------------------
# UnitStats
# ---------------------------------------------------------------------------

class TestUnitStats:
    def test_construction(self):
        us = UnitStats(target_id="u1", name="Rover", alliance="friendly", asset_type="rover")
        assert us.shots_fired == 0
        assert us.kills == 0
        assert us.health_remaining == 0.0

    def test_accuracy_no_shots(self):
        us = UnitStats(target_id="u1", name="T", alliance="friendly", asset_type="rover")
        assert us.accuracy == 0.0

    def test_accuracy_calculation(self):
        us = UnitStats(target_id="u1", name="T", alliance="friendly", asset_type="rover",
                       shots_fired=10, shots_hit=7)
        assert abs(us.accuracy - 0.7) < 0.001

    def test_kd_ratio_no_deaths(self):
        us = UnitStats(target_id="u1", name="T", alliance="friendly", asset_type="rover",
                       kills=5, deaths=0)
        assert us.kd_ratio == 5.0

    def test_kd_ratio_with_deaths(self):
        us = UnitStats(target_id="u1", name="T", alliance="friendly", asset_type="rover",
                       kills=10, deaths=5)
        assert abs(us.kd_ratio - 2.0) < 0.001

    def test_damage_efficiency_no_taken(self):
        us = UnitStats(target_id="u1", name="T", alliance="friendly", asset_type="rover",
                       damage_dealt=100.0, damage_taken=0.0)
        assert us.damage_efficiency == float("inf")

    def test_damage_efficiency_no_dealt_no_taken(self):
        us = UnitStats(target_id="u1", name="T", alliance="friendly", asset_type="rover")
        assert us.damage_efficiency == 0.0

    def test_damage_efficiency_normal(self):
        us = UnitStats(target_id="u1", name="T", alliance="friendly", asset_type="rover",
                       damage_dealt=200.0, damage_taken=100.0)
        assert abs(us.damage_efficiency - 2.0) < 0.001

    def test_to_dict(self):
        us = UnitStats(target_id="u1", name="Rover", alliance="friendly",
                       asset_type="rover", kills=3, shots_fired=10, shots_hit=7)
        d = us.to_dict()
        assert d["target_id"] == "u1"
        assert d["kills"] == 3
        assert d["accuracy"] == 0.7
        assert isinstance(d["kd_ratio"], float)

    def test_to_dict_inf_damage_efficiency(self):
        us = UnitStats(target_id="u1", name="T", alliance="friendly", asset_type="rover",
                       damage_dealt=100.0, damage_taken=0.0)
        d = us.to_dict()
        assert d["damage_efficiency"] == "inf"


# ---------------------------------------------------------------------------
# WaveStats
# ---------------------------------------------------------------------------

class TestWaveStats:
    def test_construction(self):
        ws = WaveStats(wave_number=1, wave_name="Raid")
        assert ws.wave_number == 1
        assert ws.hostiles_spawned == 0
        assert ws.score_earned == 0

    def test_to_dict(self):
        ws = WaveStats(wave_number=2, wave_name="Assault",
                       hostiles_spawned=10, hostiles_eliminated=8,
                       score_earned=500, duration=45.0)
        d = ws.to_dict()
        assert d["wave_number"] == 2
        assert d["hostiles_eliminated"] == 8
        assert d["score_earned"] == 500
        assert d["duration"] == 45.0


# ---------------------------------------------------------------------------
# StatsTracker
# ---------------------------------------------------------------------------

class TestStatsTrackerRegistration:
    def test_register_unit(self):
        st = StatsTracker()
        st.register_unit("u1", "Rover", "friendly", "rover")
        assert st.get_unit_stats("u1") is not None
        assert st.get_unit_stats("u1").name == "Rover"

    def test_register_overwrites(self):
        st = StatsTracker()
        st.register_unit("u1", "Old Name", "friendly", "rover")
        st.register_unit("u1", "New Name", "friendly", "rover")
        assert st.get_unit_stats("u1").name == "New Name"

    def test_unregistered_returns_none(self):
        st = StatsTracker()
        assert st.get_unit_stats("unknown") is None


class TestStatsTrackerCombat:
    def test_record_shot(self):
        st = StatsTracker()
        st.register_unit("u1", "Rover", "friendly", "rover")
        st.record_shot("u1")
        assert st.get_unit_stats("u1").shots_fired == 1

    def test_record_shot_auto_registers(self):
        st = StatsTracker()
        st.record_shot("unknown_unit")
        assert st.get_unit_stats("unknown_unit") is not None
        assert st.get_unit_stats("unknown_unit").shots_fired == 1

    def test_on_shot_fired(self):
        st = StatsTracker()
        st.register_unit("u1", "Rover", "friendly", "rover")
        st.on_shot_fired("u1")
        assert st.get_unit_stats("u1").shots_fired == 1

    def test_on_shot_fired_unregistered_no_crash(self):
        st = StatsTracker()
        st.on_shot_fired("ghost")  # Should not raise

    def test_on_shot_hit(self):
        st = StatsTracker()
        st.register_unit("shooter", "S", "friendly", "rover")
        st.register_unit("target", "T", "hostile", "person")
        st.on_shot_hit("shooter", "target", 25.0)
        assert st.get_unit_stats("shooter").shots_hit == 1
        assert st.get_unit_stats("shooter").damage_dealt == 25.0
        assert st.get_unit_stats("target").damage_taken == 25.0

    def test_on_kill(self):
        st = StatsTracker()
        st.register_unit("killer", "K", "friendly", "rover")
        st.register_unit("victim", "V", "hostile", "person")
        st.on_kill("killer", "victim")
        assert st.get_unit_stats("killer").kills == 1
        assert st.get_unit_stats("victim").deaths == 1

    def test_assist_tracking(self):
        st = StatsTracker()
        st.register_unit("attacker_a", "A", "friendly", "rover")
        st.register_unit("attacker_b", "B", "friendly", "rover")
        st.register_unit("victim", "V", "hostile", "person")
        # A damages the victim
        now = time.monotonic()
        st.on_shot_hit("attacker_a", "victim", 20.0, timestamp=now)
        # B kills the victim
        st.on_kill("attacker_b", "victim")
        assert st.get_unit_stats("attacker_a").assists == 1
        assert st.get_unit_stats("attacker_b").assists == 0

    def test_on_damage_taken(self):
        st = StatsTracker()
        st.register_unit("u1", "U", "friendly", "rover")
        st.on_damage_taken("u1", 30.0)
        assert st.get_unit_stats("u1").damage_taken == 30.0


class TestStatsTrackerWaves:
    def test_wave_lifecycle(self):
        st = StatsTracker()
        st.on_wave_start(1, "Attack", 10)
        assert len(st.get_wave_stats()) == 1
        ws = st.get_wave_stats()[0]
        assert ws.wave_number == 1
        assert ws.hostiles_spawned == 10

    def test_wave_complete_records_score(self):
        st = StatsTracker()
        st.on_wave_start(1, "Attack", 10)
        st.on_wave_complete(500)
        ws = st.get_wave_stats()[0]
        assert ws.score_earned == 500

    def test_wave_complete_without_start_is_safe(self):
        st = StatsTracker()
        st.on_wave_complete(100)  # No wave started — should not raise

    def test_hostile_escaped(self):
        st = StatsTracker()
        st.on_wave_start(1, "Test", 5)
        st.on_hostile_escaped()
        st.on_hostile_escaped()
        ws = st.get_wave_stats()[0]
        assert ws.hostiles_escaped == 2

    def test_friendly_loss(self):
        st = StatsTracker()
        st.on_wave_start(1, "Test", 5)
        st.on_friendly_loss()
        ws = st.get_wave_stats()[0]
        assert ws.friendly_losses == 1

    def test_wave_shots_accumulated(self):
        st = StatsTracker()
        st.register_unit("u1", "U", "friendly", "rover")
        st.on_wave_start(1, "Test", 5)
        st.record_shot("u1")
        st.record_shot("u1")
        ws = st.get_wave_stats()[0]
        assert ws.total_shots_fired == 2


class TestStatsTrackerTick:
    def _make_target(self, tid, x, y, alliance="friendly", status="active",
                     health=100, weapon_range=15.0, speed=1.0):
        return SimpleNamespace(
            target_id=tid, position=(x, y), alliance=alliance,
            status=status, health=health, weapon_range=weapon_range,
            speed=speed,
        )

    def test_tick_updates_time_alive(self):
        st = StatsTracker()
        st.register_unit("u1", "U", "friendly", "rover")
        targets = {"u1": self._make_target("u1", 0, 0)}
        st.tick(0.1, targets)
        assert st.get_unit_stats("u1").time_alive == pytest.approx(0.1, abs=0.01)

    def test_tick_tracks_distance(self):
        st = StatsTracker()
        st.register_unit("u1", "U", "friendly", "rover")
        targets1 = {"u1": self._make_target("u1", 0, 0)}
        st.tick(0.1, targets1)
        targets2 = {"u1": self._make_target("u1", 10, 0)}
        st.tick(0.1, targets2)
        assert st.get_unit_stats("u1").distance_traveled == pytest.approx(10.0, abs=0.1)

    def test_tick_records_health(self):
        st = StatsTracker()
        st.register_unit("u1", "U", "friendly", "rover")
        targets = {"u1": self._make_target("u1", 0, 0, health=75.0)}
        st.tick(0.1, targets)
        assert st.get_unit_stats("u1").health_remaining == 75.0

    def test_tick_no_targets_only_game_elapsed(self):
        st = StatsTracker()
        st.tick(0.5, None)
        assert st._game_elapsed == pytest.approx(0.5, abs=0.01)


class TestStatsTrackerQueries:
    def test_get_all_unit_stats_sorted(self):
        st = StatsTracker()
        st.register_unit("a", "A", "friendly", "rover")
        st.register_unit("b", "B", "friendly", "rover")
        st.get_unit_stats("a").kills = 2
        st.get_unit_stats("b").kills = 5
        all_stats = st.get_all_unit_stats()
        assert all_stats[0].target_id == "b"

    def test_get_mvp(self):
        st = StatsTracker()
        st.register_unit("a", "A", "friendly", "rover")
        st.register_unit("b", "B", "friendly", "rover")
        st.get_unit_stats("a").kills = 10
        st.get_unit_stats("b").kills = 3
        mvp = st.get_mvp()
        assert mvp.target_id == "a"

    def test_get_mvp_empty(self):
        st = StatsTracker()
        assert st.get_mvp() is None

    def test_get_summary(self):
        st = StatsTracker()
        st.register_unit("u1", "U1", "friendly", "rover")
        st.get_unit_stats("u1").kills = 5
        st.get_unit_stats("u1").shots_fired = 20
        st.get_unit_stats("u1").shots_hit = 15
        summary = st.get_summary()
        assert summary["total_kills"] == 5
        assert summary["unit_count"] == 1
        assert summary["mvp"]["target_id"] == "u1"


class TestStatsTrackerLifecycle:
    def test_remove_unit(self):
        st = StatsTracker()
        st.register_unit("u1", "U", "friendly", "rover")
        st.remove_unit("u1")
        assert st.get_unit_stats("u1") is None

    def test_reset(self):
        st = StatsTracker()
        st.register_unit("u1", "U", "friendly", "rover")
        st.on_wave_start(1, "Test", 5)
        st.reset()
        assert st.get_unit_stats("u1") is None
        assert len(st.get_wave_stats()) == 0
        assert st._game_elapsed == 0.0

    def test_to_dict(self):
        st = StatsTracker()
        st.register_unit("u1", "U1", "friendly", "rover")
        d = st.to_dict()
        assert "units" in d
        assert "waves" in d
        assert "summary" in d
        assert len(d["units"]) == 1

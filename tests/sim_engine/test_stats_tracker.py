# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.game.stats — StatsTracker."""

import time

from tritium_lib.sim_engine.game.stats import StatsTracker, UnitStats, WaveStats


class TestUnitStats:
    def test_accuracy_no_shots(self):
        s = UnitStats(target_id="u1", name="Unit", alliance="friendly", asset_type="rover")
        assert s.accuracy == 0.0

    def test_accuracy_calculation(self):
        s = UnitStats(target_id="u1", name="Unit", alliance="friendly", asset_type="rover",
                      shots_fired=10, shots_hit=7)
        assert s.accuracy == 0.7

    def test_kd_ratio_no_deaths(self):
        s = UnitStats(target_id="u1", name="Unit", alliance="friendly", asset_type="rover",
                      kills=5, deaths=0)
        assert s.kd_ratio == 5.0

    def test_kd_ratio_with_deaths(self):
        s = UnitStats(target_id="u1", name="Unit", alliance="friendly", asset_type="rover",
                      kills=6, deaths=2)
        assert s.kd_ratio == 3.0

    def test_damage_efficiency_no_taken(self):
        s = UnitStats(target_id="u1", name="Unit", alliance="friendly", asset_type="rover",
                      damage_dealt=100.0, damage_taken=0.0)
        assert s.damage_efficiency == float("inf")

    def test_damage_efficiency_both_zero(self):
        s = UnitStats(target_id="u1", name="Unit", alliance="friendly", asset_type="rover")
        assert s.damage_efficiency == 0.0

    def test_to_dict(self):
        s = UnitStats(target_id="u1", name="Test", alliance="friendly", asset_type="rover",
                      kills=3, shots_fired=10, shots_hit=5)
        d = s.to_dict()
        assert d["target_id"] == "u1"
        assert d["kills"] == 3
        assert d["accuracy"] == 0.5
        assert "kd_ratio" in d


class TestWaveStats:
    def test_creation(self):
        ws = WaveStats(wave_number=1, wave_name="Scout Party")
        assert ws.wave_number == 1
        assert ws.hostiles_eliminated == 0

    def test_to_dict(self):
        ws = WaveStats(wave_number=1, wave_name="Test", hostiles_spawned=5,
                       hostiles_eliminated=3, score_earned=300)
        d = ws.to_dict()
        assert d["hostiles_spawned"] == 5
        assert d["score_earned"] == 300


class TestStatsTracker:
    def test_register_unit(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover 1", "friendly", "rover")
        stats = tracker.get_unit_stats("r1")
        assert stats is not None
        assert stats.name == "Rover 1"

    def test_record_shot(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover", "friendly", "rover")
        tracker.record_shot("r1")
        assert tracker.get_unit_stats("r1").shots_fired == 1

    def test_record_shot_auto_registers(self):
        tracker = StatsTracker()
        tracker.record_shot("unknown_unit")
        stats = tracker.get_unit_stats("unknown_unit")
        assert stats is not None
        assert stats.shots_fired == 1

    def test_on_shot_hit(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover", "friendly", "rover")
        tracker.register_unit("h1", "Hostile", "hostile", "person")
        tracker.on_shot_hit("r1", "h1", 25.0, timestamp=time.monotonic())
        assert tracker.get_unit_stats("r1").shots_hit == 1
        assert tracker.get_unit_stats("r1").damage_dealt == 25.0
        assert tracker.get_unit_stats("h1").damage_taken == 25.0

    def test_on_kill(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover", "friendly", "rover")
        tracker.register_unit("h1", "Hostile", "hostile", "person")
        tracker.on_kill("r1", "h1")
        assert tracker.get_unit_stats("r1").kills == 1
        assert tracker.get_unit_stats("h1").deaths == 1

    def test_assist_tracking(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover 1", "friendly", "rover")
        tracker.register_unit("r2", "Rover 2", "friendly", "rover")
        tracker.register_unit("h1", "Hostile", "hostile", "person")

        now = time.monotonic()
        # r1 damages h1
        tracker.on_shot_hit("r1", "h1", 10.0, timestamp=now)
        # r2 kills h1 within 5s
        tracker.on_kill("r2", "h1")

        assert tracker.get_unit_stats("r1").assists == 1
        assert tracker.get_unit_stats("r2").kills == 1

    def test_wave_tracking(self):
        tracker = StatsTracker()
        tracker.on_wave_start(1, "Scout Party", 5)
        assert len(tracker.get_wave_stats()) == 1

        tracker.on_wave_complete(500)
        wave = tracker.get_wave_stats()[0]
        assert wave.score_earned == 500
        assert wave.duration > 0

    def test_hostile_escaped(self):
        tracker = StatsTracker()
        tracker.on_wave_start(1, "Test", 3)
        tracker.on_hostile_escaped()
        assert tracker.get_wave_stats()[0].hostiles_escaped == 1

    def test_friendly_loss(self):
        tracker = StatsTracker()
        tracker.on_wave_start(1, "Test", 3)
        tracker.on_friendly_loss()
        assert tracker.get_wave_stats()[0].friendly_losses == 1

    def test_get_mvp(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover 1", "friendly", "rover")
        tracker.register_unit("r2", "Rover 2", "friendly", "rover")
        tracker.get_unit_stats("r1").kills = 5
        tracker.get_unit_stats("r2").kills = 3
        mvp = tracker.get_mvp()
        assert mvp.target_id == "r1"

    def test_get_mvp_empty(self):
        tracker = StatsTracker()
        assert tracker.get_mvp() is None

    def test_get_summary(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover", "friendly", "rover")
        tracker.get_unit_stats("r1").kills = 3
        tracker.get_unit_stats("r1").shots_fired = 10
        tracker.get_unit_stats("r1").shots_hit = 7
        summary = tracker.get_summary()
        assert summary["total_kills"] == 3
        assert summary["unit_count"] == 1
        assert summary["mvp"]["target_id"] == "r1"

    def test_reset(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover", "friendly", "rover")
        tracker.on_wave_start(1, "Test", 5)
        tracker.reset()
        assert len(tracker.get_all_unit_stats()) == 0
        assert len(tracker.get_wave_stats()) == 0

    def test_remove_unit(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover", "friendly", "rover")
        tracker.remove_unit("r1")
        assert tracker.get_unit_stats("r1") is None

    def test_to_dict(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover", "friendly", "rover")
        d = tracker.to_dict()
        assert "units" in d
        assert "waves" in d
        assert "summary" in d

    def test_wave_shots_tracking(self):
        tracker = StatsTracker()
        tracker.register_unit("r1", "Rover", "friendly", "rover")
        tracker.on_wave_start(1, "Test", 5)
        tracker.record_shot("r1")
        tracker.on_shot_hit("r1", "h1", 10.0, timestamp=time.monotonic())
        wave = tracker.get_wave_stats()[0]
        assert wave.total_shots_fired == 1
        assert wave.total_shots_hit == 1
        assert wave.total_damage_dealt == 10.0

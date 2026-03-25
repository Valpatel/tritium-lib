# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.game.difficulty — DifficultyScaler."""

from tritium_lib.sim_engine.game.difficulty import DifficultyScaler


class TestDifficultyScaler:
    def test_initial_multiplier(self):
        ds = DifficultyScaler()
        assert ds.get_multiplier() == 1.0

    def test_good_performance_increases_difficulty(self):
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

    def test_poor_performance_decreases_difficulty(self):
        ds = DifficultyScaler()
        ds.record_wave({
            "eliminations": 1,
            "hostiles_spawned": 10,
            "wave_time": 90.0,
            "friendly_damage_taken": 90.0,
            "friendly_max_health": 100.0,
            "escapes": 8,
        })
        assert ds.get_multiplier() < 1.0

    def test_multiplier_capped_at_max(self):
        ds = DifficultyScaler()
        # Record many perfect waves
        for _ in range(50):
            ds.record_wave({
                "eliminations": 10,
                "hostiles_spawned": 10,
                "wave_time": 5.0,
                "friendly_damage_taken": 0.0,
                "friendly_max_health": 100.0,
                "escapes": 0,
            })
        assert ds.get_multiplier() <= 2.0

    def test_multiplier_capped_at_min(self):
        ds = DifficultyScaler()
        # Record many terrible waves
        for _ in range(50):
            ds.record_wave({
                "eliminations": 0,
                "hostiles_spawned": 10,
                "wave_time": 120.0,
                "friendly_damage_taken": 100.0,
                "friendly_max_health": 100.0,
                "escapes": 10,
            })
        assert ds.get_multiplier() >= 0.5

    def test_wave_adjustments_normal(self):
        ds = DifficultyScaler()
        adj = ds.get_wave_adjustments(10)
        assert adj["hostile_count"] == 10
        assert adj["hardened"] is False
        assert adj["easy"] is False
        assert adj["hostile_health_bonus"] == 0.0

    def test_wave_adjustments_hardened(self):
        ds = DifficultyScaler()
        ds._multiplier = 1.8
        adj = ds.get_wave_adjustments(10)
        assert adj["hardened"] is True
        assert adj["use_cover_seeking"] is True
        assert adj["elite_count"] >= 1
        assert adj["hostile_count"] > 10
        assert adj["hostile_health_bonus"] > 0

    def test_wave_adjustments_easy(self):
        ds = DifficultyScaler()
        ds._multiplier = 0.6
        adj = ds.get_wave_adjustments(10)
        assert adj["easy"] is True
        assert adj["disable_flanking"] is True
        assert adj["speed_reduction"] > 0
        assert adj["hostile_count"] < 10

    def test_hostile_count_minimum_1(self):
        ds = DifficultyScaler()
        ds._multiplier = 0.5
        adj = ds.get_wave_adjustments(1)
        assert adj["hostile_count"] >= 1

    def test_reset(self):
        ds = DifficultyScaler()
        ds.record_wave({
            "eliminations": 10,
            "hostiles_spawned": 10,
            "wave_time": 10.0,
            "friendly_damage_taken": 0.0,
            "friendly_max_health": 100.0,
            "escapes": 0,
        })
        assert ds.get_multiplier() != 1.0
        ds.reset()
        assert ds.get_multiplier() == 1.0
        assert len(ds.wave_history) == 0

    def test_wave_history_tracked(self):
        ds = DifficultyScaler()
        ds.record_wave({
            "eliminations": 5,
            "hostiles_spawned": 10,
            "wave_time": 30.0,
            "friendly_damage_taken": 20.0,
            "friendly_max_health": 100.0,
            "escapes": 2,
        })
        assert len(ds.wave_history) == 1
        assert ds.wave_history[0].elimination_rate == 0.5
        assert ds.wave_history[0].escapes == 2

    def test_zero_hostiles_spawned(self):
        """Edge case: no hostiles spawned should not crash."""
        ds = DifficultyScaler()
        ds.record_wave({
            "eliminations": 0,
            "hostiles_spawned": 0,
            "wave_time": 10.0,
            "friendly_damage_taken": 0.0,
            "friendly_max_health": 100.0,
            "escapes": 0,
        })
        # Should not crash, multiplier should still be valid
        assert 0.5 <= ds.get_multiplier() <= 2.0

    def test_zero_friendly_health(self):
        """Edge case: zero max health should not crash."""
        ds = DifficultyScaler()
        ds.record_wave({
            "eliminations": 5,
            "hostiles_spawned": 10,
            "wave_time": 20.0,
            "friendly_damage_taken": 0.0,
            "friendly_max_health": 0.0,
            "escapes": 0,
        })
        assert 0.5 <= ds.get_multiplier() <= 2.0

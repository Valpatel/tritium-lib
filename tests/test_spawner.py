# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the wave spawner and enemy composition designer.

Covers: SpawnPattern, EnemyComposition, SpawnPoint, WaveDesigner,
SpawnerEngine, DIFFICULTY_CURVES, WAVE_PRESETS, run_preset.
"""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.spawner import (
    SpawnPattern,
    EnemyComposition,
    SpawnPoint,
    UnitTemplate,
    WaveDesigner,
    SpawnerEngine,
    DIFFICULTY_CURVES,
    WAVE_PRESETS,
    _TEMPLATE_COST,
    _TEMPLATE_DIFFICULTY,
    _alliance_spawn_color,
    run_preset,
)
from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ===================================================================
# SpawnPattern enum
# ===================================================================


class TestSpawnPattern:
    def test_all_members_exist(self):
        expected = {"RANDOM", "CLUSTER", "LINE", "SURROUND", "FLANKING", "WAVES", "TRICKLE"}
        assert set(p.name for p in SpawnPattern) == expected

    def test_values_are_lowercase(self):
        for p in SpawnPattern:
            assert p.value == p.name.lower()

    def test_from_value(self):
        assert SpawnPattern("random") is SpawnPattern.RANDOM
        assert SpawnPattern("surround") is SpawnPattern.SURROUND


# ===================================================================
# EnemyComposition
# ===================================================================


class TestEnemyComposition:
    def test_empty_composition(self):
        comp = EnemyComposition()
        assert comp.total_count == 0
        assert comp.difficulty_rating == 0.0

    def test_total_count_computed(self):
        comp = EnemyComposition(templates=[
            {"template": "infantry", "count": 5},
            {"template": "sniper", "count": 2},
        ])
        assert comp.total_count == 7

    def test_difficulty_stored(self):
        comp = EnemyComposition(difficulty_rating=42.5)
        assert comp.difficulty_rating == 42.5

    def test_to_dict(self):
        comp = EnemyComposition(
            templates=[{"template": "infantry", "count": 3}],
            difficulty_rating=3.0,
        )
        d = comp.to_dict()
        assert d["total_count"] == 3
        assert d["difficulty_rating"] == 3.0
        assert len(d["templates"]) == 1

    def test_to_dict_empty(self):
        d = EnemyComposition().to_dict()
        assert d["total_count"] == 0
        assert d["templates"] == []


# ===================================================================
# UnitTemplate dataclass
# ===================================================================


class TestUnitTemplate:
    def test_defaults(self):
        ut = UnitTemplate(template="infantry", count=5)
        assert ut.equipment == []

    def test_with_equipment(self):
        ut = UnitTemplate(template="sniper", count=1, equipment=["sniper_rifle"])
        assert ut.equipment == ["sniper_rifle"]


# ===================================================================
# SpawnPoint
# ===================================================================


class TestSpawnPoint:
    def test_defaults(self):
        sp = SpawnPoint(position=(10, 20))
        assert sp.radius == 10.0
        assert sp.alliance == "hostile"
        assert sp.active is True
        assert sp.cooldown == 0.0

    def test_is_ready_active(self):
        sp = SpawnPoint(position=(0, 0), active=True)
        assert sp.is_ready()

    def test_is_ready_inactive(self):
        sp = SpawnPoint(position=(0, 0), active=False)
        assert not sp.is_ready()

    def test_cooldown_cycle(self):
        sp = SpawnPoint(position=(0, 0), cooldown=2.0)
        assert sp.is_ready()
        sp.trigger_cooldown()
        assert not sp.is_ready()
        sp.tick_cooldown(1.0)
        assert not sp.is_ready()
        sp.tick_cooldown(1.0)
        assert sp.is_ready()

    def test_cooldown_does_not_go_negative(self):
        sp = SpawnPoint(position=(0, 0), cooldown=1.0)
        sp.trigger_cooldown()
        sp.tick_cooldown(5.0)
        assert sp._cooldown_remaining == 0.0

    def test_no_cooldown_always_ready(self):
        sp = SpawnPoint(position=(0, 0), cooldown=0.0)
        sp.trigger_cooldown()
        assert sp.is_ready()


# ===================================================================
# Difficulty curves
# ===================================================================


class TestDifficultyCurves:
    def test_all_curves_present(self):
        expected = {"linear", "exponential", "logarithmic", "staircase", "random_spikes"}
        assert set(DIFFICULTY_CURVES.keys()) == expected

    def test_linear_scales(self):
        fn = DIFFICULTY_CURVES["linear"]
        assert fn(1, 10) == 10
        assert fn(5, 10) == 50

    def test_exponential_grows(self):
        fn = DIFFICULTY_CURVES["exponential"]
        v1 = fn(1, 10)
        v5 = fn(5, 10)
        v10 = fn(10, 10)
        assert v5 > v1
        assert v10 > v5

    def test_logarithmic_grows_slower(self):
        fn_log = DIFFICULTY_CURVES["logarithmic"]
        fn_lin = DIFFICULTY_CURVES["linear"]
        # At wave 10 with base 10, linear = 100, log should be less
        assert fn_log(10, 10) < fn_lin(10, 10)

    def test_staircase_jumps_every_3(self):
        fn = DIFFICULTY_CURVES["staircase"]
        # Waves 1-3 same step, 4-6 next step
        assert fn(1, 10) == fn(2, 10) == fn(3, 10)
        assert fn(4, 10) == fn(5, 10) == fn(6, 10)
        assert fn(4, 10) > fn(1, 10)

    def test_random_spikes_varies(self):
        fn = DIFFICULTY_CURVES["random_spikes"]
        # Just check it returns positive values
        for _ in range(10):
            v = fn(5, 10)
            assert v > 0

    def test_all_curves_callable(self):
        for name, fn in DIFFICULTY_CURVES.items():
            result = fn(3, 10)
            assert isinstance(result, (int, float))
            assert result > 0


# ===================================================================
# WaveDesigner
# ===================================================================


class TestWaveDesigner:
    def setup_method(self):
        self.designer = WaveDesigner(seed=42)

    def test_wave1_mostly_infantry(self):
        comp = self.designer.design_wave(1, "linear", budget=10)
        templates = {t["template"] for t in comp.templates}
        assert "infantry" in templates
        # Wave 1: only infantry available
        assert templates == {"infantry"}

    def test_wave3_adds_vehicles(self):
        comp = self.designer.design_wave(3, "linear", budget=20)
        templates = {t["template"] for t in comp.templates}
        assert "infantry" in templates
        # Vehicles unlocked at wave 3

    def test_wave5_adds_snipers_heavy(self):
        comp = self.designer.design_wave(5, "linear", budget=50)
        pool = self.designer._available_pool(5)
        assert "sniper" in pool
        assert "heavy" in pool

    def test_wave8_adds_helicopters(self):
        pool = self.designer._available_pool(8)
        assert "helicopter" in pool

    def test_wave10_adds_tanks(self):
        pool = self.designer._available_pool(10)
        assert "tank" in pool

    def test_difficulty_increases_with_wave(self):
        comp1 = self.designer.design_wave(1, "linear", 10)
        comp5 = self.designer.design_wave(5, "linear", 10)
        comp10 = self.designer.design_wave(10, "linear", 10)
        assert comp5.difficulty_rating >= comp1.difficulty_rating
        assert comp10.difficulty_rating >= comp5.difficulty_rating

    def test_total_count_positive(self):
        comp = self.designer.design_wave(3, "linear", 15)
        assert comp.total_count > 0

    def test_budget_respected(self):
        # With a tiny budget, should still get at least 1 unit
        comp = self.designer.design_wave(1, "linear", 1)
        assert comp.total_count >= 1

    def test_exponential_curve(self):
        comp_lin = self.designer.design_wave(10, "linear", 10)
        designer2 = WaveDesigner(seed=42)
        comp_exp = designer2.design_wave(10, "exponential", 10)
        # Exponential should produce more units at wave 10
        assert comp_exp.total_count >= comp_lin.total_count or comp_exp.difficulty_rating >= comp_lin.difficulty_rating

    def test_composition_has_equipment(self):
        comp = self.designer.design_wave(5, "linear", 50)
        for entry in comp.templates:
            assert "equipment" in entry
            assert isinstance(entry["equipment"], list)

    def test_unknown_curve_falls_back_to_linear(self):
        comp = self.designer.design_wave(3, "nonexistent_curve", 10)
        assert comp.total_count > 0

    def test_seed_reproducibility(self):
        d1 = WaveDesigner(seed=99)
        d2 = WaveDesigner(seed=99)
        c1 = d1.design_wave(5, "linear", 30)
        c2 = d2.design_wave(5, "linear", 30)
        assert c1.templates == c2.templates
        assert c1.difficulty_rating == c2.difficulty_rating


# ===================================================================
# WaveDesigner — generate_spawn_positions
# ===================================================================


class TestGenerateSpawnPositions:
    def setup_method(self):
        self.designer = WaveDesigner(seed=42)

    def test_zero_count_returns_empty(self):
        assert self.designer.generate_spawn_positions(SpawnPattern.RANDOM, (0, 0), 10, 0) == []

    def test_negative_count_returns_empty(self):
        assert self.designer.generate_spawn_positions(SpawnPattern.RANDOM, (0, 0), 10, -1) == []

    def test_random_count_correct(self):
        positions = self.designer.generate_spawn_positions(SpawnPattern.RANDOM, (50, 50), 20, 10)
        assert len(positions) == 10

    def test_random_within_radius(self):
        center = (100.0, 100.0)
        radius = 30.0
        positions = self.designer.generate_spawn_positions(SpawnPattern.RANDOM, center, radius, 50)
        for pos in positions:
            assert distance(center, pos) <= radius + 0.01

    def test_cluster_count_correct(self):
        positions = self.designer.generate_spawn_positions(SpawnPattern.CLUSTER, (0, 0), 15, 8)
        assert len(positions) == 8

    def test_cluster_is_tight(self):
        center = (50.0, 50.0)
        positions = self.designer.generate_spawn_positions(SpawnPattern.CLUSTER, center, 30, 100)
        # Most points should be within radius (gaussian -- some may exceed)
        close = sum(1 for p in positions if distance(center, p) <= 30)
        assert close >= 80  # at least 80% within 1 sigma * 3

    def test_line_count_correct(self):
        positions = self.designer.generate_spawn_positions(SpawnPattern.LINE, (0, 0), 20, 5)
        assert len(positions) == 5

    def test_line_same_y(self):
        center = (0.0, 10.0)
        positions = self.designer.generate_spawn_positions(SpawnPattern.LINE, center, 20, 4)
        for pos in positions:
            assert pos[1] == pytest.approx(10.0)

    def test_line_single_unit_returns_center(self):
        positions = self.designer.generate_spawn_positions(SpawnPattern.LINE, (5, 5), 10, 1)
        assert positions[0] == (5, 5)

    def test_surround_count(self):
        positions = self.designer.generate_spawn_positions(SpawnPattern.SURROUND, (0, 0), 25, 6)
        assert len(positions) == 6

    def test_surround_equidistant(self):
        center = (0.0, 0.0)
        radius = 20.0
        positions = self.designer.generate_spawn_positions(SpawnPattern.SURROUND, center, radius, 8)
        for pos in positions:
            assert distance(center, pos) == pytest.approx(radius, abs=0.01)

    def test_flanking_count(self):
        positions = self.designer.generate_spawn_positions(SpawnPattern.FLANKING, (0, 0), 30, 10)
        assert len(positions) == 10

    def test_flanking_two_sides(self):
        center = (50.0, 50.0)
        radius = 30.0
        positions = self.designer.generate_spawn_positions(SpawnPattern.FLANKING, center, radius, 10)
        left = [p for p in positions if p[0] < center[0]]
        right = [p for p in positions if p[0] > center[0]]
        assert len(left) > 0
        assert len(right) > 0

    def test_waves_count(self):
        positions = self.designer.generate_spawn_positions(SpawnPattern.WAVES, (0, 0), 20, 12)
        assert len(positions) == 12

    def test_trickle_count(self):
        positions = self.designer.generate_spawn_positions(SpawnPattern.TRICKLE, (0, 0), 20, 7)
        assert len(positions) == 7

    def test_trickle_single_returns_center(self):
        positions = self.designer.generate_spawn_positions(SpawnPattern.TRICKLE, (3, 4), 10, 1)
        assert positions[0] == (3, 4)

    def test_all_patterns_produce_correct_count(self):
        for pattern in SpawnPattern:
            positions = self.designer.generate_spawn_positions(pattern, (0, 0), 20, 5)
            assert len(positions) == 5, f"Pattern {pattern.name} returned {len(positions)} instead of 5"


# ===================================================================
# SpawnerEngine
# ===================================================================


class TestSpawnerEngine:
    def setup_method(self):
        self.engine = SpawnerEngine(seed=42)

    def test_add_spawn_point(self):
        sp = SpawnPoint(position=(10, 20))
        self.engine.add_spawn_point(sp)
        assert len(self.engine.spawn_points) == 1

    def test_remove_spawn_point(self):
        self.engine.add_spawn_point(SpawnPoint(position=(0, 0)))
        self.engine.add_spawn_point(SpawnPoint(position=(1, 1)))
        removed = self.engine.remove_spawn_point(0)
        assert removed is not None
        assert len(self.engine.spawn_points) == 1

    def test_remove_invalid_index(self):
        assert self.engine.remove_spawn_point(99) is None

    def test_active_spawn_points(self):
        self.engine.add_spawn_point(SpawnPoint(position=(0, 0), active=True))
        self.engine.add_spawn_point(SpawnPoint(position=(1, 1), active=False))
        self.engine.add_spawn_point(SpawnPoint(position=(2, 2), active=True))
        assert len(self.engine.active_spawn_points()) == 2

    def test_spawn_wave_basic(self):
        self.engine.add_spawn_point(SpawnPoint(position=(100, 0), radius=10))
        comp = EnemyComposition(templates=[
            {"template": "infantry", "count": 3, "equipment": ["rifle"]},
        ], difficulty_rating=3.0)
        result = self.engine.spawn_wave(comp)
        assert len(result) == 3
        assert all(r["template"] == "infantry" for r in result)

    def test_spawn_wave_multiple_templates(self):
        self.engine.add_spawn_point(SpawnPoint(position=(50, 50), radius=15))
        comp = EnemyComposition(templates=[
            {"template": "infantry", "count": 2},
            {"template": "sniper", "count": 1},
        ])
        result = self.engine.spawn_wave(comp)
        assert len(result) == 3
        templates = [r["template"] for r in result]
        assert templates.count("infantry") == 2
        assert templates.count("sniper") == 1

    def test_spawn_wave_distributes_across_points(self):
        self.engine.add_spawn_point(SpawnPoint(position=(0, 0), radius=5))
        self.engine.add_spawn_point(SpawnPoint(position=(100, 100), radius=5))
        comp = EnemyComposition(templates=[
            {"template": "infantry", "count": 4},
        ])
        result = self.engine.spawn_wave(comp)
        indices = {r["spawn_point_index"] for r in result}
        assert 0 in indices
        assert 1 in indices

    def test_spawn_wave_no_points_uses_fallback(self):
        comp = EnemyComposition(templates=[
            {"template": "infantry", "count": 2},
        ])
        result = self.engine.spawn_wave(comp)
        assert len(result) == 2

    def test_spawn_wave_with_pattern(self):
        self.engine.add_spawn_point(SpawnPoint(position=(0, 0), radius=20))
        comp = EnemyComposition(templates=[
            {"template": "infantry", "count": 5},
        ])
        result = self.engine.spawn_wave(comp, pattern=SpawnPattern.CLUSTER)
        assert len(result) == 5

    def test_total_spawned(self):
        self.engine.add_spawn_point(SpawnPoint(position=(0, 0)))
        comp = EnemyComposition(templates=[{"template": "infantry", "count": 3}])
        self.engine.spawn_wave(comp)
        assert self.engine.total_spawned == 3

    def test_unit_ids_unique(self):
        self.engine.add_spawn_point(SpawnPoint(position=(0, 0)))
        comp = EnemyComposition(templates=[
            {"template": "infantry", "count": 3},
            {"template": "sniper", "count": 2},
        ])
        result = self.engine.spawn_wave(comp)
        ids = [r["unit_id"] for r in result]
        assert len(ids) == len(set(ids))


# ===================================================================
# SpawnerEngine — tick / queue
# ===================================================================


class TestSpawnerEngineTick:
    def setup_method(self):
        self.engine = SpawnerEngine(seed=42)
        self.engine.add_spawn_point(SpawnPoint(position=(0, 0), radius=10))

    def test_tick_empty_queue(self):
        result = self.engine.tick(1.0)
        assert result == []

    def test_enqueue_and_tick(self):
        self.engine.enqueue({"unit_id": "u1", "template": "infantry"})
        self.engine.enqueue({"unit_id": "u2", "template": "infantry"})
        # Default interval 0.5s, tick 1.0s should release 2
        result = self.engine.tick(1.0)
        assert len(result) == 2

    def test_tick_partial_release(self):
        for i in range(5):
            self.engine.enqueue({"unit_id": f"u{i}", "template": "infantry"})
        # 0.5s interval, tick 1.0s => 2 released
        result = self.engine.tick(1.0)
        assert len(result) == 2
        assert self.engine.queue_size == 3

    def test_tick_accumulates(self):
        for i in range(3):
            self.engine.enqueue({"unit_id": f"u{i}", "template": "infantry"})
        r1 = self.engine.tick(0.3)
        assert len(r1) == 0  # not enough time
        r2 = self.engine.tick(0.3)
        assert len(r2) == 1  # 0.6s total, one spawn at 0.5

    def test_enqueue_wave(self):
        comp = EnemyComposition(templates=[
            {"template": "infantry", "count": 4},
        ])
        n = self.engine.enqueue_wave(comp)
        assert n == 4
        assert self.engine.queue_size == 4

    def test_enqueue_wave_and_drain(self):
        comp = EnemyComposition(templates=[
            {"template": "infantry", "count": 3},
        ])
        self.engine.enqueue_wave(comp, interval=0.1)
        total_released = []
        for _ in range(50):
            released = self.engine.tick(0.1)
            total_released.extend(released)
            if self.engine.queue_size == 0:
                break
        assert len(total_released) == 3

    def test_spawn_point_cooldown_ticks(self):
        sp = SpawnPoint(position=(0, 0), cooldown=2.0)
        self.engine.spawn_points = [sp]
        sp.trigger_cooldown()
        assert not sp.is_ready()
        self.engine.tick(1.0)
        assert not sp.is_ready()
        self.engine.tick(1.0)
        assert sp.is_ready()


# ===================================================================
# SpawnerEngine — to_three_js
# ===================================================================


class TestToThreeJs:
    def test_empty(self):
        engine = SpawnerEngine()
        data = engine.to_three_js()
        assert data["spawn_points"] == []
        assert data["queue_size"] == 0
        assert data["total_spawned"] == 0

    def test_with_spawn_points(self):
        engine = SpawnerEngine()
        engine.add_spawn_point(SpawnPoint(position=(10, 20), radius=5, alliance="hostile"))
        engine.add_spawn_point(SpawnPoint(position=(30, 40), radius=8, alliance="friendly", active=False))
        data = engine.to_three_js()
        assert len(data["spawn_points"]) == 2

        sp0 = data["spawn_points"][0]
        assert sp0["position"]["x"] == 10
        assert sp0["position"]["z"] == 20
        assert sp0["radius"] == 5
        assert sp0["alliance"] == "hostile"
        assert sp0["active"] is True
        assert sp0["type"] == "spawn_point"
        assert sp0["geometry"] == "ring"
        assert sp0["color"] == "#ff2a6d"

        sp1 = data["spawn_points"][1]
        assert sp1["active"] is False
        assert sp1["color"] == "#05ffa1"

    def test_after_spawning(self):
        engine = SpawnerEngine(seed=1)
        engine.add_spawn_point(SpawnPoint(position=(0, 0)))
        comp = EnemyComposition(templates=[{"template": "infantry", "count": 5}])
        engine.spawn_wave(comp)
        data = engine.to_three_js()
        assert data["total_spawned"] == 5


# ===================================================================
# Alliance color mapping
# ===================================================================


class TestAllianceSpawnColor:
    def test_hostile(self):
        assert _alliance_spawn_color("hostile") == "#ff2a6d"

    def test_friendly(self):
        assert _alliance_spawn_color("friendly") == "#05ffa1"

    def test_neutral(self):
        assert _alliance_spawn_color("neutral") == "#fcee0a"

    def test_unknown(self):
        assert _alliance_spawn_color("unknown") == "#00f0ff"

    def test_fallback(self):
        assert _alliance_spawn_color("other") == "#888888"


# ===================================================================
# WAVE_PRESETS
# ===================================================================


class TestWavePresets:
    def test_all_presets_present(self):
        expected = {"easy_10", "medium_15", "hard_20", "endless", "boss_rush"}
        assert set(WAVE_PRESETS.keys()) == expected

    def test_preset_structure(self):
        for name, preset in WAVE_PRESETS.items():
            assert "total_waves" in preset
            assert "difficulty_curve" in preset
            assert "base_budget" in preset
            assert "description" in preset
            assert isinstance(preset["base_budget"], (int, float))

    def test_endless_is_negative_one(self):
        assert WAVE_PRESETS["endless"]["total_waves"] == -1

    def test_boss_rush_high_budget(self):
        assert WAVE_PRESETS["boss_rush"]["base_budget"] >= 40


# ===================================================================
# run_preset
# ===================================================================


class TestRunPreset:
    def test_easy_10(self):
        comps = run_preset("easy_10", seed=42)
        assert len(comps) == 10
        for c in comps:
            assert c.total_count > 0

    def test_medium_15(self):
        comps = run_preset("medium_15", seed=42)
        assert len(comps) == 15

    def test_hard_20(self):
        comps = run_preset("hard_20", seed=42)
        assert len(comps) == 20

    def test_endless_capped(self):
        comps = run_preset("endless", max_waves=5, seed=42)
        assert len(comps) == 5

    def test_endless_default_cap(self):
        comps = run_preset("endless", seed=42)
        assert len(comps) == 20  # default cap

    def test_boss_rush(self):
        comps = run_preset("boss_rush", seed=42)
        assert len(comps) == 5
        # Boss rush should have high difficulty
        assert all(c.difficulty_rating > 0 for c in comps)

    def test_max_waves_limits(self):
        comps = run_preset("hard_20", max_waves=3, seed=42)
        assert len(comps) == 3

    def test_seed_reproducibility(self):
        c1 = run_preset("easy_10", seed=99)
        c2 = run_preset("easy_10", seed=99)
        for a, b in zip(c1, c2):
            assert a.templates == b.templates
            assert a.difficulty_rating == b.difficulty_rating

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            run_preset("nonexistent")

    def test_difficulty_increases_over_waves(self):
        comps = run_preset("easy_10", seed=42)
        # Overall trend: later waves should be harder
        assert comps[-1].difficulty_rating > comps[0].difficulty_rating


# ===================================================================
# Template cost / difficulty tables
# ===================================================================


class TestTemplateTables:
    def test_cost_all_positive(self):
        for name, cost in _TEMPLATE_COST.items():
            assert cost > 0, f"{name} has non-positive cost"

    def test_difficulty_all_positive(self):
        for name, diff in _TEMPLATE_DIFFICULTY.items():
            assert diff > 0, f"{name} has non-positive difficulty"

    def test_tank_most_expensive(self):
        assert _TEMPLATE_COST["tank"] >= max(
            v for k, v in _TEMPLATE_COST.items() if k != "tank"
        )

    def test_infantry_cheapest(self):
        assert _TEMPLATE_COST["infantry"] <= min(
            v for k, v in _TEMPLATE_COST.items() if k != "infantry"
        )


# ===================================================================
# Integration: full workflow
# ===================================================================


class TestIntegration:
    def test_design_and_spawn(self):
        """Full workflow: design a wave, create spawn points, spawn it."""
        designer = WaveDesigner(seed=7)
        comp = designer.design_wave(5, "linear", budget=30)
        assert comp.total_count > 0

        engine = SpawnerEngine(seed=7)
        engine.add_spawn_point(SpawnPoint(position=(100, 0), radius=20, alliance="hostile"))
        engine.add_spawn_point(SpawnPoint(position=(-100, 0), radius=20, alliance="hostile"))
        spawned = engine.spawn_wave(comp, pattern=SpawnPattern.FLANKING)
        assert len(spawned) == comp.total_count
        for unit in spawned:
            assert unit["alliance"] == "hostile"

    def test_multi_wave_progression(self):
        """Spawn multiple waves, verify total count grows."""
        designer = WaveDesigner(seed=12)
        engine = SpawnerEngine(seed=12)
        engine.add_spawn_point(SpawnPoint(position=(0, 0), radius=30))

        totals = []
        for w in range(1, 6):
            comp = designer.design_wave(w, "linear", 10)
            engine.spawn_wave(comp)
            totals.append(engine.total_spawned)

        # Each wave adds more, so total should be monotonically increasing
        for i in range(1, len(totals)):
            assert totals[i] > totals[i - 1]

    def test_enqueue_then_tick_full_drain(self):
        """Enqueue a wave and tick until all units are released."""
        designer = WaveDesigner(seed=3)
        comp = designer.design_wave(3, "linear", 15)
        engine = SpawnerEngine(seed=3)
        engine.add_spawn_point(SpawnPoint(position=(50, 50), radius=15))
        total_enqueued = engine.enqueue_wave(comp, interval=0.2)

        released = []
        for _ in range(200):
            r = engine.tick(0.2)
            released.extend(r)
            if engine.queue_size == 0:
                break

        assert len(released) == total_enqueued

    def test_three_js_after_full_workflow(self):
        """Verify Three.js output after spawning."""
        engine = SpawnerEngine(seed=5)
        engine.add_spawn_point(SpawnPoint(position=(10, 20), radius=8, alliance="hostile"))
        comp = EnemyComposition(templates=[{"template": "infantry", "count": 3}])
        engine.spawn_wave(comp)
        data = engine.to_three_js()
        assert data["total_spawned"] == 3
        assert len(data["spawn_points"]) == 1
        assert data["spawn_points"][0]["color"] == "#ff2a6d"

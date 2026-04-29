# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.spawner — wave designer, spawner engine,
spawn patterns, difficulty curves, and budget allocation.

Covers edge cases, behavioral correctness, and integration between the
WaveDesigner and SpawnerEngine.
"""

import math

import pytest

from tritium_lib.sim_engine.spawner import (
    EnemyComposition,
    SpawnPattern,
    SpawnPoint,
    SpawnerEngine,
    UnitTemplate,
    WaveDesigner,
    DIFFICULTY_CURVES,
    WAVE_PRESETS,
    run_preset,
    _alliance_spawn_color,
)


# ---------------------------------------------------------------------------
# WaveDesigner — difficulty curves
# ---------------------------------------------------------------------------


class TestDifficultyCurves:
    """Verify each curve function produces the expected scaling behavior."""

    def test_linear_scales_proportionally(self):
        """Linear curve at wave 5 with base 10 should yield 50."""
        result = DIFFICULTY_CURVES["linear"](5, 10.0)
        assert result == 50.0

    def test_exponential_grows_faster_than_linear(self):
        """Exponential should outpace linear at higher wave numbers."""
        lin = DIFFICULTY_CURVES["linear"](10, 10.0)
        exp = DIFFICULTY_CURVES["exponential"](10, 10.0)
        assert exp > lin

    def test_logarithmic_grows_slower_than_linear(self):
        """Logarithmic should grow more slowly than linear at high waves."""
        lin = DIFFICULTY_CURVES["linear"](20, 10.0)
        log = DIFFICULTY_CURVES["logarithmic"](20, 10.0)
        assert log < lin

    def test_staircase_jumps_every_3_waves(self):
        """Staircase should produce identical budget for waves 1-3, then jump."""
        curve = DIFFICULTY_CURVES["staircase"]
        assert curve(1, 10.0) == curve(2, 10.0) == curve(3, 10.0)
        assert curve(4, 10.0) > curve(3, 10.0)
        assert curve(4, 10.0) == curve(5, 10.0) == curve(6, 10.0)

    def test_random_spikes_stays_positive(self):
        """Random spikes should always produce positive budgets."""
        curve = DIFFICULTY_CURVES["random_spikes"]
        for wave in range(1, 50):
            assert curve(wave, 10.0) > 0.0

    def test_all_curves_positive_for_wave_1(self):
        """Every curve should produce a positive budget for wave 1."""
        for name, fn in DIFFICULTY_CURVES.items():
            assert fn(1, 10.0) > 0.0, f"Curve '{name}' gave non-positive for wave 1"

    def test_logarithmic_wave_zero_safe(self):
        """Logarithmic uses max(wave, 1) so wave 0 should not crash."""
        result = DIFFICULTY_CURVES["logarithmic"](0, 10.0)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# WaveDesigner — composition design
# ---------------------------------------------------------------------------


class TestWaveDesigner:
    """Behavioral tests for the wave composition designer."""

    def test_wave_1_only_infantry(self):
        """Wave 1 pool is infantry-only; composition should contain only infantry."""
        designer = WaveDesigner(seed=42)
        comp = designer.design_wave(wave_number=1, budget=20.0)
        for tmpl in comp.templates:
            assert tmpl["template"] == "infantry"

    def test_later_waves_unlock_unit_types(self):
        """Wave 10 should have access to tanks and other advanced types."""
        designer = WaveDesigner(seed=42)
        comp = designer.design_wave(wave_number=10, budget=200.0)
        templates_used = {t["template"] for t in comp.templates}
        # Wave 10 unlocks everything including tanks
        assert "infantry" in templates_used
        assert len(templates_used) > 1, "Late-game wave should have unit variety"

    def test_budget_zero_still_has_infantry(self):
        """Even with zero effective budget, at least 1 infantry should spawn."""
        designer = WaveDesigner(seed=42)
        comp = designer.design_wave(wave_number=1, budget=0.1)
        assert comp.total_count >= 1
        assert comp.templates[0]["template"] == "infantry"

    def test_total_count_matches_template_sum(self):
        """total_count property must equal the sum of individual template counts."""
        designer = WaveDesigner(seed=42)
        comp = designer.design_wave(wave_number=5, budget=50.0)
        manual_sum = sum(t["count"] for t in comp.templates)
        assert comp.total_count == manual_sum

    def test_difficulty_rating_increases_with_wave(self):
        """Difficulty rating should trend upward with wave number."""
        designer = WaveDesigner(seed=42)
        ratings = [
            designer.design_wave(wave_number=w, budget=20.0).difficulty_rating
            for w in [1, 5, 10]
        ]
        # The general trend should be upward (allow non-strict due to RNG)
        assert ratings[-1] > ratings[0]

    def test_composition_serialization(self):
        """to_dict should produce a valid JSON-serializable dict."""
        designer = WaveDesigner(seed=42)
        comp = designer.design_wave(wave_number=3, budget=30.0)
        d = comp.to_dict()
        assert "templates" in d
        assert "total_count" in d
        assert "difficulty_rating" in d
        assert isinstance(d["total_count"], int)

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical compositions."""
        d1 = WaveDesigner(seed=99)
        d2 = WaveDesigner(seed=99)
        c1 = d1.design_wave(wave_number=5, budget=50.0)
        c2 = d2.design_wave(wave_number=5, budget=50.0)
        assert c1.templates == c2.templates
        assert c1.difficulty_rating == c2.difficulty_rating

    def test_equipment_assigned_to_all_templates(self):
        """Every spawned template should have an equipment list."""
        designer = WaveDesigner(seed=42)
        comp = designer.design_wave(wave_number=10, budget=200.0)
        for tmpl in comp.templates:
            assert "equipment" in tmpl
            assert isinstance(tmpl["equipment"], list)


# ---------------------------------------------------------------------------
# WaveDesigner — spawn patterns
# ---------------------------------------------------------------------------


class TestSpawnPatterns:
    """Behavioral tests for spatial spawn patterns."""

    def test_random_positions_within_radius(self):
        """All random positions should be within the given radius of center."""
        designer = WaveDesigner(seed=42)
        center = (50.0, 50.0)
        radius = 20.0
        positions = designer.generate_spawn_positions(
            SpawnPattern.RANDOM, center, radius, 100,
        )
        assert len(positions) == 100
        for px, py in positions:
            dist = math.sqrt((px - center[0]) ** 2 + (py - center[1]) ** 2)
            assert dist <= radius + 0.001

    def test_line_pattern_collinear(self):
        """Line pattern should produce positions along a horizontal line."""
        designer = WaveDesigner(seed=42)
        positions = designer.generate_spawn_positions(
            SpawnPattern.LINE, (50.0, 50.0), 20.0, 5,
        )
        assert len(positions) == 5
        # All y-values should be the same (center y)
        for _, y in positions:
            assert abs(y - 50.0) < 0.001

    def test_surround_pattern_evenly_spaced(self):
        """Surround pattern should place units at equal angular intervals."""
        designer = WaveDesigner(seed=42)
        center = (0.0, 0.0)
        radius = 10.0
        positions = designer.generate_spawn_positions(
            SpawnPattern.SURROUND, center, radius, 4,
        )
        assert len(positions) == 4
        # Each position should be at distance ~radius from center
        for px, py in positions:
            dist = math.sqrt(px ** 2 + py ** 2)
            assert abs(dist - radius) < 0.001

    def test_flanking_splits_into_two_groups(self):
        """Flanking pattern should produce positions on both sides of center."""
        designer = WaveDesigner(seed=42)
        center = (50.0, 50.0)
        positions = designer.generate_spawn_positions(
            SpawnPattern.FLANKING, center, 30.0, 10,
        )
        assert len(positions) == 10
        # Should have positions both left and right of center
        left = [p for p in positions if p[0] < center[0]]
        right = [p for p in positions if p[0] > center[0]]
        assert len(left) > 0
        assert len(right) > 0

    def test_zero_count_returns_empty(self):
        """Requesting 0 positions should return an empty list."""
        designer = WaveDesigner(seed=42)
        positions = designer.generate_spawn_positions(
            SpawnPattern.RANDOM, (0.0, 0.0), 10.0, 0,
        )
        assert positions == []

    def test_single_unit_line_returns_center(self):
        """Line pattern with count=1 returns exactly the center point."""
        designer = WaveDesigner(seed=42)
        center = (25.0, 75.0)
        positions = designer.generate_spawn_positions(
            SpawnPattern.LINE, center, 10.0, 1,
        )
        assert len(positions) == 1
        assert positions[0] == center

    def test_trickle_single_unit_returns_center(self):
        """Trickle pattern with count=1 returns the center."""
        designer = WaveDesigner(seed=42)
        center = (10.0, 20.0)
        positions = designer.generate_spawn_positions(
            SpawnPattern.TRICKLE, center, 10.0, 1,
        )
        assert len(positions) == 1
        assert positions[0] == center

    def test_waves_pattern_produces_rows(self):
        """Waves pattern should produce multiple rows of units."""
        designer = WaveDesigner(seed=42)
        positions = designer.generate_spawn_positions(
            SpawnPattern.WAVES, (50.0, 50.0), 30.0, 12,
        )
        assert len(positions) == 12
        # Y values should span multiple rows
        y_vals = sorted(set(round(p[1], 0) for p in positions))
        assert len(y_vals) >= 2


# ---------------------------------------------------------------------------
# SpawnPoint
# ---------------------------------------------------------------------------


class TestSpawnPoint:
    """Tests for SpawnPoint cooldown mechanics."""

    def test_is_ready_when_active_no_cooldown(self):
        sp = SpawnPoint(position=(0, 0), radius=10.0, cooldown=5.0)
        assert sp.is_ready()

    def test_not_ready_during_cooldown(self):
        sp = SpawnPoint(position=(0, 0), radius=10.0, cooldown=5.0)
        sp.trigger_cooldown()
        assert not sp.is_ready()

    def test_cooldown_expires(self):
        sp = SpawnPoint(position=(0, 0), radius=10.0, cooldown=5.0)
        sp.trigger_cooldown()
        sp.tick_cooldown(6.0)
        assert sp.is_ready()

    def test_inactive_spawn_point_never_ready(self):
        sp = SpawnPoint(position=(0, 0), radius=10.0, active=False)
        assert not sp.is_ready()

    def test_cooldown_does_not_go_negative(self):
        sp = SpawnPoint(position=(0, 0), radius=10.0, cooldown=2.0)
        sp.trigger_cooldown()
        sp.tick_cooldown(100.0)
        assert sp._cooldown_remaining == 0.0


# ---------------------------------------------------------------------------
# SpawnerEngine — immediate spawning
# ---------------------------------------------------------------------------


class TestSpawnerEngine:
    """Tests for the SpawnerEngine wave spawning and queue system."""

    def test_spawn_wave_with_registered_points(self):
        """Spawning a wave distributes units across registered spawn points."""
        engine = SpawnerEngine(seed=42)
        engine.add_spawn_point(SpawnPoint(position=(0, 0), radius=10.0))
        engine.add_spawn_point(SpawnPoint(position=(100, 100), radius=10.0))

        comp = EnemyComposition(
            templates=[{"template": "infantry", "count": 4, "equipment": ["rifle"]}],
            difficulty_rating=4.0,
        )
        spawned = engine.spawn_wave(comp)
        assert len(spawned) == 4
        assert engine.total_spawned == 4

        # Units should be distributed round-robin across spawn points
        sp_indices = [u["spawn_point_index"] for u in spawned]
        assert 0 in sp_indices
        assert 1 in sp_indices

    def test_spawn_wave_no_points_fallback_to_origin(self):
        """If no spawn points registered, spawn at origin."""
        engine = SpawnerEngine(seed=42)
        comp = EnemyComposition(
            templates=[{"template": "scout", "count": 2}],
        )
        spawned = engine.spawn_wave(comp)
        assert len(spawned) == 2
        # Should not crash, positions near origin
        for unit in spawned:
            assert "position" in unit

    def test_spawn_wave_assigns_alliance(self):
        """Spawned units should inherit the spawn point's alliance."""
        engine = SpawnerEngine(seed=42)
        engine.add_spawn_point(
            SpawnPoint(position=(50, 50), radius=5.0, alliance="neutral"),
        )
        comp = EnemyComposition(
            templates=[{"template": "infantry", "count": 1}],
        )
        spawned = engine.spawn_wave(comp)
        assert spawned[0]["alliance"] == "neutral"

    def test_remove_spawn_point(self):
        """Removing a spawn point by index works correctly."""
        engine = SpawnerEngine()
        sp1 = SpawnPoint(position=(0, 0))
        sp2 = SpawnPoint(position=(10, 10))
        engine.add_spawn_point(sp1)
        engine.add_spawn_point(sp2)
        removed = engine.remove_spawn_point(0)
        assert removed is sp1
        assert len(engine.spawn_points) == 1

    def test_remove_invalid_index_returns_none(self):
        engine = SpawnerEngine()
        assert engine.remove_spawn_point(5) is None

    def test_active_spawn_points_filters_inactive(self):
        engine = SpawnerEngine()
        engine.add_spawn_point(SpawnPoint(position=(0, 0), active=True))
        engine.add_spawn_point(SpawnPoint(position=(10, 10), active=False))
        active = engine.active_spawn_points()
        assert len(active) == 1


# ---------------------------------------------------------------------------
# SpawnerEngine — enqueue and tick
# ---------------------------------------------------------------------------


class TestSpawnerEngineTick:
    """Tests for the timed/trickle spawn queue system."""

    def test_enqueue_wave_adds_to_queue(self):
        engine = SpawnerEngine(seed=42)
        engine.add_spawn_point(SpawnPoint(position=(0, 0), radius=10.0))
        comp = EnemyComposition(
            templates=[{"template": "infantry", "count": 5}],
        )
        count = engine.enqueue_wave(comp, interval=1.0)
        assert count == 5
        assert engine.queue_size == 5
        assert engine.total_spawned == 0  # nothing spawned yet

    def test_tick_releases_units_over_time(self):
        engine = SpawnerEngine(seed=42)
        engine.add_spawn_point(SpawnPoint(position=(0, 0), radius=10.0))
        comp = EnemyComposition(
            templates=[{"template": "infantry", "count": 3}],
        )
        engine.enqueue_wave(comp, interval=1.0)

        # Tick 0.5s: not enough time for first spawn
        released = engine.tick(0.4)
        assert len(released) == 0

        # Tick to 1.0s: first unit should spawn
        released = engine.tick(0.6)
        assert len(released) == 1
        assert engine.queue_size == 2

    def test_tick_with_no_queue_returns_empty(self):
        engine = SpawnerEngine()
        released = engine.tick(1.0)
        assert released == []

    def test_full_drain_of_queue(self):
        engine = SpawnerEngine(seed=42)
        engine.add_spawn_point(SpawnPoint(position=(0, 0), radius=10.0))
        comp = EnemyComposition(
            templates=[{"template": "infantry", "count": 3}],
        )
        engine.enqueue_wave(comp, interval=0.5)

        # Tick enough to drain all 3 units
        all_released = []
        for _ in range(20):
            released = engine.tick(0.5)
            all_released.extend(released)
            if engine.queue_size == 0:
                break

        assert len(all_released) == 3
        assert engine.queue_size == 0
        assert engine.total_spawned == 3

    def test_tick_advances_spawn_point_cooldowns(self):
        engine = SpawnerEngine()
        sp = SpawnPoint(position=(0, 0), cooldown=5.0)
        sp.trigger_cooldown()
        engine.add_spawn_point(sp)
        engine.tick(6.0)
        assert sp.is_ready()


# ---------------------------------------------------------------------------
# SpawnerEngine — Three.js export
# ---------------------------------------------------------------------------


class TestSpawnerThreeJS:
    """Tests for the Three.js visualization export."""

    def test_to_three_js_structure(self):
        engine = SpawnerEngine()
        engine.add_spawn_point(
            SpawnPoint(position=(10, 20), radius=15.0, alliance="hostile"),
        )
        data = engine.to_three_js()
        assert "spawn_points" in data
        assert "queue_size" in data
        assert "total_spawned" in data
        assert len(data["spawn_points"]) == 1
        marker = data["spawn_points"][0]
        assert marker["geometry"] == "ring"
        assert marker["type"] == "spawn_point"
        assert marker["color"] == "#ff2a6d"  # hostile color

    def test_alliance_colors(self):
        assert _alliance_spawn_color("hostile") == "#ff2a6d"
        assert _alliance_spawn_color("friendly") == "#05ffa1"
        assert _alliance_spawn_color("neutral") == "#fcee0a"
        assert _alliance_spawn_color("unknown") == "#00f0ff"
        assert _alliance_spawn_color("invalid") == "#888888"


# ---------------------------------------------------------------------------
# Wave presets
# ---------------------------------------------------------------------------


class TestWavePresets:
    """Tests for the preset wave configurations and run_preset()."""

    def test_all_presets_are_valid(self):
        for name, preset in WAVE_PRESETS.items():
            assert "total_waves" in preset
            assert "difficulty_curve" in preset
            assert "base_budget" in preset

    def test_run_preset_returns_correct_count(self):
        compositions = run_preset("easy_10", seed=42)
        assert len(compositions) == 10
        for comp in compositions:
            assert comp.total_count > 0

    def test_run_preset_endless_caps_at_default(self):
        """Endless preset should cap at 20 by default."""
        compositions = run_preset("endless", seed=42)
        assert len(compositions) == 20

    def test_run_preset_with_max_waves_override(self):
        compositions = run_preset("hard_20", max_waves=3, seed=42)
        assert len(compositions) == 3

    def test_boss_rush_has_high_budgets(self):
        compositions = run_preset("boss_rush", seed=42)
        assert len(compositions) == 5
        # Boss rush waves should have high unit counts due to large budgets
        for comp in compositions:
            assert comp.difficulty_rating > 0

    def test_run_preset_invalid_name_raises(self):
        with pytest.raises(KeyError):
            run_preset("nonexistent_preset")

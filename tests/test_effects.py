# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for game_effects particle system."""

import math
import time

import numpy as np
import pytest

from tritium_lib.game_effects import (
    EffectsManager,
    Particle,
    ParticleEmitter,
    blood_splatter,
    debris,
    explosion,
    fire,
    muzzle_flash,
    smoke,
    sparks,
    tracer,
)
from tritium_lib.game_effects.particles import (
    GRAVITY,
    _lerp_color,
    _multi_lerp_color,
)


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

class TestColorUtils:
    def test_lerp_color_endpoints(self):
        c1 = (0, 0, 0, 255)
        c2 = (255, 255, 255, 0)
        assert _lerp_color(c1, c2, 0.0) == c1
        assert _lerp_color(c1, c2, 1.0) == c2

    def test_lerp_color_midpoint(self):
        c1 = (0, 0, 0, 0)
        c2 = (200, 100, 50, 200)
        mid = _lerp_color(c1, c2, 0.5)
        assert mid == (100, 50, 25, 100)

    def test_lerp_color_clamps(self):
        c1 = (0, 0, 0, 0)
        c2 = (100, 100, 100, 100)
        assert _lerp_color(c1, c2, -1.0) == c1
        assert _lerp_color(c1, c2, 2.0) == c2

    def test_multi_lerp_three_stops(self):
        colors = [
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
        ]
        start = _multi_lerp_color(colors, 0.0)
        assert start == (255, 0, 0, 255)
        end = _multi_lerp_color(colors, 1.0)
        assert end == (0, 0, 255, 255)
        mid = _multi_lerp_color(colors, 0.5)
        assert mid == (0, 255, 0, 255)


# ---------------------------------------------------------------------------
# Particle dataclass
# ---------------------------------------------------------------------------

class TestParticle:
    def test_creation(self):
        p = Particle(
            position=np.array([1.0, 2.0]),
            velocity=np.array([3.0, 4.0]),
            color=(255, 0, 0, 255),
            size=2.0,
            lifetime=1.0,
            max_lifetime=1.0,
        )
        assert p.alive()
        assert p.age_ratio == pytest.approx(0.0)

    def test_age_ratio(self):
        p = Particle(
            position=np.zeros(2),
            velocity=np.zeros(2),
            color=(255, 255, 255, 255),
            size=1.0,
            lifetime=0.5,
            max_lifetime=1.0,
        )
        assert p.age_ratio == pytest.approx(0.5)

    def test_dead_particle(self):
        p = Particle(
            position=np.zeros(2),
            velocity=np.zeros(2),
            color=(0, 0, 0, 0),
            size=1.0,
            lifetime=-0.1,
            max_lifetime=1.0,
        )
        assert not p.alive()
        assert p.age_ratio == 1.0

    def test_to_dict(self):
        p = Particle(
            position=np.array([10.0, 20.0]),
            velocity=np.array([1.0, -1.0]),
            color=(255, 128, 64, 200),
            size=3.0,
            lifetime=0.7,
            max_lifetime=1.0,
        )
        d = p.to_dict()
        assert d["x"] == 10.0
        assert d["y"] == 20.0
        assert d["r"] == 255
        assert d["g"] == 128
        assert d["b"] == 64
        assert d["a"] == 200
        assert d["size"] == 3.0
        assert d["lifetime"] == pytest.approx(0.7)
        assert d["age"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# ParticleEmitter
# ---------------------------------------------------------------------------

class TestParticleEmitter:
    def test_emit_burst(self):
        em = ParticleEmitter(
            (0, 0),
            emit_burst=20,
            one_shot=True,
        )
        assert len(em.particles) == 20
        assert em.done is True

    def test_emit_manual(self):
        em = ParticleEmitter((0, 0))
        spawned = em.emit(10)
        assert len(spawned) == 10
        assert len(em.particles) == 10

    def test_max_particles_cap(self):
        em = ParticleEmitter((0, 0), max_particles=5)
        em.emit(100)
        assert len(em.particles) == 5

    def test_tick_removes_dead(self):
        em = ParticleEmitter(
            (0, 0),
            lifetime_range=(0.1, 0.1),
            emit_burst=10,
            one_shot=True,
        )
        assert len(em.particles) == 10
        em.tick(0.2)  # all should die
        assert len(em.particles) == 0

    def test_tick_moves_particles(self):
        em = ParticleEmitter(
            (0, 0),
            speed_range=(100.0, 100.0),
            lifetime_range=(2.0, 2.0),
            spread=0.01,
            base_angle=0.0,
            gravity_scale=0.0,
            drag=0.0,
            emit_burst=1,
            one_shot=True,
        )
        p = em.particles[0]
        initial_x = p.position[0]
        em.tick(0.1)
        assert em.particles[0].position[0] > initial_x

    def test_gravity_affects_velocity(self):
        em = ParticleEmitter(
            (0, 0),
            speed_range=(0.0, 0.0),
            lifetime_range=(5.0, 5.0),
            gravity_scale=1.0,
            emit_burst=1,
            one_shot=True,
        )
        em.tick(1.0)
        p = em.particles[0]
        # Gravity pushes Y positive (downward)
        assert p.velocity[1] > 0

    def test_continuous_emission(self):
        em = ParticleEmitter(
            (0, 0),
            emit_rate=100.0,
            lifetime_range=(2.0, 2.0),
        )
        assert len(em.particles) == 0
        em.tick(0.1)  # should emit ~10 particles
        assert len(em.particles) >= 5

    def test_is_finished(self):
        em = ParticleEmitter(
            (0, 0),
            lifetime_range=(0.05, 0.05),
            emit_burst=5,
            one_shot=True,
        )
        assert not em.is_finished()
        em.tick(0.1)
        assert em.is_finished()

    def test_get_particles_returns_dicts(self):
        em = ParticleEmitter(
            (0, 0),
            emit_burst=3,
            one_shot=True,
        )
        data = em.get_particles()
        assert len(data) == 3
        assert all(isinstance(d, dict) for d in data)
        assert all("x" in d and "y" in d and "r" in d for d in data)

    def test_position_jitter(self):
        positions = set()
        for _ in range(20):
            em = ParticleEmitter(
                (100, 100),
                position_jitter=10.0,
                emit_burst=1,
                one_shot=True,
            )
            p = em.particles[0]
            positions.add((round(p.position[0], 1), round(p.position[1], 1)))
        # With jitter, not all positions should be identical
        assert len(positions) > 1

    def test_drag_slows_particles(self):
        em = ParticleEmitter(
            (0, 0),
            speed_range=(100.0, 100.0),
            lifetime_range=(5.0, 5.0),
            drag=0.5,
            gravity_scale=0.0,
            spread=0.01,
            base_angle=0.0,
            emit_burst=1,
            one_shot=True,
        )
        initial_speed = float(np.linalg.norm(em.particles[0].velocity))
        em.tick(1.0)
        final_speed = float(np.linalg.norm(em.particles[0].velocity))
        assert final_speed < initial_speed


# ---------------------------------------------------------------------------
# Effect factories
# ---------------------------------------------------------------------------

class TestEffectFactories:
    """Verify each factory creates a working emitter with sane defaults."""

    def test_explosion(self):
        em = explosion((50, 50), radius=15.0, num_particles=30)
        assert len(em.particles) == 30
        assert em.one_shot is True
        # Tick it — particles should move
        em.tick(0.1)
        assert len(em.particles) > 0

    def test_muzzle_flash(self):
        em = muzzle_flash((10, 10), heading=0.0)
        assert len(em.particles) == 12
        assert em.one_shot is True
        # Very short-lived
        em.tick(0.2)
        assert em.is_finished()

    def test_tracer(self):
        em = tracer((0, 0), (100, 0), speed=300.0)
        assert len(em.particles) == 8
        assert em.one_shot is True
        # Should move toward target
        em.tick(0.05)
        for p in em.particles:
            assert p.position[0] > 0

    def test_tracer_zero_distance(self):
        em = tracer((5, 5), (5, 5), speed=300.0)
        assert len(em.particles) == 8

    def test_smoke(self):
        em = smoke((30, 30), duration=5.0)
        assert em.one_shot is False
        assert em.emit_rate > 0
        em.tick(0.5)
        assert len(em.particles) > 0

    def test_debris(self):
        em = debris((20, 20), num_pieces=15)
        assert len(em.particles) == 15
        assert em.gravity_scale == 1.0
        # Record initial Y velocities before gravity acts
        initial_vy = [p.velocity[1] for p in em.particles]
        em.tick(0.5)
        # Gravity should have increased Y velocity (downward) for all particles
        for i, p in enumerate(em.particles):
            assert p.velocity[1] > initial_vy[i]

    def test_blood_splatter(self):
        em = blood_splatter((50, 50), (1, 0))
        assert len(em.particles) == 15
        assert em.one_shot is True
        em.tick(0.1)
        # Most should have moved rightward
        rightward = sum(1 for p in em.particles if p.position[0] > 50)
        assert rightward > len(em.particles) // 2

    def test_blood_splatter_zero_direction(self):
        em = blood_splatter((0, 0), (0, 0))
        assert len(em.particles) == 15

    def test_fire(self):
        em = fire((40, 40), size=8.0)
        assert em.one_shot is False
        assert em.emit_rate > 0
        em.tick(0.2)
        assert len(em.particles) > 0

    def test_sparks(self):
        em = sparks((10, 10), (1, -1))
        assert len(em.particles) == 10
        assert em.one_shot is True
        em.tick(0.05)
        assert len(em.particles) > 0

    def test_sparks_zero_direction(self):
        em = sparks((0, 0), (0, 0))
        assert len(em.particles) == 10


# ---------------------------------------------------------------------------
# EffectsManager
# ---------------------------------------------------------------------------

class TestEffectsManager:
    def test_add_and_tick(self):
        mgr = EffectsManager()
        mgr.add(explosion((0, 0)))
        mgr.add(debris((10, 10)))
        assert mgr.active_count() == 2
        mgr.tick(0.1)
        assert mgr.active_count() >= 1

    def test_max_emitters(self):
        mgr = EffectsManager(max_emitters=3)
        for i in range(5):
            mgr.add(explosion((i * 10, 0)))
        assert mgr.active_count() == 3

    def test_culls_finished_emitters(self):
        mgr = EffectsManager()
        mgr.add(muzzle_flash((0, 0), 0.0))  # very short lived
        assert mgr.active_count() == 1
        mgr.tick(0.5)  # should finish
        assert mgr.active_count() == 0

    def test_get_all_particles(self):
        mgr = EffectsManager()
        mgr.add(explosion((0, 0), num_particles=10))
        mgr.add(debris((10, 10), num_pieces=5))
        particles = mgr.get_all_particles()
        assert len(particles) == 15
        assert all(isinstance(p, dict) for p in particles)

    def test_total_particles(self):
        mgr = EffectsManager()
        mgr.add(explosion((0, 0), num_particles=20))
        assert mgr.total_particles() == 20

    def test_clear(self):
        mgr = EffectsManager()
        mgr.add(explosion((0, 0)))
        mgr.add(smoke((10, 10)))
        mgr.clear()
        assert mgr.active_count() == 0
        assert mgr.total_particles() == 0

    def test_mixed_effects(self):
        """Smoke (continuous) + explosion (one-shot) coexist."""
        mgr = EffectsManager()
        mgr.add(smoke((0, 0)))
        mgr.add(explosion((50, 50), num_particles=20))
        for _ in range(10):
            mgr.tick(0.1)
        # Smoke should still be active, explosion may be done
        assert mgr.active_count() >= 1


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_1000_particles_under_5ms(self):
        """1000 particles should tick in <5ms."""
        em = ParticleEmitter(
            (0, 0),
            speed_range=(10.0, 50.0),
            lifetime_range=(2.0, 5.0),
            gravity_scale=1.0,
            drag=0.1,
            emit_burst=1000,
            max_particles=1000,
            one_shot=True,
        )
        assert len(em.particles) == 1000

        # Warm up
        em.tick(0.001)

        # Timed run
        start = time.perf_counter()
        em.tick(0.016)  # ~60fps frame
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5.0, f"Tick took {elapsed_ms:.2f}ms (limit 5ms)"

    def test_manager_many_emitters(self):
        """50 emitters with 20 particles each should tick quickly."""
        mgr = EffectsManager(max_emitters=256)
        for i in range(50):
            mgr.add(explosion((i * 10, 0), num_particles=20))

        start = time.perf_counter()
        mgr.tick(0.016)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0, f"Manager tick took {elapsed_ms:.2f}ms"

    def test_export_performance(self):
        """Exporting 1000 particles to dicts should be fast."""
        em = ParticleEmitter(
            (0, 0),
            lifetime_range=(5.0, 5.0),
            emit_burst=1000,
            max_particles=1000,
            one_shot=True,
        )

        start = time.perf_counter()
        data = em.get_particles()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(data) == 1000
        assert elapsed_ms < 10.0, f"Export took {elapsed_ms:.2f}ms"


# ---------------------------------------------------------------------------
# Integration / round-trip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_lifecycle(self):
        """Create effects, tick through lifecycle, verify cleanup."""
        mgr = EffectsManager()
        mgr.add(explosion((100, 100), num_particles=30))
        mgr.add(muzzle_flash((90, 100), heading=0.0))
        mgr.add(tracer((90, 100), (200, 100)))
        mgr.add(debris((100, 100), num_pieces=10))

        initial = mgr.total_particles()
        assert initial > 0

        # Simulate 3 seconds at 60fps
        for _ in range(180):
            mgr.tick(1.0 / 60.0)

        # All one-shot effects should be finished
        assert mgr.total_particles() == 0
        assert mgr.active_count() == 0

    def test_continuous_fire_persists(self):
        """Fire keeps emitting until removed."""
        mgr = EffectsManager()
        mgr.add(fire((50, 50)))

        for _ in range(60):
            mgr.tick(1.0 / 60.0)

        assert mgr.active_count() == 1
        assert mgr.total_particles() > 0

    def test_import_from_package(self):
        """Verify public API is accessible from package."""
        from tritium_lib.game_effects import (
            EffectsManager,
            Particle,
            ParticleEmitter,
            blood_splatter,
            debris,
            explosion,
            fire,
            muzzle_flash,
            smoke,
            sparks,
            tracer,
        )
        # All should be callable
        assert callable(explosion)
        assert callable(EffectsManager)

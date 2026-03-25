# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the particle effects system — emitters, lifecycle, factories, manager."""

import math

import pytest
import numpy as np

from tritium_lib.sim_engine.effects.particles import (
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
    _lerp_color,
    _multi_lerp_color,
)


# ---------------------------------------------------------------------------
# Particle dataclass
# ---------------------------------------------------------------------------

class TestParticle:
    def test_basic_creation(self):
        p = Particle(
            position=np.array([10.0, 20.0]),
            velocity=np.array([1.0, -1.0]),
            color=(255, 0, 0, 255),
            size=3.0,
            lifetime=2.0,
            max_lifetime=2.0,
        )
        assert p.alive()
        assert p.age_ratio == pytest.approx(0.0)
        assert p.size == 3.0

    def test_age_ratio_halfway(self):
        p = Particle(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            color=(255, 255, 255, 255),
            size=1.0,
            lifetime=1.0,
            max_lifetime=2.0,
        )
        assert p.age_ratio == pytest.approx(0.5)

    def test_age_ratio_dead(self):
        p = Particle(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            color=(255, 255, 255, 255),
            size=1.0,
            lifetime=0.0,
            max_lifetime=2.0,
        )
        assert p.age_ratio == pytest.approx(1.0)
        assert not p.alive()

    def test_age_ratio_zero_max_lifetime(self):
        p = Particle(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            color=(255, 255, 255, 255),
            size=1.0,
            lifetime=0.5,
            max_lifetime=0.0,
        )
        assert p.age_ratio == 1.0

    def test_to_dict(self):
        p = Particle(
            position=np.array([5.0, 10.0]),
            velocity=np.array([2.0, -3.0]),
            color=(100, 150, 200, 250),
            size=4.0,
            lifetime=1.5,
            max_lifetime=3.0,
        )
        d = p.to_dict()
        assert d["x"] == 5.0
        assert d["y"] == 10.0
        assert d["vx"] == 2.0
        assert d["vy"] == -3.0
        assert d["r"] == 100
        assert d["g"] == 150
        assert d["b"] == 200
        assert d["a"] == 250
        assert d["size"] == 4.0
        assert d["lifetime"] == 1.5
        assert d["max_lifetime"] == 3.0
        assert d["age"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

class TestColorLerp:
    def test_lerp_start(self):
        c = _lerp_color((0, 0, 0, 0), (255, 255, 255, 255), 0.0)
        assert c == (0, 0, 0, 0)

    def test_lerp_end(self):
        c = _lerp_color((0, 0, 0, 0), (255, 255, 255, 255), 1.0)
        assert c == (255, 255, 255, 255)

    def test_lerp_mid(self):
        c = _lerp_color((0, 0, 0, 0), (200, 100, 50, 255), 0.5)
        assert c == (100, 50, 25, 127)

    def test_lerp_clamp_above(self):
        c = _lerp_color((0, 0, 0, 0), (255, 255, 255, 255), 2.0)
        assert c == (255, 255, 255, 255)

    def test_lerp_clamp_below(self):
        c = _lerp_color((100, 100, 100, 100), (200, 200, 200, 200), -1.0)
        assert c == (100, 100, 100, 100)

    def test_multi_lerp_single_color(self):
        c = _multi_lerp_color([(128, 64, 32, 255)], 0.5)
        assert c == (128, 64, 32, 255)

    def test_multi_lerp_empty_colors(self):
        c = _multi_lerp_color([], 0.5)
        assert c == (255, 255, 255, 255)

    def test_multi_lerp_three_stops(self):
        colors = [(0, 0, 0, 255), (128, 128, 128, 255), (255, 255, 255, 255)]
        c_start = _multi_lerp_color(colors, 0.0)
        assert c_start == (0, 0, 0, 255)
        c_mid = _multi_lerp_color(colors, 0.5)
        assert c_mid == (128, 128, 128, 255)
        c_end = _multi_lerp_color(colors, 1.0)
        assert c_end == (255, 255, 255, 255)


# ---------------------------------------------------------------------------
# ParticleEmitter
# ---------------------------------------------------------------------------

class TestParticleEmitter:
    def test_burst_creates_particles(self):
        em = ParticleEmitter((0, 0), emit_burst=20, one_shot=True)
        assert len(em.particles) == 20
        assert em.done is True

    def test_burst_without_one_shot(self):
        em = ParticleEmitter((0, 0), emit_burst=10, one_shot=False)
        assert len(em.particles) == 10
        assert em.done is False

    def test_emit_manual(self):
        em = ParticleEmitter((0, 0))
        assert len(em.particles) == 0
        spawned = em.emit(5)
        assert len(spawned) == 5
        assert len(em.particles) == 5

    def test_max_particles_cap(self):
        em = ParticleEmitter((0, 0), max_particles=10)
        em.emit(15)
        assert len(em.particles) <= 10

    def test_tick_removes_dead_particles(self):
        em = ParticleEmitter(
            (0, 0),
            emit_burst=10,
            one_shot=True,
            lifetime_range=(0.01, 0.02),
        )
        assert len(em.particles) == 10
        em.tick(0.1)  # All should die after 0.1s (lifetime is 0.01-0.02)
        assert len(em.particles) == 0

    def test_tick_updates_positions(self):
        em = ParticleEmitter(
            (0, 0),
            emit_burst=1,
            one_shot=True,
            speed_range=(100.0, 100.0),
            lifetime_range=(5.0, 5.0),
            spread=0.001,
            base_angle=0.0,
        )
        initial_x = em.particles[0].position[0]
        em.tick(0.1)
        # Particle should have moved significantly in X direction
        if em.particles:
            assert em.particles[0].position[0] != initial_x

    def test_continuous_emission(self):
        em = ParticleEmitter(
            (0, 0),
            emit_rate=100.0,  # 100 particles per second
            lifetime_range=(5.0, 5.0),
        )
        assert len(em.particles) == 0
        em.tick(0.1)  # Should emit ~10 particles
        assert len(em.particles) > 0

    def test_gravity_pulls_down(self):
        em = ParticleEmitter(
            (50, 50),
            emit_burst=1,
            one_shot=True,
            speed_range=(0.0, 0.0),
            lifetime_range=(10.0, 10.0),
            gravity_scale=1.0,
        )
        initial_y = em.particles[0].position[1]
        em.tick(0.5)
        if em.particles:
            # Y should increase (gravity is positive Y = down)
            assert em.particles[0].position[1] > initial_y

    def test_drag_slows_particles(self):
        em_no_drag = ParticleEmitter(
            (0, 0),
            emit_burst=1,
            one_shot=True,
            speed_range=(100.0, 100.0),
            lifetime_range=(10.0, 10.0),
            drag=0.0,
            spread=0.001,
            base_angle=0.0,
        )
        em_drag = ParticleEmitter(
            (0, 0),
            emit_burst=1,
            one_shot=True,
            speed_range=(100.0, 100.0),
            lifetime_range=(10.0, 10.0),
            drag=0.5,
            spread=0.001,
            base_angle=0.0,
        )
        em_no_drag.tick(1.0)
        em_drag.tick(1.0)
        if em_no_drag.particles and em_drag.particles:
            speed_no_drag = np.linalg.norm(em_no_drag.particles[0].velocity)
            speed_drag = np.linalg.norm(em_drag.particles[0].velocity)
            assert speed_drag < speed_no_drag

    def test_get_particles_returns_dicts(self):
        em = ParticleEmitter((0, 0), emit_burst=3, one_shot=True)
        data = em.get_particles()
        assert len(data) == 3
        for d in data:
            assert "x" in d and "y" in d
            assert "r" in d and "g" in d and "b" in d and "a" in d
            assert "size" in d and "lifetime" in d

    def test_is_finished_one_shot(self):
        em = ParticleEmitter(
            (0, 0),
            emit_burst=5,
            one_shot=True,
            lifetime_range=(0.01, 0.01),
        )
        assert not em.is_finished()  # Still has live particles
        em.tick(0.1)
        assert em.is_finished()  # All dead, one_shot = done

    def test_position_jitter(self):
        positions = set()
        for _ in range(20):
            em = ParticleEmitter(
                (50, 50),
                emit_burst=1,
                one_shot=True,
                position_jitter=10.0,
            )
            p = em.particles[0]
            positions.add((round(p.position[0], 1), round(p.position[1], 1)))
        # With jitter, not all particles should land exactly at (50, 50)
        assert len(positions) > 1

    def test_elapsed_time_tracks(self):
        em = ParticleEmitter((0, 0))
        assert em.elapsed == 0.0
        em.tick(0.5)
        assert em.elapsed == pytest.approx(0.5)
        em.tick(0.3)
        assert em.elapsed == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Effect factories
# ---------------------------------------------------------------------------

class TestEffectFactories:
    def test_explosion_creates_particles(self):
        em = explosion((50, 50), radius=20.0, num_particles=30)
        assert len(em.particles) == 30
        assert em.one_shot is True
        assert em.done is True
        assert em.color_gradient is not None

    def test_muzzle_flash(self):
        em = muzzle_flash((10, 10), heading=math.pi / 4)
        assert len(em.particles) > 0
        assert em.one_shot is True

    def test_tracer(self):
        em = tracer((0, 0), (100, 0), speed=300.0)
        assert len(em.particles) > 0
        assert em.one_shot is True

    def test_tracer_zero_distance(self):
        em = tracer((50, 50), (50, 50))
        assert em.one_shot is True

    def test_smoke_continuous(self):
        em = smoke((0, 0), duration=5.0)
        assert em.emit_rate > 0
        assert em.one_shot is False

    def test_debris_with_gravity(self):
        em = debris((30, 30), num_pieces=15)
        assert len(em.particles) == 15
        assert em.gravity_scale == 1.0

    def test_blood_splatter_direction(self):
        em = blood_splatter((0, 0), (1, 0))
        assert len(em.particles) > 0
        assert em.one_shot is True

    def test_blood_splatter_zero_direction(self):
        em = blood_splatter((10, 10), (0, 0))
        assert len(em.particles) > 0

    def test_fire_continuous(self):
        em = fire((20, 20), size=8.0)
        assert em.emit_rate > 0
        assert em.gravity_scale < 0  # updraft

    def test_sparks_direction(self):
        em = sparks((0, 0), (1, 1))
        assert len(em.particles) > 0
        assert em.one_shot is True

    def test_sparks_zero_direction(self):
        em = sparks((10, 10), (0, 0))
        assert len(em.particles) > 0


# ---------------------------------------------------------------------------
# EffectsManager
# ---------------------------------------------------------------------------

class TestEffectsManager:
    def test_add_and_count(self):
        mgr = EffectsManager()
        mgr.add(explosion((0, 0)))
        mgr.add(smoke((50, 50)))
        assert mgr.active_count() == 2

    def test_tick_removes_finished(self):
        mgr = EffectsManager()
        em = explosion((0, 0), num_particles=5)
        mgr.add(em)
        # Advance past all particle lifetimes (max ~1.2s)
        for _ in range(30):
            mgr.tick(0.1)
        assert mgr.active_count() == 0

    def test_get_all_particles(self):
        mgr = EffectsManager()
        mgr.add(explosion((0, 0), num_particles=10))
        mgr.add(debris((10, 10), num_pieces=5))
        particles = mgr.get_all_particles()
        assert len(particles) == 15
        for p in particles:
            assert "x" in p and "y" in p

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

    def test_max_emitters_evicts_oldest(self):
        mgr = EffectsManager(max_emitters=3)
        for i in range(5):
            mgr.add(explosion((i * 10, 0)))
        assert mgr.active_count() == 3

    def test_to_three_js(self):
        mgr = EffectsManager()
        mgr.add(explosion((50, 50), num_particles=10))
        data = mgr.to_three_js()
        assert data["type"] == "particles"
        assert data["count"] == 10
        assert data["active_emitters"] == 1
        assert len(data["positions"]) == 10
        assert len(data["colors"]) == 10
        assert len(data["sizes"]) == 10
        assert len(data["ages"]) == 10
        # Positions should be [x, 0.0, y] format for Three.js
        for pos in data["positions"]:
            assert len(pos) == 3
            assert pos[1] == 0.0

    def test_to_three_js_caps_particles(self):
        mgr = EffectsManager()
        mgr.add(explosion((0, 0), num_particles=200))
        mgr.add(explosion((10, 10), num_particles=200))
        data = mgr.to_three_js(max_particles=50)
        assert data["count"] == 50
        assert data["total"] == 400

    def test_to_three_js_colors_are_hex(self):
        mgr = EffectsManager()
        mgr.add(explosion((0, 0), num_particles=5))
        data = mgr.to_three_js()
        for color in data["colors"]:
            assert color.startswith("#")
            assert len(color) == 7

    def test_tick_with_continuous_emitter(self):
        mgr = EffectsManager()
        mgr.add(fire((0, 0)))
        mgr.tick(0.1)
        assert mgr.total_particles() > 0
        mgr.tick(0.1)
        # Fire should keep generating particles
        assert mgr.active_count() == 1

    def test_multiple_effect_types_together(self):
        mgr = EffectsManager()
        mgr.add(explosion((0, 0)))
        mgr.add(smoke((10, 10)))
        mgr.add(fire((20, 20)))
        mgr.add(debris((30, 30)))
        assert mgr.active_count() == 4
        mgr.tick(0.05)
        assert mgr.total_particles() > 0

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Particle system for combat visual effects.

Produces particle data that the frontend renders (Canvas 2D or Three.js).
The backend computes positions/colors/lifetimes, frontend draws them.

Uses NumPy arrays internally for vectorized updates — 1000 particles
should tick in <5ms.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tritium_lib.sim_engine.debug.streams import DebugStream

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Vec2 = tuple[float, float] | np.ndarray


# ---------------------------------------------------------------------------
# Particle dataclass (individual particle state)
# ---------------------------------------------------------------------------
@dataclass
class Particle:
    """Single particle with position, velocity, color, size, and lifetime."""

    position: np.ndarray        # shape (2,)
    velocity: np.ndarray        # shape (2,)
    color: tuple[int, int, int, int]  # RGBA 0-255
    size: float
    lifetime: float             # seconds remaining
    max_lifetime: float

    @property
    def age_ratio(self) -> float:
        """0.0 = just born, 1.0 = about to die."""
        if self.max_lifetime <= 0:
            return 1.0
        return 1.0 - max(0.0, self.lifetime / self.max_lifetime)

    def alive(self) -> bool:
        return self.lifetime > 0.0

    def to_dict(self) -> dict:
        """Export for JSON serialization to frontend."""
        return {
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "vx": float(self.velocity[0]),
            "vy": float(self.velocity[1]),
            "r": self.color[0],
            "g": self.color[1],
            "b": self.color[2],
            "a": self.color[3],
            "size": self.size,
            "lifetime": self.lifetime,
            "max_lifetime": self.max_lifetime,
            "age": self.age_ratio,
        }


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------
def _lerp_color(
    c1: tuple[int, int, int, int],
    c2: tuple[int, int, int, int],
    t: float,
) -> tuple[int, int, int, int]:
    """Linear interpolation between two RGBA colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
        int(c1[3] + (c2[3] - c1[3]) * t),
    )


def _multi_lerp_color(
    colors: list[tuple[int, int, int, int]],
    t: float,
) -> tuple[int, int, int, int]:
    """Interpolate through a color gradient defined by multiple stops."""
    if len(colors) < 2:
        return colors[0] if colors else (255, 255, 255, 255)
    t = max(0.0, min(1.0, t))
    segment = t * (len(colors) - 1)
    idx = min(int(segment), len(colors) - 2)
    local_t = segment - idx
    return _lerp_color(colors[idx], colors[idx + 1], local_t)


# ---------------------------------------------------------------------------
# Particle Emitter
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 9.8], dtype=np.float64)  # positive Y = down


class ParticleEmitter:
    """Spawns and updates particles with configurable physics.

    Parameters
    ----------
    position : Vec2
        Emitter world position.
    spread : float
        Angular spread in radians for emission direction.
    base_angle : float
        Central emission angle (radians, 0 = right, pi/2 = down).
    speed_range : tuple[float, float]
        Min/max initial speed for emitted particles.
    lifetime_range : tuple[float, float]
        Min/max particle lifetime in seconds.
    size_range : tuple[float, float]
        Min/max initial particle size.
    color_start : RGBA tuple
        Color at birth.
    color_end : RGBA tuple
        Color at death (interpolated over lifetime).
    color_gradient : list of RGBA tuples
        If provided, overrides color_start/color_end with multi-stop gradient.
    gravity_scale : float
        Multiplier for gravity (0 = no gravity, 1 = normal).
    drag : float
        Velocity damping per second (0 = no drag, 1 = full stop).
    size_over_life : str
        "constant", "shrink", "grow_then_shrink", "grow".
    alpha_over_life : str
        "constant", "fade_out", "fade_in_out".
    emit_rate : float
        Continuous emission: particles per second (0 = manual emit only).
    emit_burst : int
        One-shot burst count on creation.
    max_particles : int
        Hard cap on particle count.
    one_shot : bool
        If True, emitter marks itself done after initial burst.
    position_jitter : float
        Random offset from emitter position on spawn.
    """

    def __init__(
        self,
        position: Vec2,
        *,
        spread: float = math.pi * 2,
        base_angle: float = 0.0,
        speed_range: tuple[float, float] = (10.0, 50.0),
        lifetime_range: tuple[float, float] = (0.5, 2.0),
        size_range: tuple[float, float] = (1.0, 4.0),
        color_start: tuple[int, int, int, int] = (255, 255, 255, 255),
        color_end: tuple[int, int, int, int] = (255, 255, 255, 0),
        color_gradient: Optional[list[tuple[int, int, int, int]]] = None,
        gravity_scale: float = 0.0,
        drag: float = 0.0,
        size_over_life: str = "shrink",
        alpha_over_life: str = "fade_out",
        emit_rate: float = 0.0,
        emit_burst: int = 0,
        max_particles: int = 512,
        one_shot: bool = False,
        position_jitter: float = 0.0,
    ):
        self.position = np.array(position, dtype=np.float64)
        self.spread = spread
        self.base_angle = base_angle
        self.speed_range = speed_range
        self.lifetime_range = lifetime_range
        self.size_range = size_range
        self.color_start = color_start
        self.color_end = color_end
        self.color_gradient = color_gradient
        self.gravity_scale = gravity_scale
        self.drag = drag
        self.size_over_life = size_over_life
        self.alpha_over_life = alpha_over_life
        self.emit_rate = emit_rate
        self.max_particles = max_particles
        self.one_shot = one_shot
        self.position_jitter = position_jitter

        self.particles: list[Particle] = []
        self._emit_accumulator = 0.0
        self.done = False
        self.elapsed = 0.0

        if emit_burst > 0:
            self.emit(emit_burst)
            if one_shot:
                self.done = True

    # -- Emission -----------------------------------------------------------

    def emit(self, count: int) -> list[Particle]:
        """Spawn *count* new particles and return them."""
        spawned: list[Particle] = []
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            angle = self.base_angle + random.uniform(
                -self.spread / 2, self.spread / 2
            )
            speed = random.uniform(*self.speed_range)
            vel = np.array(
                [math.cos(angle) * speed, math.sin(angle) * speed],
                dtype=np.float64,
            )
            lt = random.uniform(*self.lifetime_range)
            sz = random.uniform(*self.size_range)
            jitter = np.array(
                [
                    random.uniform(-self.position_jitter, self.position_jitter),
                    random.uniform(-self.position_jitter, self.position_jitter),
                ],
                dtype=np.float64,
            )
            p = Particle(
                position=self.position.copy() + jitter,
                velocity=vel,
                color=self.color_start,
                size=sz,
                lifetime=lt,
                max_lifetime=lt,
            )
            self.particles.append(p)
            spawned.append(p)
        return spawned

    # -- Update -------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """Advance all particles by *dt* seconds, remove dead ones."""
        self.elapsed += dt

        # Continuous emission
        if self.emit_rate > 0 and not self.done:
            self._emit_accumulator += self.emit_rate * dt
            burst = int(self._emit_accumulator)
            if burst > 0:
                self.emit(burst)
                self._emit_accumulator -= burst

        grav = GRAVITY * self.gravity_scale * dt
        drag_factor = max(0.0, 1.0 - self.drag * dt)

        alive: list[Particle] = []
        for p in self.particles:
            p.lifetime -= dt
            if p.lifetime <= 0:
                continue

            # Physics
            p.velocity += grav
            p.velocity *= drag_factor
            p.position += p.velocity * dt

            # Visual interpolation
            age = p.age_ratio

            # Color
            if self.color_gradient:
                base = _multi_lerp_color(self.color_gradient, age)
            else:
                base = _lerp_color(self.color_start, self.color_end, age)

            # Alpha over life
            if self.alpha_over_life == "fade_out":
                alpha = int(base[3] * (1.0 - age))
            elif self.alpha_over_life == "fade_in_out":
                alpha = int(base[3] * (1.0 - abs(2.0 * age - 1.0)))
            else:
                alpha = base[3]
            p.color = (base[0], base[1], base[2], max(0, min(255, alpha)))

            # Size over life
            if self.size_over_life == "shrink":
                p.size = random.uniform(*self.size_range) * (1.0 - age)
            elif self.size_over_life == "grow":
                p.size = random.uniform(*self.size_range) * (0.2 + 0.8 * age)
            elif self.size_over_life == "grow_then_shrink":
                curve = 1.0 - abs(2.0 * age - 1.0)
                p.size = random.uniform(*self.size_range) * curve
            # else: constant — keep original size

            alive.append(p)

        self.particles = alive

    # -- Export -------------------------------------------------------------

    def get_particles(self) -> list[dict]:
        """Export all living particles as dicts for frontend rendering."""
        return [p.to_dict() for p in self.particles]

    def is_finished(self) -> bool:
        """True when one-shot emitter has no living particles."""
        return self.done and len(self.particles) == 0


# ---------------------------------------------------------------------------
# Pre-built effect factories
# ---------------------------------------------------------------------------

def explosion(
    position: Vec2,
    radius: float = 10.0,
    num_particles: int = 50,
) -> ParticleEmitter:
    """Fireball + debris + smoke ring."""
    return ParticleEmitter(
        position,
        spread=math.pi * 2,
        speed_range=(radius * 2, radius * 8),
        lifetime_range=(0.3, 1.2),
        size_range=(2.0, radius * 0.8),
        color_gradient=[
            (255, 255, 200, 255),   # white-hot center
            (255, 200, 50, 255),    # yellow
            (255, 120, 20, 255),    # orange
            (200, 50, 10, 200),     # red
            (80, 80, 80, 100),      # dark smoke
            (40, 40, 40, 0),        # fade out
        ],
        gravity_scale=0.3,
        drag=0.4,
        size_over_life="grow_then_shrink",
        alpha_over_life="fade_out",
        emit_burst=num_particles,
        one_shot=True,
        position_jitter=radius * 0.3,
    )


def muzzle_flash(position: Vec2, heading: float) -> ParticleEmitter:
    """Brief bright flash at barrel."""
    return ParticleEmitter(
        position,
        spread=math.pi / 6,  # narrow cone
        base_angle=heading,
        speed_range=(40.0, 120.0),
        lifetime_range=(0.03, 0.12),
        size_range=(2.0, 6.0),
        color_gradient=[
            (255, 255, 240, 255),   # white flash
            (255, 220, 100, 255),   # yellow
            (255, 150, 30, 200),    # orange
            (255, 100, 10, 0),      # fade
        ],
        gravity_scale=0.0,
        drag=0.2,
        size_over_life="shrink",
        alpha_over_life="fade_out",
        emit_burst=12,
        one_shot=True,
        position_jitter=1.0,
    )


def tracer(
    start: Vec2,
    end: Vec2,
    speed: float = 300.0,
) -> ParticleEmitter:
    """Glowing line from shooter to target."""
    s = np.array(start, dtype=np.float64)
    e = np.array(end, dtype=np.float64)
    delta = e - s
    dist = float(np.linalg.norm(delta))
    if dist < 0.001:
        heading = 0.0
    else:
        heading = float(math.atan2(delta[1], delta[0]))

    travel_time = dist / speed if speed > 0 else 0.1

    return ParticleEmitter(
        start,
        spread=0.02,  # nearly straight line
        base_angle=heading,
        speed_range=(speed * 0.95, speed * 1.05),
        lifetime_range=(travel_time * 0.8, travel_time * 1.2),
        size_range=(1.0, 2.0),
        color_start=(255, 200, 50, 255),   # bright yellow
        color_end=(255, 100, 10, 0),
        gravity_scale=0.0,
        drag=0.0,
        size_over_life="shrink",
        alpha_over_life="fade_out",
        emit_burst=8,
        one_shot=True,
        position_jitter=0.5,
    )


def smoke(
    position: Vec2,
    duration: float = 5.0,
) -> ParticleEmitter:
    """Rising smoke cloud — grows then fades."""
    return ParticleEmitter(
        position,
        spread=math.pi / 3,
        base_angle=-math.pi / 2,  # upward
        speed_range=(5.0, 20.0),
        lifetime_range=(1.5, 4.0),
        size_range=(3.0, 12.0),
        color_gradient=[
            (180, 180, 180, 200),   # light gray
            (120, 120, 120, 160),   # mid gray
            (80, 80, 80, 80),       # dark gray
            (60, 60, 60, 0),        # fade
        ],
        gravity_scale=-0.1,  # slight updraft
        drag=0.3,
        size_over_life="grow",
        alpha_over_life="fade_out",
        emit_rate=15.0,
        max_particles=256,
        position_jitter=3.0,
    )


def debris(
    position: Vec2,
    num_pieces: int = 20,
) -> ParticleEmitter:
    """Chunks flying outward with gravity."""
    return ParticleEmitter(
        position,
        spread=math.pi * 2,
        speed_range=(30.0, 100.0),
        lifetime_range=(0.8, 2.5),
        size_range=(1.5, 5.0),
        color_gradient=[
            (160, 140, 100, 255),   # sandy brown
            (120, 100, 70, 255),    # darker
            (80, 70, 50, 200),      # dark brown
            (60, 50, 40, 0),        # fade
        ],
        gravity_scale=1.0,  # full gravity
        drag=0.1,
        size_over_life="constant",
        alpha_over_life="fade_out",
        emit_burst=num_pieces,
        one_shot=True,
        position_jitter=2.0,
    )


def blood_splatter(
    position: Vec2,
    direction: Vec2,
) -> ParticleEmitter:
    """Impact spray in direction of hit."""
    d = np.array(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm < 0.001:
        heading = 0.0
    else:
        heading = float(math.atan2(d[1], d[0]))

    return ParticleEmitter(
        position,
        spread=math.pi / 4,  # 45-degree cone
        base_angle=heading,
        speed_range=(20.0, 80.0),
        lifetime_range=(0.2, 0.8),
        size_range=(1.0, 4.0),
        color_gradient=[
            (220, 20, 20, 255),     # bright red
            (180, 10, 10, 240),     # darker red
            (120, 5, 5, 180),       # deep red
            (80, 0, 0, 0),          # fade
        ],
        gravity_scale=0.8,
        drag=0.3,
        size_over_life="shrink",
        alpha_over_life="fade_out",
        emit_burst=15,
        one_shot=True,
        position_jitter=1.5,
    )


def fire(
    position: Vec2,
    size: float = 5.0,
) -> ParticleEmitter:
    """Flickering fire effect — yellow to orange to red to black."""
    return ParticleEmitter(
        position,
        spread=math.pi / 4,
        base_angle=-math.pi / 2,  # upward
        speed_range=(8.0, 25.0),
        lifetime_range=(0.3, 1.0),
        size_range=(size * 0.3, size),
        color_gradient=[
            (255, 255, 150, 255),   # yellow core
            (255, 200, 50, 255),    # yellow-orange
            (255, 120, 20, 230),    # orange
            (200, 50, 10, 180),     # red
            (40, 10, 5, 0),         # black / fade
        ],
        gravity_scale=-0.2,  # updraft
        drag=0.2,
        size_over_life="grow_then_shrink",
        alpha_over_life="fade_out",
        emit_rate=30.0,
        max_particles=200,
        position_jitter=size * 0.4,
    )


def sparks(
    position: Vec2,
    direction: Vec2,
) -> ParticleEmitter:
    """Ricochet sparks — bright, fast, short-lived."""
    d = np.array(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm < 0.001:
        heading = 0.0
    else:
        heading = float(math.atan2(d[1], d[0]))

    return ParticleEmitter(
        position,
        spread=math.pi / 3,
        base_angle=heading,
        speed_range=(50.0, 150.0),
        lifetime_range=(0.1, 0.5),
        size_range=(0.5, 2.0),
        color_gradient=[
            (255, 255, 240, 255),   # white-hot
            (255, 220, 100, 255),   # yellow
            (255, 150, 30, 200),    # orange
            (200, 80, 10, 0),       # fade
        ],
        gravity_scale=0.6,
        drag=0.1,
        size_over_life="shrink",
        alpha_over_life="fade_out",
        emit_burst=10,
        one_shot=True,
        position_jitter=0.5,
    )


# ---------------------------------------------------------------------------
# Effects Manager
# ---------------------------------------------------------------------------

class EffectsManager:
    """Manages all active particle emitters, ticks them, culls dead ones.

    Usage::

        mgr = EffectsManager()
        mgr.add(explosion((100, 200)))
        mgr.add(smoke((150, 250)))

        # Game loop
        while running:
            mgr.tick(dt)
            particles = mgr.get_all_particles()  # send to frontend
    """

    def __init__(self, max_emitters: int = 256):
        self.max_emitters = max_emitters
        self.emitters: list[ParticleEmitter] = []

        # Debug data stream (disabled by default, zero overhead)
        self.debug = DebugStream("effects")

    def add(self, emitter: ParticleEmitter) -> ParticleEmitter:
        """Add an emitter. Drops oldest if at capacity."""
        if len(self.emitters) >= self.max_emitters:
            self.emitters.pop(0)
        self.emitters.append(emitter)
        return emitter

    def tick(self, dt: float) -> None:
        """Update all emitters and remove finished ones."""
        alive: list[ParticleEmitter] = []
        for em in self.emitters:
            em.tick(dt)
            if not em.is_finished():
                alive.append(em)
        self.emitters = alive

        # Emit debug data
        if self.debug.enabled:
            frame = self.debug.begin_frame()
            if frame is not None:
                frame.entries.append({
                    "type": "effects_summary",
                    "active_emitters": len(self.emitters),
                    "total_particles": self.total_particles(),
                })
                for idx, em in enumerate(self.emitters):
                    frame.entries.append({
                        "type": "emitter",
                        "id": idx,
                        "pos": em.position.tolist(),
                        "particle_count": len(em.particles),
                        "done": em.done,
                        "elapsed": round(em.elapsed, 3),
                    })
                self.debug.end_frame(frame)

    def get_all_particles(self) -> list[dict]:
        """Export every living particle from all emitters."""
        result: list[dict] = []
        for em in self.emitters:
            result.extend(em.get_particles())
        return result

    def active_count(self) -> int:
        """Number of currently active emitters."""
        return len(self.emitters)

    def total_particles(self) -> int:
        """Total living particles across all emitters."""
        return sum(len(em.particles) for em in self.emitters)

    def clear(self) -> None:
        """Remove all emitters and particles."""
        self.emitters.clear()

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Wave spawner and enemy composition designer for the sim engine.

Provides a wave-by-wave difficulty designer that scales enemy composition
based on wave number and difficulty curve, plus a spawner engine that
distributes units across spawn points with configurable spatial patterns.

Usage::

    from tritium_lib.sim_engine.spawner import (
        WaveDesigner, SpawnerEngine, SpawnPoint, SpawnPattern,
        DIFFICULTY_CURVES, WAVE_PRESETS,
    )

    designer = WaveDesigner()
    comp = designer.design_wave(wave_number=3, difficulty_curve="exponential", budget=100)
    engine = SpawnerEngine()
    engine.add_spawn_point(SpawnPoint(position=(100, 0), radius=20, alliance="hostile"))
    spawned = engine.spawn_wave(comp, pattern=SpawnPattern.CLUSTER)

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from tritium_lib.sim_engine.ai.steering import Vec2, distance, normalize, _sub, _add, _scale


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SpawnPattern(Enum):
    """Spatial distribution patterns for spawning units."""

    RANDOM = "random"
    CLUSTER = "cluster"
    LINE = "line"
    SURROUND = "surround"
    FLANKING = "flanking"
    WAVES = "waves"
    TRICKLE = "trickle"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UnitTemplate:
    """A single unit entry in a composition."""

    template: str
    count: int
    equipment: list[str] = field(default_factory=list)


@dataclass
class EnemyComposition:
    """Describes the full enemy force for a wave.

    ``total_count`` is computed from the sum of template counts.
    ``difficulty_rating`` is a 0-100 score indicating overall lethality.
    """

    templates: list[dict] = field(default_factory=list)
    difficulty_rating: float = 0.0

    @property
    def total_count(self) -> int:
        """Total units across all template entries."""
        return sum(t.get("count", 0) for t in self.templates)

    def to_dict(self) -> dict:
        """Serialize for JSON / Three.js transport."""
        return {
            "templates": self.templates,
            "total_count": self.total_count,
            "difficulty_rating": self.difficulty_rating,
        }


@dataclass
class SpawnPoint:
    """A location where units can be spawned."""

    position: Vec2
    radius: float = 10.0
    alliance: str = "hostile"
    active: bool = True
    cooldown: float = 0.0
    _cooldown_remaining: float = 0.0

    def is_ready(self) -> bool:
        """True if this point is active and off cooldown."""
        return self.active and self._cooldown_remaining <= 0.0

    def trigger_cooldown(self) -> None:
        """Start the cooldown timer after a spawn."""
        self._cooldown_remaining = self.cooldown

    def tick_cooldown(self, dt: float) -> None:
        """Advance cooldown timer by *dt* seconds."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining = max(0.0, self._cooldown_remaining - dt)


# ---------------------------------------------------------------------------
# Difficulty curves
# ---------------------------------------------------------------------------


def _curve_linear(wave: int, base: float) -> float:
    return base * wave


def _curve_exponential(wave: int, base: float) -> float:
    return base * (1.3 ** wave)


def _curve_logarithmic(wave: int, base: float) -> float:
    return base * (1.0 + math.log(max(wave, 1)))


def _curve_staircase(wave: int, base: float) -> float:
    """Jumps every 3 waves."""
    step = (wave - 1) // 3 + 1
    return base * step


def _curve_random_spikes(wave: int, base: float) -> float:
    """Linear with random +-30% spikes."""
    spike = random.uniform(0.7, 1.3)
    return base * wave * spike


DIFFICULTY_CURVES: dict[str, Callable[[int, float], float]] = {
    "linear": _curve_linear,
    "exponential": _curve_exponential,
    "logarithmic": _curve_logarithmic,
    "staircase": _curve_staircase,
    "random_spikes": _curve_random_spikes,
}


# ---------------------------------------------------------------------------
# Difficulty weights per template (used for budget allocation)
# ---------------------------------------------------------------------------

_TEMPLATE_COST: dict[str, float] = {
    "infantry": 1.0,
    "scout": 1.2,
    "medic": 1.5,
    "sniper": 2.5,
    "heavy": 3.0,
    "drone": 2.0,
    "turret": 4.0,
    "vehicle": 5.0,       # generic armored vehicle
    "helicopter": 8.0,
    "tank": 10.0,
}

_TEMPLATE_DIFFICULTY: dict[str, float] = {
    "infantry": 1.0,
    "scout": 1.5,
    "medic": 0.8,
    "sniper": 3.0,
    "heavy": 4.0,
    "drone": 2.5,
    "turret": 5.0,
    "vehicle": 6.0,
    "helicopter": 9.0,
    "tank": 12.0,
}


# ---------------------------------------------------------------------------
# WaveDesigner
# ---------------------------------------------------------------------------


class WaveDesigner:
    """Designs enemy compositions that scale with wave number and difficulty curve.

    The designer uses a budget system: each wave gets a budget proportional to
    the difficulty curve, then allocates units from the available pool for that
    wave tier.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # -- public API --

    def design_wave(
        self,
        wave_number: int,
        difficulty_curve: str = "linear",
        budget: float = 10.0,
    ) -> EnemyComposition:
        """Create an ``EnemyComposition`` for *wave_number*.

        Parameters
        ----------
        wave_number:
            1-based wave index.  Higher waves unlock tougher unit types.
        difficulty_curve:
            One of the keys in :data:`DIFFICULTY_CURVES`.
        budget:
            Base budget per wave (scaled by the curve).

        Returns
        -------
        EnemyComposition
            Fully specified composition with difficulty rating.
        """
        curve_fn = DIFFICULTY_CURVES.get(difficulty_curve, _curve_linear)
        effective_budget = curve_fn(wave_number, budget)

        pool = self._available_pool(wave_number)
        templates = self._allocate_budget(effective_budget, pool, wave_number)
        difficulty = sum(
            t["count"] * _TEMPLATE_DIFFICULTY.get(t["template"], 1.0)
            for t in templates
        )
        return EnemyComposition(templates=templates, difficulty_rating=round(difficulty, 2))

    def generate_spawn_positions(
        self,
        pattern: SpawnPattern,
        center: Vec2,
        radius: float,
        count: int,
    ) -> list[Vec2]:
        """Generate *count* positions arranged by *pattern* around *center*.

        Parameters
        ----------
        pattern:
            Spatial pattern to use.
        center:
            Center point (x, y) in meters.
        radius:
            Spread radius in meters.
        count:
            Number of positions to generate.

        Returns
        -------
        list[Vec2]
            List of (x, y) positions.
        """
        if count <= 0:
            return []
        if pattern == SpawnPattern.RANDOM:
            return self._pattern_random(center, radius, count)
        elif pattern == SpawnPattern.CLUSTER:
            return self._pattern_cluster(center, radius, count)
        elif pattern == SpawnPattern.LINE:
            return self._pattern_line(center, radius, count)
        elif pattern == SpawnPattern.SURROUND:
            return self._pattern_surround(center, radius, count)
        elif pattern == SpawnPattern.FLANKING:
            return self._pattern_flanking(center, radius, count)
        elif pattern == SpawnPattern.WAVES:
            return self._pattern_waves(center, radius, count)
        elif pattern == SpawnPattern.TRICKLE:
            return self._pattern_trickle(center, radius, count)
        return self._pattern_random(center, radius, count)

    # -- wave tier unlocks --

    def _available_pool(self, wave_number: int) -> list[str]:
        """Return template names available at this wave tier."""
        pool = ["infantry"]
        if wave_number >= 2:
            pool.extend(["scout", "medic"])
        if wave_number >= 3:
            pool.append("vehicle")
        if wave_number >= 4:
            pool.append("drone")
        if wave_number >= 5:
            pool.extend(["sniper", "heavy"])
        if wave_number >= 7:
            pool.append("turret")
        if wave_number >= 8:
            pool.append("helicopter")
        if wave_number >= 10:
            pool.append("tank")
        return pool

    # -- budget allocation --

    def _allocate_budget(
        self,
        budget: float,
        pool: list[str],
        wave_number: int,
    ) -> list[dict]:
        """Spend *budget* on units from *pool*, preferring variety."""
        remaining = budget
        result: dict[str, dict] = {}

        # Always include at least some infantry
        infantry_count = max(2, int(remaining * 0.3 / _TEMPLATE_COST["infantry"]))
        infantry_cost = infantry_count * _TEMPLATE_COST["infantry"]
        if infantry_cost > remaining:
            infantry_count = max(1, int(remaining / _TEMPLATE_COST["infantry"]))
            infantry_cost = infantry_count * _TEMPLATE_COST["infantry"]
        result["infantry"] = {
            "template": "infantry",
            "count": infantry_count,
            "equipment": ["rifle"],
        }
        remaining -= infantry_cost

        # Fill remaining budget with other available types
        other_pool = [t for t in pool if t != "infantry"]
        self._rng.shuffle(other_pool)

        for template in other_pool:
            if remaining <= 0:
                break
            cost = _TEMPLATE_COST.get(template, 1.0)
            max_affordable = int(remaining / cost)
            if max_affordable <= 0:
                continue

            # Scale count: expensive units come in smaller numbers
            if cost >= 8:
                count = min(max_affordable, self._rng.randint(1, 2))
            elif cost >= 4:
                count = min(max_affordable, self._rng.randint(1, 3))
            elif cost >= 2:
                count = min(max_affordable, self._rng.randint(1, max(2, wave_number // 2)))
            else:
                count = min(max_affordable, self._rng.randint(1, max(3, wave_number)))

            equipment = self._default_equipment(template)
            result[template] = {
                "template": template,
                "count": count,
                "equipment": equipment,
            }
            remaining -= count * cost

        return list(result.values())

    def _default_equipment(self, template: str) -> list[str]:
        """Return default equipment for a template."""
        defaults: dict[str, list[str]] = {
            "infantry": ["rifle"],
            "scout": ["smg", "binoculars"],
            "medic": ["pistol", "medkit"],
            "sniper": ["sniper_rifle"],
            "heavy": ["lmg", "body_armor"],
            "drone": ["camera"],
            "turret": ["hmg"],
            "vehicle": ["mounted_gun", "armor_plating"],
            "helicopter": ["rockets", "minigun"],
            "tank": ["cannon", "coaxial_mg", "reactive_armor"],
        }
        return defaults.get(template, [])

    # -- pattern generators --

    def _pattern_random(self, center: Vec2, radius: float, count: int) -> list[Vec2]:
        positions: list[Vec2] = []
        for _ in range(count):
            angle = self._rng.uniform(0, 2 * math.pi)
            r = self._rng.uniform(0, radius)
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            positions.append((x, y))
        return positions

    def _pattern_cluster(self, center: Vec2, radius: float, count: int) -> list[Vec2]:
        """Tight cluster using gaussian distribution (sigma = radius/3)."""
        positions: list[Vec2] = []
        sigma = radius / 3.0
        for _ in range(count):
            x = center[0] + self._rng.gauss(0, sigma)
            y = center[1] + self._rng.gauss(0, sigma)
            positions.append((x, y))
        return positions

    def _pattern_line(self, center: Vec2, radius: float, count: int) -> list[Vec2]:
        """Units arranged in a horizontal line centered at *center*."""
        if count == 1:
            return [center]
        spacing = (2 * radius) / (count - 1)
        start_x = center[0] - radius
        return [(start_x + i * spacing, center[1]) for i in range(count)]

    def _pattern_surround(self, center: Vec2, radius: float, count: int) -> list[Vec2]:
        """Units evenly spaced around a circle."""
        positions: list[Vec2] = []
        for i in range(count):
            angle = (2 * math.pi * i) / count
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            positions.append((x, y))
        return positions

    def _pattern_flanking(self, center: Vec2, radius: float, count: int) -> list[Vec2]:
        """Two groups on opposite sides (left/right flanks)."""
        left_count = count // 2
        right_count = count - left_count
        positions: list[Vec2] = []
        # Left flank
        left_center = (center[0] - radius, center[1])
        for i in range(left_count):
            offset_y = (i - left_count / 2) * (radius / max(left_count, 1))
            positions.append((left_center[0] + self._rng.uniform(-radius * 0.2, radius * 0.2),
                              left_center[1] + offset_y))
        # Right flank
        right_center = (center[0] + radius, center[1])
        for i in range(right_count):
            offset_y = (i - right_count / 2) * (radius / max(right_count, 1))
            positions.append((right_center[0] + self._rng.uniform(-radius * 0.2, radius * 0.2),
                              right_center[1] + offset_y))
        return positions

    def _pattern_waves(self, center: Vec2, radius: float, count: int) -> list[Vec2]:
        """Multiple rows advancing from *center*, spaced by radius/3."""
        row_size = max(3, int(math.sqrt(count)))
        positions: list[Vec2] = []
        row = 0
        placed = 0
        row_spacing = radius / 3.0
        while placed < count:
            in_this_row = min(row_size, count - placed)
            col_spacing = (2 * radius) / max(in_this_row, 1)
            start_x = center[0] - radius
            y = center[1] + row * row_spacing
            for c in range(in_this_row):
                x = start_x + c * col_spacing + self._rng.uniform(-col_spacing * 0.1, col_spacing * 0.1)
                positions.append((x, y))
                placed += 1
            row += 1
        return positions

    def _pattern_trickle(self, center: Vec2, radius: float, count: int) -> list[Vec2]:
        """Single file, evenly spaced along a line from center outward."""
        if count == 1:
            return [center]
        angle = self._rng.uniform(0, 2 * math.pi)
        dx = math.cos(angle)
        dy = math.sin(angle)
        spacing = (2 * radius) / count
        positions: list[Vec2] = []
        for i in range(count):
            dist = i * spacing
            x = center[0] + dx * dist
            y = center[1] + dy * dist
            positions.append((x, y))
        return positions


# ---------------------------------------------------------------------------
# SpawnerEngine
# ---------------------------------------------------------------------------


class SpawnerEngine:
    """Manages spawn points, queues units, and distributes them over time.

    The engine holds a list of spawn points and a queue of pending spawns.
    Call ``spawn_wave`` to enqueue a full composition, then ``tick(dt)`` to
    release units according to timing.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.spawn_points: list[SpawnPoint] = []
        self.spawn_queue: list[dict] = []
        self._spawned: list[dict] = []
        self._rng = random.Random(seed)
        self._designer = WaveDesigner(seed)
        self._time_accumulator: float = 0.0
        self._spawn_interval: float = 0.5  # seconds between trickle spawns

    # -- spawn point management --

    def add_spawn_point(self, sp: SpawnPoint) -> None:
        """Register a spawn point."""
        self.spawn_points.append(sp)

    def remove_spawn_point(self, index: int) -> SpawnPoint | None:
        """Remove spawn point by index, returns removed point or None."""
        if 0 <= index < len(self.spawn_points):
            return self.spawn_points.pop(index)
        return None

    def active_spawn_points(self) -> list[SpawnPoint]:
        """Return all spawn points that are active and off cooldown."""
        return [sp for sp in self.spawn_points if sp.is_ready()]

    # -- wave spawning --

    def spawn_wave(
        self,
        composition: EnemyComposition,
        spawn_points: list[SpawnPoint] | None = None,
        pattern: SpawnPattern = SpawnPattern.RANDOM,
    ) -> list[dict]:
        """Distribute a full composition across spawn points immediately.

        Parameters
        ----------
        composition:
            The enemy composition to spawn.
        spawn_points:
            Specific points to use.  If None, uses all registered active points.
        pattern:
            Spatial pattern for positioning within each spawn point's radius.

        Returns
        -------
        list[dict]
            List of spawned unit descriptors with position, template, equipment.
        """
        points = spawn_points or self.active_spawn_points()
        if not points:
            # Fallback: spawn at origin
            points = [SpawnPoint(position=(0.0, 0.0), radius=20.0, alliance="hostile")]

        spawned: list[dict] = []
        unit_index = 0

        for entry in composition.templates:
            template = entry["template"]
            count = entry.get("count", 1)
            equipment = entry.get("equipment", [])

            # Distribute units round-robin across spawn points
            for i in range(count):
                sp = points[unit_index % len(points)]
                # Generate position within spawn point radius using pattern
                positions = self._designer.generate_spawn_positions(
                    pattern, sp.position, sp.radius, 1,
                )
                pos = positions[0] if positions else sp.position

                unit_desc = {
                    "unit_id": f"spawn_{template}_{unit_index}",
                    "template": template,
                    "equipment": equipment,
                    "position": pos,
                    "alliance": sp.alliance,
                    "spawn_point_index": unit_index % len(points),
                }
                spawned.append(unit_desc)
                sp.trigger_cooldown()
                unit_index += 1

        self._spawned.extend(spawned)
        return spawned

    def enqueue(self, unit_desc: dict) -> None:
        """Add a single unit descriptor to the spawn queue for timed release."""
        self.spawn_queue.append(unit_desc)

    def enqueue_wave(
        self,
        composition: EnemyComposition,
        spawn_points: list[SpawnPoint] | None = None,
        pattern: SpawnPattern = SpawnPattern.RANDOM,
        interval: float = 0.5,
    ) -> int:
        """Enqueue a composition for gradual spawning via ``tick()``.

        Returns the number of units enqueued.
        """
        self._spawn_interval = interval
        points = spawn_points or self.active_spawn_points()
        if not points:
            points = [SpawnPoint(position=(0.0, 0.0), radius=20.0, alliance="hostile")]

        count = 0
        unit_index = len(self._spawned)
        for entry in composition.templates:
            template = entry["template"]
            n = entry.get("count", 1)
            equipment = entry.get("equipment", [])
            for i in range(n):
                sp = points[(unit_index + count) % len(points)]
                positions = self._designer.generate_spawn_positions(
                    pattern, sp.position, sp.radius, 1,
                )
                pos = positions[0] if positions else sp.position
                self.spawn_queue.append({
                    "unit_id": f"spawn_{template}_{unit_index + count}",
                    "template": template,
                    "equipment": equipment,
                    "position": pos,
                    "alliance": sp.alliance,
                })
                count += 1
        return count

    def tick(self, dt: float) -> list[dict]:
        """Advance time by *dt* seconds, releasing queued units at the spawn interval.

        Returns a list of unit descriptors that spawned this tick.
        """
        released: list[dict] = []

        # Advance cooldowns on all spawn points
        for sp in self.spawn_points:
            sp.tick_cooldown(dt)

        if not self.spawn_queue:
            return released

        self._time_accumulator += dt
        while self._time_accumulator >= self._spawn_interval and self.spawn_queue:
            unit = self.spawn_queue.pop(0)
            released.append(unit)
            self._spawned.append(unit)
            self._time_accumulator -= self._spawn_interval

        return released

    @property
    def total_spawned(self) -> int:
        """Total units spawned so far (immediate + ticked)."""
        return len(self._spawned)

    @property
    def queue_size(self) -> int:
        """Units still waiting in the queue."""
        return len(self.spawn_queue)

    # -- serialization --

    def to_three_js(self) -> dict:
        """Export spawn point markers for Three.js rendering.

        Returns a dict with ``spawn_points`` list, each containing
        position, radius, alliance, and active status for visualization.
        """
        markers = []
        for i, sp in enumerate(self.spawn_points):
            markers.append({
                "id": f"spawn_point_{i}",
                "position": {"x": sp.position[0], "y": 0, "z": sp.position[1]},
                "radius": sp.radius,
                "alliance": sp.alliance,
                "active": sp.active,
                "cooldown_remaining": round(sp._cooldown_remaining, 2),
                "color": _alliance_spawn_color(sp.alliance),
                "type": "spawn_point",
                "geometry": "ring",
            })
        return {
            "spawn_points": markers,
            "queue_size": self.queue_size,
            "total_spawned": self.total_spawned,
        }


def _alliance_spawn_color(alliance: str) -> str:
    """Map alliance to a hex color for spawn point visualization."""
    colors = {
        "hostile": "#ff2a6d",
        "friendly": "#05ffa1",
        "neutral": "#fcee0a",
        "unknown": "#00f0ff",
    }
    return colors.get(alliance, "#888888")


# ---------------------------------------------------------------------------
# Wave presets
# ---------------------------------------------------------------------------


WAVE_PRESETS: dict[str, dict[str, Any]] = {
    "easy_10": {
        "total_waves": 10,
        "difficulty_curve": "linear",
        "base_budget": 8.0,
        "description": "10 easy waves with linear difficulty.",
    },
    "medium_15": {
        "total_waves": 15,
        "difficulty_curve": "logarithmic",
        "base_budget": 12.0,
        "description": "15 medium waves with logarithmic scaling.",
    },
    "hard_20": {
        "total_waves": 20,
        "difficulty_curve": "exponential",
        "base_budget": 15.0,
        "description": "20 hard waves with exponential scaling.",
    },
    "endless": {
        "total_waves": -1,  # -1 = infinite
        "difficulty_curve": "linear",
        "base_budget": 10.0,
        "description": "Endless mode -- waves never stop, linear scaling.",
    },
    "boss_rush": {
        "total_waves": 5,
        "difficulty_curve": "staircase",
        "base_budget": 50.0,
        "description": "5 boss waves with massive budgets.",
    },
}


# ---------------------------------------------------------------------------
# Convenience: run a full preset
# ---------------------------------------------------------------------------


def run_preset(
    preset_name: str,
    max_waves: int | None = None,
    seed: int | None = None,
) -> list[EnemyComposition]:
    """Generate all wave compositions for a named preset.

    Parameters
    ----------
    preset_name:
        Key into :data:`WAVE_PRESETS`.
    max_waves:
        Override to cap the number of waves (useful for endless mode).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    list[EnemyComposition]
        One composition per wave.
    """
    preset = WAVE_PRESETS[preset_name]
    total = preset["total_waves"]
    if total == -1:
        total = max_waves or 20  # cap endless at 20 by default
    if max_waves is not None:
        total = min(total, max_waves)

    designer = WaveDesigner(seed=seed)
    return [
        designer.design_wave(
            wave_number=w,
            difficulty_curve=preset["difficulty_curve"],
            budget=preset["base_budget"],
        )
        for w in range(1, total + 1)
    ]

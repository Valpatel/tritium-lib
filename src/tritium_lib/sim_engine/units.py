"""Simulation unit/entity system.

Defines the things that exist in the combat sim — soldiers, vehicles, drones,
turrets, civilians.  Each unit has a type, alliance, base stats, and mutable
runtime state.  A factory function creates units from predefined templates.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from tritium_lib.sim_engine.ai.steering import Vec2, distance

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UnitType(Enum):
    """Categories of simulation entities."""

    INFANTRY = "infantry"
    SNIPER = "sniper"
    HEAVY = "heavy"
    MEDIC = "medic"
    ENGINEER = "engineer"
    SCOUT = "scout"
    VEHICLE = "vehicle"
    DRONE = "drone"
    TURRET = "turret"
    CIVILIAN = "civilian"


class Alliance(Enum):
    """Which side a unit fights for."""

    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Stats & state
# ---------------------------------------------------------------------------


@dataclass
class UnitStats:
    """Immutable baseline stats for a unit type."""

    max_health: float = 100.0
    armor: float = 0.0  # damage reduction 0-1
    speed: float = 5.0  # m/s base speed
    detection_range: float = 50.0  # meters
    attack_range: float = 30.0  # meters
    attack_damage: float = 10.0
    attack_cooldown: float = 1.0  # seconds between attacks
    accuracy: float = 0.7  # 0-1 hit probability at optimal range


@dataclass
class UnitState:
    """Mutable runtime state — changes every tick."""

    health: float = 100.0
    morale: float = 1.0  # 0-1, affects accuracy and flee threshold
    suppression: float = 0.0  # 0-1, incoming fire suppression
    ammo: int = -1  # -1 = unlimited
    is_alive: bool = True
    is_visible: bool = True
    status: str = "idle"  # idle, moving, attacking, retreating, dead, suppressed
    kill_count: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    last_attack_time: float = -999.0  # sim-time of last attack


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    """A single simulation entity."""

    unit_id: str
    name: str
    unit_type: UnitType
    alliance: Alliance
    position: Vec2
    heading: float = 0.0  # radians
    stats: UnitStats = field(default_factory=UnitStats)
    state: UnitState = field(default_factory=UnitState)
    weapon: str = "rifle"  # key into WEAPONS dict
    squad_id: str | None = None

    # -- queries ----------------------------------------------------------

    def is_alive(self) -> bool:
        """Return True if the unit is still alive."""
        return self.state.is_alive

    def effective_accuracy(self) -> float:
        """Accuracy factoring in morale and suppression."""
        return self.stats.accuracy * self.state.morale * (1.0 - self.state.suppression)

    def effective_speed(self) -> float:
        """Speed with morale modifier (low morale = slower, capped at 0.5x)."""
        morale_factor = 0.5 + 0.5 * self.state.morale  # range [0.5, 1.0]
        return self.stats.speed * morale_factor

    def distance_to(self, other: Unit) -> float:
        """Euclidean distance to another unit."""
        return distance(self.position, other.position)

    def can_see(self, other: Unit) -> bool:
        """Whether this unit can detect *other*."""
        if not other.state.is_visible:
            return False
        return self.distance_to(other) <= self.stats.detection_range

    def can_attack(self, sim_time: float = 0.0) -> bool:
        """Whether this unit can fire right now."""
        if not self.state.is_alive:
            return False
        if self.state.suppression >= 0.9:
            return False
        if self.state.ammo == 0:
            return False
        if sim_time - self.state.last_attack_time < self.stats.attack_cooldown:
            return False
        return True

    # -- mutations --------------------------------------------------------

    def take_damage(self, amount: float, source_dir: Vec2 | None = None) -> float:
        """Apply *amount* damage after armor reduction.  Returns actual damage dealt."""
        if not self.state.is_alive or amount <= 0:
            return 0.0
        actual = amount * (1.0 - self.stats.armor)
        actual = max(actual, 0.0)
        self.state.health -= actual
        self.state.damage_taken += actual
        if self.state.health <= 0:
            self.state.health = 0.0
            self.state.is_alive = False
            self.state.status = "dead"
        return actual

    def heal(self, amount: float) -> float:
        """Heal up to max_health.  Returns actual amount healed."""
        if not self.state.is_alive or amount <= 0:
            return 0.0
        headroom = self.stats.max_health - self.state.health
        actual = min(amount, headroom)
        self.state.health += actual
        return actual

    def apply_suppression(self, amount: float) -> None:
        """Increase suppression, clamped to [0, 1]."""
        self.state.suppression = max(0.0, min(1.0, self.state.suppression + amount))

    def recover_suppression(self, dt: float, rate: float = 0.3) -> None:
        """Decay suppression over time."""
        self.state.suppression = max(0.0, self.state.suppression - rate * dt)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

UNIT_TEMPLATES: dict[str, tuple[UnitType, UnitStats]] = {
    "infantry": (
        UnitType.INFANTRY,
        UnitStats(
            max_health=100.0, armor=0.1, speed=5.0, detection_range=50.0,
            attack_range=30.0, attack_damage=10.0, attack_cooldown=1.0, accuracy=0.7,
        ),
    ),
    "sniper": (
        UnitType.SNIPER,
        UnitStats(
            max_health=60.0, armor=0.0, speed=3.0, detection_range=100.0,
            attack_range=80.0, attack_damage=25.0, attack_cooldown=3.0, accuracy=0.9,
        ),
    ),
    "heavy": (
        UnitType.HEAVY,
        UnitStats(
            max_health=200.0, armor=0.4, speed=3.0, detection_range=40.0,
            attack_range=25.0, attack_damage=20.0, attack_cooldown=1.0, accuracy=0.7,
        ),
    ),
    "medic": (
        UnitType.MEDIC,
        UnitStats(
            max_health=100.0, armor=0.1, speed=5.5, detection_range=50.0,
            attack_range=20.0, attack_damage=8.0, attack_cooldown=1.0, accuracy=0.7,
        ),
    ),
    "scout": (
        UnitType.SCOUT,
        UnitStats(
            max_health=70.0, armor=0.0, speed=8.0, detection_range=80.0,
            attack_range=20.0, attack_damage=10.0, attack_cooldown=1.0, accuracy=0.7,
        ),
    ),
    "drone": (
        UnitType.DRONE,
        UnitStats(
            max_health=30.0, armor=0.0, speed=12.0, detection_range=100.0,
            attack_range=40.0, attack_damage=8.0, attack_cooldown=1.0, accuracy=0.7,
        ),
    ),
    "turret": (
        UnitType.TURRET,
        UnitStats(
            max_health=500.0, armor=0.6, speed=0.0, detection_range=60.0,
            attack_range=50.0, attack_damage=30.0, attack_cooldown=0.5, accuracy=0.7,
        ),
    ),
    "civilian": (
        UnitType.CIVILIAN,
        UnitStats(
            max_health=50.0, armor=0.0, speed=4.0, detection_range=20.0,
            attack_range=0.0, attack_damage=0.0, attack_cooldown=1.0, accuracy=0.0,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_unit(
    template: str,
    unit_id: str,
    name: str,
    alliance: Alliance,
    position: Vec2,
) -> Unit:
    """Create a unit from a template name.

    Raises ``KeyError`` if *template* is not in :data:`UNIT_TEMPLATES`.
    """
    unit_type, base_stats = UNIT_TEMPLATES[template]
    stats = UnitStats(
        max_health=base_stats.max_health,
        armor=base_stats.armor,
        speed=base_stats.speed,
        detection_range=base_stats.detection_range,
        attack_range=base_stats.attack_range,
        attack_damage=base_stats.attack_damage,
        attack_cooldown=base_stats.attack_cooldown,
        accuracy=base_stats.accuracy,
    )
    state = UnitState(health=stats.max_health)
    return Unit(
        unit_id=unit_id,
        name=name,
        unit_type=unit_type,
        alliance=alliance,
        position=position,
        stats=stats,
        state=state,
    )

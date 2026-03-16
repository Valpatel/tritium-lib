# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Naval combat module for the Tritium sim engine.

Ship physics with momentum and wide turning circles, torpedo homing,
gun batteries, submarine detection, sea state effects, and Three.js-compatible
serialization for real-time 3D rendering.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import enum
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    magnitude,
    normalize,
    heading_to_vec,
    _sub,
    _add,
    _scale,
)


# ---------------------------------------------------------------------------
# Ship classification
# ---------------------------------------------------------------------------

class ShipClass(enum.Enum):
    """Classification of naval vessel types."""
    PATROL_BOAT = "patrol_boat"
    FRIGATE = "frigate"
    DESTROYER = "destroyer"
    CRUISER = "cruiser"
    CARRIER = "carrier"
    SUBMARINE = "submarine"
    SPEEDBOAT = "speedboat"
    CARGO = "cargo"


# ---------------------------------------------------------------------------
# Ship state
# ---------------------------------------------------------------------------

@dataclass
class ShipState:
    """Mutable state of a single ship in the simulation."""

    ship_id: str
    name: str
    ship_class: ShipClass
    alliance: str

    position: Vec2
    heading: float          # radians, 0 = +x, CCW positive
    speed: float            # current speed m/s
    max_speed: float        # max speed m/s

    turn_rate: float        # max turn rate rad/s (ships turn slowly)
    health: float
    max_health: float
    armor: float            # 0-1, fraction of damage absorbed

    weapons: list[str] = field(default_factory=list)
    radar_range: float = 5000.0     # detection range meters
    sonar_range: float = 2000.0     # submarine detection range meters
    depth: float = 0.0              # negative = submerged
    is_submerged: bool = False
    wake_intensity: float = 0.0     # 0-1, computed from speed
    crew: int = 50

    # Internal physics state
    _throttle: float = 0.0
    _rudder: float = 0.0
    _velocity: float = 0.0         # actual velocity (momentum)

    def is_alive(self) -> bool:
        """Return True if the ship has positive health."""
        return self.health > 0.0

    def health_pct(self) -> float:
        """Return health as 0-1 fraction."""
        if self.max_health <= 0:
            return 0.0
        return max(0.0, min(1.0, self.health / self.max_health))


# ---------------------------------------------------------------------------
# Torpedo
# ---------------------------------------------------------------------------

@dataclass
class Torpedo:
    """A torpedo projectile in the water."""

    torpedo_id: str
    position: Vec2
    heading: float          # radians
    speed: float            # m/s
    target_id: str | None   # homing target ship_id, or None for dumb-fire
    damage: float
    range_remaining: float  # meters of fuel left
    source_ship_id: str = ""
    is_active: bool = True
    _trail: list[Vec2] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shell projectile (gun fire)
# ---------------------------------------------------------------------------

@dataclass
class ShellProjectile:
    """A gun shell in flight."""

    shell_id: str
    position: Vec2
    heading: float
    speed: float            # m/s
    damage: float
    range_remaining: float
    source_ship_id: str = ""
    is_active: bool = True


# ---------------------------------------------------------------------------
# Combat effect
# ---------------------------------------------------------------------------

@dataclass
class CombatEffect:
    """A visual/audio effect for Three.js rendering."""

    effect_type: str        # "splash", "explosion", "smoke", "wake", "sonar_ping"
    position: Vec2
    radius: float = 5.0
    duration: float = 1.0
    intensity: float = 1.0

    def to_dict(self) -> dict:
        return {
            "type": self.effect_type,
            "x": round(self.position[0], 2),
            "y": round(self.position[1], 2),
            "radius": round(self.radius, 2),
            "duration": round(self.duration, 2),
            "intensity": round(self.intensity, 2),
        }


# ---------------------------------------------------------------------------
# Ship templates — realistic-ish stats for 8 classes
# ---------------------------------------------------------------------------

SHIP_TEMPLATES: dict[ShipClass, dict] = {
    ShipClass.SPEEDBOAT: {
        "max_speed": 25.0,      # ~50 knots
        "turn_rate": 0.8,       # nimble
        "max_health": 200.0,
        "armor": 0.05,
        "weapons": ["mg_50cal"],
        "radar_range": 2000.0,
        "sonar_range": 0.0,
        "crew": 4,
    },
    ShipClass.PATROL_BOAT: {
        "max_speed": 18.0,      # ~35 knots
        "turn_rate": 0.5,
        "max_health": 500.0,
        "armor": 0.10,
        "weapons": ["gun_25mm", "mg_50cal"],
        "radar_range": 5000.0,
        "sonar_range": 1000.0,
        "crew": 25,
    },
    ShipClass.FRIGATE: {
        "max_speed": 15.0,      # ~30 knots
        "turn_rate": 0.25,
        "max_health": 1500.0,
        "armor": 0.20,
        "weapons": ["gun_76mm", "torpedo_tube", "ciws"],
        "radar_range": 15000.0,
        "sonar_range": 5000.0,
        "crew": 150,
    },
    ShipClass.DESTROYER: {
        "max_speed": 16.0,      # ~32 knots
        "turn_rate": 0.20,
        "max_health": 2500.0,
        "armor": 0.25,
        "weapons": ["gun_127mm", "torpedo_tube", "missile_vls", "ciws"],
        "radar_range": 25000.0,
        "sonar_range": 8000.0,
        "crew": 300,
    },
    ShipClass.CRUISER: {
        "max_speed": 16.0,      # ~32 knots
        "turn_rate": 0.15,
        "max_health": 4000.0,
        "armor": 0.35,
        "weapons": ["gun_127mm", "gun_127mm", "missile_vls", "torpedo_tube", "ciws", "ciws"],
        "radar_range": 40000.0,
        "sonar_range": 10000.0,
        "crew": 400,
    },
    ShipClass.CARRIER: {
        "max_speed": 15.0,      # ~30 knots
        "turn_rate": 0.08,      # very wide turning circle
        "max_health": 8000.0,
        "armor": 0.40,
        "weapons": ["ciws", "ciws", "ciws", "missile_sam"],
        "radar_range": 60000.0,
        "sonar_range": 5000.0,
        "crew": 5000,
    },
    ShipClass.SUBMARINE: {
        "max_speed": 12.0,      # ~24 knots submerged
        "turn_rate": 0.15,
        "max_health": 1200.0,
        "armor": 0.15,
        "weapons": ["torpedo_tube", "torpedo_tube", "torpedo_tube", "torpedo_tube"],
        "radar_range": 3000.0,
        "sonar_range": 15000.0,  # subs have excellent sonar
        "crew": 130,
    },
    ShipClass.CARGO: {
        "max_speed": 8.0,       # ~16 knots
        "turn_rate": 0.06,      # sluggish
        "max_health": 3000.0,
        "armor": 0.10,
        "weapons": [],
        "radar_range": 10000.0,
        "sonar_range": 0.0,
        "crew": 30,
    },
}


def create_ship(
    ship_class: ShipClass,
    name: str,
    alliance: str,
    position: Vec2 = (0.0, 0.0),
    heading: float = 0.0,
    ship_id: str | None = None,
) -> ShipState:
    """Create a ship from a template with default stats for its class."""
    t = SHIP_TEMPLATES[ship_class]
    return ShipState(
        ship_id=ship_id or f"ship_{uuid.uuid4().hex[:8]}",
        name=name,
        ship_class=ship_class,
        alliance=alliance,
        position=position,
        heading=heading,
        speed=0.0,
        max_speed=t["max_speed"],
        turn_rate=t["turn_rate"],
        health=t["max_health"],
        max_health=t["max_health"],
        armor=t["armor"],
        weapons=list(t["weapons"]),
        radar_range=t["radar_range"],
        sonar_range=t["sonar_range"],
        crew=t["crew"],
    )


# ---------------------------------------------------------------------------
# Naval physics
# ---------------------------------------------------------------------------

class NavalPhysics:
    """Ship movement with momentum, inertia, and sea-state effects.

    Ships accelerate and decelerate gradually. Turning radius increases
    with speed (faster = wider turns). Sea state adds rocking motion
    and reduces accuracy.
    """

    # Acceleration / deceleration rates (fraction of max_speed per second)
    ACCEL_RATE: float = 0.15
    DECEL_RATE: float = 0.10
    # How much speed reduces turn effectiveness (higher = wider turns at speed)
    SPEED_TURN_PENALTY: float = 0.4

    @staticmethod
    def update(ship: ShipState, throttle: float, rudder: float, dt: float) -> None:
        """Update ship position and heading with momentum physics.

        Args:
            ship: Ship to update (modified in place).
            throttle: -1 (full reverse) to 1 (full ahead).
            rudder: -1 (hard port/left) to 1 (hard starboard/right).
            dt: Time step in seconds.
        """
        throttle = max(-1.0, min(1.0, throttle))
        rudder = max(-1.0, min(1.0, rudder))

        ship._throttle = throttle
        ship._rudder = rudder

        # Target speed from throttle
        target_speed = throttle * ship.max_speed

        # Accelerate / decelerate toward target (momentum)
        speed_diff = target_speed - ship.speed
        if speed_diff > 0:
            accel = NavalPhysics.ACCEL_RATE * ship.max_speed * dt
            ship.speed = min(target_speed, ship.speed + accel)
        else:
            decel = NavalPhysics.DECEL_RATE * ship.max_speed * dt
            ship.speed = max(target_speed, ship.speed - decel)

        # Turning: effective turn rate decreases with speed
        speed_fraction = abs(ship.speed) / ship.max_speed if ship.max_speed > 0 else 0.0
        turn_penalty = 1.0 - NavalPhysics.SPEED_TURN_PENALTY * speed_fraction
        turn_penalty = max(0.3, turn_penalty)
        effective_turn = ship.turn_rate * turn_penalty * rudder * dt

        # Must be moving to turn (no pivot in place for ships)
        if abs(ship.speed) < 0.1:
            effective_turn = 0.0

        ship.heading += effective_turn
        # Normalize heading to [0, 2pi)
        ship.heading = ship.heading % (2.0 * math.pi)

        # Move in heading direction
        dx = math.cos(ship.heading) * ship.speed * dt
        dy = math.sin(ship.heading) * ship.speed * dt
        ship.position = (ship.position[0] + dx, ship.position[1] + dy)

        # Update wake intensity based on speed
        if ship.max_speed > 0:
            ship.wake_intensity = min(1.0, abs(ship.speed) / ship.max_speed)
        else:
            ship.wake_intensity = 0.0

    @staticmethod
    def calculate_wake(ship: ShipState) -> dict:
        """Calculate wake visual parameters for Three.js rendering.

        Returns a dict with wake geometry data based on ship speed and class.
        """
        speed_frac = abs(ship.speed) / ship.max_speed if ship.max_speed > 0 else 0.0

        # Larger ships produce larger wakes
        class_scale = {
            ShipClass.SPEEDBOAT: 0.5,
            ShipClass.PATROL_BOAT: 0.7,
            ShipClass.FRIGATE: 1.0,
            ShipClass.DESTROYER: 1.2,
            ShipClass.CRUISER: 1.5,
            ShipClass.CARRIER: 2.5,
            ShipClass.SUBMARINE: 0.3 if not ship.is_submerged else 0.0,
            ShipClass.CARGO: 1.8,
        }
        scale = class_scale.get(ship.ship_class, 1.0)

        wake_length = speed_frac * 50.0 * scale      # meters behind ship
        wake_width = speed_frac * 15.0 * scale        # meters at widest
        foam_intensity = speed_frac ** 1.5             # non-linear foam

        # Wake origin is behind the ship
        behind = heading_to_vec(ship.heading + math.pi)
        wake_origin = _add(ship.position, _scale(behind, 5.0 * scale))

        return {
            "origin_x": round(wake_origin[0], 2),
            "origin_y": round(wake_origin[1], 2),
            "heading": round(ship.heading, 4),
            "length": round(wake_length, 2),
            "width": round(wake_width, 2),
            "foam_intensity": round(foam_intensity, 3),
            "speed_fraction": round(speed_frac, 3),
        }

    @staticmethod
    def wave_effect(position: Vec2, sea_state: float, time: float = 0.0) -> Vec2:
        """Calculate wave-induced displacement at a position.

        Args:
            position: World position.
            sea_state: 0 (calm) to 1 (storm).
            time: Simulation time for wave phase.

        Returns:
            Displacement vector (dx, dy) in meters.
        """
        if sea_state <= 0:
            return (0.0, 0.0)

        # Two overlapping sine waves at different frequencies
        amp = sea_state * 3.0  # max 3m displacement in storm
        freq1 = 0.05
        freq2 = 0.08
        phase1 = position[0] * freq1 + time * 0.5
        phase2 = position[1] * freq2 + time * 0.7

        dx = amp * math.sin(phase1) * 0.3
        dy = amp * math.sin(phase2) * 0.3
        return (dx, dy)


# ---------------------------------------------------------------------------
# Naval formations
# ---------------------------------------------------------------------------

class FormationType(enum.Enum):
    """Standard naval formation types."""
    LINE_AHEAD = "line_ahead"       # Single file behind leader
    LINE_ABREAST = "line_abreast"   # Side by side
    DIAMOND = "diamond"             # Diamond/box pattern
    SCREEN = "screen"               # Defensive screen around capital ship
    WEDGE = "wedge"                 # V-formation


class NavalFormation:
    """Compute formation positions for a group of ships.

    The leader is always index 0. Remaining ships are assigned slots
    based on the formation type and spacing parameter.
    """

    @staticmethod
    def get_offsets(
        formation: FormationType,
        count: int,
        spacing: float = 200.0,
    ) -> list[Vec2]:
        """Return local-frame offsets for each ship slot.

        Index 0 is the leader at (0, 0). Positive x = forward,
        positive y = port (left).

        Args:
            formation: Formation type.
            count: Number of ships (including leader).
            spacing: Distance between ships in meters.

        Returns:
            List of (x, y) offsets in the leader's local frame.
        """
        if count <= 0:
            return []

        offsets: list[Vec2] = [(0.0, 0.0)]  # leader

        if formation == FormationType.LINE_AHEAD:
            for i in range(1, count):
                offsets.append((-spacing * i, 0.0))

        elif formation == FormationType.LINE_ABREAST:
            for i in range(1, count):
                # Alternate port/starboard
                side = 1 if i % 2 == 1 else -1
                rank = (i + 1) // 2
                offsets.append((0.0, side * spacing * rank))

        elif formation == FormationType.DIAMOND:
            # Diamond: 1 front, then port/starboard, then rear
            slots = [
                (0.0, 0.0),
                (-spacing, -spacing * 0.7),     # starboard rear
                (-spacing, spacing * 0.7),       # port rear
                (-spacing * 2, 0.0),             # dead astern
                (-spacing * 0.5, -spacing * 1.2),
                (-spacing * 0.5, spacing * 1.2),
                (-spacing * 1.5, -spacing * 0.7),
                (-spacing * 1.5, spacing * 0.7),
            ]
            offsets = []
            for i in range(count):
                if i < len(slots):
                    offsets.append(slots[i])
                else:
                    # Overflow: line astern behind diamond
                    offsets.append((-spacing * (2 + (i - len(slots))), 0.0))

        elif formation == FormationType.SCREEN:
            # Escorts form a semicircle ahead of the capital ship (index 0)
            # Angles from -60 to +60 degrees forward arc
            for i in range(1, count):
                # Spread escorts across the forward arc
                t = (i - 1) / max(1, count - 2)  # 0 to 1
                angle = -math.pi / 3 + (2 * math.pi / 3) * t  # -60 to +60 deg
                offsets.append((
                    math.cos(angle) * spacing,
                    math.sin(angle) * spacing,
                ))

        elif formation == FormationType.WEDGE:
            for i in range(1, count):
                side = 1 if i % 2 == 1 else -1
                rank = (i + 1) // 2
                offsets.append((
                    -spacing * rank * 0.7,
                    side * spacing * rank,
                ))

        return offsets

    @staticmethod
    def world_positions(
        leader_pos: Vec2,
        leader_heading: float,
        formation: FormationType,
        count: int,
        spacing: float = 200.0,
    ) -> list[Vec2]:
        """Compute world-space positions for a formation.

        Args:
            leader_pos: Leader's world position.
            leader_heading: Leader's heading in radians.
            formation: Formation type.
            count: Number of ships.
            spacing: Formation spacing in meters.

        Returns:
            List of world positions, one per ship.
        """
        offsets = NavalFormation.get_offsets(formation, count, spacing)
        cos_h = math.cos(leader_heading)
        sin_h = math.sin(leader_heading)
        positions: list[Vec2] = []

        for ox, oy in offsets:
            wx = ox * cos_h - oy * sin_h
            wy = ox * sin_h + oy * cos_h
            positions.append((leader_pos[0] + wx, leader_pos[1] + wy))

        return positions


# ---------------------------------------------------------------------------
# Naval combat engine
# ---------------------------------------------------------------------------

class NavalCombatEngine:
    """Manages all ships, torpedoes, shells, and combat resolution.

    Provides a ``tick()`` method for simulation stepping and
    ``to_three_js()`` for rendering state export.
    """

    # Weapon stats: (range_m, damage, speed_m_s, reload_s, accuracy)
    GUN_STATS: dict[str, dict] = {
        "mg_50cal": {"range": 1200, "damage": 10, "speed": 900, "reload": 0.1, "accuracy": 0.3},
        "gun_25mm": {"range": 2500, "damage": 30, "speed": 1100, "reload": 0.5, "accuracy": 0.4},
        "gun_76mm": {"range": 12000, "damage": 80, "speed": 900, "reload": 1.5, "accuracy": 0.35},
        "gun_127mm": {"range": 24000, "damage": 150, "speed": 800, "reload": 3.0, "accuracy": 0.3},
        "ciws": {"range": 1500, "damage": 20, "speed": 1100, "reload": 0.05, "accuracy": 0.7},
    }

    TORPEDO_STATS: dict = {
        "speed": 25.0,         # m/s (~50 knots)
        "damage": 800.0,
        "range": 10000.0,      # meters
        "turn_rate": 0.15,     # rad/s for homing
    }

    def __init__(
        self,
        sea_state: float = 0.3,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.ships: list[ShipState] = []
        self.torpedoes: list[Torpedo] = []
        self.shells: list[ShellProjectile] = []
        self.effects: list[CombatEffect] = []
        self.sea_state: float = max(0.0, min(1.0, sea_state))
        self._time: float = 0.0
        self._rng = rng or random.Random()
        self._cooldowns: dict[str, float] = {}  # ship_id -> seconds until can fire again
        self._events: list[dict] = []           # combat log events for current tick

    # -- Ship management ---------------------------------------------------

    def add_ship(self, ship: ShipState) -> None:
        """Add a ship to the simulation."""
        self.ships.append(ship)

    def get_ship(self, ship_id: str) -> ShipState | None:
        """Find a ship by ID."""
        for s in self.ships:
            if s.ship_id == ship_id:
                return s
        return None

    def remove_ship(self, ship_id: str) -> bool:
        """Remove a ship. Returns True if found and removed."""
        for i, s in enumerate(self.ships):
            if s.ship_id == ship_id:
                self.ships.pop(i)
                return True
        return False

    # -- Weapons -----------------------------------------------------------

    def fire_torpedo(
        self,
        ship_id: str,
        target_id: str | None = None,
        heading: float | None = None,
    ) -> Torpedo | None:
        """Launch a torpedo from a ship.

        If target_id is given, the torpedo homes on that target.
        If heading is given instead, it travels in that direction (dumb-fire).
        Returns the Torpedo, or None if the ship cannot fire.
        """
        ship = self.get_ship(ship_id)
        if ship is None or not ship.is_alive():
            return None

        # Check ship has torpedo tubes
        if "torpedo_tube" not in ship.weapons:
            return None

        launch_heading = heading if heading is not None else ship.heading
        stats = self.TORPEDO_STATS

        torp = Torpedo(
            torpedo_id=f"torp_{uuid.uuid4().hex[:8]}",
            position=ship.position,
            heading=launch_heading,
            speed=stats["speed"],
            target_id=target_id,
            damage=stats["damage"],
            range_remaining=stats["range"],
            source_ship_id=ship_id,
        )
        self.torpedoes.append(torp)

        self._events.append({
            "event": "torpedo_launched",
            "ship_id": ship_id,
            "torpedo_id": torp.torpedo_id,
            "target_id": target_id,
        })

        return torp

    def fire_guns(
        self,
        ship_id: str,
        target_pos: Vec2,
    ) -> list[ShellProjectile]:
        """Fire all available gun weapons at a target position.

        Returns list of shell projectiles created.
        """
        ship = self.get_ship(ship_id)
        if ship is None or not ship.is_alive():
            return []

        shells_fired: list[ShellProjectile] = []

        for weapon_id in ship.weapons:
            if weapon_id not in self.GUN_STATS:
                continue

            # Check cooldown
            cooldown_key = f"{ship_id}_{weapon_id}"
            if self._cooldowns.get(cooldown_key, 0.0) > 0.0:
                continue

            stats = self.GUN_STATS[weapon_id]
            dist = distance(ship.position, target_pos)

            # Range check
            if dist > stats["range"]:
                continue

            # Compute firing heading toward target with accuracy spread
            dx = target_pos[0] - ship.position[0]
            dy = target_pos[1] - ship.position[1]
            base_heading = math.atan2(dy, dx)

            # Sea state reduces accuracy
            accuracy_mod = 1.0 - self.sea_state * 0.4
            spread = (1.0 - stats["accuracy"] * accuracy_mod) * 0.1  # radians
            actual_heading = base_heading + self._rng.uniform(-spread, spread)

            shell = ShellProjectile(
                shell_id=f"shell_{uuid.uuid4().hex[:8]}",
                position=ship.position,
                heading=actual_heading,
                speed=stats["speed"],
                damage=stats["damage"],
                range_remaining=stats["range"],
                source_ship_id=ship_id,
            )
            shells_fired.append(shell)
            self.shells.append(shell)

            # Set cooldown
            self._cooldowns[cooldown_key] = stats["reload"]

            self._events.append({
                "event": "guns_fired",
                "ship_id": ship_id,
                "weapon": weapon_id,
                "shell_id": shell.shell_id,
            })

        return shells_fired

    # -- Sonar detection ---------------------------------------------------

    def _detect_submarines(self) -> list[dict]:
        """Check which surface ships detect which submarines."""
        detections: list[dict] = []
        subs = [s for s in self.ships if s.is_submerged and s.is_alive()]
        surface = [s for s in self.ships if not s.is_submerged and s.is_alive()]

        for detector in surface:
            if detector.sonar_range <= 0:
                continue
            for sub in subs:
                if detector.alliance == sub.alliance:
                    continue
                dist = distance(detector.position, sub.position)
                # Detection probability decreases with distance and sea state
                if dist <= detector.sonar_range:
                    # Deeper subs are harder to detect
                    depth_penalty = min(1.0, abs(sub.depth) / 200.0) * 0.3
                    # Sea state adds noise, making detection harder
                    noise_penalty = self.sea_state * 0.3
                    detect_prob = max(0.0, 1.0 - (dist / detector.sonar_range) - depth_penalty - noise_penalty)
                    if self._rng.random() < detect_prob:
                        detections.append({
                            "detector_id": detector.ship_id,
                            "submarine_id": sub.ship_id,
                            "distance": dist,
                            "confidence": detect_prob,
                        })
                        self.effects.append(CombatEffect(
                            effect_type="sonar_ping",
                            position=detector.position,
                            radius=dist,
                            duration=2.0,
                            intensity=detect_prob,
                        ))

        return detections

    # -- Tick --------------------------------------------------------------

    def tick(self, dt: float) -> dict:
        """Advance the simulation by dt seconds.

        Returns a dict summarizing what happened:
        - ship_positions: updated positions
        - torpedo_hits: list of torpedo impact events
        - shell_hits: list of shell impact events
        - sonar_detections: submarine detections
        - sunk: list of ship_ids that sank this tick
        - effects: visual effects generated
        - events: combat log entries
        """
        self._time += dt
        self._events = []
        self.effects = []
        torpedo_hits: list[dict] = []
        shell_hits: list[dict] = []
        sunk: list[str] = []

        # 1. Update cooldowns
        expired = []
        for key in self._cooldowns:
            self._cooldowns[key] -= dt
            if self._cooldowns[key] <= 0:
                expired.append(key)
        for key in expired:
            del self._cooldowns[key]

        # 2. Update ship positions (using stored throttle/rudder)
        for ship in self.ships:
            if not ship.is_alive():
                continue
            NavalPhysics.update(ship, ship._throttle, ship._rudder, dt)

        # 3. Update torpedoes
        for torp in self.torpedoes:
            if not torp.is_active:
                continue

            # Homing: adjust heading toward target
            if torp.target_id is not None:
                target = self.get_ship(torp.target_id)
                if target is not None and target.is_alive():
                    # Lead pursuit: aim where target will be
                    dx = target.position[0] - torp.position[0]
                    dy = target.position[1] - torp.position[1]
                    desired_heading = math.atan2(dy, dx)
                    # Limited turn rate
                    angle_diff = desired_heading - torp.heading
                    # Normalize to [-pi, pi]
                    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                    max_turn = self.TORPEDO_STATS["turn_rate"] * dt
                    if abs(angle_diff) > max_turn:
                        angle_diff = max_turn if angle_diff > 0 else -max_turn
                    torp.heading += angle_diff

            # Move torpedo
            move_dist = torp.speed * dt
            dx = math.cos(torp.heading) * move_dist
            dy = math.sin(torp.heading) * move_dist
            torp.position = (torp.position[0] + dx, torp.position[1] + dy)
            torp.range_remaining -= move_dist

            # Record trail for rendering
            torp._trail.append(torp.position)
            if len(torp._trail) > 50:
                torp._trail = torp._trail[-50:]

            # Range exhausted
            if torp.range_remaining <= 0:
                torp.is_active = False
                continue

            # Check collision with ships
            for ship in self.ships:
                if not ship.is_alive():
                    continue
                if ship.ship_id == torp.source_ship_id:
                    continue
                hit_radius = 15.0  # ship collision radius
                if distance(torp.position, ship.position) < hit_radius:
                    # Apply damage (armor reduces torpedo damage)
                    raw_dmg = torp.damage
                    absorbed = raw_dmg * ship.armor
                    effective = raw_dmg - absorbed
                    ship.health -= effective
                    torp.is_active = False

                    torpedo_hits.append({
                        "torpedo_id": torp.torpedo_id,
                        "target_id": ship.ship_id,
                        "damage": round(effective, 1),
                        "armor_absorbed": round(absorbed, 1),
                    })
                    self.effects.append(CombatEffect(
                        effect_type="explosion",
                        position=torp.position,
                        radius=20.0,
                        duration=2.0,
                        intensity=1.0,
                    ))
                    self._events.append({
                        "event": "torpedo_hit",
                        "torpedo_id": torp.torpedo_id,
                        "target_id": ship.ship_id,
                        "damage": round(effective, 1),
                    })
                    break

        # 4. Update shells
        for shell in self.shells:
            if not shell.is_active:
                continue

            move_dist = shell.speed * dt
            dx = math.cos(shell.heading) * move_dist
            dy = math.sin(shell.heading) * move_dist
            shell.position = (shell.position[0] + dx, shell.position[1] + dy)
            shell.range_remaining -= move_dist

            if shell.range_remaining <= 0:
                shell.is_active = False
                # Shell splash at end of range
                self.effects.append(CombatEffect(
                    effect_type="splash",
                    position=shell.position,
                    radius=5.0,
                    duration=0.5,
                ))
                continue

            # Check collision with ships
            for ship in self.ships:
                if not ship.is_alive():
                    continue
                if ship.ship_id == shell.source_ship_id:
                    continue
                hit_radius = 12.0
                if distance(shell.position, ship.position) < hit_radius:
                    raw_dmg = shell.damage
                    absorbed = raw_dmg * ship.armor
                    effective = raw_dmg - absorbed
                    ship.health -= effective
                    shell.is_active = False

                    shell_hits.append({
                        "shell_id": shell.shell_id,
                        "target_id": ship.ship_id,
                        "damage": round(effective, 1),
                        "source_ship_id": shell.source_ship_id,
                    })
                    self.effects.append(CombatEffect(
                        effect_type="explosion",
                        position=shell.position,
                        radius=8.0,
                        duration=1.0,
                        intensity=0.7,
                    ))
                    self._events.append({
                        "event": "shell_hit",
                        "shell_id": shell.shell_id,
                        "target_id": ship.ship_id,
                        "damage": round(effective, 1),
                    })
                    break

        # 5. Sonar detection
        sonar_detections = self._detect_submarines()

        # 6. Check for sunk ships
        for ship in self.ships:
            if ship.health <= 0 and ship.ship_id not in sunk:
                sunk.append(ship.ship_id)
                self.effects.append(CombatEffect(
                    effect_type="explosion",
                    position=ship.position,
                    radius=30.0,
                    duration=5.0,
                    intensity=1.0,
                ))
                self._events.append({
                    "event": "ship_sunk",
                    "ship_id": ship.ship_id,
                    "name": ship.name,
                })

        # 7. Clean up inactive projectiles
        self.torpedoes = [t for t in self.torpedoes if t.is_active]
        self.shells = [s for s in self.shells if s.is_active]

        return {
            "time": round(self._time, 3),
            "dt": dt,
            "torpedo_hits": torpedo_hits,
            "shell_hits": shell_hits,
            "sonar_detections": sonar_detections,
            "sunk": sunk,
            "effects": [e.to_dict() for e in self.effects],
            "events": self._events,
            "ship_count": len([s for s in self.ships if s.is_alive()]),
            "torpedo_count": len(self.torpedoes),
            "shell_count": len(self.shells),
        }

    # -- Three.js export ---------------------------------------------------

    def to_three_js(self) -> dict:
        """Export full simulation state for Three.js rendering.

        Returns a dict suitable for JSON serialization and consumption
        by a Three.js naval scene renderer.
        """
        ships_data: list[dict] = []
        for ship in self.ships:
            wake = NavalPhysics.calculate_wake(ship)
            ships_data.append({
                "id": ship.ship_id,
                "name": ship.name,
                "x": round(ship.position[0], 2),
                "y": round(ship.position[1], 2),
                "heading": round(ship.heading, 4),
                "speed": round(ship.speed, 2),
                "class": ship.ship_class.value,
                "alliance": ship.alliance,
                "health_pct": round(ship.health_pct(), 3),
                "wake": round(ship.wake_intensity, 3),
                "wake_data": wake,
                "is_submerged": ship.is_submerged,
                "depth": round(ship.depth, 1),
                "is_alive": ship.is_alive(),
                "turret_angles": [],  # placeholder for future turret tracking
                "crew": ship.crew,
            })

        torpedoes_data: list[dict] = []
        for torp in self.torpedoes:
            torpedoes_data.append({
                "id": torp.torpedo_id,
                "x": round(torp.position[0], 2),
                "y": round(torp.position[1], 2),
                "heading": round(torp.heading, 4),
                "speed": round(torp.speed, 2),
                "trail": [
                    (round(p[0], 1), round(p[1], 1))
                    for p in torp._trail[-20:]
                ],
            })

        shells_data: list[dict] = []
        for shell in self.shells:
            shells_data.append({
                "id": shell.shell_id,
                "x": round(shell.position[0], 2),
                "y": round(shell.position[1], 2),
                "heading": round(shell.heading, 4),
                "speed": round(shell.speed, 2),
            })

        return {
            "ships": ships_data,
            "torpedoes": torpedoes_data,
            "shells": shells_data,
            "effects": [e.to_dict() for e in self.effects],
            "sea_state": round(self.sea_state, 3),
            "time": round(self._time, 3),
        }

    # -- Utility -----------------------------------------------------------

    def set_ship_controls(
        self,
        ship_id: str,
        throttle: float | None = None,
        rudder: float | None = None,
    ) -> bool:
        """Set throttle and/or rudder for a ship. Returns True if found."""
        ship = self.get_ship(ship_id)
        if ship is None:
            return False
        if throttle is not None:
            ship._throttle = max(-1.0, min(1.0, throttle))
        if rudder is not None:
            ship._rudder = max(-1.0, min(1.0, rudder))
        return True

    def submerge(self, ship_id: str, depth: float = -50.0) -> bool:
        """Submerge a submarine. Returns False if not a submarine."""
        ship = self.get_ship(ship_id)
        if ship is None or ship.ship_class != ShipClass.SUBMARINE:
            return False
        ship.is_submerged = True
        ship.depth = min(0.0, depth)  # must be negative
        return True

    def surface(self, ship_id: str) -> bool:
        """Surface a submarine."""
        ship = self.get_ship(ship_id)
        if ship is None or ship.ship_class != ShipClass.SUBMARINE:
            return False
        ship.is_submerged = False
        ship.depth = 0.0
        return True

    def ships_by_alliance(self, alliance: str) -> list[ShipState]:
        """Return all living ships of a given alliance."""
        return [s for s in self.ships if s.alliance == alliance and s.is_alive()]

    def detect_targets(self, ship_id: str) -> list[dict]:
        """Return all targets detectable by a ship's radar/sonar."""
        ship = self.get_ship(ship_id)
        if ship is None or not ship.is_alive():
            return []

        targets: list[dict] = []
        for other in self.ships:
            if other.ship_id == ship_id or not other.is_alive():
                continue
            dist = distance(ship.position, other.position)

            # Surface targets detected by radar
            if not other.is_submerged and dist <= ship.radar_range:
                targets.append({
                    "target_id": other.ship_id,
                    "distance": round(dist, 1),
                    "bearing": round(math.atan2(
                        other.position[1] - ship.position[1],
                        other.position[0] - ship.position[0],
                    ), 4),
                    "sensor": "radar",
                    "class": other.ship_class.value,
                    "alliance": other.alliance,
                })
            # Submerged targets detected by sonar
            elif other.is_submerged and dist <= ship.sonar_range:
                depth_penalty = min(1.0, abs(other.depth) / 200.0) * 0.3
                detect_prob = max(0.0, 1.0 - (dist / ship.sonar_range) - depth_penalty)
                if self._rng.random() < detect_prob:
                    targets.append({
                        "target_id": other.ship_id,
                        "distance": round(dist, 1),
                        "bearing": round(math.atan2(
                            other.position[1] - ship.position[1],
                            other.position[0] - ship.position[0],
                        ), 4),
                        "sensor": "sonar",
                        "confidence": round(detect_prob, 3),
                        "class": other.ship_class.value,
                        "alliance": other.alliance,
                    })

        return targets

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Air combat module for the Tritium sim engine.

Fixed-wing aircraft dogfights, missiles with proportional navigation,
anti-air batteries, countermeasures, stall physics, g-force modelling,
and Three.js-compatible serialization for real-time 3D rendering.

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
# Aircraft classification
# ---------------------------------------------------------------------------

class AircraftClass(enum.Enum):
    """Classification of fixed-wing aircraft types."""
    FIGHTER = "fighter"
    BOMBER = "bomber"
    TRANSPORT = "transport"
    GUNSHIP = "gunship"
    RECON = "recon"
    STEALTH = "stealth"


# ---------------------------------------------------------------------------
# Aircraft state
# ---------------------------------------------------------------------------

@dataclass
class AircraftState:
    """Mutable state of a single aircraft in the simulation."""

    aircraft_id: str
    name: str
    aircraft_class: AircraftClass
    alliance: str

    position: Vec2              # ground-plane position (x, y) meters
    altitude: float             # meters AGL

    heading: float              # radians, 0 = +x, CCW positive
    pitch: float                # radians, positive = climbing

    speed: float                # m/s (current)
    max_speed: float            # m/s
    min_speed: float            # stall speed m/s

    turn_rate: float            # max yaw rate rad/s
    climb_rate: float           # max vertical rate m/s

    health: float
    max_health: float
    armor: float                # 0-1 fraction of damage absorbed

    fuel: float = 1.0           # 0-1
    weapons: list[str] = field(default_factory=list)
    countermeasures: int = 20   # flare/chaff count
    is_destroyed: bool = False
    g_force: float = 1.0

    radar_cross_section: float = 1.0  # 0-1, stealth aircraft have low RCS
    _trail: list[tuple[float, float, float]] = field(default_factory=list)
    _throttle: float = 1.0     # 0-1
    _target_heading: float | None = None
    _target_altitude: float | None = None

    def is_alive(self) -> bool:
        return self.health > 0.0 and not self.is_destroyed

    def health_pct(self) -> float:
        if self.max_health <= 0:
            return 0.0
        return max(0.0, min(1.0, self.health / self.max_health))

    def afterburner_active(self) -> bool:
        """True when at max throttle and speed above 80% max."""
        return self._throttle >= 0.95 and self.speed > self.max_speed * 0.8


# ---------------------------------------------------------------------------
# Missile
# ---------------------------------------------------------------------------

@dataclass
class Missile:
    """An air-launched or ground-launched missile."""

    missile_id: str
    position: Vec2
    altitude: float

    heading: float              # radians
    speed: float                # m/s current
    max_speed: float            # m/s

    target_id: str | None       # aircraft_id being tracked
    seeker_type: str            # "heat", "radar", "laser"
    damage: float
    range_remaining: float      # meters of fuel left

    turn_rate: float            # max turn rad/s
    is_active: bool = True
    countermeasure_vulnerable: bool = True  # heat seekers fooled by flares

    source_id: str = ""         # who fired it
    _trail: list[tuple[float, float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Anti-air battery
# ---------------------------------------------------------------------------

@dataclass
class AntiAir:
    """A ground-based anti-aircraft system."""

    aa_id: str
    position: Vec2
    alliance: str

    range_m: float              # engagement range meters
    damage: float               # per hit
    fire_rate: float            # shots per second
    tracking_speed: float       # rad/s

    ammo: int
    cooldown: float = 0.0       # seconds until next shot
    aa_type: str = "sam"        # "sam", "flak", "ciws", "manpad"


# ---------------------------------------------------------------------------
# Combat effect (visual)
# ---------------------------------------------------------------------------

@dataclass
class AirCombatEffect:
    """A visual effect for Three.js rendering."""

    effect_type: str    # "explosion", "flare", "smoke", "contrail", "muzzle_flash"
    position: Vec2
    altitude: float = 0.0
    radius: float = 5.0
    duration: float = 1.0
    intensity: float = 1.0

    def to_dict(self) -> dict:
        return {
            "type": self.effect_type,
            "x": round(self.position[0], 2),
            "y": round(self.position[1], 2),
            "z": round(self.altitude, 2),
            "radius": round(self.radius, 2),
            "duration": round(self.duration, 2),
            "intensity": round(self.intensity, 2),
        }


# ---------------------------------------------------------------------------
# Aircraft templates
# ---------------------------------------------------------------------------

AIRCRAFT_TEMPLATES: dict[str, dict] = {
    "f16": {
        "name": "F-16 Fighting Falcon",
        "aircraft_class": AircraftClass.FIGHTER,
        "max_speed": 600.0,
        "min_speed": 80.0,
        "turn_rate": 0.6,
        "climb_rate": 250.0,
        "max_health": 500.0,
        "armor": 0.1,
        "weapons": ["sidewinder", "amraam", "gun"],
        "countermeasures": 20,
        "radar_cross_section": 0.5,
    },
    "f22": {
        "name": "F-22 Raptor",
        "aircraft_class": AircraftClass.STEALTH,
        "max_speed": 650.0,
        "min_speed": 75.0,
        "turn_rate": 0.7,
        "climb_rate": 300.0,
        "max_health": 600.0,
        "armor": 0.15,
        "weapons": ["amraam", "sidewinder", "gun"],
        "countermeasures": 40,
        "radar_cross_section": 0.05,
    },
    "a10": {
        "name": "A-10 Thunderbolt II",
        "aircraft_class": AircraftClass.GUNSHIP,
        "max_speed": 200.0,
        "min_speed": 55.0,
        "turn_rate": 0.35,
        "climb_rate": 60.0,
        "max_health": 1200.0,
        "armor": 0.45,
        "weapons": ["gau8", "maverick"],
        "countermeasures": 10,
        "radar_cross_section": 0.7,
    },
    "b52": {
        "name": "B-52 Stratofortress",
        "aircraft_class": AircraftClass.BOMBER,
        "max_speed": 250.0,
        "min_speed": 80.0,
        "turn_rate": 0.08,
        "climb_rate": 30.0,
        "max_health": 2000.0,
        "armor": 0.2,
        "weapons": ["jdam", "jdam", "jdam", "jdam"],
        "countermeasures": 30,
        "radar_cross_section": 1.0,
    },
    "c130": {
        "name": "C-130 Hercules",
        "aircraft_class": AircraftClass.TRANSPORT,
        "max_speed": 180.0,
        "min_speed": 55.0,
        "turn_rate": 0.15,
        "climb_rate": 40.0,
        "max_health": 1500.0,
        "armor": 0.1,
        "weapons": [],
        "countermeasures": 15,
        "radar_cross_section": 0.9,
    },
    "u2": {
        "name": "U-2 Dragon Lady",
        "aircraft_class": AircraftClass.RECON,
        "max_speed": 200.0,
        "min_speed": 60.0,
        "turn_rate": 0.12,
        "climb_rate": 50.0,
        "max_health": 300.0,
        "armor": 0.05,
        "weapons": ["camera"],
        "countermeasures": 5,
        "radar_cross_section": 0.3,
    },
}


# ---------------------------------------------------------------------------
# Anti-air templates
# ---------------------------------------------------------------------------

AA_TEMPLATES: dict[str, dict] = {
    "patriot": {
        "aa_type": "sam",
        "range_m": 160000.0,
        "damage": 800.0,
        "fire_rate": 0.1,       # 1 shot per 10 seconds
        "tracking_speed": 1.5,
        "ammo": 16,
    },
    "stinger": {
        "aa_type": "manpad",
        "range_m": 5000.0,
        "damage": 200.0,
        "fire_rate": 0.5,
        "tracking_speed": 2.0,
        "ammo": 1,
    },
    "phalanx": {
        "aa_type": "ciws",
        "range_m": 2000.0,
        "damage": 50.0,
        "fire_rate": 75.0,      # CIWS fires very fast
        "tracking_speed": 3.0,
        "ammo": 1550,
    },
    "flak_88": {
        "aa_type": "flak",
        "range_m": 8000.0,
        "damage": 150.0,
        "fire_rate": 2.0,
        "tracking_speed": 0.5,
        "ammo": 200,
    },
}


# ---------------------------------------------------------------------------
# Missile templates
# ---------------------------------------------------------------------------

MISSILE_TEMPLATES: dict[str, dict] = {
    "sidewinder": {
        "seeker_type": "heat",
        "max_speed": 900.0,
        "damage": 250.0,
        "range": 18000.0,
        "turn_rate": 1.2,
        "countermeasure_vulnerable": True,
    },
    "amraam": {
        "seeker_type": "radar",
        "max_speed": 1200.0,
        "damage": 400.0,
        "range": 75000.0,
        "turn_rate": 0.8,
        "countermeasure_vulnerable": False,
    },
    "maverick": {
        "seeker_type": "laser",
        "max_speed": 350.0,
        "damage": 600.0,
        "range": 25000.0,
        "turn_rate": 0.5,
        "countermeasure_vulnerable": False,
    },
    "sam_missile": {
        "seeker_type": "radar",
        "max_speed": 1500.0,
        "damage": 800.0,
        "range": 160000.0,
        "turn_rate": 1.0,
        "countermeasure_vulnerable": False,
    },
    "stinger_missile": {
        "seeker_type": "heat",
        "max_speed": 700.0,
        "damage": 200.0,
        "range": 5000.0,
        "turn_rate": 1.5,
        "countermeasure_vulnerable": True,
    },
}


# ---------------------------------------------------------------------------
# Gun stats
# ---------------------------------------------------------------------------

GUN_STATS: dict[str, dict] = {
    "gun": {
        "damage": 30.0,
        "range": 1500.0,
        "rounds_per_burst": 20,
        "accuracy": 0.3,
        "cooldown": 0.5,
    },
    "gau8": {
        "damage": 80.0,
        "range": 2000.0,
        "rounds_per_burst": 50,
        "accuracy": 0.5,
        "cooldown": 0.2,
    },
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _distance_3d(
    pos_a: Vec2, alt_a: float,
    pos_b: Vec2, alt_b: float,
) -> float:
    """3D Euclidean distance."""
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    dz = alt_b - alt_a
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _bearing_to(src: Vec2, dst: Vec2) -> float:
    """Heading from src to dst in radians."""
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


def _pitch_to(
    src_pos: Vec2, src_alt: float,
    dst_pos: Vec2, dst_alt: float,
) -> float:
    """Pitch angle from source to destination."""
    horiz = distance(src_pos, dst_pos)
    if horiz < 1e-6:
        return math.copysign(math.pi / 2, dst_alt - src_alt)
    return math.atan2(dst_alt - src_alt, horiz)


# ---------------------------------------------------------------------------
# Air combat engine
# ---------------------------------------------------------------------------

class AirCombatEngine:
    """Manages aircraft, missiles, anti-air, and all combat resolution.

    Provides ``tick()`` for simulation stepping and ``to_three_js()``
    for Three.js-compatible state export.
    """

    # Fuel burn rates (fraction per second at full throttle)
    FUEL_BURN_BASE: float = 0.002
    FUEL_BURN_AFTERBURNER: float = 0.005
    # Gravity for stall dive
    GRAVITY: float = 9.81
    # Trail length (position history entries)
    MAX_TRAIL: int = 40
    # Flare defeat chance against heat seekers
    FLARE_DEFEAT_CHANCE: float = 0.5
    # Minimum altitude before crash
    GROUND_LEVEL: float = 0.0
    # G-force limits
    MAX_G: float = 9.0
    BLACKOUT_G: float = 8.0

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.aircraft: dict[str, AircraftState] = {}
        self.missiles: list[Missile] = []
        self.anti_air: list[AntiAir] = []
        self.effects: list[AirCombatEffect] = []
        self._time: float = 0.0
        self._rng = rng or random.Random()
        self._events: list[dict] = []
        self._gun_cooldowns: dict[str, float] = {}

    # -- Aircraft management -----------------------------------------------

    def spawn_aircraft(
        self,
        template: str,
        aircraft_id: str,
        alliance: str,
        position: Vec2 = (0.0, 0.0),
        altitude: float = 3000.0,
    ) -> AircraftState:
        """Spawn an aircraft from a template.

        Args:
            template: Key into AIRCRAFT_TEMPLATES (e.g. "f16").
            aircraft_id: Unique identifier.
            alliance: Alliance tag (e.g. "friendly", "hostile").
            position: Ground-plane spawn position.
            altitude: Initial altitude in meters AGL.

        Returns:
            The created AircraftState.

        Raises:
            KeyError: If template is not found.
        """
        t = AIRCRAFT_TEMPLATES[template]
        ac = AircraftState(
            aircraft_id=aircraft_id,
            name=t["name"],
            aircraft_class=t["aircraft_class"],
            alliance=alliance,
            position=position,
            altitude=max(0.0, altitude),
            heading=0.0,
            pitch=0.0,
            speed=t["max_speed"] * 0.5,  # spawn at half speed
            max_speed=t["max_speed"],
            min_speed=t["min_speed"],
            turn_rate=t["turn_rate"],
            climb_rate=t["climb_rate"],
            health=t["max_health"],
            max_health=t["max_health"],
            armor=t["armor"],
            weapons=list(t["weapons"]),
            countermeasures=t["countermeasures"],
            radar_cross_section=t.get("radar_cross_section", 1.0),
        )
        self.aircraft[aircraft_id] = ac
        self._events.append({
            "event": "aircraft_spawned",
            "aircraft_id": aircraft_id,
            "template": template,
            "alliance": alliance,
        })
        return ac

    def get_aircraft(self, aircraft_id: str) -> AircraftState | None:
        return self.aircraft.get(aircraft_id)

    def remove_aircraft(self, aircraft_id: str) -> bool:
        if aircraft_id in self.aircraft:
            del self.aircraft[aircraft_id]
            return True
        return False

    # -- Weapons -----------------------------------------------------------

    def fire_missile(
        self,
        aircraft_id: str,
        target_id: str,
        missile_type: str,
    ) -> Missile | None:
        """Fire a missile from an aircraft at a target.

        Args:
            aircraft_id: Launching aircraft ID.
            target_id: Target aircraft ID.
            missile_type: Key into MISSILE_TEMPLATES or weapon name
                          that maps to a missile template.

        Returns:
            The launched Missile, or None if unable to fire.
        """
        ac = self.get_aircraft(aircraft_id)
        if ac is None or not ac.is_alive():
            return None

        # Resolve missile type: weapon name -> template name
        template_key = missile_type
        if template_key not in MISSILE_TEMPLATES:
            return None

        # Check aircraft has this weapon
        if missile_type not in ac.weapons:
            return None

        # Consume the weapon
        ac.weapons.remove(missile_type)

        t = MISSILE_TEMPLATES[template_key]
        m = Missile(
            missile_id=f"msl_{uuid.uuid4().hex[:8]}",
            position=ac.position,
            altitude=ac.altitude,
            heading=ac.heading,
            speed=ac.speed + 50.0,  # initial boost from launch platform
            max_speed=t["max_speed"],
            target_id=target_id,
            seeker_type=t["seeker_type"],
            damage=t["damage"],
            range_remaining=t["range"],
            turn_rate=t["turn_rate"],
            countermeasure_vulnerable=t["countermeasure_vulnerable"],
            source_id=aircraft_id,
        )
        self.missiles.append(m)
        self._events.append({
            "event": "missile_fired",
            "aircraft_id": aircraft_id,
            "missile_id": m.missile_id,
            "target_id": target_id,
            "type": missile_type,
        })
        return m

    def deploy_countermeasures(self, aircraft_id: str) -> bool:
        """Deploy a flare/chaff from an aircraft.

        Returns True if deployed, False if no countermeasures remaining.
        Each deployment creates a flare effect and has a chance to
        defeat incoming heat-seeking missiles.
        """
        ac = self.get_aircraft(aircraft_id)
        if ac is None or not ac.is_alive():
            return False
        if ac.countermeasures <= 0:
            return False

        ac.countermeasures -= 1
        self.effects.append(AirCombatEffect(
            effect_type="flare",
            position=ac.position,
            altitude=ac.altitude,
            radius=15.0,
            duration=3.0,
            intensity=1.0,
        ))

        # Check all incoming heat-seeking missiles targeting this aircraft
        defeated = []
        for m in self.missiles:
            if not m.is_active:
                continue
            if m.target_id != aircraft_id:
                continue
            if not m.countermeasure_vulnerable:
                continue
            # Heat seekers can be fooled by flares
            if self._rng.random() < self.FLARE_DEFEAT_CHANCE:
                m.is_active = False
                defeated.append(m.missile_id)
                self.effects.append(AirCombatEffect(
                    effect_type="smoke",
                    position=m.position,
                    altitude=m.altitude,
                    radius=5.0,
                    duration=2.0,
                ))

        self._events.append({
            "event": "countermeasures_deployed",
            "aircraft_id": aircraft_id,
            "remaining": ac.countermeasures,
            "missiles_defeated": defeated,
        })
        return True

    def fire_guns(
        self,
        aircraft_id: str,
        target_pos: Vec2,
        target_alt: float = 0.0,
    ) -> list[dict]:
        """Fire guns from an aircraft toward a target position.

        Returns a list of hit result dicts.
        """
        ac = self.get_aircraft(aircraft_id)
        if ac is None or not ac.is_alive():
            return []

        results: list[dict] = []

        for weapon_name in ac.weapons:
            if weapon_name not in GUN_STATS:
                continue

            # Check cooldown
            cd_key = f"{aircraft_id}_{weapon_name}"
            if self._gun_cooldowns.get(cd_key, 0.0) > 0.0:
                continue

            stats = GUN_STATS[weapon_name]
            dist = _distance_3d(ac.position, ac.altitude, target_pos, target_alt)
            if dist > stats["range"]:
                continue

            # Set cooldown
            self._gun_cooldowns[cd_key] = stats["cooldown"]

            self.effects.append(AirCombatEffect(
                effect_type="muzzle_flash",
                position=ac.position,
                altitude=ac.altitude,
                radius=3.0,
                duration=0.2,
            ))

            # Roll hits for each round in burst
            hits = 0
            for _ in range(stats["rounds_per_burst"]):
                # Accuracy degrades with distance
                range_factor = 1.0 - (dist / stats["range"])
                hit_chance = stats["accuracy"] * range_factor
                if self._rng.random() < hit_chance:
                    hits += 1

            total_damage = hits * stats["damage"]
            results.append({
                "weapon": weapon_name,
                "hits": hits,
                "total_damage": round(total_damage, 1),
                "rounds_fired": stats["rounds_per_burst"],
                "distance": round(dist, 1),
            })

            self._events.append({
                "event": "guns_fired",
                "aircraft_id": aircraft_id,
                "weapon": weapon_name,
                "hits": hits,
                "damage": round(total_damage, 1),
            })

        return results

    # -- Anti-air management -----------------------------------------------

    def add_anti_air(
        self,
        template: str,
        aa_id: str,
        alliance: str,
        position: Vec2,
    ) -> AntiAir:
        """Add an anti-air battery from a template.

        Raises:
            KeyError: If template is not found.
        """
        t = AA_TEMPLATES[template]
        aa = AntiAir(
            aa_id=aa_id,
            position=position,
            alliance=alliance,
            range_m=t["range_m"],
            damage=t["damage"],
            fire_rate=t["fire_rate"],
            tracking_speed=t["tracking_speed"],
            ammo=t["ammo"],
            aa_type=t["aa_type"],
        )
        self.anti_air.append(aa)
        return aa

    # -- Aircraft controls -------------------------------------------------

    def set_controls(
        self,
        aircraft_id: str,
        throttle: float | None = None,
        target_heading: float | None = None,
        target_altitude: float | None = None,
    ) -> bool:
        """Set flight controls for an aircraft.

        Args:
            aircraft_id: Aircraft to control.
            throttle: 0-1 throttle setting.
            target_heading: Desired heading in radians.
            target_altitude: Desired altitude in meters.

        Returns:
            True if aircraft found, False otherwise.
        """
        ac = self.get_aircraft(aircraft_id)
        if ac is None:
            return False
        if throttle is not None:
            ac._throttle = max(0.0, min(1.0, throttle))
        if target_heading is not None:
            ac._target_heading = target_heading
        if target_altitude is not None:
            ac._target_altitude = max(0.0, target_altitude)
        return True

    # -- Tick (main simulation step) ---------------------------------------

    def tick(self, dt: float) -> dict:
        """Advance the simulation by *dt* seconds.

        Simulation order:
        1. Aircraft physics (speed, heading, altitude, stall, g-force)
        2. Missile homing (proportional navigation, limited turn rate)
        3. Missile vs countermeasure resolution
        4. AA engagement (auto-fire at aircraft in range)
        5. Collision detection (missiles hitting aircraft)
        6. Stall recovery (auto-dive)
        7. Fuel consumption
        8. Ground collision / crash detection

        Returns:
            Dict summarizing tick results.
        """
        self._time += dt
        self._events = []
        self.effects = []
        missile_hits: list[dict] = []
        aa_hits: list[dict] = []
        destroyed: list[str] = []

        # Update gun cooldowns
        expired_cds = []
        for key in self._gun_cooldowns:
            self._gun_cooldowns[key] -= dt
            if self._gun_cooldowns[key] <= 0:
                expired_cds.append(key)
        for key in expired_cds:
            del self._gun_cooldowns[key]

        # 1. Aircraft physics
        for ac in self.aircraft.values():
            if not ac.is_alive():
                continue
            self._update_aircraft_physics(ac, dt)

        # 2. Missile homing + movement
        for m in self.missiles:
            if not m.is_active:
                continue
            self._update_missile(m, dt)

        # 3. AA engagement
        for aa in self.anti_air:
            aa_result = self._update_anti_air(aa, dt)
            if aa_result:
                aa_hits.append(aa_result)

        # 4. Missile collision detection
        for m in self.missiles:
            if not m.is_active:
                continue
            hit = self._check_missile_hit(m)
            if hit:
                missile_hits.append(hit)

        # 5. Check for destroyed aircraft
        for ac in self.aircraft.values():
            if ac.health <= 0 and not ac.is_destroyed:
                ac.is_destroyed = True
                destroyed.append(ac.aircraft_id)
                self.effects.append(AirCombatEffect(
                    effect_type="explosion",
                    position=ac.position,
                    altitude=ac.altitude,
                    radius=30.0,
                    duration=3.0,
                    intensity=1.0,
                ))
                self._events.append({
                    "event": "aircraft_destroyed",
                    "aircraft_id": ac.aircraft_id,
                    "name": ac.name,
                })

        # 6. Ground collision
        for ac in self.aircraft.values():
            if ac.is_destroyed:
                continue
            if ac.altitude <= self.GROUND_LEVEL:
                ac.altitude = self.GROUND_LEVEL
                ac.is_destroyed = True
                ac.health = 0.0
                destroyed.append(ac.aircraft_id)
                self.effects.append(AirCombatEffect(
                    effect_type="explosion",
                    position=ac.position,
                    altitude=0.0,
                    radius=40.0,
                    duration=5.0,
                    intensity=1.0,
                ))
                self._events.append({
                    "event": "aircraft_crashed",
                    "aircraft_id": ac.aircraft_id,
                })

        # 7. Clean up inactive missiles
        self.missiles = [m for m in self.missiles if m.is_active]

        return {
            "time": round(self._time, 3),
            "dt": dt,
            "missile_hits": missile_hits,
            "aa_hits": aa_hits,
            "destroyed": destroyed,
            "effects": [e.to_dict() for e in self.effects],
            "events": self._events,
            "aircraft_count": len([a for a in self.aircraft.values() if a.is_alive()]),
            "missile_count": len(self.missiles),
            "aa_count": len(self.anti_air),
        }

    def _update_aircraft_physics(self, ac: AircraftState, dt: float) -> None:
        """Update a single aircraft's physics."""
        # Speed: approach throttle * max_speed
        target_speed = ac._throttle * ac.max_speed
        speed_diff = target_speed - ac.speed
        # Accelerate/decelerate at ~10% of max_speed per second
        accel_rate = ac.max_speed * 0.1
        if speed_diff > 0:
            ac.speed = min(target_speed, ac.speed + accel_rate * dt)
        else:
            ac.speed = max(target_speed, ac.speed - accel_rate * dt)

        # Stall check: below stall speed, force nose-down dive
        is_stalling = ac.speed < ac.min_speed
        if is_stalling:
            # Nose down to recover speed
            ac.pitch = max(ac.pitch - 0.5 * dt, -math.pi / 4)
            # Gravity accelerates the aircraft in a dive
            ac.speed += self.GRAVITY * 0.5 * dt
            self._events.append({
                "event": "stall",
                "aircraft_id": ac.aircraft_id,
                "speed": round(ac.speed, 1),
            })

        # Heading: turn toward target heading if set
        old_heading = ac.heading
        if ac._target_heading is not None:
            angle_diff = _normalize_angle(ac._target_heading - ac.heading)
            max_turn = ac.turn_rate * dt
            if abs(angle_diff) <= max_turn:
                ac.heading = ac._target_heading
            else:
                ac.heading += max_turn if angle_diff > 0 else -max_turn
            ac.heading = ac.heading % (2.0 * math.pi)

        # Altitude: climb/descend toward target altitude if set
        if ac._target_altitude is not None and not is_stalling:
            alt_diff = ac._target_altitude - ac.altitude
            max_climb = ac.climb_rate * dt
            if abs(alt_diff) <= max_climb:
                ac.altitude = ac._target_altitude
                ac.pitch = 0.0
            else:
                if alt_diff > 0:
                    ac.altitude += max_climb
                    ac.pitch = min(0.5, alt_diff / (ac.climb_rate * 2))
                else:
                    ac.altitude -= max_climb
                    ac.pitch = max(-0.5, alt_diff / (ac.climb_rate * 2))

        # G-force from turning
        if dt > 0:
            heading_rate = abs(_normalize_angle(ac.heading - old_heading)) / dt
            # G = v * omega / g  (centripetal)
            centripetal_g = (ac.speed * heading_rate) / self.GRAVITY
            ac.g_force = max(1.0, 1.0 + centripetal_g)
        else:
            ac.g_force = 1.0

        # Move along heading on ground plane
        dx = math.cos(ac.heading) * ac.speed * dt
        dy = math.sin(ac.heading) * ac.speed * dt
        ac.position = (ac.position[0] + dx, ac.position[1] + dy)

        # Altitude update from pitch if not using target_altitude
        if ac._target_altitude is None and not is_stalling:
            ac.altitude += math.sin(ac.pitch) * ac.speed * dt

        # Stall dive altitude change
        if is_stalling:
            ac.altitude += math.sin(ac.pitch) * ac.speed * dt

        # Fuel consumption
        if ac.fuel > 0:
            burn = self.FUEL_BURN_BASE * ac._throttle * dt
            if ac.afterburner_active():
                burn = self.FUEL_BURN_AFTERBURNER * dt
            ac.fuel = max(0.0, ac.fuel - burn)
            if ac.fuel <= 0:
                # No fuel: can't maintain speed, slowly decelerate
                ac._throttle = 0.0

        # Trail for rendering
        ac._trail.append((
            round(ac.position[0], 1),
            round(ac.position[1], 1),
            round(ac.altitude, 1),
        ))
        if len(ac._trail) > self.MAX_TRAIL:
            ac._trail = ac._trail[-self.MAX_TRAIL:]

    def _update_missile(self, m: Missile, dt: float) -> None:
        """Update a single missile: homing + movement."""
        # Accelerate toward max speed
        if m.speed < m.max_speed:
            m.speed = min(m.max_speed, m.speed + m.max_speed * 0.3 * dt)

        # Proportional navigation toward target
        if m.target_id is not None:
            target = self.get_aircraft(m.target_id)
            if target is not None and target.is_alive():
                desired_heading = _bearing_to(m.position, target.position)
                angle_diff = _normalize_angle(desired_heading - m.heading)
                max_turn = m.turn_rate * dt
                if abs(angle_diff) > max_turn:
                    angle_diff = max_turn if angle_diff > 0 else -max_turn
                m.heading += angle_diff

                # Altitude homing
                desired_pitch = _pitch_to(
                    m.position, m.altitude,
                    target.position, target.altitude,
                )
                pitch_diff = desired_pitch  # simplified: directly adjust
                max_pitch_change = m.turn_rate * dt
                alt_change = math.sin(max(-max_pitch_change, min(max_pitch_change, pitch_diff))) * m.speed * dt
                m.altitude += alt_change

        # Move
        move_dist = m.speed * dt
        dx = math.cos(m.heading) * move_dist
        dy = math.sin(m.heading) * move_dist
        m.position = (m.position[0] + dx, m.position[1] + dy)
        m.range_remaining -= move_dist

        # Trail
        m._trail.append((
            round(m.position[0], 1),
            round(m.position[1], 1),
            round(m.altitude, 1),
        ))
        if len(m._trail) > 30:
            m._trail = m._trail[-30:]

        # Range exhausted
        if m.range_remaining <= 0:
            m.is_active = False

        # Ground collision
        if m.altitude <= self.GROUND_LEVEL:
            m.is_active = False
            self.effects.append(AirCombatEffect(
                effect_type="explosion",
                position=m.position,
                altitude=0.0,
                radius=10.0,
                duration=1.0,
            ))

    def _update_anti_air(self, aa: AntiAir, dt: float) -> dict | None:
        """Update a single AA battery. Returns hit dict or None."""
        # Cooldown
        if aa.cooldown > 0:
            aa.cooldown -= dt
            return None

        if aa.ammo <= 0:
            return None

        # Find closest hostile aircraft in range
        best_target: AircraftState | None = None
        best_dist = float("inf")

        for ac in self.aircraft.values():
            if not ac.is_alive():
                continue
            if ac.alliance == aa.alliance:
                continue
            dist = _distance_3d(aa.position, 0.0, ac.position, ac.altitude)
            if dist <= aa.range_m and dist < best_dist:
                # Stealth check: low RCS aircraft harder to detect
                detect_range = aa.range_m * ac.radar_cross_section
                if dist <= detect_range:
                    best_target = ac
                    best_dist = dist

        if best_target is None:
            return None

        # Fire
        aa.ammo -= 1
        aa.cooldown = 1.0 / aa.fire_rate

        # Hit probability: decreases with distance, increases with tracking speed
        range_factor = 1.0 - (best_dist / aa.range_m)
        hit_chance = range_factor * min(1.0, aa.tracking_speed / 2.0)
        # Flak: area damage, always "hits" but reduced
        if aa.aa_type == "flak":
            hit_chance = max(0.3, range_factor * 0.6)

        if self._rng.random() < hit_chance:
            raw_dmg = aa.damage
            absorbed = raw_dmg * best_target.armor
            effective = raw_dmg - absorbed
            best_target.health -= effective

            self.effects.append(AirCombatEffect(
                effect_type="explosion",
                position=best_target.position,
                altitude=best_target.altitude,
                radius=10.0,
                duration=0.5,
            ))

            result = {
                "aa_id": aa.aa_id,
                "target_id": best_target.aircraft_id,
                "damage": round(effective, 1),
                "distance": round(best_dist, 1),
                "aa_type": aa.aa_type,
            }
            self._events.append({
                "event": "aa_hit",
                **result,
            })
            return result

        # Miss
        if aa.aa_type == "flak":
            self.effects.append(AirCombatEffect(
                effect_type="explosion",
                position=best_target.position,
                altitude=best_target.altitude + self._rng.uniform(-50, 50),
                radius=15.0,
                duration=0.3,
                intensity=0.4,
            ))
        return None

    def _check_missile_hit(self, m: Missile) -> dict | None:
        """Check if a missile has hit its target or any nearby aircraft."""
        hit_radius = 25.0  # meters proximity fuse

        for ac in self.aircraft.values():
            if not ac.is_alive():
                continue
            if ac.aircraft_id == m.source_id:
                continue
            dist = _distance_3d(m.position, m.altitude, ac.position, ac.altitude)
            if dist < hit_radius:
                raw_dmg = m.damage
                absorbed = raw_dmg * ac.armor
                effective = raw_dmg - absorbed
                ac.health -= effective
                m.is_active = False

                self.effects.append(AirCombatEffect(
                    effect_type="explosion",
                    position=ac.position,
                    altitude=ac.altitude,
                    radius=20.0,
                    duration=2.0,
                    intensity=1.0,
                ))

                result = {
                    "missile_id": m.missile_id,
                    "target_id": ac.aircraft_id,
                    "damage": round(effective, 1),
                    "armor_absorbed": round(absorbed, 1),
                    "seeker_type": m.seeker_type,
                }
                self._events.append({
                    "event": "missile_hit",
                    **result,
                })
                return result
        return None

    # -- Three.js export ---------------------------------------------------

    def to_three_js(self) -> dict:
        """Export full simulation state for Three.js rendering.

        Returns a dict suitable for JSON serialization and consumption
        by a Three.js air combat scene renderer.
        """
        aircraft_data: list[dict] = []
        for ac in self.aircraft.values():
            aircraft_data.append({
                "id": ac.aircraft_id,
                "name": ac.name,
                "x": round(ac.position[0], 2),
                "y": round(ac.position[1], 2),
                "z": round(ac.altitude, 2),
                "heading": round(ac.heading, 4),
                "pitch": round(ac.pitch, 4),
                "speed": round(ac.speed, 2),
                "class": ac.aircraft_class.value,
                "alliance": ac.alliance,
                "health_pct": round(ac.health_pct(), 3),
                "fuel": round(ac.fuel, 3),
                "g_force": round(ac.g_force, 2),
                "trail": list(ac._trail[-20:]),
                "afterburner": ac.afterburner_active(),
                "is_destroyed": ac.is_destroyed,
                "countermeasures": ac.countermeasures,
                "weapons": ac.weapons,
                "rcs": round(ac.radar_cross_section, 3),
            })

        missiles_data: list[dict] = []
        for m in self.missiles:
            color_map = {"heat": "#ff4400", "radar": "#4488ff", "laser": "#00ff44"}
            missiles_data.append({
                "id": m.missile_id,
                "x": round(m.position[0], 2),
                "y": round(m.position[1], 2),
                "z": round(m.altitude, 2),
                "heading": round(m.heading, 4),
                "speed": round(m.speed, 2),
                "type": m.seeker_type,
                "target_id": m.target_id,
                "trail": list(m._trail[-20:]),
                "color": color_map.get(m.seeker_type, "#ffffff"),
            })

        aa_data: list[dict] = []
        for aa in self.anti_air:
            aa_data.append({
                "id": aa.aa_id,
                "x": round(aa.position[0], 2),
                "y": round(aa.position[1], 2),
                "range": round(aa.range_m, 2),
                "type": aa.aa_type,
                "alliance": aa.alliance,
                "ammo": aa.ammo,
            })

        return {
            "aircraft": aircraft_data,
            "missiles": missiles_data,
            "aa_sites": aa_data,
            "effects": [e.to_dict() for e in self.effects],
            "time": round(self._time, 3),
        }

    # -- Utility -----------------------------------------------------------

    def aircraft_by_alliance(self, alliance: str) -> list[AircraftState]:
        """Return all living aircraft of a given alliance."""
        return [
            a for a in self.aircraft.values()
            if a.alliance == alliance and a.is_alive()
        ]

    def detect_targets(self, aircraft_id: str, radar_range: float = 50000.0) -> list[dict]:
        """Return all targets detectable by an aircraft's radar.

        Stealth aircraft are harder to detect (shorter effective range).
        """
        ac = self.get_aircraft(aircraft_id)
        if ac is None or not ac.is_alive():
            return []

        targets: list[dict] = []
        for other in self.aircraft.values():
            if other.aircraft_id == aircraft_id or not other.is_alive():
                continue
            dist = _distance_3d(ac.position, ac.altitude, other.position, other.altitude)
            effective_range = radar_range * other.radar_cross_section
            if dist <= effective_range:
                targets.append({
                    "target_id": other.aircraft_id,
                    "distance": round(dist, 1),
                    "bearing": round(_bearing_to(ac.position, other.position), 4),
                    "altitude": round(other.altitude, 1),
                    "class": other.aircraft_class.value,
                    "alliance": other.alliance,
                    "speed": round(other.speed, 1),
                })
        return targets

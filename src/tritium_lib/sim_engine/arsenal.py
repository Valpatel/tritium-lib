# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Comprehensive weapons database, projectile physics, and area effects.

Provides an expanded arsenal of 35+ weapons with realistic ballistics,
a ProjectileSimulator for in-flight projectile management, and an
AreaEffectManager for grenades/smoke/fire.  All output is JSON-dict
compatible so Three.js can render tracers, impacts, muzzle flashes,
and persistent area effects directly.

Usage::

    from tritium_lib.sim_engine.arsenal import (
        ARSENAL, ProjectileSimulator, AreaEffectManager,
    )

    sim = ProjectileSimulator()
    weapon = ARSENAL["m4a1"]
    proj = sim.fire(weapon, origin=(0, 0), target=(100, 50))
    impacts = sim.tick(0.016)       # 60 fps
    three_data = sim.to_three_js()  # send to frontend

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    magnitude,
    normalize,
    _sub,
    _add,
    _scale,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WeaponCategory(Enum):
    """Broad weapon classification."""
    PISTOL = "pistol"
    RIFLE = "rifle"
    SMG = "smg"
    SHOTGUN = "shotgun"
    SNIPER = "sniper"
    LMG = "lmg"
    LAUNCHER = "launcher"
    MELEE = "melee"
    THROWN = "thrown"
    TURRET = "turret"
    DRONE_MOUNTED = "drone_mounted"


class ProjectileType(Enum):
    """Type of projectile fired."""
    BULLET = "bullet"
    SHELL = "shell"
    ROCKET = "rocket"
    GRENADE = "grenade"
    FLASHBANG = "flashbang"
    SMOKE = "smoke"
    TEARGAS = "teargas"
    MOLOTOV = "molotov"
    ARROW = "arrow"


# Projectile types that are affected by gravity (arced trajectory)
_ARCED_TYPES = {
    ProjectileType.GRENADE,
    ProjectileType.FLASHBANG,
    ProjectileType.SMOKE,
    ProjectileType.TEARGAS,
    ProjectileType.MOLOTOV,
    ProjectileType.ROCKET,
}

# Impact visual effect per projectile type
_IMPACT_EFFECTS: dict[ProjectileType, str] = {
    ProjectileType.BULLET: "spark",
    ProjectileType.SHELL: "explosion",
    ProjectileType.ROCKET: "explosion",
    ProjectileType.GRENADE: "explosion",
    ProjectileType.FLASHBANG: "flash",
    ProjectileType.SMOKE: "smoke_cloud",
    ProjectileType.TEARGAS: "gas_cloud",
    ProjectileType.MOLOTOV: "fire",
    ProjectileType.ARROW: "stick",
}


# ---------------------------------------------------------------------------
# Weapon dataclass
# ---------------------------------------------------------------------------

@dataclass
class Weapon:
    """Static definition of a weapon and its characteristics."""

    weapon_id: str
    name: str
    category: WeaponCategory
    projectile_type: ProjectileType
    damage: float
    fire_rate: float            # rounds per second
    magazine_size: int
    reload_time: float          # seconds
    muzzle_velocity: float      # m/s
    effective_range: float      # meters
    max_range: float            # meters
    accuracy: float             # 0-1 base hit probability
    spread_deg: float           # cone of fire in degrees
    recoil: float               # 0-1, affects sustained accuracy
    weight_kg: float
    sound_radius: float         # meters — detection range of gunfire
    tracer_color: str = "#ffaa00"
    muzzle_flash_size: float = 1.0

    def to_dict(self) -> dict:
        """Serialize for API/frontend consumption."""
        return {
            "weapon_id": self.weapon_id,
            "name": self.name,
            "category": self.category.value,
            "projectile_type": self.projectile_type.value,
            "damage": self.damage,
            "fire_rate": self.fire_rate,
            "magazine_size": self.magazine_size,
            "reload_time": self.reload_time,
            "muzzle_velocity": self.muzzle_velocity,
            "effective_range": self.effective_range,
            "max_range": self.max_range,
            "accuracy": self.accuracy,
            "spread_deg": self.spread_deg,
            "recoil": self.recoil,
            "weight_kg": self.weight_kg,
            "sound_radius": self.sound_radius,
            "tracer_color": self.tracer_color,
            "muzzle_flash_size": self.muzzle_flash_size,
        }


# ---------------------------------------------------------------------------
# Projectile dataclass (in-flight)
# ---------------------------------------------------------------------------

@dataclass
class Projectile:
    """A projectile currently in flight."""

    projectile_id: str
    weapon_id: str
    origin: Vec2
    position: Vec2
    velocity: Vec2
    damage: float
    projectile_type: ProjectileType
    time_alive: float = 0.0
    is_active: bool = True
    tracer_color: str = "#ffaa00"
    max_range: float = 1000.0
    source_id: str = ""  # unit_id of the shooter for kill attribution

    def distance_traveled(self) -> float:
        """Distance from origin to current position."""
        return distance(self.origin, self.position)


# ---------------------------------------------------------------------------
# AreaEffect dataclass
# ---------------------------------------------------------------------------

@dataclass
class AreaEffect:
    """A persistent area effect (explosion, smoke, fire, etc.)."""

    effect_type: str            # explosion, smoke, fire, flashbang, teargas
    position: Vec2
    radius: float
    duration: float
    intensity: float            # 0-1
    time_remaining: float
    damage_per_second: float = 0.0
    color: str = "#ff4400"


# ---------------------------------------------------------------------------
# ProjectileSimulator
# ---------------------------------------------------------------------------

class ProjectileSimulator:
    """Manages in-flight projectiles with physics simulation.

    Supports gravity for arced projectiles (grenades, rockets), air
    resistance decay, and range-limit expiry.  Produces Three.js-ready
    dicts for frontend rendering.
    """

    def __init__(
        self,
        gravity: float = 9.81,
        air_resistance: float = 0.01,
    ) -> None:
        self.gravity = gravity
        self.air_resistance = air_resistance
        self.projectiles: list[Projectile] = []
        self._pending_impacts: list[dict] = []
        self._pending_flashes: list[dict] = []
        self._id_counter = 0

    # -- Fire ---------------------------------------------------------------

    def fire(
        self,
        weapon: Weapon,
        origin: Vec2,
        target: Vec2,
        accuracy_modifier: float = 1.0,
        rng: Optional[random.Random] = None,
        source_id: str = "",
    ) -> Projectile:
        """Fire a projectile from *origin* toward *target*.

        The accuracy_modifier scales the weapon's spread cone.  Values
        < 1.0 tighten spread (better aim), > 1.0 widen it.

        Returns the newly created Projectile (also appended to
        ``self.projectiles``).
        """
        r = rng or random.Random()
        self._id_counter += 1
        pid = f"p_{self._id_counter}"

        # Direction from origin to target
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        base_angle = math.atan2(dy, dx)

        # Apply spread
        spread_rad = math.radians(weapon.spread_deg * accuracy_modifier)
        angle = base_angle + r.uniform(-spread_rad / 2, spread_rad / 2)

        vx = math.cos(angle) * weapon.muzzle_velocity
        vy = math.sin(angle) * weapon.muzzle_velocity

        proj = Projectile(
            projectile_id=pid,
            weapon_id=weapon.weapon_id,
            origin=origin,
            position=origin,
            velocity=(vx, vy),
            damage=weapon.damage,
            projectile_type=weapon.projectile_type,
            tracer_color=weapon.tracer_color,
            max_range=weapon.max_range,
            source_id=source_id,
        )
        self.projectiles.append(proj)

        # Record muzzle flash
        self._pending_flashes.append({
            "x": origin[0],
            "y": origin[1],
            "size": weapon.muzzle_flash_size,
            "weapon": weapon.weapon_id,
        })

        return proj

    # -- Tick ---------------------------------------------------------------

    def tick(self, dt: float) -> list[dict]:
        """Advance all projectiles by *dt* seconds.

        Returns a list of impact event dicts for projectiles that
        expired or reached max range this tick.
        """
        impacts: list[dict] = []

        for proj in self.projectiles:
            if not proj.is_active:
                continue

            vx, vy = proj.velocity

            # Gravity for arced projectile types
            if proj.projectile_type in _ARCED_TYPES:
                vy += self.gravity * dt

            # Air resistance (velocity decay)
            drag = 1.0 - self.air_resistance * dt
            drag = max(0.0, drag)
            vx *= drag
            vy *= drag

            proj.velocity = (vx, vy)

            # Update position
            new_x = proj.position[0] + vx * dt
            new_y = proj.position[1] + vy * dt
            proj.position = (new_x, new_y)
            proj.time_alive += dt

            # Check range
            if proj.distance_traveled() >= proj.max_range:
                proj.is_active = False
                effect = _IMPACT_EFFECTS.get(proj.projectile_type, "spark")
                impacts.append({
                    "x": proj.position[0],
                    "y": proj.position[1],
                    "type": proj.projectile_type.value,
                    "damage": proj.damage,
                    "effect": effect,
                    "weapon_id": proj.weapon_id,
                })

        # Clean up dead projectiles
        self.projectiles = [p for p in self.projectiles if p.is_active]
        self._pending_impacts.extend(impacts)

        return impacts

    # -- Three.js export ----------------------------------------------------

    def to_three_js(self) -> dict:
        """Export current state for Three.js rendering.

        Returns a dict with keys:
            projectiles — active in-flight projectiles
            impacts     — impacts since last call to to_three_js
            muzzle_flashes — flashes since last call
        """
        projectiles = []
        for p in self.projectiles:
            if p.is_active:
                projectiles.append({
                    "id": p.projectile_id,
                    "x": p.position[0],
                    "y": p.position[1],
                    "vx": p.velocity[0],
                    "vy": p.velocity[1],
                    "type": p.projectile_type.value,
                    "color": p.tracer_color,
                })

        result = {
            "projectiles": projectiles,
            "impacts": list(self._pending_impacts),
            "muzzle_flashes": list(self._pending_flashes),
        }

        # Clear pending events after export
        self._pending_impacts.clear()
        self._pending_flashes.clear()

        return result


# ---------------------------------------------------------------------------
# AreaEffectManager
# ---------------------------------------------------------------------------

class AreaEffectManager:
    """Manages persistent area effects — smoke, fire, gas, explosions.

    Effects decay over time and can be queried to determine what
    affects a given position.
    """

    def __init__(self) -> None:
        self.effects: list[AreaEffect] = []

    def add(self, effect: AreaEffect) -> None:
        """Add a new area effect."""
        self.effects.append(effect)

    def tick(self, dt: float) -> list[dict]:
        """Advance all effects by *dt* seconds.

        Returns a list of dicts for effects that expired this tick.
        """
        expired: list[dict] = []
        still_active: list[AreaEffect] = []

        for eff in self.effects:
            eff.time_remaining -= dt
            # Intensity decays linearly toward end of life
            if eff.duration > 0:
                eff.intensity = max(0.0, eff.time_remaining / eff.duration)

            if eff.time_remaining <= 0:
                expired.append({
                    "type": eff.effect_type,
                    "x": eff.position[0],
                    "y": eff.position[1],
                    "radius": eff.radius,
                })
            else:
                still_active.append(eff)

        self.effects = still_active
        return expired

    def affects_position(self, pos: Vec2) -> list[AreaEffect]:
        """Return all active effects whose radius covers *pos*."""
        result: list[AreaEffect] = []
        for eff in self.effects:
            if distance(eff.position, pos) <= eff.radius:
                result.append(eff)
        return result

    def to_three_js(self) -> dict:
        """Export active effects for Three.js rendering."""
        effects = []
        for eff in self.effects:
            effects.append({
                "type": eff.effect_type,
                "x": eff.position[0],
                "y": eff.position[1],
                "radius": eff.radius,
                "intensity": round(eff.intensity, 3),
                "color": eff.color,
                "remaining": round(eff.time_remaining, 2),
            })
        return {"effects": effects}


# ---------------------------------------------------------------------------
# ARSENAL — 35+ weapons with realistic stats
# ---------------------------------------------------------------------------

ARSENAL: dict[str, Weapon] = {
    # ========= PISTOLS =========
    "m9_beretta": Weapon(
        weapon_id="m9_beretta", name="M9 Beretta",
        category=WeaponCategory.PISTOL, projectile_type=ProjectileType.BULLET,
        damage=18, fire_rate=2.5, magazine_size=15, reload_time=2.0,
        muzzle_velocity=375, effective_range=50, max_range=200,
        accuracy=0.75, spread_deg=3.5, recoil=0.35, weight_kg=0.95,
        sound_radius=400, tracer_color="#ffcc00", muzzle_flash_size=0.5,
    ),
    "glock17": Weapon(
        weapon_id="glock17", name="Glock 17",
        category=WeaponCategory.PISTOL, projectile_type=ProjectileType.BULLET,
        damage=17, fire_rate=3.0, magazine_size=17, reload_time=1.8,
        muzzle_velocity=375, effective_range=50, max_range=200,
        accuracy=0.78, spread_deg=3.0, recoil=0.30, weight_kg=0.63,
        sound_radius=380, tracer_color="#ffcc00", muzzle_flash_size=0.5,
    ),
    "desert_eagle": Weapon(
        weapon_id="desert_eagle", name="Desert Eagle .50 AE",
        category=WeaponCategory.PISTOL, projectile_type=ProjectileType.BULLET,
        damage=45, fire_rate=1.5, magazine_size=7, reload_time=2.5,
        muzzle_velocity=470, effective_range=60, max_range=250,
        accuracy=0.65, spread_deg=4.0, recoil=0.80, weight_kg=2.05,
        sound_radius=500, tracer_color="#ffaa00", muzzle_flash_size=0.9,
    ),
    "m1911": Weapon(
        weapon_id="m1911", name="M1911 .45 ACP",
        category=WeaponCategory.PISTOL, projectile_type=ProjectileType.BULLET,
        damage=25, fire_rate=2.0, magazine_size=7, reload_time=2.2,
        muzzle_velocity=260, effective_range=50, max_range=200,
        accuracy=0.72, spread_deg=3.5, recoil=0.50, weight_kg=1.10,
        sound_radius=420, tracer_color="#ffcc00", muzzle_flash_size=0.6,
    ),

    # ========= RIFLES =========
    "m4a1": Weapon(
        weapon_id="m4a1", name="M4A1 Carbine",
        category=WeaponCategory.RIFLE, projectile_type=ProjectileType.BULLET,
        damage=28, fire_rate=14.0, magazine_size=30, reload_time=2.5,
        muzzle_velocity=910, effective_range=500, max_range=800,
        accuracy=0.82, spread_deg=2.0, recoil=0.40, weight_kg=3.40,
        sound_radius=600, tracer_color="#ffaa00", muzzle_flash_size=1.0,
    ),
    "ak47": Weapon(
        weapon_id="ak47", name="AK-47",
        category=WeaponCategory.RIFLE, projectile_type=ProjectileType.BULLET,
        damage=33, fire_rate=10.0, magazine_size=30, reload_time=2.8,
        muzzle_velocity=715, effective_range=400, max_range=700,
        accuracy=0.75, spread_deg=3.0, recoil=0.55, weight_kg=3.47,
        sound_radius=650, tracer_color="#ff8800", muzzle_flash_size=1.1,
    ),
    "m16": Weapon(
        weapon_id="m16", name="M16A4",
        category=WeaponCategory.RIFLE, projectile_type=ProjectileType.BULLET,
        damage=30, fire_rate=12.5, magazine_size=30, reload_time=2.5,
        muzzle_velocity=960, effective_range=550, max_range=850,
        accuracy=0.85, spread_deg=1.5, recoil=0.35, weight_kg=3.26,
        sound_radius=600, tracer_color="#ffaa00", muzzle_flash_size=1.0,
    ),
    "scar_h": Weapon(
        weapon_id="scar_h", name="FN SCAR-H",
        category=WeaponCategory.RIFLE, projectile_type=ProjectileType.BULLET,
        damage=38, fire_rate=10.0, magazine_size=20, reload_time=2.8,
        muzzle_velocity=820, effective_range=600, max_range=900,
        accuracy=0.83, spread_deg=1.8, recoil=0.50, weight_kg=3.86,
        sound_radius=650, tracer_color="#ffaa00", muzzle_flash_size=1.1,
    ),
    "famas": Weapon(
        weapon_id="famas", name="FAMAS F1",
        category=WeaponCategory.RIFLE, projectile_type=ProjectileType.BULLET,
        damage=27, fire_rate=16.0, magazine_size=25, reload_time=2.6,
        muzzle_velocity=925, effective_range=450, max_range=750,
        accuracy=0.78, spread_deg=2.2, recoil=0.42, weight_kg=3.61,
        sound_radius=580, tracer_color="#ffaa00", muzzle_flash_size=1.0,
    ),

    # ========= SMGs =========
    "mp5": Weapon(
        weapon_id="mp5", name="HK MP5",
        category=WeaponCategory.SMG, projectile_type=ProjectileType.BULLET,
        damage=20, fire_rate=13.3, magazine_size=30, reload_time=2.0,
        muzzle_velocity=400, effective_range=200, max_range=400,
        accuracy=0.80, spread_deg=2.5, recoil=0.25, weight_kg=2.54,
        sound_radius=350, tracer_color="#ffdd00", muzzle_flash_size=0.6,
    ),
    "p90": Weapon(
        weapon_id="p90", name="FN P90",
        category=WeaponCategory.SMG, projectile_type=ProjectileType.BULLET,
        damage=22, fire_rate=15.0, magazine_size=50, reload_time=2.4,
        muzzle_velocity=715, effective_range=200, max_range=400,
        accuracy=0.78, spread_deg=2.8, recoil=0.20, weight_kg=2.54,
        sound_radius=380, tracer_color="#ffdd00", muzzle_flash_size=0.6,
    ),
    "uzi": Weapon(
        weapon_id="uzi", name="IMI Uzi",
        category=WeaponCategory.SMG, projectile_type=ProjectileType.BULLET,
        damage=18, fire_rate=10.0, magazine_size=32, reload_time=1.8,
        muzzle_velocity=400, effective_range=150, max_range=300,
        accuracy=0.70, spread_deg=4.0, recoil=0.35, weight_kg=3.50,
        sound_radius=350, tracer_color="#ffdd00", muzzle_flash_size=0.5,
    ),
    "mac10": Weapon(
        weapon_id="mac10", name="MAC-10",
        category=WeaponCategory.SMG, projectile_type=ProjectileType.BULLET,
        damage=16, fire_rate=18.0, magazine_size=30, reload_time=1.6,
        muzzle_velocity=366, effective_range=100, max_range=250,
        accuracy=0.60, spread_deg=5.0, recoil=0.50, weight_kg=2.84,
        sound_radius=320, tracer_color="#ffdd00", muzzle_flash_size=0.5,
    ),

    # ========= SNIPERS =========
    "m24": Weapon(
        weapon_id="m24", name="M24 SWS",
        category=WeaponCategory.SNIPER, projectile_type=ProjectileType.BULLET,
        damage=75, fire_rate=0.5, magazine_size=5, reload_time=3.5,
        muzzle_velocity=790, effective_range=800, max_range=1500,
        accuracy=0.95, spread_deg=0.3, recoil=0.60, weight_kg=5.40,
        sound_radius=800, tracer_color="#ffffff", muzzle_flash_size=1.2,
    ),
    "barrett_m82": Weapon(
        weapon_id="barrett_m82", name="Barrett M82A1",
        category=WeaponCategory.SNIPER, projectile_type=ProjectileType.BULLET,
        damage=120, fire_rate=0.4, magazine_size=10, reload_time=4.0,
        muzzle_velocity=890, effective_range=1800, max_range=2500,
        accuracy=0.92, spread_deg=0.2, recoil=0.85, weight_kg=14.0,
        sound_radius=1200, tracer_color="#ffffff", muzzle_flash_size=1.5,
    ),
    "svd_dragunov": Weapon(
        weapon_id="svd_dragunov", name="SVD Dragunov",
        category=WeaponCategory.SNIPER, projectile_type=ProjectileType.BULLET,
        damage=65, fire_rate=1.0, magazine_size=10, reload_time=3.0,
        muzzle_velocity=830, effective_range=800, max_range=1300,
        accuracy=0.90, spread_deg=0.5, recoil=0.55, weight_kg=4.30,
        sound_radius=750, tracer_color="#ffffff", muzzle_flash_size=1.1,
    ),
    "awp": Weapon(
        weapon_id="awp", name="AI Arctic Warfare",
        category=WeaponCategory.SNIPER, projectile_type=ProjectileType.BULLET,
        damage=100, fire_rate=0.4, magazine_size=5, reload_time=3.8,
        muzzle_velocity=850, effective_range=1000, max_range=2000,
        accuracy=0.96, spread_deg=0.15, recoil=0.70, weight_kg=6.50,
        sound_radius=900, tracer_color="#ffffff", muzzle_flash_size=1.3,
    ),

    # ========= LMGs =========
    "m249_saw": Weapon(
        weapon_id="m249_saw", name="M249 SAW",
        category=WeaponCategory.LMG, projectile_type=ProjectileType.BULLET,
        damage=30, fire_rate=14.2, magazine_size=200, reload_time=5.0,
        muzzle_velocity=915, effective_range=600, max_range=1000,
        accuracy=0.72, spread_deg=3.0, recoil=0.45, weight_kg=7.50,
        sound_radius=700, tracer_color="#ff6600", muzzle_flash_size=1.3,
    ),
    "pkm": Weapon(
        weapon_id="pkm", name="PKM",
        category=WeaponCategory.LMG, projectile_type=ProjectileType.BULLET,
        damage=35, fire_rate=12.5, magazine_size=100, reload_time=5.5,
        muzzle_velocity=825, effective_range=600, max_range=1000,
        accuracy=0.70, spread_deg=3.2, recoil=0.50, weight_kg=7.50,
        sound_radius=720, tracer_color="#ff6600", muzzle_flash_size=1.3,
    ),
    "m60": Weapon(
        weapon_id="m60", name="M60",
        category=WeaponCategory.LMG, projectile_type=ProjectileType.BULLET,
        damage=38, fire_rate=9.2, magazine_size=100, reload_time=6.0,
        muzzle_velocity=853, effective_range=600, max_range=1100,
        accuracy=0.68, spread_deg=3.5, recoil=0.55, weight_kg=10.5,
        sound_radius=750, tracer_color="#ff6600", muzzle_flash_size=1.4,
    ),

    # ========= SHOTGUNS =========
    "m870": Weapon(
        weapon_id="m870", name="Remington 870",
        category=WeaponCategory.SHOTGUN, projectile_type=ProjectileType.SHELL,
        damage=90, fire_rate=1.0, magazine_size=8, reload_time=4.5,
        muzzle_velocity=410, effective_range=40, max_range=100,
        accuracy=0.85, spread_deg=12.0, recoil=0.70, weight_kg=3.60,
        sound_radius=500, tracer_color="#ffcc44", muzzle_flash_size=1.4,
    ),
    "spas12": Weapon(
        weapon_id="spas12", name="Franchi SPAS-12",
        category=WeaponCategory.SHOTGUN, projectile_type=ProjectileType.SHELL,
        damage=95, fire_rate=1.2, magazine_size=8, reload_time=4.2,
        muzzle_velocity=420, effective_range=45, max_range=110,
        accuracy=0.82, spread_deg=11.0, recoil=0.75, weight_kg=4.40,
        sound_radius=520, tracer_color="#ffcc44", muzzle_flash_size=1.5,
    ),

    # ========= LAUNCHERS =========
    "rpg7": Weapon(
        weapon_id="rpg7", name="RPG-7",
        category=WeaponCategory.LAUNCHER, projectile_type=ProjectileType.ROCKET,
        damage=200, fire_rate=0.15, magazine_size=1, reload_time=6.0,
        muzzle_velocity=115, effective_range=200, max_range=500,
        accuracy=0.60, spread_deg=3.0, recoil=0.90, weight_kg=7.0,
        sound_radius=1000, tracer_color="#ff4400", muzzle_flash_size=2.0,
    ),
    "m203": Weapon(
        weapon_id="m203", name="M203 Grenade Launcher",
        category=WeaponCategory.LAUNCHER, projectile_type=ProjectileType.GRENADE,
        damage=150, fire_rate=0.2, magazine_size=1, reload_time=4.0,
        muzzle_velocity=76, effective_range=150, max_range=350,
        accuracy=0.65, spread_deg=2.0, recoil=0.60, weight_kg=1.36,
        sound_radius=600, tracer_color="#ff6600", muzzle_flash_size=1.5,
    ),
    "at4": Weapon(
        weapon_id="at4", name="AT4",
        category=WeaponCategory.LAUNCHER, projectile_type=ProjectileType.ROCKET,
        damage=250, fire_rate=0.1, magazine_size=1, reload_time=0.0,
        muzzle_velocity=290, effective_range=300, max_range=500,
        accuracy=0.70, spread_deg=1.5, recoil=0.95, weight_kg=6.70,
        sound_radius=1100, tracer_color="#ff3300", muzzle_flash_size=2.5,
    ),

    # ========= MELEE =========
    "knife": Weapon(
        weapon_id="knife", name="Combat Knife",
        category=WeaponCategory.MELEE, projectile_type=ProjectileType.BULLET,
        damage=40, fire_rate=2.0, magazine_size=999, reload_time=0.0,
        muzzle_velocity=0, effective_range=2.0, max_range=2.0,
        accuracy=0.95, spread_deg=0.0, recoil=0.0, weight_kg=0.30,
        sound_radius=5, tracer_color="#000000", muzzle_flash_size=0.0,
    ),
    "baton": Weapon(
        weapon_id="baton", name="Tactical Baton",
        category=WeaponCategory.MELEE, projectile_type=ProjectileType.BULLET,
        damage=25, fire_rate=1.5, magazine_size=999, reload_time=0.0,
        muzzle_velocity=0, effective_range=2.5, max_range=2.5,
        accuracy=0.90, spread_deg=0.0, recoil=0.0, weight_kg=0.50,
        sound_radius=8, tracer_color="#000000", muzzle_flash_size=0.0,
    ),

    # ========= THROWN =========
    "frag_grenade": Weapon(
        weapon_id="frag_grenade", name="M67 Frag Grenade",
        category=WeaponCategory.THROWN, projectile_type=ProjectileType.GRENADE,
        damage=180, fire_rate=0.25, magazine_size=1, reload_time=0.0,
        muzzle_velocity=15, effective_range=35, max_range=50,
        accuracy=0.70, spread_deg=5.0, recoil=0.0, weight_kg=0.40,
        sound_radius=800, tracer_color="#556b2f", muzzle_flash_size=0.0,
    ),
    "flashbang": Weapon(
        weapon_id="flashbang", name="M84 Stun Grenade",
        category=WeaponCategory.THROWN, projectile_type=ProjectileType.FLASHBANG,
        damage=5, fire_rate=0.25, magazine_size=1, reload_time=0.0,
        muzzle_velocity=14, effective_range=30, max_range=45,
        accuracy=0.72, spread_deg=5.0, recoil=0.0, weight_kg=0.50,
        sound_radius=600, tracer_color="#ffffff", muzzle_flash_size=0.0,
    ),
    "smoke_grenade": Weapon(
        weapon_id="smoke_grenade", name="M18 Smoke Grenade",
        category=WeaponCategory.THROWN, projectile_type=ProjectileType.SMOKE,
        damage=0, fire_rate=0.25, magazine_size=1, reload_time=0.0,
        muzzle_velocity=13, effective_range=30, max_range=45,
        accuracy=0.72, spread_deg=5.0, recoil=0.0, weight_kg=0.54,
        sound_radius=50, tracer_color="#888888", muzzle_flash_size=0.0,
    ),
    "molotov": Weapon(
        weapon_id="molotov", name="Molotov Cocktail",
        category=WeaponCategory.THROWN, projectile_type=ProjectileType.MOLOTOV,
        damage=40, fire_rate=0.2, magazine_size=1, reload_time=0.0,
        muzzle_velocity=12, effective_range=25, max_range=40,
        accuracy=0.65, spread_deg=8.0, recoil=0.0, weight_kg=0.70,
        sound_radius=200, tracer_color="#ff4400", muzzle_flash_size=0.0,
    ),
    "teargas": Weapon(
        weapon_id="teargas", name="CS Gas Grenade",
        category=WeaponCategory.THROWN, projectile_type=ProjectileType.TEARGAS,
        damage=2, fire_rate=0.25, magazine_size=1, reload_time=0.0,
        muzzle_velocity=14, effective_range=30, max_range=45,
        accuracy=0.72, spread_deg=5.0, recoil=0.0, weight_kg=0.45,
        sound_radius=100, tracer_color="#cccc00", muzzle_flash_size=0.0,
    ),

    # ========= TURRETS =========
    "m2_browning": Weapon(
        weapon_id="m2_browning", name="M2 Browning .50 Cal",
        category=WeaponCategory.TURRET, projectile_type=ProjectileType.BULLET,
        damage=80, fire_rate=8.5, magazine_size=100, reload_time=8.0,
        muzzle_velocity=890, effective_range=1800, max_range=2500,
        accuracy=0.78, spread_deg=2.0, recoil=0.60, weight_kg=38.0,
        sound_radius=1500, tracer_color="#ff4400", muzzle_flash_size=2.0,
    ),
    "mk19": Weapon(
        weapon_id="mk19", name="Mk 19 Grenade Launcher",
        category=WeaponCategory.TURRET, projectile_type=ProjectileType.GRENADE,
        damage=160, fire_rate=5.8, magazine_size=48, reload_time=7.0,
        muzzle_velocity=241, effective_range=1500, max_range=2200,
        accuracy=0.65, spread_deg=3.0, recoil=0.50, weight_kg=35.2,
        sound_radius=1200, tracer_color="#ff6600", muzzle_flash_size=1.8,
    ),
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_weapon(weapon_id: str) -> Weapon:
    """Look up a weapon by ID. Raises KeyError if not found."""
    return ARSENAL[weapon_id]


def weapons_by_category(category: WeaponCategory) -> list[Weapon]:
    """Return all weapons in a given category."""
    return [w for w in ARSENAL.values() if w.category == category]


def create_explosion_effect(
    position: Vec2,
    radius: float = 10.0,
    damage_per_second: float = 0.0,
    duration: float = 0.5,
) -> AreaEffect:
    """Create a standard explosion area effect."""
    return AreaEffect(
        effect_type="explosion",
        position=position,
        radius=radius,
        duration=duration,
        intensity=1.0,
        time_remaining=duration,
        damage_per_second=damage_per_second,
        color="#ff4400",
    )


def create_smoke_effect(
    position: Vec2,
    radius: float = 8.0,
    duration: float = 30.0,
) -> AreaEffect:
    """Create a smoke cloud area effect."""
    return AreaEffect(
        effect_type="smoke",
        position=position,
        radius=radius,
        duration=duration,
        intensity=1.0,
        time_remaining=duration,
        damage_per_second=0.0,
        color="#888888",
    )


def create_fire_effect(
    position: Vec2,
    radius: float = 5.0,
    duration: float = 15.0,
    damage_per_second: float = 10.0,
) -> AreaEffect:
    """Create a fire/incendiary area effect."""
    return AreaEffect(
        effect_type="fire",
        position=position,
        radius=radius,
        duration=duration,
        intensity=1.0,
        time_remaining=duration,
        damage_per_second=damage_per_second,
        color="#ff6600",
    )


def create_teargas_effect(
    position: Vec2,
    radius: float = 7.0,
    duration: float = 25.0,
    damage_per_second: float = 3.0,
) -> AreaEffect:
    """Create a teargas area effect."""
    return AreaEffect(
        effect_type="teargas",
        position=position,
        radius=radius,
        duration=duration,
        intensity=1.0,
        time_remaining=duration,
        damage_per_second=damage_per_second,
        color="#cccc00",
    )


def create_flashbang_effect(
    position: Vec2,
    radius: float = 12.0,
    duration: float = 3.0,
) -> AreaEffect:
    """Create a flashbang stun area effect."""
    return AreaEffect(
        effect_type="flashbang",
        position=position,
        radius=radius,
        duration=duration,
        intensity=1.0,
        time_remaining=duration,
        damage_per_second=0.0,
        color="#ffffff",
    )

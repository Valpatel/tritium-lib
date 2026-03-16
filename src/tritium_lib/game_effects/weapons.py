# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Weapon fire patterns, projectile sequences, and firing effects.

Each weapon type defines: fire rate, burst pattern, projectile speed,
muzzle effects, tracer frequency, sound characteristics, and recoil.

The backend computes firing state (cooldowns, bursts, ammo, spread bloom),
producing per-round event dicts that the frontend uses to spawn particles
and play spatial audio.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .particles import ParticleEmitter, muzzle_flash, tracer
from ..game_audio.spatial import SoundEvent


# ---------------------------------------------------------------------------
# Fire modes
# ---------------------------------------------------------------------------

class FireMode(Enum):
    """How a weapon fires when the trigger is pulled."""
    SEMI = "semi"       # One shot per trigger pull
    BURST = "burst"     # N-round burst per trigger pull
    AUTO = "auto"       # Continuous fire while triggered
    BOLT = "bolt"       # Bolt action — slow, powerful, manual cycle
    PUMP = "pump"       # Pump action — shotgun style
    MELEE = "melee"     # Swing/stab — no projectile, contact damage
    THROWN = "thrown"    # Single throw — grenade, knife, rock


# ---------------------------------------------------------------------------
# Weapon profile
# ---------------------------------------------------------------------------

@dataclass
class WeaponProfile:
    """Defines a weapon's static characteristics."""

    name: str
    fire_mode: FireMode
    rpm: float                          # Rounds per minute (600 = 10/sec)
    burst_count: int = 3                # Rounds per burst (BURST mode only)
    projectile_speed: float = 300.0     # m/s
    effective_range: float = 300.0      # meters
    damage: float = 25.0
    spread_deg: float = 2.0            # Base accuracy cone in degrees
    spread_bloom_per_shot: float = 0.3  # Degrees added per consecutive shot
    spread_max_deg: float = 15.0        # Maximum bloom cap
    spread_recovery_rate: float = 5.0   # Degrees/sec recovery toward base
    tracer_every: int = 5               # Every Nth round is a tracer
    muzzle_flash_size: float = 1.0      # Relative size multiplier
    sound_id: str = "rifle_shot"
    recoil_force: float = 1.0

    # Firing sound characteristics
    sound_pitch_base: float = 1.0       # Pitch multiplier
    sound_pitch_variance: float = 0.05  # Random per-shot variation
    sound_volume: float = 1.0

    # Shotgun pellets (only meaningful for PUMP mode)
    pellet_count: int = 1               # Number of projectiles per shot

    @property
    def seconds_per_round(self) -> float:
        """Time between consecutive rounds."""
        if self.rpm <= 0:
            return 1.0
        return 60.0 / self.rpm


# ---------------------------------------------------------------------------
# Pre-built weapon profiles
# ---------------------------------------------------------------------------

WEAPONS: dict[str, WeaponProfile] = {
    # --- Rifles ---
    "m4": WeaponProfile(
        "M4 Carbine", FireMode.AUTO, rpm=700, spread_deg=2.0,
        sound_id="rifle_auto", tracer_every=5,
        spread_bloom_per_shot=0.25, spread_max_deg=10.0,
    ),
    "ak47": WeaponProfile(
        "AK-47", FireMode.AUTO, rpm=600, spread_deg=3.0, damage=30,
        sound_id="rifle_auto", sound_pitch_base=0.85,
        spread_bloom_per_shot=0.4, spread_max_deg=12.0, recoil_force=1.3,
    ),
    "m16_burst": WeaponProfile(
        "M16A4", FireMode.BURST, rpm=800, burst_count=3,
        spread_deg=1.5, sound_id="rifle_burst",
        spread_bloom_per_shot=0.15, spread_max_deg=6.0,
    ),

    # --- Pistols ---
    "pistol_9mm": WeaponProfile(
        "9mm Pistol", FireMode.SEMI, rpm=120,
        effective_range=50, damage=15, spread_deg=4.0,
        sound_id="pistol_shot", muzzle_flash_size=0.5,
        spread_bloom_per_shot=0.5, spread_recovery_rate=8.0,
    ),
    "deagle": WeaponProfile(
        "Desert Eagle", FireMode.SEMI, rpm=80,
        damage=45, spread_deg=3.0, recoil_force=2.5,
        sound_id="pistol_heavy", sound_pitch_base=0.7,
        spread_bloom_per_shot=1.0, spread_max_deg=12.0,
    ),

    # --- SMGs ---
    "mp5": WeaponProfile(
        "MP5", FireMode.AUTO, rpm=800, effective_range=100,
        damage=18, spread_deg=3.0, sound_id="smg_auto",
        muzzle_flash_size=0.6, sound_pitch_base=1.1,
        spread_bloom_per_shot=0.2, spread_max_deg=10.0,
    ),
    "uzi": WeaponProfile(
        "Uzi", FireMode.AUTO, rpm=950, effective_range=80,
        damage=15, spread_deg=5.0, sound_id="smg_auto",
        sound_pitch_base=1.2,
        spread_bloom_per_shot=0.3, spread_max_deg=14.0,
    ),

    # --- Shotguns ---
    "shotgun": WeaponProfile(
        "Pump Shotgun", FireMode.PUMP, rpm=40,
        effective_range=30, damage=80, spread_deg=15.0,
        sound_id="shotgun_pump", muzzle_flash_size=1.5,
        sound_pitch_base=0.6, recoil_force=3.0,
        pellet_count=8, spread_bloom_per_shot=0.0,
    ),

    # --- Sniper ---
    "sniper": WeaponProfile(
        "Sniper Rifle", FireMode.BOLT, rpm=20,
        effective_range=800, damage=90, spread_deg=0.3,
        projectile_speed=900, sound_id="sniper_shot",
        muzzle_flash_size=1.2, sound_pitch_base=0.5,
        spread_bloom_per_shot=0.0, recoil_force=2.0,
    ),

    # --- Machine guns ---
    "m249": WeaponProfile(
        "M249 SAW", FireMode.AUTO, rpm=850,
        effective_range=600, damage=28, spread_deg=3.0,
        tracer_every=4, sound_id="lmg_auto",
        muzzle_flash_size=1.3, recoil_force=1.5,
        spread_bloom_per_shot=0.15, spread_max_deg=8.0,
    ),
    "minigun": WeaponProfile(
        "Minigun", FireMode.AUTO, rpm=3000,
        damage=20, spread_deg=5.0, tracer_every=3,
        sound_id="minigun_spin", muzzle_flash_size=1.5,
        spread_bloom_per_shot=0.0, spread_max_deg=8.0,
    ),

    # --- Pistols (more variety) ---
    "revolver": WeaponProfile(
        "Revolver .357", FireMode.SEMI, rpm=90,
        effective_range=50, damage=40, spread_deg=3.0,
        sound_id="pistol_heavy", muzzle_flash_size=0.8,
        sound_pitch_base=0.75, recoil_force=2.0,
    ),
    "glock": WeaponProfile(
        "Glock 17", FireMode.SEMI, rpm=150,
        effective_range=50, damage=18, spread_deg=3.5,
        sound_id="pistol_shot", muzzle_flash_size=0.5,
        sound_pitch_base=1.05,
    ),

    # --- Melee ---
    "knife": WeaponProfile(
        "Combat Knife", FireMode.MELEE, rpm=120,
        projectile_speed=0, effective_range=2.0, damage=35,
        spread_deg=0, sound_id="knife_slash", muzzle_flash_size=0.0,
        sound_volume=0.6, recoil_force=0.0,
    ),
    "machete": WeaponProfile(
        "Machete", FireMode.MELEE, rpm=80,
        projectile_speed=0, effective_range=2.5, damage=50,
        spread_deg=0, sound_id="blade_swing", muzzle_flash_size=0.0,
        sound_volume=0.7, recoil_force=0.0,
    ),
    "bat": WeaponProfile(
        "Baseball Bat", FireMode.MELEE, rpm=60,
        projectile_speed=0, effective_range=2.5, damage=30,
        spread_deg=0, sound_id="bat_swing", muzzle_flash_size=0.0,
        sound_volume=0.8, recoil_force=0.0, sound_pitch_base=0.6,
    ),
    "fists": WeaponProfile(
        "Fists", FireMode.MELEE, rpm=180,
        projectile_speed=0, effective_range=1.5, damage=10,
        spread_deg=0, sound_id="punch", muzzle_flash_size=0.0,
        sound_volume=0.5, recoil_force=0.0, sound_pitch_base=1.3,
    ),
    "taser": WeaponProfile(
        "Taser", FireMode.SEMI, rpm=10,
        projectile_speed=50, effective_range=5.0, damage=5,
        spread_deg=2.0, sound_id="taser_zap", muzzle_flash_size=0.3,
        sound_volume=0.7, sound_pitch_base=1.8,
    ),

    # --- Thrown ---
    "grenade": WeaponProfile(
        "Frag Grenade", FireMode.THROWN, rpm=15,
        projectile_speed=15, effective_range=30, damage=100,
        spread_deg=5.0, sound_id="grenade_throw", muzzle_flash_size=0.0,
        sound_volume=0.4, recoil_force=0.0,
    ),
    "molotov": WeaponProfile(
        "Molotov Cocktail", FireMode.THROWN, rpm=12,
        projectile_speed=12, effective_range=25, damage=40,
        spread_deg=8.0, sound_id="bottle_throw", muzzle_flash_size=0.0,
        sound_volume=0.4, recoil_force=0.0,
    ),
    "throwing_knife": WeaponProfile(
        "Throwing Knife", FireMode.THROWN, rpm=40,
        projectile_speed=20, effective_range=15, damage=45,
        spread_deg=3.0, sound_id="knife_throw", muzzle_flash_size=0.0,
        sound_volume=0.3, recoil_force=0.0,
    ),
    "rock": WeaponProfile(
        "Rock", FireMode.THROWN, rpm=30,
        projectile_speed=10, effective_range=20, damage=8,
        spread_deg=10.0, sound_id="rock_throw", muzzle_flash_size=0.0,
        sound_volume=0.3, recoil_force=0.0, sound_pitch_base=0.7,
    ),

    # --- Special ---
    "nerf": WeaponProfile(
        "Nerf Blaster", FireMode.SEMI, rpm=60,
        projectile_speed=15, effective_range=15, damage=5,
        spread_deg=8.0, sound_id="nerf_pop",
        muzzle_flash_size=0.0, sound_pitch_base=2.0,
        sound_volume=0.3, spread_bloom_per_shot=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Fired round event
# ---------------------------------------------------------------------------

@dataclass
class FiredRound:
    """Data produced each time a round leaves the barrel."""

    position: tuple[float, float]
    heading: float              # radians
    speed: float                # m/s
    spread_angle: float         # actual angle after spread applied (radians)
    is_tracer: bool
    damage: float
    sound_event: SoundEvent
    muzzle_flash_emitter: Optional[ParticleEmitter]
    weapon_name: str
    round_number: int           # lifetime counter

    def to_dict(self) -> dict:
        """Serialize for WebSocket transport."""
        return {
            "position": list(self.position),
            "heading": round(self.heading, 4),
            "speed": self.speed,
            "spread_angle": round(self.spread_angle, 4),
            "is_tracer": self.is_tracer,
            "damage": self.damage,
            "sound": self.sound_event.to_dict(),
            "weapon": self.weapon_name,
            "round_number": self.round_number,
        }


# ---------------------------------------------------------------------------
# Weapon firer — manages firing state
# ---------------------------------------------------------------------------

class WeaponFirer:
    """Manages firing state for a weapon — cooldown, burst counting, ammo.

    Usage::

        firer = WeaponFirer(WEAPONS["m4"], ammo=30)
        firer.pull_trigger()

        # Game loop
        while running:
            rounds = firer.tick(dt)
            for r in rounds:
                # spawn particles, play sound, apply damage
                ...

        firer.release_trigger()
    """

    def __init__(self, weapon: WeaponProfile, ammo: int = 30,
                 position: tuple[float, float] = (0.0, 0.0),
                 heading: float = 0.0):
        self.weapon = weapon
        self.ammo = ammo
        self.max_ammo = ammo
        self.cooldown = 0.0             # Seconds until next round can fire
        self.burst_remaining = 0        # Rounds left in current burst
        self.is_firing = False          # Trigger held
        self.total_rounds_fired = 0
        self.position = position
        self.heading = heading

        # Spread bloom tracking
        self._current_spread_bloom = 0.0  # Additional spread from sustained fire
        self._trigger_consumed = False    # For semi/bolt/pump: one shot per pull

    # -- Trigger control ----------------------------------------------------

    def pull_trigger(self) -> bool:
        """Start firing. Returns True if the weapon can fire."""
        if self.ammo <= 0:
            return False
        self.is_firing = True
        self._trigger_consumed = False

        # For burst mode, start a new burst
        if self.weapon.fire_mode == FireMode.BURST and self.burst_remaining <= 0:
            self.burst_remaining = self.weapon.burst_count

        return True

    def release_trigger(self) -> None:
        """Stop firing."""
        self.is_firing = False
        self._trigger_consumed = False
        # Burst in progress continues until complete
        # (burst_remaining will drain naturally)

    # -- Tick ---------------------------------------------------------------

    def tick(self, dt: float) -> list[FiredRound]:
        """Advance by *dt* seconds. Returns list of rounds fired this tick.

        Call every frame. The firer respects RPM timing — at 600 RPM,
        rounds fire every 0.1s regardless of tick rate.
        """
        fired: list[FiredRound] = []

        # Recover spread bloom over time
        if self._current_spread_bloom > 0:
            recovery = self.weapon.spread_recovery_rate * dt
            self._current_spread_bloom = max(
                0.0, self._current_spread_bloom - recovery
            )

        # Drain cooldown
        self.cooldown = max(0.0, self.cooldown - dt)

        # Determine if we should fire
        # Accumulate time-budget so multiple rounds can fire in one tick
        # (important for high-RPM weapons like minigun at low tick rates)
        time_budget = dt if self.cooldown <= 0 else 0.0

        while self._should_fire() and self.cooldown <= 0 and self.ammo > 0:
            round_event = self._fire_one_round()
            fired.append(round_event)

            # Add cooldown for next round
            self.cooldown += self.weapon.seconds_per_round

            # For semi/bolt/pump — only one shot per trigger pull
            if self.weapon.fire_mode in (
                FireMode.SEMI, FireMode.BOLT, FireMode.PUMP
            ):
                self._trigger_consumed = True

            # For burst mode, decrement burst counter
            if self.weapon.fire_mode == FireMode.BURST:
                self.burst_remaining -= 1
                if self.burst_remaining <= 0:
                    self._trigger_consumed = True

            # Safety: don't fire more rounds than time allows in one tick
            # Allow up to dt / seconds_per_round rounds
            spr = self.weapon.seconds_per_round
            max_rounds_this_tick = max(1, int(dt / spr) + 1) if spr > 0 else 1
            if len(fired) >= max_rounds_this_tick:
                break

        return fired

    def reload(self, ammo: Optional[int] = None) -> None:
        """Reload the weapon. Defaults to max_ammo if not specified."""
        self.ammo = ammo if ammo is not None else self.max_ammo

    # -- Internal -----------------------------------------------------------

    def _should_fire(self) -> bool:
        """Determine if the weapon should fire on the next opportunity."""
        if self.ammo <= 0:
            return False

        mode = self.weapon.fire_mode

        if mode == FireMode.AUTO:
            return self.is_firing

        if mode in (FireMode.SEMI, FireMode.BOLT, FireMode.PUMP):
            return self.is_firing and not self._trigger_consumed

        if mode == FireMode.BURST:
            # Fire if burst is in progress, OR trigger just pulled
            if self.burst_remaining > 0:
                return True
            return self.is_firing and not self._trigger_consumed

        return False

    def _fire_one_round(self) -> FiredRound:
        """Produce one round with spread, sound, and muzzle flash."""
        self.ammo -= 1
        self.total_rounds_fired += 1

        # Compute spread angle
        total_spread_deg = (
            self.weapon.spread_deg + self._current_spread_bloom
        )
        total_spread_deg = min(total_spread_deg, self.weapon.spread_max_deg)
        spread_rad = math.radians(total_spread_deg)
        shot_angle = self.heading + random.uniform(
            -spread_rad / 2, spread_rad / 2
        )

        # Bloom increases after each shot
        self._current_spread_bloom += self.weapon.spread_bloom_per_shot

        # Is this a tracer round?
        is_tracer = (
            self.weapon.tracer_every > 0
            and self.total_rounds_fired % self.weapon.tracer_every == 0
        )

        # Sound event with per-shot pitch variation
        pitch = self.weapon.sound_pitch_base + random.uniform(
            -self.weapon.sound_pitch_variance,
            self.weapon.sound_pitch_variance,
        )
        sound = SoundEvent(
            sound_id=self.weapon.sound_id,
            position=self.position,
            volume=self.weapon.sound_volume,
            pitch=pitch,
            category="weapon",
        )

        # Muzzle flash particle emitter (skip if size is zero, e.g. nerf)
        flash = None
        if self.weapon.muzzle_flash_size > 0:
            flash = muzzle_flash(self.position, self.heading)
            # Scale the flash particles by the weapon's muzzle_flash_size
            if self.weapon.muzzle_flash_size != 1.0:
                scale = self.weapon.muzzle_flash_size
                flash.size_range = (
                    flash.size_range[0] * scale,
                    flash.size_range[1] * scale,
                )

        # Handle shotgun pellets — damage split across pellets
        damage = self.weapon.damage
        if self.weapon.pellet_count > 1:
            damage = self.weapon.damage / self.weapon.pellet_count

        return FiredRound(
            position=self.position,
            heading=shot_angle,
            speed=self.weapon.projectile_speed,
            spread_angle=shot_angle - self.heading,
            is_tracer=is_tracer,
            damage=damage,
            sound_event=sound,
            muzzle_flash_emitter=flash,
            weapon_name=self.weapon.name,
            round_number=self.total_rounds_fired,
        )

    # -- Queries ------------------------------------------------------------

    @property
    def can_fire(self) -> bool:
        """True if the weapon has ammo and is not on cooldown."""
        return self.ammo > 0 and self.cooldown <= 0

    @property
    def current_spread_deg(self) -> float:
        """Current total spread including bloom."""
        return min(
            self.weapon.spread_deg + self._current_spread_bloom,
            self.weapon.spread_max_deg,
        )

    def to_dict(self) -> dict:
        """Export firer state for frontend HUD."""
        return {
            "weapon": self.weapon.name,
            "fire_mode": self.weapon.fire_mode.value,
            "ammo": self.ammo,
            "max_ammo": self.max_ammo,
            "is_firing": self.is_firing,
            "cooldown": round(self.cooldown, 4),
            "spread_deg": round(self.current_spread_deg, 2),
            "total_rounds_fired": self.total_rounds_fired,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_firer(weapon_id: str, ammo: Optional[int] = None,
                 position: tuple[float, float] = (0.0, 0.0),
                 heading: float = 0.0) -> WeaponFirer:
    """Create a WeaponFirer from a weapon ID string.

    Raises KeyError if weapon_id is not in WEAPONS.
    """
    profile = WEAPONS[weapon_id]
    default_ammo = {
        FireMode.SEMI: 15,
        FireMode.BURST: 30,
        FireMode.AUTO: 30,
        FireMode.BOLT: 5,
        FireMode.PUMP: 8,
    }
    if ammo is None:
        ammo = default_ammo.get(profile.fire_mode, 30)
    return WeaponFirer(profile, ammo=ammo, position=position, heading=heading)

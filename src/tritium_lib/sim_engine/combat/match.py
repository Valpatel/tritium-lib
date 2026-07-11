# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MatchReferee — transport-agnostic nerf-match duel scoring.

North Star, both halves
-----------------------
FUN: robot-dog nerf matches on the tactical map — seed two (or more) dogs,
feed the referee their poses, resolve shots, watch a scoreboard climb to a
KO.  PRODUCTION: the *same* referee arbitrates hits for real nerf-hardware
matches — physical robot dogs report fire via their ``WeaponStatus`` / fire
telemetry over MQTT, a camera or operator feeds poses, and this class scores
the match with the identical ballistics calibration the simulator uses.  One
scoring brain for sim units, wire robots, and future physical nerf matches.

This module is pure stdlib + intra-package imports: no MQTT, no asyncio, no
framework deps.  Whatever transport delivers poses and trigger pulls
(MQTT bridge, sim tick loop, REST endpoint, camera scorer) simply calls
``update_pose()`` and ``resolve_shot()``.

Pan convention (matches examples/robot-template + WeaponStatus)
---------------------------------------------------------------
``turret_pan_deg`` is the BODY-RELATIVE servo angle — 0 means the turret
points straight down the chassis heading, positive clockwise, physically
clamped to ``[PAN_MIN_DEG, PAN_MAX_DEG]``.  The world-frame aim bearing is::

    world_aim = heading_deg + turret_pan_deg

This mirrors the examples/robot-template turret servo and the
``WeaponStatus.pan_deg`` telemetry contract in tritium-sc: what a real dog
reports is directly what the referee consumes.

Ballistics reuse
----------------
Hit/miss uses the EXISTING combat dispersion calibration
(:func:`~tritium_lib.sim_engine.combat.combat.dispersion_sigma`): the lateral
miss at the target is ``distance * tan(aim_error) + N(0, sigma)`` and a shot
lands when that offset is within ``hit_radius``.  With perfect aim, the hit
probability equals the weapon's ``accuracy`` — the same self-calibration
proven by the projectile-flight sim's goldens.

This is Tritium stand-in logic (basic video-game-style scoring support), NOT
Graphling cognition — the referee arbitrates outcomes; it never decides for
a machine.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from tritium_lib.models.fire_control import (
    FireSolution,
    PAN_MAX_DEG,
    PAN_MIN_DEG,
    TILT_MAX_DEG,
    TILT_MIN_DEG,
)

from .combat import HIT_RADIUS, dispersion_sigma
from .weapons import _DEFAULT_WEAPONS, Weapon


def _normalize180(angle_deg: float) -> float:
    """Normalize an angle in degrees into ``(-180, 180]``."""
    a = math.fmod(angle_deg, 360.0)
    if a > 180.0:
        a -= 360.0
    elif a <= -180.0:
        a += 360.0
    return a


def relative_fire_solution(
    source_pos: tuple[float, float],
    heading_deg: float,
    target_pos: tuple[float, float],
    arc_peak: float = 0.0,
) -> tuple[FireSolution, bool]:
    """Compute a BODY-RELATIVE turret aim solution for a heading-bearing chassis.

    Like :func:`~tritium_lib.models.fire_control.compute_fire_solution`, but
    the returned ``pan`` is the servo angle relative to the chassis heading
    (robot dog turret), not the world compass bearing::

        rel = normalize180(world_bearing - heading_deg)

    Args:
        source_pos: ``(x, y)`` world position of the shooter.
        heading_deg: chassis heading — sim compass convention, ``0 = +y``
            (north), increasing clockwise.
        target_pos: ``(x, y)`` world position of the target.
        arc_peak: peak Z height of the projectile arc in world units
            (mirrors ``compute_fire_solution``); ``0`` for a flat shot.

    Returns:
        ``(solution, in_arc)`` — the :class:`FireSolution` with ``pan``
        clamped to ``[PAN_MIN_DEG, PAN_MAX_DEG]`` (and tilt to its servo
        window), and ``in_arc`` computed BEFORE clamping:
        ``|rel| <= PAN_MAX_DEG``.  A target behind the shoulder is out of
        the servo arc — the chassis must turn before the turret can bear.
    """
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]
    distance = math.hypot(dx, dy)

    world_bearing = math.degrees(math.atan2(dx, dy))
    rel = _normalize180(world_bearing - heading_deg)
    in_arc = abs(rel) <= PAN_MAX_DEG
    pan = max(PAN_MIN_DEG, min(PAN_MAX_DEG, rel))

    if arc_peak > 0.0:
        tilt = math.degrees(math.atan2(4.0 * arc_peak, distance))
        tilt = max(TILT_MIN_DEG, min(TILT_MAX_DEG, tilt))
    else:
        tilt = 0.0

    return FireSolution(pan=pan, tilt=tilt, distance=distance), in_arc


@dataclass
class MatchCombatant:
    """One entrant in a nerf match — pose, weapon, hitpoints, and stats.

    Pose fields use the sim compass convention (``heading_deg``: 0 = +y
    north, clockwise) and the body-relative turret servo convention
    (``turret_pan_deg``: 0 = straight ahead) — the same frame a real dog's
    ``WeaponStatus.pan_deg`` reports.
    """

    combatant_id: str
    weapon: Weapon
    hp: float = 40.0
    max_hp: float = 40.0
    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0
    turret_pan_deg: float = 0.0
    turret_tilt_deg: float = 0.0
    shots_fired: int = 0
    hits_landed: int = 0
    hits_taken: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0

    @property
    def alive(self) -> bool:
        """A combatant fights while its hitpoints are above zero."""
        return self.hp > 0


@dataclass
class ShotOutcome:
    """The referee's verdict on a single trigger pull."""

    shooter_id: str
    target_id: str
    hit: bool
    distance_m: float
    aim_error_deg: float
    lateral_offset_m: float
    damage_applied: float
    target_hp_after: float
    target_destroyed: bool
    reason: str  # "hit" | "dispersion_miss" | "out_of_range" | "aim_off"


class MatchReferee:
    """Scores a nerf duel/match between registered combatants.

    Transport-agnostic: callers push poses (from the sim tick, MQTT
    telemetry, or a camera tracker) via :meth:`update_pose` and adjudicate
    trigger pulls via :meth:`resolve_shot`.  All randomness flows through
    the injected ``rng`` so a match is bit-for-bit reproducible under a
    fixed seed — the same golden-replay determinism contract as
    :class:`~tritium_lib.sim_engine.combat.combat.CombatSystem`.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
        hit_radius: float = HIT_RADIUS,
    ) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._hit_radius = hit_radius
        self._combatants: dict[str, MatchCombatant] = {}

    # ------------------------------------------------------------------
    # Registration & pose feed
    # ------------------------------------------------------------------

    def add_combatant(
        self,
        combatant_id: str,
        weapon: Weapon | None = None,
        asset_type: str = "robot_dog",
        hp: float = 40.0,
    ) -> MatchCombatant:
        """Register a combatant.  ``weapon=None`` equips the default loadout
        for *asset_type* (a fresh copy, same table :class:`WeaponSystem`
        uses — unknown types get the generic blaster)."""
        if weapon is None:
            template = _DEFAULT_WEAPONS.get(asset_type)
            weapon = replace(template) if template is not None else Weapon()
        combatant = MatchCombatant(
            combatant_id=combatant_id, weapon=weapon, hp=hp, max_hp=hp,
        )
        self._combatants[combatant_id] = combatant
        return combatant

    def get_combatant(self, combatant_id: str) -> MatchCombatant | None:
        """Return the registered combatant, or ``None`` if unknown."""
        return self._combatants.get(combatant_id)

    def update_pose(
        self,
        combatant_id: str,
        x: float,
        y: float,
        heading_deg: float,
        turret_pan_deg: float | None = None,
        turret_tilt_deg: float | None = None,
    ) -> None:
        """Update a combatant's chassis pose and (optionally) turret servos.

        ``turret_pan_deg`` / ``turret_tilt_deg`` left as ``None`` preserve
        the last reported servo angles — a pose-only telemetry packet must
        not re-center the turret.
        """
        c = self._combatants[combatant_id]
        c.x = x
        c.y = y
        c.heading_deg = heading_deg
        if turret_pan_deg is not None:
            c.turret_pan_deg = turret_pan_deg
        if turret_tilt_deg is not None:
            c.turret_tilt_deg = turret_tilt_deg

    # ------------------------------------------------------------------
    # Shot adjudication
    # ------------------------------------------------------------------

    def resolve_shot(
        self,
        shooter_id: str,
        target_id: str,
        spread_factor: float = 1.0,
    ) -> ShotOutcome:
        """Adjudicate one trigger pull from *shooter_id* at *target_id*.

        Miss gates, in order:

        1. ``out_of_range`` — distance beyond ``weapon.weapon_range``.  NO
           rng draw is consumed: range gating must not shift the dispersion
           draw sequence, so a match replayed with different arena geometry
           keeps deterministic per-shot draws for the shots that do engage.
        2. ``aim_off`` — the world aim bearing (``heading + turret_pan``) is
           90 deg or more off the true bearing; the dart physically cannot
           arc back.  Also no rng draw, same determinism rationale.
        3. Otherwise the lateral offset at the target is
           ``distance * tan(aim_error) + N(0, dispersion_sigma(accuracy)
           * spread_factor)``; within ``hit_radius`` scores a ``hit``,
           outside is a ``dispersion_miss``.

        ``shots_fired`` increments on EVERY pull, whatever the outcome —
        a real trigger pull is a real trigger pull.
        """
        shooter = self._combatants[shooter_id]
        target = self._combatants[target_id]
        shooter.shots_fired += 1

        dx = target.x - shooter.x
        dy = target.y - shooter.y
        distance = math.hypot(dx, dy)

        if distance > shooter.weapon.weapon_range:
            return ShotOutcome(
                shooter_id=shooter_id, target_id=target_id, hit=False,
                distance_m=distance, aim_error_deg=0.0, lateral_offset_m=0.0,
                damage_applied=0.0, target_hp_after=target.hp,
                target_destroyed=False, reason="out_of_range",
            )

        world_aim = shooter.heading_deg + shooter.turret_pan_deg
        true_bearing = math.degrees(math.atan2(dx, dy))
        aim_error = _normalize180(world_aim - true_bearing)

        if abs(aim_error) >= 90.0:
            return ShotOutcome(
                shooter_id=shooter_id, target_id=target_id, hit=False,
                distance_m=distance, aim_error_deg=aim_error,
                lateral_offset_m=0.0, damage_applied=0.0,
                target_hp_after=target.hp, target_destroyed=False,
                reason="aim_off",
            )

        sigma = dispersion_sigma(shooter.weapon.accuracy) * spread_factor
        offset = distance * math.tan(math.radians(aim_error))
        offset += self._rng.gauss(0.0, sigma)

        if abs(offset) <= self._hit_radius:
            damage = shooter.weapon.damage
            target.hp = max(0.0, target.hp - damage)
            shooter.hits_landed += 1
            shooter.damage_dealt += damage
            target.hits_taken += 1
            target.damage_taken += damage
            return ShotOutcome(
                shooter_id=shooter_id, target_id=target_id, hit=True,
                distance_m=distance, aim_error_deg=aim_error,
                lateral_offset_m=offset, damage_applied=damage,
                target_hp_after=target.hp,
                target_destroyed=not target.alive, reason="hit",
            )

        return ShotOutcome(
            shooter_id=shooter_id, target_id=target_id, hit=False,
            distance_m=distance, aim_error_deg=aim_error,
            lateral_offset_m=offset, damage_applied=0.0,
            target_hp_after=target.hp, target_destroyed=False,
            reason="dispersion_miss",
        )

    # ------------------------------------------------------------------
    # Match state
    # ------------------------------------------------------------------

    def winner(self) -> str | None:
        """ID of the sole living combatant — only when at least two are
        registered and exactly one remains alive; otherwise ``None``."""
        if len(self._combatants) < 2:
            return None
        living = [c for c in self._combatants.values() if c.alive]
        if len(living) == 1:
            return living[0].combatant_id
        return None

    def active(self) -> bool:
        """The match is live while at least two combatants stand."""
        return sum(1 for c in self._combatants.values() if c.alive) >= 2

    def scoreboard(self) -> dict:
        """Match snapshot — winner (if decided) plus per-combatant stats.

        Shape is JSON-safe for any transport (HUD panel, MQTT publish,
        REST response)."""
        return {
            "winner": self.winner(),
            "combatants": {
                cid: {
                    "hp": c.hp,
                    "max_hp": c.max_hp,
                    "alive": c.alive,
                    "shots_fired": c.shots_fired,
                    "hits_landed": c.hits_landed,
                    "hits_taken": c.hits_taken,
                    "damage_dealt": c.damage_dealt,
                    "damage_taken": c.damage_taken,
                    "weapon": c.weapon.name,
                }
                for cid, c in self._combatants.items()
            },
        }

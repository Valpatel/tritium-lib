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

Hit feedback (wire matches)
---------------------------
In a wire match the referee only ADJUDICATES — the dog owns its health.  A
``resolve_shot`` verdict becomes a ``register_hit`` command on the target's
command topic (:func:`register_hit_command`); the dog applies the damage
with its own ``HealthTracker``, publishes a ``HitReport`` on
``tritium/{site}/robots/{id}/hit``, and embeds ``HealthStatus`` in its
telemetry.  That reported health is AUTHORITATIVE: the referee pins its
book to it via :meth:`MatchReferee.sync_health`, and KO resolves on the
dog-reported health, not the referee's internal ledger.  Hits the referee
never saw (physical hit sensor, camera impact) enter the book through
:meth:`MatchReferee.register_external_hit`.

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
from tritium_lib.models.hits import RegisterHitCommand

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

    ``team`` groups combatants for a team match; ``None`` (the default) means
    the combatant is its OWN team — a free-for-all entrant.  ``defeated_seq``
    is a KO-order stamp (0 while standing; the referee sets it to a strictly
    increasing value the instant the body first reaches 0 hp) so a
    simultaneous multi-KO is broken deterministically: the body that fell
    LAST — the higher ``defeated_seq`` — survived longest and takes the tie.
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
    team: str | None = None
    defeated_seq: int = 0

    @property
    def alive(self) -> bool:
        """A combatant fights while its hitpoints are above zero."""
        return self.hp > 0

    @property
    def team_id(self) -> str:
        """Effective team — the explicit ``team``, else the combatant's own id
        (free-for-all: every dog is its own side)."""
        return self.team if self.team is not None else self.combatant_id


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
        # Monotonic KO-order counter — stamps ``defeated_seq`` so the winner
        # of a simultaneous multi-KO is decided deterministically (last to
        # fall wins), never by dict/registration order.  No rng: the golden
        # dispersion sequence is untouched.
        self._ko_counter: int = 0

    # ------------------------------------------------------------------
    # Registration & pose feed
    # ------------------------------------------------------------------

    def add_combatant(
        self,
        combatant_id: str,
        weapon: Weapon | None = None,
        asset_type: str = "robot_dog",
        hp: float = 40.0,
        team: str | None = None,
    ) -> MatchCombatant:
        """Register a combatant.  ``weapon=None`` equips the default loadout
        for *asset_type* (a fresh copy, same table :class:`WeaponSystem`
        uses — unknown types get the generic blaster).

        ``team`` groups the combatant for a team match (last team standing
        wins); ``None`` (default) makes it its own side — a free-for-all
        entrant.  Any number of combatants may be registered: two for a duel,
        N for a free-for-all, or several teams.
        """
        if weapon is None:
            template = _DEFAULT_WEAPONS.get(asset_type)
            weapon = replace(template) if template is not None else Weapon()
        combatant = MatchCombatant(
            combatant_id=combatant_id, weapon=weapon, hp=hp, max_hp=hp,
            team=team,
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
            self._stamp_defeats()
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
    # Hit feedback — the dog owns its health, the referee mirrors it
    # ------------------------------------------------------------------

    def sync_health(
        self,
        combatant_id: str,
        hp: float,
        alive: bool | None = None,
        max_hp: float | None = None,
    ) -> None:
        """Pin the referee's book to a dog's AUTHORITATIVE reported health.

        Wire matches: the dog's own health telemetry (its ``HealthTracker``
        -> ``HealthStatus`` block) is the ground truth; the referee's ledger
        is a mirror.  ``hp`` is clamped ``>= 0``.  ``alive=False`` forces
        the book to 0 hp even if the reported hp is positive — a dog
        declaring itself dead is dead; ``alive=True`` / ``None`` leaves the
        clamped hp authoritative (``MatchCombatant.alive`` derives from it).

        ``max_hp`` mirrors the dog's OWN pool when its reported ``max_hp``
        differs from the seed the referee was constructed with (a dog whose
        config sets a different hitpoint pool than the match's ``--hp``).
        The dog owns its body: when it reports a ``max_hp`` the scoreboard
        should reflect that, not the driver's guess.  ``None`` leaves the
        seeded pool untouched; a supplied value is clamped ``>= hp`` so the
        book never shows more hp than pool.  Unknown *combatant_id* raises
        ``KeyError``, same contract as :meth:`update_pose`.
        """
        c = self._combatants[combatant_id]
        c.hp = max(0.0, hp)
        if alive is False:
            c.hp = 0.0
        if max_hp is not None:
            c.max_hp = max(c.hp, float(max_hp))
        self._stamp_defeats()

    def register_external_hit(
        self,
        target_id: str,
        damage: float,
        shooter_id: str | None = None,
    ) -> float:
        """Book a hit the referee did NOT adjudicate — hit sensor / camera.

        A physical dog's hit sensor or a camera-detected foam impact
        reports damage the referee never rolled for: drain the target's hp
        (clamped at 0) and count ``hits_taken`` / ``damage_taken``.  If
        *shooter_id* names a registered combatant, credit its
        ``hits_landed`` / ``damage_dealt`` — but NOT ``shots_fired``: the
        trigger pull was already counted when the ammo drained.  Unknown or
        ``None`` shooters (a camera saw an impact but not who fired) book
        the target side only.  No rng is involved — the dispersion draw
        sequence, and therefore golden replays, are untouched.

        Returns the target's hp after the hit.  Unknown *target_id* raises
        ``KeyError``, same contract as the other methods.
        """
        target = self._combatants[target_id]
        damage = max(0.0, damage)
        target.hp = max(0.0, target.hp - damage)
        target.hits_taken += 1
        target.damage_taken += damage
        shooter = self._combatants.get(shooter_id) if shooter_id else None
        if shooter is not None:
            shooter.hits_landed += 1
            shooter.damage_dealt += damage
        self._stamp_defeats()
        return target.hp

    def _stamp_defeats(self) -> None:
        """Stamp ``defeated_seq`` on any body that has newly reached 0 hp.

        Called after every hp mutation (``resolve_shot`` hit, ``sync_health``,
        ``register_external_hit``): a body dropping to 0 for the FIRST time is
        given the next KO-order number.  A single hp mutation kills at most one
        body, so at most one stamp is assigned per call and iteration order is
        immaterial.  This is the deterministic backbone of the multi-KO
        tie-break — no rng, so golden dispersion replays are untouched.
        """
        for c in self._combatants.values():
            if c.hp <= 0.0 and c.defeated_seq == 0:
                self._ko_counter += 1
                c.defeated_seq = self._ko_counter

    # ------------------------------------------------------------------
    # Match state
    # ------------------------------------------------------------------

    def _living_teams(self) -> set[str]:
        """Team ids with at least one living member (free-for-all: each living
        combatant is its own team)."""
        return {c.team_id for c in self._combatants.values() if c.alive}

    def winner(self) -> str | None:
        """ID of the sole living combatant — only when at least two are
        registered and exactly one remains alive; otherwise ``None``.

        This is the free-for-all, combatant-level notion (unchanged contract).
        For a team match where the winning team keeps more than one survivor,
        this returns ``None`` — use :meth:`winning_team` or :meth:`decide_winner`.
        A simultaneous multi-KO (nobody alive) also returns ``None`` here;
        :meth:`decide_winner` applies the tie-break for that case.
        """
        if len(self._combatants) < 2:
            return None
        living = [c for c in self._combatants.values() if c.alive]
        if len(living) == 1:
            return living[0].combatant_id
        return None

    def winning_team(self) -> str | None:
        """The sole surviving team id — only when >= 2 combatants are
        registered and exactly one team still has a living member; otherwise
        ``None`` (match still contested, or a simultaneous multi-KO left no
        team standing — :meth:`decide_winner` breaks that tie)."""
        if len(self._combatants) < 2:
            return None
        teams = self._living_teams()
        if len(teams) == 1:
            return next(iter(teams))
        return None

    def _rank_key(self, c: MatchCombatant) -> tuple:
        """Sort key ranking combatants best-first (ascending sort).

        A living body always outranks a dead one; among the living, more hp
        wins; among the fallen, the one that fell LAST (higher ``defeated_seq``)
        wins — it survived longest.  The combatant id is the final, fully
        deterministic backstop so the order never depends on dict insertion.
        """
        return (
            0 if c.alive else 1,      # living beats dead
            -c.hp,                    # more hp is better
            -c.defeated_seq,          # fell later (higher seq) is better
            c.combatant_id,           # deterministic tie-breaker of last resort
        )

    def decide_winner(self) -> str | None:
        """Definitive match-winner id — deterministic even on a simultaneous
        multi-KO, so a caller never has to fall back to dict/list order.

        Returns ``None`` while the match is still contested (>= 2 teams alive)
        or when fewer than two combatants are registered.  Once the match is
        settled (0 or 1 living team) it ranks ALL combatants with
        :meth:`_rank_key` and returns the champion: the sole survivor, the
        winning team's healthiest member, or — if everyone fell together — the
        body that fell last.  The result is stable across runs and independent
        of registration order or ``PYTHONHASHSEED``.
        """
        if len(self._combatants) < 2:
            return None
        if len(self._living_teams()) >= 2:
            return None
        return min(self._combatants.values(), key=self._rank_key).combatant_id

    def active(self) -> bool:
        """The match is live while at least two distinct teams still stand
        (free-for-all: at least two combatants alive)."""
        return len(self._living_teams()) >= 2

    def scoreboard(self) -> dict:
        """Match snapshot — winner (if decided) plus per-combatant stats.

        ``winner`` keeps its free-for-all, sole-survivor meaning (``None``
        while contested).  ``decided_winner`` is the tie-broken definitive
        winner (``None`` only while the match is genuinely still contested), so
        a consumer never sees a null/order-dependent result at match end.
        ``winning_team`` names the surviving team (== ``decided_winner``'s team
        for a decided match).  Shape is JSON-safe for any transport (HUD panel,
        MQTT publish, REST response)."""
        return {
            "winner": self.winner(),
            "decided_winner": self.decide_winner(),
            "winning_team": self.winning_team(),
            "combatants": {
                cid: {
                    "hp": c.hp,
                    "max_hp": c.max_hp,
                    "alive": c.alive,
                    "team": c.team_id,
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


def register_hit_command(outcome: ShotOutcome) -> RegisterHitCommand:
    """Build the wire command that tells the TARGET dog it was hit.

    Maps a referee verdict onto the hit-feedback wire contract: the command
    is published on the target's EXISTING command topic
    (``tritium/{site}/robots/{target_id}/command``) so the dog applies the
    damage to its own ``HealthTracker`` and closes the loop with a
    ``HitReport`` + health telemetry.  ``source`` is always ``"referee"``
    here — this helper is the referee's mouth; sensor and camera paths
    build their own :class:`~tritium_lib.models.hits.RegisterHitCommand`.
    """
    return RegisterHitCommand(
        shooter_id=outcome.shooter_id,
        damage=outcome.damage_applied,
        source="referee",
    )

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Hit-feedback models — the wire contract for taking damage and owning health.

North Star, both halves
-----------------------
FUN: a dog that KNOWS it has been hit can yelp, limp, and drop on the
tactical map — the scoreboard climbs from the dog's own telemetry, not a
hidden referee ledger.  PRODUCTION: a real nerf dog carries a physical hit
sensor, a camera scorer detects foam impacts, an operator can adjudicate
manually.  All of them report through THIS schema, so the sim referee and a
physical arena exercise the identical wire path — digital-twin parity for
taking damage, not just dealing it.

Direction of authority
----------------------
An adjudicator (sim :class:`~tritium_lib.sim_engine.combat.match.MatchReferee`,
camera scorer, or operator) tells a robot it was hit with a
:class:`RegisterHitCommand` on the EXISTING command topic::

    tritium/{site}/robots/{robot_id}/command   (QoS 1, retain False)

— the same topic :class:`~tritium_lib.models.fire_control.TurretAimCommand` /
:class:`~tritium_lib.models.fire_control.FireCommand` ride, keyed by the
``command`` field, so the reference brain in ``examples/robot-template/``
(robot.py keys on ``"command"``) needs no new subscription.

The robot then OWNS its health.  It applies the damage through a
:class:`HealthTracker`, publishes a :class:`HitReport` on the NEW hit topic::

    tritium/{site}/robots/{robot_id}/hit       (QoS 1, retain False)

whenever damage lands (whatever the source — referee verdict, onboard hit
sensor, camera impact), and embeds a :class:`HealthStatus` block under the
``"health"`` key of its regular telemetry payload, exactly as
:class:`~tritium_lib.models.fire_control.WeaponStatus` embeds under
``"weapon_status"``.  The dog's reported health is AUTHORITATIVE: the
referee pins its book to it via ``MatchReferee.sync_health`` — a physical
dog with a real hit sensor is the ground truth for its own body, and KO
resolves on what the dog reports, not on referee bookkeeping.

This module is pure pydantic v2 + stdlib: no MQTT, no asyncio, no framework
deps.  Whatever transport carries the payloads simply serialises these
models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field

# Default hitpoint pool — mirrors MatchCombatant's default hp/max_hp in
# sim_engine.combat.match so a wire dog and a sim dog start the same fight.
DEFAULT_HP: float = 40.0

# Hp FRACTION at/below which the body limps (inclusive boundary: a dog at
# exactly this fraction is limping).
LIMP_THRESHOLD: float = 0.35

# Locomotion factor applied while limping (multiplies normal speed).
LIMP_MOBILITY: float = 0.45

# Every adjudication path a hit can arrive from.  The Literal on the models
# below mirrors this tuple — keep them in lockstep.
HIT_SOURCES: tuple[str, ...] = ("referee", "hit_sensor", "camera", "operator", "sim")

HitSource = Literal["referee", "hit_sensor", "camera", "operator", "sim"]


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string (default timestamp)."""
    return datetime.now(timezone.utc).isoformat()


def _new_hit_id() -> str:
    """Short unique hit id — 12 hex chars is plenty for one match's hits."""
    return uuid4().hex[:12]


class RegisterHitCommand(BaseModel):
    """Adjudicator -> robot: "you were hit — apply this damage".

    Wire form (as parsed by examples/robot-template/robot.py handle_command)::

        {"command": "register_hit", "hit_id": "a1b2c3d4e5f6",
         "shooter_id": "dog_a", "damage": 8.0, "location": "chassis",
         "source": "referee", "timestamp": "2026-07-11T00:00:00+00:00"}

    Published on the EXISTING command topic
    ``tritium/{site}/robots/{robot_id}/command`` — the robot keys on the
    ``command`` field just as it does for ``turret_aim`` / ``fire``, so a
    real dog needs no new subscription to start taking damage.  ``hit_id``
    is provenance: the dog echoes it back in its :class:`HitReport` so the
    adjudicator can match verdict to acknowledgement.  ``source`` records
    WHICH adjudication path called the hit (see :data:`HIT_SOURCES`) —
    a sim referee, a physical hit sensor, a camera scorer, or an operator
    all emit the same schema.
    """

    command: Literal["register_hit"] = "register_hit"
    hit_id: str = Field(default_factory=_new_hit_id)
    shooter_id: str | None = None
    damage: float = Field(default=10.0, ge=0.0)
    location: str | None = None  # e.g. "chassis", "front_left", "turret"
    source: HitSource = "referee"
    timestamp: str = Field(default_factory=_now_iso)


class HitReport(BaseModel):
    """Robot -> world: the TARGET dog's own account of damage that landed.

    Wire form::

        {"hit_id": "a1b2c3d4e5f6", "target_id": "dog_b",
         "shooter_id": "dog_a", "damage": 8.0, "hp_after": 32.0,
         "max_hp": 40.0, "alive": true, "location": "chassis",
         "source": "referee", "ts": "2026-07-11T00:00:00+00:00"}

    Published by the target dog itself on the NEW hit topic
    ``tritium/{site}/robots/{robot_id}/hit`` (QoS 1) whenever damage lands,
    whatever the source — a referee's :class:`RegisterHitCommand`, its own
    physical hit sensor, or a camera-detected impact.  ``hp_after`` /
    ``alive`` are the dog's AUTHORITATIVE post-hit state; scoreboards and
    KO calls resolve on this report, not on adjudicator bookkeeping.
    """

    hit_id: str
    target_id: str
    shooter_id: str | None = None
    damage: float = Field(ge=0.0)
    hp_after: float = Field(ge=0.0)
    max_hp: float = Field(ge=1.0)
    alive: bool
    location: str | None = None
    source: HitSource = "referee"
    ts: str = Field(default_factory=_now_iso)


class HealthStatus(BaseModel):
    """Health telemetry — the dog's standing account of its own body.

    Embedded in the existing robot telemetry payload under the ``"health"``
    key, exactly as :class:`~tritium_lib.models.fire_control.WeaponStatus`
    embeds under ``"weapon_status"``::

        {"device_id": "dog_b", "hp": 12.0, "max_hp": 40.0,
         "hits_taken": 4, "mobility": 0.45,
         "last_hit_ts": "2026-07-11T00:00:00+00:00",
         "ts": "2026-07-11T00:00:01+00:00"}

    ``mobility`` is the locomotion factor the body is actually applying
    (``1.0`` healthy, :data:`LIMP_MOBILITY` while limping, ``0.0`` dead) —
    reported, like everything here, by the dog itself.  ``alive`` is
    computed (``hp > 0``) so the wire payload always carries it without a
    publisher having to remember to set it.
    """

    device_id: str
    hp: float = Field(ge=0.0)
    max_hp: float = Field(ge=1.0)
    hits_taken: int = Field(default=0, ge=0)
    mobility: float = Field(default=1.0, ge=0.0, le=1.0)
    last_hit_ts: str | None = None
    ts: str = Field(default_factory=_now_iso)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def alive(self) -> bool:
        """A body fights while its hitpoints are above zero."""
        return self.hp > 0

    def to_health_state(self) -> dict:
        """Collapse this telemetry into the HUD health-snapshot shape.

        Returns exactly the five keys the combat HUD renders per unit —
        ``{"hp", "max_hp", "alive", "hits_taken", "mobility"}`` — mirroring
        the role ``WeaponStatus.to_ammo_state()`` plays for ammo: a real
        dog's health telemetry and a sim unit's health land in the SAME
        snapshot shape and render through the identical HUD path.
        Single-sourcing the shape here keeps the wire and sim surfaces from
        drifting apart.
        """
        return {
            "hp": self.hp,
            "max_hp": self.max_hp,
            "alive": self.alive,
            "hits_taken": self.hits_taken,
            "mobility": self.mobility,
        }


class HealthTracker:
    """Owns ONE body's health authority — the dog-side half of the contract.

    Plain stdlib class (no pydantic, no framework deps) so it runs on the
    smallest robot brain.  The adjudicator only *requests* damage via
    :class:`RegisterHitCommand`; this tracker is where the body actually
    keeps its own book — apply a hit, emit the :class:`HitReport`, and
    surface :class:`HealthStatus` for the telemetry loop.  A dead body
    keeps accepting hits (foam keeps flying after a KO): hp stays pinned
    at 0, ``hits_taken`` still counts, ``alive`` stays ``False``.
    """

    def __init__(self, device_id: str, max_hp: float = DEFAULT_HP) -> None:
        self.device_id = device_id
        self._max_hp = float(max_hp)
        self._hp = float(max_hp)
        self._hits_taken = 0
        self._last_hit_ts: str | None = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def hp(self) -> float:
        """Current hitpoints (never below 0)."""
        return self._hp

    @property
    def max_hp(self) -> float:
        """The full pool this body started with."""
        return self._max_hp

    @property
    def alive(self) -> bool:
        """A body fights while its hitpoints are above zero."""
        return self._hp > 0

    @property
    def hp_fraction(self) -> float:
        """Remaining fraction of the pool, ``0.0`` (dead) .. ``1.0`` (full)."""
        return self._hp / self._max_hp

    def mobility_factor(self) -> float:
        """Locomotion factor for the body's current condition.

        * ``1.0`` — healthy (fraction above :data:`LIMP_THRESHOLD`).
        * :data:`LIMP_MOBILITY` — limping (``0 < fraction <= LIMP_THRESHOLD``;
          the boundary itself limps).
        * ``0.0`` — dead.
        """
        if self._hp <= 0.0:
            return 0.0
        if self.hp_fraction <= LIMP_THRESHOLD:
            return LIMP_MOBILITY
        return 1.0

    # ------------------------------------------------------------------
    # Taking damage
    # ------------------------------------------------------------------

    def apply_hit(
        self,
        damage: float,
        shooter_id: str | None = None,
        location: str | None = None,
        source: str = "referee",
        hit_id: str | None = None,
    ) -> HitReport:
        """Apply one hit to this body and return the wire-ready report.

        ``damage`` is clamped at ``>= 0`` (a hit never heals) and hp is
        clamped at ``0``.  ``hits_taken`` increments on EVERY hit — a foam
        dart on a dead chassis is still an impact — and ``last_hit_ts`` is
        stamped.  ``hit_id`` supplied (echoing a
        :class:`RegisterHitCommand`) is preserved so the adjudicator can
        match verdict to acknowledgement; ``None`` generates a fresh id
        (onboard sensor hits have no upstream command to echo).
        """
        damage = max(0.0, float(damage))
        self._hp = max(0.0, self._hp - damage)
        self._hits_taken += 1
        self._last_hit_ts = _now_iso()
        return HitReport(
            hit_id=hit_id if hit_id is not None else _new_hit_id(),
            target_id=self.device_id,
            shooter_id=shooter_id,
            damage=damage,
            hp_after=self._hp,
            max_hp=self._max_hp,
            alive=self.alive,
            location=location,
            source=source,  # type: ignore[arg-type]
            ts=self._last_hit_ts,
        )

    def status(self) -> HealthStatus:
        """Snapshot this body as the telemetry ``"health"`` block."""
        return HealthStatus(
            device_id=self.device_id,
            hp=self._hp,
            max_hp=self._max_hp,
            hits_taken=self._hits_taken,
            mobility=self.mobility_factor(),
            last_hit_ts=self._last_hit_ts,
        )

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""PoliceTacticsController -- squad-level stand-in AI for riot police.

This is a **video-game-style FSM stand-in**, NOT Graphling cognition.  It
drives friendly ``police`` units in the ``civil_unrest`` game mode with the
crowd-control doctrine the mode is built around: form a line, push the crowd
back (dispersal), make non-lethal arrests of worn-down ringleaders, rout the
weak, and feel the grievance flare back up when the operator fires
indiscriminately.

Squad FSM
---------
The controller runs one squad through a four-state machine each tick:

    hold  -> form  -> advance  -> engage

  * **hold**   -- no violent targets on the field; officers stand easy.
  * **form**   -- a LINE (or WEDGE) formation is established between the
                  squad centroid and the violent centroid, facing the crowd.
  * **advance**-- the line anchor steps toward the violent centroid at
                  ~1 m/s: the dispersal push.
  * **engage** -- the violent frontline is within ``engage_range`` of the
                  line; officers hold contact, make arrests, and rout.

Formation slots are re-planned every ``replan_interval`` sim-seconds via the
pure ``get_formation_positions`` helper and assigned to officers by a stable
sort of ``target_id`` (deterministic under the golden-replay global seed).
A tight cluster (>= ``_WEDGE_CLUSTER_MIN`` violent within
``_WEDGE_CLUSTER_RADIUS`` of the violent centroid) switches LINE -> WEDGE so
the squad splits a massed crowd instead of stalling against it.

Operator command path
---------------------
``command_tactic(tactic, corridor=None, faction=None)`` is the same interface a
real squad lead drives the stand-in with: ``"auto"`` (the FSM above), ``"line"``
/ ``"wedge"`` (force that formation), or ``"kettle"``.  Under **kettle** the FSM
is replaced by a fifth state ``kettle``: officers form an ARC cordon around
the local violent cluster with a single open corridor (facing an
operator-supplied point, or auto-set to the far side away from the line), the
ring tightens each tick, and rioters still inside the ring are shoved out
through the gap.

**Faction-aware kettling** (the three-way headline): pass ``faction`` with a
kettle command ("kettle the RED bloc") and the cordon is built ONLY around
that faction's nearest violent cluster — the centre, ring, corridor drive, and
arrests all scope to targeted-faction members.  The untargeted bloc is never
cordoned (it is left to the autonomous line, dispersal, or a second command).
When the targeted faction is fully contained/arrested/gone the cordon disbands
and the squad stands easy even while a rival bloc is still violent elsewhere.
``faction=None`` keeps the legacy behaviour (kettle whatever violent cluster is
nearest, regardless of bloc).

Autonomous faction-kettle doctrine
----------------------------------
For a **rival-faction** riot (2+ declared blocs) the stand-in commander does
not wait for an operator: ``enable_auto_kettle(blocs)`` arms a doctrine that
reads the LIVE per-bloc violent strength each tick and, when one bloc is
decisively overwhelming its rival (a strength ``ratio`` AND absolute
``margin`` lead, with the dominant bloc massed past ``min_strength``),
AUTONOMOUSLY issues ``command_tactic("kettle", faction=<dominant>)`` on its own
— the production-half essence of a squad AI making the tactical decision a real
commander would.  Hysteresis (a lower ``release_ratio``) plus a ``min_hold``
floor stop it thrashing; if the balance FLIPS the doctrine switches the cordon
to the newly dominant bloc; when dominance decays it releases back to the FSM.
It is deterministic (integer count thresholds, no RNG) and seedable.

An OPERATOR command hard-overrides the autonomous choice: any specific
``command_tactic`` from the operator (source ``"operator"``) locks the doctrine
out until the operator returns to ``"auto"`` (or ``reset()``).  With fewer than
two declared blocs the doctrine stays disabled, so a single-faction riot is
byte-identical (the doctrine is never armed).

``get_status()`` exposes the live squad state / formation / commanded tactic /
agitation / corridor / target_faction / arrests for the operator UI.
Switching back to ``auto``/``line``/``wedge`` cleanly resumes the FSM at
``form`` and clears any faction target.  ``reset()`` restores ``auto``.

Arrests / routs
---------------
  * **arrest** -- a violent target worn down to ``arrest_health`` with >= 2
    officers inside ``arrest_range`` is non-lethally detained: converted to a
    neutral ``calmed`` non-combatant, weapon zeroed, +25 de-escalation.
  * **rout**   -- a violent target below ``rout_health`` that could NOT be
    arrested this tick breaks and flees ~30 m away from the squad; +10
    de-escalation.

Grievance feedback (Epstein proxy)
----------------------------------
``agitation`` (0..1, starts 0.35) is the crowd's grievance level:

  * every police shot observed raises it (indiscriminate force inflames),
  * every arrest / rout lowers it (decisive de-escalation cools the crowd),
  * while **engaged**, un-identified civilians near a violent-vs-police melee
    contact radicalize into rioters with per-second probability
    ``0.02 * agitation`` -- the grievance flare-up arc.

Graphling boundary
------------------
Officers whose embodiment slot is OCCUPIED (``occupancy_check(id)`` True) are
excluded from the roster entirely: no slot assignment, no controller-driven
arrests -- the occupant (a Graphling) decides its own actions and Tritium
never puppets it.

Crowd beats
-----------
Beat transitions are announced on the ``crowd_event`` bus topic (the same
topic the announcer already drains) -- one publish per transition, never
per-tick spam:

  * ``police_line``   once when the formation is first established,
  * ``police_push``   once when the advance begins,
  * ``arrest_surge``  on every 3rd arrest,
  * ``crowd_broken``  once when the violent count first drops below 25 % of
                      its observed peak while engaged,
  * ``kettle_formed`` once (per kettle command) when >= 75 % of the ARC cordon
                      slots are manned by DISTINCT officers (within the
                      arc-segment tolerance) — the ring has CLOSED,
  * ``corridor_flow`` once (per kettle command) when >= 3 distinct rioters have
                      been driven out through the dispersal gap.

All duck-typed dependencies (``event_bus.publish(topic, dict)``,
``game_mode`` with ``de_escalation_score`` / ``arrest_count`` / ``rout_count``)
match the InstigatorDetector contract in ``game_mode.py``.
"""

from __future__ import annotations

import math
import random
from typing import Any

from tritium_lib.models.target_status import is_terminal
from tritium_lib.sim_engine.ai.formations import (
    FormationConfig,
    FormationType,
    get_formation_positions,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Officer statuses that count as "alive and drivable".  Mirrors the friendly
# dispatch set in behaviors.tick (a police officer that reaches its slot lands
# in "arrived" and must stay in the roster).
_ALIVE_STATUSES: frozenset[str] = frozenset(
    {"active", "idle", "stationary", "arrived"}
)

# Grievance level a fresh riot starts at (0 = calm, 1 = boiling).
_INITIAL_AGITATION: float = 0.35

# Dispersal push speed of the line anchor during "advance" (m/s).
# 1.0 left the line arriving after the ISR detain loop had already resolved
# the riot (zero contact); 1.6 is a brisk walking push that reaches the
# crowd while it is still fighting (lane/riot rebalance 2026-07-10).
_ADVANCE_SPEED: float = 1.6

# A tight crowd cluster (this many violent within the radius of the violent
# centroid) switches the formation from LINE to WEDGE to split the mass.
# Lowered 6 -> 4 (lane/riot tick 2): the push objective is the LOCAL nearest
# cluster, whose membership rarely reached 6 in a street-distributed riot, so
# WEDGE never actually triggered in play.  4 is a genuine "massed knot worth
# splitting" without demanding the whole crowd stack on one point.
_WEDGE_CLUSTER_MIN: int = 4
_WEDGE_CLUSTER_RADIUS: float = 12.0

# A violent target within this range of any officer is "in melee contact".
_MELEE_CONTACT_RANGE: float = 5.0

# The advancing line stops this far from the nearest violent target.  Must be
# INSIDE both the pepper_ball range (8 m) and the arrest range (4 m) or the
# push stalls in a permanent stand-off: holding at engage_range (12 m) left
# officers 1 m out of arrest reach with zero shots fired either way
# (2026-07-10 probe: engage state reached, 0 arrests, 0 eliminations).
_CONTACT_HOLD: float = 3.0

# Un-identified civilians within this range of a melee contact can radicalize.
_RADICALIZE_RANGE: float = 10.0
# Per-second radicalization probability base (scaled by agitation, sampled dt).
_RADICALIZE_RATE: float = 0.02

# Agitation deltas.
_SHOT_AGITATION_RISE: float = 0.01
_ARREST_AGITATION_DROP: float = 0.05
_ROUT_AGITATION_DROP: float = 0.03

# De-escalation score awarded per non-lethal arrest / per rout.
_ARREST_DEESCALATION: int = 25
_ROUT_DEESCALATION: int = 10

# Distance a routed rioter flees away from the squad centroid (m).
_ROUT_FLEE_DIST: float = 30.0

# A rout is only credited to the squad when an officer is actually near the
# broken rioter.  Without this, rioters worn down by robot fire on the far
# side of the map "routed" and paid de-escalation score to a police line
# that never touched them (credit-washing; caught in the 2026-07-10 probe:
# 3 routs, 0 arrests, squad never left "advance").  6 m = at-the-line only:
# at 15 m targets routed while the line was still closing, starving the
# arrest mechanic (arrest window passed before officers reached 4 m).
_ROUT_CREDIT_RANGE: float = 6.0

# arrest_surge beat fires on every Nth cumulative arrest.
_ARREST_SURGE_INTERVAL: int = 3

# crowd_broken fires when violent count drops below this fraction of its peak,
# and only once a peak of at least this many was observed (so a stray pair of
# rioters never trips the "crowd broken" fanfare).
_CROWD_BROKEN_FRACTION: float = 0.25
_CROWD_BROKEN_MIN_PEAK: int = 4

# Operator-commandable tactics (the production command path a squad lead uses).
# "auto"   -- the automatic hold->form->advance->engage FSM (default).
# "line"   -- force a LINE formation regardless of cluster size.
# "wedge"  -- force a WEDGE formation regardless of cluster size.
# "kettle" -- surround the local violent cluster in an ARC cordon with one
#             open dispersal corridor and push rioters out through the gap.
_VALID_TACTICS: frozenset[str] = frozenset({"auto", "line", "wedge", "kettle"})

# Kettle cordon geometry.  The ring starts wide enough to enclose the observed
# cluster spread, then tightens each tick to squeeze the crowd toward the gap.
_KETTLE_START_RADIUS: float = 10.0     # minimum initial ring radius (m)
_KETTLE_MIN_RADIUS: float = 7.0        # floor the ring never shrinks past (m)
_KETTLE_SHRINK_RATE: float = 0.3       # ring tighten speed (m/s)
_KETTLE_SPREAD_MARGIN: float = 4.0     # extra radius over the cluster spread (m)
# A kettle is GEOGRAPHIC: once engaged, the cordon center follows the crowd
# still inside the ring at most this fast (m/s) — slower than an officer's
# walk, so the ring never outruns its own cordon.  It NEVER re-targets a
# different cluster mid-kettle (the seed-14 probe showed the per-tick
# nearest-cluster objective teleporting ~70 m to a fresh wave spawn, dragging
# the half-closed ring off the officers — manned 4/8 -> 0/8 in five seconds).
_KETTLE_CENTER_DRIFT: float = 1.0
# Rioters this far beyond the current ring radius no longer drag the center
# (they escaped or were shoved out the gap); when NOBODY violent remains
# inside, the cordon's work here is done and the squad disbands it.
_KETTLE_CONTAIN_MARGIN: float = 6.0
# REACH gate for the autonomous doctrine: only CALL a kettle when the target
# bloc's nearest violent cluster is within this range of the squad — a cordon
# the officers can actually close before the crowd walks out of it.  Out of
# reach, the FSM line keeps advancing (which is what closes the distance) and
# the doctrine re-evaluates every tick.  (Seed-14 probe: kettles called on
# clusters 70-180 m out spent their whole hold window marching, ring never
# manned, kettle_formed 0/10 seeds.)
_AUTO_KETTLE_ENGAGE_RANGE: float = 35.0
_KETTLE_GAP_ANGLE: float = 75.0        # open corridor width (degrees)
# Officer "on its slot" tolerance (m).  5.0 ~= the inter-slot spacing of the
# ARC ring (2*pi*8m * 285/360 / 8 slots), i.e. an officer within tolerance is
# holding the arc SEGMENT its slot anchors.  Engine officers halt in firing
# posture at pepper-ball range (8 m >= the 7-10 m ring), typically 3.5-5 m off
# the slot point (seed-14 probe: median officer->slot stalled at 4.4-4.6 m
# under the old 3.0 m tolerance while the cordon was visibly closed).
_KETTLE_ARRIVE_DIST: float = 5.0
_KETTLE_FORMED_FRACTION: float = 0.75  # roster fraction on-slot -> kettle_formed

# Corridor drive: a still-violent target inside the ring is shoved out the gap
# to a point this far beyond the ring, at most once per interval per target.
_CORRIDOR_EXIT_DIST: float = 25.0      # push waypoint distance past the ring (m)
_CORRIDOR_PUSH_INTERVAL: float = 5.0   # min seconds between pushes of one target
_CORRIDOR_FLOW_MIN: int = 3            # distinct pushes -> corridor_flow beat

# --- Autonomous faction-kettle doctrine (rival-faction riots only) ----------
# The stand-in commander reads the live per-bloc violent strength and cordons a
# bloc that is decisively overwhelming its rival WITHOUT an operator command.
# Deterministic integer-count thresholds (no RNG); hysteresis + a minimum hold
# stop it thrashing.  Only armed when 2+ blocs are declared (enable_auto_kettle)
# so single-faction riots are byte-identical.  An operator command overrides.
_AUTO_KETTLE_MIN_STRENGTH: int = 5     # dominant bloc must have >= this violent
_AUTO_KETTLE_RATIO: float = 1.5        # dominant/rival strength ratio to ARM
_AUTO_KETTLE_MARGIN: int = 3           # AND absolute strength lead to ARM
_AUTO_KETTLE_RELEASE_RATIO: float = 1.15  # hold while >= this, else release
_AUTO_KETTLE_MIN_HOLD: float = 12.0    # min sim-seconds a kettle holds (anti-thrash)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_targets(targets: Any) -> list:
    """Return a flat list of target objects from a dict or an iterable."""
    if isinstance(targets, dict):
        return list(targets.values())
    return list(targets)


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Mean (x, y) of a list of points; (0, 0) when empty."""
    n = len(points)
    if n == 0:
        return (0.0, 0.0)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    return (sx / n, sy / n)


def _is_violent(t: Any) -> bool:
    """True for an active threat the police must engage.

    A violent target is an alive ``rioter``, or an alive, un-identified
    ``instigator`` whose activation cycle is currently ``active`` (throwing
    objects).  Hidden/activating instigators, calmed/identified units, and
    dead crowd members are not violent.
    """
    if is_terminal(getattr(t, "status", "")):
        return False
    role = getattr(t, "crowd_role", None)
    if role == "rioter":
        return True
    if (
        role == "instigator"
        and getattr(t, "instigator_state", None) == "active"
        and not getattr(t, "identified", False)
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# PoliceTacticsController
# ---------------------------------------------------------------------------


class PoliceTacticsController:
    """Squad-level stand-in AI for friendly ``police`` units (civil_unrest)."""

    def __init__(
        self,
        event_bus: Any,
        game_mode: Any = None,
        *,
        formation_spacing: float = 2.5,
        engage_range: float = 12.0,
        # 7 m, not literal grab distance: the engine's crowd-separation floor
        # keeps entities ~5 m apart, so a 4 m arrest reach was PHYSICALLY
        # unreachable — chargers crossed the whole arrest window at 5-7 m and
        # always routed instead (2026-07-10 trace).  Two officers inside 7 m
        # is "step out of the line and grab".
        arrest_range: float = 7.0,
        arrest_health: float = 40.0,
        rout_health: float = 15.0,
        replan_interval: float = 1.0,
        occupancy_check: Any = None,
        slot_validator: Any = None,
    ) -> None:
        self._event_bus = event_bus
        self._game_mode = game_mode
        self._formation_spacing = formation_spacing
        self._engage_range = engage_range
        self._arrest_range = arrest_range
        self._arrest_health = arrest_health
        self._rout_health = rout_health
        self._replan_interval = replan_interval
        self._occupancy_check = occupancy_check
        # Optional terrain-aware slot validator (the SAME injected-callable
        # pattern as occupancy_check).  ``slot_validator(from_pt, slot) -> pt``
        # returns a non-lethal, reachable waypoint for a commanded slot, nudging
        # one that lands in a wall / lethal cell to the nearest clear cell so a
        # formation / kettle-ARC cordon actually CLOSES on real AO terrain
        # (buildings / flood) instead of officers beelining into an obstacle.
        # ``None`` (the default, and every current sim-battle map — no costmap)
        # is an identity pass-through, so goldens stay byte-identical.  The
        # engine builds one over ``tritium_lib.planning.segment_clear`` + the
        # live costmap when a real AO is loaded.
        self._slot_validator = slot_validator

        # Squad FSM state.
        self._squad_state: str = "hold"
        self._anchor: tuple[float, float] | None = None
        self._slots: list[tuple[float, float]] | None = None
        self._formation_type: FormationType | None = None
        self._replan_accum: float = 0.0

        # Absolute sim clock (accumulated dt) for per-target corridor throttling.
        self._sim_clock: float = 0.0

        # Operator command override (the production squad-lead command path).
        self._commanded_tactic: str = "auto"
        self._corridor: tuple[float, float] | None = None
        # Faction-aware kettle target: when set, a kettle command cordons ONLY
        # this bloc's nearest violent cluster (the untargeted bloc is left be).
        # None => legacy kettle (nearest violent cluster of any faction).
        self._kettle_faction: str | None = None
        # Source of the current commanded tactic: "auto" (FSM, AI free to act),
        # "autonomous" (the doctrine auto-issued this kettle), or "operator" (a
        # human command; the autonomous doctrine must NOT override it).
        self._tactic_source: str = "auto"

        # Autonomous faction-kettle doctrine (rival-faction riots only).
        # Disabled by default => single-faction riots are byte-identical; the
        # engine arms it via enable_auto_kettle() only when 2+ blocs are wired.
        self._auto_kettle_enabled: bool = False
        self._auto_kettle_blocs: frozenset[str] = frozenset()
        self._auto_kettle_ratio: float = _AUTO_KETTLE_RATIO
        self._auto_kettle_margin: int = _AUTO_KETTLE_MARGIN
        self._auto_kettle_min_strength: int = _AUTO_KETTLE_MIN_STRENGTH
        self._auto_kettle_release_ratio: float = _AUTO_KETTLE_RELEASE_RATIO
        self._auto_kettle_min_hold: float = _AUTO_KETTLE_MIN_HOLD
        # The bloc the doctrine is currently kettling (None => not auto-kettling).
        self._auto_kettle_target: str | None = None
        self._auto_kettle_since: float = -1e9

        # Kettle-cordon transient state (cleared on exit / reset).
        self._kettle_gap_dir: tuple[float, float] | None = None
        self._kettle_center: tuple[float, float] | None = None
        self._kettle_radius: float = _KETTLE_START_RADIUS
        self._kettle_formed_announced: bool = False
        self._corridor_flow_announced: bool = False
        # target_id -> sim-clock time of its last corridor push (throttle).
        self._corridor_pushed: dict[str, float] = {}

        # Grievance level.
        self._agitation: float = _INITIAL_AGITATION

        # Shot observation: officer_id -> last_fired value seen last tick.
        self._last_fired_seen: dict[str, float] = {}

        # Beat-transition guards (reset() clears them).
        self._line_announced: bool = False
        self._push_announced: bool = False
        self._crowd_broken_announced: bool = False

        # Arrest / crowd-break bookkeeping.
        self._arrest_total: int = 0
        self._peak_violent: int = 0

    # -- Public read-only state -------------------------------------------------

    @property
    def squad_state(self) -> str:
        """Current squad FSM state: hold / form / advance / engage."""
        return self._squad_state

    @property
    def agitation(self) -> float:
        """Crowd grievance level, 0.0 (calm) .. 1.0 (boiling)."""
        return self._agitation

    @property
    def formation_type(self) -> FormationType | None:
        """The FormationType last planned (None until a line/wedge forms)."""
        return self._formation_type

    @property
    def commanded_tactic(self) -> str:
        """Operator-commanded tactic: auto / line / wedge / kettle."""
        return self._commanded_tactic

    @property
    def corridor(self) -> tuple[float, float] | None:
        """Operator-supplied dispersal-corridor point, or None (auto gap)."""
        return self._corridor

    @property
    def target_faction(self) -> str | None:
        """The bloc a kettle is targeting, or None (legacy nearest-cluster)."""
        return self._kettle_faction

    @property
    def tactic_source(self) -> str:
        """Who set the current tactic: auto / autonomous / operator."""
        return self._tactic_source

    @property
    def auto_kettle_target(self) -> str | None:
        """Bloc the autonomous doctrine is kettling, or None (not auto-kettling)."""
        return self._auto_kettle_target

    # -- Operator command path --------------------------------------------------

    def command_tactic(
        self,
        tactic: str,
        corridor: tuple[float, float] | None = None,
        faction: str | None = None,
        *,
        _source: str = "operator",
    ) -> bool:
        """Command a squad tactic (the production squad-lead interface).

        ``tactic`` must be one of ``_VALID_TACTICS`` ("auto", "line", "wedge",
        "kettle"); an unknown tactic is rejected with no state change and
        returns False.  A valid command stores the override, publishes a single
        ``police_tactic_commanded`` event, and returns True.  ``corridor`` (a
        world point) only matters for ``kettle`` — it aims the dispersal gap
        from the kettle centre toward that point; omit it to auto-aim the gap
        away from the squad.  ``faction`` also only matters for ``kettle`` —
        name a bloc ("kettle the RED bloc") to cordon ONLY that faction's
        nearest violent cluster; omit it (or command any non-kettle tactic) to
        kettle the nearest violent cluster of any faction.

        ``_source`` is the internal command origin (default ``"operator"``): the
        autonomous doctrine passes ``"autonomous"``.  A ``"operator"`` command
        for any specific tactic hard-locks out the doctrine; commanding
        ``"auto"`` always returns control to the automatic AI (the doctrine may
        resume) regardless of source.
        """
        if tactic not in _VALID_TACTICS:
            return False

        prev = self._commanded_tactic
        prev_faction = self._kettle_faction
        self._commanded_tactic = tactic
        self._corridor = (
            (float(corridor[0]), float(corridor[1]))
            if corridor is not None else None
        )
        # A faction target is only meaningful for a kettle; any other tactic
        # clears it so the FSM / forced formation never scopes to one bloc.
        self._kettle_faction = (
            str(faction) if (tactic == "kettle" and faction) else None
        )
        # Commanding "auto" (from anyone) returns control to the automatic AI;
        # any other specific command records its source so an OPERATOR override
        # locks the autonomous doctrine out until it returns to auto / reset().
        self._tactic_source = "auto" if tactic == "auto" else _source
        # Leaving kettle, re-entering it fresh, or SWITCHING the kettled bloc
        # clears the cordon transient state so the ring re-forms on the new
        # target (or the FSM resumes cleanly from "form").
        if (
            tactic != "kettle"
            or prev != "kettle"
            or self._kettle_faction != prev_faction
        ):
            self._exit_kettle()

        self._publish("police_tactic_commanded", {
            "tactic": tactic,
            "corridor": (
                {"x": self._corridor[0], "y": self._corridor[1]}
                if self._corridor is not None else None
            ),
            "faction": self._kettle_faction,
        })
        return True

    def enable_auto_kettle(
        self,
        blocs: Any,
        *,
        ratio: float | None = None,
        margin: int | None = None,
        min_strength: int | None = None,
        release_ratio: float | None = None,
        min_hold: float | None = None,
    ) -> None:
        """Arm the autonomous faction-kettle doctrine for a rival-faction riot.

        ``blocs`` is the set of declared bloc faction ids.  With fewer than two
        distinct blocs the doctrine stays DISABLED (a single-faction riot is
        byte-identical — the engine only calls this when it wires 2+ factions).
        Thresholds default to the module constants; override for tuning/tests.
        Deterministic (integer count thresholds, no RNG); operator commands
        still override the doctrine's choice.
        """
        blocs = frozenset(str(b) for b in (blocs or []) if b)
        if len(blocs) < 2:
            self._auto_kettle_enabled = False
            self._auto_kettle_blocs = frozenset()
            return
        self._auto_kettle_enabled = True
        self._auto_kettle_blocs = blocs
        if ratio is not None:
            self._auto_kettle_ratio = float(ratio)
        if margin is not None:
            self._auto_kettle_margin = int(margin)
        if min_strength is not None:
            self._auto_kettle_min_strength = int(min_strength)
        if release_ratio is not None:
            self._auto_kettle_release_ratio = float(release_ratio)
        if min_hold is not None:
            self._auto_kettle_min_hold = float(min_hold)

    def set_slot_validator(self, validator: Any) -> None:
        """Inject (or clear) the terrain-aware slot validator.

        ``validator(from_pt, slot) -> (x, y)`` returns a reachable, non-lethal
        waypoint for a commanded formation / kettle-cordon slot, nudging one that
        lands in a wall / lethal cell to the nearest clear cell so the cordon
        closes on real terrain.  ``None`` restores the identity pass-through (the
        default, and every costmap-free sim map — so goldens stay byte-identical).
        The engine wires one built over ``tritium_lib.planning.validate_slot`` +
        the live costmap only when a real AO / obstacle terrain is loaded.
        """
        self._slot_validator = validator

    def get_status(self) -> dict:
        """Operator-facing squad status snapshot (stable API contract).

        Exactly these keys: ``squad_state``, ``formation_type`` (string value
        or None), ``commanded_tactic``, ``agitation``, ``corridor``
        ({"x","y"} or None), ``target_faction`` (bloc id or None), ``arrests``,
        ``tactic_source`` (auto / autonomous / operator — who set the tactic, so
        the operator can SEE when the stand-in commander is acting on its own),
        ``kettle_formed`` (bool — the cordon ring has CLOSED: >=75% of the arc
        slots manned; False whenever not kettling.  Lets the operator — and the
        headed proof — see closure without scraping the event bus).
        """
        return {
            "squad_state": self._squad_state,
            "kettle_formed": self._kettle_formed_announced,
            "formation_type": (
                self._formation_type.value
                if self._formation_type is not None else None
            ),
            "commanded_tactic": self._commanded_tactic,
            "agitation": self._agitation,
            "corridor": (
                {"x": self._corridor[0], "y": self._corridor[1]}
                if self._corridor is not None else None
            ),
            "target_faction": self._kettle_faction,
            "arrests": self._arrest_total,
            "tactic_source": self._tactic_source,
        }

    # -- Occupancy (Graphling boundary) -----------------------------------------

    def _is_occupied(self, target_id: str) -> bool:
        if self._occupancy_check is None:
            return False
        try:
            return bool(self._occupancy_check(target_id))
        except Exception:
            return False

    # -- Terrain-aware slot validation (optional injected seam) -----------------

    def _validate_slot(
        self,
        from_pt: tuple[float, float],
        slot: tuple[float, float],
    ) -> tuple[float, float]:
        """Return a reachable, non-lethal waypoint for ``slot``.

        Identity pass-through when no ``slot_validator`` was injected (the
        default, and every sim-battle map — so goldens are byte-identical).  A
        wired validator (built over the live costmap + ``segment_clear``) nudges
        a slot that lands in a wall / lethal cell to the nearest clear cell so
        the cordon actually closes on real terrain.  Any validator error falls
        back to the raw slot (never breaks the tick).
        """
        if self._slot_validator is None:
            return slot
        try:
            v = self._slot_validator(from_pt, slot)
            if v is not None:
                return (float(v[0]), float(v[1]))
        except Exception:
            pass
        return slot

    # -- Tick -------------------------------------------------------------------

    def tick(self, dt: float, targets: Any, game_mode_type: str) -> None:
        """Advance the squad AI one step. No-op outside ``civil_unrest``."""
        if game_mode_type != "civil_unrest":
            return

        self._sim_clock += dt

        all_targets = _iter_targets(targets)

        roster = [
            t
            for t in all_targets
            if getattr(t, "alliance", None) == "friendly"
            and getattr(t, "asset_type", None) == "police"
            and getattr(t, "status", None) in _ALIVE_STATUSES
            and not self._is_occupied(getattr(t, "target_id", ""))
        ]

        # Observe police shots -> agitation (grievance rises with force).
        self._observe_shots(roster)

        violent = [t for t in all_targets if _is_violent(t)]

        # No squad or nobody to disperse: stand easy, drop the formation.
        if not roster or not violent:
            self._squad_state = "hold"
            self._anchor = None
            self._slots = None
            self._formation_type = None
            # A cordon with nothing left to contain is disbanded; the cordon
            # transient state (gap dir, ring, formed/flow beats, push timers)
            # resets so a fresh kettle command starts clean.
            self._exit_kettle()
            return

        squad_centroid = _centroid([tuple(o.position[:2]) for o in roster])

        # -- Squad FSM transition ----------------------------------------------
        prev_state = self._squad_state
        if self._anchor is None:
            self._anchor = squad_centroid

        # Push objective: the NEAREST violent cluster, not the global violent
        # centroid.  A street-distributed riot has rioters scattered across
        # the district; their centroid is an empty point in the middle — a
        # line marching there never makes contact (2026-07-10 geometry probe:
        # min officer-to-rioter distance oscillated 27..77 m for a full run).
        # Instead: walk to the violent target nearest the line, treat every
        # violent within _WEDGE_CLUSTER_RADIUS of it as the local cluster,
        # and push at the cluster's centroid.  Clear it, then the next.
        frontline, cluster, objective = self._nearest_cluster(self._anchor, violent)

        # -- Autonomous faction-kettle doctrine --------------------------------
        # Before acting on the commanded tactic, let the stand-in commander make
        # its OWN call: for a rival-faction riot it reads the live per-bloc
        # strength and may auto-issue a kettle of a dominant bloc (or release
        # one), updating _commanded_tactic below.  No-op for single-faction
        # riots and while an operator holds manual control.
        self._auto_kettle_decision(violent, squad_centroid)

        # -- Operator override: kettle cordon ----------------------------------
        # A commanded kettle replaces the advance/engage flow entirely: cordon
        # the local cluster in an ARC, tighten the ring, and push rioters out
        # the gap.  Arrests / routs still run inside the cordon.
        if self._commanded_tactic == "kettle":
            # Faction-aware kettle: with a bloc targeted, cordon ONLY that
            # bloc's nearest violent cluster; the untargeted bloc is left be.
            if self._kettle_faction is not None:
                kettle_violent = [
                    v for v in violent
                    if getattr(v, "faction", None) == self._kettle_faction
                ]
                if not kettle_violent:
                    # Targeted bloc fully contained / absent: disband the
                    # cordon and stand easy even if a rival bloc still riots
                    # elsewhere (that is a separate command's problem).
                    self._squad_state = "hold"
                    self._anchor = None
                    self._slots = None
                    self._formation_type = None
                    self._exit_kettle()
                    return
                _fl, k_cluster, k_objective = self._nearest_cluster(
                    self._anchor, kettle_violent
                )
                self._tick_kettle(
                    dt, roster, kettle_violent, all_targets, squad_centroid,
                    k_cluster, k_objective,
                )
                return
            self._tick_kettle(
                dt, roster, violent, all_targets, squad_centroid, cluster, objective,
            )
            return

        if prev_state == "hold":
            new_state = "form"
            self._anchor = squad_centroid  # establish the line at the squad
        elif prev_state == "form":
            new_state = "advance"
        elif prev_state in ("advance", "engage"):
            new_state = "engage" if frontline <= self._engage_range else "advance"
        else:
            new_state = "form"
        self._squad_state = new_state

        # Dispersal push: step the anchor toward the local cluster.  The
        # line keeps stepping THROUGH the engage transition until it is at
        # true contact distance (_CONTACT_HOLD), so officers actually close
        # to pepper-ball / arrest range instead of standing off at
        # engage_range.
        if new_state in ("advance", "engage") and frontline > _CONTACT_HOLD:
            dx = objective[0] - self._anchor[0]
            dy = objective[1] - self._anchor[1]
            d = math.hypot(dx, dy)
            if d > 1e-6:
                step = min(_ADVANCE_SPEED * dt, d)
                self._anchor = (
                    self._anchor[0] + dx / d * step,
                    self._anchor[1] + dy / d * step,
                )

        # -- Formation plan + waypoint command ---------------------------------
        self._replan_accum += dt
        if self._slots is None or self._replan_accum >= self._replan_interval:
            self._replan_accum = 0.0
            self._plan_formation(roster, violent, objective)

        # -- Beat transitions ---------------------------------------------------
        if new_state == "form" and not self._line_announced:
            self._line_announced = True
            self._publish("crowd_event", {"beat": "police_line", "officers": len(roster)})
        if new_state == "advance" and not self._push_announced:
            self._push_announced = True
            self._publish("crowd_event", {"beat": "police_push", "officers": len(roster)})

        # -- Arrests + routs ----------------------------------------------------
        self._process_arrests_routs(violent, roster, squad_centroid)

        # -- Grievance flare-up: radicalize bystanders while engaged -----------
        if new_state == "engage":
            self._radicalize_bystanders(all_targets, violent, roster, dt)

        # -- crowd_broken beat --------------------------------------------------
        if new_state == "engage":
            cur_violent = sum(1 for t in all_targets if _is_violent(t))
            if cur_violent > self._peak_violent:
                self._peak_violent = cur_violent
            if (
                not self._crowd_broken_announced
                and self._peak_violent >= _CROWD_BROKEN_MIN_PEAK
                and cur_violent < self._peak_violent * _CROWD_BROKEN_FRACTION
            ):
                self._crowd_broken_announced = True
                self._publish("crowd_event", {"beat": "crowd_broken", "rioters": cur_violent})

    # -- Cluster targeting ------------------------------------------------------

    def _nearest_cluster(
        self,
        anchor: tuple[float, float],
        violent: list,
    ) -> tuple[float, list, tuple[float, float]]:
        """Local violent cluster nearest ``anchor``.

        Returns ``(frontline, cluster, objective)``: the distance to the
        nearest violent target, every violent target within
        ``_WEDGE_CLUSTER_RADIUS`` of it (the local knot), and that knot's
        centroid.  ``violent`` must be non-empty (callers guarantee it).  The
        faction-aware kettle passes a bloc-scoped violent list so the cordon
        centres on one faction; the autonomous FSM passes the full list.
        """
        nearest_v = min(
            violent,
            key=lambda v: math.hypot(v.position[0] - anchor[0],
                                     v.position[1] - anchor[1]),
        )
        frontline = math.hypot(nearest_v.position[0] - anchor[0],
                               nearest_v.position[1] - anchor[1])
        cluster = [
            v for v in violent
            if math.hypot(v.position[0] - nearest_v.position[0],
                          v.position[1] - nearest_v.position[1])
            <= _WEDGE_CLUSTER_RADIUS
        ]
        objective = _centroid([tuple(v.position[:2]) for v in cluster])
        return frontline, cluster, objective

    # -- Autonomous faction-kettle doctrine ------------------------------------

    def _bloc_violent_counts(self, violent: list) -> dict[str, int]:
        """Per-declared-bloc violent count from ``violent`` (deterministic)."""
        counts: dict[str, int] = {b: 0 for b in self._auto_kettle_blocs}
        for v in violent:
            fac = getattr(v, "faction", None)
            if fac in counts:
                counts[fac] += 1
        return counts

    def _is_dominant(self, a_n: int, b_n: int) -> bool:
        """True when strength ``a_n`` decisively overwhelms rival ``b_n``."""
        return (
            a_n >= self._auto_kettle_min_strength
            and a_n >= b_n * self._auto_kettle_ratio
            and a_n - b_n >= self._auto_kettle_margin
        )

    def _bloc_in_reach(
        self,
        bloc: str,
        violent: list,
        squad_centroid: tuple[float, float],
    ) -> bool:
        """True when ``bloc``'s nearest violent cluster is close enough to
        cordon (within ``_AUTO_KETTLE_ENGAGE_RANGE`` of the squad) — the
        commander only calls a kettle its officers can actually CLOSE."""
        members = [
            v for v in violent if getattr(v, "faction", None) == bloc
        ]
        if not members:
            return False
        frontline, _cluster, _objective = self._nearest_cluster(
            squad_centroid, members
        )
        return frontline <= _AUTO_KETTLE_ENGAGE_RANGE

    def _auto_kettle_decision(
        self,
        violent: list,
        squad_centroid: tuple[float, float],
    ) -> None:
        """Autonomously kettle a bloc that is overwhelming its rival.

        The production-half essence: the stand-in commander reads the live
        per-bloc violent strength and decides ON ITS OWN — no operator command —
        to cordon a decisively dominant bloc, switch the cordon if the balance
        flips, or release it when dominance decays.  Hysteresis (a lower release
        ratio) plus a minimum hold stop it thrashing.  Deterministic (integer
        counts, id-stable tiebreaks; no RNG).

        Strength picks the TARGET; reach picks the MOMENT: a dominant bloc out
        of ``_AUTO_KETTLE_ENGAGE_RANGE`` is not kettled yet — the FSM line keeps
        advancing toward it and the doctrine re-evaluates every tick, calling
        the cordon when the squad is close enough to close it.

        No-op unless the doctrine is armed for a rival-faction riot AND the
        operator has not taken manual control (an operator command hard-locks
        the doctrine out until it returns to "auto" / reset()).
        """
        if not self._auto_kettle_enabled:
            return
        if self._tactic_source == "operator":
            return  # a human command overrides the autonomous choice

        counts = self._bloc_violent_counts(violent)
        # Two strongest blocs (deterministic tiebreak by faction id).
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        (a_id, a_n), (b_id, b_n) = ordered[0], ordered[1]

        if self._auto_kettle_target is None:
            # Not currently auto-kettling: ARM on a decisively dominant bloc,
            # once it is in reach.
            if self._is_dominant(a_n, b_n) and self._bloc_in_reach(
                a_id, violent, squad_centroid
            ):
                self._auto_kettle_target = a_id
                self._auto_kettle_since = self._sim_clock
                self.command_tactic("kettle", faction=a_id, _source="autonomous")
            return

        # Already auto-kettling a bloc.  REACH is re-checked EVERY tick — even
        # inside the min-hold window: a bloc that broke contact (fled beyond
        # closing range, or its local cluster dissolved) invalidates the hold,
        # because a cordon that cannot close is a march, not a kettle.  Release
        # to auto so the FSM line pursues; the doctrine re-arms the moment a
        # dominant bloc is back in reach.  (Seed-14 probe: gating this behind
        # min_hold let the persisting kettle command chase clusters 70-180 m
        # out for the whole hold window and the ring never manned.)
        tgt = self._auto_kettle_target
        if not self._bloc_in_reach(tgt, violent, squad_centroid):
            self._auto_kettle_target = None
            self.command_tactic("auto", _source="autonomous")
            return

        # Strength re-decisions (switch / release) wait out the min-hold floor
        # (anti-thrash).
        if self._sim_clock - self._auto_kettle_since < self._auto_kettle_min_hold:
            return

        tgt_n = counts.get(tgt, 0)
        others = [(fid, n) for fid, n in ordered if fid != tgt]
        rival_id, rival_n = others[0] if others else (b_id, b_n)

        if self._is_dominant(rival_n, tgt_n):
            # The balance FLIPPED: the rival now decisively dominates the bloc
            # we were kettling.  Switch the cordon if the new bloc is in reach;
            # otherwise RELEASE — the line advances toward it and the doctrine
            # re-arms once it can actually close the new cordon.
            if self._bloc_in_reach(rival_id, violent, squad_centroid):
                self._auto_kettle_target = rival_id
                self._auto_kettle_since = self._sim_clock
                self.command_tactic(
                    "kettle", faction=rival_id, _source="autonomous"
                )
            else:
                self._auto_kettle_target = None
                self.command_tactic("auto", _source="autonomous")
        elif (
            tgt_n < self._auto_kettle_min_strength
            or tgt_n < rival_n * self._auto_kettle_release_ratio
        ):
            # Dominance decayed (target contained, or the rival caught up past
            # the hysteresis floor): release the cordon back to the FSM.
            self._auto_kettle_target = None
            self.command_tactic("auto", _source="autonomous")
        # else: still dominant enough — HOLD the current cordon (no re-issue).

    # -- Shared arrest / rout loop ---------------------------------------------

    def _process_arrests_routs(
        self,
        violent: list,
        roster: list,
        squad_centroid: tuple[float, float],
    ) -> None:
        """Run the non-lethal arrest + rout pass (shared by all tactics).

        A worn-down violent target with >= 2 officers inside ``arrest_range``
        is detained; a weaker one broken only by a NEARBY officer routs.  Used
        by both the automatic FSM and the kettle cordon so arrests keep landing
        while the crowd is contained.
        """
        for v in violent:
            dists = [
                math.hypot(o.position[0] - v.position[0],
                           o.position[1] - v.position[1])
                for o in roster
            ]
            near = [o for o, d in zip(roster, dists) if d <= self._arrest_range]
            health = getattr(v, "health", 0.0)
            if health <= self._arrest_health and len(near) >= 2:
                self._arrest(v, near)
            elif (
                health <= self._rout_health
                and dists
                and min(dists) <= _ROUT_CREDIT_RANGE
            ):
                # Only the pressure of a NEARBY officer breaks a rioter into
                # flight — no credit for rioters worn down far from the line.
                self._rout(v, squad_centroid)

    # -- FSM internals ----------------------------------------------------------

    def _plan_formation(
        self,
        roster: list,
        violent: list,
        objective: tuple[float, float],
    ) -> None:
        """Compute formation slots and issue movement waypoints.

        Facing points from the line anchor toward the push objective (the
        local violent cluster); LINE spreads officers perpendicular to that
        axis, WEDGE fans them back from a tip.  Slots are assigned to
        officers by a stable sort of ``target_id``.  An operator ``line`` /
        ``wedge`` command forces that shape regardless of cluster size.
        """
        anchor = self._anchor if self._anchor is not None else _centroid(
            [tuple(o.position[:2]) for o in roster]
        )
        facing = math.atan2(
            objective[1] - anchor[1],
            objective[0] - anchor[0],
        )
        if self._commanded_tactic == "line":
            ftype = FormationType.LINE
        elif self._commanded_tactic == "wedge":
            ftype = FormationType.WEDGE
        else:
            cluster = sum(
                1
                for v in violent
                if math.hypot(v.position[0] - objective[0],
                              v.position[1] - objective[1]) <= _WEDGE_CLUSTER_RADIUS
            )
            ftype = (
                FormationType.WEDGE if cluster >= _WEDGE_CLUSTER_MIN
                else FormationType.LINE
            )

        config = FormationConfig(
            formation_type=ftype,
            spacing=self._formation_spacing,
            facing=facing,
            leader_pos=anchor,
            num_members=len(roster),
        )
        slots = get_formation_positions(config)
        self._slots = slots
        self._formation_type = ftype

        ordered = sorted(roster, key=lambda o: o.target_id)
        for officer, slot in zip(ordered, slots):
            # Replace the list object (not mutate) so the entity re-syncs its
            # movement controller to the new slot.  A wired slot_validator nudges
            # a slot off a wall / lethal cell (identity pass-through by default).
            officer.waypoints = [self._validate_slot(anchor, slot)]

    # -- Kettle cordon ----------------------------------------------------------

    def _tick_kettle(
        self,
        dt: float,
        roster: list,
        violent: list,
        all_targets: list,
        squad_centroid: tuple[float, float],
        cluster: list,
        objective: tuple[float, float],
    ) -> None:
        """Kettle a local violent cluster and drive it out a single corridor.

        Officers ring the cluster centroid in an ARC cordon with one open gap;
        the ring tightens each tick; once formed, rioters still inside the ring
        are shoved out through the gap.  Arrests / routs keep running inside the
        cordon.  The gap direction is fixed at command time (deterministic).

        The cordon is GEOGRAPHIC: engaged on the cluster handed in, then
        anchored — the center drifts after the rioters still INSIDE the ring at
        most ``_KETTLE_CENTER_DRIFT`` m/s and never re-targets a different
        cluster (a fresh wave spawning across the map must not teleport a
        half-closed ring off its officers).  When no violent rioter remains
        inside the ring the cordon is DONE here: disband, and the FSM (or the
        autonomous doctrine) re-evaluates from fresh state next tick.
        """
        self._squad_state = "kettle"
        center = objective

        # Fix the gap direction + initial ring on the first kettled tick.
        if self._kettle_gap_dir is None:
            if self._corridor is not None:
                gx = self._corridor[0] - center[0]
                gy = self._corridor[1] - center[1]
            else:
                # Auto: gap opens on the FAR side of the cluster from the squad,
                # so rioters flee away from the line and out the corridor.
                gx = center[0] - squad_centroid[0]
                gy = center[1] - squad_centroid[1]
            gmag = math.hypot(gx, gy)
            self._kettle_gap_dir = (
                (gx / gmag, gy / gmag) if gmag > 1e-6 else (1.0, 0.0)
            )
            spread = max(
                (math.hypot(v.position[0] - center[0], v.position[1] - center[1])
                 for v in cluster),
                default=0.0,
            )
            self._kettle_radius = max(
                _KETTLE_START_RADIUS, spread + _KETTLE_SPREAD_MARGIN
            )
        else:
            # ANCHORED center: follow only the rioters still inside the ring,
            # rate-limited — ignore the handed-in objective entirely (it is the
            # per-tick nearest cluster and teleports when a new wave spawns).
            kc = self._kettle_center if self._kettle_center is not None else center
            contain = self._kettle_radius + _KETTLE_CONTAIN_MARGIN
            inside = [
                v for v in violent
                if math.hypot(v.position[0] - kc[0],
                              v.position[1] - kc[1]) <= contain
            ]
            if not inside:
                # Nobody violent left inside the cordon (arrested, routed, or
                # shoved out the gap): this kettle's work is done.  Disband and
                # stand easy; the doctrine / operator command re-evaluates on
                # fresh state next tick (a persisting "kettle" command simply
                # engages the next cluster as a NEW geographic cordon).
                self._squad_state = "hold"
                self._anchor = None
                self._slots = None
                self._formation_type = None
                self._exit_kettle()
                return
            tx = sum(v.position[0] for v in inside) / len(inside)
            ty = sum(v.position[1] for v in inside) / len(inside)
            dx, dy = tx - kc[0], ty - kc[1]
            d = math.hypot(dx, dy)
            if d > 1e-9:
                step = min(_KETTLE_CENTER_DRIFT * dt, d)
                center = (kc[0] + dx / d * step, kc[1] + dy / d * step)
            else:
                center = kc
            # Tighten the ring toward the floor to squeeze the crowd out.
            self._kettle_radius = max(
                _KETTLE_MIN_RADIUS,
                self._kettle_radius - _KETTLE_SHRINK_RATE * dt,
            )

        self._kettle_center = center
        gap_dir = self._kettle_gap_dir
        facing = math.atan2(gap_dir[1], gap_dir[0])

        # Replan the ARC cordon on the usual cadence.
        self._replan_accum += dt
        if self._slots is None or self._replan_accum >= self._replan_interval:
            self._replan_accum = 0.0
            self._plan_kettle_formation(roster, center, facing)

        # kettle_formed beat: >= 75% of the cordon RING is manned.  Counting
        # MANNED SLOTS, not each officer's sorted-assignment slot, is what "the
        # ring closed" means and is invariant to the officer<->slot churn a
        # moving cluster + tightening ring produce (officers settle on the
        # NEAREST slot, not their assigned one).  Matching is greedy and
        # DISTINCT — each officer mans at most ONE slot (slot order, nearest
        # unclaimed officer; deterministic) — so the ~5 m tolerance (~= the
        # inter-slot spacing) can never let one officer double-count.
        if not self._kettle_formed_announced and self._slots:
            unclaimed = list(roster)
            manned = 0
            for slot in self._slots:
                best = None
                best_d = _KETTLE_ARRIVE_DIST
                for o in unclaimed:
                    d = math.hypot(o.position[0] - slot[0],
                                   o.position[1] - slot[1])
                    if d <= best_d:
                        best, best_d = o, d
                if best is not None:
                    unclaimed.remove(best)
                    manned += 1
            if roster and manned >= _KETTLE_FORMED_FRACTION * len(self._slots):
                self._kettle_formed_announced = True
                self._publish("crowd_event",
                              {"beat": "kettle_formed", "officers": len(roster)})

        # Corridor drive: once formed, shove still-violent targets inside the
        # ring out through the gap, throttled per target.
        if self._kettle_formed_announced:
            exit_r = self._kettle_radius + _CORRIDOR_EXIT_DIST
            gap_exit = (center[0] + gap_dir[0] * exit_r,
                        center[1] + gap_dir[1] * exit_r)
            for v in violent:
                if not _is_violent(v):
                    continue
                if math.hypot(v.position[0] - center[0],
                              v.position[1] - center[1]) > self._kettle_radius:
                    continue
                last = self._corridor_pushed.get(v.target_id, -1e9)
                if self._sim_clock - last >= _CORRIDOR_PUSH_INTERVAL:
                    v.waypoints = [gap_exit]
                    self._corridor_pushed[v.target_id] = self._sim_clock

            if (
                not self._corridor_flow_announced
                and len(self._corridor_pushed) >= _CORRIDOR_FLOW_MIN
            ):
                self._corridor_flow_announced = True
                self._publish("crowd_event",
                              {"beat": "corridor_flow",
                               "pushed": len(self._corridor_pushed)})

        # Arrests / routs keep landing inside the cordon.
        self._process_arrests_routs(violent, roster, squad_centroid)

    def _plan_kettle_formation(
        self,
        roster: list,
        center: tuple[float, float],
        facing: float,
    ) -> None:
        """Place officers on an ARC cordon around ``center`` (gap toward facing)."""
        config = FormationConfig(
            formation_type=FormationType.ARC,
            spacing=self._formation_spacing,
            facing=facing,
            leader_pos=center,
            num_members=len(roster),
            gap_angle=_KETTLE_GAP_ANGLE,
            radius=self._kettle_radius,
        )
        slots = get_formation_positions(config)
        self._formation_type = FormationType.ARC

        # Store the VALIDATED (as-commanded) cordon positions, not the raw ring
        # slots: a slot nudged off a wall is where the officer is actually sent,
        # so the kettle_formed arrival check must measure against it (otherwise a
        # nudged officer never counts as "on its slot" and the cordon can never
        # be reported closed on real terrain).  Identity pass-through when the
        # validator is unwired => byte-identical for costmap-free sim maps.
        #
        # Assignment is PROXIMITY-GREEDY (slot takes the nearest unassigned
        # officer; ties broken by target_id — deterministic, no RNG), not a
        # sorted-id zip: id-order sent near officers across the ring to far
        # slots and vice versa, roughly doubling closure time — and a cordon
        # that cannot close before the crowd walks out never closes at all.
        validated: list[tuple[float, float]] = []
        remaining = sorted(roster, key=lambda o: o.target_id)
        for slot in slots:
            v = self._validate_slot(center, slot)
            validated.append(v)
            if not remaining:
                continue
            officer = min(
                remaining,
                key=lambda o: (
                    math.hypot(o.position[0] - v[0], o.position[1] - v[1]),
                    o.target_id,
                ),
            )
            remaining.remove(officer)
            officer.waypoints = [v]
        self._slots = validated

    def _exit_kettle(self) -> None:
        """Clear all kettle-cordon transient state (FSM resumes at ``form``)."""
        self._kettle_gap_dir = None
        self._kettle_center = None
        self._kettle_radius = _KETTLE_START_RADIUS
        self._kettle_formed_announced = False
        self._corridor_flow_announced = False
        self._corridor_pushed.clear()

    def _observe_shots(self, roster: list) -> None:
        """Raise agitation once per officer that fired since the last tick."""
        for o in roster:
            oid = o.target_id
            last_fired = getattr(o, "last_fired", -1e9)
            prev = self._last_fired_seen.get(oid, last_fired)
            if last_fired > prev:
                self._agitation = min(1.0, self._agitation + _SHOT_AGITATION_RISE)
            self._last_fired_seen[oid] = last_fired

    def _arrest(self, v: Any, near: list) -> None:
        """Non-lethally detain a worn-down violent target."""
        officer_ids = [o.target_id for o in near]
        self._publish("arrest_made", {
            "target_id": v.target_id,
            "officer_ids": officer_ids,
            "position": {"x": v.position[0], "y": v.position[1]},
        })
        v.crowd_role = "calmed"
        v.alliance = "neutral"
        v.is_combatant = False
        v.identified = True
        v.weapon_range = 0.0
        v.weapon_damage = 0.0
        v.weapon_cooldown = 0.0

        if self._game_mode is not None:
            self._game_mode.de_escalation_score += _ARREST_DEESCALATION
            self._game_mode.arrest_count += 1

        self._agitation = max(0.0, self._agitation - _ARREST_AGITATION_DROP)

        self._arrest_total += 1
        if self._arrest_total % _ARREST_SURGE_INTERVAL == 0:
            self._publish("crowd_event", {"beat": "arrest_surge", "arrests": self._arrest_total})

    def _rout(self, v: Any, squad_centroid: tuple[float, float]) -> None:
        """Break a weak violent target: it flees away from the squad."""
        self._publish("rioter_routed", {
            "target_id": v.target_id,
            "position": {"x": v.position[0], "y": v.position[1]},
        })
        v.crowd_role = "civilian"
        v.is_combatant = False

        dx = v.position[0] - squad_centroid[0]
        dy = v.position[1] - squad_centroid[1]
        d = math.hypot(dx, dy)
        if d > 1e-6:
            flee = (
                v.position[0] + dx / d * _ROUT_FLEE_DIST,
                v.position[1] + dy / d * _ROUT_FLEE_DIST,
            )
        else:
            flee = (v.position[0], v.position[1] + _ROUT_FLEE_DIST)
        v.waypoints = [flee]

        if self._game_mode is not None:
            self._game_mode.de_escalation_score += _ROUT_DEESCALATION
            self._game_mode.rout_count += 1

        self._agitation = max(0.0, self._agitation - _ROUT_AGITATION_DROP)

    def _radicalize_bystanders(
        self,
        all_targets: list,
        violent: list,
        roster: list,
        dt: float,
    ) -> None:
        """Un-identified civilians near a live melee contact may radicalize.

        The grievance flare-up: while engaged, a civilian within
        ``_RADICALIZE_RANGE`` of a violent-vs-police melee contact flips to a
        rioter with per-second probability ``_RADICALIZE_RATE * agitation``,
        sampled once per tick via the module-global RNG so the golden-replay
        seed keeps it deterministic.
        """
        # Melee contact points: still-violent targets within melee range of any
        # officer (arrested/routed ones this tick no longer qualify).
        contacts: list[tuple[float, float]] = []
        for v in violent:
            if not _is_violent(v):
                continue
            for o in roster:
                if math.hypot(o.position[0] - v.position[0],
                              o.position[1] - v.position[1]) <= _MELEE_CONTACT_RANGE:
                    contacts.append((v.position[0], v.position[1]))
                    break

        if not contacts or self._agitation <= 0.0:
            return

        threshold = _RADICALIZE_RATE * self._agitation * dt
        for t in all_targets:
            if getattr(t, "crowd_role", None) != "civilian":
                continue
            if getattr(t, "identified", False):
                continue
            if is_terminal(getattr(t, "status", "")):
                continue
            tx, ty = t.position[0], t.position[1]
            if not any(math.hypot(tx - cx, ty - cy) <= _RADICALIZE_RANGE
                       for cx, cy in contacts):
                continue
            if random.random() < threshold:
                t.crowd_role = "rioter"
                t.is_combatant = True

    # -- Lifecycle --------------------------------------------------------------

    def _publish(self, topic: str, data: dict) -> None:
        try:
            self._event_bus.publish(topic, data)
        except Exception:
            pass

    def remove_unit(self, target_id: str) -> None:
        """Drop per-unit state for a removed officer (mirrors InstigatorDetector)."""
        self._last_fired_seen.pop(target_id, None)
        # A removed target can never be corridor-pushed again; clear its throttle.
        self._corridor_pushed.pop(target_id, None)

    def reset(self) -> None:
        """Reset the squad AI to its initial state (clears beat transitions)."""
        self._squad_state = "hold"
        self._anchor = None
        self._slots = None
        self._formation_type = None
        self._replan_accum = 0.0
        self._sim_clock = 0.0
        self._agitation = _INITIAL_AGITATION
        self._last_fired_seen.clear()
        self._line_announced = False
        self._push_announced = False
        self._crowd_broken_announced = False
        self._arrest_total = 0
        self._peak_violent = 0
        # Operator override + kettle cordon back to defaults ("auto").
        self._commanded_tactic = "auto"
        self._corridor = None
        self._kettle_faction = None
        self._tactic_source = "auto"
        # Autonomous doctrine target clears; its ARMED config (blocs/thresholds)
        # persists across a reset so a re-run of the same rival riot keeps it.
        self._auto_kettle_target = None
        self._auto_kettle_since = -1e9
        self._exit_kettle()

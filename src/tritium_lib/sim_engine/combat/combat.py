# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""CombatSystem — projectile flight, hit detection, and damage resolution.

Architecture
------------
CombatSystem manages the lifecycle of Projectile instances:

  1. ``fire()`` creates a Projectile if the source unit passes ``can_fire()``.
     The projectile starts at the source's position and flies toward the
     target's position at the time of firing.

  2. ``tick()`` advances each projectile toward its target_pos.  When the
     projectile enters the hit radius (``HIT_RADIUS`` = 5.0 units) of the
     *target* (tracked by ID, not by frozen position), damage is applied.
     If the target is eliminated (health <= 0), a ``target_eliminated``
     event is published and the interceptor's ``eliminations`` counter is
     incremented.

  3. Projectiles that exceed their max flight time (5 sim seconds) without
     hitting anything are marked as missed and removed.

Elimination streaks are tracked per source_id.  Consecutive neutralizations
within a single game session trigger escalating announcements (3=KILLING
SPREE, 5=RAMPAGE, 7=DOMINATING, 10=GODLIKE).  The streak counter resets
when the source is eliminated.

Ballistics (WP1 — real dispersion, unguided direct fire)
--------------------------------------------------------
There is NO accuracy pre-roll: every shot that passes cooldown / ammo /
range / line-of-sight produces a real projectile.  A weapon's ``accuracy``
is realised as *angular dispersion* about the source->aim bearing rather
than a coin flip that deletes the shot.

The dispersion is self-calibrated so that, at any fire distance, the
probability of landing within ``HIT_RADIUS`` of the aim point equals the
weapon's ``accuracy``.  We draw a lateral miss ``L ~ N(0, sigma)`` and
rotate the aim vector by ``angle_error = atan2(L, dist)``.  Because the
lateral offset at the target is ``dist * tan(angle_error) ~= L``, the hit
probability is ``P(|L| < HIT_RADIUS) = accuracy`` when::

    sigma = HIT_RADIUS / z,   z = NormalDist().inv_cdf((1 + accuracy) / 2)

``z`` is the standard-normal quantile with ``P(|Z| < z) = accuracy``.
An accuracy of 1.0 (>= ``_MAX_ACCURACY``) yields zero dispersion (a perfect
shot at the aim point); an accuracy near 0 yields an enormous spread.

Direct (ballistic) fire is UNGUIDED: the projectile commits to its fired
(dispersed) ``target_pos`` and flies straight, exactly like a mortar arc.
Only ``missile``-class weapons produce ``guided`` projectiles that home
toward the target's current position.  A dispersed near-miss can still
connect if the target walks into it during flight — that is correct.

All randomness in CombatSystem flows through ``self._rng`` (a seedable
``random.Random``) so a battle is bit-for-bit reproducible under a fixed
seed (golden-replay determinism).

Occlusion
---------
Each tick, a non-mortar / non-aerial projectile that moved is raycast
against the stored ``TerrainMap``; if it crosses a building cell it
detonates at the wall (``projectile_impact`` event) and is removed.  Mortar
rounds arc over terrain and aerial shots (flying source or target) are
exempt — the same rule as the fire-time line-of-sight check.

Events are published on the EventBus for the frontend and Amy's announcer:
  - ``projectile_fired``: new dart/rocket in the air (target_pos is the
    dispersed aim point; ``aim_error_deg`` reports the angular error)
  - ``projectile_hit``: damage applied
  - ``projectile_impact``: struck a building mid-flight
  - ``target_eliminated``: health reached zero
  - ``elimination_streak``: milestone reached
"""

from __future__ import annotations

import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tritium_lib.sim_engine.world.intercept import lead_target, target_velocity

if TYPE_CHECKING:
    from tritium_lib.sim_engine.core.entity import SimulationTarget

# Hit detection radius — projectile is "close enough" to count as a hit.
# Must be large enough to account for target movement during projectile flight.
HIT_RADIUS = 5.0

# Miss distance — projectile has overshot target by this much
MISS_OVERSHOOT = 8.0

# Terminal fuse window (sim seconds): brief fuse window after arrival during
# which a near-miss can still clip a mover crossing the impact point; after
# that the round is spent.  Prevents a landed, physically-spent dart from
# sitting on its impact point and damaging a walker who strolls over it
# seconds later (no "ghost darts on the floor").
TERMINAL_WINDOW = 0.3

# Standard flight speed for direct-fire projectiles (m/s).  Also the
# assumed projectile speed used by the internal lead solver when a caller
# does not supply an explicit aim point.
_PROJECTILE_SPEED = 80.0

# Accuracy at or above this is treated as perfect (zero dispersion) — avoids
# the divergent NormalDist quantile as accuracy -> 1.0.
_MAX_ACCURACY = 0.999

# Fallback baseline accuracy when neither a weapon-system weapon nor a
# target-level accuracy attribute is available.
_DEFAULT_ACCURACY = 0.85

# Weather accuracy modifier floor.  An Environment.accuracy_modifier() is
# already clamped to [0.1, 1.0] (1.0 == clear/no-wind), but combat clamps
# again defensively so a pathological modifier can neither divide by zero
# nor blow the dispersion sigma up without bound.  0.1 -> spread x10.
_MIN_ACCURACY_MODIFIER = 0.1


def weather_spread_factor(accuracy_modifier: float) -> float:
    """Map a weather *accuracy modifier* to a dispersion spread multiplier.

    ``accuracy_modifier`` is a value in ``(0, 1]`` where ``1.0`` means clear
    conditions (no degradation) and smaller values mean wind/rain are pulling
    shots wide.  The returned factor multiplies the angular-dispersion sigma:

        sigma_effective = base_sigma * weather_spread_factor(mod)

    so worse weather (lower modifier) -> larger factor -> wider spread ->
    more rounds fly wide of the aim point -> a measurably lower hit rate.

    A clear modifier of ``1.0`` returns exactly ``1.0`` (byte-identical no-op),
    which is what keeps the weather-off path and the canonical goldens
    unchanged.  The modifier is clamped to ``[_MIN_ACCURACY_MODIFIER, 1.0]``
    so the factor is always finite and >= 1.0.  This is the reusable clamp
    lib owns; the engine just supplies the live modifier.
    """
    m = max(_MIN_ACCURACY_MODIFIER, min(1.0, float(accuracy_modifier)))
    return 1.0 / m


def dispersion_sigma(accuracy: float) -> float:
    """Lateral dispersion sigma (world units) calibrated so that
    ``P(|N(0, sigma)| < HIT_RADIUS) == accuracy``.

    Uses the standard-normal quantile ``z = inv_cdf((1 + accuracy) / 2)``
    (the value with ``P(|Z| < z) = accuracy``) so ``sigma = HIT_RADIUS / z``.
    Accuracy is clamped to ``[0, _MAX_ACCURACY]``; an accuracy near 0
    (``z -> 0``) yields an enormous spread rather than dividing by zero.

    This is the single calibration shared by :class:`CombatSystem` (full
    projectile-flight sim) and the transport-agnostic
    :class:`~tritium_lib.sim_engine.combat.match.MatchReferee` (instant-resolve
    duel scoring) — one accuracy model for sim units and wire robots alike.
    """
    a = max(0.0, min(accuracy, _MAX_ACCURACY))
    z = statistics.NormalDist().inv_cdf((1.0 + a) / 2.0)
    if z <= 1e-9:
        return HIT_RADIUS * 1e3
    return HIT_RADIUS / z


# Weapon class whose projectiles home (track the target's live position).
_MISSILE_WEAPON_CLASS = "missile"

# Flying unit types exempt from fire-time LOS and in-flight occlusion —
# a flying source or target shoots/arcs over buildings.  Defined here (not
# imported from world.terrain_map's private _FLYING_TYPES) so combat owns
# its own contract; this set is a superset of that one plus the swarm/plane
# variants that also fly.
_AERIAL_TYPES = frozenset({
    "drone", "scout_drone", "heavy_drone", "recon_drone",
    "swarm_drone", "quad_drone", "plane_drone",
})

# Mortar fire: turrets lob arcing rounds at targets beyond this fraction of
# their weapon_range.  Below the threshold they fire direct (flat trajectory).
MORTAR_RANGE_FRACTION = 0.3  # 30% of range = switch to mortar
# Mortar arc types that use indirect fire
_MORTAR_CAPABLE_TYPES = frozenset({"turret", "heavy_turret", "missile_turret", "tank"})

# Elimination streak thresholds and names
_STREAK_NAMES: list[tuple[int, str]] = [
    (10, "GODLIKE"),
    (7, "DOMINATING"),
    (5, "RAMPAGE"),
    (3, "ON A STREAK"),
]


@dataclass
class Projectile:
    """A single projectile in flight."""

    id: str
    source_id: str
    source_name: str
    target_id: str
    position: tuple[float, float]
    target_pos: tuple[float, float]
    speed: float = 25.0
    damage: float = 10.0
    projectile_type: str = "nerf_dart"  # nerf_dart, nerf_rocket, water_balloon
    source_type: str = ""  # asset_type of the firing unit
    source_pos: tuple[float, float] = (0.0, 0.0)  # origin position at time of fire
    # Sim-time of firing (G-1): CombatSystem stamps this from its own
    # dt-advanced clock; flight expiry compares against the same clock.
    created_at: float = 0.0
    hit: bool = False
    missed: bool = False
    # Terminal window (unguided rounds only): sim-time at which a ballistic
    # round (flat dart OR mortar) first reached its committed aim point.  None
    # while still in flight; once set, the round is spent TERMINAL_WINDOW
    # seconds later.  Guided (missile) rounds home and never "arrive" at a
    # committed point, so they leave this None.
    arrived_at: float | None = None
    # Guidance: guided projectiles (missile-class weapons) home toward the
    # target's live position each tick; unguided direct fire commits to the
    # fired target_pos and flies straight (ballistic), like a mortar arc.
    guided: bool = False
    # Aerial: fired by (or at) a flying unit — exempt from in-flight building
    # occlusion (the shot arcs/passes over terrain), mirroring fire-time LOS.
    aerial: bool = False
    # Mortar/indirect fire fields
    is_mortar: bool = False  # True for arcing mortar rounds
    arc_peak: float = 0.0  # Peak Z height of the arc (world units)
    flight_progress: float = 0.0  # 0.0 = just fired, 1.0 = arrived
    total_flight_dist: float = 0.0  # Total 2D distance source→target

    @property
    def z_height(self) -> float:
        """Current Z height along parabolic arc. 0 at launch/impact, peak at midpoint."""
        if not self.is_mortar or self.arc_peak <= 0:
            return 0.0
        # Parabola: z = 4 * peak * t * (1 - t) where t = flight_progress
        t = max(0.0, min(1.0, self.flight_progress))
        return 4.0 * self.arc_peak * t * (1.0 - t)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "target_id": self.target_id,
            "position": {"x": self.position[0], "y": self.position[1]},
            "target_pos": {"x": self.target_pos[0], "y": self.target_pos[1]},
            "source_pos": {"x": self.source_pos[0], "y": self.source_pos[1]},
            "speed": self.speed,
            "damage": self.damage,
            "projectile_type": self.projectile_type,
            "hit": self.hit,
            "missed": self.missed,
        }
        if self.is_mortar:
            d["is_mortar"] = True
            d["z_height"] = round(self.z_height, 1)
            d["arc_peak"] = round(self.arc_peak, 1)
            d["flight_progress"] = round(self.flight_progress, 2)
        return d


class CombatSystem:
    """Manages projectiles, hit detection, and damage resolution."""

    def __init__(self, event_bus: EventBus, stats_tracker=None,
                 weapon_system=None, upgrade_system: UpgradeSystem | None = None,
                 *, terrain_map=None, rng: random.Random | None = None) -> None:
        self._projectiles: dict[str, Projectile] = {}
        self._event_bus = event_bus
        self._elimination_streaks: dict[str, int] = {}
        self._stats_tracker = stats_tracker
        self._weapon_system = weapon_system
        self._upgrade_system = upgrade_system
        # Stored terrain map for fire-time LOS (when fire() gets no explicit
        # terrain_map) and in-flight occlusion (tick() receives no terrain
        # arg from the engine).
        self._terrain_map = terrain_map
        # Seedable RNG — ALL randomness in CombatSystem (dispersion) flows
        # through this so a battle is reproducible under a fixed seed.
        self._rng: random.Random = rng if rng is not None else random.Random()
        # Combat sim clock (gap G-1): advanced only in tick(dt).
        # Projectile created_at / flight expiry use this clock so flight
        # lifetimes scale with replay speed exactly like movement does.
        self._sim_time: float = 0.0
        # Environmental weather layer (duck-typed: exposes accuracy_modifier()
        # -> float in (0, 1]).  None == weather OFF == no accuracy degradation,
        # so the spread factor is exactly 1.0 and every fire() draw is
        # byte-identical to the pre-weather path (goldens run weather-off).
        # The engine attaches the live Environment only when weather is
        # explicitly enabled, mirroring VisionSystem.environment.
        self._weather_env = None

    @property
    def projectile_count(self) -> int:
        return len(self._projectiles)

    def set_terrain_map(self, terrain_map) -> None:
        """Set the stored TerrainMap used for fire-time LOS and in-flight
        occlusion.  An explicit ``terrain_map`` kwarg on ``fire()`` overrides
        this per-call; ``tick()`` occlusion always uses the stored map."""
        self._terrain_map = terrain_map

    def set_rng(self, rng: random.Random) -> None:
        """Set the RNG used for dispersion (seedable determinism)."""
        self._rng = rng

    def set_weather_environment(self, environment) -> None:
        """Attach (or clear) the environmental weather layer.

        Pass the live :class:`Environment` when weather is enabled, or ``None``
        to disable weather-driven accuracy degradation.  Only the duck-typed
        ``accuracy_modifier() -> float`` is used, read fresh at each ``fire()``
        so a battle whose weather evolves mid-fight tracks the change.  ``None``
        restores the byte-identical clear-weather dispersion.
        """
        self._weather_env = environment

    def _weather_spread_factor(self) -> float:
        """Dispersion multiplier from the current weather (1.0 when off).

        Reads the live ``accuracy_modifier()`` and maps it through the reusable
        :func:`weather_spread_factor` clamp.  Returns exactly ``1.0`` when no
        environment is attached — the no-op that keeps weather-off identical.
        """
        env = self._weather_env
        if env is None:
            return 1.0
        return weather_spread_factor(env.accuracy_modifier())

    @staticmethod
    def _dispersion_sigma(accuracy: float) -> float:
        """Delegate to the module-level :func:`dispersion_sigma` (same math,
        same clamps — kept as a staticmethod so existing callers and the
        golden-replay RNG draw sequence stay byte-identical)."""
        return dispersion_sigma(accuracy)

    def fire(
        self,
        source: SimulationTarget,
        target: SimulationTarget,
        projectile_type: str = "nerf_dart",
        terrain_map=None,
        aim_pos: tuple[float, float] | None = None,
    ) -> Projectile | None:
        """Fire a projectile from *source* at *target*.

        Returns the Projectile if fired, None if source cannot fire.
        Updates source.last_fired timestamp.
        When terrain_map is provided, checks line_of_sight before firing.
        """
        if not source.can_fire():
            return None

        # Ammo check: empty magazine cannot fire.  The DECREMENT happens at
        # the bottom of this method, after every refusal path (range / LOS /
        # reload) — a REFUSED fire solution must not burn a round (a real
        # turret doesn't eject a dart when it declines to shoot).
        if source.ammo_count == 0:
            return None

        # Check range (with upgrade modifier)
        dx = target.position[0] - source.position[0]
        dy = target.position[1] - source.position[1]
        dist = math.hypot(dx, dy)
        effective_range = source.weapon_range
        if self._upgrade_system is not None:
            effective_range *= self._upgrade_system.get_stat_modifier(
                source.target_id, "weapon_range"
            )
        if dist > effective_range:
            return None

        # Determine if this is a mortar (indirect fire) shot.
        # Mortar-capable units lob rounds over obstacles at long range.
        use_mortar = (
            source.asset_type in _MORTAR_CAPABLE_TYPES
            and dist > effective_range * MORTAR_RANGE_FRACTION
        )

        # Aerial shot: a flying source or a flying target.  Exempt from
        # fire-time LOS and in-flight occlusion (arcs/passes over terrain).
        aerial = (
            source.asset_type in _AERIAL_TYPES
            or target.asset_type in _AERIAL_TYPES
        )

        # Check LOS — use the explicit kwarg when given, else the stored map.
        # Skip for mortar arcs and aerial shots (they clear terrain).
        active_terrain = terrain_map if terrain_map is not None else self._terrain_map
        if active_terrain is not None and not use_mortar and not aerial:
            if not active_terrain.line_of_sight(source.position, target.position):
                return None

        # Cooldown stamps are in the SOURCE's sim clock (G-1); duck-typed
        # test stubs without a sim_time fall back to 0.0 harmlessly.
        source_now = getattr(source, "sim_time", 0.0)

        # Weapon system integration: reload check, ammo, and accuracy/guidance.
        # NOTE: no accuracy pre-roll — every qualifying shot spawns a real
        # projectile; accuracy is realised below as angular dispersion.
        accuracy = _DEFAULT_ACCURACY
        guided = False
        ws_weapon = None
        if self._weapon_system is not None:
            if self._weapon_system.is_reloading(source.target_id):
                return None
            ws_weapon = self._weapon_system.get_weapon(source.target_id)
            if ws_weapon is not None:
                accuracy = ws_weapon.accuracy
                guided = ws_weapon.weapon_class == _MISSILE_WEAPON_CLASS
                self._weapon_system.consume_ammo(source.target_id)
        if ws_weapon is None:
            # No weapon-system weapon: prefer a target-level accuracy attr,
            # then weapon_accuracy, else the baseline default.
            acc_attr = getattr(source, "accuracy", None)
            if acc_attr is None:
                acc_attr = getattr(source, "weapon_accuracy", _DEFAULT_ACCURACY)
            accuracy = acc_attr

        # All refusal paths are behind us — this shot is really going out.
        # Decrement target-level ammo and the synced inventory weapon now.
        if source.ammo_count > 0:
            source.ammo_count -= 1
        if hasattr(source, 'inventory') and source.inventory is not None:
            active_wp = source.inventory.get_active_weapon()
            if active_wp is not None and active_wp.ammo > 0:
                active_wp.ammo -= 1
                # Auto-switch when active weapon runs dry
                if active_wp.ammo <= 0:
                    source.inventory.auto_switch_weapon()

        source.last_fired = source_now

        # Determine effective damage from the best available source:
        #   1. Weapon system weapon (synced from inventory at add_target time)
        #   2. target.weapon_damage (flat combat profile fallback)
        # The weapon system is the canonical source of weapon stats during
        # engine-integrated combat.  Direct inventory damage lookup is NOT
        # used here because __post_init__ auto-builds inventory with catalog
        # stats that may differ from the combat profile weapon_damage.
        effective_damage = source.weapon_damage
        if self._weapon_system is not None:
            ws_weapon = self._weapon_system.get_weapon(source.target_id)
            if ws_weapon is not None and ws_weapon.damage > 0:
                effective_damage = ws_weapon.damage
        if self._upgrade_system is not None:
            effective_damage *= self._upgrade_system.get_stat_modifier(
                source.target_id, "weapon_damage"
            )

        # Mortar arc: peak height scales with distance (higher lob for longer shots)
        arc_peak = 0.0
        mortar_speed = 40.0
        if use_mortar:
            arc_peak = max(10.0, dist * 0.4)  # 40% of distance as peak height

        # Intended aim point.  An explicit aim_pos wins; otherwise lead the
        # target internally (live fire sites don't pass a lead point, so the
        # solver here keeps unguided fire on-target for movers).
        if aim_pos is not None:
            intended = (float(aim_pos[0]), float(aim_pos[1]))
        else:
            tvel = target_velocity(
                getattr(target, "heading", 0.0),
                getattr(target, "speed", 0.0),
            )
            intended = lead_target(
                source.position, target.position, tvel, _PROJECTILE_SPEED,
            )

        # Angular dispersion about the source->aim bearing, calibrated so the
        # hit probability equals the weapon accuracy at this fire distance.
        aim_dx = intended[0] - source.position[0]
        aim_dy = intended[1] - source.position[1]
        aim_dist = math.hypot(aim_dx, aim_dy)
        angle_error = 0.0
        target_point = intended
        if accuracy < _MAX_ACCURACY and aim_dist > 1e-9:
            # Weather widens the angular dispersion: worse accuracy_modifier
            # -> larger spread factor -> the same seeded gauss draw lands the
            # round further off the aim bearing.  Clear weather (or weather
            # OFF) -> factor 1.0 -> the sigma and the draw are byte-identical.
            sigma = self._dispersion_sigma(accuracy) * self._weather_spread_factor()
            lateral = self._rng.gauss(0.0, sigma)
            angle_error = math.atan2(lateral, aim_dist)
            cos_e = math.cos(angle_error)
            sin_e = math.sin(angle_error)
            rot_dx = aim_dx * cos_e - aim_dy * sin_e
            rot_dy = aim_dx * sin_e + aim_dy * cos_e
            target_point = (
                source.position[0] + rot_dx,
                source.position[1] + rot_dy,
            )

        proj = Projectile(
            id=str(uuid.uuid4()),
            source_id=source.target_id,
            source_name=source.name,
            target_id=target.target_id,
            position=source.position,
            target_pos=target_point,
            speed=mortar_speed if use_mortar else _PROJECTILE_SPEED,
            damage=effective_damage,
            projectile_type=projectile_type,
            source_type=source.asset_type,
            source_pos=source.position,
            created_at=self._sim_time,
            guided=guided,
            aerial=aerial,
            is_mortar=use_mortar,
            arc_peak=arc_peak,
            total_flight_dist=dist,
        )
        self._projectiles[proj.id] = proj

        self._event_bus.publish("projectile_fired", {
            "id": proj.id,
            "source_id": source.target_id,
            "source_name": source.name,
            "source_type": source.asset_type,
            "source_pos": {"x": source.position[0], "y": source.position[1]},
            "target_id": target.target_id,
            # Dispersed aim point (not the target's live position) so frontend
            # tracers visibly miss when a shot is off.
            "target_pos": {"x": target_point[0], "y": target_point[1]},
            "aim_error_deg": round(math.degrees(angle_error), 2),
            "projectile_type": projectile_type,
            "damage": proj.damage,
            "fire_distance": round(dist, 1),
            "is_mortar": use_mortar,
            "arc_peak": round(arc_peak, 1) if use_mortar else 0,
        })
        # Record shot in stats tracker
        if self._stats_tracker is not None:
            self._stats_tracker.on_shot_fired(source.target_id)
        return proj

    def tick(self, dt: float, targets: dict[str, SimulationTarget],
             cover_system=None) -> None:
        """Advance all projectiles, resolve hits and misses.

        When *cover_system* is provided, damage is reduced by the target's
        cover bonus (0.0-0.8) on each hit.
        """
        self._sim_time += dt
        to_remove: list[str] = []

        for proj in self._projectiles.values():
            if proj.hit or proj.missed:
                to_remove.append(proj.id)
                continue

            # Move projectile toward its aim point.
            #   - Mortars commit to their fired (dispersed) target_pos arc.
            #   - Guided (missile) projectiles home toward the target's
            #     CURRENT position.
            #   - Unguided direct fire is ballistic: it commits to the fired
            #     (dispersed) target_pos and flies straight, exactly like a
            #     mortar — it does NOT curve to follow a moving target.
            prev_pos = proj.position
            target = targets.get(proj.target_id)
            if proj.guided and target is not None and target.status in (
                "active", "idle", "stationary"
            ):
                aim_pos = target.position  # homing munition tracks live pos
            else:
                aim_pos = proj.target_pos  # ballistic: committed aim point
            dx = aim_pos[0] - proj.position[0]
            dy = aim_pos[1] - proj.position[1]
            dist_to_aim = math.hypot(dx, dy)

            if dist_to_aim > 0:
                step = proj.speed * dt
                if step >= dist_to_aim:
                    proj.position = aim_pos
                    # Arrival: an unguided ballistic round (flat dart OR mortar)
                    # reaching its committed aim point is physically spent.
                    if not proj.guided and proj.arrived_at is None:
                        proj.arrived_at = self._sim_time
                else:
                    proj.position = (
                        proj.position[0] + (dx / dist_to_aim) * step,
                        proj.position[1] + (dy / dist_to_aim) * step,
                    )
            elif not proj.guided and proj.arrived_at is None:
                # dist_to_aim == 0: already sitting on the committed aim point.
                proj.arrived_at = self._sim_time

            # Update mortar flight progress (0→1) for arc height calculation
            if proj.is_mortar and proj.total_flight_dist > 0:
                dist_from_source = math.hypot(
                    proj.position[0] - proj.source_pos[0],
                    proj.position[1] - proj.source_pos[1],
                )
                proj.flight_progress = min(1.0, dist_from_source / proj.total_flight_dist)

            # In-flight occlusion: a flat-trajectory ground shot that crosses a
            # building this tick detonates AT the wall.  Mortars arc over and
            # aerial shots pass over — both exempt (same rule as fire-time LOS).
            if (
                self._terrain_map is not None
                and not proj.is_mortar
                and not proj.aerial
                and proj.position != prev_pos
            ):
                impact = self._terrain_map.raycast(prev_pos, proj.position)
                if impact is not None:
                    proj.position = impact
                    proj.missed = True
                    self._event_bus.publish("projectile_impact", {
                        "projectile_id": proj.id,
                        "source_id": proj.source_id,
                        "source_type": proj.source_type,
                        "projectile_type": proj.projectile_type,
                        "position": {"x": impact[0], "y": impact[1]},
                        "surface": "building",
                    })
                    to_remove.append(proj.id)
                    continue

            # Landed-dart terminal window: once an unguided round has been
            # spent on its impact point for longer than TERMINAL_WINDOW, remove
            # it BEFORE the hit test — a mover strolling onto the landing point
            # later takes NO damage (no ghost darts on the floor).  A hit WITHIN
            # the window still counts because the hit check below runs while the
            # round is still fresh.  The 5.0s max-flight expiry remains the
            # outer bound for rounds that never arrive.
            if (
                proj.arrived_at is not None
                and self._sim_time - proj.arrived_at > TERMINAL_WINDOW
            ):
                proj.missed = True
                to_remove.append(proj.id)
                continue

            # Check hit: is the projectile within HIT_RADIUS of the actual target?
            if target is not None and target.status in ("active", "idle", "stationary"):
                tdx = proj.position[0] - target.position[0]
                tdy = proj.position[1] - target.position[1]
                dist_to_target = math.hypot(tdx, tdy)

                if dist_to_target <= HIT_RADIUS:
                    proj.hit = True
                    # Apply cover damage reduction
                    effective_damage = proj.damage
                    if cover_system is not None:
                        cover_bonus = cover_system.get_cover_bonus(
                            target.position, proj.position, target.target_id
                        )
                        effective_damage = proj.damage * (1.0 - cover_bonus)
                    # Apply upgrade damage reduction
                    if self._upgrade_system is not None:
                        reduction = self._upgrade_system.get_stat_modifier(
                            target.target_id, "damage_reduction"
                        )
                        effective_damage *= (1.0 - reduction)
                    # Apply inventory armor damage reduction
                    if hasattr(target, 'inventory') and target.inventory is not None:
                        armor_reduction = target.inventory.total_damage_reduction()
                        if armor_reduction > 0:
                            effective_damage *= (1.0 - armor_reduction)
                            target.inventory.damage_armor(1)
                    # Cap total damage reduction at 80% (minimum 20% of original damage)
                    min_damage = proj.damage * 0.2
                    if effective_damage < min_damage:
                        effective_damage = min_damage
                    eliminated = target.apply_damage(effective_damage)
                    self._event_bus.publish("projectile_hit", {
                        "projectile_id": proj.id,
                        "target_id": target.target_id,
                        "target_name": target.name,
                        "damage": effective_damage,
                        "remaining_health": target.health,
                        "source_id": proj.source_id,
                        "source_type": proj.source_type,
                        "source_name": proj.source_name,
                        "source_pos": {"x": proj.source_pos[0], "y": proj.source_pos[1]},
                        "projectile_type": proj.projectile_type,
                        "position": {"x": target.position[0], "y": target.position[1]},
                    })
                    # Record hit in stats tracker
                    if self._stats_tracker is not None:
                        self._stats_tracker.on_shot_hit(
                            proj.source_id, target.target_id, effective_damage
                        )

                    if eliminated:
                        # Increment interceptor stats
                        interceptor = targets.get(proj.source_id)
                        interceptor_name = proj.source_name
                        if interceptor is not None:
                            interceptor.kills += 1
                            interceptor_name = interceptor.name

                        self._event_bus.publish("target_eliminated", {
                            "target_id": target.target_id,
                            "target_name": target.name,
                            "target_type": target.asset_type,
                            "target_alliance": target.alliance,
                            "alliance": target.alliance,
                            "interceptor_id": proj.source_id,
                            "interceptor_name": interceptor_name,
                            "interceptor_type": proj.source_type,
                            "interceptor_alliance": (
                                interceptor.alliance if interceptor is not None else None
                            ),
                            "position": {"x": target.position[0], "y": target.position[1]},
                            "method": proj.projectile_type,
                        })
                        # Record kill in stats tracker
                        if self._stats_tracker is not None:
                            self._stats_tracker.on_kill(
                                proj.source_id, target.target_id
                            )

                        # Elimination streak tracking
                        self._elimination_streaks[proj.source_id] = (
                            self._elimination_streaks.get(proj.source_id, 0) + 1
                        )
                        streak = self._elimination_streaks[proj.source_id]
                        streak_name = self._get_streak_name(streak)
                        if streak_name:
                            self._event_bus.publish("elimination_streak", {
                                "interceptor_id": proj.source_id,
                                "interceptor_name": interceptor_name,
                                "streak": streak,
                                "streak_name": streak_name,
                            })

                    to_remove.append(proj.id)
                    continue

            # Check miss: projectile exceeded max flight time (5 SIM
            # seconds — wall-clock here would never expire darts at
            # faster-than-real-time replay, G-1)
            flight_time = self._sim_time - proj.created_at
            if flight_time > 5.0:
                proj.missed = True
                to_remove.append(proj.id)

        for pid in to_remove:
            self._projectiles.pop(pid, None)

    def reset_streaks(self) -> None:
        """Reset all elimination streak counters."""
        self._elimination_streaks.clear()

    def reset_streak(self, target_id: str) -> None:
        """Reset elimination streak for a specific unit (e.g. when eliminated)."""
        self._elimination_streaks.pop(target_id, None)

    def get_active_projectiles(self) -> list[dict]:
        """Return serializable list of active projectiles for frontend rendering."""
        return [p.to_dict() for p in self._projectiles.values()
                if not p.hit and not p.missed]

    def detonate_bomber(
        self,
        bomber: SimulationTarget,
        targets: dict[str, SimulationTarget],
        radius: float = 5.0,
    ) -> list[str]:
        """Detonate a bomber drone, applying AoE damage.

        Applies bomber's weapon_damage to all targets within *radius*
        (excluding the bomber itself). Returns list of damaged target IDs.

        Args:
            bomber: The bomber drone detonating.
            targets: All targets in the simulation.
            radius: Blast radius in meters.

        Returns:
            List of target IDs that were damaged.
        """
        damage = bomber.weapon_damage
        damaged: list[str] = []
        r2 = radius * radius

        for tid, t in targets.items():
            if tid == bomber.target_id:
                continue
            dx = t.position[0] - bomber.position[0]
            dy = t.position[1] - bomber.position[1]
            if dx * dx + dy * dy <= r2:
                t.apply_damage(damage)
                damaged.append(tid)

        # Publish detonation event
        self._event_bus.publish("bomber_detonation", {
            "bomber_id": bomber.target_id,
            "position": {"x": bomber.position[0], "y": bomber.position[1]},
            "radius": radius,
            "damage": damage,
        })

        # Mark bomber as eliminated
        bomber.health = 0
        bomber.status = "eliminated"

        return damaged

    def clear(self) -> None:
        """Remove all projectiles."""
        self._projectiles.clear()

    @staticmethod
    def _get_streak_name(streak: int) -> str | None:
        """Return the streak announcement name, or None if not a milestone."""
        for threshold, name in _STREAK_NAMES:
            if streak == threshold:
                return name
        return None

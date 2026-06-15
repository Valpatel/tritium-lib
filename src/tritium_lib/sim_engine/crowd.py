# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Crowd / riot / protest simulation module.

Simulates crowds from peaceful protests to full riots with mood propagation,
event injection, escalation cascades, and JSON-serializable output for
Three.js rendering.

All spatial units are meters. Positions are Vec2 = tuple[float, float].
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    magnitude,
    normalize,
    truncate,
    _add,
    _sub,
    _scale,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CrowdMood(IntEnum):
    """Crowd mood levels, ordered by intensity."""
    CALM = 0
    UNEASY = 1
    AGITATED = 2
    RIOTING = 3
    PANICKED = 4
    FLEEING = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CrowdMember:
    """A single individual in the crowd."""
    member_id: str
    position: Vec2
    velocity: Vec2 = (0.0, 0.0)
    mood: CrowdMood = CrowdMood.CALM
    aggression: float = 0.0
    fear: float = 0.0
    has_weapon: bool = False
    is_leader: bool = False
    group_id: str = ""
    # Organic-movement state (riot-quality rework, 2026-06-14)
    exit_target: Vec2 | None = None   # sticky flee destination (dispersed exits)
    dwell: float = 0.0                 # CALM/UNEASY: seconds left standing still (milling)
    wander: Vec2 = (0.0, 0.0)          # persistent, slowly-turning wander heading
    # Stable per-cluster sector point seeded at spawn (riot legibility rework,
    # 2026-06-15). Steering targets this anchor (NOT the live group centroid,
    # which implodes the crowd to the centre) so clusters hold their distinct
    # sectors across the full bounds. None for non-riot presets (no behaviour
    # change there). Consumed by _move_members / _update_group_objectives.
    anchor: Vec2 | None = None


@dataclass
class CrowdEvent:
    """A discrete event that affects the crowd."""
    event_type: str
    position: Vec2
    radius: float
    intensity: float
    timestamp: float
    _age: float = field(default=0.0, repr=False)


# ---------------------------------------------------------------------------
# Event effect tables
# ---------------------------------------------------------------------------

# How each event type affects aggression and fear within its radius.
# (aggression_delta, fear_delta)
_EVENT_EFFECTS: dict[str, tuple[float, float]] = {
    "gunshot":      (0.1,  0.6),
    "teargas":      (-0.1, 0.5),
    "flashbang":    (-0.1, 0.4),
    "arrest":       (0.2,  0.15),
    "speech":       (0.0,  -0.2),   # calming by default; intensity scales it
    "chant":        (0.15, -0.05),
    "throw_object": (0.2,  0.05),
    "charge":       (0.3,  0.2),
    "retreat":      (-0.2, 0.1),
    "stampede":     (0.0,  0.7),
}

# Mood thresholds derived from aggression/fear
_MOOD_FEAR_FLEE = 0.7
_MOOD_FEAR_PANIC = 0.5
_MOOD_AGG_RIOT = 0.6
_MOOD_AGG_AGITATED = 0.3
_MOOD_AGG_UNEASY = 0.1

# Movement speeds (m/s)
_SPEED_CALM = 0.8
_SPEED_AGITATED = 1.5
_SPEED_RIOTING = 2.0
_SPEED_PANICKED = 3.0
_SPEED_FLEEING = 3.5

# Mood propagation
_NEIGHBOR_RADIUS = 5.0
_LEADER_RADIUS_MULT = 3.0
_PROPAGATION_STRENGTH = 0.05
_LEADER_INFLUENCE_MULT = 2.0

# Event decay rate per second
_EVENT_DECAY_RATE = 0.15

# Escalation thresholds
_ESCALATION_AGITATED_RATIO = 0.30
_ESCALATION_RIOTING_RATIO = 0.50


# ---------------------------------------------------------------------------
# Spatial hash for O(1) neighbor queries
# ---------------------------------------------------------------------------

class _CrowdGrid:
    """Lightweight spatial hash for CrowdMember neighbor queries.

    Avoids O(n^2) scans in mood propagation and separation by partitioning
    members into grid cells.  Rebuilt once per tick, queried many times.
    """

    __slots__ = ("_cell_size", "_inv_cell_size", "_cells")

    def __init__(self, cell_size: float = 5.0) -> None:
        if cell_size <= 0:
            cell_size = 5.0
        self._cell_size = cell_size
        self._inv_cell_size = 1.0 / cell_size
        self._cells: dict[tuple[int, int], list[CrowdMember]] = {}

    def rebuild(self, members: list[CrowdMember]) -> None:
        """Clear and re-insert all members."""
        cells: dict[tuple[int, int], list[CrowdMember]] = {}
        inv = self._inv_cell_size
        for m in members:
            key = (int(math.floor(m.position[0] * inv)),
                   int(math.floor(m.position[1] * inv)))
            bucket = cells.get(key)
            if bucket is None:
                bucket = []
                cells[key] = bucket
            bucket.append(m)
        self._cells = cells

    def query_radius(self, pos: Vec2, radius: float) -> list[CrowdMember]:
        """Return all members within *radius* of *pos*."""
        px, py = pos
        r2 = radius * radius
        inv = self._inv_cell_size

        min_cx = int(math.floor((px - radius) * inv))
        max_cx = int(math.floor((px + radius) * inv))
        min_cy = int(math.floor((py - radius) * inv))
        max_cy = int(math.floor((py + radius) * inv))

        result: list[CrowdMember] = []
        cells = self._cells
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                bucket = cells.get((cx, cy))
                if bucket is None:
                    continue
                for m in bucket:
                    dx = m.position[0] - px
                    dy = m.position[1] - py
                    if dx * dx + dy * dy <= r2:
                        result.append(m)
        return result


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class CrowdSimulator:
    """Crowd simulation engine with mood propagation, events, and escalation.

    Parameters
    ----------
    bounds : tuple of four floats (x_min, y_min, x_max, y_max)
        World-space bounding box that the crowd is confined to.
    max_members : int
        Hard cap on crowd size.
    """

    def __init__(self, bounds: tuple[float, float, float, float], max_members: int = 500,
                 obstacles: Any = None, street_nodes: list[Vec2] | None = None,
                 seed: int | None = None) -> None:
        self.bounds = bounds
        self.max_members = max_members
        self.members: list[CrowdMember] = []
        self.events: list[CrowdEvent] = []
        self.overall_mood: CrowdMood = CrowdMood.CALM
        self._time: float = 0.0
        # Optional BuildingObstacles (duck-typed: needs point_in_building(x, y)).
        # When set, members never walk into buildings — the crowd is confined to
        # streets / open space instead of drifting through footprints.
        self.obstacles = obstacles
        # Optional list of (x, y) street-node positions. When set, group gather
        # points snap to the nearest street node so crowds form on real streets /
        # junctions instead of collapsing onto an arbitrary centroid, AND fleeing
        # crowds disperse toward real perimeter streets (set before exits so the
        # exit zones are street-aware from construction).
        self.street_nodes = street_nodes
        self._exits: list[Vec2] = self._compute_exits()
        self._grid = _CrowdGrid(cell_size=5.0)
        # Per-group objective cycle (riot legibility rework, 2026-06-15): each
        # group marches to a stable sector anchor, holds for a dwell, then
        # rotates its objective to an ADJACENT sector — so clusters roam the
        # whole map instead of imploding to the centre. Seeded so occupancy
        # tests are deterministic. Empty / unused for non-riot presets.
        self._group_anchors: dict[str, Vec2] = {}
        self._group_phase: dict[str, str] = {}        # 'march' | 'hold'
        self._group_hold_until: dict[str, float] = {}
        self._objective_rng = random.Random(seed)

    # -- helpers -------------------------------------------------------------

    def _compute_exits(self) -> list[Vec2]:
        """Exit zones distributed around the boundary so fleeing crowds disperse
        in many directions instead of streaming onto 4 midpoints (the old
        4-exit model produced the conspicuous "lines of units").

        When street nodes are loaded, prefer real PERIMETER street nodes — the
        roads leading out of the area — so a fleeing crowd disperses DOWN actual
        streets (makes the "scatters down the streets" narration literally true),
        not toward arbitrary geometric boundary points. Falls back to the
        geometric ring when there's no road graph or too few perimeter nodes.
        """
        x0, y0, x1, y1 = self.bounds
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        nodes = self.street_nodes
        if nodes:
            rad = max(rx, ry) or 1.0
            perimeter = [
                (nx, ny) for (nx, ny) in nodes
                if math.hypot(nx - cx, ny - cy) >= 0.55 * rad
            ]
            if len(perimeter) >= 4:
                return perimeter
        n = 12
        return [
            (cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]

    def set_street_nodes(self, nodes: list[Vec2] | None) -> None:
        """Set the street-node positions and recompute exit zones so fleeing
        crowds disperse toward real perimeter streets. Use this (not direct
        attribute assignment) when street data arrives after construction, e.g.
        the live crowd worker loading the road graph on an existing sim."""
        self.street_nodes = nodes
        self._exits = self._compute_exits()

    def _clamp_to_bounds(self, pos: Vec2) -> Vec2:
        x0, y0, x1, y1 = self.bounds
        return (
            max(x0, min(x1, pos[0])),
            max(y0, min(y1, pos[1])),
        )

    def _resolve_mood(self, member: CrowdMember) -> CrowdMood:
        """Derive mood from aggression/fear levels."""
        if member.fear >= _MOOD_FEAR_FLEE:
            return CrowdMood.FLEEING
        if member.fear >= _MOOD_FEAR_PANIC:
            return CrowdMood.PANICKED
        if member.aggression >= _MOOD_AGG_RIOT:
            return CrowdMood.RIOTING
        if member.aggression >= _MOOD_AGG_AGITATED:
            return CrowdMood.AGITATED
        if member.aggression >= _MOOD_AGG_UNEASY:
            return CrowdMood.UNEASY
        return CrowdMood.CALM

    def _pick_exit(self, pos: Vec2) -> Vec2:
        """Pick a stochastic exit from the nearest few zones, with jitter, so
        fleeing members spread into a dispersed cloud instead of collapsing
        onto the single nearest exit (which produced 4 streaming lines)."""
        ranked = sorted(self._exits, key=lambda ex: distance(pos, ex))
        choice = random.choice(ranked[: min(3, len(ranked))])
        return (choice[0] + random.uniform(-6.0, 6.0),
                choice[1] + random.uniform(-6.0, 6.0))

    # -- public API ----------------------------------------------------------

    def spawn_crowd(
        self,
        center: Vec2,
        count: int,
        radius: float,
        mood: CrowdMood = CrowdMood.CALM,
        leader_ratio: float = 0.05,
        rng: random.Random | None = None,
        anchor: Vec2 | None = None,
    ) -> list[str]:
        """Spawn *count* crowd members around *center* within *radius*.

        *rng* — optional seeded ``random.Random`` for deterministic placement
        (used by the riot preset so occupancy tests are repeatable). Falls back
        to the module ``random`` when None (unchanged behaviour).
        *anchor* — optional stable sector point assigned to every member's
        ``anchor`` field, so the cluster steers toward its seeded sector instead
        of the live centroid (riot legibility rework, 2026-06-15).

        Returns list of member IDs created.
        """
        rnd = rng if rng is not None else random
        ids: list[str] = []
        available = self.max_members - len(self.members)
        actual = min(count, available)
        # Group id from the seeded RNG when one is supplied, so a seeded preset
        # is fully deterministic (occupancy tests). Falls back to a uuid for
        # live, unseeded runs.
        group_id = (f"g_{rnd.getrandbits(32):08x}" if rng is not None
                    else uuid.uuid4().hex[:8])

        aggression = 0.0
        fear = 0.0
        if mood == CrowdMood.UNEASY:
            aggression = 0.15
        elif mood == CrowdMood.AGITATED:
            aggression = 0.4
        elif mood == CrowdMood.RIOTING:
            aggression = 0.7
        elif mood == CrowdMood.PANICKED:
            fear = 0.55
        elif mood == CrowdMood.FLEEING:
            fear = 0.75

        for i in range(actual):
            angle = rnd.uniform(0, 2 * math.pi)
            r = radius * math.sqrt(rnd.random())
            px = center[0] + r * math.cos(angle)
            py = center[1] + r * math.sin(angle)
            pos = self._clamp_to_bounds((px, py))

            mid = f"c_{len(self.members)}"
            is_leader = (i / max(actual, 1)) < leader_ratio
            member = CrowdMember(
                member_id=mid,
                position=pos,
                mood=mood,
                aggression=aggression + rnd.uniform(-0.05, 0.05),
                fear=fear + rnd.uniform(-0.05, 0.05),
                is_leader=is_leader,
                group_id=group_id,
                anchor=anchor,
            )
            member.aggression = max(0.0, min(1.0, member.aggression))
            member.fear = max(0.0, min(1.0, member.fear))
            member.mood = self._resolve_mood(member)
            self.members.append(member)
            ids.append(mid)

        return ids

    def inject_event(self, event: CrowdEvent) -> None:
        """Inject a crowd event that affects nearby members."""
        event._age = 0.0
        self.events.append(event)
        self._apply_event(event)

    def _apply_event(self, event: CrowdEvent) -> None:
        """Apply an event's effect to members within its radius."""
        if event.radius <= 0:
            return
        effects = _EVENT_EFFECTS.get(event.event_type, (0.0, 0.0))
        agg_delta, fear_delta = effects

        for m in self.members:
            d = distance(m.position, event.position)
            if d > event.radius:
                continue
            # Effect falls off with distance
            falloff = 1.0 - (d / event.radius)
            strength = falloff * event.intensity

            m.aggression = max(0.0, min(1.0, m.aggression + agg_delta * strength))
            m.fear = max(0.0, min(1.0, m.fear + fear_delta * strength))
            m.mood = self._resolve_mood(m)

    def tick(self, dt: float, route_fn: Any = None) -> None:
        """Advance simulation by *dt* seconds.

        1. Mood propagation among neighbors
        2. Per-group objective cycle (march -> hold -> rotate sector)
        3. Movement based on mood
        4. Collision avoidance + boundary enforcement
        5. Event decay
        6. Escalation checks

        *route_fn* — optional ``callable(start_xy, end_xy) -> list[xy] | None``
        (e.g. ``StreetGraph.shortest_path``) so group marches follow roads. Kept
        as a param to keep this module independent of SC — never imported here.
        """
        self._time += dt

        # Rebuild spatial grid once per tick for O(1) neighbor queries
        self._grid.rebuild(self.members)

        # Precompute group centers (avoids O(n) scan per member)
        self._group_center_cache = self._compute_group_centers()

        # 1. Mood propagation
        self._propagate_moods(dt)

        # 2. Per-group objective cycle (BEFORE movement) — marches clusters to
        # stable sector anchors and rotates objectives to adjacent sectors so
        # the crowd roams the whole map instead of imploding to the centre.
        self._update_group_objectives(route_fn=route_fn)

        # 3 & 4. Movement
        self._move_members(dt)

        # 4. Event decay
        self._decay_events(dt)

        # 5. Escalation
        self._check_escalation()

        # Update overall mood
        self._update_overall_mood()

    def _propagate_moods(self, dt: float) -> None:
        """Neighbors within range influence each other's aggression/fear.

        Uses the spatial grid for O(k) neighbor lookups instead of O(n) scans.
        """
        n = len(self.members)
        if n == 0:
            return

        # Accumulate deltas then apply (avoid order-dependent bias)
        agg_deltas: dict[int, float] = {}
        fear_deltas: dict[int, float] = {}

        # Build id->index map for delta accumulation
        id_to_idx: dict[int, int] = {id(m): i for i, m in enumerate(self.members)}

        grid = self._grid
        for i in range(n):
            mi = self.members[i]
            radius = _NEIGHBOR_RADIUS * (_LEADER_RADIUS_MULT if mi.is_leader else 1.0)
            influence = _PROPAGATION_STRENGTH * (_LEADER_INFLUENCE_MULT if mi.is_leader else 1.0)

            neighbors = grid.query_radius(mi.position, radius)
            for mj in neighbors:
                if mj is mi:
                    continue
                d = distance(mi.position, mj.position)
                if d < 1e-6:
                    continue

                falloff = 1.0 - (d / radius)
                j = id_to_idx[id(mj)]
                # i influences j
                agg_deltas[j] = agg_deltas.get(j, 0.0) + (mi.aggression - mj.aggression) * influence * falloff * dt
                fear_deltas[j] = fear_deltas.get(j, 0.0) + (mi.fear - mj.fear) * influence * falloff * dt

        for j, delta in agg_deltas.items():
            m = self.members[j]
            m.aggression = max(0.0, min(1.0, m.aggression + delta))
        for j, delta in fear_deltas.items():
            m = self.members[j]
            m.fear = max(0.0, min(1.0, m.fear + delta))
        for m in self.members:
            m.mood = self._resolve_mood(m)

    def _blocked(self, pos: Vec2) -> bool:
        """True if pos is inside a building footprint (guarded, no-op if no obstacles)."""
        if self.obstacles is None:
            return False
        try:
            return bool(self.obstacles.point_in_building(pos[0], pos[1]))
        except Exception:
            return False

    def _apply_velocity(self, member: CrowdMember, dt: float) -> None:
        """Advance a member by its velocity, but never INTO a building. If the
        diagonal move is blocked, SLIDE along the wall — move on whichever axis
        is still free — so members flow around buildings instead of piling up at
        the wall (riot-mode rework, 2026-06-14)."""
        # Already inside a building (spawned/snapped in, or steered to a gather
        # point that sat on a building node)? Eject to the nearest open space.
        # Wall-slide only stops members ENTERING; it cannot free one that begins
        # the tick inside (every slide axis is blocked -> frozen inside forever).
        if self._blocked(member.position):
            out = self._nearest_open(member.position)
            if out is not None:
                member.position = out
            member.velocity = (0.0, 0.0)
            return
        vx, vy = member.velocity
        full = self._clamp_to_bounds((member.position[0] + vx * dt, member.position[1] + vy * dt))
        if not self._blocked(full):
            member.position = full
            return
        # Wall-slide: take whichever single-axis move is unobstructed.
        slide_x = self._clamp_to_bounds((member.position[0] + vx * dt, member.position[1]))
        if not self._blocked(slide_x):
            member.position = slide_x
            member.velocity = (vx, 0.0)
            return
        slide_y = self._clamp_to_bounds((member.position[0], member.position[1] + vy * dt))
        if not self._blocked(slide_y):
            member.position = slide_y
            member.velocity = (0.0, vy)
            return
        member.velocity = (0.0, 0.0)   # truly cornered

    def _span(self) -> float:
        """Smaller side of the world bounds — the spatial scale for sector
        radii / formation jitter (matches the riot preset's ``span``)."""
        x0, y0, x1, y1 = self.bounds
        return min(x1 - x0, y1 - y0)

    def _group_centroid(self, gid: str) -> Vec2 | None:
        """Raw (un-snapped) centroid of a group's members, or None if empty.
        Distinct from _group_center (leader-biased + street-snapped) — used only
        to test arrival at an objective anchor, so it must NOT snap."""
        sx = sy = 0.0
        cnt = 0
        for m in self.members:
            if m.group_id == gid:
                sx += m.position[0]
                sy += m.position[1]
                cnt += 1
        if cnt == 0:
            return None
        return (sx / cnt, sy / cnt)

    def _update_group_objectives(self, route_fn: Any = None) -> None:
        """Per-group objective cycle (riot legibility rework, 2026-06-15).

        For each group: seed a stable sector anchor from its members' spawn
        sector; when the group centroid reaches that anchor, HOLD for an 8-12s
        dwell, then ROTATE the objective to an adjacent sector (never a
        diametric centre-crossing) at radius >= span*0.35. On a high-fear event
        nearby (rank 12) the group flows to the perimeter node farthest from the
        event; with low probability a group splinters. All RNG is the seeded
        ``self._objective_rng`` so occupancy tests stay deterministic.

        Only does work for AGITATED / RIOTING members (the riot crowd). PANICKED
        / FLEEING use the exit logic, CALM / UNEASY mill — untouched here.
        """
        if not self.members:
            return
        span = self._span()
        cx = (self.bounds[0] + self.bounds[2]) / 2.0
        cy = (self.bounds[1] + self.bounds[3]) / 2.0
        rng = self._objective_rng

        # Group ids that have at least one agitated/rioting member. Sorted so
        # the objective-update order is deterministic (occupancy tests) — set
        # iteration order over string keys is process-randomised otherwise.
        active_gids: set[str] = set()
        for m in self.members:
            if m.group_id and m.mood in (CrowdMood.AGITATED, CrowdMood.RIOTING):
                active_gids.add(m.group_id)

        for gid in sorted(active_gids):
            # 1. Seed the anchor from the members' spawn sector (their stable
            #    per-member anchor) the first time we see this group.
            if gid not in self._group_anchors:
                seed_anchor = None
                for m in self.members:
                    if m.group_id == gid and m.anchor is not None:
                        seed_anchor = m.anchor
                        break
                if seed_anchor is None:
                    seed_anchor = self._group_centroid(gid)
                if seed_anchor is None:
                    continue
                self._group_anchors[gid] = seed_anchor
                self._group_phase[gid] = "march"

            anchor = self._group_anchors[gid]
            centroid = self._group_centroid(gid)
            if centroid is None:
                continue

            # 2. High-fear event nearby (rank 12): flow to the perimeter node
            #    FARTHEST from the event for a short window (down a road away
            #    from the police line), then regroup.
            fear_ev = self._nearest_high_fear_event(centroid)
            if fear_ev is not None and distance(centroid, fear_ev) <= 30.0:
                self._group_anchors[gid] = self._farthest_exit(fear_ev)
                self._group_phase[gid] = "march"
                self._group_hold_until[gid] = 0.0
                continue

            phase = self._group_phase.get(gid, "march")
            if phase == "march":
                # Arrived? -> hold for a seeded dwell.
                if distance(centroid, anchor) <= 10.0:
                    self._group_phase[gid] = "hold"
                    self._group_hold_until[gid] = self._time + rng.uniform(8.0, 12.0)
            else:  # hold
                if self._time >= self._group_hold_until.get(gid, 0.0):
                    # Rotate to an ADJACENT sector and march there.
                    self._group_anchors[gid] = self._rotate_objective(
                        anchor, cx, cy, span, route_fn=route_fn,
                    )
                    self._group_phase[gid] = "march"
                    # Low-probability splinter: peel ~30% of the group off with a
                    # fresh group_id and its own adjacent-sector objective.
                    if rng.random() < 0.15:
                        self._splinter_group(gid, cx, cy, span, rng)

    def _rotate_objective(
        self,
        anchor: Vec2,
        cx: float,
        cy: float,
        span: float,
        route_fn: Any = None,
    ) -> Vec2:
        """Rotate the group's objective to an ADJACENT compass sector (index
        +/-1 of 6, never a diametric centre-crossing) at radius >= span*0.35.
        Snaps to a street node near that bearing (biased toward recent events)
        when street data exists, else a deterministic ring point. Optionally
        validates a road route via *route_fn* (recompute only here, on change)."""
        rng = self._objective_rng
        cur_angle = math.atan2(anchor[1] - cy, anchor[0] - cx)
        step = math.pi / 3.0  # 60 deg = one of 6 sectors
        new_angle = cur_angle + (step if rng.random() < 0.5 else -step)
        radius = span * (0.35 + rng.uniform(0.0, 0.07))
        ring_point = self._clamp_to_bounds(
            (cx + radius * math.cos(new_angle), cy + radius * math.sin(new_angle))
        )
        target = ring_point
        nodes = self.street_nodes
        if nodes:
            # Prefer a street node near the new bearing; bias toward nodes close
            # to a recent event so marches drift toward the action.
            ev_pos = None
            if self.events:
                ev_pos = self.events[-1].position
            best = None
            best_score = float("inf")
            for nx, ny in nodes:
                if math.hypot(nx - cx, ny - cy) < span * 0.35:
                    continue
                score = distance((nx, ny), ring_point)
                if ev_pos is not None:
                    score += 0.3 * distance((nx, ny), ev_pos)
                if score < best_score:
                    best_score = score
                    best = (nx, ny)
            if best is not None:
                target = best
        # Validate a road route if one was supplied — guarded, falls back to the
        # direct sector point. We don't store the path (steering seeks the
        # anchor); this just ensures the objective is reachable by road.
        if route_fn is not None:
            try:
                route = route_fn(anchor, target)
                if route:
                    target = route[-1]
            except Exception:
                pass
        return target

    def _splinter_group(self, gid: str, cx: float, cy: float, span: float,
                        rng: random.Random) -> None:
        """Peel ~30% of *gid*'s members into a fresh splinter group with its own
        adjacent-sector objective (riot splinter, rank 12)."""
        members = [m for m in self.members if m.group_id == gid]
        if len(members) < 4:
            return
        n_split = max(1, int(len(members) * 0.30))
        new_gid = f"g_{rng.getrandbits(32):08x}"
        base_anchor = self._group_anchors.get(gid)
        if base_anchor is None:
            return
        new_anchor = self._rotate_objective(base_anchor, cx, cy, span)
        for m in members[:n_split]:
            m.group_id = new_gid
            m.anchor = new_anchor
        self._group_anchors[new_gid] = new_anchor
        self._group_phase[new_gid] = "march"

    def _nearest_high_fear_event(self, pos: Vec2) -> Vec2 | None:
        """Position of the nearest high-fear event (teargas / gunshot / charge /
        flashbang) — used to flow a group away from police pressure (rank 12)."""
        high_fear = {"teargas", "gunshot", "charge", "flashbang"}
        cands = [e for e in self.events if e.event_type in high_fear]
        if not cands:
            return None
        nearest = min(cands, key=lambda e: distance(pos, e.position))
        return nearest.position

    def _farthest_exit(self, event_pos: Vec2) -> Vec2:
        """Perimeter node FARTHEST from *event_pos* (reuses the exit ring), so a
        group flows down a road away from the threat. Falls back to the bounds
        centre offset away from the event if no exits exist."""
        if self._exits:
            return max(self._exits, key=lambda ex: distance(ex, event_pos))
        cx = (self.bounds[0] + self.bounds[2]) / 2.0
        cy = (self.bounds[1] + self.bounds[3]) / 2.0
        away = _sub((cx, cy), event_pos)
        if magnitude(away) < 1e-6:
            return (cx, cy)
        span = self._span()
        return self._clamp_to_bounds(_add((cx, cy), _scale(normalize(away), span * 0.4)))

    def _member_anchor(self, m: CrowdMember) -> Vec2 | None:
        """The member's stable sector anchor — its own ``anchor`` field, else the
        live per-group objective anchor. None when neither is set (so the caller
        falls back to the old group-centre behaviour, e.g. operator-spawned
        crowds with no seeded sector)."""
        if m.anchor is not None:
            return m.anchor
        if m.group_id:
            return self._group_anchors.get(m.group_id)
        return None

    def _mill_target(self, m: CrowdMember, dt: float, speed: float) -> Vec2 | None:
        """CALM/UNEASY-style milling: occasionally stand still (dwell), else
        drift on a slowly-turning wander heading. Returns a seek target, or None
        when the member is dwelling (velocity already damped + applied here)."""
        if m.dwell > 0.0:
            m.dwell -= dt
            m.velocity = (m.velocity[0] * 0.5, m.velocity[1] * 0.5)
            self._apply_velocity(m, dt)
            return None
        if magnitude(m.wander) < 1e-6 or random.random() < 0.05:
            a = random.uniform(0, 2 * math.pi)
            m.wander = (math.cos(a), math.sin(a))
            if random.random() < 0.25:
                m.dwell = random.uniform(0.5, 2.0)   # pause and mill
        a = math.atan2(m.wander[1], m.wander[0]) + random.uniform(-0.4, 0.4)
        m.wander = (math.cos(a), math.sin(a))
        return _add(m.position, _scale(m.wander, speed))

    def _move_members(self, dt: float) -> None:
        """Move each member according to their mood — with organic milling,
        dispersed flight, jittered cluster targets, and exponentially-smoothed
        velocity, so the crowd reads as a believable gathering rather than
        lines streaming onto a handful of points."""
        # Loose formation jitter / arrival radius around a group's sector anchor
        # (riot legibility rework, 2026-06-15) — agitated/rioting clusters hold
        # their seeded sector across the full bounds instead of imploding onto
        # the live group centroid (which collapsed the crowd to the centre).
        formation_r = self._span() * 0.06
        for m in self.members:
            target: Vec2 | None = None
            speed = _SPEED_CALM

            if m.mood in (CrowdMood.CALM, CrowdMood.UNEASY):
                m.exit_target = None
                speed = _SPEED_CALM * (1.2 if m.mood == CrowdMood.UNEASY else 1.0)
                # Mill: occasionally stand still, else drift on a slowly-turning
                # heading (not a fresh random angle every tick, which spins in place).
                target = self._mill_target(m, dt, speed)
                if target is None:
                    continue

            elif m.mood == CrowdMood.AGITATED:
                m.exit_target = None
                speed = _SPEED_AGITATED
                anchor = self._member_anchor(m)
                if anchor is not None:
                    # Loiter once we've reached the sector — fall through to
                    # CALM/UNEASY-style milling so the cluster holds its sector
                    # instead of converging tighter.
                    if distance(m.position, anchor) <= formation_r:
                        target = self._mill_target(m, dt, _SPEED_CALM * 1.2)
                        if target is None:
                            continue
                    else:
                        # Steer to the STABLE anchor (+ loose jitter), NOT the
                        # live group centre, which implodes the crowd.
                        target = (anchor[0] + random.uniform(-formation_r, formation_r),
                                  anchor[1] + random.uniform(-formation_r, formation_r))
                else:
                    gc = self._group_center(m)
                    # Loose cluster AROUND the center (jittered), not a snap onto it.
                    target = (gc[0] + random.uniform(-3.0, 3.0), gc[1] + random.uniform(-3.0, 3.0))

            elif m.mood == CrowdMood.RIOTING:
                m.exit_target = None
                speed = _SPEED_RIOTING
                # Surge toward a NEARBY flashpoint; otherwise hold the group's
                # STABLE sector anchor so scattered groups clash across the area
                # instead of every rioter rushing one central point (tight blob).
                near_ev = None
                if self.events:
                    ev = min(self.events, key=lambda e: distance(m.position, e.position))
                    if distance(m.position, ev.position) < 25.0:
                        near_ev = ev
                if near_ev is not None:
                    target = (near_ev.position[0] + random.uniform(-5.0, 5.0),
                              near_ev.position[1] + random.uniform(-5.0, 5.0))
                else:
                    anchor = self._member_anchor(m)
                    if anchor is not None:
                        if distance(m.position, anchor) <= formation_r:
                            target = self._mill_target(m, dt, _SPEED_CALM * 1.2)
                            if target is None:
                                continue
                        else:
                            target = (anchor[0] + random.uniform(-formation_r, formation_r),
                                      anchor[1] + random.uniform(-formation_r, formation_r))
                    else:
                        gc = self._group_center(m)
                        target = (gc[0] + random.uniform(-4.0, 4.0), gc[1] + random.uniform(-4.0, 4.0))

            elif m.mood == CrowdMood.PANICKED:
                speed = _SPEED_PANICKED
                threat = self._nearest_threat(m)
                if threat and distance(m.position, threat) < 30.0:
                    away = _sub(m.position, threat)
                    if magnitude(away) > 1e-6:
                        target = _add(m.position, _scale(normalize(away), speed))
                if target is None:
                    if m.exit_target is None:
                        m.exit_target = self._pick_exit(m.position)
                    target = m.exit_target

            elif m.mood == CrowdMood.FLEEING:
                speed = _SPEED_FLEEING
                if m.exit_target is None:
                    m.exit_target = self._pick_exit(m.position)
                target = m.exit_target

            if target is None:
                continue

            desired = _sub(target, m.position)
            d = magnitude(desired)
            if d < 1e-6:
                m.velocity = (m.velocity[0] * 0.5, m.velocity[1] * 0.5)
                continue

            desired_vel = _scale(normalize(desired), speed)
            # Separation spreads the crowd into a cluster instead of a stack.
            sep = self._separation_force(m)
            combined = _add(desired_vel, _scale(sep, speed * 1.2))
            combined = truncate(combined, speed)
            # Exponential velocity smoothing -> no per-tick heading snaps/jitter.
            m.velocity = (0.7 * m.velocity[0] + 0.3 * combined[0],
                          0.7 * m.velocity[1] + 0.3 * combined[1])
            self._apply_velocity(m, dt)

    def _compute_group_centers(self) -> dict[str, Vec2]:
        """Precompute center position for each group, biased toward leaders.

        Returns a dict mapping group_id -> center Vec2.  Called once per tick.
        """
        # Accumulate sums per group for leaders and all members separately
        leader_sums: dict[str, tuple[float, float, int]] = {}  # (sum_x, sum_y, count)
        all_sums: dict[str, tuple[float, float, int]] = {}

        for m in self.members:
            gid = m.group_id
            if not gid:
                continue
            px, py = m.position
            prev = all_sums.get(gid, (0.0, 0.0, 0))
            all_sums[gid] = (prev[0] + px, prev[1] + py, prev[2] + 1)
            if m.is_leader:
                prev_l = leader_sums.get(gid, (0.0, 0.0, 0))
                leader_sums[gid] = (prev_l[0] + px, prev_l[1] + py, prev_l[2] + 1)

        centers: dict[str, Vec2] = {}
        for gid, (sx, sy, cnt) in all_sums.items():
            ldata = leader_sums.get(gid)
            if ldata and ldata[2] > 0:
                c = (ldata[0] / ldata[2], ldata[1] / ldata[2])
            elif cnt > 0:
                c = (sx / cnt, sy / cnt)
            else:
                continue
            # Anchor the gather point to the nearest street node so the group
            # forms on a real street/junction, not an arbitrary centroid (which
            # can sit inside a building). One scan per group per tick.
            centers[gid] = self._nearest_street(c)
        return centers

    def _nearest_street(self, pos: Vec2) -> Vec2:
        """Snap a point to the nearest street node, if street data is loaded."""
        nodes = self.street_nodes
        if not nodes:
            return pos
        best = pos
        best_d = float("inf")
        px, py = pos
        for nx, ny in nodes:
            d = (nx - px) * (nx - px) + (ny - py) * (ny - py)
            if d < best_d:
                best_d = d
                best = (nx, ny)
        return best

    def _nearest_open(self, pos: Vec2) -> Vec2 | None:
        """Nearest open (non-building) point to pos, via a radial search outward.
        Used to eject a member that ended up inside a building — the smallest pop
        that clears the footprint, so they reappear just outside the wall rather
        than teleporting across the map. Returns None if nothing open is found."""
        if self.obstacles is None:
            return pos
        px, py = pos
        for r in (1.5, 3.0, 4.5, 6.0, 8.0, 11.0, 15.0, 20.0, 28.0, 40.0):
            for k in range(12):
                ang = k * (math.pi / 6.0)
                cand = self._clamp_to_bounds((px + math.cos(ang) * r, py + math.sin(ang) * r))
                if not self._blocked(cand):
                    return cand
        return None

    def _group_center(self, member: CrowdMember) -> Vec2:
        """Center of this member's group, biased toward leaders.

        Uses precomputed cache from _compute_group_centers().
        """
        cache = getattr(self, "_group_center_cache", None)
        if cache and member.group_id in cache:
            return cache[member.group_id]
        return member.position

    def _nearest_threat(self, member: CrowdMember) -> Vec2 | None:
        """Position of the nearest fear-inducing event."""
        threats = [e for e in self.events if _EVENT_EFFECTS.get(e.event_type, (0, 0))[1] > 0]
        if not threats:
            return None
        nearest = min(threats, key=lambda e: distance(member.position, e.position))
        return nearest.position

    def _separation_force(self, member: CrowdMember) -> Vec2:
        """Repulsion force from nearby crowd members.

        Uses the spatial grid for O(k) neighbor lookups instead of O(n) scans.
        """
        sep_radius = 3.0
        force = (0.0, 0.0)
        count = 0
        neighbors = self._grid.query_radius(member.position, sep_radius)
        for m in neighbors:
            if m is member:
                continue
            d = distance(member.position, m.position)
            if 1e-6 < d < sep_radius:
                diff = normalize(_sub(member.position, m.position))
                # Linear falloff (strongest when close) — spreads the crowd into
                # a believable cluster instead of letting members stack into lines.
                w = (sep_radius - d) / sep_radius
                force = _add(force, _scale(diff, w))
                count += 1
        if count > 0:
            force = _scale(force, 1.0 / count)
            # De-clump dense pockets: averaging by count weakens separation
            # exactly where it's needed most, so amplify when many neighbours
            # are crowded in close — the crowd actively spreads instead of stacking.
            if count > 3:
                force = _scale(force, 1.0 + 0.4 * (count - 3))
        return force

    def _decay_events(self, dt: float) -> None:
        """Age events and remove fully decayed ones."""
        alive: list[CrowdEvent] = []
        for e in self.events:
            e._age += dt
            e.intensity = max(0.0, e.intensity - _EVENT_DECAY_RATE * dt)
            if e.intensity > 0.01:
                alive.append(e)
        self.events = alive

    def _check_escalation(self) -> None:
        """Trigger cascading effects at escalation thresholds."""
        n = len(self.members)
        if n == 0:
            return

        mood_counts = self._mood_counts()
        agitated_plus = mood_counts.get("agitated", 0) + mood_counts.get("rioting", 0)
        rioting_count = mood_counts.get("rioting", 0)

        agitated_ratio = agitated_plus / n
        rioting_ratio = rioting_count / n

        # 30% agitated+rioting → cascade: bump UNEASY → AGITATED
        if agitated_ratio >= _ESCALATION_AGITATED_RATIO:
            for m in self.members:
                if m.mood == CrowdMood.UNEASY:
                    m.aggression = min(1.0, m.aggression + 0.05)
                    m.mood = self._resolve_mood(m)

        # 50% rioting → stampede risk: inject fear
        if rioting_ratio >= _ESCALATION_RIOTING_RATIO:
            for m in self.members:
                if m.mood in (CrowdMood.CALM, CrowdMood.UNEASY):
                    m.fear = min(1.0, m.fear + 0.03)
                    m.mood = self._resolve_mood(m)

    def _mood_counts(self) -> dict[str, int]:
        """Count members by mood name."""
        counts: dict[str, int] = {}
        for m in self.members:
            name = m.mood.name.lower()
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _update_overall_mood(self) -> None:
        """Set overall_mood to the most common mood."""
        if not self.members:
            self.overall_mood = CrowdMood.CALM
            return
        counts = self._mood_counts()
        most_common = max(counts, key=lambda k: counts[k])
        self.overall_mood = CrowdMood[most_common.upper()]

    # -- output --------------------------------------------------------------

    def get_hotspots(self) -> list[dict[str, Any]]:
        """Identify clusters of high aggression for heatmap rendering.

        Uses a simple grid-based clustering: divides the bounds into cells
        and reports cells where average aggression exceeds 0.3.
        """
        x0, y0, x1, y1 = self.bounds
        cell_size = 10.0
        cols = max(1, int((x1 - x0) / cell_size))
        rows = max(1, int((y1 - y0) / cell_size))

        grid_agg: dict[tuple[int, int], list[float]] = {}
        grid_pos: dict[tuple[int, int], list[Vec2]] = {}

        for m in self.members:
            c = int((m.position[0] - x0) / cell_size)
            r = int((m.position[1] - y0) / cell_size)
            c = max(0, min(cols - 1, c))
            r = max(0, min(rows - 1, r))
            key = (c, r)
            grid_agg.setdefault(key, []).append(m.aggression)
            grid_pos.setdefault(key, []).append(m.position)

        hotspots: list[dict[str, Any]] = []
        for key, agg_list in grid_agg.items():
            avg_agg = sum(agg_list) / len(agg_list)
            if avg_agg < 0.3 or len(agg_list) < 3:
                continue
            positions = grid_pos[key]
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
            # Radius proportional to member count in cell
            radius = min(cell_size, 2.0 + len(agg_list) * 0.5)
            hotspots.append({
                "x": round(cx, 2),
                "y": round(cy, 2),
                "radius": round(radius, 2),
                "intensity": round(avg_agg, 3),
            })

        return hotspots

    def to_three_js(self) -> dict[str, Any]:
        """Full frame data for Three.js rendering."""
        members_out = []
        for m in self.members:
            heading = math.atan2(m.velocity[1], m.velocity[0]) if magnitude(m.velocity) > 1e-6 else 0.0
            members_out.append({
                "id": m.member_id,
                "x": round(m.position[0], 2),
                "y": round(m.position[1], 2),
                "mood": m.mood.name.lower(),
                "aggression": round(m.aggression, 3),
                "heading": round(heading, 3),
            })

        events_out = []
        for e in self.events:
            events_out.append({
                "type": e.event_type,
                "x": round(e.position[0], 2),
                "y": round(e.position[1], 2),
                "radius": round(e.radius, 2),
                "intensity": round(e.intensity, 3),
                "age": round(e._age, 3),
            })

        counts = self._mood_counts()
        stats = {
            "total": len(self.members),
            "calm": counts.get("calm", 0),
            "uneasy": counts.get("uneasy", 0),
            "agitated": counts.get("agitated", 0),
            "rioting": counts.get("rioting", 0),
            "panicked": counts.get("panicked", 0),
            "fleeing": counts.get("fleeing", 0),
        }

        return {
            "members": members_out,
            "events": events_out,
            "hotspots": self.get_hotspots(),
            "stats": stats,
        }

    def snapshot(self) -> dict[str, Any]:
        """Full serializable state snapshot."""
        return {
            "time": self._time,
            "bounds": self.bounds,
            "max_members": self.max_members,
            "overall_mood": self.overall_mood.name.lower(),
            "members": [
                {
                    "member_id": m.member_id,
                    "position": m.position,
                    "velocity": m.velocity,
                    "mood": m.mood.name.lower(),
                    "aggression": round(m.aggression, 4),
                    "fear": round(m.fear, 4),
                    "has_weapon": m.has_weapon,
                    "is_leader": m.is_leader,
                    "group_id": m.group_id,
                }
                for m in self.members
            ],
            "events": [
                {
                    "event_type": e.event_type,
                    "position": e.position,
                    "radius": e.radius,
                    "intensity": round(e.intensity, 4),
                    "timestamp": e.timestamp,
                    "age": round(e._age, 4),
                }
                for e in self.events
            ],
            "stats": self._mood_counts(),
        }


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------

def _build_peaceful_protest(bounds: tuple[float, float, float, float]) -> CrowdSimulator:
    """~200 calm protesters gathering in several scattered clusters (organic
    groups, not one blob). Slow escalation if police arrive."""
    sim = CrowdSimulator(bounds, max_members=500)
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    span = min(bounds[2] - bounds[0], bounds[3] - bounds[1])
    off = span * 0.15
    for dx, dy in [(0.0, 0.0), (-off, off * 0.6), (off, off * 0.5),
                   (-off * 0.6, -off), (off * 0.7, -off * 0.7)]:
        sim.spawn_crowd((cx + dx, cy + dy), 40, radius=span * 0.08,
                        mood=CrowdMood.CALM, leader_ratio=0.05)
    return sim


def _build_riot(
    bounds: tuple[float, float, float, float],
    seed: int | None = None,
) -> CrowdSimulator:
    """~200 agitated members seeded into 6 distinct compass SECTORS spread
    across the FULL bounds (riot legibility rework, 2026-06-15) — not a pile of
    units in the centre. Two of the six sectors are the RIOTING cores; the rest
    are AGITATED. Each cluster is a tight knot anchored to its seeded sector
    point; steering holds that sector so the crowd uses the whole map.

    *seed* makes ALL placement deterministic (occupancy tests). When None the
    module RNG is used (live runs vary).
    """
    sim = CrowdSimulator(bounds, max_members=500, seed=seed)
    rng = random.Random(seed)
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    span = min(bounds[2] - bounds[0], bounds[3] - bounds[1])

    # 6 groups, one per compass sector. Place group i at angle 2*pi*i/N + 0.2,
    # alternating radius rings (0.30 / 0.42 of span) so adjacent sectors sit on
    # different rings and don't read as a single annulus. Two of the six are the
    # rioting cores (indices 1 and 4 — opposite sectors, so the clash spans the
    # map rather than hugging the centre).
    n = 6
    riot_sectors = {1, 4}
    sector_points: list[Vec2] = []
    for i in range(n):
        angle = 2 * math.pi * i / n + 0.2
        ring = 0.30 if (i % 2 == 0) else 0.42
        radius = span * ring
        sx = cx + radius * math.cos(angle)
        sy = cy + radius * math.sin(angle)
        sector_points.append(sim._clamp_to_bounds((sx, sy)))

    rioting_core_pos: Vec2 | None = None
    for i, anchor in enumerate(sector_points):
        if i in riot_sectors:
            sim.spawn_crowd(anchor, 30, radius=span * 0.05,
                            mood=CrowdMood.RIOTING, leader_ratio=0.10,
                            rng=rng, anchor=anchor)
            if rioting_core_pos is None:
                rioting_core_pos = anchor
        else:
            sim.spawn_crowd(anchor, 35, radius=span * 0.05,
                            mood=CrowdMood.AGITATED, leader_ratio=0.06,
                            rng=rng, anchor=anchor)

    # Initial thrown object on a rioting core's seeded sector (not dead centre),
    # so the flashpoint is where a riot core actually is.
    throw_pos = rioting_core_pos if rioting_core_pos is not None else (cx, cy)
    sim.inject_event(CrowdEvent(
        event_type="throw_object",
        position=throw_pos,
        radius=10.0,
        intensity=0.6,
        timestamp=0.0,
    ))
    return sim


def _build_stampede(bounds: tuple[float, float, float, float]) -> CrowdSimulator:
    """300 panicked people fleeing a gunshot."""
    sim = CrowdSimulator(bounds, max_members=500)
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    sim.spawn_crowd((cx, cy), 300, radius=35.0, mood=CrowdMood.PANICKED, leader_ratio=0.0)
    sim.inject_event(CrowdEvent(
        event_type="gunshot",
        position=(cx, cy),
        radius=50.0,
        intensity=0.9,
        timestamp=0.0,
    ))
    return sim


def _build_standoff(bounds: tuple[float, float, float, float]) -> CrowdSimulator:
    """100 agitated facing a line, tension rising."""
    sim = CrowdSimulator(bounds, max_members=500)
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    sim.spawn_crowd((cx, cy - 10), 100, radius=20.0, mood=CrowdMood.AGITATED, leader_ratio=0.05)
    return sim


CROWD_SCENARIOS: dict[str, Any] = {
    "peaceful_protest": _build_peaceful_protest,
    "riot": _build_riot,
    "stampede": _build_stampede,
    "standoff": _build_standoff,
}
"""Scenario name -> factory function(bounds) -> CrowdSimulator."""

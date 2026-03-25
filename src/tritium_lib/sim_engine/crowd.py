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

    def __init__(self, bounds: tuple[float, float, float, float], max_members: int = 500) -> None:
        self.bounds = bounds
        self.max_members = max_members
        self.members: list[CrowdMember] = []
        self.events: list[CrowdEvent] = []
        self.overall_mood: CrowdMood = CrowdMood.CALM
        self._time: float = 0.0
        self._exits: list[Vec2] = self._compute_exits()
        self._grid = _CrowdGrid(cell_size=5.0)

    # -- helpers -------------------------------------------------------------

    def _compute_exits(self) -> list[Vec2]:
        """Compute exit points at midpoints of each boundary edge."""
        x0, y0, x1, y1 = self.bounds
        return [
            ((x0 + x1) / 2, y0),  # bottom
            ((x0 + x1) / 2, y1),  # top
            (x0, (y0 + y1) / 2),  # left
            (x1, (y0 + y1) / 2),  # right
        ]

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

    def _nearest_exit(self, pos: Vec2) -> Vec2:
        best = self._exits[0]
        best_d = distance(pos, best)
        for ex in self._exits[1:]:
            d = distance(pos, ex)
            if d < best_d:
                best_d = d
                best = ex
        return best

    # -- public API ----------------------------------------------------------

    def spawn_crowd(
        self,
        center: Vec2,
        count: int,
        radius: float,
        mood: CrowdMood = CrowdMood.CALM,
        leader_ratio: float = 0.05,
    ) -> list[str]:
        """Spawn *count* crowd members around *center* within *radius*.

        Returns list of member IDs created.
        """
        ids: list[str] = []
        available = self.max_members - len(self.members)
        actual = min(count, available)
        group_id = uuid.uuid4().hex[:8]

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
            angle = random.uniform(0, 2 * math.pi)
            r = radius * math.sqrt(random.random())
            px = center[0] + r * math.cos(angle)
            py = center[1] + r * math.sin(angle)
            pos = self._clamp_to_bounds((px, py))

            mid = f"c_{len(self.members)}"
            is_leader = (i / max(actual, 1)) < leader_ratio
            member = CrowdMember(
                member_id=mid,
                position=pos,
                mood=mood,
                aggression=aggression + random.uniform(-0.05, 0.05),
                fear=fear + random.uniform(-0.05, 0.05),
                is_leader=is_leader,
                group_id=group_id,
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

    def tick(self, dt: float) -> None:
        """Advance simulation by *dt* seconds.

        1. Mood propagation among neighbors
        2. Movement based on mood
        3. Collision avoidance + boundary enforcement
        4. Event decay
        5. Escalation checks
        """
        self._time += dt

        # Rebuild spatial grid once per tick for O(1) neighbor queries
        self._grid.rebuild(self.members)

        # Precompute group centers (avoids O(n) scan per member)
        self._group_center_cache = self._compute_group_centers()

        # 1. Mood propagation
        self._propagate_moods(dt)

        # 2 & 3. Movement
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

    def _move_members(self, dt: float) -> None:
        """Move each member according to their mood."""
        for m in self.members:
            target: Vec2 | None = None
            speed = _SPEED_CALM

            if m.mood == CrowdMood.CALM:
                # Random walk
                speed = _SPEED_CALM
                angle = random.uniform(0, 2 * math.pi)
                jitter = (math.cos(angle) * speed * 0.5, math.sin(angle) * speed * 0.5)
                target = _add(m.position, jitter)

            elif m.mood == CrowdMood.UNEASY:
                speed = _SPEED_CALM * 1.2
                angle = random.uniform(0, 2 * math.pi)
                jitter = (math.cos(angle) * speed * 0.3, math.sin(angle) * speed * 0.3)
                target = _add(m.position, jitter)

            elif m.mood == CrowdMood.AGITATED:
                speed = _SPEED_AGITATED
                # Move toward group center / leaders
                target = self._group_center(m)

            elif m.mood == CrowdMood.RIOTING:
                speed = _SPEED_RIOTING
                # Move toward nearest event or random aggressive direction
                if self.events:
                    nearest_evt = min(self.events, key=lambda e: distance(m.position, e.position))
                    target = nearest_evt.position
                else:
                    target = self._group_center(m)

            elif m.mood == CrowdMood.PANICKED:
                speed = _SPEED_PANICKED
                # Move away from nearest threat event
                threat = self._nearest_threat(m)
                if threat:
                    away = _sub(m.position, threat)
                    if magnitude(away) > 1e-6:
                        target = _add(m.position, _scale(normalize(away), speed))
                    else:
                        target = self._nearest_exit(m.position)
                else:
                    target = self._nearest_exit(m.position)

            elif m.mood == CrowdMood.FLEEING:
                speed = _SPEED_FLEEING
                target = self._nearest_exit(m.position)

            if target is None:
                continue

            desired = _sub(target, m.position)
            d = magnitude(desired)
            if d < 1e-6:
                m.velocity = (0.0, 0.0)
                continue

            desired_vel = _scale(normalize(desired), speed)

            # Separation force — avoid bumping into others
            sep = self._separation_force(m)
            combined = _add(desired_vel, _scale(sep, 2.0))
            combined = truncate(combined, speed)

            m.velocity = combined
            new_pos = _add(m.position, _scale(m.velocity, dt))
            m.position = self._clamp_to_bounds(new_pos)

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
                centers[gid] = (ldata[0] / ldata[2], ldata[1] / ldata[2])
            elif cnt > 0:
                centers[gid] = (sx / cnt, sy / cnt)
        return centers

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
        sep_radius = 1.5
        force = (0.0, 0.0)
        count = 0
        neighbors = self._grid.query_radius(member.position, sep_radius)
        for m in neighbors:
            if m is member:
                continue
            d = distance(member.position, m.position)
            if 1e-6 < d < sep_radius:
                diff = normalize(_sub(member.position, m.position))
                force = _add(force, _scale(diff, 1.0 / d))
                count += 1
        if count > 0:
            force = _scale(force, 1.0 / count)
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
    """200 calm protesters with leaders. Slow escalation if police arrive."""
    sim = CrowdSimulator(bounds, max_members=500)
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    sim.spawn_crowd((cx, cy), 200, radius=30.0, mood=CrowdMood.CALM, leader_ratio=0.05)
    return sim


def _build_riot(bounds: tuple[float, float, float, float]) -> CrowdSimulator:
    """150 agitated + 50 rioting, leaders, objects thrown."""
    sim = CrowdSimulator(bounds, max_members=500)
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    sim.spawn_crowd((cx, cy), 150, radius=25.0, mood=CrowdMood.AGITATED, leader_ratio=0.03)
    sim.spawn_crowd((cx + 10, cy), 50, radius=15.0, mood=CrowdMood.RIOTING, leader_ratio=0.06)
    # Initial thrown objects
    sim.inject_event(CrowdEvent(
        event_type="throw_object",
        position=(cx + 5, cy + 5),
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

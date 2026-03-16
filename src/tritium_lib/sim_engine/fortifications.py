# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fortification and engineering module for the Tritium sim engine.

Simulates defensive positions, minefields, barricades, trenches, and
construction.  The EngineeringEngine manages placement, construction
progress, mine arming/triggering, and unit cover.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FortificationType(enum.Enum):
    SANDBAG = "sandbag"
    BUNKER = "bunker"
    TRENCH = "trench"
    BARRICADE = "barricade"
    WATCHTOWER = "watchtower"
    MINEFIELD = "minefield"
    WIRE = "wire"
    CHECKPOINT = "checkpoint"
    FOXHOLE = "foxhole"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Fortification:
    """A defensive structure on the battlefield."""

    fort_id: str
    fort_type: FortificationType
    position: Vec2
    facing: float  # radians — direction it provides cover from

    width: float
    depth: float
    height: float

    health: float
    max_health: float

    cover_value: float       # 0-1 damage reduction for units inside
    concealment: float       # 0-1 visual detection reduction
    capacity: int            # max units that fit inside
    occupants: list[str] = field(default_factory=list)

    build_progress: float = 1.0   # 0-1, 1 = complete
    is_destroyed: bool = False

    # Optional template metadata
    build_time: float = 0.0       # seconds to construct
    detection_bonus: float = 0.0  # multiplier for occupant detection range
    movement_penalty: float = 0.0 # slowdown factor for units passing through
    blocks_vehicles: bool = False

    @property
    def is_complete(self) -> bool:
        return self.build_progress >= 1.0

    @property
    def health_pct(self) -> float:
        if self.max_health <= 0:
            return 0.0
        return self.health / self.max_health

    @property
    def effective_cover(self) -> float:
        """Cover scaled by build progress and health."""
        if self.is_destroyed:
            return 0.0
        return self.cover_value * min(self.build_progress, 1.0) * self.health_pct


@dataclass
class Mine:
    """A placed mine on the battlefield."""

    mine_id: str
    position: Vec2
    mine_type: str  # anti_personnel, anti_vehicle, claymore
    damage: float
    blast_radius: float
    trigger_radius: float
    alliance: str  # won't trigger on friendlies

    is_armed: bool = True
    is_triggered: bool = False

    # Directional mines (claymore)
    facing: float = 0.0       # radians
    cone_angle: float = 360.0 # degrees — 360 = omnidirectional
    weight_threshold: float = 0.0  # kg — anti_vehicle only triggers above this


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

FORTIFICATION_TEMPLATES: dict[str, dict] = {
    "sandbag": {
        "fort_type": FortificationType.SANDBAG,
        "width": 2.0,
        "depth": 0.5,
        "height": 1.0,
        "max_health": 50.0,
        "cover_value": 0.5,
        "concealment": 0.3,
        "capacity": 2,
        "build_time": 10.0,
    },
    "foxhole": {
        "fort_type": FortificationType.FOXHOLE,
        "width": 1.5,
        "depth": 1.5,
        "height": 0.3,
        "max_health": 80.0,
        "cover_value": 0.7,
        "concealment": 0.6,
        "capacity": 2,
        "build_time": 30.0,
    },
    "trench": {
        "fort_type": FortificationType.TRENCH,
        "width": 10.0,
        "depth": 1.5,
        "height": 0.2,
        "max_health": 200.0,
        "cover_value": 0.8,
        "concealment": 0.7,
        "capacity": 6,
        "build_time": 120.0,
    },
    "bunker": {
        "fort_type": FortificationType.BUNKER,
        "width": 6.0,
        "depth": 4.0,
        "height": 2.5,
        "max_health": 500.0,
        "cover_value": 0.95,
        "concealment": 0.9,
        "capacity": 4,
        "build_time": 300.0,
    },
    "watchtower": {
        "fort_type": FortificationType.WATCHTOWER,
        "width": 3.0,
        "depth": 3.0,
        "height": 5.0,
        "max_health": 150.0,
        "cover_value": 0.3,
        "concealment": 0.2,
        "capacity": 2,
        "build_time": 180.0,
        "detection_bonus": 0.5,
    },
    "barricade": {
        "fort_type": FortificationType.BARRICADE,
        "width": 4.0,
        "depth": 1.0,
        "height": 1.2,
        "max_health": 120.0,
        "cover_value": 0.4,
        "concealment": 0.2,
        "capacity": 0,
        "build_time": 15.0,
        "blocks_vehicles": True,
    },
    "wire": {
        "fort_type": FortificationType.WIRE,
        "width": 10.0,
        "depth": 2.0,
        "height": 0.8,
        "max_health": 30.0,
        "cover_value": 0.0,
        "concealment": 0.0,
        "capacity": 0,
        "build_time": 20.0,
        "movement_penalty": 0.7,
    },
    "checkpoint": {
        "fort_type": FortificationType.CHECKPOINT,
        "width": 6.0,
        "depth": 3.0,
        "height": 2.0,
        "max_health": 100.0,
        "cover_value": 0.5,
        "concealment": 0.3,
        "capacity": 4,
        "build_time": 60.0,
        "blocks_vehicles": True,
    },
}

MINE_TEMPLATES: dict[str, dict] = {
    "anti_personnel": {
        "mine_type": "anti_personnel",
        "damage": 40.0,
        "blast_radius": 3.0,
        "trigger_radius": 2.0,
        "cone_angle": 360.0,
        "weight_threshold": 0.0,
    },
    "anti_vehicle": {
        "mine_type": "anti_vehicle",
        "damage": 200.0,
        "blast_radius": 5.0,
        "trigger_radius": 1.0,
        "cone_angle": 360.0,
        "weight_threshold": 500.0,  # kg — only triggers on vehicles
    },
    "claymore": {
        "mine_type": "claymore",
        "damage": 60.0,
        "blast_radius": 5.0,
        "trigger_radius": 3.0,
        "cone_angle": 60.0,
        "weight_threshold": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Engineering Engine
# ---------------------------------------------------------------------------

class EngineeringEngine:
    """Manages fortifications, minefields, and construction on the battlefield."""

    def __init__(self) -> None:
        self.fortifications: dict[str, Fortification] = {}
        self.minefields: list[Mine] = {}
        self.construction_queue: list[dict] = []
        self._pending_effects: list[dict] = []

        # Reset minefields to proper type
        self.minefields = []

    # -- Construction -------------------------------------------------------

    def build(
        self,
        fort_type: str,
        position: Vec2,
        facing: float = 0.0,
        builder_id: Optional[str] = None,
    ) -> Fortification:
        """Start construction of a fortification from a template.

        If *builder_id* is provided, progress starts at 0 and must be
        advanced via ``advance_construction``.  Otherwise the fortification
        is placed fully built.
        """
        template = FORTIFICATION_TEMPLATES.get(fort_type)
        if template is None:
            raise ValueError(f"Unknown fortification type: {fort_type!r}")

        fort_id = f"fort_{uuid.uuid4().hex[:8]}"
        progress = 0.0 if builder_id is not None else 1.0

        fort = Fortification(
            fort_id=fort_id,
            fort_type=template["fort_type"],
            position=position,
            facing=facing,
            width=template["width"],
            depth=template["depth"],
            height=template["height"],
            health=template["max_health"],
            max_health=template["max_health"],
            cover_value=template["cover_value"],
            concealment=template["concealment"],
            capacity=template["capacity"],
            build_progress=progress,
            build_time=template.get("build_time", 0.0),
            detection_bonus=template.get("detection_bonus", 0.0),
            movement_penalty=template.get("movement_penalty", 0.0),
            blocks_vehicles=template.get("blocks_vehicles", False),
        )

        self.fortifications[fort_id] = fort

        if builder_id is not None:
            self.construction_queue.append({
                "fort_id": fort_id,
                "builder_id": builder_id,
                "engineers": 1,
            })

        return fort

    def advance_construction(
        self,
        fort_id: str,
        dt: float,
        engineers: int = 1,
    ) -> float:
        """Advance construction progress.  More engineers = faster.

        Returns the new build_progress value (clamped to 1.0).
        """
        fort = self.fortifications.get(fort_id)
        if fort is None:
            raise KeyError(f"Fortification {fort_id!r} not found")
        if fort.is_destroyed:
            return fort.build_progress
        if fort.build_progress >= 1.0:
            return 1.0

        build_time = fort.build_time if fort.build_time > 0 else 1.0
        rate = engineers / build_time  # progress per second
        fort.build_progress = min(1.0, fort.build_progress + rate * dt)
        return fort.build_progress

    # -- Mines --------------------------------------------------------------

    def place_mine(
        self,
        position: Vec2,
        mine_type: str,
        alliance: str,
        facing: float = 0.0,
    ) -> Mine:
        """Place a mine on the battlefield from a template."""
        template = MINE_TEMPLATES.get(mine_type)
        if template is None:
            raise ValueError(f"Unknown mine type: {mine_type!r}")

        mine = Mine(
            mine_id=f"mine_{uuid.uuid4().hex[:8]}",
            position=position,
            mine_type=template["mine_type"],
            damage=template["damage"],
            blast_radius=template["blast_radius"],
            trigger_radius=template["trigger_radius"],
            alliance=alliance,
            facing=facing,
            cone_angle=template["cone_angle"],
            weight_threshold=template["weight_threshold"],
        )
        self.minefields.append(mine)
        return mine

    def clear_mines(
        self,
        position: Vec2,
        radius: float,
        alliance: str,
    ) -> int:
        """Disarm enemy mines within *radius* of *position*.

        Only clears mines belonging to a different alliance.
        Returns the number of mines disarmed.
        """
        cleared = 0
        remaining: list[Mine] = []
        for mine in self.minefields:
            dist = distance(position, mine.position)
            if dist <= radius and mine.alliance != alliance and mine.is_armed:
                cleared += 1
                # Mine is removed (disarmed and collected)
            else:
                remaining.append(mine)
        self.minefields = remaining
        return cleared

    # -- Occupancy ----------------------------------------------------------

    def enter_fortification(self, fort_id: str, unit_id: str) -> bool:
        """Place a unit inside a fortification.  Returns False if full or
        incomplete."""
        fort = self.fortifications.get(fort_id)
        if fort is None:
            return False
        if fort.is_destroyed:
            return False
        if not fort.is_complete:
            return False
        if len(fort.occupants) >= fort.capacity:
            return False
        if unit_id in fort.occupants:
            return True  # already inside
        fort.occupants.append(unit_id)
        return True

    def exit_fortification(self, fort_id: str, unit_id: str) -> None:
        """Remove a unit from a fortification."""
        fort = self.fortifications.get(fort_id)
        if fort is None:
            return
        if unit_id in fort.occupants:
            fort.occupants.remove(unit_id)

    def get_cover_bonus(self, unit_id: str) -> float:
        """Return cover value if the unit is inside a fortification, else 0."""
        for fort in self.fortifications.values():
            if unit_id in fort.occupants:
                return fort.effective_cover
        return 0.0

    def get_detection_bonus(self, unit_id: str) -> float:
        """Return detection range bonus if unit is in a watchtower-type
        fortification."""
        for fort in self.fortifications.values():
            if unit_id in fort.occupants and fort.detection_bonus > 0:
                return fort.detection_bonus
        return 0.0

    # -- Damage to fortifications ------------------------------------------

    def damage_fortification(self, fort_id: str, damage: float) -> bool:
        """Apply damage to a fortification.  Returns True if destroyed."""
        fort = self.fortifications.get(fort_id)
        if fort is None or fort.is_destroyed:
            return False
        fort.health = max(0.0, fort.health - damage)
        if fort.health <= 0:
            fort.is_destroyed = True
            fort.occupants.clear()
            return True
        return False

    # -- Tick ---------------------------------------------------------------

    def tick(
        self,
        dt: float,
        unit_positions: dict[str, tuple[Vec2, str]],
    ) -> list[dict]:
        """Advance the engineering state by *dt* seconds.

        Args:
            dt: time step in seconds
            unit_positions: mapping of unit_id -> (position, alliance).
                For anti_vehicle mines, provide a third element (weight)
                as ``(pos, alliance, weight)`` but tuple[Vec2, str] is the
                minimum.

        Returns:
            List of event dicts (mine_triggered, construction_complete, etc.)
        """
        events: list[dict] = []

        # --- Mine checks ---
        for mine in self.minefields:
            if not mine.is_armed or mine.is_triggered:
                continue
            for uid, info in unit_positions.items():
                pos = info[0]
                alliance = info[1]
                weight = info[2] if len(info) > 2 else 80.0  # type: ignore[arg-type]

                # Skip friendlies
                if alliance == mine.alliance:
                    continue

                dist = distance(pos, mine.position)
                if dist > mine.trigger_radius:
                    continue

                # Weight threshold (anti-vehicle)
                if mine.weight_threshold > 0 and weight < mine.weight_threshold:
                    continue

                # Directional check (claymore)
                if mine.cone_angle < 360.0:
                    if not _in_cone(mine.position, mine.facing, mine.cone_angle, pos):
                        continue

                # TRIGGERED
                mine.is_triggered = True
                mine.is_armed = False

                # Find all units in blast radius
                casualties: list[dict] = []
                for vid, vinfo in unit_positions.items():
                    vpos = vinfo[0]
                    vdist = distance(mine.position, vpos)
                    if vdist <= mine.blast_radius:
                        # Damage falls off linearly with distance
                        falloff = 1.0 - (vdist / mine.blast_radius) if mine.blast_radius > 0 else 1.0
                        dmg = mine.damage * max(0.0, falloff)
                        if dmg > 0:
                            casualties.append({
                                "unit_id": vid,
                                "damage": round(dmg, 1),
                                "distance": round(vdist, 2),
                            })

                events.append({
                    "type": "mine_triggered",
                    "mine_id": mine.mine_id,
                    "mine_type": mine.mine_type,
                    "position": mine.position,
                    "triggered_by": uid,
                    "casualties": casualties,
                })
                self._pending_effects.append({
                    "type": "mine_explosion",
                    "x": mine.position[0],
                    "y": mine.position[1],
                    "radius": mine.blast_radius,
                })
                break  # mine already triggered, move on

        # --- Construction progress ---
        completed: list[str] = []
        for item in self.construction_queue:
            fid = item["fort_id"]
            fort = self.fortifications.get(fid)
            if fort is None or fort.is_destroyed:
                completed.append(fid)
                continue
            if fort.build_progress >= 1.0:
                completed.append(fid)
                continue

            eng_count = item.get("engineers", 1)
            prev = fort.build_progress
            self.advance_construction(fid, dt, engineers=eng_count)

            if fort.build_progress >= 1.0 and prev < 1.0:
                events.append({
                    "type": "construction_complete",
                    "fort_id": fid,
                    "fort_type": fort.fort_type.value,
                    "position": fort.position,
                })
                completed.append(fid)

        # Remove completed items from queue
        self.construction_queue = [
            item for item in self.construction_queue
            if item["fort_id"] not in completed
        ]

        return events

    # -- Serialization ------------------------------------------------------

    def to_three_js(self) -> dict:
        """Export state for Three.js / frontend rendering."""
        forts_out = []
        for fort in self.fortifications.values():
            forts_out.append({
                "id": fort.fort_id,
                "type": fort.fort_type.value,
                "x": fort.position[0],
                "y": fort.position[1],
                "facing": fort.facing,
                "w": fort.width,
                "d": fort.depth,
                "h": fort.height,
                "health_pct": round(fort.health_pct, 3),
                "occupants": len(fort.occupants),
                "capacity": fort.capacity,
                "build_progress": round(fort.build_progress, 3),
                "is_destroyed": fort.is_destroyed,
                "cover_value": fort.cover_value,
                "concealment": fort.concealment,
            })

        mines_out = []
        for mine in self.minefields:
            if mine.is_triggered:
                continue  # don't render triggered mines
            mines_out.append({
                "id": mine.mine_id,
                "x": mine.position[0],
                "y": mine.position[1],
                "type": mine.mine_type,
                "armed": mine.is_armed,
                "alliance": mine.alliance,
                "trigger_radius": mine.trigger_radius,
            })

        effects = list(self._pending_effects)
        self._pending_effects.clear()

        return {
            "fortifications": forts_out,
            "mines": mines_out,
            "effects": effects,
        }

    # -- Query helpers ------------------------------------------------------

    def get_fortification(self, fort_id: str) -> Optional[Fortification]:
        return self.fortifications.get(fort_id)

    def get_fortifications_near(
        self, position: Vec2, radius: float
    ) -> list[Fortification]:
        """Return all non-destroyed fortifications within radius."""
        results = []
        for fort in self.fortifications.values():
            if fort.is_destroyed:
                continue
            if distance(position, fort.position) <= radius:
                results.append(fort)
        return results

    def get_mines_near(
        self, position: Vec2, radius: float
    ) -> list[Mine]:
        """Return all armed mines within radius."""
        return [
            m for m in self.minefields
            if m.is_armed and not m.is_triggered
            and distance(position, m.position) <= radius
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_cone(
    origin: Vec2,
    facing: float,
    cone_degrees: float,
    target: Vec2,
) -> bool:
    """Check if *target* is within a directional cone from *origin*."""
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    angle_to_target = math.atan2(dy, dx)
    diff = abs(_angle_diff(facing, angle_to_target))
    half_cone = math.radians(cone_degrees / 2.0)
    return diff <= half_cone


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference, result in [-pi, pi]."""
    d = b - a
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d

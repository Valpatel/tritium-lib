# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 -- see LICENSE for details.
"""IED, trap, and asymmetric warfare simulation for the Tritium sim engine.

Simulates improvised weapons, traps, ambushes, guerrilla tactics, and
insurgent behavior.  All spatial math uses Vec2 from the steering module.
"""

from __future__ import annotations

import enum
import math
import random
import uuid
from dataclasses import dataclass, field

from tritium_lib.sim_engine.ai.steering import Vec2, distance

# ---------------------------------------------------------------------------
# Constants / enums
# ---------------------------------------------------------------------------


class TrapType(enum.Enum):
    IED_ROADSIDE = "ied_roadside"
    IED_VEHICLE = "ied_vehicle"
    BOOBY_TRAP = "booby_trap"
    TRIP_WIRE = "trip_wire"
    SNARE = "snare"
    DECOY = "decoy"
    AMBUSH_POINT = "ambush_point"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Trap:
    """A placed trap or IED on the battlefield."""

    trap_id: str
    trap_type: TrapType
    position: Vec2
    facing: float  # radians, relevant for directional traps
    damage: float
    blast_radius: float
    trigger_type: str  # proximity, pressure, remote, timer, tripwire
    trigger_radius: float
    is_armed: bool = True
    is_hidden: bool = True
    detection_difficulty: float = 0.5  # 0-1, higher = harder to find
    placer_alliance: str = "hostile"
    timer_remaining: float | None = None


@dataclass
class GuerrillaCell:
    """An insurgent cell operating in an area."""

    cell_id: str
    members: list[str] = field(default_factory=list)
    base_position: Vec2 = (0.0, 0.0)
    operating_radius: float = 100.0
    weapons_cache: Vec2 | None = None
    morale: float = 0.8
    aggression: float = 0.5
    knowledge: dict = field(default_factory=dict)
    state: str = "hiding"  # hiding, preparing, attacking, fleeing, disbanded

    @property
    def member_count(self) -> int:
        return len(self.members)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TRAP_TEMPLATES: dict[str, dict] = {
    "ied_small": {
        "trap_type": TrapType.IED_ROADSIDE,
        "damage": 60.0,
        "blast_radius": 5.0,
        "trigger_type": "proximity",
        "trigger_radius": 3.0,
        "detection_difficulty": 0.6,
    },
    "ied_large": {
        "trap_type": TrapType.IED_ROADSIDE,
        "damage": 150.0,
        "blast_radius": 10.0,
        "trigger_type": "proximity",
        "trigger_radius": 5.0,
        "detection_difficulty": 0.4,
    },
    "vbied": {
        "trap_type": TrapType.IED_VEHICLE,
        "damage": 300.0,
        "blast_radius": 20.0,
        "trigger_type": "remote",
        "trigger_radius": 8.0,
        "detection_difficulty": 0.3,
    },
    "booby_trap": {
        "trap_type": TrapType.BOOBY_TRAP,
        "damage": 30.0,
        "blast_radius": 2.0,
        "trigger_type": "tripwire",
        "trigger_radius": 1.0,
        "detection_difficulty": 0.8,
    },
    "decoy": {
        "trap_type": TrapType.DECOY,
        "damage": 0.0,
        "blast_radius": 0.0,
        "trigger_type": "proximity",
        "trigger_radius": 5.0,
        "detection_difficulty": 0.2,
    },
    "ambush_point": {
        "trap_type": TrapType.AMBUSH_POINT,
        "damage": 0.0,
        "blast_radius": 0.0,
        "trigger_type": "proximity",
        "trigger_radius": 15.0,
        "detection_difficulty": 0.7,
    },
}

CELL_BEHAVIORS: dict[str, dict] = {
    "hit_and_run": {
        "description": "Attack then flee after 10 seconds",
        "attack_duration": 10.0,
        "flee_after": True,
        "min_morale": 0.3,
    },
    "ied_ambush": {
        "description": "Place IED, wait, detonate on patrol, flee",
        "requires_traps": True,
        "flee_after": True,
        "min_morale": 0.2,
    },
    "sniper_harassment": {
        "description": "Single shots from distance, relocate",
        "attack_duration": 3.0,
        "flee_after": True,
        "min_morale": 0.2,
    },
    "mob_attack": {
        "description": "Overwhelm when outnumbering target",
        "min_morale": 0.8,
        "min_ratio": 2.0,
        "flee_after": False,
    },
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AsymmetricEngine:
    """Manages traps, IEDs, guerrilla cells, and asymmetric warfare logic."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.traps: dict[str, Trap] = {}
        self.cells: dict[str, GuerrillaCell] = {}
        self.detected_traps: set[str] = set()
        self._detonation_effects: list[dict] = []
        self._rng = rng or random.Random()

    # -- Trap placement -----------------------------------------------------

    def place_trap(
        self,
        trap_type: TrapType,
        position: Vec2,
        alliance: str,
        trigger_type: str = "proximity",
        facing: float = 0.0,
        *,
        damage: float = 50.0,
        blast_radius: float = 5.0,
        trigger_radius: float = 3.0,
        detection_difficulty: float = 0.5,
        timer_remaining: float | None = None,
    ) -> Trap:
        """Place a single trap at *position*."""
        trap = Trap(
            trap_id=f"trap_{uuid.uuid4().hex[:8]}",
            trap_type=trap_type,
            position=position,
            facing=facing,
            damage=damage,
            blast_radius=blast_radius,
            trigger_type=trigger_type,
            trigger_radius=trigger_radius,
            detection_difficulty=detection_difficulty,
            placer_alliance=alliance,
            timer_remaining=timer_remaining,
        )
        self.traps[trap.trap_id] = trap
        return trap

    def place_from_template(
        self,
        template_name: str,
        position: Vec2,
        alliance: str,
        facing: float = 0.0,
    ) -> Trap:
        """Place a trap using a predefined template from TRAP_TEMPLATES."""
        tmpl = TRAP_TEMPLATES[template_name]
        return self.place_trap(
            trap_type=tmpl["trap_type"],
            position=position,
            alliance=alliance,
            trigger_type=tmpl["trigger_type"],
            facing=facing,
            damage=tmpl["damage"],
            blast_radius=tmpl["blast_radius"],
            trigger_radius=tmpl["trigger_radius"],
            detection_difficulty=tmpl["detection_difficulty"],
        )

    def place_ied_pattern(
        self,
        road_points: list[Vec2],
        count: int,
        alliance: str,
    ) -> list[Trap]:
        """Scatter *count* IEDs along a route defined by *road_points*.

        IEDs are placed at random positions along route segments with slight
        lateral offset to simulate realistic placement.
        """
        if not road_points or count <= 0:
            return []

        # Build cumulative distances along the route
        segment_lengths: list[float] = []
        for i in range(len(road_points) - 1):
            segment_lengths.append(distance(road_points[i], road_points[i + 1]))
        total_length = sum(segment_lengths)
        if total_length < 1e-6:
            return []

        placed: list[Trap] = []
        for _ in range(count):
            # Pick a random distance along the route
            d = self._rng.uniform(0, total_length)
            accum = 0.0
            for i, seg_len in enumerate(segment_lengths):
                if accum + seg_len >= d or i == len(segment_lengths) - 1:
                    t = (d - accum) / seg_len if seg_len > 0 else 0.0
                    ax, ay = road_points[i]
                    bx, by = road_points[i + 1]
                    px = ax + t * (bx - ax)
                    py = ay + t * (by - ay)
                    # Lateral offset (up to 3 m)
                    offset = self._rng.uniform(-3.0, 3.0)
                    dx = bx - ax
                    dy = by - ay
                    seg_mag = math.hypot(dx, dy)
                    if seg_mag > 0:
                        nx, ny = -dy / seg_mag, dx / seg_mag
                        px += nx * offset
                        py += ny * offset
                    trap = self.place_from_template(
                        self._rng.choice(["ied_small", "ied_large"]),
                        (px, py),
                        alliance,
                        facing=math.atan2(dy, dx) if seg_mag > 0 else 0.0,
                    )
                    placed.append(trap)
                    break
                accum += seg_len
        return placed

    # -- Sweep / detection --------------------------------------------------

    def sweep_area(
        self, sweeper_pos: Vec2, radius: float, skill: float = 0.5
    ) -> list[Trap]:
        """Sweep for traps near *sweeper_pos*.

        Detection chance per trap = skill * (1 - detection_difficulty).
        Returns newly detected traps.
        """
        found: list[Trap] = []
        for trap in self.traps.values():
            if not trap.is_armed or not trap.is_hidden:
                continue
            if trap.trap_id in self.detected_traps:
                continue
            if distance(sweeper_pos, trap.position) > radius:
                continue
            chance = skill * (1.0 - trap.detection_difficulty)
            if self._rng.random() < chance:
                self.detected_traps.add(trap.trap_id)
                found.append(trap)
        return found

    # -- Disarm -------------------------------------------------------------

    def disarm_trap(self, trap_id: str, engineer_skill: float = 0.7) -> bool:
        """Attempt to disarm a trap.

        Success probability equals *engineer_skill*.  On failure the trap
        detonates on the engineer.  Returns True if safely disarmed.
        """
        trap = self.traps.get(trap_id)
        if trap is None or not trap.is_armed:
            return True  # nothing to disarm
        if self._rng.random() < engineer_skill:
            trap.is_armed = False
            return True
        # Failure: trap detonates
        self._trigger_trap(trap, cause="disarm_failure")
        return False

    # -- Remote detonation --------------------------------------------------

    def detonate_remote(self, trap_id: str) -> dict:
        """Remotely trigger a trap.  Returns blast result dict."""
        trap = self.traps.get(trap_id)
        if trap is None or not trap.is_armed:
            return {"detonated": False, "reason": "not_armed_or_missing"}
        return self._trigger_trap(trap, cause="remote")

    # -- Cell management ----------------------------------------------------

    def create_cell(
        self,
        position: Vec2,
        member_count: int,
        operating_radius: float,
        alliance: str = "hostile",
    ) -> GuerrillaCell:
        """Spawn a new guerrilla cell at *position*."""
        cell_id = f"cell_{uuid.uuid4().hex[:8]}"
        members = [f"{cell_id}_m{i}" for i in range(member_count)]
        cell = GuerrillaCell(
            cell_id=cell_id,
            members=members,
            base_position=position,
            operating_radius=operating_radius,
            weapons_cache=position,  # cache at base by default
        )
        self.cells[cell_id] = cell
        return cell

    # -- Tick ---------------------------------------------------------------

    def tick(
        self,
        dt: float,
        unit_positions: dict[str, tuple[Vec2, str]],
        patrol_routes: list[list[Vec2]] | None = None,
    ) -> list[dict]:
        """Advance the simulation by *dt* seconds.

        *unit_positions* maps unit_id -> (position, alliance).
        Returns a list of event dicts that occurred during the tick.
        """
        events: list[dict] = []
        self._detonation_effects.clear()

        # 1. Proximity triggers
        events.extend(self._check_proximity_triggers(unit_positions))

        # 2. Timer triggers
        events.extend(self._check_timer_triggers(dt))

        # 3. Guerrilla cell AI
        events.extend(self._tick_cells(dt, unit_positions, patrol_routes))

        return events

    # -- Three.js export ----------------------------------------------------

    def to_three_js(self, alliance: str) -> dict:
        """Export state for rendering.

        Only traps that are detected (or placed by *alliance*) are visible.
        Hidden enemy traps are invisible -- true fog of war.
        """
        visible_traps: list[dict] = []
        for trap in self.traps.values():
            if not trap.is_armed:
                continue
            is_own = trap.placer_alliance == alliance
            is_detected = trap.trap_id in self.detected_traps
            if not is_own and not is_detected:
                continue
            visible_traps.append({
                "id": trap.trap_id,
                "x": trap.position[0],
                "y": trap.position[1],
                "type": trap.trap_type.value,
                "detected": is_detected,
                "armed": trap.is_armed,
                "radius": trap.blast_radius,
            })

        visible_cells: list[dict] = []
        for cell in self.cells.values():
            if cell.state == "disbanded":
                continue
            visible_cells.append({
                "id": cell.cell_id,
                "x": cell.base_position[0],
                "y": cell.base_position[1],
                "state": cell.state,
                "members": cell.member_count,
            })

        effects = list(self._detonation_effects)

        return {
            "traps": visible_traps,
            "cells": visible_cells,
            "sweep_areas": [],
            "effects": effects,
        }

    # -- Internal -----------------------------------------------------------

    def _trigger_trap(self, trap: Trap, cause: str = "proximity", victim_id: str = "") -> dict:
        """Detonate a trap and produce a blast result."""
        trap.is_armed = False
        trap.is_hidden = False
        result = {
            "detonated": True,
            "trap_id": trap.trap_id,
            "trap_type": trap.trap_type.value,
            "position": trap.position,
            "damage": trap.damage,
            "blast_radius": trap.blast_radius,
            "cause": cause,
            "victim_id": victim_id,
        }
        self._detonation_effects.append({
            "type": "ied_explosion" if "ied" in trap.trap_type.value else "trap_trigger",
            "x": trap.position[0],
            "y": trap.position[1],
            "radius": trap.blast_radius,
        })
        return result

    def _check_proximity_triggers(
        self, unit_positions: dict[str, tuple[Vec2, str]]
    ) -> list[dict]:
        events: list[dict] = []
        for trap in list(self.traps.values()):
            if not trap.is_armed:
                continue
            if trap.trigger_type not in ("proximity", "pressure", "tripwire"):
                continue
            for uid, (pos, alliance) in unit_positions.items():
                if alliance == trap.placer_alliance:
                    continue  # don't trigger on friendlies
                dist = distance(pos, trap.position)
                if dist <= trap.trigger_radius:
                    result = self._trigger_trap(trap, cause=trap.trigger_type, victim_id=uid)
                    events.append({
                        "event": "trap_triggered",
                        **result,
                    })
                    break  # trap consumed
        return events

    def _check_timer_triggers(self, dt: float) -> list[dict]:
        events: list[dict] = []
        for trap in list(self.traps.values()):
            if not trap.is_armed or trap.trigger_type != "timer":
                continue
            if trap.timer_remaining is None:
                continue
            trap.timer_remaining -= dt
            if trap.timer_remaining <= 0:
                result = self._trigger_trap(trap, cause="timer")
                events.append({"event": "trap_triggered", **result})
        return events

    def _tick_cells(
        self,
        dt: float,
        unit_positions: dict[str, tuple[Vec2, str]],
        patrol_routes: list[list[Vec2]] | None,
    ) -> list[dict]:
        events: list[dict] = []
        for cell in list(self.cells.values()):
            if cell.state == "disbanded":
                continue

            # Disband check
            if cell.morale < 0.2 or cell.member_count == 0:
                cell.state = "disbanded"
                events.append({
                    "event": "cell_disbanded",
                    "cell_id": cell.cell_id,
                    "reason": "low_morale" if cell.morale < 0.2 else "no_members",
                })
                continue

            # Gather intel on nearby enemies
            enemies_nearby: list[tuple[str, Vec2]] = []
            for uid, (pos, alliance) in unit_positions.items():
                if alliance == "hostile":
                    continue  # same side
                if distance(pos, cell.base_position) <= cell.operating_radius:
                    enemies_nearby.append((uid, pos))

            # Store patrol knowledge
            if patrol_routes:
                cell.knowledge["patrol_routes"] = patrol_routes

            # State machine
            if cell.state == "hiding":
                if enemies_nearby and cell.aggression > 0.4 and cell.morale > 0.5:
                    cell.state = "preparing"
                    events.append({
                        "event": "cell_state_change",
                        "cell_id": cell.cell_id,
                        "new_state": "preparing",
                    })
            elif cell.state == "preparing":
                if not enemies_nearby:
                    cell.state = "hiding"
                elif cell.morale > 0.6:
                    cell.state = "attacking"
                    events.append({
                        "event": "cell_attack",
                        "cell_id": cell.cell_id,
                        "targets": [uid for uid, _ in enemies_nearby[:3]],
                    })
            elif cell.state == "attacking":
                # Hit-and-run: flee after engaging
                cell.state = "fleeing"
                cell.morale -= 0.05  # combat stress
                events.append({
                    "event": "cell_state_change",
                    "cell_id": cell.cell_id,
                    "new_state": "fleeing",
                })
            elif cell.state == "fleeing":
                cell.state = "hiding"
                cell.morale += 0.02  # survived, small recovery
                events.append({
                    "event": "cell_state_change",
                    "cell_id": cell.cell_id,
                    "new_state": "hiding",
                })

        return events

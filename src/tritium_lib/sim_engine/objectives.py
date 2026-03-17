# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Mission objective system with multi-step objectives, triggers, and dynamic events.

Provides a rich objective engine for combat simulations: chained objectives
with prerequisites, trigger conditions, dynamic events that reshape the
battlefield, and preset mission templates (assault, defense, stealth, rescue).

Usage::

    from tritium_lib.sim_engine.objectives import (
        ObjectiveEngine, MissionObjective, ObjectiveType, ObjectiveStatus,
        TriggerCondition, DynamicEvent, OBJECTIVE_TEMPLATES,
    )

    engine = ObjectiveEngine()
    engine.load_template("assault_chain")
    while not engine.all_required_complete() and not engine.any_required_failed():
        events = engine.tick(0.1, world_state)
        for ev in events:
            print(ev)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ObjectiveType(str, Enum):
    """Kinds of mission objectives."""
    ELIMINATE = "eliminate"
    CAPTURE = "capture"
    DEFEND = "defend"
    ESCORT = "escort"
    EXTRACT = "extract"
    DESTROY = "destroy"
    COLLECT = "collect"
    SURVIVE = "survive"
    STEALTH = "stealth"
    PATROL = "patrol"
    RESCUE = "rescue"
    SABOTAGE = "sabotage"


class ObjectiveStatus(str, Enum):
    """Lifecycle states for an objective."""
    LOCKED = "locked"
    AVAILABLE = "available"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    OPTIONAL = "optional"


# ---------------------------------------------------------------------------
# Trigger conditions
# ---------------------------------------------------------------------------

@dataclass
class TriggerCondition:
    """A single condition that can fire an event or unlock/fail an objective.

    Supported ``condition_type`` values:

    - ``time_elapsed`` — params: ``{"seconds": float}``
    - ``units_killed`` — params: ``{"count": int, "alliance": str | None}``
    - ``position_reached`` — params: ``{"target": Vec2, "radius": float, "unit_id": str | None}``
    - ``structure_destroyed`` — params: ``{"structure_id": str}``
    - ``wave_completed`` — params: ``{"wave": int}``
    - ``objective_completed`` — params: ``{"objective_id": str}``
    - ``unit_health_below`` — params: ``{"unit_id": str, "threshold": float}``
    - ``all_hostiles_dead`` — params: ``{}``
    - ``zone_entered`` — params: ``{"zone_center": Vec2, "zone_radius": float, "alliance": str | None}``
    """

    condition_type: str
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mission objective
# ---------------------------------------------------------------------------

@dataclass
class MissionObjective:
    """A single objective within a mission.

    Objectives can be chained via ``prerequisites`` (other objective IDs that
    must be completed first) and enriched with ``unlock_triggers`` and
    ``fail_triggers``.  ``on_complete`` lists actions that fire when the
    objective transitions to COMPLETED (e.g. spawn reinforcements).
    """

    objective_id: str
    name: str
    description: str
    objective_type: ObjectiveType
    status: ObjectiveStatus = ObjectiveStatus.LOCKED
    target_position: Vec2 | None = None
    target_id: str | None = None
    radius: float = 20.0
    time_limit: float | None = None
    required: bool = True
    points: int = 100
    prerequisites: list[str] = field(default_factory=list)
    unlock_triggers: list[TriggerCondition] = field(default_factory=list)
    fail_triggers: list[TriggerCondition] = field(default_factory=list)
    progress: float = 0.0
    on_complete: list[dict[str, Any]] = field(default_factory=list)

    # Internal bookkeeping
    _elapsed: float = field(default=0.0, repr=False)


# ---------------------------------------------------------------------------
# Dynamic event
# ---------------------------------------------------------------------------

@dataclass
class DynamicEvent:
    """A one-shot (or repeating) event that fires when its trigger is met."""

    event_id: str
    name: str
    trigger: TriggerCondition
    actions: list[dict[str, Any]] = field(default_factory=list)
    one_shot: bool = True
    fired: bool = False


# ---------------------------------------------------------------------------
# Trigger evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_trigger(cond: TriggerCondition, world: dict[str, Any]) -> bool:
    """Return True when *cond* is satisfied by *world*.

    ``world`` is a dict with keys like:
    - ``elapsed`` (float): total sim time in seconds
    - ``units_killed`` (int): total hostile kills
    - ``units_killed_by_alliance`` (dict[str, int])
    - ``units`` (list[dict]): each with id, pos, health, alliance
    - ``structures_destroyed`` (set[str])
    - ``wave`` (int): current wave number
    - ``completed_objectives`` (set[str])
    - ``hostiles_alive`` (int)
    """
    ct = cond.condition_type
    p = cond.params

    if ct == "time_elapsed":
        return world.get("elapsed", 0.0) >= p.get("seconds", math.inf)

    if ct == "units_killed":
        needed = p.get("count", 1)
        alliance = p.get("alliance")
        if alliance:
            return world.get("units_killed_by_alliance", {}).get(alliance, 0) >= needed
        return world.get("units_killed", 0) >= needed

    if ct == "position_reached":
        target: Vec2 = tuple(p.get("target", (0, 0)))  # type: ignore[assignment]
        radius = p.get("radius", 5.0)
        unit_id = p.get("unit_id")
        for u in world.get("units", []):
            if unit_id and u.get("id") != unit_id:
                continue
            pos = u.get("pos")
            if pos and distance(tuple(pos), target) <= radius:  # type: ignore[arg-type]
                return True
        return False

    if ct == "structure_destroyed":
        sid = p.get("structure_id", "")
        return sid in world.get("structures_destroyed", set())

    if ct == "wave_completed":
        return world.get("wave", 0) >= p.get("wave", 1)

    if ct == "objective_completed":
        oid = p.get("objective_id", "")
        return oid in world.get("completed_objectives", set())

    if ct == "unit_health_below":
        uid = p.get("unit_id", "")
        threshold = p.get("threshold", 0.5)
        for u in world.get("units", []):
            if u.get("id") == uid:
                return u.get("health", 1.0) < threshold
        return False

    if ct == "all_hostiles_dead":
        return world.get("hostiles_alive", 1) == 0

    if ct == "zone_entered":
        center: Vec2 = tuple(p.get("zone_center", (0, 0)))  # type: ignore[assignment]
        zr = p.get("zone_radius", 10.0)
        alliance = p.get("alliance")
        for u in world.get("units", []):
            if alliance and u.get("alliance") != alliance:
                continue
            pos = u.get("pos")
            if pos and distance(tuple(pos), center) <= zr:  # type: ignore[arg-type]
                return True
        return False

    return False


def _count_units_in_radius(
    center: Vec2, radius: float, world: dict[str, Any], alliance: str | None = None,
) -> int:
    """Count world units within *radius* of *center*."""
    count = 0
    for u in world.get("units", []):
        if alliance and u.get("alliance") != alliance:
            continue
        pos = u.get("pos")
        if pos and distance(tuple(pos), center) <= radius:  # type: ignore[arg-type]
            count += 1
    return count


# ---------------------------------------------------------------------------
# Objective engine
# ---------------------------------------------------------------------------

class ObjectiveEngine:
    """Manages mission objectives and dynamic events for a simulation."""

    def __init__(self) -> None:
        self.objectives: dict[str, MissionObjective] = {}
        self.events: list[DynamicEvent] = []
        self._total_elapsed: float = 0.0

    # -- mutation ---------------------------------------------------------

    def add_objective(self, obj: MissionObjective) -> None:
        """Register an objective."""
        self.objectives[obj.objective_id] = obj

    def add_event(self, event: DynamicEvent) -> None:
        """Register a dynamic event."""
        self.events.append(event)

    def load_template(self, template_name: str) -> None:
        """Load objectives and events from ``OBJECTIVE_TEMPLATES``."""
        tmpl = OBJECTIVE_TEMPLATES.get(template_name)
        if tmpl is None:
            raise KeyError(f"Unknown template: {template_name}")
        for obj in tmpl.get("objectives", []):
            self.add_objective(copy.deepcopy(obj))
        for ev in tmpl.get("events", []):
            self.add_event(copy.deepcopy(ev))

    # -- queries ----------------------------------------------------------

    def get_active(self) -> list[MissionObjective]:
        """Return objectives with ACTIVE status."""
        return [o for o in self.objectives.values() if o.status == ObjectiveStatus.ACTIVE]

    def get_available(self) -> list[MissionObjective]:
        """Return objectives with AVAILABLE status."""
        return [o for o in self.objectives.values() if o.status == ObjectiveStatus.AVAILABLE]

    def all_required_complete(self) -> bool:
        """True when every required objective is COMPLETED."""
        required = [o for o in self.objectives.values() if o.required]
        if not required:
            return False
        return all(o.status == ObjectiveStatus.COMPLETED for o in required)

    def any_required_failed(self) -> bool:
        """True when any required objective has FAILED."""
        return any(
            o.status == ObjectiveStatus.FAILED
            for o in self.objectives.values()
            if o.required
        )

    def total_points(self) -> int:
        """Sum points from completed objectives."""
        return sum(o.points for o in self.objectives.values() if o.status == ObjectiveStatus.COMPLETED)

    # -- core loop --------------------------------------------------------

    def check_triggers(self, world_state: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate all triggers against *world_state* and return a list of
        change-event dicts describing what happened."""
        changes: list[dict[str, Any]] = []
        completed_ids = {
            oid for oid, o in self.objectives.items() if o.status == ObjectiveStatus.COMPLETED
        }
        # Inject into world so objective_completed triggers can reference it
        world_state.setdefault("completed_objectives", set())
        world_state["completed_objectives"] |= completed_ids

        # --- unlock locked objectives whose prerequisites are met ---
        for obj in self.objectives.values():
            if obj.status != ObjectiveStatus.LOCKED:
                continue
            prereqs_met = all(pid in completed_ids for pid in obj.prerequisites)
            triggers_met = (
                not obj.unlock_triggers
                or any(_evaluate_trigger(t, world_state) for t in obj.unlock_triggers)
            )
            if prereqs_met and triggers_met:
                obj.status = ObjectiveStatus.AVAILABLE
                changes.append({
                    "type": "objective_unlocked",
                    "objective_id": obj.objective_id,
                })

        # --- auto-activate available objectives (first available required,
        #     or all optional) ---
        for obj in self.objectives.values():
            if obj.status == ObjectiveStatus.AVAILABLE:
                obj.status = ObjectiveStatus.ACTIVE
                changes.append({
                    "type": "objective_activated",
                    "objective_id": obj.objective_id,
                })

        # --- check fail triggers on active objectives ---
        for obj in list(self.objectives.values()):
            if obj.status != ObjectiveStatus.ACTIVE:
                continue
            for ft in obj.fail_triggers:
                if _evaluate_trigger(ft, world_state):
                    obj.status = ObjectiveStatus.FAILED
                    changes.append({
                        "type": "objective_failed",
                        "objective_id": obj.objective_id,
                        "reason": ft.condition_type,
                    })
                    break

        # --- fire dynamic events ---
        for ev in self.events:
            if ev.fired and ev.one_shot:
                continue
            if _evaluate_trigger(ev.trigger, world_state):
                ev.fired = True
                changes.append({
                    "type": "event_fired",
                    "event_id": ev.event_id,
                    "name": ev.name,
                    "actions": ev.actions,
                })

        return changes

    def _progress_objectives(self, dt: float, world_state: dict[str, Any]) -> list[dict[str, Any]]:
        """Update progress on active objectives and complete/fail them."""
        changes: list[dict[str, Any]] = []

        for obj in list(self.objectives.values()):
            if obj.status != ObjectiveStatus.ACTIVE:
                continue

            # Track time
            obj._elapsed += dt

            # For SURVIVE objectives, time_limit is the goal duration — reaching
            # it means success, not failure.
            if obj.time_limit is not None and obj._elapsed >= obj.time_limit:
                if obj.objective_type == ObjectiveType.SURVIVE:
                    obj.progress = 1.0
                    obj.status = ObjectiveStatus.COMPLETED
                    changes.append({
                        "type": "objective_completed",
                        "objective_id": obj.objective_id,
                        "points": obj.points,
                        "on_complete": obj.on_complete,
                    })
                    continue
                else:
                    obj.status = ObjectiveStatus.FAILED
                    changes.append({
                        "type": "objective_failed",
                        "objective_id": obj.objective_id,
                        "reason": "time_limit_exceeded",
                    })
                    continue

            # Type-specific progress
            otype = obj.objective_type

            if otype == ObjectiveType.ELIMINATE:
                # Progress = fraction of hostiles killed
                total = world_state.get("total_hostiles", 1)
                killed = world_state.get("units_killed", 0)
                if obj.target_id:
                    # Specific target elimination
                    dead_ids = world_state.get("dead_unit_ids", set())
                    obj.progress = 1.0 if obj.target_id in dead_ids else 0.0
                else:
                    obj.progress = min(1.0, killed / max(total, 1))

            elif otype in (ObjectiveType.CAPTURE, ObjectiveType.DEFEND):
                # Progress based on friendly units in radius
                if obj.target_position:
                    count = _count_units_in_radius(
                        obj.target_position, obj.radius, world_state, alliance="friendly",
                    )
                    enemy_count = _count_units_in_radius(
                        obj.target_position, obj.radius, world_state, alliance="hostile",
                    )
                    if count > 0 and enemy_count == 0:
                        # Capture rate: proportional to unit count, 1 unit = 10s full capture
                        rate = min(count * 0.1, 0.5)
                        obj.progress = min(1.0, obj.progress + rate * dt)
                    elif enemy_count > 0:
                        # Contested — progress decays
                        obj.progress = max(0.0, obj.progress - 0.05 * dt)

            elif otype == ObjectiveType.ESCORT:
                # Escort target must reach destination
                if obj.target_id and obj.target_position:
                    for u in world_state.get("units", []):
                        if u.get("id") == obj.target_id:
                            pos = u.get("pos")
                            if pos:
                                d = distance(tuple(pos), obj.target_position)  # type: ignore[arg-type]
                                max_dist = world_state.get("escort_start_dist", 200.0)
                                obj.progress = max(0.0, 1.0 - d / max(max_dist, 1.0))
                            break

            elif otype == ObjectiveType.EXTRACT:
                # Any friendly unit reaches extraction point
                if obj.target_position:
                    count = _count_units_in_radius(
                        obj.target_position, obj.radius, world_state, alliance="friendly",
                    )
                    obj.progress = 1.0 if count > 0 else obj.progress

            elif otype == ObjectiveType.DESTROY:
                destroyed = world_state.get("structures_destroyed", set())
                if obj.target_id:
                    obj.progress = 1.0 if obj.target_id in destroyed else 0.0
                else:
                    total = world_state.get("total_structures", 1)
                    obj.progress = min(1.0, len(destroyed) / max(total, 1))

            elif otype == ObjectiveType.COLLECT:
                collected = world_state.get("items_collected", 0)
                needed = world_state.get("items_needed", 1)
                obj.progress = min(1.0, collected / max(needed, 1))

            elif otype == ObjectiveType.SURVIVE:
                # Progress = elapsed / time_limit
                if obj.time_limit:
                    obj.progress = min(1.0, obj._elapsed / obj.time_limit)

            elif otype == ObjectiveType.STEALTH:
                # Fail if detected
                detected = world_state.get("player_detected", False)
                if detected:
                    obj.status = ObjectiveStatus.FAILED
                    changes.append({
                        "type": "objective_failed",
                        "objective_id": obj.objective_id,
                        "reason": "detected",
                    })
                    continue
                # Progress toward destination
                if obj.target_position:
                    for u in world_state.get("units", []):
                        if u.get("alliance") == "friendly":
                            pos = u.get("pos")
                            if pos:
                                d = distance(tuple(pos), obj.target_position)  # type: ignore[arg-type]
                                obj.progress = max(obj.progress, 1.0 - d / 200.0)
                            break

            elif otype == ObjectiveType.PATROL:
                # Visit all waypoints
                waypoints = world_state.get("patrol_waypoints", [])
                visited = world_state.get("patrol_visited", set())
                if waypoints:
                    obj.progress = len(visited) / len(waypoints)
                else:
                    obj.progress = 1.0

            elif otype == ObjectiveType.RESCUE:
                # Like escort but target starts captive
                rescued = world_state.get("rescued_units", set())
                if obj.target_id:
                    obj.progress = 1.0 if obj.target_id in rescued else 0.0

            elif otype == ObjectiveType.SABOTAGE:
                sabotaged = world_state.get("sabotaged_targets", set())
                if obj.target_id:
                    obj.progress = 1.0 if obj.target_id in sabotaged else 0.0

            # Check for completion
            if obj.progress >= 1.0 and obj.status == ObjectiveStatus.ACTIVE:
                obj.progress = 1.0
                obj.status = ObjectiveStatus.COMPLETED
                changes.append({
                    "type": "objective_completed",
                    "objective_id": obj.objective_id,
                    "points": obj.points,
                    "on_complete": obj.on_complete,
                })

        return changes

    def tick(self, dt: float, world_state: dict[str, Any]) -> list[dict[str, Any]]:
        """Advance the objective engine by *dt* seconds.

        Returns a list of event dicts describing everything that changed.
        """
        self._total_elapsed += dt
        world_state["elapsed"] = self._total_elapsed

        changes: list[dict[str, Any]] = []
        changes.extend(self.check_triggers(world_state))
        changes.extend(self._progress_objectives(dt, world_state))
        return changes

    # -- serialization for Three.js ---------------------------------------

    def to_three_js(self) -> dict[str, Any]:
        """Export objective state for frontend rendering.

        Returns a dict with objective markers, zone indicators, and progress
        bars suitable for Three.js overlay rendering.
        """
        markers: list[dict[str, Any]] = []
        zones: list[dict[str, Any]] = []
        progress_bars: list[dict[str, Any]] = []

        for obj in self.objectives.values():
            if obj.status in (ObjectiveStatus.LOCKED,):
                continue

            color = _STATUS_COLORS.get(obj.status, "#ffffff")

            if obj.target_position:
                markers.append({
                    "id": obj.objective_id,
                    "name": obj.name,
                    "type": obj.objective_type.value,
                    "status": obj.status.value,
                    "position": list(obj.target_position),
                    "color": color,
                })

            if obj.objective_type in (
                ObjectiveType.CAPTURE, ObjectiveType.DEFEND,
                ObjectiveType.EXTRACT,
            ) and obj.target_position:
                zones.append({
                    "id": obj.objective_id,
                    "center": list(obj.target_position),
                    "radius": obj.radius,
                    "color": color,
                    "progress": obj.progress,
                })

            progress_bars.append({
                "id": obj.objective_id,
                "name": obj.name,
                "progress": obj.progress,
                "status": obj.status.value,
                "required": obj.required,
                "color": color,
            })

        return {
            "markers": markers,
            "zones": zones,
            "progress_bars": progress_bars,
            "all_complete": self.all_required_complete(),
            "any_failed": self.any_required_failed(),
            "total_points": self.total_points(),
        }


_STATUS_COLORS: dict[ObjectiveStatus, str] = {
    ObjectiveStatus.AVAILABLE: "#00f0ff",   # cyan
    ObjectiveStatus.ACTIVE: "#fcee0a",      # yellow
    ObjectiveStatus.COMPLETED: "#05ffa1",   # green
    ObjectiveStatus.FAILED: "#ff2a6d",      # magenta
    ObjectiveStatus.OPTIONAL: "#888888",
}


# ---------------------------------------------------------------------------
# Preset objective templates
# ---------------------------------------------------------------------------

OBJECTIVE_TEMPLATES: dict[str, dict[str, Any]] = {
    "assault_chain": {
        "objectives": [
            MissionObjective(
                objective_id="assault_1",
                name="Eliminate Guards",
                description="Neutralize the outer perimeter guards.",
                objective_type=ObjectiveType.ELIMINATE,
                status=ObjectiveStatus.AVAILABLE,
                points=100,
                prerequisites=[],
                on_complete=[{"action": "spawn_units", "params": {"count": 4, "alliance": "hostile", "location": "building"}}],
            ),
            MissionObjective(
                objective_id="assault_2",
                name="Breach Building",
                description="Advance to the target building and clear it.",
                objective_type=ObjectiveType.CAPTURE,
                target_position=(150.0, 80.0),
                radius=15.0,
                points=150,
                prerequisites=["assault_1"],
            ),
            MissionObjective(
                objective_id="assault_3",
                name="Secure Intel",
                description="Collect intelligence materials from the building.",
                objective_type=ObjectiveType.COLLECT,
                target_position=(150.0, 80.0),
                points=200,
                prerequisites=["assault_2"],
            ),
            MissionObjective(
                objective_id="assault_4",
                name="Extract",
                description="Reach the extraction point with the intel.",
                objective_type=ObjectiveType.EXTRACT,
                target_position=(0.0, 0.0),
                radius=10.0,
                points=150,
                prerequisites=["assault_3"],
            ),
        ],
        "events": [
            DynamicEvent(
                event_id="assault_reinforcements",
                name="Enemy Reinforcements",
                trigger=TriggerCondition("objective_completed", {"objective_id": "assault_2"}),
                actions=[
                    {"action": "spawn_units", "params": {"count": 6, "alliance": "hostile", "type": "rifle"}},
                    {"action": "play_narration", "params": {"text": "Enemy reinforcements incoming!"}},
                ],
            ),
        ],
    },
    "defense_chain": {
        "objectives": [
            MissionObjective(
                objective_id="defense_1",
                name="Fortify Position",
                description="Set up defenses before the assault begins.",
                objective_type=ObjectiveType.DEFEND,
                target_position=(100.0, 100.0),
                radius=30.0,
                status=ObjectiveStatus.AVAILABLE,
                time_limit=60.0,
                points=100,
                prerequisites=[],
            ),
            MissionObjective(
                objective_id="defense_2",
                name="Survive 3 Waves",
                description="Hold your position against three assault waves.",
                objective_type=ObjectiveType.SURVIVE,
                target_position=(100.0, 100.0),
                radius=30.0,
                time_limit=180.0,
                points=300,
                prerequisites=["defense_1"],
                fail_triggers=[
                    TriggerCondition("zone_entered", {"zone_center": (100.0, 100.0), "zone_radius": 30.0, "alliance": "hostile"}),
                ],
            ),
            MissionObjective(
                objective_id="defense_3",
                name="Hold Until Extraction",
                description="Maintain position until extraction arrives.",
                objective_type=ObjectiveType.SURVIVE,
                target_position=(100.0, 100.0),
                time_limit=120.0,
                points=200,
                prerequisites=["defense_2"],
            ),
        ],
        "events": [
            DynamicEvent(
                event_id="defense_wave1",
                name="Wave 1",
                trigger=TriggerCondition("objective_completed", {"objective_id": "defense_1"}),
                actions=[{"action": "spawn_units", "params": {"count": 8, "alliance": "hostile", "wave": 1}}],
            ),
            DynamicEvent(
                event_id="defense_wave2",
                name="Wave 2",
                trigger=TriggerCondition("time_elapsed", {"seconds": 120.0}),
                actions=[{"action": "spawn_units", "params": {"count": 12, "alliance": "hostile", "wave": 2}}],
            ),
            DynamicEvent(
                event_id="defense_wave3",
                name="Wave 3",
                trigger=TriggerCondition("time_elapsed", {"seconds": 240.0}),
                actions=[
                    {"action": "spawn_units", "params": {"count": 16, "alliance": "hostile", "wave": 3}},
                    {"action": "play_narration", "params": {"text": "Final wave incoming! Hold the line!"}},
                ],
            ),
        ],
    },
    "stealth_chain": {
        "objectives": [
            MissionObjective(
                objective_id="stealth_1",
                name="Infiltrate",
                description="Reach the compound without being detected.",
                objective_type=ObjectiveType.STEALTH,
                target_position=(200.0, 150.0),
                status=ObjectiveStatus.AVAILABLE,
                points=150,
                prerequisites=[],
            ),
            MissionObjective(
                objective_id="stealth_2",
                name="Disable Communications",
                description="Destroy the communications array.",
                objective_type=ObjectiveType.SABOTAGE,
                target_id="comms_array",
                points=200,
                prerequisites=["stealth_1"],
            ),
            MissionObjective(
                objective_id="stealth_3",
                name="Plant Charges",
                description="Place demolition charges on the target structure.",
                objective_type=ObjectiveType.SABOTAGE,
                target_id="main_building",
                target_position=(210.0, 160.0),
                points=200,
                prerequisites=["stealth_2"],
            ),
            MissionObjective(
                objective_id="stealth_4",
                name="Exfiltrate Undetected",
                description="Leave the compound without triggering any alarms.",
                objective_type=ObjectiveType.STEALTH,
                target_position=(0.0, 0.0),
                points=250,
                prerequisites=["stealth_3"],
                fail_triggers=[
                    TriggerCondition("units_killed", {"count": 1, "alliance": "hostile"}),
                ],
            ),
        ],
        "events": [
            DynamicEvent(
                event_id="stealth_patrol_shift",
                name="Patrol Shift Change",
                trigger=TriggerCondition("time_elapsed", {"seconds": 90.0}),
                actions=[
                    {"action": "change_patrol_routes", "params": {}},
                    {"action": "play_narration", "params": {"text": "Guards are changing shifts."}},
                ],
            ),
        ],
    },
    "rescue_chain": {
        "objectives": [
            MissionObjective(
                objective_id="rescue_1",
                name="Locate Hostage",
                description="Find the hostage's location.",
                objective_type=ObjectiveType.PATROL,
                status=ObjectiveStatus.AVAILABLE,
                points=100,
                prerequisites=[],
            ),
            MissionObjective(
                objective_id="rescue_2",
                name="Clear Area",
                description="Eliminate all hostiles guarding the hostage.",
                objective_type=ObjectiveType.ELIMINATE,
                target_position=(180.0, 120.0),
                radius=25.0,
                points=200,
                prerequisites=["rescue_1"],
            ),
            MissionObjective(
                objective_id="rescue_3",
                name="Escort to Extraction",
                description="Safely escort the hostage to the extraction point.",
                objective_type=ObjectiveType.ESCORT,
                target_id="hostage_1",
                target_position=(0.0, 0.0),
                radius=10.0,
                points=300,
                prerequisites=["rescue_2"],
                fail_triggers=[
                    TriggerCondition("unit_health_below", {"unit_id": "hostage_1", "threshold": 0.0}),
                ],
            ),
        ],
        "events": [
            DynamicEvent(
                event_id="rescue_alarm",
                name="Alarm Triggered",
                trigger=TriggerCondition("units_killed", {"count": 3, "alliance": "hostile"}),
                actions=[
                    {"action": "spawn_units", "params": {"count": 6, "alliance": "hostile", "type": "qrf"}},
                    {"action": "play_narration", "params": {"text": "Alarm raised! Quick reaction force deployed!"}},
                ],
            ),
        ],
    },
}

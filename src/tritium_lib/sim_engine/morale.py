# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Unit morale and psychology system for the Tritium sim engine.

Simulates morale as a 0-100 value per unit, affected by combat events
(taking damage, seeing allies die, killing enemies, suppression, commander
proximity) with downstream effects on accuracy, speed, and willingness
to fight.

Usage::

    from tritium_lib.sim_engine.morale import MoraleEngine, MoraleEvent

    engine = MoraleEngine()
    engine.register_unit("alpha-1", alliance="friendly")
    engine.apply_event(MoraleEvent(unit_id="alpha-1", event_type="took_damage", magnitude=30.0))
    engine.tick(0.1, unit_positions={"alpha-1": (100.0, 200.0)})
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MoraleEventType(Enum):
    """Types of events that affect morale."""
    TOOK_DAMAGE = "took_damage"
    ALLY_KILLED = "ally_killed"
    ENEMY_KILLED = "enemy_killed"
    SUPPRESSED = "suppressed"
    ENGAGEMENT_WON = "engagement_won"
    ENGAGEMENT_LOST = "engagement_lost"
    COMMANDER_NEARBY = "commander_nearby"
    FLANKED = "flanked"
    AMBUSHED = "ambushed"
    REINFORCEMENTS = "reinforcements"
    RETREAT_ORDER = "retreat_order"


class MoraleState(Enum):
    """Qualitative morale levels affecting behavior."""
    FANATICAL = "fanatical"      # 90-100 — fight to the death
    HIGH = "high"                # 70-89  — aggressive, confident
    STEADY = "steady"            # 50-69  — fights normally
    SHAKEN = "shaken"            # 30-49  — reduced effectiveness
    BROKEN = "broken"            # 10-29  — retreating
    ROUTED = "routed"            # 0-9    — surrender or flee wildly


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MoraleEvent:
    """A discrete event that modifies a unit's morale."""
    unit_id: str
    event_type: MoraleEventType | str
    magnitude: float = 1.0       # multiplier on the base effect
    source_position: Vec2 | None = None


@dataclass
class UnitMorale:
    """Per-unit morale tracking."""
    unit_id: str
    alliance: str
    morale: float = 75.0         # 0-100
    base_morale: float = 75.0    # starting value for recovery target
    is_commander: bool = False
    is_alive: bool = True
    # Modifiers
    accuracy_modifier: float = 1.0    # applied to unit accuracy
    speed_modifier: float = 1.0       # applied to unit speed
    state: MoraleState = MoraleState.HIGH
    # History
    recent_events: list[str] = field(default_factory=list)
    time_since_contact: float = 0.0   # seconds since last combat event


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base morale deltas per event type
_BASE_DELTAS: dict[str, float] = {
    "took_damage": -15.0,
    "ally_killed": -20.0,
    "enemy_killed": 10.0,
    "suppressed": -8.0,
    "engagement_won": 15.0,
    "engagement_lost": -25.0,
    "commander_nearby": 5.0,
    "flanked": -12.0,
    "ambushed": -18.0,
    "reinforcements": 20.0,
    "retreat_order": -10.0,
}

# Commander aura radius in meters
COMMANDER_AURA_RADIUS: float = 50.0

# Morale recovery rate per second when not in contact
RECOVERY_RATE: float = 1.5

# Time (seconds) of no combat events before recovery starts
RECOVERY_DELAY: float = 10.0

# Morale thresholds for state transitions
_STATE_THRESHOLDS: list[tuple[float, MoraleState]] = [
    (90.0, MoraleState.FANATICAL),
    (70.0, MoraleState.HIGH),
    (50.0, MoraleState.STEADY),
    (30.0, MoraleState.SHAKEN),
    (10.0, MoraleState.BROKEN),
    (0.0, MoraleState.ROUTED),
]

# Accuracy and speed modifiers per morale state
_STATE_MODIFIERS: dict[MoraleState, tuple[float, float]] = {
    # (accuracy_mult, speed_mult)
    MoraleState.FANATICAL: (1.1, 1.1),
    MoraleState.HIGH: (1.0, 1.0),
    MoraleState.STEADY: (0.9, 0.95),
    MoraleState.SHAKEN: (0.7, 0.8),
    MoraleState.BROKEN: (0.4, 1.2),   # broken troops run fast
    MoraleState.ROUTED: (0.1, 1.3),   # routed troops flee at max speed
}


# ---------------------------------------------------------------------------
# MoraleEngine
# ---------------------------------------------------------------------------

class MoraleEngine:
    """Manages morale for all units in the simulation.

    Each unit is tracked independently. Events push morale up or down;
    the engine computes derived modifiers (accuracy, speed) and qualitative
    state (steady, shaken, broken...) every tick.
    """

    def __init__(self) -> None:
        self.units: dict[str, UnitMorale] = {}
        self._pending_events: list[MoraleEvent] = []

    # -- registration -------------------------------------------------------

    def register_unit(
        self,
        unit_id: str,
        alliance: str = "friendly",
        starting_morale: float = 75.0,
        is_commander: bool = False,
    ) -> UnitMorale:
        """Register a unit for morale tracking."""
        um = UnitMorale(
            unit_id=unit_id,
            alliance=alliance,
            morale=max(0.0, min(100.0, starting_morale)),
            base_morale=max(0.0, min(100.0, starting_morale)),
            is_commander=is_commander,
        )
        self.units[unit_id] = um
        return um

    def remove_unit(self, unit_id: str) -> None:
        """Remove a unit from morale tracking."""
        self.units.pop(unit_id, None)

    def mark_dead(self, unit_id: str) -> None:
        """Mark a unit as dead — stops morale updates for it."""
        um = self.units.get(unit_id)
        if um is not None:
            um.is_alive = False
            um.morale = 0.0
            um.state = MoraleState.ROUTED

    # -- events -------------------------------------------------------------

    def apply_event(self, event: MoraleEvent) -> None:
        """Queue a morale-affecting event for processing on next tick."""
        self._pending_events.append(event)

    def _process_event(self, event: MoraleEvent) -> None:
        """Apply a single event to the target unit."""
        um = self.units.get(event.unit_id)
        if um is None or not um.is_alive:
            return

        etype = event.event_type
        if isinstance(etype, MoraleEventType):
            etype = etype.value

        base_delta = _BASE_DELTAS.get(etype, 0.0)
        delta = base_delta * event.magnitude
        um.morale = max(0.0, min(100.0, um.morale + delta))
        um.time_since_contact = 0.0

        # Keep last 10 events
        um.recent_events.append(etype)
        if len(um.recent_events) > 10:
            um.recent_events = um.recent_events[-10:]

    # -- queries ------------------------------------------------------------

    def get_morale(self, unit_id: str) -> float:
        """Return current morale for a unit (0-100)."""
        um = self.units.get(unit_id)
        return um.morale if um is not None else 0.0

    def get_state(self, unit_id: str) -> MoraleState:
        """Return the qualitative morale state for a unit."""
        um = self.units.get(unit_id)
        return um.state if um is not None else MoraleState.ROUTED

    def get_accuracy_modifier(self, unit_id: str) -> float:
        """Return the accuracy modifier from morale (0-1.1)."""
        um = self.units.get(unit_id)
        return um.accuracy_modifier if um is not None else 0.0

    def get_speed_modifier(self, unit_id: str) -> float:
        """Return the speed modifier from morale."""
        um = self.units.get(unit_id)
        return um.speed_modifier if um is not None else 1.0

    def should_retreat(self, unit_id: str) -> bool:
        """Return True if the unit's morale dictates retreat."""
        um = self.units.get(unit_id)
        if um is None:
            return False
        return um.state in (MoraleState.BROKEN, MoraleState.ROUTED)

    def should_surrender(self, unit_id: str) -> bool:
        """Return True if the unit is routed (potential surrender)."""
        um = self.units.get(unit_id)
        if um is None:
            return False
        return um.state == MoraleState.ROUTED

    def alliance_average_morale(self, alliance: str) -> float:
        """Return average morale for all alive units in an alliance."""
        values = [
            um.morale for um in self.units.values()
            if um.alliance == alliance and um.is_alive
        ]
        return sum(values) / len(values) if values else 0.0

    # -- tick ---------------------------------------------------------------

    def tick(
        self,
        dt: float,
        unit_positions: dict[str, Vec2] | None = None,
        events: list[MoraleEvent] | None = None,
    ) -> list[dict[str, Any]]:
        """Advance the morale simulation by *dt* seconds.

        Processes queued events, applies commander aura, morale recovery,
        and updates derived modifiers and state.

        Returns a list of state-change notifications (for feeding to
        the UI or scoring system).
        """
        if events:
            self._pending_events.extend(events)

        # 1. Process all pending events
        for event in self._pending_events:
            self._process_event(event)
        self._pending_events.clear()

        if unit_positions is None:
            unit_positions = {}

        notifications: list[dict[str, Any]] = []

        # Find commander positions per alliance
        commander_positions: dict[str, list[Vec2]] = {}
        for uid, um in self.units.items():
            if um.is_commander and um.is_alive and uid in unit_positions:
                cmds = commander_positions.setdefault(um.alliance, [])
                cmds.append(unit_positions[uid])

        for uid, um in self.units.items():
            if not um.is_alive:
                continue

            old_state = um.state

            # 2. Commander proximity bonus
            pos = unit_positions.get(uid)
            if pos is not None and not um.is_commander:
                cmds = commander_positions.get(um.alliance, [])
                for cmd_pos in cmds:
                    d = distance(pos, cmd_pos)
                    if d <= COMMANDER_AURA_RADIUS:
                        # Closer to commander = stronger bonus
                        bonus = 2.0 * (1.0 - d / COMMANDER_AURA_RADIUS) * dt
                        um.morale = min(100.0, um.morale + bonus)
                        break

            # 3. Natural recovery when not in contact
            um.time_since_contact += dt
            if um.time_since_contact >= RECOVERY_DELAY:
                # Recover toward base morale
                if um.morale < um.base_morale:
                    recovery = RECOVERY_RATE * dt
                    um.morale = min(um.base_morale, um.morale + recovery)

            # 4. Update state
            um.state = MoraleState.ROUTED
            for threshold, state in _STATE_THRESHOLDS:
                if um.morale >= threshold:
                    um.state = state
                    break

            # 5. Update modifiers
            acc_mod, spd_mod = _STATE_MODIFIERS.get(
                um.state, (1.0, 1.0),
            )
            um.accuracy_modifier = acc_mod
            um.speed_modifier = spd_mod

            # 6. Emit notification if state changed
            if um.state != old_state:
                notifications.append({
                    "type": "morale_change",
                    "unit_id": uid,
                    "alliance": um.alliance,
                    "old_state": old_state.value,
                    "new_state": um.state.value,
                    "morale": round(um.morale, 1),
                })

        return notifications

    # -- Three.js visualization ---------------------------------------------

    def to_three_js(self) -> dict[str, Any]:
        """Export morale state for Three.js visualization.

        Returns per-unit morale indicators with color-coded auras:
        - Green (#05ffa1) for high morale
        - Yellow (#fcee0a) for steady
        - Orange (#ff8800) for shaken
        - Red (#ff2a6d) for broken/routed
        """
        _STATE_COLORS: dict[MoraleState, str] = {
            MoraleState.FANATICAL: "#05ffa1",
            MoraleState.HIGH: "#05ffa1",
            MoraleState.STEADY: "#fcee0a",
            MoraleState.SHAKEN: "#ff8800",
            MoraleState.BROKEN: "#ff2a6d",
            MoraleState.ROUTED: "#ff0000",
        }

        units_out: list[dict[str, Any]] = []
        for uid, um in self.units.items():
            if not um.is_alive:
                continue
            units_out.append({
                "id": uid,
                "morale": round(um.morale, 1),
                "state": um.state.value,
                "alliance": um.alliance,
                "is_commander": um.is_commander,
                "accuracy_mod": round(um.accuracy_modifier, 2),
                "speed_mod": round(um.speed_modifier, 2),
                "aura_color": _STATE_COLORS.get(um.state, "#ffffff"),
                "aura_intensity": um.morale / 100.0,
                "retreating": um.state in (MoraleState.BROKEN, MoraleState.ROUTED),
            })

        alliance_averages: dict[str, float] = {}
        alliances: set[str] = {um.alliance for um in self.units.values() if um.is_alive}
        for alliance in alliances:
            alliance_averages[alliance] = round(
                self.alliance_average_morale(alliance), 1,
            )

        return {
            "units": units_out,
            "alliance_averages": alliance_averages,
        }

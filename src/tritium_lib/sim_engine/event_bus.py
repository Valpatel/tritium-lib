# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Global event bus and timeline system for the sim engine.

A centralized event system so sim engine modules can communicate without
direct imports. Every module publishes SimEvents to the bus; any module
can subscribe to specific event types or listen to the full stream.

Usage::

    from tritium_lib.sim_engine.event_bus import (
        SimEventBus, SimEvent, SimEventType, EventFilter,
    )

    bus = SimEventBus()
    bus.on(SimEventType.UNIT_SPAWNED, lambda e: print(f"Spawned {e.source_id}"))
    bus.emit(SimEvent(
        event_type=SimEventType.UNIT_SPAWNED,
        tick=1, time=0.0,
        source_id="unit_001",
        data={"template": "infantry"},
    ))
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


# ---------------------------------------------------------------------------
# SimEventType — comprehensive event taxonomy
# ---------------------------------------------------------------------------


class SimEventType(Enum):
    """All event types that can flow through the sim event bus."""

    # --- Units ---
    UNIT_SPAWNED = auto()
    UNIT_KILLED = auto()
    UNIT_DAMAGED = auto()
    UNIT_HEALED = auto()
    UNIT_MOVED = auto()

    # --- Combat ---
    SHOT_FIRED = auto()
    PROJECTILE_IMPACT = auto()
    EXPLOSION = auto()

    # --- Vehicles ---
    VEHICLE_SPAWNED = auto()
    VEHICLE_DESTROYED = auto()
    VEHICLE_BOARDED = auto()

    # --- Structures ---
    STRUCTURE_DAMAGED = auto()
    STRUCTURE_DESTROYED = auto()
    FIRE_STARTED = auto()
    FIRE_EXTINGUISHED = auto()

    # --- Objectives ---
    OBJECTIVE_UPDATED = auto()
    OBJECTIVE_COMPLETED = auto()
    OBJECTIVE_FAILED = auto()

    # --- Waves / Game flow ---
    WAVE_STARTED = auto()
    WAVE_COMPLETED = auto()
    GAME_OVER = auto()

    # --- Detection ---
    DETECTION_NEW = auto()
    DETECTION_LOST = auto()

    # --- Radio / Comms ---
    RADIO_TRANSMITTED = auto()
    RADIO_INTERCEPTED = auto()
    RADIO_JAMMED = auto()

    # --- Supply ---
    SUPPLY_LOW = auto()
    SUPPLY_DEPLETED = auto()
    RESUPPLY = auto()

    # --- Traps / IEDs ---
    MINE_TRIGGERED = auto()
    IED_DETONATED = auto()
    TRAP_DETECTED = auto()

    # --- Medical ---
    CASUALTY = auto()
    EVAC_REQUESTED = auto()
    TREATMENT_COMPLETE = auto()

    # --- Scoring ---
    ACHIEVEMENT_EARNED = auto()
    SCORE_UPDATED = auto()

    # --- Environment ---
    WEATHER_CHANGED = auto()
    TIME_ADVANCED = auto()

    # --- Diplomacy ---
    FACTION_RELATION_CHANGED = auto()
    CEASEFIRE = auto()
    WAR_DECLARED = auto()

    # --- Civilian ---
    CROWD_ESCALATION = auto()
    CIVILIAN_CASUALTY = auto()

    # --- Abilities / Effects ---
    ABILITY_ACTIVATED = auto()
    EFFECT_APPLIED = auto()
    EFFECT_EXPIRED = auto()


# ---------------------------------------------------------------------------
# SimEvent dataclass
# ---------------------------------------------------------------------------


@dataclass
class SimEvent:
    """A single discrete event published to the sim event bus.

    Attributes:
        event_type: The category of event (from SimEventType enum).
        tick: Simulation tick when this event occurred.
        time: Wall-clock-equivalent time in seconds.
        source_id: Entity that caused the event (unit id, vehicle id, etc.).
        target_id: Entity that was affected by the event.
        position: (x, y) world position where the event occurred.
        data: Arbitrary payload with event-specific details.
        priority: 1 = critical (game-changing), 10 = trivial (cosmetic).
    """

    event_type: SimEventType
    tick: int
    time: float
    source_id: str = ""
    target_id: str = ""
    position: tuple[float, float] | None = None
    data: dict = field(default_factory=dict)
    priority: int = 5


# ---------------------------------------------------------------------------
# Callback type
# ---------------------------------------------------------------------------

EventListener = Callable[[SimEvent], None]


# ---------------------------------------------------------------------------
# SimEventBus
# ---------------------------------------------------------------------------


class SimEventBus:
    """Centralized pub/sub event bus for the sim engine.

    Modules subscribe to specific event types (or all events) and the bus
    dispatches published events to matching listeners. An internal log
    stores recent events for timeline queries and HUD rendering.
    """

    def __init__(self, max_log: int = 10_000) -> None:
        self._listeners: dict[SimEventType, list[EventListener]] = defaultdict(list)
        self._global_listeners: list[EventListener] = []
        self._event_log: deque[SimEvent] = deque(maxlen=max_log)
        self._max_log: int = max_log

    # -- Subscribe / unsubscribe ------------------------------------------------

    def on(self, event_type: SimEventType, callback: EventListener) -> None:
        """Subscribe *callback* to a specific event type."""
        self._listeners[event_type].append(callback)

    def on_any(self, callback: EventListener) -> None:
        """Subscribe *callback* to receive every event regardless of type."""
        self._global_listeners.append(callback)

    def off(self, event_type: SimEventType, callback: EventListener) -> None:
        """Unsubscribe *callback* from *event_type*."""
        listeners = self._listeners.get(event_type)
        if listeners:
            self._listeners[event_type] = [
                cb for cb in listeners if cb is not callback
            ]

    def off_any(self, callback: EventListener) -> None:
        """Unsubscribe a global listener."""
        self._global_listeners = [
            cb for cb in self._global_listeners if cb is not callback
        ]

    # -- Publish ----------------------------------------------------------------

    def emit(self, event: SimEvent) -> None:
        """Publish a single event to all matching listeners and log it."""
        self._log_event(event)
        self._dispatch(event)

    def emit_many(self, events: list[SimEvent]) -> None:
        """Publish a batch of events in order."""
        for event in events:
            self.emit(event)

    # -- Query / timeline -------------------------------------------------------

    def get_log(
        self,
        event_type: SimEventType | None = None,
        since_tick: int | None = None,
        limit: int = 100,
    ) -> list[SimEvent]:
        """Return logged events, optionally filtered by type and/or tick.

        Results are returned in chronological order (oldest first), limited
        to the most recent *limit* matching entries.
        """
        result: list[SimEvent] | deque[SimEvent] = self._event_log
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        if since_tick is not None:
            result = [e for e in result if e.tick >= since_tick]
        return list(result)[-limit:]

    def get_timeline(self, start_tick: int, end_tick: int) -> list[SimEvent]:
        """Return all logged events in the tick range [start_tick, end_tick]."""
        return [
            e for e in self._event_log
            if start_tick <= e.tick <= end_tick
        ]

    def clear_log(self) -> None:
        """Discard all logged events."""
        self._event_log.clear()

    def stats(self) -> dict[str, int]:
        """Return event counts per type across the entire log."""
        counts: dict[str, int] = {}
        for event in self._event_log:
            key = event.event_type.name
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_three_js(self, last_n: int = 20) -> list[dict]:
        """Return the *last_n* events formatted for a Three.js / HUD overlay.

        Each dict contains string-safe keys suitable for JSON serialization
        and frontend consumption.
        """
        recent = list(self._event_log)[-last_n:] if last_n else []
        out: list[dict] = []
        for e in recent:
            entry: dict = {
                "type": e.event_type.name,
                "tick": e.tick,
                "time": round(e.time, 3),
                "source": e.source_id,
                "target": e.target_id,
                "priority": e.priority,
            }
            if e.position is not None:
                entry["x"] = round(e.position[0], 2)
                entry["y"] = round(e.position[1], 2)
            if e.data:
                entry["data"] = e.data
            out.append(entry)
        return out

    # -- Internal ---------------------------------------------------------------

    def _log_event(self, event: SimEvent) -> None:
        """Append to the ring-buffer log; deque enforces the cap in O(1)."""
        self._event_log.append(event)

    def _dispatch(self, event: SimEvent) -> None:
        """Invoke all matching listeners for *event*."""
        for cb in self._listeners.get(event.event_type, []):
            try:
                cb(event)
            except Exception:
                pass  # One bad listener must not break the bus
        for cb in self._global_listeners:
            try:
                cb(event)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# EventFilter — stateless filtering utilities
# ---------------------------------------------------------------------------


class EventFilter:
    """Stateless helper to filter lists of SimEvents."""

    @staticmethod
    def filter_by_type(
        events: list[SimEvent],
        types: list[SimEventType] | set[SimEventType],
    ) -> list[SimEvent]:
        """Keep only events whose type is in *types*."""
        type_set = set(types)
        return [e for e in events if e.event_type in type_set]

    @staticmethod
    def filter_by_area(
        events: list[SimEvent],
        center: tuple[float, float],
        radius: float,
    ) -> list[SimEvent]:
        """Keep events within *radius* of *center*. Events without position are excluded."""
        cx, cy = center
        out: list[SimEvent] = []
        for e in events:
            if e.position is None:
                continue
            dx = e.position[0] - cx
            dy = e.position[1] - cy
            if math.sqrt(dx * dx + dy * dy) <= radius:
                out.append(e)
        return out

    @staticmethod
    def filter_by_alliance(
        events: list[SimEvent],
        alliance: str,
    ) -> list[SimEvent]:
        """Keep events whose ``data["alliance"]`` matches *alliance*."""
        return [
            e for e in events
            if e.data.get("alliance") == alliance
        ]

    @staticmethod
    def filter_by_priority(
        events: list[SimEvent],
        max_priority: int,
    ) -> list[SimEvent]:
        """Keep events with priority <= *max_priority* (lower = more important)."""
        return [e for e in events if e.priority <= max_priority]

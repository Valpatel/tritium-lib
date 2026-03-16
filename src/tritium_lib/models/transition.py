# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Transition event model — state change tracking for targets.

Records when a target transitions between states such as:
  - indoor <-> outdoor (positioning method changes)
  - zone entry/exit (geofence crossings)
  - speed changes (stationary -> moving -> fast)
  - classification changes (unknown -> identified)

Used by tritium-sc transition detection plugins and edge firmware
for behavioral analytics and situational awareness.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TransitionType(str, Enum):
    """Categories of target state transitions."""
    INDOOR_OUTDOOR = "indoor_outdoor"
    ZONE_CROSSING = "zone_crossing"
    SPEED_CHANGE = "speed_change"
    CLASSIFICATION_CHANGE = "classification_change"
    ALLIANCE_CHANGE = "alliance_change"
    POSITIONING_METHOD = "positioning_method"
    VISIBILITY = "visibility"  # appeared / disappeared
    CUSTOM = "custom"


@dataclass
class TransitionEvent:
    """A structured record of a target state change.

    Attributes:
        event_id: Unique event identifier.
        target_id: Target that transitioned (e.g. ble_AA:BB:CC, det_person_3).
        transition_type: Category of transition.
        from_state: Previous state label (e.g. 'outdoor', 'stationary', 'unknown').
        to_state: New state label (e.g. 'indoor', 'moving', 'hostile').
        position: Target position at transition time as (lat, lng), if available.
        timestamp: Unix epoch time of the transition.
        confidence: Confidence in the transition detection (0.0-1.0).
        source: What detected the transition (e.g. 'edge_tracker', 'geofence_engine').
        node_id: Edge node that observed the transition, if applicable.
        metadata: Extra context (zone_id, speed, positioning method, etc.).
    """

    target_id: str
    from_state: str
    to_state: str
    transition_type: TransitionType | str = TransitionType.CUSTOM
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    position: tuple[float, float] | None = None
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    source: str = ""
    node_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON transport."""
        d: dict[str, Any] = {
            "event_id": self.event_id,
            "target_id": self.target_id,
            "transition_type": self.transition_type.value if isinstance(self.transition_type, TransitionType) else str(self.transition_type),
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }
        if self.position is not None:
            d["position"] = list(self.position)
        if self.node_id:
            d["node_id"] = self.node_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionEvent:
        """Deserialize from dict."""
        pos = data.get("position")
        tt = data.get("transition_type", "custom")
        try:
            transition_type = TransitionType(tt)
        except ValueError:
            transition_type = tt  # type: ignore[assignment]

        return cls(
            target_id=data["target_id"],
            from_state=data["from_state"],
            to_state=data["to_state"],
            transition_type=transition_type,
            event_id=data.get("event_id", uuid.uuid4().hex[:12]),
            position=tuple(pos) if pos else None,
            timestamp=data.get("timestamp", time.time()),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            node_id=data.get("node_id", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TransitionHistory:
    """Accumulated transition history for a single target.

    Tracks recent transitions and computes summary statistics like
    how often a target enters/exits buildings or changes zones.
    """

    target_id: str
    transitions: list[TransitionEvent] = field(default_factory=list)
    max_history: int = 200

    def add(self, event: TransitionEvent) -> None:
        """Record a transition event."""
        self.transitions.append(event)
        if len(self.transitions) > self.max_history:
            self.transitions = self.transitions[-self.max_history:]

    def count_by_type(self, transition_type: TransitionType | str) -> int:
        """Count transitions of a specific type."""
        tt = transition_type.value if isinstance(transition_type, TransitionType) else str(transition_type)
        return sum(
            1 for t in self.transitions
            if (t.transition_type.value if isinstance(t.transition_type, TransitionType) else str(t.transition_type)) == tt
        )

    def last_transition(self, transition_type: TransitionType | str | None = None) -> TransitionEvent | None:
        """Get the most recent transition, optionally filtered by type."""
        if transition_type is None:
            return self.transitions[-1] if self.transitions else None
        tt = transition_type.value if isinstance(transition_type, TransitionType) else str(transition_type)
        for t in reversed(self.transitions):
            t_type = t.transition_type.value if isinstance(t.transition_type, TransitionType) else str(t.transition_type)
            if t_type == tt:
                return t
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "transition_count": len(self.transitions),
            "transitions": [t.to_dict() for t in self.transitions[-20:]],
        }

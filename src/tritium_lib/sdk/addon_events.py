# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Inter-Addon Event protocol for the Tritium Addon SDK.

Provides AddonEvent (typed event dataclass) and AddonEventBus
(lightweight pub/sub with wildcard pattern matching) so addons
can communicate without tight coupling.

Usage::

    bus = AddonEventBus()
    bus.subscribe("addon:hackrf:*", lambda evt: print(evt))
    bus.publish("hackrf", "signal_detected", {"freq_mhz": 433.92})
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Callable


@dataclass
class AddonEvent:
    """A single inter-addon event."""

    source_addon: str
    event_type: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    device_id: str = ""

    @property
    def topic(self) -> str:
        """MQTT-style topic string for routing."""
        return f"addon:{self.source_addon}:{self.event_type}"

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "source_addon": self.source_addon,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "topic": self.topic,
        }


class AddonEventBus:
    """Lightweight event bus for inter-addon communication.

    Supports exact and wildcard pattern subscriptions using fnmatch-style
    ``*`` wildcards on colon-separated topic segments.

    Can optionally wrap an existing ``EventBus`` instance so addon events
    also flow through the main Tritium event system.
    """

    def __init__(self, event_bus: Any = None) -> None:
        self._event_bus = event_bus
        self._handlers: dict[str, list[Callable[[AddonEvent], None]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish(
        self,
        source_addon: str,
        event_type: str,
        data: dict,
        device_id: str = "",
    ) -> AddonEvent:
        """Create and publish an addon event.

        Returns the created ``AddonEvent``.
        """
        event = AddonEvent(
            source_addon=source_addon,
            event_type=event_type,
            data=data,
            device_id=device_id,
        )

        topic = event.topic

        # Dispatch to local handlers whose pattern matches the topic.
        for pattern, callbacks in list(self._handlers.items()):
            if self._matches(topic, pattern):
                for cb in list(callbacks):
                    cb(event)

        # Forward to the wrapped EventBus if available.
        if self._event_bus is not None and hasattr(self._event_bus, "emit"):
            self._event_bus.emit(topic, event.to_dict())

        return event

    def subscribe(
        self,
        pattern: str,
        callback: Callable[[AddonEvent], None],
    ) -> Callable:
        """Subscribe to addon events matching *pattern*.

        Pattern examples::

            addon:hackrf:signal_detected   — exact match
            addon:hackrf:*                 — all events from hackrf
            addon:*:signal_detected        — signal_detected from any addon
            addon:*:*                      — all addon events

        Returns an unsubscribe callable for convenience.
        """
        self._handlers.setdefault(pattern, []).append(callback)
        return lambda: self.unsubscribe(pattern, callback)

    def unsubscribe(self, pattern: str, callback: Callable) -> bool:
        """Remove *callback* from *pattern*.  Returns ``True`` if found."""
        callbacks = self._handlers.get(pattern)
        if callbacks is None:
            return False
        try:
            callbacks.remove(callback)
            if not callbacks:
                del self._handlers[pattern]
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(topic: str, pattern: str) -> bool:
        """Return ``True`` if *topic* matches *pattern* (fnmatch-style)."""
        return fnmatch(topic, pattern)

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Thread-safe event bus — publish/subscribe for internal events.

Modeled after tritium-sc's engine/comms/event_bus.py but simplified
for shared use across the ecosystem.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    """An event on the bus."""
    topic: str
    data: Any = None
    source: str = ""
    timestamp: float = field(default_factory=time.time)


# Subscriber callback type
Subscriber = Callable[[Event], None]


class EventBus:
    """Thread-safe pub/sub event bus.

    Supports exact topic matching and wildcard subscriptions:
      - "device.heartbeat"  — exact match
      - "device.*"          — single-level wildcard
      - "device.#"          — multi-level wildcard
    """

    def __init__(self):
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: Subscriber) -> None:
        """Subscribe to a topic pattern."""
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Subscriber) -> None:
        """Unsubscribe from a topic pattern."""
        with self._lock:
            if topic in self._subscribers:
                self._subscribers[topic] = [
                    cb for cb in self._subscribers[topic] if cb is not callback
                ]

    def publish(self, topic: str, data: Any = None, source: str = "") -> Event:
        """Publish an event. Returns the event object."""
        event = Event(topic=topic, data=data, source=source)
        callbacks = self._match(topic)
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass  # Don't let one bad subscriber break the bus
        return event

    def _match(self, topic: str) -> list[Subscriber]:
        """Find all subscribers matching a topic."""
        with self._lock:
            matched = []
            parts = topic.split(".")
            for pattern, callbacks in self._subscribers.items():
                if self._pattern_matches(pattern.split("."), parts):
                    matched.extend(callbacks)
            return matched

    @staticmethod
    def _pattern_matches(pattern: list[str], topic: list[str]) -> bool:
        """Check if a pattern matches a topic."""
        if not pattern:
            return not topic
        if pattern[0] == "#":
            return True
        if not topic:
            return False
        if pattern[0] == "*" or pattern[0] == topic[0]:
            return EventBus._pattern_matches(pattern[1:], topic[1:])
        return False

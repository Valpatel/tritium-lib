# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Thread-safe event bus — publish/subscribe for internal events.

Provides both synchronous (EventBus) and async (AsyncEventBus) variants.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class Event:
    """An event on the bus."""
    topic: str
    data: Any = None
    source: str = ""
    timestamp: float = field(default_factory=time.time)


# Subscriber callback types
Subscriber = Callable[[Event], None]
AsyncSubscriber = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Thread-safe pub/sub event bus.

    Supports exact topic matching and wildcard subscriptions:
      - "device.heartbeat"  -- exact match
      - "device.*"          -- single-level wildcard
      - "device.#"          -- multi-level wildcard
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


class AsyncEventBus:
    """Async pub/sub event bus for use within an asyncio event loop.

    Supports the same wildcard patterns as EventBus:
      - "device.heartbeat"  -- exact match
      - "device.*"          -- single-level wildcard
      - "device.#"          -- multi-level wildcard
    """

    def __init__(self):
        self._subscribers: dict[str, list[AsyncSubscriber]] = {}

    def subscribe(self, topic: str, callback: AsyncSubscriber) -> None:
        """Subscribe an async callback to a topic pattern."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: AsyncSubscriber) -> None:
        """Unsubscribe an async callback from a topic pattern."""
        if topic in self._subscribers:
            self._subscribers[topic] = [
                cb for cb in self._subscribers[topic] if cb is not callback
            ]

    async def publish(self, topic: str, data: Any = None, source: str = "") -> Event:
        """Publish an event and await all matching async subscribers."""
        event = Event(topic=topic, data=data, source=source)
        callbacks = self._match(topic)
        for cb in callbacks:
            try:
                await cb(event)
            except Exception:
                pass  # Don't let one bad subscriber break the bus
        return event

    async def publish_concurrent(
        self, topic: str, data: Any = None, source: str = ""
    ) -> Event:
        """Publish an event and run all matching subscribers concurrently."""
        event = Event(topic=topic, data=data, source=source)
        callbacks = self._match(topic)
        if callbacks:
            tasks = []
            for cb in callbacks:
                tasks.append(asyncio.create_task(self._safe_call(cb, event)))
            await asyncio.gather(*tasks)
        return event

    @staticmethod
    async def _safe_call(cb: AsyncSubscriber, event: Event) -> None:
        try:
            await cb(event)
        except Exception:
            pass

    def _match(self, topic: str) -> list[AsyncSubscriber]:
        """Find all subscribers matching a topic."""
        matched = []
        parts = topic.split(".")
        for pattern, callbacks in self._subscribers.items():
            if EventBus._pattern_matches(pattern.split("."), parts):
                matched.extend(callbacks)
        return matched

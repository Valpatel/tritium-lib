# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Thread-safe event bus — publish/subscribe for internal events.

Provides both synchronous (EventBus) and async (AsyncEventBus) variants,
plus QueueEventBus for queue-based pub/sub (used by tritium-sc).

Advanced features (opt-in, backward compatible):
  - Wildcard topic matching: '#' multi-level, '*' single-level
  - Event history: retain last N events per topic for late subscribers
  - Event filtering: subscribers can provide a predicate on event data
  - Priority ordering: higher-priority subscribers are called first
"""

import asyncio
import queue
import threading
import time
from collections import deque
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

# Filter predicate type — takes an Event, returns True to deliver
EventFilter = Callable[[Event], bool]

# Default priority (middle of range)
DEFAULT_PRIORITY = 0


@dataclass
class _SubscriberEntry:
    """Internal record for a subscriber with optional priority and filter."""
    callback: Subscriber
    priority: int = DEFAULT_PRIORITY
    filter_fn: EventFilter | None = None


@dataclass
class _AsyncSubscriberEntry:
    """Internal record for an async subscriber with optional priority and filter."""
    callback: AsyncSubscriber
    priority: int = DEFAULT_PRIORITY
    filter_fn: EventFilter | None = None


class EventBus:
    """Thread-safe pub/sub event bus.

    Supports exact topic matching and wildcard subscriptions:
      - "device.heartbeat"  -- exact match
      - "device.*"          -- single-level wildcard
      - "device.#"          -- multi-level wildcard

    Advanced opt-in features:
      - history_size: retain last N events per topic (default 0 = off)
      - priority: subscribers with higher priority are called first
      - filter_fn: predicate to filter events before delivery
    """

    def __init__(self, history_size: int = 0):
        self._subscribers: dict[str, list[_SubscriberEntry]] = {}
        self._lock = threading.Lock()
        self._history_size = max(0, history_size)
        self._history: dict[str, deque[Event]] = {}
        self._history_lock = threading.Lock()

    # -- History management ---------------------------------------------------

    @property
    def history_size(self) -> int:
        """Maximum number of events retained per topic."""
        return self._history_size

    def get_history(self, topic: str) -> list[Event]:
        """Return a copy of the event history for a specific topic.

        Only returns events whose topic exactly equals the given topic.
        For wildcard retrieval, use get_history_matching().
        """
        with self._history_lock:
            if topic in self._history:
                return list(self._history[topic])
            return []

    def get_history_matching(self, pattern: str) -> list[Event]:
        """Return events from history matching a wildcard pattern.

        Events are returned sorted by timestamp (oldest first).
        """
        pattern_parts = pattern.split(".")
        result: list[Event] = []
        with self._history_lock:
            for topic, events in self._history.items():
                if self._pattern_matches(pattern_parts, topic.split(".")):
                    result.extend(events)
        result.sort(key=lambda e: e.timestamp)
        return result

    def clear_history(self, topic: str | None = None) -> None:
        """Clear event history. If topic is None, clear all history."""
        with self._history_lock:
            if topic is None:
                self._history.clear()
            elif topic in self._history:
                self._history[topic].clear()

    def _record_history(self, event: Event) -> None:
        """Record an event in the history buffer if history is enabled."""
        if self._history_size <= 0:
            return
        with self._history_lock:
            if event.topic not in self._history:
                self._history[event.topic] = deque(maxlen=self._history_size)
            self._history[event.topic].append(event)

    def replay_history(self, topic: str, callback: Subscriber) -> int:
        """Replay stored history for a topic to a callback.

        Returns the number of events replayed. Useful for late subscribers
        that want to catch up on recent events.
        """
        events = self.get_history(topic)
        for event in events:
            try:
                callback(event)
            except Exception:
                pass
        return len(events)

    def replay_history_matching(self, pattern: str, callback: Subscriber) -> int:
        """Replay history for all topics matching a wildcard pattern.

        Returns the number of events replayed.
        """
        events = self.get_history_matching(pattern)
        for event in events:
            try:
                callback(event)
            except Exception:
                pass
        return len(events)

    # -- Subscribe / Unsubscribe ----------------------------------------------

    def subscribe(
        self,
        topic: str,
        callback: Subscriber,
        priority: int = DEFAULT_PRIORITY,
        filter_fn: EventFilter | None = None,
    ) -> None:
        """Subscribe to a topic pattern.

        Args:
            topic: Topic pattern (supports '*' and '#' wildcards).
            callback: Function called with Event when a matching event fires.
            priority: Higher values are called first (default 0).
            filter_fn: Optional predicate — callback only invoked if
                       filter_fn(event) returns True.
        """
        entry = _SubscriberEntry(
            callback=callback, priority=priority, filter_fn=filter_fn
        )
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(entry)

    def unsubscribe(self, topic: str, callback: Subscriber) -> None:
        """Unsubscribe from a topic pattern."""
        with self._lock:
            if topic in self._subscribers:
                self._subscribers[topic] = [
                    e for e in self._subscribers[topic] if e.callback is not callback
                ]

    # -- Publish --------------------------------------------------------------

    def publish(self, topic: str, data: Any = None, source: str = "") -> Event:
        """Publish an event. Returns the event object."""
        event = Event(topic=topic, data=data, source=source)
        self._record_history(event)
        entries = self._match(topic)
        for entry in entries:
            if entry.filter_fn is not None:
                try:
                    if not entry.filter_fn(event):
                        continue
                except Exception:
                    continue
            try:
                entry.callback(event)
            except Exception:
                pass  # Don't let one bad subscriber break the bus
        return event

    # -- Matching -------------------------------------------------------------

    def _match(self, topic: str) -> list[_SubscriberEntry]:
        """Find all subscriber entries matching a topic, sorted by priority (high first)."""
        with self._lock:
            matched: list[_SubscriberEntry] = []
            parts = topic.split(".")
            for pattern, entries in self._subscribers.items():
                if self._pattern_matches(pattern.split("."), parts):
                    matched.extend(entries)
            # Sort by priority descending (highest first), stable sort
            matched.sort(key=lambda e: e.priority, reverse=True)
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

    Advanced opt-in features:
      - history_size: retain last N events per topic (default 0 = off)
      - priority: subscribers with higher priority are called first
      - filter_fn: predicate to filter events before delivery
    """

    def __init__(self, history_size: int = 0):
        self._subscribers: dict[str, list[_AsyncSubscriberEntry]] = {}
        self._history_size = max(0, history_size)
        self._history: dict[str, deque[Event]] = {}

    # -- History management ---------------------------------------------------

    @property
    def history_size(self) -> int:
        """Maximum number of events retained per topic."""
        return self._history_size

    def get_history(self, topic: str) -> list[Event]:
        """Return a copy of the event history for a specific topic."""
        if topic in self._history:
            return list(self._history[topic])
        return []

    def get_history_matching(self, pattern: str) -> list[Event]:
        """Return events from history matching a wildcard pattern."""
        pattern_parts = pattern.split(".")
        result: list[Event] = []
        for topic, events in self._history.items():
            if EventBus._pattern_matches(pattern_parts, topic.split(".")):
                result.extend(events)
        result.sort(key=lambda e: e.timestamp)
        return result

    def clear_history(self, topic: str | None = None) -> None:
        """Clear event history. If topic is None, clear all history."""
        if topic is None:
            self._history.clear()
        elif topic in self._history:
            self._history[topic].clear()

    def _record_history(self, event: Event) -> None:
        """Record an event in the history buffer if history is enabled."""
        if self._history_size <= 0:
            return
        if event.topic not in self._history:
            self._history[event.topic] = deque(maxlen=self._history_size)
        self._history[event.topic].append(event)

    async def replay_history(self, topic: str, callback: AsyncSubscriber) -> int:
        """Replay stored history for a topic to an async callback."""
        events = self.get_history(topic)
        for event in events:
            try:
                await callback(event)
            except Exception:
                pass
        return len(events)

    async def replay_history_matching(
        self, pattern: str, callback: AsyncSubscriber
    ) -> int:
        """Replay history for all topics matching a wildcard pattern."""
        events = self.get_history_matching(pattern)
        for event in events:
            try:
                await callback(event)
            except Exception:
                pass
        return len(events)

    # -- Subscribe / Unsubscribe ----------------------------------------------

    def subscribe(
        self,
        topic: str,
        callback: AsyncSubscriber,
        priority: int = DEFAULT_PRIORITY,
        filter_fn: EventFilter | None = None,
    ) -> None:
        """Subscribe an async callback to a topic pattern.

        Args:
            topic: Topic pattern (supports '*' and '#' wildcards).
            callback: Async function called with Event.
            priority: Higher values are called first (default 0).
            filter_fn: Optional predicate — callback only invoked if
                       filter_fn(event) returns True.
        """
        entry = _AsyncSubscriberEntry(
            callback=callback, priority=priority, filter_fn=filter_fn
        )
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(entry)

    def unsubscribe(self, topic: str, callback: AsyncSubscriber) -> None:
        """Unsubscribe an async callback from a topic pattern."""
        if topic in self._subscribers:
            self._subscribers[topic] = [
                e for e in self._subscribers[topic] if e.callback is not callback
            ]

    # -- Publish --------------------------------------------------------------

    async def publish(self, topic: str, data: Any = None, source: str = "") -> Event:
        """Publish an event and await all matching async subscribers."""
        event = Event(topic=topic, data=data, source=source)
        self._record_history(event)
        entries = self._match(topic)
        for entry in entries:
            if entry.filter_fn is not None:
                try:
                    if not entry.filter_fn(event):
                        continue
                except Exception:
                    continue
            try:
                await entry.callback(event)
            except Exception:
                pass  # Don't let one bad subscriber break the bus
        return event

    async def publish_concurrent(
        self, topic: str, data: Any = None, source: str = ""
    ) -> Event:
        """Publish an event and run all matching subscribers concurrently."""
        event = Event(topic=topic, data=data, source=source)
        self._record_history(event)
        entries = self._match(topic)
        if entries:
            tasks = []
            for entry in entries:
                if entry.filter_fn is not None:
                    try:
                        if not entry.filter_fn(event):
                            continue
                    except Exception:
                        continue
                tasks.append(asyncio.create_task(self._safe_call(entry.callback, event)))
            if tasks:
                await asyncio.gather(*tasks)
        return event

    @staticmethod
    async def _safe_call(cb: AsyncSubscriber, event: Event) -> None:
        try:
            await cb(event)
        except Exception:
            pass

    def _match(self, topic: str) -> list[_AsyncSubscriberEntry]:
        """Find all subscribers matching a topic, sorted by priority (high first)."""
        matched: list[_AsyncSubscriberEntry] = []
        parts = topic.split(".")
        for pattern, entries in self._subscribers.items():
            if EventBus._pattern_matches(pattern.split("."), parts):
                matched.extend(entries)
        matched.sort(key=lambda e: e.priority, reverse=True)
        return matched


class QueueEventBus:
    """Thread-safe queue-based pub/sub for pushing events to subscribers.

    Each subscriber gets a ``queue.Queue`` that receives all published events
    as plain dicts: ``{"type": event_type, "data": ...}``.

    This is the API originally used by tritium-sc's ``engine.comms.event_bus``.
    It is kept here in tritium-lib so that SC can shim to it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []

    def subscribe(self, _filter: str | None = None) -> queue.Queue:
        """Subscribe to events. Returns a Queue that receives all events.

        The optional ``_filter`` parameter is accepted for API compatibility
        but is currently ignored — the caller must filter events itself.
        """
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def publish(self, event_type: str, data: dict | None = None) -> None:
        msg = {"type": event_type}
        if data is not None:
            msg["data"] = data
        with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(msg)
                    except queue.Full:
                        pass

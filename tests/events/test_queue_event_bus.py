# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for QueueEventBus and AsyncEventBus in tritium_lib.events.bus."""

import asyncio
import queue
import threading

import pytest

from tritium_lib.events.bus import AsyncEventBus, Event, EventBus, QueueEventBus


# ---------------------------------------------------------------------------
# QueueEventBus tests
# ---------------------------------------------------------------------------


class TestQueueEventBus:
    def setup_method(self):
        self.bus = QueueEventBus()

    def test_subscribe_returns_queue(self):
        q = self.bus.subscribe()
        assert isinstance(q, queue.Queue)

    def test_publish_delivers_to_subscriber(self):
        q = self.bus.subscribe()
        self.bus.publish("test_event", {"key": "value"})
        msg = q.get_nowait()
        assert msg["type"] == "test_event"
        assert msg["data"] == {"key": "value"}

    def test_publish_without_data(self):
        q = self.bus.subscribe()
        self.bus.publish("simple_event")
        msg = q.get_nowait()
        assert msg["type"] == "simple_event"
        assert "data" not in msg

    def test_multiple_subscribers_receive_same_event(self):
        q1 = self.bus.subscribe()
        q2 = self.bus.subscribe()
        self.bus.publish("shared_event", {"x": 1})
        msg1 = q1.get_nowait()
        msg2 = q2.get_nowait()
        assert msg1["type"] == "shared_event"
        assert msg2["type"] == "shared_event"

    def test_unsubscribe_removes_queue(self):
        q = self.bus.subscribe()
        self.bus.unsubscribe(q)
        self.bus.publish("after_unsub")
        assert q.empty()

    def test_unsubscribe_nonexistent_queue_no_error(self):
        fake_q = queue.Queue()
        self.bus.unsubscribe(fake_q)  # Should not raise

    def test_queue_full_drops_oldest(self):
        """When queue is full, oldest message is dropped to make room."""
        q = self.bus.subscribe()
        # Fill the queue (maxsize=1000)
        for i in range(1000):
            self.bus.publish(f"fill_{i}")
        # Publish one more — should drop oldest
        self.bus.publish("overflow_event")
        # Queue should still have 1000 items
        count = 0
        while not q.empty():
            q.get_nowait()
            count += 1
        assert count == 1000

    def test_thread_safety(self):
        """Publish and subscribe from multiple threads concurrently."""
        q = self.bus.subscribe()
        errors = []

        def publisher():
            try:
                for i in range(100):
                    self.bus.publish(f"thread_event_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=publisher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Should have received 500 events (5 threads * 100 events)
        count = 0
        while not q.empty():
            q.get_nowait()
            count += 1
        assert count == 500

    def test_filter_parameter_accepted(self):
        """The _filter parameter is accepted for API compat but ignored."""
        q = self.bus.subscribe(_filter="some_filter")
        self.bus.publish("event")
        assert not q.empty()

    def test_publish_with_none_data(self):
        q = self.bus.subscribe()
        self.bus.publish("event", data=None)
        msg = q.get_nowait()
        assert msg["type"] == "event"
        assert "data" not in msg


# ---------------------------------------------------------------------------
# AsyncEventBus tests
# ---------------------------------------------------------------------------


class TestAsyncEventBus:
    def test_async_subscribe_and_publish(self):
        bus = AsyncEventBus()
        received = []

        async def handler(e):
            received.append(e)

        bus.subscribe("topic", handler)

        async def run():
            await bus.publish("topic", data="hello")

        asyncio.run(run())
        assert len(received) == 1
        assert received[0].data == "hello"

    def test_async_unsubscribe(self):
        bus = AsyncEventBus()
        received = []

        async def handler(e):
            received.append(e)

        bus.subscribe("test", handler)
        bus.unsubscribe("test", handler)

        async def run():
            await bus.publish("test")

        asyncio.run(run())
        assert len(received) == 0

    def test_async_wildcard_matching(self):
        bus = AsyncEventBus()
        received = []

        async def handler(e):
            received.append(e)

        bus.subscribe("device.#", handler)

        async def run():
            await bus.publish("device.heartbeat")
            await bus.publish("device.sensor.temp")
            await bus.publish("other.thing")

        asyncio.run(run())
        assert len(received) == 2

    def test_async_publish_concurrent(self):
        bus = AsyncEventBus()
        received = []

        async def handler(e):
            received.append(e.topic)

        bus.subscribe("event", handler)

        async def run():
            await bus.publish_concurrent("event", data="concurrent")

        asyncio.run(run())
        assert len(received) == 1

    def test_async_bad_handler_doesnt_break_bus(self):
        bus = AsyncEventBus()
        received = []

        async def bad_handler(e):
            raise RuntimeError("boom")

        async def good_handler(e):
            received.append(e)

        bus.subscribe("topic", bad_handler)
        bus.subscribe("topic", good_handler)

        async def run():
            await bus.publish("topic")

        asyncio.run(run())
        assert len(received) == 1

    def test_async_publish_concurrent_bad_handler(self):
        bus = AsyncEventBus()
        received = []

        async def bad_handler(e):
            raise RuntimeError("fail")

        async def good_handler(e):
            received.append(e)

        bus.subscribe("ev", bad_handler)
        bus.subscribe("ev", good_handler)

        async def run():
            await bus.publish_concurrent("ev")

        asyncio.run(run())
        assert len(received) == 1


# ---------------------------------------------------------------------------
# EventBus additional edge case tests
# ---------------------------------------------------------------------------


class TestEventBusEdgeCases:
    def test_pattern_star_no_match_deeper(self):
        """'device.*' should NOT match 'device.a.b'."""
        bus = EventBus()
        received = []
        bus.subscribe("device.*", lambda e: received.append(e))
        bus.publish("device.a.b")
        assert len(received) == 0

    def test_hash_at_root_matches_everything(self):
        """'#' should match any topic."""
        bus = EventBus()
        received = []
        bus.subscribe("#", lambda e: received.append(e))
        bus.publish("anything.at.all")
        assert len(received) == 1

    def test_exact_match_only(self):
        """Exact match should only match exact topic."""
        bus = EventBus()
        received = []
        bus.subscribe("a.b.c", lambda e: received.append(e))
        bus.publish("a.b.c")
        bus.publish("a.b")
        bus.publish("a.b.c.d")
        assert len(received) == 1

    def test_unsubscribe_nonexistent_topic(self):
        """Unsubscribe from topic with no subscribers doesn't raise."""
        bus = EventBus()
        bus.unsubscribe("no_such_topic", lambda e: None)

    def test_event_dataclass_fields(self):
        """Event has all expected fields."""
        event = Event(topic="test", data=42, source="unit")
        assert event.topic == "test"
        assert event.data == 42
        assert event.source == "unit"
        assert event.timestamp > 0

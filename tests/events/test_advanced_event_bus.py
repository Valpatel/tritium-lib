# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for advanced EventBus features: history, filtering, priority ordering.

Covers EventBus (sync) and AsyncEventBus (async) variants.
"""

import asyncio
import threading
import time

import pytest

from tritium_lib.events.bus import (
    AsyncEventBus,
    Event,
    EventBus,
    EventFilter,
    DEFAULT_PRIORITY,
)


# ===========================================================================
# Event History — EventBus (sync)
# ===========================================================================


class TestEventBusHistory:
    """Tests for event history retention and replay."""

    def test_history_disabled_by_default(self):
        """EventBus with no history_size stores nothing."""
        bus = EventBus()
        assert bus.history_size == 0
        bus.publish("topic.a", data="hello")
        assert bus.get_history("topic.a") == []

    def test_history_enabled_stores_events(self):
        """Events are stored when history_size > 0."""
        bus = EventBus(history_size=10)
        bus.publish("topic.a", data="one")
        bus.publish("topic.a", data="two")
        history = bus.get_history("topic.a")
        assert len(history) == 2
        assert history[0].data == "one"
        assert history[1].data == "two"

    def test_history_per_topic_isolation(self):
        """History is stored per exact topic, not mixed."""
        bus = EventBus(history_size=5)
        bus.publish("device.heartbeat", data="hb1")
        bus.publish("device.sighting", data="sg1")
        bus.publish("device.heartbeat", data="hb2")
        hb = bus.get_history("device.heartbeat")
        sg = bus.get_history("device.sighting")
        assert len(hb) == 2
        assert len(sg) == 1
        assert hb[0].data == "hb1"
        assert sg[0].data == "sg1"

    def test_history_evicts_oldest(self):
        """When history is full, oldest events are dropped."""
        bus = EventBus(history_size=3)
        for i in range(5):
            bus.publish("t", data=i)
        history = bus.get_history("t")
        assert len(history) == 3
        assert [e.data for e in history] == [2, 3, 4]

    def test_history_returns_copy(self):
        """get_history returns a copy; mutating it does not affect the bus."""
        bus = EventBus(history_size=5)
        bus.publish("t", data="a")
        h1 = bus.get_history("t")
        h1.clear()
        h2 = bus.get_history("t")
        assert len(h2) == 1

    def test_history_matching_wildcard(self):
        """get_history_matching retrieves events across matching topics."""
        bus = EventBus(history_size=10)
        bus.publish("device.heartbeat", data="hb")
        bus.publish("device.sighting", data="sg")
        bus.publish("other.thing", data="ot")
        matched = bus.get_history_matching("device.#")
        assert len(matched) == 2
        topics = {e.topic for e in matched}
        assert topics == {"device.heartbeat", "device.sighting"}

    def test_history_matching_star(self):
        """Single-level wildcard matching in history."""
        bus = EventBus(history_size=10)
        bus.publish("a.b", data=1)
        bus.publish("a.c", data=2)
        bus.publish("a.b.c", data=3)
        matched = bus.get_history_matching("a.*")
        assert len(matched) == 2
        assert {e.data for e in matched} == {1, 2}

    def test_history_matching_sorted_by_timestamp(self):
        """get_history_matching returns events sorted by timestamp."""
        bus = EventBus(history_size=10)
        bus.publish("a.x", data="first")
        bus.publish("a.y", data="second")
        bus.publish("a.x", data="third")
        matched = bus.get_history_matching("a.#")
        assert len(matched) == 3
        for i in range(len(matched) - 1):
            assert matched[i].timestamp <= matched[i + 1].timestamp

    def test_clear_history_specific_topic(self):
        """clear_history with topic only clears that topic."""
        bus = EventBus(history_size=5)
        bus.publish("a", data=1)
        bus.publish("b", data=2)
        bus.clear_history("a")
        assert bus.get_history("a") == []
        assert len(bus.get_history("b")) == 1

    def test_clear_history_all(self):
        """clear_history with no args clears everything."""
        bus = EventBus(history_size=5)
        bus.publish("a", data=1)
        bus.publish("b", data=2)
        bus.clear_history()
        assert bus.get_history("a") == []
        assert bus.get_history("b") == []

    def test_replay_history_to_callback(self):
        """replay_history sends stored events to a callback."""
        bus = EventBus(history_size=10)
        bus.publish("t", data="old1")
        bus.publish("t", data="old2")
        received = []
        count = bus.replay_history("t", lambda e: received.append(e.data))
        assert count == 2
        assert received == ["old1", "old2"]

    def test_replay_history_matching_to_callback(self):
        """replay_history_matching sends matched events to a callback."""
        bus = EventBus(history_size=10)
        bus.publish("sensor.temp", data="t1")
        bus.publish("sensor.humidity", data="h1")
        bus.publish("other.thing", data="x")
        received = []
        count = bus.replay_history_matching(
            "sensor.#", lambda e: received.append(e.data)
        )
        assert count == 2
        assert set(received) == {"t1", "h1"}

    def test_replay_history_bad_callback_doesnt_crash(self):
        """replay_history tolerates exceptions in callback."""
        bus = EventBus(history_size=5)
        bus.publish("t", data="a")
        bus.publish("t", data="b")

        def bad_cb(e):
            if e.data == "a":
                raise ValueError("boom")

        # Should not raise
        count = bus.replay_history("t", bad_cb)
        assert count == 2

    def test_replay_empty_history(self):
        """replay_history returns 0 when no history exists."""
        bus = EventBus(history_size=5)
        count = bus.replay_history("nonexistent", lambda e: None)
        assert count == 0

    def test_history_negative_size_treated_as_zero(self):
        """Negative history_size is clamped to 0."""
        bus = EventBus(history_size=-5)
        assert bus.history_size == 0
        bus.publish("t", data="x")
        assert bus.get_history("t") == []

    def test_history_thread_safety(self):
        """History is safe to use from multiple threads."""
        bus = EventBus(history_size=1000)
        errors = []

        def writer(prefix):
            try:
                for i in range(100):
                    bus.publish(f"topic.{prefix}", data=f"{prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        total = sum(len(bus.get_history(f"topic.t{i}")) for i in range(5))
        assert total == 500

    def test_clear_history_nonexistent_topic(self):
        """Clearing history for a topic that was never published is a no-op."""
        bus = EventBus(history_size=5)
        bus.clear_history("never_published")  # Should not raise


# ===========================================================================
# Event Filtering — EventBus (sync)
# ===========================================================================


class TestEventBusFiltering:
    """Tests for subscriber-level event filtering."""

    def test_filter_allows_matching_events(self):
        """Subscriber receives events that pass filter."""
        bus = EventBus()
        received = []
        bus.subscribe(
            "data",
            lambda e: received.append(e.data),
            filter_fn=lambda e: isinstance(e.data, dict) and e.data.get("level") == "critical",
        )
        bus.publish("data", data={"level": "critical", "msg": "disk full"})
        bus.publish("data", data={"level": "info", "msg": "ok"})
        assert len(received) == 1
        assert received[0]["msg"] == "disk full"

    def test_filter_blocks_non_matching_events(self):
        """Subscriber does not receive events that fail filter."""
        bus = EventBus()
        received = []
        bus.subscribe(
            "metrics",
            lambda e: received.append(e),
            filter_fn=lambda e: e.data and e.data.get("value", 0) > 100,
        )
        bus.publish("metrics", data={"value": 50})
        bus.publish("metrics", data={"value": 200})
        bus.publish("metrics", data={"value": 10})
        assert len(received) == 1
        assert received[0].data["value"] == 200

    def test_no_filter_receives_all(self):
        """Subscriber without filter receives all events (backward compat)."""
        bus = EventBus()
        received = []
        bus.subscribe("topic", lambda e: received.append(e))
        bus.publish("topic", data="a")
        bus.publish("topic", data="b")
        assert len(received) == 2

    def test_filter_on_source_field(self):
        """Filter can inspect any Event field, not just data."""
        bus = EventBus()
        received = []
        bus.subscribe(
            "event",
            lambda e: received.append(e),
            filter_fn=lambda e: e.source == "sensor_1",
        )
        bus.publish("event", data="x", source="sensor_1")
        bus.publish("event", data="y", source="sensor_2")
        assert len(received) == 1
        assert received[0].source == "sensor_1"

    def test_filter_exception_treated_as_reject(self):
        """If filter_fn raises, the event is not delivered (safe)."""
        bus = EventBus()
        received = []

        def bad_filter(e):
            raise RuntimeError("filter crash")

        bus.subscribe("t", lambda e: received.append(e), filter_fn=bad_filter)
        bus.publish("t", data="hello")
        assert len(received) == 0

    def test_multiple_subscribers_different_filters(self):
        """Multiple subscribers with different filters get correct events."""
        bus = EventBus()
        evens = []
        odds = []
        bus.subscribe(
            "num",
            lambda e: evens.append(e.data),
            filter_fn=lambda e: e.data % 2 == 0,
        )
        bus.subscribe(
            "num",
            lambda e: odds.append(e.data),
            filter_fn=lambda e: e.data % 2 == 1,
        )
        for i in range(6):
            bus.publish("num", data=i)
        assert evens == [0, 2, 4]
        assert odds == [1, 3, 5]

    def test_filter_with_wildcard_subscription(self):
        """Filter works with wildcard topic subscriptions."""
        bus = EventBus()
        received = []
        bus.subscribe(
            "device.#",
            lambda e: received.append(e),
            filter_fn=lambda e: isinstance(e.data, dict) and e.data.get("battery", 100) < 20,
        )
        bus.publish("device.heartbeat", data={"battery": 15})
        bus.publish("device.heartbeat", data={"battery": 80})
        bus.publish("device.sighting", data={"battery": 5})
        assert len(received) == 2

    def test_filter_none_data(self):
        """Filter that checks data handles None gracefully."""
        bus = EventBus()
        received = []
        bus.subscribe(
            "t",
            lambda e: received.append(e),
            filter_fn=lambda e: e.data is not None,
        )
        bus.publish("t", data=None)
        bus.publish("t", data="something")
        assert len(received) == 1


# ===========================================================================
# Priority Ordering — EventBus (sync)
# ===========================================================================


class TestEventBusPriority:
    """Tests for priority-based subscriber ordering."""

    def test_higher_priority_called_first(self):
        """Subscribers with higher priority are called before lower ones."""
        bus = EventBus()
        call_order = []
        bus.subscribe("t", lambda e: call_order.append("low"), priority=-10)
        bus.subscribe("t", lambda e: call_order.append("high"), priority=10)
        bus.subscribe("t", lambda e: call_order.append("medium"), priority=0)
        bus.publish("t")
        assert call_order == ["high", "medium", "low"]

    def test_default_priority_is_zero(self):
        """Subscribers with no explicit priority get DEFAULT_PRIORITY (0)."""
        bus = EventBus()
        call_order = []
        bus.subscribe("t", lambda e: call_order.append("default"))
        bus.subscribe("t", lambda e: call_order.append("high"), priority=5)
        bus.publish("t")
        assert call_order == ["high", "default"]

    def test_same_priority_preserves_subscribe_order(self):
        """Subscribers at the same priority are called in subscribe order (stable sort)."""
        bus = EventBus()
        call_order = []
        bus.subscribe("t", lambda e: call_order.append("first"), priority=0)
        bus.subscribe("t", lambda e: call_order.append("second"), priority=0)
        bus.subscribe("t", lambda e: call_order.append("third"), priority=0)
        bus.publish("t")
        assert call_order == ["first", "second", "third"]

    def test_priority_across_wildcard_patterns(self):
        """Priority ordering works across different wildcard patterns."""
        bus = EventBus()
        call_order = []
        bus.subscribe("device.#", lambda e: call_order.append("wildcard_low"), priority=-1)
        bus.subscribe("device.heartbeat", lambda e: call_order.append("exact_high"), priority=10)
        bus.subscribe("device.*", lambda e: call_order.append("star_mid"), priority=5)
        bus.publish("device.heartbeat")
        assert call_order == ["exact_high", "star_mid", "wildcard_low"]

    def test_negative_priority(self):
        """Negative priorities are valid and ordered correctly."""
        bus = EventBus()
        call_order = []
        bus.subscribe("t", lambda e: call_order.append("a"), priority=-100)
        bus.subscribe("t", lambda e: call_order.append("b"), priority=-1)
        bus.subscribe("t", lambda e: call_order.append("c"), priority=-50)
        bus.publish("t")
        assert call_order == ["b", "c", "a"]

    def test_priority_with_filter(self):
        """Priority ordering applies after filtering."""
        bus = EventBus()
        call_order = []
        bus.subscribe(
            "t",
            lambda e: call_order.append("low_pass"),
            priority=-5,
            filter_fn=lambda e: True,
        )
        bus.subscribe(
            "t",
            lambda e: call_order.append("high_pass"),
            priority=5,
            filter_fn=lambda e: True,
        )
        bus.subscribe(
            "t",
            lambda e: call_order.append("high_blocked"),
            priority=100,
            filter_fn=lambda e: False,
        )
        bus.publish("t")
        # high_blocked is filtered out despite highest priority
        assert call_order == ["high_pass", "low_pass"]

    def test_unsubscribe_removes_prioritized_subscriber(self):
        """Unsubscribing a prioritized subscriber works correctly."""
        bus = EventBus()
        call_order = []
        high_cb = lambda e: call_order.append("high")
        low_cb = lambda e: call_order.append("low")
        bus.subscribe("t", high_cb, priority=10)
        bus.subscribe("t", low_cb, priority=-10)
        bus.unsubscribe("t", high_cb)
        bus.publish("t")
        assert call_order == ["low"]


# ===========================================================================
# Combined Features — EventBus (sync)
# ===========================================================================


class TestEventBusCombined:
    """Tests that combine history, filtering, and priority."""

    def test_history_plus_filter(self):
        """History stores all events even if some subscribers filter them."""
        bus = EventBus(history_size=10)
        received = []
        bus.subscribe(
            "t",
            lambda e: received.append(e.data),
            filter_fn=lambda e: e.data > 5,
        )
        for i in range(10):
            bus.publish("t", data=i)
        # Subscriber only got filtered events
        assert received == [6, 7, 8, 9]
        # But history has all events
        history = bus.get_history("t")
        assert len(history) == 10

    def test_history_plus_priority(self):
        """History is recorded regardless of subscriber priority."""
        bus = EventBus(history_size=5)
        call_order = []
        bus.subscribe("t", lambda e: call_order.append("high"), priority=10)
        bus.subscribe("t", lambda e: call_order.append("low"), priority=-10)
        bus.publish("t", data="x")
        assert call_order == ["high", "low"]
        assert len(bus.get_history("t")) == 1

    def test_replay_history_with_late_subscriber(self):
        """Late subscriber can replay history and then get live events."""
        bus = EventBus(history_size=5)
        # Publish some events before subscriber joins
        bus.publish("updates", data="v1")
        bus.publish("updates", data="v2")

        # Late subscriber joins and replays
        received = []
        cb = lambda e: received.append(e.data)
        bus.replay_history("updates", cb)
        assert received == ["v1", "v2"]

        # Now subscribe for live events
        bus.subscribe("updates", cb)
        bus.publish("updates", data="v3")
        assert received == ["v1", "v2", "v3"]

    def test_all_three_features_together(self):
        """History + filter + priority all work in combination."""
        bus = EventBus(history_size=20)
        call_order = []

        bus.subscribe(
            "sensor.#",
            lambda e: call_order.append(("critical", e.data)),
            priority=100,
            filter_fn=lambda e: isinstance(e.data, dict) and e.data.get("level") == "critical",
        )
        bus.subscribe(
            "sensor.#",
            lambda e: call_order.append(("all", e.data)),
            priority=0,
        )

        bus.publish("sensor.temp", data={"level": "info", "val": 22})
        bus.publish("sensor.temp", data={"level": "critical", "val": 99})

        # Critical subscriber only got the critical event
        critical_calls = [c for c in call_order if c[0] == "critical"]
        assert len(critical_calls) == 1
        assert critical_calls[0][1]["val"] == 99

        # "all" subscriber got both
        all_calls = [c for c in call_order if c[0] == "all"]
        assert len(all_calls) == 2

        # History has both events
        assert len(bus.get_history("sensor.temp")) == 2

        # Order: for the critical event, "critical" (pri 100) before "all" (pri 0)
        # Find the calls for the critical event
        critical_event_calls = [
            c for c in call_order
            if isinstance(c[1], dict) and c[1].get("level") == "critical"
        ]
        assert critical_event_calls[0][0] == "critical"
        assert critical_event_calls[1][0] == "all"


# ===========================================================================
# Event History — AsyncEventBus
# ===========================================================================


class TestAsyncEventBusHistory:
    """Tests for async event bus history."""

    def test_async_history_disabled_by_default(self):
        bus = AsyncEventBus()
        assert bus.history_size == 0

    def test_async_history_stores_events(self):
        bus = AsyncEventBus(history_size=5)

        async def run():
            await bus.publish("t", data="a")
            await bus.publish("t", data="b")

        asyncio.run(run())
        history = bus.get_history("t")
        assert len(history) == 2
        assert history[0].data == "a"

    def test_async_history_eviction(self):
        bus = AsyncEventBus(history_size=2)

        async def run():
            for i in range(5):
                await bus.publish("t", data=i)

        asyncio.run(run())
        history = bus.get_history("t")
        assert len(history) == 2
        assert [e.data for e in history] == [3, 4]

    def test_async_get_history_matching(self):
        bus = AsyncEventBus(history_size=10)

        async def run():
            await bus.publish("a.x", data=1)
            await bus.publish("a.y", data=2)
            await bus.publish("b.z", data=3)

        asyncio.run(run())
        matched = bus.get_history_matching("a.#")
        assert len(matched) == 2

    def test_async_replay_history(self):
        bus = AsyncEventBus(history_size=5)
        received = []

        async def cb(e):
            received.append(e.data)

        async def run():
            await bus.publish("t", data="old")
            count = await bus.replay_history("t", cb)
            return count

        count = asyncio.run(run())
        assert count == 1
        assert received == ["old"]

    def test_async_replay_history_matching(self):
        bus = AsyncEventBus(history_size=10)
        received = []

        async def cb(e):
            received.append(e.data)

        async def run():
            await bus.publish("s.a", data=1)
            await bus.publish("s.b", data=2)
            await bus.publish("other", data=3)
            count = await bus.replay_history_matching("s.#", cb)
            return count

        count = asyncio.run(run())
        assert count == 2
        assert set(received) == {1, 2}

    def test_async_clear_history(self):
        bus = AsyncEventBus(history_size=5)

        async def run():
            await bus.publish("t", data=1)

        asyncio.run(run())
        bus.clear_history("t")
        assert bus.get_history("t") == []

    def test_async_publish_concurrent_records_history(self):
        """publish_concurrent also records to history."""
        bus = AsyncEventBus(history_size=5)
        received = []

        async def cb(e):
            received.append(e.data)

        bus.subscribe("t", cb)

        async def run():
            await bus.publish_concurrent("t", data="conc")

        asyncio.run(run())
        assert len(bus.get_history("t")) == 1
        assert bus.get_history("t")[0].data == "conc"


# ===========================================================================
# Event Filtering — AsyncEventBus
# ===========================================================================


class TestAsyncEventBusFiltering:
    """Tests for async event bus filtering."""

    def test_async_filter_allows(self):
        bus = AsyncEventBus()
        received = []

        async def handler(e):
            received.append(e.data)

        bus.subscribe("t", handler, filter_fn=lambda e: e.data > 5)

        async def run():
            await bus.publish("t", data=3)
            await bus.publish("t", data=10)

        asyncio.run(run())
        assert received == [10]

    def test_async_filter_blocks(self):
        bus = AsyncEventBus()
        received = []

        async def handler(e):
            received.append(e.data)

        bus.subscribe("t", handler, filter_fn=lambda e: False)

        async def run():
            await bus.publish("t", data="x")

        asyncio.run(run())
        assert received == []

    def test_async_filter_exception_safe(self):
        bus = AsyncEventBus()
        received = []

        async def handler(e):
            received.append(e.data)

        bus.subscribe("t", handler, filter_fn=lambda e: 1 / 0)

        async def run():
            await bus.publish("t", data="x")

        asyncio.run(run())
        assert received == []

    def test_async_filter_on_concurrent_publish(self):
        """Filtering also works with publish_concurrent."""
        bus = AsyncEventBus()
        received = []

        async def handler(e):
            received.append(e.data)

        bus.subscribe("t", handler, filter_fn=lambda e: e.data == "yes")

        async def run():
            await bus.publish_concurrent("t", data="no")
            await bus.publish_concurrent("t", data="yes")

        asyncio.run(run())
        assert received == ["yes"]


# ===========================================================================
# Priority Ordering — AsyncEventBus
# ===========================================================================


class TestAsyncEventBusPriority:
    """Tests for async event bus priority ordering."""

    def test_async_priority_ordering(self):
        bus = AsyncEventBus()
        call_order = []

        async def low(e):
            call_order.append("low")

        async def high(e):
            call_order.append("high")

        async def mid(e):
            call_order.append("mid")

        bus.subscribe("t", low, priority=-5)
        bus.subscribe("t", high, priority=10)
        bus.subscribe("t", mid, priority=0)

        async def run():
            await bus.publish("t")

        asyncio.run(run())
        assert call_order == ["high", "mid", "low"]

    def test_async_priority_same_level_stable(self):
        bus = AsyncEventBus()
        call_order = []

        async def first(e):
            call_order.append("first")

        async def second(e):
            call_order.append("second")

        bus.subscribe("t", first, priority=0)
        bus.subscribe("t", second, priority=0)

        async def run():
            await bus.publish("t")

        asyncio.run(run())
        assert call_order == ["first", "second"]

    def test_async_priority_with_filter_combined(self):
        bus = AsyncEventBus()
        call_order = []

        async def high_filtered(e):
            call_order.append("high_filtered")

        async def low_unfiltered(e):
            call_order.append("low_unfiltered")

        bus.subscribe("t", high_filtered, priority=100, filter_fn=lambda e: False)
        bus.subscribe("t", low_unfiltered, priority=-100)

        async def run():
            await bus.publish("t")

        asyncio.run(run())
        # high_filtered is blocked by filter
        assert call_order == ["low_unfiltered"]


# ===========================================================================
# Backward Compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Ensure existing API continues to work unchanged."""

    def test_subscribe_with_positional_args(self):
        """Old-style subscribe(topic, callback) still works."""
        bus = EventBus()
        received = []
        bus.subscribe("t", lambda e: received.append(e))
        bus.publish("t")
        assert len(received) == 1

    def test_eventbus_no_args(self):
        """EventBus() with no args works (history disabled)."""
        bus = EventBus()
        received = []
        bus.subscribe("t", lambda e: received.append(e))
        bus.publish("t", data="hello")
        assert len(received) == 1
        assert received[0].data == "hello"

    def test_asynceventbus_no_args(self):
        """AsyncEventBus() with no args works (history disabled)."""
        bus = AsyncEventBus()
        received = []

        async def handler(e):
            received.append(e)

        bus.subscribe("t", handler)

        async def run():
            await bus.publish("t", data="world")

        asyncio.run(run())
        assert len(received) == 1

    def test_imports_unchanged(self):
        """Existing import patterns still work."""
        from tritium_lib.events import EventBus, AsyncEventBus, Event, QueueEventBus
        assert EventBus is not None
        assert AsyncEventBus is not None
        assert Event is not None
        assert QueueEventBus is not None

    def test_new_exports_available(self):
        """New exports are accessible."""
        from tritium_lib.events import EventFilter, DEFAULT_PRIORITY
        assert DEFAULT_PRIORITY == 0
        assert EventFilter is not None

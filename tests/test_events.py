"""Tests for tritium_lib.events."""

import asyncio

from tritium_lib.events import EventBus, AsyncEventBus, Event


class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("device.heartbeat", lambda e: received.append(e))
        bus.publish("device.heartbeat", {"id": "esp32-001"})
        assert len(received) == 1
        assert received[0].data["id"] == "esp32-001"

    def test_exact_match(self):
        bus = EventBus()
        received = []
        bus.subscribe("device.heartbeat", lambda e: received.append(e))
        bus.publish("device.command", {"id": "esp32-001"})
        assert len(received) == 0

    def test_single_wildcard(self):
        bus = EventBus()
        received = []
        bus.subscribe("device.*", lambda e: received.append(e))
        bus.publish("device.heartbeat", "hb")
        bus.publish("device.command", "cmd")
        bus.publish("device.sensor.temp", "nope")  # 2 levels deep — no match
        assert len(received) == 2

    def test_multi_wildcard(self):
        bus = EventBus()
        received = []
        bus.subscribe("device.#", lambda e: received.append(e))
        bus.publish("device.heartbeat", 1)
        bus.publish("device.sensor.temp", 2)
        bus.publish("other.thing", 3)
        assert len(received) == 2

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.subscribe("test", cb)
        bus.publish("test", 1)
        bus.unsubscribe("test", cb)
        bus.publish("test", 2)
        assert len(received) == 1

    def test_bad_subscriber_doesnt_break_bus(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: 1 / 0)  # raises
        bus.subscribe("test", lambda e: received.append(e))
        bus.publish("test", "data")
        assert len(received) == 1

    def test_event_has_timestamp(self):
        bus = EventBus()
        event = bus.publish("test", "data", source="unit-test")
        assert event.timestamp > 0
        assert event.source == "unit-test"


class TestAsyncEventBus:
    def test_publish_subscribe(self):
        async def _run():
            bus = AsyncEventBus()
            received = []

            async def handler(e: Event):
                received.append(e)

            bus.subscribe("test.event", handler)
            await bus.publish("test.event", {"key": "val"})
            assert len(received) == 1
            assert received[0].data["key"] == "val"

        asyncio.run(_run())

    def test_wildcard(self):
        async def _run():
            bus = AsyncEventBus()
            received = []

            async def handler(e: Event):
                received.append(e)

            bus.subscribe("device.#", handler)
            await bus.publish("device.heartbeat", 1)
            await bus.publish("device.sensor.temp", 2)
            await bus.publish("other.thing", 3)
            assert len(received) == 2

        asyncio.run(_run())

    def test_concurrent_publish(self):
        async def _run():
            bus = AsyncEventBus()
            received = []

            async def slow_handler(e: Event):
                await asyncio.sleep(0.01)
                received.append(("slow", e.data))

            async def fast_handler(e: Event):
                received.append(("fast", e.data))

            bus.subscribe("test", slow_handler)
            bus.subscribe("test", fast_handler)
            await bus.publish_concurrent("test", "data")
            assert len(received) == 2

        asyncio.run(_run())

    def test_unsubscribe(self):
        async def _run():
            bus = AsyncEventBus()
            received = []

            async def handler(e: Event):
                received.append(e)

            bus.subscribe("test", handler)
            await bus.publish("test", 1)
            bus.unsubscribe("test", handler)
            await bus.publish("test", 2)
            assert len(received) == 1

        asyncio.run(_run())

    def test_bad_subscriber(self):
        async def _run():
            bus = AsyncEventBus()
            received = []

            async def bad_handler(e: Event):
                raise ValueError("boom")

            async def good_handler(e: Event):
                received.append(e)

            bus.subscribe("test", bad_handler)
            bus.subscribe("test", good_handler)
            await bus.publish("test", "data")
            assert len(received) == 1

        asyncio.run(_run())

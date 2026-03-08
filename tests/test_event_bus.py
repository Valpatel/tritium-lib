# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the EventBus publish/subscribe system."""

from tritium_lib.events import Event, EventBus


class TestEventBus:
    def setup_method(self):
        self.bus = EventBus()

    def test_publish_returns_event(self):
        event = self.bus.publish("test.topic", data={"key": "value"})
        assert isinstance(event, Event)
        assert event.topic == "test.topic"
        assert event.data == {"key": "value"}

    def test_subscribe_receives_event(self):
        received = []
        self.bus.subscribe("device.heartbeat", lambda e: received.append(e))
        self.bus.publish("device.heartbeat", data="ping")
        assert len(received) == 1
        assert received[0].data == "ping"

    def test_no_match(self):
        received = []
        self.bus.subscribe("device.heartbeat", lambda e: received.append(e))
        self.bus.publish("sensor.reading")
        assert len(received) == 0

    def test_multiple_subscribers(self):
        r1, r2 = [], []
        self.bus.subscribe("event", lambda e: r1.append(e))
        self.bus.subscribe("event", lambda e: r2.append(e))
        self.bus.publish("event")
        assert len(r1) == 1
        assert len(r2) == 1

    def test_unsubscribe(self):
        received = []
        cb = lambda e: received.append(e)
        self.bus.subscribe("topic", cb)
        self.bus.publish("topic")
        assert len(received) == 1
        self.bus.unsubscribe("topic", cb)
        self.bus.publish("topic")
        assert len(received) == 1  # no new events

    def test_wildcard_star(self):
        received = []
        self.bus.subscribe("device.*", lambda e: received.append(e))
        self.bus.publish("device.heartbeat")
        self.bus.publish("device.command")
        self.bus.publish("sensor.reading")  # should not match
        assert len(received) == 2

    def test_wildcard_hash(self):
        received = []
        self.bus.subscribe("device.#", lambda e: received.append(e))
        self.bus.publish("device.heartbeat")
        self.bus.publish("device.sensor.temperature")
        self.bus.publish("sensor.reading")  # should not match
        assert len(received) == 2

    def test_wildcard_hash_matches_deep(self):
        received = []
        self.bus.subscribe("a.#", lambda e: received.append(e))
        self.bus.publish("a.b.c.d")
        assert len(received) == 1

    def test_event_source(self):
        event = self.bus.publish("test", source="fleet-server")
        assert event.source == "fleet-server"

    def test_event_timestamp(self):
        event = self.bus.publish("test")
        assert event.timestamp > 0

    def test_bad_subscriber_doesnt_break_bus(self):
        received = []

        def bad_handler(e):
            raise RuntimeError("boom")

        self.bus.subscribe("topic", bad_handler)
        self.bus.subscribe("topic", lambda e: received.append(e))
        self.bus.publish("topic")
        # Second subscriber should still receive despite first raising
        assert len(received) == 1

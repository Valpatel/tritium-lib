# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Tests for the Inter-Addon Event protocol."""

import asyncio
import time

import pytest

from tritium_lib.sdk.addon_events import AddonEvent, AddonEventBus
from tritium_lib.sdk.addon_base import AddonBase, AddonInfo


# ---------------------------------------------------------------------------
# AddonEvent basics
# ---------------------------------------------------------------------------

class TestAddonEvent:
    def test_creation(self):
        evt = AddonEvent(
            source_addon="hackrf",
            event_type="signal_detected",
            data={"freq_mhz": 433.92},
        )
        assert evt.source_addon == "hackrf"
        assert evt.event_type == "signal_detected"
        assert evt.data == {"freq_mhz": 433.92}
        assert evt.device_id == ""
        assert isinstance(evt.timestamp, float)

    def test_topic_property(self):
        evt = AddonEvent(
            source_addon="hackrf",
            event_type="signal_detected",
            data={},
        )
        assert evt.topic == "addon:hackrf:signal_detected"

    def test_to_dict_serialization(self):
        ts = time.time()
        evt = AddonEvent(
            source_addon="sdr",
            event_type="sweep_complete",
            data={"bands": [433, 868]},
            timestamp=ts,
            device_id="sdr-001",
        )
        d = evt.to_dict()
        assert d["source_addon"] == "sdr"
        assert d["event_type"] == "sweep_complete"
        assert d["data"] == {"bands": [433, 868]}
        assert d["timestamp"] == ts
        assert d["device_id"] == "sdr-001"
        assert d["topic"] == "addon:sdr:sweep_complete"

    def test_to_dict_is_json_serializable(self):
        import json

        evt = AddonEvent(source_addon="a", event_type="b", data={"x": 1})
        json.dumps(evt.to_dict())  # should not raise


# ---------------------------------------------------------------------------
# AddonEventBus
# ---------------------------------------------------------------------------

class TestAddonEventBus:
    def test_publish_creates_event(self):
        bus = AddonEventBus()
        evt = bus.publish("hackrf", "signal_detected", {"freq_mhz": 915})
        assert isinstance(evt, AddonEvent)
        assert evt.source_addon == "hackrf"
        assert evt.event_type == "signal_detected"

    def test_subscribe_exact_match(self):
        bus = AddonEventBus()
        received = []
        bus.subscribe("addon:hackrf:signal_detected", received.append)

        bus.publish("hackrf", "signal_detected", {"freq": 433})
        assert len(received) == 1
        assert received[0].data == {"freq": 433}

    def test_subscribe_wildcard_source(self):
        bus = AddonEventBus()
        received = []
        bus.subscribe("addon:*:signal_detected", received.append)

        bus.publish("hackrf", "signal_detected", {})
        bus.publish("sdr", "signal_detected", {})
        bus.publish("hackrf", "sweep_complete", {})  # should NOT match
        assert len(received) == 2

    def test_subscribe_wildcard_event_type(self):
        bus = AddonEventBus()
        received = []
        bus.subscribe("addon:hackrf:*", received.append)

        bus.publish("hackrf", "signal_detected", {})
        bus.publish("hackrf", "sweep_complete", {})
        bus.publish("sdr", "signal_detected", {})  # should NOT match
        assert len(received) == 2

    def test_subscribe_all(self):
        bus = AddonEventBus()
        received = []
        bus.subscribe("addon:*:*", received.append)

        bus.publish("hackrf", "signal_detected", {})
        bus.publish("sdr", "sweep_complete", {})
        bus.publish("camera", "frame_ready", {})
        assert len(received) == 3

    def test_non_matching_pattern(self):
        bus = AddonEventBus()
        received = []
        bus.subscribe("addon:hackrf:signal_detected", received.append)

        bus.publish("sdr", "sweep_complete", {})
        assert len(received) == 0

    def test_unsubscribe_stops_delivery(self):
        bus = AddonEventBus()
        received = []
        bus.subscribe("addon:hackrf:*", received.append)

        bus.publish("hackrf", "signal_detected", {})
        assert len(received) == 1

        result = bus.unsubscribe("addon:hackrf:*", received.append)
        assert result is True

        bus.publish("hackrf", "sweep_complete", {})
        assert len(received) == 1  # no new delivery

    def test_unsubscribe_unknown_returns_false(self):
        bus = AddonEventBus()
        assert bus.unsubscribe("addon:x:y", lambda e: None) is False

    def test_publish_with_device_id(self):
        bus = AddonEventBus()
        evt = bus.publish("hackrf", "signal_detected", {}, device_id="hrf-01")
        assert evt.device_id == "hrf-01"

    def test_wraps_existing_event_bus(self):
        """When an external EventBus is provided, events are forwarded."""

        class FakeEventBus:
            def __init__(self):
                self.emitted = []

            def emit(self, topic, data):
                self.emitted.append((topic, data))

        fake = FakeEventBus()
        bus = AddonEventBus(event_bus=fake)
        bus.publish("hackrf", "signal_detected", {"f": 1})
        assert len(fake.emitted) == 1
        assert fake.emitted[0][0] == "addon:hackrf:signal_detected"


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

class TestPatternMatching:
    @pytest.mark.parametrize(
        "topic,pattern,expected",
        [
            ("addon:hackrf:signal_detected", "addon:hackrf:signal_detected", True),
            ("addon:hackrf:signal_detected", "addon:hackrf:*", True),
            ("addon:hackrf:signal_detected", "addon:*:signal_detected", True),
            ("addon:hackrf:signal_detected", "addon:*:*", True),
            ("addon:hackrf:signal_detected", "addon:sdr:signal_detected", False),
            ("addon:hackrf:signal_detected", "addon:hackrf:sweep_complete", False),
            ("addon:hackrf:signal_detected", "addon:sdr:*", False),
        ],
    )
    def test_matches(self, topic, pattern, expected):
        assert AddonEventBus._matches(topic, pattern) is expected


# ---------------------------------------------------------------------------
# AddonBase integration
# ---------------------------------------------------------------------------

class TestAddonBaseIntegration:
    def test_publish_addon_event(self):
        bus = AddonEventBus()
        received = []
        bus.subscribe("addon:my-sensor:*", received.append)

        addon = AddonBase()
        addon.info = AddonInfo(id="my-sensor", name="My Sensor")
        addon.set_event_bus(bus)

        evt = addon.publish_addon_event("reading", {"value": 42})
        assert evt is not None
        assert evt.source_addon == "my-sensor"
        assert len(received) == 1

    def test_publish_without_bus_returns_none(self):
        addon = AddonBase()
        addon.info = AddonInfo(id="x", name="X")
        assert addon.publish_addon_event("evt", {}) is None

    def test_subscribe_addon_event_cleanup(self):
        bus = AddonEventBus()
        received = []

        addon = AddonBase()
        addon.info = AddonInfo(id="my-addon", name="My Addon")
        addon.set_event_bus(bus)
        addon.subscribe_addon_event("addon:other:*", received.append)

        # Event delivered before unregister
        bus.publish("other", "ping", {})
        assert len(received) == 1

        # Unregister should clean up subscriptions
        asyncio.run(addon.unregister(None))

        bus.publish("other", "ping", {})
        assert len(received) == 1  # no new delivery

    def test_subscribe_without_bus_is_noop(self):
        addon = AddonBase()
        addon.info = AddonInfo(id="x", name="X")
        # Should not raise
        addon.subscribe_addon_event("addon:*:*", lambda e: None)

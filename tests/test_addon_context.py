# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Tests for AddonContext, protocol interfaces, and AddonConfig."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import pytest

from tritium_lib.sdk import (
    AddonBase,
    AddonConfig,
    AddonContext,
    AddonInfo,
    IEventBus,
    IMQTTClient,
    IRouterHandler,
    ITargetTracker,
)
from tritium_lib.sdk.addon_events import AddonEventBus


# ---------------------------------------------------------------------------
# Concrete implementations for protocol tests
# ---------------------------------------------------------------------------


class FakeTargetTracker:
    """Minimal implementation satisfying ITargetTracker."""

    def __init__(self):
        self._targets: dict[str, dict] = {}

    def update_target(self, target_id: str, data: dict) -> None:
        self._targets[target_id] = data

    def get_target(self, target_id: str) -> dict | None:
        return self._targets.get(target_id)

    def get_all_targets(self) -> list[dict]:
        return list(self._targets.values())

    def remove_target(self, target_id: str) -> bool:
        return self._targets.pop(target_id, None) is not None


class FakeEventBus:
    """Minimal implementation satisfying IEventBus."""

    def __init__(self):
        self._subs: dict[str, list] = {}

    def publish(self, topic: str, data: Any = None, source: str = "") -> Any:
        for cb in self._subs.get(topic, []):
            cb(data)

    def subscribe(self, topic: str, callback: Callable) -> Any:
        self._subs.setdefault(topic, []).append(callback)


class FakeMQTTClient:
    """Minimal implementation satisfying IMQTTClient."""

    def __init__(self):
        self.published: list[tuple] = []

    def publish(self, topic: str, payload: Any, **kwargs) -> None:
        self.published.append((topic, payload))

    def subscribe(self, topic: str, callback: Callable | None = None) -> None:
        pass


class FakeRouterHandler:
    """Minimal implementation satisfying IRouterHandler."""

    def __init__(self):
        self.routers: list = []

    def include_router(
        self,
        router: Any,
        prefix: str = "",
        tags: list[str] | None = None,
    ) -> None:
        self.routers.append((router, prefix, tags))


# ---------------------------------------------------------------------------
# Protocol isinstance checks
# ---------------------------------------------------------------------------


class TestProtocols:
    def test_target_tracker_protocol(self):
        tracker = FakeTargetTracker()
        assert isinstance(tracker, ITargetTracker)

    def test_event_bus_protocol(self):
        bus = FakeEventBus()
        assert isinstance(bus, IEventBus)

    def test_mqtt_client_protocol(self):
        mqtt = FakeMQTTClient()
        assert isinstance(mqtt, IMQTTClient)

    def test_router_handler_protocol(self):
        handler = FakeRouterHandler()
        assert isinstance(handler, IRouterHandler)

    def test_plain_object_not_target_tracker(self):
        assert not isinstance(object(), ITargetTracker)

    def test_plain_object_not_event_bus(self):
        assert not isinstance(object(), IEventBus)


# ---------------------------------------------------------------------------
# AddonContext
# ---------------------------------------------------------------------------


class TestAddonContext:
    def test_default_creation(self):
        ctx = AddonContext()
        assert ctx.target_tracker is None
        assert ctx.event_bus is None
        assert ctx.mqtt_client is None
        assert ctx.router_handler is None
        assert ctx.site_id == "home"
        assert ctx.data_dir == "data"
        assert ctx.state == {}
        assert ctx.app is None

    def test_creation_with_services(self):
        tracker = FakeTargetTracker()
        bus = FakeEventBus()
        mqtt = FakeMQTTClient()
        router = FakeRouterHandler()
        ctx = AddonContext(
            target_tracker=tracker,
            event_bus=bus,
            mqtt_client=mqtt,
            router_handler=router,
            site_id="alpha",
            data_dir="/tmp/addon_data",
        )
        assert ctx.target_tracker is tracker
        assert ctx.event_bus is bus
        assert ctx.mqtt_client is mqtt
        assert ctx.router_handler is router
        assert ctx.site_id == "alpha"
        assert ctx.data_dir == "/tmp/addon_data"

    def test_get_set_state(self):
        ctx = AddonContext()
        assert ctx.get_state("foo") is None
        assert ctx.get_state("foo", 42) == 42
        ctx.set_state("foo", "bar")
        assert ctx.get_state("foo") == "bar"

    def test_has_service(self):
        ctx = AddonContext(target_tracker=FakeTargetTracker())
        assert ctx.has_service("target_tracker") is True
        assert ctx.has_service("event_bus") is False
        assert ctx.has_service("nonexistent") is False

    def test_state_isolation(self):
        ctx1 = AddonContext()
        ctx2 = AddonContext()
        ctx1.set_state("x", 1)
        assert ctx2.get_state("x") is None


# ---------------------------------------------------------------------------
# AddonBase with context
# ---------------------------------------------------------------------------


class SampleAddon(AddonBase):
    info = AddonInfo(id="sample", name="Sample Addon", version="1.0.0")


class TestAddonBaseContext:
    def test_register_legacy(self):
        """Legacy register(app) still works."""
        addon = SampleAddon()
        asyncio.run(addon.register("fake_app"))
        assert addon._registered is True
        assert addon._context is None

    def test_register_with_context(self):
        """register() with context stores it and wires addon event bus."""
        tracker = FakeTargetTracker()
        bus = FakeEventBus()
        addon_bus = AddonEventBus()
        ctx = AddonContext(
            target_tracker=tracker,
            event_bus=bus,
            addon_event_bus=addon_bus,
        )
        addon = SampleAddon()
        asyncio.run(addon.register(context=ctx))
        assert addon._registered is True
        assert addon._context is ctx
        assert addon._addon_event_bus is addon_bus

    def test_convenience_properties_with_context(self):
        tracker = FakeTargetTracker()
        bus = FakeEventBus()
        mqtt = FakeMQTTClient()
        ctx = AddonContext(
            target_tracker=tracker,
            event_bus=bus,
            mqtt_client=mqtt,
            site_id="bravo",
        )
        addon = SampleAddon()
        asyncio.run(addon.register(context=ctx))
        assert addon.target_tracker is tracker
        assert addon.event_bus is bus
        assert addon.mqtt_client is mqtt
        assert addon.site_id == "bravo"

    def test_convenience_properties_without_context(self):
        addon = SampleAddon()
        assert addon.target_tracker is None
        assert addon.event_bus is None
        assert addon.mqtt_client is None
        assert addon.site_id == "home"

    def test_context_property(self):
        ctx = AddonContext(site_id="charlie")
        addon = SampleAddon()
        asyncio.run(addon.register(context=ctx))
        assert addon.context is ctx


# ---------------------------------------------------------------------------
# AddonConfig
# ---------------------------------------------------------------------------


class TestAddonConfig:
    def test_empty_schema(self):
        cfg = AddonConfig()
        assert cfg.to_dict() == {}

    def test_defaults_from_schema(self):
        schema = {
            "refresh_interval": {"type": "int", "default": 30},
            "enabled": {"type": "bool", "default": True},
            "label": "default_label",  # non-dict value treated as default
        }
        cfg = AddonConfig(config_schema=schema)
        assert cfg.get("refresh_interval") == 30
        assert cfg.get("enabled") is True
        assert cfg.get("label") == "default_label"

    def test_overrides(self):
        schema = {
            "refresh_interval": {"type": "int", "default": 30},
        }
        cfg = AddonConfig(config_schema=schema, overrides={"refresh_interval": 5})
        assert cfg.get("refresh_interval") == 5

    def test_getattr_access(self):
        cfg = AddonConfig(config_schema={"color": "cyan"})
        assert cfg.color == "cyan"
        assert cfg.nonexistent is None

    def test_getattr_private_raises(self):
        cfg = AddonConfig()
        with pytest.raises(AttributeError):
            _ = cfg._internal

    def test_to_dict(self):
        schema = {"a": 1, "b": {"default": 2}}
        cfg = AddonConfig(config_schema=schema)
        d = cfg.to_dict()
        assert d == {"a": 1, "b": 2}

    def test_repr(self):
        cfg = AddonConfig(config_schema={"x": 10})
        assert "AddonConfig" in repr(cfg)

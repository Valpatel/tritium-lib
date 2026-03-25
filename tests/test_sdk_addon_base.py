# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for AddonBase lifecycle, properties, event helpers."""

import asyncio

import pytest

from tritium_lib.sdk.addon_base import AddonBase, AddonInfo
from tritium_lib.sdk.addon_events import AddonEventBus
from tritium_lib.sdk.context import AddonContext
from tritium_lib.sdk.geo_layer import AddonGeoLayer


# ── AddonInfo ───────────────────────────────────────────────────────

class TestAddonInfo:
    """Tests for the AddonInfo dataclass."""

    def test_required_fields(self):
        info = AddonInfo(id="test-addon", name="Test Addon")
        assert info.id == "test-addon"
        assert info.name == "Test Addon"

    def test_default_version(self):
        info = AddonInfo(id="x", name="X")
        assert info.version == "0.0.0"

    def test_default_license(self):
        info = AddonInfo(id="x", name="X")
        assert info.license == "AGPL-3.0"

    def test_default_category(self):
        info = AddonInfo(id="x", name="X")
        assert info.category == "system"

    def test_all_fields(self):
        info = AddonInfo(
            id="my-addon", name="My Addon", version="2.1.0",
            description="Desc", author="Author", license="MIT",
            category="sensors", icon="sensor.svg",
            min_sdk_version="2.0.0",
        )
        assert info.description == "Desc"
        assert info.author == "Author"
        assert info.icon == "sensor.svg"
        assert info.min_sdk_version == "2.0.0"


# ── AddonBase lifecycle ─────────────────────────────────────────────

class TestAddonBaseLifecycle:
    """Tests for AddonBase register/unregister."""

    def test_initial_state(self):
        addon = AddonBase()
        assert addon._registered is False
        assert addon.context is None

    def test_register_sets_registered(self):
        addon = AddonBase()
        asyncio.run(addon.register())
        assert addon._registered is True

    def test_unregister_clears_registered(self):
        addon = AddonBase()
        asyncio.run(addon.register())
        asyncio.run(addon.unregister(None))
        assert addon._registered is False

    def test_register_with_context(self):
        addon = AddonBase()
        ctx = AddonContext(site_id="test-site")
        asyncio.run(addon.register(context=ctx))
        assert addon.context is not None
        assert addon.context.site_id == "test-site"

    def test_register_with_context_and_event_bus(self):
        addon = AddonBase()
        bus = AddonEventBus()
        ctx = AddonContext(site_id="test", addon_event_bus=bus)
        asyncio.run(addon.register(context=ctx))
        assert addon._addon_event_bus is bus

    def test_unregister_clears_subscriptions(self):
        addon = AddonBase()
        addon._event_subscriptions.append(lambda: None)
        addon._mqtt_subscriptions.append(lambda: None)
        asyncio.run(addon.unregister(None))
        assert len(addon._event_subscriptions) == 0
        assert len(addon._mqtt_subscriptions) == 0


# ── Context convenience properties ──────────────────────────────────

class TestAddonBaseProperties:
    """Tests for context shortcut properties."""

    def test_target_tracker_none_without_context(self):
        addon = AddonBase()
        assert addon.target_tracker is None

    def test_event_bus_none_without_context(self):
        addon = AddonBase()
        assert addon.event_bus is None

    def test_mqtt_client_none_without_context(self):
        addon = AddonBase()
        assert addon.mqtt_client is None

    def test_site_id_default_home(self):
        addon = AddonBase()
        assert addon.site_id == "home"

    def test_site_id_from_context(self):
        addon = AddonBase()
        ctx = AddonContext(site_id="warehouse")
        asyncio.run(addon.register(context=ctx))
        assert addon.site_id == "warehouse"

    def test_target_tracker_setter(self):
        addon = AddonBase()
        mock_tracker = object()
        addon.target_tracker = mock_tracker
        assert addon.target_tracker is mock_tracker

    def test_event_bus_setter(self):
        addon = AddonBase()
        mock_bus = object()
        addon.event_bus = mock_bus
        assert addon.event_bus is mock_bus

    def test_mqtt_client_setter(self):
        addon = AddonBase()
        mock_client = object()
        addon.mqtt_client = mock_client
        assert addon.mqtt_client is mock_client

    def test_site_id_setter(self):
        addon = AddonBase()
        addon.site_id = "custom"
        assert addon.site_id == "custom"


# ── Event helpers ───────────────────────────────────────────────────

class TestAddonBaseEventHelpers:
    """Tests for addon event bus integration."""

    def test_set_event_bus(self):
        addon = AddonBase()
        bus = AddonEventBus()
        addon.set_event_bus(bus)
        assert addon._addon_event_bus is bus

    def test_publish_addon_event_without_bus_returns_none(self):
        addon = AddonBase()
        result = addon.publish_addon_event("test_event", {"key": "val"})
        assert result is None

    def test_publish_addon_event_with_bus(self):
        addon = AddonBase()
        addon.info = AddonInfo(id="my-addon", name="My Addon")
        bus = AddonEventBus()
        addon.set_event_bus(bus)
        event = addon.publish_addon_event("data_ready", {"count": 5})
        assert event is not None
        assert event.source_addon == "my-addon"
        assert event.event_type == "data_ready"
        assert event.data["count"] == 5

    def test_subscribe_addon_event_without_bus(self):
        addon = AddonBase()
        # Should not raise
        addon.subscribe_addon_event("*", lambda e: None)
        assert len(addon._addon_event_unsubs) == 0

    def test_subscribe_addon_event_with_bus(self):
        addon = AddonBase()
        bus = AddonEventBus()
        addon.set_event_bus(bus)
        received = []
        addon.subscribe_addon_event("*", lambda e: received.append(e))
        assert len(addon._addon_event_unsubs) == 1


# ── Default method returns ──────────────────────────────────────────

class TestAddonBaseDefaults:
    """Tests for default method return values."""

    def test_get_panels_empty(self):
        addon = AddonBase()
        assert addon.get_panels() == []

    def test_get_layers_empty(self):
        addon = AddonBase()
        assert addon.get_layers() == []

    def test_get_geojson_layers_empty(self):
        addon = AddonBase()
        assert addon.get_geojson_layers() == []

    def test_get_context_menu_items_empty(self):
        addon = AddonBase()
        assert addon.get_context_menu_items() == []

    def test_get_shortcuts_empty(self):
        addon = AddonBase()
        assert addon.get_shortcuts() == []

    def test_health_check_not_registered(self):
        addon = AddonBase()
        health = addon.health_check()
        assert health["status"] == "not_registered"

    def test_health_check_registered(self):
        addon = AddonBase()
        asyncio.run(addon.register())
        health = addon.health_check()
        assert health["status"] == "ok"

    def test_repr(self):
        addon = AddonBase()
        addon.info = AddonInfo(id="test", name="Test", version="1.0.0")
        r = repr(addon)
        assert "AddonBase" in r
        assert "test" in r
        assert "1.0.0" in r

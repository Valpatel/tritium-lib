# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the Tritium Addon SDK."""

import asyncio
import pytest
from pathlib import Path

from tritium_lib.sdk import (
    AddonBase, AddonInfo, SDK_VERSION,
    SensorAddon, ProcessorAddon, CommanderAddon, BridgeAddon,
    DataSourceAddon, PanelAddon, ToolAddon, AggregatorAddon,
)
from tritium_lib.sdk.manifest import AddonManifest, load_manifest, validate_manifest


class TestAddonInfo:
    def test_defaults(self):
        info = AddonInfo(id="test", name="Test")
        assert info.version == "0.0.0"
        assert info.category == "system"

    def test_full(self):
        info = AddonInfo(id="mesh", name="Meshtastic", version="2.1.0",
                        category="radio", icon="📡", author="Valpatel")
        assert info.id == "mesh"
        assert info.icon == "📡"


class TestAddonBase:
    def test_create(self):
        addon = AddonBase()
        assert not addon._registered

    def test_register_sets_flag(self):
        addon = AddonBase()
        asyncio.run(addon.register(None))
        assert addon._registered

    def test_unregister_clears_flag(self):
        addon = AddonBase()
        asyncio.run(addon.register(None))
        asyncio.run(addon.unregister(None))
        assert not addon._registered

    def test_health_check(self):
        addon = AddonBase()
        assert addon.health_check()["status"] == "not_registered"
        asyncio.run(addon.register(None))
        assert addon.health_check()["status"] == "ok"

    def test_get_panels_default_empty(self):
        assert AddonBase().get_panels() == []

    def test_get_layers_default_empty(self):
        assert AddonBase().get_layers() == []

    def test_repr(self):
        addon = AddonBase()
        addon.info = AddonInfo(id="test", name="Test", version="1.0.0")
        assert "test" in repr(addon)


class TestSensorAddon:
    def test_gather_default(self):
        sensor = SensorAddon()
        result = asyncio.run(sensor.gather())
        assert result == []

    def test_subclass(self):
        class MySensor(SensorAddon):
            info = AddonInfo(id="my-sensor", name="My Sensor")
            async def gather(self):
                return [{"target_id": "t1", "source": "test", "position": {"x": 0, "y": 0}}]

        sensor = MySensor()
        targets = asyncio.run(sensor.gather())
        assert len(targets) == 1
        assert targets[0]["target_id"] == "t1"


class TestProcessorAddon:
    def test_passthrough(self):
        proc = ProcessorAddon()
        target = {"target_id": "t1", "source": "ble"}
        result = asyncio.run(proc.process(target))
        assert result == target


class TestCommanderAddon:
    def test_think_default(self):
        cmd = CommanderAddon()
        actions = asyncio.run(cmd.think({}))
        assert actions == []


class TestBridgeAddon:
    def test_send_receive_default(self):
        bridge = BridgeAddon()
        asyncio.run(bridge.send([]))
        result = asyncio.run(bridge.receive())
        assert result == []


class TestManifest:
    def test_empty_manifest(self):
        m = AddonManifest()
        assert m.id == ""
        errors = validate_manifest(m)
        assert len(errors) >= 2  # missing id and name

    def test_valid_manifest(self):
        m = AddonManifest(id="test-addon", name="Test Addon", version="1.0.0")
        errors = validate_manifest(m)
        assert errors == []

    def test_invalid_id(self):
        m = AddonManifest(id="Test Addon!", name="Test", version="1.0.0")
        errors = validate_manifest(m)
        assert any("Invalid addon.id" in e for e in errors)

    def test_panel_validation(self):
        m = AddonManifest(id="test", name="Test", version="1.0.0",
                         panels=[{"id": "p1", "title": "Panel 1"}, {"title": "Missing ID"}])
        errors = validate_manifest(m)
        assert any("missing 'id'" in e for e in errors)

    def test_to_frontend_json(self):
        m = AddonManifest(id="mesh", name="Meshtastic", version="2.0.0",
                         category_window="radio", category_icon="📡",
                         panels=[{"id": "mesh-net", "title": "MESH"}],
                         layers=[{"id": "meshNodes", "label": "Nodes"}])
        j = m.to_frontend_json()
        assert j["id"] == "mesh"
        assert j["category"] == "radio"
        assert len(j["panels"]) == 1
        assert len(j["layers"]) == 1

    def test_load_manifest_file(self, tmp_path):
        """Test loading a real TOML manifest."""
        toml_content = '''
[addon]
id = "test-sensor"
name = "Test Sensor"
version = "1.0.0"
description = "A test sensor addon"

[addon.category]
window = "sensors"
tab_order = 5
icon = "🔬"

[dependencies]
requires = []
python_packages = ["some-package>=1.0"]

[permissions]
serial = true
network = true

[backend]
module = "test_sensor_addon"
router_prefix = "/api/addons/test-sensor"

[frontend]
panels = [
    { id = "test-panel", title = "TEST SENSOR", file = "panels/test.js" },
]
layers = [
    { id = "testLayer", label = "Test Data", category = "SENSORS", color = "#00ff00" },
]
'''
        manifest_path = tmp_path / "tritium_addon.toml"
        manifest_path.write_text(toml_content)

        m = load_manifest(manifest_path)
        assert m.id == "test-sensor"
        assert m.name == "Test Sensor"
        assert m.category_window == "sensors"
        assert m.category_tab_order == 5
        assert m.perm_serial is True
        assert m.perm_network is True
        assert len(m.panels) == 1
        assert len(m.layers) == 1
        assert m.panels[0]["id"] == "test-panel"
        errors = validate_manifest(m)
        assert errors == []

    def test_load_manifest_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nope.toml")


class TestSDKVersion:
    def test_version_exists(self):
        assert SDK_VERSION == "1.0.0"

    def test_all_interfaces_importable(self):
        """All addon types can be imported."""
        from tritium_lib.sdk import (
            AddonBase, SensorAddon, ProcessorAddon, AggregatorAddon,
            CommanderAddon, BridgeAddon, DataSourceAddon, PanelAddon, ToolAddon,
        )
        # All should be subclasses of AddonBase
        for cls in [SensorAddon, ProcessorAddon, AggregatorAddon,
                    CommanderAddon, BridgeAddon, DataSourceAddon, PanelAddon, ToolAddon]:
            assert issubclass(cls, AddonBase)

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sdk.manifest — addon manifest parsing and validation."""

import tempfile
from pathlib import Path

import pytest

from tritium_lib.sdk.manifest import AddonManifest, load_manifest, validate_manifest


class TestAddonManifest:
    def test_default_manifest(self):
        m = AddonManifest()
        assert m.id == ""
        assert m.name == ""
        assert m.version == "0.0.0"
        assert m.license == "AGPL-3.0"

    def test_to_frontend_json(self):
        m = AddonManifest(id="test-addon", name="Test", version="1.0.0")
        d = m.to_frontend_json()
        assert d["id"] == "test-addon"
        assert d["name"] == "Test"
        assert d["version"] == "1.0.0"
        assert "panels" in d
        assert "layers" in d

    def test_permissions_defaults(self):
        m = AddonManifest()
        assert m.perm_serial is False
        assert m.perm_network is False
        assert m.perm_mqtt is False
        assert m.perm_storage is False

    def test_dependencies_defaults(self):
        m = AddonManifest()
        assert m.requires == []
        assert m.optional == []
        assert m.python_packages == []


class TestValidateManifest:
    def test_valid_manifest(self):
        m = AddonManifest(id="my-addon", name="My Addon", version="1.0.0")
        errors = validate_manifest(m)
        assert errors == []

    def test_missing_id(self):
        m = AddonManifest(name="Test", version="1.0")
        errors = validate_manifest(m)
        assert any("addon.id" in e for e in errors)

    def test_missing_name(self):
        m = AddonManifest(id="test", version="1.0")
        errors = validate_manifest(m)
        assert any("addon.name" in e for e in errors)

    def test_missing_version(self):
        m = AddonManifest(id="test", name="Test", version="")
        errors = validate_manifest(m)
        assert any("addon.version" in e for e in errors)

    def test_invalid_id_format(self):
        m = AddonManifest(id="my addon!", name="Test", version="1.0")
        errors = validate_manifest(m)
        assert any("Invalid addon.id" in e for e in errors)

    def test_valid_id_with_hyphens(self):
        m = AddonManifest(id="my-cool-addon", name="Test", version="1.0")
        errors = validate_manifest(m)
        assert errors == []

    def test_panel_missing_id(self):
        m = AddonManifest(
            id="test", name="Test", version="1.0",
            panels=[{"title": "Panel 1"}],
        )
        errors = validate_manifest(m)
        assert any("Panel 0 missing 'id'" in e for e in errors)

    def test_panel_missing_title(self):
        m = AddonManifest(
            id="test", name="Test", version="1.0",
            panels=[{"id": "panel1"}],
        )
        errors = validate_manifest(m)
        assert any("Panel 0 missing 'title'" in e for e in errors)

    def test_layer_missing_id(self):
        m = AddonManifest(
            id="test", name="Test", version="1.0",
            layers=[{"label": "My Layer"}],
        )
        errors = validate_manifest(m)
        assert any("Layer 0 missing 'id'" in e for e in errors)

    def test_layer_missing_label(self):
        m = AddonManifest(
            id="test", name="Test", version="1.0",
            layers=[{"id": "layer1"}],
        )
        errors = validate_manifest(m)
        assert any("Layer 0 missing 'label'" in e for e in errors)


class TestLoadManifest:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/tritium_addon.toml")

    def test_load_valid_toml(self):
        toml_content = b"""
[addon]
id = "sdr-monitor"
name = "SDR Monitor"
version = "0.5.0"
description = "HackRF spectrum analysis"
author = "Test Author"

[dependencies]
requires = ["hackrf"]
python_packages = ["numpy"]

[hardware]
devices = ["hackrf_one"]
auto_detect = true

[permissions]
serial = true
network = true
mqtt = true

[backend]
module = "sdr_monitor"
router_prefix = "/api/addons/sdr-monitor"

[frontend]
panels = [
    {id = "sdr-spectrum", title = "Spectrum Analyzer"},
]

[config]
center_freq = {type = "int", default = 100000000}
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            m = load_manifest(f.name)

        assert m.id == "sdr-monitor"
        assert m.name == "SDR Monitor"
        assert m.version == "0.5.0"
        assert m.perm_serial is True
        assert m.perm_network is True
        assert m.perm_mqtt is True
        assert "hackrf" in m.requires
        assert "numpy" in m.python_packages
        assert len(m.panels) == 1
        assert m.panels[0]["id"] == "sdr-spectrum"
        assert m.hardware_devices == ["hackrf_one"]
        assert m.auto_detect is True

    def test_load_minimal_toml(self):
        toml_content = b"""
[addon]
id = "minimal"
name = "Minimal Addon"
version = "0.1.0"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            m = load_manifest(f.name)

        assert m.id == "minimal"
        assert m.requires == []
        assert m.panels == []
        assert m.perm_serial is False

    def test_category_as_string(self):
        toml_content = b"""
[addon]
id = "test"
name = "Test"
version = "1.0"
category = "sensors"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            m = load_manifest(f.name)
        assert m.category_window == "sensors"

    def test_category_as_dict(self):
        toml_content = b"""
[addon]
id = "test"
name = "Test"
version = "1.0"

[addon.category]
window = "intelligence"
tab_order = 5
icon = "brain"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            m = load_manifest(f.name)
        assert m.category_window == "intelligence"
        assert m.category_tab_order == 5
        assert m.category_icon == "brain"

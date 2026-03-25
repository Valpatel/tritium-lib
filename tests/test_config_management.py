# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.config — configuration management system.

Covers: TOML loading, env overrides, per-addon sections, validation,
config dump, source info, subclassing.
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from tritium_lib.config import (
    ConfigError,
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_FILE,
    TritiumBaseSettings,
    TritiumSettings,
    get_addon_config,
    load_toml,
    validate_settings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_toml(tmp_path: Path, content: str) -> Path:
    """Write a TOML file and return its path."""
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(textwrap.dedent(content))
    return toml_file


def _clean_env(monkeypatch):
    """Remove all TRITIUM_ env vars to isolate tests."""
    for k in list(os.environ):
        if k.startswith("TRITIUM_"):
            monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# 1. Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_values(self, monkeypatch):
        """Settings should have sensible defaults when nothing is configured."""
        _clean_env(monkeypatch)
        settings = TritiumBaseSettings()
        assert settings.app_name == "Tritium"
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.site_id == "home"
        assert settings.log_level == "INFO"
        assert settings.mqtt_enabled is False
        assert settings.mqtt_host == "localhost"
        assert settings.mqtt_port == 1883

    def test_default_config_paths(self):
        """Default config dir and file should point to ~/.tritium/."""
        assert DEFAULT_CONFIG_DIR == Path.home() / ".tritium"
        assert DEFAULT_CONFIG_FILE == Path.home() / ".tritium" / "config.toml"


# ---------------------------------------------------------------------------
# 2. Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_env_prefix_overrides(self, monkeypatch):
        """Environment variables with TRITIUM_ prefix should override defaults."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("TRITIUM_DEBUG", "true")
        monkeypatch.setenv("TRITIUM_PORT", "9090")
        monkeypatch.setenv("TRITIUM_SITE_ID", "base-alpha")
        settings = TritiumBaseSettings()
        assert settings.debug is True
        assert settings.port == 9090
        assert settings.site_id == "base-alpha"

    def test_env_mqtt_override(self, monkeypatch):
        """MQTT settings should be overridable via env vars."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("TRITIUM_MQTT_ENABLED", "true")
        monkeypatch.setenv("TRITIUM_MQTT_HOST", "mqtt.example.com")
        monkeypatch.setenv("TRITIUM_MQTT_PORT", "8883")
        settings = TritiumBaseSettings()
        assert settings.mqtt_enabled is True
        assert settings.mqtt_host == "mqtt.example.com"
        assert settings.mqtt_port == 8883

    def test_env_log_level_normalized(self, monkeypatch):
        """Log level from env should be uppercased."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("TRITIUM_LOG_LEVEL", "debug")
        settings = TritiumBaseSettings()
        assert settings.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# 3. TOML file loading
# ---------------------------------------------------------------------------

class TestTomlLoading:
    def test_load_toml_valid(self, tmp_path):
        """load_toml should parse a valid TOML file."""
        toml_file = _write_toml(tmp_path, """\
            [core]
            debug = true
            port = 9000
        """)
        data = load_toml(toml_file)
        assert data["core"]["debug"] is True
        assert data["core"]["port"] == 9000

    def test_load_toml_missing_file(self, tmp_path):
        """load_toml should return empty dict for missing files."""
        result = load_toml(tmp_path / "nonexistent.toml")
        assert result == {}

    def test_load_toml_invalid_raises_config_error(self, tmp_path):
        """load_toml should raise ConfigError for invalid TOML."""
        bad_file = tmp_path / "bad.toml"
        bad_file.write_text("this is not [valid toml = ")
        with pytest.raises(ConfigError, match="Invalid TOML"):
            load_toml(bad_file)

    def test_settings_from_toml_file(self, tmp_path, monkeypatch):
        """TritiumBaseSettings should load values from a TOML file."""
        _clean_env(monkeypatch)
        toml_file = _write_toml(tmp_path, """\
            debug = true
            port = 7777
            site_id = "toml-site"
            log_level = "WARNING"
        """)
        with patch("tritium_lib.config.DEFAULT_CONFIG_FILE", toml_file):
            settings = TritiumBaseSettings()
        assert settings.debug is True
        assert settings.port == 7777
        assert settings.site_id == "toml-site"
        assert settings.log_level == "WARNING"

    def test_env_takes_priority_over_toml(self, tmp_path, monkeypatch):
        """Env vars should take priority over TOML values."""
        _clean_env(monkeypatch)
        toml_file = _write_toml(tmp_path, """\
            port = 7777
            debug = false
        """)
        monkeypatch.setenv("TRITIUM_PORT", "5555")
        monkeypatch.setenv("TRITIUM_DEBUG", "true")
        with patch("tritium_lib.config.DEFAULT_CONFIG_FILE", toml_file):
            settings = TritiumBaseSettings()
        assert settings.port == 5555
        assert settings.debug is True


# ---------------------------------------------------------------------------
# 4. Per-addon config sections
# ---------------------------------------------------------------------------

class TestAddonConfig:
    def test_addon_config_present(self, tmp_path):
        """get_addon_config should return the addon's section."""
        toml_file = _write_toml(tmp_path, """\
            [addons.meshtastic]
            port = "/dev/ttyACM0"
            baud_rate = 115200

            [addons.sdr]
            device = "hackrf"
            gain = 40
        """)
        mesh_cfg = get_addon_config("meshtastic", toml_file)
        assert mesh_cfg["port"] == "/dev/ttyACM0"
        assert mesh_cfg["baud_rate"] == 115200

        sdr_cfg = get_addon_config("sdr", toml_file)
        assert sdr_cfg["device"] == "hackrf"
        assert sdr_cfg["gain"] == 40

    def test_addon_config_missing_addon(self, tmp_path):
        """Missing addon section should return empty dict."""
        toml_file = _write_toml(tmp_path, """\
            [addons.meshtastic]
            port = "/dev/ttyACM0"
        """)
        result = get_addon_config("nonexistent", toml_file)
        assert result == {}

    def test_addon_config_no_addons_section(self, tmp_path):
        """Config with no [addons] section should return empty dict."""
        toml_file = _write_toml(tmp_path, """\
            [core]
            debug = true
        """)
        result = get_addon_config("anything", toml_file)
        assert result == {}

    def test_addon_config_empty_name_raises(self, tmp_path):
        """Empty addon name should raise ConfigError."""
        toml_file = _write_toml(tmp_path, """\
            [addons.x]
            a = 1
        """)
        with pytest.raises(ConfigError, match="non-empty"):
            get_addon_config("", toml_file)
        with pytest.raises(ConfigError, match="non-empty"):
            get_addon_config("   ", toml_file)

    def test_addon_config_missing_file_returns_empty(self, tmp_path):
        """Missing TOML file should return empty dict for addon config."""
        result = get_addon_config("anything", tmp_path / "nope.toml")
        assert result == {}


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_port_too_high(self, monkeypatch):
        """Port > 65535 should fail validation."""
        _clean_env(monkeypatch)
        with pytest.raises(Exception):
            TritiumBaseSettings(port=70000)

    def test_port_zero(self, monkeypatch):
        """Port 0 should fail validation."""
        _clean_env(monkeypatch)
        with pytest.raises(Exception):
            TritiumBaseSettings(port=0)

    def test_invalid_log_level(self, monkeypatch):
        """Invalid log level should fail validation."""
        _clean_env(monkeypatch)
        with pytest.raises(Exception):
            TritiumBaseSettings(log_level="VERBOSE")

    def test_site_id_special_chars_rejected(self, monkeypatch):
        """Site ID with special characters should fail validation."""
        _clean_env(monkeypatch)
        with pytest.raises(Exception):
            TritiumBaseSettings(site_id="bad site!")

    def test_site_id_empty_rejected(self, monkeypatch):
        """Empty site ID should fail validation."""
        _clean_env(monkeypatch)
        with pytest.raises(Exception):
            TritiumBaseSettings(site_id="")

    def test_mqtt_enabled_empty_host_rejected(self, monkeypatch):
        """MQTT enabled with empty host should fail validation."""
        _clean_env(monkeypatch)
        with pytest.raises(Exception):
            TritiumBaseSettings(mqtt_enabled=True, mqtt_host="")

    def test_validate_settings_error_message(self, monkeypatch):
        """validate_settings should raise ConfigError with field details."""
        _clean_env(monkeypatch)
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            validate_settings(TritiumBaseSettings, port=70000)

    def test_validate_settings_success(self, monkeypatch):
        """validate_settings should return a valid instance on success."""
        _clean_env(monkeypatch)
        s = validate_settings(TritiumBaseSettings, port=9090, site_id="test-site")
        assert s.port == 9090
        assert s.site_id == "test-site"


# ---------------------------------------------------------------------------
# 6. Config dump (to_dict)
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_contains_all_fields(self, monkeypatch):
        """to_dict should include all defined fields."""
        _clean_env(monkeypatch)
        settings = TritiumBaseSettings()
        d = settings.to_dict()
        assert "app_name" in d
        assert "debug" in d
        assert "host" in d
        assert "port" in d
        assert "site_id" in d
        assert "log_level" in d
        assert "mqtt_enabled" in d
        assert "mqtt_host" in d
        assert "mqtt_port" in d

    def test_to_dict_masks_password(self, monkeypatch):
        """to_dict should mask password fields by default."""
        _clean_env(monkeypatch)
        settings = TritiumBaseSettings(mqtt_password="super-secret")
        d = settings.to_dict(mask_secrets=True)
        assert d["mqtt_password"] == "***"

    def test_to_dict_no_mask(self, monkeypatch):
        """to_dict with mask_secrets=False should show raw values."""
        _clean_env(monkeypatch)
        settings = TritiumBaseSettings(mqtt_password="visible")
        d = settings.to_dict(mask_secrets=False)
        assert d["mqtt_password"] == "visible"

    def test_to_dict_empty_secret_not_masked(self, monkeypatch):
        """Empty secret fields should stay empty, not become '***'."""
        _clean_env(monkeypatch)
        settings = TritiumBaseSettings(mqtt_password="")
        d = settings.to_dict(mask_secrets=True)
        assert d["mqtt_password"] == ""


# ---------------------------------------------------------------------------
# 7. Source info
# ---------------------------------------------------------------------------

class TestSourceInfo:
    def test_get_source_info_structure(self, monkeypatch):
        """get_source_info should report env prefix and file paths."""
        _clean_env(monkeypatch)
        settings = TritiumBaseSettings()
        info = settings.get_source_info()
        assert info["env_prefix"] == "TRITIUM_"
        assert ".env" in info["env_file"]
        assert "config.toml" in info["toml_file"]


# ---------------------------------------------------------------------------
# 8. TritiumSettings alias
# ---------------------------------------------------------------------------

class TestTritiumSettings:
    def test_is_subclass(self):
        """TritiumSettings should be a subclass of TritiumBaseSettings."""
        assert issubclass(TritiumSettings, TritiumBaseSettings)

    def test_works_identically(self, monkeypatch):
        """TritiumSettings should behave the same as TritiumBaseSettings."""
        _clean_env(monkeypatch)
        s = TritiumSettings()
        assert s.app_name == "Tritium"
        assert s.to_dict()["port"] == 8000


# ---------------------------------------------------------------------------
# 9. Subclass extension
# ---------------------------------------------------------------------------

class TestSubclassing:
    def test_service_specific_fields(self, monkeypatch):
        """Subclasses can add service-specific fields."""
        _clean_env(monkeypatch)

        class MyServiceSettings(TritiumBaseSettings):
            my_feature_enabled: bool = False
            my_feature_url: str = "http://localhost:5000"

        settings = MyServiceSettings(my_feature_enabled=True)
        assert settings.my_feature_enabled is True
        assert settings.app_name == "Tritium"
        d = settings.to_dict()
        assert "my_feature_enabled" in d
        assert "app_name" in d


# ---------------------------------------------------------------------------
# 10. ConfigError
# ---------------------------------------------------------------------------

class TestConfigError:
    def test_config_error_has_field(self):
        """ConfigError can carry a field name."""
        err = ConfigError("port must be > 0", field="port")
        assert err.field == "port"
        assert "port must be > 0" in str(err)

    def test_config_error_no_field(self):
        """ConfigError without a field should still work."""
        err = ConfigError("generic error")
        assert err.field is None
        assert "generic error" in str(err)

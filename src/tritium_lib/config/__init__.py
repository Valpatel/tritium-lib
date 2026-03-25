# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared configuration management for the Tritium ecosystem.

Features:
    - TOML file loading (~/.tritium/config.toml by default)
    - Environment variable overrides (TRITIUM_ prefix)
    - .env file support
    - Per-addon configuration sections
    - Validation with helpful error messages
    - Config dump for debugging (to_dict())

Priority order (highest wins):
    1. Constructor kwargs (init_settings)
    2. Environment variables (TRITIUM_*)
    3. .env file
    4. TOML config file (~/.tritium/config.toml)
    5. Field defaults

Both tritium-sc and tritium-edge extend these base settings
for their specific needs.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Default TOML config file path
DEFAULT_CONFIG_DIR = Path.home() / ".tritium"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.toml"


def _resolve_toml_path(path: str | Path | None = None) -> Path:
    """Resolve the TOML config file path.

    Args:
        path: Explicit path, or None for the default.

    Returns:
        Resolved Path object.
    """
    if path is not None:
        return Path(path).expanduser().resolve()
    return DEFAULT_CONFIG_FILE


def load_toml(path: str | Path | None = None) -> dict[str, Any]:
    """Load a TOML config file and return its contents as a dict.

    Args:
        path: Path to the TOML file. Defaults to ~/.tritium/config.toml.

    Returns:
        Parsed TOML contents, or empty dict if file doesn't exist.

    Raises:
        ConfigError: If the file exists but contains invalid TOML.
    """
    resolved = _resolve_toml_path(path)
    if not resolved.exists():
        return {}
    try:
        with open(resolved, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(
            f"Invalid TOML in {resolved}: {e}"
        ) from e


def get_addon_config(
    addon_name: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load configuration for a specific addon from the TOML file.

    Addon configs live under [addons.<name>] in the TOML file:

        [addons.meshtastic]
        port = "/dev/ttyACM0"
        baud_rate = 115200

    Args:
        addon_name: Name of the addon (e.g., "meshtastic", "sdr").
        path: Path to the TOML file. Defaults to ~/.tritium/config.toml.

    Returns:
        Dict of addon-specific config, or empty dict if not found.

    Raises:
        ConfigError: If addon_name is empty or the TOML file is invalid.
    """
    if not addon_name or not addon_name.strip():
        raise ConfigError("addon_name must be a non-empty string")

    data = load_toml(path)
    addons = data.get("addons", {})
    if not isinstance(addons, dict):
        raise ConfigError(
            "The [addons] section in config.toml must be a table, "
            f"got {type(addons).__name__}"
        )
    return addons.get(addon_name, {})


class ConfigError(Exception):
    """Raised for configuration-related errors with helpful messages."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)


def validate_settings(settings_cls: type[BaseSettings], **kwargs: Any) -> BaseSettings:
    """Create and validate a settings instance with helpful error messages.

    Args:
        settings_cls: The settings class to instantiate.
        **kwargs: Keyword arguments passed to the constructor.

    Returns:
        Validated settings instance.

    Raises:
        ConfigError: With a human-readable description of what went wrong.
    """
    try:
        return settings_cls(**kwargs)
    except ValidationError as e:
        errors = e.errors()
        messages = []
        for err in errors:
            loc = " -> ".join(str(part) for part in err["loc"])
            msg = err["msg"]
            typ = err["type"]
            messages.append(f"  {loc}: {msg} (type={typ})")
        detail = "\n".join(messages)
        raise ConfigError(
            f"Configuration validation failed for {settings_cls.__name__}:\n{detail}"
        ) from e


class TritiumBaseSettings(BaseSettings):
    """Base settings class for all Tritium services.

    Loads configuration from (highest priority first):
        1. Constructor kwargs
        2. Environment variables with TRITIUM_ prefix
        3. .env file in the working directory
        4. TOML file at ~/.tritium/config.toml (or custom path)
        5. Field defaults

    Example TOML (~/.tritium/config.toml):
        [core]
        debug = true
        site_id = "base-alpha"
        host = "0.0.0.0"
        port = 9000

        [mqtt]
        enabled = true
        host = "mqtt.local"
        port = 1883

        [addons.meshtastic]
        port = "/dev/ttyACM0"
    """

    model_config = SettingsConfigDict(
        env_prefix="TRITIUM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Common settings shared across all Tritium services
    app_name: str = "Tritium"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    site_id: str = "home"
    log_level: str = "INFO"

    # MQTT (shared bus)
    mqtt_enabled: bool = False
    mqtt_host: str = "localhost"
    mqtt_port: int = Field(default=1883, ge=1, le=65535)
    mqtt_site_id: str = "home"
    mqtt_username: str = ""
    mqtt_password: str = ""

    # TOML config file path (not loaded from env/toml itself)
    config_file: str | None = None

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(
                f"log_level must be one of {sorted(allowed)}, got '{v}'"
            )
        return upper

    @field_validator("site_id")
    @classmethod
    def _validate_site_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("site_id must be a non-empty string")
        # Only allow alphanumeric, hyphens, underscores
        cleaned = v.strip()
        if not all(c.isalnum() or c in "-_" for c in cleaned):
            raise ValueError(
                f"site_id must contain only alphanumeric characters, "
                f"hyphens, and underscores, got '{v}'"
            )
        return cleaned

    @model_validator(mode="after")
    def _validate_mqtt_consistency(self) -> "TritiumBaseSettings":
        """If MQTT is enabled, host must be non-empty."""
        if self.mqtt_enabled and not self.mqtt_host.strip():
            raise ValueError(
                "mqtt_host must be set when mqtt_enabled is True"
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        """Insert TOML file source below dotenv but above secrets.

        Priority (highest first):
            init_settings > env_settings > dotenv_settings > toml > secrets
        """
        from pydantic_settings import TomlConfigSettingsSource

        # Build TOML source — looks for the file, returns empty if missing
        toml_source = TomlConfigSettingsSource(
            settings_cls,
            toml_file=DEFAULT_CONFIG_FILE,
        )

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            toml_source,
            file_secret_settings,
        )

    def to_dict(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        """Dump all settings as a plain dict for debugging.

        Args:
            mask_secrets: If True, mask fields containing 'password',
                'secret', or 'token' in their name.

        Returns:
            Dictionary of all current setting values.
        """
        secret_keywords = {"password", "secret", "token", "key"}
        data = self.model_dump()
        if mask_secrets:
            for k, v in data.items():
                if any(kw in k.lower() for kw in secret_keywords):
                    if v:
                        data[k] = "***"
        return data

    def get_source_info(self) -> dict[str, str]:
        """Return info about where config was loaded from.

        Returns:
            Dict with keys like 'env_prefix', 'env_file', 'toml_file'.
        """
        return {
            "env_prefix": "TRITIUM_",
            "env_file": str(self.model_config.get("env_file", ".env")),
            "toml_file": str(
                self.config_file or DEFAULT_CONFIG_FILE
            ),
        }


class TritiumSettings(TritiumBaseSettings):
    """Convenience alias — the standard settings class to use.

    Provides all TritiumBaseSettings fields plus the full config
    management system. Subclass this for service-specific settings.
    """

    pass


__all__ = [
    "TritiumBaseSettings",
    "TritiumSettings",
    "ConfigError",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_CONFIG_FILE",
    "get_addon_config",
    "load_toml",
    "validate_settings",
]

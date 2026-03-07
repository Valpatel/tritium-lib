# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared configuration base classes.

Both tritium-sc and tritium-edge extend these base settings
for their specific needs.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class TritiumBaseSettings(BaseSettings):
    """Base settings class for all Tritium services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Common settings shared across all Tritium services
    app_name: str = "Tritium"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # MQTT (shared bus)
    mqtt_enabled: bool = False
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_site_id: str = "home"
    mqtt_username: str = ""
    mqtt_password: str = ""

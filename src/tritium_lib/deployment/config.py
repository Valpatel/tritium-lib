# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Deployment configuration — target host, credentials, components to deploy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComponentType(str, Enum):
    """Known Tritium component types."""

    SC = "sc"
    LIB = "lib"
    EDGE = "edge"
    ADDONS = "addons"
    MQTT = "mqtt"
    DATABASE = "database"


class DeploymentEnvironment(str, Enum):
    """Deployment target environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    FIELD = "field"


@dataclass
class DeploymentComponent:
    """Specification for a single component to deploy.

    Attributes
    ----------
    name:
        Component identifier (e.g., "sc", "mqtt", "edge").
    component_type:
        The type of component from ComponentType enum.
    version:
        Version string to deploy (e.g., "0.1.0").
    port:
        Network port the component listens on (0 = not applicable).
    config_overrides:
        Key-value pairs to override in the component's config.
    enabled:
        Whether this component should be deployed.
    """

    name: str
    component_type: ComponentType = ComponentType.SC
    version: str = "0.1.0"
    port: int = 0
    config_overrides: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "component_type": self.component_type.value,
            "version": self.version,
            "port": self.port,
            "config_overrides": dict(self.config_overrides),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentComponent:
        """Deserialize from a dictionary."""
        return cls(
            name=data["name"],
            component_type=ComponentType(data.get("component_type", "sc")),
            version=data.get("version", "0.1.0"),
            port=data.get("port", 0),
            config_overrides=data.get("config_overrides", {}),
            enabled=data.get("enabled", True),
        )


@dataclass
class DeploymentConfig:
    """Full deployment configuration for a Tritium instance.

    Attributes
    ----------
    name:
        Human-readable name for this deployment (e.g., "field-station-1").
    host:
        Target hostname or IP address.
    environment:
        Target environment (development, staging, production, field).
    components:
        List of component names to deploy. When strings are provided,
        they are auto-converted to DeploymentComponent objects.
    component_specs:
        Detailed component specifications (populated from components).
    credentials:
        Credential key-value pairs (API keys, tokens). Never logged.
    mqtt_broker:
        MQTT broker address for inter-component communication.
    mqtt_port:
        MQTT broker port (default 1883).
    data_dir:
        Root directory for persistent data on the target.
    log_dir:
        Root directory for log files on the target.
    created_at:
        Timestamp when the config was created.
    tags:
        Arbitrary tags for grouping/filtering deployments.
    """

    name: str
    host: str = "localhost"
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    components: list[str] = field(default_factory=list)
    component_specs: list[DeploymentComponent] = field(default_factory=list)
    credentials: dict[str, str] = field(default_factory=dict)
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    data_dir: str = "/var/lib/tritium"
    log_dir: str = "/var/log/tritium"
    created_at: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build component_specs from component name list if not provided."""
        if self.components and not self.component_specs:
            for name in self.components:
                comp_type = ComponentType(name) if name in ComponentType.__members__.values() else ComponentType.SC
                # Assign default ports for known components
                port = _DEFAULT_PORTS.get(name, 0)
                self.component_specs.append(
                    DeploymentComponent(
                        name=name,
                        component_type=comp_type,
                        port=port,
                    )
                )

    @property
    def enabled_components(self) -> list[DeploymentComponent]:
        """Return only enabled component specs."""
        return [c for c in self.component_specs if c.enabled]

    @property
    def component_names(self) -> list[str]:
        """Return names of all component specs."""
        return [c.name for c in self.component_specs]

    def get_component(self, name: str) -> DeploymentComponent | None:
        """Look up a component spec by name."""
        for comp in self.component_specs:
            if comp.name == name:
                return comp
        return None

    def validate(self) -> list[str]:
        """Validate the configuration and return a list of error messages.

        Returns an empty list if the config is valid.
        """
        errors: list[str] = []
        if not self.name:
            errors.append("Deployment name is required")
        if not self.host:
            errors.append("Host is required")
        if not self.component_specs:
            errors.append("At least one component must be specified")
        if self.mqtt_port < 1 or self.mqtt_port > 65535:
            errors.append(f"Invalid MQTT port: {self.mqtt_port}")
        # Check for duplicate component names
        names = [c.name for c in self.component_specs]
        if len(names) != len(set(names)):
            errors.append("Duplicate component names found")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Credentials are intentionally excluded for safety.
        """
        return {
            "name": self.name,
            "host": self.host,
            "environment": self.environment.value,
            "components": list(self.components),
            "component_specs": [c.to_dict() for c in self.component_specs],
            "mqtt_broker": self.mqtt_broker,
            "mqtt_port": self.mqtt_port,
            "data_dir": self.data_dir,
            "log_dir": self.log_dir,
            "created_at": self.created_at,
            "tags": dict(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentConfig:
        """Deserialize from a dictionary."""
        specs = [
            DeploymentComponent.from_dict(s)
            for s in data.get("component_specs", [])
        ]
        return cls(
            name=data["name"],
            host=data.get("host", "localhost"),
            environment=DeploymentEnvironment(
                data.get("environment", "development")
            ),
            components=data.get("components", []),
            component_specs=specs,
            mqtt_broker=data.get("mqtt_broker", "localhost"),
            mqtt_port=data.get("mqtt_port", 1883),
            data_dir=data.get("data_dir", "/var/lib/tritium"),
            log_dir=data.get("log_dir", "/var/log/tritium"),
            created_at=data.get("created_at", time.time()),
            tags=data.get("tags", {}),
        )


# Default ports for known component types
_DEFAULT_PORTS: dict[str, int] = {
    "sc": 8000,
    "mqtt": 1883,
    "database": 5432,
    "edge": 0,
    "lib": 0,
    "addons": 0,
}

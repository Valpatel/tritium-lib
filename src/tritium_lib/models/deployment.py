# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Deployment models — track what services are running where and what
each host needs to operate the Tritium system."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ServiceName(str, Enum):
    """Well-known Tritium service names."""
    SC_SERVER = "sc_server"
    MQTT_BROKER = "mqtt_broker"
    MESHTASTIC_BRIDGE = "meshtastic_bridge"
    OLLAMA = "ollama"
    EDGE_FLEET_SERVER = "edge_fleet_server"
    ROS2_BRIDGE = "ros2_bridge"
    GO2RTC = "go2rtc"


class ServiceState(str, Enum):
    """Runtime state of a service."""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"
    UNKNOWN = "unknown"


class ServiceStatus(BaseModel):
    """Runtime status of a single service."""
    name: str
    display_name: str = ""
    state: ServiceState = ServiceState.UNKNOWN
    pid: Optional[int] = None
    uptime_s: float = 0.0
    port: Optional[int] = None
    version: str = ""
    error_message: str = ""
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    can_start: bool = False
    can_stop: bool = False
    start_command: str = ""
    stop_command: str = ""
    log_file: str = ""


class SystemRequirements(BaseModel):
    """What a host needs to run Tritium."""
    python_version: str = "3.12+"
    system_packages: list[str] = Field(default_factory=lambda: [
        "mosquitto",
        "git",
        "python3-venv",
        "python3-pip",
        "libopencv-dev",
    ])
    python_packages: list[str] = Field(default_factory=lambda: [
        "fastapi",
        "uvicorn",
        "paho-mqtt",
        "pydantic",
        "sqlalchemy",
        "aiosqlite",
    ])
    optional_packages: list[str] = Field(default_factory=lambda: [
        "ollama",
        "platformio",
        "meshtastic",
    ])
    min_ram_mb: int = 2048
    min_disk_mb: int = 4096
    ports_needed: list[int] = Field(default_factory=lambda: [
        8000,   # SC server
        1883,   # MQTT broker
        8080,   # Fleet server
    ])


class DeployedService(BaseModel):
    """A service deployed on a specific host."""
    service: str
    host: str = "localhost"
    status: ServiceStatus = Field(default_factory=lambda: ServiceStatus(name="unknown"))
    config_path: str = ""
    installed: bool = False
    autostart: bool = False


class DeploymentConfig(BaseModel):
    """Full deployment configuration for a Tritium installation."""
    site_id: str = "default"
    hostname: str = ""
    services: list[DeployedService] = Field(default_factory=list)
    requirements: SystemRequirements = Field(default_factory=SystemRequirements)
    edge_devices: int = 0
    last_deploy: Optional[datetime] = None
    notes: str = ""

    def service_by_name(self, name: str) -> Optional[DeployedService]:
        """Find a deployed service by name."""
        for svc in self.services:
            if svc.service == name:
                return svc
        return None

    def all_running(self) -> bool:
        """Check if all deployed services are running."""
        return all(
            s.status.state == ServiceState.RUNNING
            for s in self.services
            if s.installed
        )

    def summary(self) -> dict:
        """Quick summary of deployment state."""
        installed = [s for s in self.services if s.installed]
        running = [s for s in installed if s.status.state == ServiceState.RUNNING]
        return {
            "site_id": self.site_id,
            "hostname": self.hostname,
            "total_services": len(self.services),
            "installed": len(installed),
            "running": len(running),
            "edge_devices": self.edge_devices,
            "healthy": len(running) == len(installed) and len(installed) > 0,
        }

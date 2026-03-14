# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for deployment models."""

import pytest

from tritium_lib.models.deployment import (
    DeployedService,
    DeploymentConfig,
    ServiceName,
    ServiceState,
    ServiceStatus,
    SystemRequirements,
)


class TestServiceStatus:
    def test_default_state(self):
        s = ServiceStatus(name="test")
        assert s.state == ServiceState.UNKNOWN
        assert s.pid is None
        assert s.uptime_s == 0.0

    def test_running_service(self):
        s = ServiceStatus(
            name="mqtt_broker",
            display_name="MQTT Broker",
            state=ServiceState.RUNNING,
            pid=12345,
            port=1883,
            uptime_s=3600.0,
        )
        assert s.state == ServiceState.RUNNING
        assert s.pid == 12345
        assert s.port == 1883

    def test_error_state(self):
        s = ServiceStatus(
            name="broken",
            state=ServiceState.ERROR,
            error_message="Connection refused",
        )
        assert s.state == ServiceState.ERROR
        assert s.error_message == "Connection refused"


class TestServiceName:
    def test_enum_values(self):
        assert ServiceName.SC_SERVER == "sc_server"
        assert ServiceName.MQTT_BROKER == "mqtt_broker"
        assert ServiceName.MESHTASTIC_BRIDGE == "meshtastic_bridge"
        assert ServiceName.OLLAMA == "ollama"

    def test_all_values(self):
        names = [e.value for e in ServiceName]
        assert len(names) == 7
        assert "sc_server" in names


class TestServiceState:
    def test_enum_values(self):
        assert ServiceState.RUNNING == "running"
        assert ServiceState.STOPPED == "stopped"
        assert ServiceState.ERROR == "error"
        assert ServiceState.STARTING == "starting"
        assert ServiceState.UNKNOWN == "unknown"


class TestSystemRequirements:
    def test_defaults(self):
        req = SystemRequirements()
        assert req.python_version == "3.12+"
        assert "mosquitto" in req.system_packages
        assert "fastapi" in req.python_packages
        assert req.min_ram_mb == 2048
        assert 8000 in req.ports_needed
        assert 1883 in req.ports_needed

    def test_custom(self):
        req = SystemRequirements(
            min_ram_mb=4096,
            system_packages=["custom-pkg"],
        )
        assert req.min_ram_mb == 4096
        assert req.system_packages == ["custom-pkg"]


class TestDeployedService:
    def test_defaults(self):
        svc = DeployedService(service="mqtt_broker")
        assert svc.host == "localhost"
        assert svc.installed is False
        assert svc.autostart is False

    def test_installed(self):
        svc = DeployedService(
            service="sc_server",
            host="192.168.1.10",
            installed=True,
            autostart=True,
        )
        assert svc.installed is True
        assert svc.host == "192.168.1.10"


class TestDeploymentConfig:
    def test_empty(self):
        cfg = DeploymentConfig()
        assert cfg.site_id == "default"
        assert cfg.services == []
        assert cfg.all_running() is True  # vacuously true

    def test_with_services(self):
        cfg = DeploymentConfig(
            site_id="hq",
            hostname="tritium-01",
            services=[
                DeployedService(
                    service="sc_server",
                    installed=True,
                    status=ServiceStatus(name="sc_server", state=ServiceState.RUNNING),
                ),
                DeployedService(
                    service="mqtt_broker",
                    installed=True,
                    status=ServiceStatus(name="mqtt_broker", state=ServiceState.STOPPED),
                ),
            ],
        )
        assert cfg.all_running() is False

    def test_all_running(self):
        cfg = DeploymentConfig(
            services=[
                DeployedService(
                    service="sc_server",
                    installed=True,
                    status=ServiceStatus(name="sc_server", state=ServiceState.RUNNING),
                ),
                DeployedService(
                    service="mqtt_broker",
                    installed=True,
                    status=ServiceStatus(name="mqtt_broker", state=ServiceState.RUNNING),
                ),
            ],
        )
        assert cfg.all_running() is True

    def test_service_by_name(self):
        cfg = DeploymentConfig(
            services=[
                DeployedService(service="sc_server"),
                DeployedService(service="mqtt_broker"),
            ],
        )
        assert cfg.service_by_name("sc_server") is not None
        assert cfg.service_by_name("sc_server").service == "sc_server"
        assert cfg.service_by_name("nonexistent") is None

    def test_summary(self):
        cfg = DeploymentConfig(
            site_id="test",
            hostname="box-01",
            edge_devices=3,
            services=[
                DeployedService(
                    service="sc_server",
                    installed=True,
                    status=ServiceStatus(name="sc_server", state=ServiceState.RUNNING),
                ),
                DeployedService(
                    service="mqtt_broker",
                    installed=True,
                    status=ServiceStatus(name="mqtt_broker", state=ServiceState.RUNNING),
                ),
                DeployedService(
                    service="ollama",
                    installed=False,
                ),
            ],
        )
        summary = cfg.summary()
        assert summary["site_id"] == "test"
        assert summary["hostname"] == "box-01"
        assert summary["installed"] == 2
        assert summary["running"] == 2
        assert summary["edge_devices"] == 3
        assert summary["healthy"] is True

    def test_summary_unhealthy(self):
        cfg = DeploymentConfig(
            services=[
                DeployedService(
                    service="sc_server",
                    installed=True,
                    status=ServiceStatus(name="sc_server", state=ServiceState.RUNNING),
                ),
                DeployedService(
                    service="mqtt_broker",
                    installed=True,
                    status=ServiceStatus(name="mqtt_broker", state=ServiceState.STOPPED),
                ),
            ],
        )
        summary = cfg.summary()
        assert summary["running"] == 1
        assert summary["healthy"] is False


class TestSerialization:
    def test_round_trip(self):
        cfg = DeploymentConfig(
            site_id="hq",
            hostname="tritium-01",
            services=[
                DeployedService(
                    service="sc_server",
                    installed=True,
                    status=ServiceStatus(
                        name="sc_server",
                        state=ServiceState.RUNNING,
                        pid=1234,
                    ),
                ),
            ],
        )
        data = cfg.model_dump()
        restored = DeploymentConfig(**data)
        assert restored.site_id == "hq"
        assert restored.services[0].service == "sc_server"
        assert restored.services[0].status.pid == 1234

    def test_json_round_trip(self):
        cfg = DeploymentConfig(site_id="test")
        json_str = cfg.model_dump_json()
        restored = DeploymentConfig.model_validate_json(json_str)
        assert restored.site_id == "test"

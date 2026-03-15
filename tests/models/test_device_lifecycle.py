# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for device lifecycle models."""

import pytest

from tritium_lib.models.device_lifecycle import (
    DeviceLifecycleEvent,
    DeviceLifecycleStatus,
    DeviceProvisioningConfig,
    DeviceState,
    FleetLifecycleSummary,
    VALID_TRANSITIONS,
    is_valid_transition,
)


class TestDeviceState:
    def test_all_states_exist(self):
        assert DeviceState.PROVISIONING == "provisioning"
        assert DeviceState.ACTIVE == "active"
        assert DeviceState.MAINTENANCE == "maintenance"
        assert DeviceState.RETIRED == "retired"
        assert DeviceState.ERROR == "error"

    def test_valid_transition_provisioning_to_active(self):
        assert is_valid_transition(DeviceState.PROVISIONING, DeviceState.ACTIVE)

    def test_valid_transition_active_to_maintenance(self):
        assert is_valid_transition(DeviceState.ACTIVE, DeviceState.MAINTENANCE)

    def test_invalid_transition_active_to_provisioning(self):
        assert not is_valid_transition(DeviceState.ACTIVE, DeviceState.PROVISIONING)

    def test_retired_can_reprovision(self):
        assert is_valid_transition(DeviceState.RETIRED, DeviceState.PROVISIONING)

    def test_error_to_maintenance(self):
        assert is_valid_transition(DeviceState.ERROR, DeviceState.MAINTENANCE)

    def test_all_states_have_transitions(self):
        for state in DeviceState:
            assert state in VALID_TRANSITIONS


class TestDeviceLifecycleEvent:
    def test_create_event(self):
        evt = DeviceLifecycleEvent(
            device_id="tritium-01",
            from_state=DeviceState.PROVISIONING,
            to_state=DeviceState.ACTIVE,
            reason="Initial setup complete",
            operator="admin",
        )
        assert evt.device_id == "tritium-01"
        assert evt.from_state == DeviceState.PROVISIONING
        assert evt.to_state == DeviceState.ACTIVE
        assert evt.timestamp is not None

    def test_default_fields(self):
        evt = DeviceLifecycleEvent(
            device_id="x",
            from_state=DeviceState.ACTIVE,
            to_state=DeviceState.MAINTENANCE,
        )
        assert evt.reason == ""
        assert evt.operator == ""


class TestDeviceProvisioningConfig:
    def test_create_config(self):
        cfg = DeviceProvisioningConfig(
            device_id="tritium-02",
            device_name="Perimeter Node 2",
            device_group="perimeter",
            firmware_target="1.2.0",
        )
        assert cfg.device_id == "tritium-02"
        assert cfg.device_group == "perimeter"
        assert cfg.site_id == "home"

    def test_optional_location(self):
        cfg = DeviceProvisioningConfig(
            device_id="x",
            lat=40.7128,
            lng=-74.0060,
        )
        assert cfg.lat == 40.7128


class TestDeviceLifecycleStatus:
    def test_defaults(self):
        status = DeviceLifecycleStatus(device_id="tritium-01")
        assert status.state == DeviceState.PROVISIONING
        assert status.transition_count == 0
        assert status.history == []


class TestFleetLifecycleSummary:
    def test_from_states(self):
        states = [
            DeviceState.ACTIVE,
            DeviceState.ACTIVE,
            DeviceState.MAINTENANCE,
            DeviceState.PROVISIONING,
            DeviceState.ERROR,
        ]
        summary = FleetLifecycleSummary.from_states(states)
        assert summary.total == 5
        assert summary.active == 2
        assert summary.maintenance == 1
        assert summary.provisioning == 1
        assert summary.error == 1
        assert summary.retired == 0

    def test_empty(self):
        summary = FleetLifecycleSummary.from_states([])
        assert summary.total == 0

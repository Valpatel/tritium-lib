# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for provisioning models."""

from datetime import datetime, timezone

from tritium_lib.models.provision import (
    FleetProvisionStatus,
    ProvisionData,
    ProvisionRecord,
    ProvisionSource,
    ProvisionState,
    compute_provision_status,
    validate_provision_data,
)


class TestProvisionData:
    def test_has_wifi(self):
        data = ProvisionData(wifi_ssid="MyNet", wifi_password="12345678")
        assert data.has_wifi

    def test_no_wifi(self):
        data = ProvisionData()
        assert not data.has_wifi

    def test_has_tls(self):
        data = ProvisionData(ca_pem="-----BEGIN CERTIFICATE-----\nMII...\n-----END CERTIFICATE-----")
        assert data.has_tls

    def test_no_tls(self):
        data = ProvisionData()
        assert not data.has_tls


class TestProvisionRecord:
    def test_active(self):
        r = ProvisionRecord(
            device_id="dev-1",
            source=ProvisionSource.WEB_PORTAL,
            state=ProvisionState.COMMISSIONED,
        )
        assert r.is_active
        assert not r.is_pending

    def test_pending(self):
        r = ProvisionRecord(
            device_id="dev-2",
            source=ProvisionSource.AUTO,
            state=ProvisionState.DISCOVERED,
        )
        assert not r.is_active
        assert r.is_pending

    def test_pending_state(self):
        r = ProvisionRecord(
            device_id="dev-3",
            source=ProvisionSource.BLE,
            state=ProvisionState.PENDING,
        )
        assert r.is_pending

    def test_decommissioned(self):
        r = ProvisionRecord(
            device_id="dev-4",
            source=ProvisionSource.MANUAL,
            state=ProvisionState.DECOMMISSIONED,
        )
        assert not r.is_active
        assert not r.is_pending


class TestComputeProvisionStatus:
    def test_empty(self):
        status = compute_provision_status([])
        assert status.total_devices == 0
        assert status.active_ratio == 1.0
        assert status.needs_attention == 0

    def test_all_commissioned(self):
        records = [
            ProvisionRecord("d1", ProvisionSource.WEB_PORTAL, ProvisionState.COMMISSIONED),
            ProvisionRecord("d2", ProvisionSource.BLE, ProvisionState.COMMISSIONED),
        ]
        status = compute_provision_status(records)
        assert status.total_devices == 2
        assert status.commissioned == 2
        assert status.active_ratio == 1.0
        assert status.needs_attention == 0

    def test_mixed_states(self):
        records = [
            ProvisionRecord("d1", ProvisionSource.WEB_PORTAL, ProvisionState.COMMISSIONED),
            ProvisionRecord("d2", ProvisionSource.AUTO, ProvisionState.DISCOVERED),
            ProvisionRecord("d3", ProvisionSource.BLE, ProvisionState.PENDING),
            ProvisionRecord("d4", ProvisionSource.MANUAL, ProvisionState.SUSPENDED),
            ProvisionRecord("d5", ProvisionSource.USB_SERIAL, ProvisionState.DECOMMISSIONED),
        ]
        status = compute_provision_status(records)
        assert status.total_devices == 5
        assert status.commissioned == 1
        assert status.discovered == 1
        assert status.pending == 1
        assert status.suspended == 1
        assert status.decommissioned == 1
        assert status.active_ratio == 0.2
        assert status.needs_attention == 2

    def test_devices_included(self):
        records = [
            ProvisionRecord("d1", ProvisionSource.SD_CARD, ProvisionState.COMMISSIONED),
        ]
        status = compute_provision_status(records)
        assert len(status.devices) == 1
        assert status.devices[0].device_id == "d1"


class TestValidateProvisionData:
    def test_valid_data(self):
        data = ProvisionData(
            wifi_ssid="MyNet",
            wifi_password="securepass123",
            server_url="http://192.168.1.1:8080",
        )
        assert validate_provision_data(data) == []

    def test_bad_server_url(self):
        data = ProvisionData(server_url="ftp://bad")
        issues = validate_provision_data(data)
        assert len(issues) == 1
        assert "server_url" in issues[0]

    def test_short_password(self):
        data = ProvisionData(wifi_password="short")
        issues = validate_provision_data(data)
        assert len(issues) == 1
        assert "wifi_password" in issues[0]

    def test_bad_cert(self):
        data = ProvisionData(ca_pem="not a cert")
        issues = validate_provision_data(data)
        assert len(issues) == 1
        assert "ca_pem" in issues[0]

    def test_multiple_issues(self):
        data = ProvisionData(
            server_url="ftp://bad",
            wifi_password="short",
            ca_pem="nope",
        )
        issues = validate_provision_data(data)
        assert len(issues) == 3

    def test_empty_data_valid(self):
        data = ProvisionData()
        assert validate_provision_data(data) == []

    def test_https_url_valid(self):
        data = ProvisionData(server_url="https://fleet.example.com")
        assert validate_provision_data(data) == []

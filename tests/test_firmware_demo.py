# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the firmware management demo."""

import json
import threading
import time
from http.server import HTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest

from tritium_lib.firmware.base import FlashStatus, DeviceDetection
from tritium_lib.firmware.demos.firmware_demo import (
    AVAILABLE_FIRMWARE,
    SEED_DEVICES,
    FirmwareFleetManager,
    SimulatedDevice,
    FirmwareDemoHandler,
    _generate_mac,
    _generate_ip,
)


# ---------------------------------------------------------------------------
# SimulatedDevice tests
# ---------------------------------------------------------------------------


class TestSimulatedDevice:
    def test_create(self):
        dev = SimulatedDevice(
            device_id="test-1",
            name="Test Node",
            chip="ESP32-S3",
            board="touch-lcd-43c-box",
            firmware_version="3.2.0",
            firmware_project="tritium-os",
            ip_address="192.168.1.100",
            mac_address="30:AE:A4:11:22:33",
            flash_size="16MB",
        )
        assert dev.device_id == "test-1"
        assert dev.online is True
        assert dev.flash_status == FlashStatus.PENDING
        assert dev.flash_progress == 0.0

    def test_to_dict_basic(self):
        dev = SimulatedDevice(
            device_id="test-2",
            name="Test",
            chip="ESP32-S3",
            board="touch-lcd-43c-box",
            firmware_version="3.2.0",
            firmware_project="tritium-os",
            ip_address="192.168.1.101",
            mac_address="30:AE:A4:44:55:66",
            flash_size="16MB",
        )
        d = dev.to_dict()
        assert d["device_id"] == "test-2"
        assert d["chip"] == "ESP32-S3"
        assert d["firmware_version"] == "3.2.0"
        assert d["online"] is True
        # No flash job active — should not have 'flash' key
        assert "flash" not in d

    def test_to_dict_with_flash_job(self):
        dev = SimulatedDevice(
            device_id="test-3",
            name="Flashing Node",
            chip="ESP32-S3",
            board="touch-lcd-43c-box",
            firmware_version="3.2.0",
            firmware_project="tritium-os",
            ip_address="192.168.1.102",
            mac_address="30:AE:A4:77:88:99",
            flash_size="16MB",
            flash_job_id="flash-abc123",
            flash_status=FlashStatus.WRITING,
            flash_progress=55.0,
            flash_target_version="3.2.1",
            flash_start_time=time.time() - 5.0,
        )
        d = dev.to_dict()
        assert "flash" in d
        assert d["flash"]["job_id"] == "flash-abc123"
        assert d["flash"]["status"] == "writing"
        assert d["flash"]["progress"] == 55.0
        assert d["flash"]["target_version"] == "3.2.1"
        assert d["flash"]["elapsed_s"] > 0

    def test_detection(self):
        dev = SimulatedDevice(
            device_id="test-4",
            name="Detection Test",
            chip="ESP32-S3",
            board="touch-lcd-43c-box",
            firmware_version="3.2.0",
            firmware_project="tritium-os",
            ip_address="192.168.1.103",
            mac_address="30:AE:A4:AA:BB:CC",
            flash_size="16MB",
        )
        det = dev.detection()
        assert isinstance(det, DeviceDetection)
        assert det.detected is True
        assert det.chip == "ESP32-S3"
        assert det.firmware_version == "3.2.0"
        assert det.board == "touch-lcd-43c-box"
        assert "192.168.1.103" in det.port


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_generate_mac(self):
        mac = _generate_mac()
        assert mac.startswith("30:AE:A4:")
        assert len(mac) == 17
        # All parts should be valid hex
        parts = mac.split(":")
        assert len(parts) == 6
        for p in parts:
            int(p, 16)

    def test_generate_mac_unique(self):
        macs = {_generate_mac() for _ in range(20)}
        # With 3 random bytes, collisions are extremely unlikely
        assert len(macs) >= 15

    def test_generate_ip(self):
        ip = _generate_ip(100)
        parts = ip.split(".")
        assert parts[0] == "192"
        assert parts[1] == "168"
        assert parts[2] == "1"
        host = int(parts[3])
        assert 100 <= host <= 150


# ---------------------------------------------------------------------------
# Firmware catalog tests
# ---------------------------------------------------------------------------


class TestFirmwareCatalog:
    def test_seed_data_exists(self):
        assert len(AVAILABLE_FIRMWARE) >= 4
        assert len(SEED_DEVICES) >= 6

    def test_firmware_has_required_fields(self):
        for fw in AVAILABLE_FIRMWARE:
            assert "version" in fw
            assert "board" in fw
            assert "size" in fw
            assert "sha256" in fw
            assert "channel" in fw

    def test_seed_devices_have_required_fields(self):
        for dev in SEED_DEVICES:
            assert "device_id" in dev
            assert "name" in dev
            assert "chip" in dev
            assert "board" in dev
            assert "firmware_version" in dev
            assert "firmware_project" in dev


# ---------------------------------------------------------------------------
# FirmwareFleetManager tests
# ---------------------------------------------------------------------------


class TestFirmwareFleetManager:
    def test_create(self):
        fm = FirmwareFleetManager()
        assert len(fm.devices) == 0
        assert len(fm.firmware_catalog) == len(AVAILABLE_FIRMWARE)
        assert len(fm.flash_history) == 0

    def test_discover_devices(self):
        fm = FirmwareFleetManager()
        devices = fm.discover_devices()
        assert len(devices) == len(SEED_DEVICES)
        for dev in devices:
            assert isinstance(dev, SimulatedDevice)
            assert dev.online is True
            assert dev.ip_address.startswith("192.168.1.")

    def test_discover_devices_idempotent(self):
        fm = FirmwareFleetManager()
        d1 = fm.discover_devices()
        d2 = fm.discover_devices()
        assert len(d1) == len(d2)
        # Same device IDs
        ids1 = {d.device_id for d in d1}
        ids2 = {d.device_id for d in d2}
        assert ids1 == ids2

    def test_get_device(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        dev = fm.get_device("edge-alpha")
        assert dev is not None
        assert dev.name == "Alpha Node (Front Gate)"

    def test_get_device_not_found(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        assert fm.get_device("nonexistent") is None

    def test_get_available_firmware(self):
        fm = FirmwareFleetManager()
        all_fw = fm.get_available_firmware()
        assert len(all_fw) == len(AVAILABLE_FIRMWARE)

    def test_get_available_firmware_by_board(self):
        fm = FirmwareFleetManager()
        fw = fm.get_available_firmware(board="tlora-pager")
        assert len(fw) >= 1
        for f in fw:
            assert f["board"] in ("tlora-pager", "any")

    def test_get_available_firmware_by_channel(self):
        fm = FirmwareFleetManager()
        stable = fm.get_available_firmware(channel="stable")
        beta = fm.get_available_firmware(channel="beta")
        assert all(f["channel"] == "stable" for f in stable)
        assert all(f["channel"] == "beta" for f in beta)
        assert len(stable) + len(beta) == len(AVAILABLE_FIRMWARE)

    def test_check_updates(self):
        fm = FirmwareFleetManager()
        updates = fm.check_updates()
        assert isinstance(updates, list)
        # At least some devices should be outdated
        assert len(updates) > 0
        for u in updates:
            assert "device_id" in u
            assert "current_version" in u
            assert "available_version" in u
            assert "update_available" in u

    def test_version_newer(self):
        assert FirmwareFleetManager._version_newer("3.2.1", "3.2.0") is True
        assert FirmwareFleetManager._version_newer("3.2.0", "3.2.1") is False
        assert FirmwareFleetManager._version_newer("3.2.0", "3.2.0") is False
        assert FirmwareFleetManager._version_newer("2.5.19", "2.5.18") is True
        assert FirmwareFleetManager._version_newer("3.0.0", "2.99.99") is True
        assert FirmwareFleetManager._version_newer("1.0.0", "1.0.0") is False

    def test_version_newer_with_prerelease(self):
        # Pre-release suffix is stripped before comparison
        assert FirmwareFleetManager._version_newer("3.3.0-beta.1", "3.2.1") is True
        assert FirmwareFleetManager._version_newer("3.2.0-rc1", "3.2.1") is False

    def test_summary(self):
        fm = FirmwareFleetManager()
        s = fm.summary()
        assert s["total_devices"] == len(SEED_DEVICES)
        assert s["online"] == len(SEED_DEVICES)
        assert s["firmware_versions_available"] == len(AVAILABLE_FIRMWARE)
        assert s["up_to_date"] + s["outdated"] + s["flashing"] == s["total_devices"]

    def test_start_flash_success(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        # edge-bravo is on 3.1.0, latest is 3.2.1
        result = fm.start_flash("edge-bravo", "3.2.1")
        assert "error" not in result
        assert "job_id" in result
        assert result["device_id"] == "edge-bravo"
        assert result["target_version"] == "3.2.1"

    def test_start_flash_nonexistent_device(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        result = fm.start_flash("nonexistent")
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_start_flash_same_version(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        # edge-charlie is already on 3.2.1
        result = fm.start_flash("edge-charlie", "3.2.1")
        assert "error" in result
        assert "already running" in result["error"].lower()

    def test_start_flash_invalid_version(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        result = fm.start_flash("edge-alpha", "99.99.99")
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_start_flash_auto_version(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        result = fm.start_flash("edge-bravo")
        assert "error" not in result
        assert result["target_version"] != ""

    def test_get_flash_status_idle(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        status = fm.get_flash_status("edge-alpha")
        assert status is not None
        assert status["status"] == "idle"

    def test_get_flash_status_not_found(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        assert fm.get_flash_status("nonexistent") is None

    def test_flash_completes(self):
        """Start a flash and wait for it to complete."""
        fm = FirmwareFleetManager()
        fm.discover_devices()
        result = fm.start_flash("edge-bravo", "3.2.1")
        assert "error" not in result
        job_id = result["job_id"]

        # Wait up to 15 seconds for flash to complete
        deadline = time.time() + 15
        while time.time() < deadline:
            status = fm.get_flash_status("edge-bravo")
            assert status is not None
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        status = fm.get_flash_status("edge-bravo")
        assert status is not None
        # The flash should have finished (completed or failed due to 5% random fail)
        assert status["status"] in ("completed", "failed")

        if status["status"] == "completed":
            # Version should be updated
            dev = fm.get_device("edge-bravo")
            assert dev is not None
            assert dev.firmware_version == "3.2.1"
            assert dev.uptime_s == 0.0  # Rebooted

    def test_flash_double_start_rejected(self):
        """Starting a flash while one is in progress should be rejected."""
        fm = FirmwareFleetManager()
        fm.discover_devices()
        r1 = fm.start_flash("edge-bravo", "3.2.1")
        assert "error" not in r1

        # Wait a tiny bit for the flash thread to start
        time.sleep(0.2)

        # Try to start another flash on the same device
        r2 = fm.start_flash("edge-bravo", "3.2.1")
        assert "error" in r2
        assert "already" in r2["error"].lower()

        # Wait for the first flash to finish
        deadline = time.time() + 15
        while time.time() < deadline:
            status = fm.get_flash_status("edge-bravo")
            if status and status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

    def test_flash_history_recorded(self):
        fm = FirmwareFleetManager()
        fm.discover_devices()
        fm.start_flash("edge-echo", "3.2.1")

        # Wait for completion
        deadline = time.time() + 15
        while time.time() < deadline:
            status = fm.get_flash_status("edge-echo")
            if status and status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert len(fm.flash_history) >= 1
        entry = fm.flash_history[-1]
        assert entry["device_id"] == "edge-echo"
        assert entry["target_version"] == "3.2.1"
        assert "success" in entry
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_server():
    """Start the demo HTTP server on a random port for testing."""
    import tritium_lib.firmware.demos.firmware_demo as demo_mod

    demo_mod.fleet = FirmwareFleetManager()
    demo_mod.fleet.discover_devices()

    server = HTTPServer(("127.0.0.1", 0), FirmwareDemoHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestHTTPEndpoints:
    def test_get_root(self, demo_server):
        resp = urlopen(f"{demo_server}/")
        assert resp.status == 200
        html = resp.read().decode()
        assert "TRITIUM FIRMWARE MANAGER" in html
        assert "device-grid" in html

    def test_get_devices(self, demo_server):
        resp = urlopen(f"{demo_server}/devices")
        data = json.loads(resp.read())
        assert isinstance(data, list)
        assert len(data) == len(SEED_DEVICES)
        for dev in data:
            assert "device_id" in dev
            assert "firmware_version" in dev
            assert "chip" in dev

    def test_get_firmware(self, demo_server):
        resp = urlopen(f"{demo_server}/firmware")
        data = json.loads(resp.read())
        assert isinstance(data, list)
        assert len(data) == len(AVAILABLE_FIRMWARE)

    def test_get_firmware_filtered(self, demo_server):
        resp = urlopen(f"{demo_server}/firmware?channel=beta")
        data = json.loads(resp.read())
        assert all(f["channel"] == "beta" for f in data)

    def test_get_health(self, demo_server):
        resp = urlopen(f"{demo_server}/health")
        data = json.loads(resp.read())
        assert "total_devices" in data
        assert "online" in data
        assert "outdated" in data
        assert data["total_devices"] == len(SEED_DEVICES)

    def test_get_updates(self, demo_server):
        resp = urlopen(f"{demo_server}/updates")
        data = json.loads(resp.read())
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_flash_status_idle(self, demo_server):
        resp = urlopen(f"{demo_server}/flash/edge-alpha/status")
        data = json.loads(resp.read())
        assert data["device_id"] == "edge-alpha"
        assert "status" in data

    def test_get_flash_status_not_found(self, demo_server):
        try:
            urlopen(f"{demo_server}/flash/nonexistent/status")
            assert False, "Expected 404"
        except HTTPError as e:
            assert e.code == 404

    def test_post_flash(self, demo_server):
        req = Request(
            f"{demo_server}/flash/edge-bravo",
            data=json.dumps({"version": "3.2.1"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req)
        data = json.loads(resp.read())
        assert "job_id" in data or "error" in data

    def test_post_flash_bad_version(self, demo_server):
        req = Request(
            f"{demo_server}/flash/edge-alpha",
            data=json.dumps({"version": "99.99.99"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urlopen(req)
            data = json.loads(resp.read())
            # Could be 400 or 200 with error field
            assert "error" in data
        except HTTPError as e:
            assert e.code == 400

    def test_404_unknown_path(self, demo_server):
        try:
            urlopen(f"{demo_server}/nonexistent")
            assert False, "Expected 404"
        except HTTPError as e:
            assert e.code == 404

    def test_post_flash_no_body(self, demo_server):
        """POST with no body should use auto-detected latest version."""
        req = Request(
            f"{demo_server}/flash/edge-echo",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req)
        data = json.loads(resp.read())
        # Should either succeed (auto-detect version) or error (already flashing/up to date)
        assert "job_id" in data or "error" in data

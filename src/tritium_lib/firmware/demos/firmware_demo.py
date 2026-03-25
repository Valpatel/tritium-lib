# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Firmware management demo — device discovery, OTA updates, flash progress tracking.

Self-contained simulation of the firmware management workflow:
  1. Simulated ESP32 device discovery on the network
  2. Firmware version comparison (current vs available)
  3. Simulated OTA flash progress with realistic phase timing
  4. REST API for fleet-wide firmware management

No real hardware required — all devices and flash operations are simulated.

Run with:
    PYTHONPATH=src python3 -m tritium_lib.firmware.demos.firmware_demo

Endpoints:
    GET  /               — HTML dashboard with live device status
    GET  /devices        — All discovered devices with firmware info
    POST /flash/{id}     — Start OTA flash for a device (JSON body: {"version": "..."})
    GET  /flash/{id}/status — Flash progress for a device
    GET  /firmware       — Available firmware versions
    GET  /health         — Service health summary
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

from tritium_lib.firmware.base import (
    DeviceDetection,
    FlashResult,
    FlashStatus,
)
from tritium_lib.models.firmware import FirmwareMeta, OTAJob, OTAStatus


# ---------------------------------------------------------------------------
# Firmware catalog — available versions for OTA
# ---------------------------------------------------------------------------

AVAILABLE_FIRMWARE: list[dict[str, Any]] = [
    {
        "version": "3.2.1",
        "board": "any",
        "family": "esp32",
        "size": 1_572_864,
        "sha256": hashlib.sha256(b"tritium-os-3.2.1").hexdigest(),
        "release_date": "2026-03-20",
        "notes": "Stable release: WiFi scan optimizer, MQTT heartbeat improvements",
        "channel": "stable",
    },
    {
        "version": "3.2.0",
        "board": "any",
        "family": "esp32",
        "size": 1_548_288,
        "sha256": hashlib.sha256(b"tritium-os-3.2.0").hexdigest(),
        "release_date": "2026-03-10",
        "notes": "BLE feature extraction, diagnostic dump support",
        "channel": "stable",
    },
    {
        "version": "3.3.0-beta.1",
        "board": "any",
        "family": "esp32",
        "size": 1_605_632,
        "sha256": hashlib.sha256(b"tritium-os-3.3.0-beta1").hexdigest(),
        "release_date": "2026-03-22",
        "notes": "Beta: CSI motion detection, indoor positioning, new HAL layer",
        "channel": "beta",
    },
    {
        "version": "2.5.19",
        "board": "tlora-pager",
        "family": "esp32",
        "size": 1_310_720,
        "sha256": hashlib.sha256(b"meshtastic-2.5.19").hexdigest(),
        "release_date": "2026-03-15",
        "notes": "Meshtastic firmware: channel improvements, GPS fix",
        "channel": "stable",
    },
]


# ---------------------------------------------------------------------------
# Simulated ESP32 devices on the network
# ---------------------------------------------------------------------------

@dataclass
class SimulatedDevice:
    """A simulated ESP32 device discovered on the network."""
    device_id: str
    name: str
    chip: str
    board: str
    firmware_version: str
    firmware_project: str
    ip_address: str
    mac_address: str
    flash_size: str
    uptime_s: float = 0.0
    rssi: int = -50
    last_seen: float = field(default_factory=time.time)
    online: bool = True

    # Flash state
    flash_job_id: str = ""
    flash_status: FlashStatus = FlashStatus.PENDING
    flash_progress: float = 0.0
    flash_target_version: str = ""
    flash_error: str = ""
    flash_start_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result = {
            "device_id": self.device_id,
            "name": self.name,
            "chip": self.chip,
            "board": self.board,
            "firmware_version": self.firmware_version,
            "firmware_project": self.firmware_project,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "flash_size": self.flash_size,
            "uptime_s": round(self.uptime_s, 1),
            "rssi": self.rssi,
            "last_seen": self.last_seen,
            "online": self.online,
        }
        if self.flash_job_id:
            result["flash"] = {
                "job_id": self.flash_job_id,
                "status": self.flash_status.value,
                "progress": round(self.flash_progress, 1),
                "target_version": self.flash_target_version,
                "error": self.flash_error,
                "elapsed_s": round(time.time() - self.flash_start_time, 1)
                if self.flash_start_time
                else 0.0,
            }
        return result

    def detection(self) -> DeviceDetection:
        """Convert to a DeviceDetection for compatibility with the flasher API."""
        return DeviceDetection(
            detected=True,
            port=f"tcp://{self.ip_address}",
            chip=self.chip,
            chip_id=self.mac_address,
            flash_size=self.flash_size,
            firmware_version=self.firmware_version,
            firmware_project=self.firmware_project,
            board=self.board,
        )


def _generate_mac() -> str:
    """Generate a random MAC address in Espressif's range."""
    return "30:AE:A4:{:02X}:{:02X}:{:02X}".format(
        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255),
    )


def _generate_ip(base: int = 100) -> str:
    """Generate a LAN IP in 192.168.1.x range."""
    return f"192.168.1.{base + random.randint(0, 50)}"


SEED_DEVICES: list[dict[str, Any]] = [
    {
        "device_id": "edge-alpha",
        "name": "Alpha Node (Front Gate)",
        "chip": "ESP32-S3",
        "board": "touch-lcd-43c-box",
        "firmware_version": "3.2.0",
        "firmware_project": "tritium-os",
        "flash_size": "16MB",
        "uptime_s": 86400.0 * 3 + 7200,
        "rssi": -42,
    },
    {
        "device_id": "edge-bravo",
        "name": "Bravo Node (Parking Lot)",
        "chip": "ESP32-S3",
        "board": "touch-lcd-43c-box",
        "firmware_version": "3.1.0",
        "firmware_project": "tritium-os",
        "flash_size": "16MB",
        "uptime_s": 172800.0 + 3600,
        "rssi": -58,
    },
    {
        "device_id": "edge-charlie",
        "name": "Charlie Node (Warehouse)",
        "chip": "ESP32-S3",
        "board": "touch-amoled-241b",
        "firmware_version": "3.2.1",
        "firmware_project": "tritium-os",
        "flash_size": "8MB",
        "uptime_s": 3600.0 * 12,
        "rssi": -65,
    },
    {
        "device_id": "mesh-delta",
        "name": "Delta Mesh (Rooftop Relay)",
        "chip": "ESP32-S3",
        "board": "tlora-pager",
        "firmware_version": "2.5.18",
        "firmware_project": "meshtastic",
        "flash_size": "8MB",
        "uptime_s": 86400.0 * 7,
        "rssi": -71,
    },
    {
        "device_id": "edge-echo",
        "name": "Echo Node (Loading Dock)",
        "chip": "ESP32-C3",
        "board": "xiao-esp32c3",
        "firmware_version": "3.0.0",
        "firmware_project": "tritium-os",
        "flash_size": "4MB",
        "uptime_s": 86400.0 + 1800,
        "rssi": -55,
    },
    {
        "device_id": "mesh-foxtrot",
        "name": "Foxtrot Mesh (Perimeter South)",
        "chip": "ESP32",
        "board": "tbeam",
        "firmware_version": "2.5.19",
        "firmware_project": "meshtastic",
        "flash_size": "4MB",
        "uptime_s": 86400.0 * 14,
        "rssi": -78,
    },
]


# ---------------------------------------------------------------------------
# Firmware Fleet Manager — orchestrates discovery and OTA
# ---------------------------------------------------------------------------

class FirmwareFleetManager:
    """Simulated firmware fleet manager.

    Manages device discovery, firmware version tracking, and OTA flash
    operations. All operations are simulated — no real hardware needed.
    """

    def __init__(self) -> None:
        self.devices: dict[str, SimulatedDevice] = {}
        self.firmware_catalog: list[dict[str, Any]] = list(AVAILABLE_FIRMWARE)
        self.flash_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._flash_tasks: dict[str, threading.Thread] = {}

    def discover_devices(self) -> list[SimulatedDevice]:
        """Simulate mDNS/UDP discovery of ESP32 devices on the network."""
        with self._lock:
            if self.devices:
                # Update last_seen and simulate minor RSSI fluctuations
                for dev in self.devices.values():
                    dev.last_seen = time.time()
                    dev.rssi += random.randint(-3, 3)
                    dev.rssi = max(-90, min(-30, dev.rssi))
                    dev.uptime_s += random.uniform(10, 60)
                return list(self.devices.values())

            # First discovery — create devices from seed data
            for seed in SEED_DEVICES:
                mac = _generate_mac()
                ip = _generate_ip(100 + len(self.devices))
                dev = SimulatedDevice(
                    device_id=seed["device_id"],
                    name=seed["name"],
                    chip=seed["chip"],
                    board=seed["board"],
                    firmware_version=seed["firmware_version"],
                    firmware_project=seed["firmware_project"],
                    ip_address=ip,
                    mac_address=mac,
                    flash_size=seed["flash_size"],
                    uptime_s=seed.get("uptime_s", 3600.0),
                    rssi=seed.get("rssi", -50),
                )
                self.devices[dev.device_id] = dev

            return list(self.devices.values())

    def get_device(self, device_id: str) -> SimulatedDevice | None:
        """Get a single device by ID."""
        with self._lock:
            return self.devices.get(device_id)

    def get_available_firmware(
        self, board: str = "", channel: str = "",
    ) -> list[dict[str, Any]]:
        """List available firmware versions, optionally filtered."""
        results = list(self.firmware_catalog)
        if board:
            results = [
                fw for fw in results
                if fw["board"] == board or fw["board"] == "any"
            ]
        if channel:
            results = [fw for fw in results if fw["channel"] == channel]
        return results

    def check_updates(self) -> list[dict[str, Any]]:
        """Check which devices have firmware updates available."""
        self.discover_devices()
        updates = []
        with self._lock:
            for dev in self.devices.values():
                latest = self._find_latest_for_device(dev)
                if latest and latest["version"] != dev.firmware_version:
                    updates.append({
                        "device_id": dev.device_id,
                        "device_name": dev.name,
                        "current_version": dev.firmware_version,
                        "available_version": latest["version"],
                        "release_date": latest["release_date"],
                        "notes": latest["notes"],
                        "update_available": self._version_newer(
                            latest["version"], dev.firmware_version,
                        ),
                    })
        return updates

    def start_flash(
        self, device_id: str, target_version: str = "",
    ) -> dict[str, Any]:
        """Start a simulated OTA flash operation for a device.

        Returns a flash job descriptor or an error dict.
        """
        with self._lock:
            dev = self.devices.get(device_id)
            if not dev:
                return {"error": f"Device not found: {device_id}"}

            if dev.flash_status in (
                FlashStatus.DOWNLOADING,
                FlashStatus.ERASING,
                FlashStatus.WRITING,
                FlashStatus.VERIFYING,
            ):
                return {
                    "error": f"Device {device_id} already has a flash in progress",
                    "current_status": dev.flash_status.value,
                    "current_progress": dev.flash_progress,
                }

            # Determine target version
            if not target_version:
                latest = self._find_latest_for_device(dev)
                if not latest:
                    return {"error": f"No firmware available for {dev.board}"}
                target_version = latest["version"]

            # Validate target version exists
            fw_match = None
            for fw in self.firmware_catalog:
                if fw["version"] == target_version:
                    if fw["board"] == dev.board or fw["board"] == "any":
                        fw_match = fw
                        break
            if not fw_match:
                return {
                    "error": f"Firmware version {target_version} not found "
                    f"for board {dev.board}",
                }

            if target_version == dev.firmware_version:
                return {
                    "error": f"Device already running version {target_version}",
                }

            # Create flash job
            job_id = f"flash-{uuid.uuid4().hex[:8]}"
            dev.flash_job_id = job_id
            dev.flash_status = FlashStatus.PENDING
            dev.flash_progress = 0.0
            dev.flash_target_version = target_version
            dev.flash_error = ""
            dev.flash_start_time = time.time()

        # Start background flash simulation
        thread = threading.Thread(
            target=self._simulate_flash,
            args=(device_id, job_id, target_version),
            daemon=True,
        )
        thread.start()
        self._flash_tasks[job_id] = thread

        return {
            "job_id": job_id,
            "device_id": device_id,
            "target_version": target_version,
            "status": "pending",
        }

    def get_flash_status(self, device_id: str) -> dict[str, Any] | None:
        """Get the current flash status for a device."""
        with self._lock:
            dev = self.devices.get(device_id)
            if not dev:
                return None
            if not dev.flash_job_id:
                return {
                    "device_id": device_id,
                    "status": "idle",
                    "firmware_version": dev.firmware_version,
                }
            return {
                "device_id": device_id,
                "job_id": dev.flash_job_id,
                "status": dev.flash_status.value,
                "progress": round(dev.flash_progress, 1),
                "target_version": dev.flash_target_version,
                "error": dev.flash_error,
                "elapsed_s": round(
                    time.time() - dev.flash_start_time, 1,
                ) if dev.flash_start_time else 0.0,
            }

    def _simulate_flash(
        self, device_id: str, job_id: str, target_version: str,
    ) -> None:
        """Simulate a multi-phase OTA flash with realistic timing.

        Phases:
          1. Downloading firmware (2-4s)
          2. Erasing flash (1-2s)
          3. Writing firmware (3-6s, progress 0-100%)
          4. Verifying (1-2s)
          5. Rebooting (1s)

        ~5% chance of simulated failure for realism.
        """
        def _update(status: FlashStatus, progress: float, error: str = "") -> None:
            with self._lock:
                dev = self.devices.get(device_id)
                if dev and dev.flash_job_id == job_id:
                    dev.flash_status = status
                    dev.flash_progress = progress
                    dev.flash_error = error

        try:
            # Phase 1: Downloading
            _update(FlashStatus.DOWNLOADING, 0.0)
            for pct in range(0, 101, random.randint(8, 15)):
                time.sleep(random.uniform(0.08, 0.15))
                _update(FlashStatus.DOWNLOADING, min(pct, 100.0))

            # Phase 2: Erasing
            _update(FlashStatus.ERASING, 0.0)
            time.sleep(random.uniform(0.5, 1.0))
            _update(FlashStatus.ERASING, 50.0)
            time.sleep(random.uniform(0.3, 0.6))
            _update(FlashStatus.ERASING, 100.0)

            # Simulate ~5% failure rate
            if random.random() < 0.05:
                _update(
                    FlashStatus.FAILED, 0.0,
                    "Connection lost during erase — device may need manual reset",
                )
                self._record_flash_result(device_id, job_id, target_version, False)
                return

            # Phase 3: Writing
            _update(FlashStatus.WRITING, 0.0)
            written = 0.0
            while written < 100.0:
                chunk = random.uniform(2.0, 8.0)
                written = min(written + chunk, 100.0)
                time.sleep(random.uniform(0.06, 0.12))
                _update(FlashStatus.WRITING, written)

            # Phase 4: Verifying
            _update(FlashStatus.VERIFYING, 0.0)
            time.sleep(random.uniform(0.4, 0.8))
            _update(FlashStatus.VERIFYING, 50.0)
            time.sleep(random.uniform(0.3, 0.5))
            _update(FlashStatus.VERIFYING, 100.0)

            # Phase 5: Complete — update firmware version
            with self._lock:
                dev = self.devices.get(device_id)
                if dev and dev.flash_job_id == job_id:
                    dev.flash_status = FlashStatus.COMPLETED
                    dev.flash_progress = 100.0
                    dev.firmware_version = target_version
                    dev.uptime_s = 0.0  # Just rebooted

            self._record_flash_result(device_id, job_id, target_version, True)

        except Exception as exc:
            _update(FlashStatus.FAILED, 0.0, str(exc))
            self._record_flash_result(device_id, job_id, target_version, False)

    def _record_flash_result(
        self, device_id: str, job_id: str, version: str, success: bool,
    ) -> None:
        """Record a flash result in history."""
        self.flash_history.append({
            "job_id": job_id,
            "device_id": device_id,
            "target_version": version,
            "success": success,
            "timestamp": time.time(),
        })

    def _find_latest_for_device(
        self, dev: SimulatedDevice,
    ) -> dict[str, Any] | None:
        """Find the latest stable firmware for a device."""
        candidates = []
        for fw in self.firmware_catalog:
            if fw["channel"] != "stable":
                continue
            if fw["board"] == "any" and dev.firmware_project == "tritium-os":
                candidates.append(fw)
            elif fw["board"] == dev.board:
                candidates.append(fw)
        if not candidates:
            return None
        # Sort by version descending (simple string sort works for semver)
        candidates.sort(key=lambda f: f["version"], reverse=True)
        return candidates[0]

    @staticmethod
    def _version_newer(candidate: str, current: str) -> bool:
        """Check if candidate version is newer than current.

        Handles semver-like strings: "3.2.1" > "3.2.0", "2.5.19" > "2.5.18".
        """
        def _parse(v: str) -> tuple[int, ...]:
            parts = v.split("-")[0].split(".")
            result = []
            for p in parts:
                try:
                    result.append(int(p))
                except ValueError:
                    result.append(0)
            return tuple(result)

        return _parse(candidate) > _parse(current)

    def summary(self) -> dict[str, Any]:
        """Fleet firmware summary."""
        self.discover_devices()
        with self._lock:
            devices = list(self.devices.values())
        total = len(devices)
        online = sum(1 for d in devices if d.online)
        up_to_date = 0
        outdated = 0
        flashing = 0
        for dev in devices:
            if dev.flash_status in (
                FlashStatus.DOWNLOADING, FlashStatus.ERASING,
                FlashStatus.WRITING, FlashStatus.VERIFYING,
            ):
                flashing += 1
            else:
                latest = self._find_latest_for_device(dev)
                if latest and latest["version"] == dev.firmware_version:
                    up_to_date += 1
                else:
                    outdated += 1
        return {
            "total_devices": total,
            "online": online,
            "up_to_date": up_to_date,
            "outdated": outdated,
            "flashing": flashing,
            "firmware_versions_available": len(self.firmware_catalog),
            "flash_history_count": len(self.flash_history),
        }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

fleet: FirmwareFleetManager | None = None


def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    """Send a JSON response."""
    body = json.dumps(data, indent=2, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, html: str, status: int = 200) -> None:
    """Send an HTML response."""
    body = html.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    """Read the request body."""
    length = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(length) if length > 0 else b""


class FirmwareDemoHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the firmware management demo."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging noise."""
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._serve_dashboard()
        elif path == "/devices":
            assert fleet is not None
            devices = fleet.discover_devices()
            _json_response(self, [d.to_dict() for d in devices])
        elif path == "/firmware":
            assert fleet is not None
            params = parse_qs(parsed.query)
            board = params.get("board", [""])[0]
            channel = params.get("channel", [""])[0]
            _json_response(self, fleet.get_available_firmware(board, channel))
        elif path == "/health":
            assert fleet is not None
            _json_response(self, fleet.summary())
        elif path == "/updates":
            assert fleet is not None
            _json_response(self, fleet.check_updates())
        elif path.startswith("/flash/") and path.endswith("/status"):
            # GET /flash/{device_id}/status
            assert fleet is not None
            parts = path.split("/")
            if len(parts) >= 3:
                device_id = parts[2]
                status = fleet.get_flash_status(device_id)
                if status:
                    _json_response(self, status)
                else:
                    _json_response(self, {"error": "Device not found"}, 404)
            else:
                _json_response(self, {"error": "Invalid path"}, 400)
        else:
            _json_response(self, {"error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.startswith("/flash/"):
            # POST /flash/{device_id}
            assert fleet is not None
            parts = path.split("/")
            if len(parts) >= 3:
                device_id = parts[2]
                body = _read_body(self)
                try:
                    data = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    data = {}
                target_version = data.get("version", "")
                result = fleet.start_flash(device_id, target_version)
                status_code = 200 if "error" not in result else 400
                _json_response(self, result, status_code)
            else:
                _json_response(self, {"error": "Invalid path"}, 400)
        else:
            _json_response(self, {"error": "Not found"}, 404)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_dashboard(self) -> None:
        """Serve the firmware management dashboard."""
        _html_response(self, DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tritium Firmware Manager</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a0f;
    color: #c8c8d0;
    font-family: 'Courier New', monospace;
    padding: 20px;
  }
  h1 { color: #00f0ff; font-size: 1.6em; margin-bottom: 4px; }
  h2 { color: #ff2a6d; font-size: 1.1em; margin: 16px 0 8px; }
  .subtitle { color: #666; font-size: 0.85em; margin-bottom: 16px; }
  .stats {
    display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px;
  }
  .stat-card {
    background: #12121a;
    border: 1px solid #1a1a2e;
    border-radius: 6px;
    padding: 12px 16px;
    min-width: 140px;
  }
  .stat-value { font-size: 1.8em; color: #00f0ff; font-weight: bold; }
  .stat-label { font-size: 0.75em; color: #666; text-transform: uppercase; }
  .device-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 12px;
  }
  .device-card {
    background: #12121a;
    border: 1px solid #1a1a2e;
    border-radius: 6px;
    padding: 14px;
    transition: border-color 0.2s;
  }
  .device-card:hover { border-color: #00f0ff; }
  .device-name { color: #05ffa1; font-weight: bold; font-size: 0.95em; }
  .device-id { color: #666; font-size: 0.75em; }
  .device-info {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 12px;
    margin: 8px 0;
    font-size: 0.8em;
  }
  .device-info span { color: #555; }
  .device-info strong { color: #c8c8d0; }
  .fw-version { color: #fcee0a; font-weight: bold; }
  .fw-outdated { color: #ff2a6d; }
  .fw-current { color: #05ffa1; }
  .progress-bar {
    width: 100%;
    height: 6px;
    background: #1a1a2e;
    border-radius: 3px;
    margin: 6px 0;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }
  .progress-fill.downloading { background: #00f0ff; }
  .progress-fill.erasing { background: #fcee0a; }
  .progress-fill.writing { background: #ff2a6d; }
  .progress-fill.verifying { background: #05ffa1; }
  .progress-fill.completed { background: #05ffa1; width: 100% !important; }
  .progress-fill.failed { background: #ff2a6d; }
  .flash-status {
    font-size: 0.75em;
    padding: 2px 6px;
    border-radius: 3px;
    display: inline-block;
    margin-top: 4px;
  }
  .status-idle { color: #666; }
  .status-downloading { color: #00f0ff; }
  .status-erasing { color: #fcee0a; }
  .status-writing { color: #ff2a6d; }
  .status-verifying { color: #05ffa1; }
  .status-completed { color: #05ffa1; font-weight: bold; }
  .status-failed { color: #ff2a6d; font-weight: bold; }
  .btn {
    background: #1a1a2e;
    color: #00f0ff;
    border: 1px solid #00f0ff;
    padding: 4px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-family: inherit;
    font-size: 0.8em;
    margin-top: 6px;
  }
  .btn:hover { background: #00f0ff; color: #0a0a0f; }
  .btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .error { color: #ff2a6d; font-size: 0.75em; margin-top: 4px; }
  .api-section {
    margin-top: 20px;
    padding: 12px;
    background: #12121a;
    border: 1px solid #1a1a2e;
    border-radius: 6px;
    font-size: 0.8em;
  }
  .api-section code {
    color: #00f0ff;
    background: #0a0a0f;
    padding: 1px 4px;
    border-radius: 2px;
  }
</style>
</head>
<body>
<h1>TRITIUM FIRMWARE MANAGER</h1>
<div class="subtitle">Fleet OTA Management &mdash; Simulated Demo</div>

<div class="stats" id="stats"></div>

<h2>DISCOVERED DEVICES</h2>
<div class="device-grid" id="devices"></div>

<div class="api-section">
  <h2>API ENDPOINTS</h2>
  <p><code>GET /devices</code> &mdash; All discovered devices</p>
  <p><code>GET /firmware</code> &mdash; Available firmware versions</p>
  <p><code>GET /updates</code> &mdash; Devices with pending updates</p>
  <p><code>POST /flash/{device_id}</code> &mdash; Start OTA flash (body: {"version":"..."})</p>
  <p><code>GET /flash/{device_id}/status</code> &mdash; Flash progress</p>
  <p><code>GET /health</code> &mdash; Fleet summary</p>
</div>

<script>
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  return r.json();
}

function formatUptime(s) {
  if (s >= 86400) return (s / 86400).toFixed(1) + 'd';
  if (s >= 3600) return (s / 3600).toFixed(1) + 'h';
  return (s / 60).toFixed(0) + 'm';
}

async function refreshStats() {
  const data = await fetchJSON('/health');
  document.getElementById('stats').innerHTML = `
    <div class="stat-card"><div class="stat-value">${data.total_devices}</div><div class="stat-label">Devices</div></div>
    <div class="stat-card"><div class="stat-value">${data.online}</div><div class="stat-label">Online</div></div>
    <div class="stat-card"><div class="stat-value" style="color:#05ffa1">${data.up_to_date}</div><div class="stat-label">Up to Date</div></div>
    <div class="stat-card"><div class="stat-value" style="color:#ff2a6d">${data.outdated}</div><div class="stat-label">Outdated</div></div>
    <div class="stat-card"><div class="stat-value" style="color:#fcee0a">${data.flashing}</div><div class="stat-label">Flashing</div></div>
  `;
}

async function refreshDevices() {
  const devices = await fetchJSON('/devices');
  const firmware = await fetchJSON('/firmware?channel=stable');
  const latestVersions = {};
  firmware.forEach(fw => {
    if (!latestVersions[fw.board] || fw.version > latestVersions[fw.board])
      latestVersions[fw.board] = fw.version;
    if (fw.board === 'any')
      latestVersions['_any'] = fw.version;
  });

  const grid = document.getElementById('devices');
  grid.innerHTML = devices.map(dev => {
    const latest = latestVersions[dev.board] || latestVersions['_any'] || dev.firmware_version;
    const isOutdated = dev.firmware_version !== latest;
    const versionClass = isOutdated ? 'fw-outdated' : 'fw-current';
    const flash = dev.flash || {};
    const hasFlash = !!flash.job_id;
    const isFlashing = hasFlash && !['completed','failed','idle','pending'].includes(flash.status);
    const isDone = flash.status === 'completed';
    const isFailed = flash.status === 'failed';

    let progressHtml = '';
    if (hasFlash) {
      const statusClass = flash.status || 'idle';
      progressHtml = `
        <div class="progress-bar">
          <div class="progress-fill ${statusClass}" style="width:${flash.progress || 0}%"></div>
        </div>
        <span class="flash-status status-${statusClass}">
          ${flash.status.toUpperCase()} ${flash.progress ? flash.progress.toFixed(0) + '%' : ''}
          ${flash.elapsed_s ? '(' + flash.elapsed_s.toFixed(1) + 's)' : ''}
        </span>
        ${flash.error ? '<div class="error">' + flash.error + '</div>' : ''}
      `;
    }

    const canFlash = isOutdated && !isFlashing && !isDone;
    const btnDisabled = !canFlash ? 'disabled' : '';
    const btnLabel = isDone ? 'UPDATED' : isFlashing ? 'FLASHING...' : isFailed ? 'RETRY' : 'FLASH UPDATE';

    return `
      <div class="device-card">
        <div class="device-name">${dev.name}</div>
        <div class="device-id">${dev.device_id} | ${dev.mac_address} | ${dev.ip_address}</div>
        <div class="device-info">
          <div><span>Chip:</span> <strong>${dev.chip}</strong></div>
          <div><span>Board:</span> <strong>${dev.board}</strong></div>
          <div><span>Firmware:</span> <strong class="${versionClass}">${dev.firmware_version}</strong></div>
          <div><span>Latest:</span> <strong>${latest}</strong></div>
          <div><span>Flash:</span> <strong>${dev.flash_size}</strong></div>
          <div><span>RSSI:</span> <strong>${dev.rssi} dBm</strong></div>
          <div><span>Uptime:</span> <strong>${formatUptime(dev.uptime_s)}</strong></div>
          <div><span>Project:</span> <strong>${dev.firmware_project}</strong></div>
        </div>
        ${progressHtml}
        <button class="btn" ${btnDisabled}
          onclick="startFlash('${dev.device_id}','${latest}')"
          ${isFailed ? '' : btnDisabled}>${btnLabel}</button>
      </div>
    `;
  }).join('');
}

async function startFlash(deviceId, version) {
  await fetchJSON('/flash/' + deviceId, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({version: version}),
  });
  refreshAll();
}

async function refreshAll() {
  await Promise.all([refreshStats(), refreshDevices()]);
}

refreshAll();
setInterval(refreshAll, 1000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(port: int = 8098) -> None:
    """Start the firmware management demo server."""
    global fleet
    fleet = FirmwareFleetManager()
    fleet.discover_devices()

    server = HTTPServer(("0.0.0.0", port), FirmwareDemoHandler)
    print(f"Tritium Firmware Manager demo running on http://localhost:{port}")
    print(f"  GET  /devices        - Discovered devices")
    print(f"  GET  /firmware       - Available firmware versions")
    print(f"  GET  /updates        - Devices with pending updates")
    print(f"  POST /flash/{{id}}     - Start OTA flash")
    print(f"  GET  /flash/{{id}}/status - Flash progress")
    print(f"  GET  /health         - Fleet summary")
    print()
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    run_demo()

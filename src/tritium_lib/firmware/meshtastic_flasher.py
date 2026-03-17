# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Meshtastic firmware flasher — downloads and flashes official Meshtastic firmware.

Like the Meshtastic Web Flasher (flasher.meshtastic.org), this can:
1. Detect the connected device (chip, board, current firmware)
2. Download the latest official Meshtastic firmware from GitHub releases
3. Flash the firmware via esptool.py
4. Verify the device boots with the new firmware

Supports all Meshtastic ESP32 devices including:
- T-LoRa Pager (ESP32-S3)
- T-Beam (ESP32)
- T-Deck (ESP32-S3)
- Heltec V3 (ESP32-S3)
- RAK WisBlock (nRF52840 — NOT supported by esptool, needs adafruit-nrfutil)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .base import (
    DeviceDetection,
    FlashResult,
    FlashStatus,
)
from .esp32 import ESP32Flasher

log = logging.getLogger("tritium.firmware.meshtastic")

# GitHub API for Meshtastic firmware releases
MESHTASTIC_RELEASES_URL = "https://api.github.com/repos/meshtastic/firmware/releases"
MESHTASTIC_LATEST_URL = "https://api.github.com/repos/meshtastic/firmware/releases/latest"

# Cache directory for downloaded firmware
FIRMWARE_CACHE_DIR = Path(os.environ.get(
    "TRITIUM_FIRMWARE_CACHE",
    Path.home() / ".cache" / "tritium" / "firmware" / "meshtastic",
))

# Board name mapping: hwModel string → firmware archive name
# The meshtastic library returns hwModel as a string like "TLORA_V2_1_1P6"
# The firmware archive contains files named like "firmware-tlora-v2_1_1p6-2.5.19.5f8df68.bin"
BOARD_FIRMWARE_MAP = {
    # ESP32-S3 devices
    "TLORA_PAGER": "tlora-pager",
    "T_DECK": "t-deck",
    "HELTEC_V3": "heltec-v3",
    "HELTEC_WSL_V3": "heltec-wsl-v3",
    "STATION_G2": "station-g2",
    "UNPHONE": "unphone",
    "PICOMPUTER_S3": "picomputer-s3",
    "T_WATCH_S3": "t-watch-s3",
    "NANO_G2_ULTRA": "nano-g2-ultra",
    # ESP32 devices
    "TBEAM": "tbeam",
    "TBEAM_V0P7": "tbeam0.7",
    "TLORA_V1": "tlora-v1",
    "TLORA_V2": "tlora-v2",
    "TLORA_V2_1_1P6": "tlora-v2-1-1.6",
    "TLORA_V2_1_1P8": "tlora-v2-1-1.8",
    "HELTEC_V1": "heltec-v1",
    "HELTEC_V2_0": "heltec-v2.0",
    "HELTEC_V2_1": "heltec-v2.1",
    # ESP32-C3 devices
    "HELTEC_HT62": "heltec-ht62",
    "XIAO_ESP32C3": "xiao-esp32c3",
}

# Chip type → flash offset for merged Meshtastic firmware
MESHTASTIC_FLASH_OFFSETS = {
    "ESP32": "0x1000",
    "ESP32-S2": "0x1000",
    "ESP32-S3": "0x0",
    "ESP32-C3": "0x0",
    "ESP32-C6": "0x0",
}


class MeshtasticFlasher(ESP32Flasher):
    """Flash official Meshtastic firmware to any supported device.

    Extends ESP32Flasher with:
    - Meshtastic firmware version detection from the device
    - Automatic firmware download from GitHub releases
    - Board-specific firmware file selection
    - Version comparison for update detection
    """

    def __init__(self, port: str = "", baud: int = 921600):
        super().__init__(port=port, baud=baud)
        self._meshtastic_cli = shutil.which("meshtastic")

    async def detect(self) -> DeviceDetection:
        """Detect Meshtastic device — uses both esptool and meshtastic library."""
        # First: basic ESP32 detection via esptool
        base_detection = await super().detect()

        if not base_detection.detected:
            return base_detection

        base_detection.firmware_project = "meshtastic"

        # Try to get more info via the meshtastic Python library
        try:
            mesh_info = await self._run_in_executor(
                self._detect_meshtastic_sync, base_detection.port,
            )
            if mesh_info:
                base_detection.firmware_version = mesh_info.get("firmware_version", "")
                base_detection.board = mesh_info.get("board", "")
                if not base_detection.chip:
                    base_detection.chip = mesh_info.get("chip", "")
        except Exception as e:
            log.debug(f"Meshtastic library detection failed: {e}")

        # Map hwModel to firmware name
        if base_detection.board and base_detection.board in BOARD_FIRMWARE_MAP:
            base_detection.board = BOARD_FIRMWARE_MAP[base_detection.board]

        return base_detection

    @staticmethod
    def _detect_meshtastic_sync(port: str) -> dict | None:
        """Try to read Meshtastic device info via the library."""
        try:
            import meshtastic.serial_interface
            iface = meshtastic.serial_interface.SerialInterface(
                port, connectTimeout=10, noNodes=True,
            )
            try:
                my_info = iface.getMyNodeInfo() or {}
                user = my_info.get("user", {})

                result = {
                    "board": user.get("hwModel", ""),
                    "firmware_version": "",
                }

                # Try to get firmware version from metadata
                if hasattr(iface, "metadata") and iface.metadata:
                    result["firmware_version"] = getattr(
                        iface.metadata, "firmware_version", ""
                    )

                return result
            finally:
                iface.close()
        except ImportError:
            log.debug("meshtastic package not installed")
            return None
        except Exception as e:
            log.debug(f"Meshtastic detection failed: {e}")
            return None

    async def download_firmware(
        self, board: str = "", chip: str = "", version: str = "latest",
    ) -> str | None:
        """Download Meshtastic firmware from GitHub releases.

        Args:
            board: Board name (e.g., "tlora-pager", "tbeam").
            chip: Chip type for offset selection.
            version: Version tag (e.g., "v2.5.19.5f8df68") or "latest".

        Returns local path to the downloaded .bin file, or None on failure.
        """
        self._emit_progress(FlashStatus.DOWNLOADING, 0, "Fetching release info...")

        try:
            release = await self._run_in_executor(
                self._fetch_release_sync, version,
            )
            if not release:
                log.error(f"Could not fetch Meshtastic release: {version}")
                return None

            tag = release.get("tag_name", "unknown")
            assets = release.get("assets", [])

            # Find the firmware zip for this board
            fw_asset = self._find_firmware_asset(assets, board)
            if not fw_asset:
                log.error(f"No firmware found for board '{board}' in release {tag}")
                log.info(f"Available assets: {[a['name'] for a in assets[:20]]}")
                return None

            # Download to cache
            download_url = fw_asset["browser_download_url"]
            filename = fw_asset["name"]
            cache_path = FIRMWARE_CACHE_DIR / tag / filename

            if cache_path.exists():
                log.info(f"Using cached firmware: {cache_path}")
                return str(cache_path)

            self._emit_progress(
                FlashStatus.DOWNLOADING, 20,
                f"Downloading {filename} ({fw_asset.get('size', 0) // 1024}KB)...",
            )

            downloaded = await self._run_in_executor(
                self._download_file_sync, download_url, str(cache_path),
            )

            if downloaded:
                self._emit_progress(FlashStatus.DOWNLOADING, 90, "Download complete")

                # If it's a zip, extract the .bin file
                if filename.endswith(".zip"):
                    bin_path = await self._run_in_executor(
                        self._extract_firmware_bin, str(cache_path), board,
                    )
                    return bin_path

                return str(cache_path)
            else:
                return None

        except Exception as e:
            log.error(f"Firmware download failed: {e}")
            return None

    @staticmethod
    def _fetch_release_sync(version: str) -> dict | None:
        """Fetch release info from GitHub API."""
        url = MESHTASTIC_LATEST_URL if version == "latest" else f"{MESHTASTIC_RELEASES_URL}/tags/{version}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except Exception as e:
            log.error(f"GitHub API request failed: {e}")
            return None

    @staticmethod
    def _find_firmware_asset(assets: list[dict], board: str) -> dict | None:
        """Find the firmware asset matching the board name."""
        board_lower = board.lower().replace("-", "").replace("_", "")

        for asset in assets:
            name = asset.get("name", "").lower()
            # Meshtastic firmware files are named like:
            # firmware-tlora-pager-2.5.19.5f8df68.bin
            # or firmware-tlora-pager-2.5.19.5f8df68.zip
            name_clean = name.replace("-", "").replace("_", "")
            if board_lower in name_clean and (name.endswith(".bin") or name.endswith(".zip")):
                # Prefer .bin over .zip
                return asset

        # Second pass: look for the zip archive containing all firmware
        for asset in assets:
            name = asset.get("name", "").lower()
            if "firmware" in name and name.endswith(".zip") and board_lower in name.replace("-", "").replace("_", ""):
                return asset

        return None

    @staticmethod
    def _download_file_sync(url: str, dest_path: str) -> bool:
        """Download a file to the given path."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use a temp file so we don't leave partial downloads
            with tempfile.NamedTemporaryFile(
                dir=dest.parent, delete=False, suffix=".tmp"
            ) as tmp:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        tmp.write(chunk)
                tmp_path = tmp.name

            # Move to final location
            shutil.move(tmp_path, dest_path)
            log.info(f"Downloaded firmware to {dest_path}")
            return True
        except Exception as e:
            log.error(f"Download failed: {e}")
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return False

    @staticmethod
    def _extract_firmware_bin(zip_path: str, board: str) -> str | None:
        """Extract the .bin firmware file from a zip archive."""
        import zipfile
        board_lower = board.lower()

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Find the firmware .bin for this board
                for name in zf.namelist():
                    if board_lower in name.lower() and name.endswith(".bin"):
                        extract_dir = Path(zip_path).parent
                        zf.extract(name, extract_dir)
                        return str(extract_dir / name)

                # If no board-specific .bin, look for a merged firmware
                for name in zf.namelist():
                    if "firmware" in name.lower() and name.endswith(".bin"):
                        extract_dir = Path(zip_path).parent
                        zf.extract(name, extract_dir)
                        return str(extract_dir / name)
        except Exception as e:
            log.error(f"Failed to extract firmware from {zip_path}: {e}")

        return None

    async def flash_latest(self, **kwargs) -> FlashResult:
        """Detect board, download latest Meshtastic firmware, and flash it."""
        detection = await self.detect()
        if not detection.detected:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error=f"No device detected: {detection.error}",
                port=self.port,
            )

        board = detection.board
        chip = detection.chip
        if not board:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="Could not determine board type. Specify firmware file manually.",
                port=detection.port,
            )

        log.info(f"Detected: {chip} board={board} fw={detection.firmware_version}")

        # Download latest firmware
        fw_path = await self.download_firmware(board=board, chip=chip)
        if not fw_path:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error=f"Failed to download firmware for {board}",
                port=detection.port,
            )

        # Determine flash offset from chip type
        flash_offset = MESHTASTIC_FLASH_OFFSETS.get(chip, "0x0")

        # Flash it
        return await self.flash(
            fw_path,
            erase_all=kwargs.get("erase_all", True),  # Meshtastic recommends erase_all
            verify=kwargs.get("verify", True),
            flash_offset=flash_offset,
            chip=chip.lower().replace("-", ""),
        )

    async def flash_with_meshtastic_cli(self, firmware_path: str = "") -> FlashResult:
        """Flash using the `meshtastic` CLI tool instead of esptool.

        The meshtastic CLI has a --flash-firmware flag that handles
        all the board-specific details automatically.
        """
        if not self._meshtastic_cli:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="meshtastic CLI not found. Install with: pip install meshtastic",
            )

        port = self.port or self._auto_detect_port()
        if not port:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="No serial port found",
            )

        import subprocess
        import time

        cmd = [self._meshtastic_cli, "--port", port]
        if firmware_path:
            cmd.extend(["--flash-firmware", firmware_path])
        else:
            # Without a path, meshtastic CLI flashes the latest for the detected board
            cmd.append("--flash-firmware")

        start = time.monotonic()
        self._emit_progress(FlashStatus.WRITING, 0, "Flashing via meshtastic CLI...")

        try:
            result = await self._run_in_executor(
                self._run_cli_sync, cmd,
            )
            result.duration_s = time.monotonic() - start
            result.port = port
            return result
        except Exception as e:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error=str(e), port=port,
            )

    @staticmethod
    def _run_cli_sync(cmd: list[str]) -> FlashResult:
        """Run meshtastic CLI flash command."""
        import subprocess
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            return FlashResult(
                success=proc.returncode == 0,
                status=FlashStatus.COMPLETED if proc.returncode == 0 else FlashStatus.FAILED,
                output=proc.stdout,
                error=proc.stderr if proc.returncode != 0 else "",
            )
        except subprocess.TimeoutExpired:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="Flash timed out after 300 seconds",
            )

    def get_available_versions(self, limit: int = 10) -> list[dict]:
        """Fetch available Meshtastic firmware versions from GitHub."""
        try:
            req = urllib.request.Request(
                f"{MESHTASTIC_RELEASES_URL}?per_page={limit}",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                releases = json.loads(resp.read())
                return [
                    {
                        "tag": r["tag_name"],
                        "name": r["name"],
                        "published": r["published_at"],
                        "prerelease": r["prerelease"],
                        "asset_count": len(r.get("assets", [])),
                    }
                    for r in releases
                ]
        except Exception as e:
            log.error(f"Failed to fetch releases: {e}")
            return []

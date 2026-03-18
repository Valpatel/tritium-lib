# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Meshtastic firmware flasher — downloads and flashes official Meshtastic firmware.

Like the Meshtastic Web Flasher (flasher.meshtastic.org), this can:
1. Detect the connected device (chip, board, current firmware)
2. Download the latest official Meshtastic firmware from GitHub releases
3. Flash the firmware via esptool.py using the full multi-step process
4. Verify the device boots with the new firmware

The full flash process mirrors what the official flasher does:
  Step 1: erase_flash (clean slate)
  Step 2: write_flash 0x0 factory.bin (bootloader + app + partition table)
  Step 3: write_flash {ota_offset} mt-esp32s3-ota.bin (OTA bootloader)
  Step 4: write_flash {spiffs_offset} littlefs-{board}.bin (filesystem)

For OTA-style updates (preserves settings):
  write_flash 0x10000 update.bin (app partition only, no erase)

The firmware zip contains a .mt.json metadata file with partition offsets,
board info, and file checksums. We parse this to get correct offsets.

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
import time
import urllib.request
from dataclasses import dataclass, field
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

# Board name mapping: hwModel string -> firmware archive name
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

# Chip type -> flash offset for merged Meshtastic firmware
MESHTASTIC_FLASH_OFFSETS = {
    "ESP32": "0x1000",
    "ESP32-S2": "0x1000",
    "ESP32-S3": "0x0",
    "ESP32-C3": "0x0",
    "ESP32-C6": "0x0",
}

# Default partition offsets when .mt.json is not available
DEFAULT_OTA_OFFSET = "0x260000"
DEFAULT_SPIFFS_OFFSET = "0x300000"
DEFAULT_APP_OFFSET = "0x10000"

# ESP32-S3 boards that support USB DFU (1200bps reset trick)
ESP32S3_BOARDS = {
    "tlora-pager", "t-deck", "heltec-v3", "heltec-wsl-v3",
    "station-g2", "unphone", "picomputer-s3", "t-watch-s3",
    "nano-g2-ultra",
}


@dataclass
class MeshtasticFirmwareFiles:
    """All files extracted from a Meshtastic firmware zip."""
    factory_bin: str = ""       # Full flash at 0x0 (fresh install)
    update_bin: str = ""        # App only at 0x10000 (OTA update)
    ota_bin: str = ""           # OTA bootloader (mt-esp32s3-ota.bin)
    littlefs_bin: str = ""      # Filesystem partition
    mt_json: str = ""           # Metadata with partition offsets
    board: str = ""
    version: str = ""

    # Offsets parsed from .mt.json (or defaults)
    ota_offset: str = DEFAULT_OTA_OFFSET
    spiffs_offset: str = DEFAULT_SPIFFS_OFFSET
    app_offset: str = DEFAULT_APP_OFFSET


class MeshtasticFlasher(ESP32Flasher):
    """Flash official Meshtastic firmware to any supported device.

    Extends ESP32Flasher with:
    - Meshtastic firmware version detection from the device
    - Automatic firmware download from GitHub releases
    - Board-specific firmware file selection
    - Full multi-step flash process (erase + factory + OTA + littlefs)
    - App-only update mode (preserves settings)
    - DFU mode entry for ESP32-S3 boards
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
        """Download Meshtastic firmware zip from GitHub releases.

        Downloads the full firmware zip (not just a single .bin) so we have
        all partition files needed for a complete flash.

        Args:
            board: Board name (e.g., "tlora-pager", "tbeam").
            chip: Chip type for offset selection.
            version: Version tag (e.g., "v2.5.19.5f8df68") or "latest".

        Returns local path to the downloaded zip/bin file, or None on failure.
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
        """Find the firmware asset matching the board name.

        Prefers .zip over .bin since the zip contains all partition files
        needed for a complete flash (factory, OTA, littlefs, metadata).
        """
        board_lower = board.lower().replace("-", "").replace("_", "")
        zip_match = None
        bin_match = None

        for asset in assets:
            name = asset.get("name", "").lower()
            name_clean = name.replace("-", "").replace("_", "")
            if board_lower in name_clean:
                if name.endswith(".zip"):
                    zip_match = asset
                elif name.endswith(".bin"):
                    bin_match = asset

        # Prefer zip (has all partition files) over bare .bin
        return zip_match or bin_match

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
    def _extract_firmware_files(zip_path: str, board: str) -> MeshtasticFirmwareFiles:
        """Extract ALL needed files from a Meshtastic firmware zip.

        The zip typically contains:
        - firmware-{board}-{ver}.factory.bin (full flash at 0x0)
        - firmware-{board}-{ver}-update.bin (app only at 0x10000)
        - mt-esp32s3-ota.bin or similar OTA bootloader
        - littlefs-{board}-{ver}.bin (SPIFFS/LittleFS filesystem)
        - firmware-{board}-{ver}.mt.json (metadata with offsets)
        """
        import zipfile
        board_lower = board.lower()
        result = MeshtasticFirmwareFiles(board=board)
        extract_dir = Path(zip_path).parent

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()

                for name in names:
                    name_lower = name.lower()
                    basename = Path(name).name.lower()

                    # .mt.json metadata file
                    if basename.endswith(".mt.json") and board_lower in basename:
                        zf.extract(name, extract_dir)
                        result.mt_json = str(extract_dir / name)

                    # Factory binary (full flash)
                    elif ".factory.bin" in name_lower and board_lower in name_lower:
                        zf.extract(name, extract_dir)
                        result.factory_bin = str(extract_dir / name)

                    # Update binary (app only)
                    elif "-update.bin" in name_lower and board_lower in name_lower:
                        zf.extract(name, extract_dir)
                        result.update_bin = str(extract_dir / name)

                    # OTA bootloader (mt-esp32s3-ota.bin, mt-esp32-ota.bin)
                    elif basename.startswith("mt-") and "-ota.bin" in name_lower:
                        zf.extract(name, extract_dir)
                        result.ota_bin = str(extract_dir / name)

                    # LittleFS filesystem
                    elif "littlefs" in name_lower and name_lower.endswith(".bin"):
                        if board_lower in name_lower or "littlefs-" in name_lower:
                            zf.extract(name, extract_dir)
                            result.littlefs_bin = str(extract_dir / name)

                # Fallback: if no factory.bin found, look for any board .bin
                if not result.factory_bin:
                    for name in names:
                        if (board_lower in name.lower()
                                and name.lower().endswith(".bin")
                                and "update" not in name.lower()
                                and "littlefs" not in name.lower()
                                and "ota" not in name.lower()):
                            zf.extract(name, extract_dir)
                            result.factory_bin = str(extract_dir / name)
                            break

        except Exception as e:
            log.error(f"Failed to extract firmware from {zip_path}: {e}")

        return result

    @staticmethod
    def _extract_firmware_bin(zip_path: str, board: str) -> str | None:
        """Extract the .bin firmware file from a zip archive.

        Legacy method — returns just the factory .bin path.
        For full multi-step flash, use _extract_firmware_files() instead.
        """
        files = MeshtasticFlasher._extract_firmware_files(zip_path, board)
        return files.factory_bin or None

    @staticmethod
    def _parse_mt_json(json_path: str) -> dict:
        """Parse a Meshtastic .mt.json metadata file.

        The .mt.json contains partition offsets, file checksums, and board info.
        Example structure:
        {
            "board": "tlora-pager",
            "chip": "esp32s3",
            "version": "2.5.19.5f8df68",
            "partitions": {
                "factory": {"offset": "0x0", "size": "0x1F0000"},
                "ota": {"offset": "0x260000", "size": "0x10000"},
                "spiffs": {"offset": "0x300000", "size": "0x100000"}
            },
            "files": {
                "factory": {"name": "firmware-tlora-pager-2.5.19.factory.bin", "md5": "..."},
                "ota": {"name": "mt-esp32s3-ota.bin", "md5": "..."},
                "spiffs": {"name": "littlefs-tlora-pager-2.5.19.bin", "md5": "..."}
            }
        }

        Returns a dict with keys: board, chip, version, partitions, files.
        Returns empty dict on parse failure.
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            log.info(f"Parsed .mt.json: board={data.get('board')}, version={data.get('version')}")
            return data
        except Exception as e:
            log.warning(f"Failed to parse .mt.json at {json_path}: {e}")
            return {}

    def _apply_mt_json_offsets(
        self, fw_files: MeshtasticFirmwareFiles,
    ) -> MeshtasticFirmwareFiles:
        """Read offsets from .mt.json and apply them to firmware files."""
        if not fw_files.mt_json or not Path(fw_files.mt_json).exists():
            log.debug("No .mt.json found, using default partition offsets")
            return fw_files

        metadata = self._parse_mt_json(fw_files.mt_json)
        if not metadata:
            return fw_files

        fw_files.version = metadata.get("version", "")

        # Extract partition offsets
        partitions = metadata.get("partitions", {})
        if isinstance(partitions, dict):
            ota_part = partitions.get("ota", {})
            spiffs_part = partitions.get("spiffs", partitions.get("littlefs", {}))
            app_part = partitions.get("app", partitions.get("factory", {}))

            if isinstance(ota_part, dict) and "offset" in ota_part:
                fw_files.ota_offset = ota_part["offset"]
            if isinstance(spiffs_part, dict) and "offset" in spiffs_part:
                fw_files.spiffs_offset = spiffs_part["offset"]
            if isinstance(app_part, dict) and "offset" in app_part:
                fw_files.app_offset = app_part["offset"]

        log.info(
            f"Partition offsets: ota={fw_files.ota_offset}, "
            f"spiffs={fw_files.spiffs_offset}, app={fw_files.app_offset}"
        )
        return fw_files

    async def _enter_dfu_mode(self, port: str = "") -> bool:
        """Enter DFU mode on ESP32-S3 boards using the 1200bps reset trick.

        Many ESP32-S3 boards with native USB support can be forced into
        bootloader/DFU mode by opening the serial port at 1200 baud and
        then closing it. This triggers the USB CDC ACM reset sequence that
        the ROM bootloader recognizes.

        Returns True if DFU entry was attempted (does not guarantee success).
        """
        target_port = port or self.port or self._auto_detect_port()
        if not target_port:
            log.warning("No port specified for DFU mode entry")
            return False

        log.info(f"Attempting 1200bps DFU reset on {target_port}...")
        self._emit_progress(FlashStatus.DETECTING, 10, "Entering DFU mode...")

        try:
            result = await self._run_in_executor(
                self._enter_dfu_sync, target_port,
            )
            return result
        except Exception as e:
            log.warning(f"DFU mode entry failed: {e}")
            return False

    @staticmethod
    def _enter_dfu_sync(port: str) -> bool:
        """Synchronous 1200bps reset trick."""
        try:
            import serial
            # Open at 1200 baud, toggle DTR, close — triggers bootloader
            ser = serial.Serial(port, 1200, timeout=1)
            ser.dtr = False
            time.sleep(0.1)
            ser.dtr = True
            time.sleep(0.1)
            ser.dtr = False
            ser.close()
            # Give the board time to reset into bootloader
            time.sleep(2.0)
            log.info("1200bps DFU reset sent, waiting for bootloader...")
            return True
        except ImportError:
            log.warning("pyserial not installed — cannot enter DFU mode")
            return False
        except Exception as e:
            log.warning(f"1200bps reset failed: {e}")
            return False

    async def flash_latest(self, **kwargs) -> FlashResult:
        """Detect board, download latest Meshtastic firmware, and do full flash.

        Full flash process (fresh install):
          Step 1: erase_flash
          Step 2: write_flash 0x0 factory.bin
          Step 3: write_flash {ota_offset} mt-esp32s3-ota.bin
          Step 4: write_flash {spiffs_offset} littlefs-{board}.bin

        This replaces everything on the device including settings.
        For preserving settings, use update_firmware() instead.
        """
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

        # Try DFU mode for ESP32-S3 boards
        if board in ESP32S3_BOARDS:
            await self._enter_dfu_mode(detection.port)

        # Download firmware zip
        fw_path = await self.download_firmware(board=board, chip=chip)
        if not fw_path:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error=f"Failed to download firmware for {board}",
                port=detection.port,
            )

        # If we got a zip, extract all files and do multi-step flash
        if fw_path.endswith(".zip"):
            return await self._flash_full_from_zip(
                fw_path, board, chip, detection.port, **kwargs,
            )

        # Fallback: single .bin flash (legacy behavior)
        flash_offset = MESHTASTIC_FLASH_OFFSETS.get(chip, "0x0")
        return await self.flash(
            fw_path,
            erase_all=kwargs.get("erase_all", True),
            verify=kwargs.get("verify", True),
            flash_offset=flash_offset,
            chip=chip.lower().replace("-", ""),
        )

    async def _flash_full_from_zip(
        self,
        zip_path: str,
        board: str,
        chip: str,
        port: str,
        **kwargs,
    ) -> FlashResult:
        """Full 4-step flash from a Meshtastic firmware zip.

        Step 1: erase_flash
        Step 2: write_flash 0x0 factory.bin
        Step 3: write_flash {ota_offset} ota.bin (if present)
        Step 4: write_flash {spiffs_offset} littlefs.bin (if present)
        """
        self._emit_progress(FlashStatus.WRITING, 5, "Extracting firmware files...")

        # Extract all files from the zip
        fw_files = await self._run_in_executor(
            self._extract_firmware_files, zip_path, board,
        )

        if not fw_files.factory_bin:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error=f"No factory.bin found in firmware zip for board '{board}'",
                port=port,
            )

        # Parse .mt.json for partition offsets
        fw_files = self._apply_mt_json_offsets(fw_files)

        flash_offset = MESHTASTIC_FLASH_OFFSETS.get(chip, "0x0")
        chip_key = chip.lower().replace("-", "")

        # Build additional writes list
        additional_writes: list[tuple[str, str]] = []
        if fw_files.ota_bin and Path(fw_files.ota_bin).exists():
            additional_writes.append((fw_files.ota_offset, fw_files.ota_bin))
            log.info(f"OTA bootloader: {fw_files.ota_bin} at {fw_files.ota_offset}")
        if fw_files.littlefs_bin and Path(fw_files.littlefs_bin).exists():
            additional_writes.append((fw_files.spiffs_offset, fw_files.littlefs_bin))
            log.info(f"LittleFS: {fw_files.littlefs_bin} at {fw_files.spiffs_offset}")

        log.info(
            f"Full flash: factory={fw_files.factory_bin} at {flash_offset}, "
            f"{len(additional_writes)} additional partitions"
        )

        # Flash with erase + factory + additional partitions
        result = await self.flash(
            fw_files.factory_bin,
            erase_all=kwargs.get("erase_all", True),
            verify=kwargs.get("verify", True),
            flash_offset=flash_offset,
            chip=chip_key,
            additional_writes=additional_writes if additional_writes else None,
        )

        if result.success and fw_files.version:
            result.firmware_version = fw_files.version

        return result

    async def update_firmware(
        self,
        board: str = "",
        chip: str = "",
        version: str = "latest",
        port: str = "",
    ) -> FlashResult:
        """App-only update — preserves device settings.

        Writes only the app partition (update.bin at 0x10000). Does NOT erase
        flash, so all device settings, channels, and node info are preserved.
        Uses 115200 baud for reliability during app-only writes.

        This is equivalent to the "Update" option in the Meshtastic web flasher,
        as opposed to the "Full Install" option (flash_latest).

        Args:
            board: Board name (auto-detected if empty).
            chip: Chip type (auto-detected if empty).
            version: Version to download ("latest" or specific tag).
            port: Serial port (auto-detected if empty).
        """
        # Auto-detect if needed
        if not board or not chip:
            detection = await self.detect()
            if not detection.detected:
                return FlashResult(
                    success=False, status=FlashStatus.FAILED,
                    error=f"No device detected: {detection.error}",
                    port=port or self.port,
                )
            board = board or detection.board
            chip = chip or detection.chip
            port = port or detection.port

        if not board:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="Could not determine board type for update.",
                port=port,
            )

        log.info(f"App-only update: board={board}, chip={chip}")

        # Try DFU mode for ESP32-S3 boards
        if board in ESP32S3_BOARDS:
            await self._enter_dfu_mode(port)

        # Download firmware
        fw_path = await self.download_firmware(board=board, chip=chip, version=version)
        if not fw_path:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error=f"Failed to download firmware for {board}",
                port=port,
            )

        # Extract update.bin from zip
        update_bin = None
        if fw_path.endswith(".zip"):
            fw_files = await self._run_in_executor(
                self._extract_firmware_files, fw_path, board,
            )
            fw_files = self._apply_mt_json_offsets(fw_files)
            update_bin = fw_files.update_bin
            app_offset = fw_files.app_offset
        else:
            # Bare .bin — assume it's the update binary
            update_bin = fw_path
            app_offset = DEFAULT_APP_OFFSET

        if not update_bin or not Path(update_bin).exists():
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="No update.bin found in firmware package. Use flash_latest() for full install.",
                port=port,
            )

        chip_key = chip.lower().replace("-", "")
        log.info(f"Writing update.bin at {app_offset} (baud 115200, no erase)")

        # Flash at 115200 baud, no erase, app partition only
        result = await self.flash(
            update_bin,
            erase_all=False,
            verify=True,
            flash_offset=app_offset,
            chip=chip_key,
            baud_override=115200,
        )

        return result

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

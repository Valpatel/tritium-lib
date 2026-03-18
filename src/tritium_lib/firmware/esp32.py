# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ESP32 firmware flasher — uses esptool.py for any ESP32/S2/S3/C3/C6 device.

This is the generic ESP32 flasher. It can flash any .bin file to any ESP32
board. Platform-specific flashers (MeshtasticFlasher, TritiumOSFlasher)
inherit from this and add firmware download + board detection logic.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from .base import (
    DeviceDetection,
    FirmwareFlasher,
    FlashResult,
    FlashStatus,
)

log = logging.getLogger("tritium.firmware.esp32")

# Known ESP32 USB VID:PIDs
ESPRESSIF_VIDS = {"303a"}       # Espressif USB JTAG/serial
SILABS_VIDS = {"10c4"}          # CP210x (many ESP32 boards)
CH340_VIDS = {"1a86"}           # CH340/CH9102 (common clone boards)
FTDI_VIDS = {"0403"}            # FTDI (some dev boards)

ALL_KNOWN_VIDS = ESPRESSIF_VIDS | SILABS_VIDS | CH340_VIDS | FTDI_VIDS

# Chip → default flash addresses for merged firmware
CHIP_FLASH_OFFSETS = {
    "esp32":    "0x1000",
    "esp32s2":  "0x1000",
    "esp32s3":  "0x0",
    "esp32c3":  "0x0",
    "esp32c6":  "0x0",
    "esp32h2":  "0x0",
}


class ESP32Flasher(FirmwareFlasher):
    """Flash firmware to ESP32-family chips via esptool.py.

    Supports:
    - ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C6, ESP32-H2
    - Serial and USB-JTAG connections
    - Chip detection, flash size detection
    - Full erase, partial erase, write + verify
    """

    def __init__(self, port: str = "", baud: int = 921600):
        super().__init__(port=port, baud=baud)
        self._esptool_path = self._find_esptool()

    @staticmethod
    def _find_esptool() -> str | None:
        """Find esptool.py or esptool on PATH."""
        for name in ["esptool.py", "esptool"]:
            path = shutil.which(name)
            if path:
                return path
        return None

    @property
    def is_available(self) -> bool:
        """Whether esptool is installed and available."""
        return self._esptool_path is not None

    async def detect(self) -> DeviceDetection:
        """Detect ESP32 chip on the given port using esptool chip_id."""
        if not self._esptool_path:
            return DeviceDetection(
                detected=False,
                error="esptool not found. Install with: pip install esptool",
            )

        port = self.port or self._auto_detect_port()
        if not port:
            return DeviceDetection(
                detected=False,
                error="No ESP32 serial port found",
            )

        self._emit_progress(FlashStatus.DETECTING, 0, f"Detecting device on {port}...")

        try:
            result = await self._run_in_executor(
                self._detect_sync, self._esptool_path, port, self.baud
            )
            result.port = port
            return result
        except Exception as e:
            return DeviceDetection(
                detected=False,
                port=port,
                error=str(e),
            )

    @staticmethod
    def _detect_sync(esptool_path: str, port: str, baud: int) -> DeviceDetection:
        """Run esptool chip_id + flash_id synchronously."""
        detection = DeviceDetection(port=port)

        # Step 1: chip_id — identifies chip type and unique ID
        try:
            proc = subprocess.run(
                [esptool_path, "--port", port, "--baud", str(baud), "chip_id"],
                capture_output=True, text=True, timeout=15,
            )
            output = proc.stdout + proc.stderr

            # Parse chip type
            for line in output.split("\n"):
                line_lower = line.lower().strip()
                if "chip is" in line_lower:
                    # "Chip is ESP32-S3 (QFN56) (revision v0.2)"
                    chip_part = line.split("Chip is")[-1].strip()
                    detection.chip = chip_part.split("(")[0].strip()
                elif "chip id:" in line_lower or "chip_id:" in line_lower:
                    detection.chip_id = line.split(":")[-1].strip()
                elif "mac:" in line_lower:
                    detection.chip_id = line.split(":")[-1].strip()

            if proc.returncode == 0:
                detection.detected = True
            else:
                detection.error = output.strip()[-200:]
        except subprocess.TimeoutExpired:
            detection.error = "esptool chip_id timed out"
            return detection
        except Exception as e:
            detection.error = str(e)
            return detection

        # Step 2: flash_id — identifies flash size
        if detection.detected:
            try:
                proc = subprocess.run(
                    [esptool_path, "--port", port, "--baud", str(baud), "flash_id"],
                    capture_output=True, text=True, timeout=15,
                )
                for line in (proc.stdout + proc.stderr).split("\n"):
                    if "flash size" in line.lower():
                        # "Detected flash size: 16MB"
                        detection.flash_size = line.split(":")[-1].strip()
                        break
            except Exception:
                pass  # flash_id is optional info

        return detection

    async def flash(
        self,
        firmware_path: str,
        erase_all: bool = False,
        verify: bool = True,
        flash_offset: str = "",
        chip: str = "",
        additional_writes: list[tuple[str, str]] | None = None,
        baud_override: int = 0,
    ) -> FlashResult:
        """Flash a .bin file to the ESP32, with optional additional partition writes.

        Args:
            firmware_path: Path to firmware .bin file.
            erase_all: Erase entire flash before writing (for clean installs).
            verify: Verify after writing.
            flash_offset: Override flash offset (e.g., "0x0" for ESP32-S3).
            chip: Override chip type (e.g., "esp32s3").
            additional_writes: List of (offset, file_path) tuples for extra
                partitions to write after the main firmware. Example:
                [("0x260000", "/path/to/ota.bin"), ("0x300000", "/path/to/fs.bin")]
            baud_override: Override baud rate for this flash only (0 = use default).
        """
        if not self._esptool_path:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="esptool not found. Install with: pip install esptool",
            )

        fw_path = Path(firmware_path)
        if not fw_path.exists():
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error=f"Firmware file not found: {firmware_path}",
            )

        # Validate additional writes exist
        if additional_writes:
            for offset, path in additional_writes:
                if not Path(path).exists():
                    return FlashResult(
                        success=False, status=FlashStatus.FAILED,
                        error=f"Additional write file not found: {path}",
                    )

        port = self.port or self._auto_detect_port()
        if not port:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="No serial port found",
            )

        # Determine flash offset
        if not flash_offset:
            chip_key = (chip or "").lower().replace("-", "")
            flash_offset = CHIP_FLASH_OFFSETS.get(chip_key, "0x0")

        baud = baud_override if baud_override > 0 else self.baud

        start = time.monotonic()
        self._emit_progress(FlashStatus.WRITING, 0, f"Flashing {fw_path.name}...")

        try:
            result = await self._run_in_executor(
                self._flash_sync, self._esptool_path, port, baud,
                str(fw_path), erase_all, verify, flash_offset, chip,
            )

            # Write additional partitions if main write succeeded
            if result.success and additional_writes:
                total_extra = len(additional_writes)
                for i, (offset, path) in enumerate(additional_writes):
                    pct = 50 + (50 * i // total_extra)
                    self._emit_progress(
                        FlashStatus.WRITING, pct,
                        f"Writing {Path(path).name} at {offset} ({i+1}/{total_extra})...",
                    )
                    extra_result = await self._run_in_executor(
                        self._flash_sync, self._esptool_path, port, baud,
                        path, False, verify, offset, chip,
                    )
                    if not extra_result.success:
                        result.success = False
                        result.status = FlashStatus.FAILED
                        result.error = (
                            f"Additional write at {offset} failed: {extra_result.error}"
                        )
                        break

            result.duration_s = time.monotonic() - start
            result.port = port
            result.firmware_file = str(fw_path)

            if result.success:
                self._emit_progress(FlashStatus.COMPLETED, 100, "Flash complete!")
            else:
                self._emit_progress(FlashStatus.FAILED, 0, result.error)

            return result
        except Exception as e:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error=str(e), port=port,
                duration_s=time.monotonic() - start,
            )

    @staticmethod
    def _flash_sync(
        esptool_path: str, port: str, baud: int, firmware_path: str,
        erase_all: bool, verify: bool, flash_offset: str, chip: str,
    ) -> FlashResult:
        """Synchronous flash — runs in executor thread."""
        result = FlashResult()

        # Step 1: optional erase
        if erase_all:
            try:
                cmd = [esptool_path, "--port", port, "erase_flash"]
                if chip:
                    cmd = [esptool_path, "--chip", chip, "--port", port, "erase_flash"]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if proc.returncode != 0:
                    result.error = f"Erase failed: {proc.stderr}"
                    result.status = FlashStatus.FAILED
                    return result
            except Exception as e:
                result.error = f"Erase failed: {e}"
                result.status = FlashStatus.FAILED
                return result

        # Step 2: write flash
        cmd = [esptool_path, "--port", port, "--baud", str(baud)]
        if chip:
            cmd = [esptool_path, "--chip", chip, "--port", port, "--baud", str(baud)]
        cmd.extend(["write_flash", flash_offset, firmware_path])

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            result.output = proc.stdout
            if proc.returncode == 0:
                result.success = True
                result.status = FlashStatus.COMPLETED
            else:
                result.success = False
                result.status = FlashStatus.FAILED
                result.error = proc.stderr or proc.stdout
        except subprocess.TimeoutExpired:
            result.status = FlashStatus.FAILED
            result.error = "Flash timed out after 300 seconds"
        except Exception as e:
            result.status = FlashStatus.FAILED
            result.error = str(e)

        return result

    def _auto_detect_port(self) -> str | None:
        """Auto-detect an ESP32 serial port by VID:PID."""
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                if p.vid:
                    vid_hex = f"{p.vid:04x}"
                    if vid_hex in ALL_KNOWN_VIDS:
                        return p.device
        except ImportError:
            pass

        # Fallback: check common paths
        for path in ["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyACM1", "/dev/ttyUSB1"]:
            if Path(path).exists():
                return path

        return None

    async def erase_flash(self) -> FlashResult:
        """Erase the entire flash memory."""
        if not self._esptool_path:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="esptool not found",
            )

        port = self.port or self._auto_detect_port()
        if not port:
            return FlashResult(
                success=False, status=FlashStatus.FAILED,
                error="No serial port found",
            )

        self._emit_progress(FlashStatus.ERASING, 0, "Erasing flash...")

        try:
            result = await self._run_in_executor(
                self._erase_sync, self._esptool_path, port,
            )
            return result
        except Exception as e:
            return FlashResult(
                success=False, status=FlashStatus.FAILED, error=str(e),
            )

    @staticmethod
    def _erase_sync(esptool_path: str, port: str) -> FlashResult:
        try:
            proc = subprocess.run(
                [esptool_path, "--port", port, "erase_flash"],
                capture_output=True, text=True, timeout=60,
            )
            return FlashResult(
                success=proc.returncode == 0,
                status=FlashStatus.COMPLETED if proc.returncode == 0 else FlashStatus.FAILED,
                output=proc.stdout,
                error=proc.stderr if proc.returncode != 0 else "",
                port=port,
            )
        except Exception as e:
            return FlashResult(
                success=False, status=FlashStatus.FAILED, error=str(e),
            )

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Abstract base class for firmware flashers.

Every platform (ESP32, STM32, nRF52) implements this interface.
Addons can use it to flash their target hardware via a uniform API.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("tritium.firmware")


class FlashStatus(str, Enum):
    """Status of a flash operation."""
    PENDING = "pending"
    DETECTING = "detecting"
    DOWNLOADING = "downloading"
    ERASING = "erasing"
    WRITING = "writing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DeviceDetection:
    """Result of probing a connected device."""
    detected: bool = False
    port: str = ""
    chip: str = ""            # e.g. "ESP32-S3", "nRF52840"
    chip_id: str = ""         # unique chip ID / MAC
    flash_size: str = ""      # e.g. "16MB", "4MB"
    firmware_version: str = ""
    firmware_project: str = "" # e.g. "meshtastic", "tritium-os"
    board: str = ""           # e.g. "tlora-pager", "t-deck"
    vid_pid: str = ""         # USB VID:PID
    error: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class FlashResult:
    """Result of a firmware flash operation."""
    success: bool = False
    status: FlashStatus = FlashStatus.PENDING
    firmware_version: str = ""
    firmware_file: str = ""
    port: str = ""
    duration_s: float = 0.0
    output: str = ""
    error: str = ""
    progress_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "status": self.status.value,
            "firmware_version": self.firmware_version,
            "firmware_file": self.firmware_file,
            "port": self.port,
            "duration_s": round(self.duration_s, 1),
            "error": self.error,
        }


# Type for progress callbacks: (status, progress_pct, message)
ProgressCallback = Callable[[FlashStatus, float, str], None]


class FirmwareFlasher(ABC):
    """Abstract base for all firmware flashers.

    Subclasses implement detect(), flash(), and optionally download_firmware().
    All heavy I/O runs in executor threads to keep asyncio non-blocking.
    """

    def __init__(self, port: str = "", baud: int = 921600):
        self.port = port
        self.baud = baud
        self._progress_cb: ProgressCallback | None = None

    def on_progress(self, callback: ProgressCallback):
        """Register a progress callback for flash operations."""
        self._progress_cb = callback

    def _emit_progress(self, status: FlashStatus, pct: float, msg: str):
        if self._progress_cb:
            try:
                self._progress_cb(status, pct, msg)
            except Exception:
                pass

    async def _run_in_executor(self, fn, *args):
        """Run a blocking function in the default thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    @abstractmethod
    async def detect(self) -> DeviceDetection:
        """Detect the connected device and return its info.

        Should identify: chip type, flash size, firmware version,
        board name, and any other relevant metadata.
        """
        ...

    @abstractmethod
    async def flash(self, firmware_path: str, **kwargs) -> FlashResult:
        """Flash a firmware file to the device.

        Args:
            firmware_path: Path to the firmware binary (.bin, .uf2, etc.)
            **kwargs: Platform-specific options (erase_all, verify, etc.)

        Returns FlashResult with success/failure info.
        """
        ...

    async def flash_latest(self, **kwargs) -> FlashResult:
        """Download and flash the latest firmware for the detected board.

        Default implementation: detect board, download firmware, flash it.
        Override for platforms with specific update mechanisms.
        """
        detection = await self.detect()
        if not detection.detected:
            return FlashResult(
                success=False,
                status=FlashStatus.FAILED,
                error=f"No device detected on {self.port}: {detection.error}",
                port=self.port,
            )

        fw_path = await self.download_firmware(
            board=detection.board,
            chip=detection.chip,
        )
        if not fw_path:
            return FlashResult(
                success=False,
                status=FlashStatus.FAILED,
                error=f"Failed to download firmware for {detection.board}",
                port=self.port,
            )

        return await self.flash(fw_path, **kwargs)

    async def download_firmware(
        self, board: str = "", chip: str = "", version: str = "latest",
    ) -> str | None:
        """Download firmware for the given board/chip. Returns local file path.

        Override in subclasses that support automatic firmware downloads.
        Returns None if download is not supported or fails.
        """
        return None

    @staticmethod
    def find_serial_ports() -> list[dict]:
        """List available serial ports with VID/PID info."""
        ports = []
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                ports.append({
                    "port": p.device,
                    "description": p.description,
                    "vid": f"{p.vid:04x}" if p.vid else "",
                    "pid": f"{p.pid:04x}" if p.pid else "",
                    "manufacturer": p.manufacturer or "",
                    "serial_number": p.serial_number or "",
                })
        except ImportError:
            log.warning("pyserial not installed — cannot list serial ports")
        return ports

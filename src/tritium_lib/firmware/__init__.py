# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Firmware management — generic flasher abstraction with platform implementations.

Architecture:
    FirmwareFlasher (abstract base)
    ├── ESP32Flasher — esptool.py-based flash for any ESP32/S2/S3/C3
    ├── MeshtasticFlasher — downloads + flashes official Meshtastic firmware
    └── (future: STM32Flasher, NRF52Flasher, etc.)

Usage:
    from tritium_lib.firmware import MeshtasticFlasher

    flasher = MeshtasticFlasher(port="/dev/ttyACM0")
    info = await flasher.detect()
    print(info)  # board, chip, firmware version

    # Flash latest official firmware
    result = await flasher.flash_latest()

    # Or flash a specific file
    result = await flasher.flash("/path/to/firmware.bin")
"""

from .base import FirmwareFlasher, FlashResult, DeviceDetection
from .esp32 import ESP32Flasher
from .meshtastic_flasher import MeshtasticFlasher

__all__ = [
    "FirmwareFlasher",
    "FlashResult",
    "DeviceDetection",
    "ESP32Flasher",
    "MeshtasticFlasher",
]

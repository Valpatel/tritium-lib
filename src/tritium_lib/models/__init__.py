# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared data models for the Tritium ecosystem.

These models are the contract between tritium-sc and tritium-edge.
Any device that speaks the Tritium protocol uses these types.
"""

from .device import Device, DeviceGroup, DeviceHeartbeat, DeviceCapabilities
from .command import Command, CommandType, CommandStatus
from .firmware import FirmwareMeta, OTAJob, OTAStatus
from .sensor import SensorReading

__all__ = [
    "Device",
    "DeviceGroup",
    "DeviceHeartbeat",
    "DeviceCapabilities",
    "Command",
    "CommandType",
    "CommandStatus",
    "FirmwareMeta",
    "OTAJob",
    "OTAStatus",
    "SensorReading",
]

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tritium sensor plugin interfaces.

Defines the generic interface hierarchy for all sensor plugins:

    SensorPlugin (base)
      +-- SDRPlugin      — software defined radio
      +-- RadarPlugin    — radar systems
      +-- CameraPlugin   — camera/vision systems

Each generic interface is subclassed by specific hardware implementations
(e.g., SDRPlugin -> HackRFPlugin, RTLSDRPlugin).
"""

from tritium_lib.interfaces.camera_plugin import CameraPlugin
from tritium_lib.interfaces.radar_plugin import RadarPlugin
from tritium_lib.interfaces.sdr_plugin import SDRMonitorConfig, SDRPlugin
from tritium_lib.interfaces.sensor_plugin import SensorPlugin

__all__ = [
    "SensorPlugin",
    "SDRPlugin",
    "SDRMonitorConfig",
    "RadarPlugin",
    "CameraPlugin",
]

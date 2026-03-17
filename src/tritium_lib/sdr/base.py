# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Abstract base class for SDR devices.

Every SDR platform (HackRF, RTL-SDR, LimeSDR) implements this interface.
Addons use it to control radios through a uniform API.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("tritium.sdr")


@dataclass
class SDRInfo:
    """Device identification and capabilities."""
    detected: bool = False
    name: str = ""           # e.g. "HackRF One"
    serial: str = ""
    firmware: str = ""
    api_version: str = ""
    hardware_id: str = ""
    hardware_rev: str = ""
    freq_min_hz: int = 0     # Minimum tunable frequency
    freq_max_hz: int = 0     # Maximum tunable frequency
    sample_rate_max: int = 0 # Maximum sample rate in Hz
    bandwidth_max: int = 0   # Maximum IF bandwidth
    has_tx: bool = False     # Can transmit
    has_bias_tee: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v or isinstance(v, bool)}


@dataclass
class SweepPoint:
    """A single power measurement at a frequency."""
    freq_hz: int = 0
    power_dbm: float = -100.0
    timestamp: float = 0.0


@dataclass
class SweepResult:
    """Result of a broadband frequency sweep."""
    points: list[SweepPoint] = field(default_factory=list)
    freq_start_hz: int = 0
    freq_end_hz: int = 0
    bin_width_hz: int = 0
    sweep_time_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "points": [{"freq": p.freq_hz, "power": round(p.power_dbm, 2)} for p in self.points],
            "freq_start": self.freq_start_hz,
            "freq_end": self.freq_end_hz,
            "bin_width": self.bin_width_hz,
            "sweep_time_ms": round(self.sweep_time_ms, 1),
            "num_points": len(self.points),
            "peak_freq": max(self.points, key=lambda p: p.power_dbm).freq_hz if self.points else 0,
            "peak_power": max(p.power_dbm for p in self.points) if self.points else -100,
            "avg_power": round(sum(p.power_dbm for p in self.points) / len(self.points), 1) if self.points else -100,
        }

    def get_peaks(self, threshold_dbm: float = -30.0) -> list[SweepPoint]:
        """Return points above the given power threshold."""
        return [p for p in self.points if p.power_dbm > threshold_dbm]


class SDRDevice(ABC):
    """Abstract base for all SDR devices."""

    def __init__(self):
        self._info: SDRInfo | None = None

    async def _run_in_executor(self, fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    @abstractmethod
    async def detect(self) -> SDRInfo:
        """Detect the connected SDR device and return its info."""
        ...

    @abstractmethod
    async def sweep(
        self,
        freq_start_hz: int,
        freq_end_hz: int,
        bin_width_hz: int = 500000,
    ) -> SweepResult:
        """Perform a single broadband frequency sweep.

        Args:
            freq_start_hz: Start frequency in Hz
            freq_end_hz: End frequency in Hz
            bin_width_hz: Width of each frequency bin in Hz

        Returns SweepResult with power measurements per bin.
        """
        ...

    @abstractmethod
    async def tune(self, freq_hz: int, sample_rate: int = 2000000, bandwidth: int = 0):
        """Tune the radio to a specific frequency for reception.

        Args:
            freq_hz: Center frequency in Hz
            sample_rate: Sample rate in Hz
            bandwidth: IF bandwidth in Hz (0 = auto)
        """
        ...

    @abstractmethod
    async def stop(self):
        """Stop any running operation (sweep, receive, etc)."""
        ...

    @property
    def info(self) -> SDRInfo | None:
        """Cached device info from last detect() call."""
        return self._info

    @property
    def is_available(self) -> bool:
        """Whether a device was detected."""
        return self._info is not None and self._info.detected

    @staticmethod
    def find_devices() -> list[dict]:
        """List available SDR devices on USB.

        Returns list of dicts with port/vid/pid/description.
        Override in subclasses for device-specific detection.
        """
        devices = []
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                if p.vid:
                    devices.append({
                        "port": p.device,
                        "vid": f"{p.vid:04x}",
                        "pid": f"{p.pid:04x}" if p.pid else "",
                        "description": p.description or "",
                        "serial_number": p.serial_number or "",
                    })
        except ImportError:
            pass
        return devices

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Generic SDR plugin interface.

Extends SensorPlugin with SDR-specific operations: tuning, gain control,
spectrum acquisition, and signal monitoring. Any SDR hardware (HackRF,
RTL-SDR, Airspy, LimeSDR, PlutoSDR) implements this interface.

Plugin Architecture:
    SensorPlugin (base)
      +-- SDRPlugin (this file — generic SDR)
            +-- HackRFPlugin (specific)
            +-- RTLSDRPlugin (specific)
            +-- AirspyPlugin (specific)
            +-- LimeSDRPlugin (specific)
            +-- PlutoSDRPlugin (specific)

MQTT topics (published by SDR plugins):
    tritium/{site}/sdr/{receiver}/signal    — detected RF signals
    tritium/{site}/sdr/{receiver}/spectrum   — spectrum scan sweeps
    tritium/{site}/sdr/{receiver}/ism        — decoded ISM devices
"""

from abc import abstractmethod
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from tritium_lib.interfaces.sensor_plugin import SensorPlugin
from tritium_lib.models.sdr import RFSignal, SpectrumScan


class SDRMonitorConfig(BaseModel):
    """Configuration for continuous SDR monitoring.

    Defines which frequencies to watch, how to tune the receiver,
    and which protocols to attempt decoding.
    """

    frequencies_mhz: list[float] = Field(default_factory=list)
    center_freq_mhz: float = 433.92
    span_mhz: float = 2.0
    sample_rate_hz: int = 2_048_000
    gain_db: float = 40.0
    squelch_db: float = -50.0
    decode_protocols: list[str] = Field(default_factory=list)
    scan_interval_s: float = 1.0
    enabled: bool = True
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class SDRPlugin(SensorPlugin):
    """Generic SDR plugin interface.

    HackRF, RTL-SDR, Airspy, LimeSDR, and other SDR hardware implement
    this interface. Provides frequency tuning, gain control, spectrum
    acquisition, and signal detection.

    Implementations must also satisfy SensorPlugin methods (get_name,
    get_sensor_type, start, stop, get_status, get_mqtt_topics, get_capabilities).
    """

    @abstractmethod
    def get_frequency_range(self) -> tuple[float, float]:
        """Return supported frequency range in Hz as (min_hz, max_hz)."""
        ...

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Return current sample rate in Hz."""
        ...

    @abstractmethod
    def tune(self, frequency_hz: float) -> None:
        """Tune the receiver to the specified center frequency in Hz."""
        ...

    @abstractmethod
    def set_gain(self, gain_db: float) -> None:
        """Set receiver gain in dB."""
        ...

    @abstractmethod
    def get_spectrum(self, center_freq_mhz: float, span_mhz: float) -> SpectrumScan:
        """Acquire a spectrum scan around center_freq_mhz with given span.

        Returns a SpectrumScan with power spectral density bins.
        """
        ...

    @abstractmethod
    def start_monitoring(self, config: SDRMonitorConfig) -> None:
        """Start continuous monitoring with the given configuration.

        The plugin will scan configured frequencies and publish detected
        signals to MQTT topics.
        """
        ...

    @abstractmethod
    def get_detected_signals(self) -> list[RFSignal]:
        """Return signals detected since last call (or since monitoring started).

        Each call may clear the internal buffer, depending on implementation.
        """
        ...

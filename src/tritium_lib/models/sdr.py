# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SDR (Software Defined Radio) models for spectrum monitoring and device decoding.

Covers raw RF signal detection, spectrum sweeps, ISM-band device decoding
(via rtl_433), and lightweight ADS-B/AIS track representations for
SDR-decoded data.

These are the raw SDR data models. For full-featured AIS/ADS-B entity
models with complete maritime/aviation fields, see models/ais.py.

MQTT topics:
    tritium/{site}/sdr/{receiver}/signal    — detected RF signals
    tritium/{site}/sdr/{receiver}/spectrum   — spectrum scan sweeps
    tritium/{site}/sdr/{receiver}/ism        — decoded ISM devices (rtl_433)
    tritium/{site}/sdr/{receiver}/adsb       — raw ADS-B track updates
    tritium/{site}/sdr/{receiver}/ais        — raw AIS track updates
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Modulation(str, Enum):
    """Common RF modulation types."""

    AM = "am"
    FM = "fm"
    USB = "usb"  # upper sideband
    LSB = "lsb"  # lower sideband
    CW = "cw"  # continuous wave
    OOK = "ook"  # on-off keying
    FSK = "fsk"  # frequency shift keying
    GFSK = "gfsk"  # gaussian FSK
    PSK = "psk"  # phase shift keying
    QPSK = "qpsk"
    OFDM = "ofdm"
    LORA = "lora"
    UNKNOWN = "unknown"


class RFSignal(BaseModel):
    """A detected RF signal/emission.

    Represents a signal observed during spectrum monitoring,
    e.g., from an RTL-SDR doing a frequency sweep.
    """

    frequency_mhz: float  # center frequency in MHz
    bandwidth_khz: float = 0.0  # signal bandwidth in kHz
    power_dbm: float = 0.0  # signal power in dBm
    modulation: Modulation = Modulation.UNKNOWN
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    source_id: str = ""  # SDR receiver ID


class SpectrumScan(BaseModel):
    """A spectrum sweep / power spectral density measurement.

    Represents a series of power measurements across a frequency span,
    as produced by rtl_power or similar tools.
    """

    center_freq_mhz: float  # center of the sweep
    span_mhz: float  # total frequency span
    bins: list[float] = Field(default_factory=list)  # power values in dBm per bin
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    source_id: str = ""

    @property
    def bin_count(self) -> int:
        """Number of frequency bins."""
        return len(self.bins)

    @property
    def bin_width_khz(self) -> float:
        """Width of each frequency bin in kHz."""
        if not self.bins:
            return 0.0
        return (self.span_mhz * 1000.0) / len(self.bins)

    @property
    def start_freq_mhz(self) -> float:
        """Start frequency of the sweep."""
        return self.center_freq_mhz - self.span_mhz / 2.0

    @property
    def end_freq_mhz(self) -> float:
        """End frequency of the sweep."""
        return self.center_freq_mhz + self.span_mhz / 2.0

    def peak_power_dbm(self) -> float:
        """Maximum power across all bins."""
        return max(self.bins) if self.bins else -999.0

    def peak_frequency_mhz(self) -> float:
        """Frequency of the strongest bin."""
        if not self.bins:
            return self.center_freq_mhz
        idx = self.bins.index(max(self.bins))
        return self.start_freq_mhz + idx * self.bin_width_khz / 1000.0


class ISMDevice(BaseModel):
    """A decoded ISM-band device (e.g., from rtl_433).

    rtl_433 decodes hundreds of sensor protocols in the 315/433/868/915 MHz
    ISM bands: weather stations, tire pressure monitors, door sensors,
    temperature/humidity probes, etc.
    """

    device_type: str = ""  # rtl_433 model name (e.g., "Acurite-5n1")
    protocol: str = ""  # protocol identifier
    frequency_mhz: float = 433.92  # ISM frequency
    device_id: str = ""  # device-specific ID
    payload: dict = Field(default_factory=dict)  # raw decoded fields
    temperature: Optional[float] = None  # degrees C
    humidity: Optional[float] = None  # percent
    battery_pct: Optional[float] = None  # 0-100
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    source_id: str = ""  # SDR receiver ID

    def to_target_dict(self) -> dict:
        """Convert to a dict suitable for TargetTracker ingestion."""
        tid = f"ism_{self.device_type}_{self.device_id}" if self.device_id else f"ism_{self.device_type}"
        return {
            "target_id": tid,
            "source": "sdr_ism",
            "classification": "ism_device",
            "alliance": "unknown",
            "metadata": {
                "device_type": self.device_type,
                "protocol": self.protocol,
                "frequency_mhz": self.frequency_mhz,
                "temperature": self.temperature,
                "humidity": self.humidity,
                "battery_pct": self.battery_pct,
                **self.payload,
            },
        }


class ADSBTrack(BaseModel):
    """Lightweight ADS-B track from SDR decoding (dump1090/readsb).

    For the full-featured ADS-B model with aircraft details,
    see models.ais.ADSBFlight.
    """

    icao_hex: str  # 6-char hex ICAO 24-bit address
    callsign: str = ""
    altitude_ft: float = 0.0
    speed_kts: float = 0.0  # ground speed in knots
    heading_deg: float = 0.0  # track heading
    lat: float = 0.0
    lng: float = 0.0
    squawk: str = ""  # 4-digit octal squawk code
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    source_id: str = ""

    def compute_target_id(self) -> str:
        """Generate Tritium target ID."""
        return f"adsb_{self.icao_hex.lower()}"


class AISTrack(BaseModel):
    """Lightweight AIS track from SDR decoding (rtl_ais/gnuais).

    For the full-featured AIS model with complete vessel details,
    see models.ais.AISVessel.
    """

    mmsi: str  # 9-digit Maritime Mobile Service Identity (as string for leading zeros)
    vessel_name: str = ""
    vessel_type: str = ""  # human-readable type
    lat: float = 0.0
    lng: float = 0.0
    course_deg: float = 0.0  # course over ground
    speed_kts: float = 0.0  # speed over ground in knots
    destination: str = ""
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    source_id: str = ""

    def compute_target_id(self) -> str:
        """Generate Tritium target ID."""
        return f"ais_{self.mmsi}"

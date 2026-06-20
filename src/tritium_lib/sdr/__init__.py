# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SDR (Software Defined Radio) abstractions for Tritium.

Provides a generic SDRDevice base class that specific hardware implementations
(HackRF, RTL-SDR, LimeSDR, etc.) extend. Each device can:
- Detect and enumerate hardware
- Tune to a frequency
- Capture IQ samples
- Run broadband sweeps
- Report device status

Architecture mirrors the firmware flasher pattern:
    SDRDevice (ABC)
    ├── HackRFDevice — hackrf_* CLI tools
    ├── RTLSDRDevice — rtl_* CLI tools (future)
    ├── LimeSDRDevice — LimeSuite tools (future)
    └── SimulatedSDR — pure software simulation (demos/testing)

Spectrum analysis layer:
    SpectrumAnalyzer — signal detection, classification, waterfall display
    SimulatedSignal — configurable RF signal source for simulation
"""

from .base import SDRDevice, SDRInfo, SweepResult, SweepPoint
from .simulator import SimulatedSDR, SimulatedSignal, default_signal_environment
from .iq_synth import (
    CRC24_GENERATOR,
    PREAMBLE_PULSE_POSITIONS,
    SAMPLES_PER_US,
    build_df17_frame,
    crc24,
    synth_modes_iq,
)
from .analyzer import (
    SpectrumAnalyzer,
    DetectedSignal,
    FrequencyBand,
    ScanPreset,
    WaterfallRow,
    KNOWN_BANDS,
    SCAN_PRESETS,
)

__all__ = [
    "SDRDevice",
    "SDRInfo",
    "SweepResult",
    "SweepPoint",
    "SimulatedSDR",
    "SimulatedSignal",
    "default_signal_environment",
    "CRC24_GENERATOR",
    "PREAMBLE_PULSE_POSITIONS",
    "SAMPLES_PER_US",
    "build_df17_frame",
    "crc24",
    "synth_modes_iq",
    "SpectrumAnalyzer",
    "DetectedSignal",
    "FrequencyBand",
    "ScanPreset",
    "WaterfallRow",
    "KNOWN_BANDS",
    "SCAN_PRESETS",
]

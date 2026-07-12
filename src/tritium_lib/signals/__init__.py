# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Signal analysis toolkit for RF signals (BLE RSSI, WiFi CSI, SDR spectrum).

Provides four core analysers:

- **RSSIAnalyzer** — RSSI time-series analysis with Kalman smoothing,
  log-distance path loss distance estimation, and variance-based motion
  detection.

- **SignalFingerprint** — Build unique RF fingerprints from BLE/WiFi
  observations (RSSI histogram, beacon interval, channel usage, service
  UUIDs) and compare fingerprints for device re-identification.

- **SignalSpectrumAnalyzer** — Pure-math frequency-domain analysis: peak
  detection, band classification, spectral entropy, and periodogram.
  (Distinct from the device-backed ``tritium_lib.sdr.SpectrumAnalyzer``.)

- **CSIProcessor** — WiFi Channel State Information processing for
  occupancy detection using subcarrier variance analysis with Hampel
  filtering and adaptive baseline.

All algorithms are pure Python (stdlib ``math`` only, no numpy required).

Usage::

    from tritium_lib.signals import RSSIAnalyzer, SignalFingerprint
    from tritium_lib.signals import SignalSpectrumAnalyzer, CSIProcessor
"""

from .rssi_analyzer import (
    RSSIAnalyzer,
    RSSIReading,
    RSSIStats,
    MotionResult,
)

from .fingerprint import (
    SignalFingerprint,
)

from .spectrum import (
    SignalSpectrumAnalyzer,
    SpectralPeak,
    SpectralSummary,
    BandClassification,
    BandDef,
    BAND_TABLE,
    WIFI_24_CHANNELS,
    WIFI_5_CHANNELS,
)

from .csi_processor import (
    CSIProcessor,
    OccupancyResult,
    CSIStats,
    SubcarrierBand,
    hampel_filter,
)

# gcc_phat needs numpy (FFT). Keep it optional so a core-only (no-numpy)
# install still imports the pure-Python analysers above.
try:
    from .gcc_phat import gcc_phat as gcc_phat
    _HAS_GCC_PHAT = True
except ImportError:  # pragma: no cover - numpy absent
    _HAS_GCC_PHAT = False

__all__ = [
    # RSSI
    "RSSIAnalyzer",
    "RSSIReading",
    "RSSIStats",
    "MotionResult",
    # Fingerprint
    "SignalFingerprint",
    # Spectrum
    "SignalSpectrumAnalyzer",
    "SpectralPeak",
    "SpectralSummary",
    "BandClassification",
    "BandDef",
    "BAND_TABLE",
    "WIFI_24_CHANNELS",
    "WIFI_5_CHANNELS",
    # CSI
    "CSIProcessor",
    "OccupancyResult",
    "CSIStats",
    "SubcarrierBand",
    "hampel_filter",
]

if _HAS_GCC_PHAT:
    __all__.append("gcc_phat")

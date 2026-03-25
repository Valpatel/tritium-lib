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

- **SpectrumAnalyzer** — Pure-math frequency-domain analysis: peak
  detection, band classification, spectral entropy, and periodogram.

- **CSIProcessor** — WiFi Channel State Information processing for
  occupancy detection using subcarrier variance analysis with Hampel
  filtering and adaptive baseline.

All algorithms are pure Python (stdlib ``math`` only, no numpy required).

Usage::

    from tritium_lib.signals import RSSIAnalyzer, SignalFingerprint
    from tritium_lib.signals import SpectrumAnalyzer, CSIProcessor
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
    SpectrumAnalyzer,
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

__all__ = [
    # RSSI
    "RSSIAnalyzer",
    "RSSIReading",
    "RSSIStats",
    "MotionResult",
    # Fingerprint
    "SignalFingerprint",
    # Spectrum
    "SpectrumAnalyzer",
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

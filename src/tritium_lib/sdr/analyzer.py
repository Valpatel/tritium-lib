# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Spectrum analyzer engine for SDR devices.

Wraps any SDRDevice (real or simulated) and provides:
  - Signal detection with classification
  - Waterfall display data (time x frequency power matrix)
  - Frequency band scanning presets
  - Peak detection and signal tracking over time

Designed for the SDR plugin UI and standalone demo use.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .base import SDRDevice, SweepResult, SweepPoint


# ---------------------------------------------------------------------------
# Known frequency bands for classification
# ---------------------------------------------------------------------------

@dataclass
class FrequencyBand:
    """A known RF frequency band."""
    name: str
    start_hz: int
    end_hz: int
    category: str
    description: str = ""


KNOWN_BANDS: list[FrequencyBand] = [
    FrequencyBand("FM Broadcast", 88_000_000, 108_000_000, "broadcast",
                  "Commercial FM radio stations"),
    FrequencyBand("Aircraft VHF", 118_000_000, 137_000_000, "aviation",
                  "Air traffic control and aircraft comms"),
    FrequencyBand("VHF Marine", 156_000_000, 162_000_000, "marine",
                  "Marine VHF channels"),
    FrequencyBand("ISM 315 MHz", 314_000_000, 316_000_000, "ism",
                  "TPMS, key fobs, garage doors"),
    FrequencyBand("ISM 390 MHz", 389_000_000, 391_000_000, "ism",
                  "Garage doors, car remotes"),
    FrequencyBand("ISM 433 MHz", 433_000_000, 435_000_000, "ism",
                  "Weather stations, sensors, remotes"),
    FrequencyBand("UHF TV", 470_000_000, 698_000_000, "broadcast",
                  "Digital TV broadcast"),
    FrequencyBand("LTE Band 13", 746_000_000, 756_000_000, "cellular",
                  "Verizon LTE 700 MHz"),
    FrequencyBand("LTE Band 5", 859_000_000, 894_000_000, "cellular",
                  "AT&T LTE 850 MHz"),
    FrequencyBand("ISM 868 MHz", 868_000_000, 868_600_000, "ism",
                  "LoRa EU, SRD"),
    FrequencyBand("ISM 902-928 MHz", 902_000_000, 928_000_000, "ism",
                  "LoRa US, Meshtastic, Z-Wave"),
    FrequencyBand("GSM 900", 935_000_000, 960_000_000, "cellular",
                  "GSM downlink 900 MHz"),
    FrequencyBand("ADS-B", 1_088_000_000, 1_092_000_000, "aviation",
                  "Aircraft transponder 1090 MHz"),
    FrequencyBand("GPS L1", 1_575_000_000, 1_576_000_000, "navigation",
                  "GPS L1 civilian"),
    FrequencyBand("LTE Band 2", 1_930_000_000, 1_990_000_000, "cellular",
                  "T-Mobile/AT&T PCS"),
    FrequencyBand("WiFi 2.4 GHz", 2_400_000_000, 2_500_000_000, "wifi",
                  "802.11b/g/n/ax channels 1-14"),
    FrequencyBand("BLE", 2_400_000_000, 2_483_500_000, "ble",
                  "Bluetooth Low Energy advertising + data"),
    FrequencyBand("WiFi 5 GHz", 5_150_000_000, 5_850_000_000, "wifi",
                  "802.11a/n/ac/ax UNII bands"),
]


# ---------------------------------------------------------------------------
# Scan presets
# ---------------------------------------------------------------------------

@dataclass
class ScanPreset:
    """A named frequency range for quick scanning."""
    name: str
    start_hz: int
    end_hz: int
    bin_width_hz: int
    description: str = ""


SCAN_PRESETS: list[ScanPreset] = [
    ScanPreset("FM Broadcast", 88_000_000, 108_000_000, 100_000,
               "Commercial FM radio band"),
    ScanPreset("ISM 433 MHz", 430_000_000, 440_000_000, 50_000,
               "ISM devices, weather stations, remotes"),
    ScanPreset("LoRa US", 900_000_000, 930_000_000, 50_000,
               "LoRa/Meshtastic US ISM band"),
    ScanPreset("ADS-B", 1_080_000_000, 1_100_000_000, 100_000,
               "Aircraft transponders"),
    ScanPreset("WiFi 2.4 GHz", 2_400_000_000, 2_500_000_000, 500_000,
               "WiFi and BLE"),
    ScanPreset("Wideband 0-3 GHz", 1_000_000, 3_000_000_000, 5_000_000,
               "Full wideband overview sweep"),
    ScanPreset("Cellular 700 MHz", 700_000_000, 800_000_000, 200_000,
               "LTE Band 12/13/14/17"),
    ScanPreset("Cellular PCS", 1_900_000_000, 2_000_000_000, 200_000,
               "PCS 1900 MHz cellular"),
]


# ---------------------------------------------------------------------------
# Detected signal
# ---------------------------------------------------------------------------

@dataclass
class DetectedSignal:
    """A signal detected during spectrum analysis."""
    freq_hz: int
    power_dbm: float
    bandwidth_hz: int = 0
    snr_db: float = 0.0
    category: str = "unknown"
    band_name: str = ""
    modulation: str = "unknown"
    timestamp: float = 0.0
    persistent: bool = False  # seen in multiple sweeps

    def to_dict(self) -> dict:
        return {
            "freq_hz": self.freq_hz,
            "freq_mhz": round(self.freq_hz / 1e6, 3),
            "power_dbm": round(self.power_dbm, 1),
            "bandwidth_hz": self.bandwidth_hz,
            "bandwidth_khz": round(self.bandwidth_hz / 1e3, 1),
            "snr_db": round(self.snr_db, 1),
            "category": self.category,
            "band_name": self.band_name,
            "modulation": self.modulation,
            "timestamp": self.timestamp,
            "persistent": self.persistent,
        }


# ---------------------------------------------------------------------------
# Waterfall row
# ---------------------------------------------------------------------------

@dataclass
class WaterfallRow:
    """One time-slice of the waterfall display."""
    timestamp: float
    powers: list[float]  # dBm values, one per frequency bin
    freq_start_hz: int = 0
    freq_end_hz: int = 0


# ---------------------------------------------------------------------------
# SpectrumAnalyzer
# ---------------------------------------------------------------------------

class SpectrumAnalyzer:
    """High-level spectrum analysis engine.

    Wraps an SDRDevice and provides signal detection, classification,
    and waterfall history.

    Usage::

        from tritium_lib.sdr.simulator import SimulatedSDR
        sdr = SimulatedSDR(seed=42)
        analyzer = SpectrumAnalyzer(sdr)
        await analyzer.initialize()

        # Run a sweep and detect signals
        result = await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=100_000)
        signals = analyzer.detect_signals(result, threshold_dbm=-50.0)
        for sig in signals:
            print(f"{sig.freq_hz/1e6:.1f} MHz  {sig.power_dbm:.1f} dBm  [{sig.category}]")

        # Get waterfall data
        waterfall = analyzer.get_waterfall()
    """

    def __init__(
        self,
        device: SDRDevice,
        waterfall_depth: int = 100,
        noise_floor_dbm: float = -95.0,
    ):
        self.device = device
        self.waterfall_depth = waterfall_depth
        self.noise_floor_dbm = noise_floor_dbm
        self._waterfall: deque[WaterfallRow] = deque(maxlen=waterfall_depth)
        self._signal_history: dict[int, list[float]] = {}  # freq -> list of powers
        self._sweep_count: int = 0
        self._total_signals_detected: int = 0
        self._last_sweep: Optional[SweepResult] = None
        self._last_signals: list[DetectedSignal] = []
        self._initialized: bool = False

    async def initialize(self) -> SDRDevice:
        """Detect and initialize the SDR device."""
        await self.device.detect()
        self._initialized = True
        return self.device

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self.device.is_available

    async def scan(
        self,
        freq_start_hz: int,
        freq_end_hz: int,
        bin_width_hz: int = 500_000,
    ) -> SweepResult:
        """Run a spectrum sweep and record to waterfall history."""
        result = await self.device.sweep(freq_start_hz, freq_end_hz, bin_width_hz)
        self._last_sweep = result
        self._sweep_count += 1

        # Append to waterfall
        powers = [p.power_dbm for p in result.points]
        self._waterfall.append(WaterfallRow(
            timestamp=result.timestamp,
            powers=powers,
            freq_start_hz=freq_start_hz,
            freq_end_hz=freq_end_hz,
        ))

        return result

    def detect_signals(
        self,
        sweep: Optional[SweepResult] = None,
        threshold_dbm: float = -60.0,
        min_snr_db: float = 6.0,
    ) -> list[DetectedSignal]:
        """Detect signals above threshold in a sweep result.

        Uses peak detection with noise-floor estimation and band classification.
        """
        if sweep is None:
            sweep = self._last_sweep
        if sweep is None or not sweep.points:
            return []

        # Estimate local noise floor from the bottom 25th percentile
        sorted_powers = sorted(p.power_dbm for p in sweep.points)
        q25_idx = max(0, len(sorted_powers) // 4)
        local_noise = sorted_powers[q25_idx] if sorted_powers else self.noise_floor_dbm

        # Find peaks: points above threshold AND above noise + min_snr
        effective_threshold = max(threshold_dbm, local_noise + min_snr_db)

        signals: list[DetectedSignal] = []
        points = sweep.points
        n = len(points)

        i = 0
        while i < n:
            if points[i].power_dbm > effective_threshold:
                # Found a signal — find its extent (contiguous above noise + 3dB)
                peak_power = points[i].power_dbm
                peak_idx = i
                start_idx = i
                end_idx = i

                # Expand right
                while end_idx + 1 < n and points[end_idx + 1].power_dbm > local_noise + 3.0:
                    end_idx += 1
                    if points[end_idx].power_dbm > peak_power:
                        peak_power = points[end_idx].power_dbm
                        peak_idx = end_idx

                # Expand left
                while start_idx > 0 and points[start_idx - 1].power_dbm > local_noise + 3.0:
                    start_idx -= 1

                # Compute bandwidth from bin extent
                bandwidth = (end_idx - start_idx + 1) * sweep.bin_width_hz

                # Classify by frequency band
                freq = points[peak_idx].freq_hz
                category, band_name = self._classify_frequency(freq)

                # Track signal persistence
                bucket = (freq // 1_000_000) * 1_000_000  # round to nearest MHz
                hist = self._signal_history.setdefault(bucket, [])
                hist.append(peak_power)
                if len(hist) > 50:
                    hist.pop(0)

                snr = peak_power - local_noise

                sig = DetectedSignal(
                    freq_hz=freq,
                    power_dbm=peak_power,
                    bandwidth_hz=bandwidth,
                    snr_db=snr,
                    category=category,
                    band_name=band_name,
                    timestamp=sweep.timestamp,
                    persistent=len(hist) >= 3,
                )
                signals.append(sig)
                self._total_signals_detected += 1

                # Skip past this signal
                i = end_idx + 1
            else:
                i += 1

        self._last_signals = signals
        return signals

    def _classify_frequency(self, freq_hz: int) -> tuple[str, str]:
        """Classify a frequency into a known band.

        Returns (category, band_name).
        """
        for band in KNOWN_BANDS:
            if band.start_hz <= freq_hz <= band.end_hz:
                return band.category, band.name
        return "unknown", ""

    def get_waterfall(self, max_rows: int = 0) -> dict:
        """Return waterfall display data as a serializable dict.

        Returns:
            {
                "rows": [[power, power, ...], ...],  # newest first
                "timestamps": [float, ...],
                "freq_start_hz": int,
                "freq_end_hz": int,
                "num_bins": int,
                "num_rows": int,
            }
        """
        if not self._waterfall:
            return {
                "rows": [],
                "timestamps": [],
                "freq_start_hz": 0,
                "freq_end_hz": 0,
                "num_bins": 0,
                "num_rows": 0,
            }

        rows_list = list(self._waterfall)
        if max_rows > 0:
            rows_list = rows_list[-max_rows:]

        # Reverse so newest is first
        rows_list.reverse()

        last = rows_list[0]
        return {
            "rows": [[round(p, 1) for p in row.powers] for row in rows_list],
            "timestamps": [row.timestamp for row in rows_list],
            "freq_start_hz": last.freq_start_hz,
            "freq_end_hz": last.freq_end_hz,
            "num_bins": len(last.powers),
            "num_rows": len(rows_list),
        }

    def get_status(self) -> dict:
        """Return analyzer status summary."""
        info = self.device.info
        return {
            "initialized": self._initialized,
            "device": info.to_dict() if info else {},
            "sweep_count": self._sweep_count,
            "total_signals_detected": self._total_signals_detected,
            "waterfall_depth": len(self._waterfall),
            "waterfall_max": self.waterfall_depth,
            "last_signals_count": len(self._last_signals),
            "noise_floor_dbm": self.noise_floor_dbm,
            "tracked_frequencies": len(self._signal_history),
        }

    @staticmethod
    def get_scan_presets() -> list[dict]:
        """Return available scan presets."""
        return [
            {
                "name": p.name,
                "start_hz": p.start_hz,
                "end_hz": p.end_hz,
                "bin_width_hz": p.bin_width_hz,
                "description": p.description,
                "start_mhz": round(p.start_hz / 1e6, 1),
                "end_mhz": round(p.end_hz / 1e6, 1),
            }
            for p in SCAN_PRESETS
        ]

    @staticmethod
    def get_known_bands() -> list[dict]:
        """Return known frequency bands for display/reference."""
        return [
            {
                "name": b.name,
                "start_hz": b.start_hz,
                "end_hz": b.end_hz,
                "category": b.category,
                "description": b.description,
                "start_mhz": round(b.start_hz / 1e6, 1),
                "end_mhz": round(b.end_hz / 1e6, 1),
            }
            for b in KNOWN_BANDS
        ]

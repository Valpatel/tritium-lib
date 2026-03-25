# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Frequency-domain spectrum analysis — peak detection and band classification.

Unlike ``tritium_lib.sdr.analyzer.SpectrumAnalyzer`` (which wraps SDR hardware
devices), this module provides *pure math* tools for analysing any frequency-
domain data: IQ magnitude arrays, FFT output, or synthetic spectra.

Key capabilities:
  - Peak detection with configurable prominence and noise floor estimation
  - Spectral band classification (ISM, cellular, WiFi, BLE, etc.)
  - Power spectral density estimation via periodogram (pure-Python DFT)
  - Spectral entropy for signal complexity measurement

Pure Python — stdlib math only.

Usage::

    from tritium_lib.signals import SpectrumAnalyzer

    analyzer = SpectrumAnalyzer()

    # Feed raw frequency/power pairs
    peaks = analyzer.find_peaks(
        frequencies_hz=[2.4e9 + i * 1e6 for i in range(100)],
        powers_dbm=[-90 + (10 if 30 < i < 35 else 0) for i in range(100)],
    )
    for p in peaks:
        print(f"{p.freq_hz / 1e6:.1f} MHz  {p.power_dbm:.1f} dBm  [{p.band}]")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Band classification table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandDef:
    """A known RF frequency band definition."""
    name: str
    start_hz: float
    end_hz: float
    category: str


BAND_TABLE: list[BandDef] = [
    BandDef("FM Broadcast", 88e6, 108e6, "broadcast"),
    BandDef("Aircraft VHF", 118e6, 137e6, "aviation"),
    BandDef("VHF Marine", 156e6, 162e6, "marine"),
    BandDef("ISM 315 MHz", 314e6, 316e6, "ism"),
    BandDef("ISM 433 MHz", 433e6, 435e6, "ism"),
    BandDef("ISM 868 MHz", 868e6, 868.6e6, "ism"),
    BandDef("ISM 902-928 MHz", 902e6, 928e6, "ism"),
    BandDef("GSM 900", 935e6, 960e6, "cellular"),
    BandDef("ADS-B 1090", 1088e6, 1092e6, "aviation"),
    BandDef("GPS L1", 1575e6, 1576e6, "navigation"),
    BandDef("LTE Band 2", 1930e6, 1990e6, "cellular"),
    BandDef("WiFi 2.4 GHz", 2400e6, 2500e6, "wifi"),
    BandDef("BLE 2.4 GHz", 2400e6, 2483.5e6, "ble"),
    BandDef("WiFi 5 GHz", 5150e6, 5850e6, "wifi"),
    BandDef("WiFi 6E", 5925e6, 7125e6, "wifi"),
]

# WiFi 2.4 GHz channel centre frequencies
WIFI_24_CHANNELS: dict[int, float] = {
    ch: (2412e6 + (ch - 1) * 5e6) for ch in range(1, 15)
}

# WiFi 5 GHz channel centres (common UNII-1 through UNII-3)
WIFI_5_CHANNELS: dict[int, float] = {
    36: 5180e6, 40: 5200e6, 44: 5220e6, 48: 5240e6,
    52: 5260e6, 56: 5280e6, 60: 5300e6, 64: 5320e6,
    100: 5500e6, 104: 5520e6, 108: 5540e6, 112: 5560e6,
    116: 5580e6, 120: 5600e6, 124: 5620e6, 128: 5640e6,
    132: 5660e6, 136: 5680e6, 140: 5700e6, 149: 5745e6,
    153: 5765e6, 157: 5785e6, 161: 5805e6, 165: 5825e6,
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SpectralPeak:
    """A peak detected in the frequency domain."""
    freq_hz: float
    power_dbm: float
    prominence_db: float
    bandwidth_hz: float
    snr_db: float
    band: str
    category: str

    def to_dict(self) -> dict:
        return {
            "freq_hz": self.freq_hz,
            "freq_mhz": round(self.freq_hz / 1e6, 3),
            "power_dbm": round(self.power_dbm, 2),
            "prominence_db": round(self.prominence_db, 2),
            "bandwidth_hz": self.bandwidth_hz,
            "snr_db": round(self.snr_db, 2),
            "band": self.band,
            "category": self.category,
        }


@dataclass
class BandClassification:
    """Result of classifying a frequency into known bands."""
    freq_hz: float
    band_name: str
    category: str
    wifi_channel: int | None = None

    def to_dict(self) -> dict:
        result: dict = {
            "freq_hz": self.freq_hz,
            "freq_mhz": round(self.freq_hz / 1e6, 3),
            "band_name": self.band_name,
            "category": self.category,
        }
        if self.wifi_channel is not None:
            result["wifi_channel"] = self.wifi_channel
        return result


@dataclass
class SpectralSummary:
    """Summary statistics for a spectrum segment."""
    num_bins: int
    noise_floor_dbm: float
    peak_power_dbm: float
    peak_freq_hz: float
    mean_power_dbm: float
    spectral_entropy: float
    num_peaks: int

    def to_dict(self) -> dict:
        return {
            "num_bins": self.num_bins,
            "noise_floor_dbm": round(self.noise_floor_dbm, 2),
            "peak_power_dbm": round(self.peak_power_dbm, 2),
            "peak_freq_hz": self.peak_freq_hz,
            "mean_power_dbm": round(self.mean_power_dbm, 2),
            "spectral_entropy": round(self.spectral_entropy, 4),
            "num_peaks": self.num_peaks,
        }


# ---------------------------------------------------------------------------
# SpectrumAnalyzer
# ---------------------------------------------------------------------------

class SpectrumAnalyzer:
    """Pure-math frequency-domain spectrum analyser.

    Provides peak detection, band classification, noise floor estimation,
    and spectral entropy calculation.  Works on any frequency/power data
    regardless of the data source (SDR, simulated, WiFi scan, etc.).

    Args:
        noise_floor_dbm: Default noise floor estimate in dBm.
        min_prominence_db: Minimum prominence for peak detection.
    """

    def __init__(
        self,
        noise_floor_dbm: float = -95.0,
        min_prominence_db: float = 6.0,
    ) -> None:
        self._noise_floor = noise_floor_dbm
        self._min_prominence = min_prominence_db

    # -- Peak detection -----------------------------------------------------

    def find_peaks(
        self,
        frequencies_hz: list[float],
        powers_dbm: list[float],
        threshold_dbm: float | None = None,
        min_prominence_db: float | None = None,
    ) -> list[SpectralPeak]:
        """Detect peaks in a frequency-domain power spectrum.

        Uses a three-point local-maximum test with noise-floor-relative
        prominence filtering.

        Args:
            frequencies_hz: Frequency values (must be same length as powers).
            powers_dbm: Power values in dBm.
            threshold_dbm: Absolute power threshold. Defaults to
                noise_floor + min_prominence.
            min_prominence_db: Override default min prominence.

        Returns:
            List of SpectralPeak, sorted by power (strongest first).
        """
        n = len(frequencies_hz)
        if n != len(powers_dbm) or n < 3:
            return []

        noise = self._estimate_noise_floor(powers_dbm)
        prom = min_prominence_db if min_prominence_db is not None else self._min_prominence
        thresh = threshold_dbm if threshold_dbm is not None else (noise + prom)

        peaks: list[SpectralPeak] = []

        i = 1
        while i < n - 1:
            p = powers_dbm[i]
            if p > powers_dbm[i - 1] and p > powers_dbm[i + 1] and p >= thresh:
                # Found a local maximum — measure its extent
                prominence = p - noise
                if prominence < prom:
                    i += 1
                    continue

                # Determine -3 dB bandwidth
                half_power = p - 3.0
                left = i
                while left > 0 and powers_dbm[left - 1] >= half_power:
                    left -= 1
                right = i
                while right < n - 1 and powers_dbm[right + 1] >= half_power:
                    right += 1

                bw = frequencies_hz[right] - frequencies_hz[left] if right > left else 0.0

                # Classify band
                band_name, category = self.classify_frequency(frequencies_hz[i])

                peaks.append(SpectralPeak(
                    freq_hz=frequencies_hz[i],
                    power_dbm=p,
                    prominence_db=prominence,
                    bandwidth_hz=abs(bw),
                    snr_db=p - noise,
                    band=band_name,
                    category=category,
                ))

                # Skip past this peak
                i = right + 1
            else:
                i += 1

        peaks.sort(key=lambda pk: pk.power_dbm, reverse=True)
        return peaks

    # -- Band classification ------------------------------------------------

    @staticmethod
    def classify_frequency(freq_hz: float) -> tuple[str, str]:
        """Classify a frequency into known RF bands.

        Returns (band_name, category). Category is one of:
        broadcast, aviation, marine, ism, cellular, navigation, wifi, ble, unknown.
        """
        for band in BAND_TABLE:
            if band.start_hz <= freq_hz <= band.end_hz:
                return band.name, band.category
        return "unknown", "unknown"

    @staticmethod
    def classify_frequency_detailed(freq_hz: float) -> BandClassification:
        """Classify with additional detail (WiFi channel number, etc.)."""
        band_name = "unknown"
        category = "unknown"
        wifi_ch: int | None = None

        for band in BAND_TABLE:
            if band.start_hz <= freq_hz <= band.end_hz:
                band_name = band.name
                category = band.category
                break

        # Try to match the closest WiFi 2.4 GHz channel (within 11 MHz)
        best_dist = float("inf")
        for ch, center in WIFI_24_CHANNELS.items():
            dist = abs(freq_hz - center)
            if dist < 11e6 and dist < best_dist:
                best_dist = dist
                wifi_ch = ch

        # Try to match the closest WiFi 5 GHz channel (within 20 MHz)
        if wifi_ch is None:
            best_dist = float("inf")
            for ch, center in WIFI_5_CHANNELS.items():
                dist = abs(freq_hz - center)
                if dist < 20e6 and dist < best_dist:
                    best_dist = dist
                    wifi_ch = ch

        return BandClassification(
            freq_hz=freq_hz,
            band_name=band_name,
            category=category,
            wifi_channel=wifi_ch,
        )

    # -- Noise floor estimation ---------------------------------------------

    @staticmethod
    def _estimate_noise_floor(powers_dbm: list[float]) -> float:
        """Estimate noise floor as the 25th percentile of power values."""
        if not powers_dbm:
            return -95.0
        sorted_p = sorted(powers_dbm)
        q25_idx = max(0, len(sorted_p) // 4)
        return sorted_p[q25_idx]

    # -- Spectral summary ---------------------------------------------------

    def summarize(
        self,
        frequencies_hz: list[float],
        powers_dbm: list[float],
    ) -> SpectralSummary:
        """Compute summary statistics for a spectrum segment.

        Includes noise floor, peak, mean power, spectral entropy,
        and the number of detected peaks.
        """
        n = len(powers_dbm)
        if n == 0:
            return SpectralSummary(
                num_bins=0,
                noise_floor_dbm=self._noise_floor,
                peak_power_dbm=self._noise_floor,
                peak_freq_hz=0.0,
                mean_power_dbm=self._noise_floor,
                spectral_entropy=0.0,
                num_peaks=0,
            )

        noise = self._estimate_noise_floor(powers_dbm)
        peak_idx = max(range(n), key=lambda i: powers_dbm[i])
        mean_p = sum(powers_dbm) / n

        # Spectral entropy (on linear power values)
        entropy = self._spectral_entropy(powers_dbm)

        peaks = self.find_peaks(frequencies_hz, powers_dbm)

        return SpectralSummary(
            num_bins=n,
            noise_floor_dbm=noise,
            peak_power_dbm=powers_dbm[peak_idx],
            peak_freq_hz=frequencies_hz[peak_idx] if frequencies_hz else 0.0,
            mean_power_dbm=mean_p,
            spectral_entropy=entropy,
            num_peaks=len(peaks),
        )

    @staticmethod
    def _spectral_entropy(powers_dbm: list[float]) -> float:
        """Compute normalised spectral entropy [0, 1].

        Convert dBm to linear power, normalise to a distribution,
        and compute Shannon entropy. Returns 0 for pure tone, ~1 for noise.
        """
        # Convert dBm to milliwatts (linear)
        linear = [10.0 ** (p / 10.0) for p in powers_dbm]
        total = sum(linear)
        if total <= 0:
            return 0.0

        # Normalise to probability distribution
        probs = [v / total for v in linear]

        # Shannon entropy
        h = 0.0
        for p in probs:
            if p > 1e-15:
                h -= p * math.log2(p)

        # Normalise by max entropy (uniform distribution)
        n = len(probs)
        h_max = math.log2(n) if n > 1 else 1.0
        return h / h_max if h_max > 0 else 0.0

    # -- Simple periodogram (pure Python DFT) -------------------------------

    @staticmethod
    def periodogram(
        samples: list[float],
        sample_rate_hz: float,
    ) -> tuple[list[float], list[float]]:
        """Compute power spectral density via periodogram (DFT magnitude squared).

        This is a pure-Python DFT — works on small sample counts (< 4096).
        For production SDR use, prefer the sdr.SpectrumAnalyzer with numpy.

        Args:
            samples: Time-domain signal values.
            sample_rate_hz: Sampling rate in Hz.

        Returns:
            (frequencies_hz, powers_dbm) — positive-frequency half only.
        """
        n = len(samples)
        if n == 0:
            return [], []

        # DFT (brute-force — O(n^2), fine for n < 4096)
        half = n // 2
        freqs: list[float] = []
        powers: list[float] = []

        for k in range(half + 1):
            re = 0.0
            im = 0.0
            for t in range(n):
                angle = -2.0 * math.pi * k * t / n
                re += samples[t] * math.cos(angle)
                im += samples[t] * math.sin(angle)
            mag_sq = (re * re + im * im) / (n * n)
            # Convert to dBm (assuming signal is in volts relative to 1mW/50ohm)
            power_dbm = 10.0 * math.log10(max(mag_sq, 1e-20))
            freqs.append(k * sample_rate_hz / n)
            powers.append(power_dbm)

        return freqs, powers

    # -- Status -------------------------------------------------------------

    def get_status(self) -> dict:
        """Return analyzer configuration as a dict."""
        return {
            "noise_floor_dbm": self._noise_floor,
            "min_prominence_db": self._min_prominence,
            "num_bands": len(BAND_TABLE),
            "num_wifi_24_channels": len(WIFI_24_CHANNELS),
            "num_wifi_5_channels": len(WIFI_5_CHANNELS),
        }

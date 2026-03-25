# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Simulated SDR device for demos and testing.

Generates realistic spectrum data with configurable signal sources.
No real SDR hardware needed — pure software simulation of:
  - FM broadcast stations (88-108 MHz)
  - WiFi access points (2.4 GHz, 5 GHz)
  - BLE advertisements (2.402-2.480 GHz)
  - LoRa transmissions (902-928 MHz US, 868 MHz EU)
  - ISM-band devices (315/433/915 MHz)
  - ADS-B aircraft transponders (1090 MHz)
  - Cellular bands (700/850/1900 MHz)

Each signal source is modeled as a Gaussian power peak with realistic
bandwidth, power level, and optional time-varying behavior (fading,
intermittent transmission, Doppler shift).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from .base import SDRDevice, SDRInfo, SweepPoint, SweepResult


@dataclass
class SimulatedSignal:
    """A simulated RF signal source.

    Attributes:
        name:          Human-readable label (e.g., "WNYC 93.9 FM")
        freq_hz:       Center frequency in Hz
        bandwidth_hz:  Signal bandwidth in Hz
        power_dbm:     Peak power in dBm
        modulation:    Modulation type string
        category:      Signal category for grouping (fm, wifi, ble, lora, ism, adsb, cellular)
        active:        Whether the signal is currently transmitting
        intermittent:  If True, signal randomly appears/disappears
        duty_cycle:    Fraction of time the signal is active (0.0-1.0)
        drift_hz:      Max frequency drift in Hz (simulates oscillator instability)
        fade_depth_db: Max fading depth in dB (simulates multipath/distance)
        fade_rate_hz:  Fading rate in Hz (how fast it fades)
    """
    name: str = ""
    freq_hz: int = 0
    bandwidth_hz: int = 200000  # 200 kHz default
    power_dbm: float = -40.0
    modulation: str = "unknown"
    category: str = "unknown"
    active: bool = True
    intermittent: bool = False
    duty_cycle: float = 1.0
    drift_hz: int = 0
    fade_depth_db: float = 0.0
    fade_rate_hz: float = 0.1

    def current_power(self, t: float, rng: random.Random) -> float:
        """Calculate instantaneous power with fading and intermittent behavior."""
        if not self.active:
            return -200.0

        if self.intermittent:
            # Use duty cycle to determine if transmitting
            cycle_period = 1.0 / max(self.fade_rate_hz, 0.01)
            phase = (t % cycle_period) / cycle_period
            if phase > self.duty_cycle:
                return -200.0

        power = self.power_dbm

        # Apply slow fading
        if self.fade_depth_db > 0:
            fade = self.fade_depth_db * math.sin(2.0 * math.pi * self.fade_rate_hz * t)
            power += fade

        # Add noise jitter (+/- 2 dB)
        power += rng.gauss(0, 0.7)

        return power

    def current_freq(self, t: float, rng: random.Random) -> int:
        """Calculate instantaneous frequency with drift."""
        freq = self.freq_hz
        if self.drift_hz > 0:
            drift = int(self.drift_hz * math.sin(0.05 * t))
            freq += drift
        return freq


# ---------------------------------------------------------------------------
# Pre-built signal databases
# ---------------------------------------------------------------------------

def _fm_broadcast_signals() -> list[SimulatedSignal]:
    """Common FM broadcast stations."""
    stations = [
        ("WNYC 93.9", 93_900_000),
        ("Z100 100.3", 100_300_000),
        ("Power 105.1", 105_100_000),
        ("WBLS 107.5", 107_500_000),
        ("NPR 90.7", 90_700_000),
        ("Classic 96.3", 96_300_000),
        ("Country 99.5", 99_500_000),
        ("Rock 104.3", 104_300_000),
    ]
    return [
        SimulatedSignal(
            name=name,
            freq_hz=freq,
            bandwidth_hz=200_000,
            power_dbm=random.uniform(-25, -10),
            modulation="fm",
            category="fm",
            fade_depth_db=random.uniform(1, 4),
            fade_rate_hz=random.uniform(0.02, 0.1),
        )
        for name, freq in stations
    ]


def _wifi_signals() -> list[SimulatedSignal]:
    """WiFi access points on 2.4 GHz channels."""
    # 2.4 GHz channels 1, 6, 11 are most common
    channels = {
        1: 2_412_000_000,
        6: 2_437_000_000,
        11: 2_462_000_000,
        3: 2_422_000_000,
        9: 2_452_000_000,
    }
    signals = []
    names = ["HomeNetwork", "CoffeeShop_5G", "NETGEAR42", "linksys", "ATT-WiFi",
             "xfinitywifi", "DIRECT-TV", "MyRouter"]
    for i, (ch, freq) in enumerate(channels.items()):
        if i < len(names):
            signals.append(SimulatedSignal(
                name=f"{names[i]} (Ch{ch})",
                freq_hz=freq,
                bandwidth_hz=22_000_000,  # WiFi is ~22 MHz wide
                power_dbm=random.uniform(-55, -30),
                modulation="ofdm",
                category="wifi",
                fade_depth_db=random.uniform(2, 8),
                fade_rate_hz=random.uniform(0.05, 0.3),
            ))
    return signals


def _ble_signals() -> list[SimulatedSignal]:
    """BLE advertisement channels."""
    # BLE advertising on channels 37, 38, 39
    adv_channels = {
        37: 2_402_000_000,
        38: 2_426_000_000,
        39: 2_480_000_000,
    }
    signals = []
    devices = ["iPhone-Matt", "Galaxy-S24", "AirTag-Keys", "FitBit", "AirPods"]
    for dev in devices:
        ch = random.choice(list(adv_channels.keys()))
        signals.append(SimulatedSignal(
            name=f"BLE:{dev}",
            freq_hz=adv_channels[ch],
            bandwidth_hz=2_000_000,  # BLE channel is 2 MHz
            power_dbm=random.uniform(-75, -45),
            modulation="gfsk",
            category="ble",
            intermittent=True,
            duty_cycle=random.uniform(0.1, 0.4),
            fade_rate_hz=random.uniform(0.5, 2.0),
        ))
    return signals


def _lora_signals() -> list[SimulatedSignal]:
    """LoRa/Meshtastic signals in ISM bands."""
    return [
        SimulatedSignal(
            name="Meshtastic-Node1",
            freq_hz=906_000_000,
            bandwidth_hz=125_000,
            power_dbm=random.uniform(-90, -60),
            modulation="lora",
            category="lora",
            intermittent=True,
            duty_cycle=0.05,
            fade_rate_hz=0.02,
        ),
        SimulatedSignal(
            name="Meshtastic-Node2",
            freq_hz=906_000_000,
            bandwidth_hz=125_000,
            power_dbm=random.uniform(-95, -65),
            modulation="lora",
            category="lora",
            intermittent=True,
            duty_cycle=0.03,
            fade_rate_hz=0.015,
        ),
        SimulatedSignal(
            name="LoRaWAN-Sensor",
            freq_hz=903_000_000,
            bandwidth_hz=125_000,
            power_dbm=random.uniform(-85, -55),
            modulation="lora",
            category="lora",
            intermittent=True,
            duty_cycle=0.01,
            fade_rate_hz=0.005,
        ),
    ]


def _ism_signals() -> list[SimulatedSignal]:
    """ISM-band devices (garage doors, weather stations, TPMS, etc.)."""
    return [
        SimulatedSignal(
            name="WeatherStation-Acurite",
            freq_hz=433_920_000,
            bandwidth_hz=50_000,
            power_dbm=-55.0,
            modulation="ook",
            category="ism",
            intermittent=True,
            duty_cycle=0.02,
            fade_rate_hz=0.017,  # ~60s cycle
        ),
        SimulatedSignal(
            name="TPMS-FrontLeft",
            freq_hz=315_000_000,
            bandwidth_hz=100_000,
            power_dbm=-65.0,
            modulation="fsk",
            category="ism",
            intermittent=True,
            duty_cycle=0.005,
            fade_rate_hz=0.01,
        ),
        SimulatedSignal(
            name="GarageDoor-Remote",
            freq_hz=390_000_000,
            bandwidth_hz=50_000,
            power_dbm=-50.0,
            modulation="ook",
            category="ism",
            intermittent=True,
            duty_cycle=0.001,
            fade_rate_hz=0.003,
        ),
        SimulatedSignal(
            name="KeyFob-315",
            freq_hz=315_000_000,
            bandwidth_hz=80_000,
            power_dbm=-60.0,
            modulation="ask",
            category="ism",
            intermittent=True,
            duty_cycle=0.002,
            fade_rate_hz=0.005,
        ),
    ]


def _adsb_signal() -> list[SimulatedSignal]:
    """ADS-B transponder at 1090 MHz."""
    return [
        SimulatedSignal(
            name="ADSB-Transponder",
            freq_hz=1_090_000_000,
            bandwidth_hz=2_000_000,
            power_dbm=-45.0,
            modulation="ppm",
            category="adsb",
            fade_depth_db=10.0,
            fade_rate_hz=0.05,
        ),
    ]


def _cellular_signals() -> list[SimulatedSignal]:
    """Cellular towers — LTE bands."""
    return [
        SimulatedSignal(
            name="LTE-Band13-Verizon",
            freq_hz=751_000_000,
            bandwidth_hz=10_000_000,
            power_dbm=-35.0,
            modulation="ofdm",
            category="cellular",
            fade_depth_db=3.0,
            fade_rate_hz=0.02,
        ),
        SimulatedSignal(
            name="LTE-Band2-TMobile",
            freq_hz=1_930_000_000,
            bandwidth_hz=15_000_000,
            power_dbm=-40.0,
            modulation="ofdm",
            category="cellular",
            fade_depth_db=4.0,
            fade_rate_hz=0.03,
        ),
        SimulatedSignal(
            name="LTE-Band5-ATT",
            freq_hz=869_000_000,
            bandwidth_hz=10_000_000,
            power_dbm=-38.0,
            modulation="ofdm",
            category="cellular",
            fade_depth_db=2.5,
            fade_rate_hz=0.025,
        ),
    ]


def default_signal_environment() -> list[SimulatedSignal]:
    """Create a complete RF environment with all signal types."""
    signals: list[SimulatedSignal] = []
    signals.extend(_fm_broadcast_signals())
    signals.extend(_wifi_signals())
    signals.extend(_ble_signals())
    signals.extend(_lora_signals())
    signals.extend(_ism_signals())
    signals.extend(_adsb_signal())
    signals.extend(_cellular_signals())
    return signals


# ---------------------------------------------------------------------------
# SimulatedSDR device
# ---------------------------------------------------------------------------

class SimulatedSDR(SDRDevice):
    """A fully simulated SDR device that generates realistic spectrum data.

    Creates spectrum sweeps from a configurable set of SimulatedSignal sources.
    Noise floor, signal peaks, and time-varying behavior are all modeled.

    Usage::

        sdr = SimulatedSDR()
        info = await sdr.detect()
        result = await sdr.sweep(88_000_000, 108_000_000, bin_width_hz=100_000)
        for peak in result.get_peaks(threshold_dbm=-40.0):
            print(f"{peak.freq_hz/1e6:.1f} MHz: {peak.power_dbm:.1f} dBm")
    """

    def __init__(
        self,
        signals: Optional[list[SimulatedSignal]] = None,
        noise_floor_dbm: float = -95.0,
        noise_variance_db: float = 3.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.signals = signals if signals is not None else default_signal_environment()
        self.noise_floor_dbm = noise_floor_dbm
        self.noise_variance_db = noise_variance_db
        self.rng = random.Random(seed)
        self._running = False
        self._tuned_freq: int = 0
        self._sample_rate: int = 0
        self._sweep_count: int = 0

    async def detect(self) -> SDRInfo:
        """Return simulated device info."""
        self._info = SDRInfo(
            detected=True,
            name="Tritium Simulated SDR",
            serial="SIM-0001",
            firmware="1.0.0-sim",
            api_version="1.0",
            hardware_id="SIM",
            hardware_rev="v1",
            freq_min_hz=1_000_000,       # 1 MHz
            freq_max_hz=6_000_000_000,   # 6 GHz
            sample_rate_max=20_000_000,  # 20 Msps
            bandwidth_max=20_000_000,
            has_tx=False,
            has_bias_tee=False,
        )
        return self._info

    async def sweep(
        self,
        freq_start_hz: int,
        freq_end_hz: int,
        bin_width_hz: int = 500_000,
    ) -> SweepResult:
        """Generate a simulated spectrum sweep.

        For each frequency bin, computes:
        1. Gaussian noise floor
        2. Contribution from each signal source in range
        3. Time-varying fading and intermittent behavior
        """
        t0 = time.time()
        t = time.monotonic()  # for signal time-variation
        self._sweep_count += 1

        num_bins = max(1, (freq_end_hz - freq_start_hz) // bin_width_hz)
        points: list[SweepPoint] = []

        for i in range(num_bins):
            bin_center = freq_start_hz + i * bin_width_hz + bin_width_hz // 2

            # Start with noise floor
            noise = self.noise_floor_dbm + self.rng.gauss(0, self.noise_variance_db)
            power_linear = 10.0 ** (noise / 10.0)

            # Add contribution from each signal
            for sig in self.signals:
                sig_freq = sig.current_freq(t, self.rng)
                sig_power = sig.current_power(t, self.rng)

                if sig_power < -150.0:
                    continue  # signal is off

                # Gaussian spectral shape: power falls off with distance from center
                freq_offset = abs(bin_center - sig_freq)
                half_bw = sig.bandwidth_hz / 2.0

                if freq_offset < half_bw * 3:  # within 3x bandwidth
                    # Gaussian rolloff
                    sigma = half_bw / 2.355  # FWHM to sigma
                    if sigma > 0:
                        attenuation_db = -0.5 * (freq_offset / sigma) ** 2
                        signal_power_db = sig_power + attenuation_db
                        signal_linear = 10.0 ** (signal_power_db / 10.0)
                        power_linear += signal_linear

            # Convert back to dBm
            if power_linear > 0:
                power_dbm = 10.0 * math.log10(power_linear)
            else:
                power_dbm = self.noise_floor_dbm

            points.append(SweepPoint(
                freq_hz=bin_center,
                power_dbm=round(power_dbm, 2),
                timestamp=t0,
            ))

        sweep_time_ms = (time.time() - t0) * 1000.0

        return SweepResult(
            points=points,
            freq_start_hz=freq_start_hz,
            freq_end_hz=freq_end_hz,
            bin_width_hz=bin_width_hz,
            sweep_time_ms=sweep_time_ms,
            timestamp=t0,
        )

    async def tune(self, freq_hz: int, sample_rate: int = 2_000_000, bandwidth: int = 0):
        """Simulate tuning to a frequency."""
        self._tuned_freq = freq_hz
        self._sample_rate = sample_rate
        self._running = True

    async def stop(self):
        """Stop the simulated receiver."""
        self._running = False
        self._tuned_freq = 0

    @property
    def sweep_count(self) -> int:
        """Number of sweeps performed."""
        return self._sweep_count

    @property
    def tuned_frequency(self) -> int:
        """Currently tuned frequency (0 if not tuned)."""
        return self._tuned_freq

    def add_signal(self, signal: SimulatedSignal) -> None:
        """Add a signal source to the simulation."""
        self.signals.append(signal)

    def remove_signal(self, name: str) -> bool:
        """Remove a signal source by name. Returns True if found."""
        before = len(self.signals)
        self.signals = [s for s in self.signals if s.name != name]
        return len(self.signals) < before

    def get_signals_in_range(self, freq_start_hz: int, freq_end_hz: int) -> list[SimulatedSignal]:
        """Return signals whose center frequency falls within the given range."""
        return [
            s for s in self.signals
            if freq_start_hz <= s.freq_hz <= freq_end_hz and s.active
        ]

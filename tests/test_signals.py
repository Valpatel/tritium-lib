# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.signals — RF signal analysis toolkit.

Covers: RSSIAnalyzer, SignalFingerprint, SpectrumAnalyzer, CSIProcessor.
"""

import math
import pytest

from tritium_lib.signals import (
    RSSIAnalyzer,
    RSSIReading,
    RSSIStats,
    MotionResult,
    SignalFingerprint,
    SpectrumAnalyzer,
    SpectralPeak,
    SpectralSummary,
    BandClassification,
    BandDef,
    BAND_TABLE,
    WIFI_24_CHANNELS,
    WIFI_5_CHANNELS,
    CSIProcessor,
    OccupancyResult,
    CSIStats,
    SubcarrierBand,
    hampel_filter,
)


# ===========================================================================
# RSSIAnalyzer Tests
# ===========================================================================

class TestRSSIAnalyzer:
    """Tests for RSSI time-series analysis."""

    def test_add_reading_returns_smoothed(self):
        """add_reading should return a Kalman-smoothed value."""
        analyzer = RSSIAnalyzer()
        val = analyzer.add_reading("dev1", -65.0, timestamp=1000.0)
        assert isinstance(val, float)
        # First reading should be close to the raw value
        assert abs(val - (-65.0)) < 5.0

    def test_smoothing_reduces_noise(self):
        """Kalman filter should produce lower variance than raw readings."""
        analyzer = RSSIAnalyzer()
        raw = [-65, -70, -64, -72, -66, -69, -63, -71, -67, -68]
        smoothed = []
        for i, r in enumerate(raw):
            s = analyzer.add_reading("dev1", float(r), timestamp=1000.0 + i)
            smoothed.append(s)

        raw_var = sum((v - sum(raw) / len(raw)) ** 2 for v in raw) / len(raw)
        smooth_var = sum((v - sum(smoothed) / len(smoothed)) ** 2
                         for v in smoothed) / len(smoothed)
        assert smooth_var < raw_var

    def test_get_smoothed_unknown_device(self):
        """get_smoothed returns None for unknown device."""
        analyzer = RSSIAnalyzer()
        assert analyzer.get_smoothed("nonexistent") is None

    def test_estimate_distance_basic(self):
        """Distance should be around 1m when RSSI equals tx_power."""
        analyzer = RSSIAnalyzer(tx_power=-59.0, path_loss_exponent=2.5)
        analyzer.add_reading("dev1", -59.0, timestamp=1000.0)
        dist = analyzer.estimate_distance("dev1")
        assert dist is not None
        assert abs(dist - 1.0) < 0.5

    def test_estimate_distance_farther(self):
        """Weaker RSSI should yield greater distance."""
        analyzer = RSSIAnalyzer(tx_power=-59.0, path_loss_exponent=2.0)
        analyzer.add_reading("close", -59.0, timestamp=1000.0)
        analyzer.add_reading("far", -79.0, timestamp=1000.0)
        d_close = analyzer.estimate_distance("close")
        d_far = analyzer.estimate_distance("far")
        assert d_close is not None and d_far is not None
        assert d_far > d_close

    def test_estimate_distance_unknown_device(self):
        """estimate_distance returns None for unknown device."""
        analyzer = RSSIAnalyzer()
        assert analyzer.estimate_distance("nope") is None

    def test_detect_motion_stationary(self):
        """Constant RSSI should be classified as stationary."""
        analyzer = RSSIAnalyzer(motion_variance_threshold=4.0)
        for i in range(20):
            analyzer.add_reading("dev1", -65.0 + (i % 2) * 0.5,
                                 timestamp=1000.0 + i)
        result = analyzer.detect_motion("dev1", window_seconds=30.0)
        assert result is not None
        assert not result.is_moving
        assert result.classification == "stationary"

    def test_detect_motion_moving(self):
        """Wide RSSI swing should be classified as moving."""
        analyzer = RSSIAnalyzer(motion_variance_threshold=4.0)
        # Simulate large RSSI changes
        rssi_values = [-50, -70, -45, -75, -48, -72, -46, -74, -50, -70,
                       -48, -73, -47, -71, -49, -68, -52, -74, -46, -70]
        for i, r in enumerate(rssi_values):
            analyzer.add_reading("dev1", float(r), timestamp=1000.0 + i)
        result = analyzer.detect_motion("dev1", window_seconds=30.0)
        assert result is not None
        assert result.is_moving
        assert result.classification in ("slow", "fast", "erratic")

    def test_detect_motion_insufficient_data(self):
        """detect_motion returns None with fewer than 3 readings."""
        analyzer = RSSIAnalyzer()
        analyzer.add_reading("dev1", -65.0, timestamp=1000.0)
        assert analyzer.detect_motion("dev1") is None

    def test_get_stats(self):
        """get_stats should return correct summary statistics."""
        analyzer = RSSIAnalyzer()
        values = [-60.0, -65.0, -70.0, -55.0, -75.0]
        for i, v in enumerate(values):
            analyzer.add_reading("dev1", v, timestamp=1000.0 + i)
        stats = analyzer.get_stats("dev1")
        assert stats is not None
        assert stats.count == 5
        assert stats.min_val == -75.0
        assert stats.max_val == -55.0
        assert abs(stats.mean - (-65.0)) < 0.01

    def test_get_stats_unknown(self):
        """get_stats returns None for unknown device."""
        analyzer = RSSIAnalyzer()
        assert analyzer.get_stats("nope") is None

    def test_get_tracked_devices(self):
        """get_tracked_devices lists all devices with readings."""
        analyzer = RSSIAnalyzer()
        analyzer.add_reading("dev_a", -65.0, timestamp=1000.0)
        analyzer.add_reading("dev_b", -70.0, timestamp=1000.0)
        devices = analyzer.get_tracked_devices()
        assert "dev_a" in devices
        assert "dev_b" in devices

    def test_remove_device(self):
        """remove_device should stop tracking a device."""
        analyzer = RSSIAnalyzer()
        analyzer.add_reading("dev1", -65.0, timestamp=1000.0)
        assert analyzer.remove_device("dev1") is True
        assert analyzer.get_smoothed("dev1") is None
        assert analyzer.remove_device("dev1") is False

    def test_clear(self):
        """clear should remove all devices."""
        analyzer = RSSIAnalyzer()
        analyzer.add_reading("a", -65.0, timestamp=1000.0)
        analyzer.add_reading("b", -70.0, timestamp=1000.0)
        analyzer.clear()
        assert len(analyzer.get_tracked_devices()) == 0

    def test_get_readings(self):
        """get_readings should return serializable dicts."""
        analyzer = RSSIAnalyzer()
        analyzer.add_reading("dev1", -65.0, timestamp=1000.0)
        analyzer.add_reading("dev1", -67.0, timestamp=1001.0)
        readings = analyzer.get_readings("dev1")
        assert len(readings) == 2
        assert "rssi_dbm" in readings[0]
        assert "smoothed_dbm" in readings[0]
        assert "timestamp" in readings[0]

    def test_motion_result_to_dict(self):
        """MotionResult.to_dict should produce a serializable dict."""
        result = MotionResult(
            is_moving=True, variance=5.5, trend_dbm_per_sec=0.3,
            confidence=0.8, classification="slow",
        )
        d = result.to_dict()
        assert d["is_moving"] is True
        assert d["classification"] == "slow"

    def test_get_status(self):
        """get_status should include configuration details."""
        analyzer = RSSIAnalyzer(tx_power=-50.0, path_loss_exponent=3.0)
        status = analyzer.get_status()
        assert status["tx_power"] == -50.0
        assert status["path_loss_exponent"] == 3.0


# ===========================================================================
# SignalFingerprint Tests
# ===========================================================================

class TestSignalFingerprint:
    """Tests for signal fingerprinting."""

    def test_basic_fingerprint(self):
        """Adding observations should update internal state."""
        fp = SignalFingerprint("dev1")
        fp.add_observation(rssi=-65.0, channel=37, timestamp=1000.0)
        fp.add_observation(rssi=-67.0, channel=38, timestamp=1001.0)
        assert fp.observation_count == 2
        assert fp.first_seen == 1000.0
        assert fp.last_seen == 1001.0

    def test_mean_rssi(self):
        """mean_rssi should reflect the histogram."""
        fp = SignalFingerprint("dev1")
        for _ in range(50):
            fp.add_observation(rssi=-65.0)
        # Mean should be near -65 (within bin quantization)
        assert abs(fp.mean_rssi - (-65.0)) < 5.0

    def test_beacon_interval_estimation(self):
        """estimated_beacon_interval should detect regular intervals."""
        fp = SignalFingerprint("dev1")
        # Simulate 100ms beacon interval
        for i in range(30):
            fp.add_observation(rssi=-65.0, timestamp=1000.0 + i * 0.1)
        bi = fp.estimated_beacon_interval
        assert bi is not None
        assert abs(bi - 0.1) < 0.02

    def test_dominant_channel(self):
        """dominant_channel should return the most-seen channel."""
        fp = SignalFingerprint("dev1")
        for _ in range(10):
            fp.add_observation(rssi=-65.0, channel=37)
        for _ in range(3):
            fp.add_observation(rssi=-65.0, channel=38)
        assert fp.dominant_channel == 37

    def test_self_similarity_is_high(self):
        """A fingerprint compared to itself should score high."""
        fp = SignalFingerprint("dev1")
        for i in range(20):
            fp.add_observation(rssi=-65.0, channel=37, timestamp=1000.0 + i,
                               service_uuids=["0000180f-0000-1000-8000-00805f9b34fb"])
        sim = fp.compare(fp)
        assert sim > 0.95

    def test_different_fingerprints_low_similarity(self):
        """Very different devices should have low similarity."""
        fp1 = SignalFingerprint("phone")
        for i in range(30):
            fp1.add_observation(rssi=-50.0, channel=37, timestamp=1000.0 + i * 0.1,
                                service_uuids=["0000180f-0000-1000-8000-00805f9b34fb"])

        fp2 = SignalFingerprint("sensor")
        for i in range(30):
            fp2.add_observation(rssi=-90.0, channel=39, timestamp=1000.0 + i * 5.0,
                                service_uuids=["00001809-0000-1000-8000-00805f9b34fb"])

        sim = fp1.compare(fp2)
        assert sim < 0.7

    def test_similar_fingerprints_high_similarity(self):
        """Similar devices should have high similarity."""
        fp1 = SignalFingerprint("dev_a")
        fp2 = SignalFingerprint("dev_b")
        for i in range(30):
            fp1.add_observation(rssi=-65.0 + (i % 3), channel=37,
                                timestamp=1000.0 + i * 0.1)
            fp2.add_observation(rssi=-66.0 + (i % 3), channel=37,
                                timestamp=1000.0 + i * 0.1)
        sim = fp1.compare(fp2)
        assert sim > 0.8

    def test_to_dict(self):
        """to_dict should produce all expected keys."""
        fp = SignalFingerprint("dev1")
        fp.add_observation(rssi=-65.0, channel=37, timestamp=1000.0,
                           device_name="TestDevice")
        d = fp.to_dict()
        assert d["device_id"] == "dev1"
        assert "rssi_histogram" in d
        assert "hour_histogram" in d
        assert "name_fragments" in d
        assert d["observation_count"] == 1

    def test_service_uuid_overlap(self):
        """Service UUID Jaccard should boost similarity."""
        fp1 = SignalFingerprint("a")
        fp2 = SignalFingerprint("b")
        uuids = ["0000180f-0000-1000-8000-00805f9b34fb",
                  "00001800-0000-1000-8000-00805f9b34fb"]
        for i in range(10):
            fp1.add_observation(rssi=-65.0, timestamp=1000.0 + i,
                                service_uuids=uuids)
            fp2.add_observation(rssi=-65.0, timestamp=1000.0 + i,
                                service_uuids=uuids)
        sim = fp1.compare(fp2)
        assert sim > 0.9


# ===========================================================================
# SpectrumAnalyzer Tests
# ===========================================================================

class TestSpectrumAnalyzer:
    """Tests for frequency-domain spectrum analysis."""

    def _make_spectrum(self, n=200, peak_idx=100, peak_power=-40.0,
                       noise=-90.0, center_hz=2.4e9, bin_hz=1e6):
        """Helper to create a synthetic spectrum with one peak."""
        freqs = [center_hz + i * bin_hz for i in range(n)]
        powers = [noise] * n
        for offset in range(-3, 4):
            idx = peak_idx + offset
            if 0 <= idx < n:
                atten = abs(offset) * 5.0
                powers[idx] = peak_power - atten
        return freqs, powers

    def test_find_peaks_basic(self):
        """find_peaks should detect a clear peak."""
        sa = SpectrumAnalyzer()
        freqs, powers = self._make_spectrum()
        peaks = sa.find_peaks(freqs, powers)
        assert len(peaks) >= 1
        # Strongest peak should be near -40 dBm
        assert peaks[0].power_dbm > -50.0

    def test_find_peaks_empty(self):
        """find_peaks returns empty list for flat noise."""
        sa = SpectrumAnalyzer()
        freqs = [2.4e9 + i * 1e6 for i in range(100)]
        powers = [-90.0] * 100
        peaks = sa.find_peaks(freqs, powers)
        assert len(peaks) == 0

    def test_find_peaks_multiple(self):
        """find_peaks should detect multiple distinct peaks."""
        sa = SpectrumAnalyzer()
        n = 200
        freqs = [2.4e9 + i * 1e6 for i in range(n)]
        powers = [-90.0] * n
        # Two peaks separated by 50 bins
        for offset in range(-2, 3):
            idx = 50 + offset
            if 0 <= idx < n:
                powers[idx] = -40.0 - abs(offset) * 5.0
            idx = 150 + offset
            if 0 <= idx < n:
                powers[idx] = -45.0 - abs(offset) * 5.0
        peaks = sa.find_peaks(freqs, powers)
        assert len(peaks) >= 2

    def test_classify_frequency_wifi(self):
        """WiFi 2.4 GHz should be classified correctly."""
        name, cat = SpectrumAnalyzer.classify_frequency(2.437e9)
        assert cat == "wifi" or cat == "ble"  # overlapping band
        assert name != "unknown"

    def test_classify_frequency_fm(self):
        """FM broadcast should be classified correctly."""
        name, cat = SpectrumAnalyzer.classify_frequency(98.5e6)
        assert cat == "broadcast"
        assert "FM" in name

    def test_classify_frequency_unknown(self):
        """Frequency outside known bands should be 'unknown'."""
        name, cat = SpectrumAnalyzer.classify_frequency(50e6)
        assert cat == "unknown"

    def test_classify_frequency_detailed_wifi_channel(self):
        """Detailed classification should identify WiFi channel."""
        result = SpectrumAnalyzer.classify_frequency_detailed(2.437e9)
        assert result.wifi_channel == 6  # 2437 MHz = channel 6

    def test_summarize(self):
        """summarize should return correct overall stats."""
        sa = SpectrumAnalyzer()
        freqs, powers = self._make_spectrum()
        summary = sa.summarize(freqs, powers)
        assert summary.num_bins == 200
        assert summary.peak_power_dbm > -50.0
        assert summary.noise_floor_dbm < -70.0
        assert 0.0 <= summary.spectral_entropy <= 1.0

    def test_spectral_entropy_pure_tone(self):
        """Pure tone should have low spectral entropy."""
        sa = SpectrumAnalyzer()
        freqs = list(range(100))
        # One dominant bin, rest very low
        powers = [-120.0] * 100
        powers[50] = 0.0
        summary = sa.summarize([float(f) for f in freqs],
                               [float(p) for p in powers])
        assert summary.spectral_entropy < 0.3

    def test_spectral_entropy_noise(self):
        """Flat noise should have high spectral entropy."""
        sa = SpectrumAnalyzer()
        freqs = [float(i) for i in range(100)]
        powers = [-90.0] * 100
        summary = sa.summarize(freqs, powers)
        assert summary.spectral_entropy > 0.9

    def test_periodogram_sine(self):
        """Periodogram of a sine wave should have a peak at the frequency."""
        sa = SpectrumAnalyzer()
        fs = 1000.0  # 1 kHz sample rate
        f_signal = 100.0  # 100 Hz signal
        n = 256
        samples = [math.sin(2 * math.pi * f_signal * t / fs) for t in range(n)]
        freqs, powers = sa.periodogram(samples, fs)
        # Find peak frequency
        peak_idx = max(range(len(powers)), key=lambda i: powers[i])
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - f_signal) < (fs / n) * 2  # within 2 bins

    def test_band_table_not_empty(self):
        """BAND_TABLE should have entries."""
        assert len(BAND_TABLE) > 10

    def test_wifi_channels(self):
        """WiFi channel mappings should have expected entries."""
        assert 1 in WIFI_24_CHANNELS
        assert 6 in WIFI_24_CHANNELS
        assert 11 in WIFI_24_CHANNELS
        assert 36 in WIFI_5_CHANNELS
        assert 165 in WIFI_5_CHANNELS

    def test_spectral_peak_to_dict(self):
        """SpectralPeak.to_dict should be serializable."""
        peak = SpectralPeak(
            freq_hz=2.437e9, power_dbm=-45.0, prominence_db=50.0,
            bandwidth_hz=22e6, snr_db=50.0, band="WiFi 2.4 GHz",
            category="wifi",
        )
        d = peak.to_dict()
        assert d["freq_mhz"] == 2437.0
        assert d["category"] == "wifi"


# ===========================================================================
# CSIProcessor Tests
# ===========================================================================

class TestCSIProcessor:
    """Tests for WiFi CSI occupancy detection."""

    def _make_quiet_frame(self, n=64, base_amp=1.0, noise=0.01):
        """Generate a quiet (empty room) CSI frame."""
        import random
        random.seed(42)
        return [base_amp + random.gauss(0, noise) for _ in range(n)]

    def _make_active_frame(self, n=64, base_amp=1.0, disturbance=0.5):
        """Generate a disturbed (occupied) CSI frame."""
        import random
        random.seed(None)
        return [base_amp + random.gauss(0, disturbance) for _ in range(n)]

    def test_add_frame_basic(self):
        """add_frame should accept correctly sized frames."""
        proc = CSIProcessor(num_subcarriers=32)
        result = proc.add_frame([1.0] * 32, timestamp=1000.0)
        assert result is not None
        assert result.mean_amplitude == 1.0

    def test_add_frame_wrong_size(self):
        """add_frame should reject wrong-sized frames."""
        proc = CSIProcessor(num_subcarriers=32)
        result = proc.add_frame([1.0] * 16, timestamp=1000.0)
        assert result is None

    def test_baseline_computation(self):
        """Baseline should be ready after enough quiet frames."""
        proc = CSIProcessor(num_subcarriers=16, baseline_frames=10)
        import random
        random.seed(42)
        for i in range(10):
            frame = [1.0 + random.gauss(0, 0.01) for _ in range(16)]
            proc.add_frame(frame, timestamp=1000.0 + i)
        assert proc.is_baseline_ready()

    def test_occupancy_detection_empty_room(self):
        """Empty room (low variance) should not be flagged as occupied."""
        import random
        random.seed(42)
        proc = CSIProcessor(num_subcarriers=16, baseline_frames=10,
                            window_size=30, occupancy_threshold=2.0)
        # Build baseline with quiet frames
        for i in range(10):
            frame = [1.0 + random.gauss(0, 0.01) for _ in range(16)]
            proc.add_frame(frame, timestamp=1000.0 + i)
        # Continue with similar quiet frames
        for i in range(20):
            frame = [1.0 + random.gauss(0, 0.01) for _ in range(16)]
            proc.add_frame(frame, timestamp=1010.0 + i)

        result = proc.detect_occupancy()
        assert result is not None
        assert not result.occupied

    def test_occupancy_detection_occupied(self):
        """Disturbed CSI (human present) should trigger occupied."""
        import random
        random.seed(42)
        proc = CSIProcessor(num_subcarriers=16, baseline_frames=10,
                            window_size=30, occupancy_threshold=2.0)
        # Build baseline with quiet frames
        for i in range(10):
            frame = [1.0 + random.gauss(0, 0.01) for _ in range(16)]
            proc.add_frame(frame, timestamp=1000.0 + i)
        # Now inject highly disturbed frames
        for i in range(20):
            frame = [1.0 + random.gauss(0, 0.5) for _ in range(16)]
            proc.add_frame(frame, timestamp=1010.0 + i)

        result = proc.detect_occupancy()
        assert result is not None
        assert result.occupied
        assert result.variance_ratio > 2.0

    def test_occupancy_result_to_dict(self):
        """OccupancyResult.to_dict should be serializable."""
        result = OccupancyResult(
            occupied=True, confidence=0.85, variance_ratio=3.5,
            mean_variance=0.25, baseline_variance=0.01,
            active_subcarriers=12, total_subcarriers=64,
        )
        d = result.to_dict()
        assert d["occupied"] is True
        assert d["confidence"] == 0.85

    def test_band_activity(self):
        """get_band_activity should return per-band results."""
        import random
        random.seed(42)
        proc = CSIProcessor(num_subcarriers=16, baseline_frames=5,
                            num_bands=4, window_size=20)
        for i in range(5):
            proc.add_frame([1.0 + random.gauss(0, 0.01) for _ in range(16)],
                           timestamp=1000.0 + i)
        for i in range(15):
            proc.add_frame([1.0 + random.gauss(0, 0.3) for _ in range(16)],
                           timestamp=1005.0 + i)
        bands = proc.get_band_activity()
        assert len(bands) == 4
        assert all(isinstance(b, SubcarrierBand) for b in bands)

    def test_reset_baseline(self):
        """reset_baseline should clear and allow recomputation."""
        import random
        random.seed(42)
        proc = CSIProcessor(num_subcarriers=8, baseline_frames=5)
        for i in range(5):
            proc.add_frame([1.0] * 8, timestamp=1000.0 + i)
        assert proc.is_baseline_ready()
        proc.reset_baseline()
        assert not proc.is_baseline_ready()

    def test_get_status(self):
        """get_status should return configuration dict."""
        proc = CSIProcessor(num_subcarriers=32, window_size=60)
        status = proc.get_status()
        assert status["num_subcarriers"] == 32
        assert status["window_size"] == 60
        assert status["baseline_ready"] is False


# ===========================================================================
# Hampel Filter Tests
# ===========================================================================

class TestHampelFilter:
    """Tests for the Hampel outlier filter."""

    def test_no_outliers(self):
        """Clean signal should pass through unchanged."""
        values = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0]
        result = hampel_filter(values)
        assert len(result) == len(values)
        # All values should be unchanged (no outliers)
        for orig, filt in zip(values, result):
            assert abs(orig - filt) < 0.01

    def test_outlier_removed(self):
        """Spike outlier should be replaced with local median."""
        values = [1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]
        result = hampel_filter(values)
        # The outlier at index 3 should be replaced
        assert result[3] < 10.0

    def test_short_input(self):
        """Very short input (< 3) should be returned as-is."""
        assert hampel_filter([5.0]) == [5.0]
        assert hampel_filter([5.0, 6.0]) == [5.0, 6.0]


# ===========================================================================
# Import / Integration Tests
# ===========================================================================

class TestImports:
    """Verify all expected symbols are importable."""

    def test_all_exports(self):
        """All __all__ entries should be importable."""
        import tritium_lib.signals as signals
        for name in signals.__all__:
            assert hasattr(signals, name), f"{name} not found in signals module"

    def test_rssi_stats_to_dict(self):
        """RSSIStats.to_dict should produce correct keys."""
        stats = RSSIStats(
            count=10, mean=-65.0, std_dev=3.0, min_val=-70.0,
            max_val=-60.0, range_val=10.0, latest=-62.0, smoothed=-64.0,
        )
        d = stats.to_dict()
        assert d["count"] == 10
        assert d["mean"] == -65.0
        assert "smoothed" in d

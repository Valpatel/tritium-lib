# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SDR simulator, spectrum analyzer, and demo endpoints."""

import asyncio
import math
import time

import pytest

from tritium_lib.sdr.base import SDRDevice, SDRInfo, SweepResult, SweepPoint
from tritium_lib.sdr.simulator import (
    SimulatedSDR,
    SimulatedSignal,
    default_signal_environment,
)
from tritium_lib.sdr.analyzer import (
    SpectrumAnalyzer,
    DetectedSignal,
    FrequencyBand,
    ScanPreset,
    WaterfallRow,
    KNOWN_BANDS,
    SCAN_PRESETS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run an async coroutine synchronously for tests."""
    return asyncio.run(coro)


@pytest.fixture
def sdr():
    return SimulatedSDR(seed=42, noise_floor_dbm=-95.0)


@pytest.fixture
def analyzer(sdr):
    return SpectrumAnalyzer(sdr, waterfall_depth=50)


# ---------------------------------------------------------------------------
# SimulatedSignal tests
# ---------------------------------------------------------------------------

class TestSimulatedSignal:
    def test_defaults(self):
        sig = SimulatedSignal()
        assert sig.freq_hz == 0
        assert sig.bandwidth_hz == 200_000
        assert sig.power_dbm == -40.0
        assert sig.active is True

    def test_current_power_active(self):
        import random
        rng = random.Random(42)
        sig = SimulatedSignal(power_dbm=-30.0, active=True, fade_depth_db=0.0)
        power = sig.current_power(0.0, rng)
        # Should be near -30 dBm (with noise jitter)
        assert -35.0 < power < -25.0

    def test_current_power_inactive(self):
        import random
        rng = random.Random(42)
        sig = SimulatedSignal(power_dbm=-30.0, active=False)
        power = sig.current_power(0.0, rng)
        assert power == -200.0

    def test_current_power_intermittent(self):
        import random
        rng = random.Random(42)
        sig = SimulatedSignal(
            power_dbm=-30.0,
            intermittent=True,
            duty_cycle=0.5,
            fade_rate_hz=1.0,
        )
        # Over many time steps, some should be active and some inactive
        active_count = 0
        for i in range(100):
            t = i * 0.01
            p = sig.current_power(t, rng)
            if p > -150.0:
                active_count += 1
        # With 50% duty cycle, roughly half should be active
        assert 20 < active_count < 80

    def test_current_power_fading(self):
        import random
        rng = random.Random(1)
        sig = SimulatedSignal(
            power_dbm=-40.0,
            fade_depth_db=10.0,
            fade_rate_hz=1.0,
        )
        powers = [sig.current_power(t * 0.1, rng) for t in range(100)]
        # Power should vary over time
        assert max(powers) - min(powers) > 5.0

    def test_current_freq_no_drift(self):
        import random
        rng = random.Random(42)
        sig = SimulatedSignal(freq_hz=100_000_000, drift_hz=0)
        assert sig.current_freq(0.0, rng) == 100_000_000
        assert sig.current_freq(100.0, rng) == 100_000_000

    def test_current_freq_with_drift(self):
        import random
        rng = random.Random(42)
        sig = SimulatedSignal(freq_hz=100_000_000, drift_hz=1000)
        freqs = set()
        for t in range(100):
            freqs.add(sig.current_freq(t, rng))
        # Should have multiple different frequencies
        assert len(freqs) > 1


# ---------------------------------------------------------------------------
# Signal environment tests
# ---------------------------------------------------------------------------

class TestSignalEnvironment:
    def test_default_environment_not_empty(self):
        signals = default_signal_environment()
        assert len(signals) > 0

    def test_default_environment_has_all_categories(self):
        signals = default_signal_environment()
        categories = {s.category for s in signals}
        assert "fm" in categories
        assert "wifi" in categories
        assert "ble" in categories
        assert "lora" in categories
        assert "ism" in categories
        assert "adsb" in categories
        assert "cellular" in categories

    def test_default_environment_has_valid_frequencies(self):
        signals = default_signal_environment()
        for sig in signals:
            assert sig.freq_hz > 0, f"Signal {sig.name} has invalid frequency"
            assert sig.bandwidth_hz > 0, f"Signal {sig.name} has invalid bandwidth"


# ---------------------------------------------------------------------------
# SimulatedSDR tests
# ---------------------------------------------------------------------------

class TestSimulatedSDR:
    def test_detect(self, sdr):
        info = run(sdr.detect())
        assert info.detected is True
        assert info.name == "Tritium Simulated SDR"
        assert info.serial == "SIM-0001"
        assert info.freq_min_hz == 1_000_000
        assert info.freq_max_hz == 6_000_000_000
        assert sdr.is_available is True

    def test_sweep_fm_band(self, sdr):
        async def go():
            await sdr.detect()
            return await sdr.sweep(88_000_000, 108_000_000, bin_width_hz=100_000)
        result = run(go())
        assert len(result.points) == 200  # (108-88) MHz / 100 kHz = 200 bins
        assert result.freq_start_hz == 88_000_000
        assert result.freq_end_hz == 108_000_000
        assert result.bin_width_hz == 100_000
        assert result.sweep_time_ms >= 0

    def test_sweep_has_signals_above_noise(self, sdr):
        async def go():
            await sdr.detect()
            return await sdr.sweep(88_000_000, 108_000_000, bin_width_hz=100_000)
        result = run(go())
        powers = [p.power_dbm for p in result.points]
        # FM band should have signals above noise floor
        assert max(powers) > -80.0

    def test_sweep_wifi_band(self, sdr):
        async def go():
            await sdr.detect()
            return await sdr.sweep(2_400_000_000, 2_500_000_000, bin_width_hz=1_000_000)
        result = run(go())
        assert len(result.points) == 100

    def test_sweep_empty_band(self):
        """Sweep a band with no signals — should be noise floor."""
        empty_sdr = SimulatedSDR(signals=[], seed=42, noise_floor_dbm=-95.0)
        async def go():
            await empty_sdr.detect()
            return await empty_sdr.sweep(500_000_000, 510_000_000, bin_width_hz=100_000)
        result = run(go())
        powers = [p.power_dbm for p in result.points]
        # All should be near noise floor
        assert all(-105 < p < -85 for p in powers)

    def test_sweep_increments_count(self, sdr):
        async def go():
            await sdr.detect()
            assert sdr.sweep_count == 0
            await sdr.sweep(88_000_000, 108_000_000)
            assert sdr.sweep_count == 1
            await sdr.sweep(88_000_000, 108_000_000)
            assert sdr.sweep_count == 2
        run(go())

    def test_tune(self, sdr):
        async def go():
            await sdr.detect()
            await sdr.tune(100_000_000, sample_rate=2_000_000)
            assert sdr.tuned_frequency == 100_000_000
            assert sdr._running is True
        run(go())

    def test_stop(self, sdr):
        async def go():
            await sdr.detect()
            await sdr.tune(100_000_000)
            await sdr.stop()
            assert sdr.tuned_frequency == 0
            assert sdr._running is False
        run(go())

    def test_add_signal(self, sdr):
        initial_count = len(sdr.signals)
        sdr.add_signal(SimulatedSignal(
            name="TestSignal",
            freq_hz=500_000_000,
            power_dbm=-20.0,
        ))
        assert len(sdr.signals) == initial_count + 1

    def test_remove_signal(self, sdr):
        sdr.add_signal(SimulatedSignal(name="ToRemove", freq_hz=123_456_789))
        assert sdr.remove_signal("ToRemove") is True
        assert sdr.remove_signal("NonExistent") is False

    def test_get_signals_in_range(self, sdr):
        fm_signals = sdr.get_signals_in_range(88_000_000, 108_000_000)
        assert len(fm_signals) > 0
        for sig in fm_signals:
            assert 88_000_000 <= sig.freq_hz <= 108_000_000

    def test_custom_noise_floor(self):
        custom_sdr = SimulatedSDR(signals=[], seed=42, noise_floor_dbm=-80.0)
        async def go():
            await custom_sdr.detect()
            return await custom_sdr.sweep(100_000_000, 110_000_000, bin_width_hz=100_000)
        result = run(go())
        avg = sum(p.power_dbm for p in result.points) / len(result.points)
        # Average should be near the custom noise floor
        assert -90 < avg < -70

    def test_deterministic_with_seed(self):
        sdr1 = SimulatedSDR(seed=42)
        sdr2 = SimulatedSDR(seed=42)
        async def go():
            await sdr1.detect()
            await sdr2.detect()
            r1 = await sdr1.sweep(88_000_000, 98_000_000, bin_width_hz=1_000_000)
            r2 = await sdr2.sweep(88_000_000, 98_000_000, bin_width_hz=1_000_000)
            return r1, r2
        r1, r2 = run(go())
        # Same seed should produce same number of bins
        assert len(r1.points) == len(r2.points)

    def test_single_bin_sweep(self, sdr):
        async def go():
            await sdr.detect()
            return await sdr.sweep(100_000_000, 100_500_000, bin_width_hz=500_000)
        result = run(go())
        assert len(result.points) == 1

    def test_info_before_detect(self, sdr):
        assert sdr.info is None
        assert sdr.is_available is False


# ---------------------------------------------------------------------------
# SpectrumAnalyzer tests
# ---------------------------------------------------------------------------

class TestSpectrumAnalyzer:
    def test_initialize(self, analyzer):
        run(analyzer.initialize())
        assert analyzer.is_initialized is True
        assert analyzer.device.is_available is True

    def test_scan(self, analyzer):
        async def go():
            await analyzer.initialize()
            return await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=100_000)
        result = run(go())
        assert len(result.points) == 200
        assert analyzer._sweep_count == 1

    def test_detect_signals_fm_band(self, analyzer):
        async def go():
            await analyzer.initialize()
            result = await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=100_000)
            return analyzer.detect_signals(result, threshold_dbm=-60.0)
        signals = run(go())
        # Should detect some FM stations
        assert len(signals) > 0
        for sig in signals:
            assert sig.power_dbm > -60.0
            assert sig.freq_hz >= 88_000_000
            assert sig.freq_hz <= 108_000_000

    def test_detect_signals_classification(self, analyzer):
        async def go():
            await analyzer.initialize()
            result = await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=100_000)
            return analyzer.detect_signals(result, threshold_dbm=-60.0)
        signals = run(go())
        if signals:
            # FM band signals should be classified as broadcast
            for sig in signals:
                assert sig.category in ("broadcast", "unknown")

    def test_detect_signals_snr(self, analyzer):
        async def go():
            await analyzer.initialize()
            result = await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=100_000)
            return analyzer.detect_signals(result, threshold_dbm=-80.0, min_snr_db=6.0)
        signals = run(go())
        for sig in signals:
            assert sig.snr_db >= 0  # SNR should be positive for detected signals

    def test_detect_signals_no_sweep(self, analyzer):
        """detect_signals with no prior sweep returns empty."""
        run(analyzer.initialize())
        signals = analyzer.detect_signals(threshold_dbm=-60.0)
        assert signals == []

    def test_waterfall_empty(self, analyzer):
        run(analyzer.initialize())
        wf = analyzer.get_waterfall()
        assert wf["num_rows"] == 0
        assert wf["rows"] == []

    def test_waterfall_accumulates(self, analyzer):
        async def go():
            await analyzer.initialize()
            for _ in range(5):
                await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=1_000_000)
        run(go())
        wf = analyzer.get_waterfall()
        assert wf["num_rows"] == 5
        assert len(wf["rows"]) == 5
        assert wf["num_bins"] == 20

    def test_waterfall_max_rows(self, analyzer):
        async def go():
            await analyzer.initialize()
            for _ in range(5):
                await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=1_000_000)
        run(go())
        wf = analyzer.get_waterfall(max_rows=3)
        assert wf["num_rows"] == 3

    def test_waterfall_depth_limit(self, sdr):
        """Waterfall should not exceed configured depth."""
        small_analyzer = SpectrumAnalyzer(sdr, waterfall_depth=3)
        async def go():
            await small_analyzer.initialize()
            for _ in range(10):
                await small_analyzer.scan(88_000_000, 98_000_000, bin_width_hz=1_000_000)
        run(go())
        wf = small_analyzer.get_waterfall()
        assert wf["num_rows"] == 3

    def test_status(self, analyzer):
        run(analyzer.initialize())
        status = analyzer.get_status()
        assert status["initialized"] is True
        assert status["sweep_count"] == 0
        assert status["waterfall_depth"] == 0
        assert status["waterfall_max"] == 50
        assert "device" in status

    def test_status_after_sweeps(self, analyzer):
        async def go():
            await analyzer.initialize()
            await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=1_000_000)
            result = await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=1_000_000)
            analyzer.detect_signals(result)
        run(go())
        status = analyzer.get_status()
        assert status["sweep_count"] == 2
        assert status["waterfall_depth"] == 2

    def test_scan_presets(self):
        presets = SpectrumAnalyzer.get_scan_presets()
        assert len(presets) > 0
        for p in presets:
            assert "name" in p
            assert "start_hz" in p
            assert "end_hz" in p
            assert "bin_width_hz" in p
            assert p["end_hz"] > p["start_hz"]

    def test_known_bands(self):
        bands = SpectrumAnalyzer.get_known_bands()
        assert len(bands) > 0
        for b in bands:
            assert "name" in b
            assert "start_hz" in b
            assert "end_hz" in b
            assert "category" in b
            assert b["end_hz"] > b["start_hz"]

    def test_classify_fm_frequency(self, analyzer):
        cat, name = analyzer._classify_frequency(93_900_000)
        assert cat == "broadcast"
        assert "FM" in name

    def test_classify_wifi_frequency(self, analyzer):
        cat, name = analyzer._classify_frequency(2_437_000_000)
        assert cat in ("wifi", "ble")

    def test_classify_unknown_frequency(self, analyzer):
        cat, name = analyzer._classify_frequency(50_000_000)
        assert cat == "unknown"

    def test_multiple_band_scans(self, analyzer):
        """Analyzer should handle switching between different bands."""
        async def go():
            await analyzer.initialize()
            r1 = await analyzer.scan(88_000_000, 108_000_000, bin_width_hz=500_000)
            r2 = await analyzer.scan(2_400_000_000, 2_500_000_000, bin_width_hz=1_000_000)
            return r1, r2
        r1, r2 = run(go())
        assert r1.freq_start_hz == 88_000_000
        assert r2.freq_start_hz == 2_400_000_000
        # Waterfall should have both
        wf = analyzer.get_waterfall()
        assert wf["num_rows"] == 2


# ---------------------------------------------------------------------------
# DetectedSignal tests
# ---------------------------------------------------------------------------

class TestDetectedSignal:
    def test_to_dict(self):
        sig = DetectedSignal(
            freq_hz=93_900_000,
            power_dbm=-25.3,
            bandwidth_hz=200_000,
            snr_db=70.0,
            category="broadcast",
            band_name="FM Broadcast",
            timestamp=1234567890.0,
            persistent=True,
        )
        d = sig.to_dict()
        assert d["freq_hz"] == 93_900_000
        assert d["freq_mhz"] == 93.9
        assert d["power_dbm"] == -25.3
        assert d["bandwidth_khz"] == 200.0
        assert d["snr_db"] == 70.0
        assert d["category"] == "broadcast"
        assert d["band_name"] == "FM Broadcast"
        assert d["persistent"] is True

    def test_defaults(self):
        sig = DetectedSignal(freq_hz=100_000_000, power_dbm=-50.0)
        assert sig.category == "unknown"
        assert sig.band_name == ""
        assert sig.persistent is False


# ---------------------------------------------------------------------------
# WaterfallRow tests
# ---------------------------------------------------------------------------

class TestWaterfallRow:
    def test_creation(self):
        row = WaterfallRow(
            timestamp=time.time(),
            powers=[-90.0, -80.0, -50.0, -80.0, -90.0],
            freq_start_hz=88_000_000,
            freq_end_hz=108_000_000,
        )
        assert len(row.powers) == 5
        assert row.freq_start_hz == 88_000_000


# ---------------------------------------------------------------------------
# FrequencyBand and ScanPreset tests
# ---------------------------------------------------------------------------

class TestFrequencyBand:
    def test_creation(self):
        band = FrequencyBand(
            name="Test Band",
            start_hz=100_000_000,
            end_hz=200_000_000,
            category="test",
        )
        assert band.name == "Test Band"
        assert band.end_hz > band.start_hz

    def test_known_bands_valid(self):
        for band in KNOWN_BANDS:
            assert band.end_hz > band.start_hz, f"{band.name}: end < start"
            assert band.category, f"{band.name}: empty category"
            assert band.start_hz > 0, f"{band.name}: invalid start"


class TestScanPreset:
    def test_creation(self):
        preset = ScanPreset(
            name="Test",
            start_hz=88_000_000,
            end_hz=108_000_000,
            bin_width_hz=100_000,
        )
        assert preset.end_hz > preset.start_hz

    def test_all_presets_valid(self):
        for preset in SCAN_PRESETS:
            assert preset.end_hz > preset.start_hz, f"{preset.name}: end < start"
            assert preset.bin_width_hz > 0, f"{preset.name}: invalid bin width"
            num_bins = (preset.end_hz - preset.start_hz) // preset.bin_width_hz
            assert num_bins > 0, f"{preset.name}: would produce 0 bins"
            assert num_bins < 100_000, f"{preset.name}: too many bins ({num_bins})"


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_simulator(self):
        from tritium_lib.sdr.simulator import SimulatedSDR, SimulatedSignal
        assert SimulatedSDR is not None
        assert SimulatedSignal is not None

    def test_import_analyzer(self):
        from tritium_lib.sdr.analyzer import SpectrumAnalyzer, DetectedSignal
        assert SpectrumAnalyzer is not None
        assert DetectedSignal is not None

    def test_import_from_package(self):
        from tritium_lib.sdr import (
            SimulatedSDR,
            SimulatedSignal,
            SpectrumAnalyzer,
            DetectedSignal,
            default_signal_environment,
            KNOWN_BANDS,
            SCAN_PRESETS,
        )
        assert all(x is not None for x in [
            SimulatedSDR, SimulatedSignal, SpectrumAnalyzer,
            DetectedSignal, default_signal_environment,
            KNOWN_BANDS, SCAN_PRESETS,
        ])


# ---------------------------------------------------------------------------
# Demo endpoint tests
# ---------------------------------------------------------------------------

class TestDemoEndpoints:
    """Test the FastAPI demo app endpoints."""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")
        from tritium_lib.sdr.demos.sdr_demo import app
        return TestClient(app)

    def test_get_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "initialized" in data
        assert "device" in data
        assert "sweep_count" in data

    def test_get_presets(self, client):
        resp = client.get("/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "presets" in data
        assert len(data["presets"]) > 0

    def test_get_bands(self, client):
        resp = client.get("/bands")
        assert resp.status_code == 200
        data = resp.json()
        assert "bands" in data
        assert len(data["bands"]) > 0

    def test_post_sweep(self, client):
        resp = client.post("/sweep", json={
            "start_hz": 88_000_000,
            "end_hz": 108_000_000,
            "bin_width_hz": 500_000,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "sweep" in data
        assert "signals" in data
        assert data["sweep"]["num_points"] == 40

    def test_post_sweep_custom_range(self, client):
        resp = client.post("/sweep", json={
            "start_hz": 2_400_000_000,
            "end_hz": 2_500_000_000,
            "bin_width_hz": 1_000_000,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["freq_start_mhz"] == 2400.0
        assert data["freq_end_mhz"] == 2500.0

    def test_post_sweep_invalid_range(self, client):
        resp = client.post("/sweep", json={
            "start_hz": 108_000_000,
            "end_hz": 88_000_000,
            "bin_width_hz": 100_000,
        })
        assert resp.status_code == 400

    def test_post_sweep_invalid_bin_width(self, client):
        resp = client.post("/sweep", json={
            "start_hz": 88_000_000,
            "end_hz": 108_000_000,
            "bin_width_hz": 100,
        })
        assert resp.status_code == 400

    def test_get_spectrum_after_sweep(self, client):
        # Trigger a sweep first
        client.post("/sweep", json={
            "start_hz": 88_000_000,
            "end_hz": 108_000_000,
            "bin_width_hz": 500_000,
        })
        resp = client.get("/spectrum")
        assert resp.status_code == 200
        data = resp.json()
        assert "sweep" in data
        assert data["sweep"]["num_points"] == 40

    def test_get_signals_after_sweep(self, client):
        client.post("/sweep", json={
            "start_hz": 88_000_000,
            "end_hz": 108_000_000,
            "bin_width_hz": 100_000,
        })
        resp = client.get("/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert "signals" in data
        assert "count" in data
        assert "categories" in data

    def test_get_waterfall_after_sweeps(self, client):
        # Run multiple sweeps
        for _ in range(3):
            client.post("/sweep", json={
                "start_hz": 88_000_000,
                "end_hz": 108_000_000,
                "bin_width_hz": 1_000_000,
            })
        resp = client.get("/waterfall?max_rows=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_rows"] >= 3
        assert len(data["rows"]) >= 3

    def test_dashboard_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "TRITIUM SDR SPECTRUM ANALYZER" in resp.text
        assert "spectrum-canvas" in resp.text
        assert "waterfall-canvas" in resp.text

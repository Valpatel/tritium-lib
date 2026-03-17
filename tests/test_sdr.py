# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the SDR device abstraction."""

import pytest
from tritium_lib.sdr.base import SDRDevice, SDRInfo, SweepResult, SweepPoint


class TestSDRInfo:
    def test_empty(self):
        info = SDRInfo()
        assert info.detected is False
        assert info.name == ""

    def test_to_dict_omits_empty(self):
        info = SDRInfo(detected=True, name="HackRF One", serial="abc123")
        d = info.to_dict()
        assert d["detected"] is True
        assert d["name"] == "HackRF One"
        assert "error" not in d

    def test_full_info(self):
        info = SDRInfo(
            detected=True, name="HackRF One", serial="abc",
            firmware="2024.02.1", freq_min_hz=1000000, freq_max_hz=6000000000,
            has_tx=True, has_bias_tee=True,
        )
        d = info.to_dict()
        assert d["freq_max_hz"] == 6000000000
        assert d["has_tx"] is True


class TestSweepPoint:
    def test_defaults(self):
        p = SweepPoint()
        assert p.freq_hz == 0
        assert p.power_dbm == -100.0

    def test_values(self):
        p = SweepPoint(freq_hz=100000000, power_dbm=-45.3, timestamp=1234567890.0)
        assert p.freq_hz == 100000000
        assert p.power_dbm == -45.3


class TestSweepResult:
    def test_empty(self):
        r = SweepResult()
        d = r.to_dict()
        assert d["num_points"] == 0
        assert d["peak_power"] == -100

    def test_with_points(self):
        points = [
            SweepPoint(freq_hz=88000000, power_dbm=-45.0),
            SweepPoint(freq_hz=89000000, power_dbm=-30.0),
            SweepPoint(freq_hz=90000000, power_dbm=-60.0),
        ]
        r = SweepResult(points=points, freq_start_hz=88000000, freq_end_hz=91000000)
        d = r.to_dict()
        assert d["num_points"] == 3
        assert d["peak_freq"] == 89000000
        assert d["peak_power"] == -30.0
        assert d["avg_power"] == -45.0

    def test_get_peaks(self):
        points = [
            SweepPoint(freq_hz=88000000, power_dbm=-45.0),
            SweepPoint(freq_hz=89000000, power_dbm=-20.0),
            SweepPoint(freq_hz=90000000, power_dbm=-60.0),
        ]
        r = SweepResult(points=points)
        peaks = r.get_peaks(threshold_dbm=-30.0)
        assert len(peaks) == 1
        assert peaks[0].freq_hz == 89000000

    def test_get_peaks_none(self):
        points = [SweepPoint(freq_hz=88000000, power_dbm=-60.0)]
        r = SweepResult(points=points)
        assert len(r.get_peaks(threshold_dbm=-30.0)) == 0


class TestSDRDeviceAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SDRDevice()

    def test_find_devices(self):
        devices = SDRDevice.find_devices()
        assert isinstance(devices, list)


class TestImports:
    def test_import_all(self):
        from tritium_lib.sdr import SDRDevice, SDRInfo, SweepResult, SweepPoint
        assert SDRDevice is not None
        assert SDRInfo is not None

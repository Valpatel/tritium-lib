# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SDR integration models."""

import json

from tritium_lib.models.sdr import (
    ADSBTrack,
    AISTrack,
    ISMDevice,
    Modulation,
    RFSignal,
    SpectrumScan,
)


class TestRFSignal:
    def test_creation_minimal(self):
        s = RFSignal(frequency_mhz=433.92)
        assert s.frequency_mhz == 433.92
        assert s.bandwidth_khz == 0.0
        assert s.power_dbm == 0.0
        assert s.modulation == Modulation.UNKNOWN

    def test_creation_full(self):
        s = RFSignal(
            frequency_mhz=915.0,
            bandwidth_khz=125.0,
            power_dbm=-45.0,
            modulation=Modulation.LORA,
            source_id="sdr_01",
        )
        assert s.bandwidth_khz == 125.0
        assert s.power_dbm == -45.0
        assert s.modulation == Modulation.LORA

    def test_serialization_roundtrip(self):
        s = RFSignal(frequency_mhz=162.55, modulation=Modulation.FM)
        data = s.model_dump()
        s2 = RFSignal(**data)
        assert s2.frequency_mhz == s.frequency_mhz
        assert s2.modulation == Modulation.FM

    def test_json_serialization(self):
        s = RFSignal(frequency_mhz=1090.0, power_dbm=-30.0)
        j = s.model_dump_json()
        parsed = json.loads(j)
        assert parsed["frequency_mhz"] == 1090.0

    def test_all_modulations(self):
        for m in Modulation:
            s = RFSignal(frequency_mhz=100.0, modulation=m)
            assert s.modulation == m


class TestSpectrumScan:
    def test_creation_minimal(self):
        s = SpectrumScan(center_freq_mhz=433.0, span_mhz=2.0)
        assert s.center_freq_mhz == 433.0
        assert s.span_mhz == 2.0
        assert s.bins == []

    def test_with_bins(self):
        bins = [-90.0, -85.0, -60.0, -85.0, -92.0]
        s = SpectrumScan(
            center_freq_mhz=915.0,
            span_mhz=1.0,
            bins=bins,
            source_id="sdr_02",
        )
        assert s.bin_count == 5
        assert s.bin_width_khz == 200.0
        assert s.start_freq_mhz == 914.5
        assert s.end_freq_mhz == 915.5
        assert s.peak_power_dbm() == -60.0

    def test_peak_frequency(self):
        bins = [-90.0, -80.0, -50.0, -80.0, -90.0]
        s = SpectrumScan(center_freq_mhz=100.0, span_mhz=10.0, bins=bins)
        # bin_width = 10000/5 = 2000 kHz = 2 MHz
        # start = 95 MHz, peak at idx 2 => 95 + 2*2 = 99 MHz
        assert s.peak_frequency_mhz() == 99.0

    def test_empty_bins_properties(self):
        s = SpectrumScan(center_freq_mhz=100.0, span_mhz=10.0)
        assert s.bin_count == 0
        assert s.bin_width_khz == 0.0
        assert s.peak_power_dbm() == -999.0
        assert s.peak_frequency_mhz() == 100.0

    def test_serialization_roundtrip(self):
        s = SpectrumScan(
            center_freq_mhz=868.0,
            span_mhz=5.0,
            bins=[-70.0, -65.0, -70.0],
        )
        data = s.model_dump()
        s2 = SpectrumScan(**data)
        assert s2.bins == s.bins
        assert s2.center_freq_mhz == s.center_freq_mhz


class TestISMDevice:
    def test_creation_minimal(self):
        d = ISMDevice()
        assert d.device_type == ""
        assert d.frequency_mhz == 433.92

    def test_creation_weather_station(self):
        d = ISMDevice(
            device_type="Acurite-5n1",
            protocol="acurite",
            frequency_mhz=433.92,
            device_id="12345",
            temperature=22.5,
            humidity=65.0,
            battery_pct=80.0,
            payload={"wind_speed_kmh": 15.0, "rain_mm": 2.3},
        )
        assert d.device_type == "Acurite-5n1"
        assert d.temperature == 22.5
        assert d.humidity == 65.0
        assert d.battery_pct == 80.0
        assert d.payload["wind_speed_kmh"] == 15.0

    def test_to_target_dict(self):
        d = ISMDevice(
            device_type="Oregon-THGR122N",
            device_id="42",
            temperature=18.0,
        )
        td = d.to_target_dict()
        assert td["target_id"] == "ism_Oregon-THGR122N_42"
        assert td["source"] == "sdr_ism"
        assert td["metadata"]["temperature"] == 18.0

    def test_to_target_dict_no_id(self):
        d = ISMDevice(device_type="generic_sensor")
        td = d.to_target_dict()
        assert td["target_id"] == "ism_generic_sensor"

    def test_serialization_roundtrip(self):
        d = ISMDevice(
            device_type="TPMS",
            protocol="tpms",
            frequency_mhz=315.0,
            payload={"pressure_kpa": 220.0, "tire_id": "FL"},
        )
        data = d.model_dump()
        d2 = ISMDevice(**data)
        assert d2.device_type == d.device_type
        assert d2.payload == d.payload


class TestADSBTrack:
    def test_creation(self):
        t = ADSBTrack(
            icao_hex="A1B2C3",
            callsign="UAL123",
            altitude_ft=35000.0,
            speed_kts=450.0,
            heading_deg=270.0,
            lat=40.6413,
            lng=-73.7781,
            squawk="1200",
        )
        assert t.icao_hex == "A1B2C3"
        assert t.callsign == "UAL123"
        assert t.altitude_ft == 35000.0
        assert t.speed_kts == 450.0

    def test_compute_target_id(self):
        t = ADSBTrack(icao_hex="A1B2C3")
        assert t.compute_target_id() == "adsb_a1b2c3"

    def test_compute_target_id_lowercase(self):
        t = ADSBTrack(icao_hex="abcdef")
        assert t.compute_target_id() == "adsb_abcdef"

    def test_serialization_roundtrip(self):
        t = ADSBTrack(
            icao_hex="FFAA00",
            callsign="DAL42",
            altitude_ft=12000.0,
            lat=33.45,
            lng=-112.07,
        )
        data = t.model_dump()
        t2 = ADSBTrack(**data)
        assert t2.icao_hex == t.icao_hex
        assert t2.callsign == t.callsign

    def test_defaults(self):
        t = ADSBTrack(icao_hex="000001")
        assert t.callsign == ""
        assert t.altitude_ft == 0.0
        assert t.squawk == ""


class TestAISTrack:
    def test_creation(self):
        t = AISTrack(
            mmsi="211000001",
            vessel_name="NORDSEE EXPRESS",
            vessel_type="cargo",
            lat=53.55,
            lng=9.99,
            course_deg=180.0,
            speed_kts=12.5,
            destination="HAMBURG",
        )
        assert t.mmsi == "211000001"
        assert t.vessel_name == "NORDSEE EXPRESS"
        assert t.speed_kts == 12.5
        assert t.destination == "HAMBURG"

    def test_compute_target_id(self):
        t = AISTrack(mmsi="366999999")
        assert t.compute_target_id() == "ais_366999999"

    def test_serialization_roundtrip(self):
        t = AISTrack(
            mmsi="123456789",
            vessel_name="TEST VESSEL",
            lat=51.5,
            lng=-0.12,
        )
        data = t.model_dump()
        t2 = AISTrack(**data)
        assert t2.mmsi == t.mmsi
        assert t2.vessel_name == t.vessel_name

    def test_defaults(self):
        t = AISTrack(mmsi="000000001")
        assert t.vessel_name == ""
        assert t.vessel_type == ""
        assert t.destination == ""
        assert t.lat == 0.0
        assert t.lng == 0.0

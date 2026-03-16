# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for radar integration models."""

import json

from tritium_lib.models.radar import (
    RadarClassification,
    RadarConfig,
    RadarMode,
    RadarScan,
    RadarTrack,
)


class TestRadarTrack:
    def test_creation_minimal(self):
        t = RadarTrack(track_id="T001", range_m=500.0, azimuth_deg=45.0)
        assert t.track_id == "T001"
        assert t.range_m == 500.0
        assert t.azimuth_deg == 45.0
        assert t.elevation_deg == 0.0
        assert t.velocity_mps == 0.0
        assert t.rcs_dbsm == 0.0
        assert t.classification == RadarClassification.UNKNOWN
        assert t.confidence == 1.0

    def test_creation_full(self):
        t = RadarTrack(
            track_id="T042",
            range_m=1200.0,
            azimuth_deg=270.5,
            elevation_deg=5.3,
            velocity_mps=-15.0,
            rcs_dbsm=10.0,
            classification=RadarClassification.VEHICLE,
            confidence=0.85,
            source_id="radar_01",
        )
        assert t.velocity_mps == -15.0
        assert t.classification == RadarClassification.VEHICLE
        assert t.source_id == "radar_01"

    def test_serialization_roundtrip(self):
        t = RadarTrack(
            track_id="T099",
            range_m=3000.0,
            azimuth_deg=180.0,
            classification=RadarClassification.AIRCRAFT,
        )
        data = t.model_dump()
        t2 = RadarTrack(**data)
        assert t2.track_id == t.track_id
        assert t2.range_m == t.range_m
        assert t2.classification == RadarClassification.AIRCRAFT

    def test_json_serialization(self):
        t = RadarTrack(track_id="T001", range_m=100.0, azimuth_deg=0.0)
        j = t.model_dump_json()
        parsed = json.loads(j)
        assert parsed["track_id"] == "T001"
        assert parsed["range_m"] == 100.0

    def test_to_target_dict(self):
        t = RadarTrack(
            track_id="T005",
            range_m=800.0,
            azimuth_deg=90.0,
            classification=RadarClassification.PERSON,
            confidence=0.7,
            source_id="radar_02",
        )
        d = t.to_target_dict()
        assert d["target_id"] == "radar_radar_02_T005"
        assert d["source"] == "radar"
        assert d["classification"] == "person"
        assert d["confidence"] == 0.7

    def test_all_classifications(self):
        for c in RadarClassification:
            t = RadarTrack(track_id="X", range_m=1.0, azimuth_deg=0.0, classification=c)
            assert t.classification == c


class TestRadarScan:
    def test_empty_scan(self):
        s = RadarScan(scan_id="S001")
        assert s.scan_id == "S001"
        assert s.tracks == []
        assert s.mode == RadarMode.SURVEILLANCE
        assert s.rotation_rate_rpm == 0.0

    def test_scan_with_tracks(self):
        tracks = [
            RadarTrack(track_id="T1", range_m=100.0, azimuth_deg=10.0),
            RadarTrack(track_id="T2", range_m=200.0, azimuth_deg=20.0),
            RadarTrack(track_id="T3", range_m=300.0, azimuth_deg=30.0),
        ]
        s = RadarScan(
            scan_id="S002",
            tracks=tracks,
            mode=RadarMode.TRACKING,
            rotation_rate_rpm=12.5,
        )
        assert len(s.tracks) == 3
        assert s.mode == RadarMode.TRACKING
        assert s.rotation_rate_rpm == 12.5

    def test_serialization_roundtrip(self):
        s = RadarScan(
            scan_id="S003",
            tracks=[RadarTrack(track_id="T1", range_m=500.0, azimuth_deg=45.0)],
            mode=RadarMode.WEATHER,
        )
        data = s.model_dump()
        s2 = RadarScan(**data)
        assert s2.scan_id == s.scan_id
        assert len(s2.tracks) == 1
        assert s2.mode == RadarMode.WEATHER

    def test_all_modes(self):
        for m in RadarMode:
            s = RadarScan(scan_id="X", mode=m)
            assert s.mode == m


class TestRadarConfig:
    def test_creation(self):
        c = RadarConfig(
            radar_id="radar_01",
            name="North Tower Radar",
            frequency_ghz=9.4,
            max_range_m=50000.0,
            beam_width_deg=1.5,
            latitude=33.45,
            longitude=-112.07,
            altitude_m=450.0,
        )
        assert c.radar_id == "radar_01"
        assert c.name == "North Tower Radar"
        assert c.frequency_ghz == 9.4
        assert c.max_range_m == 50000.0
        assert c.beam_width_deg == 1.5
        assert c.latitude == 33.45
        assert c.enabled is True

    def test_defaults(self):
        c = RadarConfig(radar_id="r1")
        assert c.name == ""
        assert c.frequency_ghz == 0.0
        assert c.max_range_m == 0.0
        assert c.enabled is True

    def test_serialization_roundtrip(self):
        c = RadarConfig(
            radar_id="r2",
            frequency_ghz=3.0,
            max_range_m=100000.0,
        )
        data = c.model_dump()
        c2 = RadarConfig(**data)
        assert c2.radar_id == c.radar_id
        assert c2.frequency_ghz == c.frequency_ghz

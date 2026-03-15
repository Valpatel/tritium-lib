# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for forensic reconstruction and incident report models."""

import time

import pytest

from tritium_lib.models.forensics import (
    EvidenceItem,
    ForensicReconstruction,
    GeoBounds,
    IncidentClassification,
    IncidentFinding,
    IncidentRecommendation,
    IncidentReport,
    ReconstructionStatus,
    SensorCoverage,
    TargetTimeline,
    TimeRange,
)


class TestTimeRange:
    def test_duration(self):
        tr = TimeRange(start=1000.0, end=1060.0)
        assert tr.duration_s == 60.0

    def test_contains(self):
        tr = TimeRange(start=1000.0, end=2000.0)
        assert tr.contains(1500.0)
        assert tr.contains(1000.0)
        assert tr.contains(2000.0)
        assert not tr.contains(999.0)
        assert not tr.contains(2001.0)


class TestGeoBounds:
    def test_contains(self):
        bounds = GeoBounds(north=40.0, south=39.0, east=-74.0, west=-75.0)
        assert bounds.contains(39.5, -74.5)
        assert not bounds.contains(41.0, -74.5)
        assert not bounds.contains(39.5, -73.0)

    def test_edge_case(self):
        bounds = GeoBounds(north=40.0, south=39.0, east=-74.0, west=-75.0)
        assert bounds.contains(40.0, -74.0)  # corner
        assert bounds.contains(39.0, -75.0)  # corner


class TestEvidenceItem:
    def test_creation(self):
        e = EvidenceItem(
            evidence_id="ev_001",
            timestamp=time.time(),
            sensor_id="node_01",
            sensor_type="ble",
            target_id="ble_aa:bb:cc",
            observation_type="sighting",
            confidence=0.9,
        )
        assert e.evidence_id == "ev_001"
        assert e.sensor_type == "ble"


class TestTargetTimeline:
    def test_duration(self):
        tt = TargetTimeline(
            target_id="ble_aa",
            first_seen=1000.0,
            last_seen=1300.0,
        )
        assert tt.duration_s == 300.0

    def test_zero_duration(self):
        tt = TargetTimeline(target_id="ble_bb")
        assert tt.duration_s == 0.0


class TestForensicReconstruction:
    def test_mark_complete(self):
        r = ForensicReconstruction(
            reconstruction_id="recon_001",
            events=[{"type": "sighting"}, {"type": "alert"}],
            targets=[TargetTimeline(target_id="t1"), TargetTimeline(target_id="t2")],
        )
        r.mark_complete()
        assert r.status == ReconstructionStatus.COMPLETE
        assert r.total_events == 2
        assert r.total_targets == 2
        assert r.completed_at is not None

    def test_mark_failed(self):
        r = ForensicReconstruction(reconstruction_id="recon_002")
        r.mark_failed("timeout")
        assert r.status == ReconstructionStatus.FAILED
        assert r.error == "timeout"

    def test_serialization(self):
        r = ForensicReconstruction(
            reconstruction_id="recon_003",
            time_range=TimeRange(start=100.0, end=200.0),
            bounds=GeoBounds(north=40, south=39, east=-74, west=-75),
        )
        d = r.model_dump()
        assert d["reconstruction_id"] == "recon_003"
        assert d["time_range"]["start"] == 100.0
        assert d["bounds"]["north"] == 40.0


class TestIncidentReport:
    def test_creation(self):
        report = IncidentReport(
            incident_id="inc_001",
            title="Unauthorized Entry",
            summary="Target detected in restricted zone",
            reconstruction_id="recon_001",
            classification=IncidentClassification.SIGNIFICANT,
        )
        assert report.incident_id == "inc_001"
        assert report.classification == IncidentClassification.SIGNIFICANT
        assert report.status == "draft"

    def test_add_finding(self):
        report = IncidentReport(incident_id="inc_002")
        finding = IncidentFinding(
            finding_id="f1",
            title="BLE device in restricted zone",
            confidence=0.85,
        )
        report.add_finding(finding)
        assert len(report.findings) == 1
        assert report.findings[0].confidence == 0.85

    def test_add_recommendation(self):
        report = IncidentReport(incident_id="inc_003")
        rec = IncidentRecommendation(
            recommendation_id="r1",
            action="Increase patrol frequency",
            priority=2,
        )
        report.add_recommendation(rec)
        assert len(report.recommendations) == 1
        assert report.recommendations[0].priority == 2

    def test_mark_final(self):
        report = IncidentReport(incident_id="inc_004")
        report.mark_final()
        assert report.status == "final"

    def test_full_incident_flow(self):
        recon = ForensicReconstruction(
            reconstruction_id="recon_010",
            time_range=TimeRange(start=1000.0, end=2000.0),
            bounds=GeoBounds(north=40, south=39, east=-74, west=-75),
            targets=[
                TargetTimeline(
                    target_id="ble_aa:bb:cc",
                    target_type="phone",
                    first_seen=1100.0,
                    last_seen=1800.0,
                    alliance="unknown",
                ),
            ],
            events=[{"type": "sighting", "ts": 1100.0}],
            evidence_chain=[
                EvidenceItem(
                    evidence_id="ev_1",
                    sensor_id="node_01",
                    sensor_type="ble",
                    target_id="ble_aa:bb:cc",
                    confidence=0.9,
                ),
            ],
        )
        recon.mark_complete()

        report = IncidentReport(
            incident_id="inc_010",
            title="Unknown Device in Perimeter",
            reconstruction_id=recon.reconstruction_id,
            reconstruction=recon,
            entities=["ble_aa:bb:cc"],
            classification=IncidentClassification.NOTABLE,
        )
        report.add_finding(IncidentFinding(
            finding_id="f1",
            title="Unrecognized phone detected",
            confidence=0.9,
            target_refs=["ble_aa:bb:cc"],
        ))
        report.mark_final()

        assert report.status == "final"
        assert len(report.entities) == 1
        assert report.reconstruction.total_targets == 1

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for IntelligenceReport model."""

from datetime import datetime, timezone

from tritium_lib.models.report import (
    ClassificationLevel,
    IntelligenceReport,
    ReportFinding,
    ReportRecommendation,
    ReportStatus,
)


def test_report_creation():
    r = IntelligenceReport(
        report_id="rpt-001",
        title="Suspicious BLE cluster",
        summary="Three unknown BLE devices appeared simultaneously",
        entities=["ble_AA:BB:CC:DD:EE:01", "ble_AA:BB:CC:DD:EE:02"],
        created_by="amy",
        classification_level=ClassificationLevel.FOUO,
    )
    assert r.report_id == "rpt-001"
    assert r.status == ReportStatus.DRAFT
    assert len(r.entities) == 2
    assert r.classification_level == ClassificationLevel.FOUO


def test_add_finding():
    r = IntelligenceReport(report_id="rpt-002", title="Test")
    f = ReportFinding(
        finding_id="f-1",
        title="Coordinated arrival",
        description="Devices arrived within 2 seconds of each other",
        confidence=0.85,
        evidence_refs=["sighting-101", "sighting-102"],
        tags=["coordinated", "ble"],
    )
    r.add_finding(f)
    assert len(r.findings) == 1
    assert r.findings[0].confidence == 0.85
    assert r.updated_at is not None


def test_add_recommendation():
    r = IntelligenceReport(report_id="rpt-003", title="Test")
    rec = ReportRecommendation(
        recommendation_id="rec-1",
        action="Deploy camera to sector 4",
        priority=2,
        rationale="BLE cluster detected but no visual confirmation",
    )
    r.add_recommendation(rec)
    assert len(r.recommendations) == 1
    assert r.recommendations[0].priority == 2


def test_mark_final():
    r = IntelligenceReport(report_id="rpt-004", title="Test")
    assert r.status == ReportStatus.DRAFT
    r.mark_final()
    assert r.status == ReportStatus.FINAL
    assert r.updated_at is not None


def test_classification_levels():
    for level in ClassificationLevel:
        r = IntelligenceReport(
            report_id=f"rpt-{level.value}",
            classification_level=level,
        )
        assert r.classification_level == level


def test_report_import_from_models():
    """Verify report models are accessible from the models package."""
    from tritium_lib.models import IntelligenceReport, ReportFinding
    r = IntelligenceReport(report_id="rpt-import", title="Import test")
    assert r.report_id == "rpt-import"

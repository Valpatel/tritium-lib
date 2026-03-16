# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for IntelligencePackage and related models."""

import time

import pytest

from tritium_lib.models.intelligence_package import (
    ChainOfCustody,
    EvidenceType,
    IntelClassification,
    IntelligencePackage,
    PackageDossier,
    PackageEvent,
    PackageEvidence,
    PackageImportResult,
    PackageStatus,
    PackageTarget,
    create_intelligence_package,
    validate_package_import,
)


class TestPackageTarget:
    def test_defaults(self):
        t = PackageTarget(target_id="ble_AA:BB:CC")
        assert t.target_id == "ble_AA:BB:CC"
        assert t.entity_type == "unknown"
        assert t.confidence == 0.5

    def test_full_target(self):
        t = PackageTarget(
            target_id="det_person_1",
            name="Suspect Alpha",
            entity_type="person",
            classification="person",
            alliance="hostile",
            source="yolo",
            lat=33.45,
            lng=-112.07,
            confidence=0.92,
            identifiers={"mac": "AA:BB:CC:DD:EE:FF"},
            threat_level="high",
            sighting_count=15,
        )
        assert t.alliance == "hostile"
        assert t.sighting_count == 15


class TestPackageEvent:
    def test_defaults(self):
        e = PackageEvent()
        assert e.event_id
        assert e.timestamp > 0

    def test_with_data(self):
        e = PackageEvent(
            event_type="geofence_breach",
            target_ids=["ble_1", "ble_2"],
            description="Two targets entered restricted zone",
            severity="warning",
        )
        assert len(e.target_ids) == 2
        assert e.severity == "warning"


class TestPackageDossier:
    def test_defaults(self):
        d = PackageDossier(target_id="ble_1")
        assert d.target_id == "ble_1"
        assert d.dossier_id
        assert d.signals == []

    def test_with_enrichments(self):
        d = PackageDossier(
            target_id="ble_1",
            signals=[{"type": "rssi", "value": -65}],
            enrichments=[{"source": "oui", "manufacturer": "Apple"}],
            analyst_notes=["Likely iPhone 15"],
            threat_assessment="Low risk personal device",
        )
        assert len(d.signals) == 1
        assert len(d.analyst_notes) == 1


class TestPackageEvidence:
    def test_defaults(self):
        e = PackageEvidence()
        assert e.evidence_type == EvidenceType.ANALYST_NOTE
        assert e.evidence_id

    def test_screenshot(self):
        e = PackageEvidence(
            evidence_type=EvidenceType.SCREENSHOT,
            title="Map capture at 14:30",
            mime_type="image/png",
            size_bytes=45000,
        )
        assert e.evidence_type == EvidenceType.SCREENSHOT
        assert e.size_bytes == 45000


class TestChainOfCustody:
    def test_defaults(self):
        c = ChainOfCustody(actor="analyst1", action="created")
        assert c.actor == "analyst1"
        assert c.timestamp > 0


class TestIntelligencePackage:
    def test_create_empty(self):
        pkg = IntelligencePackage()
        assert pkg.package_id
        assert pkg.status == PackageStatus.DRAFT
        assert pkg.classification == IntelClassification.UNCLASSIFIED
        assert pkg.target_count == 0

    def test_add_target(self):
        pkg = IntelligencePackage()
        t = PackageTarget(target_id="ble_1", name="Phone A")
        pkg.add_target(t)
        assert pkg.target_count == 1
        assert pkg.targets[0].target_id == "ble_1"

    def test_add_event(self):
        pkg = IntelligencePackage()
        e = PackageEvent(event_type="detection", description="New target detected")
        pkg.add_event(e)
        assert pkg.event_count == 1

    def test_add_dossier(self):
        pkg = IntelligencePackage()
        d = PackageDossier(target_id="ble_1")
        pkg.add_dossier(d)
        assert pkg.dossier_count == 1

    def test_add_evidence(self):
        pkg = IntelligencePackage()
        e = PackageEvidence(title="Map screenshot")
        pkg.add_evidence(e)
        assert pkg.evidence_count == 1

    def test_finalize(self):
        pkg = IntelligencePackage()
        pkg.add_target(PackageTarget(target_id="t1"))
        pkg.add_target(PackageTarget(target_id="t2"))
        pkg.add_event(PackageEvent(event_type="alert"))
        pkg.finalize()
        assert pkg.status == PackageStatus.FINALIZED
        assert pkg.finalized_at is not None
        assert pkg.target_count == 2
        assert pkg.event_count == 1

    def test_chain_of_custody(self):
        pkg = IntelligencePackage()
        pkg.add_custody_entry(
            actor="analyst1",
            action="created",
            site_id="site-a",
            site_name="Alpha Base",
        )
        pkg.add_custody_entry(
            actor="system",
            action="transmitted",
            site_id="site-a",
        )
        assert len(pkg.custody_chain) == 2
        assert pkg.custody_chain[0].actor == "analyst1"

    def test_target_ids(self):
        pkg = IntelligencePackage()
        pkg.add_target(PackageTarget(target_id="a"))
        pkg.add_target(PackageTarget(target_id="b"))
        ids = pkg.target_ids()
        assert ids == ["a", "b"]

    def test_is_expired_not_set(self):
        pkg = IntelligencePackage()
        assert not pkg.is_expired()

    def test_is_expired_past(self):
        pkg = IntelligencePackage(expires_at=time.time() - 100)
        assert pkg.is_expired()

    def test_is_expired_future(self):
        pkg = IntelligencePackage(expires_at=time.time() + 3600)
        assert not pkg.is_expired()

    def test_full_lifecycle(self):
        pkg = create_intelligence_package(
            source_site_id="site-a",
            source_site_name="Alpha Base",
            title="Perimeter Breach Investigation",
            description="Three unknown devices detected at north perimeter",
            created_by="analyst1",
            classification=IntelClassification.RESTRICTED,
            tags=["perimeter", "breach"],
        )
        assert pkg.source_site_id == "site-a"
        assert pkg.classification == IntelClassification.RESTRICTED
        assert len(pkg.custody_chain) == 1

        # Add content
        pkg.add_target(PackageTarget(
            target_id="ble_unknown_1",
            entity_type="phone",
            alliance="unknown",
        ))
        pkg.add_dossier(PackageDossier(
            target_id="ble_unknown_1",
            threat_assessment="Unknown device, possible surveillance",
        ))
        pkg.add_evidence(PackageEvidence(
            evidence_type=EvidenceType.SCREENSHOT,
            title="Heatmap showing device positions",
        ))
        pkg.finalize()
        assert pkg.status == PackageStatus.FINALIZED
        assert pkg.target_count == 1
        assert pkg.dossier_count == 1
        assert pkg.evidence_count == 1


class TestCreateIntelligencePackage:
    def test_basic(self):
        pkg = create_intelligence_package(
            source_site_id="hq",
            title="Test Package",
            created_by="operator",
        )
        assert pkg.source_site_id == "hq"
        assert pkg.title == "Test Package"
        assert len(pkg.custody_chain) == 1
        assert pkg.custody_chain[0].action == "created"

    def test_with_tags(self):
        pkg = create_intelligence_package(tags=["urgent", "perimeter"])
        assert "urgent" in pkg.tags


class TestValidatePackageImport:
    def test_valid_package(self):
        pkg = IntelligencePackage()
        pkg.add_target(PackageTarget(target_id="t1"))
        result = validate_package_import(pkg)
        assert result.success

    def test_expired_package(self):
        pkg = IntelligencePackage(expires_at=time.time() - 100)
        result = validate_package_import(pkg)
        assert not result.success
        assert any("expired" in e for e in result.errors)

    def test_rejected_package(self):
        pkg = IntelligencePackage(status=PackageStatus.REJECTED)
        result = validate_package_import(pkg)
        assert not result.success

    def test_draft_warning(self):
        pkg = IntelligencePackage(status=PackageStatus.DRAFT)
        pkg.add_target(PackageTarget(target_id="t1"))
        result = validate_package_import(pkg)
        assert result.success
        assert any("draft" in w for w in result.warnings)

    def test_wrong_destination(self):
        pkg = IntelligencePackage(destination_site_id="site-b")
        pkg.add_target(PackageTarget(target_id="t1"))
        result = validate_package_import(pkg, local_site_id="site-c")
        assert result.success
        assert any("addressed" in w for w in result.warnings)

    def test_empty_package(self):
        pkg = IntelligencePackage()
        result = validate_package_import(pkg)
        assert result.success
        assert any("no intelligence" in w.lower() for w in result.warnings)

    def test_duplicate_target_ids(self):
        pkg = IntelligencePackage()
        pkg.add_target(PackageTarget(target_id="t1"))
        pkg.add_target(PackageTarget(target_id="t1"))
        result = validate_package_import(pkg)
        assert any("duplicate" in w.lower() for w in result.warnings)


class TestPackageImportResult:
    def test_defaults(self):
        r = PackageImportResult()
        assert r.success
        assert r.targets_imported == 0

    def test_with_counts(self):
        r = PackageImportResult(
            package_id="pkg-1",
            targets_imported=5,
            targets_merged=2,
            events_imported=10,
            dossiers_imported=3,
            evidence_imported=1,
        )
        assert r.targets_imported == 5
        assert r.targets_merged == 2

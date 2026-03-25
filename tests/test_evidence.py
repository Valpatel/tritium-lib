# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.evidence — evidence collection and chain-of-custody."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tritium_lib.evidence import (
    AssociationData,
    ClassificationData,
    CustodyAction,
    CustodyEvent,
    Evidence,
    EvidenceChain,
    EvidenceCollection,
    EvidenceExporter,
    EvidenceStatus,
    EvidenceType,
    ExportEntry,
    InvestigationStatus,
    SignalCaptureData,
    TrackLogData,
    TrackLogEntry,
    ZoneEventData,
    collect_from_target,
    compute_sha256,
    hash_bytes,
    hash_evidence,
    verify_hash,
    verify_integrity,
)


# ---------------------------------------------------------------------------
# Evidence model tests
# ---------------------------------------------------------------------------

class TestEvidenceModel:
    """Tests for the Evidence data model."""

    def test_create_evidence_defaults(self):
        ev = Evidence(evidence_type=EvidenceType.SIGNAL_CAPTURE)
        assert ev.evidence_id  # auto-generated
        assert ev.evidence_type == EvidenceType.SIGNAL_CAPTURE
        assert ev.status == EvidenceStatus.COLLECTED
        assert ev.target_id == ""
        assert ev.sha256 == ""
        assert ev.data == {}
        assert ev.tags == []
        assert ev.notes == ""
        assert ev.investigation_id == ""

    def test_create_evidence_full(self):
        ev = Evidence(
            evidence_type=EvidenceType.TRACK_LOG,
            target_id="ble_aa:bb:cc",
            collected_by="analyst",
            source_sensor="edge-01",
            data={"entries": [{"lat": 40.0, "lng": -74.0}]},
            tags=["priority", "hostile"],
            notes="Suspicious movement pattern",
        )
        assert ev.target_id == "ble_aa:bb:cc"
        assert ev.collected_by == "analyst"
        assert ev.source_sensor == "edge-01"
        assert len(ev.data["entries"]) == 1
        assert "priority" in ev.tags
        assert ev.notes == "Suspicious movement pattern"

    def test_evidence_seal(self):
        ev = Evidence(evidence_type=EvidenceType.ZONE_EVENT)
        assert ev.status == EvidenceStatus.COLLECTED
        ev.seal()
        assert ev.status == EvidenceStatus.SEALED

    def test_evidence_mark_verified(self):
        ev = Evidence(evidence_type=EvidenceType.SIGNAL_CAPTURE)
        ev.mark_verified()
        assert ev.status == EvidenceStatus.VERIFIED

    def test_evidence_mark_challenged(self):
        ev = Evidence(evidence_type=EvidenceType.SIGNAL_CAPTURE)
        ev.mark_challenged()
        assert ev.status == EvidenceStatus.CHALLENGED

    def test_evidence_to_summary(self):
        ev = Evidence(
            evidence_type=EvidenceType.CLASSIFICATION,
            target_id="det_person_1",
            sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        )
        summary = ev.to_summary()
        assert summary["type"] == "classification"
        assert summary["target_id"] == "det_person_1"
        assert summary["sha256"].endswith("...")
        assert summary["status"] == "collected"

    def test_evidence_unique_ids(self):
        ev1 = Evidence(evidence_type=EvidenceType.SIGNAL_CAPTURE)
        ev2 = Evidence(evidence_type=EvidenceType.SIGNAL_CAPTURE)
        assert ev1.evidence_id != ev2.evidence_id

    def test_evidence_collected_at_is_utc(self):
        ev = Evidence(evidence_type=EvidenceType.SIGNAL_CAPTURE)
        assert ev.collected_at.tzinfo is not None


class TestEvidenceTypes:
    """Tests for evidence type enums."""

    def test_all_evidence_types(self):
        expected = {
            "signal_capture", "track_log", "zone_event", "association",
            "classification", "screenshot", "audio_capture", "manual_note",
            "sensor_raw", "communication",
        }
        actual = {t.value for t in EvidenceType}
        assert actual == expected

    def test_all_evidence_statuses(self):
        expected = {
            "collected", "verified", "sealed", "challenged", "archived", "expunged",
        }
        actual = {s.value for s in EvidenceStatus}
        assert actual == expected


class TestDataModels:
    """Tests for typed data models."""

    def test_signal_capture_data(self):
        d = SignalCaptureData(
            signal_type="ble_advertisement",
            mac_address="aa:bb:cc:dd:ee:ff",
            rssi=-65.0,
            sensor_id="edge-01",
        )
        assert d.signal_type == "ble_advertisement"
        assert d.rssi == -65.0

    def test_track_log_entry(self):
        e = TrackLogEntry(
            timestamp=1700000000.0,
            lat=40.7128,
            lng=-74.0060,
            source="gps",
            confidence=0.95,
        )
        assert e.lat == 40.7128
        assert e.confidence == 0.95

    def test_track_log_data(self):
        d = TrackLogData(
            target_id="ble_aa:bb:cc",
            entries=[TrackLogEntry(lat=40.0, lng=-74.0)],
            total_distance_m=150.0,
        )
        assert len(d.entries) == 1
        assert d.total_distance_m == 150.0

    def test_zone_event_data(self):
        d = ZoneEventData(
            zone_id="zone-alpha",
            zone_name="Zone Alpha",
            target_id="ble_aa:bb:cc",
            event_type="entry",
            entry_time=1700000000.0,
        )
        assert d.zone_id == "zone-alpha"
        assert d.event_type == "entry"

    def test_association_data(self):
        d = AssociationData(
            target_a="ble_aa:bb:cc",
            target_b="det_person_1",
            association_type="co_located",
            distance_m=2.5,
            confidence=0.85,
        )
        assert d.association_type == "co_located"
        assert d.confidence == 0.85

    def test_classification_data(self):
        d = ClassificationData(
            target_id="ble_aa:bb:cc",
            classifier="device_classifier",
            label="phone",
            confidence=0.92,
            features_used=["oui", "rssi_pattern", "adv_interval"],
        )
        assert d.label == "phone"
        assert len(d.features_used) == 3


# ---------------------------------------------------------------------------
# Integrity tests
# ---------------------------------------------------------------------------

class TestIntegrity:
    """Tests for SHA-256 integrity operations."""

    def test_compute_sha256_deterministic(self):
        data = {"key": "value", "number": 42}
        h1 = compute_sha256(data)
        h2 = compute_sha256(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_sha256_key_order_independent(self):
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert compute_sha256(d1) == compute_sha256(d2)

    def test_compute_sha256_different_data(self):
        d1 = {"key": "value1"}
        d2 = {"key": "value2"}
        assert compute_sha256(d1) != compute_sha256(d2)

    def test_hash_evidence_sets_sha256(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            data={"rssi": -65, "mac": "aa:bb:cc"},
        )
        assert ev.sha256 == ""
        h = hash_evidence(ev)
        assert ev.sha256 == h
        assert len(h) == 64

    def test_verify_integrity_pass(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            data={"rssi": -65, "mac": "aa:bb:cc"},
        )
        hash_evidence(ev)
        assert verify_integrity(ev) is True
        assert ev.status == EvidenceStatus.VERIFIED

    def test_verify_integrity_fail_tampered(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            data={"rssi": -65, "mac": "aa:bb:cc"},
        )
        hash_evidence(ev)
        # Tamper with the data
        ev.data["rssi"] = -30
        assert verify_integrity(ev) is False
        assert ev.status == EvidenceStatus.CHALLENGED

    def test_verify_integrity_no_hash(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            data={"rssi": -65},
        )
        assert verify_integrity(ev) is False

    def test_verify_hash_standalone(self):
        data = {"test": "data"}
        h = compute_sha256(data)
        assert verify_hash(data, h) is True
        assert verify_hash(data, "wrong_hash") is False

    def test_hash_bytes(self):
        raw = b"raw binary evidence data"
        h = hash_bytes(raw)
        assert len(h) == 64
        # Same input = same hash
        assert hash_bytes(raw) == h
        # Different input = different hash
        assert hash_bytes(b"different") != h

    def test_hash_evidence_empty_data(self):
        ev = Evidence(
            evidence_type=EvidenceType.MANUAL_NOTE,
            data={},
        )
        h = hash_evidence(ev)
        assert len(h) == 64
        assert verify_integrity(ev) is True


# ---------------------------------------------------------------------------
# Chain of custody tests
# ---------------------------------------------------------------------------

class TestEvidenceChain:
    """Tests for chain of custody tracking."""

    def test_create_chain(self):
        chain = EvidenceChain(evidence_id="ev_001")
        assert chain.evidence_id == "ev_001"
        assert chain.custodian == ""
        assert chain.event_count == 0

    def test_record_collection(self):
        chain = EvidenceChain(evidence_id="ev_001")
        event = chain.record_collection(actor="analyst", sha256="abc123")
        assert event.action == CustodyAction.COLLECTED
        assert event.actor == "analyst"
        assert event.sha256_at_time == "abc123"
        assert chain.custodian == "analyst"
        assert chain.event_count == 1

    def test_record_access(self):
        chain = EvidenceChain(evidence_id="ev_001")
        chain.record_collection(actor="analyst")
        event = chain.record_access(
            actor="supervisor",
            details="Reviewed evidence",
            ip_address="10.0.0.5",
        )
        assert event.action == CustodyAction.ACCESSED
        assert event.ip_address == "10.0.0.5"
        assert chain.event_count == 2

    def test_record_transfer(self):
        chain = EvidenceChain(evidence_id="ev_001")
        chain.record_collection(actor="analyst")
        event = chain.record_transfer(
            from_custodian="analyst",
            to_custodian="legal",
            details="For court proceedings",
        )
        assert event.action == CustodyAction.TRANSFERRED
        assert event.from_custodian == "analyst"
        assert event.to_custodian == "legal"
        assert chain.custodian == "legal"

    def test_record_verification_pass(self):
        chain = EvidenceChain(evidence_id="ev_001")
        event = chain.record_verification(
            actor="system",
            passed=True,
            sha256="abc123",
        )
        assert event.action == CustodyAction.VERIFIED
        assert "PASSED" in event.details

    def test_record_verification_fail(self):
        chain = EvidenceChain(evidence_id="ev_001")
        event = chain.record_verification(
            actor="system",
            passed=False,
            sha256="abc123",
        )
        assert event.action == CustodyAction.CHALLENGED
        assert "FAILED" in event.details

    def test_record_seal(self):
        chain = EvidenceChain(evidence_id="ev_001")
        event = chain.record_seal(actor="supervisor", sha256="def456")
        assert event.action == CustodyAction.SEALED
        assert event.sha256_at_time == "def456"

    def test_record_export(self):
        chain = EvidenceChain(evidence_id="ev_001")
        event = chain.record_export(actor="analyst", details="ZIP export")
        assert event.action == CustodyAction.EXPORTED

    def test_record_annotation(self):
        chain = EvidenceChain(evidence_id="ev_001")
        event = chain.record_annotation(
            actor="analyst",
            details="Added hostile tag",
        )
        assert event.action == CustodyAction.ANNOTATED
        assert event.details == "Added hostile tag"

    def test_get_events_by_action(self):
        chain = EvidenceChain(evidence_id="ev_001")
        chain.record_collection(actor="analyst")
        chain.record_access(actor="supervisor")
        chain.record_access(actor="operator")
        accesses = chain.get_events_by_action(CustodyAction.ACCESSED)
        assert len(accesses) == 2

    def test_get_actors(self):
        chain = EvidenceChain(evidence_id="ev_001")
        chain.record_collection(actor="analyst")
        chain.record_access(actor="supervisor")
        chain.record_access(actor="analyst")  # duplicate
        actors = chain.get_actors()
        assert actors == ["analyst", "supervisor"]

    def test_last_event(self):
        chain = EvidenceChain(evidence_id="ev_001")
        assert chain.last_event is None
        chain.record_collection(actor="analyst")
        chain.record_access(actor="supervisor")
        assert chain.last_event.actor == "supervisor"

    def test_chain_to_summary(self):
        chain = EvidenceChain(evidence_id="ev_001")
        chain.record_collection(actor="analyst")
        summary = chain.to_summary()
        assert summary["evidence_id"] == "ev_001"
        assert summary["custodian"] == "analyst"
        assert summary["event_count"] == 1
        assert summary["last_action"] == "collected"


# ---------------------------------------------------------------------------
# Collection tests
# ---------------------------------------------------------------------------

class TestEvidenceCollection:
    """Tests for evidence collection management."""

    def _make_evidence(self, etype=EvidenceType.SIGNAL_CAPTURE, target="t1"):
        return Evidence(
            evidence_type=etype,
            target_id=target,
            data={"test": "data"},
        )

    def test_create_collection(self):
        c = EvidenceCollection(
            title="Test Investigation",
            created_by="analyst",
        )
        assert c.title == "Test Investigation"
        assert c.status == InvestigationStatus.OPEN
        assert c.evidence_count == 0

    def test_add_evidence(self):
        c = EvidenceCollection(title="Test")
        ev = self._make_evidence()
        result = c.add_evidence(ev, collector="analyst")
        assert result.investigation_id == c.collection_id
        assert result.sha256 != ""  # hash was computed
        assert c.evidence_count == 1
        assert ev.evidence_id in c.chains  # chain was created

    def test_get_evidence(self):
        c = EvidenceCollection(title="Test")
        ev = self._make_evidence()
        c.add_evidence(ev)
        found = c.get_evidence(ev.evidence_id)
        assert found is ev
        assert c.get_evidence("nonexistent") is None

    def test_get_chain(self):
        c = EvidenceCollection(title="Test")
        ev = self._make_evidence()
        c.add_evidence(ev, collector="analyst")
        chain = c.get_chain(ev.evidence_id)
        assert chain is not None
        assert chain.event_count == 1  # collection event
        assert c.get_chain("nonexistent") is None

    def test_remove_evidence(self):
        c = EvidenceCollection(title="Test")
        ev = self._make_evidence()
        c.add_evidence(ev)
        assert c.remove_evidence(ev.evidence_id) is True
        assert c.evidence_count == 0
        assert c.get_evidence(ev.evidence_id) is None

    def test_remove_sealed_evidence_fails(self):
        c = EvidenceCollection(title="Test")
        ev = self._make_evidence()
        c.add_evidence(ev)
        ev.seal()
        assert c.remove_evidence(ev.evidence_id) is False
        assert c.evidence_count == 1

    def test_remove_nonexistent(self):
        c = EvidenceCollection(title="Test")
        assert c.remove_evidence("nope") is False

    def test_find_by_type(self):
        c = EvidenceCollection(title="Test")
        c.add_evidence(self._make_evidence(EvidenceType.SIGNAL_CAPTURE))
        c.add_evidence(self._make_evidence(EvidenceType.TRACK_LOG))
        c.add_evidence(self._make_evidence(EvidenceType.SIGNAL_CAPTURE))
        signals = c.find_by_type(EvidenceType.SIGNAL_CAPTURE)
        assert len(signals) == 2

    def test_find_by_target(self):
        c = EvidenceCollection(title="Test")
        c.add_evidence(self._make_evidence(target="ble_aa"))
        c.add_evidence(self._make_evidence(target="ble_bb"))
        c.add_evidence(self._make_evidence(target="ble_aa"))
        found = c.find_by_target("ble_aa")
        assert len(found) == 2

    def test_find_by_tag(self):
        c = EvidenceCollection(title="Test")
        ev1 = self._make_evidence()
        ev1.tags = ["hostile", "priority"]
        ev2 = self._make_evidence()
        ev2.tags = ["routine"]
        c.add_evidence(ev1)
        c.add_evidence(ev2)
        found = c.find_by_tag("hostile")
        assert len(found) == 1

    def test_find_by_status(self):
        c = EvidenceCollection(title="Test")
        ev1 = self._make_evidence()
        ev2 = self._make_evidence()
        c.add_evidence(ev1)
        c.add_evidence(ev2)
        ev1.seal()
        sealed = c.find_by_status(EvidenceStatus.SEALED)
        assert len(sealed) == 1

    def test_verify_all(self):
        c = EvidenceCollection(title="Test")
        ev1 = self._make_evidence()
        ev2 = self._make_evidence()
        c.add_evidence(ev1)
        c.add_evidence(ev2)
        results = c.verify_all()
        assert all(results.values())
        # Chains should have verification events
        chain = c.get_chain(ev1.evidence_id)
        assert chain.event_count == 2  # collected + verified

    def test_verify_all_detects_tampering(self):
        c = EvidenceCollection(title="Test")
        ev = self._make_evidence()
        c.add_evidence(ev)
        # Tamper
        ev.data["test"] = "tampered"
        results = c.verify_all()
        assert results[ev.evidence_id] is False

    def test_seal_all(self):
        c = EvidenceCollection(title="Test")
        c.add_evidence(self._make_evidence())
        c.add_evidence(self._make_evidence())
        count = c.seal_all(actor="supervisor")
        assert count == 2
        for ev in c.evidence.values():
            assert ev.status == EvidenceStatus.SEALED

    def test_seal_all_skips_already_sealed(self):
        c = EvidenceCollection(title="Test")
        ev = self._make_evidence()
        c.add_evidence(ev)
        ev.seal()
        count = c.seal_all()
        assert count == 0

    def test_close_and_archive(self):
        c = EvidenceCollection(title="Test")
        c.close()
        assert c.status == InvestigationStatus.CLOSED
        c.archive()
        assert c.status == InvestigationStatus.ARCHIVED

    def test_target_count(self):
        c = EvidenceCollection(title="Test")
        c.add_evidence(self._make_evidence(target="ble_aa"))
        c.add_evidence(self._make_evidence(target="ble_bb"))
        c.add_evidence(self._make_evidence(target="ble_aa"))
        assert c.target_count == 2

    def test_get_type_counts(self):
        c = EvidenceCollection(title="Test")
        c.add_evidence(self._make_evidence(EvidenceType.SIGNAL_CAPTURE))
        c.add_evidence(self._make_evidence(EvidenceType.TRACK_LOG))
        c.add_evidence(self._make_evidence(EvidenceType.SIGNAL_CAPTURE))
        counts = c.get_type_counts()
        assert counts["signal_capture"] == 2
        assert counts["track_log"] == 1

    def test_to_manifest(self):
        c = EvidenceCollection(
            title="Test Investigation",
            created_by="analyst",
            tags=["urgent"],
        )
        ev = self._make_evidence(target="ble_aa")
        c.add_evidence(ev, collector="analyst")
        manifest = c.to_manifest()
        assert manifest["title"] == "Test Investigation"
        assert manifest["evidence_count"] == 1
        assert len(manifest["items"]) == 1
        assert manifest["items"][0]["evidence_id"] == ev.evidence_id
        assert manifest["items"][0]["sha256"] != ""
        assert manifest["tags"] == ["urgent"]


# ---------------------------------------------------------------------------
# Exporter tests
# ---------------------------------------------------------------------------

class TestEvidenceExporter:
    """Tests for evidence export functionality."""

    def _make_collection(self) -> EvidenceCollection:
        c = EvidenceCollection(
            title="Export Test",
            created_by="analyst",
        )
        ev1 = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            target_id="ble_aa:bb:cc",
            data={"rssi": -65, "mac": "aa:bb:cc"},
        )
        ev2 = Evidence(
            evidence_type=EvidenceType.TRACK_LOG,
            target_id="ble_aa:bb:cc",
            data={"entries": [{"lat": 40.0, "lng": -74.0}]},
        )
        c.add_evidence(ev1, collector="analyst")
        c.add_evidence(ev2, collector="analyst")
        return c

    def test_export_collection(self):
        c = self._make_collection()
        exporter = EvidenceExporter()
        entries = exporter.export_collection(c, actor="analyst")
        filenames = {e.filename for e in entries}
        # Should have: 2 evidence files, 2 chain files, manifest, package_hash
        assert len(entries) == 6
        assert "manifest.json" in filenames
        assert "package_hash.json" in filenames
        # All evidence files
        ev_files = [f for f in filenames if f.startswith("evidence/")]
        assert len(ev_files) == 2
        # All chain files
        chain_files = [f for f in filenames if f.startswith("chains/")]
        assert len(chain_files) == 2

    def test_export_without_chains(self):
        c = self._make_collection()
        exporter = EvidenceExporter()
        entries = exporter.export_collection(c, actor="analyst", include_chains=False)
        filenames = {e.filename for e in entries}
        chain_files = [f for f in filenames if f.startswith("chains/")]
        assert len(chain_files) == 0
        # 2 evidence + manifest + package_hash
        assert len(entries) == 4

    def test_export_entries_are_valid_json(self):
        c = self._make_collection()
        exporter = EvidenceExporter()
        entries = exporter.export_collection(c, actor="analyst")
        for entry in entries:
            data = json.loads(entry.content)
            assert isinstance(data, dict)

    def test_export_entries_have_hashes(self):
        c = self._make_collection()
        exporter = EvidenceExporter()
        entries = exporter.export_collection(c, actor="analyst")
        for entry in entries:
            assert entry.sha256 != ""
            assert len(entry.sha256) == 64

    def test_export_records_in_chain(self):
        c = self._make_collection()
        exporter = EvidenceExporter()
        exporter.export_collection(c, actor="analyst")
        # Each chain should have an export event
        for chain in c.chains.values():
            export_events = chain.get_events_by_action(CustodyAction.EXPORTED)
            assert len(export_events) == 1

    def test_export_single(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            target_id="ble_aa:bb:cc",
            data={"rssi": -65},
        )
        hash_evidence(ev)
        chain = EvidenceChain(evidence_id=ev.evidence_id)
        chain.record_collection(actor="analyst", sha256=ev.sha256)

        exporter = EvidenceExporter()
        entries = exporter.export_single(ev, chain=chain, actor="analyst")
        assert len(entries) == 2  # evidence + chain
        filenames = {e.filename for e in entries}
        assert f"evidence/{ev.evidence_id}.json" in filenames
        assert f"chains/{ev.evidence_id}.json" in filenames

    def test_export_single_no_chain(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            data={"rssi": -65},
        )
        hash_evidence(ev)
        exporter = EvidenceExporter()
        entries = exporter.export_single(ev)
        assert len(entries) == 1

    def test_manifest_contains_evidence_hashes(self):
        c = self._make_collection()
        exporter = EvidenceExporter()
        entries = exporter.export_collection(c, actor="analyst")
        manifest_entry = [e for e in entries if e.filename == "manifest.json"][0]
        manifest = json.loads(manifest_entry.content)
        assert "evidence_hashes" in manifest
        assert len(manifest["evidence_hashes"]) == 2


# ---------------------------------------------------------------------------
# Collector tests
# ---------------------------------------------------------------------------

class TestCollector:
    """Tests for auto-collection from targets."""

    def test_collect_signals(self):
        c = EvidenceCollection(title="Test")

        def signal_provider(target_id):
            return [
                {"signal_type": "ble_adv", "rssi": -65, "sensor_id": "edge-01"},
                {"signal_type": "wifi_probe", "rssi": -70, "sensor_id": "edge-02"},
            ]

        collected = collect_from_target(
            target_id="ble_aa:bb:cc",
            collection=c,
            collector="analyst",
            signal_provider=signal_provider,
        )
        assert len(collected) == 2
        assert c.evidence_count == 2
        assert all(e.evidence_type == EvidenceType.SIGNAL_CAPTURE for e in collected)
        assert "ble_aa:bb:cc" in c.target_ids

    def test_collect_track(self):
        c = EvidenceCollection(title="Test")

        def track_provider(target_id):
            return [
                {"lat": 40.0, "lng": -74.0, "ts": 1700000000},
                {"lat": 40.001, "lng": -74.001, "ts": 1700000060},
            ]

        collected = collect_from_target(
            target_id="ble_aa:bb:cc",
            collection=c,
            collector="analyst",
            track_provider=track_provider,
        )
        assert len(collected) == 1  # Single track log item
        assert collected[0].evidence_type == EvidenceType.TRACK_LOG
        assert collected[0].data["entry_count"] == 2

    def test_collect_zones(self):
        c = EvidenceCollection(title="Test")

        def zone_provider(target_id):
            return [
                {"zone_id": "z1", "event_type": "entry", "entry_time": 1700000000},
                {"zone_id": "z1", "event_type": "exit", "exit_time": 1700001000},
            ]

        collected = collect_from_target(
            target_id="ble_aa",
            collection=c,
            zone_provider=zone_provider,
        )
        assert len(collected) == 2
        assert all(e.evidence_type == EvidenceType.ZONE_EVENT for e in collected)

    def test_collect_associations(self):
        c = EvidenceCollection(title="Test")

        def assoc_provider(target_id):
            return [{"target_b": "det_person_1", "type": "co_located", "distance": 3.0}]

        collected = collect_from_target(
            target_id="ble_aa",
            collection=c,
            association_provider=assoc_provider,
        )
        assert len(collected) == 1
        assert collected[0].evidence_type == EvidenceType.ASSOCIATION

    def test_collect_classifications(self):
        c = EvidenceCollection(title="Test")

        def class_provider(target_id):
            return [{"classifier": "device_classifier", "label": "phone", "confidence": 0.92}]

        collected = collect_from_target(
            target_id="ble_aa",
            collection=c,
            classification_provider=class_provider,
        )
        assert len(collected) == 1
        assert collected[0].evidence_type == EvidenceType.CLASSIFICATION

    def test_collect_all_providers(self):
        c = EvidenceCollection(title="Test")

        collected = collect_from_target(
            target_id="ble_aa",
            collection=c,
            collector="analyst",
            signal_provider=lambda tid: [{"rssi": -65}],
            track_provider=lambda tid: [{"lat": 40.0, "lng": -74.0}],
            zone_provider=lambda tid: [{"zone_id": "z1", "event_type": "entry"}],
            association_provider=lambda tid: [{"target_b": "t2"}],
            classification_provider=lambda tid: [{"label": "phone"}],
        )
        assert len(collected) == 5
        assert c.evidence_count == 5
        types = {e.evidence_type for e in collected}
        assert types == {
            EvidenceType.SIGNAL_CAPTURE,
            EvidenceType.TRACK_LOG,
            EvidenceType.ZONE_EVENT,
            EvidenceType.ASSOCIATION,
            EvidenceType.CLASSIFICATION,
        }

    def test_collect_no_providers(self):
        c = EvidenceCollection(title="Test")
        collected = collect_from_target(target_id="ble_aa", collection=c)
        assert len(collected) == 0
        assert c.evidence_count == 0
        assert "ble_aa" in c.target_ids

    def test_collect_empty_track_skipped(self):
        c = EvidenceCollection(title="Test")
        collected = collect_from_target(
            target_id="ble_aa",
            collection=c,
            track_provider=lambda tid: [],  # empty
        )
        assert len(collected) == 0

    def test_collect_adds_target_id_to_collection(self):
        c = EvidenceCollection(title="Test")
        collect_from_target(target_id="ble_aa", collection=c)
        collect_from_target(target_id="ble_aa", collection=c)  # duplicate
        assert c.target_ids.count("ble_aa") == 1

    def test_collect_all_evidence_has_hashes(self):
        c = EvidenceCollection(title="Test")
        collected = collect_from_target(
            target_id="ble_aa",
            collection=c,
            signal_provider=lambda tid: [{"rssi": -65}],
        )
        for ev in collected:
            assert ev.sha256 != ""
            assert len(ev.sha256) == 64


# ---------------------------------------------------------------------------
# Integration / end-to-end tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_full_investigation_workflow(self):
        """Simulate a complete investigation: collect, verify, seal, export."""
        # 1. Open investigation
        collection = EvidenceCollection(
            title="Hostile device near perimeter",
            created_by="analyst",
            target_ids=["ble_aa:bb:cc:dd"],
        )
        assert collection.status == InvestigationStatus.OPEN

        # 2. Collect evidence
        collected = collect_from_target(
            target_id="ble_aa:bb:cc:dd",
            collection=collection,
            collector="analyst",
            signal_provider=lambda tid: [
                {"signal_type": "ble_advertisement", "rssi": -55, "sensor_id": "edge-01"},
                {"signal_type": "ble_advertisement", "rssi": -60, "sensor_id": "edge-02"},
            ],
            track_provider=lambda tid: [
                {"lat": 40.7128, "lng": -74.0060, "ts": 1700000000},
                {"lat": 40.7130, "lng": -74.0058, "ts": 1700000060},
            ],
            zone_provider=lambda tid: [
                {"zone_id": "perimeter", "event_type": "entry", "entry_time": 1700000000},
            ],
            classification_provider=lambda tid: [
                {"classifier": "device_classifier", "label": "phone", "confidence": 0.88},
            ],
        )
        assert len(collected) == 5
        assert collection.evidence_count == 5

        # 3. Verify all evidence
        results = collection.verify_all()
        assert all(results.values())

        # 4. Seal all evidence
        sealed_count = collection.seal_all(actor="supervisor")
        assert sealed_count == 5

        # 5. Close investigation
        collection.close()
        assert collection.status == InvestigationStatus.CLOSED

        # 6. Export
        exporter = EvidenceExporter()
        entries = exporter.export_collection(collection, actor="analyst")
        assert len(entries) > 0

        # Verify manifest
        manifest_entry = [e for e in entries if e.filename == "manifest.json"][0]
        manifest = json.loads(manifest_entry.content)
        assert manifest["evidence_count"] == 5
        assert manifest["status"] == "closed"

    def test_tamper_detection_workflow(self):
        """Detect tampering in evidence after collection."""
        collection = EvidenceCollection(title="Tamper test")
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            target_id="ble_aa",
            data={"rssi": -65, "mac": "aa:bb:cc"},
        )
        collection.add_evidence(ev, collector="analyst")

        # Verify passes initially
        results = collection.verify_all()
        assert results[ev.evidence_id] is True

        # Tamper with evidence
        ev.data["rssi"] = -30

        # Verification now fails
        results = collection.verify_all()
        assert results[ev.evidence_id] is False
        assert ev.status == EvidenceStatus.CHALLENGED

        # Chain records the failure
        chain = collection.get_chain(ev.evidence_id)
        challenged = chain.get_events_by_action(CustodyAction.CHALLENGED)
        assert len(challenged) == 1

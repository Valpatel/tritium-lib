# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.evidence.collection — EvidenceCollection."""

from tritium_lib.evidence.collection import (
    EvidenceCollection,
    InvestigationStatus,
)
from tritium_lib.evidence.models import Evidence, EvidenceStatus, EvidenceType


def _make_evidence(etype=EvidenceType.SIGNAL_CAPTURE, target_id="", tags=None, data=None):
    return Evidence(
        evidence_type=etype,
        target_id=target_id,
        tags=tags or [],
        data=data or {"test": True},
    )


class TestInvestigationStatus:
    def test_values(self):
        assert InvestigationStatus.OPEN == "open"
        assert InvestigationStatus.ACTIVE == "active"
        assert InvestigationStatus.CLOSED == "closed"
        assert InvestigationStatus.ARCHIVED == "archived"
        assert InvestigationStatus.SUSPENDED == "suspended"


class TestEvidenceCollection:
    def test_default_state(self):
        coll = EvidenceCollection(title="Test Investigation")
        assert coll.title == "Test Investigation"
        assert coll.status == InvestigationStatus.OPEN
        assert coll.evidence_count == 0
        assert coll.target_count == 0
        assert len(coll.collection_id) > 0

    def test_add_evidence(self):
        coll = EvidenceCollection(title="Test")
        ev = _make_evidence()
        result = coll.add_evidence(ev, collector="officer_a")
        assert result.investigation_id == coll.collection_id
        assert result.sha256 != ""
        assert coll.evidence_count == 1
        assert ev.evidence_id in coll.evidence
        assert ev.evidence_id in coll.chains

    def test_add_evidence_creates_chain(self):
        coll = EvidenceCollection(title="Test")
        ev = _make_evidence()
        coll.add_evidence(ev, collector="alice")
        chain = coll.get_chain(ev.evidence_id)
        assert chain is not None
        assert chain.event_count == 1
        assert chain.custodian == "alice"

    def test_get_evidence(self):
        coll = EvidenceCollection()
        ev = _make_evidence()
        coll.add_evidence(ev)
        fetched = coll.get_evidence(ev.evidence_id)
        assert fetched is ev
        assert coll.get_evidence("nonexistent") is None

    def test_remove_evidence(self):
        coll = EvidenceCollection()
        ev = _make_evidence()
        coll.add_evidence(ev)
        assert coll.remove_evidence(ev.evidence_id) is True
        assert coll.evidence_count == 0
        assert coll.get_evidence(ev.evidence_id) is None
        assert coll.get_chain(ev.evidence_id) is None

    def test_remove_sealed_evidence_fails(self):
        coll = EvidenceCollection()
        ev = _make_evidence()
        coll.add_evidence(ev)
        ev.seal()
        assert coll.remove_evidence(ev.evidence_id) is False
        assert coll.evidence_count == 1

    def test_remove_nonexistent(self):
        coll = EvidenceCollection()
        assert coll.remove_evidence("nope") is False

    def test_find_by_type(self):
        coll = EvidenceCollection()
        coll.add_evidence(_make_evidence(EvidenceType.SIGNAL_CAPTURE))
        coll.add_evidence(_make_evidence(EvidenceType.TRACK_LOG))
        coll.add_evidence(_make_evidence(EvidenceType.SIGNAL_CAPTURE))
        results = coll.find_by_type(EvidenceType.SIGNAL_CAPTURE)
        assert len(results) == 2

    def test_find_by_target(self):
        coll = EvidenceCollection()
        coll.add_evidence(_make_evidence(target_id="ble_AA"))
        coll.add_evidence(_make_evidence(target_id="ble_BB"))
        coll.add_evidence(_make_evidence(target_id="ble_AA"))
        results = coll.find_by_target("ble_AA")
        assert len(results) == 2

    def test_find_by_tag(self):
        coll = EvidenceCollection()
        coll.add_evidence(_make_evidence(tags=["suspicious", "vehicle"]))
        coll.add_evidence(_make_evidence(tags=["routine"]))
        coll.add_evidence(_make_evidence(tags=["suspicious"]))
        results = coll.find_by_tag("suspicious")
        assert len(results) == 2

    def test_find_by_status(self):
        coll = EvidenceCollection()
        ev1 = _make_evidence()
        ev2 = _make_evidence()
        coll.add_evidence(ev1)
        coll.add_evidence(ev2)
        ev1.seal()
        sealed = coll.find_by_status(EvidenceStatus.SEALED)
        assert len(sealed) == 1
        collected = coll.find_by_status(EvidenceStatus.COLLECTED)
        assert len(collected) == 1

    def test_verify_all(self):
        coll = EvidenceCollection()
        ev1 = _make_evidence(data={"a": 1})
        ev2 = _make_evidence(data={"b": 2})
        coll.add_evidence(ev1)
        coll.add_evidence(ev2)
        results = coll.verify_all()
        assert all(v is True for v in results.values())
        assert len(results) == 2

    def test_verify_all_tampered(self):
        coll = EvidenceCollection()
        ev = _make_evidence(data={"original": True})
        coll.add_evidence(ev)
        ev.data["tampered"] = True
        results = coll.verify_all()
        assert results[ev.evidence_id] is False

    def test_seal_all(self):
        coll = EvidenceCollection()
        coll.add_evidence(_make_evidence())
        coll.add_evidence(_make_evidence())
        count = coll.seal_all(actor="admin")
        assert count == 2
        for ev in coll.evidence.values():
            assert ev.status == EvidenceStatus.SEALED

    def test_seal_all_skips_already_sealed(self):
        coll = EvidenceCollection()
        ev = _make_evidence()
        coll.add_evidence(ev)
        ev.seal()
        count = coll.seal_all()
        assert count == 0

    def test_close(self):
        coll = EvidenceCollection()
        coll.close()
        assert coll.status == InvestigationStatus.CLOSED

    def test_archive(self):
        coll = EvidenceCollection()
        coll.archive()
        assert coll.status == InvestigationStatus.ARCHIVED

    def test_target_count(self):
        coll = EvidenceCollection()
        coll.add_evidence(_make_evidence(target_id="ble_AA"))
        coll.add_evidence(_make_evidence(target_id="ble_BB"))
        coll.add_evidence(_make_evidence(target_id="ble_AA"))
        coll.add_evidence(_make_evidence(target_id=""))
        assert coll.target_count == 2

    def test_get_type_counts(self):
        coll = EvidenceCollection()
        coll.add_evidence(_make_evidence(EvidenceType.SIGNAL_CAPTURE))
        coll.add_evidence(_make_evidence(EvidenceType.TRACK_LOG))
        coll.add_evidence(_make_evidence(EvidenceType.SIGNAL_CAPTURE))
        counts = coll.get_type_counts()
        assert counts["signal_capture"] == 2
        assert counts["track_log"] == 1

    def test_to_manifest(self):
        coll = EvidenceCollection(title="Case 42", created_by="admin")
        ev = _make_evidence(target_id="ble_AA")
        coll.add_evidence(ev, collector="admin")
        manifest = coll.to_manifest()
        assert manifest["title"] == "Case 42"
        assert manifest["evidence_count"] == 1
        assert len(manifest["items"]) == 1
        item = manifest["items"][0]
        assert item["evidence_id"] == ev.evidence_id
        assert item["custody_events"] == 1

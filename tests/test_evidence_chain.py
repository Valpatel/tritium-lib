# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.evidence.chain — chain of custody tracking."""

from tritium_lib.evidence.chain import (
    CustodyAction,
    CustodyEvent,
    EvidenceChain,
)


class TestCustodyAction:
    def test_all_values(self):
        actions = [
            CustodyAction.COLLECTED, CustodyAction.ACCESSED,
            CustodyAction.TRANSFERRED, CustodyAction.VERIFIED,
            CustodyAction.SEALED, CustodyAction.CHALLENGED,
            CustodyAction.ANNOTATED, CustodyAction.EXPORTED,
            CustodyAction.ARCHIVED, CustodyAction.EXPUNGED,
        ]
        assert len(actions) == 10

    def test_string_values(self):
        assert CustodyAction.COLLECTED == "collected"
        assert CustodyAction.TRANSFERRED == "transferred"
        assert CustodyAction.SEALED == "sealed"


class TestCustodyEvent:
    def test_creation(self):
        evt = CustodyEvent(
            evidence_id="ev123",
            action=CustodyAction.COLLECTED,
            actor="officer_smith",
        )
        assert evt.evidence_id == "ev123"
        assert evt.action == CustodyAction.COLLECTED
        assert evt.actor == "officer_smith"
        assert len(evt.event_id) == 16

    def test_auto_timestamp(self):
        evt = CustodyEvent(
            evidence_id="ev123",
            action=CustodyAction.ACCESSED,
        )
        assert evt.timestamp is not None

    def test_transfer_fields(self):
        evt = CustodyEvent(
            evidence_id="ev123",
            action=CustodyAction.TRANSFERRED,
            from_custodian="alice",
            to_custodian="bob",
        )
        assert evt.from_custodian == "alice"
        assert evt.to_custodian == "bob"


class TestEvidenceChain:
    def test_empty_chain(self):
        chain = EvidenceChain(evidence_id="ev123")
        assert chain.evidence_id == "ev123"
        assert chain.event_count == 0
        assert chain.last_event is None
        assert chain.get_actors() == []
        assert chain.custodian == ""

    def test_record_collection(self):
        chain = EvidenceChain(evidence_id="ev123")
        evt = chain.record_collection(
            actor="officer_jones",
            details="Collected BLE signal data",
            sha256="abc123",
        )
        assert evt.action == CustodyAction.COLLECTED
        assert evt.actor == "officer_jones"
        assert evt.to_custodian == "officer_jones"
        assert evt.sha256_at_time == "abc123"
        assert chain.custodian == "officer_jones"
        assert chain.event_count == 1

    def test_record_access(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        evt = chain.record_access(
            actor="bob",
            details="Reviewed evidence",
            ip_address="10.0.0.5",
        )
        assert evt.action == CustodyAction.ACCESSED
        assert evt.ip_address == "10.0.0.5"
        assert chain.event_count == 2

    def test_record_transfer(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        evt = chain.record_transfer(
            from_custodian="alice",
            to_custodian="bob",
            details="Handover for analysis",
        )
        assert evt.action == CustodyAction.TRANSFERRED
        assert evt.from_custodian == "alice"
        assert evt.to_custodian == "bob"
        assert chain.custodian == "bob"

    def test_record_transfer_default_actor(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        evt = chain.record_transfer(
            from_custodian="alice",
            to_custodian="bob",
        )
        assert evt.actor == "alice"

    def test_record_verification_pass(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        evt = chain.record_verification(
            actor="system",
            passed=True,
            sha256="hash123",
        )
        assert evt.action == CustodyAction.VERIFIED
        assert "PASSED" in evt.details

    def test_record_verification_fail(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        evt = chain.record_verification(
            actor="system",
            passed=False,
        )
        assert evt.action == CustodyAction.CHALLENGED
        assert "FAILED" in evt.details

    def test_record_seal(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        evt = chain.record_seal(actor="admin", sha256="final_hash")
        assert evt.action == CustodyAction.SEALED
        assert "sealed" in evt.details.lower()

    def test_record_export(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        evt = chain.record_export(actor="alice", details="PDF report")
        assert evt.action == CustodyAction.EXPORTED
        assert evt.details == "PDF report"

    def test_record_annotation(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        evt = chain.record_annotation(
            actor="bob", details="Flagged as suspicious"
        )
        assert evt.action == CustodyAction.ANNOTATED
        assert evt.details == "Flagged as suspicious"

    def test_last_event(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        chain.record_access(actor="bob")
        assert chain.last_event.action == CustodyAction.ACCESSED

    def test_get_events_by_action(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        chain.record_access(actor="bob")
        chain.record_access(actor="charlie")
        accesses = chain.get_events_by_action(CustodyAction.ACCESSED)
        assert len(accesses) == 2

    def test_get_actors(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        chain.record_access(actor="bob")
        chain.record_access(actor="alice")
        actors = chain.get_actors()
        assert actors == ["alice", "bob"]

    def test_to_summary(self):
        chain = EvidenceChain(evidence_id="ev123")
        chain.record_collection(actor="alice")
        summary = chain.to_summary()
        assert summary["evidence_id"] == "ev123"
        assert summary["custodian"] == "alice"
        assert summary["event_count"] == 1
        assert "alice" in summary["actors"]
        assert summary["last_action"] == "collected"
        assert "created_at" in summary

    def test_to_summary_empty(self):
        chain = EvidenceChain(evidence_id="ev123")
        summary = chain.to_summary()
        assert summary["last_action"] is None
        assert summary["event_count"] == 0

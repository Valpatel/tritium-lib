# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.privacy.consent — consent management."""

import time

from tritium_lib.privacy.consent import (
    ConsentManager,
    ConsentRecord,
    ConsentStatus,
    LegalBasis,
    ProcessingPurpose,
)


class TestEnums:
    def test_processing_purposes(self):
        assert ProcessingPurpose.TRACKING == "tracking"
        assert ProcessingPurpose.SENSOR_FUSION == "sensor_fusion"
        assert len(ProcessingPurpose) == 10

    def test_legal_basis(self):
        assert LegalBasis.CONSENT == "consent"
        assert LegalBasis.LEGITIMATE_INTEREST == "legitimate_interest"
        assert len(LegalBasis) == 6

    def test_consent_status(self):
        assert ConsentStatus.GRANTED == "granted"
        assert ConsentStatus.WITHDRAWN == "withdrawn"
        assert ConsentStatus.EXPIRED == "expired"
        assert ConsentStatus.PENDING == "pending"


class TestConsentRecord:
    def test_default(self):
        r = ConsentRecord()
        assert r.status == "pending"
        assert r.legal_basis == "consent"

    def test_is_active_granted(self):
        r = ConsentRecord(status=ConsentStatus.GRANTED)
        assert r.is_active() is True

    def test_is_active_pending(self):
        r = ConsentRecord(status=ConsentStatus.PENDING)
        assert r.is_active() is False

    def test_is_active_withdrawn(self):
        r = ConsentRecord(status=ConsentStatus.WITHDRAWN)
        assert r.is_active() is False

    def test_is_active_expired(self):
        r = ConsentRecord(
            status=ConsentStatus.GRANTED,
            expires_at=time.time() - 1000,
        )
        assert r.is_active() is False

    def test_is_active_not_expired(self):
        r = ConsentRecord(
            status=ConsentStatus.GRANTED,
            expires_at=time.time() + 10000,
        )
        assert r.is_active() is True

    def test_is_active_no_expiry(self):
        r = ConsentRecord(status=ConsentStatus.GRANTED, expires_at=0.0)
        assert r.is_active() is True

    def test_to_dict(self):
        r = ConsentRecord(
            consent_id="c1",
            subject_id="target_1",
            purpose="tracking",
            status=ConsentStatus.GRANTED,
        )
        d = r.to_dict()
        assert d["consent_id"] == "c1"
        assert d["subject_id"] == "target_1"
        assert d["purpose"] == "tracking"

    def test_create_factory(self):
        r = ConsentRecord.create(
            subject_id="target_1",
            purpose="tracking",
            evidence="web_form",
        )
        assert r.status == ConsentStatus.GRANTED
        assert r.subject_id == "target_1"
        assert r.purpose == "tracking"
        assert r.evidence == "web_form"
        assert r.granted_at > 0
        assert len(r.consent_id) > 0


class TestConsentManager:
    def test_grant(self):
        mgr = ConsentManager()
        record = mgr.grant("target_1", "tracking", evidence="web_form")
        assert record.status == ConsentStatus.GRANTED
        assert record.subject_id == "target_1"
        assert record.purpose == "tracking"

    def test_has_consent_after_grant(self):
        mgr = ConsentManager()
        mgr.grant("target_1", "tracking")
        assert mgr.has_consent("target_1", "tracking") is True

    def test_has_consent_no_record(self):
        mgr = ConsentManager()
        assert mgr.has_consent("target_1", "tracking") is False

    def test_withdraw(self):
        mgr = ConsentManager()
        mgr.grant("target_1", "tracking")
        result = mgr.withdraw("target_1", "tracking")
        assert result is not None
        assert result.status == ConsentStatus.WITHDRAWN
        assert mgr.has_consent("target_1", "tracking") is False

    def test_withdraw_nonexistent(self):
        mgr = ConsentManager()
        assert mgr.withdraw("target_1", "tracking") is None

    def test_withdraw_all(self):
        mgr = ConsentManager()
        mgr.grant("target_1", "tracking")
        mgr.grant("target_1", "analytics")
        results = mgr.withdraw_all("target_1")
        assert len(results) == 2
        assert mgr.has_consent("target_1", "tracking") is False
        assert mgr.has_consent("target_1", "analytics") is False

    def test_get_consent(self):
        mgr = ConsentManager()
        mgr.grant("target_1", "tracking")
        r = mgr.get_consent("target_1", "tracking")
        assert r is not None
        assert r.purpose == "tracking"

    def test_get_consent_none(self):
        mgr = ConsentManager()
        assert mgr.get_consent("nope", "tracking") is None

    def test_get_all_consents(self):
        mgr = ConsentManager()
        mgr.grant("target_1", "tracking")
        mgr.grant("target_1", "analytics")
        all_c = mgr.get_all_consents("target_1")
        assert len(all_c) == 2

    def test_get_active_consents(self):
        mgr = ConsentManager()
        mgr.grant("target_1", "tracking")
        mgr.grant("target_1", "analytics")
        mgr.withdraw("target_1", "analytics")
        active = mgr.get_active_consents("target_1")
        assert len(active) == 1
        assert active[0].purpose == "tracking"

    def test_list_subjects(self):
        mgr = ConsentManager()
        mgr.grant("target_b", "tracking")
        mgr.grant("target_a", "tracking")
        subjects = mgr.list_subjects()
        assert subjects == ["target_a", "target_b"]

    def test_count_by_purpose(self):
        mgr = ConsentManager()
        mgr.grant("t1", "tracking")
        mgr.grant("t2", "tracking")
        mgr.grant("t1", "analytics")
        counts = mgr.count_by_purpose()
        assert counts["tracking"] == 2
        assert counts["analytics"] == 1

    def test_history(self):
        mgr = ConsentManager()
        mgr.grant("target_1", "tracking")
        mgr.withdraw("target_1", "tracking")
        assert len(mgr.history) == 2

    def test_export(self):
        mgr = ConsentManager()
        mgr.grant("target_1", "tracking")
        exported = mgr.export()
        assert exported["subject_count"] == 1
        assert "target_1" in exported["records"]

    def test_import_records(self):
        mgr = ConsentManager()
        r = ConsentRecord.create("target_1", "tracking")
        count = mgr.import_records([r])
        assert count == 1
        assert mgr.has_consent("target_1", "tracking") is True

    def test_clear(self):
        mgr = ConsentManager()
        mgr.grant("target_1", "tracking")
        mgr.clear()
        assert mgr.has_consent("target_1", "tracking") is False
        assert len(mgr.history) == 0
        assert mgr.list_subjects() == []

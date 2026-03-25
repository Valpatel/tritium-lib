# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.privacy — data retention, anonymization, GDPR compliance."""

import time

import pytest

from tritium_lib.privacy import (
    # Retention
    DataCategory,
    DEFAULT_RETENTION,
    RetentionPolicy,
    PurgeResult,
    RetentionManager,
    # Anonymization
    AnonymizationLevel,
    AnonymizationResult,
    Anonymizer,
    PII_FIELDS,
    # Consent
    ProcessingPurpose,
    LegalBasis,
    ConsentStatus,
    ConsentRecord,
    ConsentManager,
    # Subject requests
    RequestType,
    RequestStatus,
    DataSubjectRequest,
    SubjectRequestManager,
    # Privacy zones
    SuppressionLevel,
    PrivacyZone,
    PrivacyZoneManager,
    ZoneCheckResult,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def retention_mgr():
    """RetentionManager with default policies."""
    return RetentionManager()


@pytest.fixture
def anonymizer():
    """Anonymizer with a fixed secret for reproducible hashes."""
    return Anonymizer(secret="test-secret-key")


@pytest.fixture
def consent_mgr():
    """Empty ConsentManager."""
    return ConsentManager()


@pytest.fixture
def request_mgr():
    """SubjectRequestManager with sample collectors/erasers."""
    mgr = SubjectRequestManager()

    # Sample data store
    store = {
        "target_123": [
            {"ts": 1000, "mac": "AA:BB:CC:DD:EE:FF"},
            {"ts": 2000, "mac": "AA:BB:CC:DD:EE:FF"},
        ],
        "target_456": [
            {"ts": 3000, "mac": "11:22:33:44:55:66"},
        ],
    }

    def collector(subject_id):
        return store.get(subject_id, [])

    def eraser(subject_id):
        data = store.pop(subject_id, [])
        return len(data)

    mgr.register_collector("sightings", collector)
    mgr.register_eraser("sightings", eraser)
    return mgr


@pytest.fixture
def zone_mgr():
    """PrivacyZoneManager with a sample zone."""
    mgr = PrivacyZoneManager()
    # Simple square around (40.0, -74.0)
    polygon = [
        (39.99, -74.01),
        (39.99, -73.99),
        (40.01, -73.99),
        (40.01, -74.01),
    ]
    mgr.add_zone("Test Zone", polygon, suppression="full", reason="testing")
    return mgr


# ===========================================================================
# 1. RetentionPolicy tests
# ===========================================================================

class TestRetentionPolicy:
    def test_default_retention_categories(self):
        """All expected categories have defaults."""
        assert DataCategory.REALTIME_SIGHTINGS in DEFAULT_RETENTION
        assert DataCategory.TARGET_HISTORY in DEFAULT_RETENTION
        assert DataCategory.DOSSIERS in DEFAULT_RETENTION
        assert DataCategory.INCIDENTS in DEFAULT_RETENTION
        assert DataCategory.AUDIT_TRAIL in DEFAULT_RETENTION

    def test_default_retention_values(self):
        """Default values match spec."""
        day = 86400
        assert DEFAULT_RETENTION[DataCategory.REALTIME_SIGHTINGS] == 7 * day
        assert DEFAULT_RETENTION[DataCategory.TARGET_HISTORY] == 30 * day
        assert DEFAULT_RETENTION[DataCategory.DOSSIERS] == 90 * day
        assert DEFAULT_RETENTION[DataCategory.INCIDENTS] == 365 * day
        assert DEFAULT_RETENTION[DataCategory.AUDIT_TRAIL] == 7 * 365 * day

    def test_policy_retention_days(self):
        """retention_days property works."""
        policy = RetentionPolicy(
            category="test",
            retention_seconds=86400 * 30,
        )
        assert policy.retention_days == 30.0

    def test_policy_is_expired_true(self):
        """Old record is expired."""
        policy = RetentionPolicy(
            category="test",
            retention_seconds=86400 * 7,
        )
        old_ts = time.time() - (86400 * 10)  # 10 days ago
        assert policy.is_expired(old_ts)

    def test_policy_is_expired_false(self):
        """Recent record is not expired."""
        policy = RetentionPolicy(
            category="test",
            retention_seconds=86400 * 7,
        )
        recent_ts = time.time() - (86400 * 3)  # 3 days ago
        assert not policy.is_expired(recent_ts)

    def test_policy_disabled_never_expires(self):
        """Disabled policy never marks records as expired."""
        policy = RetentionPolicy(
            category="test",
            retention_seconds=1,
            enabled=False,
        )
        old_ts = time.time() - 10000
        assert not policy.is_expired(old_ts)

    def test_policy_to_dict(self):
        """Serialization includes all fields."""
        policy = RetentionPolicy(
            category="test",
            retention_seconds=86400,
            description="Test policy",
            legal_basis="consent",
        )
        d = policy.to_dict()
        assert d["category"] == "test"
        assert d["retention_seconds"] == 86400
        assert d["retention_days"] == 1.0
        assert d["description"] == "Test policy"
        assert d["legal_basis"] == "consent"
        assert d["enabled"] is True


# ===========================================================================
# 2. RetentionManager tests
# ===========================================================================

class TestRetentionManager:
    def test_default_policies_loaded(self, retention_mgr):
        """Manager starts with default policies for all categories."""
        policies = retention_mgr.list_policies()
        categories = {p.category for p in policies}
        assert DataCategory.REALTIME_SIGHTINGS in categories
        assert DataCategory.AUDIT_TRAIL in categories

    def test_set_custom_policy(self, retention_mgr):
        """Can add or replace a policy."""
        custom = RetentionPolicy(
            category="custom_data",
            retention_seconds=86400 * 14,
            description="Two weeks",
        )
        retention_mgr.set_policy(custom)
        assert retention_mgr.get_policy("custom_data") is custom

    def test_remove_policy(self, retention_mgr):
        """Removing a policy returns True, second call returns False."""
        assert retention_mgr.remove_policy(DataCategory.REALTIME_SIGHTINGS)
        assert not retention_mgr.remove_policy(DataCategory.REALTIME_SIGHTINGS)

    def test_enforce_calls_handlers(self, retention_mgr):
        """enforce() invokes registered handlers and returns results."""
        purged = []

        def handler(category, cutoff):
            purged.append((category, cutoff))
            return 42

        retention_mgr.register_handler(DataCategory.REALTIME_SIGHTINGS, handler)
        results = retention_mgr.enforce()

        # At least one result (for the category with a handler)
        sighting_results = [r for r in results if r.category == DataCategory.REALTIME_SIGHTINGS]
        assert len(sighting_results) == 1
        assert sighting_results[0].purged_count == 42
        assert sighting_results[0].success
        assert len(purged) == 1

    def test_enforce_skips_disabled(self):
        """Disabled policies are skipped during enforcement."""
        policy = RetentionPolicy(
            category="disabled",
            retention_seconds=1,
            enabled=False,
        )
        mgr = RetentionManager(policies={"disabled": policy})
        called = []
        mgr.register_handler("disabled", lambda c, t: called.append(1) or 0)
        results = mgr.enforce()
        assert len(results) == 0
        assert len(called) == 0

    def test_enforce_skips_no_handler(self, retention_mgr):
        """Categories without handlers produce no results."""
        results = retention_mgr.enforce()
        assert len(results) == 0  # no handlers registered

    def test_enforce_handles_errors(self, retention_mgr):
        """Handler errors are captured, not raised."""
        def bad_handler(category, cutoff):
            raise RuntimeError("DB offline")

        retention_mgr.register_handler(DataCategory.REALTIME_SIGHTINGS, bad_handler)
        results = retention_mgr.enforce()
        assert len(results) == 1
        assert not results[0].success
        assert "DB offline" in results[0].errors[0]

    def test_enforce_category(self, retention_mgr):
        """enforce_category() runs a single category."""
        retention_mgr.register_handler(
            DataCategory.DOSSIERS, lambda c, t: 10
        )
        result = retention_mgr.enforce_category(DataCategory.DOSSIERS)
        assert result is not None
        assert result.purged_count == 10

    def test_enforce_category_no_policy(self, retention_mgr):
        """enforce_category() returns None for unknown category."""
        result = retention_mgr.enforce_category("nonexistent")
        assert result is None

    def test_history_tracking(self, retention_mgr):
        """Enforcement results are stored in history."""
        retention_mgr.register_handler(DataCategory.REALTIME_SIGHTINGS, lambda c, t: 5)
        retention_mgr.enforce()
        assert len(retention_mgr.history) == 1
        retention_mgr.enforce()
        assert len(retention_mgr.history) == 2

    def test_clear_history(self, retention_mgr):
        retention_mgr.register_handler(DataCategory.REALTIME_SIGHTINGS, lambda c, t: 0)
        retention_mgr.enforce()
        assert retention_mgr.clear_history() == 1
        assert len(retention_mgr.history) == 0

    def test_unregister_handler(self, retention_mgr):
        retention_mgr.register_handler("x", lambda c, t: 0)
        assert retention_mgr.unregister_handler("x")
        assert not retention_mgr.unregister_handler("x")

    def test_export(self, retention_mgr):
        d = retention_mgr.export()
        assert "policies" in d
        assert "handlers_registered" in d
        assert isinstance(d["history_count"], int)


# ===========================================================================
# 3. Anonymizer tests
# ===========================================================================

class TestAnonymizer:
    def test_pseudonymize_record(self, anonymizer):
        """PII fields are hashed, non-PII fields are preserved."""
        record = {
            "target_id": "ble_AA:BB:CC",
            "mac_address": "AA:BB:CC:DD:EE:FF",
            "rssi": -65,
            "device_name": "iPhone 15",
        }
        result, meta = anonymizer.anonymize_record(record)
        assert result["rssi"] == -65  # preserved
        assert result["mac_address"] != "AA:BB:CC:DD:EE:FF"  # hashed
        assert result["device_name"] != "iPhone 15"  # hashed
        assert meta.anonymized_field_count >= 2

    def test_anonymize_none_level(self, anonymizer):
        """NONE level returns data unchanged."""
        record = {"mac_address": "AA:BB:CC:DD:EE:FF"}
        result, meta = anonymizer.anonymize_record(
            record, level=AnonymizationLevel.NONE
        )
        assert result["mac_address"] == "AA:BB:CC:DD:EE:FF"
        assert meta.level == "none"

    def test_redact_level(self, anonymizer):
        """REDACT level replaces PII with [REDACTED]."""
        record = {"mac_address": "AA:BB:CC:DD:EE:FF", "name": "Bob"}
        result, meta = anonymizer.anonymize_record(
            record, level=AnonymizationLevel.REDACT
        )
        assert result["mac_address"] == "[REDACTED]"
        assert result["name"] == "[REDACTED]"

    def test_anonymize_level_removes(self, anonymizer):
        """ANONYMIZE level replaces PII with [REMOVED]."""
        record = {"mac_address": "AA:BB:CC:DD:EE:FF", "rssi": -70}
        result, meta = anonymizer.anonymize_record(
            record, level=AnonymizationLevel.ANONYMIZE
        )
        assert result["mac_address"] == "[REMOVED]"
        assert result["rssi"] == -70

    def test_anonymize_identifier(self, anonymizer):
        """Hash of the same input is deterministic."""
        h1 = anonymizer.anonymize_identifier("test123")
        h2 = anonymizer.anonymize_identifier("test123")
        assert h1 == h2
        assert h1 != "test123"

    def test_different_secrets_different_hashes(self):
        """Different secrets produce different pseudonyms."""
        a1 = Anonymizer(secret="secret-a")
        a2 = Anonymizer(secret="secret-b")
        h1 = a1.anonymize_identifier("target_123")
        h2 = a2.anonymize_identifier("target_123")
        assert h1 != h2

    def test_anonymize_mac(self, anonymizer):
        """MAC pseudonymization produces anon_ prefixed ID."""
        result = anonymizer.anonymize_mac("AA:BB:CC:DD:EE:FF")
        assert result.startswith("anon_")
        assert len(result) == 17  # "anon_" + 12 hex chars

    def test_hash_target_id_preserves_prefix(self, anonymizer):
        """Target ID prefix is preserved, rest is hashed."""
        result = anonymizer.hash_target_id("ble_AA:BB:CC:DD:EE:FF")
        assert result.startswith("ble_anon_")

    def test_hash_target_id_no_prefix(self, anonymizer):
        """Target ID without underscore gets anon_ prefix."""
        result = anonymizer.hash_target_id("nounderscore")
        assert result.startswith("anon_")

    def test_is_pii_field(self, anonymizer):
        assert anonymizer.is_pii_field("mac_address")
        assert anonymizer.is_pii_field("email")
        assert not anonymizer.is_pii_field("rssi")

    def test_add_pii_field(self, anonymizer):
        assert not anonymizer.is_pii_field("custom_id")
        anonymizer.add_pii_field("custom_id")
        assert anonymizer.is_pii_field("custom_id")

    def test_scrub_embedded_mac(self, anonymizer):
        """MACs embedded in free text are pseudonymized."""
        record = {"notes": "Seen device AA:BB:CC:DD:EE:FF near gate"}
        result, meta = anonymizer.anonymize_record(record)
        assert "AA:BB:CC:DD:EE:FF" not in result["notes"]
        assert "anon_" in result["notes"]

    def test_scrub_embedded_ip_redact(self, anonymizer):
        """IPs embedded in free text are redacted."""
        record = {"notes": "From 192.168.1.100"}
        result, _ = anonymizer.anonymize_record(
            record, level=AnonymizationLevel.REDACT
        )
        assert "192.168.1.100" not in result["notes"]
        assert "[IP_REDACTED]" in result["notes"]

    def test_location_anonymization(self):
        """ANONYMIZE level reduces location precision."""
        anon = Anonymizer(location_precision=2)
        record = {"lat": 40.712776, "lng": -74.005974, "rssi": -70}
        result, _ = anon.anonymize_record(
            record, level=AnonymizationLevel.ANONYMIZE
        )
        # lat/lng should be rounded to 2 decimals
        assert result["lat"] == 40.71
        assert result["lng"] == -74.01
        assert result["rssi"] == -70

    def test_stats(self, anonymizer):
        """Stats track processed records and fields."""
        anonymizer.reset_stats()
        anonymizer.anonymize_record({"mac_address": "AA:BB", "rssi": -60})
        anonymizer.anonymize_record({"name": "Bob", "email": "bob@x.com"})
        s = anonymizer.stats
        assert s["records_processed"] == 2
        assert s["fields_anonymized"] >= 3

    def test_extra_pii_fields_constructor(self):
        """Extra PII fields passed at construction are recognized."""
        anon = Anonymizer(extra_pii_fields={"badge_number"})
        assert anon.is_pii_field("badge_number")


# ===========================================================================
# 4. Consent tests
# ===========================================================================

class TestConsentManager:
    def test_grant_consent(self, consent_mgr):
        record = consent_mgr.grant("target_1", "tracking", evidence="web_form")
        assert record.subject_id == "target_1"
        assert record.purpose == "tracking"
        assert record.status == ConsentStatus.GRANTED
        assert record.consent_id  # non-empty

    def test_has_consent_true(self, consent_mgr):
        consent_mgr.grant("target_1", "tracking")
        assert consent_mgr.has_consent("target_1", "tracking")

    def test_has_consent_false_no_record(self, consent_mgr):
        assert not consent_mgr.has_consent("target_1", "tracking")

    def test_withdraw_consent(self, consent_mgr):
        consent_mgr.grant("target_1", "tracking")
        result = consent_mgr.withdraw("target_1", "tracking")
        assert result is not None
        assert result.status == ConsentStatus.WITHDRAWN
        assert result.withdrawn_at > 0
        assert not consent_mgr.has_consent("target_1", "tracking")

    def test_withdraw_nonexistent(self, consent_mgr):
        result = consent_mgr.withdraw("nobody", "tracking")
        assert result is None

    def test_withdraw_all(self, consent_mgr):
        consent_mgr.grant("target_1", "tracking")
        consent_mgr.grant("target_1", "analytics")
        results = consent_mgr.withdraw_all("target_1")
        assert len(results) == 2
        assert not consent_mgr.has_consent("target_1", "tracking")
        assert not consent_mgr.has_consent("target_1", "analytics")

    def test_expired_consent(self, consent_mgr):
        """Consent with past expiry is not active."""
        consent_mgr.grant("target_1", "tracking", expires_at=time.time() - 100)
        assert not consent_mgr.has_consent("target_1", "tracking")

    def test_consent_not_yet_expired(self, consent_mgr):
        """Consent with future expiry is still active."""
        consent_mgr.grant("target_1", "tracking", expires_at=time.time() + 3600)
        assert consent_mgr.has_consent("target_1", "tracking")

    def test_get_all_consents(self, consent_mgr):
        consent_mgr.grant("target_1", "tracking")
        consent_mgr.grant("target_1", "analytics")
        all_c = consent_mgr.get_all_consents("target_1")
        assert len(all_c) == 2

    def test_get_active_consents(self, consent_mgr):
        consent_mgr.grant("target_1", "tracking")
        consent_mgr.grant("target_1", "analytics")
        consent_mgr.withdraw("target_1", "analytics")
        active = consent_mgr.get_active_consents("target_1")
        assert len(active) == 1
        assert active[0].purpose == "tracking"

    def test_list_subjects(self, consent_mgr):
        consent_mgr.grant("alice", "tracking")
        consent_mgr.grant("bob", "tracking")
        subjects = consent_mgr.list_subjects()
        assert subjects == ["alice", "bob"]

    def test_count_by_purpose(self, consent_mgr):
        consent_mgr.grant("a", "tracking")
        consent_mgr.grant("b", "tracking")
        consent_mgr.grant("a", "analytics")
        counts = consent_mgr.count_by_purpose()
        assert counts["tracking"] == 2
        assert counts["analytics"] == 1

    def test_history(self, consent_mgr):
        consent_mgr.grant("a", "tracking")
        consent_mgr.withdraw("a", "tracking")
        assert len(consent_mgr.history) == 2

    def test_consent_record_to_dict(self):
        record = ConsentRecord.create("subj", "purpose", evidence="api")
        d = record.to_dict()
        assert d["subject_id"] == "subj"
        assert d["purpose"] == "purpose"
        assert d["evidence"] == "api"

    def test_export(self, consent_mgr):
        consent_mgr.grant("a", "tracking")
        d = consent_mgr.export()
        assert d["subject_count"] == 1
        assert "a" in d["records"]

    def test_import_records(self, consent_mgr):
        record = ConsentRecord.create("imported", "tracking")
        count = consent_mgr.import_records([record])
        assert count == 1
        assert consent_mgr.has_consent("imported", "tracking")

    def test_clear(self, consent_mgr):
        consent_mgr.grant("a", "tracking")
        consent_mgr.clear()
        assert not consent_mgr.has_consent("a", "tracking")
        assert len(consent_mgr.history) == 0


# ===========================================================================
# 5. DataSubjectRequest tests
# ===========================================================================

class TestSubjectRequestManager:
    def test_submit_access(self, request_mgr):
        req = request_mgr.submit_access("target_123", reason="My data please")
        assert req.request_type == RequestType.ACCESS
        assert req.status == RequestStatus.PENDING
        assert req.is_open

    def test_process_access(self, request_mgr):
        req = request_mgr.submit_access("target_123")
        result = request_mgr.process(req.request_id)
        assert result.status == RequestStatus.COMPLETED
        assert "sightings" in result.response_data
        assert len(result.response_data["sightings"]) == 2

    def test_process_access_no_data(self, request_mgr):
        req = request_mgr.submit_access("unknown_target")
        result = request_mgr.process(req.request_id)
        assert result.status == RequestStatus.COMPLETED
        assert result.response_data["sightings"] == []

    def test_submit_and_process_erasure(self, request_mgr):
        req = request_mgr.submit_erasure("target_123")
        result = request_mgr.process(req.request_id)
        assert result.status == RequestStatus.COMPLETED
        assert result.affected_records == 2
        # Second erasure finds nothing
        req2 = request_mgr.submit_erasure("target_123")
        result2 = request_mgr.process(req2.request_id)
        assert result2.affected_records == 0

    def test_process_nonexistent_raises(self, request_mgr):
        with pytest.raises(KeyError):
            request_mgr.process("fake-id")

    def test_process_completed_raises(self, request_mgr):
        req = request_mgr.submit_access("target_123")
        request_mgr.process(req.request_id)
        with pytest.raises(ValueError, match="already.*cannot process"):
            request_mgr.process(req.request_id)

    def test_deny_request(self, request_mgr):
        req = request_mgr.submit_erasure("target_123")
        denied = request_mgr.deny(req.request_id, denial_reason="Ongoing investigation")
        assert denied.status == RequestStatus.DENIED
        assert denied.denial_reason == "Ongoing investigation"
        assert not denied.is_open

    def test_complete_manual(self, request_mgr):
        req = request_mgr.submit_rectification("target_123", reason="Wrong name")
        request_mgr.process(req.request_id)
        completed = request_mgr.complete(req.request_id, notes="Fixed name", affected_records=1)
        assert completed.status == RequestStatus.COMPLETED
        assert completed.affected_records == 1

    def test_list_requests(self, request_mgr):
        request_mgr.submit_access("a")
        request_mgr.submit_erasure("b")
        all_reqs = request_mgr.list_requests()
        assert len(all_reqs) == 2

    def test_list_requests_filter_subject(self, request_mgr):
        request_mgr.submit_access("a")
        request_mgr.submit_access("b")
        filtered = request_mgr.list_requests(subject_id="a")
        assert len(filtered) == 1

    def test_list_requests_filter_status(self, request_mgr):
        req = request_mgr.submit_access("a")
        request_mgr.process(req.request_id)
        request_mgr.submit_access("b")
        pending = request_mgr.list_requests(status=RequestStatus.PENDING)
        assert len(pending) == 1

    def test_overdue_detection(self, request_mgr):
        """Requests older than 30 days are flagged as overdue."""
        req = request_mgr.submit_access("target_123")
        # Not overdue now
        assert len(request_mgr.overdue()) == 0
        # Fake it by checking with a future timestamp
        future = time.time() + (31 * 86400)
        overdue = request_mgr.overdue(now=future)
        assert len(overdue) == 1
        assert overdue[0].request_id == req.request_id

    def test_stats(self, request_mgr):
        request_mgr.submit_access("a")
        request_mgr.submit_erasure("b")
        s = request_mgr.stats()
        assert s["total"] == 2
        assert s["by_type"]["access"] == 1
        assert s["by_type"]["erasure"] == 1

    def test_portability_request(self, request_mgr):
        req = request_mgr.submit_portability("target_123", reason="Switching provider")
        result = request_mgr.process(req.request_id)
        assert result.status == RequestStatus.COMPLETED
        assert "sightings" in result.response_data

    def test_restriction_request(self, request_mgr):
        req = request_mgr.submit_restriction("target_123", reason="Disputed accuracy")
        result = request_mgr.process(req.request_id)
        # Restriction stays in PROCESSING (needs manual intervention)
        assert result.status == RequestStatus.PROCESSING

    def test_response_time(self, request_mgr):
        req = request_mgr.submit_access("target_123")
        assert req.response_time_seconds is None  # not yet completed
        result = request_mgr.process(req.request_id)
        assert result.response_time_seconds is not None
        assert result.response_time_seconds >= 0

    def test_request_to_dict(self):
        req = DataSubjectRequest.create("subj", "access", reason="test")
        d = req.to_dict()
        assert d["subject_id"] == "subj"
        assert d["request_type"] == "access"
        assert d["status"] == "pending"


# ===========================================================================
# 6. Privacy zone tests
# ===========================================================================

class TestPrivacyZone:
    def test_point_inside_zone(self, zone_mgr):
        """Point inside the polygon is detected."""
        result = zone_mgr.check_point(40.0, -74.0)
        assert result.suppressed
        assert result.zone_count == 1

    def test_point_outside_zone(self, zone_mgr):
        """Point outside is not suppressed."""
        result = zone_mgr.check_point(41.0, -74.0)
        assert not result.suppressed
        assert result.zone_count == 0

    def test_zone_suppression_level(self, zone_mgr):
        """Check result reports the correct suppression level."""
        result = zone_mgr.check_point(40.0, -74.0)
        assert result.suppression_level == "full"

    def test_disabled_zone_ignored(self, zone_mgr):
        """Disabled zones don't suppress."""
        zones = zone_mgr.list_zones()
        zone_mgr.update_zone(zones[0].zone_id, enabled=False)
        result = zone_mgr.check_point(40.0, -74.0)
        assert not result.suppressed

    def test_expired_zone_ignored(self):
        """Expired zones don't suppress."""
        mgr = PrivacyZoneManager()
        polygon = [(39.99, -74.01), (39.99, -73.99), (40.01, -73.99), (40.01, -74.01)]
        mgr.add_zone("Expired", polygon, expires_at=time.time() - 100)
        result = mgr.check_point(40.0, -74.0)
        assert not result.suppressed

    def test_sensor_type_filtering(self):
        """Zone only affecting 'camera' doesn't suppress 'ble'."""
        mgr = PrivacyZoneManager()
        polygon = [(39.99, -74.01), (39.99, -73.99), (40.01, -73.99), (40.01, -74.01)]
        mgr.add_zone("Camera Only", polygon, affected_sensors=["camera"])
        # BLE not affected
        result = mgr.check_point(40.0, -74.0, sensor_type="ble")
        assert not result.suppressed
        # Camera is affected
        result = mgr.check_point(40.0, -74.0, sensor_type="camera")
        assert result.suppressed

    def test_all_sensors_when_empty(self):
        """Empty affected_sensors means all sensors are affected."""
        mgr = PrivacyZoneManager()
        polygon = [(39.99, -74.01), (39.99, -73.99), (40.01, -73.99), (40.01, -74.01)]
        mgr.add_zone("All", polygon)
        result = mgr.check_point(40.0, -74.0, sensor_type="ble")
        assert result.suppressed

    def test_remove_zone(self, zone_mgr):
        zones = zone_mgr.list_zones()
        assert zone_mgr.remove_zone(zones[0].zone_id)
        result = zone_mgr.check_point(40.0, -74.0)
        assert not result.suppressed

    def test_list_zones_active_only(self, zone_mgr):
        zones = zone_mgr.list_zones()
        zone_mgr.update_zone(zones[0].zone_id, enabled=False)
        assert len(zone_mgr.list_zones(active_only=True)) == 0
        assert len(zone_mgr.list_zones(active_only=False)) == 1

    def test_check_target(self, zone_mgr):
        result = zone_mgr.check_target("ble_123", 40.0, -74.0)
        assert result.suppressed
        assert result.target_id == "ble_123"

    def test_zone_to_dict(self):
        zone = PrivacyZone.create(
            "Test", [(0, 0), (1, 0), (1, 1)], reason="privacy"
        )
        d = zone.to_dict()
        assert d["name"] == "Test"
        assert d["reason"] == "privacy"
        assert len(d["polygon"]) == 3

    def test_zone_check_result_to_dict(self, zone_mgr):
        result = zone_mgr.check_point(40.0, -74.0)
        d = result.to_dict()
        assert d["suppressed"] is True
        assert d["zone_count"] == 1
        assert len(d["zone_names"]) == 1

    def test_multiple_overlapping_zones(self):
        """Highest suppression level wins among overlapping zones."""
        mgr = PrivacyZoneManager()
        polygon = [(39.99, -74.01), (39.99, -73.99), (40.01, -73.99), (40.01, -74.01)]
        mgr.add_zone("Anon", polygon, suppression="anonymize")
        mgr.add_zone("Full", polygon, suppression="full")
        result = mgr.check_point(40.0, -74.0)
        assert result.zone_count == 2
        assert result.suppression_level == "full"

    def test_export(self, zone_mgr):
        d = zone_mgr.export()
        assert d["total"] == 1
        assert d["active"] == 1

    def test_privacy_zone_contains_point_degenerate(self):
        """Polygon with fewer than 3 vertices never contains a point."""
        zone = PrivacyZone(polygon=[(0, 0), (1, 1)])
        assert not zone.contains_point(0.5, 0.5)

    def test_update_zone_returns_none_for_missing(self, zone_mgr):
        assert zone_mgr.update_zone("nonexistent", name="X") is None

    def test_get_zone(self, zone_mgr):
        zones = zone_mgr.list_zones()
        found = zone_mgr.get_zone(zones[0].zone_id)
        assert found is not None
        assert found.name == "Test Zone"


# ===========================================================================
# 7. Integration / cross-module tests
# ===========================================================================

class TestPrivacyIntegration:
    def test_consent_before_processing(self, consent_mgr, anonymizer):
        """Without consent, data should be anonymized."""
        target_id = "ble_AA:BB:CC"
        record = {"mac_address": "AA:BB:CC:DD:EE:FF", "rssi": -60}

        if not consent_mgr.has_consent(target_id, "tracking"):
            result, _ = anonymizer.anonymize_record(
                record, level=AnonymizationLevel.ANONYMIZE
            )
            assert result["mac_address"] == "[REMOVED]"

    def test_full_privacy_workflow(self, consent_mgr, anonymizer, zone_mgr, request_mgr):
        """End-to-end: zone check -> consent check -> anonymize -> subject request."""
        subject = "target_123"
        lat, lng = 40.0, -74.0

        # 1. Check privacy zone
        zone_result = zone_mgr.check_point(lat, lng)
        assert zone_result.suppressed  # Inside zone

        # 2. No consent
        assert not consent_mgr.has_consent(subject, "tracking")

        # 3. Anonymize
        data = {"mac_address": "AA:BB:CC", "lat": lat, "lng": lng}
        anon_data, _ = anonymizer.anonymize_record(
            data, level=AnonymizationLevel.ANONYMIZE
        )
        assert anon_data["mac_address"] == "[REMOVED]"

        # 4. Subject requests their data
        req = request_mgr.submit_access(subject)
        result = request_mgr.process(req.request_id)
        assert result.status == RequestStatus.COMPLETED

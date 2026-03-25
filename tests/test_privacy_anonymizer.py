# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.privacy.anonymizer — PII anonymization."""

from tritium_lib.privacy.anonymizer import (
    AnonymizationLevel,
    AnonymizationResult,
    Anonymizer,
    PII_FIELDS,
    LOCATION_FIELDS,
    MAC_RE,
    IPV4_RE,
)


class TestAnonymizationLevel:
    def test_values(self):
        assert AnonymizationLevel.NONE == "none"
        assert AnonymizationLevel.PSEUDONYMIZE == "pseudonymize"
        assert AnonymizationLevel.ANONYMIZE == "anonymize"
        assert AnonymizationLevel.REDACT == "redact"


class TestAnonymizationResult:
    def test_fields_affected(self):
        r = AnonymizationResult(
            original_field_count=10,
            anonymized_field_count=3,
            level="pseudonymize",
        )
        assert r.fields_affected == 3

    def test_to_dict(self):
        r = AnonymizationResult(
            original_field_count=5,
            anonymized_field_count=2,
            level="redact",
            timestamp=12345.0,
        )
        d = r.to_dict()
        assert d["original_field_count"] == 5
        assert d["anonymized_field_count"] == 2
        assert d["level"] == "redact"
        assert d["timestamp"] == 12345.0


class TestRegexPatterns:
    def test_mac_regex(self):
        assert MAC_RE.search("AA:BB:CC:DD:EE:FF")
        assert MAC_RE.search("aa:bb:cc:dd:ee:ff")
        assert MAC_RE.search("AA-BB-CC-DD-EE-FF")
        assert not MAC_RE.search("not_a_mac")

    def test_ipv4_regex(self):
        assert IPV4_RE.search("192.168.1.1")
        assert IPV4_RE.search("10.0.0.1")
        assert not IPV4_RE.search("just text")


class TestAnonymizer:
    def test_none_level_passthrough(self):
        anon = Anonymizer()
        record = {"mac_address": "AA:BB:CC:DD:EE:FF", "name": "phone"}
        out, result = anon.anonymize_record(record, level=AnonymizationLevel.NONE)
        assert out["mac_address"] == "AA:BB:CC:DD:EE:FF"
        assert out["name"] == "phone"
        assert result.anonymized_field_count == 0

    def test_pseudonymize_pii(self):
        anon = Anonymizer(secret="test_secret")
        record = {"mac_address": "AA:BB:CC:DD:EE:FF", "rssi": -55}
        out, result = anon.anonymize_record(
            record, level=AnonymizationLevel.PSEUDONYMIZE
        )
        assert out["mac_address"] != "AA:BB:CC:DD:EE:FF"
        assert len(out["mac_address"]) == 16  # truncated hash
        assert out["rssi"] == -55
        assert result.anonymized_field_count == 1

    def test_redact_pii(self):
        anon = Anonymizer()
        record = {"mac_address": "AA:BB:CC:DD:EE:FF", "email": "test@test.com"}
        out, result = anon.anonymize_record(
            record, level=AnonymizationLevel.REDACT
        )
        assert out["mac_address"] == "[REDACTED]"
        assert out["email"] == "[REDACTED]"
        assert result.anonymized_field_count == 2

    def test_anonymize_level(self):
        anon = Anonymizer()
        record = {"mac_address": "AA:BB:CC", "name": "device_1"}
        out, result = anon.anonymize_record(
            record, level=AnonymizationLevel.ANONYMIZE
        )
        assert out["mac_address"] == "[REMOVED]"
        assert out["name"] == "[REMOVED]"

    def test_anonymize_location_generalization(self):
        anon = Anonymizer(location_precision=2)
        record = {"lat": 38.8976633, "lng": -77.0365739}
        out, result = anon.anonymize_record(
            record, level=AnonymizationLevel.ANONYMIZE
        )
        assert out["lat"] == 38.90
        assert out["lng"] == -77.04

    def test_scrub_embedded_mac(self):
        anon = Anonymizer()
        record = {"notes": "Device seen with MAC AA:BB:CC:DD:EE:FF nearby"}
        out, result = anon.anonymize_record(
            record, level=AnonymizationLevel.REDACT
        )
        assert "[MAC_REDACTED]" in out["notes"]
        assert "AA:BB:CC:DD:EE:FF" not in out["notes"]

    def test_scrub_embedded_ip(self):
        anon = Anonymizer()
        record = {"notes": "Accessed from 192.168.1.1 at 10:00"}
        out, result = anon.anonymize_record(
            record, level=AnonymizationLevel.REDACT
        )
        assert "[IP_REDACTED]" in out["notes"]

    def test_anonymize_identifier(self):
        anon = Anonymizer(secret="my_secret")
        h1 = anon.anonymize_identifier("AA:BB:CC:DD:EE:FF")
        h2 = anon.anonymize_identifier("AA:BB:CC:DD:EE:FF")
        assert h1 == h2  # deterministic
        assert h1 != "AA:BB:CC:DD:EE:FF"

    def test_anonymize_mac(self):
        anon = Anonymizer()
        result = anon.anonymize_mac("aa:bb:cc:dd:ee:ff")
        assert result.startswith("anon_")
        assert len(result) == 17  # "anon_" + 12 chars

    def test_hash_target_id_with_prefix(self):
        anon = Anonymizer()
        result = anon.hash_target_id("ble_AA:BB:CC:DD:EE:FF")
        assert result.startswith("ble_anon_")

    def test_hash_target_id_no_prefix(self):
        anon = Anonymizer()
        result = anon.hash_target_id("device123")
        assert result.startswith("anon_")

    def test_is_pii_field(self):
        anon = Anonymizer()
        assert anon.is_pii_field("mac_address") is True
        assert anon.is_pii_field("rssi") is False
        assert anon.is_pii_field("email") is True

    def test_add_pii_field(self):
        anon = Anonymizer()
        assert anon.is_pii_field("custom_field") is False
        anon.add_pii_field("custom_field")
        assert anon.is_pii_field("custom_field") is True

    def test_extra_pii_fields(self):
        anon = Anonymizer(extra_pii_fields={"badge_id"})
        assert anon.is_pii_field("badge_id") is True

    def test_stats(self):
        anon = Anonymizer()
        anon.anonymize_record({"mac_address": "AA:BB"}, level=AnonymizationLevel.REDACT)
        anon.anonymize_record({"name": "test"}, level=AnonymizationLevel.REDACT)
        stats = anon.stats
        assert stats["records_processed"] == 2
        assert stats["fields_anonymized"] == 2

    def test_reset_stats(self):
        anon = Anonymizer()
        anon.anonymize_record({"mac_address": "AA:BB"}, level=AnonymizationLevel.REDACT)
        anon.reset_stats()
        assert anon.stats["records_processed"] == 0
        assert anon.stats["fields_anonymized"] == 0

    def test_original_not_modified(self):
        anon = Anonymizer()
        record = {"mac_address": "AA:BB:CC:DD:EE:FF"}
        out, _ = anon.anonymize_record(record, level=AnonymizationLevel.REDACT)
        assert record["mac_address"] == "AA:BB:CC:DD:EE:FF"

    def test_different_secrets_different_hashes(self):
        anon1 = Anonymizer(secret="secret_a")
        anon2 = Anonymizer(secret="secret_b")
        h1 = anon1.anonymize_identifier("same_input")
        h2 = anon2.anonymize_identifier("same_input")
        assert h1 != h2

    def test_pseudonymize_mac_in_text(self):
        anon = Anonymizer()
        record = {"notes": "Device AA:BB:CC:DD:EE:FF spotted"}
        out, _ = anon.anonymize_record(
            record, level=AnonymizationLevel.PSEUDONYMIZE
        )
        assert "AA:BB:CC:DD:EE:FF" not in out["notes"]
        assert "anon_" in out["notes"]

    def test_anonymize_none_pii_value(self):
        anon = Anonymizer()
        record = {"mac_address": None}
        out, _ = anon.anonymize_record(
            record, level=AnonymizationLevel.PSEUDONYMIZE
        )
        assert out["mac_address"] is None

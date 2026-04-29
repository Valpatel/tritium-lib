# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the SQLite-backed DossierStore."""

import time

import pytest

from tritium_lib.store.dossiers import DossierStore


@pytest.fixture
def store():
    """In-memory DossierStore for testing."""
    s = DossierStore(":memory:")
    yield s
    s.close()


# ------------------------------------------------------------------
# CRUD basics
# ------------------------------------------------------------------


class TestCreateAndGet:
    def test_create_dossier(self, store: DossierStore):
        did = store.create_dossier(
            "Alpha Target",
            entity_type="person",
            identifiers={"mac": "AA:BB:CC:DD:EE:FF"},
            alliance="hostile",
            threat_level="high",
            tags=["armed", "mobile"],
            notes=["First spotted near building 7"],
        )
        assert did  # non-empty UUID string

        dossier = store.get_dossier(did)
        assert dossier is not None
        assert dossier["name"] == "Alpha Target"
        assert dossier["entity_type"] == "person"
        assert dossier["alliance"] == "hostile"
        assert dossier["threat_level"] == "high"
        assert dossier["identifiers"] == {"mac": "AA:BB:CC:DD:EE:FF"}
        assert dossier["tags"] == ["armed", "mobile"]
        assert dossier["notes"] == ["First spotted near building 7"]
        assert dossier["signals"] == []
        assert dossier["enrichments"] == []

    def test_create_defaults(self, store: DossierStore):
        did = store.create_dossier("Minimal")
        dossier = store.get_dossier(did)
        assert dossier["entity_type"] == "unknown"
        assert dossier["alliance"] == "unknown"
        assert dossier["threat_level"] == "none"
        assert dossier["confidence"] == 0.0
        assert dossier["identifiers"] == {}
        assert dossier["tags"] == []
        assert dossier["notes"] == []

    def test_get_nonexistent(self, store: DossierStore):
        assert store.get_dossier("no-such-id") is None

    def test_delete_dossier(self, store: DossierStore):
        did = store.create_dossier("Delete Me")
        store.add_signal(did, "ble", "mac_sighting")
        store.add_enrichment(did, "oui", "manufacturer")
        assert store.delete_dossier(did) is True
        assert store.get_dossier(did) is None

    def test_delete_nonexistent(self, store: DossierStore):
        assert store.delete_dossier("nope") is False


# ------------------------------------------------------------------
# Signal accumulation
# ------------------------------------------------------------------


class TestSignals:
    def test_add_signal(self, store: DossierStore):
        did = store.create_dossier("Signal Test")
        sid = store.add_signal(
            did, "ble", "mac_sighting",
            data={"rssi": -42, "mac": "AA:BB:CC:DD:EE:FF"},
            position_x=10.0, position_y=20.0,
            confidence=0.8,
        )
        assert sid  # non-empty UUID

        dossier = store.get_dossier(did)
        assert len(dossier["signals"]) == 1
        sig = dossier["signals"][0]
        assert sig["signal_id"] == sid
        assert sig["source"] == "ble"
        assert sig["signal_type"] == "mac_sighting"
        assert sig["data"] == {"rssi": -42, "mac": "AA:BB:CC:DD:EE:FF"}
        assert sig["position_x"] == 10.0
        assert sig["position_y"] == 20.0
        assert sig["confidence"] == 0.8

    def test_multiple_signals(self, store: DossierStore):
        did = store.create_dossier("Multi Signal")
        store.add_signal(did, "ble", "mac_sighting", timestamp=100.0)
        store.add_signal(did, "wifi", "probe_request", timestamp=200.0)
        store.add_signal(did, "camera", "visual_detection", timestamp=300.0)

        dossier = store.get_dossier(did)
        assert len(dossier["signals"]) == 3
        # Newest first
        assert dossier["signals"][0]["source"] == "camera"
        assert dossier["signals"][2]["source"] == "ble"

    def test_signal_updates_last_seen(self, store: DossierStore):
        ts1 = time.time() - 100
        did = store.create_dossier("Timestamp Test", timestamp=ts1)
        ts2 = time.time()
        store.add_signal(did, "ble", "sighting", timestamp=ts2)

        dossier = store.get_dossier(did)
        assert dossier["last_seen"] == pytest.approx(ts2, abs=1)
        assert dossier["first_seen"] == pytest.approx(ts1, abs=1)


# ------------------------------------------------------------------
# Enrichments
# ------------------------------------------------------------------


class TestEnrichments:
    def test_add_enrichment(self, store: DossierStore):
        did = store.create_dossier("Enrich Test")
        eid = store.add_enrichment(
            did, "oui_lookup", "manufacturer",
            data={"vendor": "Apple Inc.", "oui": "AA:BB:CC"},
        )
        assert eid is not None

        dossier = store.get_dossier(did)
        assert len(dossier["enrichments"]) == 1
        enr = dossier["enrichments"][0]
        assert enr["provider"] == "oui_lookup"
        assert enr["enrichment_type"] == "manufacturer"
        assert enr["data"]["vendor"] == "Apple Inc."


# ------------------------------------------------------------------
# Identifier lookup
# ------------------------------------------------------------------


class TestFindByIdentifier:
    def test_find_by_mac(self, store: DossierStore):
        did = store.create_dossier(
            "MAC Target",
            identifiers={"mac": "AA:BB:CC:DD:EE:FF"},
        )
        result = store.find_by_identifier("mac", "AA:BB:CC:DD:EE:FF")
        assert result is not None
        assert result["dossier_id"] == did
        assert result["name"] == "MAC Target"

    def test_find_not_found(self, store: DossierStore):
        store.create_dossier("Other", identifiers={"mac": "11:22:33:44:55:66"})
        assert store.find_by_identifier("mac", "FF:FF:FF:FF:FF:FF") is None

    def test_find_by_different_types(self, store: DossierStore):
        did = store.create_dossier(
            "Multi ID",
            identifiers={"mac": "AA:BB:CC:DD:EE:FF", "ssid": "CoolNetwork"},
        )
        assert store.find_by_identifier("ssid", "CoolNetwork") is not None
        assert store.find_by_identifier("mac", "AA:BB:CC:DD:EE:FF") is not None


# ------------------------------------------------------------------
# Full-text search
# ------------------------------------------------------------------


class TestSearch:
    def test_search_by_name(self, store: DossierStore):
        store.create_dossier("Red Falcon", entity_type="person")
        store.create_dossier("Blue Eagle", entity_type="vehicle")
        results = store.search("Falcon")
        assert len(results) == 1
        assert results[0]["name"] == "Red Falcon"

    def test_search_by_entity_type(self, store: DossierStore):
        store.create_dossier("Target A", entity_type="vehicle")
        store.create_dossier("Target B", entity_type="person")
        results = store.search("vehicle")
        assert len(results) == 1
        assert results[0]["name"] == "Target A"

    def test_search_by_identifier_content(self, store: DossierStore):
        store.create_dossier(
            "MAC Device",
            identifiers={"mac": "AA:BB:CC:DD:EE:FF"},
        )
        # FTS indexes the JSON string of identifiers
        results = store.search("AA:BB:CC")
        assert len(results) >= 1

    def test_search_empty_query(self, store: DossierStore):
        store.create_dossier("Test")
        assert store.search("") == []
        assert store.search("   ") == []

    def test_search_no_results(self, store: DossierStore):
        store.create_dossier("Alpha")
        assert store.search("zzzznotfound") == []


# ------------------------------------------------------------------
# Recent listing
# ------------------------------------------------------------------


class TestGetRecent:
    def test_get_recent(self, store: DossierStore):
        store.create_dossier("Old", timestamp=100.0)
        store.create_dossier("New", timestamp=200.0)
        results = store.get_recent(limit=10)
        assert len(results) == 2
        assert results[0]["name"] == "New"
        assert results[1]["name"] == "Old"

    def test_get_recent_with_since(self, store: DossierStore):
        store.create_dossier("Old", timestamp=100.0)
        store.create_dossier("New", timestamp=200.0)
        results = store.get_recent(limit=10, since=150.0)
        assert len(results) == 1
        assert results[0]["name"] == "New"

    def test_get_recent_limit(self, store: DossierStore):
        for i in range(10):
            store.create_dossier(f"Target {i}", timestamp=float(i))
        results = store.get_recent(limit=3)
        assert len(results) == 3


# ------------------------------------------------------------------
# Threat level update
# ------------------------------------------------------------------


class TestUpdateThreatLevel:
    def test_update_threat_level(self, store: DossierStore):
        did = store.create_dossier("Threat Test", threat_level="low")
        assert store.update_threat_level(did, "critical") is True
        dossier = store.get_dossier(did)
        assert dossier["threat_level"] == "critical"

    def test_update_nonexistent(self, store: DossierStore):
        assert store.update_threat_level("nope", "high") is False


# ------------------------------------------------------------------
# Merge
# ------------------------------------------------------------------


class TestMerge:
    def test_merge_dossiers(self, store: DossierStore):
        primary_id = store.create_dossier(
            "Primary",
            identifiers={"mac": "AA:BB:CC:DD:EE:FF"},
            tags=["armed"],
            notes=["Note from primary"],
            timestamp=100.0,
        )
        secondary_id = store.create_dossier(
            "Secondary",
            identifiers={"ssid": "CoolNetwork", "mac": "11:22:33:44:55:66"},
            tags=["mobile", "armed"],
            notes=["Note from secondary"],
            timestamp=50.0,
        )
        # Add signals/enrichments to secondary
        store.add_signal(secondary_id, "wifi", "probe", timestamp=75.0)
        store.add_enrichment(secondary_id, "wigle", "location")

        assert store.merge_dossiers(primary_id, secondary_id) is True

        # Secondary should be gone
        assert store.get_dossier(secondary_id) is None

        # Primary should have merged data
        merged = store.get_dossier(primary_id)
        assert merged is not None
        # Primary MAC wins over secondary MAC
        assert merged["identifiers"]["mac"] == "AA:BB:CC:DD:EE:FF"
        # Secondary SSID is inherited
        assert merged["identifiers"]["ssid"] == "CoolNetwork"
        # Tags deduplicated
        assert "armed" in merged["tags"]
        assert "mobile" in merged["tags"]
        assert len(merged["tags"]) == 2  # no duplicate "armed"
        # Notes concatenated
        assert len(merged["notes"]) == 2
        # Earliest first_seen
        assert merged["first_seen"] == pytest.approx(50.0, abs=1)
        # Signals and enrichments moved
        assert len(merged["signals"]) == 1
        assert merged["signals"][0]["source"] == "wifi"
        assert len(merged["enrichments"]) == 1
        assert merged["enrichments"][0]["provider"] == "wigle"

    def test_merge_nonexistent(self, store: DossierStore):
        did = store.create_dossier("Solo")
        assert store.merge_dossiers(did, "nope") is False
        assert store.merge_dossiers("nope", did) is False


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------


class TestThreadSafety:
    """Verify concurrent writes don't corrupt the DossierStore."""

    def test_concurrent_creates(self, store: DossierStore):
        import threading

        errors: list[Exception] = []
        num_threads = 5
        writes_per_thread = 10
        dossier_ids: list[str] = []
        ids_lock = threading.Lock()

        def writer(thread_id: int):
            try:
                for i in range(writes_per_thread):
                    did = store.create_dossier(
                        f"Target-{thread_id}-{i}",
                        entity_type="person",
                        tags=[f"thread-{thread_id}"],
                    )
                    with ids_lock:
                        dossier_ids.append(did)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent creates: {errors}"
        assert len(dossier_ids) == num_threads * writes_per_thread

    def test_concurrent_signals(self, store: DossierStore):
        """Multiple threads adding signals to the same dossier."""
        import threading

        did = store.create_dossier("Shared Dossier")
        errors: list[Exception] = []

        def writer(thread_id: int):
            try:
                for i in range(10):
                    store.add_signal(
                        did,
                        source=f"source-{thread_id}",
                        signal_type="sighting",
                        data={"i": i},
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(t,))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent signals: {errors}"
        dossier = store.get_dossier(did)
        assert len(dossier["signals"]) == 50


# ------------------------------------------------------------------
# Update JSON fields
# ------------------------------------------------------------------


class TestUpdateJsonField:
    def test_update_tags(self, store: DossierStore):
        did = store.create_dossier("Tag Test", tags=["old"])
        assert store._update_json_field(did, "tags", ["new", "updated"]) is True
        dossier = store.get_dossier(did)
        assert dossier["tags"] == ["new", "updated"]

    def test_update_notes(self, store: DossierStore):
        did = store.create_dossier("Notes Test")
        store._update_json_field(did, "notes", ["First note", "Second note"])
        dossier = store.get_dossier(did)
        assert dossier["notes"] == ["First note", "Second note"]

    def test_update_identifiers(self, store: DossierStore):
        did = store.create_dossier("ID Test")
        store._update_json_field(did, "identifiers", {"mac": "FF:FF:FF"})
        dossier = store.get_dossier(did)
        assert dossier["identifiers"] == {"mac": "FF:FF:FF"}

    def test_update_invalid_field_raises(self, store: DossierStore):
        did = store.create_dossier("Bad Field")
        with pytest.raises(ValueError, match="Cannot update field"):
            store._update_json_field(did, "name", "evil")

    def test_update_nonexistent_dossier(self, store: DossierStore):
        assert store._update_json_field("nope", "tags", []) is False


# ------------------------------------------------------------------
# Close and reopen
# ------------------------------------------------------------------


class TestCloseAndReopen:
    def test_file_backed_persistence(self):
        import tempfile
        import os

        tmpfile = os.path.join(tempfile.mkdtemp(), "dossier_test.db")
        s = DossierStore(tmpfile)
        did = s.create_dossier("Persisted", entity_type="vehicle")
        s.close()

        s2 = DossierStore(tmpfile)
        dossier = s2.get_dossier(did)
        assert dossier is not None
        assert dossier["name"] == "Persisted"
        assert dossier["entity_type"] == "vehicle"
        s2.close()


# ------------------------------------------------------------------
# Retention policy (Gap-fix C M-8)
# ------------------------------------------------------------------


class TestRetention:
    """Periodic prune drops old signals; pinned dossiers are exempt."""

    def test_pinned_default_false(self, store: DossierStore):
        did = store.create_dossier("New", entity_type="device")
        dossier = store.get_dossier(did)
        assert dossier["pinned"] is False

    def test_set_pinned(self, store: DossierStore):
        did = store.create_dossier("Pin Me", entity_type="person")
        assert store.set_pinned(did, True) is True
        dossier = store.get_dossier(did)
        assert dossier["pinned"] is True

        assert store.set_pinned(did, False) is True
        dossier = store.get_dossier(did)
        assert dossier["pinned"] is False

    def test_set_pinned_nonexistent(self, store: DossierStore):
        assert store.set_pinned("does-not-exist", True) is False

    def test_prune_drops_old_signals(self, store: DossierStore):
        """Signals older than max_signal_age_s are dropped from
        unpinned dossiers."""
        did = store.create_dossier("Old", entity_type="person")
        now = time.time()
        # Add 3 signals: one ancient, one borderline, one fresh.
        store.add_signal(did, "ble", "presence", {"rssi": -60},
                         timestamp=now - 60 * 86400.0)   # 60 days old
        store.add_signal(did, "ble", "presence", {"rssi": -70},
                         timestamp=now - 31 * 86400.0)   # 31 days old
        store.add_signal(did, "ble", "presence", {"rssi": -80},
                         timestamp=now - 1.0)            # fresh

        stats = store.prune(now=now)
        assert stats["aged_out"] == 2

        dossier = store.get_dossier(did)
        assert len(dossier["signals"]) == 1
        assert dossier["signals"][0]["data"]["rssi"] == -80

    def test_prune_skips_pinned_dossier(self, store: DossierStore):
        """Pinned dossiers retain their full history."""
        did = store.create_dossier("Pinned", entity_type="person")
        store.set_pinned(did, True)

        now = time.time()
        for i in range(5):
            store.add_signal(did, "ble", "presence", {"i": i},
                             timestamp=now - 100 * 86400.0)  # all very old

        stats = store.prune(now=now)
        assert stats["aged_out"] == 0

        dossier = store.get_dossier(did)
        assert len(dossier["signals"]) == 5

    def test_prune_caps_signal_count(self, store: DossierStore):
        """Per-dossier signal cap drops the oldest first."""
        did = store.create_dossier("Cap Me", entity_type="person")
        now = time.time()
        for i in range(15):
            store.add_signal(did, "ble", "presence", {"i": i},
                             timestamp=now - 60.0 + i)  # all recent

        stats = store.prune(
            now=now,
            max_signals_per_dossier=10,
        )
        assert stats["capped_out"] == 5

        dossier = store.get_dossier(did)
        assert len(dossier["signals"]) == 10
        # Oldest 5 (i=0..4) should be gone; i=5..14 should remain.
        retained = sorted(s["data"]["i"] for s in dossier["signals"])
        assert retained == list(range(5, 15))

    def test_prune_caps_apply_to_pinned(self, store: DossierStore):
        """The per-dossier cap applies even to pinned dossiers — only
        the age-based prune is skipped."""
        did = store.create_dossier("Pinned & Capped", entity_type="person")
        store.set_pinned(did, True)
        now = time.time()
        for i in range(12):
            store.add_signal(did, "ble", "presence", {"i": i},
                             timestamp=now - 60.0 + i)

        stats = store.prune(
            now=now,
            max_signals_per_dossier=10,
        )
        assert stats["capped_out"] == 2

        dossier = store.get_dossier(did)
        assert len(dossier["signals"]) == 10

    def test_prune_empty_store_no_error(self, store: DossierStore):
        stats = store.prune()
        assert stats == {"aged_out": 0, "capped_out": 0}

    def test_prune_disabled_with_zero_age(self, store: DossierStore):
        """max_signal_age_s=0 means everything counts as 'too old' for
        unpinned dossiers — sanity check the math."""
        did = store.create_dossier("Recent", entity_type="person")
        now = time.time()
        store.add_signal(did, "ble", "presence", {}, timestamp=now)

        # Age threshold of 0 seconds drops anything older than 'now'.
        # The signal at exactly 'now' is NOT < cutoff (cutoff=now), so it
        # should survive.
        stats = store.prune(max_signal_age_s=0.0, now=now)
        assert stats["aged_out"] == 0

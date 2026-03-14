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

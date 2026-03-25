# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for DossierStore — CRUD, signals, enrichments, search, merge."""

import time

import pytest

from tritium_lib.store.dossiers import DossierStore


@pytest.fixture
def store():
    """Create an in-memory DossierStore for testing."""
    s = DossierStore(":memory:")
    yield s
    s.close()


# ── Dossier CRUD ────────────────────────────────────────────────────

class TestDossierCRUD:
    """Basic create/read/delete operations."""

    def test_create_returns_id(self, store):
        did = store.create_dossier("John Doe", entity_type="person")
        assert isinstance(did, str)
        assert len(did) > 0

    def test_get_dossier(self, store):
        did = store.create_dossier("Vehicle X", entity_type="vehicle")
        d = store.get_dossier(did)
        assert d is not None
        assert d["name"] == "Vehicle X"
        assert d["entity_type"] == "vehicle"
        assert d["dossier_id"] == did

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_dossier("fake-id") is None

    def test_create_with_all_fields(self, store):
        did = store.create_dossier(
            "Target Alpha",
            entity_type="person",
            identifiers={"mac": "AA:BB:CC:DD:EE:FF"},
            alliance="hostile",
            threat_level="high",
            confidence=0.85,
            tags=["armed", "surveillance"],
            notes=["Spotted near east gate"],
            timestamp=1000.0,
        )
        d = store.get_dossier(did)
        assert d["alliance"] == "hostile"
        assert d["threat_level"] == "high"
        assert d["confidence"] == pytest.approx(0.85)
        assert d["identifiers"]["mac"] == "AA:BB:CC:DD:EE:FF"
        assert "armed" in d["tags"]
        assert "Spotted near east gate" in d["notes"]
        assert d["first_seen"] == 1000.0
        assert d["last_seen"] == 1000.0

    def test_delete_dossier(self, store):
        did = store.create_dossier("To Delete")
        assert store.delete_dossier(did) is True
        assert store.get_dossier(did) is None

    def test_delete_nonexistent(self, store):
        assert store.delete_dossier("fake") is False

    def test_delete_cascades_signals(self, store):
        did = store.create_dossier("Target")
        store.add_signal(did, "ble", "rssi", {"value": -65})
        store.delete_dossier(did)
        assert store.get_dossier(did) is None

    def test_default_values(self, store):
        did = store.create_dossier("Minimal")
        d = store.get_dossier(did)
        assert d["entity_type"] == "unknown"
        assert d["alliance"] == "unknown"
        assert d["threat_level"] == "none"
        assert d["confidence"] == pytest.approx(0.0)
        assert d["identifiers"] == {}
        assert d["tags"] == []
        assert d["notes"] == []


# ── Signals ─────────────────────────────────────────────────────────

class TestDossierSignals:
    """Tests for signal recording and retrieval."""

    def test_add_signal_returns_id(self, store):
        did = store.create_dossier("Target")
        sid = store.add_signal(did, "ble", "rssi")
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_signals_in_dossier(self, store):
        did = store.create_dossier("Target")
        store.add_signal(did, "ble", "rssi", {"value": -65})
        store.add_signal(did, "camera", "detection", {"class": "person"})
        d = store.get_dossier(did)
        assert len(d["signals"]) == 2

    def test_signal_data_parsed(self, store):
        did = store.create_dossier("Target")
        store.add_signal(did, "wifi", "probe", {"ssid": "MyNetwork", "rssi": -45})
        d = store.get_dossier(did)
        sig = d["signals"][0]
        assert sig["data"]["ssid"] == "MyNetwork"
        assert sig["source"] == "wifi"
        assert sig["signal_type"] == "probe"

    def test_signal_with_position(self, store):
        did = store.create_dossier("Target")
        store.add_signal(
            did, "camera", "detection",
            position_x=100.5, position_y=200.3, confidence=0.9,
        )
        d = store.get_dossier(did)
        sig = d["signals"][0]
        assert sig["position_x"] == pytest.approx(100.5)
        assert sig["position_y"] == pytest.approx(200.3)
        assert sig["confidence"] == pytest.approx(0.9)

    def test_signal_updates_last_seen(self, store):
        did = store.create_dossier("Target", timestamp=1000.0)
        store.add_signal(did, "ble", "rssi", timestamp=2000.0)
        d = store.get_dossier(did)
        assert d["last_seen"] == pytest.approx(2000.0)

    def test_signal_does_not_backdate_last_seen(self, store):
        did = store.create_dossier("Target", timestamp=2000.0)
        store.add_signal(did, "ble", "rssi", timestamp=1000.0)
        d = store.get_dossier(did)
        assert d["last_seen"] == pytest.approx(2000.0)


# ── Enrichments ─────────────────────────────────────────────────────

class TestDossierEnrichments:
    """Tests for enrichment recording."""

    def test_add_enrichment(self, store):
        did = store.create_dossier("Target")
        eid = store.add_enrichment(did, "osint", "social_media", {"handle": "@target"})
        assert isinstance(eid, int)

    def test_enrichments_in_dossier(self, store):
        did = store.create_dossier("Target")
        store.add_enrichment(did, "osint", "social", {"handle": "@foo"})
        store.add_enrichment(did, "geoip", "location", {"city": "NYC"})
        d = store.get_dossier(did)
        assert len(d["enrichments"]) == 2

    def test_enrichment_data_parsed(self, store):
        did = store.create_dossier("Target")
        store.add_enrichment(did, "oui", "vendor", {"vendor": "Apple Inc"})
        d = store.get_dossier(did)
        e = d["enrichments"][0]
        assert e["data"]["vendor"] == "Apple Inc"
        assert e["provider"] == "oui"


# ── Lookup ──────────────────────────────────────────────────────────

class TestDossierLookup:
    """Tests for identifier-based lookup."""

    def test_find_by_identifier(self, store):
        did = store.create_dossier(
            "Known Device",
            identifiers={"mac": "AA:BB:CC:DD:EE:FF"},
        )
        result = store.find_by_identifier("mac", "AA:BB:CC:DD:EE:FF")
        assert result is not None
        assert result["dossier_id"] == did

    def test_find_by_identifier_not_found(self, store):
        store.create_dossier("Other", identifiers={"mac": "11:22:33:44:55:66"})
        assert store.find_by_identifier("mac", "FF:FF:FF:FF:FF:FF") is None


# ── Full-text search ────────────────────────────────────────────────

class TestDossierSearch:
    """Tests for FTS5 search."""

    def test_search_by_name(self, store):
        store.create_dossier("John Smith", entity_type="person")
        store.create_dossier("Jane Doe", entity_type="person")
        results = store.search("John")
        assert len(results) == 1
        assert results[0]["name"] == "John Smith"

    def test_search_by_entity_type(self, store):
        store.create_dossier("Car 1", entity_type="vehicle")
        store.create_dossier("Person 1", entity_type="person")
        results = store.search("vehicle")
        assert len(results) == 1

    def test_search_empty_query(self, store):
        store.create_dossier("Something")
        assert store.search("") == []
        assert store.search("   ") == []

    def test_search_no_results(self, store):
        store.create_dossier("Alpha")
        assert store.search("zzzznotfound") == []


# ── Listing ─────────────────────────────────────────────────────────

class TestDossierListing:
    """Tests for get_recent()."""

    def test_get_recent(self, store):
        store.create_dossier("Old", timestamp=1000.0)
        store.create_dossier("New", timestamp=2000.0)
        results = store.get_recent(limit=10)
        assert len(results) == 2
        assert results[0]["name"] == "New"

    def test_get_recent_with_limit(self, store):
        for i in range(10):
            store.create_dossier(f"D{i}", timestamp=float(1000 + i))
        results = store.get_recent(limit=3)
        assert len(results) == 3

    def test_get_recent_since(self, store):
        store.create_dossier("Old", timestamp=1000.0)
        store.create_dossier("New", timestamp=2000.0)
        results = store.get_recent(since=1500.0)
        assert len(results) == 1
        assert results[0]["name"] == "New"


# ── Updates ─────────────────────────────────────────────────────────

class TestDossierUpdates:
    """Tests for field updates."""

    def test_update_threat_level(self, store):
        did = store.create_dossier("Target", threat_level="none")
        assert store.update_threat_level(did, "critical") is True
        d = store.get_dossier(did)
        assert d["threat_level"] == "critical"

    def test_update_threat_level_nonexistent(self, store):
        assert store.update_threat_level("fake", "high") is False


# ── Merge ───────────────────────────────────────────────────────────

class TestDossierMerge:
    """Tests for dossier merging."""

    def test_merge_basic(self, store):
        d1 = store.create_dossier(
            "Primary", tags=["tag1"],
            identifiers={"mac": "AA:BB"},
            timestamp=1000.0,
        )
        d2 = store.create_dossier(
            "Secondary", tags=["tag2"],
            identifiers={"ssid": "network"},
            timestamp=500.0,
        )
        store.add_signal(d2, "wifi", "probe")

        assert store.merge_dossiers(d1, d2) is True

        merged = store.get_dossier(d1)
        assert merged is not None
        assert "tag1" in merged["tags"]
        assert "tag2" in merged["tags"]
        assert merged["identifiers"]["mac"] == "AA:BB"
        assert merged["identifiers"]["ssid"] == "network"
        assert merged["first_seen"] == pytest.approx(500.0)
        assert len(merged["signals"]) == 1

        # Secondary should be deleted
        assert store.get_dossier(d2) is None

    def test_merge_nonexistent_returns_false(self, store):
        d1 = store.create_dossier("Real")
        assert store.merge_dossiers(d1, "fake") is False
        assert store.merge_dossiers("fake", d1) is False

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.store.dossiers — SQLite-backed dossier store."""

import time

import pytest

from tritium_lib.store.dossiers import DossierStore


@pytest.fixture
def store():
    s = DossierStore(":memory:")
    yield s
    s.close()


class TestDossierStore:
    def test_create_dossier(self, store):
        did = store.create_dossier("Target Alpha", entity_type="person")
        assert did is not None
        assert len(did) > 0

    def test_get_dossier(self, store):
        did = store.create_dossier(
            "Target Alpha",
            entity_type="person",
            identifiers={"mac": "AA:BB:CC"},
            tags=["suspicious"],
            notes=["First sighting at mall"],
        )
        d = store.get_dossier(did)
        assert d is not None
        assert d["name"] == "Target Alpha"
        assert d["entity_type"] == "person"
        assert d["identifiers"]["mac"] == "AA:BB:CC"
        assert "suspicious" in d["tags"]
        assert "First sighting at mall" in d["notes"]
        assert d["signals"] == []
        assert d["enrichments"] == []

    def test_get_dossier_not_found(self, store):
        assert store.get_dossier("nonexistent") is None

    def test_add_signal(self, store):
        did = store.create_dossier("Target")
        sid = store.add_signal(
            did,
            source="ble_scanner_01",
            signal_type="ble_advertisement",
            data={"rssi": -55, "mac": "AA:BB:CC"},
            position_x=10.5,
            position_y=20.3,
            confidence=0.8,
        )
        assert sid is not None
        d = store.get_dossier(did)
        assert len(d["signals"]) == 1
        sig = d["signals"][0]
        assert sig["source"] == "ble_scanner_01"
        assert sig["data"]["rssi"] == -55

    def test_add_signal_updates_last_seen(self, store):
        ts1 = time.time()
        did = store.create_dossier("Target", timestamp=ts1)
        ts2 = ts1 + 1000
        store.add_signal(did, "sensor", "ble", timestamp=ts2)
        d = store.get_dossier(did)
        assert d["last_seen"] >= ts2

    def test_add_enrichment(self, store):
        did = store.create_dossier("Target")
        eid = store.add_enrichment(
            did,
            provider="threat_feed",
            enrichment_type="reputation",
            data={"score": 0.7, "category": "suspicious"},
        )
        assert eid is not None
        d = store.get_dossier(did)
        assert len(d["enrichments"]) == 1
        assert d["enrichments"][0]["provider"] == "threat_feed"

    def test_find_by_identifier(self, store):
        did = store.create_dossier(
            "Phone User",
            identifiers={"mac": "AA:BB:CC:DD:EE:FF"},
        )
        found = store.find_by_identifier("mac", "AA:BB:CC:DD:EE:FF")
        assert found is not None
        assert found["dossier_id"] == did

    def test_find_by_identifier_not_found(self, store):
        assert store.find_by_identifier("mac", "ZZ:ZZ:ZZ") is None

    def test_search(self, store):
        store.create_dossier("John Doe", entity_type="person")
        store.create_dossier("Delivery Van", entity_type="vehicle")
        results = store.search("John")
        assert len(results) >= 1
        assert any(r["name"] == "John Doe" for r in results)

    def test_search_empty(self, store):
        assert store.search("") == []
        assert store.search("  ") == []

    def test_get_recent(self, store):
        store.create_dossier("Old", timestamp=1000.0)
        store.create_dossier("New", timestamp=2000.0)
        recent = store.get_recent(limit=10)
        assert len(recent) == 2
        assert recent[0]["name"] == "New"

    def test_get_recent_since(self, store):
        store.create_dossier("Old", timestamp=1000.0)
        store.create_dossier("New", timestamp=2000.0)
        recent = store.get_recent(since=1500.0)
        assert len(recent) == 1
        assert recent[0]["name"] == "New"

    def test_update_threat_level(self, store):
        did = store.create_dossier("Target")
        assert store.update_threat_level(did, "high") is True
        d = store.get_dossier(did)
        assert d["threat_level"] == "high"

    def test_update_threat_level_not_found(self, store):
        assert store.update_threat_level("nope", "high") is False

    def test_delete_dossier(self, store):
        did = store.create_dossier("Target")
        store.add_signal(did, "s1", "ble")
        store.add_enrichment(did, "p1", "rep")
        assert store.delete_dossier(did) is True
        assert store.get_dossier(did) is None

    def test_delete_dossier_not_found(self, store):
        assert store.delete_dossier("nope") is False

    def test_merge_dossiers(self, store):
        did1 = store.create_dossier(
            "Target A",
            identifiers={"mac": "AA:BB"},
            tags=["tag_a"],
            notes=["Note A"],
            timestamp=1000.0,
        )
        did2 = store.create_dossier(
            "Target B",
            identifiers={"phone": "123"},
            tags=["tag_b", "tag_a"],
            notes=["Note B"],
            timestamp=500.0,
        )
        store.add_signal(did2, "sensor", "ble")
        store.add_enrichment(did2, "feed", "rep")

        assert store.merge_dossiers(did1, did2) is True

        merged = store.get_dossier(did1)
        assert merged is not None
        assert merged["identifiers"]["mac"] == "AA:BB"
        assert merged["identifiers"]["phone"] == "123"
        assert "tag_a" in merged["tags"]
        assert "tag_b" in merged["tags"]
        assert "Note A" in merged["notes"]
        assert "Note B" in merged["notes"]
        assert merged["first_seen"] == 500.0
        assert len(merged["signals"]) == 1
        assert len(merged["enrichments"]) == 1

        assert store.get_dossier(did2) is None

    def test_merge_dossiers_missing(self, store):
        did1 = store.create_dossier("A")
        assert store.merge_dossiers(did1, "nonexistent") is False
        assert store.merge_dossiers("nonexistent", did1) is False

    def test_multiple_signals_ordered(self, store):
        did = store.create_dossier("Target")
        store.add_signal(did, "s1", "ble", timestamp=100.0)
        store.add_signal(did, "s2", "wifi", timestamp=200.0)
        store.add_signal(did, "s3", "camera", timestamp=150.0)
        d = store.get_dossier(did)
        timestamps = [s["timestamp"] for s in d["signals"]]
        assert timestamps == sorted(timestamps, reverse=True)

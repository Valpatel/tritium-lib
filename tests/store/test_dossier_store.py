# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for DossierStore — CRUD, signals, enrichments, search, merge,
signal-history retention."""

import time

import pytest

from tritium_lib.store import dossiers as dossiers_mod
from tritium_lib.store.dossiers import DossierStore

DAY = 86400.0


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


# ── Signal-history retention (QUESTIONS.md 2026-04-29) ──────────────

def _signal_count(store, dossier_id):
    d = store.get_dossier(dossier_id)
    return len(d["signals"])


class TestSignalRetentionTTL:
    """30-day TTL on dossier_signals with VIP/pinned exemption."""

    def test_expired_signals_pruned(self, store):
        did = store.create_dossier("Stale Device", entity_type="device")
        now = time.time()
        store.add_signal(did, "ble", "rssi", {"v": 1}, timestamp=now - 31 * DAY)
        store.add_signal(did, "ble", "rssi", {"v": 2}, timestamp=now - 90 * DAY)

        stats = store.prune(now=now)
        assert stats["aged_out"] == 2
        assert _signal_count(store, did) == 0

    def test_fresh_signals_kept(self, store):
        did = store.create_dossier("Active Device", entity_type="device")
        now = time.time()
        store.add_signal(did, "ble", "rssi", {"v": 1}, timestamp=now - 1.0)
        store.add_signal(did, "ble", "rssi", {"v": 2}, timestamp=now - 29 * DAY)

        stats = store.prune(now=now)
        assert stats["aged_out"] == 0
        assert _signal_count(store, did) == 2

    def test_vip_tagged_dossier_immune_at_any_age(self, store):
        """Dossiers carrying the 'vip' tag keep signals forever."""
        did = store.create_dossier(
            "Person of Interest", entity_type="person", tags=["vip"],
        )
        now = time.time()
        for age_days in (31, 90, 365, 3650):
            store.add_signal(
                did, "ble", "presence", {"age": age_days},
                timestamp=now - age_days * DAY,
            )

        stats = store.prune(now=now)
        assert stats["aged_out"] == 0
        assert _signal_count(store, did) == 4

    def test_vip_tag_added_later_exempts(self, store):
        """Tagging an existing dossier 'vip' protects its history."""
        did = store.create_dossier("Upgraded", entity_type="person")
        now = time.time()
        store.add_signal(did, "ble", "presence", {}, timestamp=now - 60 * DAY)
        store._update_json_field(did, "tags", ["surveillance", "vip"])

        stats = store.prune(now=now)
        assert stats["aged_out"] == 0
        assert _signal_count(store, did) == 1

    def test_pinned_flag_exempts(self, store):
        did = store.create_dossier("Pinned", entity_type="person")
        store.set_pinned(did, True)
        now = time.time()
        store.add_signal(did, "ble", "presence", {}, timestamp=now - 400 * DAY)

        stats = store.prune(now=now)
        assert stats["aged_out"] == 0
        assert _signal_count(store, did) == 1

    def test_unpinned_neighbor_still_pruned(self, store):
        """VIP exemption is per-dossier, not global."""
        vip = store.create_dossier("VIP", tags=["vip"])
        normie = store.create_dossier("Normie")
        now = time.time()
        store.add_signal(vip, "ble", "presence", {}, timestamp=now - 60 * DAY)
        store.add_signal(normie, "ble", "presence", {}, timestamp=now - 60 * DAY)

        stats = store.prune(now=now)
        assert stats["aged_out"] == 1
        assert _signal_count(store, vip) == 1
        assert _signal_count(store, normie) == 0


class TestSignalRetentionBatching:
    """prune() drains in bounded batches — no multi-second lock holds."""

    def test_single_call_drains_multiple_batches(self):
        s = DossierStore(":memory:", prune_batch_size=10)
        try:
            did = s.create_dossier("Chatty", entity_type="device")
            now = time.time()
            for i in range(25):  # > 2 batches of 10
                s.add_signal(did, "ble", "rssi", {"i": i},
                             timestamp=now - 60 * DAY)
            s.add_signal(did, "ble", "rssi", {"fresh": True},
                         timestamp=now - 1.0)

            stats = s.prune(now=now)
            assert stats["aged_out"] == 25
            assert stats["batches"] >= 3
            assert _signal_count(s, did) == 1
        finally:
            s.close()

    def test_batch_size_call_override(self, store):
        did = store.create_dossier("Chatty", entity_type="device")
        now = time.time()
        for i in range(12):
            store.add_signal(did, "ble", "rssi", {"i": i},
                             timestamp=now - 60 * DAY)

        stats = store.prune(now=now, batch_size=5)
        assert stats["aged_out"] == 12
        assert stats["batches"] >= 3
        assert _signal_count(store, did) == 0

    def test_prune_safe_to_repeat(self, store):
        did = store.create_dossier("Once", entity_type="device")
        now = time.time()
        store.add_signal(did, "ble", "rssi", {}, timestamp=now - 60 * DAY)

        first = store.prune(now=now)
        second = store.prune(now=now)
        assert first["aged_out"] == 1
        assert second["aged_out"] == 0
        assert second["capped_out"] == 0

    def test_vacuum_off_by_default(self, store):
        stats = store.prune()
        assert stats["vacuumed"] is False

    def test_vacuum_flag_runs_vacuum(self, tmp_path):
        s = DossierStore(tmp_path / "dossiers.db")
        try:
            did = s.create_dossier("Bulky", entity_type="device")
            now = time.time()
            for i in range(50):
                s.add_signal(did, "ble", "rssi", {"i": i, "pad": "x" * 256},
                             timestamp=now - 60 * DAY)
            stats = s.prune(now=now, vacuum=True)
            assert stats["aged_out"] == 50
            assert stats["vacuumed"] is True
        finally:
            s.close()


class TestSignalRetentionConfig:
    """TTL config: constructor param > env override > module default."""

    def test_module_default_30_days(self):
        assert dossiers_mod.DEFAULT_SIGNAL_TTL_DAYS == 30.0
        s = DossierStore(":memory:")
        try:
            assert s.signal_ttl_days == 30.0
        finally:
            s.close()

    def test_constructor_ttl_override(self):
        s = DossierStore(":memory:", signal_ttl_days=7)
        try:
            did = s.create_dossier("Short-lived")
            now = time.time()
            s.add_signal(did, "ble", "rssi", {}, timestamp=now - 8 * DAY)
            s.add_signal(did, "ble", "rssi", {}, timestamp=now - 6 * DAY)

            stats = s.prune(now=now)
            assert stats["aged_out"] == 1
            assert _signal_count(s, did) == 1
        finally:
            s.close()

    def test_env_override_honored(self, monkeypatch):
        monkeypatch.setenv("TRITIUM_DOSSIER_SIGNAL_TTL_DAYS", "1")
        s = DossierStore(":memory:")
        try:
            assert s.signal_ttl_days == 1.0
            did = s.create_dossier("Ephemeral")
            now = time.time()
            s.add_signal(did, "ble", "rssi", {}, timestamp=now - 2 * DAY)
            s.add_signal(did, "ble", "rssi", {}, timestamp=now - 3600.0)

            stats = s.prune(now=now)
            assert stats["aged_out"] == 1
            assert _signal_count(s, did) == 1
        finally:
            s.close()

    def test_constructor_beats_env(self, monkeypatch):
        monkeypatch.setenv("TRITIUM_DOSSIER_SIGNAL_TTL_DAYS", "1")
        s = DossierStore(":memory:", signal_ttl_days=100)
        try:
            assert s.signal_ttl_days == 100.0
            did = s.create_dossier("Long-lived")
            now = time.time()
            s.add_signal(did, "ble", "rssi", {}, timestamp=now - 2 * DAY)

            stats = s.prune(now=now)
            assert stats["aged_out"] == 0
        finally:
            s.close()

    def test_garbage_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("TRITIUM_DOSSIER_SIGNAL_TTL_DAYS", "not-a-number")
        s = DossierStore(":memory:")
        try:
            assert s.signal_ttl_days == dossiers_mod.DEFAULT_SIGNAL_TTL_DAYS
        finally:
            s.close()

    def test_explicit_age_kwarg_still_wins(self):
        """SC's DossierManager passes max_signal_age_s explicitly —
        the legacy kwarg must override the configured TTL."""
        s = DossierStore(":memory:", signal_ttl_days=1)
        try:
            did = s.create_dossier("Legacy caller")
            now = time.time()
            s.add_signal(did, "ble", "rssi", {}, timestamp=now - 2 * DAY)

            stats = s.prune(now=now, max_signal_age_s=10 * DAY)
            assert stats["aged_out"] == 0
        finally:
            s.close()


class TestMaybePrune:
    """maybe_prune() — auto-prune hook, internally rate-limited."""

    def test_first_call_prunes(self, store):
        did = store.create_dossier("Old", entity_type="device")
        now = time.time()
        store.add_signal(did, "ble", "rssi", {}, timestamp=now - 60 * DAY)

        stats = store.maybe_prune(now=now)
        assert stats is not None
        assert stats["aged_out"] == 1

    def test_rate_limited_within_interval(self, store):
        now = time.time()
        assert store.maybe_prune(now=now) is not None
        assert store.maybe_prune(now=now + 10.0) is None
        assert store.maybe_prune(now=now + 3599.0) is None

    def test_runs_again_after_interval(self, store):
        now = time.time()
        assert store.maybe_prune(now=now) is not None
        assert store.maybe_prune(now=now + 3601.0) is not None

    def test_interval_configurable(self):
        s = DossierStore(":memory:", maybe_prune_interval_s=5.0)
        try:
            now = time.time()
            assert s.maybe_prune(now=now) is not None
            assert s.maybe_prune(now=now + 1.0) is None
            assert s.maybe_prune(now=now + 6.0) is not None
        finally:
            s.close()

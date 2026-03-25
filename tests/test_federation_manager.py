# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.federation — multi-site federation manager."""

import time
import pytest

from tritium_lib.federation import (
    FederatedSite,
    FederationManager,
    SearchResult,
    ShareCategory,
    SharePolicy,
    SyncResult,
    TrustLevel,
    default_policy_for_trust,
    federated_search,
    receive_targets,
    sync_targets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_target(
    target_id: str = "ble_aabb",
    classification: str = "person",
    alliance: str = "friendly",
    confidence: float = 0.9,
    source: str = "ble",
    name: str = "Test Target",
    **kwargs,
) -> dict:
    """Build a minimal target dict for testing."""
    t = {
        "target_id": target_id,
        "classification": classification,
        "alliance": alliance,
        "confidence": confidence,
        "source": source,
        "name": name,
        "lat": 37.7749,
        "lng": -122.4194,
        "identifiers": {"mac": "AA:BB:CC:DD:EE:FF"},
        "dossier_id": "dossier-001",
    }
    t.update(kwargs)
    return t


def _make_manager_with_sites():
    """Create a FederationManager with multiple sites at different trust levels."""
    mgr = FederationManager(local_site_id="local-hq", local_site_name="Local HQ")

    mgr.add_site(FederatedSite(
        site_id="site-full",
        name="Full Trust Site",
        url="https://full.tritium.local",
        trust_level=TrustLevel.FULL,
    ))
    mgr.add_site(FederatedSite(
        site_id="site-limited",
        name="Limited Trust Site",
        url="https://limited.tritium.local",
        trust_level=TrustLevel.LIMITED,
    ))
    mgr.add_site(FederatedSite(
        site_id="site-receive",
        name="Receive Only Site",
        url="https://receive.tritium.local",
        trust_level=TrustLevel.RECEIVE_ONLY,
    ))
    mgr.add_site(FederatedSite(
        site_id="site-blocked",
        name="Blocked Site",
        url="https://blocked.tritium.local",
        trust_level=TrustLevel.BLOCKED,
    ))
    return mgr


# ---------------------------------------------------------------------------
# TrustLevel enum
# ---------------------------------------------------------------------------


class TestTrustLevel:
    def test_values(self):
        assert TrustLevel.FULL.value == "full"
        assert TrustLevel.LIMITED.value == "limited"
        assert TrustLevel.RECEIVE_ONLY.value == "receive_only"
        assert TrustLevel.BLOCKED.value == "blocked"

    def test_all_levels_exist(self):
        assert len(TrustLevel) == 4


# ---------------------------------------------------------------------------
# SharePolicy
# ---------------------------------------------------------------------------


class TestSharePolicy:
    def test_default_policy(self):
        policy = SharePolicy()
        assert ShareCategory.TARGETS in policy.categories
        assert policy.min_confidence == 0.0
        assert not policy.redact_identifiers

    def test_allows_category(self):
        policy = SharePolicy(categories={ShareCategory.TARGETS, ShareCategory.ALERTS})
        assert policy.allows_category(ShareCategory.TARGETS)
        assert policy.allows_category(ShareCategory.ALERTS)
        assert not policy.allows_category(ShareCategory.DOSSIERS)

    def test_filter_target_passes_all(self):
        policy = SharePolicy()
        target = _make_target()
        result = policy.filter_target(target)
        assert result is not None
        assert result["target_id"] == "ble_aabb"

    def test_filter_target_classification(self):
        policy = SharePolicy(allowed_classifications={"vehicle"})
        person = _make_target(classification="person")
        vehicle = _make_target(classification="vehicle")
        assert policy.filter_target(person) is None
        assert policy.filter_target(vehicle) is not None

    def test_filter_target_alliance(self):
        policy = SharePolicy(allowed_alliances={"hostile"})
        friendly = _make_target(alliance="friendly")
        hostile = _make_target(alliance="hostile")
        assert policy.filter_target(friendly) is None
        assert policy.filter_target(hostile) is not None

    def test_filter_target_confidence(self):
        policy = SharePolicy(min_confidence=0.8)
        low = _make_target(confidence=0.5)
        high = _make_target(confidence=0.9)
        assert policy.filter_target(low) is None
        assert policy.filter_target(high) is not None

    def test_filter_target_redact_identifiers(self):
        policy = SharePolicy(redact_identifiers=True)
        target = _make_target()
        result = policy.filter_target(target)
        assert result is not None
        assert "identifiers" not in result

    def test_filter_target_redact_dossier_ids(self):
        policy = SharePolicy(redact_dossier_ids=True)
        target = _make_target()
        result = policy.filter_target(target)
        assert result is not None
        assert "dossier_id" not in result

    def test_filter_does_not_mutate_original(self):
        policy = SharePolicy(redact_identifiers=True)
        target = _make_target()
        policy.filter_target(target)
        assert "identifiers" in target  # original unchanged


# ---------------------------------------------------------------------------
# default_policy_for_trust
# ---------------------------------------------------------------------------


class TestDefaultPolicy:
    def test_full_trust_has_all_categories(self):
        policy = default_policy_for_trust(TrustLevel.FULL)
        assert ShareCategory.TARGETS in policy.categories
        assert ShareCategory.DOSSIERS in policy.categories
        assert ShareCategory.ALERTS in policy.categories
        assert ShareCategory.ZONES in policy.categories
        assert ShareCategory.EVENTS in policy.categories
        assert not policy.redact_identifiers
        assert not policy.redact_dossier_ids

    def test_limited_trust_redacts(self):
        policy = default_policy_for_trust(TrustLevel.LIMITED)
        assert ShareCategory.TARGETS in policy.categories
        assert ShareCategory.DOSSIERS not in policy.categories
        assert policy.redact_identifiers
        assert policy.redact_dossier_ids

    def test_receive_only_empty_policy(self):
        policy = default_policy_for_trust(TrustLevel.RECEIVE_ONLY)
        assert len(policy.categories) == 0

    def test_blocked_empty_policy(self):
        policy = default_policy_for_trust(TrustLevel.BLOCKED)
        assert len(policy.categories) == 0


# ---------------------------------------------------------------------------
# FederatedSite
# ---------------------------------------------------------------------------


class TestFederatedSite:
    def test_default_site(self):
        site = FederatedSite()
        assert site.name == "Unknown Site"
        assert site.trust_level == TrustLevel.LIMITED
        assert site.enabled
        assert site.share_policy is not None
        assert len(site.site_id) > 0

    def test_custom_site(self):
        site = FederatedSite(
            site_id="alpha",
            url="https://alpha.local",
            name="Alpha",
            trust_level=TrustLevel.FULL,
        )
        assert site.site_id == "alpha"
        assert site.url == "https://alpha.local"
        assert site.trust_level == TrustLevel.FULL

    def test_auto_policy_from_trust(self):
        full_site = FederatedSite(trust_level=TrustLevel.FULL)
        assert ShareCategory.DOSSIERS in full_site.effective_policy.categories

        limited_site = FederatedSite(trust_level=TrustLevel.LIMITED)
        assert limited_site.effective_policy.redact_identifiers

    def test_can_send_full(self):
        site = FederatedSite(trust_level=TrustLevel.FULL)
        assert site.can_send()

    def test_can_send_limited(self):
        site = FederatedSite(trust_level=TrustLevel.LIMITED)
        assert site.can_send()

    def test_cannot_send_receive_only(self):
        site = FederatedSite(trust_level=TrustLevel.RECEIVE_ONLY)
        assert not site.can_send()

    def test_cannot_send_blocked(self):
        site = FederatedSite(trust_level=TrustLevel.BLOCKED)
        assert not site.can_send()

    def test_can_receive_full(self):
        site = FederatedSite(trust_level=TrustLevel.FULL)
        assert site.can_receive()

    def test_can_receive_receive_only(self):
        site = FederatedSite(trust_level=TrustLevel.RECEIVE_ONLY)
        assert site.can_receive()

    def test_cannot_receive_blocked(self):
        site = FederatedSite(trust_level=TrustLevel.BLOCKED)
        assert not site.can_receive()

    def test_disabled_site_cannot_send_or_receive(self):
        site = FederatedSite(trust_level=TrustLevel.FULL, enabled=False)
        assert not site.can_send()
        assert not site.can_receive()


# ---------------------------------------------------------------------------
# FederationManager — site management
# ---------------------------------------------------------------------------


class TestFederationManagerSites:
    def test_create_manager(self):
        mgr = FederationManager(local_site_id="hq", local_site_name="HQ")
        assert mgr.local_site_id == "hq"
        assert mgr.local_site_name == "HQ"
        assert mgr.site_count == 0

    def test_auto_id(self):
        mgr = FederationManager()
        assert len(mgr.local_site_id) > 0

    def test_add_site(self):
        mgr = FederationManager()
        site = FederatedSite(site_id="alpha", name="Alpha")
        mgr.add_site(site)
        assert mgr.site_count == 1
        assert mgr.get_site("alpha") is site

    def test_add_duplicate_raises(self):
        mgr = FederationManager()
        site = FederatedSite(site_id="alpha")
        mgr.add_site(site)
        with pytest.raises(ValueError, match="already registered"):
            mgr.add_site(FederatedSite(site_id="alpha"))

    def test_remove_site(self):
        mgr = FederationManager()
        mgr.add_site(FederatedSite(site_id="alpha"))
        assert mgr.remove_site("alpha")
        assert mgr.site_count == 0
        assert mgr.get_site("alpha") is None

    def test_remove_nonexistent(self):
        mgr = FederationManager()
        assert not mgr.remove_site("nope")

    def test_list_sites_excludes_disabled(self):
        mgr = FederationManager()
        mgr.add_site(FederatedSite(site_id="a", enabled=True))
        mgr.add_site(FederatedSite(site_id="b", enabled=False))
        assert len(mgr.list_sites()) == 1
        assert len(mgr.list_sites(include_disabled=True)) == 2

    def test_update_trust(self):
        mgr = FederationManager()
        mgr.add_site(FederatedSite(site_id="alpha", trust_level=TrustLevel.FULL))
        assert mgr.update_trust("alpha", TrustLevel.BLOCKED)
        site = mgr.get_site("alpha")
        assert site.trust_level == TrustLevel.BLOCKED

    def test_update_trust_nonexistent(self):
        mgr = FederationManager()
        assert not mgr.update_trust("nope", TrustLevel.FULL)

    def test_update_policy(self):
        mgr = FederationManager()
        mgr.add_site(FederatedSite(site_id="alpha"))
        custom = SharePolicy(categories={ShareCategory.ALERTS}, min_confidence=0.5)
        assert mgr.update_policy("alpha", custom)
        site = mgr.get_site("alpha")
        assert site.effective_policy.min_confidence == 0.5

    def test_update_policy_nonexistent(self):
        mgr = FederationManager()
        assert not mgr.update_policy("nope", SharePolicy())


# ---------------------------------------------------------------------------
# FederationManager — sync_targets (outbound)
# ---------------------------------------------------------------------------


class TestSyncTargets:
    def test_sync_to_full_trust(self):
        mgr = _make_manager_with_sites()
        targets = [_make_target("t1"), _make_target("t2")]
        result = mgr.sync_targets("site-full", targets)
        assert result.success
        assert result.targets_accepted == 2
        assert result.targets_rejected == 0
        assert result.direction == "outbound"

    def test_sync_preserves_source_tags(self):
        mgr = _make_manager_with_sites()
        targets = [_make_target("t1")]
        result = mgr.sync_targets("site-full", targets)
        outbound = mgr.get_outbound_targets(result)
        assert len(outbound) == 1
        assert outbound[0]["_source_site_id"] == "local-hq"
        assert outbound[0]["_source_site_name"] == "Local HQ"
        assert "_shared_at" in outbound[0]

    def test_sync_to_limited_redacts(self):
        mgr = _make_manager_with_sites()
        targets = [_make_target("t1")]
        result = mgr.sync_targets("site-limited", targets)
        outbound = mgr.get_outbound_targets(result)
        assert len(outbound) == 1
        assert "identifiers" not in outbound[0]
        assert "dossier_id" not in outbound[0]

    def test_sync_to_receive_only_fails(self):
        mgr = _make_manager_with_sites()
        targets = [_make_target("t1")]
        result = mgr.sync_targets("site-receive", targets)
        assert not result.success
        assert result.targets_accepted == 0

    def test_sync_to_blocked_fails(self):
        mgr = _make_manager_with_sites()
        targets = [_make_target("t1")]
        result = mgr.sync_targets("site-blocked", targets)
        assert not result.success
        assert result.targets_accepted == 0

    def test_sync_to_unknown_site(self):
        mgr = FederationManager()
        result = mgr.sync_targets("unknown", [_make_target()])
        assert not result.success
        assert "Unknown site" in result.errors[0]

    def test_sync_with_classification_filter(self):
        mgr = FederationManager(local_site_id="hq")
        mgr.add_site(FederatedSite(
            site_id="alpha",
            trust_level=TrustLevel.FULL,
            share_policy=SharePolicy(
                categories={ShareCategory.TARGETS},
                allowed_classifications={"vehicle"},
            ),
        ))
        targets = [
            _make_target("t1", classification="person"),
            _make_target("t2", classification="vehicle"),
            _make_target("t3", classification="vehicle"),
        ]
        result = mgr.sync_targets("alpha", targets)
        assert result.targets_accepted == 2
        assert result.targets_rejected == 1

    def test_sync_with_max_cap(self):
        mgr = FederationManager(local_site_id="hq")
        mgr.add_site(FederatedSite(
            site_id="alpha",
            trust_level=TrustLevel.FULL,
            share_policy=SharePolicy(
                categories={ShareCategory.TARGETS},
                max_targets_per_sync=2,
            ),
        ))
        targets = [_make_target(f"t{i}") for i in range(5)]
        result = mgr.sync_targets("alpha", targets)
        assert result.targets_accepted == 2
        assert result.targets_rejected == 3
        outbound = mgr.get_outbound_targets(result)
        assert len(outbound) == 2

    def test_sync_empty_list(self):
        mgr = _make_manager_with_sites()
        result = mgr.sync_targets("site-full", [])
        assert result.success
        assert result.targets_processed == 0


# ---------------------------------------------------------------------------
# FederationManager — receive_targets (inbound)
# ---------------------------------------------------------------------------


class TestReceiveTargets:
    def test_receive_from_full_trust(self):
        mgr = _make_manager_with_sites()
        targets = [_make_target("r1"), _make_target("r2")]
        result = mgr.receive_targets("site-full", targets)
        assert result.success
        assert result.targets_accepted == 2
        assert result.direction == "inbound"

    def test_receive_stores_targets(self):
        mgr = _make_manager_with_sites()
        targets = [_make_target("r1", name="Remote Target")]
        mgr.receive_targets("site-full", targets)
        received = mgr.get_received_target("site-full", "r1")
        assert received is not None
        assert received["name"] == "Remote Target"
        assert received["_source_site_id"] == "site-full"

    def test_receive_from_limited_strips_sensitive(self):
        mgr = _make_manager_with_sites()
        targets = [_make_target("r1")]
        mgr.receive_targets("site-limited", targets)
        received = mgr.get_received_target("site-limited", "r1")
        assert "identifiers" not in received
        assert "dossier_id" not in received

    def test_receive_from_receive_only(self):
        """RECEIVE_ONLY means we receive from them — this should work."""
        mgr = _make_manager_with_sites()
        targets = [_make_target("r1")]
        result = mgr.receive_targets("site-receive", targets)
        assert result.success
        assert result.targets_accepted == 1

    def test_receive_from_blocked_fails(self):
        mgr = _make_manager_with_sites()
        targets = [_make_target("r1")]
        result = mgr.receive_targets("site-blocked", targets)
        assert not result.success

    def test_receive_from_unknown_site(self):
        mgr = FederationManager()
        result = mgr.receive_targets("unknown", [_make_target()])
        assert not result.success

    def test_receive_rejects_missing_target_id(self):
        mgr = _make_manager_with_sites()
        targets = [{"name": "No ID"}]
        result = mgr.receive_targets("site-full", targets)
        assert result.targets_rejected == 1
        assert result.targets_accepted == 0

    def test_receive_updates_last_seen(self):
        mgr = _make_manager_with_sites()
        site = mgr.get_site("site-full")
        assert site.last_seen == 0.0
        mgr.receive_targets("site-full", [_make_target("r1")])
        assert site.last_seen > 0


# ---------------------------------------------------------------------------
# FederationManager — federated_search
# ---------------------------------------------------------------------------


class TestFederatedSearch:
    def _setup_with_data(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [
            _make_target("t1", classification="person", alliance="friendly",
                         confidence=0.9, source="ble", name="Alice"),
            _make_target("t2", classification="vehicle", alliance="hostile",
                         confidence=0.7, source="yolo", name="Truck"),
            _make_target("t3", classification="person", alliance="hostile",
                         confidence=0.4, source="wifi", name="Bob"),
        ])
        mgr.receive_targets("site-receive", [
            _make_target("t4", classification="animal", alliance="neutral",
                         confidence=0.6, source="yolo", name="Dog"),
        ])
        return mgr

    def test_search_all(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({})
        assert result.total_matches == 4

    def test_search_by_classification(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({"classification": "person"})
        assert result.total_matches == 2
        ids = {m["target_id"] for m in result.matches}
        assert "t1" in ids
        assert "t3" in ids

    def test_search_by_alliance(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({"alliance": "hostile"})
        assert result.total_matches == 2

    def test_search_by_source(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({"source": "yolo"})
        assert result.total_matches == 2

    def test_search_by_name_substring(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({"name": "ali"})
        assert result.total_matches == 1
        assert result.matches[0]["target_id"] == "t1"

    def test_search_by_min_confidence(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({"min_confidence": 0.8})
        assert result.total_matches == 1
        assert result.matches[0]["target_id"] == "t1"

    def test_search_by_target_id(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({"target_id": "t2"})
        assert result.total_matches == 1
        assert result.matches[0]["name"] == "Truck"

    def test_search_by_source_site(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({"source_site_id": "site-full"})
        assert result.total_matches == 3

    def test_search_combined_filters(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({
            "classification": "person",
            "alliance": "hostile",
        })
        assert result.total_matches == 1
        assert result.matches[0]["target_id"] == "t3"

    def test_search_no_matches(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({"classification": "robot"})
        assert result.total_matches == 0
        assert result.success

    def test_search_unknown_source_site(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({"source_site_id": "nonexistent"})
        assert result.total_matches == 0
        assert "nonexistent" in result.sites_failed

    def test_search_reports_time(self):
        mgr = self._setup_with_data()
        result = mgr.federated_search({})
        assert result.search_time_ms >= 0


# ---------------------------------------------------------------------------
# Received target management
# ---------------------------------------------------------------------------


class TestReceivedTargetManagement:
    def test_get_received_targets_all(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [_make_target("t1"), _make_target("t2")])
        mgr.receive_targets("site-receive", [_make_target("t3")])
        all_targets = mgr.get_received_targets()
        assert len(all_targets) == 3

    def test_get_received_targets_by_site(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [_make_target("t1"), _make_target("t2")])
        mgr.receive_targets("site-receive", [_make_target("t3")])
        full_targets = mgr.get_received_targets("site-full")
        assert len(full_targets) == 2

    def test_clear_all(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [_make_target("t1")])
        mgr.receive_targets("site-receive", [_make_target("t2")])
        count = mgr.clear_received_targets()
        assert count == 2
        assert len(mgr.get_received_targets()) == 0

    def test_clear_by_site(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [_make_target("t1")])
        mgr.receive_targets("site-receive", [_make_target("t2")])
        count = mgr.clear_received_targets("site-full")
        assert count == 1
        assert len(mgr.get_received_targets()) == 1

    def test_purge_stale(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [_make_target("t1")])
        # Manually backdate the received_at
        key = ("site-full", "t1")
        mgr._received_targets[key]["_received_at"] = time.time() - 600
        purged = mgr.purge_stale_targets(max_age_s=300)
        assert purged == 1
        assert len(mgr.get_received_targets()) == 0

    def test_purge_keeps_fresh(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [_make_target("t1")])
        purged = mgr.purge_stale_targets(max_age_s=300)
        assert purged == 0
        assert len(mgr.get_received_targets()) == 1

    def test_remove_site_cleans_targets(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [_make_target("t1"), _make_target("t2")])
        mgr.remove_site("site-full")
        assert len(mgr.get_received_targets("site-full")) == 0


# ---------------------------------------------------------------------------
# Alert sync
# ---------------------------------------------------------------------------


class TestAlertSync:
    def test_sync_alerts_full_trust(self):
        mgr = _make_manager_with_sites()
        alerts = [{"alert_id": "a1", "severity": "high"}]
        result = mgr.sync_alerts("site-full", alerts)
        assert result.success
        assert result.alerts_accepted == 1

    def test_sync_alerts_limited_trust(self):
        mgr = _make_manager_with_sites()
        alerts = [{"alert_id": "a1"}]
        result = mgr.sync_alerts("site-limited", alerts)
        assert result.success  # LIMITED allows alerts

    def test_sync_alerts_blocked_fails(self):
        mgr = _make_manager_with_sites()
        alerts = [{"alert_id": "a1"}]
        result = mgr.sync_alerts("site-blocked", alerts)
        assert not result.success


# ---------------------------------------------------------------------------
# Stats and export
# ---------------------------------------------------------------------------


class TestStatsAndExport:
    def test_get_stats(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [_make_target("t1")])
        stats = mgr.get_stats()
        assert stats["local_site_id"] == "local-hq"
        assert stats["total_sites"] == 4
        assert stats["received_targets"] == 1
        assert stats["trust_levels"]["full"] == 1
        assert stats["trust_levels"]["blocked"] == 1

    def test_export_config(self):
        mgr = _make_manager_with_sites()
        config = mgr.export_config()
        assert config["local_site_id"] == "local-hq"
        assert len(config["sites"]) == 4
        # Check a site has the expected structure
        full_site = next(s for s in config["sites"] if s["site_id"] == "site-full")
        assert full_site["trust_level"] == "full"
        assert "targets" in full_site["share_policy"]["categories"]

    def test_sync_log(self):
        mgr = _make_manager_with_sites()
        mgr.sync_targets("site-full", [_make_target()])
        mgr.receive_targets("site-full", [_make_target()])
        log = mgr.get_sync_log()
        assert len(log) == 2
        assert log[0].direction == "outbound"
        assert log[1].direction == "inbound"

    def test_sync_log_filtered(self):
        mgr = _make_manager_with_sites()
        mgr.sync_targets("site-full", [_make_target()])
        mgr.sync_targets("site-limited", [_make_target()])
        log = mgr.get_sync_log(site_id="site-limited")
        assert len(log) == 1


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_sync_targets_function(self):
        mgr = _make_manager_with_sites()
        result = sync_targets(mgr, "site-full", [_make_target()])
        assert result.success
        assert result.targets_accepted == 1

    def test_receive_targets_function(self):
        mgr = _make_manager_with_sites()
        result = receive_targets(mgr, "site-full", [_make_target()])
        assert result.success
        assert result.targets_accepted == 1

    def test_federated_search_function(self):
        mgr = _make_manager_with_sites()
        mgr.receive_targets("site-full", [_make_target()])
        result = federated_search(mgr, {"classification": "person"})
        assert result.total_matches == 1


# ---------------------------------------------------------------------------
# SyncResult / SearchResult
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_sync_result_success(self):
        r = SyncResult(site_id="x")
        assert r.success
        r.errors.append("boom")
        assert not r.success

    def test_search_result_success(self):
        r = SearchResult()
        assert r.success
        r.sites_failed.append("x")
        assert not r.success

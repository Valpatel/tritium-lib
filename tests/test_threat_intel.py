# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.threat_intel — threat intelligence feed parser."""
from __future__ import annotations

import json
import time

import pytest

from tritium_lib.threat_intel import (
    FeedManager,
    IndicatorMatcher,
    IndicatorType,
    MatchResult,
    Severity,
    ThreatFeed,
    ThreatIndicator,
    from_stix,
    severity_from_score,
    to_stix,
)


# ===================================================================
# ThreatIndicator
# ===================================================================

class TestThreatIndicator:
    """ThreatIndicator dataclass and matching logic."""

    def test_default_values(self):
        ind = ThreatIndicator()
        assert ind.indicator_type == IndicatorType.MAC_WATCHLIST
        assert ind.value == ""
        assert ind.severity == 0.5
        assert ind.expires == 0.0
        assert ind.id.startswith("indicator--")

    def test_not_expired_no_expiry(self):
        ind = ThreatIndicator(expires=0.0)
        assert not ind.is_expired()

    def test_expired_past_timestamp(self):
        ind = ThreatIndicator(expires=time.time() - 100)
        assert ind.is_expired()

    def test_not_expired_future_timestamp(self):
        ind = ThreatIndicator(expires=time.time() + 3600)
        assert not ind.is_expired()

    def test_severity_level_low(self):
        ind = ThreatIndicator(severity=0.1)
        assert ind.severity_level == Severity.LOW

    def test_severity_level_medium(self):
        ind = ThreatIndicator(severity=0.5)
        assert ind.severity_level == Severity.MEDIUM

    def test_severity_level_high(self):
        ind = ThreatIndicator(severity=0.7)
        assert ind.severity_level == Severity.HIGH

    def test_severity_level_critical(self):
        ind = ThreatIndicator(severity=0.9)
        assert ind.severity_level == Severity.CRITICAL

    def test_normalized_mac_uppercase(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="aa:bb:cc:dd:ee:ff",
        )
        assert ind.normalized_value == "AA:BB:CC:DD:EE:FF"

    def test_normalized_mac_dash_to_colon(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA-BB-CC-DD-EE-FF",
        )
        assert ind.normalized_value == "AA:BB:CC:DD:EE:FF"

    def test_normalized_ble_uuid_lowercase(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.BLE_UUID,
            value="0000FE95-0000-1000-8000-00805F9B34FB",
        )
        assert ind.normalized_value == "0000fe95-0000-1000-8000-00805f9b34fb"

    def test_normalized_oui_prefix(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.OUI_WATCHLIST,
            value="aa-bb-cc-dd-ee-ff",
        )
        assert ind.normalized_value == "AA:BB:CC"

    def test_matches_mac_exact(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
        )
        assert ind.matches_mac("aa:bb:cc:dd:ee:ff")
        assert ind.matches_mac("AA:BB:CC:DD:EE:FF")
        assert not ind.matches_mac("11:22:33:44:55:66")

    def test_matches_mac_oui_prefix(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.OUI_WATCHLIST,
            value="AA:BB:CC",
        )
        assert ind.matches_mac("AA:BB:CC:DD:EE:FF")
        assert ind.matches_mac("AA:BB:CC:11:22:33")
        assert not ind.matches_mac("11:22:33:44:55:66")

    def test_matches_ssid_regex(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.SSID_PATTERN,
            value=r"^Evil.*Twin$",
        )
        assert ind.matches_ssid("EvilTwin")
        assert ind.matches_ssid("Evil Network Twin")
        assert not ind.matches_ssid("GoodNetwork")

    def test_matches_ssid_invalid_regex(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.SSID_PATTERN,
            value="[invalid",
        )
        assert not ind.matches_ssid("anything")

    def test_matches_ble_uuid(self):
        uid = "0000fe95-0000-1000-8000-00805f9b34fb"
        ind = ThreatIndicator(
            indicator_type=IndicatorType.BLE_UUID,
            value=uid,
        )
        assert ind.matches_ble_uuid(uid.upper())
        assert not ind.matches_ble_uuid("0000aaaa-0000-1000-8000-00805f9b34fb")

    def test_matches_ip(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.IP_WATCHLIST,
            value="192.168.1.100",
        )
        assert ind.matches_ip("192.168.1.100")
        assert not ind.matches_ip("10.0.0.1")

    def test_to_dict_roundtrip(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.SSID_PATTERN,
            value="FreeWifi.*",
            description="Suspicious hotspot pattern",
            severity=0.8,
            tags=["rogue-ap", "deauth"],
            source="analyst",
        )
        d = ind.to_dict()
        restored = ThreatIndicator.from_dict(d)
        assert restored.indicator_type == ind.indicator_type
        assert restored.value == ind.value
        assert restored.severity == ind.severity
        assert restored.tags == ind.tags
        assert restored.source == ind.source

    def test_to_stix_object_mac(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
            description="Known bad device",
            severity=0.9,
        )
        stix = ind.to_stix_object()
        assert stix["type"] == "indicator"
        assert stix["spec_version"] == "2.1"
        assert "mac-addr:value" in stix["pattern"]
        assert "'AA:BB:CC:DD:EE:FF'" in stix["pattern"]
        assert stix["confidence"] == 90
        assert stix["x_tritium_indicator_type"] == "mac_watchlist"

    def test_from_stix_object_roundtrip(self):
        ind = ThreatIndicator(
            indicator_type=IndicatorType.BLE_UUID,
            value="0000fe95-0000-1000-8000-00805f9b34fb",
            description="Xiaomi tracker UUID",
            severity=0.6,
            tags=["tracker", "ble"],
            source="research",
        )
        stix = ind.to_stix_object()
        restored = ThreatIndicator.from_stix_object(stix)
        assert restored.indicator_type == IndicatorType.BLE_UUID
        assert restored.value == ind.value
        assert restored.severity == pytest.approx(0.6, abs=0.01)
        assert restored.source == "research"


# ===================================================================
# ThreatFeed
# ===================================================================

class TestThreatFeed:
    """ThreatFeed collection management."""

    def test_empty_feed(self):
        feed = ThreatFeed(name="test", source="unit-test")
        assert feed.count == 0
        assert feed.active_count == 0
        assert feed.enabled

    def test_add_indicator(self):
        feed = ThreatFeed(name="test")
        ind = ThreatIndicator(value="AA:BB:CC:DD:EE:FF")
        feed.add_indicator(ind)
        assert feed.count == 1
        assert ind.source == "test"  # auto-set from feed name

    def test_remove_indicator(self):
        feed = ThreatFeed(name="test")
        ind = ThreatIndicator(id="indicator--abc", value="10.0.0.1")
        feed.add_indicator(ind)
        assert feed.count == 1
        assert feed.remove_indicator("indicator--abc")
        assert feed.count == 0

    def test_remove_nonexistent(self):
        feed = ThreatFeed(name="test")
        assert not feed.remove_indicator("indicator--nope")

    def test_get_indicator(self):
        feed = ThreatFeed(name="test")
        ind = ThreatIndicator(id="indicator--xyz", value="1.2.3.4")
        feed.add_indicator(ind)
        found = feed.get_indicator("indicator--xyz")
        assert found is not None
        assert found.value == "1.2.3.4"

    def test_get_indicators_by_type(self):
        feed = ThreatFeed(name="test")
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST, value="AA:BB:CC:DD:EE:FF",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.IP_WATCHLIST, value="10.0.0.1",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST, value="11:22:33:44:55:66",
        ))
        macs = feed.get_indicators_by_type(IndicatorType.MAC_WATCHLIST)
        assert len(macs) == 2

    def test_purge_expired(self):
        feed = ThreatFeed(name="test")
        feed.add_indicator(ThreatIndicator(
            value="AA:BB:CC:DD:EE:FF", expires=time.time() - 100,
        ))
        feed.add_indicator(ThreatIndicator(
            value="11:22:33:44:55:66", expires=0,  # never expires
        ))
        removed = feed.purge_expired()
        assert removed == 1
        assert feed.count == 1

    def test_to_dict_roundtrip(self):
        feed = ThreatFeed(name="watchlist", source="hq", tags=["internal"])
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
            severity=0.8,
        ))
        d = feed.to_dict()
        restored = ThreatFeed.from_dict(d)
        assert restored.name == "watchlist"
        assert restored.source == "hq"
        assert restored.count == 1
        assert restored.indicators[0].value == "AA:BB:CC:DD:EE:FF"


# ===================================================================
# Severity helpers
# ===================================================================

class TestSeverity:

    def test_severity_from_score_boundaries(self):
        assert severity_from_score(0.0) == Severity.LOW
        assert severity_from_score(0.29) == Severity.LOW
        assert severity_from_score(0.3) == Severity.MEDIUM
        assert severity_from_score(0.59) == Severity.MEDIUM
        assert severity_from_score(0.6) == Severity.HIGH
        assert severity_from_score(0.79) == Severity.HIGH
        assert severity_from_score(0.8) == Severity.CRITICAL
        assert severity_from_score(1.0) == Severity.CRITICAL


# ===================================================================
# FeedManager
# ===================================================================

class TestFeedManager:
    """FeedManager multi-feed management."""

    def test_add_and_get_feed(self):
        mgr = FeedManager()
        feed = ThreatFeed(name="alpha", feed_id="identity--alpha")
        mgr.add_feed(feed)
        assert mgr.feed_count == 1
        assert mgr.get_feed("identity--alpha") is feed

    def test_get_feed_by_name(self):
        mgr = FeedManager()
        feed = ThreatFeed(name="bravo", feed_id="identity--bravo")
        mgr.add_feed(feed)
        assert mgr.get_feed_by_name("bravo") is feed
        assert mgr.get_feed_by_name("charlie") is None

    def test_remove_feed(self):
        mgr = FeedManager()
        feed = ThreatFeed(feed_id="identity--rm")
        mgr.add_feed(feed)
        assert mgr.remove_feed("identity--rm")
        assert mgr.feed_count == 0
        assert not mgr.remove_feed("identity--rm")

    def test_all_indicators(self):
        mgr = FeedManager()
        f1 = ThreatFeed(name="f1")
        f1.add_indicator(ThreatIndicator(value="A"))
        f2 = ThreatFeed(name="f2")
        f2.add_indicator(ThreatIndicator(value="B"))
        f2.add_indicator(ThreatIndicator(value="C"))
        mgr.add_feed(f1)
        mgr.add_feed(f2)
        assert len(mgr.all_indicators()) == 3

    def test_all_indicators_excludes_disabled(self):
        mgr = FeedManager()
        f1 = ThreatFeed(name="f1", enabled=False)
        f1.add_indicator(ThreatIndicator(value="A"))
        mgr.add_feed(f1)
        assert len(mgr.all_indicators(enabled_only=True)) == 0
        assert len(mgr.all_indicators(enabled_only=False)) == 1

    def test_all_indicators_excludes_expired(self):
        mgr = FeedManager()
        f1 = ThreatFeed(name="f1")
        f1.add_indicator(ThreatIndicator(value="A", expires=time.time() - 100))
        f1.add_indicator(ThreatIndicator(value="B", expires=0))
        mgr.add_feed(f1)
        assert len(mgr.all_indicators(exclude_expired=True)) == 1
        assert len(mgr.all_indicators(exclude_expired=False)) == 2

    def test_indicators_by_type(self):
        mgr = FeedManager()
        feed = ThreatFeed(name="mixed")
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST, value="AA:BB:CC:DD:EE:FF",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.IP_WATCHLIST, value="10.0.0.1",
        ))
        mgr.add_feed(feed)
        macs = mgr.indicators_by_type(IndicatorType.MAC_WATCHLIST)
        assert len(macs) == 1
        assert macs[0].value == "AA:BB:CC:DD:EE:FF"

    def test_deduplicate(self):
        mgr = FeedManager()
        f1 = ThreatFeed(name="f1")
        f1.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
            severity=0.5,
        ))
        f2 = ThreatFeed(name="f2")
        f2.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
            severity=0.9,
        ))
        mgr.add_feed(f1)
        mgr.add_feed(f2)
        removed = mgr.deduplicate()
        assert removed == 1
        # The one with higher severity should remain
        remaining = mgr.all_indicators()
        assert len(remaining) == 1
        assert remaining[0].severity == 0.9

    def test_merge_feed_new(self):
        mgr = FeedManager()
        feed = ThreatFeed(name="new-feed", feed_id="identity--new")
        feed.add_indicator(ThreatIndicator(value="X"))
        added = mgr.merge_feed(feed)
        assert added == 1
        assert mgr.feed_count == 1

    def test_merge_feed_existing_no_duplicates(self):
        mgr = FeedManager()
        feed = ThreatFeed(name="base", feed_id="identity--base")
        ind1 = ThreatIndicator(id="indicator--1", value="A")
        feed.add_indicator(ind1)
        mgr.add_feed(feed)

        incoming = ThreatFeed(name="base", feed_id="identity--base")
        incoming.indicators.append(ThreatIndicator(id="indicator--1", value="A"))
        incoming.indicators.append(ThreatIndicator(id="indicator--2", value="B"))
        added = mgr.merge_feed(incoming)
        assert added == 1  # only indicator--2 is new
        assert mgr.get_feed("identity--base").count == 2

    def test_purge_expired_across_feeds(self):
        mgr = FeedManager()
        f1 = ThreatFeed(name="f1")
        f1.add_indicator(ThreatIndicator(value="A", expires=time.time() - 100))
        f2 = ThreatFeed(name="f2")
        f2.add_indicator(ThreatIndicator(value="B", expires=time.time() - 200))
        f2.add_indicator(ThreatIndicator(value="C", expires=0))
        mgr.add_feed(f1)
        mgr.add_feed(f2)
        purged = mgr.purge_expired()
        assert purged == 2
        assert mgr.total_indicator_count() == 1

    def test_stats(self):
        mgr = FeedManager()
        feed = ThreatFeed(name="stats-feed")
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST, value="AA:BB:CC:DD:EE:FF",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.IP_WATCHLIST, value="10.0.0.1",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="11:22:33:44:55:66",
            expires=time.time() - 100,
        ))
        mgr.add_feed(feed)
        s = mgr.stats()
        assert s["feed_count"] == 1
        assert s["total_indicators"] == 3
        assert s["expired_indicators"] == 1
        assert s["active_indicators"] == 2
        assert s["by_type"]["mac_watchlist"] == 2
        assert s["by_type"]["ip_watchlist"] == 1


# ===================================================================
# IndicatorMatcher
# ===================================================================

class TestIndicatorMatcher:
    """IndicatorMatcher — matching live targets against indicators."""

    def _build_matcher(self) -> IndicatorMatcher:
        """Build a matcher with a variety of indicators."""
        mgr = FeedManager()
        feed = ThreatFeed(name="test-feed")
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
            severity=0.9,
            description="Known surveillance device",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.OUI_WATCHLIST,
            value="DE:AD:BE",
            severity=0.6,
            description="Suspicious OUI prefix",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.SSID_PATTERN,
            value=r"^Free.*WiFi$",
            severity=0.7,
            description="Rogue AP pattern",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.BLE_UUID,
            value="0000fe95-0000-1000-8000-00805f9b34fb",
            severity=0.8,
            description="Xiaomi tracker",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.IP_WATCHLIST,
            value="192.168.1.100",
            severity=0.5,
            description="Suspicious host",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.BEHAVIORAL,
            value="loitering",
            severity=0.7,
            description="Loitering in restricted zone",
            metadata={"zone": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}},
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.BEHAVIORAL,
            value="convoy",
            severity=0.85,
            description="Convoy of 3+ vehicles",
            metadata={"min_count": 3},
        ))
        mgr.add_feed(feed)
        return IndicatorMatcher(mgr)

    def test_match_mac_hit(self):
        m = self._build_matcher()
        result = m.match_mac("aa:bb:cc:dd:ee:ff")
        assert result.matched
        assert result.hit_count == 1
        assert result.max_severity == 0.9

    def test_match_mac_miss(self):
        m = self._build_matcher()
        result = m.match_mac("11:22:33:44:55:66")
        assert not result.matched

    def test_match_mac_oui_prefix(self):
        m = self._build_matcher()
        result = m.match_mac("DE:AD:BE:11:22:33")
        assert result.matched
        assert result.hit_count == 1
        assert result.max_severity == 0.6

    def test_match_ssid_hit(self):
        m = self._build_matcher()
        result = m.match_ssid("FreePublicWiFi")
        assert result.matched
        assert result.max_severity == 0.7

    def test_match_ssid_miss(self):
        m = self._build_matcher()
        result = m.match_ssid("HomeNetwork5G")
        assert not result.matched

    def test_match_ble_uuid_hit(self):
        m = self._build_matcher()
        result = m.match_ble_uuid("0000FE95-0000-1000-8000-00805F9B34FB")
        assert result.matched
        assert result.max_severity == 0.8

    def test_match_ip_hit(self):
        m = self._build_matcher()
        result = m.match_ip("192.168.1.100")
        assert result.matched

    def test_match_ip_miss(self):
        m = self._build_matcher()
        result = m.match_ip("10.0.0.1")
        assert not result.matched

    def test_match_behavior_loitering_in_zone(self):
        m = self._build_matcher()
        result = m.match_behavior("loitering", location=(50.0, 50.0))
        assert result.matched
        assert result.max_severity == 0.7

    def test_match_behavior_loitering_outside_zone(self):
        m = self._build_matcher()
        result = m.match_behavior("loitering", location=(200.0, 200.0))
        assert not result.matched

    def test_match_behavior_convoy_sufficient_count(self):
        m = self._build_matcher()
        result = m.match_behavior("convoy", count=5)
        assert result.matched
        assert result.max_severity == 0.85

    def test_match_behavior_convoy_insufficient_count(self):
        m = self._build_matcher()
        result = m.match_behavior("convoy", count=2)
        assert not result.matched

    def test_match_behavior_unknown(self):
        m = self._build_matcher()
        result = m.match_behavior("teleporting")
        assert not result.matched

    def test_match_target_multi_attribute(self):
        m = self._build_matcher()
        result = m.match_target(
            mac="AA:BB:CC:DD:EE:FF",
            ssid="FreePublicWiFi",
            ip="192.168.1.100",
        )
        assert result.matched
        assert result.hit_count == 3
        assert result.max_severity == 0.9

    def test_match_target_no_attributes(self):
        m = self._build_matcher()
        result = m.match_target()
        assert not result.matched

    def test_disabled_feed_skipped(self):
        mgr = FeedManager()
        feed = ThreatFeed(name="disabled", enabled=False)
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
            severity=0.9,
        ))
        mgr.add_feed(feed)
        matcher = IndicatorMatcher(mgr)
        result = matcher.match_mac("AA:BB:CC:DD:EE:FF")
        assert not result.matched

    def test_expired_indicator_skipped(self):
        mgr = FeedManager()
        feed = ThreatFeed(name="expired")
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
            severity=0.9,
            expires=time.time() - 100,
        ))
        mgr.add_feed(feed)
        matcher = IndicatorMatcher(mgr)
        result = matcher.match_mac("AA:BB:CC:DD:EE:FF")
        assert not result.matched


# ===================================================================
# MatchResult
# ===================================================================

class TestMatchResult:

    def test_empty_result(self):
        r = MatchResult()
        assert not r.matched
        assert r.hit_count == 0
        assert r.max_severity == 0.0

    def test_add_hit(self):
        r = MatchResult()
        ind = ThreatIndicator(severity=0.7)
        r.add_hit(ind, "feed-a")
        assert r.matched
        assert r.hit_count == 1
        assert r.max_severity == 0.7
        assert "feed-a" in r.feed_names

    def test_to_dict(self):
        r = MatchResult()
        r.add_hit(ThreatIndicator(severity=0.5), "f1")
        d = r.to_dict()
        assert d["matched"] is True
        assert d["hit_count"] == 1
        assert d["max_severity"] == 0.5


# ===================================================================
# STIX 2.1 export/import
# ===================================================================

class TestSTIX:
    """STIX 2.1 JSON bundle round-trip."""

    def test_to_stix_produces_valid_json(self):
        feed = ThreatFeed(name="test-export", source="unit-test")
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
            severity=0.9,
        ))
        bundle_json = to_stix(feed)
        data = json.loads(bundle_json)
        assert data["type"] == "bundle"
        assert data["id"].startswith("bundle--")
        assert len(data["objects"]) == 2  # identity + indicator

    def test_to_stix_bundle_structure(self):
        feed = ThreatFeed(name="structure-test")
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.IP_WATCHLIST,
            value="10.0.0.1",
            severity=0.5,
            tags=["c2", "botnet"],
        ))
        data = json.loads(to_stix(feed))
        identity = [o for o in data["objects"] if o["type"] == "identity"][0]
        assert identity["name"] == "structure-test"
        indicator = [o for o in data["objects"] if o["type"] == "indicator"][0]
        assert indicator["spec_version"] == "2.1"
        assert "ipv4-addr:value" in indicator["pattern"]
        assert indicator["confidence"] == 50
        assert "c2" in indicator["labels"]

    def test_roundtrip_stix_full(self):
        feed = ThreatFeed(name="roundtrip", source="test", description="Full roundtrip test")
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="AA:BB:CC:DD:EE:FF",
            severity=0.9,
            description="Bad MAC",
            tags=["surveillance"],
            source="analyst",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.SSID_PATTERN,
            value="EvilTwin.*",
            severity=0.7,
            description="Rogue AP",
        ))
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.BEHAVIORAL,
            value="loitering",
            severity=0.6,
            description="Loitering behavior",
            metadata={"zone": {"x_min": 0, "y_min": 0, "x_max": 50, "y_max": 50}},
        ))

        bundle_json = to_stix(feed)
        restored = from_stix(bundle_json)

        assert restored.name == "roundtrip"
        assert len(restored.indicators) == 3

        mac_ind = next(
            i for i in restored.indicators
            if i.indicator_type == IndicatorType.MAC_WATCHLIST
        )
        assert mac_ind.value == "AA:BB:CC:DD:EE:FF"
        assert mac_ind.severity == pytest.approx(0.9, abs=0.01)

        ssid_ind = next(
            i for i in restored.indicators
            if i.indicator_type == IndicatorType.SSID_PATTERN
        )
        assert ssid_ind.value == "EvilTwin.*"

        behav_ind = next(
            i for i in restored.indicators
            if i.indicator_type == IndicatorType.BEHAVIORAL
        )
        assert behav_ind.value == "loitering"
        assert behav_ind.metadata.get("zone", {}).get("x_max") == 50

    def test_from_stix_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            from_stix("not json at all")

    def test_from_stix_not_a_bundle(self):
        with pytest.raises(ValueError, match="Expected STIX bundle type"):
            from_stix(json.dumps({"type": "report", "objects": []}))

    def test_from_stix_not_an_object(self):
        with pytest.raises(ValueError, match="STIX bundle must be a JSON object"):
            from_stix(json.dumps([1, 2, 3]))

    def test_stix_with_expiry(self):
        feed = ThreatFeed(name="expiry-test")
        future = time.time() + 86400
        feed.add_indicator(ThreatIndicator(
            indicator_type=IndicatorType.MAC_WATCHLIST,
            value="11:22:33:44:55:66",
            severity=0.5,
            expires=future,
        ))
        bundle_json = to_stix(feed)
        data = json.loads(bundle_json)
        indicator = [o for o in data["objects"] if o["type"] == "indicator"][0]
        assert "valid_until" in indicator

        restored = from_stix(bundle_json)
        assert restored.indicators[0].expires > 0

    def test_from_stix_pattern_fallback(self):
        """When x_tritium custom props are missing, parse from STIX pattern."""
        bundle = {
            "type": "bundle",
            "id": "bundle--test",
            "objects": [
                {
                    "type": "indicator",
                    "spec_version": "2.1",
                    "id": "indicator--fallback",
                    "created": "2026-01-01T00:00:00.000Z",
                    "modified": "2026-01-01T00:00:00.000Z",
                    "name": "Test",
                    "pattern": "[mac-addr:value = 'DE:AD:BE:EF:00:01']",
                    "pattern_type": "stix",
                    "valid_from": "2026-01-01T00:00:00.000Z",
                    "confidence": 75,
                    "indicator_types": ["malicious-activity"],
                    "labels": [],
                },
            ],
        }
        feed = from_stix(json.dumps(bundle))
        assert len(feed.indicators) == 1
        ind = feed.indicators[0]
        assert ind.indicator_type == IndicatorType.MAC_WATCHLIST
        assert ind.value == "DE:AD:BE:EF:00:01"
        assert ind.severity == pytest.approx(0.75, abs=0.01)

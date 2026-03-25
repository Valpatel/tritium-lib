# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.threat_intel — Threat intelligence feed parser and manager.

Parse, manage, and match threat intelligence indicators in STIX 2.1 format.
Supports MAC address watchlists, SSID pattern watchlists, BLE service UUID
watchlists, and behavioral pattern indicators (loitering, convoy).

No external STIX library required — implements the minimal STIX 2.1 subset
needed for indicator exchange.

Key classes:

  - **ThreatIndicator** — a single indicator of compromise (MAC, IP, SSID,
    BLE UUID, or behavioral pattern).
  - **ThreatFeed** — a named collection of indicators from a single source.
  - **FeedManager** — manage multiple feeds, merge, deduplicate, query.
  - **IndicatorMatcher** — match live targets against loaded indicators.
  - **to_stix()** / **from_stix()** — STIX 2.1 JSON bundle import/export.

Indicator types:

  - ``mac_watchlist``     — known-bad MAC addresses
  - ``ssid_pattern``      — suspicious SSID regex patterns
  - ``ble_uuid``          — BLE service UUID watchlist
  - ``behavioral``        — behavioral patterns (loitering, convoy, etc.)
  - ``ip_watchlist``      — known-bad IP addresses
  - ``oui_watchlist``     — known-bad OUI prefixes

Usage::

    from tritium_lib.threat_intel import (
        ThreatIndicator, ThreatFeed, FeedManager, IndicatorMatcher,
        to_stix, from_stix, IndicatorType,
    )

    # Build a feed
    feed = ThreatFeed(name="local-watchlist", source="manual")
    feed.add_indicator(ThreatIndicator(
        indicator_type=IndicatorType.MAC_WATCHLIST,
        value="AA:BB:CC:DD:EE:FF",
        description="Known surveillance device",
        severity=0.9,
    ))

    # Manage feeds
    mgr = FeedManager()
    mgr.add_feed(feed)

    # Match live targets
    matcher = IndicatorMatcher(mgr)
    hits = matcher.match_mac("AA:BB:CC:DD:EE:FF")

    # STIX export/import
    bundle_json = to_stix(feed)
    imported_feed = from_stix(bundle_json)
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STIX_SPEC_VERSION = "2.1"
STIX_BUNDLE_TYPE = "bundle"
STIX_INDICATOR_TYPE = "indicator"
STIX_IDENTITY_TYPE = "identity"

# STIX pattern prefixes for our indicator types
_STIX_PATTERN_MAP = {
    "mac_watchlist": "mac-addr:value",
    "ssid_pattern": "network-traffic:extensions.'wifi-ext'.ssid",
    "ble_uuid": "network-traffic:extensions.'ble-ext'.service_uuid",
    "behavioral": "x-tritium-behavior:pattern",
    "ip_watchlist": "ipv4-addr:value",
    "oui_watchlist": "mac-addr:value",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IndicatorType(str, Enum):
    """Types of threat indicators supported by Tritium."""
    MAC_WATCHLIST = "mac_watchlist"
    SSID_PATTERN = "ssid_pattern"
    BLE_UUID = "ble_uuid"
    BEHAVIORAL = "behavioral"
    IP_WATCHLIST = "ip_watchlist"
    OUI_WATCHLIST = "oui_watchlist"


class Severity(str, Enum):
    """Threat severity levels aligned with STIX confidence."""
    LOW = "low"           # 0.0 - 0.3
    MEDIUM = "medium"     # 0.3 - 0.6
    HIGH = "high"         # 0.6 - 0.8
    CRITICAL = "critical" # 0.8 - 1.0


def severity_from_score(score: float) -> Severity:
    """Convert a 0.0-1.0 score to a Severity enum."""
    if score >= 0.8:
        return Severity.CRITICAL
    elif score >= 0.6:
        return Severity.HIGH
    elif score >= 0.3:
        return Severity.MEDIUM
    return Severity.LOW


# ---------------------------------------------------------------------------
# ThreatIndicator
# ---------------------------------------------------------------------------

@dataclass
class ThreatIndicator:
    """A single indicator of compromise.

    Represents one threat signal: a MAC address, SSID pattern, BLE service
    UUID, behavioral pattern, IP address, or OUI prefix.

    Attributes:
        id: Unique indicator ID (auto-generated UUID if not provided).
        indicator_type: Category of indicator (IndicatorType enum).
        value: The indicator value (MAC, SSID regex, UUID, behavior name, IP, OUI).
        description: Human-readable description of the indicator.
        severity: Threat severity score 0.0 (benign) to 1.0 (maximum threat).
        tags: Free-form tags for grouping/filtering.
        created: Unix timestamp when the indicator was created.
        expires: Optional Unix timestamp when the indicator expires (0 = never).
        source: Origin of this indicator (feed name, analyst, etc.).
        metadata: Additional key-value data attached to the indicator.
    """
    indicator_type: IndicatorType = IndicatorType.MAC_WATCHLIST
    value: str = ""
    description: str = ""
    severity: float = 0.5
    tags: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    expires: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"indicator--{uuid.uuid4()}")

    def is_expired(self, now: float | None = None) -> bool:
        """Return True if this indicator has expired."""
        if self.expires <= 0:
            return False
        return (now or time.time()) > self.expires

    @property
    def severity_level(self) -> Severity:
        """Return the discrete severity level."""
        return severity_from_score(self.severity)

    @property
    def normalized_value(self) -> str:
        """Return the value normalized for comparison.

        - MAC addresses: uppercase, colon-separated.
        - SSID patterns: as-is (regex).
        - BLE UUIDs: lowercase.
        - IPs: as-is.
        - OUI: uppercase first 8 chars (AA:BB:CC).
        """
        itype = self.indicator_type
        val = self.value.strip()
        if itype == IndicatorType.MAC_WATCHLIST:
            return val.upper().replace("-", ":")
        elif itype == IndicatorType.BLE_UUID:
            return val.lower()
        elif itype == IndicatorType.OUI_WATCHLIST:
            return val.upper().replace("-", ":")[:8]
        return val

    def matches_mac(self, mac: str) -> bool:
        """Check if a MAC address matches this indicator."""
        norm_mac = mac.upper().replace("-", ":").strip()
        if self.indicator_type == IndicatorType.MAC_WATCHLIST:
            return self.normalized_value == norm_mac
        elif self.indicator_type == IndicatorType.OUI_WATCHLIST:
            return norm_mac.startswith(self.normalized_value)
        return False

    def matches_ssid(self, ssid: str) -> bool:
        """Check if an SSID matches this indicator's regex pattern."""
        if self.indicator_type != IndicatorType.SSID_PATTERN:
            return False
        try:
            return bool(re.search(self.value, ssid, re.IGNORECASE))
        except re.error:
            log.warning("Invalid SSID regex pattern: %s", self.value)
            return False

    def matches_ble_uuid(self, service_uuid: str) -> bool:
        """Check if a BLE service UUID matches this indicator."""
        if self.indicator_type != IndicatorType.BLE_UUID:
            return False
        return self.normalized_value == service_uuid.lower().strip()

    def matches_ip(self, ip: str) -> bool:
        """Check if an IP address matches this indicator."""
        if self.indicator_type != IndicatorType.IP_WATCHLIST:
            return False
        return self.value.strip() == ip.strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "id": self.id,
            "indicator_type": self.indicator_type.value,
            "value": self.value,
            "description": self.description,
            "severity": self.severity,
            "tags": list(self.tags),
            "created": self.created,
            "expires": self.expires,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ThreatIndicator:
        """Deserialize from a plain dict."""
        return cls(
            id=d.get("id", f"indicator--{uuid.uuid4()}"),
            indicator_type=IndicatorType(d.get("indicator_type", "mac_watchlist")),
            value=d.get("value", ""),
            description=d.get("description", ""),
            severity=float(d.get("severity", 0.5)),
            tags=list(d.get("tags", [])),
            created=float(d.get("created", time.time())),
            expires=float(d.get("expires", 0.0)),
            source=d.get("source", ""),
            metadata=dict(d.get("metadata", {})),
        )

    def to_stix_object(self) -> dict[str, Any]:
        """Convert this indicator to a STIX 2.1 indicator SDO."""
        pattern_key = _STIX_PATTERN_MAP.get(
            self.indicator_type.value, "x-tritium:value"
        )
        # Build STIX pattern string
        if self.indicator_type == IndicatorType.SSID_PATTERN:
            # STIX patterns use LIKE for regex-like matching
            pattern = f"[{pattern_key} LIKE '{self.value}']"
        else:
            pattern = f"[{pattern_key} = '{self.normalized_value}']"

        stix_obj: dict[str, Any] = {
            "type": STIX_INDICATOR_TYPE,
            "spec_version": STIX_SPEC_VERSION,
            "id": self.id if self.id.startswith("indicator--") else f"indicator--{self.id}",
            "created": _unix_to_stix_ts(self.created),
            "modified": _unix_to_stix_ts(self.created),
            "name": self.description or f"{self.indicator_type.value}: {self.value}",
            "description": self.description,
            "pattern": pattern,
            "pattern_type": "stix",
            "valid_from": _unix_to_stix_ts(self.created),
            "indicator_types": [_indicator_type_to_stix(self.indicator_type)],
            "confidence": int(self.severity * 100),
            "labels": self.tags,
        }
        if self.expires > 0:
            stix_obj["valid_until"] = _unix_to_stix_ts(self.expires)

        # Preserve tritium-specific data in custom properties
        stix_obj["x_tritium_indicator_type"] = self.indicator_type.value
        stix_obj["x_tritium_value"] = self.value
        stix_obj["x_tritium_source"] = self.source
        if self.metadata:
            stix_obj["x_tritium_metadata"] = self.metadata

        return stix_obj

    @classmethod
    def from_stix_object(cls, stix_obj: dict[str, Any]) -> ThreatIndicator:
        """Parse a STIX 2.1 indicator SDO into a ThreatIndicator.

        Prefers x_tritium_* custom properties when present. Falls back to
        parsing the STIX pattern string.
        """
        # Try custom properties first
        itype_str = stix_obj.get("x_tritium_indicator_type", "")
        value = stix_obj.get("x_tritium_value", "")

        if not itype_str or not value:
            # Parse from STIX pattern
            itype_str, value = _parse_stix_pattern(stix_obj.get("pattern", ""))

        try:
            itype = IndicatorType(itype_str)
        except ValueError:
            itype = IndicatorType.MAC_WATCHLIST

        created = _stix_ts_to_unix(stix_obj.get("created", ""))
        expires_str = stix_obj.get("valid_until", "")
        expires = _stix_ts_to_unix(expires_str) if expires_str else 0.0

        return cls(
            id=stix_obj.get("id", f"indicator--{uuid.uuid4()}"),
            indicator_type=itype,
            value=value,
            description=stix_obj.get("description", stix_obj.get("name", "")),
            severity=stix_obj.get("confidence", 50) / 100.0,
            tags=list(stix_obj.get("labels", [])),
            created=created,
            expires=expires,
            source=stix_obj.get("x_tritium_source", ""),
            metadata=dict(stix_obj.get("x_tritium_metadata", {})),
        )


# ---------------------------------------------------------------------------
# ThreatFeed
# ---------------------------------------------------------------------------

@dataclass
class ThreatFeed:
    """A named collection of threat indicators from a single source.

    Attributes:
        name: Human-readable feed name.
        source: Origin URL or description of the feed.
        description: Optional longer description.
        indicators: List of ThreatIndicator objects.
        created: Unix timestamp of feed creation.
        updated: Unix timestamp of last update.
        feed_id: Unique feed identifier.
        tags: Feed-level tags.
        enabled: Whether this feed is active for matching.
    """
    name: str = ""
    source: str = ""
    description: str = ""
    indicators: list[ThreatIndicator] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    feed_id: str = field(default_factory=lambda: f"identity--{uuid.uuid4()}")
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def add_indicator(self, indicator: ThreatIndicator) -> None:
        """Add an indicator to this feed, setting its source if empty."""
        if not indicator.source:
            indicator.source = self.name
        self.indicators.append(indicator)
        self.updated = time.time()

    def remove_indicator(self, indicator_id: str) -> bool:
        """Remove an indicator by ID. Returns True if found and removed."""
        before = len(self.indicators)
        self.indicators = [i for i in self.indicators if i.id != indicator_id]
        removed = len(self.indicators) < before
        if removed:
            self.updated = time.time()
        return removed

    def get_indicator(self, indicator_id: str) -> ThreatIndicator | None:
        """Look up an indicator by ID."""
        for ind in self.indicators:
            if ind.id == indicator_id:
                return ind
        return None

    def get_indicators_by_type(
        self, indicator_type: IndicatorType
    ) -> list[ThreatIndicator]:
        """Return all indicators of a given type."""
        return [i for i in self.indicators if i.indicator_type == indicator_type]

    def purge_expired(self, now: float | None = None) -> int:
        """Remove expired indicators. Returns the count removed."""
        ts = now or time.time()
        before = len(self.indicators)
        self.indicators = [i for i in self.indicators if not i.is_expired(ts)]
        removed = before - len(self.indicators)
        if removed:
            self.updated = time.time()
        return removed

    @property
    def count(self) -> int:
        """Number of indicators in this feed."""
        return len(self.indicators)

    @property
    def active_count(self) -> int:
        """Number of non-expired indicators."""
        now = time.time()
        return sum(1 for i in self.indicators if not i.is_expired(now))

    def to_dict(self) -> dict[str, Any]:
        """Serialize feed to a plain dict."""
        return {
            "feed_id": self.feed_id,
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "created": self.created,
            "updated": self.updated,
            "tags": list(self.tags),
            "enabled": self.enabled,
            "indicators": [i.to_dict() for i in self.indicators],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ThreatFeed:
        """Deserialize from a plain dict."""
        feed = cls(
            feed_id=d.get("feed_id", f"identity--{uuid.uuid4()}"),
            name=d.get("name", ""),
            source=d.get("source", ""),
            description=d.get("description", ""),
            created=float(d.get("created", time.time())),
            updated=float(d.get("updated", time.time())),
            tags=list(d.get("tags", [])),
            enabled=d.get("enabled", True),
        )
        for ind_dict in d.get("indicators", []):
            feed.indicators.append(ThreatIndicator.from_dict(ind_dict))
        return feed


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Result of matching a target attribute against threat indicators.

    Attributes:
        matched: Whether any indicator matched.
        indicators: List of matching ThreatIndicator objects.
        max_severity: Highest severity score among matches.
        feed_names: Set of feed names that produced matches.
    """
    matched: bool = False
    indicators: list[ThreatIndicator] = field(default_factory=list)
    max_severity: float = 0.0
    feed_names: set[str] = field(default_factory=set)

    def add_hit(self, indicator: ThreatIndicator, feed_name: str) -> None:
        """Record a matching indicator."""
        self.matched = True
        self.indicators.append(indicator)
        self.feed_names.add(feed_name)
        if indicator.severity > self.max_severity:
            self.max_severity = indicator.severity

    @property
    def hit_count(self) -> int:
        return len(self.indicators)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "hit_count": self.hit_count,
            "max_severity": self.max_severity,
            "feed_names": sorted(self.feed_names),
            "indicators": [i.to_dict() for i in self.indicators],
        }


# ---------------------------------------------------------------------------
# FeedManager
# ---------------------------------------------------------------------------

class FeedManager:
    """Manage multiple threat feeds — add, remove, merge, deduplicate, query.

    Thread-safe. Feeds are indexed by feed_id.
    """

    def __init__(self) -> None:
        self._feeds: dict[str, ThreatFeed] = {}
        self._lock = threading.Lock()

    def add_feed(self, feed: ThreatFeed) -> None:
        """Register a feed. Overwrites any existing feed with the same feed_id."""
        with self._lock:
            self._feeds[feed.feed_id] = feed

    def remove_feed(self, feed_id: str) -> bool:
        """Remove a feed by ID. Returns True if found and removed."""
        with self._lock:
            return self._feeds.pop(feed_id, None) is not None

    def get_feed(self, feed_id: str) -> ThreatFeed | None:
        """Look up a feed by ID."""
        with self._lock:
            return self._feeds.get(feed_id)

    def get_feed_by_name(self, name: str) -> ThreatFeed | None:
        """Look up a feed by name (first match)."""
        with self._lock:
            for feed in self._feeds.values():
                if feed.name == name:
                    return feed
        return None

    @property
    def feed_count(self) -> int:
        """Number of registered feeds."""
        with self._lock:
            return len(self._feeds)

    @property
    def feeds(self) -> list[ThreatFeed]:
        """List all registered feeds."""
        with self._lock:
            return list(self._feeds.values())

    def all_indicators(
        self,
        *,
        enabled_only: bool = True,
        exclude_expired: bool = True,
    ) -> list[ThreatIndicator]:
        """Return all indicators across all feeds.

        Args:
            enabled_only: If True, skip disabled feeds.
            exclude_expired: If True, skip expired indicators.
        """
        now = time.time()
        result: list[ThreatIndicator] = []
        with self._lock:
            for feed in self._feeds.values():
                if enabled_only and not feed.enabled:
                    continue
                for ind in feed.indicators:
                    if exclude_expired and ind.is_expired(now):
                        continue
                    result.append(ind)
        return result

    def indicators_by_type(
        self,
        indicator_type: IndicatorType,
        *,
        enabled_only: bool = True,
        exclude_expired: bool = True,
    ) -> list[ThreatIndicator]:
        """Return all indicators of a specific type across all feeds."""
        return [
            i for i in self.all_indicators(
                enabled_only=enabled_only,
                exclude_expired=exclude_expired,
            )
            if i.indicator_type == indicator_type
        ]

    def total_indicator_count(self, *, enabled_only: bool = True) -> int:
        """Count all indicators across all feeds."""
        with self._lock:
            total = 0
            for feed in self._feeds.values():
                if enabled_only and not feed.enabled:
                    continue
                total += len(feed.indicators)
            return total

    def deduplicate(self) -> int:
        """Remove duplicate indicators across all feeds.

        Duplicates are defined as indicators with the same type and
        normalized_value. When duplicates exist, the one with the highest
        severity is kept.

        Returns the number of duplicates removed.
        """
        with self._lock:
            # Build index: (type, normalized_value) -> (best_indicator, feed)
            seen: dict[tuple[str, str], tuple[ThreatIndicator, ThreatFeed]] = {}
            to_remove: list[tuple[ThreatFeed, str]] = []  # (feed, indicator_id)

            for feed in self._feeds.values():
                for ind in feed.indicators:
                    key = (ind.indicator_type.value, ind.normalized_value)
                    if key in seen:
                        existing_ind, existing_feed = seen[key]
                        if ind.severity > existing_ind.severity:
                            # New one is better — remove old, keep new
                            to_remove.append((existing_feed, existing_ind.id))
                            seen[key] = (ind, feed)
                        else:
                            # Old one is better or same — remove new
                            to_remove.append((feed, ind.id))
                    else:
                        seen[key] = (ind, feed)

            removed = 0
            for feed, ind_id in to_remove:
                if feed.remove_indicator(ind_id):
                    removed += 1

            return removed

    def purge_expired(self) -> int:
        """Purge expired indicators from all feeds."""
        with self._lock:
            total = 0
            for feed in self._feeds.values():
                total += feed.purge_expired()
            return total

    def merge_feed(self, incoming: ThreatFeed) -> int:
        """Merge indicators from an incoming feed into an existing feed.

        If a feed with the same feed_id exists, new indicators are appended
        (skipping duplicates by ID). If no matching feed exists, the incoming
        feed is added as a new feed.

        Returns the number of new indicators added.
        """
        with self._lock:
            existing = self._feeds.get(incoming.feed_id)
            if existing is None:
                self._feeds[incoming.feed_id] = incoming
                return len(incoming.indicators)

            existing_ids = {i.id for i in existing.indicators}
            added = 0
            for ind in incoming.indicators:
                if ind.id not in existing_ids:
                    existing.indicators.append(ind)
                    existing_ids.add(ind.id)
                    added += 1

            if added:
                existing.updated = time.time()
            return added

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about all managed feeds."""
        with self._lock:
            type_counts: dict[str, int] = defaultdict(int)
            total = 0
            expired = 0
            now = time.time()
            for feed in self._feeds.values():
                for ind in feed.indicators:
                    total += 1
                    type_counts[ind.indicator_type.value] += 1
                    if ind.is_expired(now):
                        expired += 1
            return {
                "feed_count": len(self._feeds),
                "total_indicators": total,
                "expired_indicators": expired,
                "active_indicators": total - expired,
                "by_type": dict(type_counts),
            }


# ---------------------------------------------------------------------------
# IndicatorMatcher
# ---------------------------------------------------------------------------

class IndicatorMatcher:
    """Match live target attributes against threat indicators.

    Uses a FeedManager as the indicator source and provides type-specific
    matching methods.
    """

    def __init__(self, feed_manager: FeedManager) -> None:
        self._mgr = feed_manager

    def match_mac(self, mac: str) -> MatchResult:
        """Match a MAC address against MAC and OUI watchlists."""
        result = MatchResult()
        for feed in self._mgr.feeds:
            if not feed.enabled:
                continue
            now = time.time()
            for ind in feed.indicators:
                if ind.is_expired(now):
                    continue
                if ind.indicator_type in (
                    IndicatorType.MAC_WATCHLIST,
                    IndicatorType.OUI_WATCHLIST,
                ):
                    if ind.matches_mac(mac):
                        result.add_hit(ind, feed.name)
        return result

    def match_ssid(self, ssid: str) -> MatchResult:
        """Match an SSID against SSID pattern indicators."""
        result = MatchResult()
        for feed in self._mgr.feeds:
            if not feed.enabled:
                continue
            now = time.time()
            for ind in feed.indicators:
                if ind.is_expired(now):
                    continue
                if ind.matches_ssid(ssid):
                    result.add_hit(ind, feed.name)
        return result

    def match_ble_uuid(self, service_uuid: str) -> MatchResult:
        """Match a BLE service UUID against BLE UUID indicators."""
        result = MatchResult()
        for feed in self._mgr.feeds:
            if not feed.enabled:
                continue
            now = time.time()
            for ind in feed.indicators:
                if ind.is_expired(now):
                    continue
                if ind.matches_ble_uuid(service_uuid):
                    result.add_hit(ind, feed.name)
        return result

    def match_ip(self, ip: str) -> MatchResult:
        """Match an IP address against IP watchlist indicators."""
        result = MatchResult()
        for feed in self._mgr.feeds:
            if not feed.enabled:
                continue
            now = time.time()
            for ind in feed.indicators:
                if ind.is_expired(now):
                    continue
                if ind.matches_ip(ip):
                    result.add_hit(ind, feed.name)
        return result

    def match_behavior(
        self,
        behavior_type: str,
        *,
        location: tuple[float, float] | None = None,
        count: int = 1,
    ) -> MatchResult:
        """Match a behavioral observation against behavioral indicators.

        Args:
            behavior_type: Type of behavior observed (e.g., "loitering",
                "convoy", "casing", "probe_burst").
            location: Optional (x, y) position where the behavior was observed.
            count: Number of entities involved (for convoy-type behaviors).
        """
        result = MatchResult()
        for feed in self._mgr.feeds:
            if not feed.enabled:
                continue
            now = time.time()
            for ind in feed.indicators:
                if ind.is_expired(now):
                    continue
                if ind.indicator_type != IndicatorType.BEHAVIORAL:
                    continue
                # Behavioral indicators store the behavior name in value
                # and optional constraints in metadata
                if ind.value.lower() != behavior_type.lower():
                    continue
                # Check optional metadata constraints
                meta = ind.metadata
                if "min_count" in meta and count < int(meta["min_count"]):
                    continue
                if location and "zone" in meta:
                    zone = meta["zone"]  # {"x_min", "y_min", "x_max", "y_max"}
                    x, y = location
                    if not (
                        zone.get("x_min", float("-inf")) <= x <= zone.get("x_max", float("inf"))
                        and zone.get("y_min", float("-inf")) <= y <= zone.get("y_max", float("inf"))
                    ):
                        continue
                result.add_hit(ind, feed.name)
        return result

    def match_target(
        self,
        *,
        mac: str | None = None,
        ssid: str | None = None,
        ble_uuid: str | None = None,
        ip: str | None = None,
        behaviors: list[str] | None = None,
    ) -> MatchResult:
        """Match a target against all available indicator types.

        Combines results from all attribute types into a single MatchResult.
        """
        combined = MatchResult()
        checks = []
        if mac:
            checks.append(self.match_mac(mac))
        if ssid:
            checks.append(self.match_ssid(ssid))
        if ble_uuid:
            checks.append(self.match_ble_uuid(ble_uuid))
        if ip:
            checks.append(self.match_ip(ip))
        if behaviors:
            for b in behaviors:
                checks.append(self.match_behavior(b))

        for r in checks:
            for ind in r.indicators:
                combined.add_hit(ind, ind.source)
        return combined


# ---------------------------------------------------------------------------
# STIX 2.1 helpers
# ---------------------------------------------------------------------------

def _unix_to_stix_ts(ts: float) -> str:
    """Convert a Unix timestamp to STIX 2.1 timestamp string."""
    if ts <= 0:
        ts = time.time()
    import datetime
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _stix_ts_to_unix(ts_str: str) -> float:
    """Convert a STIX 2.1 timestamp string to Unix timestamp."""
    if not ts_str:
        return time.time()
    import datetime
    # Handle with or without fractional seconds
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.datetime.strptime(ts_str, fmt).replace(
                tzinfo=datetime.timezone.utc
            )
            return dt.timestamp()
        except ValueError:
            continue
    return time.time()


def _indicator_type_to_stix(itype: IndicatorType) -> str:
    """Map our IndicatorType to a STIX 2.1 indicator_types vocabulary term."""
    mapping = {
        IndicatorType.MAC_WATCHLIST: "malicious-activity",
        IndicatorType.SSID_PATTERN: "anomalous-activity",
        IndicatorType.BLE_UUID: "anomalous-activity",
        IndicatorType.BEHAVIORAL: "anomalous-activity",
        IndicatorType.IP_WATCHLIST: "malicious-activity",
        IndicatorType.OUI_WATCHLIST: "anomalous-activity",
    }
    return mapping.get(itype, "unknown")


def _parse_stix_pattern(pattern: str) -> tuple[str, str]:
    """Parse a STIX pattern string to extract indicator type and value.

    Returns (indicator_type_str, value). Falls back to ("mac_watchlist", "")
    if parsing fails.
    """
    # Pattern format: [key = 'value'] or [key LIKE 'value']
    # Build ordered lookup: check more-specific keys first to avoid
    # ambiguity between mac_watchlist and oui_watchlist (both use mac-addr:value).
    # Order: specific extension keys first, then generic ones.
    _ORDERED_PATTERNS: list[tuple[str, str]] = [
        (_STIX_PATTERN_MAP["ssid_pattern"], "ssid_pattern"),
        (_STIX_PATTERN_MAP["ble_uuid"], "ble_uuid"),
        (_STIX_PATTERN_MAP["behavioral"], "behavioral"),
        (_STIX_PATTERN_MAP["ip_watchlist"], "ip_watchlist"),
        (_STIX_PATTERN_MAP["mac_watchlist"], "mac_watchlist"),
        # oui_watchlist also uses mac-addr:value — disambiguated below
    ]

    for pkey, itype_str in _ORDERED_PATTERNS:
        if pkey in pattern:
            # Extract value between quotes
            match = re.search(r"'([^']*)'", pattern)
            if match:
                val = match.group(1)
                # Disambiguate MAC vs OUI: OUI prefixes are <= 8 chars (AA:BB:CC)
                if itype_str == "mac_watchlist" and len(val) <= 8:
                    return "oui_watchlist", val
                return itype_str, val

    return "mac_watchlist", ""


# ---------------------------------------------------------------------------
# STIX 2.1 bundle export / import
# ---------------------------------------------------------------------------

def to_stix(feed: ThreatFeed) -> str:
    """Export a ThreatFeed as a STIX 2.1 JSON bundle string.

    Produces a valid STIX 2.1 bundle containing:
      - An identity SDO representing the feed source.
      - One indicator SDO per ThreatIndicator.

    Args:
        feed: The ThreatFeed to export.

    Returns:
        A JSON string containing the STIX 2.1 bundle.
    """
    objects: list[dict[str, Any]] = []

    # Identity for the feed source
    identity_obj: dict[str, Any] = {
        "type": STIX_IDENTITY_TYPE,
        "spec_version": STIX_SPEC_VERSION,
        "id": feed.feed_id if feed.feed_id.startswith("identity--") else f"identity--{feed.feed_id}",
        "created": _unix_to_stix_ts(feed.created),
        "modified": _unix_to_stix_ts(feed.updated),
        "name": feed.name,
        "description": feed.description or f"Threat feed: {feed.name}",
        "identity_class": "system",
    }
    objects.append(identity_obj)

    # Indicator SDOs
    for ind in feed.indicators:
        objects.append(ind.to_stix_object())

    bundle: dict[str, Any] = {
        "type": STIX_BUNDLE_TYPE,
        "id": f"bundle--{uuid.uuid4()}",
        "objects": objects,
    }

    return json.dumps(bundle, indent=2, default=str)


def from_stix(json_str: str) -> ThreatFeed:
    """Import a STIX 2.1 JSON bundle into a ThreatFeed.

    Parses identity SDOs as the feed source and indicator SDOs as
    ThreatIndicator objects.

    Args:
        json_str: A JSON string containing a STIX 2.1 bundle.

    Returns:
        A ThreatFeed populated with the parsed indicators.

    Raises:
        ValueError: If the JSON is not a valid STIX bundle.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("STIX bundle must be a JSON object")

    if data.get("type") != STIX_BUNDLE_TYPE:
        raise ValueError(
            f"Expected STIX bundle type, got: {data.get('type', 'missing')}"
        )

    objects = data.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("STIX bundle 'objects' must be a list")

    # Extract identity (feed metadata)
    feed_name = ""
    feed_description = ""
    feed_id = f"identity--{uuid.uuid4()}"
    feed_created = time.time()

    for obj in objects:
        if obj.get("type") == STIX_IDENTITY_TYPE:
            feed_name = obj.get("name", "")
            feed_description = obj.get("description", "")
            feed_id = obj.get("id", feed_id)
            feed_created = _stix_ts_to_unix(obj.get("created", ""))
            break

    feed = ThreatFeed(
        name=feed_name,
        source=feed_name,
        description=feed_description,
        feed_id=feed_id,
        created=feed_created,
    )

    # Extract indicators
    for obj in objects:
        if obj.get("type") == STIX_INDICATOR_TYPE:
            ind = ThreatIndicator.from_stix_object(obj)
            feed.indicators.append(ind)

    return feed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "IndicatorType",
    "Severity",
    "severity_from_score",
    "ThreatIndicator",
    "ThreatFeed",
    "MatchResult",
    "FeedManager",
    "IndicatorMatcher",
    "to_stix",
    "from_stix",
]

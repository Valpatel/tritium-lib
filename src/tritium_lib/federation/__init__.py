# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.federation — Multi-site federation for Tritium deployments.

Manages connections between multiple Tritium instances, controlling what data
is shared with whom based on trust levels and share policies.

Trust levels:
  - FULL: share everything bidirectionally
  - LIMITED: share positions only, no dossiers
  - RECEIVE_ONLY: accept data but don't share
  - BLOCKED: no data exchange

Architecture:
  - FederationManager holds all known sites and their trust/share config
  - sync_targets() pushes local targets to a remote site (filtered by policy)
  - receive_targets() ingests targets from a remote site (filtered by trust)
  - federated_search() queries across all connected federated sites
  - HTTP/MQTT transport is NOT included — this is pure data model + logic

Quick start::

    from tritium_lib.federation import (
        FederationManager,
        FederatedSite,
        TrustLevel,
        SharePolicy,
    )

    mgr = FederationManager(local_site_id="hq-east")

    # Register a remote site
    site = FederatedSite(
        url="https://west.tritium.local",
        name="HQ West",
        trust_level=TrustLevel.FULL,
    )
    mgr.add_site(site)

    # Share targets with that site
    outgoing = mgr.sync_targets(site.site_id, local_targets)

    # Receive targets from that site
    accepted = mgr.receive_targets(site.site_id, remote_targets)

    # Search across all sites
    results = mgr.federated_search({"classification": "person"})
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrustLevel(str, Enum):
    """Trust level for a federated site.

    Controls the maximum data that can flow to/from this site.
    """
    FULL = "full"                  # Share everything bidirectionally
    LIMITED = "limited"            # Share positions only, no dossiers
    RECEIVE_ONLY = "receive_only"  # Accept data but don't share
    BLOCKED = "blocked"            # No data exchange


class ShareCategory(str, Enum):
    """Categories of data that can be shared."""
    TARGETS = "targets"
    ALERTS = "alerts"
    ZONES = "zones"
    DOSSIERS = "dossiers"
    EVENTS = "events"


# ---------------------------------------------------------------------------
# SharePolicy
# ---------------------------------------------------------------------------


@dataclass
class SharePolicy:
    """Defines what data categories to share and optional filters.

    A SharePolicy is attached to a FederatedSite to control exactly what
    data flows to that site beyond the coarse TrustLevel.

    Attributes
    ----------
    categories:
        Set of ShareCategory values this policy allows.
    allowed_classifications:
        If non-empty, only targets with these classifications are shared.
        Empty means all classifications are allowed.
    allowed_alliances:
        If non-empty, only targets with these alliances are shared.
        Empty means all alliances are allowed.
    min_confidence:
        Minimum confidence threshold for shared targets (0.0 - 1.0).
    max_targets_per_sync:
        Cap on number of targets sent in a single sync operation.
        0 means unlimited.
    redact_identifiers:
        If True, strip identifiers dict from shared targets.
    redact_dossier_ids:
        If True, strip dossier_id from shared targets.
    """
    categories: set[ShareCategory] = field(
        default_factory=lambda: {ShareCategory.TARGETS}
    )
    allowed_classifications: set[str] = field(default_factory=set)
    allowed_alliances: set[str] = field(default_factory=set)
    min_confidence: float = 0.0
    max_targets_per_sync: int = 0
    redact_identifiers: bool = False
    redact_dossier_ids: bool = False

    def allows_category(self, category: ShareCategory) -> bool:
        """Check if this policy allows sharing a given category."""
        return category in self.categories

    def filter_target(self, target: dict[str, Any]) -> dict[str, Any] | None:
        """Apply policy filters to a target dict.

        Returns the (possibly redacted) target dict if it passes filters,
        or None if it should be excluded.
        """
        # Classification filter
        if self.allowed_classifications:
            classification = target.get("classification", "unknown")
            if classification not in self.allowed_classifications:
                return None

        # Alliance filter
        if self.allowed_alliances:
            alliance = target.get("alliance", "unknown")
            if alliance not in self.allowed_alliances:
                return None

        # Confidence filter
        confidence = target.get("confidence", 0.0)
        if isinstance(confidence, (int, float)) and confidence < self.min_confidence:
            return None

        # Build output (possibly redacted)
        result = dict(target)
        if self.redact_identifiers:
            result.pop("identifiers", None)
        if self.redact_dossier_ids:
            result.pop("dossier_id", None)
        return result


# ---------------------------------------------------------------------------
# Default policies per trust level
# ---------------------------------------------------------------------------


def default_policy_for_trust(trust: TrustLevel) -> SharePolicy:
    """Return a sensible default SharePolicy for a given trust level."""
    if trust == TrustLevel.FULL:
        return SharePolicy(
            categories={
                ShareCategory.TARGETS,
                ShareCategory.ALERTS,
                ShareCategory.ZONES,
                ShareCategory.DOSSIERS,
                ShareCategory.EVENTS,
            },
            redact_identifiers=False,
            redact_dossier_ids=False,
        )
    elif trust == TrustLevel.LIMITED:
        return SharePolicy(
            categories={ShareCategory.TARGETS, ShareCategory.ALERTS},
            redact_identifiers=True,
            redact_dossier_ids=True,
        )
    elif trust == TrustLevel.RECEIVE_ONLY:
        # Receive-only sites get an empty outbound policy
        return SharePolicy(categories=set())
    else:
        # BLOCKED
        return SharePolicy(categories=set())


# ---------------------------------------------------------------------------
# FederatedSite
# ---------------------------------------------------------------------------


@dataclass
class FederatedSite:
    """Represents a remote Tritium instance in the federation.

    Each site has a URL (for future HTTP transport), a human-readable name,
    a trust level that governs coarse data flow, and a SharePolicy for
    fine-grained control.

    Attributes
    ----------
    site_id:
        Unique identifier for this site.  Auto-generated if not provided.
    url:
        Base URL of the remote Tritium instance (e.g. https://west.tritium.local).
    name:
        Human-readable name for display.
    trust_level:
        Coarse trust classification.
    share_policy:
        Fine-grained policy for outbound data sharing.  If None, a default
        is derived from trust_level.
    enabled:
        Whether this site is active in the federation.
    last_seen:
        Unix timestamp of the last heartbeat or data exchange.
    metadata:
        Arbitrary key-value metadata about the site.
    """
    site_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    name: str = "Unknown Site"
    trust_level: TrustLevel = TrustLevel.LIMITED
    share_policy: SharePolicy | None = None
    enabled: bool = True
    last_seen: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.share_policy is None:
            self.share_policy = default_policy_for_trust(self.trust_level)

    @property
    def effective_policy(self) -> SharePolicy:
        """Return the share policy, guaranteed non-None."""
        if self.share_policy is None:
            return default_policy_for_trust(self.trust_level)
        return self.share_policy

    def can_send(self) -> bool:
        """Whether we are allowed to send data TO this site."""
        if not self.enabled:
            return False
        if self.trust_level == TrustLevel.BLOCKED:
            return False
        if self.trust_level == TrustLevel.RECEIVE_ONLY:
            # RECEIVE_ONLY means *we* receive from them, not send to them
            return False
        return True

    def can_receive(self) -> bool:
        """Whether we are allowed to receive data FROM this site."""
        if not self.enabled:
            return False
        if self.trust_level == TrustLevel.BLOCKED:
            return False
        return True


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------


@dataclass
class SyncResult:
    """Result of a sync operation (send or receive)."""
    site_id: str = ""
    direction: str = "outbound"  # "outbound" or "inbound"
    targets_processed: int = 0
    targets_accepted: int = 0
    targets_rejected: int = 0
    alerts_processed: int = 0
    alerts_accepted: int = 0
    alerts_rejected: int = 0
    timestamp: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Result of a federated search query."""
    query: dict[str, Any] = field(default_factory=dict)
    matches: list[dict[str, Any]] = field(default_factory=list)
    sites_searched: list[str] = field(default_factory=list)
    sites_failed: list[str] = field(default_factory=list)
    total_matches: int = 0
    search_time_ms: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.sites_failed) == 0


# ---------------------------------------------------------------------------
# FederationManager
# ---------------------------------------------------------------------------


class FederationManager:
    """Manages connections to federated Tritium sites.

    The FederationManager is the central coordinator for multi-site
    federation.  It maintains a registry of known sites, their trust
    levels and share policies, and provides methods for syncing targets,
    receiving data, and running federated searches.

    Parameters
    ----------
    local_site_id:
        Unique identifier for this Tritium instance.
    local_site_name:
        Human-readable name for this instance.
    """

    def __init__(
        self,
        local_site_id: str = "",
        local_site_name: str = "Local",
    ) -> None:
        self._local_site_id = local_site_id or str(uuid.uuid4())
        self._local_site_name = local_site_name
        self._sites: dict[str, FederatedSite] = {}
        # Received targets from remote sites, keyed by (source_site_id, target_id)
        self._received_targets: dict[tuple[str, str], dict[str, Any]] = {}
        # Outbound sync log: site_id -> list of SyncResult
        self._sync_log: list[SyncResult] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def local_site_id(self) -> str:
        return self._local_site_id

    @property
    def local_site_name(self) -> str:
        return self._local_site_name

    @property
    def site_count(self) -> int:
        return len(self._sites)

    # ------------------------------------------------------------------
    # Site management
    # ------------------------------------------------------------------

    def add_site(self, site: FederatedSite) -> None:
        """Register a remote site in the federation.

        Raises ValueError if a site with the same site_id already exists.
        """
        if site.site_id in self._sites:
            raise ValueError(f"Site '{site.site_id}' already registered")
        self._sites[site.site_id] = site
        logger.info("Federation: added site '%s' (%s) trust=%s",
                     site.name, site.site_id, site.trust_level.value)

    def remove_site(self, site_id: str) -> bool:
        """Remove a site from the federation.

        Returns True if the site was found and removed.
        """
        if site_id in self._sites:
            del self._sites[site_id]
            # Clean up received targets from this site
            keys_to_remove = [
                k for k in self._received_targets if k[0] == site_id
            ]
            for k in keys_to_remove:
                del self._received_targets[k]
            logger.info("Federation: removed site '%s'", site_id)
            return True
        return False

    def get_site(self, site_id: str) -> FederatedSite | None:
        """Look up a site by ID."""
        return self._sites.get(site_id)

    def list_sites(self, *, include_disabled: bool = False) -> list[FederatedSite]:
        """Return all registered sites.

        By default only enabled sites are returned.
        """
        if include_disabled:
            return list(self._sites.values())
        return [s for s in self._sites.values() if s.enabled]

    def update_trust(self, site_id: str, trust_level: TrustLevel) -> bool:
        """Change the trust level for a site.

        Also resets the share policy to the default for the new trust level.
        Returns False if the site is not found.
        """
        site = self._sites.get(site_id)
        if site is None:
            return False
        site.trust_level = trust_level
        site.share_policy = default_policy_for_trust(trust_level)
        logger.info("Federation: site '%s' trust updated to %s",
                     site_id, trust_level.value)
        return True

    def update_policy(self, site_id: str, policy: SharePolicy) -> bool:
        """Set a custom share policy for a site.

        Returns False if the site is not found.
        """
        site = self._sites.get(site_id)
        if site is None:
            return False
        site.share_policy = policy
        return True

    # ------------------------------------------------------------------
    # Target sync — outbound
    # ------------------------------------------------------------------

    def sync_targets(
        self,
        site_id: str,
        targets: list[dict[str, Any]],
    ) -> SyncResult:
        """Share local targets with a remote site, filtered by policy.

        This method applies the site's share policy to filter and
        optionally redact the targets before they would be sent.  The
        actual network transport is not included — callers should take
        the returned SyncResult.targets_accepted count and the filtered
        list to send via their chosen transport (HTTP, MQTT, etc.).

        Parameters
        ----------
        site_id:
            ID of the remote site to sync with.
        targets:
            List of target dicts (must have at least 'target_id').

        Returns
        -------
        SyncResult with counts and the list of accepted targets in
        the ``_outbound_targets`` attribute (non-standard, for callers).
        """
        result = SyncResult(site_id=site_id, direction="outbound")

        site = self._sites.get(site_id)
        if site is None:
            result.errors.append(f"Unknown site: {site_id}")
            return result

        if not site.can_send():
            result.errors.append(
                f"Cannot send to site '{site_id}' "
                f"(trust={site.trust_level.value}, enabled={site.enabled})"
            )
            return result

        policy = site.effective_policy
        if not policy.allows_category(ShareCategory.TARGETS):
            result.errors.append(
                f"Share policy for '{site_id}' does not allow targets"
            )
            return result

        accepted: list[dict[str, Any]] = []
        for t in targets:
            result.targets_processed += 1
            filtered = policy.filter_target(t)
            if filtered is not None:
                # Tag with source site info
                filtered["_source_site_id"] = self._local_site_id
                filtered["_source_site_name"] = self._local_site_name
                filtered["_shared_at"] = time.time()
                accepted.append(filtered)
                result.targets_accepted += 1
            else:
                result.targets_rejected += 1

        # Apply max cap
        if policy.max_targets_per_sync > 0:
            overflow = len(accepted) - policy.max_targets_per_sync
            if overflow > 0:
                accepted = accepted[:policy.max_targets_per_sync]
                result.targets_rejected += overflow
                result.targets_accepted -= overflow

        # Update last_seen
        site.last_seen = time.time()

        # Store in sync log
        self._sync_log.append(result)

        # Attach filtered targets for caller (not part of the dataclass)
        result._outbound_targets = accepted  # type: ignore[attr-defined]

        logger.info(
            "Federation: sync_targets to '%s': %d/%d accepted",
            site_id, result.targets_accepted, result.targets_processed,
        )
        return result

    def get_outbound_targets(self, result: SyncResult) -> list[dict[str, Any]]:
        """Extract the filtered outbound targets from a SyncResult.

        Convenience method for callers that need the actual target list
        after sync_targets() has filtered/redacted them.
        """
        return getattr(result, "_outbound_targets", [])

    # ------------------------------------------------------------------
    # Target sync — inbound
    # ------------------------------------------------------------------

    def receive_targets(
        self,
        site_id: str,
        targets: list[dict[str, Any]],
    ) -> SyncResult:
        """Receive shared targets from a remote site.

        Validates the source site exists and is trusted, then stores the
        targets in the received target pool.

        Parameters
        ----------
        site_id:
            ID of the remote site sending the targets.
        targets:
            List of target dicts from the remote site.

        Returns
        -------
        SyncResult with acceptance/rejection counts.
        """
        result = SyncResult(site_id=site_id, direction="inbound")

        site = self._sites.get(site_id)
        if site is None:
            result.errors.append(f"Unknown site: {site_id}")
            return result

        if not site.can_receive():
            result.errors.append(
                f"Cannot receive from site '{site_id}' "
                f"(trust={site.trust_level.value}, enabled={site.enabled})"
            )
            return result

        for t in targets:
            result.targets_processed += 1
            target_id = t.get("target_id")
            if not target_id:
                result.targets_rejected += 1
                continue

            # Store with provenance
            entry = dict(t)
            entry["_source_site_id"] = site_id
            entry["_received_at"] = time.time()

            # For LIMITED trust, strip sensitive fields from inbound too
            if site.trust_level == TrustLevel.LIMITED:
                entry.pop("identifiers", None)
                entry.pop("dossier_id", None)

            self._received_targets[(site_id, target_id)] = entry
            result.targets_accepted += 1

        # Update last_seen
        site.last_seen = time.time()

        self._sync_log.append(result)

        logger.info(
            "Federation: receive_targets from '%s': %d/%d accepted",
            site_id, result.targets_accepted, result.targets_processed,
        )
        return result

    # ------------------------------------------------------------------
    # Alert sync
    # ------------------------------------------------------------------

    def sync_alerts(
        self,
        site_id: str,
        alerts: list[dict[str, Any]],
    ) -> SyncResult:
        """Share alerts with a remote site, filtered by policy.

        Parameters
        ----------
        site_id:
            ID of the remote site.
        alerts:
            List of alert dicts.

        Returns
        -------
        SyncResult with counts.
        """
        result = SyncResult(site_id=site_id, direction="outbound")

        site = self._sites.get(site_id)
        if site is None:
            result.errors.append(f"Unknown site: {site_id}")
            return result

        if not site.can_send():
            result.errors.append(
                f"Cannot send alerts to site '{site_id}'"
            )
            return result

        policy = site.effective_policy
        if not policy.allows_category(ShareCategory.ALERTS):
            result.errors.append(
                f"Share policy for '{site_id}' does not allow alerts"
            )
            return result

        for a in alerts:
            result.alerts_processed += 1
            # Tag with source
            a_copy = dict(a)
            a_copy["_source_site_id"] = self._local_site_id
            a_copy["_shared_at"] = time.time()
            result.alerts_accepted += 1

        site.last_seen = time.time()
        self._sync_log.append(result)
        return result

    # ------------------------------------------------------------------
    # Federated search
    # ------------------------------------------------------------------

    def federated_search(
        self,
        query: dict[str, Any],
    ) -> SearchResult:
        """Search across all federated sites' received targets.

        The query dict supports the following keys (all optional):
          - target_id: exact match on target ID
          - classification: exact match on classification
          - alliance: exact match on alliance
          - source: exact match on source sensor type
          - name: substring match on target name (case-insensitive)
          - min_confidence: minimum confidence threshold
          - source_site_id: restrict to a specific source site

        Parameters
        ----------
        query:
            Dictionary of search criteria.

        Returns
        -------
        SearchResult with matching targets and metadata.
        """
        start = time.time()
        search_result = SearchResult(query=dict(query))

        # Determine which sites to search
        target_site = query.get("source_site_id")
        sites_to_search: list[str] = []

        if target_site:
            if target_site in self._sites:
                sites_to_search = [target_site]
            else:
                search_result.sites_failed.append(target_site)
        else:
            sites_to_search = [
                s.site_id for s in self._sites.values()
                if s.enabled and s.can_receive()
            ]

        search_result.sites_searched = list(sites_to_search)

        # Search through received targets
        for key, target in self._received_targets.items():
            source_site, _ = key

            if source_site not in sites_to_search:
                continue

            if self._matches_query(target, query):
                search_result.matches.append(dict(target))

        search_result.total_matches = len(search_result.matches)
        search_result.search_time_ms = (time.time() - start) * 1000.0
        return search_result

    def _matches_query(
        self, target: dict[str, Any], query: dict[str, Any]
    ) -> bool:
        """Check if a target matches the search query."""
        # target_id: exact match
        if "target_id" in query:
            if target.get("target_id") != query["target_id"]:
                return False

        # classification: exact match
        if "classification" in query:
            if target.get("classification") != query["classification"]:
                return False

        # alliance: exact match
        if "alliance" in query:
            if target.get("alliance") != query["alliance"]:
                return False

        # source: exact match
        if "source" in query:
            if target.get("source") != query["source"]:
                return False

        # name: substring (case-insensitive)
        if "name" in query:
            name = target.get("name", "")
            if query["name"].lower() not in name.lower():
                return False

        # min_confidence: threshold
        if "min_confidence" in query:
            confidence = target.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)):
                return False
            if confidence < query["min_confidence"]:
                return False

        return True

    # ------------------------------------------------------------------
    # Received target management
    # ------------------------------------------------------------------

    def get_received_targets(
        self, site_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get all received targets, optionally filtered by source site.

        Parameters
        ----------
        site_id:
            If provided, only return targets from this specific site.
        """
        if site_id is not None:
            return [
                dict(v)
                for (s, _), v in self._received_targets.items()
                if s == site_id
            ]
        return [dict(v) for v in self._received_targets.values()]

    def get_received_target(
        self, site_id: str, target_id: str,
    ) -> dict[str, Any] | None:
        """Look up a specific received target by site and target ID."""
        entry = self._received_targets.get((site_id, target_id))
        return dict(entry) if entry is not None else None

    def clear_received_targets(self, site_id: str | None = None) -> int:
        """Remove received targets, optionally only from one site.

        Returns the number of targets removed.
        """
        if site_id is None:
            count = len(self._received_targets)
            self._received_targets.clear()
            return count

        keys = [k for k in self._received_targets if k[0] == site_id]
        for k in keys:
            del self._received_targets[k]
        return len(keys)

    def purge_stale_targets(self, max_age_s: float = 300.0) -> int:
        """Remove received targets older than max_age_s seconds.

        Returns the number of targets purged.
        """
        now = time.time()
        stale_keys = [
            k for k, v in self._received_targets.items()
            if now - v.get("_received_at", 0) > max_age_s
        ]
        for k in stale_keys:
            del self._received_targets[k]
        return len(stale_keys)

    # ------------------------------------------------------------------
    # Sync log / stats
    # ------------------------------------------------------------------

    def get_sync_log(
        self, site_id: str | None = None, limit: int = 100,
    ) -> list[SyncResult]:
        """Get recent sync results, optionally filtered by site."""
        if site_id is not None:
            filtered = [r for r in self._sync_log if r.site_id == site_id]
        else:
            filtered = list(self._sync_log)
        return filtered[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics for the federation."""
        sites = list(self._sites.values())
        return {
            "local_site_id": self._local_site_id,
            "local_site_name": self._local_site_name,
            "total_sites": len(sites),
            "enabled_sites": sum(1 for s in sites if s.enabled),
            "trust_levels": {
                level.value: sum(1 for s in sites if s.trust_level == level)
                for level in TrustLevel
            },
            "received_targets": len(self._received_targets),
            "sync_operations": len(self._sync_log),
        }

    def export_config(self) -> dict[str, Any]:
        """Export the federation configuration as a serializable dict."""
        sites = []
        for site in self._sites.values():
            policy = site.effective_policy
            sites.append({
                "site_id": site.site_id,
                "url": site.url,
                "name": site.name,
                "trust_level": site.trust_level.value,
                "enabled": site.enabled,
                "last_seen": site.last_seen,
                "metadata": site.metadata,
                "share_policy": {
                    "categories": [c.value for c in policy.categories],
                    "allowed_classifications": list(policy.allowed_classifications),
                    "allowed_alliances": list(policy.allowed_alliances),
                    "min_confidence": policy.min_confidence,
                    "max_targets_per_sync": policy.max_targets_per_sync,
                    "redact_identifiers": policy.redact_identifiers,
                    "redact_dossier_ids": policy.redact_dossier_ids,
                },
            })
        return {
            "local_site_id": self._local_site_id,
            "local_site_name": self._local_site_name,
            "sites": sites,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def sync_targets(
    manager: FederationManager,
    site_id: str,
    targets: list[dict[str, Any]],
) -> SyncResult:
    """Convenience: share targets with a remote site via a FederationManager."""
    return manager.sync_targets(site_id, targets)


def receive_targets(
    manager: FederationManager,
    site_id: str,
    targets: list[dict[str, Any]],
) -> SyncResult:
    """Convenience: receive shared targets from a remote site."""
    return manager.receive_targets(site_id, targets)


def federated_search(
    manager: FederationManager,
    query: dict[str, Any],
) -> SearchResult:
    """Convenience: search across all federated sites."""
    return manager.federated_search(query)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "TrustLevel",
    "ShareCategory",
    "SharePolicy",
    "FederatedSite",
    "FederationManager",
    "SyncResult",
    "SearchResult",
    "default_policy_for_trust",
    "sync_targets",
    "receive_targets",
    "federated_search",
]

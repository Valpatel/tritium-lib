# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""AssociationNetwork — build and analyze networks of associated targets.

Builds a weighted graph of target relationships from four evidence types:

  1. **Co-location** — two targets observed at the same place/time
  2. **Communication** — BLE pairing, WiFi association metadata
  3. **Travel pattern** — shared routes or schedules
  4. **Device sharing** — seen on the same WiFi network

The resulting graph enables group discovery, key-player identification,
and association-strength queries.  Integrates with the existing COMINT
module (``comint.CommNetwork``) and evidence models (``evidence.models``).

Usage
-----
    from tritium_lib.intelligence.association_network import (
        AssociationNetwork, NetworkAnalyzer, build_from_tracking,
    )

    net = AssociationNetwork()
    net.add_association("ble_AA:BB", "det_person_1", "co_location",
                        score=0.8, evidence={"distance_m": 2.5})
    analyzer = NetworkAnalyzer(net)
    groups = analyzer.find_groups()
    key_players = analyzer.find_key_players()
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ── Enums & constants ────────────────────────────────────────────────

class EvidenceKind(str, Enum):
    """Category of evidence supporting an association."""

    CO_LOCATION = "co_location"
    COMMUNICATION = "communication"
    TRAVEL_PATTERN = "travel_pattern"
    DEVICE_SHARING = "device_sharing"


# Base weights per evidence kind — used for combined scoring
_KIND_WEIGHTS: dict[EvidenceKind, float] = {
    EvidenceKind.CO_LOCATION: 0.6,
    EvidenceKind.COMMUNICATION: 0.9,
    EvidenceKind.TRAVEL_PATTERN: 0.7,
    EvidenceKind.DEVICE_SHARING: 0.8,
}


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class Association:
    """A weighted link between two targets with supporting evidence.

    Attributes:
        target_a: First target identifier.
        target_b: Second target identifier.
        kind: Category of evidence (co_location, communication, etc.).
        score: Strength of this association, 0.0–1.0.
        timestamp: When the association was observed (unix time).
        evidence: Arbitrary metadata supporting the association.
            For co_location: distance_m, overlap_seconds, zone_id.
            For communication: medium (ble, wifi), link_count.
            For travel_pattern: route_similarity, schedule_overlap.
            For device_sharing: ssid, network_id.
    """

    target_a: str
    target_b: str
    kind: str  # one of EvidenceKind values or free-form string
    score: float = 0.5
    timestamp: float = field(default_factory=time.time)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, self.score))

    @property
    def pair_key(self) -> tuple[str, str]:
        """Canonical (sorted) pair key for undirected association."""
        return tuple(sorted([self.target_a, self.target_b]))  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_a": self.target_a,
            "target_b": self.target_b,
            "kind": self.kind,
            "score": round(self.score, 4),
            "timestamp": self.timestamp,
            "evidence": self.evidence,
        }


@dataclass
class AssociationSummary:
    """Aggregated summary of all associations between two targets.

    Attributes:
        target_a: First target identifier.
        target_b: Second target identifier.
        combined_score: Overall association strength (0.0–1.0).
        association_count: Number of individual association observations.
        kinds: Set of evidence kinds observed.
        first_seen: Earliest association timestamp.
        last_seen: Latest association timestamp.
    """

    target_a: str
    target_b: str
    combined_score: float = 0.0
    association_count: int = 0
    kinds: set[str] = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_a": self.target_a,
            "target_b": self.target_b,
            "combined_score": round(self.combined_score, 4),
            "association_count": self.association_count,
            "kinds": sorted(self.kinds),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class TargetGroup:
    """A cluster of associated targets.

    Attributes:
        group_id: Unique integer label for this group.
        members: Set of target IDs in the group.
        internal_score: Average association strength within the group.
        evidence_kinds: Set of evidence kinds linking group members.
    """

    group_id: int
    members: set[str] = field(default_factory=set)
    internal_score: float = 0.0
    evidence_kinds: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "members": sorted(self.members),
            "member_count": len(self.members),
            "internal_score": round(self.internal_score, 4),
            "evidence_kinds": sorted(self.evidence_kinds),
        }


@dataclass
class KeyPlayer:
    """A target identified as central in the association network.

    Attributes:
        target_id: The target identifier.
        degree: Number of distinct associates.
        weighted_degree: Sum of association strengths.
        betweenness: Approximate betweenness centrality (0.0–1.0).
        group_count: Number of groups this target connects.
        overall_score: Composite key-player score (0.0–1.0).
    """

    target_id: str
    degree: int = 0
    weighted_degree: float = 0.0
    betweenness: float = 0.0
    group_count: int = 0
    overall_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "degree": self.degree,
            "weighted_degree": round(self.weighted_degree, 4),
            "betweenness": round(self.betweenness, 4),
            "group_count": self.group_count,
            "overall_score": round(self.overall_score, 4),
        }


@dataclass
class WeakLink:
    """A connection between targets or groups that is fragile.

    Attributes:
        target_a: First endpoint.
        target_b: Second endpoint.
        score: Association strength (lower = weaker).
        is_bridge: True if this link connects otherwise-separate groups.
        group_a: Group ID of target_a (if groups computed).
        group_b: Group ID of target_b (if groups computed).
    """

    target_a: str
    target_b: str
    score: float = 0.0
    is_bridge: bool = False
    group_a: int = -1
    group_b: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_a": self.target_a,
            "target_b": self.target_b,
            "score": round(self.score, 4),
            "is_bridge": self.is_bridge,
            "group_a": self.group_a,
            "group_b": self.group_b,
        }


# ── AssociationNetwork — the weighted graph ──────────────────────────

class AssociationNetwork:
    """Weighted graph of target relationships.

    Thread-safe.  Stores individual Association observations and maintains
    aggregated adjacency data for efficient queries.

    Parameters
    ----------
    decay_hours:
        Exponential half-life for score decay in hours.  Older associations
        contribute less to combined scores.  0 disables decay.
    """

    def __init__(self, decay_hours: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._decay_half_life = decay_hours * 3600.0 if decay_hours > 0 else 0.0

        # All association observations
        self._associations: list[Association] = []

        # Adjacency: target_id -> {peer_id -> list of Association}
        self._adj: dict[str, dict[str, list[Association]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Per-target stats
        self._target_assoc_count: dict[str, int] = defaultdict(int)
        self._target_first_seen: dict[str, float] = {}
        self._target_last_seen: dict[str, float] = {}

    # ── Properties ────────────────────────────────────────────────────

    @property
    def target_count(self) -> int:
        """Number of unique targets in the network."""
        with self._lock:
            return len(self._adj)

    @property
    def association_count(self) -> int:
        """Total number of recorded association observations."""
        with self._lock:
            return len(self._associations)

    @property
    def edge_count(self) -> int:
        """Number of unique target-pair edges."""
        with self._lock:
            seen: set[tuple[str, str]] = set()
            for target, peers in self._adj.items():
                for peer in peers:
                    key = tuple(sorted([target, peer]))
                    seen.add(key)  # type: ignore[arg-type]
            return len(seen)

    # ── Add associations ──────────────────────────────────────────────

    def add_association(
        self,
        target_a: str,
        target_b: str,
        kind: str,
        score: float = 0.5,
        timestamp: float | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> Association:
        """Record an association between two targets.

        Args:
            target_a: First target ID.
            target_b: Second target ID.
            kind: Evidence kind (co_location, communication, etc.).
            score: Association strength 0.0–1.0.
            timestamp: Unix timestamp (default now).
            evidence: Supporting metadata.

        Returns:
            The created Association object.
        """
        assoc = Association(
            target_a=target_a,
            target_b=target_b,
            kind=kind,
            score=score,
            timestamp=timestamp or time.time(),
            evidence=evidence or {},
        )
        self._record(assoc)
        return assoc

    def add(self, assoc: Association) -> None:
        """Record a pre-built Association object."""
        self._record(assoc)

    def add_many(self, associations: Sequence[Association]) -> None:
        """Record multiple associations at once."""
        for a in associations:
            self._record(a)

    def _record(self, assoc: Association) -> None:
        """Internal: store association and update indices."""
        with self._lock:
            self._associations.append(assoc)
            self._adj[assoc.target_a][assoc.target_b].append(assoc)
            self._adj[assoc.target_b][assoc.target_a].append(assoc)

            for tid in (assoc.target_a, assoc.target_b):
                self._target_assoc_count[tid] += 1
                if tid not in self._target_first_seen:
                    self._target_first_seen[tid] = assoc.timestamp
                self._target_first_seen[tid] = min(
                    self._target_first_seen[tid], assoc.timestamp
                )
                self._target_last_seen[tid] = max(
                    self._target_last_seen.get(tid, 0.0), assoc.timestamp
                )

    # ── Query ─────────────────────────────────────────────────────────

    def get_targets(self) -> list[str]:
        """Return all target IDs in the network, sorted."""
        with self._lock:
            return sorted(self._adj.keys())

    def get_peers(self, target_id: str) -> list[str]:
        """Return all peers for a target, sorted."""
        with self._lock:
            return sorted(self._adj.get(target_id, {}).keys())

    def get_associations_between(
        self, target_a: str, target_b: str
    ) -> list[Association]:
        """Return all association observations between two targets."""
        with self._lock:
            return list(self._adj.get(target_a, {}).get(target_b, []))

    def get_associations_for(
        self,
        target_id: str,
        kind: str | None = None,
        since: float | None = None,
    ) -> list[Association]:
        """Return all associations involving a target.

        Args:
            target_id: Target to query.
            kind: Optional filter by evidence kind.
            since: Optional minimum timestamp filter.
        """
        with self._lock:
            result: list[Association] = []
            for peer_assocs in self._adj.get(target_id, {}).values():
                for a in peer_assocs:
                    if kind is not None and a.kind != kind:
                        continue
                    if since is not None and a.timestamp < since:
                        continue
                    result.append(a)
            return sorted(result, key=lambda a: a.timestamp)

    def get_target_stats(self, target_id: str) -> dict[str, Any]:
        """Return statistics for a specific target."""
        with self._lock:
            peers = self._adj.get(target_id, {})
            kinds: set[str] = set()
            for peer_assocs in peers.values():
                for a in peer_assocs:
                    kinds.add(a.kind)

            return {
                "target_id": target_id,
                "peer_count": len(peers),
                "association_count": self._target_assoc_count.get(target_id, 0),
                "first_seen": self._target_first_seen.get(target_id),
                "last_seen": self._target_last_seen.get(target_id),
                "evidence_kinds": sorted(kinds),
            }

    def strength(self, target_a: str, target_b: str) -> float:
        """Compute the combined association strength between two targets.

        Combines all observations between the pair, weighting by evidence
        kind and applying optional time decay.

        Returns:
            Association strength 0.0–1.0.  Returns 0.0 if no association.
        """
        with self._lock:
            assocs = self._adj.get(target_a, {}).get(target_b, [])
            if not assocs:
                return 0.0
            return self._compute_combined_score(assocs)

    def _compute_combined_score(self, assocs: list[Association]) -> float:
        """Compute a combined score from a list of associations.

        Uses a softmax-like combination: each observation contributes
        to pulling the combined score toward 1.0.  More observations
        and higher individual scores mean a stronger combined result.
        Time decay is applied when configured — older observations
        contribute less to the final score (not just to relative weight).
        """
        if not assocs:
            return 0.0

        now = time.time()
        effective_scores: list[float] = []

        for a in assocs:
            # Kind-based weight
            try:
                kind_enum = EvidenceKind(a.kind)
                kind_weight = _KIND_WEIGHTS.get(kind_enum, 0.5)
            except ValueError:
                kind_weight = 0.5

            # Time decay — directly reduces the effective score
            decay = 1.0
            if self._decay_half_life > 0:
                elapsed = max(0.0, now - a.timestamp)
                decay = math.exp(-math.log(2) / self._decay_half_life * elapsed)

            effective_scores.append(a.score * kind_weight * decay)

        if not effective_scores:
            return 0.0

        # Average effective score (kind-weighted, time-decayed)
        avg = sum(effective_scores) / len(effective_scores)
        # Diminishing-returns boost: more observations increase confidence
        count_boost = 1.0 - math.exp(-0.3 * len(assocs))
        combined = avg * 0.6 + count_boost * 0.4
        return max(0.0, min(1.0, combined))

    def summarize_pair(self, target_a: str, target_b: str) -> AssociationSummary:
        """Build a summary of all associations between two targets."""
        with self._lock:
            assocs = self._adj.get(target_a, {}).get(target_b, [])

        if not assocs:
            return AssociationSummary(target_a=target_a, target_b=target_b)

        kinds: set[str] = set()
        timestamps: list[float] = []
        for a in assocs:
            kinds.add(a.kind)
            timestamps.append(a.timestamp)

        return AssociationSummary(
            target_a=target_a,
            target_b=target_b,
            combined_score=self.strength(target_a, target_b),
            association_count=len(assocs),
            kinds=kinds,
            first_seen=min(timestamps),
            last_seen=max(timestamps),
        )

    # ── Graph export ──────────────────────────────────────────────────

    def to_graph_dict(self) -> dict[str, Any]:
        """Export the network as nodes and weighted edges.

        Returns:
            Dict with 'nodes' and 'edges' lists, plus summary counts.
        """
        with self._lock:
            nodes = []
            for tid in sorted(self._adj.keys()):
                nodes.append({
                    "id": tid,
                    "peer_count": len(self._adj[tid]),
                    "association_count": self._target_assoc_count.get(tid, 0),
                    "first_seen": self._target_first_seen.get(tid),
                    "last_seen": self._target_last_seen.get(tid),
                })

            edges = []
            seen: set[tuple[str, str]] = set()
            for target, peers in self._adj.items():
                for peer, assocs in peers.items():
                    key = tuple(sorted([target, peer]))
                    if key in seen:
                        continue
                    seen.add(key)  # type: ignore[arg-type]

                    kinds: set[str] = set()
                    for a in assocs:
                        kinds.add(a.kind)

                    edges.append({
                        "source": key[0],
                        "target": key[1],
                        "weight": self._compute_combined_score(assocs),
                        "count": len(assocs),
                        "kinds": sorted(kinds),
                    })

            return {
                "nodes": nodes,
                "edges": edges,
                "target_count": len(nodes),
                "edge_count": len(edges),
            }

    # ── Maintenance ───────────────────────────────────────────────────

    def prune(self, before: float) -> int:
        """Remove associations older than a timestamp.

        Returns the number of associations removed.
        """
        with self._lock:
            original = len(self._associations)
            kept = [a for a in self._associations if a.timestamp >= before]
            removed = original - len(kept)
            if removed > 0:
                self._rebuild(kept)
            return removed

    def clear(self) -> None:
        """Remove all data."""
        with self._lock:
            self._associations.clear()
            self._adj.clear()
            self._target_assoc_count.clear()
            self._target_first_seen.clear()
            self._target_last_seen.clear()

    def _rebuild(self, associations: list[Association]) -> None:
        """Rebuild all indices from a filtered list.  Caller holds lock."""
        self._associations.clear()
        self._adj.clear()
        self._target_assoc_count.clear()
        self._target_first_seen.clear()
        self._target_last_seen.clear()

        # Temporarily release lock pattern not needed since caller holds lock
        for assoc in associations:
            self._associations.append(assoc)
            self._adj[assoc.target_a][assoc.target_b].append(assoc)
            self._adj[assoc.target_b][assoc.target_a].append(assoc)

            for tid in (assoc.target_a, assoc.target_b):
                self._target_assoc_count[tid] += 1
                if tid not in self._target_first_seen:
                    self._target_first_seen[tid] = assoc.timestamp
                self._target_first_seen[tid] = min(
                    self._target_first_seen[tid], assoc.timestamp
                )
                self._target_last_seen[tid] = max(
                    self._target_last_seen.get(tid, 0.0), assoc.timestamp
                )


# ── NetworkAnalyzer — high-level graph analytics ─────────────────────

class NetworkAnalyzer:
    """Analyze an AssociationNetwork to find clusters, key players, and weak links.

    Parameters
    ----------
    network:
        The AssociationNetwork to analyze.  If None a new empty one is created.
    """

    def __init__(self, network: AssociationNetwork | None = None) -> None:
        self.network = network or AssociationNetwork()

    # ── Associates ────────────────────────────────────────────────────

    def find_associates(
        self,
        target_id: str,
        min_score: float = 0.0,
        limit: int = 50,
    ) -> list[AssociationSummary]:
        """Find all associates of a target, ranked by association strength.

        Args:
            target_id: The target to query.
            min_score: Minimum combined score to include.
            limit: Maximum number of results.

        Returns:
            Ranked list of AssociationSummary objects (strongest first).
        """
        peers = self.network.get_peers(target_id)
        results: list[AssociationSummary] = []

        for peer in peers:
            summary = self.network.summarize_pair(target_id, peer)
            if summary.combined_score >= min_score:
                results.append(summary)

        results.sort(key=lambda s: s.combined_score, reverse=True)
        return results[:limit]

    # ── Group detection (label propagation) ───────────────────────────

    def find_groups(
        self,
        min_size: int = 2,
        min_edge_score: float = 0.1,
        max_iterations: int = 100,
    ) -> list[TargetGroup]:
        """Detect clusters of associated targets using weighted label propagation.

        Mirrors the algorithm used in ``comint.CommAnalyzer.find_communities``
        but operates on association strength rather than communication counts.

        Args:
            min_size: Minimum group size to return.
            min_edge_score: Minimum edge weight to consider for propagation.
            max_iterations: Maximum iterations before stopping.

        Returns:
            List of TargetGroup objects sorted by size descending.
        """
        graph = self.network.to_graph_dict()
        nodes = [n["id"] for n in graph["nodes"]]
        if not nodes:
            return []

        # Build weighted adjacency
        adj: dict[str, dict[str, float]] = defaultdict(dict)
        edge_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)

        for edge in graph["edges"]:
            if edge["weight"] < min_edge_score:
                continue
            adj[edge["source"]][edge["target"]] = edge["weight"]
            adj[edge["target"]][edge["source"]] = edge["weight"]
            key = tuple(sorted([edge["source"], edge["target"]]))
            for k in edge.get("kinds", []):
                edge_kinds[key].add(k)  # type: ignore[arg-type]

        # Label propagation
        labels: dict[str, int] = {node: i for i, node in enumerate(nodes)}

        for _ in range(max_iterations):
            changed = False
            for node in nodes:
                neighbours = adj.get(node, {})
                if not neighbours:
                    continue

                label_votes: dict[int, float] = defaultdict(float)
                for peer, weight in neighbours.items():
                    label_votes[labels[peer]] += weight

                if not label_votes:
                    continue

                best = max(label_votes, key=lambda l: label_votes[l])
                if labels[node] != best:
                    labels[node] = best
                    changed = True

            if not changed:
                break

        # Group by label
        groups_map: dict[int, set[str]] = defaultdict(set)
        for node, label in labels.items():
            groups_map[label].add(node)

        results: list[TargetGroup] = []
        for label, members in groups_map.items():
            if len(members) < min_size:
                continue

            # Compute internal score and evidence kinds
            internal_scores: list[float] = []
            kinds: set[str] = set()
            for edge in graph["edges"]:
                if edge["source"] in members and edge["target"] in members:
                    internal_scores.append(edge["weight"])
                    for k in edge.get("kinds", []):
                        kinds.add(k)

            avg_score = (
                sum(internal_scores) / len(internal_scores)
                if internal_scores
                else 0.0
            )

            results.append(TargetGroup(
                group_id=label,
                members=members,
                internal_score=avg_score,
                evidence_kinds=kinds,
            ))

        results.sort(key=lambda g: len(g.members), reverse=True)
        return results

    # ── Key players ──────────────────────────────────────────────────

    def find_key_players(
        self,
        limit: int = 10,
        groups: list[TargetGroup] | None = None,
    ) -> list[KeyPlayer]:
        """Identify the most central and connected targets in the network.

        Computes degree, weighted degree, approximate betweenness centrality,
        and group connectivity to produce a composite key-player score.

        Args:
            limit: Maximum number of key players to return.
            groups: Pre-computed groups (calls find_groups if None).

        Returns:
            List of KeyPlayer objects sorted by overall_score descending.
        """
        graph = self.network.to_graph_dict()
        if not graph["nodes"]:
            return []

        if groups is None:
            groups = self.find_groups()

        # Build target -> group membership map
        target_groups: dict[str, set[int]] = defaultdict(set)
        for g in groups:
            for member in g.members:
                target_groups[member].add(g.group_id)

        # Build weighted adjacency
        adj: dict[str, dict[str, float]] = defaultdict(dict)
        for edge in graph["edges"]:
            adj[edge["source"]][edge["target"]] = edge["weight"]
            adj[edge["target"]][edge["source"]] = edge["weight"]

        # Compute per-target metrics
        all_targets = [n["id"] for n in graph["nodes"]]
        max_degree = max(len(adj.get(t, {})) for t in all_targets) if all_targets else 1
        max_wdeg = 0.0

        raw: list[dict[str, Any]] = []
        for tid in all_targets:
            neighbours = adj.get(tid, {})
            degree = len(neighbours)
            wdeg = sum(neighbours.values())
            if wdeg > max_wdeg:
                max_wdeg = wdeg
            gc = len(target_groups.get(tid, set()))
            raw.append({
                "target_id": tid,
                "degree": degree,
                "weighted_degree": wdeg,
                "group_count": gc,
            })

        # Approximate betweenness: fraction of shortest paths through this node.
        # Full betweenness is O(n^3); we use a simplified heuristic based on
        # how many unique group pairs this target bridges.
        total_group_pairs = max(1, len(groups) * (len(groups) - 1) // 2)

        results: list[KeyPlayer] = []
        for entry in raw:
            tid = entry["target_id"]
            degree_norm = entry["degree"] / max(max_degree, 1)
            wdeg_norm = entry["weighted_degree"] / max(max_wdeg, 0.001)

            # Bridge betweenness: count group pairs connected through this node
            my_groups = target_groups.get(tid, set())
            bridged_pairs = len(my_groups) * (len(my_groups) - 1) // 2
            betweenness = min(1.0, bridged_pairs / max(total_group_pairs, 1))

            # Composite score: weighted combination
            overall = (
                degree_norm * 0.25
                + wdeg_norm * 0.35
                + betweenness * 0.25
                + min(1.0, entry["group_count"] / max(len(groups), 1)) * 0.15
            )

            results.append(KeyPlayer(
                target_id=tid,
                degree=entry["degree"],
                weighted_degree=entry["weighted_degree"],
                betweenness=betweenness,
                group_count=entry["group_count"],
                overall_score=min(1.0, overall),
            ))

        results.sort(key=lambda kp: kp.overall_score, reverse=True)
        return results[:limit]

    # ── Weak links ───────────────────────────────────────────────────

    def find_weak_links(
        self,
        max_score: float = 0.3,
        groups: list[TargetGroup] | None = None,
    ) -> list[WeakLink]:
        """Find fragile connections — low-strength edges that bridge groups.

        Args:
            max_score: Maximum edge weight to consider "weak".
            groups: Pre-computed groups (calls find_groups if None).

        Returns:
            List of WeakLink objects sorted by score ascending (weakest first).
        """
        graph = self.network.to_graph_dict()
        if not graph["edges"]:
            return []

        if groups is None:
            groups = self.find_groups()

        # Build target -> group_id map
        target_group: dict[str, int] = {}
        for g in groups:
            for member in g.members:
                target_group[member] = g.group_id

        results: list[WeakLink] = []
        for edge in graph["edges"]:
            if edge["weight"] > max_score:
                continue

            ga = target_group.get(edge["source"], -1)
            gb = target_group.get(edge["target"], -1)
            is_bridge = ga != gb and ga >= 0 and gb >= 0

            results.append(WeakLink(
                target_a=edge["source"],
                target_b=edge["target"],
                score=edge["weight"],
                is_bridge=is_bridge,
                group_a=ga,
                group_b=gb,
            ))

        results.sort(key=lambda w: w.score)
        return results

    # ── Reachability ─────────────────────────────────────────────────

    def reachable_within(
        self, target_id: str, max_hops: int = 2, min_score: float = 0.0
    ) -> set[str]:
        """Find all targets reachable within N hops.

        Args:
            target_id: Starting target.
            max_hops: Maximum hop count (1–5).
            min_score: Minimum edge strength to traverse.

        Returns:
            Set of reachable target IDs (excluding the start).
        """
        max_hops = max(1, min(max_hops, 5))
        visited: set[str] = set()
        current: set[str] = {target_id}

        for _ in range(max_hops):
            next_layer: set[str] = set()
            for tid in current:
                for peer in self.network.get_peers(tid):
                    if peer in visited or peer == target_id:
                        continue
                    if min_score > 0 and self.network.strength(tid, peer) < min_score:
                        continue
                    next_layer.add(peer)
            visited.update(next_layer)
            current = next_layer

        return visited

    # ── Statistics ────────────────────────────────────────────────────

    def get_statistics(self) -> dict[str, Any]:
        """Return summary statistics about the association network."""
        graph = self.network.to_graph_dict()

        kind_dist: dict[str, int] = defaultdict(int)
        weights: list[float] = []
        for edge in graph["edges"]:
            weights.append(edge["weight"])
            for k in edge.get("kinds", []):
                kind_dist[k] += 1

        avg_weight = sum(weights) / len(weights) if weights else 0.0
        avg_peers = 0.0
        if graph["target_count"] > 0:
            total_peers = sum(n["peer_count"] for n in graph["nodes"])
            avg_peers = total_peers / graph["target_count"]

        return {
            "target_count": graph["target_count"],
            "edge_count": graph["edge_count"],
            "association_count": self.network.association_count,
            "avg_edge_weight": round(avg_weight, 4),
            "avg_peers_per_target": round(avg_peers, 2),
            "evidence_kind_distribution": dict(kind_dist),
        }

    # ── Full export ──────────────────────────────────────────────────

    def export(self) -> dict[str, Any]:
        """Export the full network analysis for API/dashboard consumption."""
        groups = self.find_groups()
        return {
            "network": self.network.to_graph_dict(),
            "groups": [g.to_dict() for g in groups],
            "key_players": [kp.to_dict() for kp in self.find_key_players(groups=groups)],
            "weak_links": [w.to_dict() for w in self.find_weak_links(groups=groups)],
            "statistics": self.get_statistics(),
        }


# ── Builder — auto-build from tracking data ──────────────────────────

def build_from_tracking(
    targets: Sequence[Any],
    co_location_distance: float = 10.0,
    co_location_time: float = 60.0,
) -> AssociationNetwork:
    """Auto-build an association network from tracked target data.

    Examines a collection of targets and infers associations based on:
      - Co-location: targets within ``co_location_distance`` meters seen
        within ``co_location_time`` seconds of each other.
      - Device sharing: targets with matching correlated IDs or shared
        WiFi network info.
      - Travel pattern: targets with the same route or destination data.

    Args:
        targets: Sequence of target-like objects.  Each must have at least:
            ``target_id`` (str), ``position`` (tuple[float, float]),
            ``last_seen`` (float).
            Optional: ``source`` (str), ``asset_type`` (str),
            ``correlated_ids`` (list), plus any extra dict attributes.
        co_location_distance: Maximum distance in meters for co-location.
        co_location_time: Maximum time difference in seconds for co-location.

    Returns:
        A populated AssociationNetwork.
    """
    network = AssociationNetwork()

    # Convert targets to dicts for uniform access
    items: list[dict[str, Any]] = []
    for t in targets:
        if isinstance(t, dict):
            items.append(t)
        else:
            d: dict[str, Any] = {}
            for attr in (
                "target_id", "position", "last_seen", "source",
                "asset_type", "correlated_ids", "confirming_sources",
            ):
                val = getattr(t, attr, None)
                if val is not None:
                    d[attr] = val
            items.append(d)

    if len(items) < 2:
        return network

    # ── Co-location detection ──────────────────────────────────────
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a = items[i]
            b = items[j]

            tid_a = a.get("target_id", "")
            tid_b = b.get("target_id", "")
            if not tid_a or not tid_b:
                continue

            pos_a = a.get("position", (0.0, 0.0))
            pos_b = b.get("position", (0.0, 0.0))

            # Distance check
            dx = pos_a[0] - pos_b[0]
            dy = pos_a[1] - pos_b[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > co_location_distance:
                continue

            # Time check
            ts_a = a.get("last_seen", 0.0)
            ts_b = b.get("last_seen", 0.0)
            time_diff = abs(ts_a - ts_b)

            if time_diff > co_location_time:
                continue

            # Compute score: closer + more recent = stronger
            dist_score = max(0.0, 1.0 - dist / co_location_distance)
            time_score = max(0.0, 1.0 - time_diff / co_location_time)
            score = (dist_score * 0.6 + time_score * 0.4)

            network.add_association(
                target_a=tid_a,
                target_b=tid_b,
                kind=EvidenceKind.CO_LOCATION.value,
                score=score,
                timestamp=max(ts_a, ts_b),
                evidence={
                    "distance_m": round(dist, 2),
                    "time_diff_s": round(time_diff, 2),
                },
            )

    # ── Device sharing (correlated IDs) ───────────────────────────
    id_to_targets: dict[str, list[str]] = defaultdict(list)
    for item in items:
        tid = item.get("target_id", "")
        if not tid:
            continue
        for cid in item.get("correlated_ids", []):
            id_to_targets[cid].append(tid)

    for cid, tids in id_to_targets.items():
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                network.add_association(
                    target_a=tids[i],
                    target_b=tids[j],
                    kind=EvidenceKind.DEVICE_SHARING.value,
                    score=0.85,
                    evidence={"shared_id": cid},
                )

    # ── Communication (shared confirming sources) ─────────────────
    source_to_targets: dict[str, list[str]] = defaultdict(list)
    for item in items:
        tid = item.get("target_id", "")
        if not tid:
            continue
        for src in item.get("confirming_sources", []):
            source_to_targets[src].append(tid)

    for src, tids in source_to_targets.items():
        if len(tids) < 2:
            continue
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                network.add_association(
                    target_a=tids[i],
                    target_b=tids[j],
                    kind=EvidenceKind.COMMUNICATION.value,
                    score=0.6,
                    evidence={"shared_source": src},
                )

    return network


def build_from_comint(comm_network: Any) -> AssociationNetwork:
    """Build an AssociationNetwork from an existing CommNetwork.

    Converts COMINT communication observations into association
    evidence of kind ``communication``.

    Args:
        comm_network: A ``comint.CommNetwork`` instance.

    Returns:
        A populated AssociationNetwork.
    """
    network = AssociationNetwork()

    graph = comm_network.to_graph_dict()
    for edge in graph.get("edges", []):
        source = edge.get("source", "")
        target = edge.get("target", "")
        weight = edge.get("weight", 1)
        media = edge.get("media", {})

        if not source or not target:
            continue

        # Normalize weight to 0–1 range (log scale)
        score = min(1.0, math.log1p(weight) / 5.0)

        primary_medium = ""
        if media:
            primary_medium = max(media, key=lambda m: media[m])

        network.add_association(
            target_a=source,
            target_b=target,
            kind=EvidenceKind.COMMUNICATION.value,
            score=score,
            evidence={
                "link_count": weight,
                "media": media,
                "primary_medium": primary_medium,
            },
        )

    return network

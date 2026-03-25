# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""COMINT analyzer — metadata-only communication network analysis.

Builds and analyzes a graph of communication relationships from sensor
metadata.  **No message content is ever stored or processed.**

Algorithm notes
---------------
Community detection uses a label-propagation approach:
  1. Each node starts with its own label.
  2. Each iteration, every node adopts the label most common among its
     neighbours (weighted by link count).
  3. Converges when no node changes label.

Bridge detection identifies nodes whose removal would disconnect
communities — computed via a simplified betweenness heuristic:
  - A node is a bridge if it belongs to two or more communities, or
  - If removing it increases the number of connected components.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class CommLink:
    """A single observed communication between two identifiers.

    Metadata only — no content is stored.

    Attributes:
        source: Originating identifier (MAC, node_id, BSSID, etc.).
        target: Destination identifier.
        medium: Communication medium (ble, wifi, meshtastic, espnow).
        timestamp: Unix timestamp of the observation.
        metadata: Optional extra metadata (RSSI, channel, hop count).
            Must **never** contain message content.
    """

    source: str
    target: str
    medium: str  # ble | wifi | meshtastic | espnow
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = self.source.upper()
        self.target = self.target.upper()
        self.medium = self.medium.lower()
        # Paranoia: strip anything that looks like content
        self.metadata.pop("content", None)
        self.metadata.pop("message", None)
        self.metadata.pop("payload", None)
        self.metadata.pop("body", None)
        self.metadata.pop("text", None)


@dataclass
class CommunityResult:
    """Result of community detection.

    Attributes:
        community_id: Unique integer label for this community.
        members: Set of entity identifiers in this community.
        link_count: Number of internal links within the community.
        primary_medium: Most common medium used within the community.
    """

    community_id: int
    members: set[str] = field(default_factory=set)
    link_count: int = 0
    primary_medium: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_id": self.community_id,
            "members": sorted(self.members),
            "member_count": len(self.members),
            "link_count": self.link_count,
            "primary_medium": self.primary_medium,
        }


@dataclass
class BridgeEntity:
    """An entity that connects different communities.

    Attributes:
        entity_id: The bridge identifier.
        communities: Set of community IDs this entity belongs to.
        bridge_score: How critical this entity is as a connector (0-1).
        link_count: Total number of communication links for this entity.
    """

    entity_id: str
    communities: set[int] = field(default_factory=set)
    bridge_score: float = 0.0
    link_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "communities": sorted(self.communities),
            "community_count": len(self.communities),
            "bridge_score": round(self.bridge_score, 3),
            "link_count": self.link_count,
        }


@dataclass
class TimelineEntry:
    """A single entry in a communication timeline.

    Attributes:
        peer: The entity communicated with.
        medium: Communication medium.
        timestamp: When the communication occurred.
        direction: 'out' if entity was source, 'in' if target, 'both'.
        count: Number of communications at this timestamp (aggregated).
    """

    peer: str
    medium: str
    timestamp: float
    direction: str  # in | out | both
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer": self.peer,
            "medium": self.medium,
            "timestamp": self.timestamp,
            "direction": self.direction,
            "count": self.count,
        }


# ── CommNetwork — the communication graph ────────────────────────────

class CommNetwork:
    """Graph of communication relationships built from CommLink observations.

    Thread-safe.  Maintains adjacency lists, per-edge link counts, and
    temporal indices.  No message content is stored.

    Parameters
    ----------
    retention_hours:
        How long to keep raw links.  Default 24 hours.  Set to 0 for
        unlimited retention.
    """

    def __init__(self, retention_hours: float = 24.0) -> None:
        self._lock = threading.Lock()
        self._retention_s = retention_hours * 3600.0 if retention_hours > 0 else 0.0

        # Adjacency: entity -> {peer -> count}
        self._adj: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Per-medium adjacency: (source, target) -> {medium -> count}
        self._medium_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Raw links (for timeline and temporal queries)
        self._links: list[CommLink] = []

        # Per-entity link counts
        self._entity_link_count: dict[str, int] = defaultdict(int)

        # Per-entity first/last seen
        self._first_seen: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}

    @property
    def entity_count(self) -> int:
        """Number of unique entities in the network."""
        with self._lock:
            return len(self._adj)

    @property
    def link_count(self) -> int:
        """Total number of recorded links."""
        with self._lock:
            return len(self._links)

    def add_link(self, link: CommLink) -> None:
        """Record a communication link.

        Args:
            link: The CommLink to add.  Content fields are automatically
                  stripped from metadata.
        """
        with self._lock:
            self._links.append(link)

            # Undirected adjacency (communication is bidirectional metadata)
            self._adj[link.source][link.target] += 1
            self._adj[link.target][link.source] += 1

            # Medium tracking
            key = tuple(sorted([link.source, link.target]))
            self._medium_counts[(key[0], key[1])][link.medium] += 1

            # Per-entity counts
            self._entity_link_count[link.source] += 1
            self._entity_link_count[link.target] += 1

            # Temporal tracking
            for entity in (link.source, link.target):
                if entity not in self._first_seen:
                    self._first_seen[entity] = link.timestamp
                self._first_seen[entity] = min(
                    self._first_seen[entity], link.timestamp
                )
                self._last_seen[entity] = max(
                    self._last_seen.get(entity, 0.0), link.timestamp
                )

    def add_links(self, links: list[CommLink]) -> None:
        """Record multiple links at once."""
        for link in links:
            self.add_link(link)

    def get_peers(self, entity_id: str) -> dict[str, int]:
        """Get communication peers for an entity and their link counts.

        Returns:
            Dict mapping peer_id to number of observed communications.
        """
        entity_id = entity_id.upper()
        with self._lock:
            return dict(self._adj.get(entity_id, {}))

    def get_link_count(self, source: str, target: str) -> int:
        """Get the number of observed communications between two entities."""
        source = source.upper()
        target = target.upper()
        with self._lock:
            return self._adj.get(source, {}).get(target, 0)

    def get_entities(self) -> list[str]:
        """Return all entity identifiers in the network."""
        with self._lock:
            return sorted(self._adj.keys())

    def get_entity_stats(self, entity_id: str) -> dict[str, Any]:
        """Get statistics for a specific entity.

        Returns:
            Dict with peer_count, total_links, first_seen, last_seen,
            media (list of media used).
        """
        entity_id = entity_id.upper()
        with self._lock:
            peers = self._adj.get(entity_id, {})
            total_links = self._entity_link_count.get(entity_id, 0)

            # Collect media used by this entity
            media: set[str] = set()
            for link in self._links:
                if link.source == entity_id or link.target == entity_id:
                    media.add(link.medium)

            return {
                "entity_id": entity_id,
                "peer_count": len(peers),
                "total_links": total_links,
                "first_seen": self._first_seen.get(entity_id),
                "last_seen": self._last_seen.get(entity_id),
                "media": sorted(media),
            }

    def get_medium_breakdown(self, source: str, target: str) -> dict[str, int]:
        """Get per-medium link counts between two entities."""
        key = tuple(sorted([source.upper(), target.upper()]))
        with self._lock:
            return dict(self._medium_counts.get((key[0], key[1]), {}))

    def get_links_for(
        self,
        entity_id: str,
        since: float | None = None,
        until: float | None = None,
    ) -> list[CommLink]:
        """Get raw links involving an entity, optionally time-filtered.

        Args:
            entity_id: Entity to query.
            since: Only include links after this timestamp.
            until: Only include links before this timestamp.

        Returns:
            List of CommLink objects (chronologically ordered).
        """
        entity_id = entity_id.upper()
        with self._lock:
            result = []
            for link in self._links:
                if link.source != entity_id and link.target != entity_id:
                    continue
                if since is not None and link.timestamp < since:
                    continue
                if until is not None and link.timestamp > until:
                    continue
                result.append(link)
            return sorted(result, key=lambda l: l.timestamp)

    def to_graph_dict(self) -> dict[str, Any]:
        """Export the network as a dict with nodes and edges.

        Returns:
            Dict with 'nodes' (list of entity dicts) and 'edges'
            (list of edge dicts with source, target, weight, media).
        """
        with self._lock:
            nodes = []
            for entity in sorted(self._adj.keys()):
                nodes.append({
                    "id": entity,
                    "peer_count": len(self._adj[entity]),
                    "total_links": self._entity_link_count.get(entity, 0),
                    "first_seen": self._first_seen.get(entity),
                    "last_seen": self._last_seen.get(entity),
                })

            edges = []
            seen_edges: set[tuple[str, str]] = set()
            for source, peers in self._adj.items():
                for target, count in peers.items():
                    key = tuple(sorted([source, target]))
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)

                    media = dict(
                        self._medium_counts.get((key[0], key[1]), {})
                    )
                    edges.append({
                        "source": key[0],
                        "target": key[1],
                        "weight": count,
                        "media": media,
                    })

            return {
                "nodes": nodes,
                "edges": edges,
                "entity_count": len(nodes),
                "edge_count": len(edges),
            }

    def prune(self, before: float | None = None) -> int:
        """Remove links older than the retention window.

        Returns the number of links removed.
        """
        if self._retention_s <= 0 and before is None:
            return 0
        cutoff = before if before is not None else (time.time() - self._retention_s)
        with self._lock:
            original = len(self._links)
            kept: list[CommLink] = []
            for link in self._links:
                if link.timestamp >= cutoff:
                    kept.append(link)
            removed = original - len(kept)
            if removed > 0:
                self._rebuild_from(kept)
            return removed

    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            self._adj.clear()
            self._medium_counts.clear()
            self._links.clear()
            self._entity_link_count.clear()
            self._first_seen.clear()
            self._last_seen.clear()

    def _rebuild_from(self, links: list[CommLink]) -> None:
        """Rebuild all indices from a filtered link list.  Caller holds lock."""
        self._adj.clear()
        self._medium_counts.clear()
        self._entity_link_count.clear()
        self._first_seen.clear()
        self._last_seen.clear()
        self._links = []

        # Temporarily release the pattern to reuse add_link logic
        # but we already hold the lock, so just inline the logic
        for link in links:
            self._links.append(link)
            self._adj[link.source][link.target] += 1
            self._adj[link.target][link.source] += 1

            key = tuple(sorted([link.source, link.target]))
            self._medium_counts[(key[0], key[1])][link.medium] += 1

            self._entity_link_count[link.source] += 1
            self._entity_link_count[link.target] += 1

            for entity in (link.source, link.target):
                if entity not in self._first_seen:
                    self._first_seen[entity] = link.timestamp
                self._first_seen[entity] = min(
                    self._first_seen[entity], link.timestamp
                )
                self._last_seen[entity] = max(
                    self._last_seen.get(entity, 0.0), link.timestamp
                )


# ── CommAnalyzer — high-level pattern analysis ───────────────────────

class CommAnalyzer:
    """High-level communications intelligence analyzer.

    Wraps a CommNetwork and provides community detection, bridge
    identification, and timeline construction.

    Parameters
    ----------
    network:
        A CommNetwork instance to analyze.  If None, a new empty
        network is created internally.
    """

    def __init__(self, network: CommNetwork | None = None) -> None:
        self.network = network or CommNetwork()

    def record(self, link: CommLink) -> None:
        """Record a communication link into the underlying network."""
        self.network.add_link(link)

    def record_ble_pairing(
        self, mac_a: str, mac_b: str, timestamp: float | None = None, **kwargs: Any
    ) -> CommLink:
        """Record a BLE pairing event between two MACs.

        Args:
            mac_a: First MAC address.
            mac_b: Second MAC address.
            timestamp: Unix timestamp (default now).
            **kwargs: Additional metadata (rssi, channel — never content).

        Returns:
            The created CommLink.
        """
        link = CommLink(
            source=mac_a,
            target=mac_b,
            medium="ble",
            timestamp=timestamp or time.time(),
            metadata=kwargs,
        )
        self.network.add_link(link)
        return link

    def record_wifi_association(
        self,
        client_mac: str,
        ap_bssid: str,
        timestamp: float | None = None,
        **kwargs: Any,
    ) -> CommLink:
        """Record a WiFi client-AP association.

        Args:
            client_mac: Client MAC address.
            ap_bssid: Access point BSSID.
            timestamp: Unix timestamp (default now).
            **kwargs: Additional metadata (rssi, channel, ssid).

        Returns:
            The created CommLink.
        """
        link = CommLink(
            source=client_mac,
            target=ap_bssid,
            medium="wifi",
            timestamp=timestamp or time.time(),
            metadata=kwargs,
        )
        self.network.add_link(link)
        return link

    def record_mesh_message(
        self,
        sender_id: str,
        recipient_id: str,
        timestamp: float | None = None,
        **kwargs: Any,
    ) -> CommLink:
        """Record a Meshtastic mesh message (metadata only).

        Args:
            sender_id: Sending node ID.
            recipient_id: Receiving node ID.
            timestamp: Unix timestamp (default now).
            **kwargs: Additional metadata (hop_count, channel — never content).

        Returns:
            The created CommLink.
        """
        # Strip content fields aggressively
        kwargs.pop("content", None)
        kwargs.pop("message", None)
        kwargs.pop("text", None)
        link = CommLink(
            source=sender_id,
            target=recipient_id,
            medium="meshtastic",
            timestamp=timestamp or time.time(),
            metadata=kwargs,
        )
        self.network.add_link(link)
        return link

    def record_espnow_frame(
        self,
        source_mac: str,
        dest_mac: str,
        timestamp: float | None = None,
        **kwargs: Any,
    ) -> CommLink:
        """Record an ESP-NOW frame observation.

        Args:
            source_mac: Source MAC.
            dest_mac: Destination MAC.
            timestamp: Unix timestamp (default now).
            **kwargs: Additional metadata (rssi, channel).

        Returns:
            The created CommLink.
        """
        link = CommLink(
            source=source_mac,
            target=dest_mac,
            medium="espnow",
            timestamp=timestamp or time.time(),
            metadata=kwargs,
        )
        self.network.add_link(link)
        return link

    # ── Community detection ──────────────────────────────────────────

    def find_communities(
        self, min_size: int = 2, max_iterations: int = 100
    ) -> list[CommunityResult]:
        """Detect groups of frequently communicating entities.

        Uses weighted label propagation.  Each entity starts with its
        own label.  On each iteration, every entity adopts the label
        most common among its neighbours (weighted by link count).
        Converges when no label changes.

        Args:
            min_size: Minimum community size to return.
            max_iterations: Maximum label-propagation iterations.

        Returns:
            List of CommunityResult objects, sorted by size descending.
        """
        graph = self.network.to_graph_dict()
        nodes = [n["id"] for n in graph["nodes"]]
        if not nodes:
            return []

        # Build weighted adjacency from edge list
        adj: dict[str, dict[str, int]] = defaultdict(dict)
        for edge in graph["edges"]:
            adj[edge["source"]][edge["target"]] = edge["weight"]
            adj[edge["target"]][edge["source"]] = edge["weight"]

        # Initialize: each node gets its own label
        labels: dict[str, int] = {node: i for i, node in enumerate(nodes)}

        for _ in range(max_iterations):
            changed = False
            for node in nodes:
                neighbours = adj.get(node, {})
                if not neighbours:
                    continue

                # Count weighted votes for each label
                label_votes: dict[int, int] = defaultdict(int)
                for peer, weight in neighbours.items():
                    label_votes[labels[peer]] += weight

                if not label_votes:
                    continue

                best_label = max(label_votes, key=lambda l: label_votes[l])
                if labels[node] != best_label:
                    labels[node] = best_label
                    changed = True

            if not changed:
                break

        # Group by label
        communities_map: dict[int, set[str]] = defaultdict(set)
        for node, label in labels.items():
            communities_map[label].add(node)

        # Build results
        results: list[CommunityResult] = []
        for label, members in communities_map.items():
            if len(members) < min_size:
                continue

            # Count internal links
            internal_links = 0
            medium_counts: dict[str, int] = defaultdict(int)
            for edge in graph["edges"]:
                if edge["source"] in members and edge["target"] in members:
                    internal_links += edge["weight"]
                    for med, cnt in edge.get("media", {}).items():
                        medium_counts[med] += cnt

            primary_medium = ""
            if medium_counts:
                primary_medium = max(medium_counts, key=lambda m: medium_counts[m])

            results.append(CommunityResult(
                community_id=label,
                members=members,
                link_count=internal_links,
                primary_medium=primary_medium,
            ))

        results.sort(key=lambda c: len(c.members), reverse=True)
        return results

    # ── Bridge detection ─────────────────────────────────────────────

    def find_bridges(
        self, communities: list[CommunityResult] | None = None
    ) -> list[BridgeEntity]:
        """Identify entities that connect different communities.

        A bridge is an entity that belongs to multiple communities or
        whose removal would increase the number of connected components.

        Args:
            communities: Pre-computed communities (calls find_communities()
                if None).

        Returns:
            List of BridgeEntity objects, sorted by bridge_score descending.
        """
        if communities is None:
            communities = self.find_communities()

        if not communities:
            return []

        # Map entity -> set of community IDs
        entity_communities: dict[str, set[int]] = defaultdict(set)
        for comm in communities:
            for member in comm.members:
                entity_communities[member].add(comm.community_id)

        bridges: list[BridgeEntity] = []
        for entity_id, comm_ids in entity_communities.items():
            if len(comm_ids) < 2:
                # Check if this entity has peers in other communities
                peers = self.network.get_peers(entity_id)
                peer_comms: set[int] = set()
                for peer in peers:
                    peer_comms.update(entity_communities.get(peer, set()))
                if len(peer_comms) < 2:
                    continue
                comm_ids = peer_comms

            # Compute bridge score: how many communities does this connect?
            total_communities = len(communities)
            community_fraction = len(comm_ids) / max(total_communities, 1)

            # Factor in link count (more links = more critical bridge)
            stats = self.network.get_entity_stats(entity_id)
            link_count = stats.get("total_links", 0)
            total_links = self.network.link_count or 1
            link_fraction = min(1.0, link_count / (total_links * 0.1))

            bridge_score = (community_fraction * 0.6 + link_fraction * 0.4)

            bridges.append(BridgeEntity(
                entity_id=entity_id,
                communities=comm_ids,
                bridge_score=round(bridge_score, 3),
                link_count=link_count,
            ))

        bridges.sort(key=lambda b: b.bridge_score, reverse=True)
        return bridges

    # ── Timeline ─────────────────────────────────────────────────────

    def communication_timeline(
        self,
        entity_id: str,
        since: float | None = None,
        until: float | None = None,
    ) -> list[TimelineEntry]:
        """Build a chronological communication timeline for an entity.

        Args:
            entity_id: The entity to build a timeline for.
            since: Start timestamp filter.
            until: End timestamp filter.

        Returns:
            List of TimelineEntry objects in chronological order.
        """
        entity_id = entity_id.upper()
        links = self.network.get_links_for(entity_id, since=since, until=until)

        entries: list[TimelineEntry] = []
        for link in links:
            if link.source == entity_id:
                peer = link.target
                direction = "out"
            else:
                peer = link.source
                direction = "in"

            entries.append(TimelineEntry(
                peer=peer,
                medium=link.medium,
                timestamp=link.timestamp,
                direction=direction,
            ))

        return entries

    # ── Convenience queries ──────────────────────────────────────────

    def top_communicators(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the top N entities by total communication count.

        Returns:
            List of dicts with entity_id, total_links, peer_count.
        """
        entities = self.network.get_entities()
        stats = []
        for eid in entities:
            s = self.network.get_entity_stats(eid)
            stats.append(s)

        stats.sort(key=lambda s: s["total_links"], reverse=True)
        return stats[:n]

    def most_active_pairs(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the top N most frequently communicating pairs.

        Returns:
            List of dicts with source, target, count, media.
        """
        graph = self.network.to_graph_dict()
        edges = sorted(graph["edges"], key=lambda e: e["weight"], reverse=True)
        return [
            {
                "source": e["source"],
                "target": e["target"],
                "count": e["weight"],
                "media": e.get("media", {}),
            }
            for e in edges[:n]
        ]

    def entity_reach(self, entity_id: str, max_hops: int = 2) -> set[str]:
        """Find all entities reachable within N hops.

        Args:
            entity_id: Starting entity.
            max_hops: Maximum number of hops (1-5).

        Returns:
            Set of reachable entity IDs (not including the start).
        """
        entity_id = entity_id.upper()
        max_hops = max(1, min(max_hops, 5))
        visited: set[str] = set()
        current_layer: set[str] = {entity_id}

        for _ in range(max_hops):
            next_layer: set[str] = set()
            for eid in current_layer:
                peers = self.network.get_peers(eid)
                for peer in peers:
                    if peer not in visited and peer != entity_id:
                        next_layer.add(peer)
            visited.update(next_layer)
            current_layer = next_layer

        return visited

    def get_statistics(self) -> dict[str, Any]:
        """Return summary statistics about the communication network.

        Returns:
            Dict with entity_count, link_count, edge_count,
            medium_distribution, avg_peers_per_entity.
        """
        graph = self.network.to_graph_dict()

        # Medium distribution
        medium_dist: dict[str, int] = defaultdict(int)
        for edge in graph["edges"]:
            for med, cnt in edge.get("media", {}).items():
                medium_dist[med] += cnt

        entity_count = graph["entity_count"]
        avg_peers = 0.0
        if entity_count > 0:
            total_peers = sum(n["peer_count"] for n in graph["nodes"])
            avg_peers = round(total_peers / entity_count, 2)

        return {
            "entity_count": entity_count,
            "link_count": self.network.link_count,
            "edge_count": graph["edge_count"],
            "medium_distribution": dict(medium_dist),
            "avg_peers_per_entity": avg_peers,
        }

    def export(self) -> dict[str, Any]:
        """Export the full analysis state for API/dashboard consumption.

        Returns:
            Dict with network graph, communities, bridges, and statistics.
        """
        communities = self.find_communities()
        bridges = self.find_bridges(communities=communities)
        return {
            "network": self.network.to_graph_dict(),
            "communities": [c.to_dict() for c in communities],
            "bridges": [b.to_dict() for b in bridges],
            "statistics": self.get_statistics(),
        }

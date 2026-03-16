# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fleet topology and connectivity models — multi-transport network graph.

These models represent the fleet as a connectivity graph where links can span
multiple transport types (WiFi, ESP-NOW, BLE, LoRa, MQTT).  The fleet server
builds a FleetTopology from heartbeat data and link advertisements, then
analyzes it for partitions, reachability, and overall connectivity health.

NetworkNode represents a single device in the mesh with its capabilities,
position, and health metrics.  NetworkLink is a directional connection
between two nodes.  FleetTopology holds the full graph and supports BFS
traversal, component detection, and connectivity analysis.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NodeRole(str, Enum):
    """Roles a network node can play in the mesh."""
    GATEWAY = "gateway"     # Bridges mesh to WiFi/server
    RELAY = "relay"         # Forwards packets between peers
    LEAF = "leaf"           # End device, does not relay
    SENSOR = "sensor"       # Deep-sleep wake-transmit-sleep node


class NetworkNode(BaseModel):
    """A single device in the fleet network graph.

    Contains identity, role, position, health metrics, and peer quality
    statistics.  Used by comm-link visualization and fleet dashboard.
    """
    node_id: str
    name: str = ""
    role: NodeRole = NodeRole.RELAY
    ip: str = ""
    mac: str = ""
    firmware: str = ""
    board_type: str = ""
    battery_pct: Optional[int] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    online: bool = True
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Peer quality stats (populated from heartbeat peer data)
    peer_count: int = 0
    avg_peer_rssi: Optional[float] = None
    packet_loss_pct: float = 0.0
    avg_latency_ms: Optional[float] = None
    transports: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}


class PeerQuality(BaseModel):
    """Quality metrics for a single peer relationship.

    Tracked per-peer by edge firmware and included in heartbeat data
    so the command center can visualize network health per link.
    """
    peer_mac: str
    rssi_current: int = -100
    rssi_avg: float = -100.0
    rssi_min: int = -100
    rssi_max: int = 0
    packet_loss_pct: float = 0.0
    avg_latency_ms: float = 0.0
    tx_count: int = 0
    rx_count: int = 0
    tx_fail: int = 0
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": False}

    @property
    def quality_score(self) -> int:
        """Compute a 0-100 quality score from RSSI and packet loss."""
        # RSSI component: map -90..-30 to 0..100
        rssi_score = max(0, min(100, int((self.rssi_avg + 90) * (100 / 60))))
        # Loss penalty: each 1% loss costs 2 points
        loss_penalty = min(100, int(self.packet_loss_pct * 2))
        return max(0, rssi_score - loss_penalty)


class NetworkLink(BaseModel):
    """A connection between two nodes over a specific transport.

    Links are directional in the sense that source_id reported the link,
    but for graph traversal they are treated as undirected edges.
    """
    source_id: str
    target_id: str
    transport: str  # wifi, espnow, ble, lora, mqtt
    rssi: Optional[int] = None
    latency_ms: Optional[float] = None
    bandwidth_kbps: Optional[float] = None
    packet_loss_pct: float = 0.0
    quality_score: int = 0  # 0-100, computed from RSSI + loss
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True


class FleetTopology(BaseModel):
    """The fleet mesh network as an undirected graph.

    Nodes are identified by string IDs.  Links connect pairs of nodes
    and may use different transports.  Graph algorithms operate on the
    active links only.

    The optional ``network_nodes`` list holds rich node metadata for
    visualization (positions, roles, health).  If empty, only ``nodes``
    (string IDs) are used for graph algorithms.
    """
    nodes: list[str] = Field(default_factory=list)
    links: list[NetworkLink] = Field(default_factory=list)
    network_nodes: list[NetworkNode] = Field(default_factory=list)

    def _adjacency(self, active_only: bool = True) -> dict[str, set[str]]:
        """Build an adjacency map from the link list."""
        adj: dict[str, set[str]] = defaultdict(set)
        for node in self.nodes:
            adj[node]  # ensure every node appears even if it has no links
        for link in self.links:
            if active_only and not link.active:
                continue
            adj[link.source_id].add(link.target_id)
            adj[link.target_id].add(link.source_id)
        return adj

    def neighbors(self, node_id: str) -> list[str]:
        """Return the IDs of nodes directly linked to *node_id*."""
        adj = self._adjacency()
        return sorted(adj.get(node_id, set()))

    def reachable(self, from_id: str, to_id: str) -> bool:
        """Return True if *to_id* is reachable from *from_id* via BFS."""
        if from_id == to_id:
            return True
        adj = self._adjacency()
        if from_id not in adj:
            return False
        visited: set[str] = {from_id}
        queue: deque[str] = deque([from_id])
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor == to_id:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def connected_components(self) -> list[list[str]]:
        """Return connected components as sorted lists of node IDs."""
        adj = self._adjacency()
        visited: set[str] = set()
        components: list[list[str]] = []
        for node in self.nodes:
            if node in visited:
                continue
            component: list[str] = []
            queue: deque[str] = deque([node])
            visited.add(node)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(sorted(component))
        return components

    def average_path_length(self) -> float:
        """Average shortest path length across all reachable pairs (BFS).

        Returns 0.0 if the graph has fewer than 2 nodes or no reachable pairs.
        """
        if len(self.nodes) < 2:
            return 0.0
        adj = self._adjacency()
        total_length = 0
        pair_count = 0
        for start in self.nodes:
            # BFS from start
            dist: dict[str, int] = {start: 0}
            queue: deque[str] = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in adj[current]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
            for target in self.nodes:
                if target != start and target in dist:
                    total_length += dist[target]
                    pair_count += 1
        if pair_count == 0:
            return 0.0
        return total_length / pair_count


class ConnectivityReport(BaseModel):
    """Summary of fleet connectivity derived from a FleetTopology."""
    total_nodes: int = 0
    connected_nodes: int = 0
    isolated_nodes: int = 0
    num_components: int = 0
    avg_links_per_node: float = 0.0
    transports_used: list[str] = Field(default_factory=list)


def build_topology(links: list[NetworkLink]) -> FleetTopology:
    """Construct a FleetTopology from a list of NetworkLinks.

    Node IDs are extracted from the links automatically.
    """
    node_set: set[str] = set()
    for link in links:
        node_set.add(link.source_id)
        node_set.add(link.target_id)
    return FleetTopology(nodes=sorted(node_set), links=links)


def build_fleet_topology_from_mesh(
    node_health_map: dict[str, list],
) -> FleetTopology:
    """Build a FleetTopology from mesh_peer_list data in health snapshots.

    Args:
        node_health_map: Mapping of node_id -> list of MeshPeer-like dicts
            (or MeshPeer objects) with ``mac``, ``rssi``, and ``hops`` fields.
            Each entry represents the mesh peers that *node_id* reported.

    Returns:
        A FleetTopology with ESP-NOW links derived from actual peer data.
        Nodes are identified by their node_id (reporter) and peer MAC.
    """
    node_set: set[str] = set()
    links: list[NetworkLink] = []
    seen_edges: set[tuple[str, str]] = set()

    for node_id, peers in node_health_map.items():
        node_set.add(node_id)
        for peer in peers:
            # Accept both dicts and objects with .mac/.rssi/.hops
            if isinstance(peer, dict):
                mac = peer.get("mac", "")
                rssi = peer.get("rssi", 0)
            else:
                mac = getattr(peer, "mac", "")
                rssi = getattr(peer, "rssi", 0)

            if not mac:
                continue

            node_set.add(mac)

            # Deduplicate edges (A->B == B->A)
            edge_key = tuple(sorted((node_id, mac)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            links.append(NetworkLink(
                source_id=node_id,
                target_id=mac,
                transport="espnow",
                rssi=rssi,
                active=True,
            ))

    return FleetTopology(nodes=sorted(node_set), links=links)


def analyze_connectivity(topology: FleetTopology) -> ConnectivityReport:
    """Analyze a FleetTopology and produce a ConnectivityReport."""
    components = topology.connected_components()
    isolated = sum(1 for c in components if len(c) == 1)
    connected = len(topology.nodes) - isolated

    # Count active links per node
    active_links = [link for link in topology.links if link.active]
    if topology.nodes:
        link_count: dict[str, int] = defaultdict(int)
        for link in active_links:
            link_count[link.source_id] += 1
            link_count[link.target_id] += 1
        avg_links = sum(link_count.values()) / len(topology.nodes)
    else:
        avg_links = 0.0

    transports = sorted({link.transport for link in active_links})

    return ConnectivityReport(
        total_nodes=len(topology.nodes),
        connected_nodes=connected,
        isolated_nodes=isolated,
        num_components=len(components),
        avg_links_per_node=round(avg_links, 2),
        transports_used=transports,
    )

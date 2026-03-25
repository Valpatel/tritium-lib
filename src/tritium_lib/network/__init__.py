# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.network — network topology discovery and analysis.

Discover and map the network topology of edge devices, sensors, and MQTT
brokers.  This is a pure data-model and algorithm module — no actual network
scanning is performed.  Instead, :class:`TopologyDiscovery` builds a topology
from simulated or pre-collected inventory data, and :class:`PathAnalysis`
provides graph algorithms for data-flow path finding and bottleneck detection.

Quick start::

    from tritium_lib.network import (
        NetworkTopology, NetworkNode, NetworkLink,
        TopologyDiscovery, PathAnalysis,
    )

    # Build a small topology
    discovery = TopologyDiscovery()
    discovery.add_node(NetworkNode(node_id="broker-1", node_type="mqtt_broker",
                                   ip="192.168.1.10"))
    discovery.add_node(NetworkNode(node_id="edge-1", node_type="edge_device",
                                   ip="192.168.1.20",
                                   capabilities=["ble", "wifi"]))
    discovery.add_link(NetworkLink(source_id="edge-1", target_id="broker-1",
                                   link_type="mqtt", bandwidth_mbps=10.0,
                                   latency_ms=5.0))
    topo = discovery.build()

    # Analyze paths
    pa = PathAnalysis(topo)
    path = pa.shortest_path("edge-1", "broker-1")
    bottlenecks = pa.find_bottlenecks()
    dot = topo.to_graphviz()
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """Types of nodes in the network topology."""
    EDGE_DEVICE = "edge_device"
    SENSOR = "sensor"
    MQTT_BROKER = "mqtt_broker"
    GATEWAY = "gateway"
    HUB = "hub"
    SERVER = "server"
    CAMERA = "camera"
    RADIO = "radio"


class LinkType(str, Enum):
    """Types of network connections."""
    MQTT = "mqtt"
    WIFI = "wifi"
    ETHERNET = "ethernet"
    BLE = "ble"
    LORA = "lora"
    ESPNOW = "espnow"
    USB = "usb"
    WEBSOCKET = "websocket"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NetworkNode:
    """A device, broker, or hub in the network topology.

    Attributes:
        node_id: Unique identifier for this node.
        node_type: Category of the node (edge_device, mqtt_broker, etc.).
        ip: IP address (empty string if not applicable).
        name: Human-readable label.
        capabilities: List of sensor/protocol capabilities (e.g. ``["ble", "wifi"]``).
        online: Whether the node is currently reachable.
        metadata: Arbitrary key-value metadata.
    """
    node_id: str
    node_type: str = "edge_device"
    ip: str = ""
    name: str = ""
    capabilities: list[str] = field(default_factory=list)
    online: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.node_id


@dataclass
class NetworkLink:
    """A connection between two nodes in the network topology.

    Links are undirected for traversal but carry a source/target to record
    which side initiated the connection.

    Attributes:
        source_id: Node ID of the connection source.
        target_id: Node ID of the connection target.
        link_type: Transport type (mqtt, wifi, ethernet, etc.).
        bandwidth_mbps: Maximum bandwidth in Mbps (None if unknown).
        latency_ms: Latency in milliseconds (None if unknown).
        packet_loss_pct: Packet loss percentage (0.0 – 100.0).
        active: Whether the link is currently up.
        metadata: Arbitrary key-value metadata.
    """
    source_id: str
    target_id: str
    link_type: str = "mqtt"
    bandwidth_mbps: Optional[float] = None
    latency_ms: Optional[float] = None
    packet_loss_pct: float = 0.0
    active: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def weight(self) -> float:
        """Composite cost weight for path-finding (lower is better).

        Combines latency and packet-loss into a single scalar.  If latency is
        unknown it defaults to 1.0 so that every link has a positive cost.
        """
        base = self.latency_ms if self.latency_ms is not None else 1.0
        loss_factor = 1.0 + (self.packet_loss_pct / 100.0)
        return base * loss_factor


# ---------------------------------------------------------------------------
# NetworkTopology — the graph
# ---------------------------------------------------------------------------

class NetworkTopology:
    """Graph of network nodes and their connections.

    Nodes are stored by ID in a dict; links are stored in a list.  The class
    exposes read-only views and graph-export helpers but delegates path-finding
    to :class:`PathAnalysis`.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NetworkNode] = {}
        self._links: list[NetworkLink] = []

    # -- Mutation ----------------------------------------------------------

    def add_node(self, node: NetworkNode) -> None:
        """Add or replace a node."""
        self._nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its links.  Returns True if the node existed."""
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        self._links = [
            lnk for lnk in self._links
            if lnk.source_id != node_id and lnk.target_id != node_id
        ]
        return True

    def add_link(self, link: NetworkLink) -> None:
        """Add a link.  Both endpoint node IDs must already exist."""
        if link.source_id not in self._nodes:
            raise KeyError(f"Source node '{link.source_id}' not in topology")
        if link.target_id not in self._nodes:
            raise KeyError(f"Target node '{link.target_id}' not in topology")
        self._links.append(link)

    def remove_link(self, source_id: str, target_id: str) -> int:
        """Remove all links between *source_id* and *target_id* (either direction).

        Returns the number of links removed.
        """
        before = len(self._links)
        self._links = [
            lnk for lnk in self._links
            if not (
                (lnk.source_id == source_id and lnk.target_id == target_id)
                or (lnk.source_id == target_id and lnk.target_id == source_id)
            )
        ]
        return before - len(self._links)

    # -- Queries -----------------------------------------------------------

    @property
    def nodes(self) -> dict[str, NetworkNode]:
        """Read-only view of nodes keyed by ID."""
        return dict(self._nodes)

    @property
    def links(self) -> list[NetworkLink]:
        """Read-only copy of all links."""
        return list(self._links)

    @property
    def node_ids(self) -> list[str]:
        """Sorted list of all node IDs."""
        return sorted(self._nodes.keys())

    def get_node(self, node_id: str) -> Optional[NetworkNode]:
        """Return a node by ID, or ``None``."""
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str) -> list[str]:
        """Return sorted neighbor IDs reachable via active links."""
        nbrs: set[str] = set()
        for lnk in self._links:
            if not lnk.active:
                continue
            if lnk.source_id == node_id:
                nbrs.add(lnk.target_id)
            elif lnk.target_id == node_id:
                nbrs.add(lnk.source_id)
        return sorted(nbrs)

    def links_for(self, node_id: str, active_only: bool = True) -> list[NetworkLink]:
        """Return all links touching *node_id*."""
        result: list[NetworkLink] = []
        for lnk in self._links:
            if active_only and not lnk.active:
                continue
            if lnk.source_id == node_id or lnk.target_id == node_id:
                result.append(lnk)
        return result

    def nodes_by_type(self, node_type: str) -> list[NetworkNode]:
        """Return all nodes matching *node_type*."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def adjacency(self, active_only: bool = True) -> dict[str, set[str]]:
        """Build an adjacency map from links."""
        adj: dict[str, set[str]] = defaultdict(set)
        for nid in self._nodes:
            adj[nid]  # ensure every node appears
        for lnk in self._links:
            if active_only and not lnk.active:
                continue
            adj[lnk.source_id].add(lnk.target_id)
            adj[lnk.target_id].add(lnk.source_id)
        return dict(adj)

    def connected_components(self) -> list[list[str]]:
        """Return connected components as sorted lists of node IDs."""
        adj = self.adjacency(active_only=True)
        visited: set[str] = set()
        components: list[list[str]] = []
        for nid in sorted(self._nodes.keys()):
            if nid in visited:
                continue
            component: list[str] = []
            queue: deque[str] = deque([nid])
            visited.add(nid)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adj.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(sorted(component))
        return components

    def is_connected(self) -> bool:
        """Return True if the entire topology forms a single connected component."""
        comps = self.connected_components()
        return len(comps) == 1

    # -- Export ------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "ip": n.ip,
                    "name": n.name,
                    "capabilities": n.capabilities,
                    "online": n.online,
                    "metadata": n.metadata,
                }
                for n in self._nodes.values()
            ],
            "links": [
                {
                    "source_id": lnk.source_id,
                    "target_id": lnk.target_id,
                    "link_type": lnk.link_type,
                    "bandwidth_mbps": lnk.bandwidth_mbps,
                    "latency_ms": lnk.latency_ms,
                    "packet_loss_pct": lnk.packet_loss_pct,
                    "active": lnk.active,
                    "metadata": lnk.metadata,
                }
                for lnk in self._links
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NetworkTopology":
        """Deserialize from a plain dict produced by :meth:`to_dict`."""
        topo = cls()
        for nd in data.get("nodes", []):
            topo.add_node(NetworkNode(**nd))
        for ld in data.get("links", []):
            topo.add_link(NetworkLink(**ld))
        return topo

    def to_graphviz(self, title: str = "Tritium Network Topology") -> str:
        """Export the topology as a Graphviz DOT string.

        Nodes are colored by type; links are labeled with their transport type.
        """
        node_colors: dict[str, str] = {
            "edge_device": "#00f0ff",   # cyan
            "sensor": "#05ffa1",        # green
            "mqtt_broker": "#ff2a6d",   # magenta
            "gateway": "#fcee0a",       # yellow
            "hub": "#ffaa00",
            "server": "#ff2a6d",
            "camera": "#05ffa1",
            "radio": "#00f0ff",
        }

        lines: list[str] = []
        lines.append(f'graph "{title}" {{')
        lines.append('    rankdir=LR;')
        lines.append('    bgcolor="#0a0a0a";')
        lines.append('    node [style=filled, fontcolor=white, fontname="monospace"];')
        lines.append('    edge [fontcolor="#aaaaaa", fontname="monospace", color="#444444"];')
        lines.append('')

        # Nodes
        for node in self._nodes.values():
            color = node_colors.get(node.node_type, "#888888")
            label = f"{node.name}\\n{node.node_type}"
            if node.ip:
                label += f"\\n{node.ip}"
            shape = "box" if node.node_type in ("mqtt_broker", "server", "hub") else "ellipse"
            status = "solid" if node.online else "dashed"
            lines.append(
                f'    "{node.node_id}" '
                f'[label="{label}", fillcolor="{color}", '
                f'shape={shape}, style="{status},filled"];'
            )

        lines.append('')

        # Edges (undirected graph)
        seen: set[tuple[str, str]] = set()
        for lnk in self._links:
            edge_key = tuple(sorted((lnk.source_id, lnk.target_id)))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            label_parts: list[str] = [lnk.link_type]
            if lnk.bandwidth_mbps is not None:
                label_parts.append(f"{lnk.bandwidth_mbps}Mbps")
            if lnk.latency_ms is not None:
                label_parts.append(f"{lnk.latency_ms}ms")
            label = " | ".join(label_parts)
            style = "solid" if lnk.active else "dashed"
            lines.append(
                f'    "{lnk.source_id}" -- "{lnk.target_id}" '
                f'[label="{label}", style={style}];'
            )

        lines.append('}')
        return "\n".join(lines)

    def summary(self) -> dict:
        """Return a concise summary of the topology."""
        active_links = [lnk for lnk in self._links if lnk.active]
        types: dict[str, int] = defaultdict(int)
        for n in self._nodes.values():
            types[n.node_type] += 1
        link_types: dict[str, int] = defaultdict(int)
        for lnk in active_links:
            link_types[lnk.link_type] += 1
        comps = self.connected_components()
        return {
            "total_nodes": len(self._nodes),
            "total_links": len(self._links),
            "active_links": len(active_links),
            "node_types": dict(types),
            "link_types": dict(link_types),
            "connected_components": len(comps),
            "is_connected": len(comps) == 1,
        }


# ---------------------------------------------------------------------------
# TopologyDiscovery — build topologies from inventory data
# ---------------------------------------------------------------------------

class TopologyDiscovery:
    """Build a :class:`NetworkTopology` from simulated or collected data.

    This class accumulates nodes and links, optionally auto-discovers
    connections based on shared capabilities, and produces a finished topology.
    """

    def __init__(self) -> None:
        self._nodes: list[NetworkNode] = []
        self._links: list[NetworkLink] = []

    def add_node(self, node: NetworkNode) -> None:
        """Register a network node."""
        self._nodes.append(node)

    def add_link(self, link: NetworkLink) -> None:
        """Register a network link."""
        self._links.append(link)

    def auto_discover_links(
        self,
        default_bandwidth_mbps: float = 10.0,
        default_latency_ms: float = 5.0,
    ) -> int:
        """Generate links between nodes that share a common capability.

        For every pair of nodes that both advertise the same capability
        (e.g. both have ``"wifi"``), a link is created with the shared
        capability as ``link_type``.

        Returns the number of links created.
        """
        count = 0
        node_list = list(self._nodes)
        for i, a in enumerate(node_list):
            for b in node_list[i + 1:]:
                shared = set(a.capabilities) & set(b.capabilities)
                for cap in sorted(shared):
                    self._links.append(NetworkLink(
                        source_id=a.node_id,
                        target_id=b.node_id,
                        link_type=cap,
                        bandwidth_mbps=default_bandwidth_mbps,
                        latency_ms=default_latency_ms,
                    ))
                    count += 1
        return count

    def discover_from_broker(
        self,
        broker_id: str,
        device_ids: list[str],
        link_type: str = "mqtt",
        bandwidth_mbps: float = 100.0,
        latency_ms: float = 2.0,
    ) -> int:
        """Create star-topology links from a broker to a list of devices.

        This simulates the common pattern where all edge devices connect to a
        central MQTT broker.  The broker node must already have been added via
        :meth:`add_node`.

        Returns the number of links created.
        """
        count = 0
        known_ids = {n.node_id for n in self._nodes}
        if broker_id not in known_ids:
            return 0
        for did in device_ids:
            if did in known_ids and did != broker_id:
                self._links.append(NetworkLink(
                    source_id=did,
                    target_id=broker_id,
                    link_type=link_type,
                    bandwidth_mbps=bandwidth_mbps,
                    latency_ms=latency_ms,
                ))
                count += 1
        return count

    def build(self) -> NetworkTopology:
        """Construct and return a :class:`NetworkTopology` from accumulated data."""
        topo = NetworkTopology()
        for node in self._nodes:
            topo.add_node(node)
        for link in self._links:
            # Only add links whose endpoints exist
            if topo.get_node(link.source_id) and topo.get_node(link.target_id):
                topo.add_link(link)
        return topo


# ---------------------------------------------------------------------------
# PathAnalysis — shortest paths, bottleneck detection
# ---------------------------------------------------------------------------

class PathAnalysis:
    """Find data-flow paths and identify bottlenecks in a :class:`NetworkTopology`.

    All algorithms operate on the active links only.
    """

    def __init__(self, topology: NetworkTopology) -> None:
        self._topo = topology

    def shortest_path(self, from_id: str, to_id: str) -> Optional[list[str]]:
        """Return the shortest path (list of node IDs) from *from_id* to *to_id*.

        Uses BFS on unweighted edges.  Returns ``None`` if unreachable.
        """
        if from_id == to_id:
            return [from_id]
        adj = self._topo.adjacency(active_only=True)
        if from_id not in adj:
            return None
        parent: dict[str, Optional[str]] = {from_id: None}
        queue: deque[str] = deque([from_id])
        while queue:
            current = queue.popleft()
            for neighbor in adj.get(current, set()):
                if neighbor not in parent:
                    parent[neighbor] = current
                    if neighbor == to_id:
                        # Reconstruct
                        path: list[str] = []
                        step: Optional[str] = to_id
                        while step is not None:
                            path.append(step)
                            step = parent[step]
                        return list(reversed(path))
                    queue.append(neighbor)
        return None

    def weighted_shortest_path(
        self, from_id: str, to_id: str,
    ) -> Optional[tuple[list[str], float]]:
        """Dijkstra shortest path using link :attr:`~NetworkLink.weight`.

        Returns ``(path, total_cost)`` or ``None`` if unreachable.
        """
        if from_id == to_id:
            return ([from_id], 0.0)
        nodes = self._topo.node_ids
        if from_id not in nodes or to_id not in nodes:
            return None

        # Build weighted adjacency
        wadj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for lnk in self._topo.links:
            if not lnk.active:
                continue
            wadj[lnk.source_id].append((lnk.target_id, lnk.weight))
            wadj[lnk.target_id].append((lnk.source_id, lnk.weight))

        # Dijkstra (simple — no heap, fine for small network topologies)
        dist: dict[str, float] = {nid: float("inf") for nid in self._topo.node_ids}
        dist[from_id] = 0.0
        prev: dict[str, Optional[str]] = {nid: None for nid in self._topo.node_ids}
        visited: set[str] = set()

        while True:
            # Pick unvisited node with smallest dist
            current: Optional[str] = None
            current_dist = float("inf")
            for nid in self._topo.node_ids:
                if nid not in visited and dist[nid] < current_dist:
                    current = nid
                    current_dist = dist[nid]
            if current is None or current_dist == float("inf"):
                return None  # unreachable
            if current == to_id:
                # Reconstruct
                path: list[str] = []
                step: Optional[str] = to_id
                while step is not None:
                    path.append(step)
                    step = prev[step]
                return (list(reversed(path)), dist[to_id])
            visited.add(current)
            for neighbor, w in wadj.get(current, []):
                if neighbor in visited:
                    continue
                alt = dist[current] + w
                if alt < dist[neighbor]:
                    dist[neighbor] = alt
                    prev[neighbor] = current
        # Unreachable (loop exits via return)

    def all_paths(self, from_id: str, to_id: str, max_depth: int = 10) -> list[list[str]]:
        """Return all simple paths up to *max_depth* hops (DFS).

        Useful for redundancy analysis — how many alternative routes exist?
        """
        if from_id == to_id:
            return [[from_id]]
        adj = self._topo.adjacency(active_only=True)
        if from_id not in adj:
            return []
        results: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = [(from_id, [from_id])]
        while stack:
            current, path = stack.pop()
            if len(path) > max_depth + 1:
                continue
            for neighbor in adj.get(current, set()):
                if neighbor in path:
                    continue  # no cycles
                new_path = path + [neighbor]
                if neighbor == to_id:
                    results.append(new_path)
                else:
                    stack.append((neighbor, new_path))
        return sorted(results, key=len)

    def find_bottlenecks(self) -> list[str]:
        """Identify articulation points (cut vertices) in the topology.

        An articulation point is a node whose removal would disconnect the
        graph (or increase the number of connected components).  These are
        single points of failure — network bottlenecks.

        Uses a simple approach: for each node, temporarily remove it and
        check if the number of components increases.
        """
        if len(self._topo.node_ids) <= 2:
            return []

        base_components = len(self._topo.connected_components())
        bottlenecks: list[str] = []

        for nid in self._topo.node_ids:
            # Build adjacency without this node
            adj: dict[str, set[str]] = defaultdict(set)
            remaining = [n for n in self._topo.node_ids if n != nid]
            for n in remaining:
                adj[n]  # ensure presence
            for lnk in self._topo.links:
                if not lnk.active:
                    continue
                if lnk.source_id == nid or lnk.target_id == nid:
                    continue
                adj[lnk.source_id].add(lnk.target_id)
                adj[lnk.target_id].add(lnk.source_id)

            # Count components
            visited: set[str] = set()
            comp_count = 0
            for node in remaining:
                if node in visited:
                    continue
                comp_count += 1
                queue: deque[str] = deque([node])
                visited.add(node)
                while queue:
                    cur = queue.popleft()
                    for nb in adj.get(cur, set()):
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)

            if comp_count > base_components:
                bottlenecks.append(nid)

        return sorted(bottlenecks)

    def find_bridges(self) -> list[tuple[str, str]]:
        """Identify bridge links whose removal would disconnect the graph.

        Returns a list of ``(source_id, target_id)`` tuples for bridge links.
        """
        base_components = len(self._topo.connected_components())
        bridges: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for lnk in self._topo.links:
            if not lnk.active:
                continue
            edge_key = tuple(sorted((lnk.source_id, lnk.target_id)))
            if edge_key in seen:
                continue
            seen.add(edge_key)

            # Build adjacency without this edge
            adj: dict[str, set[str]] = defaultdict(set)
            for n in self._topo.node_ids:
                adj[n]
            for other in self._topo.links:
                if not other.active:
                    continue
                other_key = tuple(sorted((other.source_id, other.target_id)))
                if other_key == edge_key:
                    continue
                adj[other.source_id].add(other.target_id)
                adj[other.target_id].add(other.source_id)

            # Count components
            visited: set[str] = set()
            comp_count = 0
            for node in self._topo.node_ids:
                if node in visited:
                    continue
                comp_count += 1
                queue: deque[str] = deque([node])
                visited.add(node)
                while queue:
                    cur = queue.popleft()
                    for nb in adj.get(cur, set()):
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)

            if comp_count > base_components:
                bridges.append(edge_key)

        return sorted(bridges)

    def hop_count(self, from_id: str, to_id: str) -> Optional[int]:
        """Return the hop count (number of links) on the shortest path.

        Returns ``None`` if unreachable.
        """
        path = self.shortest_path(from_id, to_id)
        if path is None:
            return None
        return len(path) - 1

    def node_centrality(self) -> dict[str, float]:
        """Compute degree centrality for each node (0.0 – 1.0).

        Degree centrality = (number of active neighbors) / (N - 1).
        """
        n = len(self._topo.node_ids)
        if n <= 1:
            return {nid: 0.0 for nid in self._topo.node_ids}
        result: dict[str, float] = {}
        for nid in self._topo.node_ids:
            deg = len(self._topo.neighbors(nid))
            result[nid] = deg / (n - 1)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "NodeType",
    "LinkType",
    "NetworkNode",
    "NetworkLink",
    "NetworkTopology",
    "TopologyDiscovery",
    "PathAnalysis",
]

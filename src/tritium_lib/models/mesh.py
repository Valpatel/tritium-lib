# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Mesh networking models — ESP-NOW mesh topology and routing.

These models represent the mesh network state: nodes, routes, topology,
and messages.  Used by both ESP32 firmware (simplified C struct mapping)
and the fleet server (full Pydantic models).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MeshNode(BaseModel):
    """A single node in the ESP-NOW mesh network.

    Each node tracks its own identity, neighbor list, and signal quality
    to adjacent nodes via RSSI measurements.
    """
    node_id: str
    mac: str = ""
    neighbors: list[str] = Field(default_factory=list)  # neighbor node_ids
    hop_count: int = 0  # hops from gateway/root
    rssi_map: dict[str, int] = Field(default_factory=dict)  # neighbor_id -> RSSI
    last_seen: Optional[datetime] = None
    firmware_version: str = ""

    @property
    def neighbor_count(self) -> int:
        return len(self.neighbors)

    def best_neighbor(self) -> Optional[str]:
        """Return the neighbor with the strongest RSSI, or None."""
        if not self.rssi_map:
            return None
        return max(self.rssi_map, key=self.rssi_map.get)


class MeshRoute(BaseModel):
    """A route through the mesh from source to destination.

    Routes are discovered via flooding and scored by hop count and
    aggregate link quality.
    """
    source: str  # node_id
    destination: str  # node_id
    hops: list[str] = Field(default_factory=list)  # intermediate node_ids
    quality_score: float = 0.0  # 0.0 = unusable, 1.0 = perfect

    @property
    def hop_count(self) -> int:
        return len(self.hops)

    @property
    def total_hops(self) -> int:
        """Total hops including source and destination."""
        return len(self.hops) + 2


class MeshEdge(BaseModel):
    """A link between two adjacent mesh nodes."""
    node_a: str
    node_b: str
    rssi: int = 0
    quality: float = 1.0  # 0.0 to 1.0


class MeshTopology(BaseModel):
    """Full mesh network topology — nodes and edges.

    The fleet server builds this from node heartbeats and uses it for
    route planning and partition detection.
    """
    nodes: list[MeshNode] = Field(default_factory=list)
    edges: list[MeshEdge] = Field(default_factory=list)
    partitions: int = 1  # number of disconnected sub-graphs

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def node_by_id(self, node_id: str) -> Optional[MeshNode]:
        """Find a node by its ID, or None."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None


class MeshMessageStatus(str, Enum):
    """Delivery status of a mesh message."""
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    DROPPED = "dropped"


class MeshMessage(BaseModel):
    """A message routed through the mesh network.

    Messages are forwarded hop-by-hop with a TTL to prevent infinite loops.
    """
    message_id: str = ""
    source: str  # source node_id
    destination: str  # destination node_id (or "broadcast")
    payload: bytes = b""
    ttl: int = 10  # time-to-live in hops
    hop_count: int = 0  # hops traversed so far
    status: MeshMessageStatus = MeshMessageStatus.PENDING

    model_config = {"arbitrary_types_allowed": True}

    @property
    def payload_size(self) -> int:
        return len(self.payload)

    @property
    def remaining_ttl(self) -> int:
        return max(0, self.ttl - self.hop_count)

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fleet management models — used by tritium-edge to track and manage
a fleet of ESP32 nodes."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    """Node connectivity status."""
    ONLINE = "online"
    STALE = "stale"
    OFFLINE = "offline"


class FleetNode(BaseModel):
    """A single node in the fleet.

    Extended in Wave 204 with the W199/W200 enriched-heartbeat fields so
    the command center can spot a misbehaving device (memory pressure,
    panic resets, hostile WiFi).  All new fields default to safe values
    so older heartbeats from un-upgraded firmware still parse cleanly.
    """
    device_id: str
    mac: str = ""
    ip: str = ""
    firmware_version: str = "unknown"
    uptime_s: int = 0
    wifi_rssi: int = 0
    free_heap: int = 0
    psram_free: int = 0
    partition: str = ""
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: NodeStatus = NodeStatus.OFFLINE
    capabilities: list[str] = Field(default_factory=list)
    ble_device_count: int = 0
    # Wave 204: W199 enriched heartbeat fields ----------------------------
    # Memory pressure: minimum heap ever observed and largest contiguous
    # free block (fragmentation indicator).  free_psram is the live PSRAM
    # remainder; psram_free above is the historical name kept for
    # backwards compatibility.
    min_free_heap: int = 0
    largest_free_block: int = 0
    free_psram: int = 0
    # Network identity at time of heartbeat.
    wifi_ssid: str = ""
    wifi_channel: int = 0
    # Last reset cause string ("poweron", "panic", "task_wdt", "sw", ...).
    reset_reason: str = ""
    # FreeRTOS task count and number of visible APs at last scan.
    task_count: int = 0
    wifi_scan_count: int = 0


class FleetStatus(BaseModel):
    """Snapshot of the entire fleet."""
    nodes: list[FleetNode] = Field(default_factory=list)
    total_nodes: int = 0
    online_count: int = 0
    ble_total: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NodeEvent(BaseModel):
    """An event from a fleet node."""
    node_id: str
    event_type: str  # online, offline, ota_start, ota_complete, error
    message: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def fleet_health_score(fleet: FleetStatus) -> float:
    """Calculate fleet health as 0.0-1.0.

    Factors:
    - Online ratio (50% weight): fraction of nodes that are ONLINE
    - Avg WiFi RSSI (25% weight): mapped from [-90, -30] to [0, 1]
    - Avg heap usage (25% weight): fraction of free heap vs 300KB typical max
    """
    if fleet.total_nodes == 0:
        return 0.0

    # Online ratio (0.0 - 1.0)
    online_ratio = fleet.online_count / fleet.total_nodes

    online_nodes = [n for n in fleet.nodes if n.status == NodeStatus.ONLINE]
    if not online_nodes:
        return online_ratio * 0.5

    # RSSI score: map [-90, -30] -> [0.0, 1.0], clamped
    avg_rssi = sum(n.wifi_rssi for n in online_nodes) / len(online_nodes)
    rssi_score = max(0.0, min(1.0, (avg_rssi + 90) / 60.0))

    # Heap score: fraction of 300KB typical max free heap
    typical_max_heap = 300_000
    avg_heap = sum(n.free_heap for n in online_nodes) / len(online_nodes)
    heap_score = max(0.0, min(1.0, avg_heap / typical_max_heap))

    score = (online_ratio * 0.5) + (rssi_score * 0.25) + (heap_score * 0.25)
    return round(max(0.0, min(1.0, score)), 3)

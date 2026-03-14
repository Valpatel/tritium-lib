# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Communication channel models.

Manages communication channels (MQTT brokers, TAK servers, federation peers,
WebSocket clients) as first-class entities.  Each channel has a type, endpoint,
authentication credentials, and connection status.

A CommChannel can represent:
- An MQTT broker used for device telemetry
- A TAK server for ATAK/CoT interoperability
- A federation peer for multi-site sharing
- A WebSocket client connection
- A serial port (Meshtastic radio, MeshCore)
- An HTTP/REST endpoint (fleet server, external API)
"""

import time
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ChannelType(str, Enum):
    """Type of communication channel."""
    MQTT = "mqtt"
    TAK = "tak"
    WEBSOCKET = "websocket"
    FEDERATION = "federation"
    SERIAL = "serial"
    HTTP = "http"
    ESPNOW = "espnow"
    LORA = "lora"


class ChannelStatus(str, Enum):
    """Connection status of a channel."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    DISABLED = "disabled"


class AuthType(str, Enum):
    """Authentication method for a channel."""
    NONE = "none"
    BASIC = "basic"           # username/password
    TOKEN = "token"           # API key or bearer token
    CERTIFICATE = "certificate"  # TLS client certificate
    PSK = "psk"               # Pre-shared key


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ChannelAuth(BaseModel):
    """Authentication configuration for a channel."""

    auth_type: AuthType = AuthType.NONE
    username: str = ""
    password: str = ""
    token: str = ""
    cert_path: str = ""
    key_path: str = ""
    ca_path: str = ""
    psk: str = ""


class CommChannel(BaseModel):
    """A managed communication channel.

    Represents any external communication endpoint that the Tritium
    system connects to for sending or receiving data.
    """

    channel_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unnamed Channel"
    description: str = ""
    channel_type: ChannelType = ChannelType.MQTT
    endpoint: str = ""          # host:port, URL, serial path, etc.
    auth: ChannelAuth = Field(default_factory=ChannelAuth)
    status: ChannelStatus = ChannelStatus.DISCONNECTED
    enabled: bool = True
    priority: int = 0           # Higher = preferred when multiple available
    tags: list[str] = Field(default_factory=list)

    # Connection metrics
    last_connected: Optional[float] = None
    last_error: str = ""
    reconnect_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    latency_ms: Optional[float] = None

    # Timestamps
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # Channel-specific config (flexible dict for type-specific settings)
    config: dict = Field(default_factory=dict)


class ChannelHealth(BaseModel):
    """Health summary for a communication channel."""

    channel_id: str
    channel_type: ChannelType
    status: ChannelStatus
    uptime_pct: float = 0.0     # Percentage of time connected
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0     # Errors per hour
    throughput_bps: float = 0.0  # Bytes per second (combined in/out)
    last_activity: Optional[float] = None


class ChannelInventory(BaseModel):
    """Summary of all communication channels."""

    total: int = 0
    connected: int = 0
    disconnected: int = 0
    error: int = 0
    disabled: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    channels: list[ChannelHealth] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def summarize_channels(channels: list[CommChannel]) -> ChannelInventory:
    """Build a summary inventory of communication channels."""
    inv = ChannelInventory(total=len(channels))
    by_type: dict[str, int] = {}

    for ch in channels:
        # Status counts
        if ch.status == ChannelStatus.CONNECTED:
            inv.connected += 1
        elif ch.status == ChannelStatus.DISCONNECTED:
            inv.disconnected += 1
        elif ch.status == ChannelStatus.ERROR:
            inv.error += 1
        elif ch.status == ChannelStatus.DISABLED:
            inv.disabled += 1

        # Type counts
        t = ch.channel_type.value
        by_type[t] = by_type.get(t, 0) + 1

        # Health entry
        inv.channels.append(ChannelHealth(
            channel_id=ch.channel_id,
            channel_type=ch.channel_type,
            status=ch.status,
            last_activity=ch.last_connected,
        ))

    inv.by_type = by_type
    return inv


def select_best_channel(
    channels: list[CommChannel],
    channel_type: Optional[ChannelType] = None,
) -> Optional[CommChannel]:
    """Select the best available channel of a given type.

    Prefers connected channels with highest priority and lowest latency.
    """
    candidates = [
        ch for ch in channels
        if ch.enabled
        and ch.status == ChannelStatus.CONNECTED
        and (channel_type is None or ch.channel_type == channel_type)
    ]

    if not candidates:
        return None

    # Sort by priority (desc), then latency (asc)
    candidates.sort(
        key=lambda c: (-c.priority, c.latency_ms or float("inf")),
    )
    return candidates[0]

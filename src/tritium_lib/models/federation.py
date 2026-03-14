# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Multi-site federation models.

Tritium installations can federate to share targets, dossiers, and
situational awareness across sites.  Each site maintains its own
database but can selectively share intelligence with peers.

A FederatedSite describes a remote Tritium installation.
A SiteConnection tracks the real-time link state.
SharedTarget wraps a target that has been exported/imported across sites.
FederationMessage is the wire-format envelope for inter-site comms.
"""

import time
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SiteRole(str, Enum):
    """Role of a site in the federation."""
    PRIMARY = "primary"      # Origin site — owns the data
    SECONDARY = "secondary"  # Receives shared data from primary
    PEER = "peer"            # Bidirectional sharing


class ConnectionState(str, Enum):
    """Real-time state of a site-to-site link."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class SharePolicy(str, Enum):
    """What data a site is willing to share."""
    NONE = "none"            # No sharing
    TARGETS_ONLY = "targets_only"  # Real-time target positions
    DOSSIERS = "dossiers"    # Full dossier intelligence
    FULL = "full"            # Everything (targets + dossiers + alerts + commands)


class FederationMessageType(str, Enum):
    """Type of federation message."""
    SITE_ANNOUNCE = "site_announce"
    SITE_HEARTBEAT = "site_heartbeat"
    TARGET_UPDATE = "target_update"
    TARGET_REMOVE = "target_remove"
    DOSSIER_SYNC = "dossier_sync"
    ALERT = "alert"
    COMMAND = "command"
    ACK = "ack"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FederatedSite(BaseModel):
    """Describes a remote Tritium installation in the federation."""

    site_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unknown Site"
    description: str = ""
    mqtt_host: str = ""
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_topic_prefix: str = "tritium"
    role: SiteRole = SiteRole.PEER
    share_policy: SharePolicy = SharePolicy.TARGETS_ONLY
    lat: Optional[float] = None
    lng: Optional[float] = None
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    registered_at: float = Field(default_factory=time.time)


class SiteConnection(BaseModel):
    """Real-time state of a connection to a federated site."""

    site_id: str
    state: ConnectionState = ConnectionState.DISCONNECTED
    last_heartbeat: Optional[float] = None
    last_error: str = ""
    latency_ms: Optional[float] = None
    targets_shared: int = 0
    targets_received: int = 0
    dossiers_synced: int = 0
    connected_since: Optional[float] = None
    bytes_sent: int = 0
    bytes_received: int = 0


class SharedTarget(BaseModel):
    """A target that has been shared across the federation.

    Wraps the essential tracking data so a remote site can display it
    on its tactical map without needing the full internal target model.
    """

    target_id: str
    source_site_id: str
    name: str = ""
    entity_type: str = "unknown"
    classification: str = "unknown"
    alliance: str = "unknown"
    lat: Optional[float] = None
    lng: Optional[float] = None
    local_x: float = 0.0
    local_y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    confidence: float = 0.5
    source: str = ""  # ble, yolo, mesh, wifi
    last_seen: float = Field(default_factory=time.time)
    identifiers: dict[str, str] = Field(default_factory=dict)
    threat_level: str = "none"
    dossier_id: Optional[str] = None


class FederationMessage(BaseModel):
    """Wire-format envelope for inter-site federation messages.

    Published on MQTT topic: tritium/federation/{source_site}/{msg_type}
    """

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: FederationMessageType
    source_site_id: str
    target_site_id: str = ""  # Empty = broadcast to all peers
    timestamp: float = Field(default_factory=time.time)
    payload: dict = Field(default_factory=dict)
    ttl: int = 3  # Max hops for multi-hop federation


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def federation_topic(site_id: str, msg_type: FederationMessageType) -> str:
    """Build the MQTT topic for a federation message.

    Format: tritium/federation/{site_id}/{msg_type}
    """
    return f"tritium/federation/{site_id}/{msg_type.value}"


def is_message_expired(msg: FederationMessage, max_age_s: float = 300.0) -> bool:
    """Check if a federation message is too old to process.

    Default max age is 5 minutes.
    """
    return (time.time() - msg.timestamp) > max_age_s

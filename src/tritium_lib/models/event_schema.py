# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Typed event schemas for the Tritium EventBus.

Every event that flows through the system should have a defined schema
here.  This enables validation, documentation, and IDE autocomplete
for all EventBus consumers.

Event naming convention:
  - domain.action (e.g., "device.heartbeat", "target.updated")
  - WebSocket events use prefix mapping (e.g., "sim_telemetry" -> "amy_sim_telemetry")
  - Fleet events use "fleet." prefix
  - Mesh events use "mesh_" prefix
  - TAK events use "tak_" prefix
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Event domain categories
# ---------------------------------------------------------------------------

class EventDomain(str, Enum):
    """Top-level event domain categories."""
    SIMULATION = "simulation"
    COMBAT = "combat"
    GAME = "game"
    NPC = "npc"
    FLEET = "fleet"
    MESH = "mesh"
    EDGE = "edge"
    TAK = "tak"
    SENSOR = "sensor"
    TARGET = "target"
    DOSSIER = "dossier"
    FEDERATION = "federation"
    AMY = "amy"
    AUDIO = "audio"
    MISSION = "mission"
    HAZARD = "hazard"
    UNIT = "unit"


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

@dataclass
class TritiumEvent:
    """Base schema for all Tritium events."""
    event_type: str
    domain: EventDomain
    description: str = ""


# ---------------------------------------------------------------------------
# Simulation events
# ---------------------------------------------------------------------------

@dataclass
class SimTelemetryEvent(TritiumEvent):
    """Per-entity position/state update from the simulation engine."""
    event_type: str = "sim_telemetry"
    domain: EventDomain = EventDomain.SIMULATION
    description: str = "Single target telemetry update (position, heading, speed, status)"
    # Data fields:
    target_id: str = ""
    position_x: float = 0.0
    position_y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    alliance: str = "unknown"
    asset_type: str = "unknown"
    status: str = "active"
    battery: float = 1.0


@dataclass
class SimTelemetryBatchEvent(TritiumEvent):
    """Batched telemetry for multiple targets (sent as array)."""
    event_type: str = "sim_telemetry_batch"
    domain: EventDomain = EventDomain.SIMULATION
    description: str = "Array of target telemetry updates, batched for bandwidth efficiency"


# ---------------------------------------------------------------------------
# Game lifecycle events
# ---------------------------------------------------------------------------

@dataclass
class GameStateChangeEvent(TritiumEvent):
    """Game state transition (idle -> playing -> paused -> game_over)."""
    event_type: str = "game_state_change"
    domain: EventDomain = EventDomain.GAME
    description: str = "Game state machine transition"
    state: str = ""
    wave: int = 0
    score: int = 0


@dataclass
class WaveStartEvent(TritiumEvent):
    """New wave begins in wave-based game mode."""
    event_type: str = "wave_start"
    domain: EventDomain = EventDomain.GAME
    description: str = "Wave N begins with hostile spawn"
    wave: int = 0
    hostile_count: int = 0


@dataclass
class WaveCompleteEvent(TritiumEvent):
    """Wave cleared — all hostiles eliminated."""
    event_type: str = "wave_complete"
    domain: EventDomain = EventDomain.GAME
    description: str = "Wave N completed"
    wave: int = 0


@dataclass
class GameOverEvent(TritiumEvent):
    """Game ended (victory or defeat)."""
    event_type: str = "game_over"
    domain: EventDomain = EventDomain.GAME
    description: str = "Game ended with final score"
    victory: bool = False
    score: int = 0
    waves_completed: int = 0


# ---------------------------------------------------------------------------
# Combat events
# ---------------------------------------------------------------------------

@dataclass
class ProjectileFiredEvent(TritiumEvent):
    """A unit fires a projectile."""
    event_type: str = "projectile_fired"
    domain: EventDomain = EventDomain.COMBAT
    description: str = "Projectile launched from source to target position"
    source_id: str = ""
    target_x: float = 0.0
    target_y: float = 0.0


@dataclass
class ProjectileHitEvent(TritiumEvent):
    """A projectile impacts a target."""
    event_type: str = "projectile_hit"
    domain: EventDomain = EventDomain.COMBAT
    description: str = "Projectile impact on target"
    target_id: str = ""
    damage: float = 0.0


@dataclass
class TargetEliminatedEvent(TritiumEvent):
    """A target is destroyed/eliminated."""
    event_type: str = "target_eliminated"
    domain: EventDomain = EventDomain.COMBAT
    description: str = "Target eliminated (health <= 0)"
    target_id: str = ""
    eliminated_by: str = ""


@dataclass
class EliminationStreakEvent(TritiumEvent):
    """Kill streak achievement."""
    event_type: str = "elimination_streak"
    domain: EventDomain = EventDomain.COMBAT
    description: str = "Consecutive eliminations without pause"
    unit_id: str = ""
    streak: int = 0


@dataclass
class WeaponJamEvent(TritiumEvent):
    """Weapon malfunction."""
    event_type: str = "weapon_jam"
    domain: EventDomain = EventDomain.COMBAT
    description: str = "Weapon jam requiring clear cycle"
    unit_id: str = ""


@dataclass
class AmmoDepletedEvent(TritiumEvent):
    """Unit out of ammo."""
    event_type: str = "ammo_depleted"
    domain: EventDomain = EventDomain.COMBAT
    description: str = "Unit ammo exhausted"
    unit_id: str = ""


@dataclass
class AmmoLowEvent(TritiumEvent):
    """Unit ammo running low."""
    event_type: str = "ammo_low"
    domain: EventDomain = EventDomain.COMBAT
    description: str = "Unit ammo below threshold"
    unit_id: str = ""
    remaining: int = 0


# ---------------------------------------------------------------------------
# NPC intelligence events
# ---------------------------------------------------------------------------

@dataclass
class NpcThoughtEvent(TritiumEvent):
    """NPC generates a context-aware thought."""
    event_type: str = "npc_thought"
    domain: EventDomain = EventDomain.NPC
    description: str = "NPC inner monologue / thought bubble"
    npc_id: str = ""
    thought: str = ""


@dataclass
class NpcThoughtClearEvent(TritiumEvent):
    """Clear NPC thought bubble."""
    event_type: str = "npc_thought_clear"
    domain: EventDomain = EventDomain.NPC
    description: str = "Dismiss NPC thought bubble"
    npc_id: str = ""


@dataclass
class NpcAllianceChangeEvent(TritiumEvent):
    """NPC changes alliance (defection, conversion, etc.)."""
    event_type: str = "npc_alliance_change"
    domain: EventDomain = EventDomain.NPC
    description: str = "NPC switches alliance"
    npc_id: str = ""
    old_alliance: str = ""
    new_alliance: str = ""


# ---------------------------------------------------------------------------
# Threat / escalation events
# ---------------------------------------------------------------------------

@dataclass
class EscalationChangeEvent(TritiumEvent):
    """Threat escalation level changed."""
    event_type: str = "escalation_change"
    domain: EventDomain = EventDomain.TARGET
    description: str = "Threat escalation level increase or decrease"
    level: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Edge / fleet events
# ---------------------------------------------------------------------------

@dataclass
class DeviceHeartbeatEvent(TritiumEvent):
    """Edge device periodic heartbeat."""
    event_type: str = "fleet.heartbeat"
    domain: EventDomain = EventDomain.FLEET
    description: str = "Edge device heartbeat with telemetry"
    device_id: str = ""
    uptime_s: int = 0
    battery_pct: float = 0.0


@dataclass
class DeviceOnlineEvent(TritiumEvent):
    """Edge device came online."""
    event_type: str = "fleet.device_online"
    domain: EventDomain = EventDomain.FLEET
    description: str = "Edge device connected to fleet"
    device_id: str = ""


@dataclass
class DeviceOfflineEvent(TritiumEvent):
    """Edge device went offline."""
    event_type: str = "fleet.device_offline"
    domain: EventDomain = EventDomain.FLEET
    description: str = "Edge device disconnected from fleet"
    device_id: str = ""


@dataclass
class BleUpdateEvent(TritiumEvent):
    """BLE scan results from edge device."""
    event_type: str = "edge:ble_update"
    domain: EventDomain = EventDomain.EDGE
    description: str = "BLE device scan results from edge scanner"
    device_id: str = ""
    devices: list = field(default_factory=list)


@dataclass
class WiFiUpdateEvent(TritiumEvent):
    """WiFi scan results from edge device."""
    event_type: str = "edge:wifi_update"
    domain: EventDomain = EventDomain.EDGE
    description: str = "WiFi network scan results from edge scanner"
    device_id: str = ""
    networks: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mesh events
# ---------------------------------------------------------------------------

@dataclass
class MeshNodeUpdateEvent(TritiumEvent):
    """Meshtastic/LoRa mesh node update."""
    event_type: str = "mesh_node_update"
    domain: EventDomain = EventDomain.MESH
    description: str = "Mesh radio node position/status update"
    node_id: str = ""


@dataclass
class MeshMessageEvent(TritiumEvent):
    """Message received over mesh network."""
    event_type: str = "mesh_message"
    domain: EventDomain = EventDomain.MESH
    description: str = "Text message received from mesh radio"
    sender: str = ""
    content: str = ""


# ---------------------------------------------------------------------------
# Sensor events
# ---------------------------------------------------------------------------

@dataclass
class SensorTriggeredEvent(TritiumEvent):
    """Sensor zone triggered (motion, proximity, etc.)."""
    event_type: str = "sensor_triggered"
    domain: EventDomain = EventDomain.SENSOR
    description: str = "Sensor zone activated by detected entity"
    sensor_id: str = ""
    trigger_type: str = ""


@dataclass
class SensorClearedEvent(TritiumEvent):
    """Sensor zone cleared."""
    event_type: str = "sensor_cleared"
    domain: EventDomain = EventDomain.SENSOR
    description: str = "Sensor zone returned to normal"
    sensor_id: str = ""


@dataclass
class DetectionEvent(TritiumEvent):
    """YOLO/camera detection result."""
    event_type: str = "detection"
    domain: EventDomain = EventDomain.SENSOR
    description: str = "Object detected by camera/YOLO pipeline"
    class_name: str = ""
    confidence: float = 0.0
    bbox: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# TAK/CoT events
# ---------------------------------------------------------------------------

@dataclass
class TakContactEvent(TritiumEvent):
    """CoT contact received from TAK network."""
    event_type: str = "tak_contact"
    domain: EventDomain = EventDomain.TAK
    description: str = "Cursor on Target contact from ATAK/WinTAK"
    uid: str = ""
    callsign: str = ""


# ---------------------------------------------------------------------------
# Dossier events
# ---------------------------------------------------------------------------

@dataclass
class DossierCreatedEvent(TritiumEvent):
    """New target dossier created."""
    event_type: str = "dossier_created"
    domain: EventDomain = EventDomain.DOSSIER
    description: str = "New persistent identity dossier created for entity"
    dossier_id: str = ""
    entity_type: str = ""


# ---------------------------------------------------------------------------
# Federation events
# ---------------------------------------------------------------------------

@dataclass
class FederationSiteAddedEvent(TritiumEvent):
    """Remote Tritium site joined federation."""
    event_type: str = "federation:site_added"
    domain: EventDomain = EventDomain.FEDERATION
    description: str = "New remote site joined the multi-site federation"
    site_id: str = ""


@dataclass
class FederationTargetSharedEvent(TritiumEvent):
    """Target shared to federation."""
    event_type: str = "federation:target_shared"
    domain: EventDomain = EventDomain.FEDERATION
    description: str = "Local target shared with federated sites"
    target_id: str = ""


@dataclass
class FederationTargetReceivedEvent(TritiumEvent):
    """Target received from federation."""
    event_type: str = "federation:target_received"
    domain: EventDomain = EventDomain.FEDERATION
    description: str = "Target received from remote federated site"
    target_id: str = ""
    source_site: str = ""


# ---------------------------------------------------------------------------
# Hazard events
# ---------------------------------------------------------------------------

@dataclass
class HazardSpawnedEvent(TritiumEvent):
    """Environmental hazard appeared."""
    event_type: str = "hazard_spawned"
    domain: EventDomain = EventDomain.HAZARD
    description: str = "Environmental hazard zone created"
    hazard_id: str = ""
    hazard_type: str = ""


@dataclass
class HazardExpiredEvent(TritiumEvent):
    """Environmental hazard expired."""
    event_type: str = "hazard_expired"
    domain: EventDomain = EventDomain.HAZARD
    description: str = "Environmental hazard zone expired"
    hazard_id: str = ""


# ---------------------------------------------------------------------------
# Unit events
# ---------------------------------------------------------------------------

@dataclass
class UnitDispatchedEvent(TritiumEvent):
    """Unit dispatched to a position or target."""
    event_type: str = "unit_dispatched"
    domain: EventDomain = EventDomain.UNIT
    description: str = "Unit dispatched by Amy or operator"
    unit_id: str = ""
    destination_x: float = 0.0
    destination_y: float = 0.0


@dataclass
class UnitSignalEvent(TritiumEvent):
    """Unit communication signal (ping, rally, etc.)."""
    event_type: str = "unit_signal"
    domain: EventDomain = EventDomain.UNIT
    description: str = "Visual communication signal from a unit"
    unit_id: str = ""
    signal_type: str = ""


@dataclass
class FormationCreatedEvent(TritiumEvent):
    """Squad formation established."""
    event_type: str = "formation_created"
    domain: EventDomain = EventDomain.UNIT
    description: str = "Squad formation pattern created"
    formation_type: str = ""
    unit_ids: list = field(default_factory=list)


@dataclass
class ModeChangeEvent(TritiumEvent):
    """Amy operational mode change."""
    event_type: str = "mode_change"
    domain: EventDomain = EventDomain.AMY
    description: str = "Amy switched operational mode"
    old_mode: str = ""
    new_mode: str = ""


# ---------------------------------------------------------------------------
# Mission events
# ---------------------------------------------------------------------------

@dataclass
class MissionProgressEvent(TritiumEvent):
    """Mission objective progress update."""
    event_type: str = "mission_progress"
    domain: EventDomain = EventDomain.MISSION
    description: str = "Mission objective progress changed"
    objective_id: str = ""
    progress: float = 0.0


@dataclass
class ScenarioGeneratedEvent(TritiumEvent):
    """New scenario generated."""
    event_type: str = "scenario_generated"
    domain: EventDomain = EventDomain.MISSION
    description: str = "New battle scenario generated"
    scenario_name: str = ""


@dataclass
class ZoneViolationEvent(TritiumEvent):
    """Target entered a restricted zone."""
    event_type: str = "zone_violation"
    domain: EventDomain = EventDomain.TARGET
    description: str = "Target crossed geofence boundary"
    target_id: str = ""
    zone_id: str = ""


# ---------------------------------------------------------------------------
# Registry — all event types for validation and documentation
# ---------------------------------------------------------------------------

ALL_EVENT_TYPES: dict[str, type[TritiumEvent]] = {
    "sim_telemetry": SimTelemetryEvent,
    "sim_telemetry_batch": SimTelemetryBatchEvent,
    "game_state_change": GameStateChangeEvent,
    "wave_start": WaveStartEvent,
    "wave_complete": WaveCompleteEvent,
    "game_over": GameOverEvent,
    "projectile_fired": ProjectileFiredEvent,
    "projectile_hit": ProjectileHitEvent,
    "target_eliminated": TargetEliminatedEvent,
    "elimination_streak": EliminationStreakEvent,
    "weapon_jam": WeaponJamEvent,
    "ammo_depleted": AmmoDepletedEvent,
    "ammo_low": AmmoLowEvent,
    "npc_thought": NpcThoughtEvent,
    "npc_thought_clear": NpcThoughtClearEvent,
    "npc_alliance_change": NpcAllianceChangeEvent,
    "escalation_change": EscalationChangeEvent,
    "fleet.heartbeat": DeviceHeartbeatEvent,
    "fleet.device_online": DeviceOnlineEvent,
    "fleet.device_offline": DeviceOfflineEvent,
    "edge:ble_update": BleUpdateEvent,
    "edge:wifi_update": WiFiUpdateEvent,
    "mesh_node_update": MeshNodeUpdateEvent,
    "mesh_message": MeshMessageEvent,
    "sensor_triggered": SensorTriggeredEvent,
    "sensor_cleared": SensorClearedEvent,
    "detection": DetectionEvent,
    "tak_contact": TakContactEvent,
    "dossier_created": DossierCreatedEvent,
    "federation:site_added": FederationSiteAddedEvent,
    "federation:target_shared": FederationTargetSharedEvent,
    "federation:target_received": FederationTargetReceivedEvent,
    "hazard_spawned": HazardSpawnedEvent,
    "hazard_expired": HazardExpiredEvent,
    "unit_dispatched": UnitDispatchedEvent,
    "unit_signal": UnitSignalEvent,
    "formation_created": FormationCreatedEvent,
    "mode_change": ModeChangeEvent,
    "mission_progress": MissionProgressEvent,
    "scenario_generated": ScenarioGeneratedEvent,
    "zone_violation": ZoneViolationEvent,
}


def validate_event_type(event_type: str) -> bool:
    """Check if an event type is registered in the schema."""
    return event_type in ALL_EVENT_TYPES


def get_event_schema(event_type: str) -> Optional[type[TritiumEvent]]:
    """Get the schema class for an event type."""
    return ALL_EVENT_TYPES.get(event_type)


def list_event_types() -> list[dict[str, str]]:
    """Return a list of all registered event types with descriptions."""
    result = []
    for name, cls in ALL_EVENT_TYPES.items():
        instance = cls()
        result.append({
            "event_type": name,
            "domain": instance.domain.value,
            "description": instance.description,
        })
    return result

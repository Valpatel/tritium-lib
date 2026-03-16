# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tritium ontology schema — typed definitions for all entity types,
relationship types, interfaces, and properties.

The ontology is the semantic backbone of the Tritium system. Every object
that flows through MQTT, gets stored in SQLite, or appears on a dashboard
is an instance of an ontology entity type.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DataType(str, Enum):
    """Property data types supported by the ontology."""
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    TIMESTAMP = "timestamp"
    GEO_POINT = "geo_point"
    JSON = "json"


class Cardinality(str, Enum):
    """Relationship cardinality from source to target."""
    ONE = "one"
    MANY = "many"


class TypeStatus(str, Enum):
    """Lifecycle status for entity and relationship types."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PropertyDef(BaseModel):
    """Definition of a single typed property on an entity or relationship."""
    name: str
    data_type: DataType
    required: bool = False
    default: Any = None
    description: str = ""


class InterfaceDef(BaseModel):
    """An interface that entity types can implement for polymorphic queries."""
    api_name: str
    display_name: str
    description: str = ""
    required_properties: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity & relationship types
# ---------------------------------------------------------------------------

class EntityType(BaseModel):
    """Schema definition for a category of ontology objects."""
    api_name: str
    display_name: str
    description: str = ""
    primary_key_field: str
    properties: dict[str, PropertyDef] = Field(default_factory=dict)
    interfaces: list[str] = Field(default_factory=list)
    status: TypeStatus = TypeStatus.ACTIVE


class RelationshipType(BaseModel):
    """Schema definition for a directed link between two entity types."""
    api_name: str
    display_name: str
    from_type: str
    to_type: str
    cardinality: Cardinality = Cardinality.MANY
    properties: dict[str, PropertyDef] = Field(default_factory=dict)
    description: str = ""
    status: TypeStatus = TypeStatus.ACTIVE


# ---------------------------------------------------------------------------
# Top-level schema container
# ---------------------------------------------------------------------------

class OntologySchema(BaseModel):
    """Complete ontology schema — all types, relationships, and interfaces."""
    version: str
    entity_types: dict[str, EntityType] = Field(default_factory=dict)
    relationship_types: dict[str, RelationshipType] = Field(default_factory=dict)
    interfaces: dict[str, InterfaceDef] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prop(name: str, dt: DataType, *, required: bool = False,
          default: Any = None, description: str = "") -> PropertyDef:
    return PropertyDef(name=name, data_type=dt, required=required,
                       default=default, description=description)


def _entity(api_name: str, display_name: str, pk: str,
            props: list[PropertyDef], *,
            interfaces: Optional[list[str]] = None,
            description: str = "") -> EntityType:
    return EntityType(
        api_name=api_name,
        display_name=display_name,
        description=description,
        primary_key_field=pk,
        properties={p.name: p for p in props},
        interfaces=interfaces or [],
    )


def _rel(api_name: str, display_name: str, from_type: str, to_type: str,
         cardinality: Cardinality = Cardinality.MANY, *,
         props: Optional[list[PropertyDef]] = None,
         description: str = "") -> RelationshipType:
    return RelationshipType(
        api_name=api_name,
        display_name=display_name,
        from_type=from_type,
        to_type=to_type,
        cardinality=cardinality,
        properties={p.name: p for p in (props or [])},
        description=description,
    )


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

_INTERFACES: dict[str, InterfaceDef] = {
    "trackable": InterfaceDef(
        api_name="trackable",
        display_name="Trackable",
        description="Entities with a spatial position and temporal presence.",
        required_properties=["latitude", "longitude", "last_seen"],
    ),
    "identifiable": InterfaceDef(
        api_name="identifiable",
        display_name="Identifiable",
        description="Entities with a unique hardware or logical identifier.",
        required_properties=["identifier", "identifier_type"],
    ),
    "classifiable": InterfaceDef(
        api_name="classifiable",
        display_name="Classifiable",
        description="Entities that can receive classification labels.",
        required_properties=["classification", "confidence"],
    ),
}

# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

_ENTITY_TYPES: dict[str, EntityType] = {}

# -- Physical entities ------------------------------------------------------

_ENTITY_TYPES["device"] = _entity(
    "device", "Device", "mac_address",
    [
        _prop("mac_address", DataType.STRING, required=True,
              description="IEEE MAC address"),
        _prop("device_type", DataType.STRING,
              description="wifi_ap, wifi_client, ble, etc."),
        _prop("manufacturer", DataType.STRING),
        _prop("name", DataType.STRING),
        _prop("first_seen", DataType.TIMESTAMP),
        _prop("last_seen", DataType.TIMESTAMP),
        _prop("rssi", DataType.INT),
        _prop("classification", DataType.STRING),
        _prop("confidence", DataType.FLOAT, default=0.0),
        _prop("identifier", DataType.STRING,
              description="Canonical unique ID (usually MAC)"),
        _prop("identifier_type", DataType.STRING, default="mac"),
        _prop("latitude", DataType.FLOAT),
        _prop("longitude", DataType.FLOAT),
    ],
    interfaces=["trackable", "identifiable", "classifiable"],
    description="A wireless device observed by edge scanners.",
)

_ENTITY_TYPES["mesh_node"] = _entity(
    "mesh_node", "Mesh Node", "node_id",
    [
        _prop("node_id", DataType.STRING, required=True),
        _prop("name", DataType.STRING),
        _prop("last_seen", DataType.TIMESTAMP),
        _prop("hop_count", DataType.INT, default=0),
        _prop("channel", DataType.INT),
        _prop("latitude", DataType.FLOAT),
        _prop("longitude", DataType.FLOAT),
        _prop("identifier", DataType.STRING),
        _prop("identifier_type", DataType.STRING, default="node_id"),
    ],
    interfaces=["trackable", "identifiable"],
    description="A node in the ESP-NOW or Meshtastic mesh network.",
)

_ENTITY_TYPES["edge_unit"] = _entity(
    "edge_unit", "Edge Unit", "device_id",
    [
        _prop("device_id", DataType.STRING, required=True),
        _prop("hostname", DataType.STRING),
        _prop("ip_address", DataType.STRING),
        _prop("firmware_version", DataType.STRING),
        _prop("uptime_s", DataType.INT),
        _prop("battery_pct", DataType.FLOAT),
        _prop("latitude", DataType.FLOAT),
        _prop("longitude", DataType.FLOAT),
        _prop("last_seen", DataType.TIMESTAMP),
        _prop("identifier", DataType.STRING),
        _prop("identifier_type", DataType.STRING, default="device_id"),
    ],
    interfaces=["trackable", "identifiable"],
    description="A Tritium edge device running Tritium-OS firmware.",
)

# -- Observations -----------------------------------------------------------

_ENTITY_TYPES["sighting"] = _entity(
    "sighting", "Sighting", "sighting_id",
    [
        _prop("sighting_id", DataType.STRING, required=True),
        _prop("mac", DataType.STRING, required=True),
        _prop("timestamp", DataType.TIMESTAMP, required=True),
        _prop("rssi", DataType.INT),
        _prop("channel", DataType.INT),
        _prop("latitude", DataType.FLOAT),
        _prop("longitude", DataType.FLOAT),
        _prop("source_device", DataType.STRING,
              description="device_id of the scanner that made this sighting"),
    ],
    description="A single observation of a device by an edge scanner.",
)

_ENTITY_TYPES["track"] = _entity(
    "track", "Track", "track_id",
    [
        _prop("track_id", DataType.STRING, required=True),
        _prop("entity_type", DataType.STRING),
        _prop("first_seen", DataType.TIMESTAMP),
        _prop("last_seen", DataType.TIMESTAMP),
        _prop("sighting_count", DataType.INT, default=0),
        _prop("confidence", DataType.FLOAT, default=0.0),
        _prop("classification", DataType.STRING),
        _prop("latitude", DataType.FLOAT),
        _prop("longitude", DataType.FLOAT),
        _prop("identifier", DataType.STRING),
        _prop("identifier_type", DataType.STRING, default="track_id"),
    ],
    interfaces=["trackable", "identifiable", "classifiable"],
    description="A correlated sequence of sightings for one real-world entity.",
)

# -- Communications ---------------------------------------------------------

_ENTITY_TYPES["chat_message"] = _entity(
    "chat_message", "Chat Message", "msg_id",
    [
        _prop("msg_id", DataType.STRING, required=True),
        _prop("sender", DataType.STRING, required=True),
        _prop("content", DataType.STRING, required=True),
        _prop("timestamp", DataType.TIMESTAMP, required=True),
        _prop("channel", DataType.STRING),
        _prop("mesh_hops", DataType.INT, default=0),
    ],
    description="A message sent over the mesh chat system.",
)

_ENTITY_TYPES["command"] = _entity(
    "command", "Command", "cmd_id",
    [
        _prop("cmd_id", DataType.STRING, required=True),
        _prop("action", DataType.STRING, required=True),
        _prop("params", DataType.JSON),
        _prop("source", DataType.STRING,
              description="Originator (SC user, Amy, automation rule)"),
        _prop("target", DataType.STRING,
              description="Target device_id"),
        _prop("timestamp", DataType.TIMESTAMP, required=True),
        _prop("status", DataType.STRING, default="pending"),
    ],
    description="A command dispatched to an edge device.",
)

# -- Intelligence -----------------------------------------------------------

_ENTITY_TYPES["classification"] = _entity(
    "classification", "Classification", "class_id",
    [
        _prop("class_id", DataType.STRING, required=True),
        _prop("entity_ref", DataType.STRING, required=True,
              description="Primary key of the classified entity"),
        _prop("label", DataType.STRING, required=True),
        _prop("confidence", DataType.FLOAT, default=0.0),
        _prop("method", DataType.STRING,
              description="How the classification was made"),
        _prop("timestamp", DataType.TIMESTAMP),
    ],
    description="A classification label applied to a track or device.",
)

_ENTITY_TYPES["investigation"] = _entity(
    "investigation", "Investigation", "inv_id",
    [
        _prop("inv_id", DataType.STRING, required=True),
        _prop("title", DataType.STRING, required=True),
        _prop("description", DataType.STRING),
        _prop("created", DataType.TIMESTAMP),
        _prop("status", DataType.STRING, default="open"),
        _prop("analyst", DataType.STRING),
    ],
    description="An analyst-driven investigation linking entities of interest.",
)

_ENTITY_TYPES["zone"] = _entity(
    "zone", "Zone", "zone_id",
    [
        _prop("zone_id", DataType.STRING, required=True),
        _prop("name", DataType.STRING, required=True),
        _prop("boundary", DataType.JSON,
              description="GeoJSON polygon defining the zone boundary"),
        _prop("zone_type", DataType.STRING,
              description="geofence, coverage, interest"),
    ],
    description="A geographic area used for geofencing and spatial queries.",
)

# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

_RELATIONSHIP_TYPES: dict[str, RelationshipType] = {}

# Detection
_RELATIONSHIP_TYPES["detected_by"] = _rel(
    "detected_by", "Detected By", "sighting", "edge_unit",
    Cardinality.ONE,
    description="Which edge scanner produced this sighting.",
)
_RELATIONSHIP_TYPES["sighting_of"] = _rel(
    "sighting_of", "Sighting Of", "sighting", "device",
    Cardinality.ONE,
    description="The device that was observed in this sighting.",
)
_RELATIONSHIP_TYPES["part_of_track"] = _rel(
    "part_of_track", "Part Of Track", "sighting", "track",
    Cardinality.ONE,
    description="The track this sighting has been correlated into.",
)

# Network
_RELATIONSHIP_TYPES["connected_to"] = _rel(
    "connected_to", "Connected To", "device", "device",
    Cardinality.MANY,
    description="Network association (e.g., client to access point).",
)
_RELATIONSHIP_TYPES["mesh_peer"] = _rel(
    "mesh_peer", "Mesh Peer", "mesh_node", "mesh_node",
    Cardinality.MANY,
    description="Mesh network adjacency between nodes.",
)
_RELATIONSHIP_TYPES["managed_by"] = _rel(
    "managed_by", "Managed By", "edge_unit", "edge_unit",
    Cardinality.ONE,
    description="Fleet hierarchy — which edge unit manages this one.",
)

# Spatial
_RELATIONSHIP_TYPES["located_at"] = _rel(
    "located_at", "Located At", "device", "zone",
    Cardinality.MANY,
    description="Device is currently within this zone.",
)
_RELATIONSHIP_TYPES["within_zone"] = _rel(
    "within_zone", "Within Zone", "edge_unit", "zone",
    Cardinality.MANY,
    description="Edge unit is deployed within this zone.",
)

# Intelligence
_RELATIONSHIP_TYPES["classified_as"] = _rel(
    "classified_as", "Classified As", "track", "classification",
    Cardinality.MANY,
    description="Classification labels assigned to this track.",
)
_RELATIONSHIP_TYPES["related_to"] = _rel(
    "related_to", "Related To", "track", "track",
    Cardinality.MANY,
    description="Analyst-linked relationship between tracks.",
)
_RELATIONSHIP_TYPES["subject_of"] = _rel(
    "subject_of", "Subject Of", "track", "investigation",
    Cardinality.MANY,
    description="This track is part of an investigation.",
)

# Communication
_RELATIONSHIP_TYPES["sent_by"] = _rel(
    "sent_by", "Sent By", "chat_message", "mesh_node",
    Cardinality.ONE,
    description="The mesh node that originated this message.",
)

# ---------------------------------------------------------------------------
# The canonical schema
# ---------------------------------------------------------------------------

TRITIUM_ONTOLOGY = OntologySchema(
    version="1.0.0",
    entity_types=_ENTITY_TYPES,
    relationship_types=_RELATIONSHIP_TYPES,
    interfaces=_INTERFACES,
)

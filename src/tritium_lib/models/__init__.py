# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared data models for the Tritium ecosystem.

These models are the contract between tritium-sc and tritium-edge.
Any device that speaks the Tritium protocol uses these types.
"""

from .device import Device, DeviceGroup, DeviceHeartbeat, DeviceCapabilities
from .command import Command, CommandType, CommandStatus
from .firmware import FirmwareMeta, OTAJob, OTAStatus
from .sensor import SensorReading
from .ble import (
    BleDevice,
    BleSighting,
    BlePresence,
    BlePresenceMap,
    triangulate_position,
    set_node_positions,
)
from .fleet import (
    FleetNode,
    FleetStatus,
    NodeEvent,
    NodeStatus,
    fleet_health_score,
)
from .gis import (
    TileCoord,
    MapLayer,
    MapLayerType,
    MapRegion,
    TilePackage,
    lat_lon_to_tile,
    tile_to_lat_lon,
    tiles_in_bounds,
)
from .seed import (
    SeedFile,
    SeedManifest,
    SeedPackage,
    SeedStatus,
)
from .cot import (
    CotEvent,
    CotPoint,
    CotDetail,
    CotContact,
    cot_to_xml,
    xml_to_cot,
    COT_FRIENDLY_GROUND_UNIT,
    COT_FRIENDLY_UAV,
    COT_FRIENDLY_GROUND_SENSOR,
    COT_HOSTILE_GROUND_UNIT,
)

__all__ = [
    "Device",
    "DeviceGroup",
    "DeviceHeartbeat",
    "DeviceCapabilities",
    "Command",
    "CommandType",
    "CommandStatus",
    "FirmwareMeta",
    "OTAJob",
    "OTAStatus",
    "SensorReading",
    "BleDevice",
    "BleSighting",
    "BlePresence",
    "BlePresenceMap",
    "triangulate_position",
    "set_node_positions",
    "FleetNode",
    "FleetStatus",
    "NodeEvent",
    "NodeStatus",
    "fleet_health_score",
    # GIS
    "TileCoord",
    "MapLayer",
    "MapLayerType",
    "MapRegion",
    "TilePackage",
    "lat_lon_to_tile",
    "tile_to_lat_lon",
    "tiles_in_bounds",
    # Seed / replication
    "SeedFile",
    "SeedManifest",
    "SeedPackage",
    "SeedStatus",
    # CoT models
    "CotEvent",
    "CotPoint",
    "CotDetail",
    "CotContact",
    "cot_to_xml",
    "xml_to_cot",
    "COT_FRIENDLY_GROUND_UNIT",
    "COT_FRIENDLY_UAV",
    "COT_FRIENDLY_GROUND_SENSOR",
    "COT_HOSTILE_GROUND_UNIT",
]

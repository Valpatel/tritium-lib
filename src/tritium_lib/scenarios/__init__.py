# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.scenarios — predefined scenario generator for training, testing, and demos.

Generate complete surveillance/security scenarios with entities, events, and
timelines.  Each scenario defines a geographic area, normal activity patterns,
threat injection points, and expected alerts/detections.

Built-in scenario templates:
    * airport_surveillance — terminal, parking, runways
    * border_crossing      — checkpoint, approach roads, staging areas
    * urban_patrol          — city blocks, intersections, parks
    * maritime_port         — docks, approaches, anchorages
    * campus_security       — buildings, pathways, parking lots

Usage::

    from tritium_lib.scenarios import ScenarioGenerator

    gen = ScenarioGenerator()
    scenario = gen.create("airport_surveillance")
    # scenario.entities, scenario.events, scenario.timeline, ...

    player = ScenarioPlayer(scenario)
    for event in player.step():
        print(event)
"""

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional


# ── Constants ────────────────────────────────────────────────────────────

_DEG_PER_METER_LAT = 1.0 / 111320.0
_DEG_PER_METER_LNG_37 = 1.0 / (111320.0 * math.cos(math.radians(37.0)))


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in meters between two lat/lon points."""
    dlat = (lat2 - lat1) * 111320.0
    dlon = (lon2 - lon1) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.sqrt(dlat * dlat + dlon * dlon)


def _random_mac(rng: random.Random) -> str:
    """Generate a locally-administered random MAC address."""
    octets = [rng.randint(0, 255) for _ in range(6)]
    octets[0] = (octets[0] | 0x02) & 0xFE
    return ":".join(f"{b:02X}" for b in octets)


# ── Enums ────────────────────────────────────────────────────────────────

class EntityAlliance(str, Enum):
    """Alliance classification for scenario entities."""
    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Types of entities that can participate in a scenario."""
    PERSON = "person"
    VEHICLE = "vehicle"
    DRONE = "drone"
    DEVICE = "device"
    SENSOR_NODE = "sensor_node"
    CAMERA = "camera"
    VESSEL = "vessel"
    AIRCRAFT = "aircraft"


class ZoneType(str, Enum):
    """Types of geographic zones within a scenario."""
    BUILDING = "building"
    ROAD = "road"
    PARKING = "parking"
    OPEN_AREA = "open_area"
    WATERWAY = "waterway"
    RESTRICTED = "restricted"
    CHECKPOINT = "checkpoint"
    STAGING = "staging"
    RUNWAY = "runway"
    DOCK = "dock"
    PATHWAY = "pathway"
    INTERSECTION = "intersection"
    PARK = "park"
    ANCHORAGE = "anchorage"
    TERMINAL = "terminal"


class EventKind(str, Enum):
    """Kinds of events in a scenario timeline."""
    SPAWN = "spawn"
    MOVE = "move"
    DETECT = "detect"
    ALERT = "alert"
    GEOFENCE_ENTER = "geofence_enter"
    GEOFENCE_EXIT = "geofence_exit"
    THREAT_INJECT = "threat_inject"
    CLASSIFY = "classify"
    DISPATCH = "dispatch"
    LOITER = "loiter"
    DEPART = "depart"


class AlertLevel(str, Enum):
    """Severity levels for expected alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ── Data Models ──────────────────────────────────────────────────────────

@dataclass
class GeoZone:
    """A named geographic zone within a scenario area.

    Zones define buildings, roads, restricted areas, waterways, etc.
    Each zone has a center position, approximate size, and type.
    """
    zone_id: str
    name: str
    zone_type: ZoneType
    center_lat: float
    center_lng: float
    radius_m: float = 50.0
    properties: dict[str, Any] = field(default_factory=dict)

    def contains(self, lat: float, lng: float) -> bool:
        """Check if a lat/lng point is within this zone's radius."""
        return _haversine_m(self.center_lat, self.center_lng, lat, lng) <= self.radius_m


@dataclass
class ScenarioEntity:
    """An entity participating in a scenario — person, vehicle, device, etc.

    Entities have an initial position, movement pattern, sensor signature,
    and optional alliance/threat classification.
    """
    entity_id: str
    name: str
    entity_type: EntityType
    alliance: EntityAlliance = EntityAlliance.UNKNOWN
    start_lat: float = 0.0
    start_lng: float = 0.0
    speed_mps: float = 0.0
    heading: float = 0.0
    # BLE/WiFi simulation properties
    mac_address: str = ""
    device_class: str = ""
    # Sensor visibility
    ble_visible: bool = False
    wifi_visible: bool = False
    camera_visible: bool = True
    acoustic_visible: bool = False
    # Movement waypoints: list of (lat, lng, dwell_s)
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)
    # Extra properties
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioEvent:
    """An event at a specific time offset within a scenario timeline."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    time_offset_s: float = 0.0
    kind: EventKind = EventKind.SPAWN
    entity_id: str = ""
    lat: float = 0.0
    lng: float = 0.0
    description: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedAlert:
    """An alert expected to be triggered during scenario playback."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    time_offset_s: float = 0.0
    level: AlertLevel = AlertLevel.WARNING
    entity_id: str = ""
    zone_id: str = ""
    description: str = ""
    sensor_type: str = ""


@dataclass
class ExpectedDetection:
    """A detection expected to occur during scenario playback."""
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    time_offset_s: float = 0.0
    entity_id: str = ""
    sensor_type: str = ""  # "ble", "camera", "wifi", "acoustic", "rf_motion"
    confidence_min: float = 0.5
    lat: float = 0.0
    lng: float = 0.0
    description: str = ""


@dataclass
class Scenario:
    """A named scenario with entities, events, and a timeline.

    A complete scenario definition that can be replayed through the
    Tritium pipeline for training, testing, or demonstration.
    """
    scenario_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    template: str = ""
    # Geographic area
    center_lat: float = 0.0
    center_lng: float = 0.0
    area_size_m: float = 500.0
    zones: list[GeoZone] = field(default_factory=list)
    # Entities
    entities: list[ScenarioEntity] = field(default_factory=list)
    # Timeline
    events: list[ScenarioEvent] = field(default_factory=list)
    duration_s: float = 600.0
    # Expected outcomes
    expected_alerts: list[ExpectedAlert] = field(default_factory=list)
    expected_detections: list[ExpectedDetection] = field(default_factory=list)
    # Metadata
    tags: list[str] = field(default_factory=list)
    seed: int = 42

    def computed_duration(self) -> float:
        """Return explicit duration or max event time offset."""
        if self.duration_s > 0:
            return self.duration_s
        if not self.events:
            return 0.0
        return max(e.time_offset_s for e in self.events)

    def sorted_events(self) -> list[ScenarioEvent]:
        """Return events sorted by time offset."""
        return sorted(self.events, key=lambda e: e.time_offset_s)

    def entity_by_id(self, entity_id: str) -> Optional[ScenarioEntity]:
        """Find an entity by ID."""
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        return None

    def zone_by_id(self, zone_id: str) -> Optional[GeoZone]:
        """Find a zone by ID."""
        for z in self.zones:
            if z.zone_id == zone_id:
                return z
        return None

    def events_for_entity(self, entity_id: str) -> list[ScenarioEvent]:
        """Get all events involving a specific entity."""
        return [e for e in self.events if e.entity_id == entity_id]

    def friendly_entities(self) -> list[ScenarioEntity]:
        """Return all friendly entities."""
        return [e for e in self.entities if e.alliance == EntityAlliance.FRIENDLY]

    def hostile_entities(self) -> list[ScenarioEntity]:
        """Return all hostile entities."""
        return [e for e in self.entities if e.alliance == EntityAlliance.HOSTILE]

    def neutral_entities(self) -> list[ScenarioEntity]:
        """Return all neutral entities."""
        return [e for e in self.entities if e.alliance == EntityAlliance.NEUTRAL]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        from dataclasses import asdict
        return asdict(self)


# ── Scenario Generator ───────────────────────────────────────────────────

# Template specifications for built-in scenarios.  Each template defines
# the zone layout, entity patterns, and threat injection points.

_SCENARIO_TEMPLATES: dict[str, dict[str, Any]] = {
    "airport_surveillance": {
        "description": "Airport terminal with parking, taxiways, and runways",
        "center": (33.9425, -118.4081),  # LAX area
        "area_size_m": 1200.0,
        "duration_s": 900.0,
        "tags": ["airport", "surveillance", "indoor-outdoor"],
        "zones": [
            ("terminal-main", "Main Terminal", ZoneType.TERMINAL, 0, 0, 200),
            ("terminal-intl", "International Terminal", ZoneType.TERMINAL, 300, 50, 180),
            ("parking-a", "Parking Garage A", ZoneType.PARKING, -200, -150, 100),
            ("parking-b", "Parking Garage B", ZoneType.PARKING, 200, -200, 100),
            ("runway-25l", "Runway 25L", ZoneType.RUNWAY, 0, 400, 400),
            ("runway-25r", "Runway 25R", ZoneType.RUNWAY, 0, 550, 400),
            ("taxiway-a", "Taxiway Alpha", ZoneType.ROAD, -100, 250, 300),
            ("checkpoint-main", "Main Security", ZoneType.CHECKPOINT, 0, -50, 30),
            ("restricted-ramp", "Aircraft Ramp", ZoneType.RESTRICTED, 100, 300, 150),
            ("staging-cargo", "Cargo Staging", ZoneType.STAGING, -300, 200, 100),
        ],
        "normal_patterns": {
            "passengers": {"count": 30, "type": "person", "alliance": "neutral",
                           "speed": (0.8, 1.5), "zones": ["terminal-main", "terminal-intl", "parking-a", "parking-b"],
                           "ble": True, "wifi": True, "camera": True},
            "vehicles": {"count": 8, "type": "vehicle", "alliance": "neutral",
                         "speed": (3.0, 8.0), "zones": ["parking-a", "parking-b", "taxiway-a"],
                         "ble": False, "wifi": False, "camera": True, "acoustic": True},
            "staff": {"count": 10, "type": "person", "alliance": "friendly",
                      "speed": (1.0, 1.8), "zones": ["terminal-main", "terminal-intl", "checkpoint-main"],
                      "ble": True, "wifi": True, "camera": True},
            "security": {"count": 4, "type": "person", "alliance": "friendly",
                         "speed": (1.2, 2.0), "zones": ["checkpoint-main", "restricted-ramp", "terminal-main"],
                         "ble": True, "wifi": True, "camera": True},
        },
        "threats": [
            {"time_offset_s": 300, "type": "person", "name": "Unauthorized Ramp Access",
             "approach_zone": "restricted-ramp", "speed": 1.5,
             "alerts": [("critical", "Unauthorized person on aircraft ramp")]},
            {"time_offset_s": 600, "type": "vehicle", "name": "Suspicious Vehicle Loitering",
             "approach_zone": "terminal-main", "speed": 2.0,
             "alerts": [("warning", "Vehicle loitering near terminal entrance")]},
        ],
    },

    "border_crossing": {
        "description": "Border checkpoint with approach roads and staging areas",
        "center": (32.5450, -117.0300),  # Tijuana border area
        "area_size_m": 800.0,
        "duration_s": 1200.0,
        "tags": ["border", "checkpoint", "vehicle-heavy"],
        "zones": [
            ("checkpoint-primary", "Primary Inspection", ZoneType.CHECKPOINT, 0, 0, 40),
            ("checkpoint-secondary", "Secondary Inspection", ZoneType.CHECKPOINT, 80, 0, 30),
            ("road-north", "Northbound Approach", ZoneType.ROAD, 0, -200, 200),
            ("road-south", "Southbound Approach", ZoneType.ROAD, 0, 200, 200),
            ("staging-commercial", "Commercial Staging", ZoneType.STAGING, -200, -100, 120),
            ("staging-pedestrian", "Pedestrian Staging", ZoneType.STAGING, 150, -80, 60),
            ("admin-building", "Admin Building", ZoneType.BUILDING, -100, 50, 40),
            ("parking-officers", "Officer Parking", ZoneType.PARKING, -150, 80, 50),
            ("restricted-holding", "Holding Area", ZoneType.RESTRICTED, 120, 60, 40),
            ("open-queuing", "Vehicle Queue", ZoneType.OPEN_AREA, 0, -120, 80),
        ],
        "normal_patterns": {
            "vehicles_north": {"count": 15, "type": "vehicle", "alliance": "neutral",
                               "speed": (2.0, 5.0), "zones": ["road-north", "checkpoint-primary", "open-queuing"],
                               "ble": False, "wifi": False, "camera": True, "acoustic": True},
            "vehicles_south": {"count": 8, "type": "vehicle", "alliance": "neutral",
                               "speed": (2.0, 5.0), "zones": ["road-south", "checkpoint-primary"],
                               "ble": False, "wifi": False, "camera": True, "acoustic": True},
            "pedestrians": {"count": 12, "type": "person", "alliance": "neutral",
                            "speed": (0.8, 1.5), "zones": ["staging-pedestrian", "checkpoint-primary"],
                            "ble": True, "wifi": True, "camera": True},
            "officers": {"count": 6, "type": "person", "alliance": "friendly",
                         "speed": (1.0, 1.5), "zones": ["checkpoint-primary", "checkpoint-secondary", "admin-building"],
                         "ble": True, "wifi": True, "camera": True},
            "commercial": {"count": 5, "type": "vehicle", "alliance": "neutral",
                           "speed": (1.0, 3.0), "zones": ["staging-commercial", "checkpoint-primary"],
                           "ble": False, "wifi": False, "camera": True, "acoustic": True},
        },
        "threats": [
            {"time_offset_s": 400, "type": "vehicle", "name": "Evading Vehicle",
             "approach_zone": "checkpoint-primary", "speed": 12.0,
             "alerts": [("critical", "Vehicle bypassing checkpoint at high speed")]},
            {"time_offset_s": 800, "type": "person", "name": "Foot Runner",
             "approach_zone": "restricted-holding", "speed": 4.0,
             "alerts": [("warning", "Person running toward restricted holding area")]},
            {"time_offset_s": 1000, "type": "person", "name": "Perimeter Breach",
             "approach_zone": "admin-building", "speed": 2.0,
             "alerts": [("critical", "Unauthorized individual near admin building")]},
        ],
    },

    "urban_patrol": {
        "description": "City blocks with intersections, parks, and commercial areas",
        "center": (40.7580, -73.9855),  # Midtown Manhattan area
        "area_size_m": 600.0,
        "duration_s": 900.0,
        "tags": ["urban", "patrol", "dense"],
        "zones": [
            ("block-north", "North Block", ZoneType.BUILDING, 0, 150, 80),
            ("block-south", "South Block", ZoneType.BUILDING, 0, -150, 80),
            ("block-east", "East Block", ZoneType.BUILDING, 150, 0, 80),
            ("block-west", "West Block", ZoneType.BUILDING, -150, 0, 80),
            ("intersection-main", "Main Intersection", ZoneType.INTERSECTION, 0, 0, 30),
            ("park-central", "Central Park Area", ZoneType.PARK, 100, 100, 60),
            ("road-main-ns", "Main St N-S", ZoneType.ROAD, 0, 0, 200),
            ("road-main-ew", "Main St E-W", ZoneType.ROAD, 0, 0, 200),
            ("parking-garage", "Public Parking", ZoneType.PARKING, -100, -100, 50),
            ("alley-east", "East Alley", ZoneType.PATHWAY, 200, 50, 40),
        ],
        "normal_patterns": {
            "pedestrians": {"count": 25, "type": "person", "alliance": "neutral",
                            "speed": (0.8, 1.8), "zones": ["block-north", "block-south", "block-east", "block-west",
                                                             "intersection-main", "park-central"],
                            "ble": True, "wifi": True, "camera": True},
            "vehicles": {"count": 10, "type": "vehicle", "alliance": "neutral",
                         "speed": (5.0, 12.0), "zones": ["road-main-ns", "road-main-ew", "intersection-main"],
                         "ble": False, "wifi": False, "camera": True, "acoustic": True},
            "patrol_units": {"count": 3, "type": "person", "alliance": "friendly",
                             "speed": (1.2, 2.5), "zones": ["intersection-main", "park-central", "alley-east",
                                                              "parking-garage"],
                             "ble": True, "wifi": True, "camera": True},
            "cyclists": {"count": 4, "type": "person", "alliance": "neutral",
                         "speed": (3.0, 6.0), "zones": ["road-main-ns", "road-main-ew", "park-central"],
                         "ble": True, "wifi": False, "camera": True},
        },
        "threats": [
            {"time_offset_s": 250, "type": "person", "name": "Suspicious Loiterer",
             "approach_zone": "alley-east", "speed": 0.5,
             "alerts": [("warning", "Person loitering in east alley for extended period")]},
            {"time_offset_s": 500, "type": "vehicle", "name": "Wrong-Way Driver",
             "approach_zone": "intersection-main", "speed": 15.0,
             "alerts": [("critical", "Vehicle traveling wrong way through intersection")]},
        ],
    },

    "maritime_port": {
        "description": "Port facility with docks, approach channels, and anchorages",
        "center": (37.8044, -122.2712),  # Port of Oakland area
        "area_size_m": 1500.0,
        "duration_s": 1800.0,
        "tags": ["maritime", "port", "waterfront"],
        "zones": [
            ("dock-1", "Dock 1 - Container", ZoneType.DOCK, -200, 100, 120),
            ("dock-2", "Dock 2 - Bulk", ZoneType.DOCK, 0, 100, 120),
            ("dock-3", "Dock 3 - Passenger", ZoneType.DOCK, 200, 100, 100),
            ("channel-main", "Main Channel", ZoneType.WATERWAY, 0, 400, 300),
            ("anchorage-outer", "Outer Anchorage", ZoneType.ANCHORAGE, 0, 600, 200),
            ("anchorage-inner", "Inner Anchorage", ZoneType.ANCHORAGE, 100, 350, 100),
            ("terminal-ops", "Operations Terminal", ZoneType.TERMINAL, -100, -50, 80),
            ("staging-container", "Container Yard", ZoneType.STAGING, -300, 0, 150),
            ("road-port-access", "Port Access Road", ZoneType.ROAD, 0, -150, 200),
            ("restricted-fuel", "Fuel Storage", ZoneType.RESTRICTED, 300, -50, 60),
            ("parking-port", "Port Parking", ZoneType.PARKING, -200, -100, 60),
            ("checkpoint-gate", "Main Gate", ZoneType.CHECKPOINT, 0, -200, 30),
        ],
        "normal_patterns": {
            "vessels": {"count": 5, "type": "vessel", "alliance": "neutral",
                        "speed": (1.0, 4.0), "zones": ["channel-main", "anchorage-outer", "anchorage-inner",
                                                         "dock-1", "dock-2"],
                        "ble": False, "wifi": False, "camera": True, "acoustic": True},
            "trucks": {"count": 8, "type": "vehicle", "alliance": "neutral",
                       "speed": (3.0, 8.0), "zones": ["road-port-access", "staging-container", "checkpoint-gate"],
                       "ble": False, "wifi": False, "camera": True, "acoustic": True},
            "workers": {"count": 15, "type": "person", "alliance": "friendly",
                        "speed": (0.8, 1.5), "zones": ["terminal-ops", "dock-1", "dock-2", "dock-3",
                                                         "staging-container"],
                        "ble": True, "wifi": True, "camera": True},
            "security_patrol": {"count": 3, "type": "person", "alliance": "friendly",
                                "speed": (1.0, 2.0), "zones": ["checkpoint-gate", "restricted-fuel",
                                                                 "dock-1", "dock-2", "dock-3"],
                                "ble": True, "wifi": True, "camera": True},
            "small_craft": {"count": 4, "type": "vessel", "alliance": "neutral",
                            "speed": (2.0, 8.0), "zones": ["channel-main", "anchorage-outer"],
                            "ble": False, "wifi": False, "camera": True, "acoustic": True},
        },
        "threats": [
            {"time_offset_s": 600, "type": "vessel", "name": "Unidentified Vessel Approach",
             "approach_zone": "dock-1", "speed": 5.0,
             "alerts": [("warning", "Unidentified vessel approaching Dock 1 without clearance")]},
            {"time_offset_s": 1000, "type": "person", "name": "Restricted Area Intrusion",
             "approach_zone": "restricted-fuel", "speed": 1.5,
             "alerts": [("critical", "Unauthorized person entering fuel storage area")]},
            {"time_offset_s": 1400, "type": "drone", "name": "Unauthorized Drone",
             "approach_zone": "dock-3", "speed": 10.0,
             "alerts": [("critical", "Unauthorized drone over passenger dock")]},
        ],
    },

    "campus_security": {
        "description": "University/corporate campus with buildings, pathways, and lots",
        "center": (37.4275, -122.1697),  # Stanford area
        "area_size_m": 800.0,
        "duration_s": 1200.0,
        "tags": ["campus", "security", "mixed-use"],
        "zones": [
            ("bldg-science", "Science Building", ZoneType.BUILDING, -100, 100, 60),
            ("bldg-admin", "Admin Building", ZoneType.BUILDING, 100, 100, 50),
            ("bldg-library", "Library", ZoneType.BUILDING, 0, 200, 70),
            ("bldg-engineering", "Engineering Lab", ZoneType.BUILDING, -200, 0, 55),
            ("bldg-student-center", "Student Center", ZoneType.BUILDING, 0, 0, 80),
            ("pathway-main", "Main Pathway", ZoneType.PATHWAY, 0, 100, 200),
            ("pathway-south", "South Pathway", ZoneType.PATHWAY, 0, -100, 150),
            ("parking-north", "North Parking Lot", ZoneType.PARKING, 0, 300, 80),
            ("parking-south", "South Parking Lot", ZoneType.PARKING, 0, -250, 80),
            ("park-quad", "Main Quad", ZoneType.PARK, 50, 50, 60),
            ("restricted-server", "Server Room", ZoneType.RESTRICTED, -200, 50, 20),
            ("intersection-central", "Central Junction", ZoneType.INTERSECTION, 0, 50, 25),
        ],
        "normal_patterns": {
            "students": {"count": 20, "type": "person", "alliance": "neutral",
                         "speed": (0.8, 2.0),
                         "zones": ["bldg-science", "bldg-admin", "bldg-library", "bldg-engineering",
                                   "bldg-student-center", "pathway-main", "park-quad"],
                         "ble": True, "wifi": True, "camera": True},
            "staff": {"count": 8, "type": "person", "alliance": "friendly",
                      "speed": (0.8, 1.5),
                      "zones": ["bldg-admin", "bldg-science", "bldg-engineering", "bldg-library"],
                      "ble": True, "wifi": True, "camera": True},
            "vehicles": {"count": 6, "type": "vehicle", "alliance": "neutral",
                         "speed": (3.0, 8.0), "zones": ["parking-north", "parking-south"],
                         "ble": False, "wifi": False, "camera": True, "acoustic": True},
            "campus_security": {"count": 2, "type": "person", "alliance": "friendly",
                                "speed": (1.0, 2.5),
                                "zones": ["intersection-central", "parking-north", "parking-south",
                                          "restricted-server", "bldg-student-center"],
                                "ble": True, "wifi": True, "camera": True},
        },
        "threats": [
            {"time_offset_s": 400, "type": "person", "name": "Server Room Intrusion",
             "approach_zone": "restricted-server", "speed": 1.2,
             "alerts": [("critical", "Unauthorized access attempt at server room")]},
            {"time_offset_s": 800, "type": "person", "name": "After-Hours Loiterer",
             "approach_zone": "parking-south", "speed": 0.5,
             "alerts": [("warning", "Person loitering in south parking lot after hours")]},
        ],
    },
}

# Public constant: names of all built-in scenario templates
TEMPLATE_NAMES: list[str] = sorted(_SCENARIO_TEMPLATES.keys())


class ScenarioGenerator:
    """Creates fully populated :class:`Scenario` instances from templates.

    Each generated scenario includes:
    - Geographic zones (buildings, roads, restricted areas, etc.)
    - Normal-activity entities with waypoints and sensor signatures
    - Threat actors injected at specific timeline offsets
    - Expected alerts and detections
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    @property
    def available_templates(self) -> list[str]:
        """Return names of all available scenario templates."""
        return list(TEMPLATE_NAMES)

    def create(
        self,
        template_name: str,
        *,
        seed: int | None = None,
        center: tuple[float, float] | None = None,
        duration_s: float | None = None,
        threat_count: int | None = None,
    ) -> Scenario:
        """Generate a scenario from a named template.

        Parameters
        ----------
        template_name : str
            One of the built-in template names.
        seed : int | None
            Override the random seed (default: generator seed).
        center : tuple[float, float] | None
            Override the geographic center (lat, lng).
        duration_s : float | None
            Override the scenario duration.
        threat_count : int | None
            Override how many threats to inject (None = use template default).

        Returns
        -------
        Scenario
            A fully populated scenario ready for playback.

        Raises
        ------
        ValueError
            If template_name is not recognized.
        """
        if template_name not in _SCENARIO_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {', '.join(TEMPLATE_NAMES)}"
            )

        tmpl = _SCENARIO_TEMPLATES[template_name]
        effective_seed = seed if seed is not None else self._seed
        rng = random.Random(effective_seed)

        effective_center = center if center is not None else tmpl["center"]
        effective_duration = duration_s if duration_s is not None else tmpl["duration_s"]
        area_size_m = tmpl["area_size_m"]

        clat, clng = effective_center

        # Build zones
        zones = self._build_zones(tmpl["zones"], clat, clng)

        # Build normal-activity entities
        entities: list[ScenarioEntity] = []
        events: list[ScenarioEvent] = []
        expected_detections: list[ExpectedDetection] = []
        entity_counter = 0

        for pattern_name, pattern in tmpl["normal_patterns"].items():
            count = pattern["count"]
            for i in range(count):
                entity_counter += 1
                eid = f"{template_name}-{pattern_name}-{entity_counter:04d}"
                etype = self._map_entity_type(pattern["type"])
                alliance = EntityAlliance(pattern["alliance"])

                # Pick a random start zone
                zone_ids = pattern["zones"]
                start_zone_id = rng.choice(zone_ids)
                start_zone = next((z for z in zones if z.zone_id == start_zone_id), zones[0])

                # Random position within the zone
                angle = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(0, start_zone.radius_m * 0.8)
                start_lat = start_zone.center_lat + math.cos(angle) * r * _DEG_PER_METER_LAT
                start_lng = start_zone.center_lng + math.sin(angle) * r * _DEG_PER_METER_LNG_37

                speed_min, speed_max = pattern["speed"]
                speed = rng.uniform(speed_min, speed_max)

                mac = _random_mac(rng) if pattern.get("ble", False) else ""

                # Build waypoints: entity walks between assigned zones
                waypoints = self._build_waypoints(
                    zones, zone_ids, rng,
                    effective_duration, speed,
                )

                entity = ScenarioEntity(
                    entity_id=eid,
                    name=f"{pattern_name.replace('_', ' ').title()} {i + 1}",
                    entity_type=etype,
                    alliance=alliance,
                    start_lat=start_lat,
                    start_lng=start_lng,
                    speed_mps=speed,
                    heading=rng.uniform(0, 2 * math.pi),
                    mac_address=mac,
                    ble_visible=pattern.get("ble", False),
                    wifi_visible=pattern.get("wifi", False),
                    camera_visible=pattern.get("camera", True),
                    acoustic_visible=pattern.get("acoustic", False),
                    waypoints=waypoints,
                )
                entities.append(entity)

                # Spawn event
                events.append(ScenarioEvent(
                    time_offset_s=0.0,
                    kind=EventKind.SPAWN,
                    entity_id=eid,
                    lat=start_lat,
                    lng=start_lng,
                    description=f"{entity.name} spawns at {start_zone.name}",
                ))

                # Expected detection for sensor-visible entities
                for sensor, visible in [("ble", entity.ble_visible),
                                        ("camera", entity.camera_visible),
                                        ("wifi", entity.wifi_visible),
                                        ("acoustic", entity.acoustic_visible)]:
                    if visible:
                        det_time = rng.uniform(5.0, min(60.0, effective_duration * 0.1))
                        expected_detections.append(ExpectedDetection(
                            time_offset_s=det_time,
                            entity_id=eid,
                            sensor_type=sensor,
                            confidence_min=0.5,
                            lat=start_lat,
                            lng=start_lng,
                            description=f"{sensor} detection of {entity.name}",
                        ))

        # Build threat entities
        threats = tmpl["threats"]
        if threat_count is not None:
            threats = threats[:threat_count]

        expected_alerts: list[ExpectedAlert] = []

        for t_idx, threat in enumerate(threats):
            entity_counter += 1
            eid = f"{template_name}-threat-{entity_counter:04d}"
            etype = self._map_entity_type(threat["type"])

            # Find the target zone
            target_zone_id = threat["approach_zone"]
            target_zone = next((z for z in zones if z.zone_id == target_zone_id), zones[0])

            # Spawn threat at the edge of the scenario area, approaching the zone
            approach_angle = rng.uniform(0, 2 * math.pi)
            spawn_dist = area_size_m * 0.4
            spawn_lat = target_zone.center_lat + math.cos(approach_angle) * spawn_dist * _DEG_PER_METER_LAT
            spawn_lng = target_zone.center_lng + math.sin(approach_angle) * spawn_dist * _DEG_PER_METER_LNG_37

            inject_time = threat["time_offset_s"]

            entity = ScenarioEntity(
                entity_id=eid,
                name=threat["name"],
                entity_type=etype,
                alliance=EntityAlliance.HOSTILE,
                start_lat=spawn_lat,
                start_lng=spawn_lng,
                speed_mps=threat["speed"],
                heading=approach_angle + math.pi,  # toward target
                mac_address=_random_mac(rng),
                ble_visible=True,
                wifi_visible=etype == EntityType.PERSON,
                camera_visible=True,
                acoustic_visible=etype in (EntityType.VEHICLE, EntityType.DRONE, EntityType.VESSEL),
                waypoints=[(target_zone.center_lat, target_zone.center_lng, 30.0)],
            )
            entities.append(entity)

            # Threat injection event
            events.append(ScenarioEvent(
                time_offset_s=inject_time,
                kind=EventKind.THREAT_INJECT,
                entity_id=eid,
                lat=spawn_lat,
                lng=spawn_lng,
                description=f"Threat: {threat['name']}",
            ))

            # Movement toward target zone
            travel_dist = _haversine_m(spawn_lat, spawn_lng,
                                       target_zone.center_lat, target_zone.center_lng)
            travel_time = travel_dist / max(threat["speed"], 0.1)

            events.append(ScenarioEvent(
                time_offset_s=inject_time + travel_time * 0.5,
                kind=EventKind.MOVE,
                entity_id=eid,
                lat=(spawn_lat + target_zone.center_lat) / 2,
                lng=(spawn_lng + target_zone.center_lng) / 2,
                description=f"{threat['name']} approaching {target_zone.name}",
            ))

            # Geofence enter event
            events.append(ScenarioEvent(
                time_offset_s=inject_time + travel_time,
                kind=EventKind.GEOFENCE_ENTER,
                entity_id=eid,
                lat=target_zone.center_lat,
                lng=target_zone.center_lng,
                description=f"{threat['name']} enters {target_zone.name}",
            ))

            # Expected detections for threat
            det_time = inject_time + travel_time * 0.3
            for sensor in ["camera", "ble"]:
                expected_detections.append(ExpectedDetection(
                    time_offset_s=det_time,
                    entity_id=eid,
                    sensor_type=sensor,
                    confidence_min=0.6,
                    lat=spawn_lat,
                    lng=spawn_lng,
                    description=f"{sensor} detection of threat {threat['name']}",
                ))

            # Expected alerts from threat
            for alert_level_str, alert_desc in threat["alerts"]:
                alert_time = inject_time + travel_time * 0.7
                expected_alerts.append(ExpectedAlert(
                    time_offset_s=alert_time,
                    level=AlertLevel(alert_level_str),
                    entity_id=eid,
                    zone_id=target_zone_id,
                    description=alert_desc,
                    sensor_type="multi",
                ))

        # Sort events by time
        events.sort(key=lambda e: e.time_offset_s)

        return Scenario(
            name=template_name.replace("_", " ").title(),
            description=tmpl["description"],
            template=template_name,
            center_lat=clat,
            center_lng=clng,
            area_size_m=area_size_m,
            zones=zones,
            entities=entities,
            events=events,
            duration_s=effective_duration,
            expected_alerts=expected_alerts,
            expected_detections=expected_detections,
            tags=list(tmpl["tags"]),
            seed=effective_seed,
        )

    def create_all(self, *, seed: int | None = None) -> list[Scenario]:
        """Generate one scenario for every built-in template.

        Returns
        -------
        list[Scenario]
            A list of fully populated scenarios.
        """
        return [self.create(name, seed=seed) for name in TEMPLATE_NAMES]

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_zones(
        zone_specs: list[tuple],
        center_lat: float,
        center_lng: float,
    ) -> list[GeoZone]:
        """Convert template zone specs to GeoZone objects."""
        zones: list[GeoZone] = []
        for spec in zone_specs:
            zone_id, name, zone_type, offset_x_m, offset_y_m, radius_m = spec
            zlat = center_lat + offset_y_m * _DEG_PER_METER_LAT
            zlng = center_lng + offset_x_m * _DEG_PER_METER_LNG_37
            zones.append(GeoZone(
                zone_id=zone_id,
                name=name,
                zone_type=zone_type,
                center_lat=zlat,
                center_lng=zlng,
                radius_m=float(radius_m),
            ))
        return zones

    @staticmethod
    def _build_waypoints(
        zones: list[GeoZone],
        zone_ids: list[str],
        rng: random.Random,
        duration_s: float,
        speed_mps: float,
    ) -> list[tuple[float, float, float]]:
        """Generate movement waypoints cycling through assigned zones."""
        # Filter to zones the entity can visit
        available = [z for z in zones if z.zone_id in zone_ids]
        if not available:
            return []

        waypoints: list[tuple[float, float, float]] = []
        elapsed = 0.0
        max_waypoints = 20  # cap to prevent unbounded generation

        for _ in range(max_waypoints):
            if elapsed >= duration_s:
                break
            target_zone = rng.choice(available)
            # Random position within the zone
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0, target_zone.radius_m * 0.6)
            wlat = target_zone.center_lat + math.cos(angle) * r * _DEG_PER_METER_LAT
            wlng = target_zone.center_lng + math.sin(angle) * r * _DEG_PER_METER_LNG_37
            dwell = rng.uniform(10.0, 60.0)

            waypoints.append((wlat, wlng, dwell))

            # Estimate time to this waypoint
            if waypoints and len(waypoints) > 1:
                prev_lat, prev_lng, _ = waypoints[-2]
                dist = _haversine_m(prev_lat, prev_lng, wlat, wlng)
                travel_time = dist / max(speed_mps, 0.1)
            else:
                travel_time = 0.0

            elapsed += travel_time + dwell

        return waypoints

    @staticmethod
    def _map_entity_type(type_str: str) -> EntityType:
        """Map a template type string to EntityType enum."""
        mapping = {
            "person": EntityType.PERSON,
            "vehicle": EntityType.VEHICLE,
            "drone": EntityType.DRONE,
            "device": EntityType.DEVICE,
            "vessel": EntityType.VESSEL,
            "aircraft": EntityType.AIRCRAFT,
            "sensor_node": EntityType.SENSOR_NODE,
            "camera": EntityType.CAMERA,
        }
        return mapping.get(type_str, EntityType.PERSON)


# ── Scenario Player ─────────────────────────────────────────────────────

@dataclass
class PlayerState:
    """Runtime state of a scenario player."""
    current_time_s: float = 0.0
    step_index: int = 0
    entity_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    triggered_alerts: list[ExpectedAlert] = field(default_factory=list)
    triggered_detections: list[ExpectedDetection] = field(default_factory=list)
    active_entity_ids: set[str] = field(default_factory=set)
    completed: bool = False


class ScenarioPlayer:
    """Replays a :class:`Scenario` through a timeline, yielding events step by step.

    The player advances time in discrete steps, positioning entities along
    their waypoints and checking for expected detections and alerts.

    Usage::

        player = ScenarioPlayer(scenario, step_interval_s=10.0)
        for events in player.step():
            for e in events:
                print(f"t={player.state.current_time_s:.0f}s: {e.description}")
    """

    def __init__(
        self,
        scenario: Scenario,
        step_interval_s: float = 10.0,
    ) -> None:
        self._scenario = scenario
        self._step_interval = step_interval_s
        self._sorted_events = scenario.sorted_events()
        self._event_cursor = 0
        self.state = PlayerState()
        self._rng = random.Random(scenario.seed)

        # Initialize entity positions
        for entity in scenario.entities:
            if entity.alliance != EntityAlliance.HOSTILE:
                self.state.entity_positions[entity.entity_id] = (
                    entity.start_lat, entity.start_lng
                )
                self.state.active_entity_ids.add(entity.entity_id)

    @property
    def scenario(self) -> Scenario:
        """The scenario being played."""
        return self._scenario

    @property
    def is_complete(self) -> bool:
        """Whether the scenario playback has finished."""
        return self.state.completed

    def step(self) -> Iterator[list[ScenarioEvent]]:
        """Advance the scenario one step at a time, yielding events.

        Yields
        ------
        list[ScenarioEvent]
            Events that occurred during this time step.
        """
        duration = self._scenario.computed_duration()

        while self.state.current_time_s <= duration:
            step_events = self._advance_step()
            yield step_events
            self.state.step_index += 1
            self.state.current_time_s += self._step_interval

        self.state.completed = True

    def step_once(self) -> list[ScenarioEvent]:
        """Advance exactly one step and return the events.

        Returns
        -------
        list[ScenarioEvent]
            Events that occurred during this time step.
        """
        if self.state.completed:
            return []

        step_events = self._advance_step()
        self.state.step_index += 1
        self.state.current_time_s += self._step_interval

        if self.state.current_time_s > self._scenario.computed_duration():
            self.state.completed = True

        return step_events

    def reset(self) -> None:
        """Reset the player to the beginning of the scenario."""
        self._event_cursor = 0
        self.state = PlayerState()
        self._rng = random.Random(self._scenario.seed)

        for entity in self._scenario.entities:
            if entity.alliance != EntityAlliance.HOSTILE:
                self.state.entity_positions[entity.entity_id] = (
                    entity.start_lat, entity.start_lng
                )
                self.state.active_entity_ids.add(entity.entity_id)

    def _advance_step(self) -> list[ScenarioEvent]:
        """Process one time step: collect events, move entities, check triggers."""
        current_t = self.state.current_time_s
        next_t = current_t + self._step_interval
        step_events: list[ScenarioEvent] = []

        # Collect timeline events that fall in this window
        while (self._event_cursor < len(self._sorted_events) and
               self._sorted_events[self._event_cursor].time_offset_s <= next_t):
            evt = self._sorted_events[self._event_cursor]
            step_events.append(evt)

            # If this is a threat injection, activate the entity
            if evt.kind == EventKind.THREAT_INJECT and evt.entity_id:
                entity = self._scenario.entity_by_id(evt.entity_id)
                if entity:
                    self.state.entity_positions[entity.entity_id] = (
                        entity.start_lat, entity.start_lng
                    )
                    self.state.active_entity_ids.add(entity.entity_id)

            self._event_cursor += 1

        # Move active entities along their waypoints
        self._move_entities()

        # Check for expected detections and alerts in this time window
        for det in self._scenario.expected_detections:
            if current_t <= det.time_offset_s < next_t:
                if det not in self.state.triggered_detections:
                    self.state.triggered_detections.append(det)

        for alert in self._scenario.expected_alerts:
            if current_t <= alert.time_offset_s < next_t:
                if alert not in self.state.triggered_alerts:
                    self.state.triggered_alerts.append(alert)

        return step_events

    def _move_entities(self) -> None:
        """Move entities toward their next waypoint."""
        for entity in self._scenario.entities:
            if entity.entity_id not in self.state.active_entity_ids:
                continue
            if not entity.waypoints:
                continue

            current_pos = self.state.entity_positions.get(entity.entity_id)
            if current_pos is None:
                continue

            # Find the nearest unvisited waypoint
            clat, clng = current_pos
            best_wp = None
            best_dist = float("inf")
            for wlat, wlng, _dwell in entity.waypoints:
                d = _haversine_m(clat, clng, wlat, wlng)
                if d < best_dist and d > 1.0:  # not already there
                    best_dist = d
                    best_wp = (wlat, wlng)

            if best_wp is None:
                continue

            # Move toward waypoint
            target_lat, target_lng = best_wp
            dist_m = _haversine_m(clat, clng, target_lat, target_lng)
            step_dist = entity.speed_mps * self._step_interval

            if step_dist >= dist_m:
                # Arrive at waypoint
                self.state.entity_positions[entity.entity_id] = (target_lat, target_lng)
            else:
                # Partial movement
                ratio = step_dist / max(dist_m, 0.1)
                new_lat = clat + (target_lat - clat) * ratio
                new_lng = clng + (target_lng - clng) * ratio
                self.state.entity_positions[entity.entity_id] = (new_lat, new_lng)


# ── Public API ───────────────────────────────────────────────────────────

__all__ = [
    # Enums
    "EntityAlliance",
    "EntityType",
    "ZoneType",
    "EventKind",
    "AlertLevel",
    # Data models
    "GeoZone",
    "ScenarioEntity",
    "ScenarioEvent",
    "ExpectedAlert",
    "ExpectedDetection",
    "Scenario",
    # Generator
    "ScenarioGenerator",
    "TEMPLATE_NAMES",
    # Player
    "ScenarioPlayer",
    "PlayerState",
]

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Scenario-oriented synthetic data generators for testing and demos.

Stateless, functional generators that produce lists of realistic synthetic
data.  Unlike the _BaseGenerator subclasses in data_generators.py (which are
background-threaded EventBus publishers), these functions return plain dicts
so any consumer — SC, addons, demos, tests — can call them directly.

Generators
----------
- generate_ble_sightings   : BLE device sightings with movement patterns
- generate_wifi_environment : WiFi APs with SSIDs, channels, signal strength
- generate_patrol_scenario  : Patrol routes with waypoints and events
- generate_threat_scenario  : Hostiles approaching a geofenced area
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────

# Realistic BLE device templates grouped by type
_BLE_DEVICE_TEMPLATES: list[dict[str, Any]] = [
    # Phones
    {"name_prefix": "iPhone", "type": "phone", "tx_power": -59, "adv_interval_ms": 200},
    {"name_prefix": "Galaxy", "type": "phone", "tx_power": -60, "adv_interval_ms": 250},
    {"name_prefix": "Pixel", "type": "phone", "tx_power": -58, "adv_interval_ms": 230},
    {"name_prefix": "OnePlus", "type": "phone", "tx_power": -62, "adv_interval_ms": 300},
    # Wearables
    {"name_prefix": "Apple-Watch", "type": "wearable", "tx_power": -65, "adv_interval_ms": 500},
    {"name_prefix": "Fitbit", "type": "wearable", "tx_power": -68, "adv_interval_ms": 700},
    {"name_prefix": "Garmin", "type": "wearable", "tx_power": -66, "adv_interval_ms": 1000},
    # Audio
    {"name_prefix": "AirPods", "type": "audio", "tx_power": -63, "adv_interval_ms": 400},
    {"name_prefix": "WH-1000", "type": "audio", "tx_power": -61, "adv_interval_ms": 450},
    {"name_prefix": "JBL", "type": "audio", "tx_power": -55, "adv_interval_ms": 300},
    # Trackers
    {"name_prefix": "AirTag", "type": "tracker", "tx_power": -67, "adv_interval_ms": 2000},
    {"name_prefix": "Tile", "type": "tracker", "tx_power": -64, "adv_interval_ms": 2000},
    # IoT
    {"name_prefix": "Nest", "type": "iot", "tx_power": -70, "adv_interval_ms": 5000},
    {"name_prefix": "Ring", "type": "iot", "tx_power": -68, "adv_interval_ms": 3000},
    {"name_prefix": "Echo", "type": "iot", "tx_power": -55, "adv_interval_ms": 1000},
    # Automotive
    {"name_prefix": "Tesla-Key", "type": "automotive", "tx_power": -58, "adv_interval_ms": 200},
    {"name_prefix": "BMW-Key", "type": "automotive", "tx_power": -60, "adv_interval_ms": 250},
    # Unknown (no name)
    {"name_prefix": "", "type": "unknown", "tx_power": -65, "adv_interval_ms": 500},
]

# Movement patterns: speed in meters/second, heading change per step (radians)
_MOVEMENT_PATTERNS: dict[str, dict[str, float]] = {
    "stationary": {"speed_mps": 0.0, "heading_jitter": 0.0, "rssi_range": (-55, -30)},
    "pedestrian": {"speed_mps": 1.4, "heading_jitter": 0.4, "rssi_range": (-80, -40)},
    "jogger": {"speed_mps": 2.8, "heading_jitter": 0.2, "rssi_range": (-85, -45)},
    "vehicle": {"speed_mps": 11.0, "heading_jitter": 0.1, "rssi_range": (-90, -50)},
    "cyclist": {"speed_mps": 5.5, "heading_jitter": 0.15, "rssi_range": (-85, -45)},
}

# Movement pattern weights by device type
_TYPE_MOVEMENT_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "phone": [("pedestrian", 0.5), ("stationary", 0.2), ("vehicle", 0.2), ("cyclist", 0.1)],
    "wearable": [("pedestrian", 0.4), ("jogger", 0.3), ("stationary", 0.2), ("cyclist", 0.1)],
    "audio": [("pedestrian", 0.4), ("stationary", 0.3), ("jogger", 0.2), ("cyclist", 0.1)],
    "tracker": [("pedestrian", 0.3), ("stationary", 0.4), ("vehicle", 0.2), ("cyclist", 0.1)],
    "iot": [("stationary", 1.0)],
    "automotive": [("vehicle", 0.8), ("stationary", 0.2)],
    "unknown": [("pedestrian", 0.4), ("stationary", 0.3), ("vehicle", 0.2), ("jogger", 0.1)],
}

# WiFi AP templates
_WIFI_SSID_PATTERNS: list[dict[str, Any]] = [
    # Residential
    {"pattern": "NETGEAR-{adj}", "auth": "WPA2", "net_type": "home", "channel_range": (1, 11)},
    {"pattern": "TP-Link_{hex4}", "auth": "WPA2", "net_type": "home", "channel_range": (1, 11)},
    {"pattern": "Linksys_{hex4}", "auth": "WPA2", "net_type": "home", "channel_range": (1, 11)},
    {"pattern": "{surname} WiFi", "auth": "WPA2", "net_type": "home", "channel_range": (1, 11)},
    {"pattern": "xfinitywifi", "auth": "open", "net_type": "hotspot", "channel_range": (1, 11)},
    # Commercial
    {"pattern": "Starbucks WiFi", "auth": "open", "net_type": "public", "channel_range": (1, 11)},
    {"pattern": "McDonald's Free WiFi", "auth": "open", "net_type": "public", "channel_range": (1, 11)},
    {"pattern": "HOTEL-{adj}-GUEST", "auth": "WPA2", "net_type": "guest", "channel_range": (1, 11)},
    {"pattern": "Airport_FreeWiFi", "auth": "open", "net_type": "public", "channel_range": (36, 165)},
    # Corporate
    {"pattern": "CORP-{dept}-5G", "auth": "WPA2-Enterprise", "net_type": "corporate", "channel_range": (36, 165)},
    {"pattern": "eduroam", "auth": "WPA2-Enterprise", "net_type": "corporate", "channel_range": (36, 165)},
    {"pattern": "OFFICE-{floor}", "auth": "WPA2-Enterprise", "net_type": "corporate", "channel_range": (36, 165)},
    # IoT / Mesh
    {"pattern": "SmartHome_{hex4}", "auth": "WPA2", "net_type": "iot", "channel_range": (1, 11)},
    {"pattern": "Ring-{hex4}", "auth": "WPA2", "net_type": "iot", "channel_range": (1, 11)},
    {"pattern": "MESHTASTIC-{hex4}", "auth": "open", "net_type": "mesh", "channel_range": (1, 11)},
    # Hidden
    {"pattern": "", "auth": "WPA2", "net_type": "unknown", "channel_range": (1, 11)},
]

_ADJECTIVES = ["Swift", "Peak", "Coral", "Ember", "Lunar", "Nova", "Frost", "Blaze", "Sage", "Dusk"]
_SURNAMES = ["Smith", "Chen", "Patel", "Garcia", "Kim", "Nguyen", "Johnson", "Williams", "Brown", "Jones"]
_DEPTS = ["ENG", "HR", "FIN", "MKT", "SEC", "OPS", "RND", "IT"]

# Patrol/threat alliance labels
ALLIANCE_FRIENDLY = "friendly"
ALLIANCE_HOSTILE = "hostile"
ALLIANCE_UNKNOWN = "unknown"
ALLIANCE_NEUTRAL = "neutral"

# Degrees per meter at mid-latitudes (~37 deg)
_DEG_PER_METER_LAT = 1.0 / 111320.0
_DEG_PER_METER_LNG_37 = 1.0 / (111320.0 * math.cos(math.radians(37.0)))


# ── Helpers ────────────────────────────────────────────────────────────

def _random_mac(rng: random.Random) -> str:
    """Generate a random MAC address (uppercase, colon-separated)."""
    octets = [rng.randint(0, 255) for _ in range(6)]
    # Set locally-administered bit to avoid collision with real OUIs
    octets[0] = (octets[0] | 0x02) & 0xFE
    return ":".join(f"{b:02X}" for b in octets)


def _random_bssid(rng: random.Random) -> str:
    """Generate a random BSSID for a WiFi AP."""
    return _random_mac(rng)


def _resolve_ssid_pattern(pattern: str, rng: random.Random) -> str:
    """Fill in template placeholders in an SSID pattern."""
    result = pattern
    result = result.replace("{adj}", rng.choice(_ADJECTIVES))
    result = result.replace("{surname}", rng.choice(_SURNAMES))
    result = result.replace("{hex4}", f"{rng.randint(0, 0xFFFF):04X}")
    result = result.replace("{dept}", rng.choice(_DEPTS))
    result = result.replace("{floor}", str(rng.randint(1, 12)))
    return result


def _pick_movement(device_type: str, rng: random.Random) -> str:
    """Choose a movement pattern for a device type using weighted random."""
    weights = _TYPE_MOVEMENT_WEIGHTS.get(device_type, _TYPE_MOVEMENT_WEIGHTS["unknown"])
    patterns = [w[0] for w in weights]
    probs = [w[1] for w in weights]
    return rng.choices(patterns, weights=probs, k=1)[0]


def _compute_rssi(distance_m: float, tx_power: float, path_loss_exp: float,
                  rng: random.Random) -> int:
    """Compute RSSI from distance using log-distance path-loss model with noise."""
    d = max(distance_m, 0.5)
    rssi = tx_power - 10.0 * path_loss_exp * math.log10(d)
    rssi += rng.gauss(0, 3.0)  # environmental noise
    return max(-100, min(-20, int(round(rssi))))


def _move_position(
    lat: float, lng: float, heading: float,
    speed_mps: float, dt_seconds: float, heading_jitter: float,
    rng: random.Random,
) -> tuple[float, float, float]:
    """Move a position forward by speed*dt along heading with jitter.

    Returns (new_lat, new_lng, new_heading).
    """
    heading += rng.gauss(0, heading_jitter)
    dist_m = speed_mps * dt_seconds
    dlat = math.cos(heading) * dist_m * _DEG_PER_METER_LAT
    dlng = math.sin(heading) * dist_m * _DEG_PER_METER_LNG_37
    return lat + dlat, lng + dlng, heading


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in meters between two lat/lon points."""
    dlat = (lat2 - lat1) * 111320.0
    dlon = (lon2 - lon1) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.sqrt(dlat * dlat + dlon * dlon)


# ── BLE Sighting Generator ────────────────────────────────────────────

@dataclass
class BLESightingRecord:
    """A single BLE sighting from one observer at one moment."""
    mac: str
    name: str
    device_type: str
    rssi: int
    tx_power: int
    lat: float
    lng: float
    observer_id: str
    observer_lat: float
    observer_lng: float
    timestamp: float
    movement_pattern: str
    heading: float
    speed_mps: float
    distance_m: float


def generate_ble_sightings(
    count: int = 20,
    area_center: tuple[float, float] = (37.7749, -122.4194),
    area_size_m: float = 500.0,
    observer_count: int = 3,
    time_steps: int = 10,
    step_interval_s: float = 5.0,
    seed: int = 42,
    base_time: float | None = None,
) -> list[BLESightingRecord]:
    """Generate realistic BLE sightings with movement patterns.

    Creates ``count`` synthetic BLE devices scattered in an area, each
    assigned a movement pattern based on its device type.  Multiple
    observers are placed in the area.  The simulation runs for
    ``time_steps`` steps, producing sightings whenever a device is within
    detection range (~100 m) of an observer.

    Parameters
    ----------
    count : int
        Number of unique BLE devices to simulate.
    area_center : tuple[float, float]
        (lat, lng) center of the simulation area.
    area_size_m : float
        Side length of the square simulation area in meters.
    observer_count : int
        Number of BLE observers (edge nodes) placed in the area.
    time_steps : int
        Number of discrete time steps to simulate.
    step_interval_s : float
        Seconds between each time step.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[BLESightingRecord]
        All sightings generated across all time steps and observers.
    """
    rng = random.Random(seed)
    results: list[BLESightingRecord] = []

    half_area_lat = (area_size_m / 2.0) * _DEG_PER_METER_LAT
    half_area_lng = (area_size_m / 2.0) * _DEG_PER_METER_LNG_37
    center_lat, center_lng = area_center

    # Place observers evenly within the area
    observers: list[dict[str, Any]] = []
    for i in range(observer_count):
        olat = center_lat + rng.uniform(-half_area_lat * 0.6, half_area_lat * 0.6)
        olng = center_lng + rng.uniform(-half_area_lng * 0.6, half_area_lng * 0.6)
        observers.append({
            "id": f"observer-{i+1:02d}",
            "lat": olat,
            "lng": olng,
        })

    # Create devices
    devices: list[dict[str, Any]] = []
    for i in range(count):
        template = rng.choice(_BLE_DEVICE_TEMPLATES)
        device_type = template["type"]
        name_prefix = template["name_prefix"]
        mac = _random_mac(rng)
        name = f"{name_prefix}-{rng.randint(100, 999)}" if name_prefix else ""
        movement = _pick_movement(device_type, rng)
        pattern = _MOVEMENT_PATTERNS[movement]

        # Initial position within the area
        dlat = center_lat + rng.uniform(-half_area_lat, half_area_lat)
        dlng = center_lng + rng.uniform(-half_area_lng, half_area_lng)
        heading = rng.uniform(0, 2.0 * math.pi)

        devices.append({
            "mac": mac,
            "name": name,
            "type": device_type,
            "tx_power": template["tx_power"],
            "movement": movement,
            "speed_mps": pattern["speed_mps"],
            "heading_jitter": pattern["heading_jitter"],
            "lat": dlat,
            "lng": dlng,
            "heading": heading,
        })

    # Simulate time steps
    if base_time is None:
        base_time = time.time()
    detection_range_m = 100.0

    for step in range(time_steps):
        ts = base_time + step * step_interval_s

        # Move devices
        for dev in devices:
            if dev["speed_mps"] > 0:
                new_lat, new_lng, new_heading = _move_position(
                    dev["lat"], dev["lng"], dev["heading"],
                    dev["speed_mps"], step_interval_s, dev["heading_jitter"],
                    rng,
                )
                dev["lat"] = new_lat
                dev["lng"] = new_lng
                dev["heading"] = new_heading

        # Check visibility from each observer
        for obs in observers:
            for dev in devices:
                dist = _haversine_m(obs["lat"], obs["lng"], dev["lat"], dev["lng"])
                if dist <= detection_range_m:
                    rssi = _compute_rssi(dist, dev["tx_power"], 2.5, rng)
                    results.append(BLESightingRecord(
                        mac=dev["mac"],
                        name=dev["name"],
                        device_type=dev["type"],
                        rssi=rssi,
                        tx_power=dev["tx_power"],
                        lat=dev["lat"],
                        lng=dev["lng"],
                        observer_id=obs["id"],
                        observer_lat=obs["lat"],
                        observer_lng=obs["lng"],
                        timestamp=ts,
                        movement_pattern=dev["movement"],
                        heading=dev["heading"],
                        speed_mps=dev["speed_mps"],
                        distance_m=round(dist, 2),
                    ))

    return results


# ── WiFi Environment Generator ────────────────────────────────────────

@dataclass
class WiFiAPRecord:
    """A single WiFi access point in a generated environment."""
    bssid: str
    ssid: str
    channel: int
    rssi: int
    auth_type: str
    network_type: str
    lat: float
    lng: float
    floor: int
    building_id: str
    hidden: bool


@dataclass
class WiFiEnvironment:
    """A complete WiFi environment with buildings and APs."""
    access_points: list[WiFiAPRecord]
    buildings: list[dict[str, Any]]
    area_center: tuple[float, float]
    area_size_m: float


def generate_wifi_environment(
    building_count: int = 10,
    area_center: tuple[float, float] = (37.7749, -122.4194),
    area_size_m: float = 500.0,
    seed: int = 123,
) -> WiFiEnvironment:
    """Generate a realistic WiFi environment with buildings and access points.

    Each building gets 1-6 WiFi APs depending on building type (residential,
    commercial, corporate).  APs have realistic SSIDs, channels, auth types,
    and signal strengths.  Some networks are hidden.

    Parameters
    ----------
    building_count : int
        Number of buildings in the environment.
    area_center : tuple[float, float]
        (lat, lng) center of the area.
    area_size_m : float
        Side length of the area in meters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    WiFiEnvironment
        Complete environment with buildings and APs.
    """
    rng = random.Random(seed)
    half_lat = (area_size_m / 2.0) * _DEG_PER_METER_LAT
    half_lng = (area_size_m / 2.0) * _DEG_PER_METER_LNG_37
    clat, clng = area_center

    building_types = ["residential", "commercial", "corporate", "industrial"]
    building_type_ap_range = {
        "residential": (1, 3),
        "commercial": (2, 5),
        "corporate": (3, 6),
        "industrial": (1, 2),
    }
    building_type_floors = {
        "residential": (1, 4),
        "commercial": (1, 3),
        "corporate": (2, 8),
        "industrial": (1, 2),
    }

    buildings: list[dict[str, Any]] = []
    all_aps: list[WiFiAPRecord] = []

    # Track used 2.4 GHz channels to create realistic co-channel interference
    # In real environments, channels 1, 6, 11 dominate 2.4 GHz
    preferred_24_channels = [1, 6, 11]

    for b_idx in range(building_count):
        b_type = rng.choice(building_types)
        b_lat = clat + rng.uniform(-half_lat, half_lat)
        b_lng = clng + rng.uniform(-half_lng, half_lng)
        floors_min, floors_max = building_type_floors[b_type]
        n_floors = rng.randint(floors_min, floors_max)
        building_id = f"bldg-{b_idx+1:03d}"

        buildings.append({
            "building_id": building_id,
            "type": b_type,
            "lat": b_lat,
            "lng": b_lng,
            "floors": n_floors,
        })

        # Assign APs for this building
        ap_min, ap_max = building_type_ap_range[b_type]
        n_aps = rng.randint(ap_min, ap_max)

        # Filter SSID patterns appropriate for building type
        if b_type == "residential":
            eligible_patterns = [p for p in _WIFI_SSID_PATTERNS
                                 if p["net_type"] in ("home", "iot", "unknown")]
        elif b_type == "commercial":
            eligible_patterns = [p for p in _WIFI_SSID_PATTERNS
                                 if p["net_type"] in ("public", "guest", "iot", "unknown")]
        elif b_type == "corporate":
            eligible_patterns = [p for p in _WIFI_SSID_PATTERNS
                                 if p["net_type"] in ("corporate", "guest", "iot", "unknown")]
        else:
            eligible_patterns = [p for p in _WIFI_SSID_PATTERNS
                                 if p["net_type"] in ("home", "iot", "mesh", "unknown")]

        if not eligible_patterns:
            eligible_patterns = list(_WIFI_SSID_PATTERNS)

        for ap_idx in range(n_aps):
            template = rng.choice(eligible_patterns)
            ssid = _resolve_ssid_pattern(template["pattern"], rng)
            hidden = ssid == ""
            ch_low, ch_high = template["channel_range"]

            # Prefer non-overlapping 2.4 GHz channels or pick from 5 GHz
            if ch_high <= 14:
                channel = rng.choice(preferred_24_channels)
            else:
                channel = rng.choice([36, 40, 44, 48, 149, 153, 157, 161, 165])

            # AP sits on a floor within the building
            ap_floor = rng.randint(0, n_floors - 1) if n_floors > 0 else 0

            # Signal strength: stronger if closer to center, with per-AP jitter
            dist_from_center = _haversine_m(clat, clng, b_lat, b_lng)
            base_rssi = -40 - int(dist_from_center / 10.0)
            rssi = max(-95, min(-25, base_rssi + rng.randint(-10, 10)))

            # Slight position offset within building footprint (~10m)
            ap_lat = b_lat + rng.gauss(0, 0.00005)
            ap_lng = b_lng + rng.gauss(0, 0.00005)

            all_aps.append(WiFiAPRecord(
                bssid=_random_bssid(rng),
                ssid=ssid,
                channel=channel,
                rssi=rssi,
                auth_type=template["auth"],
                network_type=template["net_type"],
                lat=ap_lat,
                lng=ap_lng,
                floor=ap_floor,
                building_id=building_id,
                hidden=hidden,
            ))

    return WiFiEnvironment(
        access_points=all_aps,
        buildings=buildings,
        area_center=area_center,
        area_size_m=area_size_m,
    )


# ── Patrol Scenario Generator ─────────────────────────────────────────

@dataclass
class PatrolWaypoint:
    """A waypoint along a patrol route."""
    waypoint_id: str
    lat: float
    lng: float
    name: str
    dwell_time_s: float  # how long to stay at this waypoint
    order: int


@dataclass
class PatrolEvent:
    """An event that occurs during a patrol."""
    event_type: str  # "arrival", "departure", "detection", "anomaly"
    waypoint_id: str
    timestamp: float
    details: dict[str, Any]


@dataclass
class PatrolUnit:
    """A patrol unit (person, vehicle, drone) following a route."""
    unit_id: str
    name: str
    asset_type: str  # "person", "vehicle", "drone"
    alliance: str
    speed_mps: float
    route: list[PatrolWaypoint]


@dataclass
class PatrolScenario:
    """Complete patrol scenario with units, routes, and events."""
    units: list[PatrolUnit]
    events: list[PatrolEvent]
    area_center: tuple[float, float]
    duration_s: float
    geofence_vertices: list[tuple[float, float]]


def generate_patrol_scenario(
    unit_count: int = 3,
    waypoints_per_route: int = 6,
    area_center: tuple[float, float] = (37.7749, -122.4194),
    area_size_m: float = 400.0,
    duration_s: float = 600.0,
    seed: int = 77,
    base_time: float | None = None,
) -> PatrolScenario:
    """Generate a patrol scenario with units following checkpoint routes.

    Creates multiple patrol units (person, vehicle, drone) each with a
    route of waypoints.  Events (arrival, departure, detection, anomaly)
    are generated at realistic intervals as units move through their
    routes.  Includes a perimeter geofence around the patrol area.

    Parameters
    ----------
    unit_count : int
        Number of patrol units.
    waypoints_per_route : int
        Number of waypoints per unit route.
    area_center : tuple[float, float]
        (lat, lng) center of patrol area.
    area_size_m : float
        Side length of the patrol area in meters.
    duration_s : float
        Total scenario duration in seconds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PatrolScenario
        Complete scenario with units, routes, waypoints, and events.
    """
    rng = random.Random(seed)
    half_lat = (area_size_m / 2.0) * _DEG_PER_METER_LAT
    half_lng = (area_size_m / 2.0) * _DEG_PER_METER_LNG_37
    clat, clng = area_center

    # Geofence: square perimeter with slight irregularity
    geofence: list[tuple[float, float]] = [
        (clat + half_lat * rng.uniform(0.9, 1.1), clng - half_lng * rng.uniform(0.9, 1.1)),
        (clat + half_lat * rng.uniform(0.9, 1.1), clng + half_lng * rng.uniform(0.9, 1.1)),
        (clat - half_lat * rng.uniform(0.9, 1.1), clng + half_lng * rng.uniform(0.9, 1.1)),
        (clat - half_lat * rng.uniform(0.9, 1.1), clng - half_lng * rng.uniform(0.9, 1.1)),
    ]

    asset_types = [
        ("person", 1.4, "foot-patrol"),
        ("vehicle", 8.0, "mobile-patrol"),
        ("drone", 15.0, "aerial-recon"),
    ]

    checkpoint_names = [
        "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
        "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima",
        "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
    ]
    name_idx = 0

    units: list[PatrolUnit] = []
    all_events: list[PatrolEvent] = []
    if base_time is None:
        base_time = time.time()

    for u_idx in range(unit_count):
        asset_type, speed, label = asset_types[u_idx % len(asset_types)]
        unit_id = f"patrol-{u_idx+1:02d}"
        unit_name = f"{label}-{u_idx+1}"

        # Generate waypoints in a rough loop
        waypoints: list[PatrolWaypoint] = []
        angles = sorted([rng.uniform(0, 2.0 * math.pi) for _ in range(waypoints_per_route)])
        for wp_idx, angle in enumerate(angles):
            radius = rng.uniform(0.3, 0.9) * area_size_m / 2.0
            wlat = clat + math.cos(angle) * radius * _DEG_PER_METER_LAT
            wlng = clng + math.sin(angle) * radius * _DEG_PER_METER_LNG_37
            dwell = rng.uniform(10.0, 60.0) if asset_type != "drone" else rng.uniform(5.0, 20.0)

            wp_name = checkpoint_names[name_idx % len(checkpoint_names)]
            name_idx += 1

            waypoints.append(PatrolWaypoint(
                waypoint_id=f"{unit_id}-wp-{wp_idx+1}",
                lat=wlat,
                lng=wlng,
                name=wp_name,
                dwell_time_s=dwell,
                order=wp_idx,
            ))

        unit = PatrolUnit(
            unit_id=unit_id,
            name=unit_name,
            asset_type=asset_type,
            alliance=ALLIANCE_FRIENDLY,
            speed_mps=speed,
            route=waypoints,
        )
        units.append(unit)

        # Generate events: arrival/departure at each waypoint, occasional detections
        current_time = base_time
        current_lat, current_lng = clat, clng

        for wp in waypoints:
            # Travel time to this waypoint
            dist = _haversine_m(current_lat, current_lng, wp.lat, wp.lng)
            travel_time = dist / speed if speed > 0 else 0.0
            current_time += travel_time

            if current_time - base_time > duration_s:
                break

            # Arrival event
            all_events.append(PatrolEvent(
                event_type="arrival",
                waypoint_id=wp.waypoint_id,
                timestamp=current_time,
                details={
                    "unit_id": unit_id,
                    "waypoint_name": wp.name,
                    "lat": wp.lat,
                    "lng": wp.lng,
                },
            ))

            # Random detection at waypoint (30% chance)
            if rng.random() < 0.3:
                det_types = ["person", "vehicle", "animal"]
                det_alliances = [ALLIANCE_UNKNOWN, ALLIANCE_NEUTRAL, ALLIANCE_HOSTILE]
                det_type = rng.choice(det_types)
                det_alliance = rng.choices(
                    det_alliances, weights=[0.5, 0.3, 0.2], k=1
                )[0]
                all_events.append(PatrolEvent(
                    event_type="detection",
                    waypoint_id=wp.waypoint_id,
                    timestamp=current_time + rng.uniform(1.0, wp.dwell_time_s * 0.5),
                    details={
                        "unit_id": unit_id,
                        "detected_type": det_type,
                        "detected_alliance": det_alliance,
                        "confidence": round(rng.uniform(0.5, 0.95), 2),
                        "lat": wp.lat + rng.gauss(0, 0.0001),
                        "lng": wp.lng + rng.gauss(0, 0.0001),
                    },
                ))

            # Random anomaly at waypoint (10% chance)
            if rng.random() < 0.1:
                anomaly_types = ["unusual_activity", "signal_interference", "perimeter_breach"]
                all_events.append(PatrolEvent(
                    event_type="anomaly",
                    waypoint_id=wp.waypoint_id,
                    timestamp=current_time + rng.uniform(0.5, wp.dwell_time_s),
                    details={
                        "unit_id": unit_id,
                        "anomaly_type": rng.choice(anomaly_types),
                        "severity": rng.choice(["low", "medium", "high"]),
                        "lat": wp.lat,
                        "lng": wp.lng,
                    },
                ))

            # Departure
            current_time += wp.dwell_time_s
            all_events.append(PatrolEvent(
                event_type="departure",
                waypoint_id=wp.waypoint_id,
                timestamp=current_time,
                details={
                    "unit_id": unit_id,
                    "waypoint_name": wp.name,
                },
            ))

            current_lat, current_lng = wp.lat, wp.lng

    # Sort events chronologically
    all_events.sort(key=lambda e: e.timestamp)

    return PatrolScenario(
        units=units,
        events=all_events,
        area_center=area_center,
        duration_s=duration_s,
        geofence_vertices=geofence,
    )


# ── Threat Scenario Generator ─────────────────────────────────────────

@dataclass
class ThreatActor:
    """A hostile entity approaching a defended area."""
    actor_id: str
    name: str
    asset_type: str  # "person", "vehicle", "drone"
    alliance: str
    lat: float
    lng: float
    heading: float
    speed_mps: float
    approach_angle: float  # direction toward the geofence center
    detection_probability: float  # per-step chance of being detected


@dataclass
class ThreatDetection:
    """A detection event during the threat scenario."""
    timestamp: float
    actor_id: str
    sensor_type: str  # "ble", "camera", "rf_motion", "acoustic"
    lat: float
    lng: float
    confidence: float
    distance_to_fence_m: float
    inside_fence: bool


@dataclass
class GeofenceDefinition:
    """A circular geofence for the threat scenario."""
    center_lat: float
    center_lng: float
    radius_m: float
    name: str


@dataclass
class ThreatScenario:
    """Complete threat scenario with actors, geofence, and detections."""
    actors: list[ThreatActor]
    geofence: GeofenceDefinition
    detections: list[ThreatDetection]
    timeline_s: float
    time_steps: int


def generate_threat_scenario(
    hostile_count: int = 4,
    area_center: tuple[float, float] = (37.7749, -122.4194),
    geofence_radius_m: float = 200.0,
    approach_distance_m: float = 800.0,
    time_steps: int = 30,
    step_interval_s: float = 10.0,
    seed: int = 666,
    base_time: float | None = None,
) -> ThreatScenario:
    """Generate a threat scenario with hostiles approaching a geofenced area.

    Hostile actors start at ``approach_distance_m`` from the geofence
    center and advance toward it over ``time_steps``.  As they approach,
    different sensor types detect them at varying ranges.  The scenario
    includes the moment each actor breaches the geofence.

    Sensor detection ranges (meters):
    - camera: 150 (high confidence close up)
    - rf_motion: 80 (requires proximity)
    - ble: 60 (requires BLE advertisement)
    - acoustic: 200 (vehicle/drone engine noise)

    Parameters
    ----------
    hostile_count : int
        Number of hostile actors.
    area_center : tuple[float, float]
        (lat, lng) center of the geofenced area.
    geofence_radius_m : float
        Radius of the defended area in meters.
    approach_distance_m : float
        Starting distance of hostiles from the center.
    time_steps : int
        Number of simulation steps.
    step_interval_s : float
        Seconds between each step.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ThreatScenario
        Complete scenario with actors, geofence, detections, and timeline.
    """
    rng = random.Random(seed)
    clat, clng = area_center

    geofence = GeofenceDefinition(
        center_lat=clat,
        center_lng=clng,
        radius_m=geofence_radius_m,
        name="Perimeter-Alpha",
    )

    # Sensor detection characteristics: (range_m, base_confidence, sensor_type)
    sensor_profiles: list[tuple[float, float, str]] = [
        (200.0, 0.55, "acoustic"),   # longest range, lowest confidence
        (150.0, 0.70, "camera"),     # visual detection
        (80.0, 0.60, "rf_motion"),   # RF motion detection
        (60.0, 0.80, "ble"),         # BLE proximity
    ]

    threat_types = [
        ("person", 2.0, 0.6),     # (type, speed_mps, detection_prob)
        ("vehicle", 8.0, 0.9),    # vehicles are easier to detect
        ("drone", 12.0, 0.7),     # drones are fast but small
    ]

    # Create actors approaching from different angles
    actors: list[ThreatActor] = []
    for i in range(hostile_count):
        # Spread approach angles around the perimeter
        base_angle = (2.0 * math.pi * i) / hostile_count
        approach_angle = base_angle + rng.uniform(-0.3, 0.3)

        t_type, t_speed, t_det_prob = rng.choice(threat_types)
        # Add speed variation
        speed = t_speed * rng.uniform(0.7, 1.3)

        # Starting position: approach_distance_m from center at approach_angle
        start_lat = clat + math.cos(approach_angle) * approach_distance_m * _DEG_PER_METER_LAT
        start_lng = clng + math.sin(approach_angle) * approach_distance_m * _DEG_PER_METER_LNG_37

        # Heading points toward center (opposite of approach angle)
        heading_toward = approach_angle + math.pi

        actor_id = f"hostile-{i+1:02d}"
        actors.append(ThreatActor(
            actor_id=actor_id,
            name=f"Threat-{t_type.title()}-{i+1}",
            asset_type=t_type,
            alliance=ALLIANCE_HOSTILE,
            lat=start_lat,
            lng=start_lng,
            heading=heading_toward,
            speed_mps=speed,
            approach_angle=approach_angle,
            detection_probability=t_det_prob,
        ))

    # Simulate approach
    detections: list[ThreatDetection] = []
    if base_time is None:
        base_time = time.time()

    for step in range(time_steps):
        ts = base_time + step * step_interval_s

        for actor in actors:
            # Move toward the center with some jitter
            heading_jitter = 0.05 if actor.asset_type == "vehicle" else 0.15
            new_lat, new_lng, new_heading = _move_position(
                actor.lat, actor.lng, actor.heading,
                actor.speed_mps, step_interval_s, heading_jitter, rng,
            )
            actor.lat = new_lat
            actor.lng = new_lng
            actor.heading = new_heading

            # Distance to geofence center
            dist_to_center = _haversine_m(clat, clng, actor.lat, actor.lng)
            dist_to_fence = dist_to_center - geofence_radius_m
            inside_fence = dist_to_fence <= 0

            # Check each sensor for detection
            for sensor_range, base_conf, sensor_type in sensor_profiles:
                if dist_to_center > sensor_range:
                    continue

                # Probability of detection increases as actor gets closer
                range_factor = 1.0 - (dist_to_center / sensor_range)
                detect_prob = actor.detection_probability * range_factor

                if rng.random() < detect_prob:
                    # Confidence: higher when closer, with noise
                    conf = base_conf + range_factor * 0.3
                    conf += rng.gauss(0, 0.05)
                    conf = max(0.3, min(0.99, conf))

                    # Position estimate has error based on sensor type
                    pos_error_m = {
                        "camera": 5.0,
                        "acoustic": 30.0,
                        "rf_motion": 15.0,
                        "ble": 8.0,
                    }.get(sensor_type, 10.0)

                    det_lat = actor.lat + rng.gauss(0, pos_error_m * _DEG_PER_METER_LAT)
                    det_lng = actor.lng + rng.gauss(0, pos_error_m * _DEG_PER_METER_LNG_37)

                    detections.append(ThreatDetection(
                        timestamp=ts,
                        actor_id=actor.actor_id,
                        sensor_type=sensor_type,
                        lat=det_lat,
                        lng=det_lng,
                        confidence=round(conf, 3),
                        distance_to_fence_m=round(max(0, dist_to_fence), 1),
                        inside_fence=inside_fence,
                    ))

    # Sort detections by timestamp
    detections.sort(key=lambda d: d.timestamp)

    return ThreatScenario(
        actors=actors,
        geofence=geofence,
        detections=detections,
        timeline_s=time_steps * step_interval_s,
        time_steps=time_steps,
    )

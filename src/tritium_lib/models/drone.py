# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Drone/UAV integration models for the Tritium ecosystem.

MQTT Topics:
    tritium/{site}/drones/{id}/telemetry  — position, battery, status
    tritium/{site}/drones/{id}/command    — waypoints, RTL, arm/disarm
    tritium/{site}/drones/{id}/mission    — mission plan upload/download
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DroneState(str, Enum):
    IDLE = "idle"
    ARMED = "armed"
    TAKEOFF = "takeoff"
    FLYING = "flying"
    HOVERING = "hovering"
    LANDING = "landing"
    LANDED = "landed"
    RTL = "rtl"
    EMERGENCY = "emergency"
    OFFLINE = "offline"


class DroneType(str, Enum):
    MULTIROTOR = "multirotor"
    FIXED_WING = "fixed_wing"
    VTOL = "vtol"
    GROUND = "ground"


class DroneTelemetry(BaseModel):
    drone_id: str
    state: DroneState = DroneState.OFFLINE
    drone_type: DroneType = DroneType.MULTIROTOR
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_msl: float = 0.0
    altitude_agl: float = 0.0
    heading_deg: float = 0.0
    ground_speed_mps: float = 0.0
    vertical_speed_mps: float = 0.0
    battery_voltage: float = 0.0
    battery_percent: float = 0.0
    gps_fix: int = 0
    satellites: int = 0
    signal_strength: float = 0.0
    mission_item: int = 0
    mission_total: int = 0
    distance_to_home_m: float = 0.0
    time_in_air_s: float = 0.0
    timestamp: float = 0.0


class DroneCommand(BaseModel):
    command: str
    params: dict = Field(default_factory=dict)


class Waypoint(BaseModel):
    seq: int = 0
    latitude: float
    longitude: float
    altitude: float = 30.0
    hold_seconds: float = 0.0
    action: str = "navigate"


class DroneMission(BaseModel):
    mission_id: str
    name: str = ""
    description: str = ""
    waypoints: list[Waypoint] = Field(default_factory=list)
    repeat: bool = False
    max_altitude_m: float = 120.0
    max_speed_mps: float = 15.0
    rtl_altitude_m: float = 30.0
    geofence_radius_m: float = 500.0


class DroneRegistration(BaseModel):
    drone_id: str
    name: str = ""
    drone_type: DroneType = DroneType.MULTIROTOR
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    max_flight_time_min: float = 25.0
    max_payload_g: float = 0.0
    has_camera: bool = True
    firmware_version: str = ""

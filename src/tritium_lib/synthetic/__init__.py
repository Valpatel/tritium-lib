# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.synthetic — synthetic data generators for testing and demos."""

from .data_generators import (
    BLEScanGenerator,
    MeshtasticNodeGenerator,
    CameraDetectionGenerator,
    TrilaterationDemoGenerator,
)
from .scenario_generators import (
    BLESightingRecord,
    WiFiAPRecord,
    WiFiEnvironment,
    PatrolWaypoint,
    PatrolEvent,
    PatrolUnit,
    PatrolScenario,
    ThreatActor,
    ThreatDetection,
    GeofenceDefinition,
    ThreatScenario,
    generate_ble_sightings,
    generate_wifi_environment,
    generate_patrol_scenario,
    generate_threat_scenario,
)

__all__ = [
    # Background generators (EventBus-based)
    "BLEScanGenerator",
    "MeshtasticNodeGenerator",
    "CameraDetectionGenerator",
    "TrilaterationDemoGenerator",
    # Scenario data records
    "BLESightingRecord",
    "WiFiAPRecord",
    "WiFiEnvironment",
    "PatrolWaypoint",
    "PatrolEvent",
    "PatrolUnit",
    "PatrolScenario",
    "ThreatActor",
    "ThreatDetection",
    "GeofenceDefinition",
    "ThreatScenario",
    # Stateless scenario generators
    "generate_ble_sightings",
    "generate_wifi_environment",
    "generate_patrol_scenario",
    "generate_threat_scenario",
]

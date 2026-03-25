# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.fusion — multi-sensor fusion engine.

High-level orchestrator that unifies BLE, WiFi, camera, acoustic, mesh,
ADS-B, and RF-motion data streams into correlated target identities.

This package does NOT re-implement tracking, correlation, geofencing, or
heatmaps.  It composes the existing components from tritium_lib.tracking
and tritium_lib.intelligence into a single ingestion API that downstream
consumers (SC dashboard, Amy AI, REST endpoints) can call.

Quick start::

    from tritium_lib.fusion import FusionEngine

    engine = FusionEngine()
    engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
    engine.ingest_camera({"class_name": "person", "confidence": 0.9,
                          "center_x": 10.0, "center_y": 5.0})
    targets = engine.get_fused_targets()
"""

from .engine import FusionEngine, FusionSnapshot, FusedTarget, SensorRecord
from .pipeline import SensorPipeline

__all__ = [
    "FusionEngine",
    "FusionSnapshot",
    "FusedTarget",
    "SensorRecord",
    "SensorPipeline",
]

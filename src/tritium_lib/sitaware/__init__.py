# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.sitaware — Situational Awareness Engine (the capstone module).

The SitAwareEngine is the top-level orchestrator that ties ALL Tritium
subsystems into a single unified operating picture.  It composes:

  - **FusionEngine**     -> fused targets with multi-sensor identity resolution
  - **AlertEngine**      -> active alerts from rule-based evaluation
  - **AnomalyEngine**    -> behavioral anomaly detection
  - **AnalyticsEngine**  -> real-time statistics and trend analysis
  - **HealthMonitor**    -> system health monitoring
  - **IncidentManager**  -> active incident tracking and lifecycle
  - **MissionPlanner**   -> active mission coordination

into a single query API that produces an **OperatingPicture** — the complete
state of every target, zone, threat, alert, incident, mission, and system
component at a point in time.

Consumers (SC dashboard, Amy AI, REST endpoints, WebSocket feeds) call
``get_picture()`` for a full snapshot, ``get_updates_since(ts)`` for deltas,
or ``subscribe(callback)`` for real-time push updates.

Quick start::

    from tritium_lib.sitaware import SitAwareEngine

    engine = SitAwareEngine()

    # Ingest sensor data through the fusion engine
    engine.fusion.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})

    # Get the unified operating picture
    picture = engine.get_picture()
    print(picture.summary)

    # Subscribe to real-time updates
    def on_update(update):
        print(f"Update: {update.update_type} at {update.timestamp}")

    engine.subscribe(on_update)

Thread-safe. All public methods are safe for concurrent access.
"""

from .engine import (
    OperatingPicture,
    PictureUpdate,
    SitAwareEngine,
    UpdateType,
)

__all__ = [
    "OperatingPicture",
    "PictureUpdate",
    "SitAwareEngine",
    "UpdateType",
]

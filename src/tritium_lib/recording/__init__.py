# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.recording — record and replay sensor data streams.

DEPRECATED (Gap-fix D D-4, 2026-04-29): no production consumers.
Only ``tests/test_recording.py`` and ``tests/test_end_to_end_pipeline.py``
import this package.  TargetTracker snapshot store + heatmap event drain
serve the same playback role for live targets.

TODO: delete this package and its dedicated tests once
``test_end_to_end_pipeline`` can be rewritten to skip the Recorder stage.



Record all sensor events (BLE sightings, WiFi probes, camera detections,
acoustic events, fusion results, alerts, zone events) to a JSON-lines file
for later replay at original or modified speed.

Usage
-----
    from tritium_lib.recording import Recorder, Player, Session

    # Record
    rec = Recorder("/tmp/patrol_2026-03-25.jsonl")
    rec.start()
    rec.record("ble_sighting", source="node_alpha", data={"mac": "AA:BB:CC:DD:EE:FF", "rssi": -45})
    rec.record("camera_detection", source="cam_01", data={"class": "person", "confidence": 0.92})
    session = rec.stop()

    # Replay at 10x speed
    player = Player("/tmp/patrol_2026-03-25.jsonl")
    player.speed = 10.0
    for event in player.replay():
        print(event)

    # Inspect session metadata
    session = Session.from_file("/tmp/patrol_2026-03-25.jsonl")
    print(session.duration, session.event_count, session.sensor_types)
"""

from .recorder import Recorder
from .player import Player, ReplayEvent
from .session import Session

__all__ = [
    "Recorder",
    "Player",
    "ReplayEvent",
    "Session",
]

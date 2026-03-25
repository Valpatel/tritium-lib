# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone tracking demo — exercises the full tracking pipeline.

Run with:
    PYTHONPATH=src python3 -m tritium_lib.tracking.demos.tracking_demo
"""

from __future__ import annotations

import random
import time

from tritium_lib.tracking import (
    TargetTracker,
    TargetHistory,
    GeofenceEngine,
    GeoZone,
    BLEClassifier,
    VehicleTrackingManager,
    ConvoyDetector,
    ThreatScorer,
    MovementPatternAnalyzer,
    HeatmapEngine,
)


class SimpleEventBus:
    """Minimal event bus for the demo."""

    def __init__(self) -> None:
        self._log: list[tuple[str, dict]] = []

    def publish(self, topic: str, data: dict) -> None:
        self._log.append((topic, data))
        print(f"  [EVENT] {topic}: {list(data.keys())}")

    @property
    def event_count(self) -> int:
        return len(self._log)


def main() -> None:
    print("=" * 60)
    print("TRITIUM TRACKING PIPELINE DEMO")
    print("=" * 60)

    bus = SimpleEventBus()
    rng = random.Random(42)

    # 1. Target Tracker
    print("\n--- TargetTracker ---")
    tracker = TargetTracker(event_bus=bus)
    tracker.update_from_ble({"mac": "AA:BB:CC:11:22:01", "name": "iPhone-Matt", "rssi": -45,
                             "node_position": {"x": 10.0, "y": 20.0}})
    tracker.update_from_ble({"mac": "AA:BB:CC:11:22:02", "name": "Galaxy-S24", "rssi": -60,
                             "node_position": {"x": 15.0, "y": 25.0}})
    tracker.update_from_simulation({
        "target_id": "sim_friendly_1", "name": "Turret-Alpha",
        "alliance": "friendly", "asset_type": "turret",
        "position": {"x": 30.0, "y": 40.0}, "heading": 90.0, "speed": 0.0,
    })
    tracker.update_from_simulation({
        "target_id": "sim_hostile_1", "name": "Hostile-1",
        "alliance": "hostile", "asset_type": "infantry",
        "position": {"x": 50.0, "y": 60.0}, "heading": 180.0, "speed": 3.0,
    })
    print(f"  Active targets: {len(tracker.get_all())}")
    print(f"  Hostiles: {len(tracker.get_hostiles())}")
    print(f"  Friendlies: {len(tracker.get_friendlies())}")

    # 2. BLE Classifier
    print("\n--- BLEClassifier ---")
    classifier = BLEClassifier(
        event_bus=bus,
        known_macs={"AA:BB:CC:11:22:01", "AA:BB:CC:11:22:02"},
    )
    for mac, name, rssi in [
        ("AA:BB:CC:11:22:01", "iPhone-Matt", -45),
        ("AA:BB:CC:11:22:02", "Galaxy-S24", -60),
        ("DD:EE:FF:11:22:01", "", -35),  # suspicious — strong signal + unknown
        ("DD:EE:FF:11:22:02", "", -80),  # new — weak signal
    ]:
        c = classifier.classify(mac, name, rssi)
        print(f"  {mac} -> {c.level} (RSSI {c.rssi})")

    # 3. Target History
    print("\n--- TargetHistory ---")
    history = TargetHistory()
    t = time.time()
    for i in range(10):
        history.record("det_car_1", (50.0 + i * 2, 60.0 + i * 1.5), t + i)
        history.record("det_car_2", (100.0 + i * 2, 80.0 + i * 1.5), t + i)
        history.record("det_car_3", (55.0 + i * 2, 65.0 + i * 1.5), t + i)
    trail = history.get_trail("det_car_1", max_points=5)
    print(f"  det_car_1 trail length: {len(trail)}")

    # 4. Vehicle Tracker
    print("\n--- VehicleTrackingManager ---")
    vehicles = VehicleTrackingManager()
    ts = time.monotonic()
    for i in range(10):
        vehicles.update_vehicle("det_car_1", 50.0 + i * 5, 60.0 + i * 3, "car", ts + i)
    vb = vehicles.get_vehicle("det_car_1")
    if vb:
        print(f"  det_car_1: speed={vb.speed_mph:.1f} mph, heading={vb.heading:.0f}, "
              f"direction={vb.direction_label}")

    # 5. Geofence
    print("\n--- GeofenceEngine ---")
    geofence = GeofenceEngine(event_bus=bus)
    geofence.add_zone(GeoZone(
        zone_id="restricted-1",
        name="Restricted Area",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        zone_type="restricted",
    ))
    inside = geofence.check("test-target", (50, 50))
    outside = geofence.check("test-target-2", (200, 200))
    print(f"  (50,50) in restricted: {len(inside) > 0}")
    print(f"  (200,200) in restricted: {len(outside) > 0}")

    # 6. Convoy Detector
    print("\n--- ConvoyDetector ---")
    convoy_det = ConvoyDetector(history=history, event_bus=bus)
    convoys = convoy_det.analyze(["det_car_1", "det_car_2", "det_car_3"])
    print(f"  Active convoys: {len(convoys)}")
    if convoys:
        c = convoys[0]
        print(f"  Convoy members: {c['member_target_ids']}")
        print(f"  Suspicious score: {c['suspicious_score']}")

    # 7. Threat Scorer
    print("\n--- ThreatScorer ---")

    def geofence_check(tid: str, pos: tuple[float, float]) -> bool:
        return len(geofence.check(tid, pos)) > 0

    scorer = ThreatScorer(geofence_checker=geofence_check)
    targets_for_scoring = [
        {"target_id": "hostile-1", "position": (50, 50), "heading": 90.0,
         "speed": 5.0, "source": "yolo", "alliance": "hostile"},
        {"target_id": "friendly-1", "position": (20, 30), "heading": 0.0,
         "speed": 2.0, "source": "ble", "alliance": "friendly"},
    ]
    scores = scorer.evaluate(targets_for_scoring)
    for tid, score in scores.items():
        print(f"  {tid}: threat_score={score:.3f}")

    # 8. Heatmap
    print("\n--- HeatmapEngine ---")
    heatmap = HeatmapEngine()
    for _ in range(50):
        heatmap.record_event("ble_activity", rng.uniform(0, 100), rng.uniform(0, 100))
    result = heatmap.get_heatmap(time_window_minutes=60, resolution=10, layer="ble_activity")
    grid = result.get("grid", [])
    total_heat = sum(sum(row) for row in grid) if grid else 0
    print(f"  Grid 10x10, total heat: {total_heat:.1f}")

    # 9. Movement Patterns
    print("\n--- MovementPatternAnalyzer ---")
    pattern_history = TargetHistory()
    for i in range(20):
        pattern_history.record("patrol-unit", (50.0 + 10 * (i % 5), 60.0 + 10 * (i % 5)), t + i * 60)
    mpa = MovementPatternAnalyzer(history=pattern_history)
    patterns = mpa.analyze("patrol-unit")
    print(f"  Patterns for patrol-unit: {len(patterns)}")

    # Summary
    print("\n" + "=" * 60)
    print(f"DEMO COMPLETE — {bus.event_count} events published")
    print("=" * 60)


if __name__ == "__main__":
    main()

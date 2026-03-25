# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Performance benchmarks for core tritium-lib modules.

Each benchmark times 1000 operations and asserts a minimum throughput
(operations per second). Uses time.perf_counter for accurate timing.
"""

import random
import time

import pytest

from tritium_lib.events.bus import EventBus
from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.tracking.geofence import GeofenceEngine, GeoZone
from tritium_lib.tracking.correlator import TargetCorrelator
from tritium_lib.fusion.engine import FusionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_OPS = 1000


def _make_sim_data(i: int) -> dict:
    """Generate a simulation target data dict."""
    return {
        "target_id": f"sim_target_{i}",
        "name": f"Target {i}",
        "alliance": random.choice(["friendly", "hostile", "unknown"]),
        "asset_type": random.choice(["rover", "drone", "person", "vehicle"]),
        "position": {"x": random.uniform(-500, 500), "y": random.uniform(-500, 500)},
        "heading": random.uniform(0, 360),
        "speed": random.uniform(0, 10),
        "battery": random.uniform(0.2, 1.0),
        "status": "active",
    }


def _make_ble_sighting(i: int) -> dict:
    """Generate a BLE sighting dict."""
    mac = f"AA:BB:CC:DD:{i // 256:02X}:{i % 256:02X}"
    return {
        "mac": mac,
        "name": f"Device {i}",
        "rssi": random.randint(-90, -30),
        "device_type": "ble_device",
        "position": {"x": random.uniform(-100, 100), "y": random.uniform(-100, 100)},
    }


def _make_detection(i: int) -> dict:
    """Generate a YOLO detection dict."""
    return {
        "class_name": random.choice(["person", "car", "motorcycle"]),
        "confidence": random.uniform(0.5, 0.99),
        "center_x": random.uniform(-200, 200),
        "center_y": random.uniform(-200, 200),
    }


def _make_square_zone(zone_id: str, cx: float, cy: float, size: float) -> GeoZone:
    """Create a square geofence zone centered at (cx, cy)."""
    half = size / 2
    return GeoZone(
        zone_id=zone_id,
        name=f"Zone {zone_id}",
        polygon=[
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ],
    )


# ---------------------------------------------------------------------------
# 1. TargetTracker: targets processed per second
# ---------------------------------------------------------------------------

class TestTargetTrackerPerformance:
    """Benchmark TargetTracker update throughput."""

    def test_simulation_update_throughput(self):
        """Time 1000 update_from_simulation calls."""
        tracker = TargetTracker()
        payloads = [_make_sim_data(i) for i in range(N_OPS)]

        start = time.perf_counter()
        for payload in payloads:
            tracker.update_from_simulation(payload)
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  TargetTracker simulation update: {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 5_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 5,000)"

    def test_ble_update_throughput(self):
        """Time 1000 update_from_ble calls."""
        tracker = TargetTracker()
        sightings = [_make_ble_sighting(i) for i in range(N_OPS)]

        start = time.perf_counter()
        for sighting in sightings:
            tracker.update_from_ble(sighting)
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  TargetTracker BLE update: {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 5_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 5,000)"

    def test_detection_update_throughput(self):
        """Time 1000 update_from_detection calls."""
        tracker = TargetTracker()
        detections = [_make_detection(i) for i in range(N_OPS)]

        start = time.perf_counter()
        for det in detections:
            tracker.update_from_detection(det)
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  TargetTracker YOLO detection update: {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 2_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 2,000)"

    def test_get_all_throughput(self):
        """Time 1000 get_all calls on a tracker with 500 targets."""
        tracker = TargetTracker()
        for i in range(500):
            tracker.update_from_simulation(_make_sim_data(i))

        start = time.perf_counter()
        for _ in range(N_OPS):
            tracker.get_all()
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  TargetTracker get_all (500 targets): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 500, f"Too slow: {ops_per_sec:.0f} ops/sec (min 500)"


# ---------------------------------------------------------------------------
# 2. FusionEngine: sensor ingestion throughput
# ---------------------------------------------------------------------------

class TestFusionEnginePerformance:
    """Benchmark FusionEngine ingest throughput."""

    def test_ble_ingest_throughput(self):
        """Time 1000 ingest_ble calls."""
        engine = FusionEngine()
        sightings = [_make_ble_sighting(i) for i in range(N_OPS)]

        start = time.perf_counter()
        for sighting in sightings:
            engine.ingest_ble(sighting)
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  FusionEngine ingest_ble: {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 2_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 2,000)"

    def test_camera_ingest_throughput(self):
        """Time 1000 ingest_camera calls."""
        engine = FusionEngine()
        detections = [_make_detection(i) for i in range(N_OPS)]

        start = time.perf_counter()
        for det in detections:
            engine.ingest_camera(det)
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  FusionEngine ingest_camera: {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 500, f"Too slow: {ops_per_sec:.0f} ops/sec (min 500)"

    def test_mixed_ingest_throughput(self):
        """Time 1000 mixed sensor ingestions (BLE + camera + mesh)."""
        engine = FusionEngine()
        ops: list[tuple[str, dict]] = []
        for i in range(N_OPS):
            kind = i % 3
            if kind == 0:
                ops.append(("ble", _make_ble_sighting(i)))
            elif kind == 1:
                ops.append(("camera", _make_detection(i)))
            else:
                ops.append(("mesh", {
                    "target_id": f"mesh_node_{i}",
                    "name": f"Mesh {i}",
                    "position": {
                        "x": random.uniform(-100, 100),
                        "y": random.uniform(-100, 100),
                    },
                }))

        start = time.perf_counter()
        for kind, data in ops:
            if kind == "ble":
                engine.ingest_ble(data)
            elif kind == "camera":
                engine.ingest_camera(data)
            else:
                engine.ingest_mesh(data)
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  FusionEngine mixed ingest: {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 500, f"Too slow: {ops_per_sec:.0f} ops/sec (min 500)"


# ---------------------------------------------------------------------------
# 3. GeofenceEngine: zone check throughput
# ---------------------------------------------------------------------------

class TestGeofencePerformance:
    """Benchmark GeofenceEngine point-in-polygon check throughput."""

    def test_single_zone_check_throughput(self):
        """Time 1000 check calls against 1 zone."""
        engine = GeofenceEngine()
        engine.add_zone(_make_square_zone("z1", 0.0, 0.0, 100.0))

        positions = [
            (random.uniform(-200, 200), random.uniform(-200, 200))
            for _ in range(N_OPS)
        ]

        start = time.perf_counter()
        for i, pos in enumerate(positions):
            engine.check(f"target_{i}", pos)
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  GeofenceEngine check (1 zone): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 10_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 10,000)"

    def test_multi_zone_check_throughput(self):
        """Time 1000 check calls against 10 zones."""
        engine = GeofenceEngine()
        for j in range(10):
            cx = (j % 5) * 100 - 200
            cy = (j // 5) * 100 - 50
            engine.add_zone(_make_square_zone(f"z{j}", cx, cy, 80.0))

        positions = [
            (random.uniform(-300, 300), random.uniform(-200, 200))
            for _ in range(N_OPS)
        ]

        start = time.perf_counter()
        for i, pos in enumerate(positions):
            engine.check(f"target_{i}", pos)
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  GeofenceEngine check (10 zones): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 5_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 5,000)"

    def test_repeated_target_zone_transitions(self):
        """Time 1000 checks for a single target moving in and out of a zone."""
        engine = GeofenceEngine()
        engine.add_zone(_make_square_zone("z1", 0.0, 0.0, 50.0))

        start = time.perf_counter()
        for i in range(N_OPS):
            # Alternate between inside and outside
            if i % 2 == 0:
                engine.check("mobile_target", (0.0, 0.0))
            else:
                engine.check("mobile_target", (100.0, 100.0))
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  GeofenceEngine transitions (1 target, 1 zone): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 5_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 5,000)"


# ---------------------------------------------------------------------------
# 4. EventBus: publish/subscribe throughput
# ---------------------------------------------------------------------------

class TestEventBusPerformance:
    """Benchmark EventBus publish throughput."""

    def test_publish_no_subscribers(self):
        """Time 1000 publishes with no subscribers."""
        bus = EventBus()

        start = time.perf_counter()
        for i in range(N_OPS):
            bus.publish(f"topic.{i % 10}", data={"value": i})
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  EventBus publish (no subscribers): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 50_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 50,000)"

    def test_publish_single_subscriber(self):
        """Time 1000 publishes with 1 subscriber per topic."""
        bus = EventBus()
        received = []
        for t in range(10):
            bus.subscribe(f"topic.{t}", lambda ev: received.append(ev))

        start = time.perf_counter()
        for i in range(N_OPS):
            bus.publish(f"topic.{i % 10}", data={"value": i})
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  EventBus publish (1 subscriber/topic): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 20_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 20,000)"
        assert len(received) == N_OPS, f"Expected {N_OPS} events, got {len(received)}"

    def test_publish_many_subscribers(self):
        """Time 1000 publishes with 10 subscribers on the same topic."""
        bus = EventBus()
        count = [0]
        for _ in range(10):
            bus.subscribe("events.data", lambda ev: count.__setitem__(0, count[0] + 1))

        start = time.perf_counter()
        for i in range(N_OPS):
            bus.publish("events.data", data={"value": i})
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  EventBus publish (10 subscribers): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 5_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 5,000)"
        assert count[0] == N_OPS * 10, f"Expected {N_OPS * 10} deliveries, got {count[0]}"

    def test_wildcard_subscriber_throughput(self):
        """Time 1000 publishes with a wildcard subscriber."""
        bus = EventBus()
        received = []
        bus.subscribe("sensor.#", lambda ev: received.append(1))

        start = time.perf_counter()
        for i in range(N_OPS):
            bus.publish(f"sensor.ble.{i % 50}", data={"rssi": -70})
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  EventBus publish (wildcard subscriber): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 10_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 10,000)"
        assert len(received) == N_OPS

    def test_publish_with_history(self):
        """Time 1000 publishes with history enabled."""
        bus = EventBus(history_size=100)
        bus.subscribe("data.stream", lambda ev: None)

        start = time.perf_counter()
        for i in range(N_OPS):
            bus.publish("data.stream", data={"seq": i})
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  EventBus publish (history=100): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 10_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 10,000)"


# ---------------------------------------------------------------------------
# 5. Correlator: correlation throughput
# ---------------------------------------------------------------------------

class TestCorrelatorPerformance:
    """Benchmark TargetCorrelator correlation pass throughput."""

    def _populate_tracker_for_correlation(self, tracker: TargetTracker, n: int) -> None:
        """Add n targets from alternating BLE and YOLO sources, nearby in pairs."""
        for i in range(n):
            cx = (i // 2) * 10.0
            cy = 0.0
            if i % 2 == 0:
                tracker.update_from_ble({
                    "mac": f"AA:BB:CC:DD:{i // 256:02X}:{i % 256:02X}",
                    "name": f"BLE {i}",
                    "rssi": -50,
                    "position": {"x": cx, "y": cy},
                })
            else:
                tracker.update_from_detection({
                    "class_name": "person",
                    "confidence": 0.8,
                    "center_x": cx + random.uniform(-2, 2),
                    "center_y": cy + random.uniform(-2, 2),
                })

    def test_correlate_small_set(self):
        """Time 1000 correlation passes over 20 targets (10 pairs)."""
        tracker = TargetTracker()
        correlator = TargetCorrelator(
            tracker,
            radius=5.0,
            confidence_threshold=0.15,
            max_age=60.0,
        )
        self._populate_tracker_for_correlation(tracker, 20)

        start = time.perf_counter()
        for _ in range(N_OPS):
            # Re-populate after each pass since correlation merges targets
            if len(tracker.get_all()) < 10:
                self._populate_tracker_for_correlation(tracker, 20)
            correlator.correlate()
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  Correlator pass (20 targets): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 100, f"Too slow: {ops_per_sec:.0f} ops/sec (min 100)"

    def test_correlate_medium_set(self):
        """Time 100 correlation passes over 100 targets (50 pairs)."""
        n_passes = 100
        tracker = TargetTracker()
        correlator = TargetCorrelator(
            tracker,
            radius=5.0,
            confidence_threshold=0.15,
            max_age=60.0,
        )

        start = time.perf_counter()
        for _ in range(n_passes):
            self._populate_tracker_for_correlation(tracker, 100)
            correlator.correlate()
        elapsed = time.perf_counter() - start

        ops_per_sec = n_passes / elapsed
        print(f"\n  Correlator pass (100 targets): {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 10, f"Too slow: {ops_per_sec:.0f} ops/sec (min 10)"

    def test_evaluate_pair_throughput(self):
        """Time 1000 individual pair evaluations."""
        tracker = TargetTracker()
        correlator = TargetCorrelator(tracker, radius=10.0)

        target_a = TrackedTarget(
            target_id="ble_aabbccddee01",
            name="BLE Device",
            alliance="unknown",
            asset_type="ble_device",
            position=(10.0, 20.0),
            source="ble",
            position_confidence=0.7,
        )
        target_b = TrackedTarget(
            target_id="det_person_1",
            name="Person #1",
            alliance="hostile",
            asset_type="person",
            position=(12.0, 21.0),
            source="yolo",
            position_confidence=0.5,
        )

        start = time.perf_counter()
        for _ in range(N_OPS):
            correlator._evaluate_pair(target_a, target_b)
        elapsed = time.perf_counter() - start

        ops_per_sec = N_OPS / elapsed
        print(f"\n  Correlator pair evaluation: {ops_per_sec:,.0f} ops/sec ({elapsed:.4f}s)")
        assert ops_per_sec > 5_000, f"Too slow: {ops_per_sec:.0f} ops/sec (min 5,000)"


# ---------------------------------------------------------------------------
# Run as standalone script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

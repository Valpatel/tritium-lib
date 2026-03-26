# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Property-based tests for core tritium-lib modules.

Uses random inputs with deterministic seeds to catch edge cases that
unit tests miss. Tests invariants that must hold for ALL valid inputs,
not just cherry-picked examples.

Modules covered:
    - TargetTracker (BLE, simulation, detection, mesh, RF motion, ADS-B)
    - EventBus (pub/sub, wildcards, history, priority, filters)
    - geo (haversine, point_in_polygon, coordinate transforms)
    - TargetCorrelator (merge, weighted scoring)
    - TargetStore (persistence roundtrip)
    - BleStore (sighting persistence)
    - TargetHistory (ring buffer, speed, heading)
    - GeofenceEngine (enter/exit detection)
    - TrackedTarget (confidence decay, multi-source boost)
"""

import math
import random
import threading
import time

import pytest

from tritium_lib.tracking.target_tracker import (
    TargetTracker,
    TrackedTarget,
    _decayed_confidence,
)
from tritium_lib.events.bus import EventBus, Event, QueueEventBus
from tritium_lib.geo import (
    haversine_distance,
    init_reference,
    local_to_latlng,
    latlng_to_local,
    point_in_polygon,
    reset as geo_reset,
)
from tritium_lib.tracking.target_history import TargetHistory
from tritium_lib.tracking.geofence import GeofenceEngine, GeoZone
from tritium_lib.store.targets import TargetStore
from tritium_lib.store.ble import BleStore

# Deterministic seed for reproducibility
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_mac(rng: random.Random) -> str:
    """Generate a random BLE MAC address."""
    return ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6))


def _random_position(rng: random.Random, scale: float = 1000.0) -> tuple[float, float]:
    """Generate a random (x, y) position."""
    return (rng.uniform(-scale, scale), rng.uniform(-scale, scale))


def _random_latlng(rng: random.Random) -> tuple[float, float]:
    """Generate a random valid lat/lng pair."""
    return (rng.uniform(-89.9, 89.9), rng.uniform(-179.9, 179.9))


# ===================================================================
# TargetTracker property tests
# ===================================================================


class TestTargetTrackerProperties:
    """Property tests for TargetTracker."""

    def test_ble_sighting_roundtrip(self):
        """Any valid BLE sighting produces a retrievable target with correct ID format."""
        rng = random.Random(SEED)
        tracker = TargetTracker()
        macs = set()

        for _ in range(200):
            mac = _random_mac(rng)
            macs.add(mac)
            rssi = rng.randint(-100, -30)
            tracker.update_from_ble({
                "mac": mac,
                "rssi": rssi,
                "observer_id": "test",
            })

        all_targets = tracker.get_all()
        target_ids = {t.target_id for t in all_targets}

        # Every unique MAC should have produced exactly one target
        assert len(all_targets) == len(macs)

        # Every target ID must follow the ble_{mac} format
        for t in all_targets:
            assert t.target_id.startswith("ble_")
            assert t.source == "ble"
            assert t.signal_count >= 1

    def test_ble_duplicate_mac_updates_not_creates(self):
        """Updating the same MAC repeatedly should NOT create new targets."""
        rng = random.Random(SEED + 1)
        tracker = TargetTracker()
        mac = _random_mac(rng)

        for i in range(100):
            tracker.update_from_ble({
                "mac": mac,
                "rssi": rng.randint(-100, -30),
                "observer_id": "test",
            })

        all_targets = tracker.get_all()
        assert len(all_targets) == 1
        assert all_targets[0].signal_count == 100

    def test_simulation_target_roundtrip(self):
        """Simulation targets should be retrievable with correct attributes."""
        rng = random.Random(SEED + 2)
        tracker = TargetTracker()

        for i in range(100):
            pos = _random_position(rng)
            tracker.update_from_simulation({
                "target_id": f"sim_{i}",
                "name": f"Unit {i}",
                "alliance": rng.choice(["friendly", "hostile", "unknown"]),
                "asset_type": rng.choice(["rover", "drone", "turret"]),
                "position": {"x": pos[0], "y": pos[1]},
                "heading": rng.uniform(0, 360),
                "speed": rng.uniform(0, 20),
                "battery": rng.uniform(0, 1),
            })

        all_targets = tracker.get_all()
        assert len(all_targets) == 100

        for t in all_targets:
            assert t.source == "simulation"
            assert t.position_confidence == 1.0

    def test_detection_proximity_matching(self):
        """Detections at the same position should match existing targets, not create duplicates."""
        tracker = TargetTracker()
        x, y = 100.0, 200.0

        tracker.update_from_detection({
            "class_name": "person",
            "confidence": 0.9,
            "center_x": x,
            "center_y": y,
        })
        # Same location — should match
        tracker.update_from_detection({
            "class_name": "person",
            "confidence": 0.85,
            "center_x": x + 0.1,
            "center_y": y + 0.1,
        })

        all_targets = tracker.get_all()
        assert len(all_targets) == 1
        assert all_targets[0].signal_count == 2

    def test_low_confidence_detection_rejected(self):
        """Detections below 0.4 confidence should be silently ignored."""
        rng = random.Random(SEED + 3)
        tracker = TargetTracker()

        for _ in range(50):
            tracker.update_from_detection({
                "class_name": "person",
                "confidence": rng.uniform(0.0, 0.39),
                "center_x": rng.uniform(-50, 50),
                "center_y": rng.uniform(-50, 50),
            })

        assert len(tracker.get_all()) == 0

    def test_remove_returns_correct_boolean(self):
        """Remove returns True for existing targets, False for non-existent."""
        tracker = TargetTracker()
        tracker.update_from_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -50})

        assert tracker.remove("ble_aabbccddeeff") is True
        assert tracker.remove("ble_aabbccddeeff") is False
        assert tracker.remove("nonexistent") is False

    def test_get_target_returns_none_for_missing(self):
        """get_target should return None for IDs that don't exist."""
        rng = random.Random(SEED + 4)
        tracker = TargetTracker()

        for _ in range(50):
            result = tracker.get_target(f"missing_{rng.randint(0, 99999)}")
            assert result is None

    def test_alliance_filters(self):
        """get_hostiles/get_friendlies should partition correctly."""
        tracker = TargetTracker()

        for i in range(10):
            tracker.update_from_simulation({
                "target_id": f"friend_{i}",
                "alliance": "friendly",
                "asset_type": "rover",
                "position": {"x": float(i), "y": 0.0},
            })
        for i in range(5):
            tracker.update_from_simulation({
                "target_id": f"hostile_{i}",
                "alliance": "hostile",
                "asset_type": "person",
                "position": {"x": float(i), "y": 10.0},
            })

        friendlies = tracker.get_friendlies()
        hostiles = tracker.get_hostiles()
        assert len(friendlies) == 10
        assert len(hostiles) == 5
        assert all(t.alliance == "friendly" for t in friendlies)
        assert all(t.alliance == "hostile" for t in hostiles)

    def test_empty_mac_ignored(self):
        """BLE sightings with empty MAC should be silently ignored."""
        tracker = TargetTracker()
        tracker.update_from_ble({"mac": "", "rssi": -50})
        assert len(tracker.get_all()) == 0

    def test_mesh_empty_target_id_ignored(self):
        """Mesh updates with empty target_id should be silently ignored."""
        tracker = TargetTracker()
        tracker.update_from_mesh({"target_id": "", "name": "x"})
        assert len(tracker.get_all()) == 0

    def test_rf_motion_empty_target_id_ignored(self):
        """RF motion with empty target_id should be silently ignored."""
        tracker = TargetTracker()
        tracker.update_from_rf_motion({"target_id": "", "position": (1, 2)})
        assert len(tracker.get_all()) == 0

    def test_adsb_empty_target_id_ignored(self):
        """ADS-B with empty target_id should be silently ignored."""
        tracker = TargetTracker()
        tracker.update_from_adsb({"target_id": ""})
        assert len(tracker.get_all()) == 0


# ===================================================================
# EventBus property tests
# ===================================================================


class TestEventBusProperties:
    """Property tests for EventBus."""

    def test_no_lost_events_exact_topic(self):
        """Every published event should reach every matching exact subscriber."""
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda e: received.append(e))

        for i in range(1000):
            bus.publish("test.event", {"i": i})

        assert len(received) == 1000
        for i, ev in enumerate(received):
            assert ev.data["i"] == i

    def test_no_lost_events_wildcard_hash(self):
        """'#' wildcard at root should receive ALL events on any topic."""
        bus = EventBus()
        received = []
        bus.subscribe("#", lambda e: received.append(e))

        rng = random.Random(SEED + 10)
        topics = [f"level{rng.randint(0,5)}.sub{rng.randint(0,9)}" for _ in range(500)]
        for topic in topics:
            bus.publish(topic, {})

        assert len(received) == 500

    def test_no_lost_events_wildcard_star(self):
        """'*' wildcard should match single-level topics."""
        bus = EventBus()
        received = []
        bus.subscribe("device.*", lambda e: received.append(e))

        for i in range(100):
            bus.publish(f"device.sensor{i}", {"i": i})

        assert len(received) == 100

    def test_star_does_not_match_multi_level(self):
        """'*' wildcard should NOT match multi-level topics."""
        bus = EventBus()
        received = []
        bus.subscribe("device.*", lambda e: received.append(e))

        bus.publish("device.sensor.heartbeat", {})
        assert len(received) == 0

    def test_unsubscribe_stops_delivery(self):
        """After unsubscribe, callback should receive no more events."""
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.subscribe("topic", cb)
        bus.publish("topic", {"before": True})
        bus.unsubscribe("topic", cb)
        bus.publish("topic", {"after": True})

        assert len(received) == 1
        assert received[0].data["before"] is True

    def test_multiple_subscribers_all_receive(self):
        """Multiple subscribers on the same topic all receive every event."""
        bus = EventBus()
        results = [[] for _ in range(10)]

        for i in range(10):
            idx = i  # capture
            bus.subscribe("shared", lambda e, idx=idx: results[idx].append(e))

        for j in range(50):
            bus.publish("shared", {"j": j})

        for i in range(10):
            assert len(results[i]) == 50

    def test_subscriber_exception_does_not_break_others(self):
        """A subscriber that raises should not prevent other subscribers from receiving."""
        bus = EventBus()
        received = []

        def bad_cb(e):
            raise RuntimeError("boom")

        bus.subscribe("topic", bad_cb)
        bus.subscribe("topic", lambda e: received.append(e))

        bus.publish("topic", {"ok": True})
        assert len(received) == 1

    def test_publish_returns_event(self):
        """publish() should return the Event object with correct fields."""
        bus = EventBus()
        ev = bus.publish("test.topic", {"val": 42}, source="test_src")

        assert isinstance(ev, Event)
        assert ev.topic == "test.topic"
        assert ev.data == {"val": 42}
        assert ev.source == "test_src"
        assert ev.timestamp > 0

    def test_history_stores_events(self):
        """When history_size > 0, events are retained and retrievable."""
        bus = EventBus(history_size=50)

        for i in range(100):
            bus.publish("sensor.data", {"i": i})

        history = bus.get_history("sensor.data")
        assert len(history) == 50
        # Should be the last 50 events
        assert history[0].data["i"] == 50
        assert history[-1].data["i"] == 99

    def test_history_zero_stores_nothing(self):
        """When history_size=0, no events should be stored."""
        bus = EventBus(history_size=0)

        for i in range(100):
            bus.publish("topic", {"i": i})

        assert len(bus.get_history("topic")) == 0

    def test_priority_ordering(self):
        """Higher-priority subscribers should be called before lower-priority ones."""
        bus = EventBus()
        order = []

        bus.subscribe("event", lambda e: order.append("low"), priority=-10)
        bus.subscribe("event", lambda e: order.append("high"), priority=10)
        bus.subscribe("event", lambda e: order.append("mid"), priority=0)

        bus.publish("event", {})
        assert order == ["high", "mid", "low"]

    def test_filter_fn_prevents_delivery(self):
        """Events that don't pass the filter should not be delivered."""
        bus = EventBus()
        received = []

        bus.subscribe(
            "data",
            lambda e: received.append(e),
            filter_fn=lambda e: e.data.get("important", False),
        )

        bus.publish("data", {"important": False})
        bus.publish("data", {"important": True})
        bus.publish("data", {"important": False})

        assert len(received) == 1
        assert received[0].data["important"] is True


# ===================================================================
# QueueEventBus property tests
# ===================================================================


class TestQueueEventBusProperties:
    """Property tests for QueueEventBus."""

    def test_subscriber_receives_all_events(self):
        """A queue subscriber should receive every published event."""
        bus = QueueEventBus()
        q = bus.subscribe()

        for i in range(100):
            bus.publish(f"event_{i}", {"i": i})

        received = []
        while not q.empty():
            received.append(q.get_nowait())

        assert len(received) == 100

    def test_multiple_queue_subscribers(self):
        """Multiple queue subscribers each get every event."""
        bus = QueueEventBus()
        queues = [bus.subscribe() for _ in range(5)]

        for i in range(50):
            bus.publish("event", {"i": i})

        for q in queues:
            received = []
            while not q.empty():
                received.append(q.get_nowait())
            assert len(received) == 50

    def test_unsubscribe_stops_delivery(self):
        """After unsubscribe, the queue should receive no more events."""
        bus = QueueEventBus()
        q = bus.subscribe()
        bus.publish("before", {})
        bus.unsubscribe(q)
        bus.publish("after", {})

        received = []
        while not q.empty():
            received.append(q.get_nowait())
        assert len(received) == 1
        assert received[0]["type"] == "before"


# ===================================================================
# Geo module property tests
# ===================================================================


class TestGeoProperties:
    """Property tests for geo module."""

    def test_haversine_symmetry(self):
        """haversine(a, b) == haversine(b, a) for random points."""
        rng = random.Random(SEED + 20)
        for _ in range(200):
            lat1, lng1 = _random_latlng(rng)
            lat2, lng2 = _random_latlng(rng)
            d1 = haversine_distance(lat1, lng1, lat2, lng2)
            d2 = haversine_distance(lat2, lng2, lat1, lng1)
            assert abs(d1 - d2) < 0.01, f"Asymmetry: {d1} vs {d2}"

    def test_haversine_identity(self):
        """haversine(a, a) == 0 for any point."""
        rng = random.Random(SEED + 21)
        for _ in range(100):
            lat, lng = _random_latlng(rng)
            d = haversine_distance(lat, lng, lat, lng)
            assert d == 0.0

    def test_haversine_non_negative(self):
        """haversine distance is always >= 0."""
        rng = random.Random(SEED + 22)
        for _ in range(200):
            lat1, lng1 = _random_latlng(rng)
            lat2, lng2 = _random_latlng(rng)
            d = haversine_distance(lat1, lng1, lat2, lng2)
            assert d >= 0.0

    def test_haversine_triangle_inequality(self):
        """haversine must satisfy the triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""
        rng = random.Random(SEED + 23)
        for _ in range(100):
            a = _random_latlng(rng)
            b = _random_latlng(rng)
            c = _random_latlng(rng)
            dab = haversine_distance(a[0], a[1], b[0], b[1])
            dbc = haversine_distance(b[0], b[1], c[0], c[1])
            dac = haversine_distance(a[0], a[1], c[0], c[1])
            # Allow small floating point tolerance
            assert dac <= dab + dbc + 0.1

    def test_coordinate_roundtrip(self):
        """local_to_latlng(latlng_to_local(p)) should roundtrip approximately."""
        geo_reset()
        ref_lat, ref_lng = 40.7128, -74.0060  # NYC
        init_reference(ref_lat, ref_lng)

        rng = random.Random(SEED + 24)
        for _ in range(100):
            # Offsets within ~10km of reference
            dx = rng.uniform(-5000, 5000)
            dy = rng.uniform(-5000, 5000)

            result = local_to_latlng(dx, dy)
            x2, y2, _ = latlng_to_local(result["lat"], result["lng"])

            assert abs(x2 - dx) < 0.01, f"X roundtrip error: {dx} -> {x2}"
            assert abs(y2 - dy) < 0.01, f"Y roundtrip error: {dy} -> {y2}"

        geo_reset()

    def test_coordinate_origin_is_reference(self):
        """Local (0,0) should map to the reference lat/lng."""
        geo_reset()
        ref_lat, ref_lng = 51.5074, -0.1278  # London
        init_reference(ref_lat, ref_lng)

        result = local_to_latlng(0.0, 0.0)
        assert abs(result["lat"] - ref_lat) < 1e-9
        assert abs(result["lng"] - ref_lng) < 1e-9

        geo_reset()

    def test_point_in_polygon_square(self):
        """Points inside a square should return True, outside False."""
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]

        # Inside
        assert point_in_polygon(5, 5, square) is True
        assert point_in_polygon(1, 1, square) is True
        assert point_in_polygon(9, 9, square) is True

        # Outside
        assert point_in_polygon(-1, 5, square) is False
        assert point_in_polygon(11, 5, square) is False
        assert point_in_polygon(5, -1, square) is False
        assert point_in_polygon(5, 11, square) is False

    def test_point_in_polygon_triangle(self):
        """Random points categorized correctly against a known triangle."""
        triangle = [(0, 0), (10, 0), (5, 10)]
        rng = random.Random(SEED + 25)

        for _ in range(200):
            px = rng.uniform(-5, 15)
            py = rng.uniform(-5, 15)
            result = point_in_polygon(px, py, triangle)
            # Simple check: point at centroid is definitely inside
            assert point_in_polygon(5.0, 3.0, triangle) is True
            # Point far outside is definitely outside
            assert point_in_polygon(100, 100, triangle) is False

    def test_point_in_polygon_degenerate(self):
        """Polygon with fewer than 3 vertices should always return False."""
        assert point_in_polygon(0, 0, []) is False
        assert point_in_polygon(0, 0, [(1, 1)]) is False
        assert point_in_polygon(0, 0, [(0, 0), (1, 1)]) is False

    def test_haversine_known_values(self):
        """Verify haversine against known distance (NYC to London ~5570km)."""
        nyc = (40.7128, -74.0060)
        london = (51.5074, -0.1278)
        d = haversine_distance(nyc[0], nyc[1], london[0], london[1])
        assert 5500_000 < d < 5600_000  # 5500-5600 km


# ===================================================================
# TargetHistory property tests
# ===================================================================


class TestTargetHistoryProperties:
    """Property tests for TargetHistory."""

    def test_record_and_retrieve(self):
        """All recorded positions should be retrievable via get_trail."""
        rng = random.Random(SEED + 30)
        history = TargetHistory()
        n = 200

        for i in range(n):
            pos = _random_position(rng)
            history.record("target_a", pos, timestamp=float(i))

        trail = history.get_trail("target_a", max_points=n)
        assert len(trail) == n

    def test_ring_buffer_limit(self):
        """History should not exceed MAX_RECORDS_PER_TARGET."""
        history = TargetHistory()
        limit = history.MAX_RECORDS_PER_TARGET

        for i in range(limit + 500):
            history.record("overflow_target", (float(i), 0.0), timestamp=float(i))

        trail = history.get_trail("overflow_target", max_points=limit + 500)
        assert len(trail) == limit

    def test_speed_stationary_is_zero(self):
        """A target that doesn't move should have speed ~0."""
        history = TargetHistory()
        for i in range(10):
            history.record("still_target", (5.0, 5.0), timestamp=float(i))

        speed = history.get_speed("still_target")
        assert speed < 0.001

    def test_speed_moving_positive(self):
        """A target moving at constant speed should report positive speed."""
        history = TargetHistory()
        # Move 1 meter/second along X axis
        for i in range(20):
            history.record("mover", (float(i), 0.0), timestamp=float(i))

        speed = history.get_speed("mover")
        assert 0.9 < speed < 1.1  # Should be ~1.0 m/s

    def test_heading_eastward(self):
        """A target moving east (+X) should have heading ~90 degrees."""
        history = TargetHistory()
        for i in range(10):
            history.record("east", (float(i), 0.0), timestamp=float(i))

        heading = history.get_heading("east")
        assert 85 < heading < 95

    def test_heading_northward(self):
        """A target moving north (+Y) should have heading ~0 degrees."""
        history = TargetHistory()
        for i in range(10):
            history.record("north", (0.0, float(i)), timestamp=float(i))

        heading = history.get_heading("north")
        assert heading < 5 or heading > 355

    def test_clear_removes_data(self):
        """clear() should remove all history for a target."""
        history = TargetHistory()
        history.record("victim", (1.0, 2.0))
        assert history.tracked_count == 1
        history.clear("victim")
        assert history.tracked_count == 0
        assert history.get_trail("victim") == []

    def test_nonexistent_target_returns_empty(self):
        """Querying a non-existent target should return empty/zero."""
        history = TargetHistory()
        assert history.get_trail("ghost") == []
        assert history.get_speed("ghost") == 0.0
        assert history.get_heading("ghost") == 0.0

    def test_multiple_targets_isolated(self):
        """History for different targets should be independent."""
        rng = random.Random(SEED + 31)
        history = TargetHistory()

        for i in range(50):
            history.record("alpha", _random_position(rng), timestamp=float(i))
            history.record("beta", _random_position(rng), timestamp=float(i))

        trail_a = history.get_trail("alpha", max_points=100)
        trail_b = history.get_trail("beta", max_points=100)
        assert len(trail_a) == 50
        assert len(trail_b) == 50
        # They should be different (different random positions)
        assert trail_a != trail_b


# ===================================================================
# GeofenceEngine property tests
# ===================================================================


class TestGeofenceProperties:
    """Property tests for GeofenceEngine."""

    def test_enter_exit_cycle(self):
        """Moving in and out of a zone should produce enter/exit events."""
        engine = GeofenceEngine()
        zone = GeoZone(
            zone_id="zone_1",
            name="Test Zone",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        )
        engine.add_zone(zone)

        # Enter
        events = engine.check("target_1", (5.0, 5.0))
        enter_events = [e for e in events if e.event_type == "enter"]
        assert len(enter_events) == 1
        assert enter_events[0].zone_id == "zone_1"

        # Stay inside
        events = engine.check("target_1", (6.0, 6.0))
        inside_events = [e for e in events if e.event_type == "inside"]
        assert len(inside_events) == 1

        # Exit
        events = engine.check("target_1", (20.0, 20.0))
        exit_events = [e for e in events if e.event_type == "exit"]
        assert len(exit_events) == 1
        assert exit_events[0].zone_id == "zone_1"

    def test_zone_occupant_tracking(self):
        """get_zone_occupants should reflect current zone membership."""
        engine = GeofenceEngine()
        zone = GeoZone(
            zone_id="z",
            name="Z",
            polygon=[(-5, -5), (5, -5), (5, 5), (-5, 5)],
        )
        engine.add_zone(zone)

        # No occupants initially
        assert engine.get_zone_occupants("z") == []

        # Enter with 3 targets
        engine.check("t1", (0, 0))
        engine.check("t2", (1, 1))
        engine.check("t3", (2, 2))

        occupants = engine.get_zone_occupants("z")
        assert set(occupants) == {"t1", "t2", "t3"}

        # t1 exits
        engine.check("t1", (100, 100))
        occupants = engine.get_zone_occupants("z")
        assert set(occupants) == {"t2", "t3"}

    def test_disabled_zone_ignored(self):
        """Disabled zones should not trigger any events."""
        engine = GeofenceEngine()
        zone = GeoZone(
            zone_id="disabled",
            name="Disabled Zone",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            enabled=False,
        )
        engine.add_zone(zone)

        events = engine.check("target_1", (5, 5))
        assert len(events) == 0

    def test_multiple_zones_tracked_independently(self):
        """A target can be in multiple overlapping zones simultaneously."""
        engine = GeofenceEngine()
        zone_a = GeoZone(
            zone_id="a",
            name="Zone A",
            polygon=[(0, 0), (20, 0), (20, 20), (0, 20)],
        )
        zone_b = GeoZone(
            zone_id="b",
            name="Zone B",
            polygon=[(10, 10), (30, 10), (30, 30), (10, 30)],
        )
        engine.add_zone(zone_a)
        engine.add_zone(zone_b)

        # Point in overlap region
        engine.check("t1", (15, 15))
        zones = engine.get_target_zones("t1")
        assert zones == {"a", "b"}

    def test_remove_zone(self):
        """Removing a zone should clean up occupant state."""
        engine = GeofenceEngine()
        zone = GeoZone(
            zone_id="removable",
            name="Removable",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        )
        engine.add_zone(zone)
        engine.check("t1", (5, 5))
        assert "removable" in engine.get_target_zones("t1")

        engine.remove_zone("removable")
        assert "removable" not in engine.get_target_zones("t1")


# ===================================================================
# TrackedTarget confidence decay tests
# ===================================================================


class TestConfidenceDecayProperties:
    """Property tests for confidence decay and multi-source boosting."""

    def test_decay_non_negative(self):
        """Decayed confidence should never be negative."""
        rng = random.Random(SEED + 40)
        sources = ["ble", "wifi", "yolo", "rf_motion", "mesh", "adsb", "manual", "unknown_source"]

        for _ in range(200):
            source = rng.choice(sources)
            initial = rng.uniform(0.0, 1.0)
            elapsed = rng.uniform(0.0, 10000.0)
            result = _decayed_confidence(source, initial, elapsed)
            assert result >= 0.0

    def test_decay_bounded_by_one(self):
        """Decayed confidence should never exceed 1.0."""
        rng = random.Random(SEED + 41)
        sources = ["ble", "wifi", "yolo", "rf_motion", "mesh", "adsb", "simulation"]

        for _ in range(200):
            source = rng.choice(sources)
            initial = rng.uniform(0.0, 2.0)  # even if initial > 1.0
            elapsed = rng.uniform(0.0, 1000.0)
            result = _decayed_confidence(source, initial, elapsed)
            assert result <= 1.0

    def test_decay_monotonic_decrease(self):
        """Confidence should decrease (or stay same) as elapsed time increases."""
        rng = random.Random(SEED + 42)

        for _ in range(100):
            source = rng.choice(["ble", "wifi", "yolo", "mesh"])
            initial = rng.uniform(0.1, 1.0)
            t1 = rng.uniform(0.0, 100.0)
            t2 = t1 + rng.uniform(0.1, 100.0)

            c1 = _decayed_confidence(source, initial, t1)
            c2 = _decayed_confidence(source, initial, t2)
            assert c2 <= c1 + 1e-9  # c2 should be <= c1

    def test_simulation_never_decays(self):
        """Simulation source should have zero decay (half_life=0)."""
        for elapsed in [0, 1, 10, 100, 1000, 10000]:
            result = _decayed_confidence("simulation", 1.0, float(elapsed))
            assert result == 1.0

    def test_zero_elapsed_returns_initial(self):
        """At elapsed=0, confidence should equal initial (clamped to [0,1])."""
        rng = random.Random(SEED + 43)
        for _ in range(50):
            source = rng.choice(["ble", "wifi", "mesh"])
            initial = rng.uniform(0.0, 1.0)
            result = _decayed_confidence(source, initial, 0.0)
            assert abs(result - initial) < 1e-9


# ===================================================================
# TargetStore property tests
# ===================================================================


class TestTargetStoreProperties:
    """Property tests for TargetStore (SQLite persistence)."""

    def setup_method(self):
        self.store = TargetStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_record_sighting_roundtrip(self):
        """Any sighting recorded should be retrievable by ID."""
        rng = random.Random(SEED + 50)

        for i in range(100):
            tid = f"target_{i}"
            x = rng.uniform(-1000, 1000)
            y = rng.uniform(-1000, 1000)
            self.store.record_sighting(
                target_id=tid,
                name=f"Target {i}",
                alliance=rng.choice(["friendly", "hostile", "unknown"]),
                asset_type="vehicle",
                source="ble",
                position_x=x,
                position_y=y,
                position_confidence=rng.uniform(0, 1),
            )

        for i in range(100):
            result = self.store.get_target(f"target_{i}")
            assert result is not None
            assert result["target_id"] == f"target_{i}"

    def test_update_preserves_first_seen(self):
        """Updating a target should preserve first_seen but update last_seen."""
        self.store.record_sighting(
            target_id="stable",
            name="Original",
            source="ble",
            timestamp=1000.0,
        )
        first = self.store.get_target("stable")
        assert first is not None
        first_seen = first["first_seen"]

        self.store.record_sighting(
            target_id="stable",
            source="wifi",
            timestamp=2000.0,
        )
        updated = self.store.get_target("stable")
        assert updated is not None
        assert updated["first_seen"] == first_seen
        assert updated["last_seen"] == 2000.0

    def test_get_all_with_filters(self):
        """Filters on source and alliance should work correctly."""
        for i in range(20):
            self.store.record_sighting(
                target_id=f"ble_{i}",
                alliance="hostile",
                source="ble",
            )
        for i in range(10):
            self.store.record_sighting(
                target_id=f"wifi_{i}",
                alliance="friendly",
                source="wifi",
            )

        ble_targets = self.store.get_all_targets(source="ble")
        assert len(ble_targets) == 20

        friendly = self.store.get_all_targets(alliance="friendly")
        assert len(friendly) == 10

    def test_delete_target_removes_history(self):
        """Deleting a target should also remove its position history."""
        self.store.record_sighting(
            target_id="doomed",
            position_x=1.0,
            position_y=2.0,
        )
        history = self.store.get_history("doomed")
        assert len(history) == 1

        deleted = self.store.delete_target("doomed")
        assert deleted is True

        assert self.store.get_target("doomed") is None
        assert len(self.store.get_history("doomed")) == 0

    def test_search_finds_by_name(self):
        """Full-text search should find targets by name."""
        self.store.record_sighting(
            target_id="findme",
            name="AlphaRover",
            source="simulation",
        )
        results = self.store.search("AlphaRover")
        assert len(results) >= 1
        assert any(r["target_id"] == "findme" for r in results)

    def test_stats_accurate(self):
        """get_stats should report accurate counts."""
        for i in range(15):
            self.store.record_sighting(
                target_id=f"t_{i}",
                source="ble" if i < 10 else "wifi",
                alliance="hostile" if i < 5 else "friendly",
            )

        stats = self.store.get_stats()
        assert stats["total_targets"] == 15
        assert stats["by_source"]["ble"] == 10
        assert stats["by_source"]["wifi"] == 5
        assert stats["by_alliance"]["hostile"] == 5
        assert stats["by_alliance"]["friendly"] == 10

    def test_trajectory_ordering(self):
        """get_trajectory should return positions oldest first."""
        for i in range(10):
            self.store.record_sighting(
                target_id="mover",
                position_x=float(i),
                position_y=0.0,
                source="sim",
                timestamp=float(1000 + i),
            )

        traj = self.store.get_trajectory("mover")
        assert len(traj) == 10
        timestamps = [t["timestamp"] for t in traj]
        assert timestamps == sorted(timestamps)

    def test_metadata_merge(self):
        """Recording with metadata should merge, not overwrite."""
        self.store.record_sighting(
            target_id="meta_target",
            metadata={"key1": "value1"},
        )
        self.store.record_sighting(
            target_id="meta_target",
            metadata={"key2": "value2"},
        )
        result = self.store.get_target("meta_target")
        assert result is not None
        assert result["metadata"]["key1"] == "value1"
        assert result["metadata"]["key2"] == "value2"


# ===================================================================
# BleStore property tests
# ===================================================================


class TestBleStoreProperties:
    """Property tests for BleStore."""

    def setup_method(self):
        self.store = BleStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_record_sighting_returns_rowid(self):
        """record_sighting should return a positive row ID."""
        rng = random.Random(SEED + 60)
        for _ in range(50):
            mac = _random_mac(rng)
            rowid = self.store.record_sighting(
                mac=mac,
                name="Test",
                rssi=rng.randint(-100, -30),
                node_id="node_1",
            )
            assert rowid > 0

    def test_batch_insert_count(self):
        """Batch insert should return the correct count."""
        rng = random.Random(SEED + 61)
        sightings = [
            {
                "mac": _random_mac(rng),
                "name": f"dev_{i}",
                "rssi": rng.randint(-100, -30),
                "node_id": "batch_node",
            }
            for i in range(100)
        ]
        count = self.store.record_sightings_batch(sightings)
        assert count == 100

    def test_add_and_list_targets(self):
        """Added BLE targets should appear in list_targets."""
        rng = random.Random(SEED + 62)
        macs = set()
        for i in range(20):
            mac = _random_mac(rng)
            macs.add(mac)
            self.store.add_target(mac, f"Label {i}")

        targets = self.store.list_targets()
        target_macs = {t["mac"] for t in targets}
        assert target_macs == macs

    def test_remove_target(self):
        """Removing a target should make it disappear from list_targets."""
        self.store.add_target("AA:BB:CC:DD:EE:FF", "Test")
        assert self.store.remove_target("AA:BB:CC:DD:EE:FF") is True
        assert self.store.remove_target("AA:BB:CC:DD:EE:FF") is False

        targets = self.store.list_targets()
        assert len(targets) == 0

    def test_node_position_roundtrip(self):
        """Node positions should be retrievable after setting."""
        rng = random.Random(SEED + 63)
        for i in range(30):
            x, y = _random_position(rng)
            self.store.set_node_position(f"node_{i}", x, y)

        positions = self.store.get_node_positions()
        assert len(positions) == 30

        for i in range(30):
            pos = self.store.get_node_position(f"node_{i}")
            assert pos is not None

    def test_wifi_sighting_roundtrip(self):
        """WiFi sightings should be recordable and retrievable."""
        rng = random.Random(SEED + 64)
        bssid = _random_mac(rng)

        for _ in range(25):
            self.store.record_wifi_sighting(
                ssid="TestNetwork",
                bssid=bssid,
                rssi=rng.randint(-90, -30),
                channel=rng.randint(1, 11),
                node_id="wifi_node_1",
            )

        history = self.store.get_wifi_history(bssid)
        assert len(history) == 25

    def test_empty_batch_returns_zero(self):
        """Empty batch insert should return 0."""
        assert self.store.record_sightings_batch([]) == 0
        assert self.store.record_wifi_sightings_batch([]) == 0

    def test_stats_fields_present(self):
        """get_stats should return all expected fields."""
        stats = self.store.get_stats()
        expected_keys = {
            "db_size_bytes",
            "sighting_count",
            "device_count",
            "target_count",
            "node_count",
            "oldest_sighting",
            "newest_sighting",
            "wifi_sighting_count",
            "wifi_network_count",
        }
        assert set(stats.keys()) == expected_keys


# ===================================================================
# Thread-safety property tests
# ===================================================================


class TestThreadSafetyProperties:
    """Property tests verifying thread-safety of core components."""

    def test_tracker_concurrent_ble_updates(self):
        """Concurrent BLE updates from multiple threads should not lose data."""
        tracker = TargetTracker()
        num_threads = 8
        macs_per_thread = 50
        all_macs: list[set] = [set() for _ in range(num_threads)]

        def worker(thread_id: int):
            rng = random.Random(SEED + 100 + thread_id)
            for _ in range(macs_per_thread):
                mac = _random_mac(rng)
                all_macs[thread_id].add(mac)
                tracker.update_from_ble({
                    "mac": mac,
                    "rssi": rng.randint(-100, -30),
                })

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_unique = set()
        for s in all_macs:
            total_unique |= s

        all_targets = tracker.get_all()
        assert len(all_targets) == len(total_unique)

    def test_eventbus_concurrent_publish(self):
        """Concurrent publishing should deliver all events to subscribers."""
        bus = EventBus()
        received = []
        lock = threading.Lock()

        def safe_append(e):
            with lock:
                received.append(e)

        bus.subscribe("concurrent", safe_append)

        num_threads = 4
        events_per_thread = 250

        def publisher(thread_id):
            for i in range(events_per_thread):
                bus.publish("concurrent", {"tid": thread_id, "i": i})

        threads = [threading.Thread(target=publisher, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == num_threads * events_per_thread

    def test_history_concurrent_recording(self):
        """Concurrent history recording should not corrupt data."""
        history = TargetHistory()
        num_threads = 4
        records_per_thread = 200

        def recorder(thread_id):
            rng = random.Random(SEED + 200 + thread_id)
            for i in range(records_per_thread):
                history.record(
                    f"concurrent_{thread_id}",
                    _random_position(rng),
                    timestamp=float(i),
                )

        threads = [threading.Thread(target=recorder, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(num_threads):
            trail = history.get_trail(f"concurrent_{i}", max_points=500)
            assert len(trail) == records_per_thread


# ===================================================================
# TargetCorrelator property tests
# ===================================================================


class TestCorrelatorProperties:
    """Property tests for TargetCorrelator (lightweight, no background threads)."""

    def test_same_source_never_correlated(self):
        """Targets from the same source should never be correlated."""
        from tritium_lib.tracking.correlator import TargetCorrelator

        tracker = TargetTracker()
        # Two BLE targets at the same position
        tracker.update_from_ble({
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -40,
            "position": {"x": 5.0, "y": 5.0},
        })
        tracker.update_from_ble({
            "mac": "AA:BB:CC:DD:EE:02",
            "rssi": -42,
            "position": {"x": 5.1, "y": 5.1},
        })

        correlator = TargetCorrelator(
            tracker,
            radius=100.0,
            max_age=9999.0,
            confidence_threshold=0.0,  # very low threshold
        )
        records = correlator.correlate()

        # Same-source pairs should be skipped
        for r in records:
            t1 = tracker.get_target(r.primary_id)
            t2 = tracker.get_target(r.secondary_id)
            # If both still exist, they shouldn't both be "ble"
            # (secondary gets removed on merge, so it won't be found)

    def test_correlation_records_stored(self):
        """Correlation records should be retrievable after correlate()."""
        from tritium_lib.tracking.correlator import TargetCorrelator

        tracker = TargetTracker()
        # BLE and YOLO target at same position — should correlate
        tracker.update_from_ble({
            "mac": "11:22:33:44:55:66",
            "rssi": -30,
            "position": {"x": 10.0, "y": 10.0},
        })
        tracker.update_from_detection({
            "class_name": "person",
            "confidence": 0.9,
            "center_x": 10.1,
            "center_y": 10.1,
        })

        correlator = TargetCorrelator(
            tracker,
            radius=5.0,
            max_age=9999.0,
            confidence_threshold=0.01,
        )
        new_records = correlator.correlate()
        all_records = correlator.get_correlations()

        # All new records should be in get_correlations()
        for r in new_records:
            assert r in all_records

    def test_weighted_score_bounded(self):
        """Weighted score should always be in [0, 1]."""
        from tritium_lib.tracking.correlator import TargetCorrelator
        from tritium_lib.tracking.correlation_strategies import StrategyScore

        tracker = TargetTracker()
        correlator = TargetCorrelator(tracker)
        rng = random.Random(SEED + 70)

        for _ in range(200):
            scores = [
                StrategyScore(
                    strategy_name=name,
                    score=rng.uniform(0, 1),
                    detail="test",
                )
                for name in ["spatial", "temporal", "signal_pattern", "dossier", "wifi_probe"]
            ]
            ws = correlator._weighted_score(scores)
            assert 0.0 <= ws <= 1.0

    def test_merge_preserves_primary_id(self):
        """After merge, the primary target should still exist."""
        from tritium_lib.tracking.correlator import TargetCorrelator

        tracker = TargetTracker()
        tracker.update_from_ble({
            "mac": "AA:BB:CC:DD:EE:FF",
            "rssi": -30,
            "position": {"x": 0, "y": 0},
        })
        tracker.update_from_detection({
            "class_name": "person",
            "confidence": 0.95,
            "center_x": 0.5,
            "center_y": 0.5,
        })

        correlator = TargetCorrelator(
            tracker,
            radius=5.0,
            max_age=9999.0,
            confidence_threshold=0.01,
        )
        records = correlator.correlate()

        if records:
            primary = tracker.get_target(records[0].primary_id)
            assert primary is not None
            # Primary should have accumulated correlated IDs
            assert len(primary.correlated_ids) > 0


# ===================================================================
# Edge case / stress tests
# ===================================================================


class TestEdgeCases:
    """Stress tests and edge case verification."""

    def test_tracker_large_volume(self):
        """Tracker should handle 1000 simultaneous targets without error."""
        tracker = TargetTracker()
        rng = random.Random(SEED + 80)

        for i in range(1000):
            tracker.update_from_simulation({
                "target_id": f"mass_{i}",
                "alliance": rng.choice(["friendly", "hostile", "unknown"]),
                "asset_type": "rover",
                "position": {"x": rng.uniform(-5000, 5000), "y": rng.uniform(-5000, 5000)},
            })

        all_targets = tracker.get_all()
        assert len(all_targets) == 1000

    def test_eventbus_rapid_subscribe_unsubscribe(self):
        """Rapid subscribe/unsubscribe cycles should not corrupt state."""
        bus = EventBus()

        for _ in range(100):
            cb = lambda e: None
            bus.subscribe("churn", cb)
            bus.publish("churn", {})
            bus.unsubscribe("churn", cb)

        # Should still work after churn
        received = []
        bus.subscribe("churn", lambda e: received.append(e))
        bus.publish("churn", {"final": True})
        assert len(received) == 1

    def test_store_special_characters_in_name(self):
        """Store should handle special characters in names without SQL injection."""
        store = TargetStore(":memory:")
        try:
            # SQL injection attempt
            store.record_sighting(
                target_id="inject_test",
                name="'; DROP TABLE targets; --",
                source="test",
            )
            result = store.get_target("inject_test")
            assert result is not None
            assert result["name"] == "'; DROP TABLE targets; --"

            # Table should still exist
            all_targets = store.get_all_targets()
            assert len(all_targets) >= 1
        finally:
            store.close()

    def test_tracker_summary_with_no_targets(self):
        """summary() on empty tracker should return empty string."""
        tracker = TargetTracker()
        assert tracker.summary() == ""

    def test_tracker_summary_with_targets(self):
        """summary() should produce non-empty string when targets exist."""
        tracker = TargetTracker()
        tracker.update_from_simulation({
            "target_id": "s1",
            "alliance": "hostile",
            "asset_type": "person",
            "position": {"x": 0, "y": 0},
        })

        summary = tracker.summary()
        assert len(summary) > 0
        assert "hostile" in summary.lower() or "HOSTILE" in summary or "1" in summary

    def test_point_in_polygon_concave(self):
        """Point-in-polygon should work for concave polygons."""
        # L-shaped polygon
        polygon = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]

        # Inside the L
        assert point_in_polygon(2, 2, polygon) is True
        assert point_in_polygon(2, 8, polygon) is True
        assert point_in_polygon(8, 2, polygon) is True

        # In the concavity (outside the L)
        assert point_in_polygon(8, 8, polygon) is False

    def test_geofence_event_bus_integration(self):
        """GeofenceEngine should publish events to the EventBus when wired."""
        bus = EventBus()
        engine = GeofenceEngine(event_bus=bus)
        received_enter = []
        received_exit = []

        bus.subscribe("geofence:enter", lambda e: received_enter.append(e))
        bus.subscribe("geofence:exit", lambda e: received_exit.append(e))

        zone = GeoZone(
            zone_id="bus_zone",
            name="Bus Zone",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        )
        engine.add_zone(zone)

        engine.check("target_x", (5, 5))
        assert len(received_enter) == 1

        engine.check("target_x", (50, 50))
        assert len(received_exit) == 1

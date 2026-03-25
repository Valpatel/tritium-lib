# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.vehicle_pipeline."""

import math
import time

import pytest

from tritium_lib.tracking.vehicle_pipeline import (
    VehiclePipeline,
    VehicleClassification,
    RouteEstimate,
    ParkingEvent,
    ConvoyResult,
    WiFiProbeRecord,
    PARKING_MIN_DURATION_S,
    PARKING_CONFIRMED_DURATION_S,
    PARKING_SPEED_MPH,
    WIFI_PROBE_RANGE_M,
    WIFI_PROBE_TEMPORAL_WINDOW_S,
    MIN_TRAIL_POINTS_FOR_ROUTE,
)
from tritium_lib.tracking.vehicle_tracker import VehicleTrackingManager


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

class FakeConvoyDetector:
    """Minimal mock for ConvoyDetector."""

    def __init__(self, convoys=None):
        self._convoys = convoys or []

    def analyze(self, target_ids=None):
        return self._convoys


class FakeStreetGraph:
    """Minimal mock for StreetGraph."""

    def __init__(self, path=None, nearest=None, road_class="residential"):
        self._path = path
        self._nearest = nearest or (0, 5.0)
        self._road_class = road_class
        # Simulate a loaded graph
        self.graph = self  # truthy, duck-typed for `graph is not None`

    def nearest_node(self, x, y):
        return self._nearest

    def shortest_path(self, start, end):
        if self._path is not None:
            return self._path
        # Default: straight line from start to end in 5 steps
        sx, sy = start
        ex, ey = end
        return [
            (sx + (ex - sx) * i / 4, sy + (ey - sy) * i / 4)
            for i in range(5)
        ]

    def edges(self, node=None, data=False):
        """Duck-type for networkx graph.edges()."""
        if data:
            return [(0, 1, {"road_class": self._road_class, "weight": 10.0})]
        return [(0, 1)]


def make_pipeline(**kwargs):
    """Create a VehiclePipeline with sensible defaults."""
    return VehiclePipeline(**kwargs)


def feed_moving_vehicle(pipeline, target_id, n_points=5, speed_mps=10.0,
                        heading_deg=90.0, start_x=0.0, start_y=0.0,
                        start_ts=1000.0, vehicle_class="car"):
    """Feed a series of camera detections simulating a moving vehicle."""
    heading_rad = math.radians(heading_deg)
    for i in range(n_points):
        ts = start_ts + i
        x = start_x + math.sin(heading_rad) * speed_mps * i
        y = start_y + math.cos(heading_rad) * speed_mps * i
        pipeline.ingest_camera_detection(
            target_id=target_id,
            x=x, y=y,
            vehicle_class=vehicle_class,
            timestamp=ts,
        )


# ===================================================================
# 1. Basic construction
# ===================================================================

def test_pipeline_construction():
    """Pipeline can be created with no arguments."""
    p = make_pipeline()
    assert p is not None
    assert p._vehicle_mgr is not None


def test_pipeline_with_custom_manager():
    """Pipeline accepts an external VehicleTrackingManager."""
    mgr = VehicleTrackingManager(max_vehicles=50)
    p = make_pipeline(vehicle_manager=mgr)
    assert p._vehicle_mgr is mgr


# ===================================================================
# 2. Camera detection ingestion
# ===================================================================

def test_ingest_camera_detection_creates_vehicle():
    p = make_pipeline()
    p.ingest_camera_detection("car-1", 10.0, 20.0, vehicle_class="car", timestamp=100.0)
    vb = p._vehicle_mgr.get_vehicle("car-1")
    assert vb is not None
    assert vb.vehicle_class == "car"


def test_ingest_camera_detection_stores_yolo_hint():
    p = make_pipeline()
    p.ingest_camera_detection("truck-1", 5.0, 5.0, vehicle_class="truck", timestamp=100.0)
    assert p._yolo_hints["truck-1"] == "truck"


def test_ingest_camera_detection_stores_size_hint():
    p = make_pipeline()
    p.ingest_camera_detection("car-1", 5.0, 5.0, bbox_area=25000.0, timestamp=100.0)
    assert p._size_hints["car-1"] == 25000.0


def test_ingest_camera_detection_stores_plate():
    p = make_pipeline()
    p.ingest_camera_detection("car-1", 5.0, 5.0, plate="ABC-1234", timestamp=100.0)
    assert p._lpr_data["car-1"] == "ABC-1234"


# ===================================================================
# 3. WiFi probe ingestion and association
# ===================================================================

def test_ingest_wifi_probe_buffered():
    p = make_pipeline()
    p.ingest_wifi_probe(mac="AA:BB:CC:DD:EE:01", rssi=-60, ssid="CarNet",
                        position=(10.0, 20.0), timestamp=100.0)
    assert "AA:BB:CC:DD:EE:01" in p._wifi_probes
    assert len(p._wifi_probes["AA:BB:CC:DD:EE:01"]) == 1


def test_wifi_probe_associated_with_nearby_vehicle():
    """WiFi probe near a camera detection should be associated."""
    p = make_pipeline()
    # First ingest a WiFi probe at position (10, 20)
    p.ingest_wifi_probe(mac="AA:BB:CC:DD:EE:01", rssi=-60,
                        position=(10.0, 20.0), timestamp=100.0)
    # Then a camera detection at (12, 20) — within WIFI_PROBE_RANGE_M
    p.ingest_camera_detection("car-1", 12.0, 20.0, timestamp=101.0)
    assert "AA:BB:CC:DD:EE:01" in p._vehicle_wifi.get("car-1", set())


def test_wifi_probe_not_associated_if_far():
    """WiFi probe far from a vehicle should not be associated."""
    p = make_pipeline()
    p.ingest_wifi_probe(mac="AA:BB:CC:DD:EE:01", rssi=-60,
                        position=(100.0, 200.0), timestamp=100.0)
    p.ingest_camera_detection("car-1", 0.0, 0.0, timestamp=101.0)
    assert "AA:BB:CC:DD:EE:01" not in p._vehicle_wifi.get("car-1", set())


def test_wifi_probe_not_associated_if_stale():
    """WiFi probe too old should not be associated."""
    p = make_pipeline()
    p.ingest_wifi_probe(mac="AA:BB:CC:DD:EE:01", rssi=-60,
                        position=(10.0, 20.0), timestamp=10.0)
    # Camera detection much later
    p.ingest_camera_detection("car-1", 10.0, 20.0,
                              timestamp=10.0 + WIFI_PROBE_TEMPORAL_WINDOW_S + 5)
    assert "AA:BB:CC:DD:EE:01" not in p._vehicle_wifi.get("car-1", set())


# ===================================================================
# 4. LPR ingestion
# ===================================================================

def test_ingest_lpr():
    p = make_pipeline()
    p.ingest_lpr("car-1", "XYZ-9999")
    assert p.get_plate("car-1") == "XYZ-9999"


def test_get_plate_unknown():
    p = make_pipeline()
    assert p.get_plate("nonexistent") is None


# ===================================================================
# 5. Vehicle classification
# ===================================================================

def test_classify_unknown_vehicle():
    """Vehicle with no signals should be classified as unknown."""
    p = make_pipeline()
    result = p.classify_vehicle("no-such-vehicle")
    assert result.vehicle_type == "unknown"
    assert result.confidence == 0.0


def test_classify_car_by_yolo():
    """YOLO 'car' class should produce a car classification."""
    p = make_pipeline()
    feed_moving_vehicle(p, "car-1", vehicle_class="car", speed_mps=15.0)
    result = p.classify_vehicle("car-1")
    assert result.vehicle_type == "car"
    assert result.confidence > 0.0
    assert "yolo" in result.signals_used


def test_classify_truck_by_yolo_and_size():
    """Truck class + large bbox should classify as truck."""
    p = make_pipeline()
    p.ingest_camera_detection("truck-1", 0.0, 0.0, vehicle_class="truck",
                              bbox_area=60000.0, timestamp=100.0)
    p.ingest_camera_detection("truck-1", 5.0, 0.0, vehicle_class="truck",
                              bbox_area=60000.0, timestamp=101.0)
    result = p.classify_vehicle("truck-1")
    assert result.vehicle_type == "truck"
    assert "size" in result.signals_used


def test_classify_motorcycle_by_small_size():
    """Small bbox + motorcycle YOLO class."""
    p = make_pipeline()
    p.ingest_camera_detection("moto-1", 0.0, 0.0, vehicle_class="motorcycle",
                              bbox_area=6000.0, timestamp=100.0)
    p.ingest_camera_detection("moto-1", 5.0, 0.0, vehicle_class="motorcycle",
                              bbox_area=6000.0, timestamp=101.0)
    result = p.classify_vehicle("moto-1")
    assert result.vehicle_type == "motorcycle"


def test_classify_bus_maps_to_truck():
    """YOLO 'bus' should contribute to truck score."""
    p = make_pipeline()
    p.ingest_camera_detection("bus-1", 0.0, 0.0, vehicle_class="bus",
                              bbox_area=70000.0, timestamp=100.0)
    p.ingest_camera_detection("bus-1", 3.0, 0.0, vehicle_class="bus",
                              bbox_area=70000.0, timestamp=101.0)
    result = p.classify_vehicle("bus-1")
    assert result.vehicle_type == "truck"


def test_classify_with_wifi_signal():
    """WiFi associations should appear in classification signals."""
    p = make_pipeline()
    p.ingest_wifi_probe(mac="AA:01", rssi=-60, position=(0.0, 0.0), timestamp=99.0)
    p.ingest_wifi_probe(mac="AA:02", rssi=-60, position=(0.0, 0.0), timestamp=99.0)
    p.ingest_wifi_probe(mac="AA:03", rssi=-60, position=(0.0, 0.0), timestamp=99.0)
    p.ingest_camera_detection("car-1", 0.0, 0.0, vehicle_class="car", timestamp=100.0)
    p.ingest_camera_detection("car-1", 5.0, 0.0, vehicle_class="car", timestamp=101.0)
    result = p.classify_vehicle("car-1")
    assert "wifi" in result.signals_used
    assert result.wifi_device_count >= 3


def test_classification_to_dict():
    p = make_pipeline()
    feed_moving_vehicle(p, "car-1", vehicle_class="car")
    result = p.classify_vehicle("car-1")
    d = result.to_dict()
    assert "target_id" in d
    assert "vehicle_type" in d
    assert "confidence" in d


# ===================================================================
# 6. Route estimation
# ===================================================================

def test_route_estimation_no_vehicle():
    """Route for unknown vehicle returns None."""
    p = make_pipeline()
    assert p.estimate_route("nope") is None


def test_route_estimation_insufficient_history():
    """Vehicle with too few points returns None."""
    p = make_pipeline()
    p.ingest_camera_detection("car-1", 0.0, 0.0, timestamp=100.0)
    assert p.estimate_route("car-1") is None


def test_route_estimation_stationary_vehicle():
    """Stationary vehicle returns a minimal route."""
    p = make_pipeline()
    # Feed enough points but all at same position (very slow)
    for i in range(5):
        p.ingest_camera_detection("car-1", 0.0, 0.0, timestamp=100.0 + i)
    route = p.estimate_route("car-1")
    assert route is not None
    assert route.total_distance_m == 0.0
    assert len(route.waypoints) >= 1


def test_route_estimation_linear_projection():
    """Moving vehicle without street graph gets linear projection."""
    p = make_pipeline()
    feed_moving_vehicle(p, "car-1", speed_mps=10.0, heading_deg=0.0)
    route = p.estimate_route("car-1")
    assert route is not None
    assert route.confidence == 0.2  # Linear projection confidence
    assert len(route.waypoints) > 1
    # Waypoints should advance northward (heading=0)
    assert route.waypoints[-1][1] > route.waypoints[0][1]


def test_route_estimation_with_street_graph():
    """Moving vehicle with a street graph gets road-network routing."""
    sg = FakeStreetGraph(
        path=[(0, 0), (10, 10), (20, 20), (30, 30)],
        nearest=(0, 2.0),
        road_class="primary",
    )
    p = make_pipeline(street_graph=sg)
    feed_moving_vehicle(p, "car-1", speed_mps=10.0, heading_deg=45.0)
    route = p.estimate_route("car-1")
    assert route is not None
    assert route.confidence > 0.2  # Higher than linear projection
    assert route.current_road_class == "primary"
    assert len(route.waypoints) >= 2


def test_route_to_dict():
    p = make_pipeline()
    feed_moving_vehicle(p, "car-1", speed_mps=10.0, heading_deg=0.0)
    route = p.estimate_route("car-1")
    d = route.to_dict()
    assert "waypoints" in d
    assert "total_distance_m" in d
    assert "confidence" in d


# ===================================================================
# 7. Convoy detection
# ===================================================================

def test_convoy_no_vehicles():
    """No vehicles tracked means no convoys."""
    p = make_pipeline()
    assert p.detect_convoy() == []


def test_convoy_too_few_vehicles():
    """A single vehicle cannot form a convoy."""
    p = make_pipeline()
    feed_moving_vehicle(p, "car-1", speed_mps=10.0)
    assert p.detect_convoy() == []


def test_convoy_internal_detection():
    """Two co-moving vehicles should be detected as a convoy (internal)."""
    p = make_pipeline()
    ts = 1000.0
    # Two vehicles moving east at similar speed, close together
    for i in range(5):
        p.ingest_camera_detection("car-1", i * 10.0, 0.0, timestamp=ts + i)
        p.ingest_camera_detection("car-2", i * 10.0, 5.0, timestamp=ts + i)
    convoys = p.detect_convoy()
    assert len(convoys) >= 1
    assert len(convoys[0].member_ids) >= 2


def test_convoy_with_external_detector():
    """Pipeline delegates to ConvoyDetector when available."""
    fake_convoys = [
        {
            "convoy_id": "c-001",
            "member_target_ids": ["car-1", "car-2", "car-3"],
            "speed_avg_mps": 12.0,
            "heading_avg_deg": 90.0,
            "duration_s": 60.0,
            "suspicious_score": 0.7,
        }
    ]
    cd = FakeConvoyDetector(fake_convoys)
    p = make_pipeline(convoy_detector=cd)
    # We need actual vehicle data for spread computation
    for tid in ["car-1", "car-2", "car-3"]:
        feed_moving_vehicle(p, tid, speed_mps=12.0, heading_deg=90.0)
    convoys = p.detect_convoy()
    assert len(convoys) == 1
    assert convoys[0].convoy_id == "c-001"
    assert convoys[0].confidence == 0.7


def test_convoy_divergent_headings_not_detected():
    """Vehicles going in opposite directions should not form a convoy."""
    p = make_pipeline()
    ts = 1000.0
    for i in range(5):
        # Car 1 going east
        p.ingest_camera_detection("car-1", i * 10.0, 0.0, timestamp=ts + i)
        # Car 2 going west
        p.ingest_camera_detection("car-2", 50.0 - i * 10.0, 5.0, timestamp=ts + i)
    convoys = p.detect_convoy()
    assert len(convoys) == 0


def test_convoy_result_to_dict():
    cr = ConvoyResult(
        convoy_id="c-test", member_ids=["a", "b"],
        avg_speed_mps=10.0, avg_heading_deg=90.0,
        spread_m=20.0, duration_s=60.0, confidence=0.8,
    )
    d = cr.to_dict()
    assert d["convoy_id"] == "c-test"
    assert d["spread_m"] == 20.0


# ===================================================================
# 8. Parking detection
# ===================================================================

def test_parking_no_vehicle():
    """Parking detection for unknown vehicle returns None."""
    p = make_pipeline()
    assert p.parking_detection("nope") is None


def test_parking_moving_vehicle():
    """Moving vehicle is not parked."""
    p = make_pipeline()
    feed_moving_vehicle(p, "car-1", speed_mps=15.0)
    assert p.parking_detection("car-1") is None


def test_parking_just_stopped():
    """Vehicle that just stopped is not yet parked."""
    p = make_pipeline()
    ts = 1000.0
    p.ingest_camera_detection("car-1", 0.0, 0.0, timestamp=ts)
    p.ingest_camera_detection("car-1", 0.01, 0.0, timestamp=ts + 1)
    result = p.parking_detection("car-1")
    # First call registers the parking state but returns None
    assert result is None


def test_parking_after_min_duration():
    """Vehicle stopped for > PARKING_MIN_DURATION_S is detected as parked."""
    p = make_pipeline()
    ts = 1000.0
    p.ingest_camera_detection("car-1", 0.0, 0.0, timestamp=ts)
    p.ingest_camera_detection("car-1", 0.01, 0.0, timestamp=ts + 1)

    # Manually set parking state to simulate time passing
    now = time.monotonic()
    p._parking_state["car-1"] = ((0.0, 0.0), now - PARKING_MIN_DURATION_S - 10)

    result = p.parking_detection("car-1")
    assert result is not None
    assert result.target_id == "car-1"
    assert result.duration_s > PARKING_MIN_DURATION_S


def test_parking_confirmed():
    """Vehicle parked > PARKING_CONFIRMED_DURATION_S is confirmed."""
    p = make_pipeline()
    ts = 1000.0
    p.ingest_camera_detection("car-1", 0.0, 0.0, timestamp=ts)
    p.ingest_camera_detection("car-1", 0.01, 0.0, timestamp=ts + 1)

    now = time.monotonic()
    p._parking_state["car-1"] = ((0.0, 0.0), now - PARKING_CONFIRMED_DURATION_S - 10)

    result = p.parking_detection("car-1")
    assert result is not None
    assert result.is_confirmed is True


def test_parking_drift_resets():
    """If vehicle drifts >5m from park spot, parking state resets."""
    p = make_pipeline()
    ts = 1000.0
    p.ingest_camera_detection("car-1", 0.0, 0.0, timestamp=ts)
    p.ingest_camera_detection("car-1", 0.01, 0.0, timestamp=ts + 1)

    now = time.monotonic()
    # Set parking state at origin, but vehicle is now at (10, 0) — drifted
    p._parking_state["car-1"] = ((0.0, 0.0), now - PARKING_MIN_DURATION_S - 10)

    # Feed a stopped detection at (10, 0)
    p.ingest_camera_detection("car-1", 10.0, 0.0, timestamp=ts + 2)
    p.ingest_camera_detection("car-1", 10.01, 0.0, timestamp=ts + 3)
    result = p.parking_detection("car-1")
    # Should have reset — not yet parked at new location
    assert result is None


def test_parking_event_to_dict():
    pe = ParkingEvent(
        target_id="car-1", position=(10.0, 20.0),
        parked_since=1000.0, duration_s=120.0,
        is_confirmed=False, nearby_wifi_devices=["AA:01"],
    )
    d = pe.to_dict()
    assert d["target_id"] == "car-1"
    assert d["duration_s"] == 120.0


def test_get_parked_vehicles():
    """get_parked_vehicles aggregates all parking events."""
    p = make_pipeline()
    ts = 1000.0
    now = time.monotonic()

    # Create two stopped vehicles
    for vid in ["car-1", "car-2"]:
        p.ingest_camera_detection(vid, 0.0, 0.0, timestamp=ts)
        p.ingest_camera_detection(vid, 0.01, 0.0, timestamp=ts + 1)
        p._parking_state[vid] = ((0.0, 0.0), now - PARKING_MIN_DURATION_S - 10)

    parked = p.get_parked_vehicles()
    assert len(parked) == 2


def test_parking_with_street_graph():
    """Parking detection includes road_class from street graph."""
    sg = FakeStreetGraph(nearest=(0, 3.0), road_class="service")
    p = make_pipeline(street_graph=sg)
    ts = 1000.0
    p.ingest_camera_detection("car-1", 0.0, 0.0, timestamp=ts)
    p.ingest_camera_detection("car-1", 0.01, 0.0, timestamp=ts + 1)

    now = time.monotonic()
    p._parking_state["car-1"] = ((0.0, 0.0), now - PARKING_MIN_DURATION_S - 10)

    result = p.parking_detection("car-1")
    assert result is not None
    assert result.road_class == "service"


# ===================================================================
# 9. Query methods
# ===================================================================

def test_get_vehicle_status():
    p = make_pipeline()
    feed_moving_vehicle(p, "car-1", vehicle_class="car", speed_mps=10.0)
    status = p.get_vehicle_status("car-1")
    assert status is not None
    assert "classification" in status
    assert status["classification"]["vehicle_type"] == "car"


def test_get_vehicle_status_unknown():
    p = make_pipeline()
    assert p.get_vehicle_status("nope") is None


def test_get_all_vehicles():
    p = make_pipeline()
    feed_moving_vehicle(p, "car-1", speed_mps=10.0)
    feed_moving_vehicle(p, "car-2", speed_mps=12.0, start_x=50.0)
    vehicles = p.get_all_vehicles()
    assert len(vehicles) == 2


def test_get_summary():
    p = make_pipeline()
    feed_moving_vehicle(p, "car-1", speed_mps=10.0)
    p.ingest_lpr("car-1", "ABC-1234")
    p.ingest_wifi_probe(mac="AA:01", rssi=-60, position=(0.0, 0.0), timestamp=999.0)
    summary = p.get_summary()
    assert summary["total"] >= 1
    assert summary["lpr_plates"] == 1
    assert "wifi_probes_buffered" in summary


# ===================================================================
# 10. WiFiProbeRecord dataclass
# ===================================================================

def test_wifi_probe_record_defaults():
    rec = WiFiProbeRecord(mac="AA:BB", ssid="Test", rssi=-70, timestamp=100.0)
    assert rec.position == (0.0, 0.0)
    assert rec.associated_vehicle_id is None


# ===================================================================
# 11. Integration: classify + route + parking combined
# ===================================================================

def test_full_pipeline_workflow():
    """End-to-end: ingest, classify, route, check parking."""
    sg = FakeStreetGraph()
    p = make_pipeline(street_graph=sg)

    ts = 1000.0
    # Ingest WiFi probes
    p.ingest_wifi_probe(mac="D1", rssi=-55, position=(0.0, 0.0), timestamp=ts - 5)
    p.ingest_wifi_probe(mac="D2", rssi=-60, position=(1.0, 0.0), timestamp=ts - 4)

    # Ingest camera detections — vehicle moving then stopping
    for i in range(5):
        p.ingest_camera_detection(
            "car-1", i * 10.0, 0.0,
            vehicle_class="car", bbox_area=20000.0,
            timestamp=ts + i,
        )

    # Classify
    cls = p.classify_vehicle("car-1")
    assert cls.vehicle_type == "car"
    assert cls.confidence > 0.0

    # Route estimation
    route = p.estimate_route("car-1")
    assert route is not None
    assert route.total_distance_m > 0

    # Not parked yet (vehicle is moving)
    assert p.parking_detection("car-1") is None

    # Summary
    summary = p.get_summary()
    assert summary["total"] >= 1

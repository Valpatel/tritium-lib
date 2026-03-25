# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""VehiclePipeline — specialized multi-sensor vehicle tracking pipeline.

Orchestrates WiFi probe analysis, camera/LPR detections, and speed/heading
analysis into a unified vehicle tracking workflow.  Integrates with
TargetTracker for identity, StreetGraph for route estimation, and
ConvoyDetector for group detection.

Key capabilities:
  - classify_vehicle: determine car/truck/motorcycle/bicycle from signal
    patterns (WiFi probe count, YOLO class, size estimate, speed profile)
  - estimate_route: predict a vehicle's likely route from its trajectory
    and the road network
  - detect_convoy: identify groups of vehicles traveling together
  - parking_detection: detect when a vehicle has parked and where

Usage::

    pipeline = VehiclePipeline(
        tracker=target_tracker,
        street_graph=street_graph,
        convoy_detector=convoy_detector,
    )
    pipeline.ingest_wifi_probe(mac="AA:BB:CC:DD:EE:FF", rssi=-65, ssid="CarNet")
    pipeline.ingest_camera_detection(target_id="det_car_5", x=10.0, y=20.0, ...)
    result = pipeline.classify_vehicle("det_car_5")
    route = pipeline.estimate_route("det_car_5")
    convoys = pipeline.detect_convoy()
    parked = pipeline.parking_detection("det_car_5")
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("vehicle_pipeline")

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

# WiFi probe thresholds for vehicle classification
WIFI_PROBE_VEHICLE_MIN = 2          # Min probes to associate WiFi with a vehicle
WIFI_PROBE_RANGE_M = 15.0           # Max distance to associate a probe with a vehicle
WIFI_PROBE_TEMPORAL_WINDOW_S = 30.0 # Time window for probe-vehicle association

# Classification signal weights
WEIGHT_YOLO_CLASS = 0.4
WEIGHT_SPEED_PROFILE = 0.25
WEIGHT_SIZE_ESTIMATE = 0.2
WEIGHT_WIFI_PATTERN = 0.15

# Speed profiles (mph) for classification heuristics
SPEED_PROFILES: dict[str, tuple[float, float]] = {
    "car": (5.0, 80.0),
    "truck": (3.0, 65.0),
    "motorcycle": (5.0, 100.0),
    "bicycle": (2.0, 25.0),
}

# Parking thresholds
PARKING_SPEED_MPH = 2.0
PARKING_MIN_DURATION_S = 60.0       # Must be stopped 60s to count as parked
PARKING_CONFIRMED_DURATION_S = 300.0  # 5 min = confirmed parking

# Route estimation
MAX_ROUTE_WAYPOINTS = 50
ROUTE_LOOKAHEAD_S = 300.0           # Predict 5 minutes ahead
MIN_TRAIL_POINTS_FOR_ROUTE = 3

# Convoy detection relay
MIN_CONVOY_VEHICLES = 2


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------

@dataclass
class VehicleClassification:
    """Result of vehicle type classification from multi-signal analysis."""

    target_id: str
    vehicle_type: str               # car, truck, motorcycle, bicycle, unknown
    confidence: float               # 0.0 to 1.0
    yolo_class: str = "unknown"     # Raw YOLO class if available
    speed_match: float = 0.0        # How well speed matches the type (0-1)
    wifi_device_count: int = 0      # Number of WiFi devices associated
    size_estimate: str = "medium"   # small, medium, large
    signals_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "vehicle_type": self.vehicle_type,
            "confidence": round(self.confidence, 3),
            "yolo_class": self.yolo_class,
            "speed_match": round(self.speed_match, 3),
            "wifi_device_count": self.wifi_device_count,
            "size_estimate": self.size_estimate,
            "signals_used": list(self.signals_used),
        }


@dataclass
class RouteEstimate:
    """Predicted route for a vehicle based on trajectory and road network."""

    target_id: str
    waypoints: list[tuple[float, float]]  # (x, y) waypoints
    total_distance_m: float               # Total route distance in meters
    estimated_time_s: float               # Estimated travel time in seconds
    current_road_class: str = "unknown"   # Highway type the vehicle is on
    heading_deg: float = 0.0              # Current heading
    avg_speed_mps: float = 0.0            # Average speed in m/s
    confidence: float = 0.0               # Route prediction confidence

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "waypoints": [list(w) for w in self.waypoints],
            "total_distance_m": round(self.total_distance_m, 1),
            "estimated_time_s": round(self.estimated_time_s, 1),
            "current_road_class": self.current_road_class,
            "heading_deg": round(self.heading_deg, 1),
            "avg_speed_mps": round(self.avg_speed_mps, 2),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ParkingEvent:
    """Describes a vehicle parking detection event."""

    target_id: str
    position: tuple[float, float]
    parked_since: float                   # Monotonic timestamp
    duration_s: float                     # How long parked so far
    is_confirmed: bool                    # True if parked > PARKING_CONFIRMED_DURATION_S
    nearby_wifi_devices: list[str] = field(default_factory=list)
    road_class: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "position": list(self.position),
            "duration_s": round(self.duration_s, 1),
            "is_confirmed": self.is_confirmed,
            "nearby_wifi_devices": list(self.nearby_wifi_devices),
            "road_class": self.road_class,
        }


@dataclass
class ConvoyResult:
    """Result of convoy detection among tracked vehicles."""

    convoy_id: str
    member_ids: list[str]
    avg_speed_mps: float
    avg_heading_deg: float
    spread_m: float                       # Max distance between members
    duration_s: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "convoy_id": self.convoy_id,
            "member_ids": list(self.member_ids),
            "avg_speed_mps": round(self.avg_speed_mps, 2),
            "avg_heading_deg": round(self.avg_heading_deg, 1),
            "spread_m": round(self.spread_m, 1),
            "duration_s": round(self.duration_s, 1),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class WiFiProbeRecord:
    """A WiFi probe observation associated with a vehicle."""

    mac: str
    ssid: str
    rssi: int
    timestamp: float
    position: tuple[float, float] = (0.0, 0.0)
    associated_vehicle_id: Optional[str] = None


# -------------------------------------------------------------------
# VehiclePipeline
# -------------------------------------------------------------------

class VehiclePipeline:
    """Orchestrates multi-sensor vehicle tracking: WiFi probes, camera
    detections, speed analysis, route estimation, and convoy detection.

    Thread-safe. All public methods acquire the internal lock.

    Parameters
    ----------
    tracker:
        A TargetTracker instance for identity resolution. Duck-typed:
        needs ``get_target(id)``, ``history`` attribute with
        ``get_trail``/``get_speed``/``get_heading``.
    street_graph:
        A StreetGraph instance for road-network routing. Optional.
    convoy_detector:
        A ConvoyDetector instance. Optional — if not provided, a
        lightweight internal convoy check is used.
    vehicle_manager:
        A VehicleTrackingManager instance. Optional — if not provided,
        one is created internally.
    """

    def __init__(
        self,
        tracker=None,
        street_graph=None,
        convoy_detector=None,
        vehicle_manager=None,
    ) -> None:
        self._tracker = tracker
        self._street_graph = street_graph
        self._convoy_detector = convoy_detector
        self._lock = threading.Lock()

        # Internal vehicle behavior manager
        if vehicle_manager is not None:
            self._vehicle_mgr = vehicle_manager
        else:
            from .vehicle_tracker import VehicleTrackingManager
            self._vehicle_mgr = VehicleTrackingManager()

        # WiFi probe buffer: mac -> list of probe records
        self._wifi_probes: dict[str, list[WiFiProbeRecord]] = {}
        # Vehicle -> associated WiFi MACs
        self._vehicle_wifi: dict[str, set[str]] = {}
        # YOLO class hints per vehicle
        self._yolo_hints: dict[str, str] = {}
        # Size estimates from bounding box area
        self._size_hints: dict[str, float] = {}
        # Parking state: target_id -> (position, parked_since)
        self._parking_state: dict[str, tuple[tuple[float, float], float]] = {}
        # LPR (license plate) data: target_id -> plate string
        self._lpr_data: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_wifi_probe(
        self,
        mac: str,
        rssi: int = -80,
        ssid: str = "",
        position: tuple[float, float] = (0.0, 0.0),
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a WiFi probe request that may be associated with a vehicle.

        Probes are buffered and later correlated with nearby vehicle
        detections by position and time.
        """
        ts = timestamp or time.monotonic()
        rec = WiFiProbeRecord(
            mac=mac, ssid=ssid, rssi=rssi,
            timestamp=ts, position=position,
        )
        with self._lock:
            self._wifi_probes.setdefault(mac, []).append(rec)
            # Trim old probes
            cutoff = ts - WIFI_PROBE_TEMPORAL_WINDOW_S * 4
            self._wifi_probes[mac] = [
                r for r in self._wifi_probes[mac] if r.timestamp > cutoff
            ]

    def ingest_camera_detection(
        self,
        target_id: str,
        x: float,
        y: float,
        vehicle_class: str = "car",
        bbox_area: float = 0.0,
        plate: str = "",
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a camera/YOLO vehicle detection.

        Updates the internal VehicleTrackingManager and stores YOLO class
        and size hints for classification.
        """
        ts = timestamp or time.monotonic()
        with self._lock:
            self._vehicle_mgr.update_vehicle(
                target_id, x, y,
                vehicle_class=vehicle_class,
                timestamp=ts,
            )
            self._yolo_hints[target_id] = vehicle_class
            if bbox_area > 0:
                self._size_hints[target_id] = bbox_area
            if plate:
                self._lpr_data[target_id] = plate

            # Associate nearby WiFi probes
            self._associate_wifi_probes(target_id, x, y, ts)

    def ingest_lpr(
        self,
        target_id: str,
        plate: str,
    ) -> None:
        """Record a license plate recognition result."""
        with self._lock:
            self._lpr_data[target_id] = plate

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_vehicle(self, target_id: str) -> VehicleClassification:
        """Classify a vehicle's type from multi-signal analysis.

        Combines:
          - YOLO detection class (car/truck/motorcycle/bicycle)
          - Speed profile matching
          - Bounding-box size estimate
          - WiFi probe pattern (device count)

        Returns a VehicleClassification with type and confidence.
        """
        with self._lock:
            return self._classify_vehicle_locked(target_id)

    def _classify_vehicle_locked(self, target_id: str) -> VehicleClassification:
        """Internal classification — must hold self._lock."""
        signals_used: list[str] = []
        scores: dict[str, float] = {
            "car": 0.0,
            "truck": 0.0,
            "motorcycle": 0.0,
            "bicycle": 0.0,
        }

        # --- Signal 1: YOLO class ---
        yolo_class = self._yolo_hints.get(target_id, "unknown")
        if yolo_class in scores:
            scores[yolo_class] += WEIGHT_YOLO_CLASS
            signals_used.append("yolo")
        elif yolo_class == "bus":
            scores["truck"] += WEIGHT_YOLO_CLASS * 0.8
            signals_used.append("yolo")

        # --- Signal 2: Speed profile ---
        speed_match = 0.0
        vb = self._vehicle_mgr.get_vehicle(target_id)
        if vb is not None and vb.speed_mph > 0:
            signals_used.append("speed")
            best_type = "car"
            best_fit = 0.0
            for vtype, (low, high) in SPEED_PROFILES.items():
                if low <= vb.speed_mph <= high:
                    # How centered is the speed in the range?
                    mid = (low + high) / 2.0
                    span = (high - low) / 2.0
                    fit = 1.0 - abs(vb.speed_mph - mid) / span if span > 0 else 0.5
                    fit = max(0.0, fit)
                    scores[vtype] += WEIGHT_SPEED_PROFILE * fit
                    if fit > best_fit:
                        best_fit = fit
                        best_type = vtype
            speed_match = best_fit

        # --- Signal 3: Size estimate from bbox ---
        size_estimate = "medium"
        bbox_area = self._size_hints.get(target_id, 0.0)
        if bbox_area > 0:
            signals_used.append("size")
            if bbox_area > 50000:
                size_estimate = "large"
                scores["truck"] += WEIGHT_SIZE_ESTIMATE
            elif bbox_area > 15000:
                size_estimate = "medium"
                scores["car"] += WEIGHT_SIZE_ESTIMATE
            elif bbox_area > 5000:
                size_estimate = "small"
                scores["motorcycle"] += WEIGHT_SIZE_ESTIMATE * 0.6
                scores["car"] += WEIGHT_SIZE_ESTIMATE * 0.4
            else:
                size_estimate = "small"
                scores["bicycle"] += WEIGHT_SIZE_ESTIMATE * 0.7
                scores["motorcycle"] += WEIGHT_SIZE_ESTIMATE * 0.3

        # --- Signal 4: WiFi device count ---
        wifi_count = len(self._vehicle_wifi.get(target_id, set()))
        if wifi_count > 0:
            signals_used.append("wifi")
            # More WiFi devices = more likely a car (passengers have phones)
            if wifi_count >= 3:
                scores["car"] += WEIGHT_WIFI_PATTERN
            elif wifi_count == 2:
                scores["car"] += WEIGHT_WIFI_PATTERN * 0.7
                scores["truck"] += WEIGHT_WIFI_PATTERN * 0.3
            else:
                # Single device — could be anything
                scores["motorcycle"] += WEIGHT_WIFI_PATTERN * 0.3
                scores["car"] += WEIGHT_WIFI_PATTERN * 0.3
                scores["bicycle"] += WEIGHT_WIFI_PATTERN * 0.2

        # Determine best classification
        if not signals_used:
            return VehicleClassification(
                target_id=target_id,
                vehicle_type="unknown",
                confidence=0.0,
                signals_used=[],
            )

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        total_weight = sum(scores.values())
        confidence = scores[best_type] / total_weight if total_weight > 0 else 0.0

        return VehicleClassification(
            target_id=target_id,
            vehicle_type=best_type,
            confidence=min(1.0, confidence),
            yolo_class=yolo_class,
            speed_match=speed_match,
            wifi_device_count=wifi_count,
            size_estimate=size_estimate,
            signals_used=signals_used,
        )

    # ------------------------------------------------------------------
    # Route estimation
    # ------------------------------------------------------------------

    def estimate_route(self, target_id: str) -> Optional[RouteEstimate]:
        """Predict a vehicle's likely route from its trajectory and the road network.

        Uses the StreetGraph to project the vehicle's current heading and
        speed onto the road network, producing a list of waypoints the
        vehicle is likely to follow.

        Returns None if the street graph is unavailable or the vehicle has
        insufficient history.
        """
        with self._lock:
            return self._estimate_route_locked(target_id)

    def _estimate_route_locked(self, target_id: str) -> Optional[RouteEstimate]:
        """Internal route estimation — must hold self._lock."""
        # Get vehicle behavior data
        vb = self._vehicle_mgr.get_vehicle(target_id)
        if vb is None or len(vb.positions) < MIN_TRAIL_POINTS_FOR_ROUTE:
            return None

        # Current position, heading, and speed
        cur_x, cur_y, cur_ts = vb.positions[-1]
        heading_rad = math.radians(vb.heading)
        speed_mps = vb.speed_mph / 2.23694  # mph to m/s

        if speed_mps < 0.5:
            # Vehicle is effectively stationary — no route to predict
            return RouteEstimate(
                target_id=target_id,
                waypoints=[(cur_x, cur_y)],
                total_distance_m=0.0,
                estimated_time_s=0.0,
                heading_deg=vb.heading,
                avg_speed_mps=speed_mps,
                confidence=0.1,
            )

        # If we have a street graph, use it for road-network routing
        if self._street_graph is not None and self._street_graph.graph is not None:
            return self._route_via_street_graph(
                target_id, cur_x, cur_y, vb.heading, speed_mps,
            )

        # Fallback: linear projection along current heading
        return self._route_linear_projection(
            target_id, cur_x, cur_y, vb.heading, speed_mps,
        )

    def _route_via_street_graph(
        self,
        target_id: str,
        cur_x: float,
        cur_y: float,
        heading_deg: float,
        speed_mps: float,
    ) -> RouteEstimate:
        """Estimate route using StreetGraph shortest-path."""
        sg = self._street_graph

        # Project a destination point along current heading
        distance_ahead = speed_mps * ROUTE_LOOKAHEAD_S
        heading_rad = math.radians(heading_deg)
        dest_x = cur_x + math.sin(heading_rad) * distance_ahead
        dest_y = cur_y + math.cos(heading_rad) * distance_ahead

        # Find current road class
        cur_node, cur_dist = sg.nearest_node(cur_x, cur_y)
        road_class = "unknown"
        if cur_node is not None and sg.graph is not None:
            for _n1, _n2, data in sg.graph.edges(cur_node, data=True):
                road_class = data.get("road_class", "unknown")
                break

        # Attempt shortest path
        path = sg.shortest_path((cur_x, cur_y), (dest_x, dest_y))

        if path and len(path) >= 2:
            # Trim to MAX_ROUTE_WAYPOINTS
            waypoints = path[:MAX_ROUTE_WAYPOINTS]
            total_dist = sum(
                math.hypot(
                    waypoints[i + 1][0] - waypoints[i][0],
                    waypoints[i + 1][1] - waypoints[i][1],
                )
                for i in range(len(waypoints) - 1)
            )
            est_time = total_dist / speed_mps if speed_mps > 0 else 0.0
            confidence = min(1.0, 0.5 + 0.3 * (1.0 - cur_dist / 50.0))
            confidence = max(0.1, confidence)

            return RouteEstimate(
                target_id=target_id,
                waypoints=waypoints,
                total_distance_m=total_dist,
                estimated_time_s=est_time,
                current_road_class=road_class,
                heading_deg=heading_deg,
                avg_speed_mps=speed_mps,
                confidence=confidence,
            )

        # Shortest path failed — fall back to linear
        return self._route_linear_projection(
            target_id, cur_x, cur_y, heading_deg, speed_mps, road_class,
        )

    def _route_linear_projection(
        self,
        target_id: str,
        cur_x: float,
        cur_y: float,
        heading_deg: float,
        speed_mps: float,
        road_class: str = "unknown",
    ) -> RouteEstimate:
        """Simple linear projection along heading."""
        heading_rad = math.radians(heading_deg)
        waypoints: list[tuple[float, float]] = [(cur_x, cur_y)]

        step_time = ROUTE_LOOKAHEAD_S / 10.0
        total_dist = 0.0
        px, py = cur_x, cur_y
        for _ in range(10):
            step_dist = speed_mps * step_time
            nx = px + math.sin(heading_rad) * step_dist
            ny = py + math.cos(heading_rad) * step_dist
            total_dist += step_dist
            waypoints.append((nx, ny))
            px, py = nx, ny

        return RouteEstimate(
            target_id=target_id,
            waypoints=waypoints,
            total_distance_m=total_dist,
            estimated_time_s=ROUTE_LOOKAHEAD_S,
            current_road_class=road_class,
            heading_deg=heading_deg,
            avg_speed_mps=speed_mps,
            confidence=0.2,  # Linear projection is low confidence
        )

    # ------------------------------------------------------------------
    # Convoy detection
    # ------------------------------------------------------------------

    def detect_convoy(
        self, target_ids: Optional[list[str]] = None,
    ) -> list[ConvoyResult]:
        """Detect groups of vehicles traveling together.

        If a ConvoyDetector is available, delegates to it. Otherwise,
        runs a lightweight internal check comparing vehicle headings
        and speeds.

        Args:
            target_ids: Specific vehicle IDs to check. If None, checks
                all tracked vehicles.

        Returns:
            List of ConvoyResult instances.
        """
        with self._lock:
            return self._detect_convoy_locked(target_ids)

    def _detect_convoy_locked(
        self, target_ids: Optional[list[str]] = None,
    ) -> list[ConvoyResult]:
        """Internal convoy detection — must hold self._lock."""
        # If we have a full ConvoyDetector, delegate
        if self._convoy_detector is not None:
            raw_convoys = self._convoy_detector.analyze(target_ids)
            results = []
            for c in raw_convoys:
                members = c.get("member_target_ids", [])
                results.append(ConvoyResult(
                    convoy_id=c.get("convoy_id", ""),
                    member_ids=members,
                    avg_speed_mps=c.get("speed_avg_mps", 0.0),
                    avg_heading_deg=c.get("heading_avg_deg", 0.0),
                    spread_m=self._compute_spread(members),
                    duration_s=c.get("duration_s", 0.0),
                    confidence=c.get("suspicious_score", 0.0),
                ))
            return results

        # Lightweight internal convoy detection
        vehicles = self._vehicle_mgr.get_all()
        if target_ids is not None:
            id_set = set(target_ids)
            vehicles = [v for v in vehicles if v.target_id in id_set]

        moving = [v for v in vehicles if v.is_moving]
        if len(moving) < MIN_CONVOY_VEHICLES:
            return []

        # Build co-movement pairs
        groups: list[set[str]] = []
        paired: set[str] = set()

        for i in range(len(moving)):
            for j in range(i + 1, len(moving)):
                a, b = moving[i], moving[j]
                if not a.positions or not b.positions:
                    continue
                ax, ay, _ = a.positions[-1]
                bx, by, _ = b.positions[-1]
                dist = math.hypot(ax - bx, ay - by)
                if dist > 100.0:
                    continue

                # Heading similarity
                h_diff = abs(a.heading - b.heading)
                if h_diff > 180:
                    h_diff = 360 - h_diff
                if h_diff > 30.0:
                    continue

                # Speed similarity
                if abs(a.speed_mph - b.speed_mph) > 5.0:
                    continue

                # Found a co-moving pair — merge into groups
                merged = False
                for g in groups:
                    if a.target_id in g or b.target_id in g:
                        g.add(a.target_id)
                        g.add(b.target_id)
                        merged = True
                        break
                if not merged:
                    groups.append({a.target_id, b.target_id})

        # Build results from groups with >= MIN_CONVOY_VEHICLES members
        results = []
        for idx, group in enumerate(groups):
            if len(group) < MIN_CONVOY_VEHICLES:
                continue
            member_vehicles = [
                v for v in moving if v.target_id in group
            ]
            speeds = [v.speed_mph / 2.23694 for v in member_vehicles]
            headings = [v.heading for v in member_vehicles]
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
            avg_heading = self._circular_mean(headings)
            spread = self._compute_spread_from_vehicles(member_vehicles)

            results.append(ConvoyResult(
                convoy_id=f"vp_convoy_{idx}",
                member_ids=sorted(group),
                avg_speed_mps=avg_speed,
                avg_heading_deg=avg_heading,
                spread_m=spread,
                duration_s=0.0,
                confidence=0.5,
            ))

        return results

    # ------------------------------------------------------------------
    # Parking detection
    # ------------------------------------------------------------------

    def parking_detection(self, target_id: str) -> Optional[ParkingEvent]:
        """Detect whether a vehicle is currently parked.

        A vehicle is considered parked when its speed has been below
        PARKING_SPEED_MPH for at least PARKING_MIN_DURATION_S.

        Returns a ParkingEvent if the vehicle is parked, or None if
        it is moving.
        """
        with self._lock:
            return self._parking_detection_locked(target_id)

    def _parking_detection_locked(self, target_id: str) -> Optional[ParkingEvent]:
        """Internal parking detection — must hold self._lock."""
        vb = self._vehicle_mgr.get_vehicle(target_id)
        if vb is None:
            return None

        now = time.monotonic()

        if vb.speed_mph >= PARKING_SPEED_MPH:
            # Vehicle is moving — clear parking state
            self._parking_state.pop(target_id, None)
            return None

        # Vehicle is stopped — check or start parking timer
        if not vb.positions:
            return None

        cur_pos = (vb.positions[-1][0], vb.positions[-1][1])

        if target_id not in self._parking_state:
            self._parking_state[target_id] = (cur_pos, now)
            return None  # Just stopped — not parked yet

        park_pos, parked_since = self._parking_state[target_id]

        # Check if vehicle has drifted from the parking position
        drift = math.hypot(cur_pos[0] - park_pos[0], cur_pos[1] - park_pos[1])
        if drift > 5.0:
            # Drifted — reset parking state
            self._parking_state[target_id] = (cur_pos, now)
            return None

        duration = now - parked_since
        if duration < PARKING_MIN_DURATION_S:
            return None  # Not parked long enough

        is_confirmed = duration >= PARKING_CONFIRMED_DURATION_S

        # Find nearby WiFi devices
        nearby_wifi = self._find_nearby_wifi(cur_pos, WIFI_PROBE_RANGE_M)

        # Determine road class if street graph available
        road_class = "unknown"
        if self._street_graph is not None and self._street_graph.graph is not None:
            node_id, dist = self._street_graph.nearest_node(cur_pos[0], cur_pos[1])
            if node_id is not None:
                for _n1, _n2, data in self._street_graph.graph.edges(node_id, data=True):
                    road_class = data.get("road_class", "unknown")
                    break

        return ParkingEvent(
            target_id=target_id,
            position=cur_pos,
            parked_since=parked_since,
            duration_s=duration,
            is_confirmed=is_confirmed,
            nearby_wifi_devices=nearby_wifi,
            road_class=road_class,
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_vehicle_status(self, target_id: str) -> Optional[dict]:
        """Get comprehensive vehicle status combining all pipeline data."""
        with self._lock:
            vb = self._vehicle_mgr.get_vehicle(target_id)
            if vb is None:
                return None

            classification = self._classify_vehicle_locked(target_id)
            parking = self._parking_detection_locked(target_id)

            result = vb.to_dict()
            result["classification"] = classification.to_dict()
            result["plate"] = self._lpr_data.get(target_id, "")
            result["wifi_associations"] = sorted(
                self._vehicle_wifi.get(target_id, set())
            )
            if parking is not None:
                result["parking"] = parking.to_dict()
            return result

    def get_all_vehicles(self) -> list[dict]:
        """Get status for all tracked vehicles."""
        with self._lock:
            return [
                vb.to_dict() for vb in self._vehicle_mgr.get_all()
            ]

    def get_parked_vehicles(self) -> list[ParkingEvent]:
        """Get all currently parked vehicles."""
        with self._lock:
            results = []
            for vb in self._vehicle_mgr.get_all():
                event = self._parking_detection_locked(vb.target_id)
                if event is not None:
                    results.append(event)
            return results

    def get_plate(self, target_id: str) -> Optional[str]:
        """Get the license plate for a vehicle, if known."""
        with self._lock:
            return self._lpr_data.get(target_id)

    def get_summary(self) -> dict:
        """Get pipeline summary statistics."""
        with self._lock:
            mgr_summary = self._vehicle_mgr.get_summary()
            wifi_assoc_count = sum(
                len(macs) for macs in self._vehicle_wifi.values()
            )
            lpr_count = len(self._lpr_data)
            parked_count = len([
                tid for tid in self._parking_state
                if self._vehicle_mgr.get_vehicle(tid) is not None
            ])

            return {
                **mgr_summary,
                "wifi_associations": wifi_assoc_count,
                "lpr_plates": lpr_count,
                "parking_candidates": parked_count,
                "wifi_probes_buffered": sum(
                    len(probes) for probes in self._wifi_probes.values()
                ),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _associate_wifi_probes(
        self,
        target_id: str,
        veh_x: float,
        veh_y: float,
        veh_ts: float,
    ) -> None:
        """Associate nearby WiFi probes with a vehicle detection.

        Called after a camera detection is ingested. Scans the WiFi probe
        buffer for probes that are close in both space and time.
        """
        cutoff = veh_ts - WIFI_PROBE_TEMPORAL_WINDOW_S

        for mac, probes in self._wifi_probes.items():
            for probe in probes:
                if probe.timestamp < cutoff:
                    continue
                if probe.associated_vehicle_id is not None:
                    continue  # Already associated
                dist = math.hypot(
                    probe.position[0] - veh_x,
                    probe.position[1] - veh_y,
                )
                if dist <= WIFI_PROBE_RANGE_M:
                    probe.associated_vehicle_id = target_id
                    self._vehicle_wifi.setdefault(target_id, set()).add(mac)

    def _find_nearby_wifi(
        self,
        position: tuple[float, float],
        radius: float,
    ) -> list[str]:
        """Find WiFi MACs with recent probes near a position."""
        macs = set()
        now = time.monotonic()
        cutoff = now - WIFI_PROBE_TEMPORAL_WINDOW_S * 2

        for mac, probes in self._wifi_probes.items():
            for probe in probes:
                if probe.timestamp < cutoff:
                    continue
                dist = math.hypot(
                    probe.position[0] - position[0],
                    probe.position[1] - position[1],
                )
                if dist <= radius:
                    macs.add(mac)
                    break

        return sorted(macs)

    def _compute_spread(self, member_ids: list[str]) -> float:
        """Compute max distance between convoy members using their last position."""
        positions = []
        for tid in member_ids:
            vb = self._vehicle_mgr.get_vehicle(tid)
            if vb is not None and vb.positions:
                x, y, _ = vb.positions[-1]
                positions.append((x, y))

        if len(positions) < 2:
            return 0.0

        max_dist = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = math.hypot(
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                )
                max_dist = max(max_dist, d)
        return max_dist

    def _compute_spread_from_vehicles(
        self, vehicles: list,
    ) -> float:
        """Compute max distance between VehicleBehavior instances."""
        positions = []
        for v in vehicles:
            if v.positions:
                x, y, _ = v.positions[-1]
                positions.append((x, y))

        if len(positions) < 2:
            return 0.0

        max_dist = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = math.hypot(
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                )
                max_dist = max(max_dist, d)
        return max_dist

    @staticmethod
    def _circular_mean(angles_deg: list[float]) -> float:
        """Compute circular mean of angles in degrees."""
        if not angles_deg:
            return 0.0
        sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
        cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
        return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0

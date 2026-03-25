# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone intelligence pipeline demo — FastAPI app.

Showcases the full Tritium intelligence + tracking stack as an integrated
pipeline, proving the lib can do sensor fusion, anomaly detection, acoustic
classification, threat assessment, and pattern recognition without the
command center.

The pipeline loop:
    Raw sensor data -> TargetTracker (identity resolution)
                    -> TargetCorrelator (cross-sensor fusion)
                    -> AnomalyDetector (RF environment monitoring)
                    -> AcousticClassifier (sound events)
                    -> ThreatModel (threat assessment)
                    -> Position estimator (trilateration)

Run with:
    PYTHONPATH=src uvicorn tritium_lib.intelligence.demos.pipeline_demo:app --port 8090
    # or
    PYTHONPATH=src python3 -m tritium_lib.intelligence.demos.pipeline_demo
"""

from __future__ import annotations

import asyncio
import math
import random
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# --- Tracking subsystem ---
from tritium_lib.tracking import (
    TargetTracker,
    TargetCorrelator,
    GeofenceEngine,
    GeoZone,
    MovementPatternAnalyzer,
    HeatmapEngine,
    BLEClassifier,
)

# --- Intelligence subsystem ---
from tritium_lib.intelligence import (
    SimpleThresholdDetector,
    Anomaly,
    ThreatModel,
    ThreatSignal,
    ThreatLevel,
    estimate_from_multiple_anchors,
    estimate_from_single_anchor,
)
from tritium_lib.intelligence.acoustic_classifier import (
    AcousticClassifier,
    AcousticEvent,
    AcousticEventType,
    AudioFeatures,
)

# --- Models ---
from tritium_lib.models.position_anchor import (
    PositionAnchor,
    DetectionEdge,
)


# ---------------------------------------------------------------------------
# Minimal event bus (no external dependencies)
# ---------------------------------------------------------------------------
class _EventBus:
    """Simple thread-safe event bus for demo purposes."""

    def __init__(self) -> None:
        self._log: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def publish(self, topic: str, data: dict) -> None:
        with self._lock:
            self._log.append((topic, data))

    @property
    def events(self) -> list[tuple[str, dict]]:
        with self._lock:
            return list(self._log)

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._log)

    def recent(self, n: int = 20) -> list[dict]:
        with self._lock:
            return [
                {"topic": t, "keys": list(d.keys()), "preview": _preview(d)}
                for t, d in self._log[-n:]
            ]


def _preview(d: dict) -> str:
    """Short preview of event data."""
    parts = []
    for k, v in list(d.items())[:3]:
        parts.append(f"{k}={v!r}"[:40])
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Synthetic sensor data generators
# ---------------------------------------------------------------------------
# Base coordinates: downtown area (~40.7128, -74.0060 — NYC-ish)
BASE_LAT = 40.7128
BASE_LNG = -74.0060

# Simulated BLE MAC addresses
_BLE_MACS = [
    "AA:BB:CC:11:22:33", "AA:BB:CC:44:55:66", "AA:BB:CC:77:88:99",
    "DD:EE:FF:11:22:33", "DD:EE:FF:44:55:66", "DD:EE:FF:77:88:99",
]

# Simulated WiFi BSSIDs
_WIFI_BSSIDS = [
    "00:11:22:33:44:55", "00:11:22:66:77:88", "00:11:22:99:AA:BB",
    "CC:DD:EE:11:22:33",
]

# Anchor nodes (GPS-equipped sensors)
_ANCHORS = [
    PositionAnchor(anchor_id="sensor_north", lat=BASE_LAT + 0.001, lng=BASE_LNG, confidence=0.95),
    PositionAnchor(anchor_id="sensor_south", lat=BASE_LAT - 0.001, lng=BASE_LNG, confidence=0.90),
    PositionAnchor(anchor_id="sensor_east", lat=BASE_LAT, lng=BASE_LNG + 0.001, confidence=0.92),
    PositionAnchor(anchor_id="sensor_west", lat=BASE_LAT, lng=BASE_LNG - 0.001, confidence=0.88),
]


@dataclass
class _SyntheticTarget:
    """A synthetic entity moving through the sensor field."""

    entity_id: str
    kind: str  # "person", "vehicle", "phone"
    lat: float = BASE_LAT
    lng: float = BASE_LNG
    speed: float = 0.001  # degrees per tick (rough)
    heading: float = 0.0  # degrees
    ble_mac: str = ""
    wifi_bssid: str = ""
    is_threat: bool = False

    def tick(self, rng: random.Random) -> None:
        """Move the target one step."""
        rad = math.radians(self.heading)
        self.lat += math.cos(rad) * self.speed * (0.5 + rng.random())
        self.lng += math.sin(rad) * self.speed * (0.5 + rng.random())
        # Random heading drift
        self.heading += rng.gauss(0, 15)
        self.heading %= 360


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------
@dataclass
class PipelineState:
    """All mutable state for the intelligence pipeline demo."""

    event_bus: _EventBus = field(default_factory=_EventBus)
    tracker: TargetTracker | None = None
    correlator: TargetCorrelator | None = None
    anomaly_detector: SimpleThresholdDetector = field(
        default_factory=lambda: SimpleThresholdDetector(threshold_sigma=2.0, min_baseline_samples=10)
    )
    acoustic_classifier: AcousticClassifier = field(
        default_factory=lambda: AcousticClassifier(enable_ml=False)
    )
    threat_model: ThreatModel = field(default_factory=ThreatModel)
    geofence: GeofenceEngine | None = None
    movement_analyzer: MovementPatternAnalyzer = field(default_factory=MovementPatternAnalyzer)
    heatmap: HeatmapEngine | None = None
    ble_classifier: BLEClassifier = field(default_factory=BLEClassifier)

    # Synthetic entities
    entities: list[_SyntheticTarget] = field(default_factory=list)
    rng: random.Random = field(default_factory=lambda: random.Random(42))

    # Collected data
    anomaly_log: list[dict] = field(default_factory=list)
    acoustic_log: list[dict] = field(default_factory=list)
    position_estimates: list[dict] = field(default_factory=list)
    rf_baseline: list[dict[str, float]] = field(default_factory=list)

    # Pipeline metrics
    ticks: int = 0
    started_at: float = field(default_factory=time.time)
    running: bool = False


_state = PipelineState()


# ---------------------------------------------------------------------------
# Pipeline initialization
# ---------------------------------------------------------------------------
def _init_pipeline() -> None:
    """Initialize all pipeline components."""
    s = _state
    s.tracker = TargetTracker(event_bus=s.event_bus)
    s.correlator = TargetCorrelator(
        tracker=s.tracker,
        confidence_threshold=0.4,
    )
    s.geofence = GeofenceEngine(event_bus=s.event_bus)
    # GeoZone requires a polygon — approximate a circle with 8 vertices
    _r = 50.0  # meters
    _poly = [
        (_r * math.cos(math.radians(a)), _r * math.sin(math.radians(a)))
        for a in range(0, 360, 45)
    ]
    s.geofence.add_zone(GeoZone(
        zone_id="restricted_zone",
        name="Restricted Area",
        polygon=_poly,
    ))
    s.tracker.set_geofence_engine(s.geofence)
    s.heatmap = HeatmapEngine()

    # Create synthetic entities
    s.entities = [
        # People walking
        _SyntheticTarget(
            entity_id="person_1", kind="person",
            lat=BASE_LAT + 0.0005, lng=BASE_LNG - 0.0003,
            speed=0.00005, heading=45.0,
            ble_mac=_BLE_MACS[0],
        ),
        _SyntheticTarget(
            entity_id="person_2", kind="person",
            lat=BASE_LAT - 0.0002, lng=BASE_LNG + 0.0004,
            speed=0.00004, heading=180.0,
            ble_mac=_BLE_MACS[1],
        ),
        # Suspicious loiterer (threat)
        _SyntheticTarget(
            entity_id="suspect_1", kind="person",
            lat=BASE_LAT, lng=BASE_LNG,
            speed=0.00001, heading=0.0,
            ble_mac=_BLE_MACS[2],
            is_threat=True,
        ),
        # Vehicles
        _SyntheticTarget(
            entity_id="vehicle_1", kind="vehicle",
            lat=BASE_LAT + 0.001, lng=BASE_LNG - 0.001,
            speed=0.0003, heading=90.0,
            wifi_bssid=_WIFI_BSSIDS[0],
        ),
        _SyntheticTarget(
            entity_id="vehicle_2", kind="vehicle",
            lat=BASE_LAT - 0.0008, lng=BASE_LNG + 0.0005,
            speed=0.0002, heading=270.0,
            wifi_bssid=_WIFI_BSSIDS[1],
        ),
        # Phone (BLE + WiFi — cross-sensor correlation candidate)
        _SyntheticTarget(
            entity_id="phone_1", kind="phone",
            lat=BASE_LAT + 0.0003, lng=BASE_LNG + 0.0002,
            speed=0.00005, heading=120.0,
            ble_mac=_BLE_MACS[3],
            wifi_bssid=_WIFI_BSSIDS[2],
        ),
    ]


# ---------------------------------------------------------------------------
# Synthetic data injection per tick
# ---------------------------------------------------------------------------
def _generate_ble_sightings() -> None:
    """Feed BLE sightings into the tracker."""
    s = _state
    assert s.tracker is not None
    for entity in s.entities:
        if not entity.ble_mac:
            continue
        # Simulate RSSI based on distance to nearest anchor
        rssi = -50 - s.rng.randint(0, 30)
        # Convert lat/lng to local coords (simple offset for demo)
        x = (entity.lat - BASE_LAT) * 111_000  # rough meters
        y = (entity.lng - BASE_LNG) * 111_000 * math.cos(math.radians(BASE_LAT))
        s.tracker.update_from_ble({
            "mac": entity.ble_mac,
            "name": f"{entity.kind}_{entity.entity_id}",
            "rssi": rssi,
            "device_type": entity.kind,
            "position": {"x": x, "y": y},
            "classification": entity.kind,
            "classification_confidence": 0.8,
        })


def _generate_camera_detections() -> None:
    """Feed YOLO-style camera detections into the tracker."""
    s = _state
    assert s.tracker is not None
    for entity in s.entities:
        if entity.kind not in ("person", "vehicle"):
            continue
        # 70% chance of detection per tick (camera coverage gaps)
        if s.rng.random() > 0.7:
            continue
        x = (entity.lat - BASE_LAT) * 111_000
        y = (entity.lng - BASE_LNG) * 111_000 * math.cos(math.radians(BASE_LAT))
        class_name = "person" if entity.kind == "person" else "car"
        s.tracker.update_from_detection({
            "class_name": class_name,
            "confidence": 0.6 + s.rng.random() * 0.35,
            "center_x": x + s.rng.gauss(0, 2),  # add noise
            "center_y": y + s.rng.gauss(0, 2),
        })


def _generate_wifi_probes() -> None:
    """Feed WiFi probe sightings into the tracker (as BLE with wifi source)."""
    s = _state
    assert s.tracker is not None
    for entity in s.entities:
        if not entity.wifi_bssid:
            continue
        # 50% chance of WiFi probe per tick
        if s.rng.random() > 0.5:
            continue
        rssi = -55 - s.rng.randint(0, 25)
        x = (entity.lat - BASE_LAT) * 111_000
        y = (entity.lng - BASE_LNG) * 111_000 * math.cos(math.radians(BASE_LAT))
        # WiFi targets register as BLE with wifi-derived ID
        s.tracker.update_from_ble({
            "mac": entity.wifi_bssid,
            "name": f"wifi_{entity.entity_id}",
            "rssi": rssi,
            "device_type": "wifi_device",
            "position": {"x": x, "y": y},
            "classification": "wifi_device",
            "classification_confidence": 0.6,
        })


def _generate_acoustic_events() -> None:
    """Generate synthetic acoustic events and classify them."""
    s = _state
    # Random acoustic event every ~5 ticks
    if s.rng.random() > 0.2:
        return

    event_templates = [
        # Vehicle engine noise
        AudioFeatures(
            rms_energy=0.3 + s.rng.random() * 0.2,
            peak_amplitude=0.4,
            zero_crossing_rate=0.05,
            spectral_centroid=200 + s.rng.random() * 200,
            spectral_bandwidth=300,
            duration_ms=3000 + s.rng.randint(0, 5000),
        ),
        # Voice
        AudioFeatures(
            rms_energy=0.15 + s.rng.random() * 0.2,
            peak_amplitude=0.3,
            zero_crossing_rate=0.08,
            spectral_centroid=500 + s.rng.random() * 1500,
            spectral_bandwidth=800,
            duration_ms=500 + s.rng.randint(0, 2000),
        ),
        # Glass break (rare)
        AudioFeatures(
            rms_energy=0.85,
            peak_amplitude=0.9,
            zero_crossing_rate=0.3,
            spectral_centroid=5000 + s.rng.random() * 1000,
            spectral_bandwidth=2000,
            duration_ms=150,
        ),
        # Footsteps
        AudioFeatures(
            rms_energy=0.1,
            peak_amplitude=0.12,
            zero_crossing_rate=0.04,
            spectral_centroid=400,
            spectral_bandwidth=200,
            duration_ms=200,
        ),
    ]

    features = s.rng.choice(event_templates)
    event = s.acoustic_classifier.classify(features)
    s.acoustic_log.append({
        "event_type": event.event_type.value,
        "confidence": round(event.confidence, 3),
        "timestamp": event.timestamp,
        "duration_ms": event.duration_ms,
        "peak_frequency_hz": round(event.peak_frequency_hz, 1),
    })
    # Cap log size
    if len(s.acoustic_log) > 200:
        s.acoustic_log = s.acoustic_log[-200:]


def _run_anomaly_detection() -> None:
    """Build RF environment baseline and check for anomalies."""
    s = _state
    assert s.tracker is not None
    targets = s.tracker.get_all()
    ble_count = sum(1 for t in targets if t.source == "ble")
    yolo_count = sum(1 for t in targets if t.source == "yolo")
    avg_confidence = (
        sum(t.effective_confidence for t in targets) / len(targets)
        if targets else 0.0
    )

    current = {
        "ble_count": float(ble_count),
        "yolo_count": float(yolo_count),
        "total_targets": float(len(targets)),
        "avg_confidence": avg_confidence,
    }

    s.rf_baseline.append(current)
    # Keep baseline manageable
    if len(s.rf_baseline) > 500:
        s.rf_baseline = s.rf_baseline[-500:]

    if len(s.rf_baseline) >= 12:
        anomalies = s.anomaly_detector.detect(current, s.rf_baseline[:-1])
        for a in anomalies:
            entry = a.to_dict()
            entry["tick"] = s.ticks
            s.anomaly_log.append(entry)
        # Cap anomaly log
        if len(s.anomaly_log) > 200:
            s.anomaly_log = s.anomaly_log[-200:]


def _run_position_estimation() -> None:
    """Estimate positions of targets using multi-anchor trilateration."""
    s = _state
    assert s.tracker is not None
    targets = s.tracker.get_all()

    for target in targets:
        if target.source != "ble":
            continue
        # Simulate detections from 2-4 anchors
        n_anchors = s.rng.randint(1, len(_ANCHORS))
        chosen_anchors = s.rng.sample(_ANCHORS, n_anchors)
        detections = []
        for anchor in chosen_anchors:
            # Simulate RSSI based on rough distance
            dx = (target.position[0] / 111_000) - (anchor.lat - BASE_LAT)
            dy = (target.position[1] / (111_000 * math.cos(math.radians(BASE_LAT)))) - (anchor.lng - BASE_LNG)
            dist_deg = math.sqrt(dx * dx + dy * dy)
            dist_m = dist_deg * 111_000
            # Inverse: rssi ~ -40 - 25*log10(dist)
            rssi = -40 - 25 * math.log10(max(dist_m, 1.0))
            detections.append(DetectionEdge(
                detector_id=anchor.anchor_id,
                detected_id=target.target_id,
                detection_type="ble",
                rssi=rssi,
            ))

        if len(chosen_anchors) >= 2:
            estimate = estimate_from_multiple_anchors(chosen_anchors, detections)
        else:
            estimate = estimate_from_single_anchor(chosen_anchors[0], detections[0])

        if estimate:
            entry = {
                "target_id": estimate.target_id,
                "lat": estimate.lat,
                "lng": estimate.lng,
                "accuracy_m": estimate.accuracy_m,
                "method": estimate.method,
                "anchor_count": estimate.anchor_count,
                "confidence": estimate.confidence,
            }
            # Update or append
            found = False
            for i, existing in enumerate(s.position_estimates):
                if existing["target_id"] == estimate.target_id:
                    s.position_estimates[i] = entry
                    found = True
                    break
            if not found:
                s.position_estimates.append(entry)

    # Cap estimates list
    if len(s.position_estimates) > 100:
        s.position_estimates = s.position_estimates[-100:]


def _run_threat_assessment() -> None:
    """Assess threats for all tracked targets."""
    s = _state
    assert s.tracker is not None
    targets = s.tracker.get_all()

    for target in targets:
        # Behavioral signal: loiterers get higher threat
        speed = target.speed
        dwell_signal = 0.0
        if speed < 0.5 and target.signal_count > 5:
            dwell_signal = min(1.0, target.signal_count / 20.0)

        if dwell_signal > 0.1:
            s.threat_model.add_signal(ThreatSignal(
                signal_type="behavior",
                score=dwell_signal,
                source="loiter_detector",
                detail=f"Low speed ({speed:.1f}), {target.signal_count} sightings",
                target_id=target.target_id,
                ttl_seconds=120.0,
            ))

        # Classification signal: unknown devices are more suspicious
        if target.classification in ("unknown", "ble_device", "wifi_device"):
            s.threat_model.add_signal(ThreatSignal(
                signal_type="classification",
                score=0.3,
                source="device_classifier",
                detail=f"Unclassified device type: {target.classification}",
                target_id=target.target_id,
                ttl_seconds=60.0,
            ))

        # Zone violation signal: targets near restricted area center
        dist = math.sqrt(target.position[0] ** 2 + target.position[1] ** 2)
        if dist < 50.0:
            s.threat_model.add_signal(ThreatSignal(
                signal_type="zone_violation",
                score=min(1.0, (50.0 - dist) / 50.0),
                source="geofence_engine",
                detail=f"Within {dist:.0f}m of restricted zone",
                target_id=target.target_id,
                ttl_seconds=60.0,
            ))


def _run_correlation() -> None:
    """Run one correlation pass to merge cross-sensor detections."""
    s = _state
    assert s.correlator is not None
    s.correlator.correlate()


# ---------------------------------------------------------------------------
# Pipeline tick — one full cycle
# ---------------------------------------------------------------------------
def _pipeline_tick() -> None:
    """Execute one full pipeline cycle."""
    s = _state

    # 1. Move synthetic entities
    for entity in s.entities:
        entity.tick(s.rng)

    # 2. Generate sensor data
    _generate_ble_sightings()
    _generate_camera_detections()
    _generate_wifi_probes()

    # 3. Acoustic classification
    _generate_acoustic_events()

    # 4. Position estimation
    _run_position_estimation()

    # 5. Anomaly detection
    _run_anomaly_detection()

    # 6. Threat assessment
    _run_threat_assessment()

    # 7. Cross-sensor correlation
    _run_correlation()

    # 8. Heatmap recording
    if s.heatmap and s.tracker:
        for target in s.tracker.get_all():
            layer = "ble_activity" if target.source == "ble" else "camera_activity"
            s.heatmap.record_event(layer, target.position[0], target.position[1])

    s.ticks += 1


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------
async def _pipeline_loop() -> None:
    """Run the pipeline at ~2 Hz in the background."""
    _state.running = True
    while _state.running:
        try:
            _pipeline_tick()
        except Exception as exc:
            import traceback
            traceback.print_exc()
        await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start pipeline on startup, stop on shutdown."""
    _init_pipeline()
    task = asyncio.create_task(_pipeline_loop())
    yield
    _state.running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Tritium Intelligence Pipeline Demo",
    description=(
        "Standalone demo of the tritium-lib intelligence + tracking pipeline. "
        "Generates synthetic multi-modal sensor data and runs the full loop: "
        "BLE/WiFi/camera -> tracking -> correlation -> anomaly detection -> "
        "acoustic classification -> threat assessment -> position estimation."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)


@app.get("/pipeline/status")
def get_status() -> JSONResponse:
    """Pipeline health and summary metrics."""
    s = _state
    targets = s.tracker.get_all() if s.tracker else []
    source_counts: dict[str, int] = {}
    for t in targets:
        source_counts[t.source] = source_counts.get(t.source, 0) + 1

    threat_stats = s.threat_model.get_stats()

    return JSONResponse({
        "status": "running" if s.running else "stopped",
        "ticks": s.ticks,
        "uptime_seconds": round(time.time() - s.started_at, 1),
        "tick_rate_hz": 2.0,
        "targets": {
            "total": len(targets),
            "by_source": source_counts,
            "hostile": sum(1 for t in targets if t.alliance == "hostile"),
            "friendly": sum(1 for t in targets if t.alliance == "friendly"),
            "unknown": sum(1 for t in targets if t.alliance == "unknown"),
        },
        "sensors": {
            "ble_macs_tracked": len([t for t in targets if t.source == "ble"]),
            "camera_detections": len([t for t in targets if t.source == "yolo"]),
            "wifi_devices": len([t for t in targets if "wifi" in t.target_id]),
            "acoustic_events_total": len(s.acoustic_log),
            "anomalies_detected": len(s.anomaly_log),
        },
        "threat_model": threat_stats,
        "correlation": {
            "total_records": len(s.correlator.get_correlations()) if s.correlator else 0,
        },
        "position_estimates": len(s.position_estimates),
        "event_bus_count": s.event_bus.event_count,
        "synthetic_entities": len(s.entities),
    })


@app.get("/pipeline/targets")
def get_targets() -> JSONResponse:
    """All tracked targets with full details."""
    s = _state
    if not s.tracker:
        return JSONResponse({"targets": [], "count": 0})

    targets = s.tracker.get_all()
    result = []
    for t in targets:
        td = t.to_dict(history=s.tracker.history)
        # Enrich with threat assessment
        assessment = s.threat_model.assess(t.target_id)
        td["threat_assessment"] = assessment.to_dict()
        # Enrich with position estimate if available
        for pe in s.position_estimates:
            if pe["target_id"] == t.target_id:
                td["fused_position"] = pe
                break
        result.append(td)

    # Sort by threat score descending
    result.sort(key=lambda x: x["threat_assessment"]["composite_score"], reverse=True)

    return JSONResponse({
        "targets": result,
        "count": len(result),
        "tick": s.ticks,
    })


@app.get("/pipeline/anomalies")
def get_anomalies() -> JSONResponse:
    """All detected anomalies in the RF environment."""
    s = _state
    return JSONResponse({
        "anomalies": s.anomaly_log[-50:],
        "total_detected": len(s.anomaly_log),
        "baseline_samples": len(s.rf_baseline),
        "detector": s.anomaly_detector.name(),
        "tick": s.ticks,
    })


@app.get("/pipeline/acoustic")
def get_acoustic() -> JSONResponse:
    """Acoustic classification event log."""
    s = _state
    # Aggregate by event type
    type_counts: dict[str, int] = {}
    for event in s.acoustic_log:
        et = event["event_type"]
        type_counts[et] = type_counts.get(et, 0) + 1

    return JSONResponse({
        "events": s.acoustic_log[-50:],
        "total_events": len(s.acoustic_log),
        "by_type": type_counts,
        "ml_available": s.acoustic_classifier.ml_available,
        "tick": s.ticks,
    })


@app.get("/pipeline/threats")
def get_threats() -> JSONResponse:
    """Threat assessments for all tracked targets."""
    s = _state
    assessments = s.threat_model.assess_all()
    return JSONResponse({
        "assessments": [a.to_dict() for a in assessments],
        "count": len(assessments),
        "level_summary": {
            level.value: sum(1 for a in assessments if a.threat_level == level)
            for level in ThreatLevel
        },
        "stats": s.threat_model.get_stats(),
        "tick": s.ticks,
    })


@app.get("/pipeline/positions")
def get_positions() -> JSONResponse:
    """Position estimates from multi-anchor trilateration."""
    s = _state
    return JSONResponse({
        "estimates": s.position_estimates,
        "count": len(s.position_estimates),
        "anchors": [
            {
                "anchor_id": a.anchor_id,
                "lat": a.lat,
                "lng": a.lng,
                "confidence": a.confidence,
            }
            for a in _ANCHORS
        ],
        "tick": s.ticks,
    })


@app.get("/pipeline/events")
def get_events() -> JSONResponse:
    """Recent event bus activity."""
    s = _state
    return JSONResponse({
        "recent": s.event_bus.recent(30),
        "total_events": s.event_bus.event_count,
        "tick": s.ticks,
    })


@app.get("/pipeline/correlations")
def get_correlations() -> JSONResponse:
    """Cross-sensor correlation records."""
    s = _state
    if not s.correlator:
        return JSONResponse({"correlations": [], "count": 0})

    records = s.correlator.get_correlations()[-50:]
    result = []
    for r in records:
        result.append({
            "primary_id": r.primary_id,
            "secondary_id": r.secondary_id,
            "confidence": round(r.confidence, 3),
            "reason": r.reason,
            "dossier_uuid": r.dossier_uuid,
            "strategy_scores": [
                {"strategy": ss.strategy_name, "score": round(ss.score, 3), "detail": ss.detail}
                for ss in r.strategy_scores
            ],
        })

    return JSONResponse({
        "correlations": result,
        "count": len(result),
        "tick": s.ticks,
    })


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("TRITIUM INTELLIGENCE PIPELINE DEMO")
    print("=" * 60)
    print("Starting on http://localhost:8090")
    print("Endpoints:")
    print("  GET /pipeline/status      — pipeline health + metrics")
    print("  GET /pipeline/targets     — all tracked targets")
    print("  GET /pipeline/anomalies   — RF anomaly detections")
    print("  GET /pipeline/acoustic    — acoustic event classifications")
    print("  GET /pipeline/threats     — threat assessments")
    print("  GET /pipeline/positions   — trilateration estimates")
    print("  GET /pipeline/events      — event bus activity")
    print("  GET /pipeline/correlations— cross-sensor fusion records")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8090)

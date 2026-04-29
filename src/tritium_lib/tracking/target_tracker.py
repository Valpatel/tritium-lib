# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""TargetTracker — unified registry of all tracked entities in the battlespace.

Merges simulation targets (friendly rovers/drones) with real-world detections
(YOLO person/vehicle) into a single view Amy can reason about.

Architecture
------------
The tracker is a *read model* — a denormalised view of targets from two
independent sources:

  1. Simulation telemetry: SimulationEngine publishes ``sim_telemetry``
     events at 10 Hz.  Commander._sim_bridge_loop forwards these to
     update_from_simulation(), which upserts TrackedTarget entries.

  2. YOLO detections: Vision pipeline publishes ``detections`` events.
     The bridge loop forwards person/vehicle detections to
     update_from_detection(), which matches by class+proximity or creates
     new entries.  Stale YOLO detections are pruned after 30s.

Why double-tracking (engine + tracker)?
  The engine owns *simulation physics* — waypoints, tick, battery drain.
  The tracker owns *Amy's perception* — what she can reason about.  These
  are different concerns:
    - The engine has targets the tracker doesn't (e.g. neutral animals
      that haven't triggered a zone yet).
    - The tracker has targets the engine doesn't (YOLO detections of real
      people and vehicles).
    - Dispatch latency is one tick (~100ms) which is invisible to
      tactical decision-making.

TrackedTarget is a lightweight projection.  It does NOT carry waypoints
or tick state — that remains on SimulationTarget in the engine.

Threat classification is NOT in the tracker.  ThreatClassifier in
escalation.py runs its own 2Hz loop over tracker.get_all() and maintains
ThreatRecord separately.  The tracker only tracks *identity and position*.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

from .target_history import TargetHistory
from .target_reappearance import TargetReappearanceMonitor


# ---------------------------------------------------------------------------
# Confidence decay — exponential decay per source type
# ---------------------------------------------------------------------------
# half-life in seconds: after this time, confidence drops to 50%
#
# Calibration notes:
#   - rf_motion (10s):  motion events are intrinsically transient — by the
#                       time the half-life elapses the entity has likely
#                       moved out of the detection cell.
#   - yolo (15s):       a frame-by-frame detector; if the camera stops
#                       seeing the box, the track should fade quickly.
#   - acoustic (20s):   ESC-50 classifier (Wave 204, 47.4% accuracy) emits
#                       events for transient sounds (gunshot, glass break,
#                       barking, vehicle pass-by).  These are short-lived
#                       events, not sustained presence — sit between
#                       rf_motion (10s) and ble (30s) so a single sound
#                       hit fades faster than a sustained BLE beacon but
#                       slower than a single RF motion blip.
#   - ble (30s):        beacons typically advertise 1–10 Hz; 30s gives ~3+
#                       missed advertisements before halving.
#   - wifi (45s):       probe-request bursts are sparser and bursty.
#   - adsb (60s):       aircraft update at ~1 Hz but can lose signal in
#                       deadzones; longer half-life avoids flicker.
#   - mesh (120s):      LoRa nodes are stationary for long periods.
#   - simulation (0):   sentinel — simulation telemetry is ground truth
#                       and never decays.
#   - manual (300s):    operator-tagged targets get a generous 5min before
#                       the system starts to question them.
_HALF_LIVES: dict[str, float] = {
    "ble": 30.0,
    "wifi": 45.0,
    "yolo": 15.0,
    "rf_motion": 10.0,
    "acoustic": 20.0,     # transient sounds — gunshot/glass-break/voice
    "mesh": 120.0,
    "adsb": 60.0,         # aircraft update frequently but can lose signal
    "simulation": 0.0,    # never decays
    "manual": 300.0,
}
_MIN_CONFIDENCE = 0.05
_LN2 = math.log(2)

# Multi-source confidence boosting — multiplicative bonus per confirming source
_MULTI_SOURCE_BOOST = 1.3  # 30% boost per additional confirming source
_MAX_BOOSTED_CONFIDENCE = 0.99

# Velocity consistency — max plausible speed in meters/second
# 50 m/s ~ 180 km/h, anything above is suspicious
_MAX_PLAUSIBLE_SPEED_MPS = 50.0
_TELEPORT_FLAG_COOLDOWN = 30.0  # seconds before re-flagging same target


def _decayed_confidence(source: str, initial: float, elapsed: float) -> float:
    """Compute exponentially decayed confidence."""
    if elapsed <= 0.0:
        return max(0.0, min(1.0, initial))
    hl = _HALF_LIVES.get(source, 300.0)
    if hl <= 0.0:
        return max(0.0, min(1.0, initial))
    decayed = initial * math.exp(-_LN2 / hl * elapsed)
    return min(1.0, decayed) if decayed >= _MIN_CONFIDENCE else 0.0


# ---------------------------------------------------------------------------
# Shared DeviceClassifier — gap-fix B-7
# ---------------------------------------------------------------------------
# Loading the multi-signal classifier hits ~10 JSON databases (~1MB) on
# the first call.  We share a single instance across every TargetTracker
# so the cost is paid exactly once for the whole process.  A failure to
# load (e.g. missing data dir) is treated as "classifier disabled" — the
# tracker keeps working, classifications just stay at their incoming
# values.
_SHARED_BLE_CLASSIFIER: object | None = None
_SHARED_BLE_CLASSIFIER_LOAD_ATTEMPTED: bool = False


def _shared_ble_classifier():
    """Return a process-wide :class:`DeviceClassifier` or ``None``.

    Lazy import so that ``tritium_lib.tracking`` does not pull in the
    classifier package (and its JSON loaders) at module import time.
    """
    global _SHARED_BLE_CLASSIFIER, _SHARED_BLE_CLASSIFIER_LOAD_ATTEMPTED
    if _SHARED_BLE_CLASSIFIER is not None:
        return _SHARED_BLE_CLASSIFIER
    if _SHARED_BLE_CLASSIFIER_LOAD_ATTEMPTED:
        return None
    _SHARED_BLE_CLASSIFIER_LOAD_ATTEMPTED = True
    try:
        from tritium_lib.classifier import DeviceClassifier
        _SHARED_BLE_CLASSIFIER = DeviceClassifier()
    except Exception:
        _SHARED_BLE_CLASSIFIER = None
    return _SHARED_BLE_CLASSIFIER


@dataclass
class TrackedTarget:
    """A target Amy is aware of — real or virtual."""

    target_id: str
    name: str
    alliance: str  # "friendly", "hostile", "unknown"
    asset_type: str  # "rover", "drone", "turret", "person", "vehicle", etc.
    position: tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    speed: float = 0.0
    battery: float = 1.0
    last_seen: float = field(default_factory=time.monotonic)
    first_seen: float = field(default_factory=time.monotonic)
    signal_count: int = 0  # number of sightings/updates received
    source: str = "manual"  # "simulation", "yolo", "manual"
    status: str = "active"
    position_source: str = "unknown"  # "gps", "simulation", "mqtt", "fixed", "yolo", "unknown"
    position_confidence: float = 0.0  # 0.0 = no confidence, 1.0 = high
    threat_score: float = 0.0  # 0.0 = no threat, 1.0 = maximum threat probability
    _initial_confidence: float = 0.0  # stored at detection time for decay
    confirming_sources: set = field(default_factory=set)  # source types that confirmed this target
    correlated_ids: list = field(default_factory=list)  # IDs of targets fused into this one
    correlation_confidence: float = 0.0  # weighted correlation score from correlator
    velocity_suspicious: bool = False  # flagged if target teleported
    _last_velocity_flag: float = 0.0  # monotonic time of last velocity flag
    classification: str = "unknown"  # RL/ML classification (person, vehicle, phone, etc.)
    classification_confidence: float = 0.0  # confidence of the classification model
    # Structured kinematic / detection metadata.  Sources that report rich
    # state (radar range/bearing/speed, RF motion direction hints, etc.)
    # store it here instead of squeezing it into the discrete ``status``
    # field — ``status`` is reserved for lifecycle states ("active",
    # "eliminated", "destroyed", "despawned", "neutralized", "escaped",
    # "idle", "stationary", "arrived", "low_battery").  See Wave 200.
    kinematics: dict | None = None

    @property
    def effective_confidence(self) -> float:
        """Position confidence with exponential time decay and multi-source boost."""
        elapsed = time.monotonic() - self.last_seen
        initial = self._initial_confidence if self._initial_confidence > 0 else self.position_confidence
        decayed = _decayed_confidence(self.source, initial, elapsed)
        # Multi-source boost: each additional confirming source multiplies confidence
        extra_sources = max(0, len(self.confirming_sources) - 1)
        if extra_sources > 0:
            boosted = decayed * (_MULTI_SOURCE_BOOST ** extra_sources)
            return min(_MAX_BOOSTED_CONFIDENCE, boosted)
        return decayed

    def to_dict(self, history: TargetHistory | None = None, geo_converter=None) -> dict:
        """Serialize to dict.

        Args:
            history: Optional TargetHistory for trail data.
            geo_converter: Optional callable(x, y) -> {"lat", "lng", "alt"} for
                coordinate conversion. If None, tries tritium_lib.geo.local_to_latlng;
                falls back to zeros if geo is not initialized.
        """
        if geo_converter is not None:
            geo = geo_converter(self.position[0], self.position[1])
        else:
            try:
                from tritium_lib.geo import local_to_latlng
                geo = local_to_latlng(self.position[0], self.position[1])
            except Exception:
                geo = {"lat": 0.0, "lng": 0.0, "alt": 0.0}
        d = {
            "target_id": self.target_id,
            "name": self.name,
            "alliance": self.alliance,
            "asset_type": self.asset_type,
            "position": {"x": self.position[0], "y": self.position[1]},
            "lat": geo["lat"],
            "lng": geo["lng"],
            "alt": geo["alt"],
            "heading": self.heading,
            "speed": self.speed,
            "battery": self.battery,
            "last_seen": self.last_seen,
            "first_seen": self.first_seen,
            "signal_count": self.signal_count,
            "source": self.source,
            "status": self.status,
            "position_source": self.position_source,
            "position_confidence": self.effective_confidence,
            "threat_score": self.threat_score,
            "confirming_sources": list(self.confirming_sources),
            "sources": list(self.confirming_sources),
            "source_count": len(self.confirming_sources),
            "correlated_ids": list(self.correlated_ids),
            "correlation_confidence": self.correlation_confidence,
            "velocity_suspicious": self.velocity_suspicious,
            "classification": self.classification,
            "classification_confidence": self.classification_confidence,
            "kinematics": dict(self.kinematics) if self.kinematics else None,
        }
        if history is not None:
            d["trail"] = history.get_trail_dicts(self.target_id, max_points=20)
        return d


class TargetTracker:
    """Thread-safe registry of all tracked targets in the battlespace."""

    # Stale timeout — remove YOLO detections older than this
    STALE_TIMEOUT = 30.0

    def __init__(self, event_bus=None, ble_classifier=None) -> None:
        self._targets: dict[str, TrackedTarget] = {}
        self._lock = threading.Lock()
        self._detection_counter: int = 0
        self._event_bus = event_bus
        self._geofence_engine = None  # Set via set_geofence_engine()
        # Gap-fix B-7: optional injected DeviceClassifier.  If left None
        # we fall back to a process-wide shared instance loaded lazily by
        # ``_shared_ble_classifier``.  Tests that want determinism can
        # pass a ``DeviceClassifier()`` instance directly, or pass
        # ``False`` to disable BLE classification entirely.
        self._ble_classifier = ble_classifier
        # Wave 201: membership counter used as a cheap "tracker
        # version" for HTTP ETag/304 caching on /api/targets.  Bumps
        # on every ADD or REMOVE — NOT on per-target position/state
        # updates (those are streamed over WebSocket telemetry; the
        # /api/targets reconcile poll only cares about set membership).
        # This keeps 304 hit-rate high in steady state where positions
        # change frequently but the active target set is stable.
        # Read with no lock — Python int read/write is atomic at the
        # bytecode level.
        self._membership_count: int = 0
        self.history = TargetHistory()
        self.reappearance_monitor = TargetReappearanceMonitor(
            event_bus=event_bus,
            min_absence_seconds=60.0,
        )

    def set_geofence_engine(self, engine) -> None:
        """Wire geofence engine for automatic zone checks on position updates."""
        self._geofence_engine = engine

    def _get_ble_classifier(self):
        """Resolve the BLE classifier for this tracker instance.

        Returns ``None`` when classification has been explicitly disabled
        (``ble_classifier=False``) or when the shared classifier failed
        to load.
        """
        if self._ble_classifier is False:
            return None
        if self._ble_classifier is not None:
            return self._ble_classifier
        return _shared_ble_classifier()

    def _check_geofence(self, target_id: str, game_x: float, game_y: float) -> None:
        """Check if a target's position triggers geofence enter/exit events."""
        if not self._geofence_engine:
            return
        try:
            self._geofence_engine.check(target_id, (game_x, game_y))
        except Exception:
            pass  # Don't let geofence errors break target tracking

    def _check_velocity(self, target: TrackedTarget, new_pos: tuple[float, float]) -> None:
        """Check if position change implies impossible velocity (teleportation)."""
        now = time.monotonic()
        dt = now - target.last_seen
        if dt <= 0.0 or dt > 120.0:  # skip if first update or very stale
            return

        dx = new_pos[0] - target.position[0]
        dy = new_pos[1] - target.position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        speed = dist / dt

        if speed > _MAX_PLAUSIBLE_SPEED_MPS:
            if (now - target._last_velocity_flag) > _TELEPORT_FLAG_COOLDOWN:
                target.velocity_suspicious = True
                target._last_velocity_flag = now
        else:
            target.velocity_suspicious = False

    def _add_confirming_source(self, target: TrackedTarget, source: str) -> None:
        """Register an additional source that confirms this target's existence.

        Multi-source confirmation is only meaningful when the new source
        differs from the target's primary ``source``.  A YOLO update on a
        YOLO-source target is the same modality re-observing itself; it is
        not cross-modal confirmation and must not inflate
        ``confirming_sources``.

        ``"simulation"`` is rejected unconditionally — simulation telemetry
        is synthetic ground truth (a fake sensor used to drive the test
        harness), not a real sensor modality.  Counting it as a
        confirming source produces fake "multi-source" metrics that mask
        the absence of genuine cross-modal fusion.  See Gap-fix A
        (post-Wave 198) for the live-system measurement that flagged this
        as a 70% artifact in the fusion headline number.
        """
        if source == "simulation":
            return
        if source == target.source:
            return
        target.confirming_sources.add(source)

    def update_from_simulation(self, sim_data: dict) -> None:
        """Update or create a tracked target from simulation telemetry."""
        tid = sim_data["target_id"]
        pos = sim_data.get("position", {})
        position = (pos.get("x", 0.0), pos.get("y", 0.0))
        with self._lock:
            if tid in self._targets:
                t = self._targets[tid]
                self._check_velocity(t, position)
                t.position = position
                t.heading = sim_data.get("heading", 0.0)
                t.speed = sim_data.get("speed", 0.0)
                t.battery = sim_data.get("battery", 1.0)
                t.status = sim_data.get("status", "active")
                t.last_seen = time.monotonic()
                t.signal_count += 1
                self._add_confirming_source(t, "simulation")
            else:
                # New target — bump membership for ETag invalidation
                self._membership_count += 1
                self._targets[tid] = TrackedTarget(
                    target_id=tid,
                    name=sim_data.get("name", tid[:8]),
                    alliance=sim_data.get("alliance", "unknown"),
                    asset_type=sim_data.get("asset_type", "unknown"),
                    position=position,
                    heading=sim_data.get("heading", 0.0),
                    speed=sim_data.get("speed", 0.0),
                    battery=sim_data.get("battery", 1.0),
                    last_seen=time.monotonic(),
                    first_seen=time.monotonic(),
                    signal_count=1,
                    source="simulation",
                    status=sim_data.get("status", "active"),
                    position_source="simulation",
                    position_confidence=1.0,
                    _initial_confidence=1.0,
                    # Gap-fix A: simulation is synthetic ground truth, not a
                    # sensor modality.  Start with no confirming sources so
                    # the multi-source headline metric only counts genuine
                    # cross-modal observations (BLE + YOLO, mesh + ADS-B,
                    # etc.).
                    confirming_sources=set(),
                )
        self.history.record(tid, position)
        self._check_geofence(tid, position[0], position[1])

    YOLO_MAX_TRACK_SPEED = 30.0
    """Upper bound on plausible target speed (m/s) used to expand the YOLO
    match radius across detection gaps.  30 m/s ≈ 67 mph covers cars on
    surface streets; faster vehicles will spawn new IDs (and that's fine —
    they're a different track regime)."""

    def update_from_detection(self, detection: dict) -> None:
        """Update or create a tracked target from a YOLO detection.

        Match logic chooses the *closest* existing target within a
        motion-aware radius — not the first that fits.  The radius grows
        with the time since each candidate was last seen so that fast
        targets do not split into a new ID every frame, while still-recent
        ghosts don't get refreshed by a detection that's actually a new
        entity.
        """
        if detection.get("confidence", 0) < 0.4:
            return

        class_name = detection.get("class_name", "unknown")
        cx = detection.get("center_x", 0.0)
        cy = detection.get("center_y", 0.0)

        if class_name == "person":
            alliance = "hostile"
            asset_type = "person"
        elif class_name in ("car", "motorcycle", "bicycle"):
            alliance = "unknown"
            asset_type = "vehicle"
        else:
            alliance = "unknown"
            asset_type = class_name

        tid = f"det_{class_name}_{self._detection_counter}"

        now = time.monotonic()
        v_max = self.YOLO_MAX_TRACK_SPEED
        base_threshold_sq = 9.0 if (abs(cx) > 2.0 or abs(cy) > 2.0) else 0.04

        with self._lock:
            matched = None
            best_dist_sq = float("inf")
            for existing in self._targets.values():
                if existing.source != "yolo":
                    continue
                if existing.asset_type != asset_type:
                    continue
                dx = existing.position[0] - cx
                dy = existing.position[1] - cy
                dist_sq = dx * dx + dy * dy
                # Motion budget: a target moving at v_max for the elapsed
                # interval can have travelled up to (dt * v_max) meters.
                dt = max(0.0, now - existing.last_seen)
                motion_budget_sq = (dt * v_max) ** 2
                threshold = max(base_threshold_sq, motion_budget_sq)
                if dist_sq < threshold and dist_sq < best_dist_sq:
                    matched = existing
                    best_dist_sq = dist_sq

            if matched:
                self._check_velocity(matched, (cx, cy))
                matched.position = (cx, cy)
                matched.last_seen = now
                matched.signal_count += 1
                self._add_confirming_source(matched, "yolo")
                tid = matched.target_id
            else:
                self._detection_counter += 1
                self._membership_count += 1
                tid = f"det_{class_name}_{self._detection_counter}"
                self._targets[tid] = TrackedTarget(
                    target_id=tid,
                    name=f"{class_name.title()} #{self._detection_counter}",
                    alliance=alliance,
                    asset_type=asset_type,
                    position=(cx, cy),
                    last_seen=now,
                    first_seen=now,
                    signal_count=1,
                    source="yolo",
                    position_source="yolo",
                    position_confidence=0.1,
                    _initial_confidence=0.1,
                    confirming_sources={"yolo"},
                    classification=class_name,
                    classification_confidence=detection.get("confidence", 0.0),
                )
        self.history.record(tid, (cx, cy))

    def update_from_camera_detection(
        self,
        detection: dict,
        camera_lat: float,
        camera_lng: float,
        latlng_to_local_fn=None,
    ) -> None:
        """Update or create a target from a camera detection, positioned near the camera.

        Args:
            detection: Dict with keys: label/class_name, confidence, bbox.
            camera_lat: Camera latitude.
            camera_lng: Camera longitude.
            latlng_to_local_fn: Optional callable(lat, lng) -> (x, y, z).
                If None, tries to import from tritium_lib.geo.
        """
        if latlng_to_local_fn is None:
            try:
                from tritium_lib.geo import latlng_to_local
                latlng_to_local_fn = latlng_to_local
            except ImportError:
                return

        label = detection.get("label") or detection.get("class_name", "unknown")
        confidence = detection.get("confidence", 0.5)
        if confidence < 0.4:
            return

        cam_x, cam_y, _ = latlng_to_local_fn(camera_lat, camera_lng)

        bbox = detection.get("bbox", {})
        if isinstance(bbox, dict):
            px = bbox.get("x", 0.5)
            py = bbox.get("y", 0.5)
        else:
            px, py = 0.5, 0.5

        offset_x = (px - 0.5) * 60.0
        offset_y = (0.5 - py) * 30.0

        game_x = cam_x + offset_x
        game_y = cam_y + offset_y

        self.update_from_detection({
            "class_name": label,
            "confidence": confidence,
            "center_x": game_x,
            "center_y": game_y,
        })

    # BLE sightings have longer stale timeout — devices can be stationary
    BLE_STALE_TIMEOUT = 120.0

    def update_from_ble(self, sighting: dict) -> None:
        """Update or create a tracked target from a BLE sighting.

        Gap-fix B-7: when the sighting does not already carry a
        classification (the common case for raw scanner events), run the
        bundled multi-signal :class:`DeviceClassifier` over the available
        identity hints — MAC, advertised name, manufacturer/company ID,
        GAP appearance, service UUIDs, Apple continuity, Fast Pair model.
        Whatever the classifier produces is written back to
        ``classification`` / ``classification_confidence`` so downstream
        consumers see device-type metadata on every BLE target instead of
        the previous ``classification_confidence == 0.0`` for raw
        sightings.
        """
        mac = sighting.get("mac", "")
        if not mac:
            return

        tid = f"ble_{mac.replace(':', '').lower()}"
        name = sighting.get("name") or mac
        rssi = sighting.get("rssi", -100)
        asset_type = sighting.get("device_type") or "ble_device"
        confidence = max(0.0, min(1.0, (rssi + 100) / 70))

        # Pre-compute classification from sighting hints unless the caller
        # already did so.  We only run the classifier when at least one
        # identity-bearing field is present — bare MAC-only sightings hit
        # OUI lookup but skip the cost when the MAC is randomized.
        sighting_class = sighting.get("classification")
        sighting_class_conf = sighting.get("classification_confidence")
        derived_class = ""
        derived_class_conf = 0.0
        derived_manufacturer = ""
        if not sighting_class:
            classifier = self._get_ble_classifier()
            if classifier is not None:
                # Coerce hints into the shapes DeviceClassifier expects.
                cid = sighting.get("company_id")
                try:
                    cid_int = int(cid) if cid is not None else None
                except (TypeError, ValueError):
                    cid_int = None
                appearance = sighting.get("appearance")
                try:
                    appearance_int = int(appearance) if appearance is not None else None
                except (TypeError, ValueError):
                    appearance_int = None
                svc_uuids = sighting.get("service_uuids") or sighting.get("services")
                if svc_uuids and not isinstance(svc_uuids, list):
                    svc_uuids = [svc_uuids]
                try:
                    result = classifier.classify_ble(
                        mac=mac,
                        name=sighting.get("name") or "",
                        company_id=cid_int,
                        appearance=appearance_int,
                        service_uuids=svc_uuids if isinstance(svc_uuids, list) else None,
                        fast_pair_model_id=sighting.get("fast_pair_model_id"),
                        apple_device_class=sighting.get("apple_device_class"),
                    )
                    if result.device_type and result.device_type != "unknown":
                        derived_class = result.device_type
                        derived_class_conf = float(result.confidence or 0.0)
                    if result.manufacturer:
                        derived_manufacturer = result.manufacturer
                except Exception:
                    # Classifier must never break tracking — degrade silently.
                    pass

        pos = sighting.get("position")
        if pos:
            position = (float(pos.get("x", 0)), float(pos.get("y", 0)))
            pos_source = "trilateration"
        else:
            node_pos = sighting.get("node_position")
            if node_pos:
                position = (float(node_pos.get("x", 0)), float(node_pos.get("y", 0)))
                pos_source = "node_proximity"
            else:
                position = (0.0, 0.0)
                pos_source = "unknown"

        # Resolve the asset_type to use: prefer the explicit sighting field,
        # else upgrade the generic "ble_device" using the classifier hint.
        effective_asset_type = asset_type
        if asset_type == "ble_device" and derived_class:
            effective_asset_type = derived_class

        # Resolve final classification fields.
        if sighting_class:
            final_class = sighting_class
            final_class_conf = float(sighting_class_conf or 0.0)
        elif derived_class:
            final_class = derived_class
            final_class_conf = derived_class_conf
        else:
            final_class = asset_type
            final_class_conf = float(sighting_class_conf or 0.0)

        with self._lock:
            if tid in self._targets:
                t = self._targets[tid]
                if pos_source != "unknown":
                    self._check_velocity(t, position)
                    t.position = position
                    t.position_source = pos_source
                t.last_seen = time.monotonic()
                t.signal_count += 1
                t.position_confidence = confidence
                t._initial_confidence = confidence
                self._add_confirming_source(t, "ble")
                if effective_asset_type != "ble_device":
                    t.asset_type = effective_asset_type
                if sighting_class:
                    t.classification = sighting_class
                    t.classification_confidence = float(sighting_class_conf or 0.0)
                elif derived_class and (
                    t.classification in ("", "unknown", "ble_device")
                    or t.classification_confidence < derived_class_conf
                ):
                    # Only overwrite an existing classification if the new
                    # derivation is more confident — preserves any earlier
                    # high-confidence tag (e.g. an explicit upstream label).
                    t.classification = derived_class
                    t.classification_confidence = derived_class_conf
            else:
                self._membership_count += 1
                self._targets[tid] = TrackedTarget(
                    target_id=tid,
                    name=name,
                    alliance="unknown",
                    asset_type=effective_asset_type,
                    position=position,
                    last_seen=time.monotonic(),
                    first_seen=time.monotonic(),
                    signal_count=1,
                    source="ble",
                    position_source=pos_source,
                    position_confidence=confidence,
                    _initial_confidence=confidence,
                    confirming_sources={"ble"},
                    classification=final_class,
                    classification_confidence=final_class_conf,
                )
                self.reappearance_monitor.check_reappearance(
                    target_id=tid,
                    name=name,
                    source="ble",
                    asset_type=effective_asset_type,
                    position=position,
                )
        if pos_source != "unknown":
            self.history.record(tid, position)
            self._check_geofence(tid, position[0], position[1])

    # Mesh radio targets — nodes can be stationary for long periods
    MESH_STALE_TIMEOUT = 300.0

    def update_from_mesh(self, mesh_data: dict, latlng_to_local_fn=None) -> None:
        """Update or create a tracked target from a Meshtastic mesh node.

        Args:
            mesh_data: Dict with keys: target_id, name, lat, lng, alt, etc.
            latlng_to_local_fn: Optional callable(lat, lng, alt) -> (x, y, z).
                If None, tries to import from tritium_lib.geo.
        """
        tid = mesh_data.get("target_id", "")
        if not tid:
            return

        name = mesh_data.get("name", tid)
        battery = mesh_data.get("battery", 1.0)
        alliance = mesh_data.get("alliance", "friendly")
        asset_type = mesh_data.get("asset_type", "mesh_radio")

        lat = mesh_data.get("lat")
        lng = mesh_data.get("lng")
        alt = mesh_data.get("alt", 0.0)

        if lat is not None and lng is not None and (lat != 0.0 or lng != 0.0):
            if latlng_to_local_fn is None:
                try:
                    from tritium_lib.geo import latlng_to_local
                    latlng_to_local_fn = latlng_to_local
                except ImportError:
                    latlng_to_local_fn = None

            if latlng_to_local_fn is not None:
                try:
                    x, y, _z = latlng_to_local_fn(lat, lng, alt or 0.0)
                    position = (x, y)
                    pos_source = "gps"
                    confidence = 0.9
                except Exception:
                    position = (0.0, 0.0)
                    pos_source = "unknown"
                    confidence = 0.0
            else:
                position = (0.0, 0.0)
                pos_source = "unknown"
                confidence = 0.0
        elif mesh_data.get("position"):
            pos = mesh_data["position"]
            position = (float(pos.get("x", 0)), float(pos.get("y", 0)))
            pos_source = "gps"
            confidence = 0.9
        else:
            position = (0.0, 0.0)
            pos_source = "unknown"
            confidence = 0.0

        with self._lock:
            if tid in self._targets:
                t = self._targets[tid]
                if pos_source != "unknown":
                    self._check_velocity(t, position)
                    t.position = position
                    t.position_source = pos_source
                t.name = name
                t.battery = battery
                t.last_seen = time.monotonic()
                t.signal_count += 1
                t.position_confidence = confidence
                t._initial_confidence = confidence
                self._add_confirming_source(t, "mesh")
            else:
                self._membership_count += 1
                self._targets[tid] = TrackedTarget(
                    target_id=tid,
                    name=name,
                    alliance=alliance,
                    asset_type=asset_type,
                    position=position,
                    last_seen=time.monotonic(),
                    first_seen=time.monotonic(),
                    signal_count=1,
                    source="mesh",
                    battery=battery,
                    position_source=pos_source,
                    position_confidence=confidence,
                    _initial_confidence=confidence,
                    confirming_sources={"mesh"},
                    classification="mesh_radio",
                )
                self.reappearance_monitor.check_reappearance(
                    target_id=tid,
                    name=name,
                    source="mesh",
                    asset_type=asset_type,
                    position=position,
                )
        if pos_source != "unknown":
            self.history.record(tid, position)
            self._check_geofence(tid, position[0], position[1])

    # ADS-B aircraft targets
    ADSB_STALE_TIMEOUT = 120.0

    def update_from_adsb(self, adsb_data: dict, latlng_to_local_fn=None) -> None:
        """Update or create a tracked target from an ADS-B aircraft detection.

        Args:
            adsb_data: Dict with keys: target_id, name, lat, lng, alt, etc.
            latlng_to_local_fn: Optional callable(lat, lng, alt) -> (x, y, z).
        """
        tid = adsb_data.get("target_id", "")
        if not tid:
            return

        name = adsb_data.get("name", tid)
        lat = adsb_data.get("lat")
        lng = adsb_data.get("lng")
        alt = adsb_data.get("alt", 0.0)
        heading = adsb_data.get("heading", 0.0)
        speed = adsb_data.get("speed", 0.0)

        if lat is not None and lng is not None and (lat != 0.0 or lng != 0.0):
            if latlng_to_local_fn is None:
                try:
                    from tritium_lib.geo import latlng_to_local
                    latlng_to_local_fn = latlng_to_local
                except ImportError:
                    latlng_to_local_fn = None

            if latlng_to_local_fn is not None:
                try:
                    x, y, _z = latlng_to_local_fn(lat, lng, alt or 0.0)
                    position = (x, y)
                    pos_source = "adsb"
                    confidence = 0.95
                except Exception:
                    position = (0.0, 0.0)
                    pos_source = "unknown"
                    confidence = 0.0
            else:
                position = (0.0, 0.0)
                pos_source = "unknown"
                confidence = 0.0
        else:
            position = (0.0, 0.0)
            pos_source = "unknown"
            confidence = 0.0

        with self._lock:
            if tid in self._targets:
                t = self._targets[tid]
                if pos_source != "unknown":
                    self._check_velocity(t, position)
                    t.position = position
                    t.position_source = pos_source
                t.name = name
                t.heading = heading
                t.speed = speed
                t.last_seen = time.monotonic()
                t.signal_count += 1
                t.position_confidence = confidence
                t._initial_confidence = confidence
                self._add_confirming_source(t, "adsb")
            else:
                self._membership_count += 1
                self._targets[tid] = TrackedTarget(
                    target_id=tid,
                    name=name,
                    alliance="unknown",
                    asset_type="aircraft",
                    position=position,
                    heading=heading,
                    speed=speed,
                    last_seen=time.monotonic(),
                    first_seen=time.monotonic(),
                    signal_count=1,
                    source="adsb",
                    position_source=pos_source,
                    position_confidence=confidence,
                    _initial_confidence=confidence,
                    confirming_sources={"adsb"},
                    classification="aircraft",
                )

        if pos_source != "unknown":
            self.history.record(tid, position)
            self._check_geofence(tid, position[0], position[1])

    # RF motion targets have shorter stale timeout
    RF_MOTION_STALE_TIMEOUT = 30.0

    def update_from_rf_motion(self, motion: dict) -> None:
        """Update or create a tracked target from an RF motion event.

        Rejects events with position (0, 0) — that indicates the detecting
        sensor has no known location and placing a target at the map origin
        is misleading.

        Also rejects NaN/Inf values, which can poison the tracker and slip
        past the (0, 0) check (NaN compares False to everything).  See
        Wave 200 security audit.
        """
        import math

        tid = motion.get("target_id", "")
        if not tid:
            return

        position = motion.get("position", (0.0, 0.0))
        if isinstance(position, dict):
            try:
                position = (float(position.get("x", 0)), float(position.get("y", 0)))
            except (TypeError, ValueError):
                return
        else:
            # Coerce tuple/list elements defensively
            try:
                position = (float(position[0]), float(position[1]))
            except (TypeError, ValueError, IndexError):
                return

        # Reject NaN / Inf — these slip past the (0, 0) check because NaN
        # compares False to everything and Inf is a finite-but-absurd value.
        # An unsanitized RF motion event with NaN coords would propagate
        # through arithmetic and corrupt every downstream consumer.
        if not (math.isfinite(position[0]) and math.isfinite(position[1])):
            return

        # Reject targets at (0, 0) — this means no real position data is
        # available from the detecting sensor.  Creating targets here would
        # place them at the map origin / Gulf of Guinea which is wrong.
        if position == (0.0, 0.0):
            return

        confidence = float(motion.get("confidence", 0.5))
        direction = motion.get("direction_hint", "unknown")
        pair_id = motion.get("pair_id", "")

        with self._lock:
            if tid in self._targets:
                t = self._targets[tid]
                self._check_velocity(t, position)
                t.position = position
                t.position_confidence = confidence
                t._initial_confidence = confidence
                t.last_seen = time.monotonic()
                t.signal_count += 1
                # Wave 200: don't poison the discrete ``status`` field with
                # the direction hint — store it in ``kinematics``.
                kinematics = dict(t.kinematics) if t.kinematics else {}
                kinematics["direction_hint"] = direction
                if pair_id:
                    kinematics["pair_id"] = pair_id
                t.kinematics = kinematics
                self._add_confirming_source(t, "rf_motion")
            else:
                self._membership_count += 1
                self._targets[tid] = TrackedTarget(
                    target_id=tid,
                    name=f"RF Motion ({pair_id})",
                    alliance="unknown",
                    asset_type="motion_detected",
                    position=position,
                    last_seen=time.monotonic(),
                    first_seen=time.monotonic(),
                    signal_count=1,
                    source="rf_motion",
                    position_source="rf_pair_midpoint",
                    position_confidence=confidence,
                    _initial_confidence=confidence,
                    status="active",
                    kinematics={
                        "direction_hint": direction,
                        **({"pair_id": pair_id} if pair_id else {}),
                    },
                    confirming_sources={"rf_motion"},
                )
        self.history.record(tid, position)

    def get_all(self) -> list[TrackedTarget]:
        """Return all tracked targets (pruning stale detections)."""
        self._prune_stale()
        with self._lock:
            return list(self._targets.values())

    @property
    def version(self) -> int:
        """Monotonic membership counter — bumps when targets are added or
        removed, but **not** on per-target position/state updates.

        Used by /api/targets ETag/304 to short-circuit unchanged polls
        (Wave 201).  The reconciliation poll only cares about set
        membership: positions/state stream over WebSocket telemetry and
        do not require the heavyweight 158 KB list refresh.
        """
        return self._membership_count

    def snapshot(self) -> tuple[list[TrackedTarget], int]:
        """Atomically read targets and version under one lock acquisition.

        Returns ``(targets_list, membership_count)``.  The list is a
        fresh copy — safe to iterate without holding the lock.  The
        membership count is the value at the moment the snapshot was
        taken; callers can compare against a previously-stored version
        to skip work when the active set has not changed.

        Calls :meth:`_prune_stale` first so the returned snapshot
        reflects the same active set that ``get_all()`` would return.
        """
        self._prune_stale()
        with self._lock:
            return list(self._targets.values()), self._membership_count

    def get_hostiles(self) -> list[TrackedTarget]:
        """Return only hostile targets."""
        return [t for t in self.get_all() if t.alliance == "hostile"]

    def get_friendlies(self) -> list[TrackedTarget]:
        """Return only friendly targets."""
        return [t for t in self.get_all() if t.alliance == "friendly"]

    def get_target(self, target_id: str) -> TrackedTarget | None:
        """Get a specific target by ID."""
        with self._lock:
            return self._targets.get(target_id)

    def remove(self, target_id: str) -> bool:
        """Remove a target from tracking."""
        with self._lock:
            removed = self._targets.pop(target_id, None) is not None
            if removed:
                self._membership_count += 1
            return removed

    def summary(self) -> str:
        """Battlespace summary for reasoning context."""
        all_targets = self.get_all()
        if not all_targets:
            return ""
        friendlies = [t for t in all_targets if t.alliance == "friendly"]
        hostiles = [t for t in all_targets if t.alliance == "hostile"]
        unknowns = [t for t in all_targets if t.alliance == "unknown"]

        parts = []
        if friendlies:
            parts.append(f"{len(friendlies)} friendly")
        if hostiles:
            parts.append(f"{len(hostiles)} hostile")
        if unknowns:
            parts.append(f"{len(unknowns)} unknown")

        result = f"BATTLESPACE: {', '.join(parts)} target(s) tracked"

        import math
        alerts = []
        _max_proximity_checks = 200
        _h_sample = hostiles[:_max_proximity_checks]
        _f_sample = friendlies[:_max_proximity_checks]
        for h in _h_sample:
            for f in _f_sample:
                dx = h.position[0] - f.position[0]
                dy = h.position[1] - f.position[1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 25.0:
                    dist = math.sqrt(dist_sq)
                    alerts.append(f"ALERT: {h.name} within {dist:.1f} units of {f.name}")
                    if len(alerts) >= 3:
                        break
            if len(alerts) >= 3:
                break
        if alerts:
            result += "\n" + "\n".join(alerts[:3])

        if hostiles:
            sectors: dict[str, list[str]] = {}
            for h in hostiles:
                sx = "E" if h.position[0] > 5 else ("W" if h.position[0] < -5 else "")
                sy = "N" if h.position[1] > 5 else ("S" if h.position[1] < -5 else "")
                sector = (sy + sx) or "center"
                sectors.setdefault(sector, []).append(h.name)
            sector_parts = [f"{len(names)} in {s}" for s, names in sectors.items()]
            result += f"\nHostile sectors: {', '.join(sector_parts)}"

        return result

    SIM_STALE_TIMEOUT = 10.0

    def _prune_stale(self) -> None:
        """Remove targets that haven't been updated recently."""
        now = time.monotonic()
        with self._lock:
            stale = [
                tid for tid, t in self._targets.items()
                if (t.source == "yolo" and (now - t.last_seen) > self.STALE_TIMEOUT)
                or (t.source == "simulation" and (now - t.last_seen) > self.SIM_STALE_TIMEOUT)
                or (t.source == "ble" and (now - t.last_seen) > self.BLE_STALE_TIMEOUT)
                or (t.source == "rf_motion" and (now - t.last_seen) > self.RF_MOTION_STALE_TIMEOUT)
                or (t.source == "mesh" and (now - t.last_seen) > self.MESH_STALE_TIMEOUT)
                or (t.source == "adsb" and (now - t.last_seen) > self.ADSB_STALE_TIMEOUT)
            ]
            if stale:
                self._membership_count += 1
            for tid in stale:
                t = self._targets[tid]
                self.reappearance_monitor.record_departure(
                    target_id=tid,
                    name=t.name,
                    source=t.source,
                    asset_type=t.asset_type,
                    last_position=t.position,
                )
                del self._targets[tid]
                self.history.clear(tid)

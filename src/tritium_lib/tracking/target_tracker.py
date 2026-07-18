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

import logging
import math
import threading
import time
from dataclasses import dataclass, field

from .target_history import TargetHistory
from .target_reappearance import TargetReappearanceMonitor
from .integrity import (
    innovation_mahalanobis_sq,
    update_velocity_ewma,
    is_spoofed,
    spoof_score,
)

logger = logging.getLogger("tritium.tracking.target_tracker")


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
    "camera": 15.0,   # frame detections decay like any vision contact
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

# Velocity consistency. The RAIM innovation gate (tracking.integrity) handles
# normal motion checks; _MAX_PLAUSIBLE_SPEED_MPS is retained as a reference for a
# typical ground vehicle (~180 km/h). _TELEPORT_SPEED_MPS is the absolute backstop:
# a single-step speed above it is a teleport/spoof regardless of motion history, and
# is set well above any plausible platform (fast aircraft ~300 m/s) so legitimate
# fast movers (e.g. ADS-B aircraft) are not flagged.
_MAX_PLAUSIBLE_SPEED_MPS = 50.0
_TELEPORT_SPEED_MPS = 600.0
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


# ---------------------------------------------------------------------------
# Alliance authority
# ---------------------------------------------------------------------------
# The canonical alliance vocabulary.  The tracker is the ONE authority for a
# target's *effective* alliance; every consumer (live map WS batches, REST
# /api/targets*, CoT/TAK affiliation, fusion) reads it from here.  Values a
# writer proposes are validated against this set — junk never clobbers a
# known alliance.
VALID_ALLIANCES = frozenset({"friendly", "hostile", "neutral", "unknown", "vip"})

# Precedence order for alliance writes (highest wins, 2026-07-11 ruling):
#   1. "operator"  — an explicit human tag (set_operator_alliance).  Pinned:
#                    telemetry can never clobber a human's decision.
#   2. "auto"      — declared telemetry (a frame that carries an "alliance"
#                    key: a red-team robot re-declaring itself, an NPC
#                    turning hostile).  Applied only while not pinned.
#   3. creation default — whatever the ingest path stamped at first sight
#                    (sim default / _resolve_alliance fallback); simply the
#                    value that holds until tier 1 or 2 writes.


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
    source: str = "manual"  # "simulation", "yolo", "camera", "manual", ...
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
    spoof_score: float = 0.0  # 0..1 RAIM innovation-gate plausibility-of-spoof
    _vx: float = 0.0  # constant-velocity estimate (m/s) for the integrity gate
    _vy: float = 0.0
    _v_samples: int = 0  # velocity deltas observed (two-point init at the first)
    classification: str = "unknown"  # RL/ML classification (person, vehicle, phone, etc.)
    classification_confidence: float = 0.0  # confidence of the classification model
    # Civil-unrest crowd sub-classification — finer grain than ``alliance``.
    # "civilian" (protected), "instigator" (agitator), "rioter" (active).
    # None for non-crowd entities (rovers, drones, vehicles, etc.).  Threaded
    # from SimulationTarget.crowd_role through update_from_simulation so the
    # tactical map / ops surface can render a protected civilian distinctly
    # from an agitator.  See Wave 213.
    crowd_role: str | None = None
    # Hit-feedback contract (tritium_lib.models.hits): combat hitpoints for
    # entities that REPORT health — sim combatants and wire robots whose
    # telemetry carries a ``health`` block (a robot dog's HealthTracker).
    # ``None`` means "this target does not report health" (most sensors),
    # NOT "zero hp" — renderers must skip the HP bar, never draw it empty.
    # Flat floats here (the tactical surface convention: health/max_health),
    # distilled from the wire block's hp/max_hp by the ingest bridge.
    health: float | None = None
    max_health: float | None = None
    # Who last wrote ``alliance`` — the alliance-authority precedence tier
    # (see module docstring above the class).  "auto" = the tracker follows
    # declared telemetry; "operator" = a human tagged this target and the
    # value is PINNED (update_from_simulation and every other telemetry
    # ingest must leave ``alliance`` alone until re-tagged).
    alliance_source: str = "auto"  # "auto" | "operator"
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
            "alliance_source": self.alliance_source,
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
            "spoof_score": round(self.spoof_score, 3),
            "classification": self.classification,
            "classification_confidence": self.classification_confidence,
            "crowd_role": self.crowd_role,
            "health": self.health,
            "max_health": self.max_health,
            "kinematics": dict(self.kinematics) if self.kinematics else None,
        }
        if history is not None:
            d["trail"] = history.get_trail_dicts(self.target_id, max_points=20)
        return d


class TargetTracker:
    """Thread-safe registry of all tracked targets in the battlespace.

    Args:
        event_bus: Optional EventBus for reappearance + capacity alarms.
        ble_classifier: Optional DeviceClassifier (``False`` disables).
        max_targets: Optional hard cap on tracker membership — the LAST
            line of defense against runaway track minting (see the
            2026-07 feedback cascade: a consumer re-ingesting its own
            republished detections grew 0 -> 1,279 tracks in 3.5 min).
            ``None`` (the default) preserves the historical unbounded
            behavior.  When set, *sensor-derived* ingest paths (vision,
            BLE, mesh, acoustic, ADS-B, RF motion, lidar) refuse to mint
            NEW tracks once ``len(targets) >= max_targets`` — refusals
            are counted in :attr:`cap_rejections`, logged at WARNING,
            and published to the event bus as ``tracker.capacity``.
            Updates to EXISTING tracks always proceed at cap.  The
            operator's own fleet (``update_from_simulation`` /
            ``update_from_robot_pose``) is exempt: those populations are
            bounded by the engine/fleet itself, and a full tracker must
            never blind the operator to their own assets.
    """

    # Stale timeout — remove vision detections (yolo/camera) older than this
    STALE_TIMEOUT = 30.0

    # Upper bound on stored detection-key aliases (stable-identity path).
    # Aliases map caller-supplied keys -> det_* track ids; entries pointing
    # at pruned tracks are purged with the tracks, and the map is LRU-capped
    # so a caller cycling through unbounded distinct keys cannot leak memory
    # here.  Evicting an old alias is safe: the worst case is that a key not
    # seen for 4096+ other keys re-mints one fresh track.
    MAX_DETECTION_KEY_ALIASES = 4096

    # Minimum seconds between capacity alarms (log + bus event).  The cap
    # must be LOUD but not a log flood at 10 Hz ingest.
    CAP_ALARM_INTERVAL = 10.0

    def __init__(
        self,
        event_bus=None,
        ble_classifier=None,
        max_targets: int | None = None,
    ) -> None:
        self._targets: dict[str, TrackedTarget] = {}
        self._lock = threading.Lock()
        self._detection_counter: int = 0
        self._event_bus = event_bus
        # Stable-identity aliases: caller-supplied detection key -> det_*
        # track id.  See update_from_detection's ``detection_key`` contract.
        self._detection_keys: dict[str, str] = {}
        # Hard-cap state (defense-in-depth; see class docstring).
        self.max_targets = max_targets
        self.cap_rejections: int = 0
        self._cap_alarm_pending: dict | None = None
        self._last_cap_alarm: float | None = None
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
        """RAIM-style integrity gate: flag fixes that defy the motion model.

        Replaces a fixed max-speed threshold (which flagged every legitimately
        fast, steady mover — e.g. an ADS-B aircraft at 200 m/s — and missed slow
        drift spoofs) with an innovation test: predict the position from the
        target's constant-velocity estimate and gate the normalized innovation
        against a chi-square threshold. A steady fast mover stays in-gate; a
        teleport / GPS-spoof jump trips the gate. Sets ``spoof_score`` (0..1) for
        downstream consumers (HUD, anomaly engine, fusion confidence).
        """
        now = time.monotonic()
        dt = now - target.last_seen
        if dt <= 0.0 or dt > 120.0:  # skip if first update or very stale
            return

        # Absolute teleport backstop: a single-step speed faster than any plausible
        # platform (well above even a fast aircraft ~300 m/s) is a teleport / GPS
        # spoof. Flag it even before a velocity baseline exists -- the innovation
        # gate below needs >=2 samples, so a teleport on a target's FIRST move would
        # otherwise slip through warmup unflagged.
        dx = new_pos[0] - target.position[0]
        dy = new_pos[1] - target.position[1]
        speed = math.hypot(dx, dy) / dt
        teleport = speed > _TELEPORT_SPEED_MPS

        m2 = innovation_mahalanobis_sq(
            target.position, new_pos, (target._vx, target._vy), dt
        )
        target.spoof_score = max(spoof_score(m2), 1.0 if teleport else 0.0)

        if teleport or is_spoofed(m2, target._v_samples):
            if (now - target._last_velocity_flag) > _TELEPORT_FLAG_COOLDOWN:
                target.velocity_suspicious = True
                target._last_velocity_flag = now
        else:
            target.velocity_suspicious = False

        # Update the constant-velocity model. Two-point init (alpha=1) on the
        # first delta so a freshly-acquired steady mover gets the right model
        # immediately, EWMA-smoothed thereafter.
        alpha = 1.0 if target._v_samples == 0 else 0.4
        target._vx, target._vy = update_velocity_ewma(
            (target._vx, target._vy), target.position, new_pos, dt, alpha=alpha
        )
        target._v_samples += 1

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
        # "camera" (posed frame detection) and "yolo" (bare vision detection)
        # are the SAME modality — vision — differing only in provenance.  A
        # camera detection refreshing a yolo track (or vice versa) is not
        # cross-modal confirmation; counting it would inflate the fusion
        # headline exactly like the simulation artifact above.  Camera
        # provenance is recorded in ``kinematics`` instead.
        if source in self.VISION_SOURCES and target.source in self.VISION_SOURCES:
            return
        target.confirming_sources.add(source)

    # ------------------------------------------------------------------
    # Stable detection identity (the republish-echo hardening, 2026-07)
    # ------------------------------------------------------------------

    def _resolve_keyed_track_locked(self, key: str) -> TrackedTarget | None:
        """Resolve a caller-supplied detection key to a live track.

        Must be called with ``self._lock`` held.  Resolution order:

        1. The alias map (``key`` was seen before and minted/claimed a
           ``det_*`` track that is still alive).  A mapping whose track
           has since been pruned is dropped — a dead track is never
           resurrected through its key.
        2. The key IS one of our own live *vision* track ids — the
           republish-echo shape: a consumer feeding the tracker's own
           output back stamps ``source_track_id=target.target_id``.
           Matching it directly is what makes the echo loop PLATEAU
           instead of minting a new generation of tracks per pass.
           Non-vision tracks (``ble_*``, ``mesh_*``, ...) are NOT
           claimed this way — updating them through the vision path
           would corrupt their source semantics; such a key instead
           mints one vision track and aliases to it thereafter.
        """
        mapped = self._detection_keys.get(key)
        if mapped is not None:
            t = self._targets.get(mapped)
            if t is not None:
                return t
            del self._detection_keys[key]
        t = self._targets.get(key)
        if t is not None and t.source in self.VISION_SOURCES:
            return t
        return None

    def _record_detection_key_locked(self, key: str, tid: str) -> None:
        """Remember ``key -> tid``.  Must be called with the lock held.

        Self-aliases (``key == tid``) are skipped — the direct-id branch
        of :meth:`_resolve_keyed_track_locked` already covers them.  The
        map is LRU-capped at :attr:`MAX_DETECTION_KEY_ALIASES`.
        """
        if key == tid:
            return
        self._detection_keys.pop(key, None)
        self._detection_keys[key] = tid
        while len(self._detection_keys) > self.MAX_DETECTION_KEY_ALIASES:
            self._detection_keys.pop(next(iter(self._detection_keys)))

    def _purge_detection_keys_locked(self) -> None:
        """Drop aliases whose target no longer exists.  Lock held."""
        if self._detection_keys:
            self._detection_keys = {
                k: v for k, v in self._detection_keys.items()
                if v in self._targets
            }

    # ------------------------------------------------------------------
    # Hard membership cap (defense-in-depth; see class docstring)
    # ------------------------------------------------------------------

    def _reject_at_cap_locked(self, source: str) -> bool:
        """Return True when a NEW track must be refused (tracker full).

        Must be called with ``self._lock`` held, at a creation site of a
        sensor-derived ingest path.  Counts the rejection and stages the
        LOUD alarm; the caller must invoke :meth:`_flush_cap_alarm`
        after releasing the lock (the alarm publishes to the event bus,
        which may re-enter the tracker — never do that under the lock).
        """
        if self.max_targets is None or len(self._targets) < self.max_targets:
            return False
        self.cap_rejections += 1
        self._cap_alarm_pending = {
            "max_targets": self.max_targets,
            "active_targets": len(self._targets),
            "rejected_total": self.cap_rejections,
            "source": source,
        }
        return True

    def _flush_cap_alarm(self) -> None:
        """Emit the staged capacity alarm (rate-limited, outside the lock).

        LOUD by contract: a WARNING log line plus a ``tracker.capacity``
        event on the bus.  Rate-limited to one alarm per
        :attr:`CAP_ALARM_INTERVAL` seconds so a 10 Hz ingest storm reads
        as a heartbeat, not a log flood — but the very first trip always
        fires immediately.  ``cap_rejections`` counts every refusal
        regardless of rate limiting, so nothing is silently dropped.
        """
        pending = self._cap_alarm_pending
        if pending is None:
            return
        now = time.monotonic()
        if (
            self._last_cap_alarm is not None
            and (now - self._last_cap_alarm) < self.CAP_ALARM_INTERVAL
        ):
            return
        self._last_cap_alarm = now
        self._cap_alarm_pending = None
        logger.warning(
            "TargetTracker at capacity (%d/%d): refused new '%s' track — "
            "%d total refusals. Runaway ingest or cap too small.",
            pending["active_targets"], pending["max_targets"],
            pending["source"], pending["rejected_total"],
        )
        if self._event_bus is not None:
            try:
                self._event_bus.publish("tracker.capacity", data=pending)
            except Exception:
                # The alarm must never break ingest — the WARNING above
                # already made the refusal visible.
                pass

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
                # Crowd sub-role can change mid-sim (a civilian radicalizes
                # into a rioter).  Only overwrite when the telemetry actually
                # carries the key — an absent key is "no opinion", not "clear
                # the known role" (throttled/partial updates omit it).
                if "crowd_role" in sim_data:
                    t.crowd_role = sim_data.get("crowd_role")
                # Health (hit-feedback contract): same absent-key rule — a
                # telemetry frame without health is "no opinion", not "heal
                # to unknown".  A frame WITH it is authoritative (the robot
                # owns its own health; see tritium_lib.models.hits).
                if "health" in sim_data:
                    t.health = sim_data.get("health")
                if "max_health" in sim_data:
                    t.max_health = sim_data.get("max_health")
                # Alliance (declared-telemetry tier): a unit CAN change sides
                # mid-run — a red-team robot re-declares itself, an NPC turns
                # hostile.  Same absent-key rule as crowd_role/health: a frame
                # without the key is "no opinion", never a reset.  Junk values
                # never clobber (VALID_ALLIANCES whitelist).  An operator tag
                # outranks the wire: once alliance_source == "operator" the
                # declared claim is ignored until the operator re-tags
                # (set_operator_alliance is the only writer that pins).
                if "alliance" in sim_data and t.alliance_source != "operator":
                    declared = sim_data.get("alliance")
                    if declared in VALID_ALLIANCES and declared != t.alliance:
                        t.alliance = declared
                        # Alliance is identity-grade state (map glyph color,
                        # hostiles/friendlies REST, CoT affiliation): bump the
                        # version counter so /api/targets ETag caches refresh.
                        self._membership_count += 1
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
                    # Civil-unrest crowd sub-role (None for non-crowd units).
                    crowd_role=sim_data.get("crowd_role"),
                    # Hit-feedback health (None = does not report health).
                    health=sim_data.get("health"),
                    max_health=sim_data.get("max_health"),
                )
        self.history.record(tid, position)
        self._check_geofence(tid, position[0], position[1])

    YOLO_MAX_TRACK_SPEED = 30.0
    """Upper bound on plausible target speed (m/s) used to expand the YOLO
    match radius across detection gaps.  30 m/s ≈ 67 mph covers cars on
    surface streets; faster vehicles will spawn new IDs (and that's fine —
    they're a different track regime)."""

    VISION_SOURCES = ("yolo", "camera")
    """The single *vision* track regime.  ``"camera"`` marks a detection
    with camera provenance (a posed security camera saw these pixels —
    ``source_camera`` in the payload); ``"yolo"`` is a bare vision
    detection with no camera identity.  Both describe the same physical
    contact population, so matching treats them as one regime: a camera
    detection refreshes a yolo track instead of spawning a duplicate id."""

    def update_from_detection(
        self, detection: dict, *, detection_key: str | None = None
    ) -> str | None:
        """Update or create a tracked target from a vision detection.

        Identity resolution has two modes:

        **Keyed (stable identity)** — when the caller supplies a key
        (the ``detection_key`` kwarg, or a ``detection_key`` /
        ``source_track_id`` field in the payload), the detection resolves
        STRICTLY by key: the same key always updates the same live track,
        a new key mints a new track, and proximity plays no part.  This
        is the contract for any consumer that re-ingests detections it
        already has an identity for — an upstream tracker id (ByteTrack,
        radar track), a per-slot camera id, or the tracker's OWN
        ``det_*`` id when republishing (the 2026-07 feedback cascade:
        without a stable key, every re-ingest of the tracker's own
        output minted a fresh ``det_*``, growing 0 -> 1,279 tracks in
        3.5 min).  Keyed resolution trusts the caller: it bypasses the
        asset-type check, so a key whose class drifts (person -> car)
        keeps updating the one track rather than re-minting.  A key
        whose track has been pruned mints a fresh track — keys never
        resurrect the dead.  Distinct keys are never merged, even when
        co-located: supplying keys asserts identity, and asserting two
        identities means two tracks (the fusion correlator, not this
        method, is where cross-track identity is argued).

        **Unkeyed (dedupe-by-proximity)** — with no key (the default,
        byte-identical to the historical behavior), match logic chooses
        the *closest* existing vision target of the same asset type
        within a motion-aware radius — not the first that fits.  The
        radius grows with the time since each candidate was last seen so
        that fast targets do not split into a new ID every frame, while
        still-recent ghosts don't get refreshed by a detection that's
        actually a new entity.  What this does and does NOT guarantee:
        it deduplicates *re-observations of the same entity* (small
        inter-frame motion, same class); it does NOT guarantee two
        genuinely distinct co-located entities stay distinct (two people
        within the match radius CAN collapse to one track — in a crowd,
        supply keys), and it does NOT guarantee a fast or teleporting
        re-observation reuses its track (outside the motion budget a new
        id is minted — which is exactly the runaway a republishing
        caller must avoid by supplying keys).

        Camera provenance: when the payload carries ``source_camera`` (set
        by :class:`tritium_lib.perception.FrameDetectionPipeline` and the
        SC camera bridges), the track is created with ``source="camera"``
        and the camera identity/geometry (``camera_id``, ``bearing_deg``,
        ``distance_m``, ``bbox``) is stamped into ``kinematics`` so the
        dossier and map can answer *which camera saw this target*.

        Returns:
            The ``det_*`` target id this detection matched or created, so
            callers (dossier signals, fusion) can link follow-on records
            to the exact track — or None if the detection was rejected
            (confidence gate, or the tracker is at ``max_targets``).
        """
        if detection.get("confidence", 0) < 0.4:
            return None

        key = (
            detection_key
            or detection.get("detection_key")
            or detection.get("source_track_id")
        )
        key = str(key) if key else None

        class_name = detection.get("class_name", "unknown")
        cx = detection.get("center_x", 0.0)
        cy = detection.get("center_y", 0.0)
        camera_id = str(detection.get("source_camera") or "")
        src = "camera" if camera_id else "yolo"

        # A vision detection carries ZERO IFF information — pixels that say
        # "person" say nothing about intent.  Land as "unknown" like every
        # other passive-sensor ingest (radar obstacles, acoustic, BLE) and
        # let the alliance-authority precedence upgrade it: an operator tag,
        # declared telemetry via fusion with a sim/wire track, or a
        # classifier/threat verdict.  The old person="hostile" hard-code
        # fabricated phantom hostiles at fleet scale (203 "hostile" tracks
        # vs 4 real hostiles measured in the 2026-07-17 battle verification)
        # and made /api/targets/hostiles and every threat count meaningless
        # in demo/city-sim, where most person detections are ambient
        # civilians.
        if class_name == "person":
            asset_type = "person"
        elif class_name in ("car", "motorcycle", "bicycle"):
            asset_type = "vehicle"
        else:
            asset_type = class_name
        alliance = "unknown"

        now = time.monotonic()
        v_max = self.YOLO_MAX_TRACK_SPEED
        base_threshold_sq = 9.0 if (abs(cx) > 2.0 or abs(cy) > 2.0) else 0.04

        capped = False
        with self._lock:
            matched = None
            if key is not None:
                # Keyed path: strict identity, no proximity, no
                # asset-type gate — the caller asserted which entity
                # this is.
                matched = self._resolve_keyed_track_locked(key)
            else:
                best_dist_sq = float("inf")
                for existing in self._targets.values():
                    if existing.source not in self.VISION_SOURCES:
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
                self._add_confirming_source(matched, src)
                if camera_id:
                    self._stamp_camera_provenance(matched, camera_id, detection)
                tid = matched.target_id
            elif self._reject_at_cap_locked(src):
                capped = True
            else:
                self._detection_counter += 1
                self._membership_count += 1
                tid = f"det_{class_name}_{self._detection_counter}"
                target = TrackedTarget(
                    target_id=tid,
                    name=f"{class_name.title()} #{self._detection_counter}",
                    alliance=alliance,
                    asset_type=asset_type,
                    position=(cx, cy),
                    last_seen=now,
                    first_seen=now,
                    signal_count=1,
                    source=src,
                    position_source=src,
                    position_confidence=0.1,
                    _initial_confidence=0.1,
                    confirming_sources={src},
                    classification=class_name,
                    classification_confidence=detection.get("confidence", 0.0),
                )
                if camera_id:
                    self._stamp_camera_provenance(target, camera_id, detection)
                self._targets[tid] = target
                if key is not None:
                    self._record_detection_key_locked(key, tid)
        if capped:
            self._flush_cap_alarm()
            return None
        self.history.record(tid, (cx, cy))
        return tid

    @staticmethod
    def _stamp_camera_provenance(
        target: TrackedTarget, camera_id: str, detection: dict
    ) -> None:
        """Record which camera saw this track (and where in its view).

        Provenance lives in ``kinematics`` — the structured detection-
        metadata field — NOT in ``confirming_sources``, because camera and
        yolo are the same vision modality (see ``_add_confirming_source``).
        Existing kinematics keys from other sources are preserved.

        Depth-fusion payloads (``tracking.depth_fusion``) additionally carry
        measured 3D fields — ``range_m`` (slant range), ``world_enu`` /
        ``elevation_m``, ``world_lat`` / ``world_lng``, ``depth_source`` —
        which ride along here so the dossier/map can answer *how far and how
        high* as well as *which camera*.
        """
        kin = dict(target.kinematics) if target.kinematics else {}
        kin["camera_id"] = camera_id
        for key in (
            "bearing_deg", "distance_m",
            "range_m", "elevation_m", "world_lat", "world_lng", "depth_source",
        ):
            value = detection.get(key)
            if value is not None:
                kin[key] = value
        enu = detection.get("world_enu")
        if isinstance(enu, (list, tuple)) and len(enu) >= 3:
            kin["world_enu"] = list(enu[:3])
        bbox = detection.get("bbox")
        if isinstance(bbox, dict):
            kin["bbox"] = dict(bbox)
        target.kinematics = kin

    def update_from_camera_detection(
        self,
        detection: dict,
        camera_lat: float,
        camera_lng: float,
        latlng_to_local_fn=None,
        camera_id: str = "",
    ) -> str | None:
        """Update or create a target from a camera detection, positioned near the camera.

        Args:
            detection: Dict with keys: label/class_name, confidence, bbox.
            camera_lat: Camera latitude.
            camera_lng: Camera longitude.
            latlng_to_local_fn: Optional callable(lat, lng) -> (x, y, z).
                If None, tries to import from tritium_lib.geo.
            camera_id: Identity of the observing camera.  When set (or when
                the detection dict carries ``camera_id``/``source_camera``),
                the resulting track gets camera provenance: ``source="camera"``
                and ``kinematics.camera_id`` for the dossier/map surface.
                Default empty keeps the legacy ``source="yolo"`` behavior.
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

        cam_id = str(
            camera_id
            or detection.get("camera_id")
            or detection.get("source_camera")
            or ""
        )
        payload = {
            "class_name": label,
            "confidence": confidence,
            "center_x": game_x,
            "center_y": game_y,
        }
        if cam_id:
            payload["source_camera"] = cam_id
        if isinstance(bbox, dict):
            payload["bbox"] = bbox
        # Stable-identity passthrough: a caller that knows which entity
        # this is (upstream tracker id, per-slot camera track) rides the
        # keyed contract of update_from_detection instead of proximity.
        key = detection.get("detection_key") or detection.get("source_track_id")
        if key:
            payload["detection_key"] = str(key)
        return self.update_from_detection(payload)

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

        capped = False
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
            elif self._reject_at_cap_locked("ble"):
                capped = True
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
        if capped:
            self._flush_cap_alarm()
            return
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

        capped = False
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
            elif self._reject_at_cap_locked("mesh"):
                capped = True
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
        if capped:
            self._flush_cap_alarm()
            return
        if pos_source != "unknown":
            self.history.record(tid, position)
            self._check_geofence(tid, position[0], position[1])

    # Confidence carried by a robot's OWN pose report.  Ground truth out of a
    # simulator is exact — it is the only position in the system that is known
    # rather than estimated.  A real robot's onboard estimate (odometry, GPS,
    # SLAM) drifts, so it sits high but deliberately below 1.0: fusion must be
    # able to prefer a corroborating fix over a confidently-wrong dead reckon.
    ROBOT_POSE_GROUND_TRUTH_CONFIDENCE = 1.0
    ROBOT_POSE_ONBOARD_CONFIDENCE = 0.85

    # Confidence carried by a clustered LiDAR return.  A range cluster proves
    # something is THERE and nothing more — no identity, no allegiance — so it
    # sits well below a self-reported robot pose (which knows what it is) and
    # below a camera detection (which at least has a class).  Fusion should be
    # free to let any classifying sensor overrule a bare obstacle.
    LIDAR_OBSTACLE_CONFIDENCE = 0.6

    # Association gate: how far a cluster centroid may move between sweeps and
    # still be judged the SAME obstacle.  Sized for a 10 Hz sweep and a body
    # under ~3 m/s; too wide and two neighbouring obstacles swap identities,
    # too narrow and a static wall re-registers as a new contact every sweep.
    LIDAR_ASSOCIATION_GATE_M = 1.5

    def update_from_laser_scan(self, scan: dict) -> list[str]:
        """A LaserScan sweep -> operator-visible obstacle tracks.

        The SC-side landing point for :mod:`tritium_lib.geo.laser_scan`, and
        the fifth ingest modality after RF / vision / mesh / acoustic.  The
        geometry (polar -> world points -> range-gap clustering) lives in the
        geo module; what this method adds is **identity across sweeps**.

        That is the whole difficulty.  A cluster index is not a track: one
        extra speckle at a low beam index renumbers every cluster behind it,
        so index-derived ids make a static wall look like a stream of contacts
        blinking in and out.  Instead each cluster is associated to the
        nearest existing track *from the same lidar* within
        :attr:`LIDAR_ASSOCIATION_GATE_M` (global nearest neighbour, the
        standard LaserScan association), and only an unmatched cluster mints a
        new id.  Ids stay namespaced per ``lidar_id`` so two sensors cannot
        silently fuse two rooms into one.

        Obstacles land as ``unknown`` alliance by design: a range return says
        something is there, never what it is or whose it is.

        Args:
            scan: a ``/scan``-shaped payload — ``ranges`` (required),
                ``angle_min``, ``angle_increment``, ``range_min``,
                ``range_max``, ``lidar_id``, the sensor pose
                (``sensor_x`` / ``sensor_y`` / ``sensor_yaw_deg``), and the
                clustering knobs ``gap_m`` / ``min_points`` /
                ``association_gate_m``.

        Returns:
            The target ids touched by this sweep, in beam order.  An empty or
            malformed sweep returns ``[]`` rather than raising — a connector
            must not be able to crash the tracker.
        """
        from ..geo.laser_scan import scan_obstacles

        ranges = scan.get("ranges")
        if not ranges:
            return []

        lidar_id = str(scan.get("lidar_id") or "lidar")
        gate = float(scan.get("association_gate_m", self.LIDAR_ASSOCIATION_GATE_M))

        try:
            obstacles = scan_obstacles(
                ranges,
                angle_min=float(scan.get("angle_min", -math.pi)),
                angle_increment=float(
                    scan.get("angle_increment", 2.0 * math.pi / len(ranges))
                ),
                sensor_x=float(scan.get("sensor_x", 0.0)),
                sensor_y=float(scan.get("sensor_y", 0.0)),
                sensor_yaw_deg=float(scan.get("sensor_yaw_deg", 0.0)),
                range_min=float(scan.get("range_min", 0.0)),
                # A beam at exactly range_max is a NO-RETURN, not a hit at the
                # horizon.  Excluding the boundary is what stops an empty room
                # from being ingested as a solid ring of obstacles around the
                # sensor.
                range_max=float(scan.get("range_max", float("inf"))) - 1e-6,
                gap_m=float(scan.get("gap_m", 0.5)),
                min_points=int(scan.get("min_points", 1)),
            )
        except (TypeError, ValueError):
            return []

        now = time.monotonic()
        touched: list[str] = []
        capped = False

        with self._lock:
            # Candidates are this lidar's own existing tracks.  Each may be
            # claimed once per sweep, so two clusters can never collapse onto
            # one track.
            prefix = f"lidar_{lidar_id}_"
            unclaimed = {
                tid for tid in self._targets if tid.startswith(prefix)
            }

            for obs in obstacles:
                best_id, best_d = None, gate
                for tid in unclaimed:
                    t = self._targets[tid]
                    d = math.hypot(t.position[0] - obs.x, t.position[1] - obs.y)
                    if d <= best_d:
                        best_id, best_d = tid, d

                if best_id is not None:
                    unclaimed.discard(best_id)
                    t = self._targets[best_id]
                    t.position = (obs.x, obs.y)
                    t.last_seen = now
                    t.signal_count += 1
                    t.position_confidence = self.LIDAR_OBSTACLE_CONFIDENCE
                    self._add_confirming_source(t, "lidar")
                    touched.append(best_id)
                else:
                    if self._reject_at_cap_locked("lidar"):
                        capped = True
                        continue
                    self._lidar_seq = getattr(self, "_lidar_seq", 0) + 1
                    tid = f"{prefix}{self._lidar_seq}"
                    self._membership_count += 1
                    self._targets[tid] = TrackedTarget(
                        target_id=tid,
                        name=f"OBSTACLE {self._lidar_seq}",
                        alliance="unknown",
                        asset_type="obstacle",
                        position=(obs.x, obs.y),
                        heading=0.0,
                        speed=0.0,
                        last_seen=now,
                        first_seen=now,
                        signal_count=1,
                        source="lidar",
                        position_source="lidar",
                        position_confidence=self.LIDAR_OBSTACLE_CONFIDENCE,
                        _initial_confidence=self.LIDAR_OBSTACLE_CONFIDENCE,
                        confirming_sources={"lidar"},
                        classification="obstacle",
                    )
                    touched.append(tid)

        if capped:
            self._flush_cap_alarm()
        for tid in touched:
            t = self._targets.get(tid)
            if t is not None:
                self.history.record(tid, t.position)
                self._check_geofence(tid, t.position[0], t.position[1])
        return touched

    def update_from_robot_pose(self, pose: dict) -> str | None:
        """Update or create a target from a body reporting its OWN pose.

        Every other ``update_from_*`` method models a sensor observing
        something else and has to infer heading from successive positions.
        This one models a body — a quadruped in Isaac, a rover on the wire —
        telling the operator where it is and which way it faces.  Heading
        therefore arrives directly, which is what lets the map icon agree with
        the simulator viewport instead of lagging a motion estimate.

        This is the SC-side landing point for
        :mod:`tritium_lib.geo.isaac_frame`: the Isaac pose bridge converts a
        stage pose to east/north + compass heading and hands the result
        straight here.

        Args:
            pose: ``target_id`` (required), ``position`` as ``{"x","y"}`` or
                ``[x, y]``, ``heading`` degrees CW from north, and optionally
                ``name`` / ``asset_type`` / ``alliance`` / ``speed`` /
                ``battery`` / ``ground_truth``.

        Returns:
            The target id, or ``None`` when the payload carries no id.
        """
        tid = str(pose.get("target_id") or "").strip()
        if not tid:
            return None

        raw_pos = pose.get("position")
        if isinstance(raw_pos, dict):
            position = (float(raw_pos.get("x", 0.0)), float(raw_pos.get("y", 0.0)))
        elif isinstance(raw_pos, (list, tuple)) and len(raw_pos) >= 2:
            position = (float(raw_pos[0]), float(raw_pos[1]))
        elif "x" in pose or "y" in pose:
            # Flat form — what the Isaac pose bridge and robot MQTT telemetry
            # already put on the wire.  Accepting it here means the ingest
            # seam does not force every producer to re-shape its payload.
            position = (float(pose.get("x", 0.0)), float(pose.get("y", 0.0)))
        else:
            position = (0.0, 0.0)

        heading = float(pose.get("heading", 0.0)) % 360.0
        speed = float(pose.get("speed", 0.0))
        battery = float(pose.get("battery", 1.0))
        name = str(pose.get("name") or tid)
        asset_type = str(pose.get("asset_type") or "robot")
        alliance = str(pose.get("alliance") or "friendly")

        ground_truth = bool(pose.get("ground_truth", False))
        pos_source = "sim_truth" if ground_truth else "onboard"
        confidence = (
            self.ROBOT_POSE_GROUND_TRUTH_CONFIDENCE if ground_truth
            else self.ROBOT_POSE_ONBOARD_CONFIDENCE
        )

        with self._lock:
            t = self._targets.get(tid)
            if t is not None:
                # The integrity gate is a check on an ESTIMATE.  Ground truth
                # cannot be spoofed and an operator repositioning a sim body
                # is legitimate, so flagging it would put a permanent false
                # alarm on the one track that is known to be correct.
                if not ground_truth:
                    self._check_velocity(t, position)
                t.position = position
                t.position_source = pos_source
                t.heading = heading
                t.speed = speed
                t.battery = battery
                t.name = name
                t.asset_type = asset_type
                # Operator alliance tags outrank declared telemetry (see
                # ``set_operator_alliance``) — never clobber a human decision.
                if t.alliance_source != "operator":
                    t.alliance = alliance
                t.last_seen = time.monotonic()
                t.signal_count += 1
                t.position_confidence = confidence
                t._initial_confidence = confidence
                self._add_confirming_source(t, "robot_pose")
            else:
                self._membership_count += 1
                self._targets[tid] = TrackedTarget(
                    target_id=tid,
                    name=name,
                    alliance=alliance,
                    asset_type=asset_type,
                    position=position,
                    heading=heading,
                    speed=speed,
                    battery=battery,
                    last_seen=time.monotonic(),
                    first_seen=time.monotonic(),
                    signal_count=1,
                    source="robot_pose",
                    position_source=pos_source,
                    position_confidence=confidence,
                    _initial_confidence=confidence,
                    confirming_sources={"robot_pose"},
                    classification=asset_type,
                )
                self.reappearance_monitor.check_reappearance(
                    target_id=tid,
                    name=name,
                    source="robot_pose",
                    asset_type=asset_type,
                    position=position,
                )

        self.history.record(tid, position)
        self._check_geofence(tid, position[0], position[1])
        return tid

    # Acoustic targets — transient sounds (gunshot/voice/vehicle) detected by a
    # mic array.  Localisation is coarse (direction-of-arrival, weak range), so
    # tracks are low-confidence and short-lived, but they still confirm an
    # entity already seen by RF/vision/mesh — the fourth north-star modality.
    ACOUSTIC_STALE_TIMEOUT = 60.0

    # Acoustic event type -> (asset_type, classification).  The classification
    # IS the sound class; asset_type is the most likely emitter.
    _ACOUSTIC_ASSET = {
        "gunshot": "person",
        "voice": "person",
        "footsteps": "person",
        "scream": "person",
        "glass_break": "person",
        "vehicle": "vehicle",
        "engine": "vehicle",
        "drone": "drone",
    }

    def update_from_acoustic(self, event: dict) -> None:
        """Update or create a tracked target from an acoustic detection.

        Args:
            event: Dict with keys: event_type (gunshot/voice/vehicle/...),
                position {x, y} (coarse DOA estimate), confidence, sensor_id,
                and optionally target_id / name / alliance.

        Acoustic is the fourth north-star modality (RF + vision + mesh +
        acoustic).  A co-located acoustic track correlates with the RF/vision
        tracks for the same entity into ONE unique multi-source ID.
        """
        event_type = event.get("event_type", "unknown")
        target_id = event.get("target_id", "")
        sensor_id = event.get("sensor_id", "mic")

        # Stable per-entity id when the caller knows which entity made the
        # sound; otherwise key by the sensor + sound class.
        if target_id:
            tid = f"acoustic_{target_id}"
        elif event.get("position"):
            tid = f"acoustic_{sensor_id}_{event_type}"
        else:
            # Nothing to localise or key on — ignore (never create junk tracks).
            return

        pos = event.get("position")
        if pos:
            position = (float(pos.get("x", 0)), float(pos.get("y", 0)))
            pos_source = "acoustic_doa"
        else:
            position = (0.0, 0.0)
            pos_source = "unknown"

        name = event.get("name") or target_id or event_type
        alliance = event.get("alliance", "unknown")
        asset_type = self._ACOUSTIC_ASSET.get(event_type, "unknown")
        # DOA localisation is coarse — cap confidence well below RF/vision.
        confidence = max(0.0, min(0.6, float(event.get("confidence", 0.4))))

        capped = False
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
                self._add_confirming_source(t, "acoustic")
            elif self._reject_at_cap_locked("acoustic"):
                capped = True
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
                    source="acoustic",
                    position_source=pos_source,
                    position_confidence=confidence,
                    _initial_confidence=confidence,
                    confirming_sources={"acoustic"},
                    classification=event_type,
                    classification_confidence=confidence,
                )
                self.reappearance_monitor.check_reappearance(
                    target_id=tid,
                    name=name,
                    source="acoustic",
                    asset_type=asset_type,
                    position=position,
                )
        if capped:
            self._flush_cap_alarm()
            return
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

        capped = False
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
            elif self._reject_at_cap_locked("adsb"):
                capped = True
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

        if capped:
            self._flush_cap_alarm()
            return
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

        capped = False
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
            elif self._reject_at_cap_locked("rf_motion"):
                capped = True
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
        if capped:
            self._flush_cap_alarm()
            return
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

        Exception: *alliance* changes (operator tag or a declared mid-run
        side-switch) also bump, because alliance is identity-grade state a
        reconciling client must not 304 past — see set_operator_alliance
        and update_from_simulation's alliance handling.

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

    def set_operator_alliance(self, target_id: str, alliance: str) -> TrackedTarget | None:
        """Apply an explicit operator alliance tag — the TOP precedence tier.

        This is the ONLY writer that sets ``alliance_source = "operator"``.
        Once pinned, declared telemetry (update_from_simulation's alliance
        handling, wire re-declarations) can never clobber the human's
        decision; only a subsequent operator tag changes it.  SC's
        POST /api/targets/{id}/tag and the classification-override route
        both come through here so map == REST == CoT == fusion.

        Returns the updated target, or ``None`` when the alliance value is
        not in :data:`VALID_ALLIANCES` or the target is unknown.
        """
        if alliance not in VALID_ALLIANCES:
            return None
        with self._lock:
            t = self._targets.get(target_id)
            if t is None:
                return None
            t.alliance = alliance
            t.alliance_source = "operator"
            # Identity-grade change: invalidate /api/targets ETag caches so a
            # reconciling client refreshes immediately instead of 304-ing on
            # a stale alliance.
            self._membership_count += 1
            return t

    def remove(self, target_id: str) -> bool:
        """Remove a target from tracking."""
        with self._lock:
            removed = self._targets.pop(target_id, None) is not None
            if removed:
                self._membership_count += 1
                self._purge_detection_keys_locked()
            return removed

    def clear_source(self, source: str) -> int:
        """Remove every target whose ``source`` field matches.

        Returns the count of targets removed.  Used by the demo router
        on POST /api/demo/stop so synthetic targets do not linger in
        the tracker after the demo is shut down (Gap-fix C GA-1).
        """
        if not source:
            return 0
        with self._lock:
            stale = [
                tid for tid, t in self._targets.items()
                if getattr(t, "source", None) == source
            ]
            for tid in stale:
                self._targets.pop(tid, None)
            if stale:
                self._membership_count += 1
                self._purge_detection_keys_locked()
            return len(stale)

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

    def tactical_brief(self) -> str:
        """Concise live situational grounding for the cognition layer.

        Unlike :meth:`summary` (combat-oriented: proximity alerts + hostile
        sectors, and which OMITS the ``neutral`` alliance), this is a
        state-of-the-board inventory meant to ground an operator/Amy question
        regardless of game state — it is a pure tracker read with no
        ``game_mode`` dependency, so it works in monitor mode too:

        - counts by alliance INCLUDING ``neutral`` (the civilian
          non-combatants the combat summary leaves invisible), and
        - a breakdown by target classification (person/vehicle/phone/animal —
          the operational mission's "track every target" taxonomy), falling
          back to ``asset_type`` when a target has no ML classification.

        Returns ``""`` when nothing is tracked.
        """
        targets = self.get_all()
        if not targets:
            return ""

        alliance_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for t in targets:
            alliance_counts[t.alliance] = alliance_counts.get(t.alliance, 0) + 1
            # Prefer the ML/RL classification; fall back to the asset type.
            kind = t.classification
            if not kind or kind == "unknown":
                kind = t.asset_type or "unknown"
            type_counts[kind] = type_counts.get(kind, 0) + 1

        # Stable, operator-meaningful ordering; unknown alliances trail.
        order = ["friendly", "hostile", "neutral", "unknown"]
        ordered = [a for a in order if a in alliance_counts]
        ordered += [a for a in alliance_counts if a not in order]
        alliance_str = ", ".join(f"{alliance_counts[a]} {a}" for a in ordered)

        lines = [f"TRACKING {len(targets)} target(s): {alliance_str}"]

        # Type breakdown — most common first (deterministic tie-break by name),
        # capped to keep the brief short for the small chat model.
        known = {k: v for k, v in type_counts.items() if k and k != "unknown"}
        if known:
            top = sorted(known.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
            lines.append("Types: " + ", ".join(f"{v} {k}" for k, v in top))

        # Zone occupancy — surface live geofence state to cognition so an
        # operator/Amy can answer "what's in the restricted zone?" instead of
        # guessing (UX Loop 5 / 10). The tracker already holds the geofence
        # engine reference (set via set_geofence_engine), and knows each
        # target's alliance/classification, so it can resolve occupants into a
        # threat-aware breakdown (a hostile in a restricted zone = BREACH).
        geo = getattr(self, "_geofence_engine", None)
        if geo is not None and hasattr(geo, "zone_brief"):
            try:
                by_id = {t.target_id: t for t in targets}

                def _resolve(tid):
                    t = by_id.get(tid)
                    if t is None:
                        return None
                    kind = t.classification
                    if not kind or kind == "unknown":
                        kind = t.asset_type or "unknown"
                    return {"alliance": t.alliance, "classification": kind}

                zb = geo.zone_brief(occupant_resolver=_resolve)
                if zb:
                    lines.append(zb)
            except Exception:
                pass

        return "\n".join(lines)

    SIM_STALE_TIMEOUT = 10.0

    def _prune_stale(self) -> None:
        """Remove targets that haven't been updated recently."""
        now = time.monotonic()
        with self._lock:
            stale = [
                tid for tid, t in self._targets.items()
                if (t.source in self.VISION_SOURCES
                    and (now - t.last_seen) > self.STALE_TIMEOUT)
                or (t.source == "simulation" and (now - t.last_seen) > self.SIM_STALE_TIMEOUT)
                or (t.source == "ble" and (now - t.last_seen) > self.BLE_STALE_TIMEOUT)
                or (t.source == "rf_motion" and (now - t.last_seen) > self.RF_MOTION_STALE_TIMEOUT)
                or (t.source == "mesh" and (now - t.last_seen) > self.MESH_STALE_TIMEOUT)
                or (t.source == "acoustic" and (now - t.last_seen) > self.ACOUSTIC_STALE_TIMEOUT)
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
            if stale:
                self._purge_detection_keys_locked()

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Sensor detection, stealth, camouflage, noise, and electronic warfare.

Simulates how different sensor types detect targets based on signature
profiles, environmental conditions, and countermeasures.  Integrates with
the environment module (TimeOfDay, Weather) for realistic modifiers.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SensorType(Enum):
    """Types of sensors available in the simulation."""
    VISUAL = "visual"
    THERMAL = "thermal"
    ACOUSTIC = "acoustic"
    RADAR = "radar"
    SONAR = "sonar"
    SEISMIC = "seismic"
    RF_PASSIVE = "rf_passive"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Sensor:
    """A sensor attached to a unit or emplacement."""
    sensor_id: str
    sensor_type: SensorType
    position: Vec2
    heading: float              # radians, 0 = east / +x
    fov_deg: float              # field of view in degrees (360 = omnidirectional)
    range_m: float              # max detection range in meters
    sensitivity: float          # 0-1, higher = better at detecting low signatures
    is_active: bool = True
    owner_id: str = ""


@dataclass
class SignatureProfile:
    """How detectable an entity is across sensor domains.

    Each value is 0-1 where 0 = undetectable and 1 = maximum signature.
    """
    visual: float = 1.0
    thermal: float = 1.0
    acoustic: float = 1.0
    radar: float = 1.0
    rf_emission: float = 0.0

    def get(self, sensor_type: SensorType) -> float:
        """Return the signature value matching *sensor_type*."""
        _MAP = {
            SensorType.VISUAL: self.visual,
            SensorType.THERMAL: self.thermal,
            SensorType.ACOUSTIC: self.acoustic,
            SensorType.RADAR: self.radar,
            SensorType.RF_PASSIVE: self.rf_emission,
            SensorType.SONAR: self.acoustic,   # sonar reuses acoustic
            SensorType.SEISMIC: self.acoustic,  # seismic reuses acoustic
        }
        return _MAP.get(sensor_type, 0.0)


@dataclass
class Detection:
    """A single sensor detection event."""
    detector_id: str
    target_id: str
    sensor_type: SensorType
    confidence: float           # 0-1
    position_accuracy: float    # meters of error
    timestamp: float
    is_confirmed: bool = False


# ---------------------------------------------------------------------------
# Signature Presets
# ---------------------------------------------------------------------------

SIGNATURE_PRESETS: dict[str, SignatureProfile] = {
    "infantry": SignatureProfile(
        visual=0.6, thermal=0.3, acoustic=0.2, radar=0.1, rf_emission=0.0,
    ),
    "sniper_ghillie": SignatureProfile(
        visual=0.1, thermal=0.2, acoustic=0.05, radar=0.05, rf_emission=0.0,
    ),
    "vehicle": SignatureProfile(
        visual=1.0, thermal=0.8, acoustic=0.9, radar=0.7, rf_emission=0.0,
    ),
    "tank": SignatureProfile(
        visual=1.0, thermal=1.0, acoustic=1.0, radar=1.0, rf_emission=0.0,
    ),
    "helicopter": SignatureProfile(
        visual=0.9, thermal=0.9, acoustic=1.0, radar=0.8, rf_emission=0.0,
    ),
    "drone_small": SignatureProfile(
        visual=0.2, thermal=0.1, acoustic=0.3, radar=0.15, rf_emission=0.0,
    ),
    "submarine_surfaced": SignatureProfile(
        visual=0.8, thermal=0.5, acoustic=0.6, radar=0.7, rf_emission=0.0,
    ),
    "submarine_submerged": SignatureProfile(
        visual=0.0, thermal=0.1, acoustic=0.3, radar=0.0, rf_emission=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Countermeasure Records
# ---------------------------------------------------------------------------

@dataclass
class _Countermeasure:
    """Active countermeasure affecting an area."""
    position: Vec2
    radius: float
    kind: str           # "smoke", "chaff", "flare"
    intensity: float    # 0-1
    remaining: float    # seconds remaining


# ---------------------------------------------------------------------------
# Detection Engine
# ---------------------------------------------------------------------------

class DetectionEngine:
    """Runs sensor-vs-entity detection every tick.

    The engine evaluates each sensor against each entity, applying range,
    FOV, signature strength, environmental modifiers, and active
    countermeasures to produce a list of Detection results.
    """

    def __init__(self) -> None:
        self.sensors: list[Sensor] = []
        self.signatures: dict[str, SignatureProfile] = {}
        self.detections: list[Detection] = []
        self.noise_sources: list[dict] = []
        self._countermeasures: list[_Countermeasure] = []
        self._radio_silent: set[str] = set()
        self._detection_decay_s: float = 5.0

    # -- signature management -----------------------------------------------

    def set_signature(self, entity_id: str, profile: SignatureProfile) -> None:
        """Register or update the signature profile for an entity."""
        self.signatures[entity_id] = profile

    # -- noise / countermeasures --------------------------------------------

    def add_noise(self, position: Vec2, intensity: float, duration: float) -> None:
        """Add a temporary acoustic noise source (gunfire, explosion, etc.)."""
        self.noise_sources.append({
            "position": position,
            "intensity": max(0.0, min(1.0, intensity)),
            "remaining": duration,
        })

    def deploy_smoke(self, position: Vec2, radius: float, duration: float = 10.0) -> None:
        """Deploy smoke that reduces visual detection in an area."""
        self._countermeasures.append(_Countermeasure(
            position=position, radius=radius, kind="smoke",
            intensity=1.0, remaining=duration,
        ))

    def deploy_chaff(self, position: Vec2, radius: float, duration: float = 8.0) -> None:
        """Deploy chaff that reduces radar detection in an area."""
        self._countermeasures.append(_Countermeasure(
            position=position, radius=radius, kind="chaff",
            intensity=1.0, remaining=duration,
        ))

    def deploy_flare(self, position: Vec2, duration: float = 6.0) -> None:
        """Deploy a thermal decoy flare."""
        self._countermeasures.append(_Countermeasure(
            position=position, radius=30.0, kind="flare",
            intensity=1.0, remaining=duration,
        ))

    def go_radio_silent(self, entity_id: str) -> None:
        """Set an entity to emit zero RF."""
        self._radio_silent.add(entity_id)

    def break_radio_silence(self, entity_id: str) -> None:
        """Allow an entity to emit RF again."""
        self._radio_silent.discard(entity_id)

    # -- core detection logic -----------------------------------------------

    def _angle_between(self, origin: Vec2, heading: float, target: Vec2) -> float:
        """Absolute angle (degrees) between sensor heading and target bearing."""
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        bearing = math.atan2(dy, dx)
        diff = abs(bearing - heading)
        if diff > math.pi:
            diff = 2.0 * math.pi - diff
        return math.degrees(diff)

    def _countermeasure_modifier(
        self, pos: Vec2, sensor_type: SensorType,
    ) -> float:
        """Return a multiplier (0-1) from active countermeasures at *pos*."""
        modifier = 1.0
        for cm in self._countermeasures:
            d = distance(cm.position, pos)
            if d > cm.radius:
                continue
            # Stronger effect closer to center
            falloff = 1.0 - (d / cm.radius)
            effect = falloff * cm.intensity
            if cm.kind == "smoke" and sensor_type == SensorType.VISUAL:
                modifier *= (1.0 - 0.9 * effect)
            elif cm.kind == "chaff" and sensor_type == SensorType.RADAR:
                modifier *= (1.0 - 0.85 * effect)
            elif cm.kind == "flare" and sensor_type == SensorType.THERMAL:
                # Flares saturate thermal sensors, reducing detection ability
                modifier *= (1.0 - 0.8 * effect)
        return max(0.0, modifier)

    def _env_modifier(self, sensor_type: SensorType, env: dict) -> float:
        """Compute an environment-based modifier for a sensor type.

        Recognised *env* keys:
            weather   — "clear", "rain", "fog", "storm", "snow", "sandstorm"
            is_night  — bool
            cover     — float 0-1 (concealment factor at target position)
        """
        mod = 1.0
        weather = env.get("weather", "clear")
        is_night = env.get("is_night", False)

        if sensor_type == SensorType.VISUAL:
            if is_night:
                mod *= 0.25
            if weather == "fog":
                mod *= 0.2
            elif weather == "rain":
                mod *= 0.6
            elif weather == "heavy_rain":
                mod *= 0.4
            elif weather == "storm":
                mod *= 0.3
            elif weather == "snow":
                mod *= 0.5
            elif weather == "sandstorm":
                mod *= 0.15

        elif sensor_type == SensorType.THERMAL:
            if is_night:
                mod *= 1.3  # thermal is *better* at night
            if weather == "fog":
                mod *= 0.6
            elif weather == "rain":
                mod *= 0.7
            elif weather == "heavy_rain":
                mod *= 0.5

        elif sensor_type == SensorType.ACOUSTIC:
            if weather == "rain":
                mod *= 0.7
            elif weather == "heavy_rain":
                mod *= 0.5
            elif weather == "storm":
                mod *= 0.3
            # wind adds noise
            wind = env.get("wind_speed", 0.0)
            if wind > 10.0:
                mod *= 0.6
            elif wind > 5.0:
                mod *= 0.8

        elif sensor_type == SensorType.RADAR:
            if weather == "rain":
                mod *= 0.85
            elif weather == "heavy_rain":
                mod *= 0.7
            elif weather == "storm":
                mod *= 0.5
            elif weather == "sandstorm":
                mod *= 0.4

        elif sensor_type == SensorType.RF_PASSIVE:
            # RF propagation barely affected by weather
            pass

        # Cover / concealment
        cover = env.get("cover", 0.0)
        if sensor_type in (SensorType.VISUAL, SensorType.RADAR):
            mod *= (1.0 - 0.8 * cover)
        elif sensor_type == SensorType.THERMAL:
            mod *= (1.0 - 0.4 * cover)

        return max(0.0, min(2.0, mod))

    def check_detection(
        self,
        sensor: Sensor,
        target_pos: Vec2,
        target_sig: SignatureProfile,
        environment: dict,
        target_id: str = "",
    ) -> Optional[Detection]:
        """Evaluate whether *sensor* detects a target at *target_pos*.

        Returns a ``Detection`` with computed confidence and accuracy, or
        ``None`` if the target is not detected.
        """
        if not sensor.is_active:
            return None

        # Range check
        dist = distance(sensor.position, target_pos)
        if dist > sensor.range_m:
            return None

        # FOV check (skip for omnidirectional sensors)
        if sensor.fov_deg < 360.0:
            angle_off = self._angle_between(
                sensor.position, sensor.heading, target_pos,
            )
            if angle_off > sensor.fov_deg / 2.0:
                return None

        # Base signature strength for this sensor type
        sig_value = target_sig.get(sensor.sensor_type)

        # Radio silence override
        if sensor.sensor_type == SensorType.RF_PASSIVE and target_id in self._radio_silent:
            sig_value = 0.0

        if sig_value <= 0.0:
            return None

        # Environment modifier
        env_mod = self._env_modifier(sensor.sensor_type, environment)

        # Countermeasure modifier at target position
        cm_mod = self._countermeasure_modifier(target_pos, sensor.sensor_type)

        # Range falloff: quadratic
        if sensor.range_m <= 0.0:
            return None
        range_factor = 1.0 - (dist / sensor.range_m) ** 2

        # Confidence calculation
        confidence = sig_value * sensor.sensitivity * env_mod * cm_mod * range_factor
        confidence = max(0.0, min(1.0, confidence))

        # Below a threshold, no detection
        if confidence < 0.05:
            return None

        # Position accuracy degrades with distance and lower confidence
        base_accuracy = 2.0 + dist * 0.05
        position_accuracy = base_accuracy / max(confidence, 0.1)

        return Detection(
            detector_id=sensor.sensor_id,
            target_id=target_id,
            sensor_type=sensor.sensor_type,
            confidence=confidence,
            position_accuracy=position_accuracy,
            timestamp=time.time(),
            is_confirmed=confidence > 0.8,
        )

    # -- tick ---------------------------------------------------------------

    def tick(
        self,
        dt: float,
        entities: dict[str, Vec2],
        env: dict,
    ) -> list[Detection]:
        """Run one detection cycle.

        Evaluates every sensor against every entity, merges multi-sensor
        detections on the same target, decays old detections, and returns
        new detections for this tick.
        """
        # Decay countermeasures and noise
        for cm in self._countermeasures:
            cm.remaining -= dt
            cm.intensity *= max(0.0, cm.remaining / (cm.remaining + dt))
        self._countermeasures = [c for c in self._countermeasures if c.remaining > 0]

        for ns in self.noise_sources:
            ns["remaining"] -= dt
        self.noise_sources = [n for n in self.noise_sources if n["remaining"] > 0]

        # Decay old detections
        now = time.time()
        self.detections = [
            d for d in self.detections
            if (now - d.timestamp) < self._detection_decay_s
        ]

        new_detections: list[Detection] = []

        for sensor in self.sensors:
            if not sensor.is_active:
                continue
            for eid, pos in entities.items():
                # Don't detect yourself
                if eid == sensor.owner_id:
                    continue
                sig = self.signatures.get(eid)
                if sig is None:
                    continue
                det = self.check_detection(sensor, pos, sig, env, target_id=eid)
                if det is not None:
                    new_detections.append(det)

        # Merge multi-sensor detections per owner: for each (owner, target)
        # pair, keep the best detection.  If multiple sensor *types* from the
        # same owner detect the target, mark it confirmed and boost confidence.
        # This preserves per-owner detections so alliance queries work.
        merged: dict[tuple[str, str], Detection] = {}
        sensor_types_seen: dict[tuple[str, str], set[SensorType]] = {}
        for det in new_detections:
            owner = self._sensor_owner(det.detector_id)
            key = (owner, det.target_id)
            stypes = sensor_types_seen.setdefault(key, set())
            stypes.add(det.sensor_type)
            if key not in merged or det.confidence > merged[key].confidence:
                merged[key] = det
            # Multi-sensor confirmation
            if len(stypes) > 1:
                merged[key].is_confirmed = True
                merged[key].confidence = min(
                    1.0, merged[key].confidence * 1.15,
                )

        result = list(merged.values())
        self.detections.extend(result)
        return result

    def _sensor_owner(self, sensor_id: str) -> str:
        """Return the owner_id of a sensor by its id."""
        for s in self.sensors:
            if s.sensor_id == sensor_id:
                return s.owner_id
        return ""

    # -- queries ------------------------------------------------------------

    def get_detection_map(self, alliance: str) -> dict:
        """Return current detections grouped by target for a given alliance.

        Returns a dict mapping target_id to the best (highest confidence)
        detection.  The *alliance* parameter filters sensors by owner
        (sensors whose ``owner_id`` starts with the alliance tag).
        """
        owned_sensors = {
            s.sensor_id for s in self.sensors
            if s.owner_id.startswith(alliance)
        }
        best: dict[str, Detection] = {}
        for det in self.detections:
            if det.detector_id not in owned_sensors:
                continue
            if det.target_id not in best or det.confidence > best[det.target_id].confidence:
                best[det.target_id] = det
        return {
            tid: {
                "target_id": d.target_id,
                "sensor_type": d.sensor_type.value,
                "confidence": round(d.confidence, 3),
                "position_accuracy": round(d.position_accuracy, 1),
                "is_confirmed": d.is_confirmed,
            }
            for tid, d in best.items()
        }

    # -- serialization for three.js -----------------------------------------

    def to_three_js(self) -> dict:
        """Export current state for Three.js visualization."""
        sensors_out = []
        for s in self.sensors:
            color_map = {
                SensorType.VISUAL: "#00f0ff33",
                SensorType.THERMAL: "#ff2a6d33",
                SensorType.ACOUSTIC: "#05ffa133",
                SensorType.RADAR: "#fcee0a33",
                SensorType.SONAR: "#4488ff33",
                SensorType.SEISMIC: "#aa660033",
                SensorType.RF_PASSIVE: "#ff00ff33",
            }
            sensors_out.append({
                "id": s.sensor_id,
                "x": s.position[0],
                "y": s.position[1],
                "fov": s.fov_deg,
                "range": s.range_m,
                "type": s.sensor_type.value,
                "cone_color": color_map.get(s.sensor_type, "#ffffff33"),
                "active": s.is_active,
            })

        detections_out = []
        for d in self.detections:
            detections_out.append({
                "target_id": d.target_id,
                "confidence": round(d.confidence, 3),
                "accuracy_circle": round(d.position_accuracy, 1),
                "sensor_type": d.sensor_type.value,
                "confirmed": d.is_confirmed,
            })

        noise_out = []
        for ns in self.noise_sources:
            noise_out.append({
                "x": ns["position"][0],
                "y": ns["position"][1],
                "intensity": round(ns["intensity"], 3),
            })

        return {
            "sensors": sensors_out,
            "detections": detections_out,
            "noise_sources": noise_out,
        }

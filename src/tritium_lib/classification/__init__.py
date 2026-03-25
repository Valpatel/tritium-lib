# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.classification — multi-sensor target classification pipeline.

Determines *what* a target IS (person, vehicle, animal, device) by fusing
evidence from multiple classifiers — BLE advertisements, WiFi probe patterns,
movement speed, behavioral patterns, and time-of-day usage.

Architecture
------------
Individual classifiers each produce a ``ClassificationVote`` with a type,
subtype, confidence, and evidence string.  The ``EnsembleClassifier`` collects
votes from all registered classifiers, applies confidence-weighted voting, and
emits a final ``ClassificationResult``.

The ``ClassificationPipeline`` orchestrates the full lifecycle: register
classifiers, ingest observations, run the ensemble, and update the target's
classification fields.

Output types
~~~~~~~~~~~~
person, vehicle, bicycle, animal, fixed_device, mobile_device, unknown

Built-in classifiers
~~~~~~~~~~~~~~~~~~~~
- ``BLETypeClassifier``   — BLE advertisement data (name, OUI, services)
- ``WiFiTypeClassifier``  — WiFi probe request patterns
- ``SpeedClassifier``     — movement speed thresholds
- ``BehaviorClassifier``  — movement patterns (loitering, patrol, transit)
- ``TimeClassifier``      — time-of-day activity patterns

Usage::

    from tritium_lib.classification import ClassificationPipeline

    pipeline = ClassificationPipeline()
    result = pipeline.classify(observations)
    print(result.target_type)   # "person"
    print(result.confidence)    # 0.82
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

logger = logging.getLogger("classification")


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class TargetType(str, Enum):
    """High-level target types the pipeline can produce."""

    PERSON = "person"
    VEHICLE = "vehicle"
    BICYCLE = "bicycle"
    ANIMAL = "animal"
    FIXED_DEVICE = "fixed_device"
    MOBILE_DEVICE = "mobile_device"
    UNKNOWN = "unknown"


# Canonical list as plain strings for quick membership checks
TARGET_TYPES: list[str] = [t.value for t in TargetType]

# Sub-types for richer classification
PERSON_SUBTYPES = ["pedestrian", "runner", "commuter", "resident", "visitor", "loiterer"]
VEHICLE_SUBTYPES = ["car", "truck", "bus", "motorcycle", "emergency"]
BICYCLE_SUBTYPES = ["bicycle", "e_bike", "scooter"]
ANIMAL_SUBTYPES = ["dog", "cat", "bird", "wildlife"]
DEVICE_SUBTYPES = ["phone", "tablet", "laptop", "watch", "earbuds", "speaker",
                   "camera", "sensor", "router", "beacon", "tag"]


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------

@dataclass
class ClassificationVote:
    """A single classifier's opinion about a target's type.

    Attributes
    ----------
    target_type : str
        One of :data:`TARGET_TYPES`.
    subtype : str
        More specific type (e.g. ``"pedestrian"`` for person).
    confidence : float
        0.0 – 1.0, how sure the classifier is.
    evidence : str
        Human-readable explanation of *why* this vote was cast.
    source : str
        Name of the classifier that produced this vote.
    """

    target_type: str = "unknown"
    subtype: str = ""
    confidence: float = 0.0
    evidence: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type,
            "subtype": self.subtype,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source": self.source,
        }


@dataclass
class ClassificationResult:
    """Final classification output after ensemble voting.

    Attributes
    ----------
    target_type : str
        Winning type from :data:`TARGET_TYPES`.
    subtype : str
        Most specific subtype (may be empty).
    confidence : float
        Aggregated confidence (0.0 – 1.0).
    evidence : list[str]
        All evidence strings from contributing votes.
    votes : list[ClassificationVote]
        Raw votes from every classifier that participated.
    timestamp : float
        Time the classification was produced (monotonic).
    """

    target_type: str = "unknown"
    subtype: str = ""
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    votes: list[ClassificationVote] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type,
            "subtype": self.subtype,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "votes": [v.to_dict() for v in self.votes],
            "timestamp": self.timestamp,
        }


@dataclass
class TargetObservation:
    """A bundle of sensor data about a target, fed to classifiers.

    Populate only the fields you have — classifiers gracefully skip missing data.

    Attributes
    ----------
    target_id : str
        Unique target identifier (e.g. ``"ble_AA:BB:CC:DD:EE:FF"``).
    ble_name : str
        BLE advertised name (e.g. ``"iPhone 15"``).
    ble_mac : str
        BLE MAC address.
    ble_services : list[str]
        BLE service UUIDs advertised.
    ble_manufacturer : str
        Manufacturer from OUI lookup.
    ble_appearance : int
        GAP appearance code (0 = unset).
    wifi_ssid : str
        WiFi SSID from probe or association.
    wifi_bssid : str
        WiFi BSSID / MAC.
    wifi_probe_count : int
        Number of distinct probe requests seen.
    wifi_is_randomized : bool
        Whether the WiFi MAC appears randomized.
    speed_mps : float
        Current speed in metres per second (negative = unknown).
    avg_speed_mps : float
        Average speed over observation window (negative = unknown).
    max_speed_mps : float
        Maximum observed speed (negative = unknown).
    is_stationary : bool
        True if target has not moved meaningfully.
    movement_pattern : str
        Detected movement pattern (loitering, patrol, transit, erratic, stationary).
    dwell_seconds : float
        How long the target has lingered at its current position.
    hour_of_day : int
        Local hour (0-23) when observed (-1 = unknown).
    day_of_week : int
        Day of week (0=Mon, 6=Sun, -1 = unknown).
    visit_count : int
        How many times this target has been seen in this area.
    device_type_hint : str
        Hint from DeviceClassifier (phone, watch, etc.) if available.
    """

    target_id: str = ""
    # BLE signals
    ble_name: str = ""
    ble_mac: str = ""
    ble_services: list[str] = field(default_factory=list)
    ble_manufacturer: str = ""
    ble_appearance: int = 0
    # WiFi signals
    wifi_ssid: str = ""
    wifi_bssid: str = ""
    wifi_probe_count: int = 0
    wifi_is_randomized: bool = False
    # Speed / movement
    speed_mps: float = -1.0
    avg_speed_mps: float = -1.0
    max_speed_mps: float = -1.0
    is_stationary: bool = False
    movement_pattern: str = ""
    dwell_seconds: float = 0.0
    # Temporal
    hour_of_day: int = -1
    day_of_week: int = -1
    visit_count: int = 0
    # Cross-classifier hint
    device_type_hint: str = ""


# ---------------------------------------------------------------------------
# Classifier protocol
# ---------------------------------------------------------------------------

class Classifier(Protocol):
    """Duck-typed protocol for classifiers.

    Any object with a ``classify`` method that accepts a
    :class:`TargetObservation` and returns a :class:`ClassificationVote`
    (or ``None`` to abstain) satisfies this protocol.
    """

    @property
    def name(self) -> str: ...  # pragma: no cover

    def classify(self, obs: TargetObservation) -> Optional[ClassificationVote]: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Built-in classifiers
# ---------------------------------------------------------------------------

# Speed thresholds (m/s) — based on real-world data
_SPEED_STATIONARY = 0.3    # below this → stationary
_SPEED_PEDESTRIAN = 2.5    # typical walking ~1.4 m/s, max ~2.5 m/s
_SPEED_RUNNER = 5.0        # running ~3-5 m/s
_SPEED_BICYCLE = 12.0      # cycling ~4-10 m/s, e-bike up to ~12
_SPEED_VEHICLE = 60.0      # cars on surface streets < 60 m/s (~135 mph)

# BLE name → (target_type, subtype, confidence)
_BLE_TARGET_PATTERNS: list[tuple[str, str, str, float]] = [
    (r"(?i)^iPhone", "mobile_device", "phone", 0.90),
    (r"(?i)^Samsung", "mobile_device", "phone", 0.85),
    (r"(?i)^Pixel", "mobile_device", "phone", 0.85),
    (r"(?i)^Galaxy\s?(S|A|Z|Note|Fold|Flip)", "mobile_device", "phone", 0.85),
    (r"(?i)^OnePlus", "mobile_device", "phone", 0.85),
    (r"(?i)^Xiaomi", "mobile_device", "phone", 0.75),
    (r"(?i)^Huawei", "mobile_device", "phone", 0.75),
    (r"(?i)^OPPO", "mobile_device", "phone", 0.75),
    (r"(?i)^Galaxy\s?Watch", "mobile_device", "watch", 0.90),
    (r"(?i)Watch", "mobile_device", "watch", 0.80),
    (r"(?i)^Fitbit", "mobile_device", "watch", 0.90),
    (r"(?i)^Garmin", "mobile_device", "watch", 0.90),
    (r"(?i)AirPod", "mobile_device", "earbuds", 0.95),
    (r"(?i)^Galaxy\s?Buds", "mobile_device", "earbuds", 0.90),
    (r"(?i)^Bose", "mobile_device", "earbuds", 0.80),
    (r"(?i)^Sony.*W[HF]", "mobile_device", "earbuds", 0.80),
    (r"(?i)^JBL", "mobile_device", "speaker", 0.80),
    (r"(?i)^Sonos", "fixed_device", "speaker", 0.85),
    (r"(?i)^HomePod", "fixed_device", "speaker", 0.90),
    (r"(?i)^Echo", "fixed_device", "speaker", 0.85),
    (r"(?i)^Google Home", "fixed_device", "speaker", 0.85),
    (r"(?i)MacBook", "mobile_device", "laptop", 0.90),
    (r"(?i)^iPad", "mobile_device", "tablet", 0.90),
    (r"(?i)^Galaxy\s?Tab", "mobile_device", "tablet", 0.85),
    (r"(?i)^Fire.*HD", "mobile_device", "tablet", 0.80),
    (r"(?i)^Tile", "mobile_device", "tag", 0.90),
    (r"(?i)^AirTag", "mobile_device", "tag", 0.95),
    (r"(?i)^Chipolo", "mobile_device", "tag", 0.90),
    (r"(?i)^Tesla", "vehicle", "car", 0.85),
    (r"(?i)^Wyze", "fixed_device", "camera", 0.80),
    (r"(?i)^Ring", "fixed_device", "camera", 0.80),
    (r"(?i)^Nest", "fixed_device", "sensor", 0.75),
    (r"(?i)^Govee", "fixed_device", "sensor", 0.75),
    (r"(?i)^ESP32", "fixed_device", "sensor", 0.85),
    (r"(?i)^Raspberry", "fixed_device", "sensor", 0.80),
]

# BLE appearance code ranges → (target_type, subtype)
_APPEARANCE_MAP: dict[int, tuple[str, str]] = {
    # Generic categories (Bluetooth SIG assigned numbers)
    64: ("mobile_device", "phone"),      # Generic Phone
    192: ("mobile_device", "watch"),     # Generic Watch
    193: ("mobile_device", "watch"),     # Sports Watch
    961: ("mobile_device", "earbuds"),   # Generic HID — often earbuds
    # More specific
    832: ("mobile_device", "watch"),     # Heart Rate Sensor → fitness device
    1024: ("fixed_device", "sensor"),    # Generic Outdoor Sports Activity
    3136: ("mobile_device", "tag"),      # Generic Tag
}

# WiFi SSID patterns → (target_type, subtype, confidence)
_WIFI_TARGET_PATTERNS: list[tuple[str, str, str, float]] = [
    (r"(?i)^iPhone", "mobile_device", "phone", 0.90),
    (r"(?i)^Android[_\- ]", "mobile_device", "phone", 0.85),
    (r"(?i)^Galaxy[_\- ]", "mobile_device", "phone", 0.85),
    (r"(?i)^Pixel[_\- ]", "mobile_device", "phone", 0.85),
    (r"(?i)MacBook", "mobile_device", "laptop", 0.85),
    (r"(?i)^LAPTOP-", "mobile_device", "laptop", 0.80),
    (r"(?i)^DESKTOP-", "fixed_device", "laptop", 0.75),
    (r"(?i)^DIRECT-.*Print", "fixed_device", "sensor", 0.80),
    (r"(?i)^HP-", "fixed_device", "sensor", 0.70),
    (r"(?i)^ChromeCast", "fixed_device", "sensor", 0.75),
    (r"(?i)^Roku", "fixed_device", "sensor", 0.75),
    (r"(?i)Tesla", "vehicle", "car", 0.75),
    (r"(?i)^Ring[_\- ]", "fixed_device", "camera", 0.80),
    (r"(?i)^Nest[_\- ]", "fixed_device", "sensor", 0.75),
]

# Manufacturer → likely target type (OUI-based)
_MANUFACTURER_MAP: dict[str, tuple[str, str, float]] = {
    "apple": ("mobile_device", "phone", 0.65),
    "samsung": ("mobile_device", "phone", 0.60),
    "google": ("mobile_device", "phone", 0.55),
    "espressif": ("fixed_device", "sensor", 0.70),
    "raspberry pi": ("fixed_device", "sensor", 0.70),
    "philips lighting": ("fixed_device", "sensor", 0.65),
    "texas instruments": ("fixed_device", "sensor", 0.55),
    "nordic semiconductor": ("fixed_device", "sensor", 0.55),
}


class BLETypeClassifier:
    """Classifies target type from BLE advertisement data.

    Uses device name patterns, GAP appearance codes, manufacturer OUI,
    and service UUIDs to determine whether a BLE signal comes from a
    phone (proxy for person), fixed sensor, vehicle beacon, etc.
    """

    name: str = "ble_type"

    def classify(self, obs: TargetObservation) -> Optional[ClassificationVote]:
        """Classify based on BLE signals.  Returns None if no BLE data."""
        if (not obs.ble_name and not obs.ble_mac and not obs.ble_services
                and obs.ble_appearance == 0 and not obs.ble_manufacturer
                and not obs.device_type_hint):
            return None

        best: Optional[ClassificationVote] = None

        # 1) Name pattern matching (highest priority)
        if obs.ble_name:
            for pattern, ttype, subtype, conf in _BLE_TARGET_PATTERNS:
                if re.search(pattern, obs.ble_name):
                    vote = ClassificationVote(
                        target_type=ttype,
                        subtype=subtype,
                        confidence=conf,
                        evidence=f"BLE name '{obs.ble_name}' matches pattern for {subtype}",
                        source=self.name,
                    )
                    if best is None or vote.confidence > best.confidence:
                        best = vote
                    break  # first match wins for name patterns

        # 2) GAP appearance code
        if obs.ble_appearance and obs.ble_appearance in _APPEARANCE_MAP:
            ttype, subtype = _APPEARANCE_MAP[obs.ble_appearance]
            conf = 0.75
            vote = ClassificationVote(
                target_type=ttype,
                subtype=subtype,
                confidence=conf,
                evidence=f"BLE appearance code 0x{obs.ble_appearance:04X} maps to {subtype}",
                source=self.name,
            )
            if best is None or vote.confidence > best.confidence:
                best = vote

        # 3) Manufacturer (from OUI)
        if obs.ble_manufacturer:
            mfr_lower = obs.ble_manufacturer.lower()
            for mfr_key, (ttype, subtype, conf) in _MANUFACTURER_MAP.items():
                if mfr_key in mfr_lower:
                    vote = ClassificationVote(
                        target_type=ttype,
                        subtype=subtype,
                        confidence=conf,
                        evidence=f"BLE manufacturer '{obs.ble_manufacturer}' suggests {ttype}/{subtype}",
                        source=self.name,
                    )
                    if best is None or vote.confidence > best.confidence:
                        best = vote
                    break

        # 4) Device type hint from DeviceClassifier
        if not best and obs.device_type_hint:
            ttype, subtype, conf = _device_hint_to_target(obs.device_type_hint)
            if ttype != "unknown":
                best = ClassificationVote(
                    target_type=ttype,
                    subtype=subtype,
                    confidence=conf,
                    evidence=f"Device type hint '{obs.device_type_hint}' mapped to {ttype}",
                    source=self.name,
                )

        return best


class WiFiTypeClassifier:
    """Classifies target type from WiFi probe / association data."""

    name: str = "wifi_type"

    def classify(self, obs: TargetObservation) -> Optional[ClassificationVote]:
        """Classify based on WiFi signals.  Returns None if no WiFi data."""
        if (not obs.wifi_ssid and not obs.wifi_bssid
                and obs.wifi_probe_count == 0 and not obs.wifi_is_randomized):
            return None

        best: Optional[ClassificationVote] = None

        # 1) SSID pattern matching
        if obs.wifi_ssid:
            for pattern, ttype, subtype, conf in _WIFI_TARGET_PATTERNS:
                if re.search(pattern, obs.wifi_ssid):
                    best = ClassificationVote(
                        target_type=ttype,
                        subtype=subtype,
                        confidence=conf,
                        evidence=f"WiFi SSID '{obs.wifi_ssid}' matches pattern for {subtype}",
                        source=self.name,
                    )
                    break

        # 2) Probe count heuristic — lots of probes suggests a mobile device
        if obs.wifi_probe_count > 0 and best is None:
            if obs.wifi_probe_count >= 5:
                best = ClassificationVote(
                    target_type="mobile_device",
                    subtype="phone",
                    confidence=0.45,
                    evidence=f"WiFi probe count {obs.wifi_probe_count} suggests mobile device",
                    source=self.name,
                )
            else:
                best = ClassificationVote(
                    target_type="unknown",
                    subtype="",
                    confidence=0.20,
                    evidence=f"WiFi probe count {obs.wifi_probe_count} is low — inconclusive",
                    source=self.name,
                )

        # 3) Randomized MAC suggests modern phone/laptop
        if obs.wifi_is_randomized and best is None:
            best = ClassificationVote(
                target_type="mobile_device",
                subtype="",
                confidence=0.40,
                evidence="WiFi MAC is randomized — likely modern phone or laptop",
                source=self.name,
            )

        return best


class SpeedClassifier:
    """Classifies target type based on movement speed.

    Speed thresholds (m/s):
      - stationary < 0.3
      - pedestrian < 2.5
      - runner < 5.0
      - bicycle < 12.0
      - vehicle >= 12.0

    Uses ``speed_mps`` (instantaneous), ``avg_speed_mps``, and ``max_speed_mps``
    to determine the most likely type.  Prefers average speed for robustness,
    falls back to instantaneous.
    """

    name: str = "speed"

    def classify(self, obs: TargetObservation) -> Optional[ClassificationVote]:
        """Classify from speed.  Returns None if no speed data."""
        # Pick best available speed metric
        speed = obs.avg_speed_mps if obs.avg_speed_mps >= 0 else obs.speed_mps
        if speed < 0:
            return None

        max_speed = obs.max_speed_mps if obs.max_speed_mps >= 0 else speed

        if speed < _SPEED_STATIONARY:
            # Stationary — could be anything; low-confidence guess
            return ClassificationVote(
                target_type="unknown",
                subtype="",
                confidence=0.15,
                evidence=f"Stationary (speed {speed:.2f} m/s) — type indeterminate",
                source=self.name,
            )

        if speed < _SPEED_PEDESTRIAN:
            # Walking speed → person
            conf = 0.70 if speed >= 0.8 else 0.50
            return ClassificationVote(
                target_type="person",
                subtype="pedestrian",
                confidence=conf,
                evidence=f"Walking speed ({speed:.1f} m/s) consistent with pedestrian",
                source=self.name,
            )

        if speed < _SPEED_RUNNER:
            # Running speed → person (runner)
            return ClassificationVote(
                target_type="person",
                subtype="runner",
                confidence=0.65,
                evidence=f"Running speed ({speed:.1f} m/s) consistent with runner",
                source=self.name,
            )

        if speed < _SPEED_BICYCLE:
            # Bicycle range — but could be slow vehicle in traffic
            conf = 0.55
            # If max speed ever exceeded bicycle range, it is a vehicle
            if max_speed >= _SPEED_BICYCLE:
                return ClassificationVote(
                    target_type="vehicle",
                    subtype="car",
                    confidence=0.60,
                    evidence=(
                        f"Current speed {speed:.1f} m/s in bicycle range but max "
                        f"{max_speed:.1f} m/s exceeds bicycle threshold"
                    ),
                    source=self.name,
                )
            return ClassificationVote(
                target_type="bicycle",
                subtype="bicycle",
                confidence=conf,
                evidence=f"Speed {speed:.1f} m/s consistent with bicycle",
                source=self.name,
            )

        # Above bicycle threshold → vehicle
        conf = min(0.85, 0.60 + (speed - _SPEED_BICYCLE) * 0.005)
        return ClassificationVote(
            target_type="vehicle",
            subtype="car",
            confidence=conf,
            evidence=f"Speed {speed:.1f} m/s consistent with motor vehicle",
            source=self.name,
        )


class BehaviorClassifier:
    """Classifies target type from movement patterns.

    Movement patterns (from ``MovementPatternAnalyzer`` or similar):
      - loitering → person
      - patrol    → person or vehicle
      - transit   → person or vehicle (depends on speed)
      - erratic   → animal or person
      - stationary → fixed_device (long) or anything (short)
    """

    name: str = "behavior"

    # Minimum dwell seconds before stationary → fixed_device
    FIXED_DWELL_THRESHOLD: float = 3600.0  # 1 hour

    def classify(self, obs: TargetObservation) -> Optional[ClassificationVote]:
        """Classify from movement pattern and dwell time."""
        pattern = obs.movement_pattern.lower().strip() if obs.movement_pattern else ""
        if not pattern and obs.dwell_seconds <= 0:
            return None

        # Long-duration stationary → fixed device
        if obs.is_stationary and obs.dwell_seconds >= self.FIXED_DWELL_THRESHOLD:
            return ClassificationVote(
                target_type="fixed_device",
                subtype="",
                confidence=0.65,
                evidence=f"Stationary for {obs.dwell_seconds:.0f}s — likely fixed device",
                source=self.name,
            )

        if pattern == "loitering":
            return ClassificationVote(
                target_type="person",
                subtype="loiterer",
                confidence=0.60,
                evidence="Loitering pattern — consistent with person lingering",
                source=self.name,
            )

        if pattern == "patrol":
            # Patrol at walking speed → person; otherwise vehicle
            speed = obs.avg_speed_mps if obs.avg_speed_mps >= 0 else obs.speed_mps
            if 0 <= speed < _SPEED_RUNNER:
                return ClassificationVote(
                    target_type="person",
                    subtype="pedestrian",
                    confidence=0.55,
                    evidence="Patrol pattern at walking speed — likely person on foot",
                    source=self.name,
                )
            return ClassificationVote(
                target_type="vehicle",
                subtype="car",
                confidence=0.50,
                evidence="Patrol pattern at driving speed — likely patrol vehicle",
                source=self.name,
            )

        if pattern == "erratic":
            return ClassificationVote(
                target_type="animal",
                subtype="",
                confidence=0.40,
                evidence="Erratic movement — possibly an animal",
                source=self.name,
            )

        if pattern == "transit":
            speed = obs.avg_speed_mps if obs.avg_speed_mps >= 0 else obs.speed_mps
            if 0 <= speed < _SPEED_BICYCLE:
                return ClassificationVote(
                    target_type="person",
                    subtype="commuter",
                    confidence=0.45,
                    evidence="Transit pattern at low speed — likely commuter on foot",
                    source=self.name,
                )
            return ClassificationVote(
                target_type="vehicle",
                subtype="car",
                confidence=0.50,
                evidence="Transit pattern at high speed — likely vehicle in transit",
                source=self.name,
            )

        if pattern == "stationary":
            # Short stationary → could be anything
            return ClassificationVote(
                target_type="unknown",
                subtype="",
                confidence=0.15,
                evidence="Stationary — insufficient data to classify",
                source=self.name,
            )

        return None


class TimeClassifier:
    """Classifies target type from time-of-day and visit frequency.

    Heuristics:
      - Repeated visits at the same hours → resident or commuter
      - Night-time movement → resident or animal
      - Single visit → visitor
      - Weekday rush-hour → commuter
    """

    name: str = "time"

    # Rush hour windows (local hour)
    MORNING_RUSH = (6, 9)
    EVENING_RUSH = (16, 19)
    NIGHT_START = 22
    NIGHT_END = 5

    def classify(self, obs: TargetObservation) -> Optional[ClassificationVote]:
        """Classify from temporal signals."""
        hour = obs.hour_of_day
        if hour < 0:
            return None

        visits = obs.visit_count

        # Night-time activity
        if hour >= self.NIGHT_START or hour <= self.NIGHT_END:
            if visits >= 5:
                return ClassificationVote(
                    target_type="person",
                    subtype="resident",
                    confidence=0.50,
                    evidence=f"Night activity (hour {hour}) with {visits} visits — likely resident",
                    source=self.name,
                )
            if visits <= 1:
                return ClassificationVote(
                    target_type="unknown",
                    subtype="",
                    confidence=0.25,
                    evidence=f"Night activity (hour {hour}) single visit — inconclusive",
                    source=self.name,
                )
            return ClassificationVote(
                target_type="person",
                subtype="visitor",
                confidence=0.35,
                evidence=f"Night activity (hour {hour}) with {visits} visits",
                source=self.name,
            )

        # Rush-hour commuter detection
        is_rush = (self.MORNING_RUSH[0] <= hour < self.MORNING_RUSH[1] or
                   self.EVENING_RUSH[0] <= hour < self.EVENING_RUSH[1])

        if is_rush and visits >= 3:
            # Weekday check: 0=Mon..4=Fri are workdays
            if 0 <= obs.day_of_week <= 4:
                return ClassificationVote(
                    target_type="person",
                    subtype="commuter",
                    confidence=0.55,
                    evidence=f"Rush hour (hour {hour}) weekday, {visits} visits — likely commuter",
                    source=self.name,
                )
            return ClassificationVote(
                target_type="person",
                subtype="visitor",
                confidence=0.35,
                evidence=f"Rush hour (hour {hour}) weekend, {visits} visits",
                source=self.name,
            )

        # Frequent visitor → person (resident)
        if visits >= 10:
            return ClassificationVote(
                target_type="person",
                subtype="resident",
                confidence=0.45,
                evidence=f"{visits} visits — frequent presence suggests resident",
                source=self.name,
            )

        # Few visits during normal hours
        if visits <= 1:
            return ClassificationVote(
                target_type="unknown",
                subtype="",
                confidence=0.10,
                evidence=f"Single visit at hour {hour} — not enough data",
                source=self.name,
            )

        return ClassificationVote(
            target_type="person",
            subtype="visitor",
            confidence=0.30,
            evidence=f"{visits} visits at hour {hour} — occasional visitor",
            source=self.name,
        )


# ---------------------------------------------------------------------------
# Helper: map DeviceClassifier device_type → target type
# ---------------------------------------------------------------------------

def _device_hint_to_target(device_type: str) -> tuple[str, str, float]:
    """Map a DeviceClassifier device_type string to (target_type, subtype, conf).

    Returns ("unknown", "", 0.0) if unmapped.
    """
    dt = device_type.lower()
    _map: dict[str, tuple[str, str, float]] = {
        "phone": ("mobile_device", "phone", 0.70),
        "tablet": ("mobile_device", "tablet", 0.70),
        "computer": ("mobile_device", "laptop", 0.65),
        "watch": ("mobile_device", "watch", 0.70),
        "fitness": ("mobile_device", "watch", 0.65),
        "earbuds": ("mobile_device", "earbuds", 0.70),
        "headphones": ("mobile_device", "earbuds", 0.65),
        "speaker": ("fixed_device", "speaker", 0.65),
        "smart_speaker": ("fixed_device", "speaker", 0.70),
        "smart_home": ("fixed_device", "sensor", 0.60),
        "camera": ("fixed_device", "camera", 0.70),
        "tag": ("mobile_device", "tag", 0.70),
        "microcontroller": ("fixed_device", "sensor", 0.65),
        "vehicle": ("vehicle", "car", 0.65),
        "printer": ("fixed_device", "sensor", 0.60),
        "media_player": ("fixed_device", "sensor", 0.55),
        "gamepad": ("mobile_device", "", 0.50),
        "vr_headset": ("mobile_device", "", 0.55),
        "hotspot": ("fixed_device", "router", 0.50),
    }
    return _map.get(dt, ("unknown", "", 0.0))


# ---------------------------------------------------------------------------
# Ensemble classifier
# ---------------------------------------------------------------------------

class EnsembleClassifier:
    """Combines votes from multiple classifiers using confidence-weighted voting.

    For each target type, sums the confidence from every vote that supports it.
    The type with the highest total confidence wins.  If two types tie, the
    one with more supporting votes wins.  If still tied, ``"unknown"`` loses
    to any concrete type.

    The final confidence is the weighted-average confidence of votes that
    agreed with the winning type, clamped to [0, 1].

    Parameters
    ----------
    classifiers : list
        Classifier instances to poll.  Each must have a ``name`` attribute
        and a ``classify(obs) -> ClassificationVote | None`` method.
    """

    name: str = "ensemble"

    def __init__(self, classifiers: list | None = None) -> None:
        self._classifiers: list = list(classifiers or [])

    def add_classifier(self, clf) -> None:
        """Register an additional classifier."""
        self._classifiers.append(clf)

    def remove_classifier(self, name: str) -> bool:
        """Remove a classifier by name.  Returns True if found."""
        before = len(self._classifiers)
        self._classifiers = [c for c in self._classifiers if getattr(c, "name", "") != name]
        return len(self._classifiers) < before

    @property
    def classifiers(self) -> list:
        """Return a copy of registered classifiers."""
        return list(self._classifiers)

    def classify(self, obs: TargetObservation) -> ClassificationResult:
        """Run all classifiers and produce a combined result."""
        votes: list[ClassificationVote] = []
        for clf in self._classifiers:
            try:
                vote = clf.classify(obs)
                if vote is not None:
                    votes.append(vote)
            except Exception as exc:
                logger.warning("Classifier %s failed: %s", getattr(clf, "name", "?"), exc)

        if not votes:
            return ClassificationResult(
                target_type="unknown",
                subtype="",
                confidence=0.0,
                evidence=["No classifiers produced votes"],
                votes=[],
            )

        # Aggregate confidence per target type
        type_scores: dict[str, float] = {}
        type_counts: dict[str, int] = {}
        type_subtypes: dict[str, dict[str, float]] = {}  # type → {subtype → total_conf}

        for v in votes:
            tt = v.target_type
            type_scores[tt] = type_scores.get(tt, 0.0) + v.confidence
            type_counts[tt] = type_counts.get(tt, 0) + 1
            if v.subtype:
                subs = type_subtypes.setdefault(tt, {})
                subs[v.subtype] = subs.get(v.subtype, 0.0) + v.confidence

        # Pick winner: highest total confidence, break ties by count,
        # then prefer non-unknown.
        def _sort_key(item: tuple[str, float]) -> tuple[float, int, int]:
            tt, score = item
            count = type_counts.get(tt, 0)
            is_known = 0 if tt == "unknown" else 1
            return (score, count, is_known)

        winner_type = max(type_scores.items(), key=_sort_key)[0]

        # Pick best subtype for winner
        winner_subtype = ""
        if winner_type in type_subtypes:
            winner_subtype = max(type_subtypes[winner_type].items(),
                                key=lambda x: x[1])[0]

        # Compute final confidence: weighted average of agreeing votes
        agreeing = [v for v in votes if v.target_type == winner_type]
        if agreeing:
            total_conf = sum(v.confidence for v in agreeing)
            final_confidence = total_conf / len(agreeing)
            # Boost slightly for multi-source agreement
            if len(agreeing) >= 2:
                agreement_boost = min(0.10, 0.03 * len(agreeing))
                final_confidence = min(1.0, final_confidence + agreement_boost)
        else:
            final_confidence = 0.0

        evidence = [v.evidence for v in votes if v.evidence]

        return ClassificationResult(
            target_type=winner_type,
            subtype=winner_subtype,
            confidence=round(final_confidence, 4),
            evidence=evidence,
            votes=votes,
        )


# ---------------------------------------------------------------------------
# Pipeline — high-level orchestrator
# ---------------------------------------------------------------------------

class ClassificationPipeline:
    """End-to-end target classification pipeline.

    Creates a default :class:`EnsembleClassifier` pre-loaded with all
    built-in classifiers.  Call :meth:`classify` with a
    :class:`TargetObservation` to get a :class:`ClassificationResult`.

    Parameters
    ----------
    classifiers : list, optional
        Override the default classifier set.  If ``None``, all built-in
        classifiers are registered.
    """

    def __init__(self, classifiers: list | None = None) -> None:
        if classifiers is not None:
            self._ensemble = EnsembleClassifier(classifiers)
        else:
            self._ensemble = EnsembleClassifier([
                BLETypeClassifier(),
                WiFiTypeClassifier(),
                SpeedClassifier(),
                BehaviorClassifier(),
                TimeClassifier(),
            ])

    @property
    def ensemble(self) -> EnsembleClassifier:
        """Access the underlying ensemble."""
        return self._ensemble

    def add_classifier(self, clf) -> None:
        """Register an additional classifier."""
        self._ensemble.add_classifier(clf)

    def classify(self, obs: TargetObservation) -> ClassificationResult:
        """Classify a target from its observations."""
        return self._ensemble.classify(obs)

    def classify_many(self, observations: list[TargetObservation]) -> list[ClassificationResult]:
        """Classify multiple targets.  Returns results in the same order."""
        return [self.classify(obs) for obs in observations]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enums / constants
    "TargetType",
    "TARGET_TYPES",
    "PERSON_SUBTYPES",
    "VEHICLE_SUBTYPES",
    "BICYCLE_SUBTYPES",
    "ANIMAL_SUBTYPES",
    "DEVICE_SUBTYPES",
    # Data carriers
    "ClassificationVote",
    "ClassificationResult",
    "TargetObservation",
    # Protocol
    "Classifier",
    # Built-in classifiers
    "BLETypeClassifier",
    "WiFiTypeClassifier",
    "SpeedClassifier",
    "BehaviorClassifier",
    "TimeClassifier",
    # Ensemble + pipeline
    "EnsembleClassifier",
    "ClassificationPipeline",
]

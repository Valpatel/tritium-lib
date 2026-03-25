# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BehaviorProfiler — builds comprehensive behavioral profiles from long-term observation.

Constructs multi-dimensional profiles for tracked targets based on accumulated
observation data across temporal, spatial, social, and device dimensions. Uses
pure statistics (histograms, entropy, centroids, z-scores) — no ML dependencies.

Profile dimensions:
  * **Temporal** — active hours, day-of-week patterns, regularity score
  * **Spatial** — home area, work area, transit corridors, frequent stops
  * **Social** — typical group size, communication frequency, associations
  * **Device** — device types carried, signal patterns, MAC rotation detection

Role classification:
  resident, worker, commuter, visitor, delivery, patrol

Change detection:
  Flags when a profile deviates significantly from its historical baseline.

Usage::

    from tritium_lib.intelligence.behavior_profiler import (
        BehaviorProfiler, BehaviorProfile, ProfileComparison,
    )

    profiler = BehaviorProfiler()

    # Feed observations over time
    profiler.add_observation("ble_aa:bb:cc", Observation(
        timestamp=time.time(), lat=40.7128, lng=-74.0060,
        source="ble", device_type="phone",
    ))

    # Build profile
    profile = profiler.build_profile("ble_aa:bb:cc")

    # Classify role
    role = profiler.classify_role(profile)

    # Detect change
    changes = profiler.detect_change(profile)

    # Compare two profiles
    comparison = ProfileComparison.compare(profile_a, profile_b)
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOUR_BINS = 24
DOW_BINS = 7

# Minimum observations to consider a profile meaningful
MIN_OBSERVATIONS_FOR_PROFILE = 5

# Minimum observations to detect change
MIN_OBSERVATIONS_FOR_CHANGE = 10

# Spatial clustering: observations within this distance (meters) are the same stop
STOP_CLUSTER_RADIUS_M = 50.0

# Home area: most visited stop during night hours (22-06)
HOME_HOURS = set(range(22, 24)) | set(range(0, 6))

# Work area: most visited stop during work hours (09-17)
WORK_HOURS = set(range(9, 17))

# Transit corridor minimum points
MIN_CORRIDOR_POINTS = 3

# Change detection: z-score threshold
CHANGE_Z_THRESHOLD = 2.5

# MAC rotation: threshold for distinct MACs per device type
MAC_ROTATION_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetRole(str, Enum):
    """Classified role based on behavioral profile."""

    RESIDENT = "resident"
    WORKER = "worker"
    COMMUTER = "commuter"
    VISITOR = "visitor"
    DELIVERY = "delivery"
    PATROL = "patrol"
    UNKNOWN = "unknown"


class ChangeSeverity(str, Enum):
    """Severity of a detected behavioral change."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A single observation of a target."""

    timestamp: float  # epoch seconds
    lat: float = 0.0
    lng: float = 0.0
    source: str = ""  # ble, wifi, camera, mesh, etc.
    device_type: str = ""  # phone, watch, laptop, vehicle, etc.
    mac_address: str = ""  # MAC or device ID for rotation detection
    group_size: int = 1  # how many targets co-located at observation time
    association_ids: list[str] = field(default_factory=list)  # other target IDs seen with


@dataclass
class SpatialStop:
    """A frequently visited spatial location."""

    lat: float
    lng: float
    visit_count: int = 0
    total_dwell_s: float = 0.0
    label: str = ""  # "home", "work", "stop_N"
    first_visit: float = 0.0
    last_visit: float = 0.0

    def to_dict(self) -> dict:
        return {
            "lat": round(self.lat, 6),
            "lng": round(self.lng, 6),
            "visit_count": self.visit_count,
            "total_dwell_s": round(self.total_dwell_s, 1),
            "label": self.label,
            "first_visit": self.first_visit,
            "last_visit": self.last_visit,
        }


@dataclass
class TransitCorridor:
    """A frequently traversed path between two stops."""

    start_stop: str  # label of origin stop
    end_stop: str  # label of destination stop
    trip_count: int = 0
    avg_duration_s: float = 0.0
    avg_speed_mps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start_stop": self.start_stop,
            "end_stop": self.end_stop,
            "trip_count": self.trip_count,
            "avg_duration_s": round(self.avg_duration_s, 1),
            "avg_speed_mps": round(self.avg_speed_mps, 2),
        }


@dataclass
class TemporalDimension:
    """Temporal behavioral profile: when the target is active."""

    hourly_histogram: list[int] = field(default_factory=lambda: [0] * HOUR_BINS)
    dow_histogram: list[int] = field(default_factory=lambda: [0] * DOW_BINS)
    regularity_score: float = 0.0  # 0 = random, 1 = perfectly predictable
    peak_hours: list[int] = field(default_factory=list)
    quiet_hours: list[int] = field(default_factory=list)
    is_daytime: bool = False
    is_nighttime: bool = False
    active_span_hours: float = 0.0  # hours between first and last active hour

    def to_dict(self) -> dict:
        return {
            "hourly_histogram": list(self.hourly_histogram),
            "dow_histogram": list(self.dow_histogram),
            "regularity_score": round(self.regularity_score, 4),
            "peak_hours": self.peak_hours,
            "quiet_hours": self.quiet_hours,
            "is_daytime": self.is_daytime,
            "is_nighttime": self.is_nighttime,
            "active_span_hours": round(self.active_span_hours, 1),
        }


@dataclass
class SpatialDimension:
    """Spatial behavioral profile: where the target goes."""

    home_area: Optional[SpatialStop] = None
    work_area: Optional[SpatialStop] = None
    frequent_stops: list[SpatialStop] = field(default_factory=list)
    transit_corridors: list[TransitCorridor] = field(default_factory=list)
    total_area_m2: float = 0.0  # bounding box area of all observations
    centroid_lat: float = 0.0
    centroid_lng: float = 0.0

    def to_dict(self) -> dict:
        return {
            "home_area": self.home_area.to_dict() if self.home_area else None,
            "work_area": self.work_area.to_dict() if self.work_area else None,
            "frequent_stops": [s.to_dict() for s in self.frequent_stops],
            "transit_corridors": [c.to_dict() for c in self.transit_corridors],
            "total_area_m2": round(self.total_area_m2, 1),
            "centroid_lat": round(self.centroid_lat, 6),
            "centroid_lng": round(self.centroid_lng, 6),
        }


@dataclass
class SocialDimension:
    """Social behavioral profile: who the target interacts with."""

    avg_group_size: float = 0.0
    max_group_size: int = 0
    unique_associates: int = 0
    top_associates: list[tuple[str, int]] = field(default_factory=list)  # (target_id, co_count)
    communication_frequency: float = 0.0  # associations per hour of activity
    is_loner: bool = True  # typically alone
    is_social: bool = False  # frequently in groups

    def to_dict(self) -> dict:
        return {
            "avg_group_size": round(self.avg_group_size, 2),
            "max_group_size": self.max_group_size,
            "unique_associates": self.unique_associates,
            "top_associates": [
                {"target_id": tid, "count": cnt}
                for tid, cnt in self.top_associates
            ],
            "communication_frequency": round(self.communication_frequency, 4),
            "is_loner": self.is_loner,
            "is_social": self.is_social,
        }


@dataclass
class DeviceDimension:
    """Device behavioral profile: what equipment the target uses."""

    device_types: list[str] = field(default_factory=list)  # unique types seen
    source_types: list[str] = field(default_factory=list)  # ble, wifi, camera, etc.
    primary_device: str = ""  # most frequently observed device type
    mac_count: int = 0  # number of distinct MACs observed
    mac_rotation_detected: bool = False  # privacy MAC rotation
    signal_patterns: dict[str, int] = field(default_factory=dict)  # source -> observation count

    def to_dict(self) -> dict:
        return {
            "device_types": list(self.device_types),
            "source_types": list(self.source_types),
            "primary_device": self.primary_device,
            "mac_count": self.mac_count,
            "mac_rotation_detected": self.mac_rotation_detected,
            "signal_patterns": dict(self.signal_patterns),
        }


@dataclass
class BehaviorChange:
    """A detected significant change in behavior."""

    dimension: str  # "temporal", "spatial", "social", "device"
    description: str
    severity: ChangeSeverity
    z_score: float = 0.0
    old_value: str = ""
    new_value: str = ""

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "description": self.description,
            "severity": self.severity.value,
            "z_score": round(self.z_score, 2),
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


@dataclass
class BehaviorProfile:
    """Comprehensive behavioral profile for a target.

    Aggregates temporal, spatial, social, and device dimensions into a
    single profile that characterizes a target's long-term behavior.
    """

    target_id: str
    temporal: TemporalDimension = field(default_factory=TemporalDimension)
    spatial: SpatialDimension = field(default_factory=SpatialDimension)
    social: SocialDimension = field(default_factory=SocialDimension)
    device: DeviceDimension = field(default_factory=DeviceDimension)
    role: TargetRole = TargetRole.UNKNOWN
    role_confidence: float = 0.0
    observation_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    profile_age_days: float = 0.0

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "temporal": self.temporal.to_dict(),
            "spatial": self.spatial.to_dict(),
            "social": self.social.to_dict(),
            "device": self.device.to_dict(),
            "role": self.role.value,
            "role_confidence": round(self.role_confidence, 3),
            "observation_count": self.observation_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "profile_age_days": round(self.profile_age_days, 1),
        }

    def export(self) -> dict:
        """Export profile as JSON-serializable dict."""
        return self.to_dict()


# ---------------------------------------------------------------------------
# Profile comparison
# ---------------------------------------------------------------------------

@dataclass
class ProfileComparison:
    """Result of comparing two behavioral profiles for similarity."""

    target_a: str
    target_b: str
    temporal_similarity: float = 0.0  # 0 = different, 1 = identical
    spatial_similarity: float = 0.0
    social_similarity: float = 0.0
    device_similarity: float = 0.0
    overall_similarity: float = 0.0
    same_role: bool = False
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target_a": self.target_a,
            "target_b": self.target_b,
            "temporal_similarity": round(self.temporal_similarity, 4),
            "spatial_similarity": round(self.spatial_similarity, 4),
            "social_similarity": round(self.social_similarity, 4),
            "device_similarity": round(self.device_similarity, 4),
            "overall_similarity": round(self.overall_similarity, 4),
            "same_role": self.same_role,
            "details": self.details,
        }

    @staticmethod
    def compare(
        a: BehaviorProfile,
        b: BehaviorProfile,
        weights: Optional[dict[str, float]] = None,
    ) -> ProfileComparison:
        """Compare two profiles and return a similarity assessment.

        Args:
            a: First profile.
            b: Second profile.
            weights: Optional dimension weights. Defaults to equal weighting.

        Returns:
            ProfileComparison with per-dimension and overall similarity scores.
        """
        w = weights or {
            "temporal": 0.3,
            "spatial": 0.3,
            "social": 0.2,
            "device": 0.2,
        }

        temporal_sim = _histogram_similarity(
            a.temporal.hourly_histogram, b.temporal.hourly_histogram
        )
        spatial_sim = _spatial_similarity(a.spatial, b.spatial)
        social_sim = _social_similarity(a.social, b.social)
        device_sim = _device_similarity(a.device, b.device)

        overall = (
            w.get("temporal", 0.25) * temporal_sim
            + w.get("spatial", 0.25) * spatial_sim
            + w.get("social", 0.25) * social_sim
            + w.get("device", 0.25) * device_sim
        )

        return ProfileComparison(
            target_a=a.target_id,
            target_b=b.target_id,
            temporal_similarity=temporal_sim,
            spatial_similarity=spatial_sim,
            social_similarity=social_sim,
            device_similarity=device_sim,
            overall_similarity=overall,
            same_role=(a.role == b.role and a.role != TargetRole.UNKNOWN),
        )


# ---------------------------------------------------------------------------
# Profiler engine
# ---------------------------------------------------------------------------

class BehaviorProfiler:
    """Builds behavioral profiles from accumulated observation data.

    Thread-safe. Feed observations over time, then build profiles
    that aggregate temporal, spatial, social, and device behavior.
    """

    def __init__(self) -> None:
        self._observations: dict[str, list[Observation]] = defaultdict(list)
        self._profiles: dict[str, BehaviorProfile] = {}
        import threading
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Observation ingestion
    # ------------------------------------------------------------------

    def add_observation(self, target_id: str, obs: Observation) -> None:
        """Record a single observation for a target.

        Args:
            target_id: Unique target identifier.
            obs: The observation to record.
        """
        with self._lock:
            self._observations[target_id].append(obs)

    def add_observations(
        self, target_id: str, observations: Sequence[Observation]
    ) -> None:
        """Record multiple observations for a target at once."""
        with self._lock:
            self._observations[target_id].extend(observations)

    def observation_count(self, target_id: str) -> int:
        """Return the number of observations recorded for a target."""
        with self._lock:
            return len(self._observations.get(target_id, []))

    def known_targets(self) -> list[str]:
        """Return list of all target IDs with observations."""
        with self._lock:
            return list(self._observations.keys())

    # ------------------------------------------------------------------
    # Profile building
    # ------------------------------------------------------------------

    def build_profile(self, target_id: str) -> BehaviorProfile:
        """Build a comprehensive behavioral profile from all observations.

        Args:
            target_id: Target to build a profile for.

        Returns:
            BehaviorProfile with all dimensions populated.
        """
        with self._lock:
            obs_list = list(self._observations.get(target_id, []))

        if not obs_list:
            return BehaviorProfile(target_id=target_id)

        obs_list.sort(key=lambda o: o.timestamp)

        temporal = self._build_temporal(obs_list)
        spatial = self._build_spatial(obs_list)
        social = self._build_social(obs_list)
        device = self._build_device(obs_list)

        first_seen = obs_list[0].timestamp
        last_seen = obs_list[-1].timestamp
        age_days = (last_seen - first_seen) / 86400.0

        profile = BehaviorProfile(
            target_id=target_id,
            temporal=temporal,
            spatial=spatial,
            social=social,
            device=device,
            observation_count=len(obs_list),
            first_seen=first_seen,
            last_seen=last_seen,
            profile_age_days=max(0.0, age_days),
        )

        # Classify role
        profile.role, profile.role_confidence = self._classify_role(profile)

        with self._lock:
            self._profiles[target_id] = profile

        return profile

    def get_profile(self, target_id: str) -> Optional[BehaviorProfile]:
        """Return a previously built profile, or None."""
        with self._lock:
            return self._profiles.get(target_id)

    def build_all_profiles(self) -> dict[str, BehaviorProfile]:
        """Build profiles for all known targets."""
        targets = self.known_targets()
        result = {}
        for tid in targets:
            result[tid] = self.build_profile(tid)
        return result

    # ------------------------------------------------------------------
    # Role classification
    # ------------------------------------------------------------------

    def classify_role(self, profile: BehaviorProfile) -> tuple[TargetRole, float]:
        """Classify a target's role from its behavioral profile.

        Returns:
            Tuple of (role, confidence).
        """
        return self._classify_role(profile)

    def _classify_role(self, profile: BehaviorProfile) -> tuple[TargetRole, float]:
        """Internal role classifier using scoring heuristics."""
        if profile.observation_count < MIN_OBSERVATIONS_FOR_PROFILE:
            return TargetRole.UNKNOWN, 0.0

        scores: dict[TargetRole, float] = {role: 0.0 for role in TargetRole}

        t = profile.temporal
        s = profile.spatial
        soc = profile.social

        # --- Resident signals ---
        # Present during night hours, has a home area
        if s.home_area is not None:
            scores[TargetRole.RESIDENT] += 0.3
        night_activity = sum(
            t.hourly_histogram[h] for h in HOME_HOURS if h < HOUR_BINS
        )
        total_activity = sum(t.hourly_histogram)
        if total_activity > 0 and night_activity / total_activity > 0.3:
            scores[TargetRole.RESIDENT] += 0.2
        if t.regularity_score > 0.5:
            scores[TargetRole.RESIDENT] += 0.1

        # --- Worker signals ---
        # Active during work hours, has a work area
        if s.work_area is not None:
            scores[TargetRole.WORKER] += 0.3
        work_activity = sum(
            t.hourly_histogram[h] for h in WORK_HOURS if h < HOUR_BINS
        )
        if total_activity > 0 and work_activity / total_activity > 0.5:
            scores[TargetRole.WORKER] += 0.2
        # Workers typically have strong day-of-week patterns (weekdays)
        weekday_activity = sum(t.dow_histogram[d] for d in range(5))
        weekend_activity = sum(t.dow_histogram[d] for d in range(5, 7))
        if total_activity > 0 and weekday_activity > weekend_activity * 2:
            scores[TargetRole.WORKER] += 0.1

        # --- Commuter signals ---
        # Has both home and work, uses transit corridors
        if s.home_area is not None and s.work_area is not None:
            scores[TargetRole.COMMUTER] += 0.3
        if len(s.transit_corridors) > 0:
            scores[TargetRole.COMMUTER] += 0.2
        if t.regularity_score > 0.6:
            scores[TargetRole.COMMUTER] += 0.1

        # --- Visitor signals ---
        # Low observation count relative to profile age, few stops
        if profile.profile_age_days > 0:
            visit_rate = profile.observation_count / max(profile.profile_age_days, 0.01)
            if visit_rate < 2.0:  # less than 2 observations per day
                scores[TargetRole.VISITOR] += 0.2
        if len(s.frequent_stops) <= 2:
            scores[TargetRole.VISITOR] += 0.15
        if s.home_area is None and s.work_area is None:
            scores[TargetRole.VISITOR] += 0.15

        # --- Delivery signals ---
        # Many frequent stops, moderate speed, brief dwell times
        if len(s.frequent_stops) > 5:
            scores[TargetRole.DELIVERY] += 0.25
        short_dwells = sum(
            1 for stop in s.frequent_stops if 0 < stop.total_dwell_s < 600
        )
        if len(s.frequent_stops) > 0 and short_dwells / len(s.frequent_stops) > 0.5:
            scores[TargetRole.DELIVERY] += 0.2
        if total_activity > 0 and work_activity / total_activity > 0.6:
            scores[TargetRole.DELIVERY] += 0.1

        # --- Patrol signals ---
        # High regularity, repeated routes, full coverage
        if t.regularity_score > 0.7:
            scores[TargetRole.PATROL] += 0.2
        if len(s.transit_corridors) >= 2:
            scores[TargetRole.PATROL] += 0.15
        # Patrols often have wide spatial coverage
        if s.total_area_m2 > 10000:
            scores[TargetRole.PATROL] += 0.15
        # Patrols tend to be loners
        if soc.is_loner:
            scores[TargetRole.PATROL] += 0.1

        # Remove UNKNOWN from scoring
        scores.pop(TargetRole.UNKNOWN, None)

        if not scores:
            return TargetRole.UNKNOWN, 0.0

        # Pick the highest-scoring role
        best_role = max(scores, key=lambda r: scores[r])
        best_score = scores[best_role]

        if best_score < 0.15:
            return TargetRole.UNKNOWN, 0.0

        # Confidence is the best score normalized by the maximum possible
        confidence = min(1.0, best_score / 0.6)
        return best_role, round(confidence, 3)

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def detect_change(self, profile: BehaviorProfile) -> list[BehaviorChange]:
        """Detect significant behavioral changes by comparing the recent
        window of observations against the historical baseline.

        The profile must have been built already. This method splits
        observations into an older baseline (first 70%) and a recent
        window (last 30%) and compares statistical properties.

        Args:
            profile: A previously built BehaviorProfile.

        Returns:
            List of detected BehaviorChange objects.
        """
        with self._lock:
            obs_list = list(self._observations.get(profile.target_id, []))

        if len(obs_list) < MIN_OBSERVATIONS_FOR_CHANGE:
            return []

        obs_list.sort(key=lambda o: o.timestamp)

        # Split 70/30
        split = max(1, int(len(obs_list) * 0.7))
        baseline_obs = obs_list[:split]
        recent_obs = obs_list[split:]

        if len(recent_obs) < 2 or len(baseline_obs) < 2:
            return []

        changes: list[BehaviorChange] = []

        # Temporal change: compare hourly distributions
        base_hours = _build_hourly_histogram(baseline_obs)
        recent_hours = _build_hourly_histogram(recent_obs)
        temporal_change = _distribution_shift(base_hours, recent_hours)
        if temporal_change > CHANGE_Z_THRESHOLD:
            changes.append(BehaviorChange(
                dimension="temporal",
                description="Activity hours have shifted significantly",
                severity=ChangeSeverity.MEDIUM if temporal_change < 4.0 else ChangeSeverity.HIGH,
                z_score=temporal_change,
                old_value=f"peak_hour={_peak_bin(base_hours)}",
                new_value=f"peak_hour={_peak_bin(recent_hours)}",
            ))

        # Spatial change: compare centroid shift
        base_centroid = _centroid(baseline_obs)
        recent_centroid = _centroid(recent_obs)
        if base_centroid and recent_centroid:
            centroid_dist_m = _haversine_m(
                base_centroid[0], base_centroid[1],
                recent_centroid[0], recent_centroid[1],
            )
            base_spread = _spatial_spread_m(baseline_obs, base_centroid)
            if base_spread > 0:
                spatial_z = centroid_dist_m / base_spread
                if spatial_z > CHANGE_Z_THRESHOLD:
                    changes.append(BehaviorChange(
                        dimension="spatial",
                        description="Spatial center has shifted significantly",
                        severity=ChangeSeverity.MEDIUM if spatial_z < 4.0 else ChangeSeverity.HIGH,
                        z_score=spatial_z,
                        old_value=f"centroid=({base_centroid[0]:.4f},{base_centroid[1]:.4f})",
                        new_value=f"centroid=({recent_centroid[0]:.4f},{recent_centroid[1]:.4f})",
                    ))

        # Social change: compare average group size
        base_group = _mean([o.group_size for o in baseline_obs])
        recent_group = _mean([o.group_size for o in recent_obs])
        base_group_std = _std([o.group_size for o in baseline_obs])
        # When baseline is constant (std=0), any change is significant
        if base_group_std > 0:
            group_z = abs(recent_group - base_group) / base_group_std
        elif abs(recent_group - base_group) > 0.5:
            # Constant baseline but recent mean differs — treat as significant
            group_z = CHANGE_Z_THRESHOLD + 1.0
        else:
            group_z = 0.0
        if group_z > CHANGE_Z_THRESHOLD:
            changes.append(BehaviorChange(
                dimension="social",
                description="Group size pattern has changed",
                severity=ChangeSeverity.LOW if group_z < 4.0 else ChangeSeverity.MEDIUM,
                z_score=group_z,
                old_value=f"avg_group={base_group:.1f}",
                new_value=f"avg_group={recent_group:.1f}",
            ))

        # Device change: new device types appearing
        base_devices = set(o.device_type for o in baseline_obs if o.device_type)
        recent_devices = set(o.device_type for o in recent_obs if o.device_type)
        new_devices = recent_devices - base_devices
        if new_devices:
            changes.append(BehaviorChange(
                dimension="device",
                description=f"New device types observed: {', '.join(sorted(new_devices))}",
                severity=ChangeSeverity.LOW,
                z_score=0.0,
                old_value=",".join(sorted(base_devices)),
                new_value=",".join(sorted(recent_devices)),
            ))

        # Device change: MAC rotation started or stopped
        base_macs = set(o.mac_address for o in baseline_obs if o.mac_address)
        recent_macs = set(o.mac_address for o in recent_obs if o.mac_address)
        base_rotating = len(base_macs) >= MAC_ROTATION_THRESHOLD
        recent_rotating = len(recent_macs) >= MAC_ROTATION_THRESHOLD
        if base_rotating != recent_rotating:
            changes.append(BehaviorChange(
                dimension="device",
                description="MAC rotation behavior changed",
                severity=ChangeSeverity.MEDIUM,
                z_score=0.0,
                old_value=f"rotating={base_rotating}",
                new_value=f"rotating={recent_rotating}",
            ))

        return changes

    # ------------------------------------------------------------------
    # Dimension builders
    # ------------------------------------------------------------------

    def _build_temporal(self, obs_list: list[Observation]) -> TemporalDimension:
        """Build temporal dimension from observations."""
        hourly = [0] * HOUR_BINS
        dow = [0] * DOW_BINS

        for obs in obs_list:
            try:
                import datetime as dt_mod
                dt_val = dt_mod.datetime.fromtimestamp(obs.timestamp)
                hourly[dt_val.hour] += 1
                dow[dt_val.weekday()] += 1
            except (OSError, ValueError, OverflowError):
                continue

        total = sum(hourly)
        regularity = _normalized_entropy_score(hourly)

        peak_hours = _top_bins(hourly, n=3)
        quiet_hours = [h for h, c in enumerate(hourly) if c == 0]

        # Daytime: all activity between 6-18
        day_sum = sum(hourly[6:18])
        night_sum = total - day_sum
        is_daytime = total > 0 and day_sum == total
        is_nighttime = total > 0 and night_sum == total

        # Active span
        active_indices = [h for h, c in enumerate(hourly) if c > 0]
        if active_indices:
            active_span = active_indices[-1] - active_indices[0] + 1
        else:
            active_span = 0

        return TemporalDimension(
            hourly_histogram=hourly,
            dow_histogram=dow,
            regularity_score=regularity,
            peak_hours=peak_hours,
            quiet_hours=quiet_hours,
            is_daytime=is_daytime,
            is_nighttime=is_nighttime,
            active_span_hours=float(active_span),
        )

    def _build_spatial(self, obs_list: list[Observation]) -> SpatialDimension:
        """Build spatial dimension from observations."""
        points = [(o.lat, o.lng, o.timestamp) for o in obs_list if o.lat != 0.0 or o.lng != 0.0]

        if not points:
            return SpatialDimension()

        # Centroid
        c_lat = _mean([p[0] for p in points])
        c_lng = _mean([p[1] for p in points])

        # Bounding box area (approximate)
        lats = [p[0] for p in points]
        lngs = [p[1] for p in points]
        lat_range = max(lats) - min(lats)
        lng_range = max(lngs) - min(lngs)
        # Convert degrees to meters approximately
        lat_m = lat_range * 111320.0
        lng_m = lng_range * 111320.0 * math.cos(math.radians(c_lat))
        area_m2 = lat_m * lng_m

        # Cluster points into stops
        stops = _cluster_stops(points, STOP_CLUSTER_RADIUS_M)

        # Identify home and work areas
        home_area = _identify_area(stops, obs_list, HOME_HOURS, "home")
        work_area = _identify_area(stops, obs_list, WORK_HOURS, "work")

        # Label remaining stops
        for i, stop in enumerate(stops):
            if stop.label == "":
                stop.label = f"stop_{i}"

        # Build transit corridors from consecutive stop visits
        corridors = _build_corridors(obs_list, stops)

        return SpatialDimension(
            home_area=home_area,
            work_area=work_area,
            frequent_stops=stops,
            transit_corridors=corridors,
            total_area_m2=area_m2,
            centroid_lat=c_lat,
            centroid_lng=c_lng,
        )

    def _build_social(self, obs_list: list[Observation]) -> SocialDimension:
        """Build social dimension from observations."""
        group_sizes = [o.group_size for o in obs_list]
        avg_group = _mean(group_sizes) if group_sizes else 0.0
        max_group = max(group_sizes) if group_sizes else 0

        # Count associations
        assoc_counter: Counter[str] = Counter()
        for obs in obs_list:
            for aid in obs.association_ids:
                assoc_counter[aid] += 1

        unique_associates = len(assoc_counter)
        top_associates = assoc_counter.most_common(10)

        # Communication frequency: associations per hour of observation
        if len(obs_list) >= 2:
            time_span_h = (obs_list[-1].timestamp - obs_list[0].timestamp) / 3600.0
            total_associations = sum(len(o.association_ids) for o in obs_list)
            comm_freq = total_associations / max(time_span_h, 0.01)
        else:
            comm_freq = 0.0

        is_loner = avg_group <= 1.2 and unique_associates < 3
        is_social = avg_group >= 2.0 or unique_associates >= 5

        return SocialDimension(
            avg_group_size=avg_group,
            max_group_size=max_group,
            unique_associates=unique_associates,
            top_associates=top_associates,
            communication_frequency=comm_freq,
            is_loner=is_loner,
            is_social=is_social,
        )

    def _build_device(self, obs_list: list[Observation]) -> DeviceDimension:
        """Build device dimension from observations."""
        device_counter: Counter[str] = Counter()
        source_counter: Counter[str] = Counter()
        mac_set: set[str] = set()

        for obs in obs_list:
            if obs.device_type:
                device_counter[obs.device_type] += 1
            if obs.source:
                source_counter[obs.source] += 1
            if obs.mac_address:
                mac_set.add(obs.mac_address)

        device_types = sorted(device_counter.keys())
        source_types = sorted(source_counter.keys())
        primary = device_counter.most_common(1)[0][0] if device_counter else ""
        mac_count = len(mac_set)
        mac_rotation = mac_count >= MAC_ROTATION_THRESHOLD

        return DeviceDimension(
            device_types=device_types,
            source_types=source_types,
            primary_device=primary,
            mac_count=mac_count,
            mac_rotation_detected=mac_rotation,
            signal_patterns=dict(source_counter),
        )


# ---------------------------------------------------------------------------
# Statistical helpers (module-level, pure functions)
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean of a sequence. Returns 0.0 if empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    """Population standard deviation. Returns 0.0 if fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _normalized_entropy_score(histogram: list[int]) -> float:
    """Compute a regularity score from a histogram via normalized entropy.

    1.0 = all activity concentrated in one bin (perfectly predictable).
    0.0 = uniform distribution (completely random).
    """
    total = sum(histogram)
    if total == 0:
        return 0.0

    n_bins = len(histogram)
    if n_bins <= 1:
        return 1.0

    max_entropy = math.log(n_bins)
    if max_entropy == 0:
        return 1.0

    entropy = 0.0
    for count in histogram:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return round(1.0 - (entropy / max_entropy), 4)


def _top_bins(histogram: list[int], n: int = 3) -> list[int]:
    """Return indices of the top-N bins by count."""
    indexed = [(count, idx) for idx, count in enumerate(histogram)]
    indexed.sort(reverse=True)
    return [idx for count, idx in indexed[:n] if count > 0]


def _peak_bin(histogram: list[int]) -> int:
    """Return the index of the bin with the highest count."""
    if not histogram or max(histogram) == 0:
        return 0
    return histogram.index(max(histogram))


def _histogram_similarity(a: list[int], b: list[int]) -> float:
    """Cosine similarity between two histograms. 0.0 to 1.0."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def _spatial_similarity(a: SpatialDimension, b: SpatialDimension) -> float:
    """Similarity between two spatial profiles based on centroid distance and area overlap."""
    if (a.centroid_lat == 0.0 and a.centroid_lng == 0.0) or \
       (b.centroid_lat == 0.0 and b.centroid_lng == 0.0):
        return 0.0

    dist_m = _haversine_m(a.centroid_lat, a.centroid_lng, b.centroid_lat, b.centroid_lng)

    # Similarity decays with distance: 1.0 at 0m, 0.5 at 500m, ~0 at 2km
    spatial_score = math.exp(-dist_m / 500.0)

    # Bonus for shared stop labels (home-home, work-work)
    a_labels = set(s.label for s in a.frequent_stops)
    b_labels = set(s.label for s in b.frequent_stops)
    shared = len(a_labels & b_labels)
    total = max(len(a_labels | b_labels), 1)
    label_score = shared / total

    return 0.7 * spatial_score + 0.3 * label_score


def _social_similarity(a: SocialDimension, b: SocialDimension) -> float:
    """Similarity between two social profiles."""
    # Compare group size patterns
    max_group = max(a.avg_group_size, b.avg_group_size, 1.0)
    group_sim = 1.0 - abs(a.avg_group_size - b.avg_group_size) / max_group

    # Compare associate overlap
    a_assoc = set(tid for tid, _ in a.top_associates)
    b_assoc = set(tid for tid, _ in b.top_associates)
    if a_assoc or b_assoc:
        shared = len(a_assoc & b_assoc)
        total = max(len(a_assoc | b_assoc), 1)
        assoc_sim = shared / total
    else:
        assoc_sim = 1.0 if a.is_loner and b.is_loner else 0.5

    # Loner/social match
    style_match = 1.0 if a.is_loner == b.is_loner and a.is_social == b.is_social else 0.0

    return 0.4 * group_sim + 0.3 * assoc_sim + 0.3 * style_match


def _device_similarity(a: DeviceDimension, b: DeviceDimension) -> float:
    """Similarity between two device profiles."""
    # Device type overlap
    a_types = set(a.device_types)
    b_types = set(b.device_types)
    if a_types or b_types:
        shared = len(a_types & b_types)
        total = max(len(a_types | b_types), 1)
        type_sim = shared / total
    else:
        type_sim = 1.0

    # Source type overlap
    a_src = set(a.source_types)
    b_src = set(b.source_types)
    if a_src or b_src:
        shared = len(a_src & b_src)
        total = max(len(a_src | b_src), 1)
        src_sim = shared / total
    else:
        src_sim = 1.0

    # MAC rotation match
    rotation_match = 1.0 if a.mac_rotation_detected == b.mac_rotation_detected else 0.0

    return 0.4 * type_sim + 0.3 * src_sim + 0.3 * rotation_match


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in meters between two lat/lng points."""
    R = 6_371_000.0  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _centroid(obs_list: list[Observation]) -> Optional[tuple[float, float]]:
    """Compute the geographic centroid of observations."""
    pts = [(o.lat, o.lng) for o in obs_list if o.lat != 0.0 or o.lng != 0.0]
    if not pts:
        return None
    return (_mean([p[0] for p in pts]), _mean([p[1] for p in pts]))


def _spatial_spread_m(obs_list: list[Observation], centroid: tuple[float, float]) -> float:
    """Compute the average distance from observations to a centroid, in meters."""
    distances = []
    for o in obs_list:
        if o.lat != 0.0 or o.lng != 0.0:
            d = _haversine_m(o.lat, o.lng, centroid[0], centroid[1])
            distances.append(d)
    return _std(distances) if distances else 0.0


def _build_hourly_histogram(obs_list: list[Observation]) -> list[int]:
    """Build an hourly histogram from observations."""
    import datetime as dt_mod
    hourly = [0] * HOUR_BINS
    for obs in obs_list:
        try:
            dt_val = dt_mod.datetime.fromtimestamp(obs.timestamp)
            hourly[dt_val.hour] += 1
        except (OSError, ValueError, OverflowError):
            continue
    return hourly


def _distribution_shift(baseline: list[int], recent: list[int]) -> float:
    """Compute a z-score-like measure of how much two distributions differ.

    Uses chi-squared-like statistic normalized by baseline variance.
    """
    base_total = sum(baseline)
    recent_total = sum(recent)

    if base_total == 0 or recent_total == 0:
        return 0.0

    # Normalize to proportions
    base_prop = [c / base_total for c in baseline]
    recent_prop = [c / recent_total for c in recent]

    # Sum of squared differences
    ssd = sum((bp - rp) ** 2 for bp, rp in zip(base_prop, recent_prop))

    # Scale: sqrt(ssd * n_bins) gives a reasonable z-like score
    return math.sqrt(ssd * len(baseline)) * 10.0


def _cluster_stops(
    points: list[tuple[float, float, float]],
    radius_m: float,
) -> list[SpatialStop]:
    """Cluster geographic points into stops using a greedy approach.

    Each cluster is represented as a SpatialStop centered on the mean
    of its member points.
    """
    if not points:
        return []

    assigned = [False] * len(points)
    stops: list[SpatialStop] = []

    for i in range(len(points)):
        if assigned[i]:
            continue

        cluster_lats = [points[i][0]]
        cluster_lngs = [points[i][1]]
        cluster_times = [points[i][2]]
        assigned[i] = True

        # Find all unassigned points within radius
        c_lat = points[i][0]
        c_lng = points[i][1]

        for j in range(i + 1, len(points)):
            if assigned[j]:
                continue
            d = _haversine_m(c_lat, c_lng, points[j][0], points[j][1])
            if d <= radius_m:
                cluster_lats.append(points[j][0])
                cluster_lngs.append(points[j][1])
                cluster_times.append(points[j][2])
                assigned[j] = True

        stop_lat = _mean(cluster_lats)
        stop_lng = _mean(cluster_lngs)
        times_sorted = sorted(cluster_times)

        # Estimate dwell time: sum of consecutive time differences
        dwell_s = 0.0
        for k in range(1, len(times_sorted)):
            gap = times_sorted[k] - times_sorted[k - 1]
            # Only count gaps < 1 hour as dwell (otherwise it's a separate visit)
            if gap < 3600:
                dwell_s += gap

        stops.append(SpatialStop(
            lat=stop_lat,
            lng=stop_lng,
            visit_count=len(cluster_lats),
            total_dwell_s=dwell_s,
            first_visit=times_sorted[0],
            last_visit=times_sorted[-1],
        ))

    # Sort by visit count descending
    stops.sort(key=lambda s: s.visit_count, reverse=True)
    return stops


def _identify_area(
    stops: list[SpatialStop],
    obs_list: list[Observation],
    target_hours: set[int],
    label: str,
) -> Optional[SpatialStop]:
    """Identify the most visited stop during specified hours and label it."""
    import datetime as dt_mod

    # Count observations per stop during target hours
    stop_scores: dict[int, int] = defaultdict(int)

    for obs in obs_list:
        if obs.lat == 0.0 and obs.lng == 0.0:
            continue
        try:
            dt_val = dt_mod.datetime.fromtimestamp(obs.timestamp)
            if dt_val.hour not in target_hours:
                continue
        except (OSError, ValueError, OverflowError):
            continue

        # Find nearest stop
        best_idx = -1
        best_dist = float("inf")
        for idx, stop in enumerate(stops):
            d = _haversine_m(obs.lat, obs.lng, stop.lat, stop.lng)
            if d < best_dist:
                best_dist = d
                best_idx = idx

        if best_idx >= 0 and best_dist < STOP_CLUSTER_RADIUS_M * 2:
            stop_scores[best_idx] += 1

    if not stop_scores:
        return None

    best_stop_idx = max(stop_scores, key=lambda k: stop_scores[k])
    best_stop = stops[best_stop_idx]

    # Only label if it has meaningful activity
    if stop_scores[best_stop_idx] < 2:
        return None

    best_stop.label = label
    return best_stop


def _build_corridors(
    obs_list: list[Observation],
    stops: list[SpatialStop],
) -> list[TransitCorridor]:
    """Identify transit corridors from sequences of stop visits."""
    if len(stops) < 2:
        return []

    # Map each observation to its nearest stop
    stop_sequence: list[tuple[str, float]] = []
    for obs in obs_list:
        if obs.lat == 0.0 and obs.lng == 0.0:
            continue
        best_label = ""
        best_dist = float("inf")
        for stop in stops:
            d = _haversine_m(obs.lat, obs.lng, stop.lat, stop.lng)
            if d < best_dist:
                best_dist = d
                best_label = stop.label
        if best_label and best_dist < STOP_CLUSTER_RADIUS_M * 2:
            stop_sequence.append((best_label, obs.timestamp))

    # Extract transitions between different consecutive stops
    corridor_stats: dict[tuple[str, str], list[float]] = defaultdict(list)
    for i in range(1, len(stop_sequence)):
        prev_label, prev_time = stop_sequence[i - 1]
        curr_label, curr_time = stop_sequence[i]
        if prev_label != curr_label and curr_time > prev_time:
            duration = curr_time - prev_time
            # Only count transitions under 2 hours
            if 0 < duration < 7200:
                corridor_stats[(prev_label, curr_label)].append(duration)

    corridors: list[TransitCorridor] = []
    for (start, end), durations in corridor_stats.items():
        if len(durations) < MIN_CORRIDOR_POINTS:
            continue
        avg_dur = _mean(durations)

        # Estimate distance between stops
        start_stop = next((s for s in stops if s.label == start), None)
        end_stop = next((s for s in stops if s.label == end), None)
        if start_stop and end_stop:
            dist_m = _haversine_m(start_stop.lat, start_stop.lng, end_stop.lat, end_stop.lng)
            avg_speed = dist_m / max(avg_dur, 1.0)
        else:
            avg_speed = 0.0

        corridors.append(TransitCorridor(
            start_stop=start,
            end_stop=end,
            trip_count=len(durations),
            avg_duration_s=avg_dur,
            avg_speed_mps=avg_speed,
        ))

    corridors.sort(key=lambda c: c.trip_count, reverse=True)
    return corridors

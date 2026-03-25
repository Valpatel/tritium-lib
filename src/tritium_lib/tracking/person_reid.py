# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Person re-identification engine — match the same person across sensors and time.

Fuses BLE MAC rotation detection, camera appearance timing, spatial
consistency, and signal fingerprint matching to resolve unique person
identities even when individual sensor IDs change (e.g. BLE MAC rotation,
camera re-assignment after occlusion).

The engine builds PersonProfile objects from TrackedTarget data and scores
candidate matches using four strategies:

  1. **BLE MAC rotation** — detects when a BLE device disappears and a new
     MAC appears with similar RSSI pattern, probe behavior, and timing.
  2. **Temporal co-occurrence** — one ID disappears at time T, a new ID
     appears at T + delta in the same vicinity.
  3. **Spatial consistency** — given last known position and speed/heading,
     is the candidate's position reachable within the time gap?
  4. **Signal fingerprint** — RSSI pattern, probe SSID set overlap, and
     advertisement interval matching.

Usage::

    engine = ReIDEngine(tracker=tracker, dossier_store=dossier_store)
    profile = engine.create_profile("ble_aabbccddeeff")
    matches = engine.find_matches(profile, candidates=engine.get_candidate_profiles())
    if matches and matches[0].score >= 0.7:
        engine.merge_identities(profile.target_id, matches[0].target_id)
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .dossier import DossierStore, TargetDossier
from .target_history import TargetHistory
from .target_tracker import TargetTracker, TrackedTarget

logger = logging.getLogger("person-reid")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum time gap (seconds) for temporal co-occurrence matching
MAX_TEMPORAL_GAP: float = 60.0

# Maximum walking speed in meters/second (~5 km/h) for spatial plausibility
MAX_WALKING_SPEED: float = 1.8

# Minimum overall match score to consider a positive re-identification
DEFAULT_MATCH_THRESHOLD: float = 0.55

# RSSI similarity tolerance (dBm) — within this delta means "close enough"
RSSI_TOLERANCE: float = 12.0

# BLE advertisement interval tolerance (ms)
ADV_INTERVAL_TOLERANCE: float = 50.0

# Strategy weights — how much each signal contributes to the final score
STRATEGY_WEIGHTS: dict[str, float] = {
    "ble_mac_rotation": 0.30,
    "temporal_cooccurrence": 0.25,
    "spatial_consistency": 0.25,
    "signal_fingerprint": 0.20,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PersonProfile:
    """Aggregated sensor profile for a tracked person or device.

    Built from TrackedTarget data plus optional enrichments (RSSI history,
    probe SSIDs, advertisement intervals).
    """

    target_id: str
    source: str = ""
    asset_type: str = ""
    alliance: str = "unknown"
    classification: str = ""

    # Spatial
    position: tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    speed: float = 0.0

    # Temporal
    first_seen: float = 0.0
    last_seen: float = 0.0
    signal_count: int = 0

    # BLE-specific
    rssi_mean: float = -80.0
    rssi_std: float = 10.0
    adv_interval_ms: float = 0.0
    probe_ssids: set[str] = field(default_factory=set)
    oui_prefix: str = ""

    # Trail (recent positions)
    trail: list[tuple[float, float, float]] = field(default_factory=list)

    # Correlated IDs already merged into this profile
    correlated_ids: list[str] = field(default_factory=list)
    confirming_sources: set[str] = field(default_factory=set)

    # Position confidence
    position_confidence: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "target_id": self.target_id,
            "source": self.source,
            "asset_type": self.asset_type,
            "alliance": self.alliance,
            "classification": self.classification,
            "position": {"x": self.position[0], "y": self.position[1]},
            "heading": self.heading,
            "speed": self.speed,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "signal_count": self.signal_count,
            "rssi_mean": self.rssi_mean,
            "rssi_std": self.rssi_std,
            "adv_interval_ms": self.adv_interval_ms,
            "probe_ssids": sorted(self.probe_ssids),
            "oui_prefix": self.oui_prefix,
            "correlated_ids": list(self.correlated_ids),
            "confirming_sources": sorted(self.confirming_sources),
            "position_confidence": self.position_confidence,
        }


@dataclass
class MatchResult:
    """Result of comparing two PersonProfiles."""

    target_id: str  # the candidate that was matched
    score: float  # overall weighted score 0.0 to 1.0
    strategy_scores: dict[str, float] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "score": round(self.score, 4),
            "strategy_scores": {k: round(v, 4) for k, v in self.strategy_scores.items()},
            "detail": self.detail,
        }


@dataclass
class MergeRecord:
    """Record of a completed identity merge."""

    merge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    primary_id: str = ""
    secondary_id: str = ""
    score: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)
    dossier_uuid: str = ""


# ---------------------------------------------------------------------------
# Scoring functions — one per strategy
# ---------------------------------------------------------------------------

def _score_ble_mac_rotation(
    query: PersonProfile,
    candidate: PersonProfile,
) -> float:
    """Detect BLE MAC rotation — same device, new randomized MAC.

    Indicators of rotation:
      - Both are BLE sources
      - One disappeared shortly before the other appeared
      - Same OUI prefix (vendor) or both random MACs
      - Similar RSSI signature (same physical distance from sensors)
      - Similar advertisement interval
    """
    # Only meaningful for BLE-sourced targets
    if query.source != "ble" or candidate.source != "ble":
        return 0.0

    score = 0.0
    components = 0

    # OUI match — if both have the same prefix, more likely same vendor
    if query.oui_prefix and candidate.oui_prefix:
        components += 1
        if query.oui_prefix == candidate.oui_prefix:
            score += 1.0
        else:
            score += 0.0
    elif not query.oui_prefix and not candidate.oui_prefix:
        # Both random MACs — mildly supportive
        components += 1
        score += 0.4

    # RSSI similarity
    components += 1
    rssi_diff = abs(query.rssi_mean - candidate.rssi_mean)
    if rssi_diff <= RSSI_TOLERANCE:
        score += 1.0 - (rssi_diff / RSSI_TOLERANCE)
    else:
        score += 0.0

    # Advertisement interval similarity
    if query.adv_interval_ms > 0 and candidate.adv_interval_ms > 0:
        components += 1
        interval_diff = abs(query.adv_interval_ms - candidate.adv_interval_ms)
        if interval_diff <= ADV_INTERVAL_TOLERANCE:
            score += 1.0 - (interval_diff / ADV_INTERVAL_TOLERANCE)

    # Temporal gap — disappearance/appearance overlap
    # query.last_seen should be close to candidate.first_seen (or vice versa)
    gap = abs(query.last_seen - candidate.first_seen)
    reverse_gap = abs(candidate.last_seen - query.first_seen)
    min_gap = min(gap, reverse_gap)
    components += 1
    if min_gap <= MAX_TEMPORAL_GAP:
        score += 1.0 - (min_gap / MAX_TEMPORAL_GAP)

    if components == 0:
        return 0.0
    return min(1.0, score / components)


def _score_temporal_cooccurrence(
    query: PersonProfile,
    candidate: PersonProfile,
) -> float:
    """Score based on disappearance/appearance timing overlap.

    High score when one ID disappears around the time another appears,
    suggesting the same entity under a new identifier.
    """
    # Check if query disappeared and candidate appeared close together
    gap_forward = candidate.first_seen - query.last_seen
    gap_reverse = query.first_seen - candidate.last_seen

    # Use whichever gap direction is smallest and non-negative
    candidates_gaps = []
    if gap_forward >= 0:
        candidates_gaps.append(gap_forward)
    if gap_reverse >= 0:
        candidates_gaps.append(gap_reverse)

    if not candidates_gaps:
        # Overlapping time windows — could be simultaneous targets
        # Less likely to be the same person; penalize but don't zero out
        overlap_start = max(query.first_seen, candidate.first_seen)
        overlap_end = min(query.last_seen, candidate.last_seen)
        overlap_duration = max(0.0, overlap_end - overlap_start)
        query_duration = max(0.001, query.last_seen - query.first_seen)
        candidate_duration = max(0.001, candidate.last_seen - candidate.first_seen)
        # If significantly overlapping, they're likely different entities
        overlap_ratio = overlap_duration / min(query_duration, candidate_duration)
        if overlap_ratio > 0.5:
            return 0.0
        # Brief overlap is OK (sensor lag)
        return max(0.0, 0.3 * (1.0 - overlap_ratio))

    min_gap = min(candidates_gaps)
    if min_gap > MAX_TEMPORAL_GAP:
        return 0.0

    return 1.0 - (min_gap / MAX_TEMPORAL_GAP)


def _score_spatial_consistency(
    query: PersonProfile,
    candidate: PersonProfile,
) -> float:
    """Score whether the candidate position is reachable given speed/heading.

    Uses the query's last known position, speed, and heading to predict
    where the person could plausibly be when the candidate appeared.
    """
    dx = candidate.position[0] - query.position[0]
    dy = candidate.position[1] - query.position[1]
    distance = math.hypot(dx, dy)

    # Time between the two observations
    time_gap = abs(candidate.first_seen - query.last_seen)
    if time_gap < 0.001:
        time_gap = 0.001

    # Maximum plausible travel distance at walking speed
    max_distance = MAX_WALKING_SPEED * time_gap

    if max_distance < 0.1:
        max_distance = 0.1

    if distance > max_distance:
        # Beyond walking speed — could still match at running/vehicle speed
        # but with reduced confidence
        overshoot_ratio = distance / max_distance
        if overshoot_ratio > 5.0:
            return 0.0
        return max(0.0, 0.3 * (1.0 - (overshoot_ratio - 1.0) / 4.0))

    # Within walking distance — higher score for closer positions
    return 1.0 - (distance / max_distance)


def _score_signal_fingerprint(
    query: PersonProfile,
    candidate: PersonProfile,
) -> float:
    """Score based on signal characteristics: RSSI pattern, probe SSIDs, etc.

    Signal fingerprinting matches devices by their radio behavior rather
    than their claimed identity.
    """
    score = 0.0
    components = 0

    # RSSI mean similarity
    components += 1
    rssi_diff = abs(query.rssi_mean - candidate.rssi_mean)
    if rssi_diff <= RSSI_TOLERANCE:
        score += 1.0 - (rssi_diff / RSSI_TOLERANCE)

    # RSSI standard deviation similarity (transmission pattern)
    components += 1
    std_diff = abs(query.rssi_std - candidate.rssi_std)
    if std_diff <= 8.0:
        score += 1.0 - (std_diff / 8.0)
    else:
        score += 0.0

    # Probe SSID overlap (Jaccard similarity)
    if query.probe_ssids and candidate.probe_ssids:
        components += 1
        intersection = query.probe_ssids & candidate.probe_ssids
        union = query.probe_ssids | candidate.probe_ssids
        if union:
            jaccard = len(intersection) / len(union)
            score += jaccard

    # Signal count similarity (probe frequency behavior)
    if query.signal_count > 0 and candidate.signal_count > 0:
        q_duration = max(1.0, query.last_seen - query.first_seen)
        c_duration = max(1.0, candidate.last_seen - candidate.first_seen)
        q_rate = query.signal_count / q_duration
        c_rate = candidate.signal_count / c_duration
        max_rate = max(q_rate, c_rate)
        min_rate = min(q_rate, c_rate)
        if max_rate > 0:
            components += 1
            ratio = min_rate / max_rate
            score += ratio

    if components == 0:
        return 0.0
    return min(1.0, score / components)


# ---------------------------------------------------------------------------
# ReIDEngine
# ---------------------------------------------------------------------------

class ReIDEngine:
    """Person re-identification engine.

    Matches persons across BLE MAC rotation, camera appearance timing,
    spatial consistency, and signal fingerprint patterns.

    Args:
        tracker: TargetTracker to read current and historical targets.
        dossier_store: DossierStore for persistent identity resolution.
        reid_store: Optional ReIDStore for appearance embedding persistence.
        match_threshold: Minimum score to consider a positive match.
        weights: Optional strategy weight overrides.
    """

    def __init__(
        self,
        tracker: TargetTracker,
        dossier_store: DossierStore | None = None,
        reid_store: object | None = None,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.tracker = tracker
        self.dossier_store = dossier_store or DossierStore()
        self.reid_store = reid_store
        self.match_threshold = match_threshold
        self.weights = weights or dict(STRATEGY_WEIGHTS)

        self._profiles: dict[str, PersonProfile] = {}
        self._merge_history: list[MergeRecord] = []
        self._lock = threading.Lock()

        # Departed target profiles — kept around for matching against new arrivals
        self._departed_profiles: dict[str, PersonProfile] = {}
        self._max_departed = 500

    # ------------------------------------------------------------------
    # Profile creation
    # ------------------------------------------------------------------

    def create_profile(self, target_id: str) -> PersonProfile | None:
        """Build a PersonProfile from all available sensor data for a target.

        Returns None if the target is not found in the tracker.
        """
        target = self.tracker.get_target(target_id)
        if target is None:
            # Check departed profiles
            with self._lock:
                return self._departed_profiles.get(target_id)

        profile = self._profile_from_target(target)
        with self._lock:
            self._profiles[target_id] = profile
        return profile

    def _profile_from_target(self, target: TrackedTarget) -> PersonProfile:
        """Convert a TrackedTarget into a PersonProfile with enrichments."""
        trail = self.tracker.history.get_trail(target.target_id, max_points=50)

        # Estimate RSSI from position confidence for BLE targets
        rssi_mean = -80.0
        rssi_std = 10.0
        if target.source == "ble":
            # Approximate RSSI from confidence (confidence = (rssi + 100) / 70)
            rssi_mean = (target.position_confidence * 70) - 100
            rssi_std = 8.0  # default, could be enriched from BLE store

        # Extract OUI prefix from BLE target ID
        oui_prefix = ""
        if target.target_id.startswith("ble_"):
            mac_hex = target.target_id[4:]  # e.g. "aabbccddeeff"
            if len(mac_hex) >= 6:
                oui_prefix = mac_hex[:6]
                # Detect random MAC (locally administered bit)
                try:
                    first_byte = int(mac_hex[:2], 16)
                    if first_byte & 0x02:  # locally administered
                        oui_prefix = ""
                except ValueError:
                    pass

        speed = self.tracker.history.get_speed(target.target_id)
        heading = self.tracker.history.get_heading(target.target_id)

        return PersonProfile(
            target_id=target.target_id,
            source=target.source,
            asset_type=target.asset_type,
            alliance=target.alliance,
            classification=target.classification,
            position=target.position,
            heading=heading,
            speed=speed,
            first_seen=target.first_seen,
            last_seen=target.last_seen,
            signal_count=target.signal_count,
            rssi_mean=rssi_mean,
            rssi_std=rssi_std,
            oui_prefix=oui_prefix,
            trail=trail,
            correlated_ids=list(target.correlated_ids),
            confirming_sources=set(target.confirming_sources),
            position_confidence=target.position_confidence,
        )

    def record_departure(self, target: TrackedTarget) -> None:
        """Record a departed target's profile for future matching.

        Called when the tracker prunes a stale target so we can match it
        against future arrivals.
        """
        profile = self._profile_from_target(target)
        with self._lock:
            self._departed_profiles[target.target_id] = profile
            # Evict oldest if over limit
            if len(self._departed_profiles) > self._max_departed:
                oldest_key = min(
                    self._departed_profiles,
                    key=lambda k: self._departed_profiles[k].last_seen,
                )
                del self._departed_profiles[oldest_key]

    def get_candidate_profiles(
        self,
        include_active: bool = True,
        include_departed: bool = True,
    ) -> list[PersonProfile]:
        """Get all available profiles for matching.

        Args:
            include_active: Include currently tracked targets.
            include_departed: Include recently departed targets.
        """
        profiles: list[PersonProfile] = []

        if include_active:
            for target in self.tracker.get_all():
                profiles.append(self._profile_from_target(target))

        if include_departed:
            with self._lock:
                profiles.extend(self._departed_profiles.values())

        return profiles

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_matches(
        self,
        profile: PersonProfile,
        candidates: list[PersonProfile] | None = None,
        threshold: float | None = None,
        limit: int = 10,
    ) -> list[MatchResult]:
        """Score a profile against candidates and return ranked matches.

        Args:
            profile: The query profile to match.
            candidates: Candidate profiles. If None, uses get_candidate_profiles().
            threshold: Minimum score to include. Defaults to self.match_threshold.
            limit: Maximum number of results.

        Returns:
            List of MatchResult sorted by descending score.
        """
        if threshold is None:
            threshold = self.match_threshold
        if candidates is None:
            candidates = self.get_candidate_profiles()

        results: list[MatchResult] = []

        for candidate in candidates:
            # Don't match against self or already-correlated IDs
            if candidate.target_id == profile.target_id:
                continue
            if candidate.target_id in profile.correlated_ids:
                continue

            scores = self._score_pair(profile, candidate)
            weighted = self._weighted_score(scores)

            if weighted >= threshold:
                detail_parts = [
                    f"{name}={s:.2f}"
                    for name, s in scores.items()
                    if s > 0
                ]
                results.append(MatchResult(
                    target_id=candidate.target_id,
                    score=weighted,
                    strategy_scores=scores,
                    detail=f"[{', '.join(detail_parts)}] combined={weighted:.2f}",
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _score_pair(
        self,
        query: PersonProfile,
        candidate: PersonProfile,
    ) -> dict[str, float]:
        """Run all strategies on a profile pair."""
        return {
            "ble_mac_rotation": _score_ble_mac_rotation(query, candidate),
            "temporal_cooccurrence": _score_temporal_cooccurrence(query, candidate),
            "spatial_consistency": _score_spatial_consistency(query, candidate),
            "signal_fingerprint": _score_signal_fingerprint(query, candidate),
        }

    def _weighted_score(self, scores: dict[str, float]) -> float:
        """Compute weighted combination of strategy scores."""
        total_weight = 0.0
        total_score = 0.0

        for name, s in scores.items():
            w = self.weights.get(name, 0.0)
            total_weight += w
            total_score += w * s

        if total_weight <= 0:
            return 0.0
        return min(1.0, total_score / total_weight)

    # ------------------------------------------------------------------
    # Identity merging
    # ------------------------------------------------------------------

    def merge_identities(
        self,
        target_a: str,
        target_b: str,
        score: float = 0.0,
    ) -> MergeRecord:
        """Merge two target IDs into a single person identity.

        The primary target (target_a) survives; target_b's data is folded
        into target_a via the DossierStore and TargetTracker.

        Args:
            target_a: Primary target ID (survives the merge).
            target_b: Secondary target ID (absorbed into target_a).
            score: The match score that prompted this merge.

        Returns:
            MergeRecord documenting the merge.
        """
        # Create/update dossier linking the two identities
        source_a = "unknown"
        source_b = "unknown"

        t_a = self.tracker.get_target(target_a)
        t_b = self.tracker.get_target(target_b)

        if t_a:
            source_a = t_a.source
        if t_b:
            source_b = t_b.source

        dossier = self.dossier_store.create_or_update(
            signal_a=target_a,
            source_a=source_a,
            signal_b=target_b,
            source_b=source_b,
            confidence=score,
            metadata={
                "merge_type": "person_reid",
                "score": score,
            },
        )

        # If both targets are live in the tracker, merge attributes
        if t_a and t_b:
            t_a.position_confidence = min(
                1.0,
                t_a.position_confidence + t_b.position_confidence * 0.5,
            )
            t_a.last_seen = max(t_a.last_seen, t_b.last_seen)

            if target_b not in t_a.correlated_ids:
                t_a.correlated_ids.append(target_b)
            for cid in t_b.correlated_ids:
                if cid not in t_a.correlated_ids:
                    t_a.correlated_ids.append(cid)

            t_a.confirming_sources.add(t_b.source)
            t_a.confirming_sources |= t_b.confirming_sources

            t_a.correlation_confidence = max(
                t_a.correlation_confidence, score
            )

            # Use better position
            if t_b.position_confidence > t_a.position_confidence:
                t_a.position = t_b.position
                t_a.position_source = t_b.position_source

            self.tracker.remove(target_b)

            logger.info(
                "Merged %s into %s (score=%.2f, dossier=%s)",
                target_b, target_a, score, dossier.uuid[:8],
            )

        record = MergeRecord(
            primary_id=target_a,
            secondary_id=target_b,
            score=score,
            dossier_uuid=dossier.uuid,
        )

        with self._lock:
            self._merge_history.append(record)
            # Cap merge history to prevent unbounded memory growth
            _MAX_MERGE_HISTORY = 2000
            if len(self._merge_history) > _MAX_MERGE_HISTORY:
                self._merge_history = self._merge_history[-_MAX_MERGE_HISTORY:]
            # Clean up profiles
            self._departed_profiles.pop(target_b, None)
            self._profiles.pop(target_b, None)

        return record

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def scan_for_matches(self) -> list[MergeRecord]:
        """Run a full re-identification scan across all tracked targets.

        Builds profiles for all current targets, checks each against all
        candidates (including departed targets), and automatically merges
        any high-confidence matches.

        Returns:
            List of MergeRecords for any merges performed.
        """
        targets = self.tracker.get_all()
        candidates = self.get_candidate_profiles()
        merges: list[MergeRecord] = []
        consumed: set[str] = set()

        for target in targets:
            if target.target_id in consumed:
                continue

            profile = self._profile_from_target(target)
            # Filter out already-consumed candidates
            valid_candidates = [
                c for c in candidates
                if c.target_id not in consumed
                and c.target_id != profile.target_id
            ]

            matches = self.find_matches(profile, valid_candidates)
            for match in matches:
                if match.target_id in consumed:
                    continue
                if match.score >= self.match_threshold:
                    record = self.merge_identities(
                        target.target_id,
                        match.target_id,
                        score=match.score,
                    )
                    merges.append(record)
                    consumed.add(match.target_id)
                    break  # Only merge one per target per scan

        return merges

    # ------------------------------------------------------------------
    # Query and statistics
    # ------------------------------------------------------------------

    def get_merge_history(self, limit: int = 100) -> list[dict]:
        """Return recent merge records."""
        with self._lock:
            records = list(reversed(self._merge_history))[:limit]
        return [
            {
                "merge_id": r.merge_id,
                "primary_id": r.primary_id,
                "secondary_id": r.secondary_id,
                "score": round(r.score, 4),
                "timestamp": r.timestamp,
                "dossier_uuid": r.dossier_uuid,
            }
            for r in records
        ]

    def get_departed_count(self) -> int:
        """Number of departed profiles being tracked for future matching."""
        with self._lock:
            return len(self._departed_profiles)

    def get_profile(self, target_id: str) -> PersonProfile | None:
        """Get a cached profile by target ID."""
        with self._lock:
            profile = self._profiles.get(target_id)
            if profile:
                return profile
            return self._departed_profiles.get(target_id)

    @property
    def stats(self) -> dict:
        """Return engine statistics."""
        with self._lock:
            return {
                "active_profiles": len(self._profiles),
                "departed_profiles": len(self._departed_profiles),
                "total_merges": len(self._merge_history),
                "match_threshold": self.match_threshold,
                "weights": dict(self.weights),
            }

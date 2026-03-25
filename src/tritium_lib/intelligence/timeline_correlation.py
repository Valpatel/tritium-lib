# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""TimelineCorrelator — correlate events across target timelines.

Consumes position history from TargetHistory and MovementPatternAnalyzer
to find temporal connections between targets:

  - **Co-occurrences**: Where and when two targets were at the same place
  - **Followers**: Targets that consistently appear shortly after a given target
  - **Causal chains**: Sequences of events suggesting cause-and-effect
  - **Temporal patterns**: Named multi-target pattern matching
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Sequence

from tritium_lib.tracking.target_history import TargetHistory, PositionRecord
from tritium_lib.tracking.movement_patterns import MovementPatternAnalyzer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TimelineEvent:
    """A single event in a target's timeline."""

    target_id: str
    timestamp: float
    position: tuple[float, float]
    event_type: str = "position"  # "position", "loitering", "deviation", etc.
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "timestamp": self.timestamp,
            "position": {"x": self.position[0], "y": self.position[1]},
            "event_type": self.event_type,
            "details": self.details,
        }


@dataclass
class EventSequence:
    """Ordered sequence of events for a target."""

    target_id: str
    events: list[TimelineEvent] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.events and self.start_time == 0.0 and self.end_time == 0.0:
            self._recompute()

    def _recompute(self) -> None:
        if self.events:
            self.events.sort(key=lambda e: e.timestamp)
            self.start_time = self.events[0].timestamp
            self.end_time = self.events[-1].timestamp
            self.duration = self.end_time - self.start_time

    def append(self, event: TimelineEvent) -> None:
        """Append an event and keep the sequence sorted."""
        self.events.append(event)
        self._recompute()

    def in_range(self, t_start: float, t_end: float) -> list[TimelineEvent]:
        """Return events within a time range (inclusive)."""
        return [e for e in self.events if t_start <= e.timestamp <= t_end]

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "event_count": len(self.events),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "events": [e.to_dict() for e in self.events],
        }


@dataclass(slots=True)
class TemporalOverlap:
    """A period where two target timelines intersect spatially and temporally."""

    target_a: str
    target_b: str
    start_time: float
    end_time: float
    duration: float
    center: tuple[float, float]
    avg_distance: float
    confidence: float
    event_count: int = 0

    def to_dict(self) -> dict:
        return {
            "target_a": self.target_a,
            "target_b": self.target_b,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "center": {"x": self.center[0], "y": self.center[1]},
            "avg_distance": round(self.avg_distance, 2),
            "confidence": round(self.confidence, 3),
            "event_count": self.event_count,
        }


@dataclass
class CausalChain:
    """A sequence of events across targets suggesting causation.

    For example: Target A arrives at location -> Target B departs shortly after
    suggests Target B was waiting for Target A.
    """

    chain_id: str
    events: list[TimelineEvent] = field(default_factory=list)
    targets_involved: list[str] = field(default_factory=list)
    confidence: float = 0.0
    pattern_type: str = "unknown"  # "follow", "meetup", "handoff", "escort"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "chain_id": self.chain_id,
            "events": [e.to_dict() for e in self.events],
            "targets_involved": self.targets_involved,
            "confidence": round(self.confidence, 3),
            "pattern_type": self.pattern_type,
            "description": self.description,
        }


@dataclass(slots=True)
class FollowerResult:
    """A target that consistently appears after another target."""

    leader_id: str
    follower_id: str
    occurrence_count: int
    avg_delay_seconds: float
    avg_distance: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "leader_id": self.leader_id,
            "follower_id": self.follower_id,
            "occurrence_count": self.occurrence_count,
            "avg_delay_seconds": round(self.avg_delay_seconds, 2),
            "avg_distance": round(self.avg_distance, 2),
            "confidence": round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# Temporal pattern definitions
# ---------------------------------------------------------------------------

@dataclass
class TemporalPattern:
    """Definition of a named temporal pattern to match against target sets.

    Attributes:
        name: Pattern name (e.g., "meetup", "surveillance", "escort").
        min_targets: Minimum number of targets that must participate.
        max_time_window: Maximum time window for the pattern in seconds.
        max_spatial_radius: Maximum spatial radius for coincidence.
        min_co_occurrences: Minimum number of co-occurrence events.
        require_movement: If True, at least one target must be moving.
    """

    name: str
    min_targets: int = 2
    max_time_window: float = 600.0  # 10 minutes
    max_spatial_radius: float = 20.0
    min_co_occurrences: int = 2
    require_movement: bool = False


# Pre-defined patterns
PATTERN_MEETUP = TemporalPattern(
    name="meetup",
    min_targets=2,
    max_time_window=300.0,
    max_spatial_radius=15.0,
    min_co_occurrences=3,
    require_movement=False,
)

PATTERN_SURVEILLANCE = TemporalPattern(
    name="surveillance",
    min_targets=2,
    max_time_window=1800.0,  # 30 minutes
    max_spatial_radius=50.0,
    min_co_occurrences=5,
    require_movement=True,
)

PATTERN_ESCORT = TemporalPattern(
    name="escort",
    min_targets=2,
    max_time_window=600.0,
    max_spatial_radius=10.0,
    min_co_occurrences=10,
    require_movement=True,
)


# ---------------------------------------------------------------------------
# Main correlator
# ---------------------------------------------------------------------------

class TimelineCorrelator:
    """Find temporal overlaps and correlations between target activity timelines.

    Thread-safe. Uses TargetHistory for position trails and optionally
    MovementPatternAnalyzer for enriched events (loitering, deviations, etc.).
    """

    def __init__(
        self,
        history: TargetHistory,
        pattern_analyzer: MovementPatternAnalyzer | None = None,
        *,
        co_occurrence_radius: float = 10.0,
        co_occurrence_min_duration: float = 5.0,
        follower_time_window: float = 120.0,
        follower_spatial_radius: float = 30.0,
    ) -> None:
        """Initialize the timeline correlator.

        Args:
            history: TargetHistory providing position trails.
            pattern_analyzer: Optional MovementPatternAnalyzer for enriched events.
            co_occurrence_radius: Max distance (units) to consider "same place".
            co_occurrence_min_duration: Min seconds of co-location to report.
            follower_time_window: Max seconds after leader for follower detection.
            follower_spatial_radius: Max distance for follower arrival matching.
        """
        self._history = history
        self._pattern_analyzer = pattern_analyzer
        self._co_occurrence_radius = co_occurrence_radius
        self._co_occurrence_min_duration = co_occurrence_min_duration
        self._follower_time_window = follower_time_window
        self._follower_spatial_radius = follower_spatial_radius
        self._lock = threading.Lock()
        self._chain_counter = 0

    def _next_chain_id(self) -> str:
        """Generate a unique chain ID."""
        with self._lock:
            self._chain_counter += 1
            return f"chain_{self._chain_counter:04d}"

    def build_event_sequence(
        self,
        target_id: str,
        max_points: int = 500,
        include_patterns: bool = True,
    ) -> EventSequence:
        """Build a full event sequence for a target from history + patterns.

        Args:
            target_id: Target to build sequence for.
            max_points: Maximum trail points to retrieve.
            include_patterns: Whether to include movement pattern events.

        Returns:
            EventSequence with all known events for this target.
        """
        seq = EventSequence(target_id=target_id)

        # Position events from history
        trail = self._history.get_trail(target_id, max_points=max_points)
        for x, y, t in trail:
            seq.events.append(TimelineEvent(
                target_id=target_id,
                timestamp=t,
                position=(x, y),
                event_type="position",
            ))

        # Movement pattern events
        if include_patterns and self._pattern_analyzer is not None:
            patterns = self._pattern_analyzer.analyze(target_id, max_points=max_points)
            for p in patterns:
                center = p.get("center", {})
                cx = center.get("x", 0.0) if isinstance(center, dict) else 0.0
                cy = center.get("y", 0.0) if isinstance(center, dict) else 0.0
                seq.events.append(TimelineEvent(
                    target_id=target_id,
                    timestamp=p.get("timestamp", 0.0),
                    position=(cx, cy),
                    event_type=p.get("pattern_type", "pattern"),
                    details=p,
                ))

        seq._recompute()
        return seq

    def find_co_occurrences(
        self,
        target_a: str,
        target_b: str,
        *,
        radius: float | None = None,
        min_duration: float | None = None,
        max_points: int = 500,
    ) -> list[TemporalOverlap]:
        """Find all time periods where target_a and target_b were at the same place.

        Scans both trails for temporal windows where both targets are within
        ``radius`` of each other for at least ``min_duration`` seconds.

        Args:
            target_a: First target ID.
            target_b: Second target ID.
            radius: Max distance to consider co-located (defaults to instance setting).
            min_duration: Min seconds to report (defaults to instance setting).
            max_points: Maximum trail points to retrieve per target.

        Returns:
            List of TemporalOverlap objects, sorted by start time.
        """
        r = radius if radius is not None else self._co_occurrence_radius
        min_dur = min_duration if min_duration is not None else self._co_occurrence_min_duration

        trail_a = self._history.get_trail(target_a, max_points=max_points)
        trail_b = self._history.get_trail(target_b, max_points=max_points)

        if len(trail_a) < 2 or len(trail_b) < 2:
            return []

        # Find temporal overlap region
        t_start = max(trail_a[0][2], trail_b[0][2])
        t_end = min(trail_a[-1][2], trail_b[-1][2])

        if t_end - t_start < min_dur:
            return []

        # Walk through trail_a, interpolate trail_b at each timestamp
        overlaps: list[TemporalOverlap] = []
        b_idx = 0
        run_start: float | None = None
        run_positions_a: list[tuple[float, float]] = []
        run_positions_b: list[tuple[float, float]] = []
        run_distances: list[float] = []
        run_count = 0

        for ax, ay, at in trail_a:
            if at < t_start or at > t_end:
                continue

            # Advance b_idx to bracket this time
            while b_idx < len(trail_b) - 1 and trail_b[b_idx + 1][2] <= at:
                b_idx += 1

            if b_idx >= len(trail_b) - 1:
                bx, by = trail_b[-1][0], trail_b[-1][1]
            else:
                b1 = trail_b[b_idx]
                b2 = trail_b[b_idx + 1]
                dt_b = b2[2] - b1[2]
                if dt_b > 0:
                    frac = (at - b1[2]) / dt_b
                    bx = b1[0] + frac * (b2[0] - b1[0])
                    by = b1[1] + frac * (b2[1] - b1[1])
                else:
                    bx, by = b1[0], b1[1]

            dist = math.hypot(ax - bx, ay - by)

            if dist <= r:
                if run_start is None:
                    run_start = at
                    run_positions_a = []
                    run_positions_b = []
                    run_distances = []
                    run_count = 0
                run_positions_a.append((ax, ay))
                run_positions_b.append((bx, by))
                run_distances.append(dist)
                run_count += 1
            else:
                # End of a co-located run
                if run_start is not None:
                    duration = at - run_start
                    if duration >= min_dur and run_count >= 2:
                        all_pts = run_positions_a + run_positions_b
                        cx = sum(p[0] for p in all_pts) / len(all_pts)
                        cy = sum(p[1] for p in all_pts) / len(all_pts)
                        avg_dist = sum(run_distances) / len(run_distances)
                        confidence = min(1.0, duration / (min_dur * 5)) * min(1.0, 1.0 - avg_dist / (r * 1.5))
                        confidence = max(0.0, confidence)

                        overlaps.append(TemporalOverlap(
                            target_a=target_a,
                            target_b=target_b,
                            start_time=run_start,
                            end_time=at,
                            duration=duration,
                            center=(cx, cy),
                            avg_distance=avg_dist,
                            confidence=confidence,
                            event_count=run_count,
                        ))
                    run_start = None

        # Close any open run
        if run_start is not None and trail_a:
            last_t = trail_a[-1][2]
            duration = last_t - run_start
            if duration >= min_dur and run_count >= 2:
                all_pts = run_positions_a + run_positions_b
                cx = sum(p[0] for p in all_pts) / len(all_pts)
                cy = sum(p[1] for p in all_pts) / len(all_pts)
                avg_dist = sum(run_distances) / len(run_distances)
                confidence = min(1.0, duration / (min_dur * 5)) * min(1.0, 1.0 - avg_dist / (r * 1.5))
                confidence = max(0.0, confidence)

                overlaps.append(TemporalOverlap(
                    target_a=target_a,
                    target_b=target_b,
                    start_time=run_start,
                    end_time=last_t,
                    duration=duration,
                    center=(cx, cy),
                    avg_distance=avg_dist,
                    confidence=confidence,
                    event_count=run_count,
                ))

        return overlaps

    def find_followers(
        self,
        target_id: str,
        candidate_ids: list[str] | None = None,
        *,
        time_window: float | None = None,
        spatial_radius: float | None = None,
        min_occurrences: int = 2,
        max_points: int = 500,
    ) -> list[FollowerResult]:
        """Find targets that consistently appear after the given target.

        For each position in target_id's trail, checks whether any candidate
        target appears at a nearby location within ``time_window`` seconds later.

        Args:
            target_id: The "leader" target ID.
            candidate_ids: List of candidate follower IDs. If None, uses all
                targets in the history (limited to those with trails).
            time_window: Max seconds after leader for follower to appear.
            spatial_radius: Max distance for follower arrival matching.
            min_occurrences: Minimum follow events to report.
            max_points: Maximum trail points to retrieve per target.

        Returns:
            List of FollowerResult sorted by confidence descending.
        """
        tw = time_window if time_window is not None else self._follower_time_window
        sr = spatial_radius if spatial_radius is not None else self._follower_spatial_radius

        leader_trail = self._history.get_trail(target_id, max_points=max_points)
        if len(leader_trail) < 2:
            return []

        # If no candidates provided, we can't enumerate all targets from TargetHistory
        # (it doesn't expose target IDs). Caller must provide candidate_ids.
        if candidate_ids is None:
            return []

        results: list[FollowerResult] = []

        for cand_id in candidate_ids:
            if cand_id == target_id:
                continue

            cand_trail = self._history.get_trail(cand_id, max_points=max_points)
            if len(cand_trail) < 2:
                continue

            follow_events: list[tuple[float, float]] = []  # (delay, distance)

            for lx, ly, lt in leader_trail:
                # Look for candidate arrivals in the time window after leader
                for cx, cy, ct in cand_trail:
                    delay = ct - lt
                    if delay < 0:
                        continue
                    if delay > tw:
                        break  # trail is sorted by time

                    dist = math.hypot(cx - lx, cy - ly)
                    if dist <= sr:
                        follow_events.append((delay, dist))
                        break  # one match per leader position

            if len(follow_events) >= min_occurrences:
                avg_delay = sum(d for d, _ in follow_events) / len(follow_events)
                avg_dist = sum(d for _, d in follow_events) / len(follow_events)

                # Confidence based on: count, consistency of delay, closeness
                count_score = min(1.0, len(follow_events) / (len(leader_trail) * 0.5))
                delay_consistency = 1.0 - min(1.0, _std_dev([d for d, _ in follow_events]) / (tw * 0.5)) if len(follow_events) >= 2 else 0.5
                proximity_score = max(0.0, 1.0 - avg_dist / sr)

                confidence = 0.4 * count_score + 0.3 * delay_consistency + 0.3 * proximity_score
                confidence = max(0.0, min(1.0, confidence))

                results.append(FollowerResult(
                    leader_id=target_id,
                    follower_id=cand_id,
                    occurrence_count=len(follow_events),
                    avg_delay_seconds=avg_delay,
                    avg_distance=avg_dist,
                    confidence=confidence,
                ))

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def detect_pattern(
        self,
        target_ids: list[str],
        pattern: TemporalPattern,
        *,
        max_points: int = 500,
    ) -> list[CausalChain]:
        """Match a temporal pattern across a set of targets.

        Checks whether the given targets exhibit the described pattern
        (e.g., meetup, surveillance, escort) within the pattern's constraints.

        Args:
            target_ids: Target IDs to check.
            pattern: TemporalPattern definition to match.
            max_points: Maximum trail points per target.

        Returns:
            List of CausalChain objects describing matched pattern instances.
        """
        if len(target_ids) < pattern.min_targets:
            return []

        # Build trails
        trails: dict[str, list[tuple[float, float, float]]] = {}
        for tid in target_ids:
            trail = self._history.get_trail(tid, max_points=max_points)
            if len(trail) >= 2:
                trails[tid] = trail

        if len(trails) < pattern.min_targets:
            return []

        chains: list[CausalChain] = []

        # Check all pairs for co-occurrences within pattern constraints
        tids = list(trails.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tid_a = tids[i]
                tid_b = tids[j]

                co_occs = self.find_co_occurrences(
                    tid_a,
                    tid_b,
                    radius=pattern.max_spatial_radius,
                    min_duration=0.1,  # very short — we'll filter by count
                    max_points=max_points,
                )

                # Filter co-occurrences within the time window
                valid_co_occs = [
                    c for c in co_occs
                    if c.duration <= pattern.max_time_window
                ]

                if len(valid_co_occs) < pattern.min_co_occurrences:
                    continue

                # Movement check
                if pattern.require_movement:
                    trail_a = trails[tid_a]
                    trail_b = trails[tid_b]
                    moved_a = _total_displacement(trail_a) > 1.0
                    moved_b = _total_displacement(trail_b) > 1.0
                    if not moved_a and not moved_b:
                        continue

                # Build a causal chain from the co-occurrences
                events: list[TimelineEvent] = []
                for co in valid_co_occs:
                    events.append(TimelineEvent(
                        target_id=tid_a,
                        timestamp=co.start_time,
                        position=co.center,
                        event_type=f"co_occurrence_{pattern.name}",
                        details={
                            "with_target": tid_b,
                            "duration": co.duration,
                            "avg_distance": co.avg_distance,
                        },
                    ))

                events.sort(key=lambda e: e.timestamp)

                # Compute confidence from co-occurrence quality
                avg_confidence = sum(c.confidence for c in valid_co_occs) / len(valid_co_occs)
                count_factor = min(1.0, len(valid_co_occs) / (pattern.min_co_occurrences * 2))
                chain_confidence = 0.5 * avg_confidence + 0.5 * count_factor

                chains.append(CausalChain(
                    chain_id=self._next_chain_id(),
                    events=events,
                    targets_involved=[tid_a, tid_b],
                    confidence=max(0.0, min(1.0, chain_confidence)),
                    pattern_type=pattern.name,
                    description=(
                        f"{pattern.name}: {tid_a} and {tid_b} had "
                        f"{len(valid_co_occs)} co-occurrences within "
                        f"{pattern.max_time_window}s window"
                    ),
                ))

        chains.sort(key=lambda c: c.confidence, reverse=True)
        return chains

    def find_causal_chains(
        self,
        target_ids: list[str],
        *,
        max_delay: float = 60.0,
        spatial_radius: float = 15.0,
        max_points: int = 500,
    ) -> list[CausalChain]:
        """Discover causal chains where one target's arrival triggers another's departure.

        Looks for patterns where:
        - Target A arrives at a location
        - Target B departs from the same location shortly after (handoff)
        - Target A loiters, then Target B approaches (meetup)

        Args:
            target_ids: Target IDs to analyze.
            max_delay: Maximum seconds between cause and effect.
            spatial_radius: Maximum distance for spatial proximity.
            max_points: Maximum trail points per target.

        Returns:
            List of CausalChain objects.
        """
        if len(target_ids) < 2:
            return []

        # Build sequences with pattern enrichment
        sequences: dict[str, EventSequence] = {}
        for tid in target_ids:
            seq = self.build_event_sequence(
                tid,
                max_points=max_points,
                include_patterns=self._pattern_analyzer is not None,
            )
            if seq.events:
                sequences[tid] = seq

        if len(sequences) < 2:
            return []

        chains: list[CausalChain] = []
        tids = list(sequences.keys())

        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tid_a = tids[i]
                tid_b = tids[j]
                seq_a = sequences[tid_a]
                seq_b = sequences[tid_b]

                # Look for "arrive then appear" patterns
                for ea in seq_a.events:
                    # Find events in B that happen shortly after at nearby location
                    for eb in seq_b.events:
                        delay = eb.timestamp - ea.timestamp
                        if delay < 0 or delay > max_delay:
                            continue

                        dist = math.hypot(
                            eb.position[0] - ea.position[0],
                            eb.position[1] - ea.position[1],
                        )
                        if dist > spatial_radius:
                            continue

                        # Found a potential causal link
                        proximity_score = max(0.0, 1.0 - dist / spatial_radius)
                        timing_score = max(0.0, 1.0 - delay / max_delay)
                        confidence = 0.5 * proximity_score + 0.5 * timing_score

                        if confidence < 0.2:
                            continue

                        chain_type = _classify_causal_event(ea, eb)

                        chains.append(CausalChain(
                            chain_id=self._next_chain_id(),
                            events=[ea, eb],
                            targets_involved=[tid_a, tid_b],
                            confidence=confidence,
                            pattern_type=chain_type,
                            description=(
                                f"{chain_type}: {tid_a} ({ea.event_type}) -> "
                                f"{tid_b} ({eb.event_type}) "
                                f"delay={delay:.1f}s dist={dist:.1f}"
                            ),
                        ))

        # De-duplicate by keeping only the highest-confidence chain per target pair per type
        chains.sort(key=lambda c: c.confidence, reverse=True)
        return chains

    def get_timeline_overlap(
        self,
        target_ids: list[str],
        max_points: int = 500,
    ) -> dict:
        """Get a summary of how multiple target timelines overlap.

        Args:
            target_ids: Target IDs to analyze.
            max_points: Maximum trail points per target.

        Returns:
            Dict with overlap statistics.
        """
        time_ranges: dict[str, tuple[float, float]] = {}

        for tid in target_ids:
            trail = self._history.get_trail(tid, max_points=max_points)
            if len(trail) >= 2:
                time_ranges[tid] = (trail[0][2], trail[-1][2])

        if len(time_ranges) < 2:
            return {
                "targets_with_data": len(time_ranges),
                "overlap_start": 0.0,
                "overlap_end": 0.0,
                "overlap_duration": 0.0,
                "individual_ranges": time_ranges,
            }

        # Find the global overlap window
        overlap_start = max(r[0] for r in time_ranges.values())
        overlap_end = min(r[1] for r in time_ranges.values())
        overlap_duration = max(0.0, overlap_end - overlap_start)

        return {
            "targets_with_data": len(time_ranges),
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_duration": overlap_duration,
            "individual_ranges": {
                tid: {"start": s, "end": e, "duration": e - s}
                for tid, (s, e) in time_ranges.items()
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std_dev(values: list[float]) -> float:
    """Compute standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _total_displacement(trail: list[tuple[float, float, float]]) -> float:
    """Total displacement from first to last point."""
    if len(trail) < 2:
        return 0.0
    dx = trail[-1][0] - trail[0][0]
    dy = trail[-1][1] - trail[0][1]
    return math.hypot(dx, dy)


def _classify_causal_event(ea: TimelineEvent, eb: TimelineEvent) -> str:
    """Classify the type of causal relationship between two events."""
    # Loitering followed by arrival => meetup
    if ea.event_type == "loitering" and eb.event_type == "position":
        return "meetup"
    # Position followed by loitering => handoff (A passes, B stays)
    if ea.event_type == "position" and eb.event_type == "loitering":
        return "handoff"
    # Both position events => follow
    if ea.event_type == "position" and eb.event_type == "position":
        return "follow"
    # Deviation events => reaction
    if ea.event_type == "deviation" or eb.event_type == "deviation":
        return "reaction"
    return "sequence"

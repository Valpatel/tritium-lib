# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Access pattern analysis — detect unauthorized and unusual area access.

Analyses how targets access zones over time to detect tailgating,
piggybacking, unusual frequency, and deviations from established
access patterns.  Integrates with the geofence engine and zone analysis
subsystems.

Key components:

  - :class:`AccessEvent`            — single enter/exit event with timestamps
  - :class:`AccessPattern`          — a target's typical access behavior for an area
  - :class:`AccessPatternAnalyzer`  — learn and detect deviations in access
  - :func:`detect_tailgating`       — targets entering in quick succession
  - :func:`detect_piggybacking`     — targets entering without proper credentials
  - :func:`frequency_analysis`      — how often a target visits an area

Events published to EventBus (if attached):
    access:tailgating   — two or more targets entered a zone in quick succession
    access:piggybacking — a target entered a zone without proper access
    access:anomaly      — a target deviated from its learned access pattern

Usage::

    from tritium_lib.intelligence.access_patterns import (
        AccessEvent, AccessPattern, AccessPatternAnalyzer,
        detect_tailgating, detect_piggybacking, frequency_analysis,
    )

    analyzer = AccessPatternAnalyzer()
    analyzer.record_access(AccessEvent(
        target_id="ble_aa:bb:cc", zone_id="zone_lobby",
        event_type="enter", timestamp=1000.0,
    ))
    # After many observations, learn patterns:
    pattern = analyzer.learn_pattern("ble_aa:bb:cc", "zone_lobby")
    # Check for tailgating:
    alerts = detect_tailgating(zone_events, threshold_seconds=3.0)
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum access events needed to learn a pattern
MIN_EVENTS_FOR_PATTERN = 5

# Default tailgating threshold — seconds between successive zone entries
DEFAULT_TAILGATE_THRESHOLD_SECONDS = 3.0

# Default piggybacking — target enters without being in the authorized set
# (analyzed via the authorized_targets parameter)

# Number of time-of-day bins for access schedule histograms
ACCESS_SCHEDULE_BINS = 24

# Standard deviation multiplier for anomaly detection
DEFAULT_ANOMALY_SIGMA = 2.0

# Maximum events stored per zone per target
MAX_EVENTS_PER_TARGET_ZONE = 10_000

# Maximum patterns stored
MAX_PATTERNS = 50_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AccessEvent:
    """A single access event — a target entering or exiting a zone.

    Attributes:
        target_id: Unique target identifier (e.g. ``ble_AA:BB:CC``).
        zone_id: Zone that was accessed.
        event_type: ``"enter"`` or ``"exit"``.
        timestamp: Unix epoch time of the event.
        authorized: Whether the target had proper credentials/authorization.
            ``None`` means authorization status is unknown.
        source: Sensor source that detected the event (e.g. ``"ble"``, ``"camera"``).
        position: Optional (lat, lng) at time of event.
    """

    target_id: str
    zone_id: str
    event_type: str  # "enter" or "exit"
    timestamp: float = 0.0
    authorized: bool | None = None
    source: str = ""
    position: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.timestamp <= 0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "target_id": self.target_id,
            "zone_id": self.zone_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "authorized": self.authorized,
            "source": self.source,
        }
        if self.position is not None:
            d["position"] = list(self.position)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessEvent:
        pos = data.get("position")
        return cls(
            target_id=data["target_id"],
            zone_id=data["zone_id"],
            event_type=data["event_type"],
            timestamp=data.get("timestamp", 0.0),
            authorized=data.get("authorized"),
            source=data.get("source", ""),
            position=tuple(pos) if pos else None,
        )


@dataclass
class AccessPattern:
    """A target's learned typical access behavior for a specific zone.

    Captures when and how often the target accesses the zone,
    typical dwell times, and expected time-of-day distribution.

    Attributes:
        target_id: Unique target identifier.
        zone_id: Zone this pattern describes.
        total_entries: Total number of observed entries.
        total_exits: Total number of observed exits.
        avg_dwell_seconds: Average dwell time (time between enter and exit).
        std_dwell_seconds: Standard deviation of dwell times.
        min_dwell_seconds: Minimum observed dwell time.
        max_dwell_seconds: Maximum observed dwell time.
        avg_interval_seconds: Average time between consecutive entries.
        std_interval_seconds: Standard deviation of inter-entry intervals.
        hourly_distribution: Entries per hour-of-day (0-23).
        day_of_week_distribution: Entries per day-of-week (0=Mon, 6=Sun).
        first_seen: Timestamp of the earliest recorded event.
        last_seen: Timestamp of the most recent recorded event.
        learned_at: Timestamp when the pattern was computed.
    """

    target_id: str = ""
    zone_id: str = ""
    total_entries: int = 0
    total_exits: int = 0
    avg_dwell_seconds: float = 0.0
    std_dwell_seconds: float = 0.0
    min_dwell_seconds: float = 0.0
    max_dwell_seconds: float = 0.0
    avg_interval_seconds: float = 0.0
    std_interval_seconds: float = 0.0
    hourly_distribution: dict[int, int] = field(default_factory=dict)
    day_of_week_distribution: dict[int, int] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0
    learned_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "zone_id": self.zone_id,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "avg_dwell_seconds": round(self.avg_dwell_seconds, 2),
            "std_dwell_seconds": round(self.std_dwell_seconds, 2),
            "min_dwell_seconds": round(self.min_dwell_seconds, 2),
            "max_dwell_seconds": round(self.max_dwell_seconds, 2),
            "avg_interval_seconds": round(self.avg_interval_seconds, 2),
            "std_interval_seconds": round(self.std_interval_seconds, 2),
            "hourly_distribution": self.hourly_distribution,
            "day_of_week_distribution": self.day_of_week_distribution,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "learned_at": self.learned_at,
        }


@dataclass
class TailgateAlert:
    """Alert raised when targets enter a zone in quick succession.

    Attributes:
        zone_id: Zone where tailgating was detected.
        leader_target_id: The first target that entered.
        follower_target_id: The target that followed closely behind.
        leader_timestamp: When the leader entered.
        follower_timestamp: When the follower entered.
        gap_seconds: Time between the two entries.
        severity: ``"low"``, ``"medium"``, or ``"high"``.
    """

    zone_id: str = ""
    leader_target_id: str = ""
    follower_target_id: str = ""
    leader_timestamp: float = 0.0
    follower_timestamp: float = 0.0
    gap_seconds: float = 0.0
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "leader_target_id": self.leader_target_id,
            "follower_target_id": self.follower_target_id,
            "leader_timestamp": self.leader_timestamp,
            "follower_timestamp": self.follower_timestamp,
            "gap_seconds": round(self.gap_seconds, 3),
            "severity": self.severity,
        }


@dataclass
class PiggybackAlert:
    """Alert raised when an unauthorized target enters a zone.

    Attributes:
        zone_id: Zone where piggybacking was detected.
        target_id: The unauthorized target.
        timestamp: When the entry occurred.
        preceding_authorized_id: The authorized target that entered
            just before (if any).
        gap_seconds: Time since the preceding authorized entry (if any).
        severity: ``"low"``, ``"medium"``, or ``"high"``.
    """

    zone_id: str = ""
    target_id: str = ""
    timestamp: float = 0.0
    preceding_authorized_id: str = ""
    gap_seconds: float | None = None
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "zone_id": self.zone_id,
            "target_id": self.target_id,
            "timestamp": self.timestamp,
            "preceding_authorized_id": self.preceding_authorized_id,
            "severity": self.severity,
        }
        if self.gap_seconds is not None:
            d["gap_seconds"] = round(self.gap_seconds, 3)
        return d


@dataclass
class FrequencyReport:
    """Result of frequency analysis for a target's visits to a zone.

    Attributes:
        target_id: Target identifier.
        zone_id: Zone identifier.
        total_visits: Number of enter events in the analysis window.
        visits_per_day: Average visits per day.
        visits_per_week: Average visits per week.
        time_range: (start, end) timestamps of the analysis window.
        peak_hours: Hours of day with the most entries.
        avg_dwell_seconds: Average dwell time across visits.
        last_visit: Timestamp of the most recent visit.
    """

    target_id: str = ""
    zone_id: str = ""
    total_visits: int = 0
    visits_per_day: float = 0.0
    visits_per_week: float = 0.0
    time_range: tuple[float, float] = (0.0, 0.0)
    peak_hours: list[int] = field(default_factory=list)
    avg_dwell_seconds: float = 0.0
    last_visit: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "zone_id": self.zone_id,
            "total_visits": self.total_visits,
            "visits_per_day": round(self.visits_per_day, 4),
            "visits_per_week": round(self.visits_per_week, 4),
            "time_range": list(self.time_range),
            "peak_hours": self.peak_hours,
            "avg_dwell_seconds": round(self.avg_dwell_seconds, 2),
            "last_visit": self.last_visit,
        }


@dataclass
class AccessAnomaly:
    """An anomaly detected in access behavior.

    Attributes:
        target_id: Target whose behavior deviated.
        zone_id: Zone where the anomaly was detected.
        anomaly_type: Type of anomaly (``"unusual_time"``, ``"unusual_frequency"``,
            ``"unusual_dwell"``, ``"new_zone"``).
        description: Human-readable description.
        score: Anomaly score 0.0 (normal) to 1.0 (extremely anomalous).
        severity: ``"low"``, ``"medium"``, ``"high"``, or ``"critical"``.
        timestamp: When the anomaly was detected.
        details: Additional context.
    """

    target_id: str = ""
    zone_id: str = ""
    anomaly_type: str = ""
    description: str = ""
    score: float = 0.0
    severity: str = "low"
    timestamp: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp <= 0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "zone_id": self.zone_id,
            "anomaly_type": self.anomaly_type,
            "description": self.description,
            "score": round(self.score, 4),
            "severity": self.severity,
            "timestamp": self.timestamp,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def detect_tailgating(
    zone_events: Sequence[AccessEvent],
    threshold_seconds: float = DEFAULT_TAILGATE_THRESHOLD_SECONDS,
) -> list[TailgateAlert]:
    """Detect when targets enter a zone in quick succession (tailgating).

    Examines enter events sorted by timestamp and flags consecutive
    entries by different targets that are closer than *threshold_seconds*.

    Parameters
    ----------
    zone_events : Sequence[AccessEvent]
        Access events to analyze. Should ideally be from a single zone,
        but multi-zone events are grouped internally.
    threshold_seconds : float
        Maximum gap between consecutive entries to flag as tailgating.

    Returns
    -------
    list[TailgateAlert]
        Detected tailgating incidents, sorted by timestamp.
    """
    if threshold_seconds <= 0:
        return []

    # Group enter events by zone
    by_zone: dict[str, list[AccessEvent]] = defaultdict(list)
    for ev in zone_events:
        if ev.event_type == "enter":
            by_zone[ev.zone_id].append(ev)

    alerts: list[TailgateAlert] = []

    for zone_id, entries in by_zone.items():
        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)

        for i in range(1, len(sorted_entries)):
            prev = sorted_entries[i - 1]
            curr = sorted_entries[i]

            # Must be different targets
            if prev.target_id == curr.target_id:
                continue

            gap = curr.timestamp - prev.timestamp
            if 0 <= gap <= threshold_seconds:
                # Severity based on how close the gap is
                if gap < threshold_seconds * 0.3:
                    severity = "high"
                elif gap < threshold_seconds * 0.7:
                    severity = "medium"
                else:
                    severity = "low"

                alerts.append(TailgateAlert(
                    zone_id=zone_id,
                    leader_target_id=prev.target_id,
                    follower_target_id=curr.target_id,
                    leader_timestamp=prev.timestamp,
                    follower_timestamp=curr.timestamp,
                    gap_seconds=gap,
                    severity=severity,
                ))

    alerts.sort(key=lambda a: a.follower_timestamp)
    return alerts


def detect_piggybacking(
    zone_events: Sequence[AccessEvent],
    authorized_targets: set[str] | None = None,
    follow_window_seconds: float = 10.0,
) -> list[PiggybackAlert]:
    """Detect targets entering a zone without proper authorization.

    A piggybacking event is when an unauthorized target enters a zone,
    especially right after an authorized target opened the way.

    Two detection modes:

    1. **Explicit authorization**: If ``authorized_targets`` is provided,
       any enter event by a target NOT in this set is flagged.
    2. **Event-level authorization**: If ``authorized_targets`` is ``None``,
       uses each event's ``authorized`` field. Events with
       ``authorized=False`` are flagged.

    Parameters
    ----------
    zone_events : Sequence[AccessEvent]
        Access events to analyze.
    authorized_targets : set[str] or None
        Set of target IDs that are authorized to enter. If ``None``,
        relies on each event's ``authorized`` attribute.
    follow_window_seconds : float
        Time window to look back for a preceding authorized entry.

    Returns
    -------
    list[PiggybackAlert]
        Detected piggybacking incidents, sorted by timestamp.
    """
    # Group enter events by zone
    by_zone: dict[str, list[AccessEvent]] = defaultdict(list)
    for ev in zone_events:
        if ev.event_type == "enter":
            by_zone[ev.zone_id].append(ev)

    alerts: list[PiggybackAlert] = []

    for zone_id, entries in by_zone.items():
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)

        for i, ev in enumerate(sorted_entries):
            # Determine if this target is unauthorized
            is_unauthorized = False
            if authorized_targets is not None:
                is_unauthorized = ev.target_id not in authorized_targets
            elif ev.authorized is False:
                is_unauthorized = True

            if not is_unauthorized:
                continue

            # Look back for the most recent authorized entry
            preceding_auth_id = ""
            gap = None
            for j in range(i - 1, -1, -1):
                prev = sorted_entries[j]
                time_diff = ev.timestamp - prev.timestamp
                if time_diff > follow_window_seconds:
                    break

                prev_authorized = False
                if authorized_targets is not None:
                    prev_authorized = prev.target_id in authorized_targets
                elif prev.authorized is True:
                    prev_authorized = True

                if prev_authorized:
                    preceding_auth_id = prev.target_id
                    gap = time_diff
                    break

            # Severity: higher if they closely followed an authorized person
            if gap is not None and gap < follow_window_seconds * 0.3:
                severity = "high"
            elif gap is not None:
                severity = "medium"
            else:
                severity = "low"

            alerts.append(PiggybackAlert(
                zone_id=zone_id,
                target_id=ev.target_id,
                timestamp=ev.timestamp,
                preceding_authorized_id=preceding_auth_id,
                gap_seconds=gap,
                severity=severity,
            ))

    alerts.sort(key=lambda a: a.timestamp)
    return alerts


def frequency_analysis(
    target_id: str,
    area_id: str,
    events: Sequence[AccessEvent],
    time_range: tuple[float, float] | None = None,
) -> FrequencyReport:
    """Analyze how often a target visits a specific area.

    Parameters
    ----------
    target_id : str
        Target to analyze.
    area_id : str
        Zone / area to analyze visits for.
    events : Sequence[AccessEvent]
        Access events to analyze. Filtered to the given target and area.
    time_range : tuple[float, float] or None
        ``(start, end)`` timestamps. If ``None``, uses the full event range.

    Returns
    -------
    FrequencyReport
    """
    # Filter to this target + zone
    filtered = [
        e for e in events
        if e.target_id == target_id and e.zone_id == area_id
    ]

    if not filtered:
        tr = time_range if time_range else (0.0, 0.0)
        return FrequencyReport(
            target_id=target_id,
            zone_id=area_id,
            time_range=tr,
        )

    filtered.sort(key=lambda e: e.timestamp)

    # Apply time range
    if time_range:
        start, end = time_range
        filtered = [e for e in filtered if start <= e.timestamp <= end]
    else:
        start = filtered[0].timestamp if filtered else 0.0
        end = filtered[-1].timestamp if filtered else 0.0

    if not filtered:
        return FrequencyReport(
            target_id=target_id,
            zone_id=area_id,
            time_range=(start, end),
        )

    entries = [e for e in filtered if e.event_type == "enter"]
    exits = [e for e in filtered if e.event_type == "exit"]
    total_visits = len(entries)

    # Time span in days
    span_seconds = max(end - start, 1.0)
    span_days = span_seconds / 86400.0

    visits_per_day = total_visits / span_days if span_days > 0 else 0.0
    visits_per_week = visits_per_day * 7.0

    # Peak hours
    hourly: dict[int, int] = defaultdict(int)
    for e in entries:
        hour = time.localtime(e.timestamp).tm_hour
        hourly[hour] += 1

    peak_hours: list[int] = []
    if hourly:
        max_count = max(hourly.values())
        peak_hours = sorted(h for h, c in hourly.items() if c == max_count)

    # Dwell times
    dwell_times = _compute_dwell_times_from_events(filtered)
    avg_dwell = statistics.mean(dwell_times) if dwell_times else 0.0

    last_visit = entries[-1].timestamp if entries else 0.0

    return FrequencyReport(
        target_id=target_id,
        zone_id=area_id,
        total_visits=total_visits,
        visits_per_day=visits_per_day,
        visits_per_week=visits_per_week,
        time_range=(start, end),
        peak_hours=peak_hours,
        avg_dwell_seconds=avg_dwell,
        last_visit=last_visit,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_dwell_times_from_events(
    events: Sequence[AccessEvent],
) -> list[float]:
    """Pair enter/exit events per target and compute dwell durations."""
    by_target: dict[str, list[AccessEvent]] = defaultdict(list)
    for e in events:
        by_target[e.target_id].append(e)

    dwell_times: list[float] = []
    for tid, tevents in by_target.items():
        tevents_sorted = sorted(tevents, key=lambda e: e.timestamp)
        enter_time: float | None = None
        for ev in tevents_sorted:
            if ev.event_type == "enter":
                enter_time = ev.timestamp
            elif ev.event_type == "exit" and enter_time is not None:
                dwell = ev.timestamp - enter_time
                if dwell >= 0:
                    dwell_times.append(dwell)
                enter_time = None

    return dwell_times


def _severity_from_score(score: float) -> str:
    """Map a 0-1 anomaly score to a severity string."""
    if score >= 0.8:
        return "critical"
    elif score >= 0.6:
        return "high"
    elif score >= 0.4:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# AccessPatternAnalyzer
# ---------------------------------------------------------------------------

class AccessPatternAnalyzer:
    """Learn and detect deviations in how targets access zones.

    Thread-safe. Maintains a history of access events and can learn
    per-target-per-zone access patterns. Once patterns are learned,
    new events can be checked for anomalies.

    Parameters
    ----------
    event_bus : optional
        An event bus with a ``.publish(topic, data)`` method for
        broadcasting access anomaly events.
    sigma_threshold : float
        Number of standard deviations before flagging an anomaly.
    max_events : int
        Maximum events retained per (target, zone) pair.
    """

    def __init__(
        self,
        event_bus: Any = None,
        sigma_threshold: float = DEFAULT_ANOMALY_SIGMA,
        max_events: int = MAX_EVENTS_PER_TARGET_ZONE,
    ) -> None:
        self._event_bus = event_bus
        self._sigma_threshold = sigma_threshold
        self._max_events = max_events
        self._lock = threading.Lock()

        # Events indexed by (target_id, zone_id) -> list[AccessEvent]
        self._events: dict[tuple[str, str], list[AccessEvent]] = defaultdict(list)

        # All events in chronological order per zone (for tailgating etc.)
        self._zone_events: dict[str, list[AccessEvent]] = defaultdict(list)

        # Learned patterns: (target_id, zone_id) -> AccessPattern
        self._patterns: dict[tuple[str, str], AccessPattern] = {}

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_access(self, event: AccessEvent) -> None:
        """Record a single access event.

        Parameters
        ----------
        event : AccessEvent
            The access event to record.
        """
        key = (event.target_id, event.zone_id)
        with self._lock:
            self._events[key].append(event)
            if len(self._events[key]) > self._max_events:
                self._events[key] = self._events[key][-self._max_events:]

            self._zone_events[event.zone_id].append(event)
            if len(self._zone_events[event.zone_id]) > self._max_events:
                self._zone_events[event.zone_id] = (
                    self._zone_events[event.zone_id][-self._max_events:]
                )

    def record_access_batch(self, events: Sequence[AccessEvent]) -> int:
        """Record multiple access events.

        Returns the number of events recorded.
        """
        count = 0
        for ev in events:
            if ev.event_type in ("enter", "exit"):
                self.record_access(ev)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Pattern learning
    # ------------------------------------------------------------------

    def learn_pattern(
        self,
        target_id: str,
        zone_id: str,
    ) -> AccessPattern | None:
        """Learn the typical access pattern for a target in a zone.

        Requires at least :data:`MIN_EVENTS_FOR_PATTERN` events.

        Parameters
        ----------
        target_id : str
        zone_id : str

        Returns
        -------
        AccessPattern or None
            The learned pattern, or ``None`` if insufficient data.
        """
        key = (target_id, zone_id)
        with self._lock:
            events = list(self._events.get(key, []))

        entries = [e for e in events if e.event_type == "enter"]
        exits = [e for e in events if e.event_type == "exit"]

        if len(entries) < MIN_EVENTS_FOR_PATTERN:
            return None

        # Dwell times
        dwell_times = _compute_dwell_times_from_events(events)

        avg_dwell = statistics.mean(dwell_times) if dwell_times else 0.0
        std_dwell = statistics.stdev(dwell_times) if len(dwell_times) >= 2 else 0.0
        min_dwell = min(dwell_times) if dwell_times else 0.0
        max_dwell = max(dwell_times) if dwell_times else 0.0

        # Inter-entry intervals
        entry_timestamps = sorted(e.timestamp for e in entries)
        intervals: list[float] = []
        for i in range(1, len(entry_timestamps)):
            intervals.append(entry_timestamps[i] - entry_timestamps[i - 1])

        avg_interval = statistics.mean(intervals) if intervals else 0.0
        std_interval = statistics.stdev(intervals) if len(intervals) >= 2 else 0.0

        # Hourly distribution
        hourly: dict[int, int] = defaultdict(int)
        for e in entries:
            hour = time.localtime(e.timestamp).tm_hour
            hourly[hour] += 1

        # Day of week distribution
        dow: dict[int, int] = defaultdict(int)
        for e in entries:
            day = time.localtime(e.timestamp).tm_wday
            dow[day] += 1

        all_timestamps = [e.timestamp for e in events]
        first_seen = min(all_timestamps) if all_timestamps else 0.0
        last_seen = max(all_timestamps) if all_timestamps else 0.0

        pattern = AccessPattern(
            target_id=target_id,
            zone_id=zone_id,
            total_entries=len(entries),
            total_exits=len(exits),
            avg_dwell_seconds=avg_dwell,
            std_dwell_seconds=std_dwell,
            min_dwell_seconds=min_dwell,
            max_dwell_seconds=max_dwell,
            avg_interval_seconds=avg_interval,
            std_interval_seconds=std_interval,
            hourly_distribution=dict(hourly),
            day_of_week_distribution=dict(dow),
            first_seen=first_seen,
            last_seen=last_seen,
            learned_at=time.time(),
        )

        with self._lock:
            self._patterns[key] = pattern

        logger.info(
            "Learned access pattern for %s in %s: %d entries, avg_dwell=%.1fs",
            target_id, zone_id, len(entries), avg_dwell,
        )
        return pattern

    def get_pattern(
        self,
        target_id: str,
        zone_id: str,
    ) -> AccessPattern | None:
        """Retrieve the learned access pattern for a target in a zone."""
        with self._lock:
            return self._patterns.get((target_id, zone_id))

    def get_all_patterns(self) -> list[AccessPattern]:
        """Return all learned patterns."""
        with self._lock:
            return list(self._patterns.values())

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def check_access(
        self,
        event: AccessEvent,
    ) -> list[AccessAnomaly]:
        """Check a new access event against the learned pattern.

        Detects:
        - **unusual_time**: Entry at an unusual hour compared to learned pattern.
        - **unusual_frequency**: Entry interval significantly shorter or longer
          than the learned average.
        - **unusual_dwell**: Dwell time significantly different from learned average.
          (Only checked on exit events.)
        - **new_zone**: Target has never been seen in this zone before.

        Parameters
        ----------
        event : AccessEvent
            The event to check.

        Returns
        -------
        list[AccessAnomaly]
            Anomalies detected (may be empty).
        """
        anomalies: list[AccessAnomaly] = []
        key = (event.target_id, event.zone_id)

        with self._lock:
            pattern = self._patterns.get(key)
            past_events = list(self._events.get(key, []))
            all_keys = set(self._events.keys())

        # Check if this is a completely new zone for this target
        target_zones = {k[1] for k in all_keys if k[0] == event.target_id}
        if event.zone_id not in target_zones and event.event_type == "enter":
            anomalies.append(AccessAnomaly(
                target_id=event.target_id,
                zone_id=event.zone_id,
                anomaly_type="new_zone",
                description=(
                    f"Target {event.target_id} entered zone {event.zone_id} "
                    f"for the first time"
                ),
                score=0.5,
                severity="medium",
                timestamp=event.timestamp,
            ))

        if pattern is None:
            # No learned pattern — can only flag new_zone
            self._publish_anomalies(anomalies)
            return anomalies

        # Check unusual time (enter events only)
        if event.event_type == "enter" and pattern.hourly_distribution:
            hour = time.localtime(event.timestamp).tm_hour
            total_dist = sum(pattern.hourly_distribution.values())
            hour_count = pattern.hourly_distribution.get(hour, 0)
            hour_fraction = hour_count / total_dist if total_dist > 0 else 0.0

            # If this hour has zero or very few observations, it is unusual
            if total_dist >= MIN_EVENTS_FOR_PATTERN and hour_fraction < 0.02:
                score = min(1.0, 1.0 - hour_fraction * 50)
                anomalies.append(AccessAnomaly(
                    target_id=event.target_id,
                    zone_id=event.zone_id,
                    anomaly_type="unusual_time",
                    description=(
                        f"Target {event.target_id} accessed zone "
                        f"{event.zone_id} at hour {hour}, which has only "
                        f"{hour_count}/{total_dist} historical entries"
                    ),
                    score=score,
                    severity=_severity_from_score(score),
                    timestamp=event.timestamp,
                    details={
                        "hour": hour,
                        "hour_count": hour_count,
                        "total_entries": total_dist,
                        "hour_fraction": round(hour_fraction, 4),
                    },
                ))

        # Check unusual frequency (enter events)
        if (
            event.event_type == "enter"
            and pattern.avg_interval_seconds > 0
        ):
            # Find the most recent entry
            past_entries = [e for e in past_events if e.event_type == "enter"]
            if past_entries:
                last_entry = max(past_entries, key=lambda e: e.timestamp)
                interval = event.timestamp - last_entry.timestamp
                # Use std if available; if std is 0 (perfectly regular),
                # use 10% of the mean as the deviation reference.
                std_ref = pattern.std_interval_seconds
                if std_ref < 1.0:
                    std_ref = max(pattern.avg_interval_seconds * 0.1, 1.0)
                z_score = abs(interval - pattern.avg_interval_seconds) / std_ref

                if z_score > self._sigma_threshold:
                    score = min(1.0, z_score / (self._sigma_threshold * 3))
                    anomalies.append(AccessAnomaly(
                        target_id=event.target_id,
                        zone_id=event.zone_id,
                        anomaly_type="unusual_frequency",
                        description=(
                            f"Target {event.target_id} interval to zone "
                            f"{event.zone_id} is {interval:.0f}s "
                            f"(expected {pattern.avg_interval_seconds:.0f}s "
                            f"+/- {pattern.std_interval_seconds:.0f}s, "
                            f"z={z_score:.1f})"
                        ),
                        score=score,
                        severity=_severity_from_score(score),
                        timestamp=event.timestamp,
                        details={
                            "interval_seconds": round(interval, 2),
                            "expected_interval": round(pattern.avg_interval_seconds, 2),
                            "std_interval": round(pattern.std_interval_seconds, 2),
                            "z_score": round(z_score, 2),
                        },
                    ))

        # Check unusual dwell (exit events — compute from last enter)
        if (
            event.event_type == "exit"
            and pattern.avg_dwell_seconds > 0
        ):
            past_entries = sorted(
                [e for e in past_events if e.event_type == "enter"],
                key=lambda e: e.timestamp,
            )
            if past_entries:
                # Find the most recent enter before this exit
                last_enter = None
                for e in reversed(past_entries):
                    if e.timestamp <= event.timestamp:
                        last_enter = e
                        break

                if last_enter is not None:
                    dwell = event.timestamp - last_enter.timestamp
                    # Use std if available; if std is 0 (perfectly regular),
                    # use 10% of the mean as the deviation reference.
                    std_ref = pattern.std_dwell_seconds
                    if std_ref < 1.0:
                        std_ref = max(pattern.avg_dwell_seconds * 0.1, 1.0)
                    z_score = abs(dwell - pattern.avg_dwell_seconds) / std_ref

                    if z_score > self._sigma_threshold:
                        score = min(1.0, z_score / (self._sigma_threshold * 3))
                        anomalies.append(AccessAnomaly(
                            target_id=event.target_id,
                            zone_id=event.zone_id,
                            anomaly_type="unusual_dwell",
                            description=(
                                f"Target {event.target_id} dwell in zone "
                                f"{event.zone_id} is {dwell:.0f}s "
                                f"(expected {pattern.avg_dwell_seconds:.0f}s "
                                f"+/- {pattern.std_dwell_seconds:.0f}s, "
                                f"z={z_score:.1f})"
                            ),
                            score=score,
                            severity=_severity_from_score(score),
                            timestamp=event.timestamp,
                            details={
                                "dwell_seconds": round(dwell, 2),
                                "expected_dwell": round(pattern.avg_dwell_seconds, 2),
                                "std_dwell": round(pattern.std_dwell_seconds, 2),
                                "z_score": round(z_score, 2),
                            },
                        ))

        self._publish_anomalies(anomalies)
        return anomalies

    # ------------------------------------------------------------------
    # Zone-level analysis helpers
    # ------------------------------------------------------------------

    def detect_tailgating_in_zone(
        self,
        zone_id: str,
        threshold_seconds: float = DEFAULT_TAILGATE_THRESHOLD_SECONDS,
        time_range: tuple[float, float] | None = None,
    ) -> list[TailgateAlert]:
        """Detect tailgating in a specific zone from recorded events.

        Parameters
        ----------
        zone_id : str
        threshold_seconds : float
        time_range : tuple[float, float] or None
            Optional time filter.

        Returns
        -------
        list[TailgateAlert]
        """
        with self._lock:
            events = list(self._zone_events.get(zone_id, []))

        if time_range:
            start, end = time_range
            events = [e for e in events if start <= e.timestamp <= end]

        return detect_tailgating(events, threshold_seconds)

    def detect_piggybacking_in_zone(
        self,
        zone_id: str,
        authorized_targets: set[str] | None = None,
        follow_window_seconds: float = 10.0,
        time_range: tuple[float, float] | None = None,
    ) -> list[PiggybackAlert]:
        """Detect piggybacking in a specific zone from recorded events.

        Parameters
        ----------
        zone_id : str
        authorized_targets : set[str] or None
        follow_window_seconds : float
        time_range : tuple[float, float] or None
            Optional time filter.

        Returns
        -------
        list[PiggybackAlert]
        """
        with self._lock:
            events = list(self._zone_events.get(zone_id, []))

        if time_range:
            start, end = time_range
            events = [e for e in events if start <= e.timestamp <= end]

        return detect_piggybacking(events, authorized_targets, follow_window_seconds)

    def frequency_analysis_for_target(
        self,
        target_id: str,
        zone_id: str,
        time_range: tuple[float, float] | None = None,
    ) -> FrequencyReport:
        """Run frequency analysis for a target in a zone using recorded events.

        Parameters
        ----------
        target_id : str
        zone_id : str
        time_range : tuple[float, float] or None

        Returns
        -------
        FrequencyReport
        """
        key = (target_id, zone_id)
        with self._lock:
            events = list(self._events.get(key, []))

        return frequency_analysis(target_id, zone_id, events, time_range)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_events(
        self,
        target_id: str | None = None,
        zone_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[AccessEvent]:
        """Retrieve recorded access events, optionally filtered.

        Parameters
        ----------
        target_id : str or None
        zone_id : str or None
        event_type : str or None
        limit : int

        Returns
        -------
        list[AccessEvent]
            Most recent events first.
        """
        with self._lock:
            if target_id and zone_id:
                events = list(self._events.get((target_id, zone_id), []))
            elif zone_id:
                events = list(self._zone_events.get(zone_id, []))
            elif target_id:
                events = []
                for key, evts in self._events.items():
                    if key[0] == target_id:
                        events.extend(evts)
            else:
                events = []
                for evts in self._zone_events.values():
                    events.extend(evts)

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        events.sort(key=lambda e: e.timestamp)
        return list(reversed(events[-limit:]))

    def get_zone_ids(self) -> list[str]:
        """Return all zone IDs that have recorded events."""
        with self._lock:
            return list(self._zone_events.keys())

    def get_target_ids(self, zone_id: str | None = None) -> list[str]:
        """Return all target IDs (optionally filtered to a zone)."""
        with self._lock:
            if zone_id:
                return list({
                    k[0] for k in self._events.keys() if k[1] == zone_id
                })
            return list({k[0] for k in self._events.keys()})

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        with self._lock:
            total_events = sum(len(v) for v in self._zone_events.values())
            zone_count = len(self._zone_events)
            target_zone_pairs = len(self._events)
            pattern_count = len(self._patterns)
        return {
            "total_events": total_events,
            "zone_count": zone_count,
            "target_zone_pairs": target_zone_pairs,
            "pattern_count": pattern_count,
            "max_events_per_pair": self._max_events,
            "sigma_threshold": self._sigma_threshold,
        }

    def clear(
        self,
        target_id: str | None = None,
        zone_id: str | None = None,
    ) -> None:
        """Clear stored events and patterns.

        If both target_id and zone_id are given, clears that pair only.
        If only zone_id, clears all events for that zone.
        If only target_id, clears all events for that target.
        If neither, clears everything.
        """
        with self._lock:
            if target_id and zone_id:
                key = (target_id, zone_id)
                self._events.pop(key, None)
                self._patterns.pop(key, None)
            elif zone_id:
                self._zone_events.pop(zone_id, None)
                keys_to_remove = [k for k in self._events if k[1] == zone_id]
                for k in keys_to_remove:
                    self._events.pop(k, None)
                    self._patterns.pop(k, None)
            elif target_id:
                keys_to_remove = [k for k in self._events if k[0] == target_id]
                for k in keys_to_remove:
                    self._events.pop(k, None)
                    self._patterns.pop(k, None)
            else:
                self._events.clear()
                self._zone_events.clear()
                self._patterns.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish_anomalies(self, anomalies: list[AccessAnomaly]) -> None:
        """Publish anomalies to event bus if attached."""
        if self._event_bus is None or not anomalies:
            return
        for anomaly in anomalies:
            try:
                self._event_bus.publish("access:anomaly", anomaly.to_dict())
            except Exception:
                logger.warning("Failed to publish access anomaly", exc_info=True)

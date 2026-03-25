# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BehavioralPatternLearner — learns normal behavioral patterns from tracking data.

Learns individual target routines (routes, schedules, zones) from historical
position data and alerts when a target deviates from its established pattern.
Uses simple statistics (mean, std, histograms) — no ML dependencies.

Integrates with:
  - TargetHistory  — position trail source
  - MovementPatternAnalyzer — loitering / route detection
  - DwellTracker — zone dwell data

Usage::

    from tritium_lib.tracking import TargetHistory, MovementPatternAnalyzer
    from tritium_lib.intelligence.behavioral_pattern_learner import BehavioralPatternLearner

    history = TargetHistory()
    analyzer = MovementPatternAnalyzer(history)
    learner = BehavioralPatternLearner(history=history, analyzer=analyzer)

    # Feed positions for a target over time...
    # Then learn its patterns:
    learner.learn_route("ble_aa:bb:cc")
    learner.learn_schedule("ble_aa:bb:cc")

    # Later, check if a new position is a deviation:
    result = learner.detect_deviation("ble_aa:bb:cc", (120.5, 45.3))

    # Full behavioral profile:
    profile = learner.get_profile("ble_aa:bb:cc")
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum observations needed to learn a route
MIN_ROUTE_POINTS = 10

# Minimum observations needed to learn a schedule
MIN_SCHEDULE_OBSERVATIONS = 5

# How far (in units) a position can deviate before flagging
DEFAULT_ROUTE_DEVIATION_THRESHOLD = 3.0  # multiples of route std

# How far (in seconds) a schedule can deviate before flagging
DEFAULT_SCHEDULE_DEVIATION_THRESHOLD = 2.0  # multiples of schedule std

# Number of time-of-day bins for schedule histograms (1 bin = 1 hour)
SCHEDULE_BINS = 24

# Number of day-of-week bins
DOW_BINS = 7

# Maximum number of waypoints stored per learned route
MAX_ROUTE_WAYPOINTS = 200

# Maximum number of zones stored per target
MAX_ZONES = 50


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LearnedWaypoint:
    """A single waypoint on a learned route with positional statistics."""

    x: float
    y: float
    std_x: float = 0.0
    std_y: float = 0.0
    observation_count: int = 1


@dataclass
class LearnedRoute:
    """A learned regular route for a target."""

    target_id: str
    waypoints: list[LearnedWaypoint] = field(default_factory=list)
    total_observations: int = 0
    mean_duration_s: float = 0.0
    std_duration_s: float = 0.0
    last_updated: float = 0.0

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "waypoint_count": len(self.waypoints),
            "total_observations": self.total_observations,
            "mean_duration_s": round(self.mean_duration_s, 1),
            "std_duration_s": round(self.std_duration_s, 1),
            "last_updated": self.last_updated,
            "waypoints": [
                {"x": w.x, "y": w.y, "std_x": w.std_x, "std_y": w.std_y}
                for w in self.waypoints
            ],
        }


@dataclass
class LearnedSchedule:
    """A learned time-of-day / day-of-week schedule for a target."""

    target_id: str
    hourly_histogram: list[int] = field(default_factory=lambda: [0] * SCHEDULE_BINS)
    dow_histogram: list[int] = field(default_factory=lambda: [0] * DOW_BINS)
    arrival_times: list[float] = field(default_factory=list)
    departure_times: list[float] = field(default_factory=list)
    mean_arrival_hour: float = 0.0
    std_arrival_hour: float = 0.0
    mean_departure_hour: float = 0.0
    std_departure_hour: float = 0.0
    total_observations: int = 0
    last_updated: float = 0.0

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "hourly_histogram": list(self.hourly_histogram),
            "dow_histogram": list(self.dow_histogram),
            "mean_arrival_hour": round(self.mean_arrival_hour, 2),
            "std_arrival_hour": round(self.std_arrival_hour, 2),
            "mean_departure_hour": round(self.mean_departure_hour, 2),
            "std_departure_hour": round(self.std_departure_hour, 2),
            "total_observations": self.total_observations,
            "last_updated": self.last_updated,
            "peak_hours": self._peak_hours(),
            "peak_days": self._peak_days(),
        }

    def _peak_hours(self, top_n: int = 3) -> list[int]:
        """Return the top-N most active hours."""
        indexed = [(count, hour) for hour, count in enumerate(self.hourly_histogram)]
        indexed.sort(reverse=True)
        return [hour for count, hour in indexed[:top_n] if count > 0]

    def _peak_days(self, top_n: int = 3) -> list[int]:
        """Return the top-N most active days (0=Mon)."""
        indexed = [(count, day) for day, count in enumerate(self.dow_histogram)]
        indexed.sort(reverse=True)
        return [day for count, day in indexed[:top_n] if count > 0]


@dataclass
class FrequentZone:
    """A zone the target frequently visits."""

    center_x: float
    center_y: float
    radius: float
    visit_count: int = 0
    total_dwell_s: float = 0.0
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "radius": round(self.radius, 2),
            "visit_count": self.visit_count,
            "total_dwell_s": round(self.total_dwell_s, 1),
            "label": self.label,
        }


@dataclass
class DeviationResult:
    """Result of a deviation check."""

    is_deviation: bool = False
    deviation_type: str = ""  # "route", "schedule", "zone", ""
    severity: float = 0.0  # 0.0 to 1.0
    distance_from_expected: float = 0.0
    nearest_waypoint_index: int = -1
    sigma: float = 0.0  # how many standard deviations away
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_deviation": self.is_deviation,
            "deviation_type": self.deviation_type,
            "severity": round(self.severity, 3),
            "distance_from_expected": round(self.distance_from_expected, 2),
            "nearest_waypoint_index": self.nearest_waypoint_index,
            "sigma": round(self.sigma, 2),
            "details": self.details,
        }


@dataclass
class BehavioralProfile:
    """Complete behavioral profile for a target."""

    target_id: str
    route: Optional[LearnedRoute] = None
    schedule: Optional[LearnedSchedule] = None
    zones: list[FrequentZone] = field(default_factory=list)
    regularity_score: float = 0.0  # 0.0 (unpredictable) to 1.0 (clockwork)
    total_observations: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "has_route": self.route is not None,
            "has_schedule": self.schedule is not None,
            "zone_count": len(self.zones),
            "regularity_score": round(self.regularity_score, 3),
            "total_observations": self.total_observations,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "route": self.route.to_dict() if self.route else None,
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "zones": [z.to_dict() for z in self.zones],
        }


# ---------------------------------------------------------------------------
# Observation record for schedule learning
# ---------------------------------------------------------------------------

@dataclass
class ScheduleObservation:
    """A single timestamp observation for schedule learning."""

    timestamp: float  # epoch seconds
    hour: float  # fractional hour (0.0 - 23.99)
    day_of_week: int  # 0=Mon, 6=Sun
    is_arrival: bool = True


# ---------------------------------------------------------------------------
# BehavioralPatternLearner
# ---------------------------------------------------------------------------

class BehavioralPatternLearner:
    """Learns normal behavioral patterns from tracking data and alerts on deviations.

    Works with TargetHistory for position trails, MovementPatternAnalyzer for
    pattern detection, and optionally DwellTracker for zone data.

    Thread-safe: all state is guarded by a lock.
    """

    def __init__(
        self,
        history=None,
        analyzer=None,
        dwell_tracker=None,
        route_deviation_threshold: float = DEFAULT_ROUTE_DEVIATION_THRESHOLD,
        schedule_deviation_threshold: float = DEFAULT_SCHEDULE_DEVIATION_THRESHOLD,
    ) -> None:
        """Initialize the behavioral pattern learner.

        Args:
            history: TargetHistory instance for position trails.
            analyzer: MovementPatternAnalyzer instance.
            dwell_tracker: Optional DwellTracker instance.
            route_deviation_threshold: Sigma threshold for route deviations.
            schedule_deviation_threshold: Sigma threshold for schedule deviations.
        """
        self._history = history
        self._analyzer = analyzer
        self._dwell_tracker = dwell_tracker
        self._route_threshold = route_deviation_threshold
        self._schedule_threshold = schedule_deviation_threshold

        # Per-target learned data
        self._routes: dict[str, LearnedRoute] = {}
        self._schedules: dict[str, LearnedSchedule] = {}
        self._zones: dict[str, list[FrequentZone]] = {}
        self._observations: dict[str, list[ScheduleObservation]] = {}

        import threading
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Route learning
    # ------------------------------------------------------------------

    def learn_route(
        self,
        target_id: str,
        trail: list[tuple[float, float, float]] | None = None,
    ) -> LearnedRoute | None:
        """Learn a target's regular route from its position trail.

        If trail is None, reads from the attached TargetHistory.

        Args:
            target_id: The target to learn a route for.
            trail: Optional explicit trail as (x, y, timestamp) tuples.

        Returns:
            LearnedRoute if enough data, None otherwise.
        """
        if trail is None:
            if self._history is None:
                log.warning("learn_route: no history source for %s", target_id)
                return None
            trail = self._history.get_trail(target_id, max_points=1000)

        if len(trail) < MIN_ROUTE_POINTS:
            log.debug(
                "learn_route: not enough points for %s (%d < %d)",
                target_id, len(trail), MIN_ROUTE_POINTS,
            )
            return None

        # Simplify trail into waypoints using Douglas-Peucker-like downsampling
        waypoints = self._simplify_trail(trail)

        # Compute positional statistics at each waypoint
        learned_waypoints = self._compute_waypoint_stats(trail, waypoints)

        # Compute duration statistics
        durations = self._compute_segment_durations(trail)
        mean_dur, std_dur = _mean_std(durations) if durations else (0.0, 0.0)

        route = LearnedRoute(
            target_id=target_id,
            waypoints=learned_waypoints,
            total_observations=len(trail),
            mean_duration_s=mean_dur,
            std_duration_s=std_dur,
            last_updated=time.time(),
        )

        with self._lock:
            existing = self._routes.get(target_id)
            if existing is not None:
                # Merge with existing route (incremental learning)
                route = self._merge_routes(existing, route)
            self._routes[target_id] = route

        log.info(
            "learn_route: %s — %d waypoints from %d observations",
            target_id, len(route.waypoints), route.total_observations,
        )
        return route

    def _simplify_trail(
        self,
        trail: list[tuple[float, float, float]],
    ) -> list[tuple[float, float]]:
        """Downsample a trail to representative waypoints.

        Uses uniform sampling with a cap at MAX_ROUTE_WAYPOINTS.
        """
        n = len(trail)
        if n <= MAX_ROUTE_WAYPOINTS:
            return [(x, y) for x, y, _ in trail]

        step = max(1, n // MAX_ROUTE_WAYPOINTS)
        result = []
        for i in range(0, n, step):
            x, y, _ = trail[i]
            result.append((x, y))
        # Always include the last point
        last_x, last_y, _ = trail[-1]
        if result[-1] != (last_x, last_y):
            result.append((last_x, last_y))
        return result[:MAX_ROUTE_WAYPOINTS]

    def _compute_waypoint_stats(
        self,
        trail: list[tuple[float, float, float]],
        waypoints: list[tuple[float, float]],
    ) -> list[LearnedWaypoint]:
        """Compute mean/std for each waypoint from nearby trail points."""
        if not waypoints:
            return []

        result = []
        n_trail = len(trail)
        n_wp = len(waypoints)

        for i, (wx, wy) in enumerate(waypoints):
            # Find trail points closest to this waypoint index
            # Map waypoint index to trail index range
            start = max(0, int((i / n_wp) * n_trail) - 2)
            end = min(n_trail, int(((i + 1) / n_wp) * n_trail) + 2)

            nearby_x = [trail[j][0] for j in range(start, end)]
            nearby_y = [trail[j][1] for j in range(start, end)]

            if nearby_x:
                mean_x = sum(nearby_x) / len(nearby_x)
                mean_y = sum(nearby_y) / len(nearby_y)
                var_x = sum((v - mean_x) ** 2 for v in nearby_x) / max(len(nearby_x), 1)
                var_y = sum((v - mean_y) ** 2 for v in nearby_y) / max(len(nearby_y), 1)
                std_x = math.sqrt(var_x) if var_x > 0 else 0.1
                std_y = math.sqrt(var_y) if var_y > 0 else 0.1
            else:
                mean_x, mean_y = wx, wy
                std_x = std_y = 0.1

            result.append(LearnedWaypoint(
                x=mean_x,
                y=mean_y,
                std_x=std_x,
                std_y=std_y,
                observation_count=len(nearby_x),
            ))

        return result

    def _compute_segment_durations(
        self,
        trail: list[tuple[float, float, float]],
    ) -> list[float]:
        """Compute durations between consecutive trail points."""
        durations = []
        for i in range(1, len(trail)):
            dt = trail[i][2] - trail[i - 1][2]
            if dt > 0:
                durations.append(dt)
        return durations

    def _merge_routes(
        self,
        existing: LearnedRoute,
        new: LearnedRoute,
    ) -> LearnedRoute:
        """Merge a new route observation with an existing one.

        Uses exponential moving average for waypoint positions and statistics.
        """
        alpha = 0.3  # learning rate for incremental updates

        # Match waypoints by index (both are uniformly sampled)
        merged_waypoints = []
        n = max(len(existing.waypoints), len(new.waypoints))

        for i in range(n):
            if i < len(existing.waypoints) and i < len(new.waypoints):
                ew = existing.waypoints[i]
                nw = new.waypoints[i]
                merged_waypoints.append(LearnedWaypoint(
                    x=ew.x * (1 - alpha) + nw.x * alpha,
                    y=ew.y * (1 - alpha) + nw.y * alpha,
                    std_x=ew.std_x * (1 - alpha) + nw.std_x * alpha,
                    std_y=ew.std_y * (1 - alpha) + nw.std_y * alpha,
                    observation_count=ew.observation_count + nw.observation_count,
                ))
            elif i < len(existing.waypoints):
                merged_waypoints.append(existing.waypoints[i])
            else:
                merged_waypoints.append(new.waypoints[i])

        return LearnedRoute(
            target_id=existing.target_id,
            waypoints=merged_waypoints,
            total_observations=existing.total_observations + new.total_observations,
            mean_duration_s=(
                existing.mean_duration_s * (1 - alpha)
                + new.mean_duration_s * alpha
            ),
            std_duration_s=(
                existing.std_duration_s * (1 - alpha)
                + new.std_duration_s * alpha
            ),
            last_updated=time.time(),
        )

    # ------------------------------------------------------------------
    # Schedule learning
    # ------------------------------------------------------------------

    def learn_schedule(
        self,
        target_id: str,
        timestamps: list[float] | None = None,
    ) -> LearnedSchedule | None:
        """Learn a target's arrival/departure schedule from observation timestamps.

        If timestamps is None, reads from the attached TargetHistory.

        Args:
            target_id: The target to learn a schedule for.
            timestamps: Optional list of epoch timestamps.

        Returns:
            LearnedSchedule if enough data, None otherwise.
        """
        if timestamps is None:
            if self._history is None:
                log.warning("learn_schedule: no history source for %s", target_id)
                return None
            trail = self._history.get_trail(target_id, max_points=1000)
            timestamps = [t for _, _, t in trail]

        if len(timestamps) < MIN_SCHEDULE_OBSERVATIONS:
            log.debug(
                "learn_schedule: not enough observations for %s (%d < %d)",
                target_id, len(timestamps), MIN_SCHEDULE_OBSERVATIONS,
            )
            return None

        # Convert timestamps to schedule observations
        observations = []
        for ts in timestamps:
            obs = self._timestamp_to_observation(ts)
            observations.append(obs)

        # Build hourly and day-of-week histograms
        hourly = [0] * SCHEDULE_BINS
        dow = [0] * DOW_BINS
        hours = []

        for obs in observations:
            hour_bin = int(obs.hour) % SCHEDULE_BINS
            hourly[hour_bin] += 1
            dow[obs.day_of_week % DOW_BINS] += 1
            hours.append(obs.hour)

        # Detect arrival/departure patterns from gaps
        arrivals, departures = self._detect_arrival_departure(timestamps)

        mean_arrival, std_arrival = _mean_std(arrivals) if arrivals else (0.0, 0.0)
        mean_departure, std_departure = _mean_std(departures) if departures else (0.0, 0.0)

        schedule = LearnedSchedule(
            target_id=target_id,
            hourly_histogram=hourly,
            dow_histogram=dow,
            arrival_times=arrivals[-50:],  # keep last 50
            departure_times=departures[-50:],
            mean_arrival_hour=mean_arrival,
            std_arrival_hour=std_arrival,
            mean_departure_hour=mean_departure,
            std_departure_hour=std_departure,
            total_observations=len(timestamps),
            last_updated=time.time(),
        )

        with self._lock:
            existing = self._schedules.get(target_id)
            if existing is not None:
                schedule = self._merge_schedules(existing, schedule)
            self._schedules[target_id] = schedule
            self._observations[target_id] = observations[-200:]

        log.info(
            "learn_schedule: %s — %d observations, arrival ~%.1fh, departure ~%.1fh",
            target_id, schedule.total_observations,
            schedule.mean_arrival_hour, schedule.mean_departure_hour,
        )
        return schedule

    @staticmethod
    def _timestamp_to_observation(ts: float) -> ScheduleObservation:
        """Convert an epoch timestamp to a ScheduleObservation."""
        import datetime
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        dow = dt.weekday()  # 0=Mon
        return ScheduleObservation(
            timestamp=ts,
            hour=hour,
            day_of_week=dow,
        )

    def _detect_arrival_departure(
        self,
        timestamps: list[float],
    ) -> tuple[list[float], list[float]]:
        """Detect arrival and departure times from a sorted timestamp list.

        An arrival is the first observation after a gap > 1 hour.
        A departure is the last observation before a gap > 1 hour.
        """
        if len(timestamps) < 2:
            return [], []

        sorted_ts = sorted(timestamps)
        gap_threshold = 3600.0  # 1 hour gap = new session

        arrivals: list[float] = []
        departures: list[float] = []

        # First timestamp is always an arrival
        arr_obs = self._timestamp_to_observation(sorted_ts[0])
        arrivals.append(arr_obs.hour)

        for i in range(1, len(sorted_ts)):
            gap = sorted_ts[i] - sorted_ts[i - 1]
            if gap >= gap_threshold:
                # End of previous session = departure
                dep_obs = self._timestamp_to_observation(sorted_ts[i - 1])
                departures.append(dep_obs.hour)
                # Start of new session = arrival
                arr_obs = self._timestamp_to_observation(sorted_ts[i])
                arrivals.append(arr_obs.hour)

        # Last timestamp is always a departure
        dep_obs = self._timestamp_to_observation(sorted_ts[-1])
        departures.append(dep_obs.hour)

        return arrivals, departures

    def _merge_schedules(
        self,
        existing: LearnedSchedule,
        new: LearnedSchedule,
    ) -> LearnedSchedule:
        """Merge a new schedule observation with an existing one."""
        merged_hourly = [
            existing.hourly_histogram[i] + new.hourly_histogram[i]
            for i in range(SCHEDULE_BINS)
        ]
        merged_dow = [
            existing.dow_histogram[i] + new.dow_histogram[i]
            for i in range(DOW_BINS)
        ]

        all_arrivals = existing.arrival_times + new.arrival_times
        all_departures = existing.departure_times + new.departure_times

        # Keep last 100
        all_arrivals = all_arrivals[-100:]
        all_departures = all_departures[-100:]

        mean_arrival, std_arrival = _mean_std(all_arrivals) if all_arrivals else (0.0, 0.0)
        mean_departure, std_departure = _mean_std(all_departures) if all_departures else (0.0, 0.0)

        return LearnedSchedule(
            target_id=existing.target_id,
            hourly_histogram=merged_hourly,
            dow_histogram=merged_dow,
            arrival_times=all_arrivals,
            departure_times=all_departures,
            mean_arrival_hour=mean_arrival,
            std_arrival_hour=std_arrival,
            mean_departure_hour=mean_departure,
            std_departure_hour=std_departure,
            total_observations=existing.total_observations + new.total_observations,
            last_updated=time.time(),
        )

    # ------------------------------------------------------------------
    # Zone learning
    # ------------------------------------------------------------------

    def learn_zones(
        self,
        target_id: str,
        trail: list[tuple[float, float, float]] | None = None,
        cluster_radius: float = 10.0,
        min_dwell_s: float = 60.0,
    ) -> list[FrequentZone]:
        """Learn frequently visited zones from dwell clusters.

        Uses a simple greedy clustering: find the densest cluster center,
        absorb nearby points, repeat.

        Args:
            target_id: Target to learn zones for.
            trail: Optional explicit trail.
            cluster_radius: Radius to cluster nearby points.
            min_dwell_s: Minimum total dwell time in a cluster.

        Returns:
            List of FrequentZone objects.
        """
        if trail is None:
            if self._history is None:
                return []
            trail = self._history.get_trail(target_id, max_points=1000)

        if len(trail) < 3:
            return []

        # Find dwell clusters: points where the target was slow/stationary
        dwell_points = self._extract_dwell_points(trail)
        if not dwell_points:
            return []

        zones = self._cluster_dwell_points(dwell_points, cluster_radius, min_dwell_s)

        with self._lock:
            self._zones[target_id] = zones[:MAX_ZONES]

        log.info("learn_zones: %s — %d zones", target_id, len(zones))
        return zones

    def _extract_dwell_points(
        self,
        trail: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        """Extract points where the target was slow or stationary."""
        speed_threshold = 0.5  # units per second
        dwell_points = []

        for i in range(1, len(trail)):
            dx = trail[i][0] - trail[i - 1][0]
            dy = trail[i][1] - trail[i - 1][1]
            dt = trail[i][2] - trail[i - 1][2]
            if dt <= 0:
                continue
            speed = math.hypot(dx, dy) / dt
            if speed < speed_threshold:
                dwell_points.append(trail[i])

        return dwell_points

    def _cluster_dwell_points(
        self,
        points: list[tuple[float, float, float]],
        radius: float,
        min_dwell_s: float,
    ) -> list[FrequentZone]:
        """Greedy clustering of dwell points into zones."""
        remaining = list(points)
        zones: list[FrequentZone] = []

        while remaining:
            # Find point with most neighbors
            best_idx = 0
            best_count = 0

            for i, (px, py, _) in enumerate(remaining):
                count = sum(
                    1 for (qx, qy, _) in remaining
                    if math.hypot(px - qx, py - qy) <= radius
                )
                if count > best_count:
                    best_count = count
                    best_idx = i

            if best_count < 2:
                break

            cx, cy, _ = remaining[best_idx]

            # Absorb all points within radius
            cluster = []
            new_remaining = []
            for pt in remaining:
                if math.hypot(pt[0] - cx, pt[1] - cy) <= radius:
                    cluster.append(pt)
                else:
                    new_remaining.append(pt)
            remaining = new_remaining

            if not cluster:
                break

            # Compute zone center and dwell time
            mean_x = sum(p[0] for p in cluster) / len(cluster)
            mean_y = sum(p[1] for p in cluster) / len(cluster)
            max_r = max(
                math.hypot(p[0] - mean_x, p[1] - mean_y) for p in cluster
            ) if cluster else 0.0

            # Estimate dwell time from the cluster's time span
            times = sorted(p[2] for p in cluster)
            dwell_time = times[-1] - times[0] if len(times) > 1 else 0.0

            if dwell_time >= min_dwell_s:
                zones.append(FrequentZone(
                    center_x=mean_x,
                    center_y=mean_y,
                    radius=max(max_r, 1.0),
                    visit_count=len(cluster),
                    total_dwell_s=dwell_time,
                ))

        # Sort by dwell time descending
        zones.sort(key=lambda z: z.total_dwell_s, reverse=True)
        return zones[:MAX_ZONES]

    # ------------------------------------------------------------------
    # Deviation detection
    # ------------------------------------------------------------------

    def detect_deviation(
        self,
        target_id: str,
        position: tuple[float, float],
        timestamp: float | None = None,
    ) -> DeviationResult:
        """Check if a target's current position deviates from its learned patterns.

        Checks against:
          1. Learned route (is the position far from the expected path?)
          2. Learned zones (is the target in an unexpected area?)
          3. Learned schedule (is this an unusual time?)

        Args:
            target_id: The target to check.
            position: Current (x, y) position.
            timestamp: Current epoch timestamp (defaults to time.time()).

        Returns:
            DeviationResult with severity and details.
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            route = self._routes.get(target_id)
            schedule = self._schedules.get(target_id)
            zones = self._zones.get(target_id, [])

        # Check route deviation
        route_result = self._check_route_deviation(route, position)

        # Check schedule deviation
        schedule_result = self._check_schedule_deviation(schedule, timestamp)

        # Check zone deviation
        zone_result = self._check_zone_deviation(zones, position)

        # Combine results: return the most severe deviation
        results = [r for r in [route_result, schedule_result, zone_result] if r.is_deviation]

        if not results:
            return DeviationResult(is_deviation=False)

        # Return the one with highest severity
        results.sort(key=lambda r: r.severity, reverse=True)
        best = results[0]

        # Enrich with all deviation types found
        all_types = [r.deviation_type for r in results]
        best.details["all_deviation_types"] = all_types
        best.details["deviation_count"] = len(results)

        return best

    def _check_route_deviation(
        self,
        route: LearnedRoute | None,
        position: tuple[float, float],
    ) -> DeviationResult:
        """Check if position deviates from learned route."""
        if route is None or not route.waypoints:
            return DeviationResult(is_deviation=False)

        px, py = position

        # Find nearest waypoint
        min_dist = float("inf")
        nearest_idx = 0

        for i, wp in enumerate(route.waypoints):
            dist = math.hypot(px - wp.x, py - wp.y)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        wp = route.waypoints[nearest_idx]

        # Compute sigma: how many standard deviations away
        combined_std = math.hypot(max(wp.std_x, 0.1), max(wp.std_y, 0.1))
        sigma = min_dist / combined_std if combined_std > 0 else 0.0

        is_deviation = sigma > self._route_threshold

        severity = 0.0
        if is_deviation:
            # Severity scales from 0 at threshold to 1 at 3x threshold
            severity = min(1.0, (sigma - self._route_threshold) / (self._route_threshold * 2))

        return DeviationResult(
            is_deviation=is_deviation,
            deviation_type="route",
            severity=severity,
            distance_from_expected=min_dist,
            nearest_waypoint_index=nearest_idx,
            sigma=sigma,
            details={
                "waypoint_x": wp.x,
                "waypoint_y": wp.y,
                "waypoint_std": combined_std,
            },
        )

    def _check_schedule_deviation(
        self,
        schedule: LearnedSchedule | None,
        timestamp: float,
    ) -> DeviationResult:
        """Check if the current time deviates from learned schedule."""
        if schedule is None or schedule.total_observations < MIN_SCHEDULE_OBSERVATIONS:
            return DeviationResult(is_deviation=False)

        obs = self._timestamp_to_observation(timestamp)
        hour = obs.hour
        hour_bin = int(hour) % SCHEDULE_BINS

        # Check if this hour has any historical activity
        total_activity = sum(schedule.hourly_histogram)
        if total_activity == 0:
            return DeviationResult(is_deviation=False)

        hour_activity = schedule.hourly_histogram[hour_bin]
        hour_fraction = hour_activity / total_activity

        # If this hour has very low historical activity, it's unusual
        expected_fraction = 1.0 / SCHEDULE_BINS  # uniform expectation
        is_unusual = hour_fraction < expected_fraction * 0.25 and total_activity >= MIN_SCHEDULE_OBSERVATIONS

        if not is_unusual:
            return DeviationResult(is_deviation=False)

        # Severity based on how unusual this time is
        severity = min(1.0, 1.0 - (hour_fraction / expected_fraction)) if expected_fraction > 0 else 0.0

        return DeviationResult(
            is_deviation=True,
            deviation_type="schedule",
            severity=severity,
            details={
                "current_hour": round(hour, 2),
                "hour_bin": hour_bin,
                "hour_activity_fraction": round(hour_fraction, 4),
                "expected_fraction": round(expected_fraction, 4),
            },
        )

    def _check_zone_deviation(
        self,
        zones: list[FrequentZone],
        position: tuple[float, float],
    ) -> DeviationResult:
        """Check if position is outside all known frequent zones."""
        if not zones:
            return DeviationResult(is_deviation=False)

        px, py = position

        # Check if position is within any known zone
        min_dist_from_zone = float("inf")
        nearest_zone_idx = 0

        for i, zone in enumerate(zones):
            dist = math.hypot(px - zone.center_x, py - zone.center_y)
            dist_from_edge = dist - zone.radius
            if dist_from_edge < min_dist_from_zone:
                min_dist_from_zone = dist_from_edge
                nearest_zone_idx = i

        if min_dist_from_zone <= 0:
            # Inside a known zone
            return DeviationResult(is_deviation=False)

        # Outside all zones — compute severity based on distance
        # Severity scales from 0 at zone edge to 1 at 5x the zone radius
        nearest_zone = zones[nearest_zone_idx]
        ref_radius = max(nearest_zone.radius, 1.0)
        severity = min(1.0, min_dist_from_zone / (ref_radius * 5))

        return DeviationResult(
            is_deviation=True,
            deviation_type="zone",
            severity=severity,
            distance_from_expected=min_dist_from_zone,
            details={
                "nearest_zone_center": (nearest_zone.center_x, nearest_zone.center_y),
                "nearest_zone_radius": nearest_zone.radius,
                "distance_from_zone_edge": round(min_dist_from_zone, 2),
            },
        )

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def get_profile(self, target_id: str) -> BehavioralProfile:
        """Get the complete behavioral profile for a target.

        Returns:
            BehavioralProfile with route, schedule, zones, and regularity score.
        """
        with self._lock:
            route = self._routes.get(target_id)
            schedule = self._schedules.get(target_id)
            zones = self._zones.get(target_id, [])

        # Compute regularity score from available data
        regularity = self._compute_regularity(route, schedule, zones)

        # Compute observation span
        total_obs = 0
        first_seen = 0.0
        last_seen = 0.0

        if route:
            total_obs += route.total_observations
            if route.last_updated > last_seen:
                last_seen = route.last_updated
        if schedule:
            total_obs += schedule.total_observations
            if schedule.last_updated > last_seen:
                last_seen = schedule.last_updated

        # Get first/last from history if available
        if self._history is not None:
            trail = self._history.get_trail(target_id, max_points=1)
            if trail:
                first_seen = trail[0][2]
            trail_last = self._history.get_trail(target_id, max_points=1000)
            if trail_last:
                last_seen = max(last_seen, trail_last[-1][2])

        return BehavioralProfile(
            target_id=target_id,
            route=route,
            schedule=schedule,
            zones=list(zones),
            regularity_score=regularity,
            total_observations=total_obs,
            first_seen=first_seen,
            last_seen=last_seen,
        )

    def _compute_regularity(
        self,
        route: LearnedRoute | None,
        schedule: LearnedSchedule | None,
        zones: list[FrequentZone],
    ) -> float:
        """Compute a regularity score from 0.0 (unpredictable) to 1.0 (clockwork).

        Combines:
          - Route consistency (low waypoint std = more regular)
          - Schedule consistency (concentrated histogram = more regular)
          - Zone concentration (fewer zones, more dwell = more regular)
        """
        scores: list[float] = []

        # Route regularity: lower average std = more regular
        if route and route.waypoints:
            avg_std = sum(
                math.hypot(w.std_x, w.std_y) for w in route.waypoints
            ) / len(route.waypoints)
            # Map: std of 0 -> 1.0, std of 50+ -> 0.0
            route_score = max(0.0, 1.0 - avg_std / 50.0)
            scores.append(route_score)

        # Schedule regularity: concentrated histogram = regular
        if schedule and schedule.total_observations >= MIN_SCHEDULE_OBSERVATIONS:
            total = sum(schedule.hourly_histogram)
            if total > 0:
                # Compute entropy of hourly histogram
                entropy = 0.0
                for count in schedule.hourly_histogram:
                    if count > 0:
                        p = count / total
                        entropy -= p * math.log(p)
                # Max entropy for 24 bins = log(24) ~ 3.178
                max_entropy = math.log(SCHEDULE_BINS) if SCHEDULE_BINS > 0 else 1.0
                # Low entropy = concentrated = regular
                schedule_score = max(0.0, 1.0 - entropy / max_entropy)
                scores.append(schedule_score)

        # Zone regularity: fewer dominant zones = more regular
        if zones:
            # If most dwell is in 1-2 zones, that's regular
            total_dwell = sum(z.total_dwell_s for z in zones)
            if total_dwell > 0 and zones:
                top_zone_fraction = zones[0].total_dwell_s / total_dwell
                zone_score = min(1.0, top_zone_fraction)
                scores.append(zone_score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def learn_all(
        self,
        target_ids: list[str],
    ) -> dict[str, BehavioralProfile]:
        """Learn route, schedule, and zones for all given targets.

        Returns:
            Dict of target_id -> BehavioralProfile.
        """
        profiles = {}
        for tid in target_ids:
            self.learn_route(tid)
            self.learn_schedule(tid)
            self.learn_zones(tid)
            profiles[tid] = self.get_profile(tid)
        return profiles

    def get_all_profiles(self) -> dict[str, BehavioralProfile]:
        """Return profiles for all targets that have any learned data."""
        with self._lock:
            all_ids = set(self._routes.keys()) | set(self._schedules.keys()) | set(self._zones.keys())

        return {tid: self.get_profile(tid) for tid in all_ids}

    def clear(self, target_id: str | None = None) -> None:
        """Clear learned data for a target, or all targets if None."""
        with self._lock:
            if target_id is None:
                self._routes.clear()
                self._schedules.clear()
                self._zones.clear()
                self._observations.clear()
            else:
                self._routes.pop(target_id, None)
                self._schedules.pop(target_id, None)
                self._zones.pop(target_id, None)
                self._observations.pop(target_id, None)

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics for the learner."""
        with self._lock:
            return {
                "targets_with_routes": len(self._routes),
                "targets_with_schedules": len(self._schedules),
                "targets_with_zones": len(self._zones),
                "total_targets": len(
                    set(self._routes.keys())
                    | set(self._schedules.keys())
                    | set(self._zones.keys())
                ),
            }

    def export(self) -> dict[str, Any]:
        """Export full learner state as a JSON-serializable dict."""
        with self._lock:
            return {
                "routes": {
                    tid: route.to_dict()
                    for tid, route in self._routes.items()
                },
                "schedules": {
                    tid: sched.to_dict()
                    for tid, sched in self._schedules.items()
                },
                "zones": {
                    tid: [z.to_dict() for z in zones]
                    for tid, zones in self._zones.items()
                },
                "stats": self.get_stats(),
            }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean and standard deviation of a list of floats."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance) if variance > 0 else 0.0

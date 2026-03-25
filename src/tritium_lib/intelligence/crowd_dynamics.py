# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""CrowdDynamicsAnalyzer — analyze crowd formation, movement, and dispersal.

Detects groups of co-located targets that form crowds, tracks crowd growth
and shrinkage over time, identifies dispersal events, analyzes crowd flow
direction and speed, and estimates crowd density per area.

Integrates with:
  - :class:`~tritium_lib.tracking.target_tracker.TargetTracker` for live target data
  - :class:`~tritium_lib.tracking.target_history.TargetHistory` for position trails

Usage::

    from tritium_lib.intelligence.crowd_dynamics import (
        CrowdDynamicsAnalyzer,
        CrowdCluster,
    )

    analyzer = CrowdDynamicsAnalyzer()
    targets = [
        {"target_id": "t1", "position": (10.0, 10.0)},
        {"target_id": "t2", "position": (11.0, 10.5)},
        ...
    ]
    clusters = analyzer.detect_clusters(targets)
    events = analyzer.update(targets, timestamp=1000.0)
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Clustering
DEFAULT_CLUSTER_RADIUS_M = 15.0  # meters — max distance between members
MIN_CROWD_SIZE = 3               # minimum targets to constitute a crowd
MAX_CLUSTER_DIAMETER_M = 100.0   # max spread of a single crowd cluster

# Formation / dispersal detection
FORMATION_GROWTH_RATIO = 1.5    # cluster grew by 50% -> formation event
DISPERSAL_SHRINK_RATIO = 0.5    # cluster shrank to 50% -> dispersal event
CLUSTER_TIMEOUT_S = 120.0       # seconds without update before cluster expires

# Density
DEFAULT_DENSITY_CELL_SIZE_M = 10.0  # meters per density grid cell


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CrowdState(str, Enum):
    """Current lifecycle state of a crowd cluster."""
    FORMING = "forming"
    STABLE = "stable"
    GROWING = "growing"
    SHRINKING = "shrinking"
    DISPERSING = "dispersing"
    DISPERSED = "dispersed"


class CrowdEventType(str, Enum):
    """Types of crowd dynamics events."""
    FORMATION = "formation"       # a new crowd formed
    GROWTH = "growth"             # an existing crowd gained members
    SHRINKAGE = "shrinkage"       # an existing crowd lost members
    DISPERSAL = "dispersal"       # a crowd broke apart
    MERGE = "merge"               # two crowds merged into one
    SPLIT = "split"               # one crowd split into two


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CrowdCluster:
    """A detected group of co-located targets forming a crowd."""

    cluster_id: str
    member_ids: list[str]
    center: tuple[float, float]
    radius: float                 # meters — bounding radius from center
    state: CrowdState = CrowdState.FORMING
    first_seen: float = 0.0
    last_seen: float = 0.0
    peak_size: int = 0
    avg_speed: float = 0.0       # mean speed of members (m/s)
    flow_heading: float = 0.0    # dominant movement direction (degrees)
    flow_speed: float = 0.0      # speed of the crowd center movement (m/s)
    density: float = 0.0         # targets per square meter
    _prev_center: tuple[float, float] = (0.0, 0.0)
    _prev_timestamp: float = 0.0

    @property
    def size(self) -> int:
        return len(self.member_ids)

    @property
    def duration_s(self) -> float:
        if self.last_seen <= self.first_seen:
            return 0.0
        return self.last_seen - self.first_seen

    @property
    def area_m2(self) -> float:
        """Approximate area of the cluster circle."""
        return math.pi * self.radius * self.radius if self.radius > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "member_ids": list(self.member_ids),
            "center": {"x": round(self.center[0], 2), "y": round(self.center[1], 2)},
            "radius": round(self.radius, 2),
            "size": self.size,
            "state": self.state.value,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_s": round(self.duration_s, 1),
            "peak_size": self.peak_size,
            "avg_speed": round(self.avg_speed, 2),
            "flow_heading": round(self.flow_heading, 1),
            "flow_speed": round(self.flow_speed, 2),
            "density": round(self.density, 4),
            "area_m2": round(self.area_m2, 1),
        }


@dataclass
class CrowdEvent:
    """A crowd dynamics event — formation, dispersal, growth, etc."""

    event_id: str
    event_type: CrowdEventType
    cluster_id: str
    timestamp: float
    member_count: int = 0
    previous_count: int = 0
    center: tuple[float, float] = (0.0, 0.0)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "cluster_id": self.cluster_id,
            "timestamp": self.timestamp,
            "member_count": self.member_count,
            "previous_count": self.previous_count,
            "center": {"x": round(self.center[0], 2), "y": round(self.center[1], 2)},
            "details": self.details,
        }


@dataclass
class DensityCell:
    """A single cell in the crowd density grid."""

    row: int
    col: int
    x: float
    y: float
    count: int = 0
    density: float = 0.0  # targets per m^2

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "count": self.count,
            "density": round(self.density, 6),
        }


@dataclass
class FlowVector:
    """Crowd flow at a location — direction and speed."""

    x: float
    y: float
    heading_deg: float
    speed_mps: float
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "heading_deg": round(self.heading_deg, 1),
            "speed_mps": round(self.speed_mps, 2),
            "sample_count": self.sample_count,
        }


# ---------------------------------------------------------------------------
# FormationDetector
# ---------------------------------------------------------------------------

class FormationDetector:
    """Detect when individual targets coalesce into a crowd.

    Compares the current set of clusters against the previous set to
    identify newly formed clusters (clusters that did not exist before
    and have at least ``min_size`` members).
    """

    def __init__(self, min_size: int = MIN_CROWD_SIZE) -> None:
        self._min_size = min_size
        self._known_ids: set[str] = set()

    def detect(
        self,
        clusters: list[CrowdCluster],
        timestamp: float,
    ) -> list[CrowdEvent]:
        """Return formation events for newly appearing clusters."""
        events: list[CrowdEvent] = []
        for c in clusters:
            if c.cluster_id not in self._known_ids and c.size >= self._min_size:
                events.append(CrowdEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    event_type=CrowdEventType.FORMATION,
                    cluster_id=c.cluster_id,
                    timestamp=timestamp,
                    member_count=c.size,
                    previous_count=0,
                    center=c.center,
                    details={"radius": c.radius},
                ))
        self._known_ids = {c.cluster_id for c in clusters}
        return events

    def reset(self) -> None:
        self._known_ids.clear()


# ---------------------------------------------------------------------------
# DispersalDetector
# ---------------------------------------------------------------------------

class DispersalDetector:
    """Detect when a crowd breaks up or significantly shrinks.

    Tracks cluster sizes across updates and fires dispersal events when
    a cluster disappears or shrinks below ``DISPERSAL_SHRINK_RATIO`` of
    its previous size.
    """

    def __init__(self, shrink_ratio: float = DISPERSAL_SHRINK_RATIO) -> None:
        self._shrink_ratio = shrink_ratio
        self._prev_sizes: dict[str, int] = {}

    def detect(
        self,
        clusters: list[CrowdCluster],
        timestamp: float,
    ) -> list[CrowdEvent]:
        """Return dispersal events for clusters that shrank or disappeared."""
        events: list[CrowdEvent] = []
        current_ids = {c.cluster_id for c in clusters}
        current_sizes = {c.cluster_id: c.size for c in clusters}
        current_centers = {c.cluster_id: c.center for c in clusters}

        # Clusters that disappeared entirely
        for cid, prev_size in self._prev_sizes.items():
            if cid not in current_ids:
                events.append(CrowdEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    event_type=CrowdEventType.DISPERSAL,
                    cluster_id=cid,
                    timestamp=timestamp,
                    member_count=0,
                    previous_count=prev_size,
                    details={"reason": "disappeared"},
                ))

        # Clusters that shrank significantly
        for c in clusters:
            prev = self._prev_sizes.get(c.cluster_id)
            if prev is not None and prev > 0:
                ratio = c.size / prev
                if ratio <= self._shrink_ratio:
                    events.append(CrowdEvent(
                        event_id=f"evt_{uuid.uuid4().hex[:8]}",
                        event_type=CrowdEventType.DISPERSAL,
                        cluster_id=c.cluster_id,
                        timestamp=timestamp,
                        member_count=c.size,
                        previous_count=prev,
                        center=c.center,
                        details={"ratio": round(ratio, 3)},
                    ))

        self._prev_sizes = dict(current_sizes)
        return events

    def reset(self) -> None:
        self._prev_sizes.clear()


# ---------------------------------------------------------------------------
# FlowAnalyzer
# ---------------------------------------------------------------------------

class FlowAnalyzer:
    """Analyze crowd flow direction and speed.

    Computes per-cluster and grid-based flow vectors from target position
    history.  Call ``compute_cluster_flow`` for a single cluster's bulk
    movement, or ``compute_grid_flow`` for a spatial flow field.
    """

    def __init__(self, cell_size: float = DEFAULT_DENSITY_CELL_SIZE_M) -> None:
        self._cell_size = cell_size
        self._prev_positions: dict[str, tuple[float, float, float]] = {}  # id -> (x, y, t)

    def record_positions(
        self,
        targets: list[dict[str, Any]],
        timestamp: float,
    ) -> None:
        """Record current target positions for velocity computation."""
        for t in targets:
            tid = t.get("target_id", "")
            pos = t.get("position")
            if tid and pos is not None:
                x, y = float(pos[0]), float(pos[1])
                self._prev_positions[tid] = (x, y, timestamp)

    def compute_cluster_flow(
        self,
        cluster: CrowdCluster,
        targets: list[dict[str, Any]],
        timestamp: float,
    ) -> tuple[float, float]:
        """Compute bulk flow heading (degrees) and speed (m/s) for a cluster.

        Returns (heading_deg, speed_mps).
        """
        member_set = set(cluster.member_ids)
        vx_sum = 0.0
        vy_sum = 0.0
        count = 0

        for t in targets:
            tid = t.get("target_id", "")
            if tid not in member_set:
                continue
            pos = t.get("position")
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            prev = self._prev_positions.get(tid)
            if prev is None:
                continue
            px, py, pt = prev
            dt = timestamp - pt
            if dt <= 0:
                continue
            vx_sum += (x - px) / dt
            vy_sum += (y - py) / dt
            count += 1

        if count == 0:
            return 0.0, 0.0

        avg_vx = vx_sum / count
        avg_vy = vy_sum / count
        speed = math.hypot(avg_vx, avg_vy)
        heading = math.degrees(math.atan2(avg_vx, avg_vy)) % 360.0
        return heading, speed

    def compute_grid_flow(
        self,
        targets: list[dict[str, Any]],
        area: tuple[float, float, float, float],
        timestamp: float,
        resolution: int = 10,
    ) -> list[FlowVector]:
        """Compute a grid of flow vectors over an area.

        Parameters
        ----------
        targets : list[dict]
            Targets with ``target_id`` and ``position`` keys.
        area : tuple
            ``(min_x, min_y, max_x, max_y)`` bounding box.
        timestamp : float
            Current timestamp.
        resolution : int
            Grid cells per axis.

        Returns
        -------
        list[FlowVector]
            Flow vectors for cells that have moving targets.
        """
        min_x, min_y, max_x, max_y = area
        range_x = max_x - min_x
        range_y = max_y - min_y
        if range_x <= 0 or range_y <= 0 or resolution < 1:
            return []

        cell_w = range_x / resolution
        cell_h = range_y / resolution

        # Accumulate velocities per cell
        cell_vx: dict[tuple[int, int], list[float]] = defaultdict(list)
        cell_vy: dict[tuple[int, int], list[float]] = defaultdict(list)

        for t in targets:
            tid = t.get("target_id", "")
            pos = t.get("position")
            if not tid or pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                continue

            prev = self._prev_positions.get(tid)
            if prev is None:
                continue
            px, py, pt = prev
            dt = timestamp - pt
            if dt <= 0:
                continue

            vx = (x - px) / dt
            vy = (y - py) / dt

            col = min(int((x - min_x) / cell_w), resolution - 1)
            row = min(int((y - min_y) / cell_h), resolution - 1)
            cell_vx[(row, col)].append(vx)
            cell_vy[(row, col)].append(vy)

        vectors: list[FlowVector] = []
        for (row, col), vxs in cell_vx.items():
            vys = cell_vy[(row, col)]
            n = len(vxs)
            avg_vx = sum(vxs) / n
            avg_vy = sum(vys) / n
            speed = math.hypot(avg_vx, avg_vy)
            heading = math.degrees(math.atan2(avg_vx, avg_vy)) % 360.0
            cx = min_x + (col + 0.5) * cell_w
            cy = min_y + (row + 0.5) * cell_h
            vectors.append(FlowVector(
                x=cx, y=cy,
                heading_deg=heading,
                speed_mps=speed,
                sample_count=n,
            ))

        return vectors

    def reset(self) -> None:
        self._prev_positions.clear()


# ---------------------------------------------------------------------------
# DensityEstimator
# ---------------------------------------------------------------------------

class DensityEstimator:
    """Estimate crowd density per area using a grid.

    Divides a bounding area into cells and counts how many targets fall
    into each cell, then converts to targets-per-square-meter.
    """

    def __init__(self, cell_size: float = DEFAULT_DENSITY_CELL_SIZE_M) -> None:
        self._cell_size = cell_size

    def estimate(
        self,
        targets: list[dict[str, Any]],
        area: tuple[float, float, float, float],
    ) -> list[DensityCell]:
        """Estimate density over an area.

        Parameters
        ----------
        targets : list[dict]
            Targets with ``position`` key ``(x, y)``.
        area : tuple
            ``(min_x, min_y, max_x, max_y)`` bounding rectangle.

        Returns
        -------
        list[DensityCell]
            Cells with nonzero density, sorted by density descending.
        """
        min_x, min_y, max_x, max_y = area
        range_x = max_x - min_x
        range_y = max_y - min_y
        if range_x <= 0 or range_y <= 0:
            return []

        cols = max(1, int(math.ceil(range_x / self._cell_size)))
        rows = max(1, int(math.ceil(range_y / self._cell_size)))
        cell_w = range_x / cols
        cell_h = range_y / rows
        cell_area = cell_w * cell_h

        grid: dict[tuple[int, int], int] = defaultdict(int)

        for t in targets:
            pos = t.get("position")
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                continue

            col = min(int((x - min_x) / cell_w), cols - 1)
            row = min(int((y - min_y) / cell_h), rows - 1)
            grid[(row, col)] += 1

        cells: list[DensityCell] = []
        for (row, col), count in grid.items():
            if count <= 0:
                continue
            cx = min_x + (col + 0.5) * cell_w
            cy = min_y + (row + 0.5) * cell_h
            density = count / cell_area if cell_area > 0 else 0.0
            cells.append(DensityCell(
                row=row, col=col,
                x=cx, y=cy,
                count=count,
                density=density,
            ))

        cells.sort(key=lambda c: c.density, reverse=True)
        return cells

    def peak_density(
        self,
        targets: list[dict[str, Any]],
        area: tuple[float, float, float, float],
    ) -> float:
        """Return the peak density (targets/m^2) in the area."""
        cells = self.estimate(targets, area)
        if not cells:
            return 0.0
        return cells[0].density

    def average_density(
        self,
        targets: list[dict[str, Any]],
        area: tuple[float, float, float, float],
    ) -> float:
        """Return the average density across occupied cells."""
        cells = self.estimate(targets, area)
        if not cells:
            return 0.0
        return sum(c.density for c in cells) / len(cells)


# ---------------------------------------------------------------------------
# CrowdDynamicsAnalyzer — main orchestrator
# ---------------------------------------------------------------------------

class CrowdDynamicsAnalyzer:
    """Detect crowds forming, growing, and dispersing among tracked targets.

    Orchestrates clustering, formation detection, dispersal detection,
    flow analysis, and density estimation into a single ``update()`` call.

    Parameters
    ----------
    cluster_radius : float
        Maximum distance (meters) between cluster members.
    min_crowd_size : int
        Minimum number of targets to form a crowd.
    growth_ratio : float
        Fractional increase threshold to emit a growth event.
    shrink_ratio : float
        Fractional decrease threshold to emit a dispersal event.
    density_cell_size : float
        Cell size (meters) for density estimation.
    """

    def __init__(
        self,
        cluster_radius: float = DEFAULT_CLUSTER_RADIUS_M,
        min_crowd_size: int = MIN_CROWD_SIZE,
        growth_ratio: float = FORMATION_GROWTH_RATIO,
        shrink_ratio: float = DISPERSAL_SHRINK_RATIO,
        density_cell_size: float = DEFAULT_DENSITY_CELL_SIZE_M,
    ) -> None:
        self._cluster_radius = cluster_radius
        self._min_crowd_size = min_crowd_size
        self._growth_ratio = growth_ratio

        # Sub-analyzers
        self._formation_detector = FormationDetector(min_size=min_crowd_size)
        self._dispersal_detector = DispersalDetector(shrink_ratio=shrink_ratio)
        self._flow_analyzer = FlowAnalyzer(cell_size=density_cell_size)
        self._density_estimator = DensityEstimator(cell_size=density_cell_size)

        # State
        self._clusters: dict[str, CrowdCluster] = {}
        self._prev_cluster_sizes: dict[str, int] = {}
        self._events: list[CrowdEvent] = []
        self._max_events = 1000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        targets: list[dict[str, Any]],
        timestamp: float,
    ) -> list[CrowdEvent]:
        """Run a full crowd dynamics analysis tick.

        Parameters
        ----------
        targets : list[dict]
            Current target positions.  Each dict must have at minimum
            ``target_id`` (str) and ``position`` (tuple of two floats).
        timestamp : float
            Current unix timestamp.

        Returns
        -------
        list[CrowdEvent]
            Events detected this tick (formations, dispersals, growth, etc.).
        """
        # 1. Cluster targets
        new_clusters = self.detect_clusters(targets, timestamp)

        # 2. Match new clusters to existing ones
        matched = self._match_clusters(new_clusters)

        # 3. Detect formations (new clusters)
        formation_events = self._formation_detector.detect(
            list(matched.values()), timestamp
        )

        # 4. Detect dispersals (disappeared / shrunk clusters)
        dispersal_events = self._dispersal_detector.detect(
            list(matched.values()), timestamp
        )

        # 5. Detect growth events
        growth_events = self._detect_growth(matched, timestamp)

        # 6. Compute flow for each cluster
        for c in matched.values():
            heading, speed = self._flow_analyzer.compute_cluster_flow(
                c, targets, timestamp
            )
            c.flow_heading = heading
            c.flow_speed = speed

        # 7. Record positions for next tick's flow computation
        self._flow_analyzer.record_positions(targets, timestamp)

        # 8. Compute per-cluster density
        for c in matched.values():
            if c.area_m2 > 0:
                c.density = c.size / c.area_m2
            else:
                c.density = 0.0

        # 9. Update state
        self._clusters = matched
        self._prev_cluster_sizes = {
            cid: c.size for cid, c in matched.items()
        }

        # 10. Collect and store events
        tick_events = formation_events + dispersal_events + growth_events
        self._events.extend(tick_events)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        return tick_events

    def detect_clusters(
        self,
        targets: list[dict[str, Any]],
        timestamp: float = 0.0,
    ) -> list[CrowdCluster]:
        """Cluster targets by proximity using single-linkage clustering.

        Parameters
        ----------
        targets : list[dict]
            Each dict must have ``target_id`` and ``position``.
        timestamp : float
            Current timestamp for bookkeeping.

        Returns
        -------
        list[CrowdCluster]
            Detected clusters with at least ``min_crowd_size`` members.
        """
        # Build adjacency using distance threshold
        positions: dict[str, tuple[float, float]] = {}
        for t in targets:
            tid = t.get("target_id", "")
            pos = t.get("position")
            if tid and pos is not None:
                positions[tid] = (float(pos[0]), float(pos[1]))

        if len(positions) < self._min_crowd_size:
            return []

        target_ids = list(positions.keys())
        adj: dict[str, set[str]] = {tid: set() for tid in target_ids}

        # O(n^2) pairwise distance — fine for hundreds of targets
        for i in range(len(target_ids)):
            for j in range(i + 1, len(target_ids)):
                a, b = target_ids[i], target_ids[j]
                ax, ay = positions[a]
                bx, by = positions[b]
                dist = math.hypot(ax - bx, ay - by)
                if dist <= self._cluster_radius:
                    adj[a].add(b)
                    adj[b].add(a)

        # BFS connected components
        visited: set[str] = set()
        clusters: list[CrowdCluster] = []

        for start in target_ids:
            if start in visited:
                continue
            component: list[str] = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) < self._min_crowd_size:
                continue

            # Check diameter constraint
            max_dist = 0.0
            for i in range(len(component)):
                for j in range(i + 1, len(component)):
                    ax, ay = positions[component[i]]
                    bx, by = positions[component[j]]
                    d = math.hypot(ax - bx, ay - by)
                    if d > max_dist:
                        max_dist = d

            if max_dist > MAX_CLUSTER_DIAMETER_M:
                continue

            # Compute centroid and bounding radius
            xs = [positions[tid][0] for tid in component]
            ys = [positions[tid][1] for tid in component]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            radius = max(
                math.hypot(positions[tid][0] - cx, positions[tid][1] - cy)
                for tid in component
            )

            cluster_id = f"crowd_{uuid.uuid4().hex[:8]}"
            clusters.append(CrowdCluster(
                cluster_id=cluster_id,
                member_ids=sorted(component),
                center=(cx, cy),
                radius=max(radius, 0.1),  # avoid zero radius
                state=CrowdState.FORMING,
                first_seen=timestamp,
                last_seen=timestamp,
                peak_size=len(component),
            ))

        return clusters

    def get_active_clusters(self) -> list[CrowdCluster]:
        """Return all currently tracked crowd clusters."""
        return list(self._clusters.values())

    def get_cluster(self, cluster_id: str) -> CrowdCluster | None:
        """Return a specific cluster by ID."""
        return self._clusters.get(cluster_id)

    def get_events(self, limit: int = 100) -> list[CrowdEvent]:
        """Return recent crowd dynamics events."""
        return list(reversed(self._events[-limit:]))

    def get_density_map(
        self,
        targets: list[dict[str, Any]],
        area: tuple[float, float, float, float],
    ) -> list[DensityCell]:
        """Compute a density map for the given targets and area."""
        return self._density_estimator.estimate(targets, area)

    def get_flow_field(
        self,
        targets: list[dict[str, Any]],
        area: tuple[float, float, float, float],
        timestamp: float,
        resolution: int = 10,
    ) -> list[FlowVector]:
        """Compute a grid-based flow field for the area."""
        return self._flow_analyzer.compute_grid_flow(
            targets, area, timestamp, resolution
        )

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        clusters = list(self._clusters.values())
        total_members = sum(c.size for c in clusters)
        return {
            "active_clusters": len(clusters),
            "total_crowd_members": total_members,
            "largest_cluster": max((c.size for c in clusters), default=0),
            "total_events": len(self._events),
            "cluster_ids": [c.cluster_id for c in clusters],
        }

    def clear(self) -> None:
        """Reset all state."""
        self._clusters.clear()
        self._prev_cluster_sizes.clear()
        self._events.clear()
        self._formation_detector.reset()
        self._dispersal_detector.reset()
        self._flow_analyzer.reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_clusters(
        self, new_clusters: list[CrowdCluster]
    ) -> dict[str, CrowdCluster]:
        """Match new clusters to existing ones by member overlap.

        If a new cluster shares >= 50% members with an existing cluster,
        it inherits the existing cluster's ID and history.
        """
        matched: dict[str, CrowdCluster] = {}
        used_existing: set[str] = set()

        for nc in new_clusters:
            new_set = set(nc.member_ids)
            best_id: str | None = None
            best_overlap = 0

            for eid, ec in self._clusters.items():
                if eid in used_existing:
                    continue
                overlap = len(new_set & set(ec.member_ids))
                threshold = max(len(new_set), len(ec.member_ids)) * 0.5
                if overlap >= threshold and overlap > best_overlap:
                    best_overlap = overlap
                    best_id = eid

            if best_id is not None:
                # Inherit existing cluster identity
                existing = self._clusters[best_id]
                nc.cluster_id = best_id
                nc.first_seen = existing.first_seen
                nc.peak_size = max(existing.peak_size, nc.size)
                nc._prev_center = existing.center
                nc._prev_timestamp = existing.last_seen

                # Determine state
                prev_size = self._prev_cluster_sizes.get(best_id, 0)
                if prev_size > 0:
                    ratio = nc.size / prev_size
                    if ratio >= self._growth_ratio:
                        nc.state = CrowdState.GROWING
                    elif ratio <= 0.7:
                        nc.state = CrowdState.SHRINKING
                    else:
                        nc.state = CrowdState.STABLE
                else:
                    nc.state = CrowdState.STABLE

                used_existing.add(best_id)
            else:
                nc.state = CrowdState.FORMING

            matched[nc.cluster_id] = nc

        return matched

    def _detect_growth(
        self,
        clusters: dict[str, CrowdCluster],
        timestamp: float,
    ) -> list[CrowdEvent]:
        """Detect significant growth in existing clusters."""
        events: list[CrowdEvent] = []
        for cid, c in clusters.items():
            prev = self._prev_cluster_sizes.get(cid)
            if prev is not None and prev > 0:
                ratio = c.size / prev
                if ratio >= self._growth_ratio:
                    events.append(CrowdEvent(
                        event_id=f"evt_{uuid.uuid4().hex[:8]}",
                        event_type=CrowdEventType.GROWTH,
                        cluster_id=cid,
                        timestamp=timestamp,
                        member_count=c.size,
                        previous_count=prev,
                        center=c.center,
                        details={"growth_ratio": round(ratio, 3)},
                    ))
        return events

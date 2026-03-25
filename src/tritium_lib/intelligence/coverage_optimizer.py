# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Sensor coverage optimization — optimal placement, gap analysis, redundancy.

Given an area and sensor constraints (type, range, FOV), computes where to
place sensors for maximum detection probability coverage.  Also identifies
gaps in existing deployments and regions of excessive redundancy.

Algorithms:
    - Grid-based detection-probability model (RSSI path-loss or simple range)
    - Greedy placement: iteratively place next sensor where coverage gap is largest
    - K-means clustering of uncovered cells to suggest placement regions
    - Per-cell redundancy counting to find over-instrumented areas

Integrates with:
    - :mod:`tritium_lib.models.sensor_config` — SensorType, SensorPlacement
    - :mod:`tritium_lib.geo` — haversine_distance for lat/lng mode
    - :mod:`tritium_lib.intelligence.position_estimator` — rssi_to_distance

Usage::

    from tritium_lib.intelligence.coverage_optimizer import (
        CoverageMap, SensorPlacement as OptSensorPlacement, optimize_placement,
        coverage_gaps, redundancy_analysis,
    )

    area = (0.0, 0.0, 200.0, 200.0)  # min_x, min_y, max_x, max_y in meters
    result = optimize_placement(area, sensor_count=4, sensor_types=["ble_radio"])
    gaps = coverage_gaps(existing_sensors, area)
    redundancy = redundancy_analysis(existing_sensors, area)
    heatmap = result.coverage_map.to_heatmap()
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — default sensor range profiles (meters)
# ---------------------------------------------------------------------------

SENSOR_RANGE_PROFILES: dict[str, dict[str, float]] = {
    "ble_radio": {
        "max_range_m": 30.0,
        "tx_power_dbm": -59.0,
        "path_loss_exp": 2.5,
    },
    "wifi_radio": {
        "max_range_m": 80.0,
        "tx_power_dbm": -30.0,
        "path_loss_exp": 2.8,
    },
    "camera": {
        "max_range_m": 50.0,
        "fov_degrees": 90.0,
    },
    "acoustic": {
        "max_range_m": 40.0,
        "tx_power_dbm": -50.0,
        "path_loss_exp": 3.0,
    },
    "mesh_radio": {
        "max_range_m": 200.0,
        "tx_power_dbm": -40.0,
        "path_loss_exp": 2.2,
    },
    "rf_monitor": {
        "max_range_m": 100.0,
        "tx_power_dbm": -45.0,
        "path_loss_exp": 2.5,
    },
    "radar": {
        "max_range_m": 150.0,
        "fov_degrees": 120.0,
    },
    "pir": {
        "max_range_m": 12.0,
        "fov_degrees": 110.0,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SensorSpec:
    """Specification for a sensor to be placed.

    Attributes:
        sensor_type: Type key matching SENSOR_RANGE_PROFILES or SensorType values.
        max_range_m: Maximum detection range in meters.
        fov_degrees: Field of view in degrees (360 = omnidirectional).
        heading_degrees: Direction the sensor faces (0 = north, clockwise).
            Only relevant for directional sensors (fov < 360).
        tx_power_dbm: Transmit power for RSSI-based probability modeling.
        path_loss_exp: Path-loss exponent for RSSI distance model.
    """

    sensor_type: str = "ble_radio"
    max_range_m: float = 30.0
    fov_degrees: float = 360.0
    heading_degrees: float = 0.0
    tx_power_dbm: float = -59.0
    path_loss_exp: float = 2.5

    @classmethod
    def from_type(cls, sensor_type: str) -> SensorSpec:
        """Create a SensorSpec from a known sensor type string."""
        profile = SENSOR_RANGE_PROFILES.get(sensor_type, {})
        return cls(
            sensor_type=sensor_type,
            max_range_m=profile.get("max_range_m", 50.0),
            fov_degrees=profile.get("fov_degrees", 360.0),
            tx_power_dbm=profile.get("tx_power_dbm", -50.0),
            path_loss_exp=profile.get("path_loss_exp", 2.5),
        )


@dataclass
class PlacedSensor:
    """A sensor with a concrete position.

    Attributes:
        sensor_id: Unique identifier.
        x: X position in local meters.
        y: Y position in local meters.
        spec: Sensor specification (type, range, FOV, etc.).
    """

    sensor_id: str = ""
    x: float = 0.0
    y: float = 0.0
    spec: SensorSpec = field(default_factory=SensorSpec)


@dataclass
class CoverageCell:
    """A single cell in the coverage grid.

    Attributes:
        row: Grid row index.
        col: Grid column index.
        center_x: World X coordinate of cell center.
        center_y: World Y coordinate of cell center.
        detection_prob: Combined detection probability (0..1).
        sensor_count: Number of sensors covering this cell.
        contributing_sensors: IDs of sensors that reach this cell.
    """

    row: int = 0
    col: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    detection_prob: float = 0.0
    sensor_count: int = 0
    contributing_sensors: list[str] = field(default_factory=list)


@dataclass
class CoverageMap:
    """Grid-based coverage map with per-cell detection probabilities.

    The grid covers the bounding area at a given resolution.  Each cell
    stores the combined detection probability from all sensors and how
    many sensors contribute.

    Attributes:
        area: ``(min_x, min_y, max_x, max_y)`` bounding rectangle.
        resolution: Number of cells along each axis.
        cells: 2D list ``[row][col]`` of :class:`CoverageCell`.
        overall_coverage: Fraction of cells with detection_prob > threshold.
        avg_detection_prob: Mean detection probability across all cells.
        min_detection_prob: Minimum detection probability.
        max_detection_prob: Maximum detection probability.
    """

    area: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0)
    resolution: int = 20
    cells: list[list[CoverageCell]] = field(default_factory=list)
    overall_coverage: float = 0.0
    avg_detection_prob: float = 0.0
    min_detection_prob: float = 0.0
    max_detection_prob: float = 0.0

    def to_heatmap(self) -> dict[str, Any]:
        """Export as heatmap data suitable for frontend visualization.

        Returns a dict with:
            - ``grid``: 2D float array of detection probabilities [row][col].
            - ``area``: bounding rectangle ``(min_x, min_y, max_x, max_y)``.
            - ``resolution``: grid size.
            - ``overall_coverage``: fraction of covered cells.
            - ``avg_detection_prob``: mean probability.
            - ``max_detection_prob``: peak probability.
            - ``sensor_count_grid``: 2D int array of sensor counts per cell.
        """
        prob_grid: list[list[float]] = []
        count_grid: list[list[int]] = []
        for row in self.cells:
            prob_row: list[float] = []
            count_row: list[int] = []
            for cell in row:
                prob_row.append(round(cell.detection_prob, 4))
                count_row.append(cell.sensor_count)
            prob_grid.append(prob_row)
            count_grid.append(count_row)

        return {
            "grid": prob_grid,
            "area": list(self.area),
            "resolution": self.resolution,
            "overall_coverage": round(self.overall_coverage, 4),
            "avg_detection_prob": round(self.avg_detection_prob, 4),
            "max_detection_prob": round(self.max_detection_prob, 4),
            "sensor_count_grid": count_grid,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "area": list(self.area),
            "resolution": self.resolution,
            "overall_coverage": round(self.overall_coverage, 4),
            "avg_detection_prob": round(self.avg_detection_prob, 4),
            "min_detection_prob": round(self.min_detection_prob, 4),
            "max_detection_prob": round(self.max_detection_prob, 4),
            "cell_count": self.resolution * self.resolution,
        }


@dataclass
class CoverageGap:
    """A region with insufficient sensor coverage.

    Attributes:
        center_x: X coordinate of the gap centroid.
        center_y: Y coordinate of the gap centroid.
        radius_m: Approximate radius of the gap area.
        avg_detection_prob: Mean detection probability in the gap cells.
        cell_count: Number of gap cells in this cluster.
        severity: 0..1 how severe the gap is (1 = zero coverage).
    """

    center_x: float = 0.0
    center_y: float = 0.0
    radius_m: float = 0.0
    avg_detection_prob: float = 0.0
    cell_count: int = 0
    severity: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "radius_m": round(self.radius_m, 2),
            "avg_detection_prob": round(self.avg_detection_prob, 4),
            "cell_count": self.cell_count,
            "severity": round(self.severity, 4),
        }


@dataclass
class RedundancyZone:
    """A region with excessive sensor overlap.

    Attributes:
        center_x: X coordinate of the zone centroid.
        center_y: Y coordinate of the zone centroid.
        radius_m: Approximate radius of the redundant area.
        avg_sensor_count: Mean number of sensors covering cells in this zone.
        max_sensor_count: Peak sensor overlap.
        cell_count: Number of cells in this redundancy zone.
    """

    center_x: float = 0.0
    center_y: float = 0.0
    radius_m: float = 0.0
    avg_sensor_count: float = 0.0
    max_sensor_count: int = 0
    cell_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "radius_m": round(self.radius_m, 2),
            "avg_sensor_count": round(self.avg_sensor_count, 2),
            "max_sensor_count": self.max_sensor_count,
            "cell_count": self.cell_count,
        }


@dataclass
class PlacementResult:
    """Result of an optimize_placement call.

    Attributes:
        sensors: List of optimally placed sensors.
        coverage_map: The resulting coverage map.
        total_coverage: Fraction of area with detection_prob > threshold.
        avg_detection_prob: Mean detection probability across all cells.
    """

    sensors: list[PlacedSensor] = field(default_factory=list)
    coverage_map: CoverageMap = field(default_factory=CoverageMap)
    total_coverage: float = 0.0
    avg_detection_prob: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_count": len(self.sensors),
            "sensors": [
                {
                    "sensor_id": s.sensor_id,
                    "x": round(s.x, 2),
                    "y": round(s.y, 2),
                    "sensor_type": s.spec.sensor_type,
                    "max_range_m": s.spec.max_range_m,
                }
                for s in self.sensors
            ],
            "total_coverage": round(self.total_coverage, 4),
            "avg_detection_prob": round(self.avg_detection_prob, 4),
            "coverage_map": self.coverage_map.to_dict(),
        }


# ---------------------------------------------------------------------------
# Detection probability model
# ---------------------------------------------------------------------------

def _detection_probability(
    distance_m: float,
    spec: SensorSpec,
) -> float:
    """Compute detection probability at a given distance from a sensor.

    Uses a log-distance path-loss model for RF sensors (BLE, WiFi, mesh, etc.)
    and a simple linear falloff for non-RF sensors (camera, PIR).

    Returns 0..1 probability.
    """
    if distance_m <= 0:
        return 1.0
    if distance_m > spec.max_range_m:
        return 0.0

    # RF sensors: RSSI-based probability using path-loss model
    if spec.sensor_type in ("ble_radio", "wifi_radio", "mesh_radio",
                            "acoustic", "rf_monitor"):
        # RSSI at distance using log-distance path-loss
        rssi = spec.tx_power_dbm - 10.0 * spec.path_loss_exp * math.log10(
            max(distance_m, 0.5)
        )
        # Probability: 1.0 at tx_power, 0.0 at sensitivity floor (-100 dBm)
        sensitivity_floor = -100.0
        if rssi >= spec.tx_power_dbm:
            return 1.0
        if rssi <= sensitivity_floor:
            return 0.0
        prob = (rssi - sensitivity_floor) / (spec.tx_power_dbm - sensitivity_floor)
        return max(0.0, min(1.0, prob))

    # Non-RF sensors: smooth falloff (inverse square feel)
    ratio = distance_m / spec.max_range_m
    # Cosine-based smooth falloff: 1.0 at center, 0.0 at max_range
    prob = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return max(0.0, min(1.0, prob))


def _is_in_fov(
    sensor_x: float,
    sensor_y: float,
    target_x: float,
    target_y: float,
    spec: SensorSpec,
) -> bool:
    """Check if a target position falls within the sensor's field of view."""
    if spec.fov_degrees >= 360.0:
        return True

    dx = target_x - sensor_x
    dy = target_y - sensor_y
    if dx == 0.0 and dy == 0.0:
        return True

    # Bearing from sensor to target (0 = north/+Y, clockwise)
    bearing = math.degrees(math.atan2(dx, dy)) % 360.0
    heading = spec.heading_degrees % 360.0

    # Angular difference
    diff = abs(bearing - heading)
    if diff > 180.0:
        diff = 360.0 - diff

    return diff <= spec.fov_degrees / 2.0


def _cell_detection_prob(
    cell_x: float,
    cell_y: float,
    sensor: PlacedSensor,
) -> float:
    """Detection probability at (cell_x, cell_y) from a single sensor."""
    dx = cell_x - sensor.x
    dy = cell_y - sensor.y
    distance = math.sqrt(dx * dx + dy * dy)

    if not _is_in_fov(sensor.x, sensor.y, cell_x, cell_y, sensor.spec):
        return 0.0

    return _detection_probability(distance, sensor.spec)


# ---------------------------------------------------------------------------
# Coverage map construction
# ---------------------------------------------------------------------------

def build_coverage_map(
    sensors: list[PlacedSensor],
    area: tuple[float, float, float, float],
    resolution: int = 20,
    coverage_threshold: float = 0.1,
) -> CoverageMap:
    """Build a grid-based coverage map for a set of placed sensors.

    Args:
        sensors: List of sensors with positions.
        area: ``(min_x, min_y, max_x, max_y)`` in local meters.
        resolution: Grid dimension (resolution x resolution cells).
        coverage_threshold: Minimum detection probability to count as "covered".

    Returns:
        A :class:`CoverageMap` with per-cell probabilities and summary stats.
    """
    resolution = max(2, min(200, resolution))
    min_x, min_y, max_x, max_y = area
    range_x = max_x - min_x
    range_y = max_y - min_y

    if range_x <= 0 or range_y <= 0:
        return CoverageMap(area=area, resolution=resolution)

    cell_w = range_x / resolution
    cell_h = range_y / resolution

    cells: list[list[CoverageCell]] = []
    total_prob = 0.0
    min_prob = 1.0
    max_prob = 0.0
    covered_count = 0
    total_cells = resolution * resolution

    for row in range(resolution):
        cell_row: list[CoverageCell] = []
        cy = min_y + (row + 0.5) * cell_h
        for col in range(resolution):
            cx = min_x + (col + 0.5) * cell_w

            # Combine detection probabilities: P(detect) = 1 - prod(1 - p_i)
            combined_miss = 1.0
            count = 0
            contributing: list[str] = []

            for s in sensors:
                p = _cell_detection_prob(cx, cy, s)
                if p > 0.0:
                    combined_miss *= (1.0 - p)
                    count += 1
                    contributing.append(s.sensor_id)

            combined_prob = 1.0 - combined_miss

            cell = CoverageCell(
                row=row,
                col=col,
                center_x=cx,
                center_y=cy,
                detection_prob=combined_prob,
                sensor_count=count,
                contributing_sensors=contributing,
            )
            cell_row.append(cell)

            total_prob += combined_prob
            min_prob = min(min_prob, combined_prob)
            max_prob = max(max_prob, combined_prob)
            if combined_prob >= coverage_threshold:
                covered_count += 1

        cells.append(cell_row)

    avg_prob = total_prob / total_cells if total_cells > 0 else 0.0
    overall_coverage = covered_count / total_cells if total_cells > 0 else 0.0

    return CoverageMap(
        area=area,
        resolution=resolution,
        cells=cells,
        overall_coverage=overall_coverage,
        avg_detection_prob=avg_prob,
        min_detection_prob=min_prob if total_cells > 0 else 0.0,
        max_detection_prob=max_prob,
    )


# ---------------------------------------------------------------------------
# Greedy placement optimizer
# ---------------------------------------------------------------------------

def optimize_placement(
    area: tuple[float, float, float, float],
    sensor_count: int,
    sensor_types: list[str] | None = None,
    resolution: int = 20,
    coverage_threshold: float = 0.1,
    existing_sensors: list[PlacedSensor] | None = None,
) -> PlacementResult:
    """Find optimal sensor positions using greedy coverage maximization.

    Places sensors one at a time, each at the grid cell with the lowest
    current detection probability (largest gap). This greedy approach
    gives a good approximation to the NP-hard coverage problem.

    Args:
        area: ``(min_x, min_y, max_x, max_y)`` bounding rectangle in meters.
        sensor_count: Number of new sensors to place.
        sensor_types: List of sensor type strings (one per sensor, or one for
            all).  Defaults to ``["ble_radio"]``.
        resolution: Grid dimension for coverage evaluation.
        coverage_threshold: Minimum probability to count as "covered".
        existing_sensors: Already-deployed sensors to account for.

    Returns:
        :class:`PlacementResult` with optimized sensor positions and coverage map.
    """
    sensor_count = max(0, min(100, sensor_count))
    resolution = max(2, min(200, resolution))

    if sensor_types is None:
        sensor_types = ["ble_radio"]

    min_x, min_y, max_x, max_y = area
    range_x = max_x - min_x
    range_y = max_y - min_y

    if range_x <= 0 or range_y <= 0 or sensor_count == 0:
        cmap = build_coverage_map(
            existing_sensors or [], area, resolution, coverage_threshold,
        )
        return PlacementResult(
            sensors=[],
            coverage_map=cmap,
            total_coverage=cmap.overall_coverage,
            avg_detection_prob=cmap.avg_detection_prob,
        )

    cell_w = range_x / resolution
    cell_h = range_y / resolution

    # Start with existing sensors
    placed: list[PlacedSensor] = list(existing_sensors or [])
    new_sensors: list[PlacedSensor] = []

    for i in range(sensor_count):
        # Determine sensor type for this placement
        type_idx = i if i < len(sensor_types) else len(sensor_types) - 1
        stype = sensor_types[type_idx]
        spec = SensorSpec.from_type(stype)

        # Build coverage map with current sensors
        cmap = build_coverage_map(placed, area, resolution, coverage_threshold)

        # Find cell with lowest detection probability (largest gap)
        best_row = 0
        best_col = 0
        best_prob = 2.0  # > 1.0 sentinel

        for row in range(resolution):
            for col in range(resolution):
                cell = cmap.cells[row][col]
                if cell.detection_prob < best_prob:
                    best_prob = cell.detection_prob
                    best_row = row
                    best_col = col

        # Place sensor at the center of the worst-covered cell
        sx = min_x + (best_col + 0.5) * cell_w
        sy = min_y + (best_row + 0.5) * cell_h

        sensor = PlacedSensor(
            sensor_id=f"opt_{stype}_{i}",
            x=sx,
            y=sy,
            spec=spec,
        )
        placed.append(sensor)
        new_sensors.append(sensor)

    # Final coverage map
    final_map = build_coverage_map(placed, area, resolution, coverage_threshold)

    return PlacementResult(
        sensors=new_sensors,
        coverage_map=final_map,
        total_coverage=final_map.overall_coverage,
        avg_detection_prob=final_map.avg_detection_prob,
    )


# ---------------------------------------------------------------------------
# Coverage gap analysis
# ---------------------------------------------------------------------------

def coverage_gaps(
    sensors: list[PlacedSensor],
    area: tuple[float, float, float, float],
    resolution: int = 20,
    gap_threshold: float = 0.1,
    max_clusters: int = 10,
) -> list[CoverageGap]:
    """Identify areas with insufficient sensor coverage.

    Uses K-means clustering on cells below the gap threshold to find
    distinct gap regions.

    Args:
        sensors: Currently deployed sensors.
        area: ``(min_x, min_y, max_x, max_y)`` bounding rectangle.
        resolution: Grid dimension.
        gap_threshold: Detection probability below which a cell is a "gap".
        max_clusters: Maximum number of gap clusters to return.

    Returns:
        List of :class:`CoverageGap` sorted by severity (worst first).
    """
    cmap = build_coverage_map(sensors, area, resolution, gap_threshold)

    # Collect gap cells
    gap_cells: list[tuple[float, float, float]] = []
    for row in cmap.cells:
        for cell in row:
            if cell.detection_prob < gap_threshold:
                gap_cells.append((cell.center_x, cell.center_y, cell.detection_prob))

    if not gap_cells:
        return []

    # K-means clustering of gap cells
    k = min(max_clusters, len(gap_cells))
    clusters = _kmeans_cluster(gap_cells, k)

    min_x, min_y, max_x, max_y = area
    range_x = max_x - min_x
    range_y = max_y - min_y
    cell_size = math.sqrt((range_x / resolution) * (range_y / resolution))

    gaps: list[CoverageGap] = []
    for cluster_cells in clusters:
        if not cluster_cells:
            continue
        cx = sum(c[0] for c in cluster_cells) / len(cluster_cells)
        cy = sum(c[1] for c in cluster_cells) / len(cluster_cells)
        avg_prob = sum(c[2] for c in cluster_cells) / len(cluster_cells)

        # Radius: max distance from centroid to any cluster cell
        max_dist = 0.0
        for c in cluster_cells:
            d = math.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2)
            max_dist = max(max_dist, d)
        radius = max_dist + cell_size * 0.5  # pad by half a cell

        severity = 1.0 - avg_prob  # higher = worse

        gaps.append(CoverageGap(
            center_x=cx,
            center_y=cy,
            radius_m=radius,
            avg_detection_prob=avg_prob,
            cell_count=len(cluster_cells),
            severity=severity,
        ))

    gaps.sort(key=lambda g: g.severity, reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# Redundancy analysis
# ---------------------------------------------------------------------------

def redundancy_analysis(
    sensors: list[PlacedSensor],
    area: tuple[float, float, float, float],
    resolution: int = 20,
    redundancy_threshold: int = 3,
    max_clusters: int = 10,
) -> list[RedundancyZone]:
    """Identify areas with excessive sensor overlap.

    Clusters cells where more sensors than ``redundancy_threshold`` contribute
    detection probability, indicating potential over-instrumentation.

    Args:
        sensors: Currently deployed sensors.
        area: ``(min_x, min_y, max_x, max_y)`` bounding rectangle.
        resolution: Grid dimension.
        redundancy_threshold: Minimum sensor count to flag as redundant.
        max_clusters: Maximum number of redundancy zones.

    Returns:
        List of :class:`RedundancyZone` sorted by max_sensor_count descending.
    """
    cmap = build_coverage_map(sensors, area, resolution)

    # Collect over-covered cells
    over_cells: list[tuple[float, float, int]] = []
    for row in cmap.cells:
        for cell in row:
            if cell.sensor_count >= redundancy_threshold:
                over_cells.append((cell.center_x, cell.center_y, cell.sensor_count))

    if not over_cells:
        return []

    k = min(max_clusters, len(over_cells))
    # Convert to (x, y, value) for clustering
    cluster_input = [(c[0], c[1], float(c[2])) for c in over_cells]
    clusters = _kmeans_cluster(cluster_input, k)

    min_x, min_y, max_x, max_y = area
    range_x = max_x - min_x
    range_y = max_y - min_y
    cell_size = math.sqrt((range_x / resolution) * (range_y / resolution))

    zones: list[RedundancyZone] = []
    for cluster_cells in clusters:
        if not cluster_cells:
            continue
        cx = sum(c[0] for c in cluster_cells) / len(cluster_cells)
        cy = sum(c[1] for c in cluster_cells) / len(cluster_cells)
        counts = [int(c[2]) for c in cluster_cells]
        avg_count = sum(counts) / len(counts)
        max_count = max(counts)

        max_dist = 0.0
        for c in cluster_cells:
            d = math.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2)
            max_dist = max(max_dist, d)
        radius = max_dist + cell_size * 0.5

        zones.append(RedundancyZone(
            center_x=cx,
            center_y=cy,
            radius_m=radius,
            avg_sensor_count=avg_count,
            max_sensor_count=max_count,
            cell_count=len(cluster_cells),
        ))

    zones.sort(key=lambda z: z.max_sensor_count, reverse=True)
    return zones


# ---------------------------------------------------------------------------
# K-means clustering helper
# ---------------------------------------------------------------------------

def _kmeans_cluster(
    points: list[tuple[float, float, float]],
    k: int,
    max_iterations: int = 50,
) -> list[list[tuple[float, float, float]]]:
    """Simple K-means clustering on 2D points (ignores the third value for distance).

    Args:
        points: List of ``(x, y, value)`` tuples.
        k: Number of clusters.
        max_iterations: Maximum iterations.

    Returns:
        List of ``k`` clusters, each a list of ``(x, y, value)`` tuples.
    """
    if not points or k <= 0:
        return []
    if k >= len(points):
        return [[p] for p in points]

    # Initialize centroids: evenly spaced from the point list
    step = max(1, len(points) // k)
    centroids: list[tuple[float, float]] = []
    for i in range(k):
        idx = min(i * step, len(points) - 1)
        centroids.append((points[idx][0], points[idx][1]))

    assignments: list[int] = [0] * len(points)

    for _iteration in range(max_iterations):
        # Assign each point to nearest centroid
        changed = False
        for i, (px, py, _) in enumerate(points):
            best_cluster = 0
            best_dist = float("inf")
            for ci, (cx, cy) in enumerate(centroids):
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_cluster = ci
            if assignments[i] != best_cluster:
                assignments[i] = best_cluster
                changed = True

        if not changed:
            break

        # Recompute centroids
        new_centroids: list[tuple[float, float]] = []
        for ci in range(k):
            cluster_pts = [
                points[i] for i in range(len(points)) if assignments[i] == ci
            ]
            if cluster_pts:
                mx = sum(p[0] for p in cluster_pts) / len(cluster_pts)
                my = sum(p[1] for p in cluster_pts) / len(cluster_pts)
                new_centroids.append((mx, my))
            else:
                new_centroids.append(centroids[ci])
        centroids = new_centroids

    # Build result clusters
    result: list[list[tuple[float, float, float]]] = [[] for _ in range(k)]
    for i, pt in enumerate(points):
        result[assignments[i]].append(pt)

    return result


# ---------------------------------------------------------------------------
# Utility: distance between two local-meter positions
# ---------------------------------------------------------------------------

def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in local meters."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

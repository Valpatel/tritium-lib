# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""CrowdDensityTracker -- crowd density grid for civil unrest mode.

Divides the map into a 10m cell grid and tracks the count of person-type
entities in each cell.  Density levels drive gameplay mechanics:

  - sparse (0-2):    normal operation, instigator identification possible
  - moderate (3-5):  normal operation, instigator identification possible
  - dense (6-10):    2x civilian conversion rate, instigator ID blocked
  - critical (11+):  3x conversion rate, instigator ID blocked, POI timer

When a POI building sits in a cell that has been at critical density for
a configurable timeout (default 60s), the tracker publishes an
``infrastructure_overwhelmed`` event signaling defeat for that POI.

Events published on the EventBus:
  - ``crowd_density``            every 1 second (grid state snapshot)
  - ``infrastructure_overwhelmed`` when a POI building is overrun
"""

from __future__ import annotations

import math
from typing import Any

from tritium_lib.models.target_status import is_terminal


# Density classification thresholds
_SPARSE_MAX = 2
_MODERATE_MAX = 5
_DENSE_MAX = 10

# Density level names
_SPARSE = "sparse"
_MODERATE = "moderate"
_DENSE = "dense"
_CRITICAL = "critical"

# Event publish interval (seconds)
_PUBLISH_INTERVAL = 1.0


def _classify(count: int) -> str:
    """Classify a cell count into a density level string."""
    if count <= _SPARSE_MAX:
        return _SPARSE
    if count <= _MODERATE_MAX:
        return _MODERATE
    if count <= _DENSE_MAX:
        return _DENSE
    return _CRITICAL


class CrowdDensityTracker:
    """Track crowd density on a 10m cell grid for civil unrest mode.

    Args:
        bounds: (x_min, y_min, x_max, y_max) -- map boundary
        event_bus: EventBus for publishing crowd_density events.
                   Optional -- if None, no events are published.
        cell_size: grid cell size in meters (default 10)
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        event_bus: object | None = None,
        cell_size: float = 10.0,
    ) -> None:
        self.bounds = bounds
        self._event_bus = event_bus
        if cell_size <= 0:
            cell_size = 10.0
        self.cell_size = cell_size

        x_min, y_min, x_max, y_max = bounds
        self.cols = max(1, math.ceil((x_max - x_min) / cell_size))
        self.rows = max(1, math.ceil((y_max - y_min) / cell_size))

        # 2D grid of counts (row-major: grid[row][col])
        self._counts: list[list[int]] = [
            [0] * self.cols for _ in range(self.rows)
        ]
        # 2D grid of density level strings
        self._levels: list[list[str]] = [
            [_SPARSE] * self.cols for _ in range(self.rows)
        ]

        # Event publish timer (accumulated dt)
        self._publish_timer: float = 0.0

        # POI buildings: list of (position, name, critical_timer)
        self._poi_buildings: list[dict] = []

    def _pos_to_cell(self, position: tuple[float, float]) -> tuple[int, int]:
        """Map a world position to grid (row, col), clamped to bounds."""
        x_min, y_min, _, _ = self.bounds
        col = int((position[0] - x_min) / self.cell_size)
        row = int((position[1] - y_min) / self.cell_size)
        # Clamp to valid range
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return row, col

    def tick(self, targets: dict[str, Any], dt: float) -> None:
        """Count person-type entities per cell and update density levels.

        Args:
            targets: dict of target_id -> target objects with
                     .asset_type, .status, .position attributes
            dt: time step in seconds
        """
        # Reset counts
        for r in range(self.rows):
            for c in range(self.cols):
                self._counts[r][c] = 0

        # Count person-type entities per cell
        for target in targets.values():
            if target.asset_type != "person":
                continue
            if is_terminal(target.status):
                continue
            row, col = self._pos_to_cell(target.position)
            self._counts[row][col] += 1

        # Classify density per cell
        for r in range(self.rows):
            for c in range(self.cols):
                self._levels[r][c] = _classify(self._counts[r][c])

        # Update POI critical density timers
        for poi in self._poi_buildings:
            row, col = self._pos_to_cell(poi["position"])
            if self._levels[row][col] == _CRITICAL:
                poi["critical_timer"] += dt
            else:
                # Reset timer if density drops below critical
                poi["critical_timer"] = 0.0

        # Publish event every _PUBLISH_INTERVAL seconds
        self._publish_timer += dt
        if self._publish_timer >= _PUBLISH_INTERVAL:
            self._publish_timer -= _PUBLISH_INTERVAL
            self._publish_event()

    def _publish_event(self) -> None:
        """Publish crowd_density event on EventBus."""
        if self._event_bus is None:
            return
        # Find max density and critical cell count
        max_density = _SPARSE
        critical_count = 0
        density_order = {_SPARSE: 0, _MODERATE: 1, _DENSE: 2, _CRITICAL: 3}

        for r in range(self.rows):
            for c in range(self.cols):
                level = self._levels[r][c]
                if density_order[level] > density_order[max_density]:
                    max_density = level
                if level == _CRITICAL:
                    critical_count += 1

        self._event_bus.publish("crowd_density", {
            "grid": [row[:] for row in self._levels],
            "cell_size": self.cell_size,
            "bounds": list(self.bounds),
            "max_density": max_density,
            "critical_count": critical_count,
        })

    def get_density_at(self, position: tuple[float, float]) -> str:
        """Return density classification at given position.

        Returns:
            One of: "sparse", "moderate", "dense", "critical"
        """
        row, col = self._pos_to_cell(position)
        return self._levels[row][col]

    def get_conversion_multiplier(self, position: tuple[float, float]) -> float:
        """Return conversion rate multiplier at given position.

        Returns:
            1.0 for sparse/moderate, 2.0 for dense, 3.0 for critical
        """
        density = self.get_density_at(position)
        if density == _DENSE:
            return 2.0
        if density == _CRITICAL:
            return 3.0
        return 1.0

    def can_identify_instigator(self, position: tuple[float, float]) -> bool:
        """Check if instigator identification is possible at given position.

        Returns:
            True for sparse/moderate, False for dense/critical
        """
        density = self.get_density_at(position)
        return density in (_SPARSE, _MODERATE)

    def add_poi_building(self, position: tuple[float, float], name: str) -> None:
        """Register a POI building position for critical density timer tracking.

        Args:
            position: (x, y) world position of the building
            name: display name of the building
        """
        self._poi_buildings.append({
            "position": position,
            "name": name,
            "critical_timer": 0.0,
        })

    def check_poi_defeat(self, timeout: float = 60.0) -> bool:
        """Check if any POI has had critical density for >= timeout seconds.

        Publishes ``infrastructure_overwhelmed`` event if triggered.

        Args:
            timeout: seconds of critical density required (default 60)

        Returns:
            True if any POI has been overwhelmed
        """
        for poi in self._poi_buildings:
            if poi["critical_timer"] >= timeout:
                if self._event_bus is not None:
                    self._event_bus.publish("infrastructure_overwhelmed", {
                        "poi_name": poi["name"],
                        "poi_position": list(poi["position"]),
                        "critical_duration": poi["critical_timer"],
                    })
                return True
        return False

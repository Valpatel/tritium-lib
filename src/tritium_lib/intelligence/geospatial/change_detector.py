# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Temporal change detection for geospatial terrain layers.

Compares two terrain layer snapshots (different dates) to detect
significant changes: new construction, vegetation clearing, flooding,
road blockages, etc.

Change detection pipeline:
    Previous terrain layer (cached) ──┐
                                       ├── Diff → Changed regions
    Current terrain layer (new)  ──────┘
                                              │
                                        ┌─────▼─────┐
                                        │  Alert:    │ "New construction"
                                        │  Update    │ "Road blocked"
                                        │  Context   │ "Vegetation cleared"
                                        └────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from tritium_lib.models.terrain import TerrainType

logger = logging.getLogger(__name__)


@dataclass
class TerrainChange:
    """A detected change between two terrain layer snapshots."""
    centroid_lat: float
    centroid_lon: float
    previous_type: TerrainType
    current_type: TerrainType
    area_m2: float
    confidence: float
    description: str
    severity: str = "info"  # info, warning, critical


@dataclass
class ChangeReport:
    """Summary of changes between two terrain snapshots."""
    ao_id: str
    previous_date: Optional[datetime] = None
    current_date: Optional[datetime] = None
    changes: list[TerrainChange] = field(default_factory=list)
    total_changed_area_m2: float = 0.0

    @property
    def change_count(self) -> int:
        return len(self.changes)

    def summary(self) -> str:
        """Human-readable change summary for commander AI."""
        if not self.changes:
            return f"No significant terrain changes detected in AO '{self.ao_id}'."

        lines = [f"TERRAIN CHANGE REPORT — AO \"{self.ao_id}\""]
        lines.append(f"Changes detected: {self.change_count}")
        lines.append(f"Total changed area: {self.total_changed_area_m2:.0f} m²")
        lines.append("")

        for change in self.changes:
            lines.append(
                f"  [{change.severity.upper()}] {change.description} "
                f"({change.area_m2:.0f} m² at {change.centroid_lat:.4f}, {change.centroid_lon:.4f})"
            )
        return "\n".join(lines)


# Change severity rules
_SEVERITY_RULES: dict[tuple[str, str], str] = {
    # Previous type → Current type → severity
    ("road", "water"): "critical",       # flooding
    ("road", "barren"): "warning",       # road damage/construction
    ("building", "barren"): "warning",   # demolition
    ("vegetation", "barren"): "info",    # clearing
    ("vegetation", "building"): "info",  # new construction
    ("water", "barren"): "warning",      # drought/dam
    ("parking", "building"): "info",     # new construction
}

# Change descriptions
_CHANGE_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("road", "water"): "Road flooding detected",
    ("road", "barren"): "Road under construction or damaged",
    ("building", "barren"): "Building demolished or collapsed",
    ("vegetation", "barren"): "Vegetation cleared",
    ("vegetation", "building"): "New construction in previously vegetated area",
    ("water", "barren"): "Water body dried up or drained",
    ("parking", "building"): "New construction on parking lot",
    ("barren", "building"): "New building construction completed",
    ("barren", "vegetation"): "Vegetation regrowth",
    ("road", "building"): "Road blocked by new structure",
}


class ChangeDetector:
    """Detects changes between two terrain layer snapshots.

    Compares regions by spatial proximity — if a region at the same
    location has changed terrain type, that's a change event.
    """

    def __init__(self, min_area_m2: float = 50.0) -> None:
        self.min_area_m2 = min_area_m2

    def detect_changes(
        self,
        previous_layer: object,
        current_layer: object,
    ) -> ChangeReport:
        """Compare two terrain layers and report changes.

        Both layers must have a `regions` property returning
        list[SegmentedRegion].
        """
        if not hasattr(previous_layer, 'regions') or not hasattr(current_layer, 'regions'):
            return ChangeReport(ao_id="unknown")

        ao_id = "unknown"
        if hasattr(previous_layer, 'metadata') and previous_layer.metadata:
            ao_id = previous_layer.metadata.ao_id

        prev_date = None
        curr_date = None
        if hasattr(previous_layer, 'metadata') and previous_layer.metadata:
            prev_date = previous_layer.metadata.created_at
        if hasattr(current_layer, 'metadata') and current_layer.metadata:
            curr_date = current_layer.metadata.created_at

        changes: list[TerrainChange] = []
        total_area = 0.0

        # Build spatial index of previous regions
        prev_by_location: dict[tuple[int, int], list] = {}
        for region in previous_layer.regions:
            cell = (
                int(region.centroid_lon * 10000),
                int(region.centroid_lat * 10000),
            )
            if cell not in prev_by_location:
                prev_by_location[cell] = []
            prev_by_location[cell].append(region)

        # Compare current regions against previous
        for region in current_layer.regions:
            if region.area_m2 < self.min_area_m2:
                continue

            cell = (
                int(region.centroid_lon * 10000),
                int(region.centroid_lat * 10000),
            )

            # Check this cell and neighbors
            matched = False
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    for prev in prev_by_location.get(neighbor, []):
                        if prev.terrain_type != region.terrain_type:
                            # Terrain type changed at this location
                            change = self._create_change(prev, region)
                            if change is not None:
                                changes.append(change)
                                total_area += change.area_m2
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break

        report = ChangeReport(
            ao_id=ao_id,
            previous_date=prev_date,
            current_date=curr_date,
            changes=changes,
            total_changed_area_m2=total_area,
        )

        if changes:
            logger.info(
                "Detected %d terrain changes in AO '%s' (%.0f m²)",
                len(changes), ao_id, total_area,
            )

        return report

    def _create_change(
        self,
        previous: object,
        current: object,
    ) -> Optional[TerrainChange]:
        """Create a TerrainChange from a previous/current region pair."""
        prev_type = previous.terrain_type
        curr_type = current.terrain_type

        if prev_type == curr_type:
            return None

        key = (prev_type.value, curr_type.value)
        description = _CHANGE_DESCRIPTIONS.get(
            key,
            f"Terrain changed from {prev_type.value} to {curr_type.value}",
        )
        severity = _SEVERITY_RULES.get(key, "info")

        return TerrainChange(
            centroid_lat=current.centroid_lat,
            centroid_lon=current.centroid_lon,
            previous_type=prev_type,
            current_type=curr_type,
            area_m2=current.area_m2,
            confidence=min(previous.confidence, current.confidence),
            description=description,
            severity=severity,
        )

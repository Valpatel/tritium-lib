# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Convoy visualization model for rendering convoy data on the frontend map.

Provides a lightweight model optimized for the tactical map renderer.
While the Convoy model (convoy.py) tracks full analytical state, this
model carries only what the frontend needs to draw convoy overlays:
bounding polygon, heading arrow, speed label, and formation shape.

MQTT topic:
    tritium/{site}/visualization/convoys — convoy overlay updates
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConvoyFormationType(str, Enum):
    """Formation type for visual rendering."""

    COLUMN = "column"      # Targets in a single-file line
    PARALLEL = "parallel"  # Targets side-by-side
    CLUSTER = "cluster"    # Targets grouped tightly


class LatLng(BaseModel):
    """A single latitude/longitude coordinate."""

    lat: float = 0.0
    lng: float = 0.0


class ConvoyVisualization(BaseModel):
    """Frontend-ready convoy rendering data.

    Carries the minimum data needed to draw a convoy overlay on the
    tactical map: the member IDs, heading arrow, speed label, formation
    shape, confidence ring, and a bounding polygon for the group.

    Attributes:
        convoy_id: Unique convoy identifier (matches Convoy.convoy_id).
        target_ids: List of target IDs belonging to this convoy.
        heading_degrees: Average heading in degrees (0=north, 90=east).
        speed_estimate: Estimated speed in m/s.
        formation_type: Visual formation shape.
        confidence: 0.0-1.0 confidence that this is a real convoy.
        bounding_box: List of lat/lng pairs forming the convex hull
            around convoy members.
        label: Optional display label for the map overlay.
        color: CSS color for rendering (defaults to yellow/warning).
    """

    convoy_id: str = ""
    target_ids: list[str] = Field(default_factory=list)
    heading_degrees: float = Field(default=0.0, ge=0.0, le=360.0)
    speed_estimate: float = Field(default=0.0, ge=0.0)
    formation_type: ConvoyFormationType = ConvoyFormationType.CLUSTER
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bounding_box: list[LatLng] = Field(default_factory=list)
    label: str = ""
    color: str = "#fcee0a"

    @property
    def member_count(self) -> int:
        """Number of targets in this convoy."""
        return len(self.target_ids)

    @property
    def has_bounding_box(self) -> bool:
        """Whether a valid bounding polygon exists (3+ points)."""
        return len(self.bounding_box) >= 3

    @property
    def speed_kmh(self) -> float:
        """Speed converted to km/h."""
        return self.speed_estimate * 3.6

    def heading_label(self) -> str:
        """Human-readable heading direction."""
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int((self.heading_degrees + 22.5) / 45.0) % 8
        return dirs[idx]

    def to_dict(self) -> dict:
        """Serialize for JSON transport to the frontend."""
        return {
            "convoy_id": self.convoy_id,
            "target_ids": self.target_ids,
            "heading_degrees": self.heading_degrees,
            "speed_estimate": self.speed_estimate,
            "formation_type": self.formation_type.value,
            "confidence": self.confidence,
            "bounding_box": [{"lat": p.lat, "lng": p.lng} for p in self.bounding_box],
            "label": self.label,
            "color": self.color,
            "member_count": self.member_count,
            "speed_kmh": self.speed_kmh,
            "heading_label": self.heading_label(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConvoyVisualization:
        """Deserialize from a plain dict."""
        bbox = [
            LatLng(lat=p.get("lat", 0.0), lng=p.get("lng", 0.0))
            for p in data.get("bounding_box", [])
        ]
        formation = data.get("formation_type", "cluster")
        if isinstance(formation, str):
            formation = ConvoyFormationType(formation)

        return cls(
            convoy_id=data.get("convoy_id", ""),
            target_ids=data.get("target_ids", []),
            heading_degrees=data.get("heading_degrees", 0.0),
            speed_estimate=data.get("speed_estimate", 0.0),
            formation_type=formation,
            confidence=data.get("confidence", 0.0),
            bounding_box=bbox,
            label=data.get("label", ""),
            color=data.get("color", "#fcee0a"),
        )

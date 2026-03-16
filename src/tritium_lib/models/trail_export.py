# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Trail export models for target movement history.

Provides the TrailPoint and TrailExport data models used to represent
a target's position history for export to GPX, KML, GeoJSON, or CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TrailFormat(str, Enum):
    """Supported trail export formats."""
    GPX = "gpx"
    KML = "kml"
    GEOJSON = "geojson"
    CSV = "csv"
    JSON = "json"


@dataclass
class TrailPoint:
    """A single point in a target's movement trail.

    Attributes:
        lat: Latitude in decimal degrees (WGS84).
        lng: Longitude in decimal degrees (WGS84).
        alt: Altitude in meters (optional).
        timestamp: UTC timestamp of the observation.
        speed: Speed in m/s at this point (optional).
        heading: Heading in degrees (0-360) at this point (optional).
        confidence: Position confidence (0.0-1.0, optional).
        source: Sensor source that produced this point (optional).
    """
    lat: float
    lng: float
    alt: Optional[float] = None
    timestamp: Optional[datetime] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    confidence: Optional[float] = None
    source: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lng": self.lng,
            "alt": self.alt,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "speed": self.speed,
            "heading": self.heading,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrailPoint:
        ts = data.get("timestamp") or data.get("time")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif isinstance(ts, (int, float)):
            from datetime import timezone
            ts = datetime.fromtimestamp(ts, tz=timezone.utc)
        return cls(
            lat=float(data.get("lat", 0.0)),
            lng=float(data.get("lng", data.get("lon", 0.0))),
            alt=data.get("alt", data.get("altitude", data.get("ele"))),
            timestamp=ts,
            speed=data.get("speed"),
            heading=data.get("heading"),
            confidence=data.get("confidence"),
            source=data.get("source"),
        )


@dataclass
class TrailExport:
    """Complete trail export for a target.

    Bundles a target's identity with its position history and the
    requested export format. Used by the backend to generate GPX/KML/etc.

    Attributes:
        target_id: Unique identifier for the target.
        format: Export format (gpx, kml, geojson, csv, json).
        points: Ordered list of trail points.
        metadata: Optional metadata dict (target name, alliance, etc.).
    """
    target_id: str
    format: TrailFormat = TrailFormat.GPX
    points: list[TrailPoint] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration from first to last point in seconds, if timestamps exist."""
        times = [p.timestamp for p in self.points if p.timestamp is not None]
        if len(times) < 2:
            return None
        return (max(times) - min(times)).total_seconds()

    @property
    def total_distance_m(self) -> float:
        """Approximate total distance in meters using haversine."""
        import math
        total = 0.0
        for i in range(1, len(self.points)):
            p1 = self.points[i - 1]
            p2 = self.points[i]
            lat1, lon1 = math.radians(p1.lat), math.radians(p1.lng)
            lat2, lon2 = math.radians(p2.lat), math.radians(p2.lng)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            total += 6371000 * c  # Earth radius in meters
        return total

    def simplify(self, tolerance: float = 0.00005) -> TrailExport:
        """Return a new TrailExport with simplified points using Ramer-Douglas-Peucker.

        Args:
            tolerance: Simplification tolerance in degrees (default ~5m).

        Returns:
            New TrailExport with reduced point count.
        """
        if len(self.points) <= 2:
            return TrailExport(
                target_id=self.target_id,
                format=self.format,
                points=list(self.points),
                metadata=dict(self.metadata),
            )

        def _perp_dist(pt: TrailPoint, start: TrailPoint, end: TrailPoint) -> float:
            """Perpendicular distance from pt to line segment start-end."""
            dx = end.lng - start.lng
            dy = end.lat - start.lat
            if dx == 0.0 and dy == 0.0:
                return ((pt.lng - start.lng) ** 2 + (pt.lat - start.lat) ** 2) ** 0.5
            t = ((pt.lng - start.lng) * dx + (pt.lat - start.lat) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            proj_lng = start.lng + t * dx
            proj_lat = start.lat + t * dy
            return ((pt.lng - proj_lng) ** 2 + (pt.lat - proj_lat) ** 2) ** 0.5

        def _rdp(pts: list[TrailPoint], tol: float) -> list[TrailPoint]:
            if len(pts) <= 2:
                return pts
            max_dist = 0.0
            max_idx = 0
            for i in range(1, len(pts) - 1):
                d = _perp_dist(pts[i], pts[0], pts[-1])
                if d > max_dist:
                    max_dist = d
                    max_idx = i
            if max_dist > tol:
                left = _rdp(pts[:max_idx + 1], tol)
                right = _rdp(pts[max_idx:], tol)
                return left[:-1] + right
            return [pts[0], pts[-1]]

        simplified = _rdp(self.points, tolerance)
        return TrailExport(
            target_id=self.target_id,
            format=self.format,
            points=simplified,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "format": self.format.value if isinstance(self.format, TrailFormat) else self.format,
            "points": [p.to_dict() for p in self.points],
            "metadata": self.metadata,
            "point_count": self.point_count,
            "duration_seconds": self.duration_seconds,
            "total_distance_m": self.total_distance_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrailExport:
        fmt = data.get("format", "gpx")
        if isinstance(fmt, str):
            try:
                fmt = TrailFormat(fmt)
            except ValueError:
                fmt = TrailFormat.GPX
        return cls(
            target_id=data.get("target_id", ""),
            format=fmt,
            points=[TrailPoint.from_dict(p) for p in data.get("points", [])],
            metadata=data.get("metadata", {}),
        )

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Time series and paginated result models for telemetry and API responses.

Generic containers for sensor data over time, fleet telemetry aggregation,
and paginated API responses.
"""

from datetime import datetime, timezone
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class TimeSeriesPoint(BaseModel):
    """A single data point in a time series."""

    timestamp: datetime
    value: float
    label: Optional[str] = None


class TimeSeries(BaseModel):
    """A named time series with metadata.

    Represents a sequence of (timestamp, value) points for a single metric
    from a single source.
    """

    metric: str = Field(description="Metric name, e.g. 'free_heap', 'wifi_rssi', 'temperature'")
    device_id: Optional[str] = None
    unit: str = ""
    points: list[TimeSeriesPoint] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def latest(self) -> Optional[TimeSeriesPoint]:
        return self.points[-1] if self.points else None

    @property
    def oldest(self) -> Optional[TimeSeriesPoint]:
        return self.points[0] if self.points else None

    def values(self) -> list[float]:
        """Return just the values as a flat list."""
        return [p.value for p in self.points]

    def stats(self) -> dict:
        """Compute basic statistics over the series."""
        vals = self.values()
        if not vals:
            return {"min": None, "max": None, "mean": None, "count": 0}
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "count": len(vals),
            "first": self.oldest.timestamp.isoformat() if self.oldest else None,
            "last": self.latest.timestamp.isoformat() if self.latest else None,
        }

    def rate_of_change(self) -> Optional[float]:
        """Compute rate of change per hour (value delta / time delta).

        Returns None if fewer than 2 points or zero time span.
        """
        if len(self.points) < 2:
            return None
        first = self.points[0]
        last = self.points[-1]
        dt_hours = (last.timestamp - first.timestamp).total_seconds() / 3600.0
        if dt_hours <= 0:
            return None
        return (last.value - first.value) / dt_hours


class FleetTimeSeries(BaseModel):
    """Aggregated time series across multiple devices for a single metric."""

    metric: str
    unit: str = ""
    series: dict[str, TimeSeries] = Field(
        default_factory=dict,
        description="device_id -> TimeSeries",
    )

    @property
    def device_count(self) -> int:
        return len(self.series)

    def fleet_stats(self) -> dict:
        """Aggregate stats across all devices."""
        all_vals = []
        for ts in self.series.values():
            all_vals.extend(ts.values())
        if not all_vals:
            return {"min": None, "max": None, "mean": None, "devices": 0, "total_points": 0}
        return {
            "min": min(all_vals),
            "max": max(all_vals),
            "mean": sum(all_vals) / len(all_vals),
            "devices": self.device_count,
            "total_points": len(all_vals),
        }

    def outlier_devices(self, threshold_stddev: float = 2.0) -> list[str]:
        """Find devices whose latest value is >N stddev from fleet mean."""
        latest_vals = {}
        for did, ts in self.series.items():
            if ts.latest:
                latest_vals[did] = ts.latest.value

        if len(latest_vals) < 3:
            return []

        vals = list(latest_vals.values())
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        stddev = variance**0.5

        if stddev <= 0:
            return []

        return [
            did
            for did, v in latest_vals.items()
            if abs(v - mean) > threshold_stddev * stddev
        ]


class PagedResult(BaseModel):
    """Paginated API response container."""

    items: list[Any] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 50
    has_more: bool = False

    @classmethod
    def from_list(
        cls,
        all_items: list,
        offset: int = 0,
        limit: int = 50,
    ) -> "PagedResult":
        """Create a paged result from a full list."""
        total = len(all_items)
        page = all_items[offset : offset + limit]
        return cls(
            items=page,
            total=total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < total,
        )

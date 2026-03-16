# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Daily pattern models — quantify how predictable a target's daily schedule is.

Provides DailyPattern for analyzing 24-hour activity histograms of targets.
Used to detect regular commuters, patrol routes, and anomalous schedule breaks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class DailyPattern(BaseModel):
    """A target's daily activity pattern — 24-bin hourly histogram.

    Attributes:
        target_id: The target this pattern describes.
        hourly_counts: List of 24 integers, one per hour (0-23).
        peak_hour: Hour of day with the most activity (0-23).
        quiet_hours: List of hours with zero or near-zero activity.
        regularity_score: 0.0-1.0 how predictable the schedule is.
            1.0 = perfectly regular (same hours every day).
            0.0 = completely random / no pattern.
        total_sightings: Total number of sightings across all hours.
        days_observed: Number of distinct days contributing data.
        created_at: When this pattern was first computed.
        updated_at: When the pattern was last refreshed.
    """

    target_id: str = ""
    hourly_counts: list[int] = Field(
        default_factory=lambda: [0] * 24,
        description="24-bin histogram, index 0 = midnight, index 23 = 11 PM",
    )
    peak_hour: int = 0
    quiet_hours: list[int] = Field(default_factory=list)
    regularity_score: float = 0.0
    total_sightings: int = 0
    days_observed: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def model_post_init(self, __context) -> None:
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    def compute_peak_hour(self) -> int:
        """Recompute peak_hour from hourly_counts."""
        if not self.hourly_counts or max(self.hourly_counts) == 0:
            self.peak_hour = 0
            return 0
        self.peak_hour = self.hourly_counts.index(max(self.hourly_counts))
        return self.peak_hour

    def compute_quiet_hours(self, threshold_pct: float = 0.05) -> list[int]:
        """Recompute quiet_hours — hours below threshold_pct of peak activity.

        Args:
            threshold_pct: Fraction of peak count below which an hour is 'quiet'.
        """
        peak_count = max(self.hourly_counts) if self.hourly_counts else 0
        if peak_count == 0:
            self.quiet_hours = list(range(24))
            return self.quiet_hours

        threshold = peak_count * threshold_pct
        self.quiet_hours = [
            h for h, c in enumerate(self.hourly_counts) if c <= threshold
        ]
        return self.quiet_hours

    def compute_regularity_score(self) -> float:
        """Compute how regular the activity pattern is.

        Uses normalized entropy — uniform distribution = 0.0 (random),
        single-spike = 1.0 (perfectly predictable).
        """
        import math

        total = sum(self.hourly_counts)
        if total == 0:
            self.regularity_score = 0.0
            return 0.0

        # Calculate normalized entropy (0 = concentrated, 1 = uniform)
        max_entropy = math.log(24)
        entropy = 0.0
        for c in self.hourly_counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log(p)

        # Invert: high entropy = low regularity
        self.regularity_score = round(1.0 - (entropy / max_entropy), 4)
        return self.regularity_score

    def recompute(self) -> None:
        """Recompute all derived fields from hourly_counts."""
        self.total_sightings = sum(self.hourly_counts)
        self.compute_peak_hour()
        self.compute_quiet_hours()
        self.compute_regularity_score()
        self.updated_at = datetime.now(timezone.utc)

    def add_sighting(self, hour: int) -> None:
        """Record a sighting at the given hour (0-23)."""
        if 0 <= hour <= 23:
            self.hourly_counts[hour] += 1
            self.total_sightings += 1

    @property
    def active_hours(self) -> list[int]:
        """Hours with at least one sighting."""
        return [h for h, c in enumerate(self.hourly_counts) if c > 0]

    @property
    def is_daytime_only(self) -> bool:
        """True if all activity is between 6 AM and 6 PM."""
        for h in range(0, 6):
            if self.hourly_counts[h] > 0:
                return False
        for h in range(18, 24):
            if self.hourly_counts[h] > 0:
                return False
        return self.total_sightings > 0

    @property
    def is_nighttime_only(self) -> bool:
        """True if all activity is between 6 PM and 6 AM."""
        for h in range(6, 18):
            if self.hourly_counts[h] > 0:
                return False
        return self.total_sightings > 0

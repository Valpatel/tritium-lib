# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Behavioral clustering models — group targets by movement similarity.

Provides BehaviorCluster for grouping targets that exhibit similar
movement patterns (same speed range, same time of day, same areas).
Used by the behavioral intelligence plugin to auto-detect target
groups based on behavioral similarity rather than operator assignment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FormationType(str, Enum):
    """How targets in the cluster move relative to each other."""

    CONVOY = "convoy"            # Targets moving in a line along the same route
    SWARM = "swarm"              # Targets clustered tightly and moving together
    PATROL = "patrol"            # Targets following a repeated route pattern
    DISPERSED = "dispersed"      # Targets in same area but spread out
    STATIONARY = "stationary"    # Targets dwelling in the same location
    UNKNOWN = "unknown"


class CommonPattern(BaseModel):
    """Shared behavioral characteristics of a cluster.

    Describes what the targets in the cluster have in common:
    speed range, active time window, and spatial region.
    """

    speed_min_mps: float = 0.0       # Min observed speed (m/s)
    speed_max_mps: float = 0.0       # Max observed speed (m/s)
    active_hour_start: int = 0       # Hour of day when cluster is active (0-23)
    active_hour_end: int = 23        # Hour of day when cluster is inactive (0-23)
    primary_heading_deg: float = -1  # Dominant heading (-1 = no dominant heading)
    avg_dwell_s: float = 0.0         # Average dwell time in seconds
    regularity_score: float = 0.0    # 0.0-1.0 how regular the pattern is


class BehaviorCluster(BaseModel):
    """A group of targets exhibiting similar movement behavior.

    Attributes:
        cluster_id: Unique identifier for this cluster.
        targets: List of target IDs in this cluster.
        common_pattern: Shared behavioral characteristics.
        centroid_lat: Center latitude of the cluster's activity area.
        centroid_lng: Center longitude of the cluster's activity area.
        radius_m: Radius in meters encompassing the cluster's activity.
        formation_type: How the targets move relative to each other.
        confidence: 0.0-1.0 confidence that the clustering is meaningful.
        created_at: When the cluster was first detected.
        updated_at: When the cluster was last updated.
        observation_count: How many sightings contributed to this cluster.
        source_patterns: Pattern IDs from the pattern detector that formed this cluster.
    """

    cluster_id: str = ""
    targets: list[str] = Field(default_factory=list)
    common_pattern: CommonPattern = Field(default_factory=CommonPattern)
    centroid_lat: float = 0.0
    centroid_lng: float = 0.0
    radius_m: float = 100.0
    formation_type: FormationType = FormationType.UNKNOWN
    confidence: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    observation_count: int = 0
    source_patterns: list[str] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    @property
    def target_count(self) -> int:
        """Number of targets in this cluster."""
        return len(self.targets)

    def add_target(self, target_id: str) -> bool:
        """Add a target to the cluster. Returns True if added."""
        if target_id not in self.targets:
            self.targets.append(target_id)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def remove_target(self, target_id: str) -> bool:
        """Remove a target from the cluster. Returns True if removed."""
        if target_id in self.targets:
            self.targets.remove(target_id)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def has_target(self, target_id: str) -> bool:
        """Check if a target belongs to this cluster."""
        return target_id in self.targets

    def merge(self, other: BehaviorCluster) -> None:
        """Merge another cluster into this one.

        Adds all targets from the other cluster, averages the centroids,
        and expands the radius to encompass both clusters.
        """
        for tid in other.targets:
            self.add_target(tid)

        # Average centroids weighted by observation count
        total_obs = self.observation_count + other.observation_count
        if total_obs > 0:
            w1 = self.observation_count / total_obs
            w2 = other.observation_count / total_obs
            self.centroid_lat = self.centroid_lat * w1 + other.centroid_lat * w2
            self.centroid_lng = self.centroid_lng * w1 + other.centroid_lng * w2

        # Expand radius to encompass both
        self.radius_m = max(self.radius_m, other.radius_m) * 1.2
        self.observation_count = total_obs
        self.confidence = min(1.0, (self.confidence + other.confidence) / 2.0)

        # Merge source patterns
        for pid in other.source_patterns:
            if pid not in self.source_patterns:
                self.source_patterns.append(pid)

        self.updated_at = datetime.now(timezone.utc)


class ClusterSummary(BaseModel):
    """Lightweight summary of a behavior cluster for listing endpoints."""

    cluster_id: str = ""
    target_count: int = 0
    formation_type: FormationType = FormationType.UNKNOWN
    centroid_lat: float = 0.0
    centroid_lng: float = 0.0
    radius_m: float = 100.0
    confidence: float = 0.0
    speed_range: str = ""
    active_hours: str = ""

    @classmethod
    def from_cluster(cls, cluster: BehaviorCluster) -> ClusterSummary:
        cp = cluster.common_pattern
        speed_range = f"{cp.speed_min_mps:.1f}-{cp.speed_max_mps:.1f} m/s"
        active_hours = f"{cp.active_hour_start:02d}:00-{cp.active_hour_end:02d}:00"
        return cls(
            cluster_id=cluster.cluster_id,
            target_count=cluster.target_count,
            formation_type=cluster.formation_type,
            centroid_lat=cluster.centroid_lat,
            centroid_lng=cluster.centroid_lng,
            radius_m=cluster.radius_m,
            confidence=cluster.confidence,
            speed_range=speed_range,
            active_hours=active_hours,
        )

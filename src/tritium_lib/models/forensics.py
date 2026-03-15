# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Forensic reconstruction and incident report models.

ForensicReconstruction captures a time-bounded, area-bounded historical
analysis of what happened: which targets were present, what events occurred,
and the chain of evidence linking sensors to conclusions.

IncidentReport extends IntelligenceReport with a reconstruction reference,
providing a structured document for after-action review.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReconstructionStatus(str, Enum):
    """Lifecycle status of a forensic reconstruction."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class GeoBounds(BaseModel):
    """Rectangular geographic area of interest."""
    north: float = 0.0  # max latitude
    south: float = 0.0  # min latitude
    east: float = 0.0   # max longitude
    west: float = 0.0   # min longitude

    model_config = {"frozen": False}

    def contains(self, lat: float, lng: float) -> bool:
        """Check if a point is within these bounds."""
        return self.south <= lat <= self.north and self.west <= lng <= self.east


class TimeRange(BaseModel):
    """Time window for reconstruction."""
    start: float = 0.0  # Unix timestamp
    end: float = 0.0    # Unix timestamp

    model_config = {"frozen": False}

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, ts: float) -> bool:
        return self.start <= ts <= self.end


class EvidenceItem(BaseModel):
    """A single piece of evidence in the chain — a sensor observation."""
    evidence_id: str = ""
    timestamp: float = 0.0
    sensor_id: str = ""       # device that observed this
    sensor_type: str = ""     # ble, wifi, camera, mesh, acoustic
    target_id: str = ""       # target this evidence concerns
    observation_type: str = ""  # sighting, detection, classification, etc.
    data: dict = Field(default_factory=dict)  # raw observation data
    confidence: float = 0.0   # 0.0 to 1.0

    model_config = {"frozen": False}


class TargetTimeline(BaseModel):
    """A single target's activity during the reconstruction window."""
    target_id: str = ""
    target_type: str = ""      # person, vehicle, phone, etc.
    first_seen: float = 0.0    # Unix timestamp
    last_seen: float = 0.0
    positions: list[dict] = Field(default_factory=list)  # [{ts, lat, lng, source}]
    events: list[dict] = Field(default_factory=list)      # relevant events
    evidence_ids: list[str] = Field(default_factory=list)  # links to EvidenceItem
    alliance: str = "unknown"  # friendly, hostile, unknown
    classification: str = "unknown"

    model_config = {"frozen": False}

    @property
    def duration_s(self) -> float:
        if self.first_seen and self.last_seen:
            return max(0.0, self.last_seen - self.first_seen)
        return 0.0


class SensorCoverage(BaseModel):
    """Summary of a sensor's contribution to the reconstruction."""
    sensor_id: str = ""
    sensor_type: str = ""
    observation_count: int = 0
    targets_observed: list[str] = Field(default_factory=list)
    time_range: TimeRange = Field(default_factory=TimeRange)

    model_config = {"frozen": False}


class ForensicReconstruction(BaseModel):
    """Full forensic reconstruction of an area over a time window.

    Produced by POST /api/forensics/reconstruct. Contains everything
    needed to understand what happened: targets present, events that
    occurred, sensor coverage, and the evidence chain linking them.
    """
    reconstruction_id: str = ""
    status: ReconstructionStatus = ReconstructionStatus.PENDING
    time_range: TimeRange = Field(default_factory=TimeRange)
    bounds: GeoBounds = Field(default_factory=GeoBounds)
    targets: list[TargetTimeline] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)  # chronological event list
    evidence_chain: list[EvidenceItem] = Field(default_factory=list)
    sensor_coverage: list[SensorCoverage] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_events: int = 0
    total_targets: int = 0
    error: str = ""

    model_config = {"frozen": False}

    def mark_complete(self) -> None:
        """Mark reconstruction as complete."""
        self.status = ReconstructionStatus.COMPLETE
        self.completed_at = datetime.now(timezone.utc)
        self.total_events = len(self.events)
        self.total_targets = len(self.targets)

    def mark_failed(self, error: str) -> None:
        """Mark reconstruction as failed."""
        self.status = ReconstructionStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc)


class IncidentFinding(BaseModel):
    """A finding specific to an incident report."""
    finding_id: str = ""
    title: str = ""
    description: str = ""
    confidence: float = 0.0
    evidence_refs: list[str] = Field(default_factory=list)
    target_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}


class IncidentRecommendation(BaseModel):
    """Actionable recommendation from incident analysis."""
    recommendation_id: str = ""
    action: str = ""
    priority: int = 3  # 1=critical, 2=high, 3=medium, 4=low
    rationale: str = ""
    assigned_to: str = ""

    model_config = {"frozen": False}


class IncidentClassification(str, Enum):
    """Incident severity classification."""
    ROUTINE = "routine"
    NOTABLE = "notable"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


class IncidentReport(BaseModel):
    """Structured incident report generated from a forensic reconstruction.

    Links back to the reconstruction that produced it and provides
    human-readable findings, recommendations, and classification.
    Extends the concept of IntelligenceReport with forensic context.
    """
    incident_id: str = ""
    title: str = ""
    summary: str = ""
    reconstruction_id: str = ""  # link to ForensicReconstruction
    reconstruction: Optional[ForensicReconstruction] = None
    classification: IncidentClassification = IncidentClassification.ROUTINE
    findings: list[IncidentFinding] = Field(default_factory=list)
    recommendations: list[IncidentRecommendation] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)  # target IDs involved
    sensor_summary: list[SensorCoverage] = Field(default_factory=list)
    timeline_summary: list[dict] = Field(default_factory=list)  # key moments
    created_at: Optional[datetime] = None
    created_by: str = ""  # user or system
    status: str = "draft"  # draft, review, final, archived
    tags: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}

    def mark_final(self) -> None:
        """Finalize the incident report."""
        self.status = "final"

    def add_finding(self, finding: IncidentFinding) -> None:
        self.findings.append(finding)

    def add_recommendation(self, rec: IncidentRecommendation) -> None:
        self.recommendations.append(rec)

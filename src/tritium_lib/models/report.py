# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Intelligence report models for structured investigation outputs.

An IntelligenceReport captures findings, entity references, recommendations,
and classification levels from investigations.  Used by the command center
to generate actionable intelligence summaries from sensor fusion data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ClassificationLevel(str, Enum):
    """Report classification levels."""
    UNCLASSIFIED = "unclassified"
    FOUO = "fouo"  # For Official Use Only
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class ReportStatus(str, Enum):
    """Report lifecycle status."""
    DRAFT = "draft"
    REVIEW = "review"
    FINAL = "final"
    ARCHIVED = "archived"


class ReportFinding(BaseModel):
    """A single finding within an intelligence report."""
    finding_id: str = ""
    title: str = ""
    description: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    evidence_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}


class ReportRecommendation(BaseModel):
    """An actionable recommendation derived from findings."""
    recommendation_id: str = ""
    action: str = ""
    priority: int = 3  # 1=critical, 2=high, 3=medium, 4=low
    rationale: str = ""
    assigned_to: str = ""

    model_config = {"frozen": False}


class IntelligenceReport(BaseModel):
    """A structured intelligence report from an investigation.

    Captures findings, referenced entities, recommendations, and
    classification metadata.  Designed for both human review and
    machine consumption via the REST API.
    """
    report_id: str = ""
    title: str = ""
    summary: str = ""
    entities: list[str] = Field(
        default_factory=list,
        description="Target/entity IDs referenced in this report",
    )
    findings: list[ReportFinding] = Field(default_factory=list)
    recommendations: list[ReportRecommendation] = Field(default_factory=list)
    created_by: str = ""  # user or system that generated the report
    classification_level: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
    status: ReportStatus = ReportStatus.DRAFT
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)
    source_investigation: str = ""  # link to investigation ID if applicable

    model_config = {"frozen": False}

    def mark_final(self) -> None:
        """Transition report to final status."""
        self.status = ReportStatus.FINAL
        self.updated_at = datetime.now(timezone.utc)

    def add_finding(self, finding: ReportFinding) -> None:
        """Append a finding to the report."""
        self.findings.append(finding)
        self.updated_at = datetime.now(timezone.utc)

    def add_recommendation(self, rec: ReportRecommendation) -> None:
        """Append a recommendation to the report."""
        self.recommendations.append(rec)
        self.updated_at = datetime.now(timezone.utc)

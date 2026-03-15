# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Correlation evidence models — structured reasoning for target fusion.

When two targets are correlated (e.g., BLE phone + camera person), we need
to record WHY they were correlated. This module provides structured storage
of the evidence that led to the correlation decision.

Evidence types:
  - spatial_proximity: targets were within X meters of each other
  - temporal_cooccurrence: targets appeared/disappeared at similar times
  - signal_pattern: RSSI or signal characteristics match
  - visual_similarity: ReID embedding cosine similarity
  - dossier_history: historical pattern of co-occurrence
  - handoff_match: target departed one sensor and arrived at another
  - behavioral: movement patterns suggest same entity
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """Types of evidence that support target correlation."""
    SPATIAL_PROXIMITY = "spatial_proximity"
    TEMPORAL_COOCCURRENCE = "temporal_cooccurrence"
    SIGNAL_PATTERN = "signal_pattern"
    VISUAL_SIMILARITY = "visual_similarity"
    DOSSIER_HISTORY = "dossier_history"
    HANDOFF_MATCH = "handoff_match"
    BEHAVIORAL = "behavioral"
    MANUAL = "manual"


class CorrelationEvidence(BaseModel):
    """A single piece of evidence supporting a target correlation.

    Each evidence record captures one reason WHY two targets were
    determined to be the same entity. Multiple evidence records can
    be stacked to build a composite confidence score.

    Attributes:
        pair_id: Deterministic ID for the target pair (sorted hash).
        evidence_type: What kind of evidence this is.
        evidence_data: Structured data specific to the evidence type.
        confidence: How strongly this evidence supports correlation (0-1).
        timestamp: When the evidence was observed.
        evidence_id: Unique ID for this evidence record.
        source: Which subsystem produced this evidence.
        notes: Optional human-readable explanation.
    """
    evidence_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Unique ID for this evidence record",
    )
    pair_id: str = Field(
        ...,
        description="Deterministic ID for the correlated target pair",
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Category of evidence",
    )
    evidence_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data specific to the evidence type",
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="How strongly this evidence supports correlation",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this evidence was observed",
    )
    source: str = Field(
        "",
        description="Subsystem that produced this evidence",
    )
    notes: str = Field(
        "",
        description="Human-readable explanation of the evidence",
    )

    model_config = {"frozen": False}


def make_pair_id(target_a: str, target_b: str) -> str:
    """Create a deterministic pair ID from two target IDs.

    The pair ID is the same regardless of argument order, ensuring
    that evidence for (A, B) and (B, A) is stored under the same key.

    Args:
        target_a: First target ID.
        target_b: Second target ID.

    Returns:
        Deterministic pair ID string.
    """
    sorted_ids = sorted([target_a, target_b])
    return f"{sorted_ids[0]}::{sorted_ids[1]}"


def compute_composite_confidence(
    evidence_list: list[CorrelationEvidence],
) -> float:
    """Compute a composite confidence from multiple evidence records.

    Uses a 1 - product(1 - ci) formula: each independent piece of
    evidence increases confidence, but with diminishing returns.

    Args:
        evidence_list: List of evidence records.

    Returns:
        Composite confidence score between 0.0 and 1.0.
    """
    if not evidence_list:
        return 0.0
    product = 1.0
    for ev in evidence_list:
        product *= (1.0 - ev.confidence)
    return round(1.0 - product, 6)


def build_spatial_evidence(
    target_a: str,
    target_b: str,
    distance_m: float,
    max_distance: float = 10.0,
    source: str = "correlator",
) -> CorrelationEvidence:
    """Create spatial proximity evidence for two co-located targets.

    Confidence is inversely proportional to distance.

    Args:
        target_a: First target ID.
        target_b: Second target ID.
        distance_m: Distance between targets in meters.
        max_distance: Maximum distance to consider relevant.
        source: Subsystem name.

    Returns:
        CorrelationEvidence with spatial proximity data.
    """
    confidence = max(0.0, 1.0 - (distance_m / max_distance))
    return CorrelationEvidence(
        pair_id=make_pair_id(target_a, target_b),
        evidence_type=EvidenceType.SPATIAL_PROXIMITY,
        evidence_data={
            "target_a": target_a,
            "target_b": target_b,
            "distance_m": round(distance_m, 2),
            "max_distance": max_distance,
        },
        confidence=round(confidence, 4),
        source=source,
        notes=f"Targets {distance_m:.1f}m apart (max {max_distance}m threshold)",
    )


def build_visual_evidence(
    target_a: str,
    target_b: str,
    similarity: float,
    camera_a: str = "",
    camera_b: str = "",
    source: str = "reid",
) -> CorrelationEvidence:
    """Create visual similarity evidence from ReID embedding match.

    Args:
        target_a: First target ID.
        target_b: Second target ID.
        similarity: Cosine similarity between embeddings (0-1).
        camera_a: Camera that saw target_a.
        camera_b: Camera that saw target_b.
        source: Subsystem name.

    Returns:
        CorrelationEvidence with visual similarity data.
    """
    return CorrelationEvidence(
        pair_id=make_pair_id(target_a, target_b),
        evidence_type=EvidenceType.VISUAL_SIMILARITY,
        evidence_data={
            "target_a": target_a,
            "target_b": target_b,
            "similarity": round(similarity, 4),
            "camera_a": camera_a,
            "camera_b": camera_b,
        },
        confidence=round(similarity, 4),
        source=source,
        notes=f"ReID cosine similarity {similarity:.3f} between cameras {camera_a} and {camera_b}",
    )


def build_handoff_evidence(
    target_a: str,
    target_b: str,
    from_sensor: str,
    to_sensor: str,
    gap_seconds: float,
    source: str = "handoff_tracker",
) -> CorrelationEvidence:
    """Create handoff evidence when a target transitions between sensors.

    Args:
        target_a: Target ID at departure sensor.
        target_b: Target ID at arrival sensor.
        from_sensor: Sensor where target departed.
        to_sensor: Sensor where target arrived.
        gap_seconds: Time gap between departure and arrival.
        source: Subsystem name.

    Returns:
        CorrelationEvidence with handoff match data.
    """
    # Confidence decreases with longer gaps
    confidence = max(0.1, 1.0 - (gap_seconds / 120.0))
    return CorrelationEvidence(
        pair_id=make_pair_id(target_a, target_b),
        evidence_type=EvidenceType.HANDOFF_MATCH,
        evidence_data={
            "target_a": target_a,
            "target_b": target_b,
            "from_sensor": from_sensor,
            "to_sensor": to_sensor,
            "gap_seconds": round(gap_seconds, 2),
        },
        confidence=round(confidence, 4),
        source=source,
        notes=f"Target handed off {from_sensor} -> {to_sensor} with {gap_seconds:.1f}s gap",
    )

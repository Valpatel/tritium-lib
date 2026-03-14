# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ML training data models for self-improving correlation and classification.

These models define the structure for collecting training examples from
the live system. Every correlation decision, classification decision,
and operator feedback is captured as a TrainingExample for future
model training and reinforcement learning.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Type of decision being recorded for training."""
    CORRELATION = "correlation"
    CLASSIFICATION = "classification"
    THREAT_ASSESSMENT = "threat_assessment"
    ALLIANCE_OVERRIDE = "alliance_override"


class TrainingExample(BaseModel):
    """A single training example from the live system.

    Captures the features, label, and metadata for one decision.
    Can be used for supervised learning, reinforcement learning,
    or active learning pipelines.
    """
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Input features for this example (sensor data, context)",
    )
    label: str = Field(
        "",
        description="Ground-truth label (may be set later by operator feedback)",
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="System confidence in its prediction (0=none, 1=certain)",
    )
    source: str = Field(
        "",
        description="Which subsystem generated this example (correlator, classifier, etc.)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this example was recorded",
    )
    confirmed_by: Optional[str] = Field(
        None,
        description="Operator who confirmed/rejected this label (None if unconfirmed)",
    )


class CorrelationTrainingData(BaseModel):
    """Training data for the target correlation model.

    Each example records a pair of targets, the features used for
    correlation scoring, the system's decision, and whether an
    operator later confirmed or rejected the decision.
    """
    target_a_id: str = Field(..., description="First target in the correlation pair")
    target_b_id: str = Field(..., description="Second target in the correlation pair")
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Correlation features: proximity, timing, co-occurrence, etc.",
    )
    score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Correlation score produced by the system",
    )
    decision: str = Field(
        "unknown",
        description="System decision: merge, related, unrelated, unknown",
    )
    outcome: Optional[str] = Field(
        None,
        description="Confirmed outcome: correct, incorrect, uncertain (set by operator)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    source: str = Field("correlator", description="Subsystem that produced this data")


class ClassificationTrainingData(BaseModel):
    """Training data for the device classification model.

    Each example records the device features observed, the system's
    predicted type and confidence, and operator corrections.
    """
    target_id: str = Field(..., description="Target being classified")
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Device features: RSSI, OUI, service UUIDs, appearance, name pattern",
    )
    predicted_type: str = Field(
        "unknown",
        description="System's predicted device type",
    )
    predicted_alliance: str = Field(
        "unknown",
        description="System's predicted alliance",
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="System confidence in the classification",
    )
    correct_type: Optional[str] = Field(
        None,
        description="Operator-corrected device type (None if unconfirmed)",
    )
    correct_alliance: Optional[str] = Field(
        None,
        description="Operator-corrected alliance (None if unconfirmed)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    source: str = Field("classifier", description="Subsystem that produced this data")


class FeedbackRecord(BaseModel):
    """Operator feedback on a system decision.

    Operators confirm or reject correlation, classification, and threat
    assessment decisions. This feedback is stored as training data for
    reinforcement learning.
    """
    target_id: str = Field(..., description="Target the feedback applies to")
    decision_type: DecisionType = Field(
        ..., description="Type of decision being evaluated",
    )
    correct: bool = Field(
        ..., description="Whether the system's decision was correct",
    )
    notes: str = Field(
        "",
        description="Operator notes explaining the feedback",
    )
    operator: str = Field(
        "",
        description="Operator who provided the feedback",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

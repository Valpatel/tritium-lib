# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ML training data models.

Wave 52 — validates TrainingExample, CorrelationTrainingData,
ClassificationTrainingData, and FeedbackRecord models.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tritium_lib.models.training import (
    ClassificationTrainingData,
    CorrelationTrainingData,
    DecisionType,
    FeedbackRecord,
    TrainingExample,
)


class TestTrainingExample:
    """Tests for TrainingExample model."""

    def test_default_values(self):
        """TrainingExample should have sensible defaults."""
        ex = TrainingExample()
        assert ex.features == {}
        assert ex.label == ""
        assert ex.confidence == 0.0
        assert ex.source == ""
        assert ex.confirmed_by is None
        assert isinstance(ex.timestamp, datetime)

    def test_full_construction(self):
        """TrainingExample with all fields set."""
        ex = TrainingExample(
            features={"rssi": -45, "oui": "Apple"},
            label="phone",
            confidence=0.92,
            source="classifier",
            confirmed_by="op1",
        )
        assert ex.features["rssi"] == -45
        assert ex.label == "phone"
        assert ex.confidence == 0.92
        assert ex.confirmed_by == "op1"

    def test_confidence_bounds(self):
        """Confidence must be between 0 and 1."""
        with pytest.raises(Exception):
            TrainingExample(confidence=1.5)
        with pytest.raises(Exception):
            TrainingExample(confidence=-0.1)

    def test_serialization(self):
        """TrainingExample should serialize to dict."""
        ex = TrainingExample(
            features={"x": 1},
            label="test",
            confidence=0.5,
        )
        d = ex.model_dump()
        assert d["features"] == {"x": 1}
        assert d["label"] == "test"


class TestCorrelationTrainingData:
    """Tests for CorrelationTrainingData model."""

    def test_required_fields(self):
        """target_a_id and target_b_id are required."""
        data = CorrelationTrainingData(
            target_a_id="ble_aa:bb:cc:dd:ee:ff",
            target_b_id="det_person_1",
        )
        assert data.target_a_id == "ble_aa:bb:cc:dd:ee:ff"
        assert data.target_b_id == "det_person_1"
        assert data.score == 0.0
        assert data.decision == "unknown"
        assert data.outcome is None

    def test_full_construction(self):
        """Full correlation training data."""
        data = CorrelationTrainingData(
            target_a_id="t1",
            target_b_id="t2",
            features={"proximity": 0.9, "timing": 0.8},
            score=0.85,
            decision="merge",
            outcome="correct",
            source="correlator",
        )
        assert data.score == 0.85
        assert data.decision == "merge"
        assert data.outcome == "correct"
        assert data.features["proximity"] == 0.9


class TestClassificationTrainingData:
    """Tests for ClassificationTrainingData model."""

    def test_required_fields(self):
        """target_id is required."""
        data = ClassificationTrainingData(target_id="ble_xx")
        assert data.target_id == "ble_xx"
        assert data.predicted_type == "unknown"
        assert data.predicted_alliance == "unknown"
        assert data.correct_type is None

    def test_with_corrections(self):
        """Classification with operator corrections."""
        data = ClassificationTrainingData(
            target_id="ble_xx",
            features={"rssi": -45, "name": "Watch"},
            predicted_type="phone",
            confidence=0.4,
            correct_type="watch",
            correct_alliance="friendly",
        )
        assert data.correct_type == "watch"
        assert data.correct_alliance == "friendly"


class TestFeedbackRecord:
    """Tests for FeedbackRecord model."""

    def test_required_fields(self):
        """target_id, decision_type, and correct are required."""
        fb = FeedbackRecord(
            target_id="ble_xx",
            decision_type=DecisionType.CLASSIFICATION,
            correct=True,
        )
        assert fb.target_id == "ble_xx"
        assert fb.decision_type == DecisionType.CLASSIFICATION
        assert fb.correct is True
        assert fb.notes == ""
        assert fb.operator == ""

    def test_rejection_feedback(self):
        """Operator rejection record."""
        fb = FeedbackRecord(
            target_id="mesh_1",
            decision_type=DecisionType.THREAT_ASSESSMENT,
            correct=False,
            notes="False positive, this is a known sensor",
            operator="supervisor",
        )
        assert fb.correct is False
        assert "False positive" in fb.notes

    def test_decision_type_enum(self):
        """DecisionType enum values."""
        assert DecisionType.CORRELATION == "correlation"
        assert DecisionType.CLASSIFICATION == "classification"
        assert DecisionType.THREAT_ASSESSMENT == "threat_assessment"
        assert DecisionType.ALLIANCE_OVERRIDE == "alliance_override"

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for PatternLearner — behavioral pattern threat prediction."""

import tempfile
from pathlib import Path

import pytest

from tritium_lib.intelligence.pattern_learning import (
    PATTERN_FEATURES,
    PatternLearner,
    PredictionResult,
    TrainingExample,
)


# ---------------------------------------------------------------------------
# Identity and basics
# ---------------------------------------------------------------------------

class TestPatternLearnerBasics:
    def test_name(self):
        learner = PatternLearner()
        assert learner.name == "pattern_learner"

    def test_not_trained_initially(self):
        learner = PatternLearner()
        assert not learner.is_trained
        assert learner.accuracy == 0.0
        assert learner.training_count == 0

    def test_add_training_example(self):
        learner = PatternLearner()
        learner.add_training_example(
            features={"time_of_day_hour": 2, "device_count": 5},
            outcome="threat",
        )
        assert learner.training_examples == 1

    def test_clear_training_data(self):
        learner = PatternLearner()
        learner.add_training_example({"x": 1.0}, outcome="threat")
        learner.add_training_example({"x": 2.0}, outcome="benign")
        assert learner.training_examples == 2
        learner.clear_training_data()
        assert learner.training_examples == 0

    def test_pattern_features_defined(self):
        assert len(PATTERN_FEATURES) > 5
        assert "time_of_day_hour" in PATTERN_FEATURES
        assert "device_count" in PATTERN_FEATURES


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestPatternLearnerTraining:
    def test_train_needs_two_examples(self):
        learner = PatternLearner()
        learner.add_training_example({"x": 1.0}, "threat")
        result = learner.train()
        assert not result["success"]

    def test_train_needs_both_classes(self):
        learner = PatternLearner()
        learner.add_training_example({"x": 1.0}, "threat")
        learner.add_training_example({"x": 2.0}, "threat")
        result = learner.train()
        assert not result["success"]
        assert "both" in result["error"].lower() or "benign" in result["error"].lower()

    def test_train_success(self):
        learner = PatternLearner()
        # Threats: high device count, late night
        for i in range(10):
            learner.add_training_example(
                {"time_of_day_hour": 2.0, "device_count": 8.0, "new_device_ratio": 0.9},
                outcome="threat",
            )
        # Benign: low device count, daytime
        for i in range(10):
            learner.add_training_example(
                {"time_of_day_hour": 14.0, "device_count": 2.0, "new_device_ratio": 0.1},
                outcome="benign",
            )

        result = learner.train()
        assert result["success"]
        assert result["accuracy"] > 0.0
        assert result["training_count"] == 20
        assert result["threat_count"] == 10
        assert result["benign_count"] == 10
        assert learner.is_trained

    def test_train_accuracy_reasonable(self):
        """With clearly separable data, accuracy should be high."""
        learner = PatternLearner()
        for _ in range(20):
            learner.add_training_example({"x": 10.0}, "threat")
            learner.add_training_example({"x": -10.0}, "benign")

        result = learner.train()
        assert result["success"]
        # With such clear separation, accuracy should be very high
        assert result["accuracy"] >= 0.8


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPatternLearnerPrediction:
    def _trained_learner(self) -> PatternLearner:
        """Create a learner trained on separable data."""
        learner = PatternLearner()
        for _ in range(15):
            learner.add_training_example({"x": 5.0, "y": 3.0}, "threat")
            learner.add_training_example({"x": -5.0, "y": -3.0}, "benign")
        learner.train()
        return learner

    def test_predict_untrained(self):
        learner = PatternLearner()
        result = learner.predict({"x": 1.0})
        assert isinstance(result, PredictionResult)
        assert result.confidence == 0.0
        assert result.recommendation == "monitor"

    def test_predict_returns_result(self):
        learner = self._trained_learner()
        result = learner.predict({"x": 5.0, "y": 3.0})
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.threat_probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.recommendation in ("monitor", "investigate", "alert")

    def test_predict_threat_like_pattern(self):
        learner = self._trained_learner()
        result = learner.predict({"x": 5.0, "y": 3.0})
        assert result.threat_probability > 0.4

    def test_predict_benign_like_pattern(self):
        learner = self._trained_learner()
        result = learner.predict({"x": -5.0, "y": -3.0})
        assert result.threat_probability < 0.6

    def test_predict_invalid_input(self):
        learner = self._trained_learner()
        result = learner.predict("not a dict")
        assert isinstance(result, PredictionResult)
        assert result.confidence == 0.0

    def test_contributing_features(self):
        learner = self._trained_learner()
        result = learner.predict({"x": 5.0, "y": 3.0})
        assert isinstance(result.contributing_features, dict)

    def test_recommendation_thresholds(self):
        learner = PatternLearner()
        # All threat
        for _ in range(20):
            learner.add_training_example({"x": 10.0}, "threat")
            learner.add_training_example({"x": -10.0}, "benign")
        learner.train()

        # Very threat-like should get "alert"
        result = learner.predict({"x": 10.0})
        assert result.recommendation in ("alert", "investigate")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestPatternLearnerStats:
    def test_get_stats_untrained(self):
        learner = PatternLearner()
        stats = learner.get_stats()
        assert stats["name"] == "pattern_learner"
        assert stats["trained"] is False
        assert stats["training_examples"] == 0
        assert "threat_prior" in stats

    def test_get_stats_trained(self):
        learner = PatternLearner()
        learner.add_training_example({"x": 1.0}, "threat")
        learner.add_training_example({"x": -1.0}, "benign")
        learner.train()

        stats = learner.get_stats()
        assert stats["trained"] is True
        assert stats["training_examples"] == 2
        assert stats["feature_count"] >= 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPatternLearnerPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pkl")
            learner = PatternLearner(model_path=path)

            learner.add_training_example({"x": 5.0}, "threat")
            learner.add_training_example({"x": -5.0}, "benign")
            learner.train()

            assert learner.save()

            # Load into new instance
            learner2 = PatternLearner(model_path=path)
            assert learner2.load()
            assert learner2.is_trained
            assert learner2.accuracy == learner.accuracy

            # Predictions should match
            r1 = learner.predict({"x": 5.0})
            r2 = learner2.predict({"x": 5.0})
            assert abs(r1.threat_probability - r2.threat_probability) < 0.01

    def test_save_no_path(self):
        learner = PatternLearner()
        assert not learner.save()

    def test_load_nonexistent(self):
        learner = PatternLearner(model_path="/nonexistent/path.pkl")
        assert not learner.load()

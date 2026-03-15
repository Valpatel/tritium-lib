# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for RLMetrics — RL model health monitoring."""

import time

import pytest

from tritium_lib.intelligence.rl_metrics import RLMetrics, TrainingSnapshot


class TestRLMetrics:
    """RLMetrics unit tests."""

    def test_init(self):
        m = RLMetrics()
        status = m.get_status()
        assert status["total_trainings"] == 0
        assert status["total_predictions"] == 0
        assert status["overall_accuracy"] == 0.0

    def test_record_training(self):
        m = RLMetrics()
        m.record_training(
            accuracy=0.78,
            training_count=200,
            feature_importance={"distance": 0.25, "rssi_delta": 0.15},
            model_name="correlation",
        )
        status = m.get_status()
        assert status["total_trainings"] == 1
        assert status["latest_training"]["accuracy"] == 0.78
        assert status["latest_training"]["training_count"] == 200

    def test_accuracy_trend(self):
        m = RLMetrics()
        for i in range(5):
            m.record_training(
                accuracy=0.5 + i * 0.1,
                training_count=100 + i * 50,
                model_name="correlation",
            )
        trend = m.get_accuracy_trend("correlation")
        assert len(trend) == 5
        assert trend[0]["accuracy"] == 0.5
        assert trend[-1]["accuracy"] == 0.9

    def test_record_prediction(self):
        m = RLMetrics()
        m.record_training(accuracy=0.8, training_count=100, model_name="correlation")
        m.record_prediction(predicted_class=1, probability=0.85, correct=True)
        m.record_prediction(predicted_class=0, probability=0.3, correct=False)

        status = m.get_status()
        assert status["total_predictions"] == 2
        assert status["total_correct"] == 1
        assert status["total_incorrect"] == 1
        assert status["overall_accuracy"] == 0.5

    def test_prediction_distribution(self):
        m = RLMetrics()
        for i in range(10):
            m.record_prediction(
                predicted_class=1 if i >= 5 else 0,
                probability=i / 10.0,
            )
        dist = m.get_prediction_distribution()
        assert dist["total"] == 10
        assert dist["class_counts"][0] == 5
        assert dist["class_counts"][1] == 5
        assert sum(dist["probability_histogram"]) == 10

    def test_feature_importance(self):
        m = RLMetrics()
        m.record_training(
            accuracy=0.8,
            training_count=100,
            feature_importance={"distance": -0.3, "rssi_delta": 0.1, "co_movement": 0.2},
            model_name="correlation",
        )
        fi = m.get_feature_importance()
        assert "distance" in fi
        # Should be sorted by absolute importance
        keys = list(fi.keys())
        assert keys[0] == "distance"  # highest absolute value

    def test_training_data_growth(self):
        m = RLMetrics()
        for i in range(3):
            m.record_training(
                accuracy=0.7,
                training_count=50 * (i + 1),
                model_name="correlation",
            )
        growth = m.get_training_data_growth("correlation")
        assert len(growth) == 3
        assert growth[0]["training_count"] == 50
        assert growth[2]["training_count"] == 150

    def test_per_model_tracking(self):
        m = RLMetrics()
        m.record_training(accuracy=0.8, training_count=100, model_name="correlation")
        m.record_training(accuracy=0.9, training_count=200, model_name="classifier")
        m.record_prediction(predicted_class=1, probability=0.9, correct=True, model_name="correlation")

        status = m.get_status()
        assert "correlation" in status["models"]
        assert "classifier" in status["models"]
        assert status["models"]["correlation"]["total_predictions"] == 1
        assert status["models"]["classifier"]["total_predictions"] == 0

    def test_reset(self):
        m = RLMetrics()
        m.record_training(accuracy=0.8, training_count=100)
        m.record_prediction(predicted_class=1, probability=0.8)
        m.reset()
        status = m.get_status()
        assert status["total_trainings"] == 0
        assert status["total_predictions"] == 0

    def test_feedback(self):
        m = RLMetrics()
        m.record_training(accuracy=0.8, training_count=100, model_name="correlation")
        m.record_feedback(correct=True, model_name="correlation")
        m.record_feedback(correct=False, model_name="correlation")

        status = m.get_status()
        assert status["total_correct"] == 1
        assert status["total_incorrect"] == 1
        assert status["models"]["correlation"]["correct_predictions"] == 1
        assert status["models"]["correlation"]["incorrect_predictions"] == 1

    def test_export_empty(self):
        m = RLMetrics()
        export = m.export()
        assert "status" in export
        assert "models_detail" in export
        assert export["total_trainings"] == 0
        assert export["total_predictions"] == 0
        assert "export_timestamp" in export
        assert isinstance(export["export_timestamp"], float)

    def test_export_with_data(self):
        m = RLMetrics()
        m.record_training(
            accuracy=0.85,
            training_count=300,
            feature_importance={"distance": 0.3, "rssi": 0.2},
            model_name="correlation",
            duration_s=1.5,
        )
        m.record_training(
            accuracy=0.90,
            training_count=500,
            model_name="classifier",
        )
        m.record_prediction(predicted_class=1, probability=0.9, correct=True, model_name="correlation")
        m.record_prediction(predicted_class=0, probability=0.3, correct=False, model_name="correlation")

        export = m.export()
        assert export["total_trainings"] == 2
        assert export["total_predictions"] == 2
        assert export["total_correct"] == 1
        assert export["total_incorrect"] == 1
        assert export["training_history_size"] == 2
        assert export["prediction_history_size"] == 2

        # Models detail includes trend and feature importance
        assert "correlation" in export["models_detail"]
        corr = export["models_detail"]["correlation"]
        assert "accuracy_trend" in corr
        assert "training_growth" in corr
        assert "feature_importance" in corr
        assert len(corr["accuracy_trend"]) >= 1
        assert corr["feature_importance"].get("distance") is not None

        assert "classifier" in export["models_detail"]

    def test_export_serializable(self):
        """Export output must be JSON-serializable."""
        import json
        m = RLMetrics()
        m.record_training(accuracy=0.75, training_count=100)
        m.record_prediction(predicted_class=1, probability=0.8, correct=True)
        export = m.export()
        # Should not raise
        json_str = json.dumps(export)
        assert len(json_str) > 10
        parsed = json.loads(json_str)
        assert parsed["total_trainings"] == 1

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for RLMetrics — RL model health monitoring."""

import time

import pytest

from tritium_lib.intelligence.rl_metrics import (
    FeatureAblation,
    RLMetrics,
    TrainingSnapshot,
)


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


class TestFeatureAblation:
    """B-10 instrumentation: per-feature ablation diagnostics."""

    def test_record_feature_stats_basic(self):
        m = RLMetrics()
        rows = [
            [0.1, 1.0, 0.0],
            [0.2, 1.0, 0.0],
            [0.4, 1.0, 0.0],
        ]
        snap = m.record_feature_stats(
            feature_names=["distance", "constant_one", "constant_zero"],
            rows=rows,
            model_name="correlation",
        )
        assert "distance" in snap
        # Distance has variance
        assert not snap["distance"].is_constant
        # Constant_one and constant_zero have zero std
        assert snap["constant_one"].is_constant
        assert snap["constant_zero"].is_constant
        assert snap["constant_one"].mean == pytest.approx(1.0)
        assert snap["constant_zero"].mean == pytest.approx(0.0)

    def test_get_constant_features(self):
        m = RLMetrics()
        rows = [
            [0.1, 1.0, 0.0, 0.5],
            [0.5, 1.0, 0.0, 0.5],
            [0.9, 1.0, 0.0, 0.5],
        ]
        m.record_feature_stats(
            feature_names=["distance", "constant_one", "constant_zero", "stuck"],
            rows=rows,
        )
        stuck = m.get_constant_features()
        assert "constant_one" in stuck
        assert "constant_zero" in stuck
        assert "stuck" in stuck
        assert "distance" not in stuck

    def test_get_constant_features_saturation(self):
        m = RLMetrics()
        # 19/20 rows are 0.0, only one is 1.0 -> saturation 0.95
        rows = [[0.0]] * 19 + [[1.0]]
        m.record_feature_stats(
            feature_names=["mostly_zero"],
            rows=rows,
        )
        stuck = m.get_constant_features(saturation_threshold=0.95)
        assert "mostly_zero" in stuck

    def test_export_includes_ablation(self):
        m = RLMetrics()
        m.record_training(
            accuracy=0.85,
            training_count=10,
            feature_importance={"distance": 0.4, "constant_one": 0.0},
            model_name="correlation",
        )
        m.record_feature_stats(
            feature_names=["distance", "constant_one"],
            rows=[[0.1, 1.0], [0.5, 1.0], [0.9, 1.0]],
            model_name="correlation",
            importance={"distance": 0.4, "constant_one": 0.0},
        )

        export = m.export()
        corr = export["models_detail"]["correlation"]
        assert "feature_ablation" in corr
        assert "constant_features" in corr
        # Feature ablation entries have full diagnostic shape
        ablation = corr["feature_ablation"]
        assert isinstance(ablation, list)
        assert len(ablation) == 2
        sample = ablation[0]
        for key in [
            "feature_name", "mean", "std", "min", "max",
            "unique_values", "is_constant", "saturation_ratio",
            "sample_count", "importance",
        ]:
            assert key in sample
        # constant_one should be flagged
        assert "constant_one" in corr["constant_features"]
        assert "distance" not in corr["constant_features"]

    def test_record_feature_stats_handles_empty(self):
        m = RLMetrics()
        snap = m.record_feature_stats(
            feature_names=["a", "b"],
            rows=[],
        )
        assert snap["a"].sample_count == 0
        assert snap["a"].is_constant
        assert snap["b"].is_constant

    def test_feature_ablation_to_dict(self):
        ab = FeatureAblation(
            feature_name="distance",
            mean=0.5,
            std=0.2,
            minimum=0.1,
            maximum=0.9,
            unique_values=10,
            is_constant=False,
            saturation_ratio=0.1,
            sample_count=100,
            importance=0.4,
        )
        d = ab.to_dict()
        assert d["feature_name"] == "distance"
        assert d["mean"] == 0.5
        assert d["is_constant"] is False
        assert d["importance"] == 0.4

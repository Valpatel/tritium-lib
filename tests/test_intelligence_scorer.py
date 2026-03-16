# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.intelligence.scorer module."""

import math
import tempfile

import pytest

from tritium_lib.intelligence.scorer import (
    FEATURE_NAMES,
    CorrelationFeatures,
    CorrelationScorer,
    LearnedScorer,
    ScorerResult,
    StaticScorer,
    _sigmoid,
)


class _PicklableModel:
    """Top-level class so pickle can serialize it."""
    def predict_proba(self, X):
        return [[0.3, 0.7]]


class TestSigmoid:
    def test_zero(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9

    def test_positive(self):
        assert _sigmoid(10.0) > 0.999

    def test_negative(self):
        assert _sigmoid(-10.0) < 0.001

    def test_symmetry(self):
        assert abs(_sigmoid(3.0) + _sigmoid(-3.0) - 1.0) < 1e-9


class TestStaticScorer:
    def test_name(self):
        s = StaticScorer()
        assert s.name == "static"

    def test_is_not_trained(self):
        s = StaticScorer()
        assert not s.is_trained

    def test_zero_features(self):
        s = StaticScorer()
        result = s.predict({})
        assert isinstance(result, ScorerResult)
        assert 0.0 <= result.probability <= 1.0
        assert result.method == "static"

    def test_high_distance_lowers_score(self):
        s = StaticScorer()
        close = s.predict({"distance": 0.1})
        far = s.predict({"distance": 100.0})
        assert close.probability > far.probability

    def test_co_movement_raises_score(self):
        s = StaticScorer()
        no_move = s.predict({"co_movement": 0.0})
        co_move = s.predict({"co_movement": 1.0})
        assert co_move.probability > no_move.probability

    def test_custom_weights(self):
        s = StaticScorer(weights={"distance": -1.0}, bias=0.5)
        result = s.predict({"distance": 5.0})
        assert result.probability < 0.5

    def test_result_has_detail(self):
        s = StaticScorer()
        result = s.predict({"distance": 2.0, "co_movement": 0.8})
        assert "logit=" in result.detail


class TestLearnedScorer:
    def test_name(self):
        s = LearnedScorer()
        assert s.name == "learned"

    def test_no_model_falls_back(self):
        s = LearnedScorer(model=None)
        assert not s.is_trained
        result = s.predict({"distance": 1.0})
        assert "fallback" in result.detail

    def test_with_mock_model(self):
        """Test with a mock sklearn-like model."""
        class MockModel:
            def predict_proba(self, X):
                return [[0.2, 0.8]]

        s = LearnedScorer(model=MockModel(), accuracy=0.95, training_count=100)
        assert s.is_trained
        assert s.accuracy == 0.95
        assert s.training_count == 100

        result = s.predict({"distance": 1.0})
        assert result.method == "learned"
        assert abs(result.probability - 0.8) < 1e-6

    def test_model_error_falls_back(self):
        class BadModel:
            def predict_proba(self, X):
                raise RuntimeError("model broken")

        s = LearnedScorer(model=BadModel())
        result = s.predict({"distance": 1.0})
        assert "model error" in result.detail

    def test_save_and_load(self):
        # Use a simple object that pickle can handle (top-level class)
        s = LearnedScorer(model=_PicklableModel(), accuracy=0.9, training_count=50)

        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/model.pkl"
            assert s.save(path)

            loaded = LearnedScorer.from_file(path)
            assert loaded is not None
            assert loaded.accuracy == 0.9
            assert loaded.training_count == 50

    def test_load_nonexistent(self):
        result = LearnedScorer.from_file("/tmp/nonexistent_model_abc123.pkl")
        assert result is None


class TestAbstractBase:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            CorrelationScorer()


class TestFeatureNames:
    def test_canonical_names(self):
        assert "distance" in FEATURE_NAMES
        assert "rssi_delta" in FEATURE_NAMES
        assert "co_movement" in FEATURE_NAMES
        assert "device_type_match" in FEATURE_NAMES
        assert "time_gap" in FEATURE_NAMES
        assert "signal_pattern" in FEATURE_NAMES

    def test_extended_names_wave126(self):
        """Wave 126: verify the 4 new features are in FEATURE_NAMES."""
        assert "co_movement_duration" in FEATURE_NAMES
        assert "time_of_day_similarity" in FEATURE_NAMES
        assert "source_diversity_score" in FEATURE_NAMES
        assert "wifi_probe_correlation" in FEATURE_NAMES
        assert len(FEATURE_NAMES) == 10

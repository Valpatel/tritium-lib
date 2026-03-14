# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for BaseLearner ABC."""
import tempfile
import os
from typing import Any

import pytest

from tritium_lib.intelligence.base_learner import BaseLearner


class DummyLearner(BaseLearner):
    """Concrete test implementation of BaseLearner."""

    def __init__(self, model_path: str = "") -> None:
        super().__init__(model_path)
        self._train_called = False

    @property
    def name(self) -> str:
        return "dummy"

    def train(self) -> dict[str, Any]:
        self._model = {"weights": [1, 2, 3]}
        self._accuracy = 0.85
        self._training_count = 100
        import time
        self._last_trained = time.time()
        self._train_called = True
        return {"success": True, "accuracy": 0.85, "training_count": 100}

    def predict(self, features: Any) -> Any:
        if self._model is None:
            return None
        return {"prediction": "test", "confidence": 0.9}


class TestBaseLearnerInterface:
    def test_initial_state(self):
        learner = DummyLearner()
        assert learner.name == "dummy"
        assert not learner.is_trained
        assert learner.accuracy == 0.0
        assert learner.training_count == 0
        assert learner.last_trained == 0.0

    def test_train_updates_state(self):
        learner = DummyLearner()
        result = learner.train()
        assert result["success"] is True
        assert learner.is_trained
        assert learner.accuracy == 0.85
        assert learner.training_count == 100
        assert learner.last_trained > 0.0

    def test_predict_before_training(self):
        learner = DummyLearner()
        assert learner.predict({}) is None

    def test_predict_after_training(self):
        learner = DummyLearner()
        learner.train()
        result = learner.predict({"feature": 1.0})
        assert result is not None
        assert result["prediction"] == "test"

    def test_get_stats(self):
        learner = DummyLearner()
        learner.train()
        stats = learner.get_stats()
        assert stats["name"] == "dummy"
        assert stats["trained"] is True
        assert stats["accuracy"] == 0.85
        assert stats["training_count"] == 100
        assert stats["last_trained_iso"] is not None


class TestBaseLearnerPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")

            learner1 = DummyLearner(model_path=model_path)
            learner1.train()
            assert learner1.save() is True

            learner2 = DummyLearner(model_path=model_path)
            assert learner2.load() is True
            assert learner2.is_trained
            assert learner2.accuracy == 0.85
            assert learner2.training_count == 100

    def test_load_nonexistent(self):
        learner = DummyLearner(model_path="/tmp/nonexistent_model_xyz.pkl")
        assert learner.load() is False

    def test_save_without_path(self):
        learner = DummyLearner(model_path="")
        learner.train()
        assert learner.save() is False

    def test_load_without_path(self):
        learner = DummyLearner(model_path="")
        assert learner.load() is False


class TestBaseLearnerSerialization:
    def test_serialize_default(self):
        learner = DummyLearner()
        learner.train()
        data = learner._serialize()
        assert "model" in data
        assert "accuracy" in data
        assert "training_count" in data
        assert "last_trained" in data

    def test_deserialize_default(self):
        learner = DummyLearner()
        learner._deserialize({
            "model": {"weights": [4, 5, 6]},
            "accuracy": 0.92,
            "training_count": 200,
            "last_trained": 1000.0,
        })
        assert learner.is_trained
        assert learner.accuracy == 0.92
        assert learner.training_count == 200
        assert learner.last_trained == 1000.0

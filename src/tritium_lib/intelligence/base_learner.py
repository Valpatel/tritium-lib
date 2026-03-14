# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BaseLearner ABC — common interface for all Tritium ML learners.

Extracts the shared pattern from CorrelationLearner and
BLEClassificationLearner: train, predict, save, load, get_stats.
Concrete learners override the abstract methods for their specific
model type while inheriting pickle-based persistence and status tracking.

Usage::

    class MyLearner(BaseLearner):
        @property
        def name(self) -> str:
            return "my_learner"

        def train(self) -> dict[str, Any]:
            ...

        def predict(self, features: Any) -> Any:
            ...
"""
from __future__ import annotations

import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("base_learner")


class BaseLearner(ABC):
    """Abstract base class for ML learners in the Tritium intelligence stack.

    Provides common persistence (pickle), status tracking (accuracy,
    training count, timestamps), and a consistent interface for the
    intelligence router to enumerate and manage all learners.
    """

    def __init__(self, model_path: str = "") -> None:
        self._model_path = model_path
        self._model: Any = None
        self._accuracy: float = 0.0
        self._training_count: int = 0
        self._last_trained: float = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this learner (e.g. 'correlation', 'ble_classifier')."""

    @property
    def is_trained(self) -> bool:
        """Whether a trained model is loaded."""
        return self._model is not None

    @property
    def accuracy(self) -> float:
        """Last measured model accuracy (0.0 to 1.0)."""
        return self._accuracy

    @property
    def training_count(self) -> int:
        """Number of examples used in the last training run."""
        return self._training_count

    @property
    def last_trained(self) -> float:
        """Unix timestamp of last training run (0.0 if never trained)."""
        return self._last_trained

    @property
    def model_path(self) -> str:
        """Path where the model is persisted."""
        return self._model_path

    @abstractmethod
    def train(self) -> dict[str, Any]:
        """Train (or retrain) the model.

        Returns:
            Dict with at least 'success' (bool) and optionally 'accuracy',
            'training_count', 'error', and learner-specific fields.
        """

    @abstractmethod
    def predict(self, features: Any) -> Any:
        """Run inference on the trained model.

        Args:
            features: Input features (type depends on learner).

        Returns:
            Prediction result (type depends on learner).
        """

    def get_stats(self) -> dict[str, Any]:
        """Return learner status for API/dashboard consumption.

        Override in subclasses to add learner-specific fields.
        """
        return {
            "name": self.name,
            "trained": self.is_trained,
            "accuracy": self._accuracy,
            "training_count": self._training_count,
            "last_trained": self._last_trained,
            "last_trained_iso": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_trained))
                if self._last_trained > 0
                else None
            ),
            "model_path": self._model_path,
        }

    def save(self) -> bool:
        """Save the current model to disk using pickle.

        Override in subclasses that need custom serialization.

        Returns:
            True on success, False on failure.
        """
        if not self._model_path:
            logger.warning("%s: no model_path configured, skipping save", self.name)
            return False

        try:
            p = Path(self._model_path)
            p.parent.mkdir(parents=True, exist_ok=True)

            data = self._serialize()
            with open(p, "wb") as f:
                pickle.dump(data, f)

            logger.info("%s: model saved to %s", self.name, self._model_path)
            return True
        except Exception as exc:
            logger.warning("%s: failed to save model: %s", self.name, exc)
            return False

    def load(self) -> bool:
        """Load a previously saved model from disk.

        Override in subclasses that need custom deserialization.

        Returns:
            True if a model was loaded, False otherwise.
        """
        if not self._model_path:
            return False

        try:
            p = Path(self._model_path)
            if not p.exists():
                return False

            with open(p, "rb") as f:
                data = pickle.load(f)

            self._deserialize(data)

            logger.info(
                "%s: loaded model (accuracy=%.3f, n=%d)",
                self.name, self._accuracy, self._training_count,
            )
            return True
        except Exception as exc:
            logger.debug("%s: no existing model to load: %s", self.name, exc)
            return False

    def _serialize(self) -> dict[str, Any]:
        """Build the dict to pickle. Override to add learner-specific fields."""
        return {
            "model": self._model,
            "accuracy": self._accuracy,
            "training_count": self._training_count,
            "last_trained": self._last_trained,
        }

    def _deserialize(self, data: dict[str, Any]) -> None:
        """Restore state from a pickled dict. Override to read learner-specific fields."""
        self._model = data.get("model")
        self._accuracy = data.get("accuracy", 0.0)
        self._training_count = data.get("training_count", 0)
        self._last_trained = data.get("last_trained", 0.0)

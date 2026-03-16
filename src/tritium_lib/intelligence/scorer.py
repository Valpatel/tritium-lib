# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""CorrelationScorer ABC with static and learned implementations.

Provides a pluggable scoring interface for target correlation. The
StaticScorer uses hand-tuned weights (the current approach). The
LearnedScorer wraps a trained model (logistic regression) for
data-driven scoring. Both accept a CorrelationFeatures dict and
return a ScorerResult with probability and confidence.

Usage by tritium-sc's TargetCorrelator:
    scorer = LearnedScorer.from_file("model.pkl")
    if scorer is None:
        scorer = StaticScorer()
    result = scorer.predict(features)
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

# Feature dict type alias
CorrelationFeatures = dict[str, float]

# Canonical feature names expected by scorers.
# Wave 126: expanded from 6 to 10 features for richer correlation signals.
FEATURE_NAMES = [
    "distance",
    "rssi_delta",
    "co_movement",
    "device_type_match",
    "time_gap",
    "signal_pattern",
    # New features (Wave 126)
    "co_movement_duration",
    "time_of_day_similarity",
    "source_diversity_score",
    "wifi_probe_correlation",
]

# Default static weights — hand-tuned baseline
DEFAULT_WEIGHTS: dict[str, float] = {
    "distance": -0.25,       # Closer = higher score (negative weight on distance)
    "rssi_delta": -0.08,     # Smaller delta = higher score
    "co_movement": 0.18,     # Co-movement is strong signal
    "device_type_match": 0.12,  # Same type pair boost
    "time_gap": -0.08,       # Smaller gap = higher score
    "signal_pattern": 0.15,  # Signal appearance timing
    # New features (Wave 126)
    "co_movement_duration": 0.12,   # Duration of co-located movement
    "time_of_day_similarity": 0.06, # Same time of day across sessions
    "source_diversity_score": 0.08, # Multiple sensor types involved
    "wifi_probe_correlation": 0.14, # WiFi probe + BLE temporal match
}

DEFAULT_BIAS = 0.5  # Baseline probability before features


@dataclass(slots=True)
class ScorerResult:
    """Result of a correlation score prediction."""
    probability: float  # 0.0 to 1.0 — predicted correlation probability
    confidence: float   # 0.0 to 1.0 — confidence in the prediction
    method: str = ""    # "static" or "learned"
    detail: str = ""    # Human-readable explanation


class CorrelationScorer(ABC):
    """Abstract base class for correlation scoring models.

    Implementations must accept a feature dict and return a ScorerResult.
    Features are normalized floats (see FEATURE_NAMES).
    """

    @abstractmethod
    def predict(self, features: CorrelationFeatures) -> ScorerResult:
        """Predict correlation probability from features.

        Args:
            features: Dict mapping feature names to float values.
                Missing features default to 0.0.

        Returns:
            ScorerResult with probability, confidence, and method info.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this scorer."""

    @property
    def is_trained(self) -> bool:
        """Whether this scorer has been trained on data."""
        return False


class StaticScorer(CorrelationScorer):
    """Hand-tuned weighted scoring using static coefficients.

    Computes a linear combination of features with configurable weights,
    then applies sigmoid to produce a probability. This is the baseline
    approach — always available, no training data required.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        bias: float = DEFAULT_BIAS,
    ) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.bias = bias

    @property
    def name(self) -> str:
        return "static"

    def predict(self, features: CorrelationFeatures) -> ScorerResult:
        """Compute weighted sum + sigmoid."""
        logit = 0.0
        contributing: list[str] = []

        for feat_name, weight in self.weights.items():
            value = features.get(feat_name, 0.0)
            contribution = weight * value
            logit += contribution
            if abs(contribution) > 0.01:
                contributing.append(f"{feat_name}={value:.2f}*{weight:.2f}")

        # Apply sigmoid centered at bias
        probability = _sigmoid(logit + (self.bias - 0.5) * 2.0)
        # Confidence is based on how far from 0.5 the result is
        confidence = min(1.0, abs(probability - 0.5) * 2.0)

        return ScorerResult(
            probability=probability,
            confidence=confidence,
            method="static",
            detail=f"logit={logit:.3f} [{', '.join(contributing)}]",
        )


class LearnedScorer(CorrelationScorer):
    """Trained model scorer using scikit-learn logistic regression.

    Wraps a fitted sklearn LogisticRegression model. Falls back to
    StaticScorer if the model is not available or prediction fails.
    """

    def __init__(
        self,
        model: Any = None,
        feature_names: Optional[list[str]] = None,
        accuracy: float = 0.0,
        training_count: int = 0,
        fallback: Optional[StaticScorer] = None,
    ) -> None:
        self._model = model
        self._feature_names = feature_names or list(FEATURE_NAMES)
        self._accuracy = accuracy
        self._training_count = training_count
        self._fallback = fallback or StaticScorer()

    @property
    def name(self) -> str:
        return "learned"

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def training_count(self) -> int:
        return self._training_count

    def predict(self, features: CorrelationFeatures) -> ScorerResult:
        """Predict using trained model, fallback to static on error."""
        if self._model is None:
            result = self._fallback.predict(features)
            result.detail = f"(no model, fallback) {result.detail}"
            return result

        try:
            # Build feature vector in canonical order
            X = [[features.get(fn, 0.0) for fn in self._feature_names]]
            proba = self._model.predict_proba(X)[0]
            # proba is [P(class=0), P(class=1)]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            confidence = min(1.0, abs(probability - 0.5) * 2.0)

            return ScorerResult(
                probability=probability,
                confidence=confidence,
                method="learned",
                detail=f"model_accuracy={self._accuracy:.3f} n={self._training_count}",
            )
        except Exception as exc:
            # Fallback to static scorer
            result = self._fallback.predict(features)
            result.detail = f"(model error: {exc}, fallback) {result.detail}"
            return result

    @classmethod
    def from_file(cls, path: str) -> Optional["LearnedScorer"]:
        """Load a trained model from a pickle file.

        Returns None if the file doesn't exist or loading fails.
        """
        try:
            import pickle
            from pathlib import Path

            p = Path(path)
            if not p.exists():
                return None

            with open(p, "rb") as f:
                data = pickle.load(f)

            return cls(
                model=data.get("model"),
                feature_names=data.get("feature_names", list(FEATURE_NAMES)),
                accuracy=data.get("accuracy", 0.0),
                training_count=data.get("training_count", 0),
            )
        except Exception:
            return None

    def save(self, path: str) -> bool:
        """Save the trained model to a pickle file.

        Returns True on success.
        """
        try:
            import pickle
            from pathlib import Path

            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "model": self._model,
                "feature_names": self._feature_names,
                "accuracy": self._accuracy,
                "training_count": self._training_count,
            }
            with open(p, "wb") as f:
                pickle.dump(data, f)
            return True
        except Exception:
            return False


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)

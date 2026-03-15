# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""PatternLearner — learns which behavioral patterns predict threats.

Trains on historical pattern-to-outcome data (e.g., a sequence of
movements, device appearances, time-of-day patterns that preceded a
threat escalation).  Uses a lightweight approach (no heavy ML deps):

1. Feature extraction from behavioral patterns
2. Logistic-regression-style scoring with learned weights
3. Bayesian updating of prior probabilities

Usage::

    learner = PatternLearner()
    # Train on historical data
    learner.add_training_example(
        features={"time_of_day_hour": 2, "device_count": 5, "new_device_ratio": 0.8},
        outcome="threat",
    )
    result = learner.train()
    # Predict
    prob = learner.predict({"time_of_day_hour": 3, "device_count": 4, "new_device_ratio": 0.6})
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tritium_lib.intelligence.base_learner import BaseLearner

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Standard feature names for behavioral patterns
PATTERN_FEATURES = [
    "time_of_day_hour",       # 0-23
    "day_of_week",            # 0-6 (Mon=0)
    "device_count",           # number of devices in area
    "new_device_ratio",       # fraction of devices not seen before (0-1)
    "dwell_time_minutes",     # how long the entity has been in area
    "movement_speed_mps",     # meters per second
    "revisit_count",          # how many times this entity has returned
    "distance_from_center_m", # distance from site center
    "signal_strength_dbm",    # average signal strength
    "correlation_count",      # number of correlated entities
    "threat_score",           # existing threat score (0-1)
    "alert_count_1h",         # alerts in the last hour
    "geofence_violations",    # number of geofence violations
]


@dataclass
class TrainingExample:
    """A single training example: features + outcome."""

    features: dict[str, float] = field(default_factory=dict)
    outcome: str = "benign"     # "threat" or "benign"
    timestamp: float = 0.0
    target_id: str = ""
    source: str = ""


@dataclass
class PredictionResult:
    """Result of a threat prediction."""

    threat_probability: float = 0.0    # 0.0 to 1.0
    confidence: float = 0.0           # 0.0 to 1.0 (based on training data)
    contributing_features: dict[str, float] = field(default_factory=dict)
    recommendation: str = "monitor"   # monitor, investigate, alert


# ---------------------------------------------------------------------------
# PatternLearner
# ---------------------------------------------------------------------------

class PatternLearner(BaseLearner):
    """Learns which behavioral patterns predict threats.

    Uses a lightweight logistic-regression-inspired approach:
      - Maintains per-feature weights (learned via gradient descent)
      - Maintains per-feature mean/std for normalization
      - Uses sigmoid activation for probability output
      - Bayesian prior from historical threat rate

    No numpy/scipy/sklearn required.
    """

    def __init__(self, model_path: str = "") -> None:
        super().__init__(model_path)
        self._training_data: list[TrainingExample] = []
        self._weights: dict[str, float] = {}
        self._bias: float = 0.0
        self._feature_means: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}
        self._threat_prior: float = 0.1  # prior probability of threat
        self._learning_rate: float = 0.01
        self._max_iterations: int = 200

    @property
    def name(self) -> str:
        return "pattern_learner"

    # ------------------------------------------------------------------
    # Training data management
    # ------------------------------------------------------------------

    def add_training_example(
        self,
        features: dict[str, float],
        outcome: str = "benign",
        target_id: str = "",
        source: str = "",
    ) -> None:
        """Add a training example.

        Args:
            features: Dict of feature_name -> float value.
            outcome: "threat" or "benign".
            target_id: Optional target this example relates to.
            source: Optional source sensor.
        """
        example = TrainingExample(
            features=dict(features),
            outcome=outcome,
            timestamp=time.time(),
            target_id=target_id,
            source=source,
        )
        self._training_data.append(example)

    def clear_training_data(self) -> None:
        """Remove all training examples."""
        self._training_data.clear()

    @property
    def training_examples(self) -> int:
        """Number of stored training examples."""
        return len(self._training_data)

    # ------------------------------------------------------------------
    # Feature normalization
    # ------------------------------------------------------------------

    def _compute_feature_stats(self) -> None:
        """Compute mean and std for each feature across training data."""
        if not self._training_data:
            return

        # Collect all feature names
        all_features: set[str] = set()
        for ex in self._training_data:
            all_features.update(ex.features.keys())

        for feat in all_features:
            values = [ex.features.get(feat, 0.0) for ex in self._training_data]
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / max(n, 1)
            std = math.sqrt(variance) if variance > 0 else 1.0
            self._feature_means[feat] = mean
            self._feature_stds[feat] = std

    def _normalize(self, features: dict[str, float]) -> dict[str, float]:
        """Normalize features using stored mean/std."""
        result = {}
        for feat, val in features.items():
            mean = self._feature_means.get(feat, 0.0)
            std = self._feature_stds.get(feat, 1.0)
            result[feat] = (val - mean) / std if std > 0 else 0.0
        return result

    # ------------------------------------------------------------------
    # Logistic regression core
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid activation, clamped to prevent overflow."""
        x = max(-500.0, min(500.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def _forward(self, features: dict[str, float]) -> float:
        """Compute the logistic regression output."""
        z = self._bias
        for feat, val in features.items():
            z += self._weights.get(feat, 0.0) * val
        return self._sigmoid(z)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Train the model on accumulated training examples.

        Uses mini-batch gradient descent on logistic regression.

        Returns:
            Dict with 'success', 'accuracy', 'training_count', etc.
        """
        if len(self._training_data) < 2:
            return {
                "success": False,
                "error": "Need at least 2 training examples",
                "training_count": len(self._training_data),
            }

        # Count outcomes
        threat_count = sum(1 for ex in self._training_data if ex.outcome == "threat")
        benign_count = len(self._training_data) - threat_count

        if threat_count == 0 or benign_count == 0:
            return {
                "success": False,
                "error": "Need both threat and benign examples",
                "training_count": len(self._training_data),
                "threat_count": threat_count,
                "benign_count": benign_count,
            }

        # Update threat prior
        self._threat_prior = threat_count / len(self._training_data)

        # Compute feature stats for normalization
        self._compute_feature_stats()

        # Initialize weights if needed
        all_features = set()
        for ex in self._training_data:
            all_features.update(ex.features.keys())
        for feat in all_features:
            if feat not in self._weights:
                self._weights[feat] = 0.0

        # Gradient descent
        for iteration in range(self._max_iterations):
            total_loss = 0.0

            for ex in self._training_data:
                normalized = self._normalize(ex.features)
                y_true = 1.0 if ex.outcome == "threat" else 0.0
                y_pred = self._forward(normalized)

                # Binary cross-entropy gradient
                error = y_pred - y_true
                total_loss += -(
                    y_true * math.log(max(y_pred, 1e-10))
                    + (1 - y_true) * math.log(max(1 - y_pred, 1e-10))
                )

                # Update weights
                for feat, val in normalized.items():
                    self._weights[feat] -= self._learning_rate * error * val
                self._bias -= self._learning_rate * error

        # Compute accuracy
        correct = 0
        for ex in self._training_data:
            normalized = self._normalize(ex.features)
            prob = self._forward(normalized)
            predicted = "threat" if prob >= 0.5 else "benign"
            if predicted == ex.outcome:
                correct += 1

        accuracy = correct / len(self._training_data)

        self._model = {
            "weights": dict(self._weights),
            "bias": self._bias,
            "feature_means": dict(self._feature_means),
            "feature_stds": dict(self._feature_stds),
            "threat_prior": self._threat_prior,
        }
        self._accuracy = accuracy
        self._training_count = len(self._training_data)
        self._last_trained = time.time()

        log.info(
            "PatternLearner trained: accuracy=%.3f, n=%d (threat=%d, benign=%d)",
            accuracy, len(self._training_data), threat_count, benign_count,
        )

        return {
            "success": True,
            "accuracy": accuracy,
            "training_count": len(self._training_data),
            "threat_count": threat_count,
            "benign_count": benign_count,
            "feature_count": len(self._weights),
            "threat_prior": self._threat_prior,
        }

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, features: dict[str, float] | Any) -> PredictionResult:
        """Predict threat probability for a new pattern.

        Args:
            features: Dict of feature_name -> float value.

        Returns:
            PredictionResult with threat_probability, confidence,
            contributing_features, and recommendation.
        """
        if not isinstance(features, dict):
            return PredictionResult(
                threat_probability=self._threat_prior,
                confidence=0.0,
                recommendation="monitor",
            )

        if not self.is_trained:
            # No model — use prior
            return PredictionResult(
                threat_probability=self._threat_prior,
                confidence=0.0,
                recommendation="monitor",
            )

        normalized = self._normalize(features)
        prob = self._forward(normalized)

        # Compute feature contributions
        contributions = {}
        for feat, val in normalized.items():
            weight = self._weights.get(feat, 0.0)
            contributions[feat] = round(weight * val, 4)

        # Confidence based on how many training examples we have
        # and how many features overlap with training
        feature_overlap = sum(1 for f in features if f in self._weights)
        total_features = max(len(self._weights), 1)
        data_confidence = min(1.0, self._training_count / 100)
        feature_confidence = feature_overlap / total_features
        confidence = data_confidence * feature_confidence

        # Recommendation thresholds
        if prob >= 0.7:
            recommendation = "alert"
        elif prob >= 0.4:
            recommendation = "investigate"
        else:
            recommendation = "monitor"

        return PredictionResult(
            threat_probability=round(prob, 4),
            confidence=round(confidence, 4),
            contributing_features=contributions,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return learner stats including pattern-specific fields."""
        base = super().get_stats()
        base.update({
            "training_examples": len(self._training_data),
            "threat_prior": self._threat_prior,
            "feature_count": len(self._weights),
            "feature_names": sorted(self._weights.keys()) if self._weights else [],
            "top_weights": dict(
                sorted(self._weights.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            ) if self._weights else {},
        })
        return base

    # ------------------------------------------------------------------
    # Persistence overrides
    # ------------------------------------------------------------------

    def _serialize(self) -> dict[str, Any]:
        """Serialize model state for persistence."""
        data = super()._serialize()
        data.update({
            "weights": self._weights,
            "bias": self._bias,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "threat_prior": self._threat_prior,
            "training_data": [
                {
                    "features": ex.features,
                    "outcome": ex.outcome,
                    "timestamp": ex.timestamp,
                    "target_id": ex.target_id,
                    "source": ex.source,
                }
                for ex in self._training_data
            ],
        })
        return data

    def _deserialize(self, data: dict[str, Any]) -> None:
        """Restore model state from persistence."""
        super()._deserialize(data)
        self._weights = data.get("weights", {})
        self._bias = data.get("bias", 0.0)
        self._feature_means = data.get("feature_means", {})
        self._feature_stds = data.get("feature_stds", {})
        self._threat_prior = data.get("threat_prior", 0.1)
        self._training_data = [
            TrainingExample(**td)
            for td in data.get("training_data", [])
        ]

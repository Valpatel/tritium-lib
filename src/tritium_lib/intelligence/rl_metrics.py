# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""RL model health metrics — tracks model accuracy, feature importance,
training data size, and prediction distribution over time.

Provides RLMetrics for monitoring reinforcement-learning model health
in the correlation and classification pipelines. Designed to integrate
with the CorrelationLearner and any future RL-based learners.

Usage::

    metrics = RLMetrics()
    metrics.record_training(accuracy=0.78, training_count=200,
                            feature_importance={"distance": 0.25, ...})
    metrics.record_prediction(predicted_class=1, probability=0.85)
    status = metrics.get_status()
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingSnapshot:
    """Record of a single training run."""

    timestamp: float
    accuracy: float
    training_count: int
    feature_importance: dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    duration_s: float = 0.0


@dataclass
class PredictionRecord:
    """Record of a single prediction."""

    timestamp: float
    predicted_class: int  # 0 or 1 for binary
    probability: float
    correct: bool | None = None  # None = no feedback yet


class RLMetrics:
    """Thread-safe metrics tracker for RL model health monitoring.

    Tracks:
    - Model accuracy over time (from training snapshots)
    - Feature importance evolution (which features matter most)
    - Training data size growth
    - Prediction distribution (class balance, confidence distribution)
    - Rolling window statistics for recent model performance

    Parameters
    ----------
    max_history:
        Maximum training snapshots and prediction records to retain.
    window_seconds:
        Time window for rolling statistics (default 1 hour).
    """

    def __init__(
        self,
        max_history: int = 1000,
        window_seconds: float = 3600.0,
    ) -> None:
        self._lock = threading.RLock()
        self._max_history = max_history
        self._window = window_seconds

        # Training history
        self._training_history: deque[TrainingSnapshot] = deque(maxlen=max_history)

        # Prediction history
        self._predictions: deque[PredictionRecord] = deque(maxlen=max_history * 10)

        # Aggregate counters
        self._total_trainings = 0
        self._total_predictions = 0
        self._total_correct = 0
        self._total_incorrect = 0

        # Feature importance accumulation (for averaging)
        self._feature_importance_sum: dict[str, float] = defaultdict(float)
        self._feature_importance_count: int = 0

        # Per-model tracking
        self._model_metrics: dict[str, _ModelMetrics] = {}

    def record_training(
        self,
        accuracy: float,
        training_count: int,
        feature_importance: dict[str, float] | None = None,
        model_name: str = "correlation",
        duration_s: float = 0.0,
    ) -> None:
        """Record a training run completion.

        Parameters
        ----------
        accuracy:
            Cross-validation accuracy (0.0 to 1.0).
        training_count:
            Number of training examples used.
        feature_importance:
            Feature name -> importance score mapping.
        model_name:
            Identifier for the model that was trained.
        duration_s:
            Training duration in seconds.
        """
        now = time.time()
        snapshot = TrainingSnapshot(
            timestamp=now,
            accuracy=accuracy,
            training_count=training_count,
            feature_importance=feature_importance or {},
            model_name=model_name,
            duration_s=duration_s,
        )

        with self._lock:
            self._training_history.append(snapshot)
            self._total_trainings += 1

            # Accumulate feature importance
            if feature_importance:
                self._feature_importance_count += 1
                for feat, imp in feature_importance.items():
                    self._feature_importance_sum[feat] += imp

            # Per-model tracking
            if model_name not in self._model_metrics:
                self._model_metrics[model_name] = _ModelMetrics(model_name)
            mm = self._model_metrics[model_name]
            mm.last_accuracy = accuracy
            mm.last_training_count = training_count
            mm.last_trained = now
            mm.total_trainings += 1
            mm.accuracy_history.append((now, accuracy))

    def record_prediction(
        self,
        predicted_class: int,
        probability: float,
        correct: bool | None = None,
        model_name: str = "correlation",
    ) -> None:
        """Record a model prediction.

        Parameters
        ----------
        predicted_class:
            Predicted class (0 or 1 for binary classification).
        probability:
            Model's predicted probability for the positive class.
        correct:
            Whether the prediction was correct (None if unknown).
        model_name:
            Which model made the prediction.
        """
        now = time.time()
        record = PredictionRecord(
            timestamp=now,
            predicted_class=predicted_class,
            probability=probability,
            correct=correct,
        )

        with self._lock:
            self._predictions.append(record)
            self._total_predictions += 1

            if correct is True:
                self._total_correct += 1
            elif correct is False:
                self._total_incorrect += 1

            if model_name in self._model_metrics:
                mm = self._model_metrics[model_name]
                mm.total_predictions += 1
                if correct is True:
                    mm.correct_predictions += 1
                elif correct is False:
                    mm.incorrect_predictions += 1

    def record_feedback(
        self,
        correct: bool,
        model_name: str = "correlation",
    ) -> None:
        """Record feedback on a past prediction.

        Parameters
        ----------
        correct:
            Whether the prediction was correct.
        model_name:
            Which model's prediction this feedback applies to.
        """
        with self._lock:
            if correct:
                self._total_correct += 1
            else:
                self._total_incorrect += 1

            if model_name in self._model_metrics:
                mm = self._model_metrics[model_name]
                if correct:
                    mm.correct_predictions += 1
                else:
                    mm.incorrect_predictions += 1

    def get_accuracy_trend(
        self,
        model_name: str = "correlation",
        max_points: int = 50,
    ) -> list[dict[str, Any]]:
        """Get accuracy over time for a model.

        Returns list of {timestamp, accuracy, training_count} dicts.
        """
        with self._lock:
            snapshots = [
                s for s in self._training_history
                if s.model_name == model_name
            ]
            # Take last max_points
            snapshots = list(snapshots)[-max_points:]
            return [
                {
                    "timestamp": s.timestamp,
                    "accuracy": round(s.accuracy, 4),
                    "training_count": s.training_count,
                    "duration_s": round(s.duration_s, 2),
                }
                for s in snapshots
            ]

    def get_feature_importance(
        self,
        model_name: str = "correlation",
    ) -> dict[str, float]:
        """Get averaged feature importance across training runs.

        Returns feature_name -> average_importance dict, sorted by importance.
        """
        with self._lock:
            # Use the most recent training snapshot's importance if available
            for snapshot in reversed(self._training_history):
                if snapshot.model_name == model_name and snapshot.feature_importance:
                    return dict(
                        sorted(
                            snapshot.feature_importance.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )
                    )

            # Fall back to averaged importance
            if self._feature_importance_count == 0:
                return {}

            avg = {
                k: round(v / self._feature_importance_count, 4)
                for k, v in self._feature_importance_sum.items()
            }
            return dict(
                sorted(avg.items(), key=lambda x: abs(x[1]), reverse=True)
            )

    def get_prediction_distribution(
        self,
        window_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Get prediction class and probability distribution.

        Parameters
        ----------
        window_seconds:
            Time window to analyze. Defaults to self._window.

        Returns
        -------
        dict with:
            class_counts: {0: N, 1: M}
            probability_histogram: counts in 10 bins [0.0-0.1, 0.1-0.2, ...]
            mean_probability: average predicted probability
            correct_rate: rate of correct predictions (where feedback exists)
        """
        window = window_seconds or self._window
        cutoff = time.time() - window

        with self._lock:
            recent = [p for p in self._predictions if p.timestamp >= cutoff]

        if not recent:
            return {
                "class_counts": {0: 0, 1: 0},
                "probability_histogram": [0] * 10,
                "mean_probability": 0.0,
                "correct_rate": 0.0,
                "total": 0,
            }

        class_counts = {0: 0, 1: 0}
        histogram = [0] * 10
        prob_sum = 0.0
        correct_count = 0
        feedback_count = 0

        for p in recent:
            class_counts[p.predicted_class] = class_counts.get(p.predicted_class, 0) + 1
            bin_idx = min(9, int(p.probability * 10))
            histogram[bin_idx] += 1
            prob_sum += p.probability
            if p.correct is not None:
                feedback_count += 1
                if p.correct:
                    correct_count += 1

        return {
            "class_counts": class_counts,
            "probability_histogram": histogram,
            "mean_probability": round(prob_sum / len(recent), 4),
            "correct_rate": round(correct_count / feedback_count, 4) if feedback_count > 0 else 0.0,
            "total": len(recent),
        }

    def get_training_data_growth(
        self,
        model_name: str = "correlation",
        max_points: int = 50,
    ) -> list[dict[str, Any]]:
        """Get training data size growth over time.

        Returns list of {timestamp, training_count} dicts.
        """
        with self._lock:
            snapshots = [
                s for s in self._training_history
                if s.model_name == model_name
            ]
            snapshots = list(snapshots)[-max_points:]
            return [
                {
                    "timestamp": s.timestamp,
                    "training_count": s.training_count,
                }
                for s in snapshots
            ]

    def get_status(self) -> dict[str, Any]:
        """Full status report for API/dashboard consumption."""
        with self._lock:
            total_feedback = self._total_correct + self._total_incorrect
            overall_accuracy = (
                round(self._total_correct / total_feedback, 4)
                if total_feedback > 0 else 0.0
            )

            models = {}
            for name, mm in self._model_metrics.items():
                models[name] = mm.to_dict()

            # Latest training snapshot
            latest_training = None
            if self._training_history:
                s = self._training_history[-1]
                latest_training = {
                    "timestamp": s.timestamp,
                    "accuracy": round(s.accuracy, 4),
                    "training_count": s.training_count,
                    "model_name": s.model_name,
                    "duration_s": round(s.duration_s, 2),
                }

            return {
                "total_trainings": self._total_trainings,
                "total_predictions": self._total_predictions,
                "total_correct": self._total_correct,
                "total_incorrect": self._total_incorrect,
                "overall_accuracy": overall_accuracy,
                "models": models,
                "latest_training": latest_training,
                "feature_importance": self.get_feature_importance(),
                "prediction_distribution": self.get_prediction_distribution(),
            }

    def export(self) -> dict[str, Any]:
        """Export all metrics as a serializable dictionary for persistence or API.

        Returns a comprehensive snapshot including:
        - Full status report
        - Accuracy trend per model
        - Training data growth per model
        - Feature importance per model
        - Raw training history and prediction counts

        Suitable for JSON serialization, dashboard consumption, or
        inter-process metrics sharing.
        """
        with self._lock:
            status = self.get_status()

            # Per-model detailed metrics
            models_detail = {}
            for name, mm in self._model_metrics.items():
                models_detail[name] = {
                    **mm.to_dict(),
                    "accuracy_trend": self.get_accuracy_trend(model_name=name),
                    "training_growth": self.get_training_data_growth(model_name=name),
                    "feature_importance": self.get_feature_importance(model_name=name),
                }

            return {
                "status": status,
                "models_detail": models_detail,
                "total_trainings": self._total_trainings,
                "total_predictions": self._total_predictions,
                "total_correct": self._total_correct,
                "total_incorrect": self._total_incorrect,
                "training_history_size": len(self._training_history),
                "prediction_history_size": len(self._predictions),
                "export_timestamp": time.time(),
            }

    def reset(self) -> None:
        """Reset all metrics. Useful for testing."""
        with self._lock:
            self._training_history.clear()
            self._predictions.clear()
            self._total_trainings = 0
            self._total_predictions = 0
            self._total_correct = 0
            self._total_incorrect = 0
            self._feature_importance_sum.clear()
            self._feature_importance_count = 0
            self._model_metrics.clear()


class _ModelMetrics:
    """Per-model tracking state (internal)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.last_accuracy: float = 0.0
        self.last_training_count: int = 0
        self.last_trained: float = 0.0
        self.total_trainings: int = 0
        self.total_predictions: int = 0
        self.correct_predictions: int = 0
        self.incorrect_predictions: int = 0
        self.accuracy_history: deque[tuple[float, float]] = deque(maxlen=100)

    @property
    def prediction_accuracy(self) -> float:
        total = self.correct_predictions + self.incorrect_predictions
        if total == 0:
            return 0.0
        return self.correct_predictions / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "last_accuracy": round(self.last_accuracy, 4),
            "last_training_count": self.last_training_count,
            "last_trained": self.last_trained,
            "total_trainings": self.total_trainings,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "incorrect_predictions": self.incorrect_predictions,
            "prediction_accuracy": round(self.prediction_accuracy, 4),
            "accuracy_trend": [
                {"timestamp": t, "accuracy": round(a, 4)}
                for t, a in list(self.accuracy_history)[-20:]
            ],
        }

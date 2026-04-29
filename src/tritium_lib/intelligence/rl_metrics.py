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


@dataclass
class FeatureAblation:
    """Per-feature health snapshot used to diagnose stuck training.

    Stores the basic descriptive statistics of one feature column across
    the training dataset.  Designed for B-10 / Wave 198 instrumentation —
    a feature that is always 0 or always 1 explains a classifier that
    plateaus at a fixed accuracy.

    Attributes
    ----------
    feature_name:
        The feature column.
    mean:
        Average value across training rows.
    std:
        Population standard deviation across training rows.
    minimum:
        Smallest observed value.
    maximum:
        Largest observed value.
    unique_values:
        Number of distinct values seen (capped at 64 to bound memory).
    is_constant:
        True if std < 1e-9, i.e. the feature contributes no signal.
    saturation_ratio:
        Fraction of rows that take the dominant value.  1.0 means the
        feature is effectively constant.
    sample_count:
        Number of training rows sampled.
    importance:
        Optional model-side importance copy (for ablation views).
    """

    feature_name: str = ""
    mean: float = 0.0
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    unique_values: int = 0
    is_constant: bool = False
    saturation_ratio: float = 0.0
    sample_count: int = 0
    importance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "unique_values": self.unique_values,
            "is_constant": self.is_constant,
            "saturation_ratio": round(self.saturation_ratio, 4),
            "sample_count": self.sample_count,
            "importance": round(self.importance, 6),
        }


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
        prediction_store: Any = None,
    ) -> None:
        """Initialise metrics tracker.

        Parameters
        ----------
        max_history:
            Maximum training snapshots and prediction records held in
            memory.
        window_seconds:
            Time window for rolling statistics (default 1 hour).
        prediction_store:
            Optional ``PredictionStore``-compatible object.  If
            supplied, every ``record_prediction`` call also writes
            through to the store, and the in-memory deque is hydrated
            from the store at construction time so historical
            predictions survive a process restart (B-6).
        """
        self._lock = threading.RLock()
        self._max_history = max_history
        self._window = window_seconds

        # Training history
        self._training_history: deque[TrainingSnapshot] = deque(maxlen=max_history)

        # Prediction history (in-memory, bounded).  When a
        # ``prediction_store`` is configured this deque is the recent
        # cache and the store is the durable record (B-6).
        self._predictions: deque[PredictionRecord] = deque(maxlen=max_history * 10)
        self._prediction_store = prediction_store

        # Hydrate the in-memory deque from the persistent store so
        # restart-survival is transparent to existing consumers.  Hard
        # cap to ``maxlen`` to respect deque semantics.
        if self._prediction_store is not None:
            try:
                cached = list(self._prediction_store)
                if cached:
                    cap = self._predictions.maxlen or len(cached)
                    self._predictions.extend(cached[-cap:])
            except Exception:  # pragma: no cover — store I/O is best-effort
                pass

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

        # Per-model feature ablation: model_name -> {feature_name -> FeatureAblation}
        # Refreshed on each call to ``record_feature_stats`` (B-10 instrumentation).
        self._feature_ablation: dict[str, dict[str, FeatureAblation]] = {}

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

            # B-6: write-through to persistent store (if configured).
            # Best-effort — store outages must not break inference.
            if self._prediction_store is not None:
                try:
                    self._prediction_store.append(record, model_name=model_name)
                except TypeError:
                    # Older store API — fall back to positional call
                    try:
                        self._prediction_store.append(record)
                    except Exception:  # pragma: no cover
                        pass
                except Exception:  # pragma: no cover - persistence is opt-in
                    pass

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

    def record_feature_stats(
        self,
        feature_names: list[str],
        rows: list[list[float]],
        model_name: str = "correlation",
        importance: dict[str, float] | None = None,
    ) -> dict[str, FeatureAblation]:
        """Compute and store per-feature ablation stats from training rows.

        Designed for B-10: when accuracy plateaus, the most common cause is
        a feature that is always 0 (or always 1) — it contributes no
        information.  This method scans the training matrix once and
        records mean / std / min / max / unique-count / saturation ratio
        per feature, plus the model's importance score if supplied.

        Parameters
        ----------
        feature_names:
            Column names in the same order as ``rows``.
        rows:
            Training feature matrix (list of feature vectors).  May be
            empty; in that case all features are reported as constant
            with sample_count=0.
        model_name:
            Which model these stats describe (default ``"correlation"``).
        importance:
            Optional ``{feature_name: importance}`` map, copied into each
            FeatureAblation for convenient ablation views.

        Returns
        -------
        dict[str, FeatureAblation]
            The newly-recorded ablation snapshot.
        """
        importance = importance or {}
        snapshot: dict[str, FeatureAblation] = {}

        n_rows = len(rows)
        if n_rows == 0:
            for fname in feature_names:
                snapshot[fname] = FeatureAblation(
                    feature_name=fname,
                    is_constant=True,
                    sample_count=0,
                    importance=float(importance.get(fname, 0.0)),
                )
        else:
            for col_idx, fname in enumerate(feature_names):
                col_values: list[float] = []
                for row in rows:
                    if col_idx < len(row):
                        try:
                            col_values.append(float(row[col_idx]))
                        except (TypeError, ValueError):
                            continue

                if not col_values:
                    snapshot[fname] = FeatureAblation(
                        feature_name=fname,
                        is_constant=True,
                        sample_count=0,
                        importance=float(importance.get(fname, 0.0)),
                    )
                    continue

                count = len(col_values)
                mean = sum(col_values) / count
                variance = sum((v - mean) ** 2 for v in col_values) / count
                std = variance ** 0.5
                minimum = min(col_values)
                maximum = max(col_values)

                # Cap unique tracking to avoid OOM on continuous features
                seen: set[float] = set()
                value_counts: dict[float, int] = {}
                for v in col_values:
                    if len(seen) < 64:
                        seen.add(v)
                    value_counts[v] = value_counts.get(v, 0) + 1
                dominant_count = max(value_counts.values()) if value_counts else 0
                saturation = dominant_count / count if count > 0 else 0.0

                snapshot[fname] = FeatureAblation(
                    feature_name=fname,
                    mean=mean,
                    std=std,
                    minimum=minimum,
                    maximum=maximum,
                    unique_values=len(seen),
                    is_constant=std < 1e-9,
                    saturation_ratio=saturation,
                    sample_count=count,
                    importance=float(importance.get(fname, 0.0)),
                )

        with self._lock:
            self._feature_ablation[model_name] = snapshot
        return snapshot

    def get_feature_ablation(
        self,
        model_name: str = "correlation",
    ) -> list[dict[str, Any]]:
        """Return the most recent feature ablation snapshot.

        Returns a list of dicts (sorted by importance, descending) so
        callers can see at-a-glance which features are doing work and
        which are constant or saturated.
        """
        with self._lock:
            snap = self._feature_ablation.get(model_name, {})
            entries = [v.to_dict() for v in snap.values()]

        entries.sort(key=lambda d: d.get("importance", 0.0), reverse=True)
        return entries

    def get_constant_features(
        self,
        model_name: str = "correlation",
        saturation_threshold: float = 0.95,
    ) -> list[str]:
        """Return features that are constant or near-constant.

        Used by B-10 instrumentation to quickly surface stuck-classifier
        causes.  A feature is considered "constant" if either its
        standard deviation is below 1e-9 or its dominant-value
        saturation ratio is at or above ``saturation_threshold``.
        """
        with self._lock:
            snap = self._feature_ablation.get(model_name, {})
            stuck = [
                a.feature_name
                for a in snap.values()
                if a.is_constant or a.saturation_ratio >= saturation_threshold
            ]
        return stuck

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
                    # B-10 instrumentation: per-feature ablation diagnostics.
                    # Surfaces constant / saturated features that explain a
                    # plateaued accuracy (e.g. RL stuck at 0.855 in W198).
                    "feature_ablation": self.get_feature_ablation(model_name=name),
                    "constant_features": self.get_constant_features(model_name=name),
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

    def reset(self, *, clear_store: bool = False) -> None:
        """Reset all metrics. Useful for testing.

        Parameters
        ----------
        clear_store:
            If True and a ``prediction_store`` was configured, also
            wipe the persistent store.  Defaults to False so test runs
            don't accidentally erase production history when a single
            ``RLMetrics`` instance is reset between scenarios.
        """
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
            self._feature_ablation.clear()

            if clear_store and self._prediction_store is not None:
                try:
                    self._prediction_store.clear()
                except Exception:  # pragma: no cover
                    pass


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

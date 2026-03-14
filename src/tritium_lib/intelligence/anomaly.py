# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Anomaly detection ABCs and implementations.

Provides a pluggable interface for anomaly detection on RF environment
metrics. Two implementations:

  - SimpleThresholdDetector: flags metrics > N standard deviations from
    the baseline mean. No external dependencies.

  - AutoencoderDetector: trains a simple autoencoder on baseline metrics
    and flags high reconstruction error as anomalous. Requires numpy.

Usage::

    from tritium_lib.intelligence.anomaly import (
        SimpleThresholdDetector,
        Anomaly,
    )

    detector = SimpleThresholdDetector(threshold_sigma=2.0)

    baseline = [
        {"ble_count": 10, "wifi_count": 5, "rssi_mean": -60},
        {"ble_count": 12, "wifi_count": 4, "rssi_mean": -58},
        # ... 288+ samples for 24h baseline ...
    ]

    current = {"ble_count": 25, "wifi_count": 5, "rssi_mean": -60}
    anomalies = detector.detect(current, baseline)
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Anomaly:
    """A single detected anomaly in a metric."""

    metric_name: str = ""
    current_value: float = 0.0
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    deviation_sigma: float = 0.0
    direction: str = ""  # "above" or "below"
    severity: str = "low"  # low, medium, high, critical
    score: float = 0.0  # Anomaly score (0.0 = normal, 1.0 = very anomalous)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_mean": round(self.baseline_mean, 4),
            "baseline_std": round(self.baseline_std, 4),
            "deviation_sigma": round(self.deviation_sigma, 2),
            "direction": self.direction,
            "severity": self.severity,
            "score": round(self.score, 4),
        }


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors.

    Subclasses implement ``detect()`` which compares current metric values
    against a baseline history and returns a list of detected anomalies.
    """

    @abstractmethod
    def detect(
        self,
        current_metrics: dict[str, float],
        baseline: list[dict[str, float]],
    ) -> list[Anomaly]:
        """Detect anomalies in current metrics relative to baseline.

        Parameters
        ----------
        current_metrics:
            Dict mapping metric names to current values.
        baseline:
            List of historical metric snapshots (same key structure).
            Should contain enough samples for meaningful statistics
            (recommended 288+ for 24h at 5-min intervals).

        Returns
        -------
        list[Anomaly]:
            Detected anomalies. Empty list if everything is normal.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the detector implementation name."""
        ...


class SimpleThresholdDetector(AnomalyDetector):
    """Flags metrics that deviate more than N standard deviations from baseline mean.

    Simple, robust, and requires no external dependencies. Good as a
    default detector when no ML libraries are available.

    Parameters
    ----------
    threshold_sigma:
        Number of standard deviations to consider anomalous (default 2.0).
    min_baseline_samples:
        Minimum baseline samples required before detection activates.
    """

    def __init__(
        self,
        threshold_sigma: float = 2.0,
        min_baseline_samples: int = 10,
    ) -> None:
        self._threshold = threshold_sigma
        self._min_samples = min_baseline_samples

    def name(self) -> str:
        return "simple_threshold"

    def detect(
        self,
        current_metrics: dict[str, float],
        baseline: list[dict[str, float]],
    ) -> list[Anomaly]:
        """Detect anomalies using simple threshold on each metric."""
        if len(baseline) < self._min_samples:
            return []

        anomalies: list[Anomaly] = []

        for metric_name, current_value in current_metrics.items():
            # Gather baseline values for this metric
            values = [
                sample[metric_name]
                for sample in baseline
                if metric_name in sample
            ]
            if len(values) < self._min_samples:
                continue

            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 0.0

            if std < 1e-6:
                # Near-zero variance — flag any notable change
                if abs(current_value - mean) > 0.5:
                    direction = "above" if current_value > mean else "below"
                    anomalies.append(Anomaly(
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_mean=mean,
                        baseline_std=std,
                        deviation_sigma=10.0,
                        direction=direction,
                        severity="high",
                        score=1.0,
                    ))
                continue

            deviation = (current_value - mean) / std
            if abs(deviation) >= self._threshold:
                direction = "above" if deviation > 0 else "below"
                abs_dev = abs(deviation)

                # Severity based on sigma deviation
                severity = "low"
                if abs_dev >= 5.0:
                    severity = "critical"
                elif abs_dev >= 4.0:
                    severity = "high"
                elif abs_dev >= 3.0:
                    severity = "medium"

                # Score normalized to 0-1
                score = min(1.0, abs_dev / 5.0)

                anomalies.append(Anomaly(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_mean=mean,
                    baseline_std=std,
                    deviation_sigma=abs_dev,
                    direction=direction,
                    severity=severity,
                    score=score,
                ))

        return anomalies


class AutoencoderDetector(AnomalyDetector):
    """Anomaly detection using a simple autoencoder on baseline metrics.

    Trains a single-hidden-layer autoencoder on baseline data. Anomalies
    are detected when the reconstruction error exceeds a threshold
    (derived from the baseline reconstruction error distribution).

    Requires numpy. Falls back gracefully if not available.

    Parameters
    ----------
    hidden_dim:
        Hidden layer dimension (default 4).
    error_percentile:
        Percentile of baseline errors above which data is anomalous
        (default 95.0).
    learning_rate:
        Learning rate for training (default 0.01).
    epochs:
        Number of training epochs (default 100).
    """

    def __init__(
        self,
        hidden_dim: int = 4,
        error_percentile: float = 95.0,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._error_percentile = error_percentile
        self._lr = learning_rate
        self._epochs = epochs
        self._trained = False
        self._threshold_error: float = 0.0
        self._encoder_w: Any = None
        self._encoder_b: Any = None
        self._decoder_w: Any = None
        self._decoder_b: Any = None
        self._mean: Any = None
        self._std: Any = None
        self._feature_names: list[str] = []

    def name(self) -> str:
        return "autoencoder"

    def detect(
        self,
        current_metrics: dict[str, float],
        baseline: list[dict[str, float]],
    ) -> list[Anomaly]:
        """Detect anomalies using autoencoder reconstruction error."""
        try:
            import numpy as np
        except ImportError:
            return []

        if len(baseline) < 20:
            return []

        # Determine feature names from first baseline sample
        if not self._feature_names:
            self._feature_names = sorted(baseline[0].keys())

        # Build baseline matrix
        X_baseline = self._build_matrix(baseline)
        if X_baseline is None or len(X_baseline) < 20:
            return []

        # Train if not yet trained
        if not self._trained:
            self._train(X_baseline)

        # Compute reconstruction error for current metrics
        current_vector = np.array([
            current_metrics.get(f, 0.0) for f in self._feature_names
        ], dtype=np.float64).reshape(1, -1)

        # Normalize
        if self._std is not None:
            safe_std = np.where(self._std < 1e-6, 1.0, self._std)
            current_norm = (current_vector - self._mean) / safe_std
        else:
            current_norm = current_vector

        # Forward pass
        hidden = np.tanh(current_norm @ self._encoder_w + self._encoder_b)
        reconstructed = hidden @ self._decoder_w + self._decoder_b
        error = float(np.mean((current_norm - reconstructed) ** 2))

        if error <= self._threshold_error:
            return []

        # Anomaly detected — find which metrics contribute most
        anomalies: list[Anomaly] = []
        per_feature_error = np.abs(current_norm[0] - reconstructed[0])

        for i, fname in enumerate(self._feature_names):
            if per_feature_error[i] > 1.5:  # Feature contributes to anomaly
                baseline_values = [s.get(fname, 0.0) for s in baseline if fname in s]
                mean_val = sum(baseline_values) / len(baseline_values)
                var_val = sum((v - mean_val) ** 2 for v in baseline_values) / len(baseline_values)
                std_val = math.sqrt(var_val) if var_val > 0 else 0.0

                current_val = current_metrics.get(fname, 0.0)
                if std_val > 1e-6:
                    deviation = abs(current_val - mean_val) / std_val
                else:
                    deviation = float(per_feature_error[i])

                direction = "above" if current_val > mean_val else "below"
                score = min(1.0, error / (self._threshold_error * 3))

                severity = "low"
                if score >= 0.8:
                    severity = "high"
                elif score >= 0.5:
                    severity = "medium"

                anomalies.append(Anomaly(
                    metric_name=fname,
                    current_value=current_val,
                    baseline_mean=mean_val,
                    baseline_std=std_val,
                    deviation_sigma=deviation,
                    direction=direction,
                    severity=severity,
                    score=score,
                ))

        # If no individual features flagged but overall error is high
        if not anomalies:
            score = min(1.0, error / (self._threshold_error * 3))
            anomalies.append(Anomaly(
                metric_name="_reconstruction_error",
                current_value=error,
                baseline_mean=self._threshold_error * 0.5,
                baseline_std=self._threshold_error * 0.2,
                deviation_sigma=(error - self._threshold_error) / max(0.01, self._threshold_error * 0.2),
                direction="above",
                severity="medium" if score < 0.7 else "high",
                score=score,
            ))

        return anomalies

    def _build_matrix(self, samples: list[dict[str, float]]) -> Any:
        """Build numpy matrix from sample dicts."""
        try:
            import numpy as np
        except ImportError:
            return None

        rows = []
        for sample in samples:
            row = [sample.get(f, 0.0) for f in self._feature_names]
            rows.append(row)

        return np.array(rows, dtype=np.float64)

    def _train(self, X: Any) -> None:
        """Train the autoencoder on baseline data."""
        try:
            import numpy as np
        except ImportError:
            return

        n_samples, n_features = X.shape

        # Normalize
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        safe_std = np.where(self._std < 1e-6, 1.0, self._std)
        X_norm = (X - self._mean) / safe_std

        # Initialize weights
        rng = np.random.RandomState(42)
        hidden_dim = min(self._hidden_dim, n_features)
        scale = 0.1

        self._encoder_w = rng.randn(n_features, hidden_dim) * scale
        self._encoder_b = np.zeros((1, hidden_dim))
        self._decoder_w = rng.randn(hidden_dim, n_features) * scale
        self._decoder_b = np.zeros((1, n_features))

        # Train with gradient descent
        for _epoch in range(self._epochs):
            # Forward
            hidden = np.tanh(X_norm @ self._encoder_w + self._encoder_b)
            reconstructed = hidden @ self._decoder_w + self._decoder_b
            error = X_norm - reconstructed

            # Backward
            d_decoder_w = -hidden.T @ error / n_samples
            d_decoder_b = -np.mean(error, axis=0, keepdims=True)

            d_hidden = -error @ self._decoder_w.T * (1 - hidden ** 2)
            d_encoder_w = X_norm.T @ d_hidden / n_samples
            d_encoder_b = np.mean(d_hidden, axis=0, keepdims=True)

            # Update
            self._encoder_w -= self._lr * d_encoder_w
            self._encoder_b -= self._lr * d_encoder_b
            self._decoder_w -= self._lr * d_decoder_w
            self._decoder_b -= self._lr * d_decoder_b

        # Compute threshold from baseline reconstruction errors
        hidden = np.tanh(X_norm @ self._encoder_w + self._encoder_b)
        reconstructed = hidden @ self._decoder_w + self._decoder_b
        errors = np.mean((X_norm - reconstructed) ** 2, axis=1)
        self._threshold_error = float(np.percentile(errors, self._error_percentile))

        self._trained = True

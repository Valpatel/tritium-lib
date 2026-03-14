# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Intelligence subsystem — scorer ABCs, anomaly detection, and implementations.

Exports:
    CorrelationScorer  — ABC for correlation scoring models
    StaticScorer       — Hand-tuned weighted scorer (baseline)
    LearnedScorer      — Trained logistic regression scorer (data-driven)
    ScorerResult       — Dataclass for scorer output (probability, confidence)
    CorrelationFeatures — Type alias for feature dicts (dict[str, float])
    FEATURE_NAMES      — Canonical list of expected feature keys
    DEFAULT_WEIGHTS    — Default static weights for each feature
    AnomalyDetector    — ABC for anomaly detectors
    SimpleThresholdDetector — Threshold-based anomaly detection (no deps)
    AutoencoderDetector — Autoencoder-based anomaly detection (numpy)
    Anomaly            — Anomaly dataclass
"""

from tritium_lib.intelligence.anomaly import (
    Anomaly,
    AnomalyDetector,
    AutoencoderDetector,
    SimpleThresholdDetector,
)
from tritium_lib.intelligence.scorer import (
    CorrelationFeatures,
    CorrelationScorer,
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    LearnedScorer,
    ScorerResult,
    StaticScorer,
)

__all__ = [
    "Anomaly",
    "AnomalyDetector",
    "AutoencoderDetector",
    "CorrelationFeatures",
    "CorrelationScorer",
    "DEFAULT_WEIGHTS",
    "FEATURE_NAMES",
    "LearnedScorer",
    "ScorerResult",
    "SimpleThresholdDetector",
    "StaticScorer",
]

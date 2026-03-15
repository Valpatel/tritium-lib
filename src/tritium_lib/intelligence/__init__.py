# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Intelligence subsystem — scorer ABCs, anomaly detection, learner base, model registry.

Exports:
    BaseLearner        — ABC for ML learners (train, predict, save, load, get_stats)
    ModelRegistry      — SQLite-backed versioned ML model storage
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

from tritium_lib.intelligence.feature_engineering import (
    EXTENDED_FEATURE_NAMES,
    build_extended_features,
    co_movement_score,
    device_type_match,
    source_diversity,
    time_similarity,
    wifi_probe_temporal_correlation,
)
from tritium_lib.intelligence.anomaly import (
    Anomaly,
    AnomalyDetector,
    AutoencoderDetector,
    SimpleThresholdDetector,
)
from tritium_lib.intelligence.base_learner import BaseLearner
from tritium_lib.intelligence.model_registry import ModelRegistry
from tritium_lib.intelligence.pattern_learning import (
    PatternLearner,
    PredictionResult,
    TrainingExample,
    PATTERN_FEATURES,
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
from tritium_lib.intelligence.rl_metrics import (
    PredictionRecord,
    RLMetrics,
    TrainingSnapshot,
)
from tritium_lib.intelligence.threat_model import (
    DEFAULT_SIGNAL_WEIGHTS,
    THREAT_THRESHOLDS,
    ThreatAssessment,
    ThreatLevel,
    ThreatModel,
    ThreatSignal,
    score_to_threat_level,
)

__all__ = [
    "EXTENDED_FEATURE_NAMES",
    "build_extended_features",
    "co_movement_score",
    "device_type_match",
    "source_diversity",
    "time_similarity",
    "wifi_probe_temporal_correlation",
    "Anomaly",
    "AnomalyDetector",
    "AutoencoderDetector",
    "BaseLearner",
    "CorrelationFeatures",
    "CorrelationScorer",
    "DEFAULT_SIGNAL_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "FEATURE_NAMES",
    "LearnedScorer",
    "ModelRegistry",
    "PATTERN_FEATURES",
    "PatternLearner",
    "PredictionRecord",
    "PredictionResult",
    "RLMetrics",
    "ScorerResult",
    "SimpleThresholdDetector",
    "StaticScorer",
    "THREAT_THRESHOLDS",
    "ThreatAssessment",
    "ThreatLevel",
    "ThreatModel",
    "ThreatSignal",
    "TrainingExample",
    "TrainingSnapshot",
    "score_to_threat_level",
]

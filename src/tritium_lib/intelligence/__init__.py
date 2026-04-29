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
from tritium_lib.intelligence.anomaly_engine import (
    AnomalyAlert,
    AnomalyEngine,
    KnownPattern,
    ZoneBaseline,
)
from tritium_lib.intelligence.base_learner import BaseLearner
from tritium_lib.intelligence.model_registry import ModelRegistry
from tritium_lib.intelligence.pattern_learning import (
    PatternLearner,
    PredictionResult,
    TrainingExample,
    PATTERN_FEATURES,
)
from tritium_lib.intelligence.behavioral_pattern_learner import (
    BehavioralPatternLearner,
    BehavioralProfile,
    DeviationResult,
    FrequentZone,
    LearnedRoute,
    LearnedSchedule,
    LearnedWaypoint,
    ScheduleObservation,
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
    FeatureAblation,
    PredictionRecord,
    RLMetrics,
    TrainingSnapshot,
)
from tritium_lib.intelligence.prediction_store import PredictionStore
from tritium_lib.intelligence.position_estimator import (
    estimate_from_multiple_anchors,
    estimate_from_single_anchor,
    rssi_to_distance as fusion_rssi_to_distance,
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
from tritium_lib.intelligence.threat_assessment import (
    ALL_INDICATOR_CATEGORIES,
    AreaAssessment,
    DEFAULT_INDICATOR_WEIGHTS,
    IndicatorCategory,
    ThreatAssessmentEngine,
    ThreatIndicator,
    ThreatMatrix,
    ThreatPrediction,
)
from tritium_lib.intelligence.zone_analysis import (
    ActivityPrediction,
    Hotspot,
    ZoneAnalyzer,
    ZoneComparison,
    ZoneEvent,
    ZoneReport,
)
from tritium_lib.intelligence.coverage_optimizer import (
    CoverageCell,
    CoverageGap,
    CoverageMap,
    PlacedSensor,
    PlacementResult,
    RedundancyZone,
    SensorSpec,
    SENSOR_RANGE_PROFILES,
    build_coverage_map,
    coverage_gaps,
    optimize_placement,
    redundancy_analysis,
)
from tritium_lib.intelligence.timeline_correlation import (
    CausalChain,
    EventSequence,
    FollowerResult,
    TemporalOverlap,
    TemporalPattern,
    TimelineCorrelator,
    TimelineEvent,
    PATTERN_ESCORT,
    PATTERN_MEETUP,
    PATTERN_SURVEILLANCE,
)
from tritium_lib.intelligence.access_patterns import (
    AccessAnomaly,
    AccessEvent,
    AccessPattern,
    AccessPatternAnalyzer,
    FrequencyReport,
    PiggybackAlert,
    TailgateAlert,
    detect_piggybacking,
    detect_tailgating,
    frequency_analysis,
)
from tritium_lib.intelligence.trajectory_predictor import (
    DestinationPrediction,
    FlockAware,
    LinearExtrapolation,
    Prediction,
    PredictionContext,
    PredictionModel,
    RoadConstrained,
    RoutineAware,
    TrajectoryPredictor,
)
from tritium_lib.intelligence.crowd_dynamics import (
    CrowdCluster,
    CrowdDynamicsAnalyzer,
    CrowdEvent as CrowdDynamicsEvent,
    CrowdEventType,
    CrowdState,
    DensityCell,
    DensityEstimator,
    DispersalDetector,
    FlowAnalyzer,
    FlowVector,
    FormationDetector,
)
from tritium_lib.intelligence.association_network import (
    Association,
    AssociationNetwork,
    AssociationSummary,
    EvidenceKind,
    KeyPlayer,
    NetworkAnalyzer as AssociationNetworkAnalyzer,
    TargetGroup,
    WeakLink,
    build_from_comint,
    build_from_tracking,
)
from tritium_lib.intelligence.behavior_profiler import (
    BehaviorChange,
    BehaviorProfile as LongTermBehaviorProfile,
    BehaviorProfiler,
    ChangeSeverity,
    DeviceDimension,
    Observation as ProfilerObservation,
    ProfileComparison,
    SocialDimension,
    SpatialDimension,
    SpatialStop,
    TargetRole,
    TemporalDimension,
    TransitCorridor,
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
    "AnomalyAlert",
    "AnomalyDetector",
    "AnomalyEngine",
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
    "FeatureAblation",
    "PredictionRecord",
    "PredictionResult",
    "PredictionStore",
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
    "KnownPattern",
    "ZoneBaseline",
    "score_to_threat_level",
    # Behavioral pattern learning
    "BehavioralPatternLearner",
    "BehavioralProfile",
    "DeviationResult",
    "FrequentZone",
    "LearnedRoute",
    "LearnedSchedule",
    "LearnedWaypoint",
    "ScheduleObservation",
    # Position estimation (sensor fusion)
    "estimate_from_multiple_anchors",
    "estimate_from_single_anchor",
    "fusion_rssi_to_distance",
    # Threat assessment engine
    "ALL_INDICATOR_CATEGORIES",
    "AreaAssessment",
    "DEFAULT_INDICATOR_WEIGHTS",
    "IndicatorCategory",
    "ThreatAssessmentEngine",
    "ThreatIndicator",
    "ThreatMatrix",
    "ThreatPrediction",
    # Zone analysis
    "ActivityPrediction",
    "Hotspot",
    "ZoneAnalyzer",
    "ZoneComparison",
    "ZoneEvent",
    "ZoneReport",
    # Coverage optimization
    "CoverageCell",
    "CoverageGap",
    "CoverageMap",
    "PlacedSensor",
    "PlacementResult",
    "RedundancyZone",
    "SensorSpec",
    "SENSOR_RANGE_PROFILES",
    "build_coverage_map",
    "coverage_gaps",
    "optimize_placement",
    "redundancy_analysis",
    # Timeline correlation
    "CausalChain",
    "EventSequence",
    "FollowerResult",
    "TemporalOverlap",
    "TemporalPattern",
    "TimelineCorrelator",
    "TimelineEvent",
    "PATTERN_ESCORT",
    "PATTERN_MEETUP",
    "PATTERN_SURVEILLANCE",
    # Access pattern analysis
    "AccessAnomaly",
    "AccessEvent",
    "AccessPattern",
    "AccessPatternAnalyzer",
    "FrequencyReport",
    "PiggybackAlert",
    "TailgateAlert",
    "detect_piggybacking",
    "detect_tailgating",
    "frequency_analysis",
    # Crowd dynamics analysis
    "CrowdCluster",
    "CrowdDynamicsAnalyzer",
    "CrowdDynamicsEvent",
    "CrowdEventType",
    "CrowdState",
    "DensityCell",
    "DensityEstimator",
    "DispersalDetector",
    "FlowAnalyzer",
    "FlowVector",
    "FormationDetector",
    # Trajectory prediction
    "DestinationPrediction",
    "FlockAware",
    "LinearExtrapolation",
    "Prediction",
    "PredictionContext",
    "PredictionModel",
    "RoadConstrained",
    "RoutineAware",
    "TrajectoryPredictor",
    # Association network analysis
    "Association",
    "AssociationNetwork",
    "AssociationNetworkAnalyzer",
    "AssociationSummary",
    "EvidenceKind",
    "KeyPlayer",
    "TargetGroup",
    "WeakLink",
    "build_from_comint",
    "build_from_tracking",
    # Behavior profiler (long-term behavioral profiling)
    "BehaviorChange",
    "BehaviorProfiler",
    "ChangeSeverity",
    "DeviceDimension",
    "LongTermBehaviorProfile",
    "ProfileComparison",
    "ProfilerObservation",
    "SocialDimension",
    "SpatialDimension",
    "SpatialStop",
    "TargetRole",
    "TemporalDimension",
    "TransitCorridor",
    # Geospatial segmentation (subpackage)
    "geospatial",
]

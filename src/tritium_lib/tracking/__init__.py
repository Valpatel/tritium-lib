# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.tracking — core target tracking and identity resolution.

This package contains the unified target registry, multi-strategy correlator,
Kalman predictor, geofence engine, trilateration, heatmaps, movement pattern
analysis, and dwell tracking.
"""

from .target_tracker import TargetTracker, TrackedTarget
from .correlator import TargetCorrelator, CorrelationRecord, start_correlator, stop_correlator
from .geofence import GeofenceEngine, GeoZone, GeoEvent
from .trilateration import TrilaterationEngine, PositionResult
from .target_history import TargetHistory, PositionRecord
from .target_reappearance import TargetReappearanceMonitor, ReappearanceEvent
from .target_prediction import PredictedPosition, predict_target, predict_all_targets
from .kalman_predictor import (
    KalmanState,
    kalman_update,
    predict_target_kalman,
    predict_all_targets_kalman,
    clear_kalman_state,
    get_kalman_state,
)
from .heatmap import HeatmapEngine, HeatmapEvent
from .movement_patterns import MovementPatternAnalyzer, MovementPattern
from .dossier import DossierStore, TargetDossier
from .correlation_strategies import (
    CorrelationStrategy,
    StrategyScore,
    SpatialStrategy,
    TemporalStrategy,
    SignalPatternStrategy,
    WiFiProbeStrategy,
    DossierStrategy,
)
from .ble_classifier import BLEClassifier, BLEClassification, CLASSIFICATION_LEVELS
from .vehicle_tracker import VehicleBehavior, VehicleTrackingManager, VEHICLE_CLASSES
from .convoy_detector import ConvoyDetector, TargetMotion
from .threat_scoring import ThreatScorer, BehaviorProfile
from .escalation import (
    ThreatRecord,
    THREAT_LEVELS,
    EscalationConfig,
    ClassifyResult,
    escalation_index,
    is_escalation,
    find_zone,
    classify_target,
    classify_all_targets,
)
from .patrol import PatrolRoute, PatrolAssignment, PatrolManager
from .network_analysis import NetworkAnalyzer, DeviceProfile, ProbeRecord, COMMON_SSIDS
from .proximity_monitor import ProximityMonitor
from .sensor_health_monitor import SensorHealthMonitor
try:
    from .obstacles import BuildingObstacles, _latlng_to_local, _segments_intersect
except ImportError:
    BuildingObstacles = None  # httpx not installed
    _latlng_to_local = None
    _segments_intersect = None
try:
    from .street_graph import StreetGraph
except ImportError:
    StreetGraph = None  # httpx not installed — street graph unavailable

__all__ = [
    # Core tracker
    "TargetTracker",
    "TrackedTarget",
    # Correlator
    "TargetCorrelator",
    "CorrelationRecord",
    "start_correlator",
    "stop_correlator",
    # Geofence
    "GeofenceEngine",
    "GeoZone",
    "GeoEvent",
    # Kalman
    "KalmanState",
    "kalman_update",
    "predict_target_kalman",
    "predict_all_targets_kalman",
    "clear_kalman_state",
    "get_kalman_state",
    # Trilateration
    "TrilaterationEngine",
    "PositionResult",
    # History
    "TargetHistory",
    "PositionRecord",
    # Reappearance
    "TargetReappearanceMonitor",
    "ReappearanceEvent",
    # Prediction
    "PredictedPosition",
    "predict_target",
    "predict_all_targets",
    # Heatmap
    "HeatmapEngine",
    "HeatmapEvent",
    # Movement patterns
    "MovementPatternAnalyzer",
    "MovementPattern",
    # Dossier
    "DossierStore",
    "TargetDossier",
    # Strategies
    "CorrelationStrategy",
    "StrategyScore",
    "SpatialStrategy",
    "TemporalStrategy",
    "SignalPatternStrategy",
    "WiFiProbeStrategy",
    "DossierStrategy",
    # BLE classifier
    "BLEClassifier",
    "BLEClassification",
    "CLASSIFICATION_LEVELS",
    # Vehicle tracker
    "VehicleBehavior",
    "VehicleTrackingManager",
    "VEHICLE_CLASSES",
    # Convoy detector
    "ConvoyDetector",
    "TargetMotion",
    # Threat scoring
    "ThreatScorer",
    "BehaviorProfile",
    # Escalation
    "ThreatRecord",
    "THREAT_LEVELS",
    "EscalationConfig",
    "ClassifyResult",
    "escalation_index",
    "is_escalation",
    "find_zone",
    "classify_target",
    "classify_all_targets",
    # Patrol
    "PatrolRoute",
    "PatrolAssignment",
    "PatrolManager",
    # Network analysis
    "NetworkAnalyzer",
    "DeviceProfile",
    "ProbeRecord",
    "COMMON_SSIDS",
    # Proximity monitor
    "ProximityMonitor",
    # Sensor health monitor
    "SensorHealthMonitor",
    # Building obstacles
    "BuildingObstacles",
    "_latlng_to_local",
    "_segments_intersect",
    # Street graph
    "StreetGraph",
]

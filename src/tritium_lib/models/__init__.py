# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared data models for the Tritium ecosystem.

These models are the contract between tritium-sc and tritium-edge.
Any device that speaks the Tritium protocol uses these types.
"""

from .device import Device, DeviceGroup, DeviceHeartbeat, DeviceCapabilities
from .capability import (
    CapabilityAdvertisement,
    CapabilityType,
    DeviceCapability,
)
from .command import Command, CommandType, CommandStatus
from .firmware import FirmwareMeta, OTAJob, OTAStatus
from .sensor import SensorReading
from .ble import (
    BleDevice,
    BleSighting,
    BlePresence,
    BlePresenceMap,
    triangulate_position,
    set_node_positions,
)
from .fleet import (
    FleetNode,
    FleetStatus,
    NodeEvent,
    NodeStatus,
    fleet_health_score,
)
from .gis import (
    TileCoord,
    TileBounds,
    MapLayer,
    MapLayerType,
    MapRegion,
    TilePackage,
    OfflineRegion,
    lat_lon_to_tile,
    tile_to_lat_lon,
    tiles_in_bounds,
)
from .seed import (
    SeedFile,
    SeedManifest,
    SeedPackage,
    SeedStatus,
    SeedTransfer,
    SeedTransferStatus,
)
from .acoustic_modem import (
    AcousticFrame,
    AcousticConfig,
    AcousticChannelStats,
    ModulationType,
)
from .mesh import (
    MeshNode,
    MeshRoute,
    MeshEdge,
    MeshTopology,
    MeshMessage,
    MeshMessageStatus,
)
from .config import (
    ConfigDrift,
    ConfigDriftSeverity,
    DeviceConfig,
    FleetConfigStatus,
    MapDefaults,
    NotificationPrefs,
    ScanIntervals,
    TritiumSystemConfig,
    compute_config_drift,
    compute_fleet_config_status,
    classify_drift_severity,
)
from .cot import (
    CotEvent,
    CotPoint,
    CotDetail,
    CotContact,
    cot_to_xml,
    xml_to_cot,
    COT_FRIENDLY_GROUND_UNIT,
    COT_FRIENDLY_UAV,
    COT_FRIENDLY_GROUND_SENSOR,
    COT_HOSTILE_GROUND_UNIT,
)
from .provision import (
    ProvisionData,
    ProvisionRecord,
    ProvisionSource,
    ProvisionState,
    FleetProvisionStatus,
    compute_provision_status,
    validate_provision_data,
)
from .diagnostics import (
    CrashInfo,
    DiagLogEntry,
    DiagLogBatch,
    DiagLogSummary,
    HeapTrend,
    I2cSlaveHealth,
    MeshPeer,
    analyze_heap_trends,
    summarize_diag_log,
)
from .topology import (
    NetworkLink,
    NetworkNode,
    NodeRole,
    PeerQuality,
    FleetTopology,
    ConnectivityReport,
    build_topology,
    build_fleet_topology_from_mesh,
    analyze_connectivity,
)
from .correlation import (
    CorrelationType,
    CorrelationEvent,
    CorrelationSummary,
    classify_correlation_severity,
    summarize_correlations,
)
from .transport import (
    TransportType,
    TransportState,
    TransportMetrics,
    TransportPreference,
    NodeTransportStatus,
    select_best_transport,
    transport_summary,
)
from .alert import (
    Alert,
    AlertDelivery,
    AlertHistory,
    AlertSeverity,
    WebhookConfig,
    classify_alert_severity,
    summarize_alerts,
)
from .timeseries import (
    TimeSeriesPoint,
    TimeSeries,
    FleetTimeSeries,
    PagedResult,
)
from .trilateration import (
    AnchorPoint,
    PositionEstimate,
    RSSIFilter,
    rssi_to_distance,
    trilaterate_2d,
    estimate_position,
)
from .meshtastic import (
    MeshtasticConnectionType,
    MeshtasticNode,
    MeshtasticMessage,
    MeshtasticWaypoint,
    MeshtasticStatus,
)
from .mesh_node_extended import (
    MeshNodeExtended,
    MeshNodePosition,
    MeshNodeDeviceMetrics,
    MeshNodeEnvironment,
    MeshNodeRadioMetrics,
)
from .wifi import (
    WiFiProbeRequest,
    WiFiNetwork,
    WiFiFingerprint,
    WiFiNetworkType,
)
from .camera import (
    CameraSourceType,
    CameraFrameFormat,
    CameraPosition,
    CameraSource,
    CameraFrame,
    BoundingBox,
    CameraDetection,
)
from .dossier import (
    DossierSignal,
    DossierEnrichment,
    PositionRecord,
    TargetDossier,
)
from .reid import (
    ReIDEmbedding,
    ReIDMatch,
)
from .radio import (
    RadioMode,
    RadioSchedulerConfig,
    RadioSchedulerStatus,
    CameraMqttConfig,
    CameraMqttStats,
)
from .drone import (
    DroneCommand,
    DroneMission,
    DroneRegistration,
    DroneState,
    DroneTelemetry,
    DroneType,
    Waypoint,
)
from .ais import (
    AISPosition,
    AISVessel,
    VesselType,
    NavigationStatus,
    ADSBPosition,
    ADSBFlight,
    FlightCategory,
    SquawkCode,
)
from .lpr import (
    PlateAlert,
    PlateColor,
    PlateDetection,
    PlateRecord,
    PlateRegion,
    PlateWatchEntry,
    PlateWatchlist,
    LPRStats,
)
from .acoustic_event import (
    AcousticEvent,
    AcousticEventType,
    AcousticSensorConfig,
    AcousticSeverity,
    AcousticSpectrum,
    AcousticStats,
    classify_event_severity,
)
from .acoustic_intelligence import (
    AcousticObserver,
    AcousticTrilateration,
    AudioFeatureVector,
    SoundClassification,
    SoundSignature,
    acoustic_trilaterate,
    SPEED_OF_SOUND_MPS,
)
from .acoustic_tdoa import (
    TDoAObservation,
    TDoAResult,
    compute_tdoa_position,
    SPEED_OF_SOUND_MPS as TDOA_SPEED_OF_SOUND_MPS,
)
from .behavior import (
    AnomalySeverity,
    AnomalyType,
    BehaviorAnomaly,
    BehaviorPattern,
    BehaviorType,
    CorrelationScore,
    PositionSample,
    TargetRoutine,
    classify_anomaly_severity,
    compute_correlation_score,
)
from .terrain import (
    CoverageAnalysis,
    CoverageCell,
    ElevationPoint,
    ElevationProfile,
    SensorPlacement,
    TerrainType,
    WeatherConditions,
    estimate_signal_strength,
    free_space_path_loss_db,
    terrain_path_loss_db,
)
from .federation import (
    ConnectionState,
    FederatedSite,
    FederationMessage,
    FederationMessageType,
    SharedTarget,
    SharePolicy,
    SiteConnection,
    SiteRole,
    federation_topic,
    is_message_expired,
)
from .notification_rules import (
    DEFAULT_RULES,
    NotificationChannel,
    NotificationRule,
    NotificationSeverity,
)
from .alert_rules import (
    AlertCondition,
    AlertRule,
    AlertTrigger,
    ConditionOperator,
    DEFAULT_ALERT_RULES,
)
from .event_schema import (
    ALL_EVENT_TYPES,
    EventDomain,
    TritiumEvent,
    validate_event_type,
    get_event_schema,
    list_event_types,
)
from .summary import (
    FleetSummary,
    SystemSummary,
    TargetCounts,
)
from .analytics import (
    DailyAnalytics,
    DeviceActivity,
)
from .analytics_dashboard import (
    DEFAULT_WIDGETS,
    DashboardWidget,
    WidgetConfig,
    WidgetType,
)
from .export import (
    ExportFormat,
    ExportManifest,
    ExportPackage,
    ExportScope,
    ExportSection,
    ExportSectionType,
    ImportResult,
    create_export_manifest,
    validate_import_compatibility,
)
from .mission import (
    GeofenceZone,
    Mission,
    MissionObjective,
    MissionStatus,
    MissionType,
)
from .report import (
    ClassificationLevel,
    IntelligenceReport,
    ReportFinding,
    ReportRecommendation,
    ReportStatus,
)
from .forensics import (
    EvidenceItem,
    ForensicReconstruction,
    GeoBounds,
    IncidentClassification,
    IncidentFinding,
    IncidentRecommendation,
    IncidentReport,
    ReconstructionStatus,
    SensorCoverage,
    TargetTimeline,
    TimeRange,
)
from .sensor_config import (
    MountingType,
    SensorArray,
    SensorPlacement as SensorPlacementConfig,
    SensorPosition,
    SensorStatus,
    SensorType,
)
from .operational import (
    OperationalObjective,
    OperationalPeriod,
    OperationalPhase,
    WeatherInfo,
)
from .comms import (
    AuthType,
    ChannelAuth,
    ChannelHealth,
    ChannelInventory,
    ChannelStatus,
    ChannelType,
    CommChannel,
    select_best_channel,
    summarize_channels,
)
from .scenario import (
    ActorAlliance,
    ActorType,
    ScenarioActor,
    ScenarioEvent,
    ScenarioEventType,
    ScenarioObjective,
    ScenarioStatus,
    TacticalScenario,
)
from .user import (
    Permission,
    ROLE_PERMISSIONS,
    User,
    UserRole,
    UserSession,
)
from .template import (
    BRIEFING_TEMPLATE,
    BUILTIN_TEMPLATES,
    INVESTIGATION_TEMPLATE,
    ReportFormat,
    ReportTemplate,
    SITREP_TEMPLATE,
    TemplateSection,
    TemplateSectionType,
    TemplateVariable,
)
from .training import (
    ClassificationTrainingData,
    CorrelationTrainingData,
    DecisionType,
    FeedbackRecord,
    TrainingExample,
)
from .swarm import (
    SwarmCommand,
    SwarmCommandType,
    SwarmFormation,
    SwarmFormationType,
    SwarmMember,
    SwarmMemberStatus,
    SwarmRole,
    SwarmStatus,
)
from .environment import (
    EnvironmentReading,
    EnvironmentSnapshot,
    EnvironmentSource,
)
from .deployment import (
    DeployedService,
    DeploymentConfig,
    ServiceName,
    ServiceState,
    ServiceStatus,
    SystemRequirements,
)
from .movement_analytics import (
    ActivityPeriod,
    DwellTime,
    FleetMetrics,
    MovementAnalytics,
)
from .feature_vector import (
    AggregatedFeatures,
    ClassificationFeedback,
    EdgeIntelligenceMetrics,
    FeatureSource,
    FeatureVector,
)
from .floorplan import (
    BuildingOccupancy,
    FloorPlan,
    FloorPlanBounds,
    FloorPlanStatus,
    GeoAnchor,
    IndoorPosition,
    PolygonPoint,
    Room,
    RoomOccupancy,
    RoomType,
    WiFiRSSIFingerprint,
)
from .ble_interrogation import (
    BleDeviceProfile as BleGATTProfile,
    BleGATTCharacteristic,
    BleGATTService,
    BleInterrogationQueue,
    BleInterrogationResult,
    STANDARD_SERVICE_UUIDS,
    classify_device_from_profile,
    lookup_service_name,
)
from .visualization import (
    AllianceColor,
    CoverageVolume,
    Scene3DConfig,
    SensorVolumeType,
    TimelineConfig,
    TrajectoryRibbon,
)
from .fleet_ops import (
    BUILTIN_TEMPLATES as FLEET_BUILTIN_TEMPLATES,
    ConfigTemplate,
    ConfigTemplateName,
    CoveragePoint,
    DeviceUptimeRecord,
    FleetAnalyticsSnapshot,
    FleetCommand,
    FleetCommandStatus,
    FleetCommandType,
    SightingRateRecord,
)
from .tactical_event import (
    EventPosition,
    TacticalEvent,
    TacticalEventType,
    TacticalSeverity,
    filter_events,
)
from .prediction import (
    PredictedPosition,
    TargetPrediction,
)
from .device_capability_matrix import (
    CapabilityMatrix,
    DeviceCapabilityEntry,
)
from .tactical_situation import (
    AmyStatus,
    FleetHealth as TacticalFleetHealth,
    TargetCountsSummary,
    TacticalSituation,
    ThreatLevel,
)
from .gpx import (
    GPXDocument,
    GPXRoute,
    GPXTrack,
    GPXWaypoint,
)
from .collaboration import (
    ChatMessageType,
    DrawingType,
    MapDrawing,
    OperatorAction,
    OperatorChatMessage,
    SharedWorkspace,
    WorkspaceEvent,
    WorkspaceEventType,
)
from .correlation_evidence import (
    CorrelationEvidence,
    EvidenceType,
    build_handoff_evidence,
    build_spatial_evidence,
    build_visual_evidence,
    compute_composite_confidence,
    make_pair_id,
)
from .autonomous import (
    AutonomousDecision,
    AutonomousDecisionLog,
    AutonomousDecisionType,
    AutonomousTrigger,
    EdgeAlertRule,
    OverrideState,
)
from .confidence import (
    ConfidenceModel,
    DEFAULT_HALF_LIVES,
    SourceType,
)
from .sensor_health import (
    SensorAlert,
    SensorArrayHealth,
    SensorBaseline,
    SensorHealthMetrics,
    SensorHealthStatus,
    classify_sensor_health,
)
from .velocity import (
    VelocityProfile,
    compute_anomaly_score,
)
from .vehicle import (
    VehicleTrack,
    compute_heading,
    compute_speed_mph,
    compute_suspicious_score,
    heading_to_label,
)
from .device_lifecycle import (
    DeviceLifecycleEvent,
    DeviceLifecycleStatus,
    DeviceProvisioningConfig,
    DeviceState,
    FleetLifecycleSummary,
    VALID_TRANSITIONS,
    is_valid_transition,
)
from .quick_action import (
    QuickAction,
    QuickActionLog,
    QuickActionType,
)
from .intelligence_package import (
    ChainOfCustody,
    EvidenceType as IntelEvidenceType,
    IntelClassification,
    IntelligencePackage,
    PackageDossier,
    PackageEvent,
    PackageEvidence,
    PackageImportResult,
    PackageStatus,
    PackageTarget,
    create_intelligence_package,
    validate_package_import,
)
from .proximity import (
    AlliancePair,
    ProximityAlert,
    ProximityAlertType,
    ProximityRule,
    ProximitySeverity,
    classify_proximity_severity,
    DEFAULT_PROXIMITY_RULES,
)
from .benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkUnit,
)
from .dwell import (
    DwellEvent,
    DwellSeverity,
    DwellState,
    DWELL_RADIUS_M,
    DWELL_THRESHOLD_S,
    classify_dwell_severity,
)
from .personality import (
    CommanderPersonality,
    PRESET_PERSONALITIES,
    PATROL_PERSONALITY,
    BATTLE_PERSONALITY,
    STEALTH_PERSONALITY,
    OBSERVER_PERSONALITY,
)
from .geofence_event import (
    GeofenceEvent,
)
from .camera_link import (
    CameraDetectionLink,
    CameraLinkSummary,
    FramePosition,
)
from .target_group import (
    TargetGroup,
    TargetGroupSummary,
)
from .transition import (
    TransitionEvent,
    TransitionHistory,
    TransitionType,
)
from .acoustic_training import (
    AcousticTrainingExample,
    AcousticTrainingSet,
    TrainingSource,
)
from .pattern import (
    BehaviorPattern as LearnedBehaviorPattern,
    CoPresenceRelationship,
    DeviationType,
    LocationCluster,
    PatternAlert,
    PatternAnomaly,
    PatternStatus,
    PatternType,
    TimeSlot,
    compute_temporal_correlation,
    detect_time_regularity,
)
from .clustering import (
    BehaviorCluster,
    ClusterSummary,
    CommonPattern,
    FormationType,
)
from .daily_pattern import DailyPattern

__all__ = [
    # Deployment
    "DeployedService",
    "DeploymentConfig",
    "ServiceName",
    "ServiceState",
    "ServiceStatus",
    "SystemRequirements",
    "Device",
    "DeviceGroup",
    "DeviceHeartbeat",
    "DeviceCapabilities",
    # Capability advertisement
    "CapabilityAdvertisement",
    "CapabilityType",
    "DeviceCapability",
    "Command",
    "CommandType",
    "CommandStatus",
    "FirmwareMeta",
    "OTAJob",
    "OTAStatus",
    "SensorReading",
    "BleDevice",
    "BleSighting",
    "BlePresence",
    "BlePresenceMap",
    "triangulate_position",
    "set_node_positions",
    "FleetNode",
    "FleetStatus",
    "NodeEvent",
    "NodeStatus",
    "fleet_health_score",
    # GIS
    "TileCoord",
    "TileBounds",
    "MapLayer",
    "MapLayerType",
    "MapRegion",
    "TilePackage",
    "OfflineRegion",
    "lat_lon_to_tile",
    "tile_to_lat_lon",
    "tiles_in_bounds",
    # Seed / replication
    "SeedFile",
    "SeedManifest",
    "SeedPackage",
    "SeedStatus",
    "SeedTransfer",
    "SeedTransferStatus",
    # Acoustic modem
    "AcousticFrame",
    "AcousticConfig",
    "AcousticChannelStats",
    "ModulationType",
    # Mesh networking
    "MeshNode",
    "MeshRoute",
    "MeshEdge",
    "MeshTopology",
    "MeshMessage",
    "MeshMessageStatus",
    # CoT models
    "CotEvent",
    "CotPoint",
    "CotDetail",
    "CotContact",
    "cot_to_xml",
    "xml_to_cot",
    "COT_FRIENDLY_GROUND_UNIT",
    "COT_FRIENDLY_UAV",
    "COT_FRIENDLY_GROUND_SENSOR",
    "COT_HOSTILE_GROUND_UNIT",
    # Config sync
    "ConfigDrift",
    "ConfigDriftSeverity",
    "DeviceConfig",
    "FleetConfigStatus",
    "MapDefaults",
    "NotificationPrefs",
    "ScanIntervals",
    "TritiumSystemConfig",
    "compute_config_drift",
    "compute_fleet_config_status",
    "classify_drift_severity",
    # Provisioning
    "ProvisionData",
    "ProvisionRecord",
    "ProvisionSource",
    "ProvisionState",
    "FleetProvisionStatus",
    "compute_provision_status",
    "validate_provision_data",
    # Alert / webhook
    "Alert",
    "AlertDelivery",
    "AlertHistory",
    "AlertSeverity",
    "WebhookConfig",
    "classify_alert_severity",
    "summarize_alerts",
    # Fleet topology
    "NetworkLink",
    "NetworkNode",
    "NodeRole",
    "PeerQuality",
    "FleetTopology",
    "ConnectivityReport",
    "build_topology",
    "build_fleet_topology_from_mesh",
    "analyze_connectivity",
    # Event correlation
    "CorrelationType",
    "CorrelationEvent",
    "CorrelationSummary",
    "classify_correlation_severity",
    "summarize_correlations",
    # Transport negotiation
    "TransportType",
    "TransportState",
    "TransportMetrics",
    "TransportPreference",
    "NodeTransportStatus",
    "select_best_transport",
    "transport_summary",
    # Diagnostic log
    "CrashInfo",
    "DiagLogEntry",
    "DiagLogBatch",
    "DiagLogSummary",
    "HeapTrend",
    "I2cSlaveHealth",
    "MeshPeer",
    "analyze_heap_trends",
    "summarize_diag_log",
    # Time series & pagination
    "TimeSeriesPoint",
    "TimeSeries",
    "FleetTimeSeries",
    "PagedResult",
    # BLE trilateration
    "AnchorPoint",
    "PositionEstimate",
    "RSSIFilter",
    "rssi_to_distance",
    "trilaterate_2d",
    "estimate_position",
    # Meshtastic BLE bridge
    "MeshtasticConnectionType",
    "MeshtasticNode",
    "MeshtasticMessage",
    "MeshtasticWaypoint",
    "MeshtasticStatus",
    # Meshtastic extended (real hardware)
    "MeshNodeExtended",
    "MeshNodePosition",
    "MeshNodeDeviceMetrics",
    "MeshNodeEnvironment",
    "MeshNodeRadioMetrics",
    # WiFi passive fingerprinting
    "WiFiProbeRequest",
    "WiFiNetwork",
    "WiFiFingerprint",
    "WiFiNetworkType",
    # Camera sources & detection
    "CameraSourceType",
    "CameraFrameFormat",
    "CameraPosition",
    "CameraSource",
    "CameraFrame",
    "BoundingBox",
    "CameraDetection",
    # Target Dossier
    "DossierSignal",
    "DossierEnrichment",
    "PositionRecord",
    "TargetDossier",
    # ReID (re-identification)
    "ReIDEmbedding",
    "ReIDMatch",
    # Radio scheduler (BLE/WiFi TDM)
    "RadioMode",
    "RadioSchedulerConfig",
    "RadioSchedulerStatus",
    "CameraMqttConfig",
    "CameraMqttStats",
    # Drone/UAV integration
    "DroneCommand",
    "DroneMission",
    "DroneRegistration",
    "DroneState",
    "DroneTelemetry",
    "DroneType",
    "Waypoint",
    # AIS/ADS-B maritime & aviation
    "AISPosition",
    "AISVessel",
    "VesselType",
    "NavigationStatus",
    "ADSBPosition",
    "ADSBFlight",
    "FlightCategory",
    "SquawkCode",
    # License Plate Recognition
    "PlateAlert",
    "PlateColor",
    "PlateDetection",
    "PlateRecord",
    "PlateRegion",
    "PlateWatchEntry",
    "PlateWatchlist",
    "LPRStats",
    # Acoustic event classification
    "AcousticEvent",
    "AcousticEventType",
    "AcousticSensorConfig",
    "AcousticSeverity",
    "AcousticSpectrum",
    "AcousticStats",
    "classify_event_severity",
    # Acoustic intelligence (ML classification + localization)
    "AcousticObserver",
    "AcousticTrilateration",
    "AudioFeatureVector",
    "SoundClassification",
    "SoundSignature",
    "acoustic_trilaterate",
    "SPEED_OF_SOUND_MPS",
    # Acoustic TDoA (Time Difference of Arrival)
    "TDoAObservation",
    "TDoAResult",
    "compute_tdoa_position",
    "TDOA_SPEED_OF_SOUND_MPS",
    # Behavioral pattern recognition
    "AnomalySeverity",
    "AnomalyType",
    "BehaviorAnomaly",
    "BehaviorPattern",
    "BehaviorType",
    "CorrelationScore",
    "PositionSample",
    "TargetRoutine",
    "classify_anomaly_severity",
    "compute_correlation_score",
    # Terrain analysis & RF propagation
    "CoverageAnalysis",
    "CoverageCell",
    "ElevationPoint",
    "ElevationProfile",
    "SensorPlacement",
    "TerrainType",
    "WeatherConditions",
    "estimate_signal_strength",
    "free_space_path_loss_db",
    "terrain_path_loss_db",
    # Multi-site federation
    "ConnectionState",
    "FederatedSite",
    "FederationMessage",
    "FederationMessageType",
    "SharedTarget",
    "SharePolicy",
    "SiteConnection",
    "SiteRole",
    "federation_topic",
    "is_message_expired",
    # Notification rules
    "DEFAULT_RULES",
    "NotificationChannel",
    "NotificationRule",
    "NotificationSeverity",
    # Alert rules
    "AlertCondition",
    "AlertRule",
    "AlertTrigger",
    "ConditionOperator",
    "DEFAULT_ALERT_RULES",
    # Event schemas
    "ALL_EVENT_TYPES",
    "EventDomain",
    "TritiumEvent",
    "validate_event_type",
    "get_event_schema",
    "list_event_types",
    # System summary
    "FleetSummary",
    "SystemSummary",
    "TargetCounts",
    # Daily analytics
    "DailyAnalytics",
    "DeviceActivity",
    # Analytics dashboard widgets
    "DEFAULT_WIDGETS",
    "DashboardWidget",
    "WidgetConfig",
    "WidgetType",
    # Export / import
    "ExportFormat",
    "ExportManifest",
    "ExportPackage",
    "ExportScope",
    "ExportSection",
    "ExportSectionType",
    "ImportResult",
    "create_export_manifest",
    "validate_import_compatibility",
    # Mission management
    "GeofenceZone",
    "Mission",
    "MissionObjective",
    "MissionStatus",
    "MissionType",
    # Intelligence reports
    "ClassificationLevel",
    "IntelligenceReport",
    "ReportFinding",
    "ReportRecommendation",
    "ReportStatus",
    # Sensor placement & configuration
    "MountingType",
    "SensorArray",
    "SensorPlacementConfig",
    "SensorPosition",
    "SensorStatus",
    "SensorType",
    # Operational periods
    "OperationalObjective",
    "OperationalPeriod",
    "OperationalPhase",
    "WeatherInfo",
    # Communication channels
    "AuthType",
    "ChannelAuth",
    "ChannelHealth",
    "ChannelInventory",
    "ChannelStatus",
    "ChannelType",
    "CommChannel",
    "select_best_channel",
    "summarize_channels",
    # Tactical scenarios
    "ActorAlliance",
    "ActorType",
    "ScenarioActor",
    "ScenarioEvent",
    "ScenarioEventType",
    "ScenarioObjective",
    "ScenarioStatus",
    "TacticalScenario",
    # User / session management
    "Permission",
    "ROLE_PERMISSIONS",
    "User",
    "UserRole",
    "UserSession",
    # Report templates
    "BRIEFING_TEMPLATE",
    "BUILTIN_TEMPLATES",
    "INVESTIGATION_TEMPLATE",
    "ReportFormat",
    "ReportTemplate",
    "SITREP_TEMPLATE",
    "TemplateSection",
    "TemplateSectionType",
    "TemplateVariable",
    # ML training data
    "ClassificationTrainingData",
    "CorrelationTrainingData",
    "DecisionType",
    "FeedbackRecord",
    "TrainingExample",
    # Swarm coordination
    "SwarmCommand",
    "SwarmCommandType",
    "SwarmFormation",
    "SwarmFormationType",
    "SwarmMember",
    "SwarmMemberStatus",
    "SwarmRole",
    "SwarmStatus",
    # Environment sensors
    "EnvironmentReading",
    "EnvironmentSnapshot",
    "EnvironmentSource",
    # BLE GATT interrogation
    "BleGATTProfile",
    "BleGATTCharacteristic",
    "BleGATTService",
    "BleInterrogationQueue",
    "BleInterrogationResult",
    "STANDARD_SERVICE_UUIDS",
    "classify_device_from_profile",
    "lookup_service_name",
    # Edge intelligence / feature vectors
    "AggregatedFeatures",
    "ClassificationFeedback",
    "EdgeIntelligenceMetrics",
    "FeatureSource",
    "FeatureVector",
    # Movement analytics
    "ActivityPeriod",
    "DwellTime",
    "FleetMetrics",
    "MovementAnalytics",
    # Indoor floor plans & spatial intelligence
    "BuildingOccupancy",
    "FloorPlan",
    "FloorPlanBounds",
    "FloorPlanStatus",
    "GeoAnchor",
    "IndoorPosition",
    "PolygonPoint",
    "Room",
    "RoomOccupancy",
    "RoomType",
    "WiFiRSSIFingerprint",
    # 3D visualization configuration
    "AllianceColor",
    "CoverageVolume",
    "Scene3DConfig",
    "SensorVolumeType",
    "TimelineConfig",
    "TrajectoryRibbon",
    # Fleet operations
    "FLEET_BUILTIN_TEMPLATES",
    "ConfigTemplate",
    "ConfigTemplateName",
    "CoveragePoint",
    "DeviceUptimeRecord",
    "FleetAnalyticsSnapshot",
    "FleetCommand",
    "FleetCommandStatus",
    "FleetCommandType",
    "SightingRateRecord",
    # Dwell detection
    "DwellEvent",
    "DwellSeverity",
    "DwellState",
    "DWELL_RADIUS_M",
    "DWELL_THRESHOLD_S",
    "classify_dwell_severity",
    # Commander personality
    "CommanderPersonality",
    "PRESET_PERSONALITIES",
    "PATROL_PERSONALITY",
    "BATTLE_PERSONALITY",
    "STEALTH_PERSONALITY",
    "OBSERVER_PERSONALITY",
    # Geofence crossing events
    "GeofenceEvent",
    # Camera-to-target detection links
    "CameraDetectionLink",
    "CameraLinkSummary",
    "FramePosition",
    # Target groups
    "TargetGroup",
    "TargetGroupSummary",
    # State transitions
    "TransitionEvent",
    "TransitionHistory",
    "TransitionType",
    # Acoustic ML training data
    "AcousticTrainingExample",
    "AcousticTrainingSet",
    "TrainingSource",
    # Behavioral pattern learning
    "LearnedBehaviorPattern",
    "CoPresenceRelationship",
    "DeviationType",
    "LocationCluster",
    "PatternAlert",
    "PatternAnomaly",
    "PatternStatus",
    "PatternType",
    "TimeSlot",
    "compute_temporal_correlation",
    "detect_time_regularity",
    # Intelligence packages
    "ChainOfCustody",
    "IntelEvidenceType",
    "IntelClassification",
    "IntelligencePackage",
    "PackageDossier",
    "PackageEvent",
    "PackageEvidence",
    "PackageImportResult",
    "PackageStatus",
    "PackageTarget",
    "create_intelligence_package",
    "validate_package_import",
    # Proximity alerts
    "AlliancePair",
    "ProximityAlert",
    "ProximityAlertType",
    "ProximityRule",
    "ProximitySeverity",
    "classify_proximity_severity",
    "DEFAULT_PROXIMITY_RULES",
    # Quick tactical actions
    "QuickAction",
    "QuickActionLog",
    "QuickActionType",
    # Benchmark results
    "BenchmarkResult",
    "BenchmarkSuite",
    "BenchmarkUnit",
    # Velocity profiling
    "VelocityProfile",
    "compute_anomaly_score",
    # Vehicle tracking
    "VehicleTrack",
    "compute_heading",
    "compute_speed_mph",
    "compute_suspicious_score",
    "heading_to_label",
    # Confidence decay
    "ConfidenceModel",
    "DEFAULT_HALF_LIVES",
    "SourceType",
    # Sensor health monitoring
    "SensorAlert",
    "SensorArrayHealth",
    "SensorBaseline",
    "SensorHealthMetrics",
    "SensorHealthStatus",
    "classify_sensor_health",
    # Autonomous edge decisions
    "AutonomousDecision",
    "AutonomousDecisionLog",
    "AutonomousDecisionType",
    "AutonomousTrigger",
    "EdgeAlertRule",
    "OverrideState",
    # Correlation evidence
    "CorrelationEvidence",
    "EvidenceType",
    "build_handoff_evidence",
    "build_spatial_evidence",
    "build_visual_evidence",
    "compute_composite_confidence",
    "make_pair_id",
    # Tactical events
    "EventPosition",
    "TacticalEvent",
    "TacticalEventType",
    "TacticalSeverity",
    "filter_events",
    # Target prediction
    "PredictedPosition",
    "TargetPrediction",
    # Device capability matrix
    "CapabilityMatrix",
    "DeviceCapabilityEntry",
    # Collaboration
    "ChatMessageType",
    "DrawingType",
    "MapDrawing",
    "OperatorAction",
    "OperatorChatMessage",
    "SharedWorkspace",
    "WorkspaceEvent",
    "WorkspaceEventType",
    # Forensics
    "EvidenceItem",
    "ForensicReconstruction",
    "GeoBounds",
    "IncidentClassification",
    "IncidentFinding",
    "IncidentRecommendation",
    "IncidentReport",
    "ReconstructionStatus",
    "SensorCoverage",
    "TargetTimeline",
    "TimeRange",
    # Device lifecycle management
    "DeviceLifecycleEvent",
    "DeviceLifecycleStatus",
    "DeviceProvisioningConfig",
    "DeviceState",
    "FleetLifecycleSummary",
    "VALID_TRANSITIONS",
    "is_valid_transition",
    # Tactical situation
    "AmyStatus",
    "TacticalFleetHealth",
    "TargetCountsSummary",
    "TacticalSituation",
    "ThreatLevel",
    # GPX export
    "GPXDocument",
    "GPXRoute",
    "GPXTrack",
    "GPXWaypoint",
    # Behavioral clustering
    "BehaviorCluster",
    "ClusterSummary",
    "CommonPattern",
    "FormationType",
    # Daily patterns
    "DailyPattern",
]

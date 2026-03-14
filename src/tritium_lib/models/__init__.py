# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared data models for the Tritium ecosystem.

These models are the contract between tritium-sc and tritium-edge.
Any device that speaks the Tritium protocol uses these types.
"""

from .device import Device, DeviceGroup, DeviceHeartbeat, DeviceCapabilities
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

__all__ = [
    "Device",
    "DeviceGroup",
    "DeviceHeartbeat",
    "DeviceCapabilities",
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
]

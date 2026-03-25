# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Evidence data models — structured types for investigation evidence.

Evidence types capture sensor observations, track logs, zone events,
association data, and classification results.  Every piece of evidence
gets a SHA-256 integrity hash so tampering can be detected later.

No file I/O happens in this module — all data is in-memory Pydantic
models suitable for serialization and mock-friendly testing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """Categories of evidence that can be collected."""
    SIGNAL_CAPTURE = "signal_capture"          # BLE advertisements, WiFi probes
    TRACK_LOG = "track_log"                    # Position history over time
    ZONE_EVENT = "zone_event"                  # Geofence entry/exit timestamps
    ASSOCIATION = "association"                 # Co-located targets
    CLASSIFICATION = "classification"          # Device type, behavior type results
    SCREENSHOT = "screenshot"                  # Visual capture from camera/UI
    AUDIO_CAPTURE = "audio_capture"            # Acoustic sensor recording
    MANUAL_NOTE = "manual_note"                # Operator-entered observation
    SENSOR_RAW = "sensor_raw"                  # Raw sensor payload (RF, SDR, etc.)
    COMMUNICATION = "communication"            # Intercepted comms (mesh, radio)


class EvidenceStatus(str, Enum):
    """Lifecycle status of an evidence item."""
    COLLECTED = "collected"      # Just gathered
    VERIFIED = "verified"        # Integrity confirmed
    SEALED = "sealed"            # Locked — no further modifications
    CHALLENGED = "challenged"    # Integrity check failed
    ARCHIVED = "archived"        # Moved to long-term storage
    EXPUNGED = "expunged"        # Marked for deletion (legal/policy)


class SignalCaptureData(BaseModel):
    """Data specific to a signal capture evidence item."""
    signal_type: str = ""         # ble_advertisement, wifi_probe, wifi_beacon, etc.
    mac_address: str = ""
    rssi: Optional[float] = None
    frequency_mhz: Optional[float] = None
    channel: Optional[int] = None
    raw_payload: str = ""         # hex-encoded raw bytes
    sensor_id: str = ""           # which sensor captured this
    duration_ms: Optional[float] = None

    model_config = {"frozen": False}


class TrackLogEntry(BaseModel):
    """A single position in a track log."""
    timestamp: float = 0.0       # Unix timestamp
    lat: float = 0.0
    lng: float = 0.0
    altitude_m: Optional[float] = None
    heading: Optional[float] = None
    speed_mps: Optional[float] = None
    source: str = ""             # gps, triangulation, yolo, etc.
    confidence: float = 0.0

    model_config = {"frozen": False}


class TrackLogData(BaseModel):
    """Data specific to a track log evidence item."""
    target_id: str = ""
    entries: list[TrackLogEntry] = Field(default_factory=list)
    total_distance_m: float = 0.0
    duration_s: float = 0.0

    model_config = {"frozen": False}


class ZoneEventData(BaseModel):
    """Data specific to a zone event evidence item."""
    zone_id: str = ""
    zone_name: str = ""
    target_id: str = ""
    event_type: str = ""         # entry, exit, dwell
    entry_time: Optional[float] = None    # Unix timestamp
    exit_time: Optional[float] = None
    dwell_seconds: Optional[float] = None

    model_config = {"frozen": False}


class AssociationData(BaseModel):
    """Data specific to an association evidence item."""
    target_a: str = ""
    target_b: str = ""
    association_type: str = ""   # co_located, traveled_with, detected_with
    distance_m: Optional[float] = None
    overlap_seconds: Optional[float] = None
    shared_zone: str = ""
    confidence: float = 0.0

    model_config = {"frozen": False}


class ClassificationData(BaseModel):
    """Data specific to a classification evidence item."""
    target_id: str = ""
    classifier: str = ""         # device_classifier, yolo, behavior_model, etc.
    label: str = ""              # person, vehicle, phone, etc.
    confidence: float = 0.0
    model_version: str = ""
    features_used: list[str] = Field(default_factory=list)
    raw_scores: dict[str, float] = Field(default_factory=dict)

    model_config = {"frozen": False}


class Evidence(BaseModel):
    """A single piece of evidence for an investigation.

    Each Evidence item represents one discrete observation or artifact:
    a signal capture, a track log, a zone event, an association record,
    or a classification result.  It carries a SHA-256 integrity hash
    computed over its content so tampering can be detected.

    Attributes:
        evidence_id: Unique identifier for this evidence item.
        evidence_type: What kind of evidence this is.
        status: Current lifecycle status.
        target_id: Primary target this evidence relates to.
        collected_at: When this evidence was collected.
        collected_by: Who or what collected this evidence.
        source_sensor: Sensor or subsystem that produced the raw data.
        data: Type-specific structured data payload.
        sha256: Integrity hash of the data payload.
        tags: Free-form tags for categorization.
        notes: Human-readable notes.
        investigation_id: Optional link to an investigation/collection.
    """
    evidence_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for this evidence item",
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Category of evidence",
    )
    status: EvidenceStatus = Field(
        default=EvidenceStatus.COLLECTED,
        description="Current lifecycle status",
    )
    target_id: str = Field(
        "",
        description="Primary target this evidence relates to",
    )
    collected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this evidence was collected (UTC)",
    )
    collected_by: str = Field(
        "",
        description="Who or what collected this evidence (user, system, sensor)",
    )
    source_sensor: str = Field(
        "",
        description="Sensor or subsystem that produced the raw data",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific structured data payload",
    )
    sha256: str = Field(
        "",
        description="SHA-256 hash of the serialized data payload",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for categorization",
    )
    notes: str = Field(
        "",
        description="Human-readable notes about this evidence",
    )
    investigation_id: str = Field(
        "",
        description="Link to the parent investigation/collection",
    )

    model_config = {"frozen": False}

    def seal(self) -> None:
        """Seal the evidence — no further modifications should be made."""
        self.status = EvidenceStatus.SEALED

    def mark_verified(self) -> None:
        """Mark the evidence as integrity-verified."""
        self.status = EvidenceStatus.VERIFIED

    def mark_challenged(self) -> None:
        """Mark the evidence as integrity-challenged (hash mismatch)."""
        self.status = EvidenceStatus.CHALLENGED

    def to_summary(self) -> dict[str, Any]:
        """Return a concise summary dict for display."""
        return {
            "evidence_id": self.evidence_id,
            "type": self.evidence_type.value,
            "target_id": self.target_id,
            "collected_at": self.collected_at.isoformat(),
            "status": self.status.value,
            "sha256": self.sha256[:16] + "..." if len(self.sha256) > 16 else self.sha256,
        }

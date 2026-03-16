# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Intelligence package models for portable inter-site intelligence sharing.

An IntelligencePackage bundles targets, events, dossiers, and evidence
into a portable format that can be exported from one Tritium installation
and imported into another.  This is the primary vehicle for sharing
tactical intelligence between federated sites or for offline analysis.

Key differences from ExportPackage:
  - ExportPackage is a generic backup/restore format for system state.
  - IntelligencePackage is purpose-built for sharing curated intelligence
    between analysts and sites, with classification levels, chain of
    custody tracking, and selective redaction support.
"""

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IntelClassification(str, Enum):
    """Classification level for an intelligence package."""
    UNCLASSIFIED = "unclassified"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class PackageStatus(str, Enum):
    """Lifecycle status of an intelligence package."""
    DRAFT = "draft"
    FINALIZED = "finalized"
    TRANSMITTED = "transmitted"
    RECEIVED = "received"
    IMPORTED = "imported"
    REJECTED = "rejected"


class EvidenceType(str, Enum):
    """Type of evidence attached to an intelligence package."""
    SCREENSHOT = "screenshot"
    VIDEO_CLIP = "video_clip"
    AUDIO_CLIP = "audio_clip"
    LOG_EXTRACT = "log_extract"
    SENSOR_DATA = "sensor_data"
    ANALYST_NOTE = "analyst_note"
    CORRELATION_MAP = "correlation_map"
    TIMELINE = "timeline"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PackageTarget(BaseModel):
    """A target included in an intelligence package.

    Contains the essential tracking data needed for the recipient site
    to create or merge a target in their own tracker.
    """
    target_id: str
    name: str = ""
    entity_type: str = "unknown"
    classification: str = "unknown"
    alliance: str = "unknown"
    source: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None
    confidence: float = 0.5
    identifiers: dict[str, str] = Field(default_factory=dict)
    threat_level: str = "none"
    first_seen: float = 0.0
    last_seen: float = 0.0
    sighting_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PackageEvent(BaseModel):
    """A tactical event included in an intelligence package."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: float = Field(default_factory=time.time)
    target_ids: list[str] = Field(default_factory=list)
    description: str = ""
    severity: str = "info"
    lat: Optional[float] = None
    lng: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PackageDossier(BaseModel):
    """A target dossier included in an intelligence package.

    Contains enrichment data, signal history, and analyst assessments
    for a specific target.
    """
    target_id: str
    dossier_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signals: list[dict[str, Any]] = Field(default_factory=list)
    enrichments: list[dict[str, Any]] = Field(default_factory=list)
    position_history: list[dict[str, Any]] = Field(default_factory=list)
    analyst_notes: list[str] = Field(default_factory=list)
    threat_assessment: str = ""
    classification: str = "unknown"
    alliance: str = "unknown"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class PackageEvidence(BaseModel):
    """An evidence item attached to an intelligence package.

    Evidence can be screenshots, sensor data extracts, analyst notes,
    or other supporting material for the intelligence assessment.
    """
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evidence_type: EvidenceType = EvidenceType.ANALYST_NOTE
    title: str = ""
    description: str = ""
    target_ids: list[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)
    mime_type: str = ""
    size_bytes: int = 0
    checksum: str = ""


class ChainOfCustody(BaseModel):
    """Tracks the chain of custody for an intelligence package.

    Each time the package is transmitted, received, or modified,
    a custody entry is appended.
    """
    actor: str = ""
    action: str = ""  # created, transmitted, received, imported, modified
    timestamp: float = Field(default_factory=time.time)
    site_id: str = ""
    site_name: str = ""
    notes: str = ""


class IntelligencePackage(BaseModel):
    """A portable intelligence package for inter-site sharing.

    Bundles targets, events, dossiers, and evidence into a single
    transferable unit with classification, chain of custody, and
    import/export metadata.
    """
    package_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_site_id: str = ""
    source_site_name: str = ""
    destination_site_id: str = ""
    destination_site_name: str = ""

    # Metadata
    title: str = ""
    description: str = ""
    classification: IntelClassification = IntelClassification.UNCLASSIFIED
    status: PackageStatus = PackageStatus.DRAFT
    created_by: str = ""
    created_at: float = Field(default_factory=time.time)
    finalized_at: Optional[float] = None
    expires_at: Optional[float] = None
    tags: list[str] = Field(default_factory=list)

    # Content
    targets: list[PackageTarget] = Field(default_factory=list)
    events: list[PackageEvent] = Field(default_factory=list)
    dossiers: list[PackageDossier] = Field(default_factory=list)
    evidence: list[PackageEvidence] = Field(default_factory=list)

    # Chain of custody
    custody_chain: list[ChainOfCustody] = Field(default_factory=list)

    # Counts (auto-computed)
    target_count: int = 0
    event_count: int = 0
    dossier_count: int = 0
    evidence_count: int = 0

    def add_target(self, target: PackageTarget) -> None:
        """Add a target and update count."""
        self.targets.append(target)
        self.target_count = len(self.targets)

    def add_event(self, event: PackageEvent) -> None:
        """Add an event and update count."""
        self.events.append(event)
        self.event_count = len(self.events)

    def add_dossier(self, dossier: PackageDossier) -> None:
        """Add a dossier and update count."""
        self.dossiers.append(dossier)
        self.dossier_count = len(self.dossiers)

    def add_evidence(self, item: PackageEvidence) -> None:
        """Add an evidence item and update count."""
        self.evidence.append(item)
        self.evidence_count = len(self.evidence)

    def finalize(self) -> None:
        """Mark the package as finalized (ready for transmission)."""
        self.status = PackageStatus.FINALIZED
        self.finalized_at = time.time()
        self.target_count = len(self.targets)
        self.event_count = len(self.events)
        self.dossier_count = len(self.dossiers)
        self.evidence_count = len(self.evidence)

    def add_custody_entry(
        self,
        actor: str,
        action: str,
        site_id: str = "",
        site_name: str = "",
        notes: str = "",
    ) -> None:
        """Append a chain of custody entry."""
        self.custody_chain.append(ChainOfCustody(
            actor=actor,
            action=action,
            site_id=site_id,
            site_name=site_name,
            notes=notes,
        ))

    def target_ids(self) -> list[str]:
        """Return all target IDs in this package."""
        return [t.target_id for t in self.targets]

    def is_expired(self) -> bool:
        """Check if the package has passed its expiration time."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class PackageImportResult(BaseModel):
    """Result of importing an intelligence package into a local site."""
    package_id: str = ""
    success: bool = True
    targets_imported: int = 0
    targets_merged: int = 0
    targets_skipped: int = 0
    events_imported: int = 0
    dossiers_imported: int = 0
    dossiers_merged: int = 0
    evidence_imported: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_intelligence_package(
    source_site_id: str = "",
    source_site_name: str = "",
    title: str = "",
    description: str = "",
    created_by: str = "",
    classification: IntelClassification = IntelClassification.UNCLASSIFIED,
    tags: Optional[list[str]] = None,
) -> IntelligencePackage:
    """Create a new intelligence package with standard metadata."""
    pkg = IntelligencePackage(
        source_site_id=source_site_id,
        source_site_name=source_site_name,
        title=title,
        description=description,
        created_by=created_by,
        classification=classification,
        tags=tags or [],
    )
    pkg.add_custody_entry(
        actor=created_by or "system",
        action="created",
        site_id=source_site_id,
        site_name=source_site_name,
    )
    return pkg


def validate_package_import(
    package: IntelligencePackage,
    local_site_id: str = "",
) -> PackageImportResult:
    """Pre-validate an intelligence package before importing.

    Checks for expiration, classification compatibility, and
    basic structural integrity.
    """
    result = PackageImportResult(package_id=package.package_id)

    if package.is_expired():
        result.errors.append("Package has expired")
        result.success = False

    if package.status == PackageStatus.REJECTED:
        result.errors.append("Package has been rejected")
        result.success = False

    if package.status == PackageStatus.DRAFT:
        result.warnings.append("Package is still in draft status")

    if package.destination_site_id and package.destination_site_id != local_site_id:
        result.warnings.append(
            f"Package was addressed to site '{package.destination_site_id}', "
            f"not this site '{local_site_id}'"
        )

    if not package.targets and not package.events and not package.dossiers:
        result.warnings.append("Package contains no intelligence data")

    # Check for duplicate target IDs within package
    tids = [t.target_id for t in package.targets]
    if len(tids) != len(set(tids)):
        result.warnings.append("Package contains duplicate target IDs")

    return result

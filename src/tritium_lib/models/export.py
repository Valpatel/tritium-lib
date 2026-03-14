# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Export/import models for portable system state bundles.

An ExportManifest describes what is included in an export package.
An ExportPackage wraps the manifest plus the actual data payloads.
Used by backup/restore workflows and federation for standardized
data exchange between Tritium installations.

ExportSection identifies a logical data section (automation rules,
target dossiers, device configs, etc.).  Each section has its own
schema version so receivers can handle version mismatches gracefully.
"""

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExportFormat(str, Enum):
    """Serialization format for the export package."""
    JSON = "json"
    MSGPACK = "msgpack"
    CBOR = "cbor"


class ExportScope(str, Enum):
    """Scope of data included in the export."""
    FULL = "full"              # Everything: config, targets, dossiers, rules, history
    CONFIG = "config"          # Configuration only: device configs, automation rules
    INTELLIGENCE = "intelligence"  # Targets, dossiers, threat data
    TARGETS = "targets"        # Real-time target snapshot
    RULES = "rules"            # Automation rules only
    CUSTOM = "custom"          # Cherry-picked sections


class ExportSectionType(str, Enum):
    """Logical data sections that can be included in an export."""
    AUTOMATION_RULES = "automation_rules"
    DEVICE_CONFIGS = "device_configs"
    TARGET_SNAPSHOT = "target_snapshot"
    DOSSIERS = "dossiers"
    THREAT_FEEDS = "threat_feeds"
    GEOFENCES = "geofences"
    MAP_LAYERS = "map_layers"
    NOTIFICATION_RULES = "notification_rules"
    FLEET_REGISTRY = "fleet_registry"
    FEDERATION_SITES = "federation_sites"
    CAMERA_CONFIGS = "camera_configs"
    KNOWN_DEVICES = "known_devices"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ExportSection(BaseModel):
    """A single data section within an export package.

    Each section is independently versioned so importers can handle
    schema evolution gracefully.
    """

    section_type: ExportSectionType
    schema_version: str = "1.0.0"
    item_count: int = 0
    data: list[dict[str, Any]] = Field(default_factory=list)
    checksum: str = ""  # SHA256 of serialized data (optional)


class ExportManifest(BaseModel):
    """Describes the contents and metadata of an export package.

    The manifest is always the first object in a serialized export,
    allowing importers to inspect what is included before processing
    the full payload.
    """

    export_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = Field(default_factory=time.time)
    tritium_version: str = ""
    source_site_id: str = ""
    source_site_name: str = ""
    scope: ExportScope = ExportScope.CUSTOM
    sections: list[ExportSectionType] = Field(default_factory=list)
    section_counts: dict[str, int] = Field(default_factory=dict)
    total_items: int = 0
    format: ExportFormat = ExportFormat.JSON
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    encrypted: bool = False
    compressed: bool = False


class ExportPackage(BaseModel):
    """A complete export bundle: manifest + data sections.

    This is the top-level structure serialized to disk or transmitted
    over the wire during backup/restore or federation sync.
    """

    manifest: ExportManifest = Field(default_factory=ExportManifest)
    sections: list[ExportSection] = Field(default_factory=list)

    def add_section(
        self,
        section_type: ExportSectionType,
        data: list[dict[str, Any]],
        schema_version: str = "1.0.0",
    ) -> None:
        """Add a data section to the package and update manifest counts."""
        section = ExportSection(
            section_type=section_type,
            schema_version=schema_version,
            item_count=len(data),
            data=data,
        )
        self.sections.append(section)
        self.manifest.sections.append(section_type)
        self.manifest.section_counts[section_type.value] = len(data)
        self.manifest.total_items += len(data)

    def get_section(self, section_type: ExportSectionType) -> Optional[ExportSection]:
        """Retrieve a section by type, or None if not present."""
        for s in self.sections:
            if s.section_type == section_type:
                return s
        return None

    def section_types(self) -> list[ExportSectionType]:
        """List all section types present in this package."""
        return [s.section_type for s in self.sections]


class ImportResult(BaseModel):
    """Result of importing an export package."""

    export_id: str = ""
    success: bool = True
    sections_imported: list[str] = Field(default_factory=list)
    sections_skipped: list[str] = Field(default_factory=list)
    items_imported: int = 0
    items_skipped: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_export_manifest(
    scope: ExportScope,
    site_id: str = "",
    site_name: str = "",
    description: str = "",
    tags: Optional[list[str]] = None,
) -> ExportManifest:
    """Create a new export manifest with standard metadata."""
    return ExportManifest(
        scope=scope,
        source_site_id=site_id,
        source_site_name=site_name,
        description=description,
        tags=tags or [],
    )


def validate_import_compatibility(
    manifest: ExportManifest,
    supported_sections: Optional[set[ExportSectionType]] = None,
) -> ImportResult:
    """Pre-check whether an export package can be imported.

    Returns an ImportResult with warnings/errors for incompatible sections.
    Does NOT actually import any data.
    """
    result = ImportResult(export_id=manifest.export_id)

    if supported_sections is None:
        # If not specified, accept all section types
        supported_sections = set(ExportSectionType)

    for section_type in manifest.sections:
        if section_type in supported_sections:
            result.sections_imported.append(section_type.value)
        else:
            result.sections_skipped.append(section_type.value)
            result.warnings.append(
                f"Section '{section_type.value}' is not supported by this installation"
            )

    if manifest.encrypted:
        result.errors.append("Encrypted exports are not yet supported")
        result.success = False

    return result

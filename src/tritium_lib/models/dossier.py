# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Target Dossier models.

A dossier is a persistent identity that accumulates evidence over time.
It is NOT a single detection — it represents accumulated intelligence about
a unique real-world entity (person, vehicle, device, animal) built from
multiple correlated signals across sensors and time.
"""

import uuid
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DossierSignal(BaseModel):
    """A single contributing detection or signal linked to a dossier."""

    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str  # ble, yolo, wifi, mesh, manual, etc.
    signal_type: str  # mac_sighting, visual_detection, probe_request, etc.
    data: dict = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    position: Optional[tuple[float, float]] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DossierEnrichment(BaseModel):
    """External intelligence enrichment attached to a dossier."""

    provider: str  # wigle, oui_lookup, social_media, etc.
    enrichment_type: str  # manufacturer, location_history, profile, etc.
    data: dict = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class PositionRecord(BaseModel):
    """A timestamped position observation."""

    x: float
    y: float
    timestamp: float = Field(default_factory=time.time)
    source: str = "unknown"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TargetDossier(BaseModel):
    """Persistent identity that accumulates evidence over time.

    A dossier aggregates signals from BLE, WiFi, YOLO vision, mesh
    intercepts, manual observations, and enrichment providers into a
    unified profile of a real-world entity.
    """

    dossier_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unknown"
    entity_type: Literal["person", "vehicle", "device", "animal", "unknown"] = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    first_seen: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    signals: list[DossierSignal] = Field(default_factory=list)
    identifiers: dict[str, str] = Field(default_factory=dict)
    enrichments: list[DossierEnrichment] = Field(default_factory=list)
    position_history: list[PositionRecord] = Field(default_factory=list)
    alliance: str = "unknown"
    threat_level: Literal["none", "low", "medium", "high", "critical"] = "none"
    notes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

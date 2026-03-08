# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Firmware metadata model."""

from datetime import datetime
from typing import Optional

from enum import Enum

from pydantic import BaseModel, Field


class FirmwareMeta(BaseModel):
    """Firmware image metadata — used by fleet server and OTA tools."""
    id: str
    version: str
    board: str = "any"
    family: str = "esp32"
    size: int = 0
    sha256: str = ""
    signed: bool = False
    encrypted: bool = False
    build_timestamp: Optional[datetime] = None
    uploaded_at: Optional[datetime] = None
    notes: str = ""


class OTAStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OTAJob(BaseModel):
    """An OTA firmware update job targeting one or more devices."""
    id: str
    firmware_url: str
    target_devices: list[str] = Field(default_factory=list)
    status: OTAStatus = OTAStatus.PENDING
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    firmware_version: str = ""
    firmware_sha256: str = ""
    error: Optional[str] = None

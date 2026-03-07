# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Firmware metadata model."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


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

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Command models — device commands shared across the ecosystem."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CommandType(str, Enum):
    REBOOT = "reboot"
    GPIO_SET = "gpio_set"
    GPIO_READ = "gpio_read"
    CONFIG_UPDATE = "config_update"
    OTA_URL = "ota_url"
    OTA_ROLLBACK = "ota_rollback"
    IDENTIFY = "identify"
    SLEEP = "sleep"
    WIFI_ADD = "wifi_add"
    MESH_SEND = "mesh_send"
    CUSTOM = "custom"


class CommandStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKED = "acked"
    FAILED = "failed"
    EXPIRED = "expired"


class Command(BaseModel):
    """A command sent to a device. Same schema everywhere."""
    id: str
    device_id: str
    type: CommandType
    payload: dict = Field(default_factory=dict)
    status: CommandStatus = CommandStatus.PENDING
    created_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    acked_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    error: Optional[str] = None

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Sensor reading model — used when edge devices report telemetry
to tritium-sc via MQTT bridge."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """A single sensor reading from an edge device.

    Maps to MQTT topic: tritium/{site}/sensors/{device_id}/{sensor_type}
    """
    device_id: str
    sensor_type: str  # temperature, humidity, imu, gps, etc.
    value: float | dict | list  # scalar or structured
    unit: str = ""  # celsius, percent, g, deg/s, etc.
    timestamp: Optional[datetime] = None
    quality: float = 1.0  # 0.0-1.0 confidence

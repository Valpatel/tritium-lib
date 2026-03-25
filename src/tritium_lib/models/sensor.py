# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Sensor reading model — used when edge devices report telemetry
to tritium-sc via MQTT bridge."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SensorReading(BaseModel):
    """A single sensor reading from an edge device.

    Maps to MQTT topic: tritium/{site}/sensors/{device_id}/{sensor_type}
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "device_id": "esp32-001",
                    "sensor_type": "temperature",
                    "value": 23.5,
                    "unit": "celsius",
                    "quality": 0.95,
                }
            ]
        }
    )

    device_id: str = Field(..., min_length=1)
    sensor_type: str = Field(..., min_length=1)  # temperature, humidity, imu, gps, etc.
    value: float | dict | list  # scalar or structured
    unit: str = ""  # celsius, percent, g, deg/s, etc.
    timestamp: Optional[datetime] = None
    quality: float = Field(1.0, ge=0.0, le=1.0)  # 0.0-1.0 confidence

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        val_str = str(self.value)
        if len(val_str) > 40:
            val_str = val_str[:37] + "..."
        unit_str = f" {self.unit}" if self.unit else ""
        return (
            f"[{self.device_id}] {self.sensor_type}={val_str}{unit_str} "
            f"q={self.quality:.2f}"
        )

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Environment reading model — standardizes environmental sensor data
across all sources (Meshtastic nodes, ESP32 BME280/BMP280/SHT31, etc.)."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EnvironmentSource(str, Enum):
    """Source type for environment readings."""
    MESHTASTIC = "meshtastic"
    EDGE_DEVICE = "edge_device"
    WEATHER_API = "weather_api"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class EnvironmentReading(BaseModel):
    """A single environmental sensor reading from any source.

    Maps to MQTT topic: tritium/{site}/environment/{source_id}
    """
    source_id: str = Field(description="Device or node ID that produced this reading")
    source_type: EnvironmentSource = EnvironmentSource.UNKNOWN
    temperature_c: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity_pct: Optional[float] = Field(None, description="Relative humidity 0-100%")
    pressure_hpa: Optional[float] = Field(None, description="Barometric pressure in hPa")
    air_quality_index: Optional[float] = Field(None, description="Air quality index (0-500)")
    light_level_lux: Optional[float] = Field(None, description="Ambient light in lux")
    noise_level_db: Optional[float] = Field(None, description="Ambient noise in dB")
    gas_resistance_ohm: Optional[float] = Field(None, description="Gas resistance (BME680)")
    uv_index: Optional[float] = Field(None, description="UV index (0-11+)")
    wind_speed_mps: Optional[float] = Field(None, description="Wind speed in m/s")
    wind_direction_deg: Optional[float] = Field(None, description="Wind direction in degrees")
    rainfall_mm: Optional[float] = Field(None, description="Rainfall in mm")
    timestamp: Optional[datetime] = None
    quality: float = Field(1.0, ge=0.0, le=1.0, description="Reading confidence 0-1")
    location_lat: Optional[float] = Field(None, description="Latitude of sensor")
    location_lng: Optional[float] = Field(None, description="Longitude of sensor")

    @property
    def temperature_f(self) -> Optional[float]:
        """Temperature in Fahrenheit."""
        if self.temperature_c is None:
            return None
        return self.temperature_c * 9.0 / 5.0 + 32.0

    @property
    def has_data(self) -> bool:
        """Whether this reading has any actual sensor data."""
        return any([
            self.temperature_c is not None,
            self.humidity_pct is not None,
            self.pressure_hpa is not None,
            self.air_quality_index is not None,
            self.light_level_lux is not None,
            self.noise_level_db is not None,
        ])

    def summary_line(self) -> str:
        """Human-readable one-line summary for dashboards."""
        parts = []
        if self.temperature_c is not None:
            parts.append(f"{self.temperature_f:.0f}F/{self.temperature_c:.1f}C")
        if self.humidity_pct is not None:
            parts.append(f"{self.humidity_pct:.0f}% humidity")
        if self.pressure_hpa is not None:
            parts.append(f"{self.pressure_hpa:.1f} hPa")
        if self.air_quality_index is not None:
            parts.append(f"AQI {self.air_quality_index:.0f}")
        if self.light_level_lux is not None:
            parts.append(f"{self.light_level_lux:.0f} lux")
        if self.noise_level_db is not None:
            parts.append(f"{self.noise_level_db:.0f} dB")
        return ", ".join(parts) if parts else "No data"


class EnvironmentSnapshot(BaseModel):
    """Aggregated environment readings from multiple sources."""
    readings: list[EnvironmentReading] = Field(default_factory=list)
    timestamp: Optional[datetime] = None

    @property
    def avg_temperature_c(self) -> Optional[float]:
        temps = [r.temperature_c for r in self.readings if r.temperature_c is not None]
        return sum(temps) / len(temps) if temps else None

    @property
    def avg_humidity_pct(self) -> Optional[float]:
        vals = [r.humidity_pct for r in self.readings if r.humidity_pct is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def avg_pressure_hpa(self) -> Optional[float]:
        vals = [r.pressure_hpa for r in self.readings if r.pressure_hpa is not None]
        return sum(vals) / len(vals) if vals else None

    def summary(self) -> str:
        """Aggregate summary across all sources."""
        if not self.readings:
            return "No environment data"
        parts = []
        t = self.avg_temperature_c
        if t is not None:
            parts.append(f"{t * 9/5 + 32:.0f}F")
        h = self.avg_humidity_pct
        if h is not None:
            parts.append(f"{h:.0f}% humidity")
        p = self.avg_pressure_hpa
        if p is not None:
            parts.append(f"{p:.1f} hPa")
        sources = len(self.readings)
        prefix = f"[{sources} source{'s' if sources != 1 else ''}]"
        return f"{prefix} {', '.join(parts)}" if parts else f"{prefix} No data"

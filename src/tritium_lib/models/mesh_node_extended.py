# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Extended Meshtastic node model with full telemetry and environment data.

Captures everything a real Meshtastic radio provides: position, device
telemetry (battery, voltage, channel utilization), environment sensors
(temperature, humidity, barometric pressure), hardware model, firmware
version, and radio metrics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MeshNodePosition(BaseModel):
    """GPS position from a Meshtastic node."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None  # meters above sea level
    precision_bits: Optional[int] = None
    time: Optional[int] = None  # POSIX timestamp of position fix
    ground_speed: Optional[int] = None  # m/s
    ground_track: Optional[int] = None  # degrees 0-359
    sats_in_view: Optional[int] = None
    fix_quality: Optional[int] = None  # 0=no fix, 1=2D, 2=3D
    fix_type: Optional[int] = None
    pdop: Optional[int] = None  # position dilution of precision

    @property
    def has_fix(self) -> bool:
        return self.latitude is not None and self.longitude is not None


class MeshNodeDeviceMetrics(BaseModel):
    """Device telemetry from a Meshtastic node."""
    battery_level: Optional[int] = None  # 0-100 percent
    voltage: Optional[float] = None  # battery voltage
    channel_utilization: Optional[float] = None  # 0-100 percent
    air_util_tx: Optional[float] = None  # 0-100 percent transmit airtime
    uptime_seconds: Optional[int] = None


class MeshNodeEnvironment(BaseModel):
    """Environment sensor data from a Meshtastic node."""
    temperature: Optional[float] = None  # Celsius
    relative_humidity: Optional[float] = None  # 0-100 percent
    barometric_pressure: Optional[float] = None  # hPa
    gas_resistance: Optional[float] = None  # Ohms (BME680)
    iaq: Optional[int] = None  # indoor air quality index
    wind_speed: Optional[float] = None  # m/s
    wind_direction: Optional[int] = None  # degrees 0-359
    lux: Optional[float] = None  # light level
    uv_index: Optional[float] = None
    weight: Optional[float] = None  # kg (for scale sensors)


class MeshNodeRadioMetrics(BaseModel):
    """Radio signal metrics for a Meshtastic node."""
    snr: Optional[float] = None  # signal-to-noise ratio dB
    rssi: Optional[int] = None  # received signal strength dBm
    hop_limit: Optional[int] = None
    hop_start: Optional[int] = None
    # Hops away = hop_start - hop_limit (if both known)

    @property
    def hops_away(self) -> Optional[int]:
        if self.hop_start is not None and self.hop_limit is not None:
            return self.hop_start - self.hop_limit
        return None


class MeshNodeExtended(BaseModel):
    """Full Meshtastic node with all available data.

    This is the extended model for real Meshtastic hardware integration.
    It captures every field the meshtastic Python library exposes:
    user info, position, device telemetry, environment sensors,
    radio metrics, and connection state.
    """
    # Identity
    node_id: str  # hex string e.g. "!ba33ff38"
    node_num: Optional[int] = None  # numeric node ID
    long_name: str = ""
    short_name: str = ""
    hw_model: str = ""  # e.g. "TLORA_V2_1_1P6", "HELTEC_V3"
    firmware_version: str = ""  # e.g. "2.7.19"
    role: Optional[str] = None  # CLIENT, ROUTER, etc.
    is_licensed: bool = False  # ham radio licensed
    mac_addr: Optional[str] = None

    # Position
    position: Optional[MeshNodePosition] = None

    # Telemetry
    device_metrics: Optional[MeshNodeDeviceMetrics] = None
    environment: Optional[MeshNodeEnvironment] = None

    # Radio
    radio: Optional[MeshNodeRadioMetrics] = None

    # Timestamps
    last_heard: Optional[int] = None  # POSIX timestamp
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Meshtastic-specific
    is_favorite: bool = False
    via_mqtt: bool = False  # reached via MQTT rather than radio
    hops_away: Optional[int] = None
    num: Optional[int] = None  # alias for node_num

    # Tritium tracking
    target_id: Optional[str] = None  # tritium target ID: mesh_{node_hex}

    model_config = {"populate_by_name": True}

    @property
    def has_position(self) -> bool:
        return self.position is not None and self.position.has_fix

    @property
    def battery_percent(self) -> Optional[int]:
        if self.device_metrics and self.device_metrics.battery_level is not None:
            return self.device_metrics.battery_level
        return None

    @property
    def display_name(self) -> str:
        return self.long_name or self.short_name or self.node_id

    @property
    def age_seconds(self) -> Optional[int]:
        """Seconds since this node was last heard, or None."""
        if self.last_heard is None:
            return None
        import time
        return int(time.time()) - self.last_heard

    @classmethod
    def from_meshtastic_node(cls, node_id: str, info: dict) -> "MeshNodeExtended":
        """Create from the meshtastic library's node dict.

        The meshtastic library returns nodes as dicts with keys:
        'user', 'position', 'deviceMetrics', 'lastHeard', 'snr', etc.
        """
        user = info.get("user", {})
        pos_raw = info.get("position", {})
        dm_raw = info.get("deviceMetrics", {})
        env_raw = info.get("environmentMetrics", {})

        position = None
        if pos_raw:
            position = MeshNodePosition(
                latitude=pos_raw.get("latitude"),
                longitude=pos_raw.get("longitude"),
                altitude=pos_raw.get("altitude"),
                precision_bits=pos_raw.get("precisionBits"),
                time=pos_raw.get("time"),
                ground_speed=pos_raw.get("groundSpeed"),
                ground_track=pos_raw.get("groundTrack"),
                sats_in_view=pos_raw.get("satsInView"),
            )

        device_metrics = None
        if dm_raw:
            device_metrics = MeshNodeDeviceMetrics(
                battery_level=dm_raw.get("batteryLevel"),
                voltage=dm_raw.get("voltage"),
                channel_utilization=dm_raw.get("channelUtilization"),
                air_util_tx=dm_raw.get("airUtilTx"),
                uptime_seconds=dm_raw.get("uptimeSeconds"),
            )

        environment = None
        if env_raw:
            environment = MeshNodeEnvironment(
                temperature=env_raw.get("temperature"),
                relative_humidity=env_raw.get("relativeHumidity"),
                barometric_pressure=env_raw.get("barometricPressure"),
                gas_resistance=env_raw.get("gasResistance"),
                iaq=env_raw.get("iaq"),
            )

        radio = MeshNodeRadioMetrics(
            snr=info.get("snr"),
            rssi=info.get("rssi"),
            hop_limit=info.get("hopLimit"),
            hop_start=info.get("hopStart"),
        )

        node_hex = node_id.replace("!", "")

        return cls(
            node_id=node_id,
            node_num=user.get("num"),
            long_name=user.get("longName", ""),
            short_name=user.get("shortName", ""),
            hw_model=user.get("hwModel", ""),
            firmware_version=info.get("firmwareVersion", ""),
            role=user.get("role"),
            is_licensed=user.get("isLicensed", False),
            mac_addr=user.get("macaddr"),
            position=position,
            device_metrics=device_metrics,
            environment=environment,
            radio=radio,
            last_heard=info.get("lastHeard"),
            is_favorite=info.get("isFavorite", False),
            via_mqtt=info.get("viaMqtt", False),
            hops_away=info.get("hopsAway"),
            target_id=f"mesh_{node_hex}",
        )

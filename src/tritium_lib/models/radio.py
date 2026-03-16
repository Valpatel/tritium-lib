# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Radio scheduler models for BLE/WiFi time-division multiplexing.

These models represent the radio scheduler state on edge devices
that need to share the ESP32-S3 radio between WiFi and BLE stacks.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RadioMode(str, Enum):
    """Current radio mode of the edge device."""

    IDLE = "idle"
    WIFI_ACTIVE = "wifi"
    BLE_SCANNING = "ble"
    TRANSITIONING = "transition"


class RadioSchedulerConfig(BaseModel):
    """Configuration for the radio scheduler."""

    wifi_slot_ms: int = Field(default=25000, ge=1000, le=300000, description="WiFi active duration")
    ble_slot_ms: int = Field(default=10000, ge=1000, le=300000, description="BLE scan duration")
    transition_ms: int = Field(default=2000, ge=500, le=10000, description="Transition teardown/startup time")
    enable_ble: bool = Field(default=True, description="Enable BLE scanning slot")
    enable_wifi: bool = Field(default=True, description="Enable WiFi slot")
    wifi_first: bool = Field(default=True, description="Start with WiFi slot")


class RadioSchedulerStatus(BaseModel):
    """Status of the radio scheduler on an edge device."""

    mode: RadioMode = RadioMode.IDLE
    wifi_cycles: int = Field(default=0, ge=0, description="Total WiFi slot cycles completed")
    ble_cycles: int = Field(default=0, ge=0, description="Total BLE slot cycles completed")
    slot_remaining_ms: int = Field(default=0, ge=0, description="Milliseconds remaining in current slot")
    config: Optional[RadioSchedulerConfig] = None


class CameraMqttConfig(BaseModel):
    """Configuration for camera MQTT frame publisher on edge devices."""

    target_fps: float = Field(default=2.0, ge=0.1, le=10.0, description="Target frames per second")
    jpeg_quality: int = Field(default=15, ge=10, le=63, description="JPEG quality (lower=better)")
    auto_start: bool = Field(default=True, description="Start publishing on init")
    topic_prefix: str = Field(default="", description="MQTT topic prefix for frames")


class CameraMqttStats(BaseModel):
    """Stats from the camera MQTT publisher on an edge device."""

    active: bool = False
    frames_published: int = Field(default=0, ge=0)
    frames_failed: int = Field(default=0, ge=0)
    avg_latency_ms: int = Field(default=0, ge=0)
    max_frame_bytes: int = Field(default=0, ge=0)
    target_fps: float = 0.0
    actual_fps: float = 0.0

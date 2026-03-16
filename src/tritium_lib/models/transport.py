# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Transport negotiation models.

Tritium nodes negotiate the best available transport automatically.
Supported transports: WiFi, ESP-NOW, BLE, LoRa, MQTT, Acoustic, USB Serial.
These models represent transport state, quality metrics, and preferences
used by the negotiation algorithm.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TransportType(str, Enum):
    """Available communication transports."""
    WIFI = "wifi"
    ESP_NOW = "esp_now"
    BLE = "ble"
    LORA = "lora"
    MQTT = "mqtt"
    ACOUSTIC = "acoustic"
    USB_SERIAL = "usb_serial"


class TransportState(str, Enum):
    """Operational state of a transport."""
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class TransportMetrics(BaseModel):
    """Quality metrics for a single transport on a device.

    Captures signal strength, throughput, latency, and reliability
    for transport selection decisions.
    """
    type: TransportType
    state: TransportState
    rssi: Optional[int] = Field(
        None, description="Signal strength in dBm (where applicable)",
    )
    bandwidth_bps: Optional[int] = Field(
        None, ge=0, description="Estimated bandwidth in bits per second",
    )
    latency_ms: Optional[float] = Field(
        None, ge=0.0, description="Round-trip latency in milliseconds",
    )
    packet_loss_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Packet loss percentage (0-100)",
    )
    last_active: Optional[datetime] = Field(
        None, description="Timestamp of last successful communication",
    )


class TransportPreference(BaseModel):
    """User/fleet preference for a transport type.

    Lower priority value means more preferred. Quality thresholds
    define minimum acceptable signal and maximum acceptable latency.
    """
    type: TransportType
    priority: int = Field(
        ..., description="Selection priority (lower = preferred)",
    )
    min_rssi: Optional[int] = Field(
        None, description="Minimum RSSI threshold for usability (dBm)",
    )
    max_latency_ms: Optional[float] = Field(
        None, ge=0.0, description="Maximum acceptable latency in ms",
    )


class NodeTransportStatus(BaseModel):
    """Transport status for a single device.

    Aggregates metrics for all transports on the device and tracks
    which transport is currently active vs. preferred.
    """
    device_id: str = Field(..., description="Unique device identifier")
    transports: list[TransportMetrics] = Field(default_factory=list)
    preferred_transport: Optional[TransportType] = Field(
        None, description="Transport the negotiation algorithm prefers",
    )
    active_transport: Optional[TransportType] = Field(
        None, description="Transport currently in use",
    )

    def get_transport(self, transport_type: TransportType) -> Optional[TransportMetrics]:
        """Look up metrics for a specific transport type."""
        for t in self.transports:
            if t.type == transport_type:
                return t
        return None

    @property
    def available_transports(self) -> list[TransportMetrics]:
        """Transports in AVAILABLE or DEGRADED state."""
        return [
            t for t in self.transports
            if t.state in (TransportState.AVAILABLE, TransportState.DEGRADED)
        ]


def select_best_transport(
    transports: list[TransportMetrics],
    preferences: list[TransportPreference],
) -> Optional[TransportType]:
    """Select the best available transport based on state, priority, and quality thresholds.

    Algorithm:
    1. Filter to transports that are AVAILABLE or DEGRADED.
    2. For each preference (sorted by priority), check if the transport
       meets the quality thresholds (min_rssi, max_latency_ms).
    3. Return the first transport that passes all checks.
    4. If no transport meets preferences, return the best available
       transport by priority alone (fallback).
    5. Return None if no transports are usable.

    Args:
        transports: Current transport metrics for the device.
        preferences: Ordered preferences with quality thresholds.

    Returns:
        The selected TransportType, or None if nothing is usable.
    """
    # Index metrics by type for fast lookup
    metrics_by_type: dict[TransportType, TransportMetrics] = {
        t.type: t for t in transports
    }

    # Only consider usable states
    usable_types = {
        t.type for t in transports
        if t.state in (TransportState.AVAILABLE, TransportState.DEGRADED)
    }

    if not usable_types:
        return None

    # Sort preferences by priority (lower = better)
    sorted_prefs = sorted(preferences, key=lambda p: p.priority)

    # First pass: find a transport that meets all quality thresholds
    for pref in sorted_prefs:
        if pref.type not in usable_types:
            continue
        metrics = metrics_by_type[pref.type]

        # Check RSSI threshold
        if pref.min_rssi is not None and metrics.rssi is not None:
            if metrics.rssi < pref.min_rssi:
                continue

        # Check latency threshold
        if pref.max_latency_ms is not None and metrics.latency_ms is not None:
            if metrics.latency_ms > pref.max_latency_ms:
                continue

        return pref.type

    # Fallback: return the highest-priority usable transport regardless of thresholds
    for pref in sorted_prefs:
        if pref.type in usable_types:
            return pref.type

    # No preferences matched any usable transport — pick any usable one
    for t in transports:
        if t.type in usable_types:
            return t.type

    return None


def transport_summary(status: NodeTransportStatus) -> dict:
    """Produce a compact summary dict for API responses.

    Returns:
        Dict with device_id, active/preferred transport, counts by state,
        and a list of available transport names.
    """
    by_state: dict[str, int] = {}
    available_names: list[str] = []

    for t in status.transports:
        by_state[t.state.value] = by_state.get(t.state.value, 0) + 1
        if t.state in (TransportState.AVAILABLE, TransportState.DEGRADED):
            available_names.append(t.type.value)

    return {
        "device_id": status.device_id,
        "active": status.active_transport.value if status.active_transport else None,
        "preferred": status.preferred_transport.value if status.preferred_transport else None,
        "total": len(status.transports),
        "by_state": by_state,
        "available": available_names,
    }

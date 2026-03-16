# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for transport negotiation models."""

from datetime import datetime, timezone

import pytest

from tritium_lib.models.transport import (
    NodeTransportStatus,
    TransportMetrics,
    TransportPreference,
    TransportState,
    TransportType,
    select_best_transport,
    transport_summary,
)


def _utc(year=2026, month=3, day=8, hour=12, minute=0, second=0):
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# TransportType enum
# ---------------------------------------------------------------------------

class TestTransportType:
    def test_all_values(self):
        expected = {"wifi", "esp_now", "ble", "lora", "mqtt", "acoustic", "usb_serial"}
        assert {t.value for t in TransportType} == expected

    def test_string_enum(self):
        assert TransportType.WIFI == "wifi"
        assert TransportType.ESP_NOW == "esp_now"
        assert TransportType.ACOUSTIC == "acoustic"


# ---------------------------------------------------------------------------
# TransportState enum
# ---------------------------------------------------------------------------

class TestTransportState:
    def test_all_values(self):
        expected = {"available", "degraded", "unavailable", "disabled"}
        assert {s.value for s in TransportState} == expected

    def test_string_enum(self):
        assert TransportState.AVAILABLE == "available"
        assert TransportState.DISABLED == "disabled"


# ---------------------------------------------------------------------------
# TransportMetrics
# ---------------------------------------------------------------------------

class TestTransportMetrics:
    def test_create_minimal(self):
        m = TransportMetrics(type=TransportType.WIFI, state=TransportState.AVAILABLE)
        assert m.type == TransportType.WIFI
        assert m.state == TransportState.AVAILABLE
        assert m.rssi is None
        assert m.bandwidth_bps is None
        assert m.latency_ms is None
        assert m.packet_loss_pct is None
        assert m.last_active is None

    def test_create_full(self):
        m = TransportMetrics(
            type=TransportType.LORA,
            state=TransportState.DEGRADED,
            rssi=-90,
            bandwidth_bps=9600,
            latency_ms=250.5,
            packet_loss_pct=5.2,
            last_active=_utc(),
        )
        assert m.rssi == -90
        assert m.bandwidth_bps == 9600
        assert m.latency_ms == 250.5
        assert m.packet_loss_pct == 5.2
        assert m.last_active == _utc()

    def test_packet_loss_bounds(self):
        # Valid at boundaries
        m0 = TransportMetrics(
            type=TransportType.BLE, state=TransportState.AVAILABLE, packet_loss_pct=0.0,
        )
        assert m0.packet_loss_pct == 0.0

        m100 = TransportMetrics(
            type=TransportType.BLE, state=TransportState.AVAILABLE, packet_loss_pct=100.0,
        )
        assert m100.packet_loss_pct == 100.0

    def test_packet_loss_over_100_rejected(self):
        with pytest.raises(Exception):
            TransportMetrics(
                type=TransportType.BLE, state=TransportState.AVAILABLE, packet_loss_pct=101.0,
            )

    def test_negative_bandwidth_rejected(self):
        with pytest.raises(Exception):
            TransportMetrics(
                type=TransportType.WIFI, state=TransportState.AVAILABLE, bandwidth_bps=-1,
            )

    def test_negative_latency_rejected(self):
        with pytest.raises(Exception):
            TransportMetrics(
                type=TransportType.WIFI, state=TransportState.AVAILABLE, latency_ms=-1.0,
            )

    def test_serialization(self):
        m = TransportMetrics(
            type=TransportType.ESP_NOW,
            state=TransportState.AVAILABLE,
            rssi=-45,
            bandwidth_bps=250000,
        )
        d = m.model_dump()
        assert d["type"] == "esp_now"
        assert d["state"] == "available"
        assert d["rssi"] == -45

    def test_json_roundtrip(self):
        m = TransportMetrics(
            type=TransportType.MQTT,
            state=TransportState.AVAILABLE,
            latency_ms=12.3,
            last_active=_utc(),
        )
        json_str = m.model_dump_json()
        m2 = TransportMetrics.model_validate_json(json_str)
        assert m2.type == m.type
        assert m2.latency_ms == m.latency_ms

    def test_all_transport_types(self):
        for t in TransportType:
            m = TransportMetrics(type=t, state=TransportState.UNAVAILABLE)
            assert m.type == t

    def test_all_states(self):
        for s in TransportState:
            m = TransportMetrics(type=TransportType.WIFI, state=s)
            assert m.state == s


# ---------------------------------------------------------------------------
# TransportPreference
# ---------------------------------------------------------------------------

class TestTransportPreference:
    def test_create_basic(self):
        p = TransportPreference(type=TransportType.WIFI, priority=1)
        assert p.type == TransportType.WIFI
        assert p.priority == 1
        assert p.min_rssi is None
        assert p.max_latency_ms is None

    def test_create_with_thresholds(self):
        p = TransportPreference(
            type=TransportType.LORA,
            priority=3,
            min_rssi=-100,
            max_latency_ms=500.0,
        )
        assert p.min_rssi == -100
        assert p.max_latency_ms == 500.0

    def test_serialization(self):
        p = TransportPreference(type=TransportType.BLE, priority=2, min_rssi=-70)
        d = p.model_dump()
        assert d["type"] == "ble"
        assert d["priority"] == 2
        assert d["min_rssi"] == -70


# ---------------------------------------------------------------------------
# NodeTransportStatus
# ---------------------------------------------------------------------------

class TestNodeTransportStatus:
    def _make_status(self):
        return NodeTransportStatus(
            device_id="node-001",
            transports=[
                TransportMetrics(
                    type=TransportType.WIFI, state=TransportState.AVAILABLE,
                    rssi=-40, bandwidth_bps=54_000_000, latency_ms=5.0,
                ),
                TransportMetrics(
                    type=TransportType.ESP_NOW, state=TransportState.AVAILABLE,
                    rssi=-55, bandwidth_bps=250_000, latency_ms=2.0,
                ),
                TransportMetrics(
                    type=TransportType.BLE, state=TransportState.DEGRADED,
                    rssi=-75, bandwidth_bps=125_000, latency_ms=20.0,
                ),
                TransportMetrics(
                    type=TransportType.LORA, state=TransportState.UNAVAILABLE,
                ),
                TransportMetrics(
                    type=TransportType.ACOUSTIC, state=TransportState.DISABLED,
                ),
            ],
            preferred_transport=TransportType.WIFI,
            active_transport=TransportType.WIFI,
        )

    def test_create(self):
        s = self._make_status()
        assert s.device_id == "node-001"
        assert len(s.transports) == 5

    def test_get_transport(self):
        s = self._make_status()
        wifi = s.get_transport(TransportType.WIFI)
        assert wifi is not None
        assert wifi.rssi == -40

        mqtt = s.get_transport(TransportType.MQTT)
        assert mqtt is None

    def test_available_transports(self):
        s = self._make_status()
        avail = s.available_transports
        types = {t.type for t in avail}
        assert types == {TransportType.WIFI, TransportType.ESP_NOW, TransportType.BLE}

    def test_empty_status(self):
        s = NodeTransportStatus(device_id="node-empty")
        assert s.transports == []
        assert s.preferred_transport is None
        assert s.active_transport is None
        assert s.available_transports == []

    def test_serialization(self):
        s = self._make_status()
        d = s.model_dump()
        assert d["device_id"] == "node-001"
        assert d["active_transport"] == "wifi"
        assert len(d["transports"]) == 5

    def test_json_roundtrip(self):
        s = self._make_status()
        json_str = s.model_dump_json()
        s2 = NodeTransportStatus.model_validate_json(json_str)
        assert s2.device_id == s.device_id
        assert len(s2.transports) == len(s.transports)
        assert s2.active_transport == s.active_transport


# ---------------------------------------------------------------------------
# select_best_transport
# ---------------------------------------------------------------------------

class TestSelectBestTransport:
    def _default_transports(self):
        return [
            TransportMetrics(
                type=TransportType.WIFI, state=TransportState.AVAILABLE,
                rssi=-40, latency_ms=5.0,
            ),
            TransportMetrics(
                type=TransportType.ESP_NOW, state=TransportState.AVAILABLE,
                rssi=-55, latency_ms=2.0,
            ),
            TransportMetrics(
                type=TransportType.BLE, state=TransportState.DEGRADED,
                rssi=-75, latency_ms=20.0,
            ),
            TransportMetrics(
                type=TransportType.LORA, state=TransportState.UNAVAILABLE,
            ),
        ]

    def _default_preferences(self):
        return [
            TransportPreference(type=TransportType.WIFI, priority=1, min_rssi=-70, max_latency_ms=50.0),
            TransportPreference(type=TransportType.ESP_NOW, priority=2, min_rssi=-80),
            TransportPreference(type=TransportType.BLE, priority=3, min_rssi=-80),
            TransportPreference(type=TransportType.LORA, priority=4, min_rssi=-110),
        ]

    def test_selects_highest_priority(self):
        result = select_best_transport(self._default_transports(), self._default_preferences())
        assert result == TransportType.WIFI

    def test_falls_back_when_preferred_unavailable(self):
        transports = self._default_transports()
        # Make WiFi unavailable
        transports[0] = TransportMetrics(
            type=TransportType.WIFI, state=TransportState.UNAVAILABLE,
        )
        result = select_best_transport(transports, self._default_preferences())
        assert result == TransportType.ESP_NOW

    def test_rssi_threshold_skips_weak_signal(self):
        transports = [
            TransportMetrics(
                type=TransportType.WIFI, state=TransportState.AVAILABLE,
                rssi=-85, latency_ms=5.0,  # Below min_rssi of -70
            ),
            TransportMetrics(
                type=TransportType.ESP_NOW, state=TransportState.AVAILABLE,
                rssi=-55, latency_ms=2.0,
            ),
        ]
        prefs = [
            TransportPreference(type=TransportType.WIFI, priority=1, min_rssi=-70),
            TransportPreference(type=TransportType.ESP_NOW, priority=2, min_rssi=-80),
        ]
        result = select_best_transport(transports, prefs)
        assert result == TransportType.ESP_NOW

    def test_latency_threshold_skips_slow(self):
        transports = [
            TransportMetrics(
                type=TransportType.WIFI, state=TransportState.AVAILABLE,
                rssi=-40, latency_ms=100.0,  # Above max_latency_ms of 50
            ),
            TransportMetrics(
                type=TransportType.ESP_NOW, state=TransportState.AVAILABLE,
                rssi=-55, latency_ms=2.0,
            ),
        ]
        prefs = [
            TransportPreference(type=TransportType.WIFI, priority=1, max_latency_ms=50.0),
            TransportPreference(type=TransportType.ESP_NOW, priority=2),
        ]
        result = select_best_transport(transports, prefs)
        assert result == TransportType.ESP_NOW

    def test_fallback_ignores_thresholds(self):
        """When no transport meets thresholds, fall back to highest-priority usable."""
        transports = [
            TransportMetrics(
                type=TransportType.WIFI, state=TransportState.AVAILABLE,
                rssi=-90, latency_ms=200.0,
            ),
            TransportMetrics(
                type=TransportType.BLE, state=TransportState.AVAILABLE,
                rssi=-95, latency_ms=300.0,
            ),
        ]
        prefs = [
            TransportPreference(type=TransportType.WIFI, priority=1, min_rssi=-50, max_latency_ms=10.0),
            TransportPreference(type=TransportType.BLE, priority=2, min_rssi=-50, max_latency_ms=10.0),
        ]
        result = select_best_transport(transports, prefs)
        # WiFi is priority 1, so it's the fallback even though it fails thresholds
        assert result == TransportType.WIFI

    def test_no_usable_transports(self):
        transports = [
            TransportMetrics(type=TransportType.WIFI, state=TransportState.UNAVAILABLE),
            TransportMetrics(type=TransportType.BLE, state=TransportState.DISABLED),
        ]
        prefs = [
            TransportPreference(type=TransportType.WIFI, priority=1),
            TransportPreference(type=TransportType.BLE, priority=2),
        ]
        result = select_best_transport(transports, prefs)
        assert result is None

    def test_empty_transports(self):
        prefs = [TransportPreference(type=TransportType.WIFI, priority=1)]
        assert select_best_transport([], prefs) is None

    def test_empty_preferences(self):
        """With no preferences, should still return a usable transport."""
        transports = [
            TransportMetrics(type=TransportType.BLE, state=TransportState.AVAILABLE),
        ]
        result = select_best_transport(transports, [])
        assert result == TransportType.BLE

    def test_degraded_still_usable(self):
        transports = [
            TransportMetrics(type=TransportType.LORA, state=TransportState.DEGRADED, rssi=-100),
        ]
        prefs = [TransportPreference(type=TransportType.LORA, priority=1)]
        result = select_best_transport(transports, prefs)
        assert result == TransportType.LORA

    def test_none_rssi_skips_threshold_check(self):
        """When metrics has no rssi, the threshold check is skipped (passes)."""
        transports = [
            TransportMetrics(type=TransportType.MQTT, state=TransportState.AVAILABLE),
        ]
        prefs = [TransportPreference(type=TransportType.MQTT, priority=1, min_rssi=-70)]
        result = select_best_transport(transports, prefs)
        assert result == TransportType.MQTT

    def test_none_latency_skips_threshold_check(self):
        """When metrics has no latency, the threshold check is skipped (passes)."""
        transports = [
            TransportMetrics(type=TransportType.USB_SERIAL, state=TransportState.AVAILABLE),
        ]
        prefs = [TransportPreference(type=TransportType.USB_SERIAL, priority=1, max_latency_ms=10.0)]
        result = select_best_transport(transports, prefs)
        assert result == TransportType.USB_SERIAL

    def test_transport_not_in_preferences_used_as_last_resort(self):
        """A transport with no preference entry is still usable as last resort."""
        transports = [
            TransportMetrics(type=TransportType.WIFI, state=TransportState.UNAVAILABLE),
            TransportMetrics(type=TransportType.ACOUSTIC, state=TransportState.AVAILABLE),
        ]
        prefs = [TransportPreference(type=TransportType.WIFI, priority=1)]
        result = select_best_transport(transports, prefs)
        assert result == TransportType.ACOUSTIC

    def test_priority_ordering(self):
        """Preferences with out-of-order priority values should still sort correctly."""
        transports = [
            TransportMetrics(type=TransportType.BLE, state=TransportState.AVAILABLE, rssi=-50),
            TransportMetrics(type=TransportType.ESP_NOW, state=TransportState.AVAILABLE, rssi=-50),
        ]
        prefs = [
            TransportPreference(type=TransportType.BLE, priority=10),
            TransportPreference(type=TransportType.ESP_NOW, priority=5),
        ]
        result = select_best_transport(transports, prefs)
        assert result == TransportType.ESP_NOW


# ---------------------------------------------------------------------------
# transport_summary
# ---------------------------------------------------------------------------

class TestTransportSummary:
    def test_full_summary(self):
        status = NodeTransportStatus(
            device_id="node-abc",
            transports=[
                TransportMetrics(type=TransportType.WIFI, state=TransportState.AVAILABLE),
                TransportMetrics(type=TransportType.ESP_NOW, state=TransportState.AVAILABLE),
                TransportMetrics(type=TransportType.BLE, state=TransportState.DEGRADED),
                TransportMetrics(type=TransportType.LORA, state=TransportState.UNAVAILABLE),
                TransportMetrics(type=TransportType.ACOUSTIC, state=TransportState.DISABLED),
            ],
            preferred_transport=TransportType.WIFI,
            active_transport=TransportType.WIFI,
        )
        s = transport_summary(status)
        assert s["device_id"] == "node-abc"
        assert s["active"] == "wifi"
        assert s["preferred"] == "wifi"
        assert s["total"] == 5
        assert s["by_state"]["available"] == 2
        assert s["by_state"]["degraded"] == 1
        assert s["by_state"]["unavailable"] == 1
        assert s["by_state"]["disabled"] == 1
        assert set(s["available"]) == {"wifi", "esp_now", "ble"}

    def test_empty_status(self):
        status = NodeTransportStatus(device_id="node-empty")
        s = transport_summary(status)
        assert s["device_id"] == "node-empty"
        assert s["active"] is None
        assert s["preferred"] is None
        assert s["total"] == 0
        assert s["by_state"] == {}
        assert s["available"] == []

    def test_all_unavailable(self):
        status = NodeTransportStatus(
            device_id="node-down",
            transports=[
                TransportMetrics(type=TransportType.WIFI, state=TransportState.UNAVAILABLE),
                TransportMetrics(type=TransportType.BLE, state=TransportState.UNAVAILABLE),
            ],
        )
        s = transport_summary(status)
        assert s["total"] == 2
        assert s["available"] == []
        assert s["by_state"]["unavailable"] == 2

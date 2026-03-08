# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for alert and webhook models."""

from datetime import datetime, timezone

from tritium_lib.models.alert import (
    Alert,
    AlertDelivery,
    AlertHistory,
    AlertSeverity,
    WebhookConfig,
    classify_alert_severity,
    summarize_alerts,
)


class TestWebhookConfig:
    def test_unfiltered(self):
        wh = WebhookConfig(id="w1", url="http://example.com/hook")
        assert not wh.is_filtered_by_device
        assert not wh.is_filtered_by_event
        assert wh.matches("node_anomaly", "dev-1", 0.5)

    def test_severity_filter(self):
        wh = WebhookConfig(id="w1", url="http://x", severity_min=0.7)
        assert not wh.matches("node_anomaly", "dev-1", 0.3)
        assert wh.matches("node_anomaly", "dev-1", 0.7)
        assert wh.matches("node_anomaly", "dev-1", 1.0)

    def test_device_filter(self):
        wh = WebhookConfig(id="w1", url="http://x", device_ids=["dev-1", "dev-2"])
        assert wh.is_filtered_by_device
        assert wh.matches("node_anomaly", "dev-1", 0.5)
        assert not wh.matches("node_anomaly", "dev-3", 0.5)

    def test_event_type_filter(self):
        wh = WebhookConfig(id="w1", url="http://x", event_types=["node_offline"])
        assert wh.is_filtered_by_event
        assert wh.matches("node_offline", "dev-1", 0.5)
        assert not wh.matches("node_anomaly", "dev-1", 0.5)

    def test_combined_filters(self):
        wh = WebhookConfig(
            id="w1", url="http://x",
            severity_min=0.5,
            device_ids=["dev-1"],
            event_types=["node_anomaly"],
        )
        # All pass
        assert wh.matches("node_anomaly", "dev-1", 0.8)
        # Severity too low
        assert not wh.matches("node_anomaly", "dev-1", 0.3)
        # Wrong device
        assert not wh.matches("node_anomaly", "dev-2", 0.8)
        # Wrong event
        assert not wh.matches("node_offline", "dev-1", 0.8)


class TestAlert:
    def _make_alert(self, severity=0.5, deliveries=None):
        return Alert(
            id="a1",
            timestamp=datetime.now(timezone.utc),
            event_type="node_anomaly",
            device_id="dev-1",
            detail="Heap low",
            severity=severity,
            deliveries=deliveries or [],
        )

    def test_severity_level_info(self):
        assert self._make_alert(0.2).severity_level == AlertSeverity.INFO

    def test_severity_level_warning(self):
        assert self._make_alert(0.5).severity_level == AlertSeverity.WARNING

    def test_severity_level_critical(self):
        assert self._make_alert(0.9).severity_level == AlertSeverity.CRITICAL

    def test_delivery_counts(self):
        deliveries = [
            AlertDelivery(webhook_id="w1", url="http://x", ok=True, status_code=200),
            AlertDelivery(webhook_id="w2", url="http://y", ok=False, error="timeout"),
        ]
        alert = self._make_alert(deliveries=deliveries)
        assert alert.delivery_count == 2
        assert alert.successful_deliveries == 1

    def test_no_deliveries(self):
        alert = self._make_alert()
        assert alert.delivery_count == 0
        assert alert.successful_deliveries == 0


class TestClassifySeverity:
    def test_info(self):
        assert classify_alert_severity(0.0) == AlertSeverity.INFO
        assert classify_alert_severity(0.39) == AlertSeverity.INFO

    def test_warning(self):
        assert classify_alert_severity(0.4) == AlertSeverity.WARNING
        assert classify_alert_severity(0.69) == AlertSeverity.WARNING

    def test_critical(self):
        assert classify_alert_severity(0.7) == AlertSeverity.CRITICAL
        assert classify_alert_severity(1.0) == AlertSeverity.CRITICAL


class TestSummarizeAlerts:
    def test_empty(self):
        summary = summarize_alerts([])
        assert summary.total_alerts == 0
        assert summary.critical_count == 0
        assert summary.warning_count == 0
        assert summary.info_count == 0

    def test_mixed(self):
        now = datetime.now(timezone.utc)
        alerts = [
            Alert(id="a1", timestamp=now, event_type="t", device_id="d",
                  detail="", severity=0.9),  # critical
            Alert(id="a2", timestamp=now, event_type="t", device_id="d",
                  detail="", severity=0.5),  # warning
            Alert(id="a3", timestamp=now, event_type="t", device_id="d",
                  detail="", severity=0.2),  # info
            Alert(id="a4", timestamp=now, event_type="t", device_id="d",
                  detail="", severity=0.8),  # critical
        ]
        summary = summarize_alerts(alerts)
        assert summary.total_alerts == 4
        assert summary.critical_count == 2
        assert summary.warning_count == 1
        assert summary.info_count == 1
        assert len(summary.recent) == 4

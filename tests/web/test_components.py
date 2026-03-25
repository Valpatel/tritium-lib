# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.web.components — HTML component library."""

from datetime import datetime

from tritium_lib.web.components import AlertBanner, MetricCard, StatusBadge, TimelineEvent


class TestStatusBadge:
    def test_render_online(self):
        html = StatusBadge.render("online")
        assert "status-dot" in html
        assert "badge" in html
        assert "online" in html

    def test_render_offline(self):
        html = StatusBadge.render("offline")
        assert "offline" in html

    def test_render_with_label(self):
        html = StatusBadge.render("online", label="Connected")
        assert "Connected" in html

    def test_render_error(self):
        html = StatusBadge.render("error")
        assert "error" in html


class TestMetricCard:
    def test_basic_metric(self):
        html = MetricCard.render("Temperature", "23.5", unit="C")
        assert "Temperature" in html
        assert "23.5" in html
        assert "C" in html
        assert "metric" in html

    def test_metric_no_unit(self):
        html = MetricCard.render("Count", "42")
        assert "42" in html
        # Should not have unit span
        assert 'class="unit"' not in html

    def test_metric_with_sparkline(self):
        html = MetricCard.render("CPU", "75", sparkline=[10, 20, 50, 75, 60])
        assert "sparkline" in html
        assert "spark-bar" in html

    def test_sparkline_max_20(self):
        """Only last 20 data points shown."""
        data = list(range(30))
        html = MetricCard.render("Load", "5", sparkline=data)
        # Count spark-bar occurrences
        assert html.count("spark-bar") == 20

    def test_sparkline_empty(self):
        html = MetricCard.render("Load", "0", sparkline=[])
        assert "sparkline" not in html


class TestTimelineEvent:
    def test_render_single_event(self):
        ts = datetime(2026, 3, 24, 14, 30, 0)
        html = TimelineEvent.render(ts, "Device connected", category="fleet")
        assert "timeline-event" in html
        assert "14:30:00" in html
        assert "Device connected" in html
        assert "fleet" in html

    def test_render_timeline(self):
        events = [
            (datetime(2026, 3, 24, 14, 0), "Start", "system"),
            (datetime(2026, 3, 24, 14, 5), "Scan complete", "sensor"),
            (datetime(2026, 3, 24, 14, 10), "Target detected", "combat"),
        ]
        html = TimelineEvent.render_timeline(events)
        assert '<div class="timeline">' in html
        assert html.count("timeline-event") == 3

    def test_html_escaping(self):
        ts = datetime(2026, 3, 24, 12, 0)
        html = TimelineEvent.render(ts, '<script>alert("xss")</script>')
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestAlertBanner:
    def test_info_banner(self):
        html = AlertBanner.render("System update available", severity="info")
        assert "alert" in html
        assert "info" in html
        assert "System update available" in html

    def test_warning_banner(self):
        html = AlertBanner.render("Low battery", severity="warning")
        assert "warning" in html

    def test_error_banner(self):
        html = AlertBanner.render("Connection lost", severity="error")
        assert "error" in html

    def test_dismissable(self):
        html = AlertBanner.render("Test", dismissable=True)
        assert "dismiss" in html

    def test_not_dismissable(self):
        html = AlertBanner.render("Test", dismissable=False)
        assert "dismiss" not in html

    def test_html_escaping(self):
        html = AlertBanner.render('<img src=x onerror="alert(1)">')
        assert "<img" not in html
        assert "&lt;img" in html

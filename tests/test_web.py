"""Tests for tritium_lib.web — theme, components, dashboard, templates."""

from datetime import datetime, timezone

import pytest

from tritium_lib.models import Device, DeviceHeartbeat, SensorReading
from tritium_lib.web import (
    TritiumTheme,
    StatusBadge,
    MetricCard,
    DeviceTable,
    TimelineEvent,
    AlertBanner,
    DashboardPage,
    full_page,
    admin_page,
    mobile_page,
    commissioning_page,
    node_dashboard_page,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def theme():
    return TritiumTheme()


@pytest.fixture
def sample_device():
    return Device(
        device_id="esp32-001",
        device_name="hallway-sensor",
        mac="20:6E:F1:9A:12:00",
        board="touch-lcd-35bc",
        firmware_version="1.2.0",
        ip_address="192.168.1.50",
        capabilities=["camera", "audio", "imu", "temperature"],
        status="online",
        last_seen=datetime(2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc),
        registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_devices():
    return [
        Device(
            device_id="esp32-001",
            device_name="hallway-sensor",
            mac="20:6E:F1:9A:12:00",
            board="touch-lcd-35bc",
            firmware_version="1.2.0",
            ip_address="192.168.1.50",
            status="online",
            capabilities=["camera", "audio"],
        ),
        Device(
            device_id="esp32-002",
            device_name="garage-node",
            mac="1C:DB:D4:9C:CD:68",
            board="touch-amoled-241b",
            firmware_version="1.1.0",
            ip_address="192.168.1.51",
            status="offline",
        ),
        Device(
            device_id="esp32-003",
            device_name="error-node",
            board="touch-lcd-349",
            status="error",
        ),
    ]


@pytest.fixture
def sample_heartbeat():
    return DeviceHeartbeat(
        device_id="esp32-001",
        firmware_version="1.2.0",
        board="touch-lcd-35bc",
        uptime_s=86400,
        free_heap=180000,
        wifi_rssi=-55,
        ip_address="192.168.1.50",
        boot_count=3,
        mesh_peers=2,
        capabilities=["camera", "audio", "imu"],
    )


@pytest.fixture
def sample_sensors():
    return [
        SensorReading(
            device_id="esp32-001",
            sensor_type="temperature",
            value=23.5,
            unit="°C",
        ),
        SensorReading(
            device_id="esp32-001",
            sensor_type="humidity",
            value=62.1,
            unit="%",
        ),
    ]


@pytest.fixture
def sample_ble_sightings():
    return [
        {
            "mac": "AA:BB:CC:DD:EE:01",
            "name": "iPhone-12",
            "rssi": -65,
            "seen_by": ["esp32-001", "esp32-002"],
            "last_seen": "2026-03-07T12:00:00Z",
        },
        {
            "mac": "AA:BB:CC:DD:EE:02",
            "name": "Unknown",
            "rssi": -90,
            "seen_by": ["esp32-001"],
            "last_seen": "2026-03-07T11:55:00Z",
        },
    ]


# ── Theme tests ─────────────────────────────────────────────────────────


class TestTheme:
    def test_css_contains_design_tokens(self, theme):
        css = theme.css()
        assert "#0a0a0a" in css  # background
        assert "#00ffd0" in css  # accent
        assert "Courier New" in css  # monospace font
        assert "monospace" in css

    def test_css_contains_scanlines(self, theme):
        css = theme.css()
        assert "scanline" in css.lower() or "repeating-linear-gradient" in css

    def test_css_responsive(self, theme):
        css = theme.css()
        assert "@media" in css
        assert "768px" in css

    def test_css_contains_neon_glow(self, theme):
        css = theme.css()
        assert "text-shadow" in css or "box-shadow" in css

    def test_nav_html(self, theme):
        pages = [("/", "Home"), ("/fleet", "Fleet"), ("/config", "Config")]
        html = theme.nav_html(pages, active="/fleet")
        assert 'class="nav"' in html
        assert 'href="/"' in html
        assert 'href="/fleet"' in html
        assert 'class="active"' in html

    def test_card(self, theme):
        html = theme.card("Test Title", "<p>Content</p>")
        assert "card" in html
        assert "Test Title" in html
        assert "<p>Content</p>" in html

    def test_table(self, theme):
        html = theme.table(["Name", "Value"], [["Temp", "23.5"]])
        assert "<table>" in html
        assert "<th>" in html
        assert "Temp" in html
        assert "23.5" in html

    def test_badge(self, theme):
        html = theme.badge("online", "online")
        assert "badge" in html
        assert "online" in html

    def test_progress_bar(self, theme):
        html = theme.progress_bar(75)
        assert "bar-bg" in html
        assert "bar-fill" in html
        assert "75%" in html

    def test_progress_bar_clamps(self, theme):
        html_low = theme.progress_bar(-10)
        assert "0%" in html_low
        html_high = theme.progress_bar(200)
        assert "100%" in html_high

    def test_device_card(self, theme, sample_device):
        html = theme.device_card(sample_device)
        assert "hallway-sensor" in html
        assert "touch-lcd-35bc" in html
        assert "20:6E:F1:9A:12:00" in html
        assert "1.2.0" in html
        assert "192.168.1.50" in html
        assert "online" in html
        assert "camera" in html

    def test_sensor_gauge(self, theme):
        reading = SensorReading(
            device_id="esp32-001",
            sensor_type="temperature",
            value=23.5,
            unit="°C",
        )
        html = theme.sensor_gauge(reading)
        assert "gauge" in html
        assert "23.5" in html
        assert "temperature" in html

    def test_rssi_bar(self, theme):
        html = theme.rssi_bar(-55)
        assert "rssi-bars" in html
        assert "-55dBm" in html

    def test_rssi_bar_strong_signal(self, theme):
        html = theme.rssi_bar(-30)
        assert "-30dBm" in html

    def test_rssi_bar_weak_signal(self, theme):
        html = theme.rssi_bar(-95)
        assert "-95dBm" in html


# ── Component tests ─────────────────────────────────────────────────────


class TestComponents:
    def test_status_badge_online(self):
        html = StatusBadge.render("online")
        assert "status-dot" in html
        assert "online" in html

    def test_status_badge_custom_label(self):
        html = StatusBadge.render("error", label="CRITICAL")
        assert "CRITICAL" in html
        assert "error" in html

    def test_metric_card(self):
        html = MetricCard.render("Temperature", "23.5", "°C")
        assert "metric" in html
        assert "23.5" in html
        assert "Temperature" in html

    def test_metric_card_with_sparkline(self):
        data = [10, 20, 30, 25, 15, 35, 40]
        html = MetricCard.render("Load", "35", "%", sparkline=data)
        assert "sparkline" in html
        assert "spark-bar" in html
        assert "35" in html

    def test_device_table(self, sample_devices):
        html = DeviceTable.render(
            sample_devices,
            rssi_map={"esp32-001": -55},
            uptime_map={"esp32-001": 3600},
        )
        assert "<table>" in html
        assert "hallway-sensor" in html
        assert "garage-node" in html
        assert "-55dBm" in html
        assert "1h 0m" in html

    def test_device_table_empty(self):
        html = DeviceTable.render([])
        assert "<table>" in html

    def test_timeline_event(self):
        ts = datetime(2026, 3, 7, 12, 30, 0)
        html = TimelineEvent.render(ts, "Device came online", "connection")
        assert "12:30:00" in html
        assert "Device came online" in html
        assert "connection" in html

    def test_timeline_render_multiple(self):
        events = [
            (datetime(2026, 3, 7, 12, 0, 0), "Boot", "system"),
            (datetime(2026, 3, 7, 12, 1, 0), "WiFi connected", "network"),
        ]
        html = TimelineEvent.render_timeline(events)
        assert "timeline" in html
        assert "Boot" in html
        assert "WiFi connected" in html

    def test_alert_banner_info(self):
        html = AlertBanner.render("System update available", "info")
        assert "alert" in html
        assert "info" in html
        assert "System update available" in html
        assert "dismiss" in html

    def test_alert_banner_error_dismissable(self):
        html = AlertBanner.render("Connection lost", "error", dismissable=True)
        assert "error" in html
        assert "&#x2715;" in html

    def test_alert_banner_not_dismissable(self):
        html = AlertBanner.render("Read only", "warning", dismissable=False)
        assert "dismiss" not in html


# ── Dashboard tests ─────────────────────────────────────────────────────


class TestDashboard:
    def test_fleet_overview(self, sample_devices):
        dash = DashboardPage()
        html = dash.render_fleet_overview(sample_devices)
        assert "Fleet Overview" in html
        assert "hallway-sensor" in html
        assert "garage-node" in html
        assert "3" in html  # total count

    def test_fleet_overview_with_heartbeats(self, sample_devices, sample_heartbeat):
        dash = DashboardPage()
        heartbeats = {"esp32-001": sample_heartbeat}
        html = dash.render_fleet_overview(sample_devices, heartbeats)
        assert "Fleet Overview" in html

    def test_device_detail(self, sample_device, sample_heartbeat, sample_sensors):
        dash = DashboardPage()
        html = dash.render_device_detail(sample_device, sample_heartbeat, sample_sensors)
        assert "hallway-sensor" in html
        assert "Uptime" in html
        assert "Free Heap" in html
        assert "temperature" in html
        assert "humidity" in html
        assert "Capabilities" in html

    def test_device_detail_minimal(self):
        device = Device(device_id="esp32-minimal")
        dash = DashboardPage()
        html = dash.render_device_detail(device)
        assert "esp32-minimal" in html

    def test_ble_presence(self, sample_ble_sightings):
        dash = DashboardPage()
        html = dash.render_ble_presence(sample_ble_sightings)
        assert "BLE Presence" in html
        assert "iPhone-12" in html
        assert "AA:BB:CC:DD:EE:01" in html

    def test_ble_presence_empty(self):
        dash = DashboardPage()
        html = dash.render_ble_presence([])
        assert "No BLE devices" in html

    def test_network_topology(self, sample_devices):
        dash = DashboardPage()
        html = dash.render_network_topology(sample_devices)
        assert "Network Topology" in html
        assert "hallway-sensor" in html

    def test_network_topology_empty(self):
        dash = DashboardPage()
        html = dash.render_network_topology([])
        assert "No devices" in html


# ── Template tests ──────────────────────────────────────────────────────


class TestTemplates:
    def test_full_page_structure(self):
        html = full_page("Test", "<p>Hello</p>")
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert 'charset="utf-8"' in html
        assert "viewport" in html
        assert "Test — Tritium" in html
        assert "<p>Hello</p>" in html
        assert "#0a0a0a" in html  # theme CSS present

    def test_full_page_has_responsive_meta(self):
        html = full_page("Test", "")
        assert "width=device-width" in html
        assert "initial-scale=1" in html

    def test_admin_page_has_sidebar(self):
        pages = [("/", "Dashboard"), ("/fleet", "Fleet")]
        html = admin_page("Fleet", pages, "<p>Content</p>", active_url="/fleet")
        assert "admin-layout" in html
        assert "sidebar" in html
        assert "Dashboard" in html
        assert "Fleet" in html
        assert 'class="active"' in html
        assert "TRITIUM" in html

    def test_mobile_page_meta_tags(self):
        html = mobile_page("Mobile", "<p>Hi</p>")
        assert "maximum-scale=1" in html
        assert "user-scalable=no" in html
        assert "apple-mobile-web-app-capable" in html
        assert "mobile-web-app-capable" in html

    def test_commissioning_page_has_wifi_form(self):
        html = commissioning_page()
        assert "wifi-form" in html or "WiFi Configuration" in html
        assert 'name="ssid"' in html
        assert 'name="password"' in html
        assert "Scan Networks" in html
        assert "Fleet Registration" in html
        assert 'name="device_name"' in html
        assert 'name="server_url"' in html

    def test_commissioning_page_full_html(self):
        html = commissioning_page()
        assert "<!DOCTYPE html>" in html
        assert "Commissioning" in html

    def test_node_dashboard_page(self, sample_device, sample_heartbeat, sample_sensors):
        html = node_dashboard_page(
            sample_device,
            heartbeat=sample_heartbeat,
            sensors=sample_sensors,
        )
        assert "<!DOCTYPE html>" in html
        assert "hallway-sensor" in html
        assert "Dashboard" in html
        assert "OTA Update" in html


# ── XSS safety tests ───────────────────────────────────────────────────


class TestSafety:
    def test_device_name_escaped(self, theme):
        evil = Device(
            device_id="evil",
            device_name='<script>alert("xss")</script>',
        )
        html = theme.device_card(evil)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_badge_text_escaped(self, theme):
        html = theme.badge('<img src=x onerror=alert(1)>')
        assert "<img" not in html

    def test_nav_url_escaped(self, theme):
        html = theme.nav_html([('"><script>alert(1)</script>', "Evil")])
        assert "<script>" not in html

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Reusable dashboard page generators for the Tritium ecosystem.

Composes device cards, sensor readouts, fleet status grids, and
network topology views from Tritium data models.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tritium_lib.models import Device, DeviceHeartbeat, SensorReading

from .theme import TritiumTheme
from .components import StatusBadge, MetricCard, DeviceTable, AlertBanner


def _esc(text: str) -> str:
    return html.escape(str(text))


class DashboardPage:
    """Composes dashboard sections from Tritium models."""

    def __init__(self, theme: TritiumTheme | None = None):
        self.theme = theme or TritiumTheme()

    def render_fleet_overview(
        self,
        devices: list["Device"],
        heartbeats: dict[str, "DeviceHeartbeat"] | None = None,
    ) -> str:
        """Render a grid of device cards with status badges.

        Args:
            devices: List of Device models.
            heartbeats: Optional dict mapping device_id -> latest heartbeat.
        """
        heartbeats = heartbeats or {}

        # Summary metrics
        total = len(devices)
        online = sum(1 for d in devices if d.status == "online")
        offline = total - online
        error_count = sum(1 for d in devices if d.status == "error")

        summary = f"""
        <div class="grid grid-4" style="margin-bottom:16px">
            {MetricCard.render("Total Nodes", str(total))}
            {MetricCard.render("Online", str(online))}
            {MetricCard.render("Offline", str(offline))}
            {MetricCard.render("Errors", str(error_count))}
        </div>
        """

        # Device cards grid
        cards = []
        for d in devices:
            hb = heartbeats.get(d.device_id)
            card = self.theme.device_card(d)
            if hb and hb.wifi_rssi is not None:
                rssi_html = self.theme.rssi_bar(hb.wifi_rssi)
                card = card.replace("</div>\n        </div>",
                    f'<div style="margin-top:6px">{rssi_html}</div></div>\n        </div>',
                    1)
            cards.append(card)

        grid = f'<div class="grid grid-2">{"".join(cards)}</div>'

        return f"""
        <h1>Fleet Overview</h1>
        {summary}
        {grid}
        """

    def render_device_detail(
        self,
        device: "Device",
        heartbeat: "DeviceHeartbeat | None" = None,
        sensors: list["SensorReading"] | None = None,
    ) -> str:
        """Render a full device detail page.

        Args:
            device: The Device model.
            heartbeat: Latest heartbeat data.
            sensors: Recent sensor readings for this device.
        """
        sensors = sensors or []

        # Device header
        header = self.theme.device_card(device)

        # Heartbeat info
        hb_section = ""
        if heartbeat:
            metrics = []
            if heartbeat.uptime_s is not None:
                h = heartbeat.uptime_s // 3600
                m = (heartbeat.uptime_s % 3600) // 60
                metrics.append(MetricCard.render("Uptime", f"{h}h {m}m"))
            if heartbeat.free_heap is not None:
                kb = heartbeat.free_heap / 1024
                metrics.append(MetricCard.render("Free Heap", f"{kb:.0f}", "KB"))
            if heartbeat.wifi_rssi is not None:
                metrics.append(MetricCard.render("WiFi RSSI", str(heartbeat.wifi_rssi), "dBm"))
            if heartbeat.boot_count is not None:
                metrics.append(MetricCard.render("Boot Count", str(heartbeat.boot_count)))
            if heartbeat.mesh_peers is not None:
                metrics.append(MetricCard.render("Mesh Peers", str(heartbeat.mesh_peers)))

            hb_section = f"""
            <h2>System Metrics</h2>
            <div class="grid grid-3">{"".join(metrics)}</div>
            """

        # Sensor readings
        sensor_section = ""
        if sensors:
            gauges = "".join(self.theme.sensor_gauge(r) for r in sensors)
            sensor_section = f"""
            <h2>Sensor Readings</h2>
            <div style="display:flex;flex-wrap:wrap;gap:8px">{gauges}</div>
            """

        # Capabilities
        caps_section = ""
        if device.capabilities:
            caps_badges = " ".join(
                self.theme.badge(c, "online") for c in device.capabilities
            )
            caps_section = f"<h2>Capabilities</h2><div>{caps_badges}</div>"

        return f"""
        <h1>{_esc(device.device_name or device.device_id)}</h1>
        {header}
        {hb_section}
        {sensor_section}
        {caps_section}
        """

    def render_ble_presence(
        self,
        ble_sightings: list[dict],
    ) -> str:
        """Render a BLE presence list for triangulation.

        Args:
            ble_sightings: List of dicts with keys: mac, name, rssi,
                           seen_by (list of device_ids), last_seen.
        """
        if not ble_sightings:
            return '<div class="card"><p>No BLE devices detected.</p></div>'

        rows = []
        for s in ble_sightings:
            mac = _esc(s.get("mac", "??:??:??"))
            name = _esc(s.get("name", "Unknown"))
            rssi = s.get("rssi", -100)
            rssi_html = TritiumTheme().rssi_bar(rssi)
            seen_by = s.get("seen_by", [])
            observers = ", ".join(_esc(o) for o in seen_by) if seen_by else "—"
            last_seen = _esc(str(s.get("last_seen", "—")))
            rows.append([mac, name, rssi_html, observers, last_seen])

        headers = ["MAC", "Name", "RSSI", "Seen By", "Last Seen"]
        table = self.theme.table(headers, rows)

        return f"""
        <h2>BLE Presence</h2>
        {table}
        """

    def render_network_topology(
        self,
        devices: list["Device"],
    ) -> str:
        """Render a mesh network topology visualization.

        Uses a simple grid layout showing connections between nodes.
        For actual mesh data, this would be enhanced with peer info.
        """
        if not devices:
            return '<div class="card"><p>No devices in network.</p></div>'

        # Build node list with visual representation
        nodes = []
        for d in devices:
            status = d.status or "offline"
            node_html = f"""
            <div class="card" style="text-align:center;min-width:120px">
                <span class="status-dot {_esc(status)}"></span><br>
                <strong style="color:{TritiumTheme.ACCENT}">{_esc(d.device_name or d.device_id)}</strong>
                <div class="label">{_esc(d.board)}</div>
                <div class="label">{_esc(d.ip_address or 'no ip')}</div>
            </div>
            """
            nodes.append(node_html)

        return f"""
        <h2>Network Topology</h2>
        <div class="grid grid-3">{"".join(nodes)}</div>
        """

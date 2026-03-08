# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone HTML component library for the Tritium ecosystem.

Every component returns a plain HTML string — no framework dependencies.
Designed for Jinja2 template injection, direct HTTP serving, or
string concatenation in microcontroller webservers.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tritium_lib.models import Device

from .theme import TritiumTheme

_theme = TritiumTheme()


def _esc(text: str) -> str:
    return html.escape(str(text))


class StatusBadge:
    """Online/offline/error badge with colored dot."""

    @staticmethod
    def render(status: str, label: str | None = None) -> str:
        """Render a status badge.

        Args:
            status: One of 'online', 'offline', 'error', 'updating'.
            label: Display text. Defaults to status value.
        """
        display = label or status
        return (
            f'<span class="status-dot {_esc(status)}"></span>'
            f'<span class="badge {_esc(status)}">{_esc(display)}</span>'
        )


class MetricCard:
    """Label + large value display with optional sparkline."""

    @staticmethod
    def render(
        label: str,
        value: str,
        unit: str = "",
        sparkline: list[float] | None = None,
    ) -> str:
        """Render a metric card.

        Args:
            label: Metric name (e.g. "Temperature").
            value: Display value (e.g. "23.5").
            unit: Optional unit suffix (e.g. "C").
            sparkline: Optional list of values for a mini bar chart.
        """
        spark_html = ""
        if sparkline:
            max_v = max(sparkline) if sparkline else 1
            max_v = max_v if max_v > 0 else 1
            bars = []
            for v in sparkline[-20:]:  # last 20 data points
                h = max(2, int((v / max_v) * 20))
                bars.append(f'<span class="spark-bar" style="height:{h}px"></span>')
            spark_html = f'<div class="sparkline" style="margin-top:8px">{"".join(bars)}</div>'

        unit_span = f'<span class="unit">{_esc(unit)}</span>' if unit else ""
        return f"""
        <div class="card metric">
            <div class="value">{_esc(value)}{unit_span}</div>
            <div class="label">{_esc(label)}</div>
            {spark_html}
        </div>
        """


class DeviceTable:
    """Sortable device table with MAC, IP, RSSI, uptime, firmware version."""

    @staticmethod
    def render(devices: list["Device"], rssi_map: dict[str, int] | None = None,
               uptime_map: dict[str, int] | None = None) -> str:
        """Render a device table.

        Args:
            devices: List of Device models.
            rssi_map: Optional dict mapping device_id -> RSSI in dBm.
            uptime_map: Optional dict mapping device_id -> uptime in seconds.
        """
        rssi_map = rssi_map or {}
        uptime_map = uptime_map or {}

        headers = ["Status", "Name", "Board", "MAC", "IP", "RSSI", "Uptime", "Firmware"]
        rows = []
        for d in devices:
            status = StatusBadge.render(d.status)
            name = _esc(d.device_name or d.device_id)
            board = _esc(d.board)
            mac = _esc(d.mac)
            ip = _esc(d.ip_address or "—")

            rssi = rssi_map.get(d.device_id)
            rssi_html = _theme.rssi_bar(rssi) if rssi is not None else "—"

            uptime_s = uptime_map.get(d.device_id)
            if uptime_s is not None:
                hours = uptime_s // 3600
                mins = (uptime_s % 3600) // 60
                uptime_html = f"{hours}h {mins}m"
            else:
                uptime_html = "—"

            fw = _esc(d.firmware_version)
            rows.append([status, name, board, mac, ip, rssi_html, uptime_html, fw])

        return _theme.table(headers, rows)


class TimelineEvent:
    """Event log entry with timestamp and category."""

    @staticmethod
    def render(
        timestamp: datetime,
        message: str,
        category: str = "system",
    ) -> str:
        """Render a timeline event entry."""
        ts = timestamp.strftime("%H:%M:%S")
        return f"""
        <div class="timeline-event">
            <span class="time">{ts}</span>
            <span class="category">{_esc(category)}</span>
            <div style="margin-top:4px">{_esc(message)}</div>
        </div>
        """

    @staticmethod
    def render_timeline(events: list[tuple[datetime, str, str]]) -> str:
        """Render a full timeline from a list of (timestamp, message, category) tuples."""
        entries = "".join(
            TimelineEvent.render(ts, msg, cat) for ts, msg, cat in events
        )
        return f'<div class="timeline">{entries}</div>'


class AlertBanner:
    """Dismissable alert banner with severity levels."""

    @staticmethod
    def render(
        message: str,
        severity: str = "info",
        dismissable: bool = True,
    ) -> str:
        """Render an alert banner.

        Args:
            message: Alert text.
            severity: One of 'info', 'warning', 'error'.
            dismissable: Whether to show dismiss button.
        """
        dismiss = (
            '<span class="dismiss" onclick="this.parentElement.remove()">&#x2715;</span>'
            if dismissable
            else ""
        )
        return (
            f'<div class="alert {_esc(severity)}">'
            f'<span>{_esc(message)}</span>{dismiss}</div>'
        )

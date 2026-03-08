# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared CSS theme system for the Tritium ecosystem.

Dark cyberpunk/neon aesthetic — consistent with the ESP32 hal_webserver
theme deployed on edge nodes. Black background, cyan accents, monospace
typography, neon glow effects, scanline overlays.

This is the canonical Tritium design language. All web surfaces — node
dashboards, fleet admin, command center — use this theme.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tritium_lib.models import Device, SensorReading


class TritiumTheme:
    """Generates CSS and HTML components in the Tritium cyberpunk style."""

    # ── Design tokens ───────────────────────────────────────────────────
    BG = "#0a0a0a"
    BG_CARD = "#111111"
    BG_INPUT = "#0a0a0a"
    ACCENT = "#00ffd0"
    ACCENT_HOVER = "#66ffe8"
    ACCENT_DIM = "#00ffd022"
    ACCENT_MID = "#00ffd044"
    DANGER = "#ff3366"
    DANGER_HOVER = "#ff6690"
    WARNING = "#ffaa00"
    TEXT = "#c0c0c0"
    TEXT_DIM = "#666666"
    BORDER = "#1a1a1a"
    FONT = "'Courier New', 'Fira Code', monospace"

    # ── CSS generation ──────────────────────────────────────────────────

    def css(self) -> str:
        """Return the full Tritium theme CSS string."""
        return f"""
/* ═══ Tritium Theme — Cyberpunk Terminal Aesthetic ═══ */
*{{margin:0;padding:0;box-sizing:border-box}}

body{{
  background:{self.BG};
  color:{self.TEXT};
  font-family:{self.FONT};
  font-size:14px;
  padding:20px;
  max-width:1200px;
  margin:0 auto;
  line-height:1.5;
}}

/* Scanline overlay */
body::after{{
  content:'';
  position:fixed;
  top:0;left:0;right:0;bottom:0;
  pointer-events:none;
  background:repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,255,208,0.015) 2px,
    rgba(0,255,208,0.015) 4px
  );
  z-index:9999;
}}

h1{{
  color:{self.ACCENT};
  font-size:22px;
  border-bottom:1px solid {self.ACCENT_MID};
  padding-bottom:8px;
  margin-bottom:16px;
  text-shadow:0 0 10px {self.ACCENT_DIM};
}}
h2{{color:{self.ACCENT};font-size:16px;margin:16px 0 8px;
  text-shadow:0 0 6px {self.ACCENT_DIM}}}
h3{{color:{self.ACCENT};font-size:14px;margin:12px 0 6px}}

a{{color:{self.ACCENT};text-decoration:none}}
a:hover{{text-decoration:underline;color:{self.ACCENT_HOVER}}}

/* Cards */
.card{{
  background:{self.BG_CARD};
  border:1px solid {self.ACCENT_DIM};
  border-radius:6px;
  padding:16px;
  margin-bottom:12px;
  transition:border-color 0.2s;
}}
.card:hover{{border-color:{self.ACCENT_MID}}}

/* Tables */
table{{width:100%;border-collapse:collapse;margin:8px 0}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid {self.BORDER}}}
th{{color:{self.ACCENT};font-weight:normal;font-size:12px;text-transform:uppercase;
  letter-spacing:1px}}

/* Forms */
input[type=text],input[type=password],select{{
  background:{self.BG_INPUT};color:{self.ACCENT};
  border:1px solid {self.ACCENT_MID};border-radius:4px;
  padding:8px 12px;font-family:{self.FONT};font-size:13px;
  width:100%;margin:4px 0;
}}
input[type=text]:focus,input[type=password]:focus{{
  outline:none;border-color:{self.ACCENT};
  box-shadow:0 0 8px {self.ACCENT_DIM};
}}
textarea{{
  background:{self.BG_INPUT};color:{self.ACCENT};
  border:1px solid {self.ACCENT_MID};border-radius:4px;
  padding:10px;width:100%;font-family:{self.FONT};font-size:13px;resize:vertical;
}}

/* Buttons */
button,input[type=submit]{{
  background:{self.ACCENT};color:{self.BG};border:none;
  padding:8px 20px;border-radius:4px;cursor:pointer;
  font-family:inherit;font-weight:bold;font-size:13px;margin:4px 2px;
  text-transform:uppercase;letter-spacing:1px;
  transition:all 0.2s;
}}
button:hover,input[type=submit]:hover{{
  background:{self.ACCENT_HOVER};
  box-shadow:0 0 12px {self.ACCENT_DIM};
}}
button.danger{{background:{self.DANGER};color:#fff}}
button.danger:hover{{background:{self.DANGER_HOVER}}}

/* Progress bars */
.bar-bg{{
  background:{self.BORDER};border-radius:3px;height:14px;
  width:120px;display:inline-block;vertical-align:middle;
  overflow:hidden;
}}
.bar-fill{{
  background:{self.ACCENT};height:100%;border-radius:3px;
  transition:width 0.3s;
  box-shadow:0 0 6px {self.ACCENT_DIM};
}}

/* Navigation */
.nav{{margin-bottom:16px;display:flex;gap:12px;flex-wrap:wrap}}
.nav a{{
  background:{self.BG_CARD};border:1px solid {self.ACCENT_DIM};
  padding:6px 14px;border-radius:4px;font-size:13px;
  transition:all 0.2s;
}}
.nav a:hover{{background:{self.BORDER};border-color:{self.ACCENT};
  text-decoration:none;box-shadow:0 0 8px {self.ACCENT_DIM}}}
.nav a.active{{border-color:{self.ACCENT};color:{self.ACCENT_HOVER}}}

/* Utility */
.label{{color:{self.TEXT_DIM};font-size:12px;text-transform:uppercase;letter-spacing:1px}}
.msg{{padding:10px;border-radius:4px;margin:8px 0}}
.msg.ok{{background:#00ffd011;border:1px solid {self.ACCENT_MID};color:{self.ACCENT}}}
.msg.err{{background:#ff336611;border:1px solid #ff336644;color:{self.DANGER}}}
.msg.warn{{background:#ffaa0011;border:1px solid #ffaa0044;color:{self.WARNING}}}

/* Badges */
.badge{{
  display:inline-block;padding:2px 8px;border-radius:10px;
  font-size:11px;text-transform:uppercase;letter-spacing:1px;
  font-weight:bold;
}}
.badge.online{{background:#00ffd022;color:{self.ACCENT};border:1px solid {self.ACCENT_MID}}}
.badge.offline{{background:#66666622;color:{self.TEXT_DIM};border:1px solid #666666}}
.badge.error{{background:#ff336622;color:{self.DANGER};border:1px solid #ff336644}}
.badge.updating{{background:#ffaa0022;color:{self.WARNING};border:1px solid #ffaa0044}}

/* Status dot */
.status-dot{{
  display:inline-block;width:8px;height:8px;border-radius:50%;
  margin-right:6px;vertical-align:middle;
}}
.status-dot.online{{background:{self.ACCENT};box-shadow:0 0 6px {self.ACCENT}}}
.status-dot.offline{{background:{self.TEXT_DIM}}}
.status-dot.error{{background:{self.DANGER};box-shadow:0 0 6px {self.DANGER}}}
.status-dot.updating{{background:{self.WARNING};box-shadow:0 0 6px {self.WARNING}}}

/* Grid layout */
.grid{{display:grid;gap:12px}}
.grid-2{{grid-template-columns:repeat(auto-fill,minmax(280px,1fr))}}
.grid-3{{grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}}
.grid-4{{grid-template-columns:repeat(auto-fill,minmax(180px,1fr))}}

/* Metric card */
.metric{{text-align:center;padding:20px}}
.metric .value{{font-size:28px;color:{self.ACCENT};font-weight:bold;
  text-shadow:0 0 10px {self.ACCENT_DIM}}}
.metric .unit{{font-size:12px;color:{self.TEXT_DIM};margin-left:4px}}
.metric .label{{margin-top:6px}}

/* Gauge */
.gauge{{
  position:relative;width:100px;height:50px;overflow:hidden;
  margin:0 auto;
}}
.gauge-bg{{
  width:100px;height:100px;border-radius:50%;
  border:6px solid {self.BORDER};border-bottom:none;
  clip-path:inset(0 0 50% 0);
}}
.gauge-fill{{
  position:absolute;top:0;left:0;
  width:100px;height:100px;border-radius:50%;
  border:6px solid {self.ACCENT};border-bottom:none;
  clip-path:inset(0 0 50% 0);
  transform-origin:center center;
  transition:transform 0.5s;
}}
.gauge-value{{
  position:absolute;bottom:0;left:50%;transform:translateX(-50%);
  font-size:14px;color:{self.ACCENT};font-weight:bold;
}}

/* RSSI signal bars */
.rssi-bars{{display:inline-flex;align-items:flex-end;gap:2px;height:16px;vertical-align:middle}}
.rssi-bars .bar{{width:4px;border-radius:1px;transition:background 0.2s}}

/* Sparkline (CSS-only mini chart) */
.sparkline{{display:inline-flex;align-items:flex-end;gap:1px;height:20px}}
.sparkline .spark-bar{{width:3px;background:{self.ACCENT};border-radius:1px;
  min-height:2px;opacity:0.7}}

/* Alert banner */
.alert{{
  padding:12px 16px;border-radius:4px;margin:8px 0;
  display:flex;align-items:center;justify-content:space-between;
}}
.alert.info{{background:#00ffd011;border:1px solid {self.ACCENT_MID};color:{self.ACCENT}}}
.alert.warning{{background:#ffaa0011;border:1px solid #ffaa0044;color:{self.WARNING}}}
.alert.error{{background:#ff336611;border:1px solid #ff336644;color:{self.DANGER}}}
.alert .dismiss{{cursor:pointer;opacity:0.6;font-size:18px}}
.alert .dismiss:hover{{opacity:1}}

/* Timeline */
.timeline{{border-left:2px solid {self.ACCENT_DIM};margin-left:8px;padding-left:16px}}
.timeline-event{{position:relative;margin-bottom:12px;padding:8px 12px;
  background:{self.BG_CARD};border-radius:4px;border:1px solid {self.ACCENT_DIM}}}
.timeline-event::before{{
  content:'';position:absolute;left:-21px;top:12px;
  width:10px;height:10px;border-radius:50%;background:{self.ACCENT};
  box-shadow:0 0 6px {self.ACCENT};
}}
.timeline-event .time{{color:{self.TEXT_DIM};font-size:11px}}
.timeline-event .category{{color:{self.ACCENT};font-size:11px;text-transform:uppercase;
  margin-left:8px}}

/* Sidebar admin layout */
.admin-layout{{display:grid;grid-template-columns:200px 1fr;gap:20px;min-height:100vh}}
.sidebar{{
  background:{self.BG_CARD};border-right:1px solid {self.ACCENT_DIM};
  padding:16px;position:sticky;top:0;height:100vh;overflow-y:auto;
}}
.sidebar a{{display:block;padding:8px 12px;margin:2px 0;border-radius:4px;
  color:{self.TEXT};font-size:13px}}
.sidebar a:hover{{background:{self.BORDER};color:{self.ACCENT};text-decoration:none}}
.sidebar a.active{{background:{self.ACCENT_DIM};color:{self.ACCENT}}}
.main-content{{padding:0 16px}}

/* Responsive */
@media (max-width:768px){{
  body{{padding:10px;font-size:13px}}
  .admin-layout{{grid-template-columns:1fr}}
  .sidebar{{position:static;height:auto;border-right:none;
    border-bottom:1px solid {self.ACCENT_DIM}}}
  .grid-2,.grid-3,.grid-4{{grid-template-columns:1fr}}
  .metric .value{{font-size:22px}}
  h1{{font-size:18px}}
}}
@media (max-width:480px){{
  body{{padding:8px}}
  .nav{{gap:6px}}
  .nav a{{padding:4px 10px;font-size:12px}}
  .card{{padding:10px}}
}}
"""

    # ── HTML component generators ───────────────────────────────────────

    def nav_html(self, pages: list[tuple[str, str]], active: str = "") -> str:
        """Render a navigation bar.

        Args:
            pages: List of (url, label) tuples.
            active: URL of the currently active page.
        """
        links = []
        for url, label in pages:
            cls = ' class="active"' if url == active else ""
            links.append(f'<a href="{_esc(url)}"{cls}>{_esc(label)}</a>')
        return f'<div class="nav">{" ".join(links)}</div>'

    def card(self, title: str, body: str, css_class: str = "") -> str:
        """Render a card component."""
        cls = f"card {css_class}".strip()
        header = f"<h3>{_esc(title)}</h3>" if title else ""
        return f'<div class="{cls}">{header}{body}</div>'

    def table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Render an HTML table."""
        ths = "".join(f"<th>{_esc(h)}</th>" for h in headers)
        trs = []
        for row in rows:
            tds = "".join(f"<td>{cell}</td>" for cell in row)
            trs.append(f"<tr>{tds}</tr>")
        return (
            f'<table><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table>'
        )

    def badge(self, text: str, variant: str = "online") -> str:
        """Render a badge. Variants: online, offline, error, updating."""
        return f'<span class="badge {_esc(variant)}">{_esc(text)}</span>'

    def progress_bar(self, percent: float, width: str = "120px") -> str:
        """Render a progress bar (0-100)."""
        pct = max(0.0, min(100.0, percent))
        return (
            f'<span class="bar-bg" style="width:{width}">'
            f'<span class="bar-fill" style="width:{pct:.0f}%"></span></span>'
        )

    def device_card(self, device: "Device") -> str:
        """Render a Device model as a status card."""
        status = device.status or "offline"
        dot = f'<span class="status-dot {_esc(status)}"></span>'
        badge = self.badge(status, status)

        caps = ""
        if device.capabilities:
            cap_badges = " ".join(
                f'<span class="badge online" style="font-size:9px">{_esc(c)}</span>'
                for c in device.capabilities[:6]
            )
            caps = f'<div style="margin-top:8px">{cap_badges}</div>'

        last_seen = ""
        if device.last_seen:
            last_seen = (
                f'<div class="label" style="margin-top:6px">'
                f'Last seen: {device.last_seen.strftime("%Y-%m-%d %H:%M:%S")}</div>'
            )

        ip_line = ""
        if device.ip_address:
            ip_line = f'<div class="label">IP: {_esc(device.ip_address)}</div>'

        return self.card("", f"""
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    {dot}<strong>{_esc(device.device_name or device.device_id)}</strong>
                </div>
                {badge}
            </div>
            <div class="label" style="margin-top:6px">
                {_esc(device.board)} &middot; {_esc(device.mac)} &middot;
                fw {_esc(device.firmware_version)}
            </div>
            {ip_line}
            {last_seen}
            {caps}
        """)

    def sensor_gauge(self, reading: "SensorReading") -> str:
        """Render a SensorReading as a visual gauge with value display."""
        val = reading.value if isinstance(reading.value, (int, float)) else 0
        # Normalize to 0-180 degrees for half-circle gauge
        # Use sensor-type heuristics for range
        ranges = {
            "temperature": (-20, 60),
            "humidity": (0, 100),
            "pressure": (900, 1100),
        }
        lo, hi = ranges.get(reading.sensor_type, (0, 100))
        ratio = max(0.0, min(1.0, (val - lo) / (hi - lo))) if hi != lo else 0.5
        deg = ratio * 180

        unit = _esc(reading.unit) if reading.unit else ""
        label = _esc(reading.sensor_type)

        display_val = f"{val:.1f}" if isinstance(reading.value, float) else str(val)

        return f"""
        <div class="card metric" style="width:140px;display:inline-block">
            <div class="gauge">
                <div class="gauge-bg"></div>
                <div class="gauge-fill" style="transform:rotate({deg:.0f}deg)"></div>
                <div class="gauge-value">{display_val}{unit}</div>
            </div>
            <div class="label" style="margin-top:8px">{label}</div>
        </div>
        """

    def rssi_bar(self, rssi_dbm: int) -> str:
        """Render a WiFi/BLE signal strength indicator (-100 to 0 dBm).

        Shows 5 bars, filled based on signal strength.
        """
        # Normalize: -100 dBm = 0%, -30 dBm = 100%
        strength = max(0.0, min(1.0, (rssi_dbm + 100) / 70))
        filled = round(strength * 5)

        bars = []
        for i in range(5):
            h = 4 + i * 3  # heights: 4, 7, 10, 13, 16
            if i < filled:
                color = self.ACCENT if strength > 0.3 else self.WARNING
                if strength <= 0.15:
                    color = self.DANGER
            else:
                color = self.BORDER
            bars.append(f'<span class="bar" style="height:{h}px;background:{color}"></span>')

        return (
            f'<span class="rssi-bars">{"".join(bars)}</span>'
            f' <span class="label">{rssi_dbm}dBm</span>'
        )


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text))

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Complete HTML page templates for the Tritium ecosystem.

Wraps content in full HTML documents with the Tritium cyberpunk theme,
responsive meta tags, and proper structure for desktop and mobile.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tritium_lib.models import Device

from .theme import TritiumTheme
from .components import StatusBadge, MetricCard
from .dashboard import DashboardPage


def _esc(text: str) -> str:
    return html.escape(str(text))


def full_page(
    title: str,
    body_html: str,
    theme: TritiumTheme | None = None,
) -> str:
    """Wrap content in a full HTML document with Tritium theme CSS.

    Args:
        title: Page title.
        body_html: Inner body HTML content.
        theme: Optional TritiumTheme instance (uses default if None).
    """
    theme = theme or TritiumTheme()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} — Tritium</title>
<style>{theme.css()}</style>
</head>
<body>
{body_html}
</body>
</html>"""


def admin_page(
    title: str,
    nav_pages: list[tuple[str, str]],
    body_html: str,
    active_url: str = "",
    theme: TritiumTheme | None = None,
) -> str:
    """Admin layout with sidebar navigation.

    Args:
        title: Page title.
        nav_pages: List of (url, label) for sidebar links.
        body_html: Main content area HTML.
        active_url: Currently active URL for highlighting.
        theme: Optional theme instance.
    """
    theme = theme or TritiumTheme()

    sidebar_links = []
    for url, label in nav_pages:
        cls = ' class="active"' if url == active_url else ""
        sidebar_links.append(f'<a href="{_esc(url)}"{cls}>{_esc(label)}</a>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} — Tritium Admin</title>
<style>{theme.css()}</style>
</head>
<body>
<div class="admin-layout">
    <div class="sidebar">
        <h2 style="margin-bottom:16px">TRITIUM</h2>
        {"".join(sidebar_links)}
    </div>
    <div class="main-content">
        <h1>{_esc(title)}</h1>
        {body_html}
    </div>
</div>
</body>
</html>"""


def mobile_page(
    title: str,
    body_html: str,
    theme: TritiumTheme | None = None,
) -> str:
    """Mobile-optimized single-column layout.

    Args:
        title: Page title.
        body_html: Body content HTML.
        theme: Optional theme instance.
    """
    theme = theme or TritiumTheme()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<title>{_esc(title)} — Tritium</title>
<style>
{theme.css()}
body{{padding:8px;max-width:100%}}
.card{{margin-bottom:8px}}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


def commissioning_page(theme: TritiumTheme | None = None) -> str:
    """WiFi setup + device registration flow for new ESP32 nodes.

    This page is served by a freshly flashed node in AP mode.
    The user connects to the node's WiFi and navigates here to
    configure network credentials and register with the fleet server.
    """
    theme = theme or TritiumTheme()
    return full_page("Node Commissioning", f"""
    <h1>Node Commissioning</h1>

    <div class="card">
        <h2>WiFi Configuration</h2>
        <form id="wifi-form" method="POST" action="/api/wifi">
            <div style="margin:8px 0">
                <label class="label">Network SSID</label>
                <input type="text" name="ssid" id="ssid" placeholder="Enter WiFi network name" required>
            </div>
            <div style="margin:8px 0">
                <label class="label">Password</label>
                <input type="password" name="password" id="password" placeholder="Enter WiFi password">
            </div>
            <div style="margin:12px 0">
                <button type="button" onclick="scanNetworks()">Scan Networks</button>
            </div>
            <div id="scan-results" style="margin:8px 0"></div>
            <button type="submit">Save &amp; Connect</button>
        </form>
    </div>

    <div class="card">
        <h2>Fleet Registration</h2>
        <form id="register-form" method="POST" action="/api/register">
            <div style="margin:8px 0">
                <label class="label">Device Name</label>
                <input type="text" name="device_name" placeholder="e.g. hallway-sensor-01">
            </div>
            <div style="margin:8px 0">
                <label class="label">Fleet Server URL</label>
                <input type="text" name="server_url" placeholder="https://fleet.example.com">
            </div>
            <div style="margin:8px 0">
                <label class="label">Registration Token</label>
                <input type="password" name="reg_token" placeholder="Fleet registration token">
            </div>
            <button type="submit">Register Node</button>
        </form>
    </div>

    <div id="status-msg"></div>

    <script>
    function scanNetworks() {{
        fetch('/api/scan').then(r=>r.json()).then(data=>{{
            let html='<table><tr><th>SSID</th><th>RSSI</th><th>Security</th></tr>';
            data.networks.forEach(n=>{{
                html+='<tr onclick="document.getElementById(\\'ssid\\').value=\\''+n.ssid+'\\'"'+
                    ' style="cursor:pointer"><td>'+n.ssid+'</td><td>'+n.rssi+'dBm</td>'+
                    '<td>'+n.security+'</td></tr>';
            }});
            html+='</table>';
            document.getElementById('scan-results').innerHTML=html;
        }});
    }}
    </script>
    """, theme)


def node_dashboard_page(
    device: "Device",
    heartbeat: "object | None" = None,
    sensors: "list | None" = None,
    theme: TritiumTheme | None = None,
) -> str:
    """Dashboard page served by each ESP32 node at its own IP.

    Shows the node's own status, sensor readings, and provides
    links to OTA update, config, and fleet server.

    Args:
        device: This node's Device model.
        heartbeat: Latest DeviceHeartbeat (optional).
        sensors: Recent SensorReading list (optional).
        theme: Optional theme instance.
    """
    theme = theme or TritiumTheme()
    dash = DashboardPage(theme)

    nav = theme.nav_html([
        ("/", "Dashboard"),
        ("/config", "Config"),
        ("/update", "OTA Update"),
        ("/api/status", "API"),
    ], active="/")

    detail = dash.render_device_detail(device, heartbeat, sensors)

    return full_page(
        f"{device.device_name or device.device_id}",
        f"{nav}{detail}",
        theme,
    )

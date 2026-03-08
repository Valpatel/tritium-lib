# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tritium web UI module — shared theme, components, dashboards, and templates.

Provides the Tritium cyberpunk design language as reusable Python functions
that generate HTML strings. No framework dependencies — works with Jinja2,
FastAPI, Flask, or direct string serving on ESP32 webservers.
"""

from .theme import TritiumTheme
from .components import (
    StatusBadge,
    MetricCard,
    DeviceTable,
    TimelineEvent,
    AlertBanner,
)
from .dashboard import DashboardPage
from .templates import (
    full_page,
    admin_page,
    mobile_page,
    commissioning_page,
    node_dashboard_page,
)

__all__ = [
    "TritiumTheme",
    "StatusBadge",
    "MetricCard",
    "DeviceTable",
    "TimelineEvent",
    "AlertBanner",
    "DashboardPage",
    "full_page",
    "admin_page",
    "mobile_page",
    "commissioning_page",
    "node_dashboard_page",
]

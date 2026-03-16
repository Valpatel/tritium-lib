# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Analytics dashboard widget models.

Configurable dashboard widgets for building custom analytics views.
Each widget has a type (counter, chart, table, map, timeline),
a data source endpoint, and display configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class WidgetType(str, Enum):
    """Types of dashboard widgets."""
    COUNTER = "counter"
    CHART = "chart"
    TABLE = "table"
    MAP = "map"
    TIMELINE = "timeline"


@dataclass
class WidgetConfig:
    """Display configuration for a widget.

    Attributes
    ----------
    color:
        Primary color for the widget (CSS hex).
    refresh_seconds:
        Auto-refresh interval in seconds (0 = no auto-refresh).
    max_items:
        Maximum items to display (for tables/timelines).
    chart_type:
        Sub-type for chart widgets: line, bar, pie, area, sparkline.
    show_legend:
        Whether to show a chart legend.
    height:
        Preferred height in grid units (1 unit = ~100px).
    width:
        Preferred width in grid units.
    extra:
        Arbitrary extension data.
    """
    color: str = "#00f0ff"
    refresh_seconds: int = 30
    max_items: int = 20
    chart_type: str = "line"
    show_legend: bool = False
    height: int = 2
    width: int = 2
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "color": self.color,
            "refresh_seconds": self.refresh_seconds,
            "max_items": self.max_items,
            "chart_type": self.chart_type,
            "show_legend": self.show_legend,
            "height": self.height,
            "width": self.width,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WidgetConfig:
        return cls(
            color=data.get("color", "#00f0ff"),
            refresh_seconds=data.get("refresh_seconds", 30),
            max_items=data.get("max_items", 20),
            chart_type=data.get("chart_type", "line"),
            show_legend=data.get("show_legend", False),
            height=data.get("height", 2),
            width=data.get("width", 2),
            extra=data.get("extra", {}),
        )


@dataclass
class DashboardWidget:
    """A configurable analytics dashboard widget.

    Attributes
    ----------
    widget_id:
        Unique identifier for the widget instance.
    title:
        Human-readable title displayed on the widget.
    widget_type:
        The type of visualization (counter, chart, table, map, timeline).
    data_source:
        API endpoint or data key to fetch widget data from.
    config:
        Display configuration.
    position:
        Grid position as {"x": int, "y": int} for drag/drop layout.
    enabled:
        Whether the widget is visible in the dashboard.
    description:
        Optional help text describing what this widget shows.
    """
    widget_id: str = ""
    title: str = ""
    widget_type: WidgetType = WidgetType.COUNTER
    data_source: str = ""
    config: WidgetConfig = field(default_factory=WidgetConfig)
    position: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "widget_id": self.widget_id,
            "title": self.title,
            "widget_type": self.widget_type.value if isinstance(self.widget_type, WidgetType) else self.widget_type,
            "data_source": self.data_source,
            "config": self.config.to_dict(),
            "position": self.position,
            "enabled": self.enabled,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DashboardWidget:
        wtype = data.get("widget_type", "counter")
        if isinstance(wtype, str):
            try:
                wtype = WidgetType(wtype)
            except ValueError:
                wtype = WidgetType.COUNTER

        config_data = data.get("config", {})
        config = WidgetConfig.from_dict(config_data) if isinstance(config_data, dict) else WidgetConfig()

        return cls(
            widget_id=data.get("widget_id", ""),
            title=data.get("title", ""),
            widget_type=wtype,
            data_source=data.get("data_source", ""),
            config=config,
            position=data.get("position", {"x": 0, "y": 0}),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
        )


# Pre-configured widget definitions for the default analytics dashboard
DEFAULT_WIDGETS: list[DashboardWidget] = [
    DashboardWidget(
        widget_id="target_count_trend",
        title="Target Count Trend",
        widget_type=WidgetType.CHART,
        data_source="/api/analytics/history?hours=24",
        config=WidgetConfig(color="#00f0ff", chart_type="area", height=2, width=3),
        position={"x": 0, "y": 0},
        description="Number of tracked targets over the last 24 hours.",
    ),
    DashboardWidget(
        widget_id="threat_level_history",
        title="Threat Level History",
        widget_type=WidgetType.CHART,
        data_source="/api/analytics/history?hours=24",
        config=WidgetConfig(color="#ff2a6d", chart_type="line", height=2, width=3),
        position={"x": 3, "y": 0},
        description="Threat level changes over the last 24 hours.",
    ),
    DashboardWidget(
        widget_id="sighting_rate",
        title="Sighting Rate",
        widget_type=WidgetType.CHART,
        data_source="/api/analytics/history?hours=12",
        config=WidgetConfig(color="#05ffa1", chart_type="bar", height=2, width=3),
        position={"x": 0, "y": 2},
        description="Sighting events per hour over the last 12 hours.",
    ),
    DashboardWidget(
        widget_id="top_devices",
        title="Top Devices",
        widget_type=WidgetType.TABLE,
        data_source="/api/analytics/history?hours=24",
        config=WidgetConfig(color="#fcee0a", max_items=10, height=3, width=3),
        position={"x": 3, "y": 2},
        description="Most active devices by sighting count.",
    ),
    DashboardWidget(
        widget_id="correlation_success_rate",
        title="Correlation Success Rate",
        widget_type=WidgetType.COUNTER,
        data_source="/api/analytics/history?hours=24",
        config=WidgetConfig(color="#05ffa1", height=1, width=2),
        position={"x": 0, "y": 4},
        description="Percentage of successful target correlations.",
    ),
]

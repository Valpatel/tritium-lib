# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for analytics dashboard widget models."""

from tritium_lib.models.analytics_dashboard import (
    DEFAULT_WIDGETS,
    DashboardWidget,
    WidgetConfig,
    WidgetType,
)


def test_widget_type_values():
    assert WidgetType.COUNTER.value == "counter"
    assert WidgetType.CHART.value == "chart"
    assert WidgetType.TABLE.value == "table"
    assert WidgetType.MAP.value == "map"
    assert WidgetType.TIMELINE.value == "timeline"


def test_widget_config_defaults():
    cfg = WidgetConfig()
    assert cfg.color == "#00f0ff"
    assert cfg.refresh_seconds == 30
    assert cfg.max_items == 20
    assert cfg.chart_type == "line"
    assert cfg.show_legend is False
    assert cfg.height == 2
    assert cfg.width == 2
    assert cfg.extra == {}


def test_widget_config_roundtrip():
    cfg = WidgetConfig(color="#ff2a6d", chart_type="bar", height=3, width=4)
    d = cfg.to_dict()
    restored = WidgetConfig.from_dict(d)
    assert restored.color == "#ff2a6d"
    assert restored.chart_type == "bar"
    assert restored.height == 3
    assert restored.width == 4


def test_dashboard_widget_defaults():
    w = DashboardWidget()
    assert w.widget_id == ""
    assert w.widget_type == WidgetType.COUNTER
    assert w.enabled is True
    assert w.position == {"x": 0, "y": 0}


def test_dashboard_widget_roundtrip():
    w = DashboardWidget(
        widget_id="test_1",
        title="Test Widget",
        widget_type=WidgetType.CHART,
        data_source="/api/test",
        config=WidgetConfig(color="#05ffa1", chart_type="sparkline"),
        position={"x": 2, "y": 3},
        enabled=False,
        description="A test widget.",
    )
    d = w.to_dict()
    assert d["widget_type"] == "chart"
    assert d["config"]["color"] == "#05ffa1"

    restored = DashboardWidget.from_dict(d)
    assert restored.widget_id == "test_1"
    assert restored.widget_type == WidgetType.CHART
    assert restored.config.chart_type == "sparkline"
    assert restored.enabled is False
    assert restored.position == {"x": 2, "y": 3}


def test_from_dict_invalid_widget_type():
    d = {"widget_id": "bad", "widget_type": "nonexistent"}
    w = DashboardWidget.from_dict(d)
    assert w.widget_type == WidgetType.COUNTER  # fallback


def test_default_widgets_exist():
    assert len(DEFAULT_WIDGETS) == 5
    ids = [w.widget_id for w in DEFAULT_WIDGETS]
    assert "target_count_trend" in ids
    assert "threat_level_history" in ids
    assert "sighting_rate" in ids
    assert "top_devices" in ids
    assert "correlation_success_rate" in ids


def test_default_widgets_serialize():
    for w in DEFAULT_WIDGETS:
        d = w.to_dict()
        assert "widget_id" in d
        assert "widget_type" in d
        restored = DashboardWidget.from_dict(d)
        assert restored.widget_id == w.widget_id
        assert restored.widget_type == w.widget_type


def test_import_from_top_level():
    """Verify models are accessible from the tritium_lib.models namespace."""
    from tritium_lib.models import DashboardWidget as DW, WidgetType as WT
    assert DW is DashboardWidget
    assert WT is WidgetType

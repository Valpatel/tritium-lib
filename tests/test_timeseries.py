# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for time series and pagination models."""

from datetime import datetime, timedelta, timezone

import pytest

from tritium_lib.models.timeseries import (
    FleetTimeSeries,
    PagedResult,
    TimeSeries,
    TimeSeriesPoint,
)


def _ts(minutes_ago: int, value: float) -> TimeSeriesPoint:
    """Create a test point at N minutes ago."""
    return TimeSeriesPoint(
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        value=value,
    )


class TestTimeSeries:
    def test_empty_series(self):
        ts = TimeSeries(metric="free_heap", unit="bytes")
        assert ts.count == 0
        assert ts.latest is None
        assert ts.oldest is None
        assert ts.values() == []
        stats = ts.stats()
        assert stats["count"] == 0
        assert stats["min"] is None

    def test_single_point(self):
        ts = TimeSeries(metric="temperature", unit="C", points=[_ts(0, 42.5)])
        assert ts.count == 1
        assert ts.latest.value == 42.5
        assert ts.oldest.value == 42.5

    def test_multiple_points_stats(self):
        ts = TimeSeries(
            metric="free_heap",
            unit="bytes",
            points=[_ts(30, 200000), _ts(20, 195000), _ts(10, 190000), _ts(0, 185000)],
        )
        assert ts.count == 4
        stats = ts.stats()
        assert stats["min"] == 185000
        assert stats["max"] == 200000
        assert stats["mean"] == 192500.0
        assert stats["count"] == 4

    def test_rate_of_change(self):
        now = datetime.now(timezone.utc)
        ts = TimeSeries(
            metric="free_heap",
            unit="bytes",
            points=[
                TimeSeriesPoint(timestamp=now - timedelta(hours=2), value=200000),
                TimeSeriesPoint(timestamp=now, value=190000),
            ],
        )
        rate = ts.rate_of_change()
        assert rate is not None
        assert rate == pytest.approx(-5000.0, rel=0.01)

    def test_rate_of_change_insufficient_data(self):
        ts = TimeSeries(metric="x", points=[_ts(0, 100)])
        assert ts.rate_of_change() is None

    def test_rate_of_change_zero_time(self):
        now = datetime.now(timezone.utc)
        ts = TimeSeries(
            metric="x",
            points=[
                TimeSeriesPoint(timestamp=now, value=100),
                TimeSeriesPoint(timestamp=now, value=200),
            ],
        )
        assert ts.rate_of_change() is None

    def test_values_flat_list(self):
        ts = TimeSeries(metric="rssi", points=[_ts(2, -45), _ts(1, -50), _ts(0, -48)])
        assert ts.values() == [-45, -50, -48]

    def test_with_device_id(self):
        ts = TimeSeries(metric="heap", device_id="esp32-aabb", points=[_ts(0, 100)])
        assert ts.device_id == "esp32-aabb"


class TestFleetTimeSeries:
    def test_empty_fleet(self):
        fts = FleetTimeSeries(metric="heap", unit="bytes")
        assert fts.device_count == 0
        stats = fts.fleet_stats()
        assert stats["devices"] == 0

    def test_fleet_stats(self):
        fts = FleetTimeSeries(
            metric="heap",
            series={
                "dev1": TimeSeries(metric="heap", points=[_ts(0, 200000)]),
                "dev2": TimeSeries(metric="heap", points=[_ts(0, 180000)]),
                "dev3": TimeSeries(metric="heap", points=[_ts(0, 190000)]),
            },
        )
        assert fts.device_count == 3
        stats = fts.fleet_stats()
        assert stats["min"] == 180000
        assert stats["max"] == 200000
        assert stats["devices"] == 3

    def test_outlier_detection(self):
        fts = FleetTimeSeries(
            metric="heap",
            series={
                "dev1": TimeSeries(metric="heap", points=[_ts(0, 200000)]),
                "dev2": TimeSeries(metric="heap", points=[_ts(0, 198000)]),
                "dev3": TimeSeries(metric="heap", points=[_ts(0, 202000)]),
                "dev4": TimeSeries(metric="heap", points=[_ts(0, 199000)]),
                "dev5": TimeSeries(metric="heap", points=[_ts(0, 201000)]),
                "dev6": TimeSeries(metric="heap", points=[_ts(0, 10000)]),  # clear outlier
            },
        )
        outliers = fts.outlier_devices(threshold_stddev=2.0)
        assert "dev6" in outliers
        assert "dev1" not in outliers

    def test_outlier_insufficient_devices(self):
        fts = FleetTimeSeries(
            metric="heap",
            series={
                "dev1": TimeSeries(metric="heap", points=[_ts(0, 200000)]),
                "dev2": TimeSeries(metric="heap", points=[_ts(0, 50000)]),
            },
        )
        # Needs at least 3 devices
        assert fts.outlier_devices() == []


class TestPagedResult:
    def test_from_list_first_page(self):
        items = list(range(100))
        page = PagedResult.from_list(items, offset=0, limit=25)
        assert len(page.items) == 25
        assert page.total == 100
        assert page.has_more is True
        assert page.items[0] == 0

    def test_from_list_last_page(self):
        items = list(range(100))
        page = PagedResult.from_list(items, offset=75, limit=25)
        assert len(page.items) == 25
        assert page.has_more is False
        assert page.items[-1] == 99

    def test_from_list_middle_page(self):
        items = list(range(100))
        page = PagedResult.from_list(items, offset=50, limit=25)
        assert page.has_more is True
        assert page.items[0] == 50

    def test_from_list_empty(self):
        page = PagedResult.from_list([], offset=0, limit=25)
        assert page.total == 0
        assert page.has_more is False
        assert page.items == []

    def test_from_list_beyond_end(self):
        items = list(range(10))
        page = PagedResult.from_list(items, offset=20, limit=25)
        assert page.items == []
        assert page.total == 10
        assert page.has_more is False

    def test_exact_page_boundary(self):
        items = list(range(50))
        page = PagedResult.from_list(items, offset=0, limit=50)
        assert len(page.items) == 50
        assert page.has_more is False

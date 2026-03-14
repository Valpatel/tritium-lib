# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for OperationalPeriod models."""

import pytest
from datetime import datetime, timezone, timedelta

from tritium_lib.models.operational import (
    OperationalObjective,
    OperationalPeriod,
    OperationalPhase,
    WeatherInfo,
)


class TestWeatherInfo:
    def test_default(self):
        w = WeatherInfo()
        assert w.condition == ""
        assert w.temperature_c is None

    def test_to_dict(self):
        w = WeatherInfo(condition="clear", temperature_c=22.5, wind_speed_kph=15.0)
        d = w.to_dict()
        assert d["condition"] == "clear"
        assert d["temperature_c"] == 22.5
        assert d["wind_speed_kph"] == 15.0


class TestOperationalObjective:
    def test_default(self):
        obj = OperationalObjective(description="Secure perimeter")
        assert obj.description == "Secure perimeter"
        assert not obj.completed
        assert obj.priority == 1

    def test_to_dict(self):
        obj = OperationalObjective(description="Test", priority=2)
        d = obj.to_dict()
        assert d["description"] == "Test"
        assert d["priority"] == 2
        assert d["completed"] is False
        assert d["completed_at"] is None


class TestOperationalPeriod:
    def test_default(self):
        op = OperationalPeriod()
        assert op.phase == OperationalPhase.PLANNED
        assert op.personnel_count == 0
        assert op.commander == ""
        assert len(op.period_id) > 0

    def test_with_commander_and_objectives(self):
        op = OperationalPeriod(
            commander="CDR Smith",
            personnel_count=12,
            objectives=[
                OperationalObjective(description="Secure north gate"),
                OperationalObjective(description="Deploy cameras"),
            ],
            weather=WeatherInfo(condition="overcast", temperature_c=18.0),
        )
        assert op.commander == "CDR Smith"
        assert op.personnel_count == 12
        assert len(op.objectives) == 2
        assert op.weather.condition == "overcast"

    def test_activate(self):
        op = OperationalPeriod()
        assert op.phase == OperationalPhase.PLANNED
        op.activate()
        assert op.phase == OperationalPhase.ACTIVE

    def test_complete(self):
        op = OperationalPeriod()
        op.activate()
        op.complete()
        assert op.phase == OperationalPhase.COMPLETED
        assert op.end is not None

    def test_cancel(self):
        op = OperationalPeriod()
        op.cancel()
        assert op.phase == OperationalPhase.CANCELLED
        assert op.end is not None

    def test_is_terminal(self):
        op = OperationalPeriod()
        assert not op.is_terminal
        op.complete()  # can't complete from planned directly, will stay planned
        # complete requires active or transition
        op2 = OperationalPeriod()
        op2.activate()
        op2.complete()
        assert op2.is_terminal

    def test_complete_objective(self):
        obj = OperationalObjective(description="Test")
        op = OperationalPeriod(objectives=[obj])
        assert op.progress == 0.0
        result = op.complete_objective(obj.objective_id)
        assert result is True
        assert op.progress == 1.0
        assert obj.completed is True
        assert obj.completed_at is not None

    def test_complete_objective_not_found(self):
        op = OperationalPeriod()
        assert op.complete_objective("nonexistent") is False

    def test_progress(self):
        objs = [
            OperationalObjective(description="A"),
            OperationalObjective(description="B"),
            OperationalObjective(description="C"),
            OperationalObjective(description="D"),
        ]
        op = OperationalPeriod(objectives=objs)
        assert op.progress == 0.0
        op.complete_objective(objs[0].objective_id)
        assert op.progress == 0.25
        op.complete_objective(objs[1].objective_id)
        assert op.progress == 0.5

    def test_duration_seconds(self):
        start = datetime(2026, 3, 14, 8, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 14, 16, 0, 0, tzinfo=timezone.utc)
        op = OperationalPeriod(start=start, end=end)
        assert op.duration_seconds == 28800.0  # 8 hours

    def test_duration_none_when_open(self):
        op = OperationalPeriod()
        assert op.duration_seconds is None

    def test_to_dict(self):
        op = OperationalPeriod(
            commander="CDR Jones",
            personnel_count=8,
            site_id="alpha",
            tags=["night-ops"],
        )
        d = op.to_dict()
        assert d["commander"] == "CDR Jones"
        assert d["personnel_count"] == 8
        assert d["site_id"] == "alpha"
        assert d["tags"] == ["night-ops"]
        assert d["phase"] == "planned"
        assert d["progress"] == 0.0
        assert "period_id" in d
        assert "start" in d

    def test_empty_objectives_progress(self):
        op = OperationalPeriod()
        assert op.progress == 0.0
